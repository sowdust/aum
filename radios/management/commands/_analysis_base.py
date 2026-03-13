"""
Shared base classes for per-stage analysis commands.

Underscore prefix prevents Django from registering this as a command.

Two base classes:
- AnalysisStageCommand: for recording-level stages (segmentation, summarization)
- SegmentStageCommand: for segment-level stages (fingerprinting, transcription)

Provides:
- Signal handling (SIGINT/SIGTERM) with graceful shutdown
- --once and --limit arguments
- Polling loop with configurable interval (ANALYZE_POLL_INTERVAL)
- Optimistic claim: UPDATE ... SET status='running' WHERE status='pending'
- Stale claim recovery on startup
- Skip logic via stream.is_stage_active()
- Completion tracking (analysis_started_at / analysis_completed_at)
- --retry-failed flag to reset failed → pending
"""

import os
import signal
import time
import traceback
import logging

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from radios.models import Recording, TranscriptionSegment

logger = logging.getLogger("broadcast_analysis")


class _ShutdownRequested(Exception):
    """Raised when a shutdown signal is received during processing."""


class AnalysisStageCommand(BaseCommand):
    """
    Abstract base for commands that process one recording-level pipeline stage.

    Subclasses must define:
        stage_name (str): e.g. "segmentation"
        upstream_done_fields (list[str]): status fields that must be done/skipped
            before this stage can run, e.g. ["segmentation_status"]
        process_one(recording, file_path, check_fn): run the stage logic
    """

    stage_name: str = ""
    upstream_done_fields: list = []

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process all eligible recordings once, then exit.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            metavar="N",
            help="Maximum recordings to process per cycle (0 = unlimited).",
        )
        parser.add_argument(
            "--retry-failed",
            action="store_true",
            help="Reset failed recordings for this stage back to pending before starting.",
        )
        parser.add_argument(
            "--retry-skipped",
            action="store_true",
            help="Reset skipped recordings for this stage back to pending before starting.",
        )

    def handle(self, *args, **options):
        once = options["once"]
        limit = options["limit"]
        retry_failed = options["retry_failed"]
        retry_skipped = options["retry_skipped"]
        poll_interval = getattr(settings, "ANALYZE_POLL_INTERVAL", 30)

        status_field = f"{self.stage_name}_status"
        error_field = f"{self.stage_name}_error"

        self._running = True

        def shutdown(signum, frame):
            if not self._running:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                self.stdout.write(
                    "\nForce shutdown requested. "
                    "Press Ctrl+C once more to kill immediately."
                )
                return
            self._running = False
            self.stdout.write(
                "\nShutdown signal received — finishing current work. "
                "Press Ctrl+C again to force exit."
            )
            logger.info("Shutdown signal (%s) received for %s.", signum, self.stage_name)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Stale claim recovery: reset any 'running' rows back to 'pending'
        stale_count = Recording.objects.filter(
            **{status_field: "running"}
        ).update(**{status_field: "pending", error_field: ""})
        if stale_count:
            logger.info(
                "Reset %d stale 'running' %s claim(s) to 'pending'.",
                stale_count, self.stage_name,
            )

        # Retry failed if requested
        if retry_failed:
            retry_count = Recording.objects.filter(
                **{status_field: "failed"}
            ).update(**{status_field: "pending", error_field: ""})
            if retry_count:
                logger.info(
                    "Reset %d failed %s recording(s) to 'pending'.",
                    retry_count, self.stage_name,
                )

        # Retry skipped if requested
        if retry_skipped:
            skipped_count = Recording.objects.filter(
                **{status_field: "skipped"}
            ).update(**{status_field: "pending", error_field: ""})
            if skipped_count:
                logger.info(
                    "Reset %d skipped %s recording(s) to 'pending'.",
                    skipped_count, self.stage_name,
                )

        logger.info(
            "%s daemon starting (once=%s, limit=%s, poll=%ss)",
            self.stage_name, once, limit or "unlimited", poll_interval,
        )

        try:
            while self._running:
                processed = self._process_cycle(status_field, error_field, limit)

                if once or not self._running:
                    break

                if not processed:
                    deadline = time.monotonic() + poll_interval
                    while self._running and time.monotonic() < deadline:
                        time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("Force killed by KeyboardInterrupt.")

        logger.info("%s daemon exited.", self.stage_name)

    def _process_cycle(self, status_field, error_field, limit):
        """Find and process eligible recordings. Returns count processed."""
        # Build queryset: this stage pending, upstream stages done/skipped
        filters = {status_field: "pending"}
        for upstream_field in self.upstream_done_fields:
            filters[f"{upstream_field}__in"] = ["done", "skipped"]

        qs = (
            Recording.objects
            .filter(**filters)
            .select_related("stream", "stream__radio", "stream__audio_feed")
            .order_by("start_time")
        )
        if limit:
            qs = qs[:limit]

        recordings = list(qs)
        if recordings:
            logger.info(
                "[%s] Found %d recording(s) to process.",
                self.stage_name, len(recordings),
            )

        processed = 0
        for recording in recordings:
            if not self._running:
                break
            self._process_one_recording(recording, status_field, error_field)
            processed += 1

        return processed

    def _process_one_recording(self, recording, status_field, error_field):
        """Claim, process, and update status for a single recording."""
        # Skip if stage is inactive for this stream
        stream = recording.stream
        if not stream.is_stage_active(self.stage_name):
            logger.info(
                "[%s] %s inactive for stream %s — skipping.",
                recording.id, self.stage_name, stream.name,
            )
            Recording.objects.filter(
                pk=recording.pk, **{status_field: "pending"}
            ).update(**{status_field: "skipped"})
            self._check_all_complete(recording)
            return

        # Optimistic claim: only succeeds if still pending (atomic on SQLite)
        claimed = Recording.objects.filter(
            pk=recording.pk, **{status_field: "pending"}
        ).update(**{status_field: "running"})

        if not claimed:
            return  # another worker got it

        # Set analysis_started_at if this is the first stage to run
        if not recording.analysis_started_at:
            Recording.objects.filter(
                pk=recording.pk, analysis_started_at__isnull=True,
            ).update(analysis_started_at=timezone.now())

        recording.refresh_from_db()
        check = lambda: self._check_shutdown()

        logger.info(
            "[%s] Processing %s for recording %s",
            self.stage_name, self.stage_name, recording.id,
        )

        try:
            # Session recordings (real-time pipeline) have no file —
            # segments carry their own files.  Pass None as file_path.
            if recording.is_session:
                file_path = None
            else:
                if not recording.file or not recording.file.name:
                    raise FileNotFoundError(
                        f"Recording {recording.id} has no file attached."
                    )
                file_path = recording.file.path
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Recording file not found on disk: {file_path}"
                    )

            self.process_one(recording, file_path, check)

            # Success
            Recording.objects.filter(pk=recording.pk).update(
                **{status_field: "done", error_field: ""}
            )
            logger.info("[%s] %s complete.", recording.id, self.stage_name)
            recording.refresh_from_db()
            self._check_all_complete(recording)

        except (_ShutdownRequested, KeyboardInterrupt):
            # Release claim — reset to pending for retry
            Recording.objects.filter(pk=recording.pk).update(
                **{status_field: "pending"}
            )
            logger.info(
                "[%s] Shutdown mid-%s — reset to pending.",
                recording.id, self.stage_name,
            )

        except Exception:
            tb = traceback.format_exc()
            logger.error("[%s] %s failed:\n%s", recording.id, self.stage_name, tb)
            Recording.objects.filter(pk=recording.pk).update(
                **{status_field: "failed", error_field: tb}
            )

    def _check_shutdown(self):
        """Raise _ShutdownRequested if a shutdown has been requested."""
        if not self._running:
            raise _ShutdownRequested()

    def _check_all_complete(self, recording):
        """Set analysis_completed_at if all stages are done/skipped."""
        recording.refresh_from_db()
        stages = [
            recording.segmentation_status,
            recording.fingerprinting_status,
            recording.transcription_status,
            recording.summarization_status,
        ]
        if all(s in ("done", "skipped") for s in stages):
            if not recording.analysis_completed_at:
                Recording.objects.filter(
                    pk=recording.pk, analysis_completed_at__isnull=True,
                ).update(analysis_completed_at=timezone.now())

    def process_one(self, recording, file_path, check_fn):
        """
        Run the stage logic on a single recording.
        Must be overridden by subclasses.

        Args:
            recording: Recording instance (refreshed from DB)
            file_path: absolute path to the audio file
            check_fn: callable that raises _ShutdownRequested on shutdown
        """
        raise NotImplementedError


class SegmentStageCommand(BaseCommand):
    """
    Abstract base for commands that process individual TranscriptionSegments.

    Used by fingerprinting and transcription — stages that operate on
    individual segments rather than whole recordings.

    Subclasses must define:
        stage_name (str): e.g. "fingerprinting" or "transcription"
        segment_types (list[str]): e.g. ["music"] or ["speech", "speech_over_music"]
        process_segment(segment, source_path, start, end, check_fn): run the stage logic
    """

    stage_name: str = ""
    segment_types: list = []

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process all eligible segments once, then exit.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            metavar="N",
            help="Maximum segments to process per cycle (0 = unlimited).",
        )
        parser.add_argument(
            "--retry-failed",
            action="store_true",
            help="Reset failed segments for this stage back to pending before starting.",
        )
        parser.add_argument(
            "--retry-skipped",
            action="store_true",
            help="Reset skipped segments for this stage back to pending before starting.",
        )

    def handle(self, *args, **options):
        once = options["once"]
        limit = options["limit"]
        retry_failed = options["retry_failed"]
        retry_skipped = options["retry_skipped"]
        poll_interval = getattr(settings, "ANALYZE_POLL_INTERVAL", 30)

        status_field = f"{self.stage_name}_status"
        error_field = f"{self.stage_name}_error"

        self._running = True

        def shutdown(signum, frame):
            if not self._running:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                self.stdout.write(
                    "\nForce shutdown requested. "
                    "Press Ctrl+C once more to kill immediately."
                )
                return
            self._running = False
            self.stdout.write(
                "\nShutdown signal received — finishing current work. "
                "Press Ctrl+C again to force exit."
            )
            logger.info("Shutdown signal (%s) received for %s.", signum, self.stage_name)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Stale claim recovery: reset any 'running' segments back to 'pending'
        stale_count = TranscriptionSegment.objects.filter(
            segment_type__in=self.segment_types,
            **{status_field: "running"},
        ).update(**{status_field: "pending", error_field: ""})
        if stale_count:
            logger.info(
                "Reset %d stale 'running' %s segment claim(s) to 'pending'.",
                stale_count, self.stage_name,
            )

        # Retry failed if requested
        if retry_failed:
            retry_count = TranscriptionSegment.objects.filter(
                segment_type__in=self.segment_types,
                **{status_field: "failed"},
            ).update(**{status_field: "pending", error_field: ""})
            if retry_count:
                logger.info(
                    "Reset %d failed %s segment(s) to 'pending'.",
                    retry_count, self.stage_name,
                )

        # Retry skipped if requested
        if retry_skipped:
            skipped_count = TranscriptionSegment.objects.filter(
                segment_type__in=self.segment_types,
                **{status_field: "skipped"},
            ).update(**{status_field: "pending", error_field: ""})
            if skipped_count:
                logger.info(
                    "Reset %d skipped %s segment(s) to 'pending'.",
                    skipped_count, self.stage_name,
                )

        logger.info(
            "%s segment daemon starting (once=%s, limit=%s, poll=%ss)",
            self.stage_name, once, limit or "unlimited", poll_interval,
        )

        try:
            while self._running:
                processed = self._process_cycle(status_field, error_field, limit)

                if once or not self._running:
                    break

                if not processed:
                    deadline = time.monotonic() + poll_interval
                    while self._running and time.monotonic() < deadline:
                        time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("Force killed by KeyboardInterrupt.")

        logger.info("%s segment daemon exited.", self.stage_name)

    def _process_cycle(self, status_field, error_field, limit):
        """Find and process eligible segments. Returns count processed."""
        qs = (
            TranscriptionSegment.objects
            .filter(
                segment_type__in=self.segment_types,
                recording__segmentation_status__in=["done", "skipped"],
                **{status_field: "pending"},
            )
            .select_related(
                "recording", "recording__stream",
                "recording__stream__radio", "recording__stream__audio_feed",
            )
            .order_by("recording__start_time", "start_offset")
        )
        if limit:
            qs = qs[:limit]

        segments = list(qs)
        if segments:
            logger.info(
                "[%s] Found %d segment(s) to process.",
                self.stage_name, len(segments),
            )

        processed = 0
        for segment in segments:
            if not self._running:
                break
            self._process_one_segment(segment, status_field, error_field)
            processed += 1

        return processed

    def _process_one_segment(self, segment, status_field, error_field):
        """Claim, process, and update status for a single segment."""
        stream = segment.recording.stream

        # Skip if stage is inactive for this stream
        if not stream.is_stage_active(self.stage_name):
            logger.info(
                "[seg %s] %s inactive for stream %s — skipping.",
                segment.id, self.stage_name, stream.name,
            )
            TranscriptionSegment.objects.filter(
                pk=segment.pk, **{status_field: "pending"}
            ).update(**{status_field: "skipped"})
            return

        # Optimistic claim: only succeeds if still pending (atomic on SQLite)
        claimed = TranscriptionSegment.objects.filter(
            pk=segment.pk, **{status_field: "pending"}
        ).update(**{status_field: "running"})

        if not claimed:
            return  # another worker got it

        # Set analysis_started_at on the recording if this is the first work
        recording = segment.recording
        if not recording.analysis_started_at:
            Recording.objects.filter(
                pk=recording.pk, analysis_started_at__isnull=True,
            ).update(analysis_started_at=timezone.now())

        segment.refresh_from_db()
        check = lambda: self._check_shutdown()

        logger.info(
            "[seg %s] Processing %s for segment [%.1f-%.1fs] of recording %s",
            segment.id, self.stage_name,
            segment.start_offset, segment.end_offset, recording.id,
        )

        try:
            # Resolve source path and time range
            if segment.file and segment.file.name:
                source_path = segment.file.path
                start = 0
                end = segment.end_offset - segment.start_offset
            elif recording.is_session:
                raise FileNotFoundError(
                    f"Session segment {segment.id} has no file attached."
                )
            else:
                if not recording.file or not recording.file.name:
                    raise FileNotFoundError(
                        f"Recording {recording.id} has no file attached."
                    )
                source_path = recording.file.path
                if not os.path.exists(source_path):
                    raise FileNotFoundError(
                        f"Recording file not found on disk: {source_path}"
                    )
                start = segment.start_offset
                end = segment.end_offset

            self.process_segment(segment, source_path, start, end, check)

            # Success
            TranscriptionSegment.objects.filter(pk=segment.pk).update(
                **{status_field: "done", error_field: ""}
            )
            logger.info("[seg %s] %s complete.", segment.id, self.stage_name)

            # Check if all segments of this recording are done
            self._check_recording_complete(recording)

        except (_ShutdownRequested, KeyboardInterrupt):
            # Release claim — reset to pending for retry
            TranscriptionSegment.objects.filter(pk=segment.pk).update(
                **{status_field: "pending"}
            )
            logger.info(
                "[seg %s] Shutdown mid-%s — reset to pending.",
                segment.id, self.stage_name,
            )

        except Exception:
            tb = traceback.format_exc()
            logger.error("[seg %s] %s failed:\n%s", segment.id, self.stage_name, tb)
            TranscriptionSegment.objects.filter(pk=segment.pk).update(
                **{status_field: "failed", error_field: tb}
            )

    def _check_shutdown(self):
        """Raise _ShutdownRequested if a shutdown has been requested."""
        if not self._running:
            raise _ShutdownRequested()

    def _check_recording_complete(self, recording):
        """Set analysis_completed_at if all stages of the recording are done/skipped."""
        recording.refresh_from_db()
        stages = [
            recording.segmentation_status,
            recording.fingerprinting_status,
            recording.transcription_status,
            recording.summarization_status,
        ]
        if all(s in ("done", "skipped") for s in stages):
            if not recording.analysis_completed_at:
                Recording.objects.filter(
                    pk=recording.pk, analysis_completed_at__isnull=True,
                ).update(analysis_completed_at=timezone.now())

    def process_segment(self, segment, source_path, start, end, check_fn):
        """
        Run the stage logic on a single segment.
        Must be overridden by subclasses.

        Args:
            segment: TranscriptionSegment instance (refreshed from DB)
            source_path: absolute path to the audio file
            start: start offset in seconds within the source file
            end: end offset in seconds within the source file
            check_fn: callable that raises _ShutdownRequested on shutdown
        """
        raise NotImplementedError
