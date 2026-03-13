"""
Convenience wrapper that runs all analysis stages sequentially for each
recording — same end-to-end behavior as the original monolithic command.

For production, use the individual stage commands as separate daemons:
    python manage.py segment_recordings
    python manage.py fingerprint_recordings
    python manage.py transcribe_recordings
    python manage.py summarize_recordings

This wrapper is useful for development, debugging, and single-recording
processing where running four separate daemons is overkill.

Usage:
    python manage.py analyze_recordings            # run as daemon
    python manage.py analyze_recordings --once     # process all pending, then exit
    python manage.py analyze_recordings --limit 5  # cap recordings per cycle
"""

import os
import signal
import time
import traceback
import logging

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Q, Exists, OuterRef
from django.utils import timezone

from radios.models import Recording, TranscriptionSegment

logger = logging.getLogger("broadcast_analysis")


class _ShutdownRequested(Exception):
    """Raised when a shutdown signal is received during processing."""


class Command(BaseCommand):
    help = (
        "Convenience wrapper: run all analysis stages sequentially per recording. "
        "For production, use the individual stage commands as separate daemons."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process all pending recordings once, then exit.",
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
            help="Reset all failed stages back to pending before starting.",
        )

    def handle(self, *args, **options):
        once = options["once"]
        limit = options["limit"]
        retry_failed = options["retry_failed"]
        poll_interval = getattr(settings, "ANALYZE_POLL_INTERVAL", 30)

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
            logger.info("Shutdown signal (%s) received.", signum)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Reset stale 'running' claims for recording-level stages
        for stage in ("segmentation", "summarization"):
            field = f"{stage}_status"
            error_field = f"{stage}_error"
            stale = Recording.objects.filter(**{field: "running"}).update(
                **{field: "pending", error_field: ""}
            )
            if stale:
                logger.info("Reset %d stale 'running' %s claims.", stale, stage)

        # Reset stale 'running' claims for segment-level stages
        for stage in ("fingerprinting", "transcription"):
            field = f"{stage}_status"
            error_field = f"{stage}_error"
            stale = TranscriptionSegment.objects.filter(
                **{field: "running"}
            ).update(**{field: "pending", error_field: ""})
            if stale:
                logger.info("Reset %d stale 'running' %s segment claims.", stale, stage)

        if retry_failed:
            # Recording-level stages
            for stage in ("segmentation", "summarization"):
                field = f"{stage}_status"
                error_field = f"{stage}_error"
                count = Recording.objects.filter(**{field: "failed"}).update(
                    **{field: "pending", error_field: ""}
                )
                if count:
                    logger.info("Reset %d failed %s recording(s).", count, stage)

            # Segment-level stages
            for stage in ("fingerprinting", "transcription"):
                field = f"{stage}_status"
                error_field = f"{stage}_error"
                count = TranscriptionSegment.objects.filter(
                    **{field: "failed"}
                ).update(**{field: "pending", error_field: ""})
                if count:
                    logger.info("Reset %d failed %s segment(s).", count, stage)

        logger.info(
            "Analysis wrapper starting (once=%s, limit=%s, poll=%ss)",
            once, limit or "unlimited", poll_interval,
        )

        try:
            while self._running:
                # Find recordings that have any stage still pending
                # For segment-level stages, check if any child segments are pending
                has_pending_fp = Exists(
                    TranscriptionSegment.objects.filter(
                        recording=OuterRef("pk"),
                        fingerprinting_status="pending",
                    )
                )
                has_pending_tx = Exists(
                    TranscriptionSegment.objects.filter(
                        recording=OuterRef("pk"),
                        transcription_status="pending",
                    )
                )

                qs = (
                    Recording.objects
                    .annotate(
                        _has_pending_fp=has_pending_fp,
                        _has_pending_tx=has_pending_tx,
                    )
                    .filter(
                        Q(segmentation_status="pending") |
                        Q(_has_pending_fp=True) |
                        Q(_has_pending_tx=True) |
                        Q(summarization_status="pending")
                    )
                    .select_related("stream", "stream__radio", "stream__audio_feed")
                    .order_by("start_time")
                )
                if limit:
                    qs = qs[:limit]

                recordings = list(qs)
                if recordings:
                    logger.info("Found %d recording(s) to process.", len(recordings))

                for recording in recordings:
                    if not self._running:
                        break
                    self._process(recording)

                if once or not self._running:
                    break

                deadline = time.monotonic() + poll_interval
                while self._running and time.monotonic() < deadline:
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.warning("Force killed by KeyboardInterrupt.")

        logger.info("Analysis wrapper exited.")

    def _check(self):
        if not self._running:
            raise _ShutdownRequested()

    def _process(self, recording):
        """Run all stages sequentially on a single recording."""
        check = self._check

        logger.info("Processing recording %s (%s)", recording.id, recording)

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

            stream = recording.stream

            # Stage 1: Segmentation (recording-level)
            if recording.segmentation_status == "pending":
                if stream.is_stage_active("segmentation"):
                    self._claim_and_run(
                        recording, "segmentation", file_path, check,
                        self._run_segmentation,
                    )
                else:
                    self._skip_stage(recording, "segmentation")

            recording.refresh_from_db()

            # Stage 2: Fingerprinting (segment-level)
            self._run_segment_stage(
                recording, file_path, check,
                stage_name="fingerprinting",
                segment_types=["music"],
                process_fn=self._run_fingerprinting_segment,
            )

            # Stage 3: Transcription + correction (segment-level)
            self._run_segment_stage(
                recording, file_path, check,
                stage_name="transcription",
                segment_types=["speech", "speech_over_music"],
                process_fn=self._run_transcription_segment,
            )

            recording.refresh_from_db()

            # Stage 4: Summarization (recording-level)
            if recording.summarization_status == "pending":
                # Check that all speech segments are transcribed
                has_pending = recording.segments.filter(
                    segment_type__in=["speech", "speech_over_music"],
                    transcription_status__in=["pending", "running"],
                ).exists()
                if has_pending:
                    logger.info(
                        "[%s] Skipping summarization — transcription not complete.",
                        recording.id,
                    )
                elif stream.is_stage_active("summarization"):
                    self._claim_and_run(
                        recording, "summarization", file_path, check,
                        self._run_summarization,
                    )
                else:
                    self._skip_stage(recording, "summarization")

            recording.refresh_from_db()
            self._check_all_complete(recording)
            logger.info("[%s] Analysis complete — status: %s", recording.id, recording.analysis_status)

        except (_ShutdownRequested, KeyboardInterrupt):
            logger.info("[%s] Shutdown mid-analysis.", recording.id)

        except Exception:
            tb = traceback.format_exc()
            logger.error("[%s] Analysis failed:\n%s", recording.id, tb)

    def _run_segment_stage(self, recording, file_path, check, stage_name, segment_types, process_fn):
        """Process all pending segments of a given type for a recording."""
        status_field = f"{stage_name}_status"
        error_field = f"{stage_name}_error"
        stream = recording.stream

        pending_segments = list(
            recording.segments.filter(
                segment_type__in=segment_types,
                **{status_field: "pending"},
            )
        )

        for seg in pending_segments:
            if not self._running:
                break

            if not stream.is_stage_active(stage_name):
                TranscriptionSegment.objects.filter(
                    pk=seg.pk, **{status_field: "pending"}
                ).update(**{status_field: "skipped"})
                continue

            # Optimistic claim
            claimed = TranscriptionSegment.objects.filter(
                pk=seg.pk, **{status_field: "pending"}
            ).update(**{status_field: "running"})
            if not claimed:
                continue

            seg.refresh_from_db()

            try:
                # Resolve source path
                if seg.file and seg.file.name:
                    source_path = seg.file.path
                    start = 0
                    end = seg.end_offset - seg.start_offset
                else:
                    source_path = file_path
                    start = seg.start_offset
                    end = seg.end_offset

                process_fn(seg, source_path, start, end, check)

                TranscriptionSegment.objects.filter(pk=seg.pk).update(
                    **{status_field: "done", error_field: ""}
                )

            except (_ShutdownRequested, KeyboardInterrupt):
                TranscriptionSegment.objects.filter(pk=seg.pk).update(
                    **{status_field: "pending"}
                )
                raise

            except Exception:
                tb = traceback.format_exc()
                logger.error("[seg %s] %s failed:\n%s", seg.id, stage_name, tb)
                TranscriptionSegment.objects.filter(pk=seg.pk).update(
                    **{status_field: "failed", error_field: tb}
                )

    def _claim_and_run(self, recording, stage, file_path, check, func):
        """Claim a recording-level stage, run it, and update status."""
        status_field = f"{stage}_status"
        error_field = f"{stage}_error"

        claimed = Recording.objects.filter(
            pk=recording.pk, **{status_field: "pending"}
        ).update(**{status_field: "running"})
        if not claimed:
            return

        if not recording.analysis_started_at:
            Recording.objects.filter(
                pk=recording.pk, analysis_started_at__isnull=True,
            ).update(analysis_started_at=timezone.now())

        recording.refresh_from_db()

        try:
            func(recording, file_path, check)
            Recording.objects.filter(pk=recording.pk).update(
                **{status_field: "done", error_field: ""}
            )
        except (_ShutdownRequested, KeyboardInterrupt):
            Recording.objects.filter(pk=recording.pk).update(
                **{status_field: "pending"}
            )
            raise
        except Exception:
            tb = traceback.format_exc()
            logger.error("[%s] %s failed:\n%s", recording.id, stage, tb)
            Recording.objects.filter(pk=recording.pk).update(
                **{status_field: "failed", error_field: tb}
            )

    def _skip_stage(self, recording, stage):
        status_field = f"{stage}_status"
        Recording.objects.filter(
            pk=recording.pk, **{status_field: "pending"}
        ).update(**{status_field: "skipped"})

    def _check_all_complete(self, recording):
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

    # --- Stage implementations (reuse existing analysis modules) ---

    def _run_segmentation(self, recording, file_path, check):
        from radios.analysis.segmenter import segment_audio
        from radios.models import TranscriptionSettings
        from radios.management.commands.segment_recordings import _initial_segment_statuses

        check()

        # Session recordings are already segmented inline by StreamProcessor
        if recording.is_session:
            logger.info("[%s] Session recording — already segmented inline.", recording.id)
            return

        logger.info("[%s] Running segmentation...", recording.id)
        audio_segments = segment_audio(file_path)
        recording.segments.all().delete()

        stream = recording.stream

        if audio_segments:
            TranscriptionSegment.objects.bulk_create([
                TranscriptionSegment(
                    recording=recording,
                    segment_type=seg.segment_type,
                    start_offset=seg.start,
                    end_offset=seg.end,
                    **_initial_segment_statuses(seg.segment_type, stream),
                )
                for seg in audio_segments
            ])
            logger.info("[%s] Segmentation done: %d segments.", recording.id, len(audio_segments))
        else:
            logger.warning("[%s] Segmentation returned no segments.", recording.id)

    def _run_fingerprinting_segment(self, segment, source_path, start, end, check):
        from radios.models import Song
        from radios.analysis.fingerprinter import fingerprint_segment_sliding
        from django.db import transaction

        duration = end - start
        if duration <= 5:
            return

        check()
        results = fingerprint_segment_sliding(source_path, start, end)

        with transaction.atomic():
            segment.song_occurrences.all().delete()
            for result in results:
                from radios.models import SongOccurrence
                song = Song.get_or_create_from_fingerprint(result)
                SongOccurrence.objects.create(
                    segment=segment,
                    song=song,
                    start_offset=result.estimated_start,
                    end_offset=result.estimated_end,
                    confidence=result.score,
                )

    def _run_transcription_segment(self, segment, source_path, start, end, check):
        from radios.analysis.transcriber import transcribe_segment

        source = segment.recording.stream.source
        language_hint = getattr(source, "languages", "") or ""

        check()
        result = transcribe_segment(source_path, start, end, language_hint=language_hint)
        if result:
            segment.text = result.text
            segment.text_english = result.text_english
            segment.language = result.language
            segment.confidence = result.confidence
            segment.save(update_fields=["text", "text_english", "language", "confidence"])

    def _run_summarization(self, recording, file_path, check):
        from django.db.models import Q
        from radios.models import ChunkSummary, Tag
        from radios.analysis.summarizer import summarize_texts

        source = recording.stream.source
        language_hint = getattr(source, "languages", "") or ""

        texts = list(
            recording.segments
            .filter(segment_type__in=["speech", "speech_over_music"])
            .exclude(Q(text="") | Q(text__isnull=True))
            .values_list("text", flat=True)
        )
        if not texts:
            logger.info("[%s] No transcribed text to summarize.", recording.id)
            return

        check()
        logger.info("[%s] Summarizing %d text segment(s)...", recording.id, len(texts))
        result = summarize_texts(texts, language_hint=language_hint)
        if not result:
            return

        chunk_summary, created = ChunkSummary.objects.update_or_create(
            recording=recording,
            defaults={"summary_text": result.summary_text},
        )
        tag_objects = [Tag.get_or_create_normalized(name)[0] for name in result.tags]
        chunk_summary.tags.set(tag_objects)
