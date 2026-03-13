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
from django.utils import timezone

from radios.models import Recording

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

        # Reset stale 'running' claims for all stages
        for stage in ("segmentation", "fingerprinting", "transcription", "summarization"):
            field = f"{stage}_status"
            error_field = f"{stage}_error"
            stale = Recording.objects.filter(**{field: "running"}).update(
                **{field: "pending", error_field: ""}
            )
            if stale:
                logger.info("Reset %d stale 'running' %s claims.", stale, stage)

        if retry_failed:
            for stage in ("segmentation", "fingerprinting", "transcription", "summarization"):
                field = f"{stage}_status"
                error_field = f"{stage}_error"
                count = Recording.objects.filter(**{field: "failed"}).update(
                    **{field: "pending", error_field: ""}
                )
                if count:
                    logger.info("Reset %d failed %s recording(s).", count, stage)

        logger.info(
            "Analysis wrapper starting (once=%s, limit=%s, poll=%ss)",
            once, limit or "unlimited", poll_interval,
        )

        try:
            while self._running:
                # Find recordings that have any stage still pending
                from django.db.models import Q
                qs = (
                    Recording.objects
                    .filter(
                        Q(segmentation_status="pending") |
                        Q(fingerprinting_status="pending") |
                        Q(transcription_status="pending") |
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

            # Stage 1: Segmentation
            if recording.segmentation_status == "pending":
                if stream.is_stage_active("segmentation"):
                    self._claim_and_run(
                        recording, "segmentation", file_path, check,
                        self._run_segmentation,
                    )
                else:
                    self._skip_stage(recording, "segmentation")

            recording.refresh_from_db()

            # Stage 2: Fingerprinting
            if recording.fingerprinting_status == "pending":
                if stream.is_stage_active("fingerprinting"):
                    self._claim_and_run(
                        recording, "fingerprinting", file_path, check,
                        self._run_fingerprinting,
                    )
                else:
                    self._skip_stage(recording, "fingerprinting")

            # Stage 3: Transcription + correction
            if recording.transcription_status == "pending":
                if stream.is_stage_active("transcription"):
                    self._claim_and_run(
                        recording, "transcription", file_path, check,
                        self._run_transcription,
                    )
                else:
                    self._skip_stage(recording, "transcription")

            recording.refresh_from_db()

            # Stage 4: Summarization
            if recording.summarization_status == "pending":
                if stream.is_stage_active("summarization"):
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

    def _claim_and_run(self, recording, stage, file_path, check, func):
        """Claim a stage, run it, and update status."""
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
        from radios.models import TranscriptionSegment

        check()

        # Session recordings are already segmented inline by StreamProcessor
        if recording.is_session:
            logger.info("[%s] Session recording — already segmented inline.", recording.id)
            return

        logger.info("[%s] Running segmentation...", recording.id)
        audio_segments = segment_audio(file_path)
        recording.segments.all().delete()

        if audio_segments:
            TranscriptionSegment.objects.bulk_create([
                TranscriptionSegment(
                    recording=recording,
                    segment_type=seg.segment_type,
                    start_offset=seg.start,
                    end_offset=seg.end,
                )
                for seg in audio_segments
            ])
            logger.info("[%s] Segmentation done: %d segments.", recording.id, len(audio_segments))
        else:
            logger.warning("[%s] Segmentation returned no segments.", recording.id)

    def _run_fingerprinting(self, recording, file_path, check):
        from radios.models import Song
        from radios.analysis.fingerprinter import fingerprint_segment

        music_segments = list(recording.segments.filter(segment_type="music"))
        logger.info("[%s] Fingerprinting %d music segment(s)...", recording.id, len(music_segments))

        for seg in music_segments:
            check()
            # Prefer per-segment file (real-time pipeline) over recording file + offsets
            if seg.file and seg.file.name:
                source_path = seg.file.path
                start = 0
                end = seg.end_offset - seg.start_offset
            else:
                source_path = file_path
                start = seg.start_offset
                end = seg.end_offset
            result = fingerprint_segment(source_path, start, end)
            if result:
                song = Song.get_or_create_from_fingerprint(result)
                seg.song = song
                seg.confidence = result.score
                seg.save(update_fields=["song", "confidence"])

    def _run_transcription(self, recording, file_path, check):
        from radios.analysis.transcriber import transcribe_segment
        from radios.analysis.corrector import correct_transcription
        from radios.models import TranscriptionSettings

        source = recording.stream.source
        language_hint = getattr(source, "languages", "") or ""

        speech_segments = list(
            recording.segments.filter(segment_type__in=["speech", "speech_over_music"])
        )
        logger.info("[%s] Transcribing %d speech segment(s)...", recording.id, len(speech_segments))

        for seg in speech_segments:
            check()
            # Prefer per-segment file (real-time pipeline) over recording file + offsets
            if seg.file and seg.file.name:
                source_path = seg.file.path
                start = 0
                end = seg.end_offset - seg.start_offset
            else:
                source_path = file_path
                start = seg.start_offset
                end = seg.end_offset
            result = transcribe_segment(source_path, start, end, language_hint=language_hint)
            if result:
                seg.text = result.text
                seg.text_english = result.text_english
                seg.language = result.language
                seg.confidence = result.confidence
                seg.save(update_fields=["text", "text_english", "language", "confidence"])

        # Correction
        cfg = TranscriptionSettings.get_settings()
        if not cfg.enable_correction:
            return

        speech_with_text = list(
            recording.segments
            .filter(segment_type__in=["speech", "speech_over_music"])
            .exclude(text="")
            .order_by("start_offset")
        )
        if not speech_with_text:
            return

        segments_data = [{"index": i, "text": seg.text} for i, seg in enumerate(speech_with_text)]
        radio_name = getattr(source, "name", "")
        city = getattr(source, "city", "")
        country = getattr(source, "country", "")
        radio_location = f"{city}, {country}" if city and country else (city or str(country) if city or country else "")
        radio_language = getattr(source, "languages", "") or ""

        check()
        corrections = correct_transcription(
            segments_data, radio_name=radio_name,
            radio_location=radio_location, radio_language=radio_language,
        )
        if corrections:
            correction_map = {c["index"]: c for c in corrections}
            for i, seg in enumerate(speech_with_text):
                if i in correction_map:
                    c = correction_map[i]
                    seg.text_original = seg.text
                    seg.text = c["text"]
                    seg.text_english = c["text_english"]
                    seg.save(update_fields=["text", "text_original", "text_english"])

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
