"""
Daemon that continuously processes new Recording objects through the
analysis pipeline: segmentation → fingerprinting → transcription.

Usage
-----
    python manage.py analyze_recordings            # run as daemon
    python manage.py analyze_recordings --once     # process all pending, then exit
    python manage.py analyze_recordings --limit 5  # cap recordings per cycle
"""

import signal
import time
import traceback
import logging

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from radios.models import Recording, TranscriptionSegment
from radios.analysis.fingerprinter import fingerprint_segment
from radios.analysis.transcriber import transcribe_segment

logger = logging.getLogger("broadcast_analysis")


class Command(BaseCommand):
    help = "Continuously analyse pending recordings (segmentation + fingerprinting + transcription)."

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

    def handle(self, *args, **options):
        once = options["once"]
        limit = options["limit"]
        poll_interval = getattr(settings, "ANALYZE_POLL_INTERVAL", 30)

        running = True

        def shutdown(signum, frame):
            nonlocal running
            logger.info("Shutdown signal (%s) received, will stop after current recording.", signum)
            running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        logger.info("Analysis daemon starting (once=%s, limit=%s, poll_interval=%ss)",
                    once, limit or "unlimited", poll_interval)

        while running:
            qs = (
                Recording.objects
                .filter(analysis_status="pending")
                .select_related("stream")
                .order_by("start_time")
            )
            if limit:
                qs = qs[:limit]

            recordings = list(qs)

            if recordings:
                logger.info("Found %d pending recording(s) to process.", len(recordings))
            else:
                logger.debug("No pending recordings.")

            for recording in recordings:
                if not running:
                    break
                _process(recording)

            if once or not running:
                break

            time.sleep(poll_interval)

        logger.info("Analysis daemon exited cleanly.")


def _process(recording: Recording) -> None:
    """
    Run segmentation + fingerprinting on a single Recording.

    Sets analysis_status to 'transcribing' while in progress, then 'done'
    on success or 'failed' on any unhandled exception.  Never raises —
    the daemon loop must not crash.
    """
    logger.info("Processing recording %s (%s)", recording.id, recording)

    # Mark as in-progress
    recording.analysis_status = "analysing"
    recording.analysis_started_at = timezone.now()
    recording.analysis_error = ""
    recording.save(update_fields=["analysis_status", "analysis_started_at", "analysis_error"])

    try:
        # Verify file exists on disk
        if not recording.file or not recording.file.name:
            raise FileNotFoundError(f"Recording {recording.id} has no file attached.")
        file_path = recording.file.path
        if not __import__("os").path.exists(file_path):
            raise FileNotFoundError(
                f"Recording file not found on disk: {file_path}"
            )

        stream = recording.stream

        # ----------------------------------------------------------------
        # Stage 1: Segmentation
        # ----------------------------------------------------------------
        if stream.is_stage_active("segmentation"):
            logger.info("[%s] Running segmentation...", recording.id)
            from radios.analysis.segmenter import segment_audio

            audio_segments = segment_audio(file_path)

            # Clear any stale segments from a previous (failed) run
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
                logger.info(
                    "[%s] Segmentation done: %d segments.", recording.id, len(audio_segments)
                )
            else:
                logger.warning("[%s] Segmentation returned no segments.", recording.id)
        else:
            logger.info("[%s] Segmentation stage inactive — skipping.", recording.id)

        # ----------------------------------------------------------------
        # Stage 2: Fingerprinting
        # ----------------------------------------------------------------
        if stream.is_stage_active("fingerprinting"):
            api_key = getattr(settings, "ACOUSTID_API_KEY", "")
            if not api_key:
                logger.warning(
                    "[%s] ACOUSTID_API_KEY is not set — skipping fingerprinting.", recording.id
                )
            else:
                music_segments = list(
                    recording.segments.filter(segment_type="music")
                )
                logger.info(
                    "[%s] Fingerprinting %d music segment(s)...",
                    recording.id, len(music_segments),
                )
                for seg in music_segments:
                    result = fingerprint_segment(
                        file_path,
                        seg.start_offset,
                        seg.end_offset,
                        api_key,
                    )
                    if result:
                        seg.song_title = result.title
                        seg.song_artist = result.artist
                        seg.confidence = result.score
                        seg.save(update_fields=["song_title", "song_artist", "confidence"])
                        logger.info(
                            "[%s] Fingerprinted segment [%.1f-%.1fs]: %s — %s (%.2f)",
                            recording.id, seg.start_offset, seg.end_offset,
                            result.artist, result.title, result.score,
                        )
        else:
            logger.info("[%s] Fingerprinting stage inactive — skipping.", recording.id)

        # ----------------------------------------------------------------
        # Stage 3: Transcription
        # ----------------------------------------------------------------
        if stream.is_stage_active("transcription"):
            recording.analysis_status = "transcribing"
            recording.save(update_fields=["analysis_status"])

            backend = getattr(settings, "TRANSCRIPTION_BACKEND", "local")
            source = stream.source
            language_hint = getattr(source, "languages", "") or ""

            speech_segments = list(
                recording.segments.filter(
                    segment_type__in=["speech", "speech_over_music"]
                )
            )
            logger.info(
                "[%s] Transcribing %d speech segment(s) (backend=%s)...",
                recording.id, len(speech_segments), backend,
            )
            for seg in speech_segments:
                result = transcribe_segment(
                    file_path, seg.start_offset, seg.end_offset,
                    backend, language_hint,
                )
                if result:
                    seg.text = result.text
                    seg.text_english = result.text_english
                    seg.language = result.language
                    seg.confidence = result.confidence
                    seg.save(update_fields=[
                        "text", "text_english", "language", "confidence",
                    ])
                    logger.info(
                        "[%s] Transcribed segment [%.1f-%.1fs]: lang=%s, %d chars",
                        recording.id, seg.start_offset, seg.end_offset,
                        result.language, len(result.text),
                    )

            recording.analysis_status = "transcribed"
            recording.save(update_fields=["analysis_status"])
        else:
            logger.info("[%s] Transcription stage inactive — skipping.", recording.id)

        # ----------------------------------------------------------------
        # Done
        # ----------------------------------------------------------------
        recording.analysis_status = "done"
        recording.analysis_completed_at = timezone.now()
        recording.save(update_fields=["analysis_status", "analysis_completed_at"])
        logger.info("[%s] Analysis complete.", recording.id)

    except Exception:
        tb = traceback.format_exc()
        logger.error("[%s] Analysis failed:\n%s", recording.id, tb)
        recording.analysis_status = "failed"
        recording.analysis_error = tb
        recording.save(update_fields=["analysis_status", "analysis_error"])
