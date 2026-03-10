"""
Daemon that continuously processes new Recording objects through the
analysis pipeline:

    segmentation → fingerprinting + transcription (parallel) → summarization

Each stage advances the recording's ``analysis_status`` only **after** the
stage completes successfully.  If the process is interrupted at any point,
the recording keeps its last completed status and that stage is simply
re-run on the next invocation — no work is lost and no stage is repeated.

Status milestones
-----------------
    pending → segmented → transcribed → done
                                         ↘ failed

Usage
-----
    python manage.py analyze_recordings            # run as daemon
    python manage.py analyze_recordings --once     # process all pending, then exit
    python manage.py analyze_recordings --limit 5  # cap recordings per cycle
"""

import os
import signal
import time
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import close_old_connections
from django.db.models import Q
from django.utils import timezone

from radios.models import Recording, TranscriptionSegment, TranscriptionSettings, ChunkSummary, Tag
from radios.analysis.fingerprinter import fingerprint_segment
from radios.analysis.transcriber import transcribe_segment
from radios.analysis.corrector import correct_transcription
from radios.analysis.summarizer import summarize_texts

logger = logging.getLogger("broadcast_analysis")

# Recordings in any of these statuses still have work to do.
INCOMPLETE_STATUSES = ["pending", "segmented", "transcribed"]


class _ShutdownRequested(Exception):
    """Raised when a shutdown signal is received during processing."""


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------

class Command(BaseCommand):
    help = (
        "Continuously analyse pending recordings "
        "(segmentation → fingerprinting + transcription → summarization)."
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

    def handle(self, *args, **options):
        once = options["once"]
        limit = options["limit"]
        poll_interval = getattr(settings, "ANALYZE_POLL_INTERVAL", 30)

        running = True

        def shutdown(signum, frame):
            nonlocal running
            if not running:
                # Second signal — restore default handlers so the next
                # Ctrl+C kills the process immediately.
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                self.stdout.write(
                    "\nForce shutdown requested. "
                    "Press Ctrl+C once more to kill immediately."
                )
                logger.warning(
                    "Second shutdown signal (%s) — default handler restored.",
                    signum,
                )
                return
            running = False
            self.stdout.write(
                "\nShutdown signal received — finishing current work. "
                "Press Ctrl+C again to force exit."
            )
            logger.info("Shutdown signal (%s) received.", signum)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        logger.info(
            "Analysis daemon starting (once=%s, limit=%s, poll_interval=%ss)",
            once, limit or "unlimited", poll_interval,
        )

        try:
            while running:
                qs = (
                    Recording.objects
                    .filter(analysis_status__in=INCOMPLETE_STATUSES)
                    .select_related("stream", "stream__radio")
                    .order_by("start_time")
                )
                if limit:
                    qs = qs[:limit]

                recordings = list(qs)

                if recordings:
                    logger.info(
                        "Found %d recording(s) to process.", len(recordings)
                    )
                else:
                    logger.debug("No recordings to process.")

                for recording in recordings:
                    if not running:
                        break
                    _process(recording, should_stop=lambda: not running)

                if once or not running:
                    break

                # Sleep in 1-second steps so Ctrl+C is noticed promptly.
                # time.sleep() restarts after signal handlers (PEP 475),
                # so a single long sleep would block shutdown for the full
                # interval.
                deadline = time.monotonic() + poll_interval
                while running and time.monotonic() < deadline:
                    time.sleep(1)

        except KeyboardInterrupt:
            # Reached when the default handler is restored (second Ctrl+C)
            # and the user presses Ctrl+C a third time.
            logger.warning("Force killed by KeyboardInterrupt.")

        logger.info("Analysis daemon exited.")


# ---------------------------------------------------------------------------
# Pipeline stages — each handles one step of the analysis
# ---------------------------------------------------------------------------

def _check_shutdown(should_stop):
    """Raise _ShutdownRequested if a shutdown has been requested."""
    if should_stop():
        raise _ShutdownRequested()


def _run_segmentation(recording, file_path, check):
    """Stage 1: Split audio into speech / music / noise segments."""
    from radios.analysis.segmenter import segment_audio

    # Check before starting — segmentation runs a TF model in C and
    # cannot be interrupted once underway.
    check()
    logger.info("[%s] Running segmentation...", recording.id)

    audio_segments = segment_audio(file_path)

    # Clear stale segments from a previous (interrupted) run
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
            "[%s] Segmentation done: %d segments.",
            recording.id, len(audio_segments),
        )
    else:
        logger.warning("[%s] Segmentation returned no segments.", recording.id)


def _run_fingerprinting(recording, file_path, check):
    """Stage 2: Identify songs in music segments via AcoustID."""
    api_key = getattr(settings, "ACOUSTID_API_KEY", "")
    if not api_key:
        logger.warning(
            "[%s] ACOUSTID_API_KEY is not set — skipping fingerprinting.",
            recording.id,
        )
        return

    music_segments = list(recording.segments.filter(segment_type="music"))
    logger.info(
        "[%s] Fingerprinting %d music segment(s)...",
        recording.id, len(music_segments),
    )

    for seg in music_segments:
        check()
        result = fingerprint_segment(
            file_path, seg.start_offset, seg.end_offset, api_key,
        )
        if result:
            seg.song_title = result.title
            seg.song_artist = result.artist
            seg.confidence = result.score
            seg.save(update_fields=["song_title", "song_artist", "confidence"])
            logger.info(
                "[%s] Fingerprinted [%.1f-%.1fs]: %s — %s (%.2f)",
                recording.id, seg.start_offset, seg.end_offset,
                result.artist, result.title, result.score,
            )


def _run_transcription(recording, file_path, check):
    """Stage 3: Transcribe speech segments to text."""
    radio = recording.stream.radio
    language_hint = getattr(radio, "languages", "") or ""

    speech_segments = list(
        recording.segments.filter(
            segment_type__in=["speech", "speech_over_music"]
        )
    )
    logger.info(
        "[%s] Transcribing %d speech segment(s)...",
        recording.id, len(speech_segments),
    )

    for seg in speech_segments:
        check()
        result = transcribe_segment(
            file_path, seg.start_offset, seg.end_offset,
            language_hint=language_hint,
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
                "[%s] Transcribed [%.1f-%.1fs]: lang=%s, %d chars",
                recording.id, seg.start_offset, seg.end_offset,
                result.language, len(result.text),
            )


def _run_correction(recording, check):
    """Stage 3b: LLM correction of transcription errors + re-translation."""
    cfg = TranscriptionSettings.get_settings()
    if not cfg.enable_correction:
        return

    speech_segments = list(
        recording.segments
        .filter(segment_type__in=["speech", "speech_over_music"])
        .exclude(text="")
        .order_by("start_offset")
    )

    if not speech_segments:
        logger.info("[%s] No speech text to correct.", recording.id)
        return

    segments_data = [
        {"index": i, "text": seg.text}
        for i, seg in enumerate(speech_segments)
    ]

    source = recording.stream.source
    radio_name = getattr(source, "name", "")
    radio_location = ""
    city = getattr(source, "city", "")
    country = getattr(source, "country", "")
    if city and country:
        radio_location = f"{city}, {country}"
    elif city or country:
        radio_location = city or str(country)
    radio_language = getattr(source, "languages", "") or ""

    check()
    logger.info("[%s] Running transcription correction on %d segment(s)...",
                recording.id, len(speech_segments))

    corrections = correct_transcription(
        segments_data,
        radio_name=radio_name,
        radio_location=radio_location,
        radio_language=radio_language,
    )

    if not corrections:
        logger.warning("[%s] Correction returned no results.", recording.id)
        return

    correction_map = {c["index"]: c for c in corrections}

    corrected_count = 0
    for i, seg in enumerate(speech_segments):
        if i in correction_map:
            c = correction_map[i]
            seg.text_original = seg.text
            seg.text = c["text"]
            seg.text_english = c["text_english"]
            seg.save(update_fields=["text", "text_original", "text_english"])
            corrected_count += 1

    logger.info("[%s] Corrected %d/%d speech segment(s).",
                recording.id, corrected_count, len(speech_segments))


def _run_summarization(recording, check):
    """Stage 4: Summarize transcribed speech into a ChunkSummary with tags."""
    radio = recording.stream.radio
    language_hint = getattr(radio, "languages", "") or ""

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
        logger.warning("[%s] Summarizer returned no result.", recording.id)
        return

    chunk_summary, created = ChunkSummary.objects.update_or_create(
        recording=recording,
        defaults={"summary_text": result.summary_text},
    )

    tag_objects = [Tag.get_or_create_normalized(name)[0] for name in result.tags]
    chunk_summary.tags.set(tag_objects)

    action = "Created" if created else "Updated"
    logger.info(
        "[%s] %s chunk summary (%d chars, %d tags).",
        recording.id, action, len(result.summary_text), len(tag_objects),
    )


# ---------------------------------------------------------------------------
# Parallel execution of fingerprinting + transcription
# ---------------------------------------------------------------------------

def _run_in_thread(func, *args):
    """
    Wrapper for running a pipeline stage in a thread.

    Django does not automatically manage DB connections for manually spawned
    threads, so we close stale connections before and after the work to
    prevent "connection already closed" errors or leaked connections.
    """
    try:
        close_old_connections()
        return func(*args)
    finally:
        close_old_connections()


def _run_parallel_fp_and_tx(recording, file_path, check, fp_active, tx_active):
    """
    Run fingerprinting and transcription concurrently.

    Both stages are I/O-bound (API calls, subprocess invocations), so
    threading provides real parallelism without GIL contention.

    If either stage raises _ShutdownRequested, it is re-raised in the
    calling thread after both futures settle.  Other exceptions are also
    propagated (shutdown takes priority).
    """
    futures = {}

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="analysis") as pool:
        if fp_active:
            futures["fingerprinting"] = pool.submit(
                _run_in_thread, _run_fingerprinting, recording, file_path, check,
            )
        if tx_active:
            futures["transcription"] = pool.submit(
                _run_in_thread, _run_transcription, recording, file_path, check,
            )

        shutdown_requested = False
        first_error = None

        for stage_name, future in futures.items():
            try:
                future.result()
            except (_ShutdownRequested, KeyboardInterrupt):
                shutdown_requested = True
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.error(
                    "[%s] %s failed: %s", recording.id, stage_name, exc,
                )

    # Shutdown takes priority — discards all partial work upstream
    if shutdown_requested:
        raise _ShutdownRequested()
    if first_error is not None:
        raise first_error


# ---------------------------------------------------------------------------
# Main per-recording processing function
# ---------------------------------------------------------------------------

def _process(recording: Recording, should_stop) -> None:
    """
    Run the analysis pipeline on a single Recording, resuming from
    whatever stage it last completed.

    Status is a completed-milestone marker — it is only advanced *after*
    a stage finishes successfully:

        pending  →  segmented  →  transcribed  →  done

    If the daemon is interrupted mid-stage, the status stays where it was
    and the stage is simply re-run next time.  No partial work is lost
    and no completed stage is repeated.

    On unhandled error the status is set to 'failed' with the traceback.

    Never raises — the daemon loop must not crash.
    """
    check = lambda: _check_shutdown(should_stop)
    status = recording.analysis_status

    logger.info(
        "Processing recording %s (%s) — resuming from '%s'",
        recording.id, recording, status,
    )

    try:
        # -- Verify the audio file exists on disk --------------------------
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

        # ------------------------------------------------------------------
        # Stage 1: Segmentation  (pending → segmented)
        # ------------------------------------------------------------------
        if status == "pending":
            if stream.is_stage_active("segmentation"):
                _run_segmentation(recording, file_path, check)
            else:
                logger.info(
                    "[%s] Segmentation stage inactive — skipping.",
                    recording.id,
                )

            recording.analysis_status = "segmented"
            recording.analysis_started_at = timezone.now()
            recording.save(update_fields=[
                "analysis_status", "analysis_started_at",
            ])
            status = "segmented"

        # ------------------------------------------------------------------
        # Stages 2 & 3: Fingerprinting + Transcription  (segmented → transcribed)
        # ------------------------------------------------------------------
        if status == "segmented":
            fp_active = stream.is_stage_active("fingerprinting")
            tx_active = stream.is_stage_active("transcription")

            if fp_active or tx_active:
                _run_parallel_fp_and_tx(
                    recording, file_path, check, fp_active, tx_active,
                )
            else:
                logger.info(
                    "[%s] Fingerprinting and transcription inactive — skipping.",
                    recording.id,
                )

            if tx_active:
                _run_correction(recording, check)

            recording.analysis_status = "transcribed"
            recording.save(update_fields=["analysis_status"])
            status = "transcribed"

        # ------------------------------------------------------------------
        # Stage 4: Summarization  (transcribed → done)
        # ------------------------------------------------------------------
        if status == "transcribed":
            if stream.is_stage_active("summarization"):
                _run_summarization(recording, check)
            else:
                logger.info(
                    "[%s] Summarization stage inactive — skipping.",
                    recording.id,
                )

            recording.analysis_status = "done"
            recording.analysis_completed_at = timezone.now()
            recording.save(update_fields=[
                "analysis_status", "analysis_completed_at",
            ])

        logger.info("[%s] Analysis complete.", recording.id)

    except (_ShutdownRequested, KeyboardInterrupt):
        logger.info(
            "[%s] Shutdown mid-analysis — keeping status '%s' for retry.",
            recording.id, recording.analysis_status,
        )

    except Exception:
        tb = traceback.format_exc()
        logger.error("[%s] Analysis failed:\n%s", recording.id, tb)
        recording.analysis_status = "failed"
        recording.analysis_error = tb
        recording.save(update_fields=["analysis_status", "analysis_error"])
