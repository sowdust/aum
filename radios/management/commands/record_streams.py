"""
record_streams — Continuously record active audio streams via ffmpeg.

Usage:
    python manage.py record_streams

Thin orchestrator that delegates all ffmpeg/file/DB work to StreamRecorder
(radios/analysis/recorder.py).  This command only handles:
    - Signal handling (SIGINT/SIGTERM)
    - Polling loop to sync active streams with StreamRecorder instances
    - Eligibility checks (global/per-stream enables, recording window, URL scheme)

Safety:
    - ffmpeg is invoked via subprocess with a list of args (never shell=True)
    - Stream URLs are validated to http/https only (prevents ffmpeg protocol abuse)
    - Global recording flag uses fail-closed semantics
"""

import logging
import signal
import time

from django.core.management.base import BaseCommand

from radios.analysis.recorder import (
    ALLOWED_STREAM_SCHEMES,
    StreamRecorder,
    _has_valid_url_scheme,
    _is_recording_enabled,
    _is_within_recording_window,
)
from radios.models import Stream

logger = logging.getLogger("stream_recorder")

POLL_INTERVAL = 5  # seconds between main-loop iterations


class Command(BaseCommand):
    help = "Continuously records active audio streams in time-limited chunks."

    def handle(self, *args, **options):
        logger.info("Recorder service starting up")

        # Reset any stale 'recording' statuses from a previous unclean shutdown
        stale = Stream.objects.filter(recording_status="recording").update(
            recording_status="idle", recording_error="", recording_started_at=None,
        )
        if stale:
            logger.info("Reset %d stale 'recording' stream(s) to idle.", stale)

        recorders = {}  # stream_id -> StreamRecorder
        running = True

        def on_shutdown(signum, _frame):
            nonlocal running
            logger.info("Shutdown signal received (signal %s)", signum)
            running = False

        signal.signal(signal.SIGINT, on_shutdown)
        signal.signal(signal.SIGTERM, on_shutdown)

        while running:
            self._sync_recorders(recorders)
            time.sleep(POLL_INTERVAL)

        # Graceful shutdown
        logger.info("Shutting down all recorders…")
        for recorder in recorders.values():
            recorder.stop()
        logger.info("Recorder service exited cleanly")

    def _sync_recorders(self, recorders: dict):
        """One main-loop iteration: keep recorders in sync with DB state."""
        active_streams = {
            s.id: s
            for s in Stream.objects.filter(is_active=True)
                                   .select_related("radio", "audio_feed")
        }

        # Give existing recorders a fresh stream reference (picks up URL changes, etc.)
        for stream_id, stream in active_streams.items():
            if stream_id in recorders:
                recorders[stream_id].stream = stream

        # Stop recorders whose stream was deactivated or had recording disabled
        for stream_id in list(recorders):
            stream = active_streams.get(stream_id)
            if stream is None:
                logger.info("Stream %s deactivated — stopping recorder", stream_id)
                recorders.pop(stream_id).stop()
            elif not _is_recording_enabled(stream):
                logger.info("Recording disabled for '%s' — stopping recorder", stream)
                recorders.pop(stream_id).stop()

        # Start recorders for newly eligible streams
        for stream_id, stream in active_streams.items():
            if stream_id in recorders:
                # Tick existing recorder (drain chunks, handle crashes)
                try:
                    recorders[stream_id].tick()
                except Exception:
                    logger.exception("Recorder for '%s' crashed in tick()", stream)
                continue
            source = stream.source
            if source is None:
                logger.warning("Stream '%s' has no source (radio/audio_feed) — skipping", stream)
                continue

            if not _has_valid_url_scheme(stream.url, ALLOWED_STREAM_SCHEMES):
                logger.error(
                    "Stream '%s' URL uses a disallowed scheme — skipping (only http/https allowed)",
                    stream,
                )
                continue

            if _is_recording_enabled(stream) and _is_within_recording_window(source):
                logger.info("Starting recorder for '%s' (id=%s)", stream, stream_id)
                recorder = StreamRecorder(stream)
                recorder.start()
                recorders[stream_id] = recorder
            else:
                logger.debug("Stream '%s' not eligible (window/flags) — skipping", stream)
