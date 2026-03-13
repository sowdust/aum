"""
record_and_segment — Real-time stream recording with inline segmentation.

Records active audio streams via ffmpeg in PCM mode, classifies audio
in real-time using inaSpeechSegmenter, and stores finalized segments
as individual MP3 files.

This is an alternative to record_streams + segment_recordings for streams
that benefit from real-time processing.  Both modes can coexist — different
streams can use different modes.

Usage:
    python manage.py record_and_segment

Safety:
    - ffmpeg invoked via subprocess with list args (never shell=True)
    - Stream URLs validated to http/https only
    - Global recording flag uses fail-closed semantics
"""

import logging
import signal
import time

from django.core.management.base import BaseCommand

from radios.analysis.recorder import (
    ALLOWED_STREAM_SCHEMES,
    _has_valid_url_scheme,
    _is_recording_enabled,
    _is_within_recording_window,
)
from radios.analysis.stream_processor import StreamProcessor
from radios.models import Stream
from django.db import close_old_connections

logger = logging.getLogger("stream_processor")

POLL_INTERVAL = 5  # seconds between main-loop iterations


class Command(BaseCommand):
    help = (
        "Record active audio streams with real-time segmentation. "
        "Alternative to record_streams + segment_recordings."
    )

    def handle(self, *args, **options):
        logger.info("Real-time stream processor starting up")

        # Reset any stale 'recording' statuses from a previous unclean shutdown
        stale = Stream.objects.filter(recording_status="recording").update(
            recording_status="idle", recording_error="", recording_started_at=None,
        )
        if stale:
            logger.info("Reset %d stale 'recording' stream(s) to idle.", stale)

        processors: dict[int, StreamProcessor] = {}  # stream_id -> StreamProcessor
        running = True

        def on_shutdown(signum, _frame):
            nonlocal running
            logger.info("Shutdown signal received (signal %s)", signum)
            running = False

        signal.signal(signal.SIGINT, on_shutdown)
        signal.signal(signal.SIGTERM, on_shutdown)

        while running:
            try:
                close_old_connections()
                self._sync_processors(processors)
            except Exception:
                logger.exception("Main loop iteration failed")
            time.sleep(POLL_INTERVAL)

        # Graceful shutdown
        logger.info("Shutting down all stream processors...")
        for processor in list(processors.values()):
            try:
                processor.stop()
            except Exception:
                logger.exception("Error stopping processor")
        logger.info("Real-time stream processor exited cleanly")

    def _stop_processor(self, processors, stream_id):
        processor = processors.pop(stream_id, None)
        if processor:
            processor.stop()

    def _sync_processors(self, processors: dict):
        """One main-loop iteration: keep processors in sync with DB state."""
        active_streams = {
            s.id: s
            for s in Stream.objects.filter(is_active=True, enable_recording=True)
                                   .select_related("radio", "audio_feed")
        }

        # Give existing processors a fresh stream reference
        for stream_id, stream in active_streams.items():
            if stream_id in processors:
                processors[stream_id].stream = stream

        # Stop processors whose stream was deactivated or had recording disabled
        for stream_id in list(processors):
            stream = active_streams.get(stream_id)
            if stream is None:
                logger.info("Stream %s deactivated — stopping processor", stream_id)
                self._stop_processor(processors, stream_id)
            elif not _is_recording_enabled(stream):
                logger.info("Recording disabled for '%s' — stopping processor", stream)
                self._stop_processor(processors, stream_id)

        # Start processors for newly eligible streams / tick existing ones
        for stream_id, stream in active_streams.items():
            if stream_id in processors:
                try:
                    processors[stream_id].tick()
                except Exception:
                    logger.exception("Processor for '%s' crashed in tick()", stream)
                    processors.pop(stream_id).stop()
                continue

            source = stream.source
            if source is None:
                logger.warning("Stream '%s' has no source — skipping", stream)
                continue

            if not _has_valid_url_scheme(stream.url, ALLOWED_STREAM_SCHEMES):
                logger.error(
                    "Stream '%s' URL uses a disallowed scheme — skipping",
                    stream,
                )
                continue

            if _is_recording_enabled(stream) and _is_within_recording_window(source):
                logger.info(
                    "Starting real-time processor for '%s' (id=%s)",
                    stream, stream_id,
                )
                try:
                    processor = StreamProcessor(stream)
                    processor.start()
                except Exception:
                    logger.exception("Failed to start processor for '%s'", stream)
                else:
                    processors[stream_id] = processor
            else:
                logger.debug("Stream '%s' not eligible — skipping", stream)
