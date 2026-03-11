"""
record_streams — Continuously record active audio streams via ffmpeg.

Usage:
    python manage.py record_streams

Lifecycle:
    1. Polls the DB every POLL_INTERVAL seconds for active, recording-enabled streams
    2. Spawns one ffmpeg subprocess per eligible stream
    3. Rotates recordings every CHUNK_SIZE seconds (from Django settings)
    4. Saves each finished chunk to Django file storage + creates a Recording row
    5. Shuts down gracefully on SIGINT or SIGTERM

Safety:
    - ffmpeg is invoked via subprocess with a list of args (never shell=True)
    - Stream URLs are validated to http/https only (prevents ffmpeg protocol abuse)
    - Global recording flag uses fail-closed semantics
"""
import datetime
import logging
import os
import signal
import subprocess
import tempfile
import time
import uuid
import zoneinfo
from urllib.parse import urlparse

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from radios.models import GlobalPipelineSettings, Recording, Stream

logger = logging.getLogger("stream_recorder")

POLL_INTERVAL = 5  # seconds between main-loop iterations

ALLOWED_STREAM_SCHEMES = ("http", "https")
ALLOWED_PROXY_SCHEMES = ("http", "https", "socks5")


# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------

def _is_recording_globally_enabled() -> bool:
    """
    Check the global recording kill-switch.
    Fail-closed: if settings cannot be read, returns False.
    """
    try:
        return GlobalPipelineSettings.get_settings().enable_recording
    except Exception:
        logger.error("Cannot read GlobalPipelineSettings — refusing to record (fail-closed)")
        return False


def _is_recording_enabled(stream: Stream) -> bool:
    """True only when recording is enabled both globally AND for this stream."""
    if not _is_recording_globally_enabled():
        return False
    return stream.enable_recording


def _is_within_recording_window(source) -> bool:
    """
    True if the current local time falls inside the source's recording window.
    A window of 0–0 means 24/7.
    """
    start_h = source.recording_start_hour
    end_h = source.recording_end_hour
    if start_h == 0 and end_h == 0:
        return True

    tz = zoneinfo.ZoneInfo(source.timezone)
    current_hour = datetime.datetime.now(tz).hour

    if start_h < end_h:
        return start_h <= current_hour < end_h
    # Overnight window, e.g. 22–06
    return current_hour >= start_h or current_hour < end_h


def _has_valid_url_scheme(url: str, allowed_schemes: tuple) -> bool:
    """True if *url* uses one of *allowed_schemes*."""
    try:
        return urlparse(url).scheme in allowed_schemes
    except Exception:
        return False


def _remove_file(path: str):
    """Best-effort removal of a file."""
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------

class Command(BaseCommand):
    help = "Continuously records active audio streams in time-limited chunks."

    def handle(self, *args, **options):
        logger.info("Recorder service starting up")

        workers = {}   # stream_id -> RecorderWorker
        running = True

        def on_shutdown(signum, _frame):
            nonlocal running
            logger.info("Shutdown signal received (signal %s)", signum)
            running = False

        signal.signal(signal.SIGINT, on_shutdown)
        signal.signal(signal.SIGTERM, on_shutdown)

        while running:
            self._sync_workers(workers)
            time.sleep(POLL_INTERVAL)

        # Graceful shutdown
        logger.info("Shutting down all workers…")
        for worker in workers.values():
            worker.stop()
        logger.info("Recorder service exited cleanly")

    def _sync_workers(self, workers: dict):
        """One main-loop iteration: keep workers in sync with DB state."""
        active_streams = {
            s.id: s
            for s in Stream.objects.filter(is_active=True)
                                   .select_related("radio", "audio_feed")
        }

        # Give existing workers a fresh stream reference (picks up URL changes, etc.)
        for stream_id, stream in active_streams.items():
            if stream_id in workers:
                workers[stream_id].stream = stream

        # Stop workers whose stream was deactivated or had recording disabled
        for stream_id in list(workers):
            stream = active_streams.get(stream_id)
            if stream is None:
                logger.info("Stream %s deactivated — stopping worker", stream_id)
                workers.pop(stream_id).stop()
            elif not _is_recording_enabled(stream):
                logger.info("Recording disabled for '%s' — stopping worker", stream)
                workers.pop(stream_id).stop()

        # Start workers for newly eligible streams
        for stream_id, stream in active_streams.items():
            if stream_id in workers:
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
                logger.info("Starting worker for '%s' (id=%s)", stream, stream_id)
                worker = RecorderWorker(stream)
                worker.start()
                workers[stream_id] = worker
            else:
                logger.debug("Stream '%s' not eligible (window/flags) — skipping", stream)

        # Tick all workers (chunk rotation, crash recovery)
        for worker in list(workers.values()):
            try:
                worker.tick()
            except Exception:
                logger.exception("Worker for '%s' crashed in tick()", worker.stream)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class RecorderWorker:
    """
    One worker per stream.  Manages an ffmpeg subprocess that captures audio
    into a temp file, rotates on the configured chunk boundary, and persists
    each finished chunk to Django storage + the Recording table.
    """

    def __init__(self, stream: Stream):
        self.stream = stream
        self._process = None          # subprocess.Popen
        self._chunk_start = None      # timezone-aware datetime
        self._temp_path = None        # audio temp file
        self._stderr_path = None      # ffmpeg stderr capture file
        self._stderr_fh = None        # open handle kept alive for the child

    # -- public API --------------------------------------------------------

    def start(self):
        logger.info("Worker[%s]: starting first chunk", self.stream)
        self._start_chunk()

    def stop(self):
        logger.info("Worker[%s]: stopping", self.stream)
        self._finalize_chunk()

    def tick(self):
        """
        Called every POLL_INTERVAL seconds.
        Handles ffmpeg crash recovery, recording-window exit, and chunk rotation.
        """
        source = self.stream.source
        in_window = _is_within_recording_window(source) if source else False

        # No running process — restart if we should be recording
        if self._process is None:
            if in_window:
                logger.warning("Worker[%s]: no ffmpeg process — restarting", self.stream)
                self._start_chunk()
            return

        # ffmpeg exited on its own — log stderr so we can see WHY
        if self._process.poll() is not None:
            rc = self._process.returncode
            stderr_tail = self._read_stderr()
            logger.warning(
                "Worker[%s]: ffmpeg exited (rc=%s)%s",
                self.stream, rc,
                f"\n  stderr: {stderr_tail}" if stderr_tail else "",
            )
            self._finalize_chunk()
            if in_window:
                self._start_chunk()
            return

        # Outside recording window — stop gracefully
        if not in_window:
            logger.info("Worker[%s]: outside recording window — finalizing", self.stream)
            self._finalize_chunk()
            return

        # Chunk boundary — rotate
        elapsed = (timezone.now() - self._chunk_start).total_seconds()
        if elapsed >= settings.CHUNK_SIZE:
            logger.info("Worker[%s]: chunk time reached — rotating", self.stream)
            self._finalize_chunk()
            self._start_chunk()

    # -- internals ---------------------------------------------------------

    def _build_ffmpeg_cmd(self, output_path: str) -> list[str]:
        """
        Assemble the ffmpeg argument list.

        Always called via subprocess.Popen with a *list* (never a shell string)
        to prevent command injection.  Stream URLs are pre-validated in
        _sync_workers() to allow only http/https schemes.
        """
        source = self.stream.source
        proxy_url = source.get_effective_proxy_url() if source else ""

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
        ]

        # Proxy — input-level option, must come before -i
        if proxy_url:
            if _has_valid_url_scheme(proxy_url, ALLOWED_PROXY_SCHEMES):
                cmd += ["-http_proxy", proxy_url]
            else:
                logger.warning(
                    "Worker[%s]: proxy URL has disallowed scheme — ignoring",
                    self.stream,
                )

        # Reconnect options — input-level, must come before -i
        cmd += [
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
        ]

        cmd += ["-i", self.stream.url]

        # Output: copy audio into MP3 container.
        # NOTE: -c:a copy only works when the source stream is already MP3.
        # If your source sends AAC/Ogg/etc, change to:  -c:a libmp3lame -b:a 192k
        cmd += [
            "-vn",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-f", "mp3",
            output_path,
        ]

        return cmd

    def _start_chunk(self):
        """Launch ffmpeg writing to a fresh temp file."""
        self._chunk_start = timezone.now()

        self._temp_path = os.path.join(
            tempfile.gettempdir(), f"rec_{uuid.uuid4().hex}.mp3"
        )
        self._stderr_path = os.path.join(
            tempfile.gettempdir(), f"rec_{uuid.uuid4().hex}.log"
        )

        cmd = self._build_ffmpeg_cmd(self._temp_path)

        source = self.stream.source
        proxy_url = source.get_effective_proxy_url() if source else ""
        logger.info(
            "Worker[%s]: ffmpeg -> %s%s",
            self.stream, self._temp_path,
            " (via proxy)" if proxy_url else "",
        )

        try:
            self._stderr_fh = open(self._stderr_path, "w")
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_fh,
            )
        except Exception:
            logger.exception("Worker[%s]: failed to launch ffmpeg", self.stream)
            self._process = None
            self._temp_path = None
            self._chunk_start = None
            self._close_stderr()

    def _read_stderr(self) -> str:
        """Return up to 2 000 chars of ffmpeg's stderr output (for logging)."""
        if not self._stderr_path:
            return ""
        # Flush the parent's handle so the file is up-to-date
        if self._stderr_fh and not self._stderr_fh.closed:
            try:
                self._stderr_fh.flush()
            except OSError:
                pass
        try:
            with open(self._stderr_path, "r") as f:
                return f.read(2000).strip()
        except OSError:
            return ""

    def _close_stderr(self):
        """Close the stderr file handle and remove the temp file."""
        if self._stderr_fh and not self._stderr_fh.closed:
            try:
                self._stderr_fh.close()
            except OSError:
                pass
        self._stderr_fh = None
        if self._stderr_path:
            _remove_file(self._stderr_path)
            self._stderr_path = None

    def _finalize_chunk(self):
        """
        Stop ffmpeg, then save the audio file to Django storage and create
        a Recording row.
        """
        if self._process is None:
            return

        proc = self._process
        temp_path = self._temp_path
        start_time = self._chunk_start

        # Clear state first to prevent double-finalize
        self._process = None
        self._temp_path = None
        self._chunk_start = None

        # Stop ffmpeg
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Worker[%s]: ffmpeg did not exit — sending SIGKILL", self.stream)
                proc.kill()
                proc.wait(timeout=5)

        # Log ffmpeg diagnostics, then clean up the stderr file
        stderr_tail = self._read_stderr()
        if stderr_tail:
            logger.info("Worker[%s]: ffmpeg stderr:\n%s", self.stream, stderr_tail)
        self._close_stderr()

        end_time = timezone.now()

        # Validate the output file
        if not temp_path or not os.path.exists(temp_path):
            logger.warning("Worker[%s]: no audio file for this chunk", self.stream)
            return

        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            logger.warning("Worker[%s]: audio file is empty (0 bytes) — discarding", self.stream)
            _remove_file(temp_path)
            return

        # Build a filesystem-safe filename
        safe_name = "".join(
            c if (c.isalnum() or c in "-_") else "_"
            for c in str(self.stream)
        )
        final_filename = (
            f"{safe_name}_"
            f"{start_time:%Y-%m-%dT%H-%M-%S}_to_"
            f"{end_time:%Y-%m-%dT%H-%M-%S}.mp3"
        )

        logger.info(
            "Worker[%s]: saving %s (%s -> %s, %d bytes)",
            self.stream, final_filename, start_time, end_time, file_size,
        )

        try:
            with transaction.atomic():
                rec = Recording(
                    stream=self.stream,
                    start_time=start_time,
                    end_time=end_time,
                )
                with open(temp_path, "rb") as f:
                    rec.file.save(final_filename, File(f), save=True)
            logger.info("Worker[%s]: saved Recording id=%s", self.stream, rec.id)
        except Exception:
            logger.exception("Worker[%s]: failed to save chunk to DB/storage", self.stream)
        finally:
            _remove_file(temp_path)
