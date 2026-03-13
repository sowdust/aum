"""
StreamRecorder — manages one long-lived ffmpeg process per stream using
the segment muxer for automatic chunk rotation.

Instead of killing/restarting ffmpeg every CHUNK_SIZE seconds, the segment
muxer writes sequential chunk files and emits a CSV line to stdout when
each chunk completes.  A reader thread drains stdout into a queue; the
main thread calls tick() periodically to persist completed chunks.

Security:
    - ffmpeg is invoked via subprocess with a list (never shell=True)
    - Stream URLs are validated to http/https only before reaching here
    - Proxy URLs are validated to http/https/socks5 only
    - Temp directory uses tempfile.mkdtemp()
"""

import datetime
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import zoneinfo
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from django.conf import settings
from django.core.files import File
from django.db import transaction
from django.utils import timezone

from radios.models import GlobalPipelineSettings, Recording, Stream

logger = logging.getLogger("stream_recorder")

ALLOWED_STREAM_SCHEMES = ("http", "https")
ALLOWED_PROXY_SCHEMES = ("http", "https", "socks5")

# Extension → codec mapping for stream-copy detection
_EXT_TO_CODEC = {
    ".mp3": "mp3",
    ".aac": "aac",
    ".ogg": "vorbis",
    ".opus": "opus",
    ".flac": "flac",
    ".m4a": "aac",
}


def _has_valid_url_scheme(url: str, allowed_schemes: tuple) -> bool:
    try:
        return urlparse(url).scheme in allowed_schemes
    except Exception:
        return False


def _remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def _guess_codec_from_url(url: str) -> Optional[str]:
    """Return codec name from URL path extension, or None if ambiguous."""
    ext = Path(urlparse(url).path).suffix.lower()
    return _EXT_TO_CODEC.get(ext)


def _probe_codec(url: str, timeout: int = 10) -> Optional[str]:
    """Use ffprobe to detect the audio codec of a stream URL."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "csv=p=0",
        url,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            codec = result.stdout.strip().split("\n")[0].strip()
            if codec:
                return codec
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def detect_stream_codec(stream: Stream) -> str:
    """
    Detect the audio codec for a stream.  Strategy (cheapest first):
    1. Check the source's stream_codec field (already detected/configured)
    2. Guess from the URL extension
    3. ffprobe the stream
    4. Fall back to empty string (will transcode)

    Persists the result in source.stream_codec for future use.
    """
    source = stream.source
    if source and source.stream_codec:
        return source.stream_codec

    # Try URL extension first
    codec = _guess_codec_from_url(stream.url)

    # Fall back to ffprobe
    if codec is None:
        codec = _probe_codec(stream.url) or ""

    # Persist for future use
    if source and codec:
        source.stream_codec = codec
        source.save(update_fields=["stream_codec"])
        logger.info(
            "Detected codec '%s' for stream '%s' — saved to source.",
            codec, stream,
        )

    return codec or ""


def _is_recording_globally_enabled() -> bool:
    """Fail-closed: if settings cannot be read, returns False."""
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
    Hours are checked against the source's configured timezone.
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


class StreamRecorder:
    """
    Manages one ffmpeg segment-muxer process for a single stream.

    Lifecycle:
        recorder = StreamRecorder(stream)
        recorder.start()
        while running:
            recorder.tick()   # drain completed chunks, save to DB
        recorder.stop()       # terminate ffmpeg, save final chunk
    """

    def __init__(self, stream: Stream):
        self.stream = stream
        self._process = None
        self._reader_thread = None
        self._chunk_queue = queue.Queue()
        self._temp_dir = None
        self._recording_start_wall = None  # wall-clock time when ffmpeg started
        self._stderr_path = None
        self._stderr_fh = None

    @property
    def last_error(self) -> str:
        return self.stream.recording_error

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self):
        """Launch ffmpeg with segment muxer."""
        logger.info("StreamRecorder[%s]: starting", self.stream)
        self._start_ffmpeg()

    def stop(self):
        """Terminate ffmpeg, save final partial chunk, clean up."""
        logger.info("StreamRecorder[%s]: stopping", self.stream)
        self._stop_ffmpeg()
        self._save_remaining_chunks()
        self._cleanup_temp_dir()
        self._update_stream_status("idle", "")

    def tick(self):
        """
        Called every poll interval.  Handles:
        - Draining completed segments from the queue and saving them
        - Detecting ffmpeg crashes and restarting if appropriate
        - Stopping if outside the recording window
        """
        source = self.stream.source
        in_window = _is_within_recording_window(source) if source else False

        # Drain any completed chunks
        self._save_completed_chunks()

        # No running process
        if self._process is None:
            if in_window:
                logger.warning("StreamRecorder[%s]: no ffmpeg — restarting", self.stream)
                self._start_ffmpeg()
            return

        # ffmpeg exited unexpectedly
        if self._process.poll() is not None:
            rc = self._process.returncode
            stderr_tail = self._read_stderr()
            error_msg = f"ffmpeg exited (rc={rc})"
            if stderr_tail:
                error_msg += f": {stderr_tail[:500]}"

            logger.warning("StreamRecorder[%s]: %s", self.stream, error_msg)

            # Save whatever chunks were completed before the crash
            self._save_remaining_chunks()
            self._cleanup_process()

            self._update_stream_status("error", error_msg)

            # Try to restart if still in window
            if in_window:
                logger.info("StreamRecorder[%s]: restarting after crash", self.stream)
                self._start_ffmpeg()
            return

        # Outside recording window — stop gracefully
        if not in_window:
            logger.info("StreamRecorder[%s]: outside recording window — stopping", self.stream)
            self._stop_ffmpeg()
            self._save_remaining_chunks()
            self._cleanup_temp_dir()
            self._update_stream_status("idle", "")

    # -- internal: ffmpeg management ----------------------------------------

    def _build_ffmpeg_cmd(self, output_pattern: str) -> list[str]:
        source = self.stream.source
        proxy_url = source.get_effective_proxy_url() if source else ""

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
        ]

        # Proxy — must come before -i
        if proxy_url:
            if _has_valid_url_scheme(proxy_url, ALLOWED_PROXY_SCHEMES):
                cmd += ["-http_proxy", proxy_url]
            else:
                logger.warning(
                    "StreamRecorder[%s]: proxy URL has disallowed scheme — ignoring",
                    self.stream,
                )

        # Reconnect options — must come before -i
        cmd += [
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
        ]

        cmd += ["-i", self.stream.url]

        # Strip video
        cmd += ["-vn"]

        # Codec selection: stream-copy if source is MP3, else transcode
        codec = detect_stream_codec(self.stream)
        if codec == "mp3":
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]

        # Segment muxer
        chunk_size = getattr(settings, "CHUNK_SIZE", 20 * 60)
        cmd += [
            "-f", "segment",
            "-segment_time", str(chunk_size),
            "-segment_list", "pipe:1",
            "-segment_list_type", "csv",
            "-reset_timestamps", "1",
            "-strftime", "0",
            output_pattern,
        ]

        return cmd

    def _start_ffmpeg(self):
        """Launch the ffmpeg segment muxer process."""
        self._temp_dir = tempfile.mkdtemp(prefix="aum_rec_")
        output_pattern = os.path.join(self._temp_dir, "chunk_%05d.mp3")

        self._stderr_path = os.path.join(self._temp_dir, "ffmpeg_stderr.log")
        self._recording_start_wall = timezone.now()

        cmd = self._build_ffmpeg_cmd(output_pattern)

        source = self.stream.source
        proxy_url = source.get_effective_proxy_url() if source else ""
        logger.info(
            "StreamRecorder[%s]: ffmpeg -> %s%s",
            self.stream, self._temp_dir,
            " (via proxy)" if proxy_url else "",
        )

        try:
            self._stderr_fh = open(self._stderr_path, "w")
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._stderr_fh,
            )
        except Exception as e:
            logger.exception("StreamRecorder[%s]: failed to launch ffmpeg", self.stream)
            self._cleanup_process()
            self._cleanup_temp_dir()
            self._update_stream_status("error", str(e))
            return

        # Start reader thread for stdout (segment list CSV lines)
        self._reader_thread = threading.Thread(
            target=self._stdout_reader,
            daemon=True,
        )
        self._reader_thread.start()

        self._update_stream_status("recording", "")

    def _stdout_reader(self):
        """
        Background thread: read CSV lines from ffmpeg's stdout.
        Each line is like: chunk_00000.mp3,0.000000,1200.000000
        """
        proc = self._process
        try:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._chunk_queue.put(line)
        except (ValueError, OSError):
            pass  # pipe closed

    def _stop_ffmpeg(self):
        """Send SIGTERM to ffmpeg, wait, then SIGKILL if needed."""
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("StreamRecorder[%s]: ffmpeg did not exit — SIGKILL", self.stream)
                self._process.kill()
                self._process.wait(timeout=5)

        # Wait for reader thread to finish draining stdout
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3)

    def _cleanup_process(self):
        """Clean up process and stderr handles without touching temp dir."""
        self._close_stderr()
        self._process = None
        self._reader_thread = None
        self._recording_start_wall = None

    def _cleanup_temp_dir(self):
        """Remove the temp directory and all contents."""
        self._close_stderr()
        self._process = None
        self._reader_thread = None
        self._recording_start_wall = None
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None

    # -- internal: chunk persistence ----------------------------------------

    def _save_completed_chunks(self):
        """Drain the queue and persist each completed chunk."""
        while True:
            try:
                line = self._chunk_queue.get_nowait()
            except queue.Empty:
                break
            self._persist_chunk_from_csv(line)

    def _save_remaining_chunks(self):
        """
        After ffmpeg stops: drain the queue, then scan the temp dir for any
        chunk file not yet persisted (the final partial chunk won't appear in
        the segment list).
        """
        self._save_completed_chunks()

        if not self._temp_dir or not os.path.isdir(self._temp_dir):
            return

        # Collect filenames already saved via the queue
        saved = set()
        # We don't track saved filenames directly, so scan for .mp3 files
        # that still exist (unsaved ones) — but we need to know which were saved.
        # Simpler approach: just try to save any remaining .mp3 file.
        for fname in sorted(os.listdir(self._temp_dir)):
            if not fname.endswith(".mp3"):
                continue
            fpath = os.path.join(self._temp_dir, fname)
            file_size = os.path.getsize(fpath)
            if file_size == 0:
                continue

            # Parse chunk number to estimate time offsets
            chunk_num = self._parse_chunk_number(fname)
            chunk_size = getattr(settings, "CHUNK_SIZE", 20 * 60)

            if self._recording_start_wall:
                start_dt = self._recording_start_wall + datetime.timedelta(
                    seconds=chunk_num * chunk_size
                )
            else:
                start_dt = timezone.now() - datetime.timedelta(seconds=chunk_size)

            end_dt = timezone.now() if chunk_num == self._find_max_chunk_number() else (
                start_dt + datetime.timedelta(seconds=chunk_size)
            )

            self._save_chunk_file(fpath, start_dt, end_dt)

    def _persist_chunk_from_csv(self, csv_line: str):
        """
        Parse a segment list CSV line and save the chunk.
        Format: chunk_00000.mp3,start_seconds,end_seconds
        """
        parts = csv_line.split(",")
        if len(parts) < 3:
            logger.warning("StreamRecorder[%s]: malformed CSV line: %s", self.stream, csv_line)
            return

        filename = parts[0].strip()
        try:
            start_secs = float(parts[1])
            end_secs = float(parts[2])
        except ValueError:
            logger.warning("StreamRecorder[%s]: bad timestamps in CSV: %s", self.stream, csv_line)
            return

        if not self._temp_dir:
            return

        fpath = os.path.join(self._temp_dir, filename)
        if not os.path.exists(fpath):
            logger.warning("StreamRecorder[%s]: chunk file missing: %s", self.stream, fpath)
            return

        file_size = os.path.getsize(fpath)
        if file_size == 0:
            logger.warning("StreamRecorder[%s]: empty chunk — discarding: %s", self.stream, filename)
            _remove_file(fpath)
            return

        # Calculate wall-clock times from recording start + offsets
        if self._recording_start_wall:
            start_dt = self._recording_start_wall + datetime.timedelta(seconds=start_secs)
            end_dt = self._recording_start_wall + datetime.timedelta(seconds=end_secs)
        else:
            duration = end_secs - start_secs
            end_dt = timezone.now()
            start_dt = end_dt - datetime.timedelta(seconds=duration)

        self._save_chunk_file(fpath, start_dt, end_dt)

    def _save_chunk_file(self, fpath: str, start_dt, end_dt):
        """Save a chunk file to Django storage and create a Recording row."""
        file_size = os.path.getsize(fpath)

        safe_name = "".join(
            c if (c.isalnum() or c in "-_") else "_"
            for c in str(self.stream)
        )
        final_filename = (
            f"{safe_name}_"
            f"{start_dt:%Y-%m-%dT%H-%M-%S}_to_"
            f"{end_dt:%Y-%m-%dT%H-%M-%S}.mp3"
        )

        logger.info(
            "StreamRecorder[%s]: saving %s (%s -> %s, %d bytes)",
            self.stream, final_filename, start_dt, end_dt, file_size,
        )

        try:
            with transaction.atomic():
                rec = Recording(
                    stream=self.stream,
                    start_time=start_dt,
                    end_time=end_dt,
                )
                with open(fpath, "rb") as f:
                    rec.file.save(final_filename, File(f), save=True)
            logger.info("StreamRecorder[%s]: saved Recording id=%s", self.stream, rec.id)
        except Exception:
            logger.exception("StreamRecorder[%s]: failed to save chunk to DB/storage", self.stream)
            self._update_stream_status(
                "error", f"Failed to save chunk {os.path.basename(fpath)}"
            )
        finally:
            _remove_file(fpath)

    def _parse_chunk_number(self, filename: str) -> int:
        """Extract chunk number from 'chunk_00003.mp3' -> 3."""
        stem = Path(filename).stem  # 'chunk_00003'
        try:
            return int(stem.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    def _find_max_chunk_number(self) -> int:
        """Find the highest chunk number in the temp dir."""
        if not self._temp_dir or not os.path.isdir(self._temp_dir):
            return 0
        max_num = 0
        for fname in os.listdir(self._temp_dir):
            if fname.endswith(".mp3"):
                max_num = max(max_num, self._parse_chunk_number(fname))
        return max_num

    # -- internal: stderr / status ------------------------------------------

    def _read_stderr(self) -> str:
        """Return up to 2000 chars of ffmpeg's stderr output."""
        if not self._stderr_path:
            return ""
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
        if self._stderr_fh and not self._stderr_fh.closed:
            try:
                self._stderr_fh.close()
            except OSError:
                pass
        self._stderr_fh = None
        self._stderr_path = None

    def _update_stream_status(self, status: str, error: str):
        """Persist recording status to the DB."""
        update_fields = {
            "recording_status": status,
            "recording_error": error,
        }
        if status == "recording":
            update_fields["recording_started_at"] = timezone.now()
        elif status == "idle":
            update_fields["recording_started_at"] = None

        try:
            Stream.objects.filter(pk=self.stream.pk).update(**update_fields)
        except Exception:
            logger.exception(
                "StreamRecorder[%s]: failed to update stream status", self.stream
            )
