"""
Real-time stream processing pipeline.

Replaces the "record then segment" approach with "stream → classify in
real-time → store finalized segments as MP3".

Architecture:
    ffmpeg (stream URL) → mono 16kHz PCM stdout → StreamProcessor
                                                       │
                                             RollingPCMBuffer (last 5 min)
                                                       │
                                             Classifier (10s micro-batches)
                                                       │
                                             SegmentStateMachine
                                                       │
                                             on_segment_finalized()
                                                       │
                                             encode MP3 + save to DB

Components:
    RollingPCMBuffer       — ring buffer of mono 16kHz int16 samples
    SegmentStateMachine    — tracks content type changes, delayed finalization
    StreamSegmentEncoder   — extracts PCM, encodes MP3, saves to Django storage
    StreamProcessor        — per-stream orchestrator (ffmpeg + threads)

Security:
    - ffmpeg invoked via subprocess with list args (never shell=True)
    - Stream URLs validated to http/https before reaching here
    - Temp files use /dev/shm for classifier (RAM-backed, auto-cleaned)
    - Proxy URLs validated to allowed schemes only
"""

import datetime
import logging
import os
import struct
import subprocess
import tempfile
import threading
import time
from typing import List, Optional
from urllib.parse import urlparse

import numpy as np

from django.conf import settings
from django.core.files import File
from django.db import transaction
from django.utils import timezone

from radios.models import Recording, Stream, TranscriptionSegment
from radios.analysis.segmenter import (
    SAMPLE_RATE,
    _compute_energy_db,
    _get_segmenter,
    _refine_boundaries,
    _spectral_flux,
    AudioSegment,
    REFINE_SEARCH_RADIUS,
)
from radios.analysis.recorder import (
    ALLOWED_STREAM_SCHEMES,
    ALLOWED_PROXY_SCHEMES,
    _has_valid_url_scheme,
)

logger = logging.getLogger("stream_processor")

# --- Configuration -----------------------------------------------------------

CLASSIFY_INTERVAL = 10.0     # seconds between classifier runs
BUFFER_DURATION = 18_000.0   # seconds of PCM to keep in ring buffer (30 min)
READ_CHUNK_SAMPLES = 8000    # 0.5s at 16kHz — PCM read chunk size
SEGMENT_MIN_DURATION = 10.0  # minimum segment duration before finalization (s)
STABILITY_WINDOW = 10.0      # seconds of stable label before finalizing
BYTES_PER_SAMPLE = 2         # int16
MAX_SEGMENT_DURATION = 900.0 # 15 minutes

# Labels from inaSpeechSegmenter that we store (others are discarded)
STORED_LABELS = {"speech", "music"}


# =============================================================================
# RollingPCMBuffer
# =============================================================================

class RollingPCMBuffer:
    """
    Thread-safe ring buffer of mono 16kHz int16 PCM samples.

    Tracks wall-clock timestamps so samples can be extracted by time range.
    Capacity is ~10 MB/min at 16kHz mono int16.
    """

    def __init__(self, max_duration: float = BUFFER_DURATION):
        self._max_samples = int(max_duration * SAMPLE_RATE)
        self._buffer = np.zeros(self._max_samples, dtype=np.int16)
        self._write_pos = 0        # next write position (wraps around)
        self._total_written = 0    # total samples ever written (monotonic)
        self._start_wall = None    # wall-clock time of sample 0
        self._lock = threading.RLock()  # RLock: re-entrant (get_range_float32 calls _available_samples under the same lock)

    @property
    def total_duration(self) -> float:
        """Total seconds of audio received since start."""
        with self._lock:
            return self._total_written / SAMPLE_RATE

    def append(self, samples: np.ndarray):
        """Append int16 PCM samples to the buffer."""
        with self._lock:
            if self._start_wall is None:
                self._start_wall = timezone.now()

            n = len(samples)
            if n == 0:
                return

            if n >= self._max_samples:
                # More data than buffer can hold — keep only the tail
                samples = samples[-self._max_samples:]
                n = self._max_samples
                self._buffer[:] = samples
                self._write_pos = 0
                self._total_written += n
                return

            # Write in up to two chunks (wrap-around)
            end_pos = self._write_pos + n
            if end_pos <= self._max_samples:
                self._buffer[self._write_pos:end_pos] = samples
            else:
                first = self._max_samples - self._write_pos
                self._buffer[self._write_pos:] = samples[:first]
                self._buffer[:n - first] = samples[first:]

            self._write_pos = end_pos % self._max_samples
            self._total_written += n

    def get_tail(self, duration: float) -> Optional[np.ndarray]:
        """Return the last *duration* seconds of PCM as float32 samples."""
        n_samples = min(int(duration * SAMPLE_RATE), self._available_samples())
        if n_samples <= 0:
            return None
        return self._extract_last_n(n_samples)

    def get_range_float32(self, start_sec: float, end_sec: float) -> Optional[np.ndarray]:
        """
        Extract PCM for [start_sec, end_sec] relative to stream start.
        Returns float32 normalized samples, or None if range is unavailable.
        """
        with self._lock:
            avail = self._available_samples()
            if avail == 0:
                return None

            start_sample = int(start_sec * SAMPLE_RATE)
            end_sample = int(end_sec * SAMPLE_RATE)

            # Oldest available sample index
            oldest = self._total_written - avail
            if start_sample < oldest or end_sample > self._total_written:
                return None

            # Convert to buffer-relative positions
            offset_start = start_sample - oldest
            offset_end = end_sample - oldest
            n = offset_end - offset_start
            if n <= 0:
                return None

            # Read position in ring buffer
            read_start = (self._write_pos - avail + offset_start) % self._max_samples
            result = np.zeros(n, dtype=np.int16)

            if read_start + n <= self._max_samples:
                result[:] = self._buffer[read_start:read_start + n]
            else:
                first = self._max_samples - read_start
                result[:first] = self._buffer[read_start:]
                result[first:] = self._buffer[:n - first]

            return result.astype(np.float32) / 32768.0

    def get_wall_time(self, stream_sec: float) -> Optional[datetime.datetime]:
        """Convert stream-relative seconds to wall-clock datetime."""
        with self._lock:
            if self._start_wall is None:
                return None
            return self._start_wall + datetime.timedelta(seconds=stream_sec)

    def _available_samples(self) -> int:
        with self._lock:
            return min(self._total_written, self._max_samples)

    def _extract_last_n(self, n: int) -> np.ndarray:
        """Extract the last n samples as float32 (caller must check n > 0)."""
        with self._lock:
            avail = self._available_samples()
            n = min(n, avail)
            start = (self._write_pos - n) % self._max_samples
            result = np.zeros(n, dtype=np.int16)

            if start + n <= self._max_samples:
                result[:] = self._buffer[start:start + n]
            else:
                first = self._max_samples - start
                result[:first] = self._buffer[start:]
                result[first:] = self._buffer[:n - first]

            return result.astype(np.float32) / 32768.0


# =============================================================================
# SegmentStateMachine
# =============================================================================

class _ActiveSegment:
    """Tracks a segment being accumulated."""
    __slots__ = ("label", "start_sec", "end_sec", "stable_since")

    def __init__(self, label: str, start_sec: float):
        self.label = label
        self.start_sec = start_sec
        self.end_sec = start_sec
        self.stable_since = start_sec

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


class SegmentStateMachine:
    """
    Receives per-bucket classifications from the classifier and decides
    when to finalize segments.

    Delayed finalization: a segment is only finalized when a different label
    has been stable for STABILITY_WINDOW seconds AND the active segment
    exceeds SEGMENT_MIN_DURATION.

    Silence (noEnergy) and noise segments are discarded (not stored).
    """

    def __init__(self, on_finalized):
        self._on_finalized = on_finalized
        self._active: Optional[_ActiveSegment] = None
        self._pending_label: Optional[str] = None
        self._pending_since: float = 0.0
        self._finalized: List[AudioSegment] = []

    def feed(self, labels: list, batch_start_sec: float, bucket_duration: float):
        """
        Process a batch of classifier labels.

        labels: list of (label, start, end) tuples from inaSpeechSegmenter
        batch_start_sec: stream-relative start time of this batch
        bucket_duration: total duration of the classified batch
        """
        for label, start, end in labels:
            # Map inaSpeechSegmenter labels
            mapped = self._map_label(label)
            seg_start = batch_start_sec + start
            seg_end = batch_start_sec + end

            if self._active is None:
                if mapped in STORED_LABELS:
                    self._active = _ActiveSegment(mapped, seg_start)
                    self._active.end_sec = seg_end
                continue

            if mapped == self._active.label:
                # Same label — extend active segment
                self._active.end_sec = seg_end
                self._pending_label = None
                continue

            # Different label
            if self._pending_label != mapped:
                # New pending label — start tracking stability
                self._pending_label = mapped
                self._pending_since = seg_start

            pending_duration = seg_end - self._pending_since

            if (pending_duration >= STABILITY_WINDOW
                    and self._active.duration >= SEGMENT_MIN_DURATION):
                # Finalize the active segment
                self._finalize_active()

                # Start new segment if the new label is stored
                if mapped in STORED_LABELS:
                    self._active = _ActiveSegment(mapped, self._pending_since)
                    self._active.end_sec = seg_end
                else:
                    self._active = None
                self._pending_label = None
            else:
                # Not stable enough yet — keep extending active
                self._active.end_sec = seg_end

    def flush(self):
        """Finalize any remaining active segment (called on stream stop)."""
        if self._active and self._active.duration >= 2.0:
            self._finalize_active()
        self._active = None
        self._pending_label = None

    def _finalize_active(self):
        """Emit the active segment via callback."""
        seg = self._active
        if seg is None:
            return

        start = seg.start_sec
        end = seg.end_sec

        audio_seg = None
        while start < end:
            chunk_end = min(start + MAX_SEGMENT_DURATION, end)

            audio_seg = AudioSegment(
                start=round(start, 2),
                end=round(chunk_end, 2),
                segment_type=seg.label,
            )

            self._on_finalized(audio_seg)
            start = chunk_end

        self._finalized.append(audio_seg)
        logger.info(
            "Segment finalized: %s [%.1f-%.1fs] (%.1fs)",
            seg.label, seg.start_sec, seg.end_sec, seg.duration,
        )

    @staticmethod
    def _map_label(label: str) -> str:
        """Map inaSpeechSegmenter labels to our categories."""
        if label == "noEnergy":
            return "silence"
        return label  # speech, music, noise pass through


# =============================================================================
# StreamSegmentEncoder
# =============================================================================

class StreamSegmentEncoder:
    """
    Given finalized segment metadata and a RollingPCMBuffer reference:
    1. Extract PCM samples for the time range
    2. Optionally refine boundary using spectral flux
    3. Encode to MP3 via ffmpeg
    4. Save file via Django storage and create TranscriptionSegment row
    """

    def __init__(self, buffer: RollingPCMBuffer, recording: Recording):
        self._buffer = buffer
        self._recording = recording

    def encode_and_save(self, segment: AudioSegment):
        """Encode a finalized segment to MP3 and persist to DB."""
        # Extract PCM with context for boundary refinement
        context_start = max(0, segment.start - 10.0)
        context_end = segment.end + 10.0
        pcm_context = self._buffer.get_range_float32(context_start, context_end)

        energy = 0.0
        if pcm_context is not None:
            # Compute energy on the segment portion only
            seg_offset_start = segment.start - context_start
            seg_offset_end = segment.end - context_start
            energy = _compute_energy_db(pcm_context, seg_offset_start, seg_offset_end)

        # Extract just the segment PCM
        pcm = self._buffer.get_range_float32(segment.start, segment.end)
        if pcm is None:
            logger.warning(
                "Cannot extract PCM for segment [%.1f-%.1fs] — buffer expired",
                segment.start, segment.end,
            )
            return

        # Encode to MP3 via ffmpeg pipe
        mp3_data = self._encode_mp3(pcm)
        if mp3_data is None:
            return

        # Build filename
        wall_start = self._buffer.get_wall_time(segment.start)
        wall_end = self._buffer.get_wall_time(segment.end)
        if wall_start is None:
            wall_start = timezone.now()
        if wall_end is None:
            wall_end = timezone.now()

        filename = (
            f"{segment.segment_type}_"
            f"{wall_start:%Y%m%d_%H%M%S}_to_"
            f"{wall_end:%H%M%S}.mp3"
        )

        # Save to DB
        try:
            with transaction.atomic():
                seg_obj = TranscriptionSegment(
                    recording=self._recording,
                    segment_type=segment.segment_type,
                    start_offset=segment.start,
                    end_offset=segment.end,
                    absolute_start_time=wall_start,
                    absolute_end_time=wall_end,
                    energy_db=round(energy, 1),
                )
                # Write MP3 data to a temp file, then save via Django
                tmp_path = os.path.join(
                    tempfile.gettempdir(), f"aum_seg_{os.getpid()}_{id(segment)}.mp3"
                )
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(mp3_data)
                    with open(tmp_path, "rb") as f:
                        seg_obj.file.save(filename, File(f), save=True)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            logger.info(
                "Saved segment: %s [%.1f-%.1fs] (%d bytes MP3)",
                segment.segment_type, segment.start, segment.end, len(mp3_data),
            )
        except Exception:
            logger.exception(
                "Failed to save segment [%.1f-%.1fs]",
                segment.start, segment.end,
            )

    @staticmethod
    def _encode_mp3(pcm_float32: np.ndarray) -> Optional[bytes]:
        """Encode float32 PCM to MP3 via ffmpeg pipe. Returns bytes or None."""
        pcm_int16 = (pcm_float32 * 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        cmd = [
            "ffmpeg",
            "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libmp3lame", "-b:a", "64k",
            "-f", "mp3", "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd,
                input=pcm_bytes,
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.error("ffmpeg MP3 encode failed: %s", result.stderr[:500])
                return None
            return result.stdout
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error("ffmpeg MP3 encode error: %s", exc)
            return None


# =============================================================================
# StreamProcessor
# =============================================================================

class StreamProcessor:
    """
    Per-stream orchestrator for the real-time segmentation pipeline.

    Starts ffmpeg outputting mono 16kHz PCM to stdout.
    Reader thread: reads PCM in 0.5s chunks, appends to RollingPCMBuffer.
    Classifier thread: every ~10s, classifies buffer tail, feeds SegmentStateMachine.
    On segment finalized: StreamSegmentEncoder saves MP3 + DB row.
    """

    def __init__(self, stream: Stream):
        self.stream = stream
        self.last_heartbeat = time.monotonic()
        self.started_at = None
        self.running = False
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._classifier_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._buffer = RollingPCMBuffer()
        self._recording: Optional[Recording] = None
        self._encoder: Optional[StreamSegmentEncoder] = None
        self._state_machine: Optional[SegmentStateMachine] = None
        self._stderr_path: Optional[str] = None
        self._stderr_fh = None
        self._last_classified_sec: float = 0.0

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self):
        """Launch ffmpeg and processing threads."""
        logger.info("StreamProcessor[%s]: starting", self.stream)

        # Create session recording (no file)
        now = timezone.now()
        self._recording = Recording.objects.create(
            stream=self.stream,
            start_time=now,
            end_time=now,
            is_session=True,
            segmentation_status="running",
        )

        self._encoder = StreamSegmentEncoder(self._buffer, self._recording)
        self._state_machine = SegmentStateMachine(
            on_finalized=self._on_segment_finalized
        )
        self._stop_event.clear()

        self._start_ffmpeg()

    def stop(self):
        """Terminate ffmpeg, flush segments, update DB."""
        logger.info("StreamProcessor[%s]: stopping", self.stream)
        self._stop_event.set()

        self._stop_ffmpeg()

        # Flush remaining segments
        if self._state_machine:
            self._state_machine.flush()

        # Update session recording end time
        if self._recording:
            Recording.objects.filter(pk=self._recording.pk).update(
                end_time=timezone.now(),
                segmentation_status="done",
            )

        self._cleanup()
        self._update_stream_status("idle", "")

    def heartbeat(self):
        self.last_heartbeat = time.monotonic()

    def tick(self):
        """
        Called periodically from the main loop.
        Detects ffmpeg crashes and restarts if needed.
        """
        if self._process is None:
            return

        if not self.is_alive():
            raise RuntimeError("Process exited")

        if self._process.poll() is not None:
            rc = self._process.returncode
            stderr_tail = self._read_stderr()
            error_msg = f"ffmpeg exited (rc={rc})"
            if stderr_tail:
                error_msg += f": {stderr_tail[:500]}"

            logger.warning("StreamProcessor[%s]: %s", self.stream, error_msg)

            # Flush any accumulated segments
            if self._state_machine:
                self._state_machine.flush()

            if self._recording:
                Recording.objects.filter(pk=self._recording.pk).update(
                    end_time=timezone.now(),
                    segmentation_status="done",
                )

            self._cleanup()
            self._update_stream_status("error", error_msg)

            # Restart with new session
            logger.info("StreamProcessor[%s]: restarting after crash", self.stream)
            self.start()
            self.heartbeat()

    # -- internal: ffmpeg management ------------------------------------------

    def _build_ffmpeg_cmd(self) -> list:
        """Build ffmpeg command for mono 16kHz PCM stdout output."""
        source = self.stream.source
        proxy_url = source.get_effective_proxy_url() if source else ""

        cmd = [
            "ffmpeg",
            "-y", "-hide_banner", "-loglevel", "warning",
        ]

        if proxy_url:
            if _has_valid_url_scheme(proxy_url, ALLOWED_PROXY_SCHEMES):
                cmd += ["-http_proxy", proxy_url]
            else:
                logger.warning(
                    "StreamProcessor[%s]: proxy URL has disallowed scheme — ignoring",
                    self.stream,
                )

        cmd += [
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
        ]

        cmd += ["-i", self.stream.url]
        cmd += ["-vn"]
        cmd += ["-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1"]

        return cmd

    def _start_ffmpeg(self):
        """Launch the ffmpeg PCM output process."""
        tmp_dir = tempfile.gettempdir()
        self._stderr_path = os.path.join(
            tmp_dir, f"aum_stream_{self.stream.pk}_stderr.log"
        )

        cmd = self._build_ffmpeg_cmd()
        logger.debug(cmd)
        logger.info("StreamProcessor[%s]: launching ffmpeg (PCM mode)", self.stream)

        try:
            self._stderr_fh = open(self._stderr_path, "w")
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._stderr_fh,
            )
            logger.debug("started ffmpeg")
        except Exception as e:
            logger.exception("StreamProcessor[%s]: failed to launch ffmpeg", self.stream)
            self._update_stream_status("error", str(e))
            return

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._pcm_reader,
            name=f"pcm-reader-{self.stream.pk}",
            daemon=True,
        )
        self._reader_thread.start()
        logger.debug("started reader thread")

        # Start classifier thread
        self._classifier_thread = threading.Thread(
            target=self._classifier_loop,
            name=f"classifier-{self.stream.pk}",
            daemon=True,
        )
        self._classifier_thread.start()

        logger.debug("Started threads")

        self._update_stream_status("recording", "")

    def _pcm_reader(self):
        """Background thread: reads PCM from ffmpeg stdout into buffer."""
        proc = self._process
        chunk_bytes = READ_CHUNK_SAMPLES * BYTES_PER_SAMPLE

        try:
            while not self._stop_event.is_set():
                data = proc.stdout.read(chunk_bytes)
                if not data:
                    break
                # Convert bytes to int16 numpy array
                samples = np.frombuffer(data, dtype=np.int16)
                self._buffer.append(samples)
        except (ValueError, OSError):
            pass  # pipe closed

    def _classifier_loop(self):
        """Background thread: periodically classify buffered audio."""
        # Wait for initial audio to accumulate
        while (not self._stop_event.is_set()
               and self._buffer.total_duration < CLASSIFY_INTERVAL):
            self._stop_event.wait(timeout=1.0)

        while not self._stop_event.is_set():
            try:
                self._classify_batch()
                logger.debug(
                    "Classifier tick: buffer=%.1fs last_classified=%.1fs",
                    self._buffer.total_duration,
                    self._last_classified_sec,
                )
            except Exception:
                logger.exception("StreamProcessor[%s]: classifier error", self.stream)

            self._stop_event.wait(timeout=CLASSIFY_INTERVAL)

    def _classify_batch(self):
        """
        Classify the latest un-classified audio via inaSpeechSegmenter.

        Writes PCM to a temp file (in /dev/shm if available, ~320KB per batch),
        runs the already-loaded CNN singleton, feeds labels to state machine.
        """
        current_duration = self._buffer.total_duration
        batch_start = self._last_classified_sec
        batch_end = current_duration

        if batch_end - batch_start < CLASSIFY_INTERVAL * 0.5:
            return  # not enough new audio

        pcm = self._buffer.get_range_float32(batch_start, batch_end)
        if pcm is None or len(pcm) < SAMPLE_RATE:
            return

        # Write to temp file for inaSpeechSegmenter
        # Prefer /dev/shm (RAM-backed) for speed
        shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".wav", prefix="aum_cls_", dir=shm_dir,
        )
        try:
            os.close(tmp_fd)
            # Write as WAV for inaSpeechSegmenter compatibility
            self._write_wav(tmp_path, pcm)

            segmenter = _get_segmenter()
            raw_labels = segmenter(tmp_path)

            logger.debug("Classifier output: %s", raw_labels)

            if raw_labels:
                self._state_machine.feed(
                    raw_labels, batch_start,
                    batch_end - batch_start,
                )
        except Exception:
            logger.exception("StreamProcessor[%s]: classification failed", self.stream)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        self._last_classified_sec = batch_end

    @staticmethod
    def _write_wav(path: str, pcm_float32: np.ndarray):
        """Write float32 PCM to a 16kHz mono WAV file."""
        import wave
        pcm_int16 = (pcm_float32 * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_int16.tobytes())

    def _on_segment_finalized(self, segment: AudioSegment):
        """Callback from state machine when a segment is finalized."""
        if self._encoder:
            self._encoder.encode_and_save(segment)

    def _stop_ffmpeg(self):
        """Send SIGTERM to ffmpeg, wait, then SIGKILL if needed."""
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("StreamProcessor[%s]: ffmpeg did not exit — SIGKILL", self.stream)
                self._process.kill()
                self._process.wait(timeout=5)

        # Wait for threads to finish
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=3)
        if self._classifier_thread and self._classifier_thread.is_alive():
            self._classifier_thread.join(timeout=3)

    def _cleanup(self):
        """Clean up process and handles."""
        self._close_stderr()
        self._process = None
        self._reader_thread = None
        self._classifier_thread = None

    def _read_stderr(self) -> str:
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
        if self._stderr_path:
            try:
                os.unlink(self._stderr_path)
            except OSError:
                pass
        self._stderr_path = None

    def _update_stream_status(self, status: str, error: str):
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
                "StreamProcessor[%s]: failed to update stream status",
                self.stream,
            )
