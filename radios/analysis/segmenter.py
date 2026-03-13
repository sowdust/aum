"""
Audio segmentation -- speech / music / noise / noEnergy.

Uses inaSpeechSegmenter, a CNN-based tool built for broadcast radio/TV
segmentation. It classifies audio into speech, music, noise, and noEnergy
(silence) segments directly from the audio file.

Previous implementation used webrtcvad + 4Hz modulation + RMS energy.
The old code is preserved in segmenter_webrtcvad.py for rollback.

Segment types produced
----------------------
speech    human talking           -> transcribe with Whisper
music     instrumental or vocal   -> fingerprint with AcoustID
noise     non-speech, non-music   -> skip or flag
noEnergy  dead air / silence      -> skip

Post-processing pipeline
------------------------

Step 1 -- Two-pass consolidation:
    inaSpeechSegmenter classifies short frames (~0.02 s) independently,
    producing many tiny fragments near content transitions.  A two-pass
    consolidation merges these into broadcast-scale blocks:

    Pass 1 -- jitter removal (< 3 s):
        Absorbs tiny CNN noise fragments (1-3 s label flickers at
        boundaries).  This sharpens transition zones before the main pass.

    Pass 2 -- broadcast consolidation (< SEGMENT_MIN_DURATION):
        Absorbs remaining short segments to produce broadcast-scale blocks.

    Both passes process segments **shortest-first** (not left-to-right)
    to avoid systematically shifting boundaries forward in time.

Step 2 -- Boundary refinement (proximity-weighted spectral flux):
    After consolidation, each boundary is fine-tuned using the spectral
    flux of the PCM audio.  Spectral flux measures how much the frequency
    content changes between consecutive frames -- peaks correspond to
    acoustic events (content transitions, beat onsets, etc.).

    The flux is smoothed over 2 s to suppress individual beat onsets and
    highlight broad content transitions.  Each boundary is refined by
    searching +/- SEGMENT_MIN_DURATION seconds for the best peak.

    Peak selection uses a Gaussian proximity weight centred on the
    consolidated boundary (sigma = 5 s).  This ensures:
      - A moderate flux peak 2 s from the boundary (likely the real
        transition) easily beats a loud beat 10 s away.
      - If the CNN boundary is already at the right spot, it barely moves.
    Without proximity weighting, argmax often picks a loud drum hit or
    cymbal crash 10-15 s away instead of the actual content transition.

Django settings
---------------
SEGMENT_MIN_DURATION  (float, default 15.0)
    Minimum segment duration in seconds.  Segments shorter than this are
    absorbed into neighbours.  Also used as the search radius for boundary
    refinement.  Increase for cleaner output with fewer segments; decrease
    if you need finer granularity.  Recommended range: 10 -- 30.

Dependencies: inaSpeechSegmenter, tensorflow-cpu, numpy, ffmpeg (+ ffprobe)
"""

import dataclasses
import json
import logging
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

logger = logging.getLogger("broadcast_analysis")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"   # suppress INFO

SAMPLE_RATE = 16000   # Hz -- PCM sample rate (do not change without re-testing)

# -----------------------------------------------------------------------
# Tuning parameters
# All knobs that affect segmentation quality are collected here so they
# can be found and adjusted in one place.
# -----------------------------------------------------------------------

# -- Consolidation -------------------------------------------------------

# Pass 1 jitter threshold (seconds).
# CNN frames shorter than this are treated as label flicker at boundaries
# and absorbed before the main consolidation pass.  Range: 1–5 s.
JITTER_THRESHOLD = 2.0

# Pass 2 minimum segment duration (seconds).
# Segments shorter than this are absorbed into neighbours to produce
# broadcast-scale blocks (full songs, long speech runs).
# Overridable via Django setting SEGMENT_MIN_DURATION.  Range: 10–30 s.
SEGMENT_MIN_DURATION_DEFAULT = 10.0

# -- Spectral flux -------------------------------------------------------

# Hop between consecutive FFT analysis frames (seconds).
# Smaller = finer time resolution but more computation.  Range: 0.1–0.5 s.
FLUX_HOP_SEC = 0.25

# FFT window size (seconds).
# Larger = more frequency resolution, less time precision.  Range: 0.25–1.0 s.
FLUX_WIN_SEC = 0.5

# Moving-average smoothing applied to raw flux (seconds).
# Suppresses individual beat onsets; keeps broad content transitions.
# Range: 0.5–4.0 s.  Larger = smoother but less precise boundary location.
FLUX_SMOOTH_SEC = 1.0

# -- Boundary refinement -------------------------------------------------

# Search radius around each consolidated boundary (seconds).
# The refinement will not look further than this from the INA boundary.
# Range: 2–20 s.  Smaller = stays closer to INA; larger = more freedom.
REFINE_SEARCH_RADIUS = 10.0

# Gaussian proximity sigma (seconds).
# Controls how strongly the refinement prefers peaks close to the boundary
# over peaks further away.
#   sigma=1.5 → very tight (almost ignores anything >3 s away)
#   sigma=5.0 → moderate (peaks 10 s away still get ~2 % weight)
#   sigma=10+ → loose (similar to raw argmax)
# Range: 1.0–10.0 s.
PROXIMITY_SIGMA = 5.0

# -----------------------------------------------------------------------

# Lazy-loaded singleton -- the Segmenter is expensive to initialise
# (loads CNN weights), so we create it once and reuse across calls.
_segmenter_instance = None


@dataclasses.dataclass
class AudioSegment:
    start: float                    # seconds from file start
    end: float                      # seconds from file start
    segment_type: str               # speech | music | noise | noEnergy
    energy_db: float = 0.0          # average RMS energy of this segment
    file_path: Optional[str] = None # absolute path if saved to disk, else None


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def segment_audio(
    audio_path: str,
    save_dir: Optional[str] = None,
) -> List[AudioSegment]:
    """
    Segment an audio file into speech / music / noise / noEnergy.

    Parameters
    ----------
    audio_path : str
        Path to the source audio file.
    save_dir : str or None
        Directory in which to save each segment as a separate MP3 file.
        If None, the Django setting ``SEGMENT_SAVE_DIR`` is used; if that is
        also unset or empty, segments are not written to disk.
        Within *save_dir* a sub-directory named after the source file stem is
        created automatically (e.g. ``<save_dir>/test_1/``).
    """
    from django.conf import settings

    duration = _get_duration(audio_path)
    if duration <= 0:
        return []

    # --- Run inaSpeechSegmenter CNN ---
    segmenter = _get_segmenter()
    try:
        raw_segments = segmenter(audio_path)
    except Exception as exc:
        logger.error("inaSpeechSegmenter failed on %s: %s", audio_path, exc)
        return []

    if not raw_segments:
        return []

    # --- Convert to PCM for energy calculation and boundary refinement ---
    pcm_samples = _load_pcm(audio_path)

    # --- Correct any timestamp drift between INA and our PCM ---
    raw_segments = _correct_timestamp_drift(raw_segments, pcm_samples, duration)

    # --- Build AudioSegment list with energy ---
    segments: List[AudioSegment] = []
    for label, start, end in raw_segments:
        energy = _compute_energy_db(pcm_samples, start, end)
        segments.append(AudioSegment(
            start=round(start, 2),
            end=round(end, 2),
            segment_type=label,
            energy_db=round(energy, 1),
        ))

    # --- Consolidate into broadcast-scale blocks ---
    min_seg = getattr(settings, "SEGMENT_MIN_DURATION", SEGMENT_MIN_DURATION_DEFAULT)
    segments = _consolidate_segments(segments, min_dur=min_seg)

    # --- Refine boundaries using spectral flux ---
    segments = _refine_boundaries(segments, pcm_samples, search_radius=REFINE_SEARCH_RADIUS)

    # --- Optionally save segments to disk ---
    effective_save_dir = save_dir or getattr(settings, "SEGMENT_SAVE_DIR", None)
    if effective_save_dir:
        segments = _save_segments_to_disk(segments, audio_path, effective_save_dir)

    _log_summary(audio_path, duration, segments)
    return segments


# -----------------------------------------------------------------------
# Segmenter singleton
# -----------------------------------------------------------------------

def _get_segmenter():
    """
    Return a lazily-initialised inaSpeechSegmenter.Segmenter instance.

    Uses vad_engine='smn' for speech/music/noise classification.
    Gender detection is off -- we only need content-type labels.
    """
    global _segmenter_instance

    if _segmenter_instance is None:
        from inaSpeechSegmenter import Segmenter
        logger.info("Initialising inaSpeechSegmenter (first call, loading CNN weights)...")
        _segmenter_instance = Segmenter(
            vad_engine='smn',
            detect_gender=False,
        )
        logger.info("inaSpeechSegmenter ready.")

    return _segmenter_instance


# -----------------------------------------------------------------------
# Energy calculation
# -----------------------------------------------------------------------

def _load_pcm(audio_path: str) -> Optional[np.ndarray]:
    """Convert audio to 16kHz mono PCM and return as float32 samples, or None on failure."""
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
        pcm_path = tempfile.mktemp(suffix=".pcm")
        try:
            if not _to_pcm(audio_path, pcm_path):
                return None
            pcm_int16 = np.fromfile(pcm_path, dtype=np.int16)
        finally:
            try:
                os.unlink(pcm_path)
            except OSError:
                pass

        if len(pcm_int16) == 0:
            return None

        return pcm_int16.astype(np.float32) / 32768.0


def _compute_energy_db(
    samples: Optional[np.ndarray],
    start: float,
    end: float,
) -> float:
    """Compute RMS energy in dB for a time range within the PCM samples."""
    if samples is None or start <= end:
        return -100.0

    lo = int(start * SAMPLE_RATE)
    hi = int(end * SAMPLE_RATE)
    lo = max(0, min(lo, len(samples)))
    hi = max(lo, min(hi, len(samples)))

    if lo >= hi:
        return -100.0

    chunk = samples[lo:hi]
    rms = np.sqrt(np.mean(chunk ** 2))
    return 20.0 * np.log10(rms + 1e-10)


# -----------------------------------------------------------------------
# Timestamp drift correction
# -----------------------------------------------------------------------

def _correct_timestamp_drift(
    raw_segments: list,
    pcm_samples: Optional[np.ndarray],
    ffprobe_duration: float,
) -> list:
    """
    Linearly rescale inaSpeechSegmenter timestamps to match the PCM time base.

    inaSpeechSegmenter and ffmpeg decode the audio independently (different
    resamplers, different MP3 frame handling).  For streaming-captured files
    the effective durations can diverge, which compounds to progressive drift.

    We use the PCM sample count as ground truth (since spectral flux and
    energy are computed on it) and linearly rescale all INA timestamps so
    the last segment's end matches the PCM duration.
    """
    if not raw_segments:
        return raw_segments

    ina_end = raw_segments[-1][2]
    if ina_end <= 0:
        return raw_segments

    # Ground truth: actual PCM duration (preferred) or ffprobe duration
    if pcm_samples is not None and len(pcm_samples) > 0:
        reference_duration = len(pcm_samples) / SAMPLE_RATE
    else:
        reference_duration = ffprobe_duration

    drift = abs(ina_end - reference_duration)
    if drift < 0.5:
        return raw_segments

    scale = reference_duration / ina_end

    # Sanity check: only correct reasonable drift (up to ~5 %)
    if not (0.95 <= scale <= 1.05):
        logger.warning(
            "Timestamp scale factor %.4f out of range (INA=%.1fs, PCM=%.1fs) "
            "-- skipping drift correction",
            scale, ina_end, reference_duration,
        )
        return raw_segments

    logger.info(
        "Correcting timestamp drift: INA=%.1fs vs PCM=%.1fs (%.1fs drift, scale=%.6f)",
        ina_end, reference_duration, drift, scale,
    )
    return [(label, start * scale, end * scale) for label, start, end in raw_segments]


# -----------------------------------------------------------------------
# Segment consolidation
# -----------------------------------------------------------------------

def _merge_adjacent(segments: List[AudioSegment]) -> List[AudioSegment]:
    """Merge consecutive segments that share the same type."""
    if not segments:
        return segments
    merged: List[AudioSegment] = []
    for seg in segments:
        if merged and merged[-1].segment_type == seg.segment_type:
            merged[-1] = dataclasses.replace(merged[-1], end=seg.end)
        else:
            merged.append(seg)
    return merged


def _absorb_shortest(
    segments: List[AudioSegment],
    min_dur: float,
) -> List[AudioSegment]:
    """
    Repeatedly absorb the shortest sub-threshold segment into a neighbour.

    Processes the **shortest** segment first each iteration (not
    left-to-right) to avoid systematically shifting transition boundaries.

    Neighbour selection priority:
    1. Both neighbours same type  -> bridge the gap.
    2. Short segment's type matches one neighbour -> extend that neighbour
       (keeps the real transition boundary in place).
    3. No type match -> absorb into the longer neighbour.
    """
    segments = _merge_adjacent(segments)

    while len(segments) > 1:
        shortest_idx = min(
            range(len(segments)),
            key=lambda i: segments[i].end - segments[i].start,
        )
        if (segments[shortest_idx].end - segments[shortest_idx].start) >= min_dur:
            break

        idx = shortest_idx
        seg = segments[idx]
        prev = segments[idx - 1] if idx > 0 else None
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None

        if prev and next_seg:
            if prev.segment_type == next_seg.segment_type:
                segments[idx - 1] = dataclasses.replace(prev, end=next_seg.end)
                del segments[idx:idx + 2]
            elif seg.segment_type == prev.segment_type:
                segments[idx - 1] = dataclasses.replace(prev, end=seg.end)
                del segments[idx]
            elif seg.segment_type == next_seg.segment_type:
                segments[idx + 1] = dataclasses.replace(next_seg, start=seg.start)
                del segments[idx]
            else:
                prev_dur = prev.end - prev.start
                next_dur = next_seg.end - next_seg.start
                if prev_dur >= next_dur:
                    segments[idx - 1] = dataclasses.replace(prev, end=seg.end)
                else:
                    segments[idx + 1] = dataclasses.replace(next_seg, start=seg.start)
                del segments[idx]
        elif prev:
            segments[idx - 1] = dataclasses.replace(prev, end=seg.end)
            del segments[idx]
        elif next_seg:
            segments[idx + 1] = dataclasses.replace(next_seg, start=seg.start)
            del segments[idx]
        else:
            break

        segments = _merge_adjacent(segments)

    return segments




def _consolidate_segments(
    segments: List[AudioSegment],
    min_dur: float = 15.0,
) -> List[AudioSegment]:
    """
    Two-pass consolidation for broadcast-scale segmentation.

    Pass 1 -- jitter removal (< 3 s):
        Absorbs tiny CNN classification noise.  These 1-3 s fragments are
        almost always wrong labels flickering at content boundaries.
        Removing them first sharpens the transition zones.

    Pass 2 -- broadcast consolidation (< min_dur):
        Absorbs remaining short segments to produce broadcast-scale blocks.
        Because pass 1 already cleaned up the transition zones, this pass
        shifts boundaries much less (typically < 3-5 s from reality).
    """
    if not segments:
        return segments

    segments = list(segments)
    segments = _absorb_shortest(segments, JITTER_THRESHOLD)
    segments = _absorb_shortest(segments, min_dur)
    return segments


# -----------------------------------------------------------------------
# Boundary refinement (proximity-weighted spectral flux)
# -----------------------------------------------------------------------



def _spectral_flux(samples: np.ndarray) -> np.ndarray:
    """
    Compute spectral flux over the entire signal.

    Spectral flux measures how much the frequency content changes between
    consecutive short-time frames.  A peak in the smoothed flux corresponds
    to a real content transition (speech -> music, music -> speech, etc.).

    Returns one value per hop (0.25 s).
    """
    hop_n = int(FLUX_HOP_SEC * SAMPLE_RATE)
    win_n = int(FLUX_WIN_SEC * SAMPLE_RATE)
    n_frames = max(0, (len(samples) - win_n) // hop_n + 1)
    if n_frames < 2:
        return np.zeros(0)

    flux = np.zeros(n_frames)
    prev_mag = None
    for f in range(n_frames):
        lo = f * hop_n
        frame = samples[lo:lo + win_n]
        mag = np.abs(np.fft.rfft(frame))
        if prev_mag is not None:
            # Half-wave rectified difference -- only count spectral increases
            # (onset-style detection, more robust than raw difference)
            flux[f] = np.sum(np.maximum(mag - prev_mag, 0.0))
        prev_mag = mag

    # Smooth to suppress individual beat/note onsets and highlight broad
    # content transitions (~2 s moving average)
    smooth_frames = max(1, int(FLUX_SMOOTH_SEC / FLUX_HOP_SEC))
    kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
    flux = np.convolve(flux, kernel, mode="same")

    return flux


def _refine_boundaries(
    segments: List[AudioSegment],
    samples: Optional[np.ndarray],
    search_radius: float = 15.0,
) -> List[AudioSegment]:
    """
    Fine-tune each boundary using proximity-weighted spectral flux.

    For each boundary, searches +/- *search_radius* seconds in the
    smoothed spectral flux.  Instead of picking the global maximum (which
    is often a loud beat far from the boundary), the flux is multiplied
    by a Gaussian centred on the current boundary position.  This ensures
    that a moderate transition peak near the boundary beats a loud beat
    10+ seconds away.
    """
    if samples is None or len(segments) < 2:
        return segments

    flux = _spectral_flux(samples)
    if len(flux) == 0:
        return segments

    hop_n = int(FLUX_HOP_SEC * SAMPLE_RATE)
    segments = list(segments)

    for i in range(1, len(segments)):
        boundary = segments[i].start

        # Clamp search window so it doesn't cross into neighbouring segments'
        # interiors (leave at least 1 s margin)
        lo_sec = max(segments[i - 1].start + 1.0, boundary - search_radius)
        hi_sec = min(segments[i].end - 1.0, boundary + search_radius)
        if lo_sec >= hi_sec:
            continue

        lo_frame = max(0, int(lo_sec / FLUX_HOP_SEC))
        hi_frame = min(len(flux), int(hi_sec / FLUX_HOP_SEC))
        if lo_frame >= hi_frame:
            continue

        # Proximity-weighted peak selection: multiply flux by a Gaussian
        # centred on the current boundary.  A moderate peak 2 s away
        # easily beats a loud peak 10 s away.
        window_flux = flux[lo_frame:hi_frame].copy()
        frame_times = np.arange(lo_frame, hi_frame) * FLUX_HOP_SEC
        proximity = np.exp(
            -((frame_times - boundary) ** 2) / (2.0 * PROXIMITY_SIGMA ** 2)
        )
        weighted_flux = window_flux * proximity

        peak_frame = lo_frame + np.argmax(weighted_flux)
        new_boundary = round(peak_frame * hop_n / SAMPLE_RATE, 2)

        segments[i - 1] = dataclasses.replace(segments[i - 1], end=new_boundary)
        segments[i] = dataclasses.replace(segments[i], start=new_boundary)

    return segments


# -----------------------------------------------------------------------
# Segment saving
# -----------------------------------------------------------------------

def _segment_filename(index: int, seg: AudioSegment) -> str:
    """Build a human-readable filename for a saved segment."""
    def ts(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}m{s:02d}s"
    return f"{index:03d}_{seg.segment_type}_{ts(seg.start)}-{ts(seg.end)}.mp3"


def _save_segment(src_path: str, start: float, end: float, out_path: str) -> bool:
    """
    Extract [start, end] seconds from *src_path* and write to *out_path*.

    Uses ``-ss``/``-to`` before ``-i`` (fast seek) with stream copy so the
    operation is nearly instantaneous.  MP3 frame alignment may shift the
    actual cut by up to ~0.026 s at each boundary — acceptable for broadcast
    segments that are typically 15+ seconds long.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-ss", str(start),
        "-to", str(end),
        "-i", src_path,
        "-c", "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.error("Failed to save segment %s: %s", out_path, exc)
        return False


def _save_segments_to_disk(
    segments: List[AudioSegment],
    src_path: str,
    save_dir: str,
) -> List[AudioSegment]:
    """
    Save each segment to *save_dir/<source_stem>/* and return updated segments
    with ``file_path`` populated.
    """
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_dir = os.path.join(save_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Saving %d segments to %s", len(segments), out_dir)

    updated = []
    for i, seg in enumerate(segments):
        filename = _segment_filename(i, seg)
        out_path = os.path.join(out_dir, filename)
        ok = _save_segment(src_path, seg.start, seg.end, out_path)
        updated.append(dataclasses.replace(seg, file_path=out_path if ok else None))
        if ok:
            logger.debug("Saved segment %s", out_path)

    saved = sum(1 for s in updated if s.file_path)
    logger.info("Saved %d/%d segments to %s", saved, len(segments), out_dir)
    return updated


# -----------------------------------------------------------------------
# Audio I/O helpers
# -----------------------------------------------------------------------

def _get_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception as exc:
        logger.error("ffprobe failed on %s: %s", path, exc)
        return 0.0


def _to_pcm(src: str, dst: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", src,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.error("PCM conversion failed: %s", exc)
        return False


def _fmt_duration(seconds: float) -> str:
    """Format seconds as 'Xm Ys' for log messages."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _log_summary(path: str, duration: float, segments: List[AudioSegment]) -> None:
    counts = {}
    durations = {}
    for s in segments:
        counts[s.segment_type] = counts.get(s.segment_type, 0) + 1
        durations[s.segment_type] = durations.get(s.segment_type, 0.0) + (s.end - s.start)
    parts = [
        f"{t}: {counts[t]} seg(s), {_fmt_duration(durations[t])}"
        for t in ("speech", "music", "noise", "noEnergy")
        if t in counts
    ]
    logger.info(
        "Segmented %s (%s) -> %s",
        os.path.basename(path), _fmt_duration(duration), " | ".join(parts),
    )
