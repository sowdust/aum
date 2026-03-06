"""
Audio segmentation — speech / music / speech_over_music / silence.

Combines three signals per 0.5-second analysis bucket:

1. **RMS energy (dB)**          → silence vs active audio
2. **webrtcvad speech ratio**   → coarse speech-candidate detection
3. **4 Hz envelope modulation** → THE key speech-vs-music differentiator

Speech naturally modulates at ~4 Hz (syllabic rate): speakers produce
roughly 4 syllables per second, creating a clear amplitude peak near 4 Hz
in the modulation spectrum.  Music — even vocal music with singing — does
not exhibit this modulation pattern because sustained vowels and musical
rhythm differ fundamentally from syllabic speech.
(Scheirer & Slaney, 1997)

Segment types produced
----------------------
speech            clear human talking → transcribe with Whisper
music             instrumental or vocal music → fingerprint with AcoustID
speech_over_music talking over background music → transcribe AND fingerprint
silence           dead air / very low energy → skip

Dependencies: numpy, webrtcvad, ffmpeg (+ ffprobe)
"""

import dataclasses
import json
import logging
import os
import subprocess
import tempfile
from typing import List

import numpy as np

logger = logging.getLogger("broadcast_analysis")

SAMPLE_RATE = 16000          # Hz — required by webrtcvad
FRAME_MS = 30                # webrtcvad frame size (10 | 20 | 30)
BYTES_PER_SAMPLE = 2         # 16-bit signed PCM

BUCKET_SEC = 0.5             # classification resolution
MOD_WINDOW_SEC = 4.0         # centred window for modulation analysis


@dataclasses.dataclass
class AudioSegment:
    start: float             # seconds from file start
    end: float               # seconds from file start
    segment_type: str        # speech | music | speech_over_music | silence
    energy_db: float = 0.0   # average RMS energy of this segment


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def segment_audio(audio_path: str) -> List[AudioSegment]:
    """Segment an audio file into speech / music / speech_over_music / silence."""
    from django.conf import settings
    silence_db = getattr(settings, "SILENCE_THRESHOLD_DB", -40)
    vad_aggressiveness = getattr(settings, "VAD_AGGRESSIVENESS", 2)

    duration = _get_duration(audio_path)
    if duration <= 0:
        return []

    pcm_path = tempfile.mktemp(suffix=".pcm")
    try:
        if not _to_pcm(audio_path, pcm_path):
            return []
        pcm_int16 = np.fromfile(pcm_path, dtype=np.int16)
    finally:
        try:
            os.unlink(pcm_path)
        except OSError:
            pass

    if len(pcm_int16) == 0:
        return []

    samples = pcm_int16.astype(np.float32) / 32768.0
    num_buckets = max(1, int(duration / BUCKET_SEC) + 1)

    # --- three feature vectors, one value per bucket ---
    energy_db = _energy_per_bucket(samples, num_buckets)
    vad_ratio = _vad_per_bucket(pcm_int16, num_buckets, vad_aggressiveness)
    mod_index = _modulation_per_bucket(samples, num_buckets)

    # --- classify ---
    labels = _classify(energy_db, vad_ratio, mod_index, silence_db)

    # --- labels → merged segments ---
    segments = _to_segments(labels, energy_db, duration)

    _log_summary(audio_path, duration, segments)
    return segments


# -----------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------

def _energy_per_bucket(samples: np.ndarray, num_buckets: int) -> np.ndarray:
    """RMS energy in dB for each 0.5 s bucket."""
    bsamp = int(BUCKET_SEC * SAMPLE_RATE)
    energy = np.full(num_buckets, -100.0)
    for i in range(num_buckets):
        lo = i * bsamp
        hi = min(lo + bsamp, len(samples))
        if lo >= len(samples):
            break
        chunk = samples[lo:hi]
        rms = np.sqrt(np.mean(chunk ** 2))
        energy[i] = 20.0 * np.log10(rms + 1e-10)
    return energy


def _vad_per_bucket(
    pcm_int16: np.ndarray,
    num_buckets: int,
    aggressiveness: int = 2,
) -> np.ndarray:
    """Fraction of 30 ms frames within each bucket where webrtcvad fires."""
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)
    frame_bytes = frame_samples * BYTES_PER_SAMPLE
    bucket_samples = int(BUCKET_SEC * SAMPLE_RATE)
    raw = pcm_int16.tobytes()

    ratio = np.zeros(num_buckets)
    for i in range(num_buckets):
        byte_lo = i * bucket_samples * BYTES_PER_SAMPLE
        byte_hi = min(byte_lo + bucket_samples * BYTES_PER_SAMPLE, len(raw))
        if byte_lo >= len(raw):
            break
        speech_frames = 0
        total_frames = 0
        pos = byte_lo
        while pos + frame_bytes <= byte_hi:
            try:
                if vad.is_speech(raw[pos : pos + frame_bytes], SAMPLE_RATE):
                    speech_frames += 1
            except Exception:
                pass
            total_frames += 1
            pos += frame_bytes
        if total_frames:
            ratio[i] = speech_frames / total_frames
    return ratio


def _modulation_per_bucket(samples: np.ndarray, num_buckets: int) -> np.ndarray:
    """
    4 Hz envelope-modulation index per bucket.

    For each bucket a MOD_WINDOW_SEC-wide centred window of the amplitude
    envelope is FFT'd.  The mean magnitude in the 3-6 Hz band (speech
    syllabic zone) divided by the envelope RMS gives a dimensionless index.

    Typical values:
        clear speech          0.15 – 0.40
        speech over music     0.07 – 0.15
        music with vocals     0.02 – 0.07
        instrumental music    0.00 – 0.03
        silence / noise       ≈ 0
    """
    # Full-signal smoothed amplitude envelope (~50 Hz LPF via 20 ms moving avg)
    smooth_n = max(1, int(SAMPLE_RATE * 0.02))
    kernel = np.ones(smooth_n, dtype=np.float32) / smooth_n
    envelope = np.convolve(np.abs(samples), kernel, mode="same")

    bucket_samples = int(BUCKET_SEC * SAMPLE_RATE)
    mod_win_samples = int(MOD_WINDOW_SEC * SAMPLE_RATE)
    half_win = mod_win_samples // 2

    mod = np.zeros(num_buckets)
    for i in range(num_buckets):
        centre = i * bucket_samples + bucket_samples // 2
        lo = max(0, centre - half_win)
        hi = min(len(envelope), centre + half_win)
        win = envelope[lo:hi]
        if len(win) < SAMPLE_RATE:          # need ≥ 1 s
            continue
        win = win - np.mean(win)            # remove DC
        rms = np.sqrt(np.mean(win ** 2))
        if rms < 1e-10:
            continue

        fft_mag = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(len(win), 1.0 / SAMPLE_RATE)

        band = (freqs >= 3.0) & (freqs <= 6.0)
        if not np.any(band):
            continue
        mod[i] = np.mean(fft_mag[band]) / (rms * len(win) + 1e-10)

    return mod


# -----------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------

def _classify(
    energy_db: np.ndarray,
    vad_ratio: np.ndarray,
    mod_index: np.ndarray,
    silence_db: float,
) -> List[str]:
    """
    Combine the three features into a label per bucket.

    Uses *adaptive* modulation thresholds computed from the non-silent
    portion of the recording, clamped to sane ranges so pathological
    recordings (all-speech or all-music) don't break the logic.
    """
    n = len(energy_db)

    active = energy_db > silence_db
    if np.any(active):
        active_mod = mod_index[active]
        # Adaptive thresholds with safety clamps
        mod_high = float(np.clip(np.percentile(active_mod, 70), 0.06, 0.30))
        mod_low = float(np.clip(np.percentile(active_mod, 35), 0.02, 0.12))
    else:
        mod_high, mod_low = 0.12, 0.05

    labels: List[str] = []
    for i in range(n):
        if energy_db[i] < silence_db:
            labels.append("silence")
        elif mod_index[i] >= mod_high and vad_ratio[i] >= 0.25:
            labels.append("speech")
        elif mod_index[i] >= mod_low and vad_ratio[i] >= 0.35:
            labels.append("speech_over_music")
        elif energy_db[i] >= silence_db:
            labels.append("music")
        else:
            labels.append("silence")

    # Two rounds of smoothing to remove isolated 1-bucket glitches
    for _ in range(2):
        labels = _smooth_labels(labels)

    return labels


def _smooth_labels(labels: List[str]) -> List[str]:
    """Replace an isolated single-bucket label with its neighbours."""
    if len(labels) <= 2:
        return labels
    out = list(labels)
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
            out[i] = labels[i - 1]
    return out


# -----------------------------------------------------------------------
# Segments
# -----------------------------------------------------------------------

def _to_segments(
    labels: List[str],
    energy_db: np.ndarray,
    duration: float,
) -> List[AudioSegment]:
    """Convert bucket labels into merged AudioSegment list."""
    if not labels:
        return []

    segments: List[AudioSegment] = []
    cur_type = labels[0]
    cur_start = 0.0
    energy_acc: List[float] = [float(energy_db[0])]

    for i in range(1, len(labels)):
        if labels[i] != cur_type:
            seg_end = round(min(i * BUCKET_SEC, duration), 2)
            segments.append(AudioSegment(
                start=round(cur_start, 2),
                end=seg_end,
                segment_type=cur_type,
                energy_db=round(float(np.mean(energy_acc)), 1),
            ))
            cur_type = labels[i]
            cur_start = i * BUCKET_SEC
            energy_acc = []
        energy_acc.append(float(energy_db[min(i, len(energy_db) - 1)]))

    # Final segment
    segments.append(AudioSegment(
        start=round(cur_start, 2),
        end=round(duration, 2),
        segment_type=cur_type,
        energy_db=round(float(np.mean(energy_acc)), 1),
    ))

    # Drop segments shorter than 1 s by absorbing into neighbours
    segments = _merge_short(segments, min_dur=1.0)

    return segments


def _merge_short(segments: List[AudioSegment], min_dur: float) -> List[AudioSegment]:
    """Absorb segments shorter than *min_dur* into their neighbour."""
    changed = True
    while changed:
        changed = False
        result: List[AudioSegment] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if (seg.end - seg.start) < min_dur:
                changed = True
                if result:
                    result[-1] = dataclasses.replace(result[-1], end=seg.end)
                elif i + 1 < len(segments):
                    segments[i + 1] = dataclasses.replace(
                        segments[i + 1], start=seg.start
                    )
                    i += 1
                    continue
                else:
                    result.append(seg)
            else:
                result.append(seg)
            i += 1
        segments = result

    # Merge adjacent same-type
    merged: List[AudioSegment] = []
    for seg in segments:
        if merged and merged[-1].segment_type == seg.segment_type:
            merged[-1] = dataclasses.replace(merged[-1], end=seg.end)
        else:
            merged.append(seg)
    return merged


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


def _log_summary(path: str, duration: float, segments: List[AudioSegment]) -> None:
    counts = {}
    durations = {}
    for s in segments:
        counts[s.segment_type] = counts.get(s.segment_type, 0) + 1
        durations[s.segment_type] = durations.get(s.segment_type, 0.0) + (s.end - s.start)
    parts = [
        f"{t}: {counts[t]} segs, {durations[t]:.0f}s"
        for t in ("speech", "speech_over_music", "music", "silence")
        if t in counts
    ]
    logger.info("Segmented %s (%.0fs) → %s", os.path.basename(path), duration, " | ".join(parts))
