"""
Music fingerprinting via AcoustID / Chromaprint.

Takes a time slice of an audio file and queries the AcoustID web service
for a song match.  Returns the best-scoring title/artist pair, or None if
no confident match is found.

Requirements
------------
- fpcalc binary on $PATH  (from the Chromaprint project)
    Ubuntu/Debian:  apt install libchromaprint-tools
    Fedora/RHEL:    dnf install chromaprint-tools
    macOS:          brew install chromaprint
- pyacoustid Python package  (pip install pyacoustid)
- settings.ACOUSTID_API_KEY  (register a free key at acoustid.org)
"""

import dataclasses
import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger("broadcast_analysis")

# Discard AcoustID matches below this confidence score (0.0–1.0).
_MIN_SCORE = 0.5

# Segments shorter than this (seconds) are too short for a reliable
# Chromaprint fingerprint and are skipped.
_MIN_DURATION = 10.0

# Cap the audio sent to AcoustID — only the first 2 minutes are needed
# for reliable identification, and this keeps ffmpeg extraction fast.
_MAX_CLIP = 120.0

# Inward offset applied to each music segment before fingerprinting (seconds).
# Segmentation boundaries are approximate: the first and last few seconds of
# a segment often contain bleed-over from adjacent speech/noise, which
# degrades fingerprint quality.  This trims both edges before extraction:
#   effective_start = start + _BOUNDARY_TRIM
#   effective_end   = end   - _BOUNDARY_TRIM
# Increase if you see many false negatives near segment boundaries.
# Set to 0.0 to disable.
_BOUNDARY_TRIM = 10.0


@dataclasses.dataclass
class FingerprintResult:
    title: str
    artist: str
    score: float   # AcoustID confidence 0.0–1.0
    mbid: str      # MusicBrainz recording ID (from acoustid.match result)


def fingerprint_segment(
    source_path: str,
    start: float,
    end: float,
    api_key: str,
) -> Optional[FingerprintResult]:
    """
    Identify the song in [start, end) seconds of source_path.

    Extracts up to _MAX_CLIP seconds starting at `start` to a temp WAV
    file with ffmpeg, then submits the Chromaprint fingerprint to the
    AcoustID web service.

    Returns the best FingerprintResult or None if:
    - The segment is shorter than _MIN_DURATION
    - fpcalc is not on $PATH
    - No match at or above _MIN_SCORE is found
    - The AcoustID service returns an error
    """
    try:
        import acoustid
    except ImportError:
        logger.error("pyacoustid is not installed — cannot fingerprint")
        return None

    # The library hardcodes http:// which AcoustID now redirects to HTTPS.
    # HTTP 301/302 redirects convert POST→GET, dropping the request body
    # and causing error 2: "missing required parameter duration".
    acoustid.set_base_url("https://api.acoustid.org/v2/")

    start = start + _BOUNDARY_TRIM
    end = end - _BOUNDARY_TRIM
    duration = end - start
    if duration < _MIN_DURATION:
        logger.debug("Segment too short to fingerprint after boundary trim (%.1fs)", duration)
        return None

    clip_duration = min(duration, _MAX_CLIP)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name

        # Extract slice to a mono WAV via ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(clip_duration),
            "-i", source_path,
            "-ar", "44100",
            "-ac", "1",
            "-f", "wav",
            tmp_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            logger.error(
                "ffmpeg slice failed for %s [%.1f–%.1f]: %s",
                source_path, start, end,
                proc.stderr.decode(errors="replace"),
            )
            return None

        # Fingerprint and look up against the AcoustID service.
        # Use parse=False so we get the raw JSON response and can log the
        # actual AcoustID error code/message instead of the generic
        # "status: error" that parse_lookup_result raises.
        try:
            response = acoustid.match(api_key, tmp_path, meta="recordings", parse=False)
        except acoustid.NoBackendError:
            logger.error(
                "fpcalc not found — install libchromaprint-tools (Debian/Ubuntu) "
                "or chromaprint-tools (Fedora)"
            )
            return None
        except acoustid.FingerprintGenerationError as exc:
            logger.error("Chromaprint fingerprint generation failed: %s", exc)
            return None
        except acoustid.WebServiceError as exc:
            logger.error("AcoustID web service error: %s", exc)
            return None

        if response.get("status") != "ok":
            error = response.get("error", {})
            code = error.get("code", "?")
            message = error.get("message", "no message")
            logger.error(
                "AcoustID API returned status=%r (error %s: %s)",
                response.get("status"), code, message,
            )
            return None

        matches = list(acoustid.parse_lookup_result(response))

        for score, mbid, title, artist in matches:
            if score >= _MIN_SCORE and title:
                return FingerprintResult(
                    title=title,
                    artist=artist or "",
                    score=score,
                    mbid=mbid or "",
                )

        return None

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
