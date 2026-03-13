"""
AcoustID-based fingerprinting (legacy).

Kept as a fallback — the active fingerprinter now uses ShazamIO.
To revert, copy this file over fingerprinter.py.
"""

import dataclasses
import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger("broadcast_analysis")

_MIN_SCORE = 0.15
_MIN_DURATION = 30.0
_MAX_CLIP = 120.0
_BOUNDARY_TRIM = 10.0


@dataclasses.dataclass
class FingerprintResult:
    title: str
    artist: str
    score: float
    mbid: str


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

    start += _BOUNDARY_TRIM
    end -= _BOUNDARY_TRIM
    duration = end - start
    if duration < _MIN_DURATION:
        logger.debug("Segment too short to fingerprint after boundary trim (%.1fs)", duration)
        return None

    clip_duration = min(duration, _MAX_CLIP)

    tmp_path = None
    try:
        # Create a temporary file path and close it immediately so ffmpeg can write
        tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_dir)
        tmp_path = tmp_file.name
        tmp_file.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", source_path,
            "-ac", "1",        # mono
            "-ar", "44100",    # standard sample rate
            "-f", "wav",
            tmp_path,
        ]
        logger.info("Running ffmpeg: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            logger.error(
                "ffmpeg slice failed for %s [%.1f–%.1f]: %s",
                source_path, start, end,
                proc.stderr.decode(errors="replace"),
            )
            return None

        # Fingerprint and lookup (let pyacoustid handle parsing)
        try:
            response = acoustid.match(api_key, tmp_path)
        except acoustid.NoBackendError:
            logger.error("fpcalc not found — install libchromaprint-tools or chromaprint-tools")
            return None
        except acoustid.FingerprintGenerationError as exc:
            logger.error("Chromaprint fingerprint generation failed: %s", exc)
            return None
        except acoustid.WebServiceError as exc:
            logger.error("AcoustID web service error: %s", exc)
            return None

        best = None
        for score, mbid, title, artist in response:
            if not title:
                continue
            if score >= _MIN_SCORE:
                if not best or score > best.score:
                    best = FingerprintResult(
                        title=title,
                        artist=artist or "",
                        score=score,
                        mbid=mbid or "",
                    )

        return best

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
