"""
Shazam-based music fingerprinting via ShazamIO.

Extracts a clip from the source audio with ffmpeg, then uses the Shazam
recognition API (reverse-engineered, no API key required) to identify it.

Previous implementation used AcoustID/MusicBrainz — see fingerprinter_acoustid.py.
"""

import asyncio
import dataclasses
import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger("broadcast_analysis")

_MIN_DURATION = 30.0
_MAX_CLIP = 120.0
_BOUNDARY_TRIM = 10.0


@dataclasses.dataclass
class FingerprintResult:
    """Identified track metadata — consumed by Song.get_or_create_from_fingerprint()."""
    title: str
    artist: str
    score: float
    mbid: str  # Shazam track key (reuses field name for backward compat)


def fingerprint_segment(
    source_path: str,
    start: float,
    end: float,
) -> Optional[FingerprintResult]:
    """
    Identify the song in [start, end) seconds of source_path.

    Extracts up to _MAX_CLIP seconds (after trimming _BOUNDARY_TRIM from
    each edge) to a temp WAV file with ffmpeg, then submits it to the
    Shazam recognition service via ShazamIO.

    Returns a FingerprintResult or None if:
    - The segment is shorter than _MIN_DURATION after boundary trim
    - ffmpeg fails to extract the clip
    - Shazam returns no match
    - shazamio is not installed
    """
    try:
        from shazamio import Shazam
    except ImportError:
        logger.error("shazamio is not installed — cannot fingerprint")
        return None

    start += _BOUNDARY_TRIM
    end -= _BOUNDARY_TRIM
    duration = end - start
    if duration < _MIN_DURATION:
        logger.debug(
            "Segment too short to fingerprint after boundary trim (%.1fs)", duration,
        )
        return None

    clip_duration = min(duration, _MAX_CLIP)

    tmp_path = None
    try:
        tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=tmp_dir,
        )
        tmp_path = tmp_file.name
        tmp_file.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-t", str(clip_duration),
            "-i", source_path,
            "-ac", "1",
            "-ar", "44100",
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

        result = _recognize_sync(Shazam, tmp_path)
        return result

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _recognize_sync(shazam_cls, audio_path: str) -> Optional[FingerprintResult]:
    """Run ShazamIO's async recognize() from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (e.g. Jupyter, Django async view).
        # Create a new loop in a thread to avoid blocking.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _recognize(shazam_cls, audio_path))
            return future.result(timeout=120)
    else:
        return asyncio.run(_recognize(shazam_cls, audio_path))


async def _recognize(shazam_cls, audio_path: str) -> Optional[FingerprintResult]:
    """Call Shazam recognition and parse the response into a FingerprintResult."""
    shazam = shazam_cls()
    try:
        response = await shazam.recognize(audio_path)
    except Exception as exc:
        logger.error("Shazam recognition failed: %s", exc)
        return None

    track = response.get("track")
    if not track:
        return None

    title = track.get("title", "").strip()
    artist = track.get("subtitle", "").strip()
    track_key = track.get("key", "")

    if not title:
        return None

    return FingerprintResult(
        title=title,
        artist=artist,
        score=1.0,
        mbid=str(track_key),
    )
