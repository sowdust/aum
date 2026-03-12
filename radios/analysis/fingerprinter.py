"""
Shazam-based music fingerprinting via ShazamIO.

Extracts clips from the source audio with ffmpeg, then uses the Shazam
recognition API (reverse-engineered, no API key required) to identify them.

Supports sliding window to detect multiple songs within long music segments.

Previous implementation used AcoustID/MusicBrainz — see fingerprinter_acoustid.py.
"""

import asyncio
import dataclasses
import logging
import os
import subprocess
import tempfile
import time
from typing import Optional

logger = logging.getLogger("broadcast_analysis")

_MIN_DURATION = 10.0   # Shazam needs ~5s; 10s gives a safe margin
_MAX_CLIP = 120.0
_BOUNDARY_TRIM = 3.0   # Trim 3s from each edge to avoid fade-in/out noise

# Sliding window constants
_WINDOW = 30.0             # Clip size for each Shazam call
_DEFAULT_TRACK_DUR = 240.0 # 4 min assumed track length
_FAIL_STEP = 15.0          # Advance on no match (was 30s, smaller = more attempts)
_MAX_ATTEMPTS = 20         # Safety cap per segment

# Rate limiting
_INTER_REQUEST_DELAY = 2.0   # Seconds to wait between Shazam API calls
_RETRY_ATTEMPTS = 3          # Retries on transient failure (covers 429s)
_RETRY_BASE_DELAY = 10.0     # Initial retry wait in seconds (doubled each retry)


@dataclasses.dataclass
class FingerprintResult:
    """Identified track metadata — consumed by Song.get_or_create_from_fingerprint()."""
    title: str
    artist: str
    score: float
    shazam_key: str
    genres: list          # genre name strings
    album_name: str
    release_year: Optional[int]
    album_cover_url: str
    estimated_start: float  # absolute offset in recording
    estimated_end: float    # absolute offset in recording


def fingerprint_segment(
    source_path: str,
    start: float,
    end: float,
) -> Optional[FingerprintResult]:
    """
    Identify the first song in [start, end) seconds of source_path.

    Backward-compatible wrapper around fingerprint_segment_sliding().
    Returns the first FingerprintResult or None.
    """
    results = fingerprint_segment_sliding(source_path, start, end)
    return results[0] if results else None


def fingerprint_segment_sliding(
    source_path: str,
    start: float,
    end: float,
) -> list[FingerprintResult]:
    """
    Identify all songs in [start, end) seconds of source_path using a sliding window.

    Trims _BOUNDARY_TRIM from each edge, then advances through the segment
    in steps, calling Shazam for each window. Deduplicates by shazam_key.
    """
    try:
        from shazamio import Shazam
    except ImportError:
        logger.error("shazamio is not installed — cannot fingerprint")
        return []

    trimmed_start = start + _BOUNDARY_TRIM
    trimmed_end = end - _BOUNDARY_TRIM
    duration = trimmed_end - trimmed_start

    if duration < _MIN_DURATION:
        logger.debug(
            "Segment too short to fingerprint after boundary trim (%.1fs)", duration,
        )
        return []

    results = []
    seen_keys = set()
    pos = trimmed_start
    attempts = 0

    while pos + _MIN_DURATION <= trimmed_end and attempts < _MAX_ATTEMPTS:
        attempts += 1

        # Skip if pos falls inside an already-identified song's range
        skip = False
        for r in results:
            if r.estimated_start <= pos < r.estimated_end:
                pos = r.estimated_end
                skip = True
                break
        if skip:
            continue

        clip_duration = min(_WINDOW, trimmed_end - pos)
        if clip_duration < _MIN_DURATION:
            break

        result = _extract_and_recognize(Shazam, source_path, pos, clip_duration)

        # Throttle: wait between API calls to avoid rate limiting
        if pos + _MIN_DURATION <= trimmed_end and attempts < _MAX_ATTEMPTS:
            time.sleep(_INTER_REQUEST_DELAY)

        if result:
            if result.shazam_key and result.shazam_key in seen_keys:
                # Same song still playing — advance past it
                pos += _DEFAULT_TRACK_DUR
                logger.debug(
                    "Duplicate shazam_key %s at %.1fs, skipping",
                    result.shazam_key, pos,
                )
            else:
                # New song found
                result.estimated_start = pos
                track_dur = _DEFAULT_TRACK_DUR
                result.estimated_end = pos + track_dur
                if result.shazam_key:
                    seen_keys.add(result.shazam_key)
                results.append(result)
                logger.info(
                    "Identified [%.1f-%.1fs]: %s — %s (key=%s)",
                    result.estimated_start, result.estimated_end,
                    result.artist, result.title, result.shazam_key,
                )
                pos += track_dur
        else:
            pos += _FAIL_STEP

    return results


def _extract_and_recognize(shazam_cls, source_path, pos, clip_duration):
    """Extract a clip and run Shazam recognition on it."""
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
            "-ss", str(pos),
            "-t", str(clip_duration),
            "-i", source_path,
            "-ac", "1",
            "-ar", "44100",
            "-f", "wav",
            tmp_path,
        ]
        logger.debug("Running ffmpeg: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            logger.error(
                "ffmpeg slice failed for %s [%.1f+%.1fs]: %s",
                source_path, pos, clip_duration,
                proc.stderr.decode(errors="replace"),
            )
            return None

        return _recognize_sync(shazam_cls, tmp_path)

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _recognize_sync(shazam_cls, audio_path: str) -> Optional[FingerprintResult]:
    """
    Run ShazamIO's async recognize() from synchronous code.

    Retries up to _RETRY_ATTEMPTS times with exponential back-off on any
    exception (covers HTTP 429 / transient network errors from Shazam).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    delay = _RETRY_BASE_DELAY
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _recognize(shazam_cls, audio_path))
                    return future.result(timeout=120)
            else:
                return asyncio.run(_recognize(shazam_cls, audio_path))
        except Exception as exc:
            if attempt == _RETRY_ATTEMPTS:
                logger.error(
                    "Shazam recognition failed after %d attempts: %s",
                    _RETRY_ATTEMPTS, exc,
                )
                return None
            logger.warning(
                "Shazam recognition failed (attempt %d/%d), retrying in %.0fs: %s",
                attempt, _RETRY_ATTEMPTS, delay, exc,
            )
            time.sleep(delay)
            delay *= 2
    return None  # unreachable, satisfies type checker


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
    shazam_key = str(track.get("key", ""))

    if not title:
        return None

    # Extract genre
    genres = []
    genres_data = track.get("genres")
    if genres_data:
        primary = genres_data.get("primary")
        if primary:
            genres.append(primary)

    # Extract album and release year from sections
    album_name = ""
    release_year = None
    sections = track.get("sections", [])
    for section in sections:
        if section.get("type") == "SONG":
            for meta in section.get("metadata", []):
                if meta.get("title") == "Album":
                    album_name = meta.get("text", "")
                elif meta.get("title") == "Released":
                    year_text = meta.get("text", "")
                    try:
                        release_year = int(year_text[:4]) if len(year_text) >= 4 else None
                    except (ValueError, TypeError):
                        pass

    # Extract cover art
    album_cover_url = ""
    images = track.get("images")
    if images:
        album_cover_url = images.get("coverart", "")

    return FingerprintResult(
        title=title,
        artist=artist,
        score=1.0,
        shazam_key=shazam_key,
        genres=genres,
        album_name=album_name,
        release_year=release_year,
        album_cover_url=album_cover_url,
        estimated_start=0.0,  # will be set by caller
        estimated_end=0.0,
    )
