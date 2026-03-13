"""
Reconstruct a radio's full broadcast day from transcription, segmentation, and
fingerprinting data. Identifies shows, summarises content, and lists songs.

Usage
-----
    from radios.analysis.broadcast_day_summarizer import summarize_broadcast_day

    result = summarize_broadcast_day(radio, date_obj)
    # result.overview, result.tags, result.shows
"""

import dataclasses
import json
import logging
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from radios.analysis._llm_backends import call_llm

logger = logging.getLogger("broadcast_analysis")

_MAX_TIMELINE_CHARS = 100_000
_MAX_TAGS = 15
_SPEECH_TEXT_PREVIEW = 300


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ShowResult:
    name: str
    show_type: str
    start_time: str   # "HH:MM"
    end_time: str      # "HH:MM"
    summary: str
    tags: list
    songs: list        # ["Title - Artist", ...]


@dataclasses.dataclass
class BroadcastDayResult:
    overview: str
    tags: list
    shows: list        # list of ShowResult


@dataclasses.dataclass
class TimelineEntry:
    """A single event in the broadcast timeline."""
    absolute_time: datetime
    entry_type: str      # "speech", "music", "noise", "silence", "song"
    text: str
    duration_seconds: float = 0.0
    recording_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_broadcast_day(radio, day: date) -> Optional[BroadcastDayResult]:
    """
    Gather data for a radio+date, build a timeline, call LLM, parse result.

    Returns BroadcastDayResult or None on failure.
    """
    from radios.models import DailySummarizationSettings

    timeline_entries = gather_broadcast_day_data(radio, day)
    if not timeline_entries:
        logger.info("No timeline data for %s on %s.", radio.name, day)
        return None

    tz = ZoneInfo(radio.timezone or "UTC")
    timeline_text = build_timeline(timeline_entries, tz)

    if not timeline_text.strip():
        return None

    # If timeline is too long, try using chunk summaries as fallback
    if len(timeline_text) > _MAX_TIMELINE_CHARS and _MAX_TIMELINE_CHARS > 0:
        timeline_text = _build_fallback_timeline(radio, day, timeline_entries, tz)

    cfg = DailySummarizationSettings.get_settings()

    location_parts = []
    if radio.city:
        location_parts.append(radio.city)
    if radio.country:
        location_parts.append(str(radio.country.name))
    location = ", ".join(location_parts) or "Unknown"

    prompt = cfg.prompt_broadcast_day.format(
        radio_name=radio.name,
        radio_location=location,
        radio_language=radio.languages or "Unknown",
        date=day.isoformat(),
        timezone=radio.timezone or "UTC",
        website=radio.website or "Unknown",
        timeline=timeline_text,
    )

    response_text = call_llm(prompt, cfg, label="Broadcast Day Summarization")
    if not response_text:
        return None

    print('-' * 78)
    print(prompt)
    print('-' * 78)
    print(response_text)

    return _parse_response(response_text)


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def gather_broadcast_day_data(radio, day: date) -> list:
    """
    Query recordings, segments, and song occurrences for a radio+date.
    Returns a list of TimelineEntry sorted chronologically.
    """
    from radios.models import Recording, TranscriptionSegment, SongOccurrence

    tz = ZoneInfo(radio.timezone or "UTC")

    # Compute UTC boundaries for the local date
    local_start = datetime(day.year, day.month, day.day, tzinfo=tz)
    local_end = local_start + timedelta(days=1)

    recordings = list(
        Recording.objects.filter(
            stream__radio=radio,
            start_time__gte=local_start,
            start_time__lt=local_end,
        )
        .select_related("stream")
        .order_by("start_time")
    )

    if not recordings:
        return []

    entries = []
    recording_ids = [r.pk for r in recordings]
    recording_map = {r.pk: r for r in recordings}

    # Gather segments
    segments = (
        TranscriptionSegment.objects
        .filter(recording_id__in=recording_ids)
        .order_by("recording__start_time", "start_offset")
    )

    # Gather song occurrences
    song_occurrences = list(
        SongOccurrence.objects
        .filter(segment__recording_id__in=recording_ids)
        .select_related("song", "song__artist_ref", "segment")
        .prefetch_related("song__genres")
    )

    # Build a set of (recording_id, approx_offset) for songs to avoid duplication
    song_times = set()
    for occ in song_occurrences:
        rec = recording_map.get(occ.segment.recording_id)
        if not rec:
            continue
        abs_time = rec.start_time + timedelta(seconds=occ.start_offset)
        abs_time_local = abs_time.astimezone(tz)
        duration = occ.end_offset - occ.start_offset if occ.end_offset > occ.start_offset else 0

        song_str = occ.song.title
        if occ.song.artist:
            song_str += f" — {occ.song.artist}"

        genre_names = [g.name for g in occ.song.genres.all()]
        if genre_names:
            song_str += f" [{', '.join(genre_names)}]"

        entries.append(TimelineEntry(
            absolute_time=abs_time_local,
            entry_type="song",
            text=song_str,
            duration_seconds=duration,
            recording_id=occ.segment.recording_id,
        ))
        song_times.add((occ.segment.recording_id, round(occ.start_offset)))

    for seg in segments:
        rec = recording_map.get(seg.recording_id)
        if not rec:
            continue

        abs_time = rec.start_time + timedelta(seconds=seg.start_offset)
        abs_time_local = abs_time.astimezone(tz)
        duration = max(0, (seg.end_offset or seg.start_offset) - seg.start_offset)

        if seg.segment_type in ("speech", "speech_over_music"):
            text = seg.text or seg.text_english or ""
            if text:
                text = text[:_SPEECH_TEXT_PREVIEW]
                if len(seg.text or seg.text_english or "") > _SPEECH_TEXT_PREVIEW:
                    text += "..."
            entries.append(TimelineEntry(
                absolute_time=abs_time_local,
                entry_type="speech",
                text=text,
                duration_seconds=duration,
                recording_id=seg.recording_id,
            ))
        elif seg.segment_type == "music":
            # Only add music entry if there's no song occurrence for this segment
            has_song = (seg.recording_id, round(seg.start_offset)) in song_times
            if not has_song:
                entries.append(TimelineEntry(
                    absolute_time=abs_time_local,
                    entry_type="music",
                    text="(unidentified music)",
                    duration_seconds=duration,
                    recording_id=seg.recording_id,
                ))
        elif seg.segment_type in ("noise", "noEnergy", "silence"):
            # Only note significant gaps (> 30s)
            if duration > 30:
                entries.append(TimelineEntry(
                    absolute_time=abs_time_local,
                    entry_type="silence",
                    text=f"{seg.get_segment_type_display()} ({duration:.0f}s)",
                    duration_seconds=duration,
                    recording_id=seg.recording_id,
                ))

    entries.sort(key=lambda e: e.absolute_time)
    return entries


# ---------------------------------------------------------------------------
# Timeline building
# ---------------------------------------------------------------------------

def build_timeline(entries: list, tz: ZoneInfo) -> str:
    """Build a chronological text timeline from TimelineEntry list."""
    lines = []
    for entry in entries:
        time_str = entry.absolute_time.strftime("%H:%M:%S")
        dur_str = f" ({entry.duration_seconds:.0f}s)" if entry.duration_seconds > 0 else ""

        if entry.entry_type == "song":
            lines.append(f"[{time_str}] SONG{dur_str}: {entry.text}")
        elif entry.entry_type == "speech":
            lines.append(f"[{time_str}] SPEECH{dur_str}: {entry.text}")
        elif entry.entry_type == "music":
            lines.append(f"[{time_str}] MUSIC{dur_str}: {entry.text}")
        elif entry.entry_type in ("silence", "noise"):
            lines.append(f"[{time_str}] {entry.text}")

    return "\n".join(lines)


def _build_fallback_timeline(radio, day, entries, tz) -> str:
    """
    If the full timeline is too long, use chunk summaries instead of raw text
    for speech segments, and keep songs as-is.
    """
    from radios.models import ChunkSummary

    # Get chunk summaries for this day
    local_start = datetime(day.year, day.month, day.day, tzinfo=tz)
    local_end = local_start + timedelta(days=1)

    chunk_map = {}
    chunks = (
        ChunkSummary.objects
        .filter(
            recording__stream__radio=radio,
            recording__start_time__gte=local_start,
            recording__start_time__lt=local_end,
        )
        .select_related("recording")
        .order_by("recording__start_time")
    )
    for cs in chunks:
        chunk_map.setdefault(cs.recording_id, []).append(cs.summary_text)

    # Build condensed timeline: chunk summaries for speech blocks, songs kept
    lines = []
    seen_recordings = set()

    for entry in entries:
        time_str = entry.absolute_time.strftime("%H:%M:%S")
        dur_str = f" ({entry.duration_seconds:.0f}s)" if entry.duration_seconds > 0 else ""

        if entry.entry_type == "song":
            lines.append(f"[{time_str}] SONG{dur_str}: {entry.text}")
        elif entry.entry_type == "speech":
            rec_id = entry.recording_id
            if rec_id is not None and rec_id in chunk_map and rec_id not in seen_recordings:
                seen_recordings.add(rec_id)
                summary = chunk_map[rec_id]
                lines.append(f"[{time_str}] SPEECH SUMMARY: {summary[:500]}")
            elif rec_id not in seen_recordings:
                lines.append(f"[{time_str}] SPEECH{dur_str}: {entry.text}")
        elif entry.entry_type == "music":
            lines.append(f"[{time_str}] MUSIC{dur_str}: {entry.text}")

    result = "\n".join(lines)

    # If still too long, truncate
    if len(result) > _MAX_TIMELINE_CHARS:
        result = result[:_MAX_TIMELINE_CHARS] + "\n... (truncated)"

    return result


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(response_text: str) -> Optional[BroadcastDayResult]:
    """Parse JSON response from the LLM into a BroadcastDayResult."""
    text = response_text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = [line for line in text.split("\n")
                 if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not parse JSON from broadcast day LLM response: %s — raw: %r",
            exc, text[:300],
        )
        return None

    overview = data.get("overview", "").strip()
    if not overview:
        logger.warning("LLM returned empty overview field.")
        return None

    raw_tags = data.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []
    tags = list(dict.fromkeys(
        t.lower().strip() for t in raw_tags
        if isinstance(t, str) and t.strip()
    ))[:_MAX_TAGS]

    shows = []
    for show_data in data.get("shows", []):
        if not isinstance(show_data, dict):
            continue
        show_type = show_data.get("type", "unknown")
        valid_types = {"music", "news", "talk", "sports", "cultural", "religious", "spot", "mixed", "unknown"}
        if show_type not in valid_types:
            show_type = "unknown"

        show_tags = show_data.get("tags", [])
        if not isinstance(show_tags, list):
            show_tags = []
        show_tags = [t.lower().strip() for t in show_tags if isinstance(t, str) and t.strip()][:10]

        show_songs = show_data.get("songs", [])
        if not isinstance(show_songs, list):
            show_songs = []
        show_songs = [str(s) for s in show_songs if s]
        shows.sort(key=lambda s: s.start_time)

        shows.append(ShowResult(
            name=show_data.get("name", "Unknown Show"),
            show_type=show_type,
            start_time=show_data.get("start_time", "00:00"),
            end_time=show_data.get("end_time", "00:00"),
            summary=show_data.get("summary", ""),
            tags=show_tags,
            songs=show_songs,
        ))

    return BroadcastDayResult(overview=overview, tags=tags, shows=shows)


# ---------------------------------------------------------------------------
# Song linking
# ---------------------------------------------------------------------------

def link_songs_to_show(show_result: ShowResult, radio, day: date,
                       show_start: datetime, show_end: datetime) -> list:
    """
    Match LLM-returned song strings to actual SongOccurrence records.
    Returns list of SongOccurrence IDs that matched.
    """
    from radios.models import SongOccurrence

    tz = ZoneInfo(radio.timezone or "UTC")
    local_start = datetime(day.year, day.month, day.day, tzinfo=tz)
    local_end = local_start + timedelta(days=1)

    # Get all song occurrences for this radio+date
    occurrences = list(
        SongOccurrence.objects
        .filter(
            segment__recording__stream__radio=radio,
            segment__recording__start_time__gte=local_start,
            segment__recording__start_time__lt=local_end,
        )
        .select_related("song", "segment__recording")
    )

    matched_ids = set()
    for song_str in show_result.songs:
        normalized = song_str.lower().strip()
        best_match = None

        for occ in occurrences:
            # Compute absolute time of this occurrence
            abs_time = occ.segment.recording.start_time + timedelta(seconds=occ.start_offset)
            abs_time_local = abs_time.astimezone(tz)

            # Check time proximity: occurrence falls within show's time range
            if not (show_start <= abs_time_local <= show_end):
                continue

            # Check string match
            artist = occ.song.artist or ""
            title = occ.song.title or ""
            occ_str = f"{title} - {artist}".lower().strip()
            occ_str_dash = f"{title} — {artist}".lower().strip()

            if (normalized in occ_str or normalized in occ_str_dash
                    or occ_str in normalized or occ_str_dash in normalized
                    or occ.song.title.lower() in normalized):
                best_match = occ
                break

        if best_match:
            matched_ids.append(best_match.pk)
        else:
            logger.debug("Unmatched song from LLM: %r", song_str)

    return matched_ids
