"""
Daemon that identifies songs in music segments via Shazam fingerprinting.

Upstream: segmentation must be done/skipped.
Independent of transcription — can run in parallel.

Uses sliding window to detect multiple songs per segment, storing results
as SongOccurrence rows.

Usage:
    python manage.py fingerprint_recordings            # run as daemon
    python manage.py fingerprint_recordings --once     # process pending, then exit
    python manage.py fingerprint_recordings --limit 5  # cap per cycle
"""

import logging

from radios.models import Song, SongOccurrence
from radios.analysis.fingerprinter import fingerprint_segment_sliding
from radios.management.commands._analysis_base import AnalysisStageCommand

logger = logging.getLogger("broadcast_analysis")


class Command(AnalysisStageCommand):
    help = "Identify songs in music segments via Shazam fingerprinting."

    stage_name = "fingerprinting"
    upstream_done_fields = ["segmentation_status"]

    def process_one(self, recording, file_path, check_fn):
        music_segments = list(recording.segments.filter(segment_type="music"))
        logger.info(
            "[%s] Fingerprinting %d music segment(s)...",
            recording.id, len(music_segments),
        )

        for seg in music_segments:
            check_fn()
            results = fingerprint_segment_sliding(
                file_path, seg.start_offset, seg.end_offset,
            )

            # Clear previous occurrences (idempotent re-processing)
            seg.song_occurrences.all().delete()

            for result in results:
                song = Song.get_or_create_from_fingerprint(result)
                SongOccurrence.objects.create(
                    segment=seg,
                    song=song,
                    start_offset=result.estimated_start,
                    end_offset=result.estimated_end,
                    confidence=result.score,
                )
                logger.info(
                    "[%s] Fingerprinted [%.1f-%.1fs]: %s — %s (%.2f)",
                    recording.id, result.estimated_start, result.estimated_end,
                    result.artist, result.title, result.score,
                )

            if not results:
                logger.info(
                    "[%s] Fingerprinted [%.1f-%.1fs]: no match found.",
                    recording.id, seg.start_offset, seg.end_offset,
                )
