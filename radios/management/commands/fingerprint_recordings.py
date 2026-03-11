"""
Daemon that identifies songs in music segments via Shazam fingerprinting.

Upstream: segmentation must be done/skipped.
Independent of transcription — can run in parallel.

Usage:
    python manage.py fingerprint_recordings            # run as daemon
    python manage.py fingerprint_recordings --once     # process pending, then exit
    python manage.py fingerprint_recordings --limit 5  # cap per cycle
"""

import logging

from radios.models import Song
from radios.analysis.fingerprinter import fingerprint_segment
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
            result = fingerprint_segment(
                file_path, seg.start_offset, seg.end_offset,
            )
            if result:
                song = Song.get_or_create_from_fingerprint(result)
                seg.song = song
                seg.confidence = result.score
                seg.save(update_fields=["song", "confidence"])
                logger.info(
                    "[%s] Fingerprinted [%.1f-%.1fs]: %s — %s (%.2f)",
                    recording.id, seg.start_offset, seg.end_offset,
                    result.artist, result.title, result.score,
                )
            else:
                logger.info(
                    "[%s] Fingerprinted [%.1f-%.1fs]: no match found.",
                    recording.id, seg.start_offset, seg.end_offset,
                )
