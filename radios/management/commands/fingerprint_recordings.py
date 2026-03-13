"""
Daemon that identifies songs in music segments via Shazam fingerprinting.

Upstream: segmentation must be done/skipped.
Independent of transcription — can run in parallel.

Uses sliding window to detect multiple songs per segment, storing results
as SongOccurrence rows.

Usage:
    python manage.py fingerprint_recordings                  # run as daemon
    python manage.py fingerprint_recordings --once           # process pending, then exit
    python manage.py fingerprint_recordings --limit 5        # cap per cycle
    python manage.py fingerprint_recordings --retry-failed   # re-queue failed recordings
    python manage.py fingerprint_recordings --retry-no-match # re-queue done-but-no-songs recordings
    python manage.py fingerprint_recordings --retry-skipped  # re-queue skipped recordings
"""

import logging

from radios.models import Recording, Song, SongOccurrence
from radios.analysis.fingerprinter import fingerprint_segment_sliding
from radios.management.commands._analysis_base import AnalysisStageCommand
import os
import time
import random
from django.conf import settings
from django.db import transaction
from django.db.models import Exists, OuterRef

MIN_DURATION = 5

logger = logging.getLogger("broadcast_analysis")


class Command(AnalysisStageCommand):
    help = "Identify songs in music segments via Shazam fingerprinting."

    stage_name = "fingerprinting"
    upstream_done_fields = ["segmentation_status"]

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--retry-no-match",
            action="store_true",
            help=(
                "Reset fingerprinting-done recordings that have no identified songs "
                "back to pending, so they are retried."
            ),
        )

    def handle(self, *args, **options):
        if options["retry_no_match"]:
            count = (
                Recording.objects
                .filter(fingerprinting_status="done")
                .filter(~Exists(SongOccurrence.objects.filter(segment__recording=OuterRef("pk"))))
                .update(fingerprinting_status="pending", fingerprinting_error="")
            )
            if count:
                logger.info(
                    "Reset %d no-match fingerprinting recording(s) to 'pending'.", count,
                )
        super().handle(*args, **options)

    def process_one(self, recording, file_path, check_fn):
        music_segments = list(
            recording.segments.filter(segment_type="music")
            .prefetch_related("song_occurrences")
        )
        logger.info(
            "[%s] Fingerprinting %d music segment(s)...",
            recording.id, len(music_segments),
        )

        for seg in music_segments:
            check_fn()
            duration = seg.end_offset - seg.start_offset
            if duration <= MIN_DURATION:
                logger.warning("[%s] Skipping invalid (empty) segment %s", recording.id, seg.id)
                continue
            # Prefer per-segment file (real-time pipeline) over recording file + offsets
            if seg.file and seg.file.name:
                if os.path.exists(seg.file.path):
                    logger.error("File path does not exist %s" % seg.file.path)
                source_path = seg.file.path
                start = 0
                end = seg.end_offset - seg.start_offset
            else:
                source_path = file_path
                start = seg.start_offset
                end = seg.end_offset
            results = fingerprint_segment_sliding(
                source_path, start, end,
            )
            sleep_time = getattr(settings, "FINGERPRINT_SLEEP_SECONDS", 0)
            if sleep_time:
                time.sleep(sleep_time + random.uniform(0, 0.5))

            # Clear previous occurrences (idempotent re-processing)
            with transaction.atomic():
                seg.song_occurrences.all().delete()
                occurrences = []
                for result in results:
                    song = Song.get_or_create_from_fingerprint(result)
                    occurrences.append(
                        SongOccurrence(
                            segment=seg,
                            song=song,
                            start_offset=result.estimated_start,
                            end_offset=result.estimated_end,
                            confidence=result.score,
                        )
                    )
                    logger.info(
                        "[%s] Fingerprinted [%.1f-%.1fs]: %s — %s (%.2f)",
                        recording.id, result.estimated_start, result.estimated_end,
                        result.artist, result.title, result.score,
                    )
                SongOccurrence.objects.bulk_create(occurrences)

            if not results:
                logger.info(
                    "[%s] Fingerprinted [%.1f-%.1fs]: no match found.",
                    recording.id, seg.start_offset, seg.end_offset,
                )
