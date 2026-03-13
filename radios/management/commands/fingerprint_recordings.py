"""
Daemon that identifies songs in music segments via Shazam fingerprinting.

Operates at the segment level — each music segment is independently claimed
and processed. Multiple workers can fingerprint segments from the same
recording in parallel.

Uses sliding window to detect multiple songs per segment, storing results
as SongOccurrence rows.

Usage:
    python manage.py fingerprint_recordings                  # run as daemon
    python manage.py fingerprint_recordings --once           # process pending, then exit
    python manage.py fingerprint_recordings --limit 5        # cap per cycle
    python manage.py fingerprint_recordings --retry-failed   # re-queue failed segments
    python manage.py fingerprint_recordings --retry-no-match # re-queue done-but-no-songs segments
    python manage.py fingerprint_recordings --retry-skipped  # re-queue skipped segments
"""

import logging
import time
import random

from django.conf import settings
from django.db import transaction
from django.db.models import Exists, OuterRef

from radios.models import TranscriptionSegment, Song, SongOccurrence
from radios.analysis.fingerprinter import fingerprint_segment_sliding
from radios.management.commands._analysis_base import SegmentStageCommand

MIN_DURATION = 5

logger = logging.getLogger("broadcast_analysis")


class Command(SegmentStageCommand):
    help = "Identify songs in music segments via Shazam fingerprinting."

    stage_name = "fingerprinting"
    segment_types = ["music"]

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--retry-no-match",
            action="store_true",
            help=(
                "Reset fingerprinting-done segments that have no identified songs "
                "back to pending, so they are retried."
            ),
        )

    def handle(self, *args, **options):
        if options.get("retry_no_match"):
            count = (
                TranscriptionSegment.objects
                .filter(
                    segment_type="music",
                    fingerprinting_status="done",
                )
                .filter(~Exists(
                    SongOccurrence.objects.filter(segment=OuterRef("pk"))
                ))
                .update(fingerprinting_status="pending", fingerprinting_error="")
            )
            if count:
                logger.info(
                    "Reset %d no-match fingerprinting segment(s) to 'pending'.", count,
                )
        super().handle(*args, **options)

    def process_segment(self, segment, source_path, start, end, check_fn):
        duration = end - start
        if duration <= MIN_DURATION:
            logger.warning(
                "[seg %s] Skipping short segment (%.1fs)", segment.id, duration,
            )
            return

        check_fn()

        results = fingerprint_segment_sliding(source_path, start, end)

        sleep_time = getattr(settings, "FINGERPRINT_SLEEP_SECONDS", 0)
        if sleep_time:
            time.sleep(sleep_time + random.uniform(0, 0.5))

        # Clear previous occurrences (idempotent re-processing)
        with transaction.atomic():
            segment.song_occurrences.all().delete()
            occurrences = []
            for result in results:
                song = Song.get_or_create_from_fingerprint(result)
                occurrences.append(
                    SongOccurrence(
                        segment=segment,
                        song=song,
                        start_offset=result.estimated_start,
                        end_offset=result.estimated_end,
                        confidence=result.score,
                    )
                )
                logger.info(
                    "[seg %s] Fingerprinted [%.1f-%.1fs]: %s — %s (%.2f)",
                    segment.id, result.estimated_start, result.estimated_end,
                    result.artist, result.title, result.score,
                )
            SongOccurrence.objects.bulk_create(occurrences)

        if not results:
            logger.info(
                "[seg %s] Fingerprinted [%.1f-%.1fs]: no match found.",
                segment.id, start, end,
            )
