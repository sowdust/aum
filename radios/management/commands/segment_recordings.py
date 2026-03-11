"""
Daemon that segments pending recordings into speech/music/noise segments.

First stage of the analysis pipeline — no upstream dependencies.

Usage:
    python manage.py segment_recordings            # run as daemon
    python manage.py segment_recordings --once     # process pending, then exit
    python manage.py segment_recordings --limit 5  # cap per cycle
"""

import logging

from radios.models import Recording, TranscriptionSegment
from radios.management.commands._analysis_base import AnalysisStageCommand
from django.db import transaction

logger = logging.getLogger("broadcast_analysis")


class Command(AnalysisStageCommand):
    help = "Segment pending recordings into speech/music/noise blocks."

    stage_name = "segmentation"
    upstream_done_fields = []  # first stage — no upstream

    def process_one(self, recording, file_path, check_fn):
        from radios.analysis.segmenter import segment_audio

        check_fn()
        logger.info("[%s] Running segmentation...", recording.id)

        try:
            audio_segments = segment_audio(file_path)
        except Exception:
            logger.exception("[%s] Segmentation crashed.", recording.id)
            return

        # Clear stale segments from a previous interrupted run
        with transaction.atomic():
            recording.segments.all().delete()

            if audio_segments:
                TranscriptionSegment.objects.bulk_create([
                    TranscriptionSegment(
                        recording=recording,
                        segment_type=seg.segment_type,
                        start_offset=seg.start,
                        end_offset=seg.end,
                    )
                    for seg in audio_segments
                ], batch_size=500,
                )
                logger.info(
                    "[%s] Segmentation done: %d segments.",
                    recording.id, len(audio_segments),
                )
            else:
                logger.warning("[%s] Segmentation returned no segments.", recording.id)

            # Check downstream stages — skip inactive ones now
            stream = recording.stream
            for stage, field in [
                ("fingerprinting", "fingerprinting_status"),
                ("transcription", "transcription_status"),
                ("summarization", "summarization_status"),
            ]:
                if not stream.is_stage_active(stage):
                    Recording.objects.filter(
                        pk=recording.pk, **{field: "pending"}
                    ).update(**{field: "skipped"})
