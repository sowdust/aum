"""
Daemon that segments pending recordings into speech/music/noise segments.

First stage of the analysis pipeline — no upstream dependencies.

When creating segments, sets initial per-segment pipeline statuses based on
segment type and stream configuration:
- Music segments: fingerprinting_status=pending (or skipped if disabled)
- Speech segments: transcription_status=pending (or skipped if disabled),
                   correction_status=pending (or skipped if correction disabled)
- Other types: all statuses skipped

Usage:
    python manage.py segment_recordings            # run as daemon
    python manage.py segment_recordings --once     # process pending, then exit
    python manage.py segment_recordings --limit 5  # cap per cycle
"""

import logging

from radios.models import Recording, TranscriptionSegment, TranscriptionSettings
from radios.management.commands._analysis_base import AnalysisStageCommand
from django.db import transaction

logger = logging.getLogger("broadcast_analysis")


def _initial_segment_statuses(segment_type, stream):
    """
    Return a dict of initial status fields for a new TranscriptionSegment
    based on its type and stream configuration.
    """
    fp_active = stream.is_stage_active("fingerprinting")
    tx_active = stream.is_stage_active("transcription")
    correction_enabled = TranscriptionSettings.get_settings().enable_correction

    if segment_type == "music":
        return {
            "fingerprinting_status": "pending" if fp_active else "skipped",
            "transcription_status": "skipped",
            "correction_status": "skipped",
        }
    elif segment_type in ("speech", "speech_over_music"):
        return {
            "fingerprinting_status": "skipped",
            "transcription_status": "pending" if tx_active else "skipped",
            "correction_status": "pending" if (tx_active and correction_enabled) else "skipped",
        }
    else:
        # noise, noEnergy, silence, unknown
        return {
            "fingerprinting_status": "skipped",
            "transcription_status": "skipped",
            "correction_status": "skipped",
        }


class Command(AnalysisStageCommand):
    help = "Segment pending recordings into speech/music/noise blocks."

    stage_name = "segmentation"
    upstream_done_fields = []  # first stage — no upstream

    def process_one(self, recording, file_path, check_fn):
        from radios.analysis.segmenter import segment_audio

        check_fn()

        # Session recordings are already segmented inline by StreamProcessor
        if recording.is_session:
            logger.info("[%s] Session recording — already segmented inline.", recording.id)
            return

        logger.info("[%s] Running segmentation...", recording.id)

        try:
            audio_segments = segment_audio(file_path)
        except Exception:
            logger.exception("[%s] Segmentation crashed.", recording.id)
            return

        stream = recording.stream

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
                        **_initial_segment_statuses(seg.segment_type, stream),
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

            # Skip summarization if inactive
            if not stream.is_stage_active("summarization"):
                Recording.objects.filter(
                    pk=recording.pk, summarization_status="pending"
                ).update(summarization_status="skipped")
