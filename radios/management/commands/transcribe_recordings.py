"""
Daemon that transcribes speech segments and optionally applies LLM correction.

Upstream: segmentation must be done/skipped.
Independent of fingerprinting — can run in parallel.
Correction is tightly coupled and always runs within this stage.

Usage:
    python manage.py transcribe_recordings            # run as daemon
    python manage.py transcribe_recordings --once     # process pending, then exit
    python manage.py transcribe_recordings --limit 5  # cap per cycle
"""

import logging

from radios.models import TranscriptionSettings
from radios.analysis.transcriber import transcribe_segment
from radios.analysis.corrector import correct_transcription
from radios.management.commands._analysis_base import AnalysisStageCommand

logger = logging.getLogger("broadcast_analysis")


class Command(AnalysisStageCommand):
    help = "Transcribe speech segments to text, with optional LLM correction."

    stage_name = "transcription"
    upstream_done_fields = ["segmentation_status"]

    def process_one(self, recording, file_path, check_fn):
        self._run_transcription(recording, file_path, check_fn)
        self._run_correction(recording, check_fn)

    def _run_transcription(self, recording, file_path, check_fn):
        source = recording.stream.source
        language_hint = getattr(source, "languages", "") or ""

        speech_segments = list(
            recording.segments.filter(
                segment_type__in=["speech", "speech_over_music"]
            )
        )
        logger.info(
            "[%s] Transcribing %d speech segment(s)...",
            recording.id, len(speech_segments),
        )

        for seg in speech_segments:
            check_fn()
            result = transcribe_segment(
                file_path, seg.start_offset, seg.end_offset,
                language_hint=language_hint,
            )
            if result:
                seg.text = result.text
                seg.text_english = result.text_english
                seg.language = result.language
                seg.confidence = result.confidence
                seg.save(update_fields=[
                    "text", "text_english", "language", "confidence",
                ])
                logger.info(
                    "[%s] Transcribed [%.1f-%.1fs]: lang=%s, %d chars",
                    recording.id, seg.start_offset, seg.end_offset,
                    result.language, len(result.text),
                )

    def _run_correction(self, recording, check_fn):
        cfg = TranscriptionSettings.get_settings()
        if not cfg.enable_correction:
            return

        speech_segments = list(
            recording.segments
            .filter(segment_type__in=["speech", "speech_over_music"])
            .exclude(text="")
            .order_by("start_offset")
        )

        if not speech_segments:
            logger.info("[%s] No speech text to correct.", recording.id)
            return

        segments_data = [
            {"index": i, "text": seg.text}
            for i, seg in enumerate(speech_segments)
        ]

        source = recording.stream.source
        radio_name = getattr(source, "name", "")
        radio_location = ""
        city = getattr(source, "city", "")
        country = getattr(source, "country", "")
        if city and country:
            radio_location = f"{city}, {country}"
        elif city or country:
            radio_location = city or str(country)
        radio_language = getattr(source, "languages", "") or ""

        check_fn()
        logger.info(
            "[%s] Running transcription correction on %d segment(s)...",
            recording.id, len(speech_segments),
        )

        corrections = correct_transcription(
            segments_data,
            radio_name=radio_name,
            radio_location=radio_location,
            radio_language=radio_language,
        )

        if not corrections:
            logger.warning("[%s] Correction returned no results.", recording.id)
            return

        correction_map = {c["index"]: c for c in corrections}

        corrected_count = 0
        for i, seg in enumerate(speech_segments):
            if i in correction_map:
                c = correction_map[i]
                seg.text_original = seg.text
                seg.text = c["text"]
                seg.text_english = c["text_english"]
                seg.save(update_fields=["text", "text_original", "text_english"])
                corrected_count += 1

        logger.info(
            "[%s] Corrected %d/%d speech segment(s).",
            recording.id, corrected_count, len(speech_segments),
        )
