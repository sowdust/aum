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
import os

from django.conf import settings as django_settings

from radios.models import TranscriptionSettings
from radios.analysis.transcriber import transcribe_segment, transcribe_runpod_batch
from radios.analysis.transcriber import _extract_audio_slice  # noqa: PLC2701 (internal helper reuse)
from radios.analysis.corrector import correct_transcription
from radios.management.commands._analysis_base import AnalysisStageCommand

logger = logging.getLogger("broadcast_analysis")


class Command(AnalysisStageCommand):
    help = "Transcribe speech segments to text, with optional LLM correction."

    stage_name = "transcription"
    upstream_done_fields = ["segmentation_status"]

    def process_one(self, recording, file_path, check_fn):
        cfg = TranscriptionSettings.get_settings()
        if cfg.backend == "runpod":
            self._run_transcription_runpod(recording, file_path, check_fn)
        else:
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
            # Prefer per-segment file (real-time pipeline) over recording file + offsets
            if seg.file and seg.file.name:
                source_path = seg.file.path
                start = 0
                end = seg.end_offset - seg.start_offset
            else:
                source_path = file_path
                start = seg.start_offset
                end = seg.end_offset
            result = transcribe_segment(
                source_path, start, end,
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

    def _run_transcription_runpod(self, recording, file_path, check_fn):
        """
        Transcribe all speech segments for a recording using RunPod serverless,
        submitting up to _RUNPOD_MAX_PARALLEL jobs in parallel.
        """
        source = recording.stream.source
        language_hint = getattr(source, "languages", "") or ""

        speech_segments = list(
            recording.segments.filter(
                segment_type__in=["speech", "speech_over_music"]
            )
        )
        logger.info(
            "[%s] RunPod: transcribing %d speech segment(s) in parallel...",
            recording.id, len(speech_segments),
        )

        check_fn()

        # Build per-segment WAV files and the segments_data list for the batch call.
        # We track temp paths so they can be cleaned up in the finally block.
        segments_data = []
        temp_paths = []
        try:
            for idx, seg in enumerate(speech_segments):
                if seg.file and seg.file.name:
                    source_path = seg.file.path
                    start = 0.0
                    end = seg.end_offset - seg.start_offset
                else:
                    source_path = file_path
                    start = seg.start_offset
                    end = seg.end_offset

                # Extract a WAV slice for the RunPod worker
                audio_path = _extract_audio_slice(source_path, start, end, fmt="wav")
                if audio_path is None:
                    logger.warning(
                        "[%s] RunPod: could not extract audio for segment idx=%d, skipping",
                        recording.id, idx,
                    )
                    continue
                temp_paths.append(audio_path)

                # Build a public URL if the segment has its own media file
                audio_url = ""
                if seg.file and seg.file.name:
                    media_url = getattr(django_settings, "MEDIA_URL", "/media/")
                    audio_url = media_url.rstrip("/") + "/" + seg.file.name.lstrip("/")

                segments_data.append({
                    "idx": idx,
                    "audio_path": audio_path,
                    "language_hint": language_hint,
                    "audio_url": audio_url,
                })

            if not segments_data:
                logger.info("[%s] RunPod: no segments to transcribe.", recording.id)
                return

            results = transcribe_runpod_batch(segments_data)

            for item in segments_data:
                idx = item["idx"]
                result = results.get(idx)
                if result:
                    seg = speech_segments[idx]
                    seg.text = result.text
                    seg.text_english = result.text_english
                    seg.language = result.language
                    seg.confidence = result.confidence
                    seg.save(update_fields=["text", "text_english", "language", "confidence"])
                    logger.info(
                        "[%s] RunPod: transcribed segment idx=%d lang=%s %d chars",
                        recording.id, idx, result.language, len(result.text),
                    )
        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

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
