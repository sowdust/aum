"""
Daemon that transcribes speech segments and optionally triggers batch correction.

Operates at the segment level — each speech segment is independently claimed
and transcribed. After marking a segment done, checks for correction batch
eligibility: N consecutive transcribed-but-uncorrected speech segments from
the same stream (N is configurable via TranscriptionSettings.correction_batch_size).

Usage:
    python manage.py transcribe_recordings            # run as daemon
    python manage.py transcribe_recordings --once     # process pending, then exit
    python manage.py transcribe_recordings --limit 5  # cap per cycle
"""

import logging
import os

from django.conf import settings as django_settings
from django.db.models import F

from radios.models import TranscriptionSegment, TranscriptionSettings
from radios.analysis.transcriber import transcribe_segment, transcribe_runpod_batch
from radios.analysis.transcriber import _extract_audio_slice  # noqa: PLC2701
from radios.analysis.corrector import correct_transcription
from radios.management.commands._analysis_base import SegmentStageCommand

logger = logging.getLogger("broadcast_analysis")


class Command(SegmentStageCommand):
    help = "Transcribe speech segments to text, with optional batch LLM correction."

    stage_name = "transcription"
    segment_types = ["speech", "speech_over_music"]

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--correction-batch-size",
            type=int,
            default=0,
            metavar="N",
            help=(
                "Override correction_batch_size from TranscriptionSettings. "
                "0 = use the DB setting."
            ),
        )

    def handle(self, *args, **options):
        self._correction_batch_size_override = options.get("correction_batch_size", 0)
        super().handle(*args, **options)

    def process_segment(self, segment, source_path, start, end, check_fn):
        cfg = TranscriptionSettings.get_settings()

        if cfg.backend == "runpod":
            self._transcribe_runpod(segment, source_path, start, end, check_fn)
        else:
            self._transcribe_single(segment, source_path, start, end, check_fn)

    def _transcribe_single(self, segment, source_path, start, end, check_fn):
        source = segment.recording.stream.source
        language_hint = getattr(source, "languages", "") or ""

        check_fn()
        result = transcribe_segment(
            source_path, start, end, language_hint=language_hint,
        )
        if result:
            segment.text = result.text
            segment.text_english = result.text_english
            segment.language = result.language
            segment.confidence = result.confidence
            segment.save(update_fields=[
                "text", "text_english", "language", "confidence",
            ])
            logger.info(
                "[seg %s] Transcribed [%.1f-%.1fs]: lang=%s, %d chars",
                segment.id, segment.start_offset, segment.end_offset,
                result.language, len(result.text),
            )

            # After transcription, check for correction batch
            self._try_correction_batch(segment, check_fn)

    def _transcribe_runpod(self, segment, source_path, start, end, check_fn):
        """Transcribe a single segment using RunPod serverless."""
        source = segment.recording.stream.source
        language_hint = getattr(source, "languages", "") or ""

        check_fn()

        audio_path = _extract_audio_slice(source_path, start, end, fmt="wav")
        if audio_path is None:
            logger.warning(
                "[seg %s] RunPod: could not extract audio, skipping", segment.id,
            )
            return

        try:
            audio_url = ""
            if segment.file and segment.file.name:
                media_url = getattr(django_settings, "MEDIA_URL", "/media/")
                audio_url = media_url.rstrip("/") + "/" + segment.file.name.lstrip("/")

            segments_data = [{
                "idx": 0,
                "audio_path": audio_path,
                "language_hint": language_hint,
                "audio_url": audio_url,
            }]

            results = transcribe_runpod_batch(segments_data)
            result = results.get(0)

            if result:
                segment.text = result.text
                segment.text_english = result.text_english
                segment.language = result.language
                segment.confidence = result.confidence
                segment.save(update_fields=[
                    "text", "text_english", "language", "confidence",
                ])
                logger.info(
                    "[seg %s] RunPod: transcribed lang=%s %d chars",
                    segment.id, result.language, len(result.text),
                )

                # After transcription, check for correction batch
                self._try_correction_batch(segment, check_fn)
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    def _try_correction_batch(self, segment, check_fn):
        """
        After a segment is transcribed, check if we have enough consecutive
        transcribed-but-uncorrected speech segments from the same stream
        to run a correction batch.
        """
        cfg = TranscriptionSettings.get_settings()
        if not cfg.enable_correction:
            # Mark this segment as correction skipped
            TranscriptionSegment.objects.filter(pk=segment.pk).update(
                correction_status="skipped"
            )
            return

        batch_size = self._correction_batch_size_override or cfg.correction_batch_size

        stream = segment.recording.stream

        # Find consecutive transcribed-but-uncorrected speech segments
        # from the same stream, ordered chronologically
        candidates = list(
            TranscriptionSegment.objects
            .filter(
                recording__stream=stream,
                segment_type__in=["speech", "speech_over_music"],
                transcription_status="done",
                correction_status="pending",
            )
            .exclude(text="")
            .order_by("recording__start_time", "start_offset")
            [:batch_size]
        )

        if len(candidates) < batch_size:
            return  # not enough segments yet

        # Run correction on this batch
        check_fn()

        source = stream.source
        radio_name = getattr(source, "name", "")
        city = getattr(source, "city", "")
        country = getattr(source, "country", "")
        radio_location = (
            f"{city}, {country}" if city and country
            else (city or str(country) if city or country else "")
        )
        radio_language = getattr(source, "languages", "") or ""

        segments_data = [
            {"index": i, "text": seg.text}
            for i, seg in enumerate(candidates)
        ]

        logger.info(
            "[stream %s] Running correction on batch of %d segment(s)...",
            stream.name, len(candidates),
        )

        corrections = correct_transcription(
            segments_data,
            radio_name=radio_name,
            radio_location=radio_location,
            radio_language=radio_language,
        )

        if not corrections:
            logger.warning(
                "[stream %s] Correction returned no results.", stream.name,
            )
            return

        correction_map = {c["index"]: c for c in corrections}

        corrected_count = 0
        for i, seg in enumerate(candidates):
            if i in correction_map:
                c = correction_map[i]
                seg.text_original = seg.text
                seg.text = c["text"]
                seg.text_english = c["text_english"]
                seg.correction_status = "done"
                seg.save(update_fields=[
                    "text", "text_original", "text_english", "correction_status",
                ])
                corrected_count += 1
            else:
                # No correction returned for this segment — mark done anyway
                TranscriptionSegment.objects.filter(pk=seg.pk).update(
                    correction_status="done"
                )

        logger.info(
            "[stream %s] Corrected %d/%d segment(s) in batch.",
            stream.name, corrected_count, len(candidates),
        )
