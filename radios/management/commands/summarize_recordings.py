"""
Daemon that summarizes transcribed recordings into ChunkSummary + DailySummary.

Upstream: transcription must be done/skipped.
Does NOT wait for fingerprinting — summarization only needs transcription output.

Usage:
    python manage.py summarize_recordings            # run as daemon
    python manage.py summarize_recordings --once     # process pending, then exit
    python manage.py summarize_recordings --limit 5  # cap per cycle
"""

import logging

from django.db.models import Q

from radios.models import Recording, ChunkSummary, DailySummary, Tag
from radios.analysis.summarizer import summarize_texts
from radios.management.commands._analysis_base import AnalysisStageCommand

logger = logging.getLogger("broadcast_analysis")


class Command(AnalysisStageCommand):
    help = "Summarize transcribed recordings and generate daily summaries."

    stage_name = "summarization"
    upstream_done_fields = ["transcription_status"]

    def process_one(self, recording, file_path, check_fn):
        self._run_chunk_summary(recording, check_fn)
        self._try_daily_summary(recording, check_fn)

    def _run_chunk_summary(self, recording, check_fn):
        source = recording.stream.source
        language_hint = getattr(source, "languages", "") or ""

        texts = list(
            recording.segments
            .filter(segment_type__in=["speech", "speech_over_music"])
            .exclude(Q(text="") | Q(text__isnull=True))
            .values_list("text", flat=True)
        )

        if not texts:
            logger.info("[%s] No transcribed text to summarize.", recording.id)
            return

        check_fn()
        logger.info("[%s] Summarizing %d text segment(s)...", recording.id, len(texts))

        result = summarize_texts(texts, language_hint=language_hint)
        if not result:
            logger.warning("[%s] Summarizer returned no result.", recording.id)
            return

        chunk_summary, created = ChunkSummary.objects.update_or_create(
            recording=recording,
            defaults={"summary_text": result.summary_text},
        )

        tag_objects = [Tag.get_or_create_normalized(name)[0] for name in result.tags]
        chunk_summary.tags.set(tag_objects)

        action = "Created" if created else "Updated"
        logger.info(
            "[%s] %s chunk summary (%d chars, %d tags).",
            recording.id, action, len(result.summary_text), len(tag_objects),
        )

    def _try_daily_summary(self, recording, check_fn):
        """Generate a daily summary if all recordings for this radio+date are done."""
        stream = recording.stream
        radio = stream.radio
        if not radio:
            return  # AudioFeed — no daily summary

        rec_date = recording.start_time.date()

        # Check if all recordings for this radio+date have summarization done/skipped
        all_recs = Recording.objects.filter(
            stream__radio=radio,
            start_time__date=rec_date,
        )
        if all_recs.exclude(summarization_status__in=["done", "skipped"]).exists():
            return  # not all done yet

        # Gather all chunk summaries for this radio+date
        chunk_summaries = list(
            ChunkSummary.objects
            .filter(
                recording__stream__radio=radio,
                recording__start_time__date=rec_date,
            )
            .order_by("recording__start_time")
            .values_list("summary_text", flat=True)
        )

        if not chunk_summaries:
            return

        check_fn()
        logger.info(
            "[%s] Generating daily summary for %s on %s (%d chunks)...",
            recording.id, radio.name, rec_date, len(chunk_summaries),
        )

        from radios.analysis.summarizer import summarize_daily_texts
        result = summarize_daily_texts(chunk_summaries)
        if not result:
            logger.warning(
                "[%s] Daily summarizer returned no result.", recording.id,
            )
            return

        daily, created = DailySummary.objects.update_or_create(
            radio=radio,
            date=rec_date,
            defaults={
                "summary_text": result.summary_text,
                "chunk_count": len(chunk_summaries),
            },
        )

        tag_objects = [Tag.get_or_create_normalized(name)[0] for name in result.tags]
        daily.tags.set(tag_objects)

        action = "Created" if created else "Updated"
        logger.info(
            "[%s] %s daily summary for %s — %s (%d chars, %d tags).",
            recording.id, action, radio.name, rec_date,
            len(result.summary_text), len(tag_objects),
        )
