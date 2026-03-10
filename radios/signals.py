"""
Django signals to keep FTS5 virtual tables in sync with model changes.

Registered in RadiosConfig.ready().
"""
import logging

from django.db import connection
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

from radios.models import TranscriptionSegment, ChunkSummary, DailySummary

logger = logging.getLogger("broadcast_analysis")


def _fts_execute(sql, params):
    """Execute raw SQL, logging errors without crashing the save."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
    except Exception:
        logger.exception("FTS sync error: %s", sql)


# --- TranscriptionSegment ---

@receiver(post_save, sender=TranscriptionSegment)
def sync_transcription_fts_on_save(sender, instance, **kwargs):
    # DELETE then INSERT (FTS5 doesn't support UPDATE)
    _fts_execute(
        "DELETE FROM radios_transcription_fts WHERE segment_id = %s",
        [instance.id],
    )
    _fts_execute(
        "INSERT INTO radios_transcription_fts (segment_id, text, text_english) VALUES (%s, %s, %s)",
        [instance.id, instance.text, instance.text_english],
    )


@receiver(post_delete, sender=TranscriptionSegment)
def sync_transcription_fts_on_delete(sender, instance, **kwargs):
    _fts_execute(
        "DELETE FROM radios_transcription_fts WHERE segment_id = %s",
        [instance.id],
    )


# --- ChunkSummary ---

@receiver(post_save, sender=ChunkSummary)
def sync_chunk_summary_fts_on_save(sender, instance, **kwargs):
    _fts_execute(
        "DELETE FROM radios_summary_fts WHERE summary_id = %s AND summary_type = 'chunk'",
        [instance.id],
    )
    _fts_execute(
        "INSERT INTO radios_summary_fts (summary_id, summary_type, summary_text) VALUES (%s, 'chunk', %s)",
        [instance.id, instance.summary_text],
    )


@receiver(post_delete, sender=ChunkSummary)
def sync_chunk_summary_fts_on_delete(sender, instance, **kwargs):
    _fts_execute(
        "DELETE FROM radios_summary_fts WHERE summary_id = %s AND summary_type = 'chunk'",
        [instance.id],
    )


# --- DailySummary ---

@receiver(post_save, sender=DailySummary)
def sync_daily_summary_fts_on_save(sender, instance, **kwargs):
    _fts_execute(
        "DELETE FROM radios_summary_fts WHERE summary_id = %s AND summary_type = 'daily'",
        [instance.id],
    )
    _fts_execute(
        "INSERT INTO radios_summary_fts (summary_id, summary_type, summary_text) VALUES (%s, 'daily', %s)",
        [instance.id, instance.summary_text],
    )


@receiver(post_delete, sender=DailySummary)
def sync_daily_summary_fts_on_delete(sender, instance, **kwargs):
    _fts_execute(
        "DELETE FROM radios_summary_fts WHERE summary_id = %s AND summary_type = 'daily'",
        [instance.id],
    )
