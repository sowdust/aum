"""
Management command to rebuild FTS5 virtual tables from existing data.

Usage:
    python manage.py rebuild_fts

Clears and repopulates radios_transcription_fts and radios_summary_fts
from all TranscriptionSegment, ChunkSummary, and DailySummary rows.
"""
from django.core.management.base import BaseCommand
from django.db import connection

from radios.models import TranscriptionSegment, ChunkSummary, DailySummary


class Command(BaseCommand):
    help = "Rebuild FTS5 search indexes from existing data"

    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            # --- Transcription segments ---
            self.stdout.write("Clearing radios_transcription_fts...")
            cursor.execute("DELETE FROM radios_transcription_fts")

            segments = TranscriptionSegment.objects.values_list(
                "id", "text", "text_english"
            )
            count = 0
            for seg_id, text, text_english in segments.iterator(chunk_size=500):
                cursor.execute(
                    "INSERT INTO radios_transcription_fts (segment_id, text, text_english) "
                    "VALUES (%s, %s, %s)",
                    [seg_id, text, text_english],
                )
                count += 1
            self.stdout.write(f"  Indexed {count} transcription segments.")

            # --- Chunk summaries ---
            self.stdout.write("Clearing radios_summary_fts...")
            cursor.execute("DELETE FROM radios_summary_fts")

            chunk_summaries = ChunkSummary.objects.values_list("id", "summary_text")
            count = 0
            for cs_id, summary_text in chunk_summaries.iterator(chunk_size=500):
                cursor.execute(
                    "INSERT INTO radios_summary_fts (summary_id, summary_type, summary_text) "
                    "VALUES (%s, 'chunk', %s)",
                    [cs_id, summary_text],
                )
                count += 1
            self.stdout.write(f"  Indexed {count} chunk summaries.")

            # --- Daily summaries ---
            daily_summaries = DailySummary.objects.values_list("id", "summary_text")
            count = 0
            for ds_id, summary_text in daily_summaries.iterator(chunk_size=500):
                cursor.execute(
                    "INSERT INTO radios_summary_fts (summary_id, summary_type, summary_text) "
                    "VALUES (%s, 'daily', %s)",
                    [ds_id, summary_text],
                )
                count += 1
            self.stdout.write(f"  Indexed {count} daily summaries.")

        self.stdout.write(self.style.SUCCESS("FTS rebuild complete."))
