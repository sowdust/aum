"""
Daemon that generates BroadcastDaySummary records for radio stations.

Finds radio+date pairs where all recordings have transcription done/skipped,
then calls the broadcast day summarizer to reconstruct the day.

Usage:
    python manage.py summarize_broadcast_days              # run as daemon
    python manage.py summarize_broadcast_days --once       # process pending, then exit
    python manage.py summarize_broadcast_days --limit 5    # cap per cycle
    python manage.py summarize_broadcast_days --radio <slug> --date 2026-03-10
    python manage.py summarize_broadcast_days --force      # re-process done days
    python manage.py summarize_broadcast_days --retry-failed
"""

import signal
import time
import traceback
import logging
from datetime import date, datetime, timedelta
from logging import shutdown
from zoneinfo import ZoneInfo

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import models
from django.db.models import Count, Q
from django.utils import timezone

from radios.models import (
    Radio, Recording, Stream, TranscriptionSegment,
    BroadcastDaySummary, ShowBlock, Tag, SongOccurrence,
)
from radios.analysis.broadcast_day_summarizer import (
    summarize_broadcast_day, link_songs_to_show,
)

logger = logging.getLogger("broadcast_analysis")


class Command(BaseCommand):
    help = "Generate daily broadcast summaries for radios with completed transcription."

    def add_arguments(self, parser):
        parser.add_argument(
            "--once", action="store_true",
            help="Process all eligible radio+date pairs once, then exit.",
        )
        parser.add_argument(
            "--limit", type=int, default=0, metavar="N",
            help="Maximum radio+date pairs to process per cycle (0 = unlimited).",
        )
        parser.add_argument(
            "--retry-failed", action="store_true",
            help="Reset failed BroadcastDaySummary records back to pending.",
        )
        parser.add_argument(
            "--radio", type=str, default="",
            help="Only process a specific radio (by slug).",
        )
        parser.add_argument(
            "--date", type=str, default="",
            help="Only process a specific date (YYYY-MM-DD).",
        )
        parser.add_argument(
            "--force", action="store_true",
            help="Re-process even if a done BroadcastDaySummary already exists.",
        )

    def handle(self, *args, **options):
        once = options["once"]
        limit = options["limit"]
        retry_failed = options["retry_failed"]
        radio_slug = options["radio"]
        target_date = options["date"]
        force = options["force"]
        poll_interval = getattr(settings, "ANALYZE_POLL_INTERVAL", 30)

        if retry_failed and force:
            logger.error("[!] Using both --retry-failed and --force will cause an infinite loop")

        self._running = True

        def shutdown(signum, frame):
            if not self._running:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                self.stdout.write(
                    "\nForce shutdown requested. Press Ctrl+C once more to kill."
                )
                return
            self._running = False
            self.stdout.write(
                "\nShutdown signal received — finishing current work."
            )
            logger.info("Shutdown signal (%s) received for broadcast day summarizer.", signum)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Stale claim recovery
        stale = BroadcastDaySummary.objects.filter(status="running").update(
            status="pending", error=""
        )
        if stale:
            logger.info("Reset %d stale 'running' broadcast day(s) to 'pending'.", stale)

        if retry_failed:
            retried = BroadcastDaySummary.objects.filter(status="failed").update(
                status="pending", error=""
            )
            if retried:
                logger.info("Reset %d failed broadcast day(s) to 'pending'.", retried)

        # Parse target date
        parsed_date = None
        if target_date:
            try:
                parsed_date = date.fromisoformat(target_date)
            except ValueError:
                self.stderr.write(f"Invalid date format: {target_date}. Use YYYY-MM-DD.")
                return

        logger.info(
            "Broadcast day summarizer starting (once=%s, limit=%s, poll=%ss)",
            once, limit or "unlimited", poll_interval,
        )

        try:
            while self._running:
                processed = self._process_cycle(
                    limit, radio_slug, parsed_date, force,
                )
                if once or not self._running:
                    break
                if not processed:
                    deadline = time.monotonic() + poll_interval
                    while self._running and time.monotonic() < deadline:
                        time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("Force killed by KeyboardInterrupt.")

        logger.info("Broadcast day summarizer exited.")

    def _process_cycle(self, limit, radio_slug, target_date, force):
        """Find and process eligible radio+date candidates. Returns count."""
        candidates = self._find_candidates(radio_slug, target_date, force)

        if limit:
            candidates = candidates[:limit]

        if candidates:
            logger.info("Found %d broadcast day candidate(s).", len(candidates))

        processed = 0
        for radio, day in candidates:
            if not self._running:
                break
            self._process_one(radio, day, force)
            processed += 1

        return processed

    def _find_candidates(self, radio_slug, target_date, force):
        """
        Find radio+date pairs ready for broadcast day summarization.

        A pair is ready when ALL recordings for that radio+date have
        transcription_status in (done, skipped).

        Only considers past days (in the radio's local timezone) unless a
        specific --date was given — today's broadcast is still in progress.
        Failed candidates are skipped (use --retry-failed to reset them).
        """
        radio_qs = Radio.objects.filter(slug=radio_slug) if radio_slug else Radio.objects.all()

        candidates = []

        for radio in radio_qs:
            # Check if any stream has daily_summarization active
            has_active = any(
                stream.is_stage_active("daily_summarization")
                for stream in radio.streams.filter(is_active=True)
            )
            if not has_active and not radio_slug:
                continue

            tz = ZoneInfo(radio.timezone or "UTC")
            today_local = timezone.now().astimezone(tz).date()

            # Find dates with recordings
            rec_qs = Recording.objects.filter(stream__radio=radio)
            if target_date:
                local_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=tz)
                local_end = local_start + timedelta(days=1)
                rec_qs = rec_qs.filter(start_time__gte=local_start, start_time__lt=local_end)

            dates = (
                rec_qs
                .values_list("start_time__date", flat=True)
                .distinct()
            )

            for rec_date in dates:
                if rec_date is None:
                    continue

                # Skip today and future unless an explicit --date was given
                if not target_date and rec_date >= today_local:
                    continue

                # All speech segments for this date must have transcription done/skipped
                from django.db.models import Exists, OuterRef
                day_recs = Recording.objects.filter(
                    stream__radio=radio,
                    start_time__date=rec_date,
                )
                has_incomplete = day_recs.filter(
                    Exists(
                        TranscriptionSegment.objects.filter(
                            recording=OuterRef("pk"),
                            segment_type__in=["speech", "speech_over_music"],
                            transcription_status__in=["pending", "running"],
                        )
                    )
                ).exists()
                if has_incomplete:
                    continue

                # Check existing BroadcastDaySummary status
                existing = BroadcastDaySummary.objects.filter(
                    radio=radio, date=rec_date,
                ).first()

                if existing:
                    if existing.status == "done" and not force:
                        continue
                    if existing.status in ("running", "failed"):
                        # running: another worker has it
                        # failed: stays failed until --retry-failed resets it
                        continue
                    # status == "pending": will be processed
                else:
                    BroadcastDaySummary.objects.get_or_create(
                        radio=radio, date=rec_date,
                        defaults={"status": "pending"},
                    )
                candidates.append((radio, rec_date))

        return candidates

    def _process_one(self, radio, day, force):
        """Process a single radio+date pair."""
        # Optimistic claim
        summary, created = BroadcastDaySummary.objects.get_or_create(
            radio=radio, date=day,
            defaults={"status": "pending"},
        )

        if force and summary.status == "done":
            summary.status = "pending"
            summary.save(update_fields=["status"])

        claimed = BroadcastDaySummary.objects.filter(
            pk=summary.pk, status="pending"
        ).update(status="running", error="")

        if not claimed:
            return

        logger.info(
            "Processing broadcast day: %s — %s", radio.name, day,
        )

        try:
            result = summarize_broadcast_day(radio, day)
            if not result:
                logger.warning("Broadcast day summarizer returned no result for %s — %s.", radio.name, day)
                BroadcastDaySummary.objects.filter(pk=summary.pk).update(
                    status="failed", error="Summarizer returned no result."
                )
                return

            # Count recordings
            tz = ZoneInfo(radio.timezone or "UTC")
            local_start = datetime(day.year, day.month, day.day, tzinfo=tz)
            local_end = local_start + timedelta(days=1)
            rec_count = Recording.objects.filter(
                stream__radio=radio,
                start_time__gte=local_start,
                start_time__lt=local_end,
            ).count()

            # Update the summary
            BroadcastDaySummary.objects.filter(pk=summary.pk).update(
                status="done",
                error="",
                overview=result.overview,
                recording_count=rec_count,
            )
            summary.refresh_from_db()

            # Set tags
            tag_objects = [Tag.get_or_create_normalized(name)[0] for name in result.tags]
            summary.tags.set(tag_objects)

            # Delete old shows and recreate
            summary.shows.all().delete()

            for idx, show in enumerate(result.shows):
                # Parse times
                try:
                    start_h, start_m = map(int, show.start_time.split(":"))
                    end_h, end_m = map(int, show.end_time.split(":"))
                except (ValueError, AttributeError):
                    start_h, start_m = 0, 0
                    end_h, end_m = 0, 0

                show_start = datetime(day.year, day.month, day.day, start_h, start_m, tzinfo=tz)
                show_end = datetime(day.year, day.month, day.day, end_h, end_m, tzinfo=tz)
                # Handle shows that cross midnight
                if show_end <= show_start:
                    show_end += timedelta(days=1)

                show_block = ShowBlock.objects.create(
                    broadcast_day=summary,
                    name=show.name,
                    show_type=show.show_type,
                    start_time=show_start,
                    end_time=show_end,
                    summary=show.summary,
                    order=idx,
                )

                # Set show tags
                show_tag_objects = [Tag.get_or_create_normalized(name)[0] for name in show.tags]
                show_block.tags.set(show_tag_objects)

                # Link songs
                song_ids = link_songs_to_show(show, radio, day, show_start, show_end)
                if song_ids:
                    show_block.songs.set(song_ids)

            logger.info(
                "Broadcast day done: %s — %s (%d shows, %d tags).",
                radio.name, day, len(result.shows), len(tag_objects),
            )

        except Exception:
            tb = traceback.format_exc()
            logger.error(
                "Broadcast day failed: %s — %s:\n%s", radio.name, day, tb,
            )
            BroadcastDaySummary.objects.filter(pk=summary.pk).update(
                status="failed", error=tb,
            )
