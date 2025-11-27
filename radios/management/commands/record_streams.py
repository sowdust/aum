

import datetime
import logging
import os
import subprocess
import tempfile
import time
import uuid

from django.core.files import File
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction
from django.conf import settings

from radios.models import Stream, Recording


logger = logging.getLogger("stream_recorder")


class Command(BaseCommand):
    help = "Continuously records active audio streams in 1-hour chunks."

    def handle(self, *args, **options):
        logger.info("Recorder service starting up")

        workers = {}

        running = True

        # graceful shutdown: SIGINT/SIGTERM
        import signal

        def shutdown(signum, frame):
            nonlocal running
            logger.info("Shutdown signal (%s) received, stopping workers...", signum)
            running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        while running:
            # Refresh active streams list
            active_streams = {
                s.id: s
                for s in Stream.objects.filter(is_active=True)
            }

            # start workers for new streams
            for stream_id, stream in active_streams.items():
                if stream_id not in workers:
                    logger.info("Starting worker for stream '%s' (%s)", stream.name, stream_id)
                    workers[stream_id] = RecorderWorker(stream)
                    workers[stream_id].start()

            # stop workers for streams that got deactivated
            for stream_id in list(workers.keys()):
                if stream_id not in active_streams:
                    logger.info("Stopping worker for deactivated stream %s", stream_id)
                    workers[stream_id].stop()
                    del workers[stream_id]

            # tick workers
            for w in list(workers.values()):
                try:
                    w.tick()
                except Exception as e:
                    logger.exception("Worker for stream '%s' crashed in tick(): %s", w.stream.name, e)

            time.sleep(5)

        # graceful shutdown
        logger.info("Stopping all workers...")
        for w in workers.values():
            w.stop()
        logger.info("Recorder service exited cleanly")


class RecorderWorker:
    """
    One RecorderWorker per AudioStream.

    Responsibilities:
    - Launch ffmpeg to capture the stream audio
    - Rotate output every hour
    - Store file and create StreamRecording
    """

    def __init__(self, stream: Stream):
        self.stream = stream
        self.current_process = None
        self.current_chunk_start = None
        self.current_tempfile_path = None

    def start(self):
        logger.info("Worker[%s]: starting first chunk", self.stream.name)
        self._start_new_chunk()

    def stop(self):
        logger.info("Worker[%s]: stopping, finalizing current chunk", self.stream.name)
        self._finalize_chunk(kill=True)

    def tick(self):
        """
        Called periodically.
        - If ffmpeg died: finalize file, start new chunk
        - If >=1h passed: rotate
        """
        if not self.current_process:
            logger.warning("Worker[%s]: ffmpeg process missing, restarting", self.stream.name)
            self._start_new_chunk()
            return

        # Check if ffmpeg exited
        if self.current_process.poll() is not None:
            logger.warning("Worker[%s]: ffmpeg exited unexpectedly (rc=%s), rotating chunk",
                           self.stream.name, self.current_process.returncode)
            self._finalize_chunk()
            self._start_new_chunk()
            return

        # Check for hour boundary
        now = timezone.now()
        elapsed = now - self.current_chunk_start
        if elapsed.total_seconds() >= settings.CHUNK_SIZE:
            logger.info("Worker[%s]: 1h reached, rotating chunk", self.stream.name)
            self._finalize_chunk()
            self._start_new_chunk()

    def _start_new_chunk(self):
        """
        Start ffmpeg writing to a new unique tempfile.
        """
        self.current_chunk_start = timezone.now()

        unique_name = f"rec_{uuid.uuid4().hex}.mp3"
        temp_path = os.path.join(tempfile.gettempdir(), unique_name)
        self.current_tempfile_path = temp_path

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # auto-overwrite if file somehow exists
            "-hide_banner",
            "-loglevel", "warning",
            "-i", self.stream.url,
            "-vn",
            "-acodec", "copy",
            "-f", "mp3",
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            temp_path,
        ]

        logger.info("Worker[%s]: starting ffmpeg -> %s", self.stream.name, temp_path)
        try:
            self.current_process = subprocess.Popen(cmd)
        except Exception:
            logger.exception("Worker[%s]: failed to start ffmpeg", self.stream.name)
            self.current_process = None
            self.current_tempfile_path = None

    def _finalize_chunk(self, kill=False):
        """
        Stop ffmpeg, save file to storage, create DB row.
        kill=True means force kill if terminate doesn't work.
        """
        if not self.current_process:
            return

        proc = self.current_process
        temp_path = self.current_tempfile_path
        start_time = self.current_chunk_start

        self.current_process = None
        self.current_tempfile_path = None
        self.current_chunk_start = None

        # Stop process
        if proc.poll() is None:
            logger.info("Worker[%s]: stopping ffmpeg for current chunk", self.stream.name)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if kill:
                    logger.warning("Worker[%s]: ffmpeg did not terminate, killing", self.stream.name)
                    proc.kill()
                    proc.wait(timeout=5)
                else:
                    logger.warning("Worker[%s]: ffmpeg still alive after terminate timeout", self.stream.name)
                    proc.kill()
                    proc.wait(timeout=5)

        end_time = timezone.now()

        if not temp_path or not os.path.exists(temp_path):
            logger.warning("Worker[%s]: no file to save for chunk", self.stream.name)
            return

        # sanitize stream name for filename
        safe_name = "".join(
            c if (c.isalnum() or c in ("-", "_")) else "_"
            for c in self.stream.name
        )
        final_filename = (
            f"{safe_name}_"
            f"{start_time:%Y-%m-%dT%H-%M-%S}_to_"
            f"{end_time:%Y-%m-%dT%H-%M-%S}.mp3"
        )

        logger.info(
            "Worker[%s]: saving chunk %s (%s -> %s)",
            self.stream.name, final_filename, start_time, end_time
        )

        try:
            with transaction.atomic():
                rec = Recording(
                    stream=self.stream,
                    start_time=start_time,
                    end_time=end_time,
                )
                with open(temp_path, "rb") as f:
                    rec.file.save(final_filename, File(f), save=True)

            logger.info(
                "Worker[%s]: chunk saved, DB id=%s, size=%d bytes",
                self.stream.name,
                rec.id,
                os.path.getsize(temp_path)
            )
        except Exception:
            logger.exception("Worker[%s]: FAILED to save chunk to DB/storage", self.stream.name)
        finally:
            # cleanup temp file
            try:
                os.remove(temp_path)
                logger.debug("Worker[%s]: removed temp file %s", self.stream.name, temp_path)
            except OSError:
                logger.warning("Worker[%s]: could not remove temp file %s", self.stream.name, temp_path)

