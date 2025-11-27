import datetime
import os
import subprocess
import tempfile
import time
from django.core.files import File
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction
from radios.models import AudioStream, StreamRecording
from django.conf import settings


class Command(BaseCommand):
    help = "Continuously records active audio streams in 1-hour chunks."

    def handle(self, *args, **options):
        """
        Strategy:
        - Loop forever
        - Fetch all active AudioStream objects
        - For each stream, ensure we have (or start) a RecorderWorker
        - Tick workers so they keep recording
        """
        self.stdout.write(self.style.SUCCESS("Recorder starting..."))

        workers = {}

        while True:
            # Refresh active streams from DB
            active_streams = {
                s.id: s
                for s in AudioStream.objects.filter(is_active=True)
            }

            # Start workers for new streams
            for stream_id, stream in active_streams.items():
                if stream_id not in workers:
                    workers[stream_id] = RecorderWorker(stream)
                    workers[stream_id].start()

            # Stop workers for streams no longer active
            for stream_id in list(workers.keys()):
                if stream_id not in active_streams:
                    workers[stream_id].stop()
                    del workers[stream_id]

            # Let workers run one step
            for w in workers.values():
                w.tick()

            # Be gentle with CPU
            time.sleep(5)


class RecorderWorker:
    """
    One RecorderWorker per AudioStream.

    Responsibilities:
    - Launch ffmpeg to capture the stream audio
    - Rotate output every hour exactly
    - Store file via StreamRecording
    """

    def __init__(self, stream: AudioStream):
        self.stream = stream
        self.current_process = None
        self.current_chunk_start = None
        self.current_tempfile = None

    def start(self):
        # Force new chunk immediately
        self._start_new_chunk()

    def stop(self):
        # Finalize current chunk (if any)
        self._finalize_chunk()
        # ensure process is gone
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()

    def tick(self):
        """
        Called periodically from the main loop.
        Checks:
        - Is ffmpeg still alive?
        - Has 1 hour passed?
        If hour passed OR process died, finalize and start new chunk.
        """
        if not self.current_process:
            # process crashed or never started -> start again
            self._start_new_chunk()
            return

        # If ffmpeg already exited, rotate immediately
        if self.current_process.poll() is not None:
            self._finalize_chunk()
            self._start_new_chunk()
            return

        now = timezone.now()
        elapsed = now - self.current_chunk_start
        if elapsed.total_seconds() >= 3600:
            # rotate hourly
            self._finalize_chunk()
            self._start_new_chunk()

    def _start_new_chunk(self):
        """
        Start a new ffmpeg process that writes raw audio data into a temp file.
        We'll later move that file into Django storage.
        """
        self.current_chunk_start = timezone.now()

        # temp file (not yet in MEDIA_ROOT, so partial files aren't exposed)
        self.current_tempfile = tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        )
        temp_path = self.current_tempfile.name
        self.current_tempfile.close()  # ffmpeg will write to it

        # Launch ffmpeg
        # Assumptions:
        # - the stream URL is something ffmpeg can read
        # - we want to save as mp3 (you can change codec/format)
        #
        # -reconnect options help survive temporary drops
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", self.stream.streaming_url,
            "-vn",              # no video
            "-acodec", "copy",  # don't transcode if source is already audio (safer: use 'libmp3lame' if needed)
            "-f", "mp3",
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            temp_path,
        ]

        self.current_process = subprocess.Popen(cmd)

    def _finalize_chunk(self):
        """
        Stop ffmpeg, create StreamRecording row, move temp file into storage.
        """
        if not self.current_process:
            return

        # Stop process nicely
        if self.current_process.poll() is None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()

        end_time = timezone.now()

        temp_path = None
        if self.current_tempfile:
            temp_path = self.current_tempfile.name

        # Reset state before we do any DB writes,
        # so if something explodes we don't double-finalize.
        proc = self.current_process
        self.current_process = None
        self.current_tempfile = None

        if not temp_path or not os.path.exists(temp_path):
            # nothing to save
            return

        # build final filename
        # example: "streamname_2025-10-26T13-00-00_to_2025-10-26T14-00-00.mp3"
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in self.stream.name
        )
        filename = (
            f"{safe_name}_"
            f"{self.current_chunk_start:%Y-%m-%dT%H-%M-%S}_to_"
            f"{end_time:%Y-%m-%dT%H-%M-%S}.mp3"
        )

        # save in a DB transaction so both file+row are consistent
        with transaction.atomic():
            rec = StreamRecording(
                stream=self.stream,
                start_time=self.current_chunk_start,
                end_time=end_time,
            )

            # open temp file and attach to FileField
            with open(temp_path, "rb") as f:
                rec.file.save(filename, File(f), save=True)

        # cleanup temp file
        os.remove(temp_path)


