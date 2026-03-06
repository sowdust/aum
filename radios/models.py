import zoneinfo

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.core.validators import URLValidator
from django_countries.fields import CountryField
from django.utils.text import slugify
from django.db import models
import uuid

TIMEZONE_CHOICES = sorted(
    [(tz, tz) for tz in zoneinfo.available_timezones()],
    key=lambda x: x[0],
)

BROADCAST_TYPE_CHOICES = [
    ("fm", "FM"),
    ("am", "AM"),
    ("sw", "Shortwave"),
    ("web", "Online"),
    ("dab", "Digital Audio Broadcasting"),
]


PIPELINE_STAGES = ["recording", "segmentation", "fingerprinting", "transcription", "summarization"]


class GlobalPipelineSettings(models.Model):
    """
    Singleton. Always pk=1. Site-wide kill switches for each pipeline stage.
    Use GlobalPipelineSettings.get_settings() — never query directly.
    """
    enable_recording = models.BooleanField(
        default=True,
        help_text="Globally enable/disable recording for all sources.",
    )
    enable_segmentation = models.BooleanField(
        default=True,
        help_text="Globally enable/disable audio segmentation.",
    )
    enable_fingerprinting = models.BooleanField(
        default=True,
        help_text="Globally enable/disable music fingerprinting (AcoustID).",
    )
    enable_transcription = models.BooleanField(
        default=True,
        help_text="Globally enable/disable speech transcription (Whisper).",
    )
    enable_summarization = models.BooleanField(
        default=True,
        help_text="Globally enable/disable LLM summarization.",
    )

    class Meta:
        verbose_name = "Global Pipeline Settings"
        verbose_name_plural = "Global Pipeline Settings"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        pass  # Prevent deletion

    @classmethod
    def get_settings(cls):
        """Return the singleton row, creating it with defaults if absent."""
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return "Global Pipeline Settings"


class BaseAudioSource(models.Model):
    """
    Abstract base for any monitored audio source (Radio or AudioFeed).
    Provides shared identity, geographic, and recording-schedule fields.
    """
    name = models.CharField(max_length=180)
    slug = models.SlugField(primary_key=True, unique=True, blank=True)
    description = models.TextField(blank=True)
    country = CountryField(blank=True, null=True)
    city = models.CharField(max_length=100, blank=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    languages = models.CharField(max_length=255, blank=True)
    website = models.URLField(blank=True, validators=[URLValidator()])
    frequencies = models.CharField(max_length=200, blank=True)
    show_archive = models.BooleanField(default=False)
    timezone = models.CharField(
        max_length=63,
        choices=TIMEZONE_CHOICES,
        default="UTC",
        help_text="IANA timezone for local time (e.g. Europe/Rome)",
    )
    recording_start_hour = models.PositiveSmallIntegerField(
        default=0,
        help_text="Hour (0-23) in local time to start recording. 0/0 = 24/7.",
    )
    recording_end_hour = models.PositiveSmallIntegerField(
        default=0,
        help_text="Hour (0-23) in local time to stop recording. 0/0 = 24/7.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
        ordering = ["name"]

    def __str__(self):
        return self.name

    def _generate_unique_slug(self):
        """
        Generate a slug from name. Falls back to adding city/country/frequencies,
        then a numeric suffix to ensure uniqueness within the same model.
        """
        base_slug = slugify(self.name)
        slug_candidate = base_slug

        def is_unique(s):
            qs = self.__class__.objects.filter(slug=s)
            if self.pk:
                qs = qs.exclude(pk=self.pk)
            return not qs.exists()

        if is_unique(slug_candidate):
            return slug_candidate

        if self.city:
            slug_candidate = slugify(f"{self.name}-{self.city}")
            if is_unique(slug_candidate):
                return slug_candidate

        if self.country:
            slug_candidate = slugify(f"{self.name}-{self.city}-{self.country}")
            if is_unique(slug_candidate):
                return slug_candidate

        if self.frequencies:
            slug_candidate = slugify(f"{self.name}-{self.city}-{self.country}-{self.frequencies}")
            if is_unique(slug_candidate):
                return slug_candidate

        counter = 2
        while True:
            slug_candidate = f"{base_slug}-{counter}"
            if is_unique(slug_candidate):
                return slug_candidate
            counter += 1

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = self._generate_unique_slug()
        super().save(*args, **kwargs)


class Radio(BaseAudioSource):
    """A radio station — has broadcast types, branding, and member management."""
    logo = models.ImageField(upload_to="logos/", null=True, blank=True)
    motto = models.CharField(max_length=255, blank=True)
    since = models.DateField(blank=True, null=True, help_text="Radio creation date")
    until = models.DateField(blank=True, null=True, help_text="Radio end date")
    is_fm = models.BooleanField(default=False, help_text="Does the radio broadcast via FM?")
    is_am = models.BooleanField(default=False)
    is_dab = models.BooleanField(default=False)
    is_sw = models.BooleanField(default=False)
    is_web = models.BooleanField(default=False)
    contact_email = models.EmailField(blank=True)
    members = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        through="RadioMembership",
        related_name="radios",
        blank=True,
    )


class AudioFeed(BaseAudioSource):
    """
    A non-radio audio source (air traffic, emergency services, public safety, etc.).
    Monitored for anomalies rather than broadcast content.
    """
    FEED_TYPE_CHOICES = [
        ("air_traffic", "Air Traffic Control"),
        ("emergency", "Emergency Services"),
        ("public_safety", "Public Safety"),
        ("weather", "Weather"),
        ("other", "Other"),
    ]
    feed_type = models.CharField(
        max_length=20,
        choices=FEED_TYPE_CHOICES,
        default="other",
    )


class RadioMembership(models.Model):
    ROLE_CHOICES = [
        ("owner", "Owner"),
        ("dj", "DJ"),
        ("viewer", "Viewer"),
    ]
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    radio = models.ForeignKey(Radio, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="viewer")
    verified = models.BooleanField(default=False)

    class Meta:
        unique_together = ("user", "radio")


class Stream(models.Model):
    """
    An audio stream URL belonging to a Radio or an AudioFeed.
    Exactly one of `radio` or `audio_feed` must be set.
    """
    radio = models.ForeignKey(
        Radio,
        on_delete=models.CASCADE,
        related_name="streams",
        null=True,
        blank=True,
    )
    audio_feed = models.ForeignKey(
        AudioFeed,
        on_delete=models.CASCADE,
        related_name="streams",
        null=True,
        blank=True,
    )
    name = models.CharField(max_length=200)
    url = models.URLField()
    is_active = models.BooleanField(default=True)

    # --- Pipeline stage enables (admin-controlled) ---
    enable_recording = models.BooleanField(default=True)
    enable_segmentation = models.BooleanField(default=True)
    enable_fingerprinting = models.BooleanField(default=True)
    enable_transcription = models.BooleanField(default=True)
    enable_summarization = models.BooleanField(default=True)

    # --- Stage result visibility (owner-controlled) ---
    recording_owner_visible = models.BooleanField(default=True)
    recording_public_visible = models.BooleanField(default=False)
    segmentation_owner_visible = models.BooleanField(default=True)
    segmentation_public_visible = models.BooleanField(default=False)
    fingerprinting_owner_visible = models.BooleanField(default=True)
    fingerprinting_public_visible = models.BooleanField(default=False)
    transcription_owner_visible = models.BooleanField(default=True)
    transcription_public_visible = models.BooleanField(default=False)
    summarization_owner_visible = models.BooleanField(default=True)
    summarization_public_visible = models.BooleanField(default=False)

    _STAGE_DEPENDENCIES = {
        "recording": None,
        "segmentation": "recording",
        "fingerprinting": "segmentation",
        "transcription": "segmentation",
        "summarization": "transcription",
    }

    def is_stage_active(self, stage: str) -> bool:
        """
        Return True if `stage` is active for this stream.
        Checks global kill switch, per-stream enable flag, and upstream dependency.
        """
        if stage not in self._STAGE_DEPENDENCIES:
            raise ValueError(f"Unknown pipeline stage: {stage!r}")
        global_settings = GlobalPipelineSettings.get_settings()
        if not getattr(global_settings, f"enable_{stage}"):
            return False
        if not getattr(self, f"enable_{stage}"):
            return False
        upstream = self._STAGE_DEPENDENCIES[stage]
        if upstream is not None:
            return self.is_stage_active(upstream)
        return True

    @property
    def source(self):
        """Return whichever of radio or audio_feed this stream belongs to."""
        return self.radio or self.audio_feed

    def __str__(self):
        return self.name


def safe_stream_folder(stream_name: str) -> str:
    """
    Produces a stable directory name from the stream name.
    Example: "Radio Rock 101.5" -> "radio-rock-101-5"
    """
    return slugify(stream_name) or str(uuid.uuid4())


def recording_upload_path(instance, filename):
    """
    Final storage path for a given chunk.
    recordings/<stream-name-sanitized>/<YYYY>/<MM>/<DD>/<filename>.mp3
    """
    stream_folder = safe_stream_folder(instance.stream.name)
    return (
        f"recordings/{stream_folder}/"
        f"{instance.start_time:%Y/%m/%d}/"
        f"{filename}"
    )


ANALYSIS_STATUS_CHOICES = [
    ("pending", "Pending"),
    ("analysing", "Analysing"),
    ("transcribing", "Transcribing"),
    ("transcribed", "Transcribed"),
    ("summarizing", "Summarizing"),
    ("done", "Done"),
    ("failed", "Failed"),
]


class Recording(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    stream = models.ForeignKey("Stream", on_delete=models.CASCADE, related_name="recordings")
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    file = models.FileField(upload_to=recording_upload_path)
    analysis_status = models.CharField(
        max_length=20,
        choices=ANALYSIS_STATUS_CHOICES,
        default="pending",
        db_index=True,
    )
    analysis_error = models.TextField(blank=True, default="")
    analysis_started_at = models.DateTimeField(null=True, blank=True)
    analysis_completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-start_time"]

    def __str__(self):
        return f"{self.stream.source} [{self.start_time} - {self.end_time}]"


class TranscriptionSegment(models.Model):
    SEGMENT_TYPE_CHOICES = [
        ("speech", "Speech"),
        ("music", "Music"),
        ("speech_over_music", "Speech over Music"),
        ("noise", "Noise"),
        ("noEnergy", "No Energy / Silence"),
        ("silence", "Silence"),
        ("unknown", "Unknown"),
    ]
    recording = models.ForeignKey(
        Recording, on_delete=models.CASCADE, related_name="segments"
    )
    segment_type = models.CharField(
        max_length=20, choices=SEGMENT_TYPE_CHOICES, default="speech"
    )
    start_offset = models.FloatField(help_text="Seconds from recording start_time")
    end_offset = models.FloatField(help_text="Seconds from recording start_time")
    text = models.TextField(blank=True, default="")
    confidence = models.FloatField(default=0.0)
    language = models.CharField(max_length=10, blank=True, default="")
    song_title = models.CharField(max_length=255, blank=True, default="")
    song_artist = models.CharField(max_length=255, blank=True, default="")

    class Meta:
        ordering = ["recording", "start_offset"]
        indexes = [
            models.Index(fields=["recording", "start_offset"]),
        ]

    def __str__(self):
        return f"{self.segment_type} [{self.start_offset:.1f}-{self.end_offset:.1f}s]"


class ChunkSummary(models.Model):
    recording = models.OneToOneField(
        Recording, on_delete=models.CASCADE, related_name="chunk_summary"
    )
    summary_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Summary for {self.recording}"


class DailySummary(models.Model):
    radio = models.ForeignKey(
        Radio, on_delete=models.CASCADE, related_name="daily_summaries"
    )
    date = models.DateField()
    summary_text = models.TextField()
    chunk_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("radio", "date")
        ordering = ["-date"]

    def __str__(self):
        return f"{self.radio.name} - {self.date}"


class FeedAnomaly(models.Model):
    ANOMALY_TYPE_CHOICES = [
        ("speech", "Speech Detected"),
        ("noise", "Unusual Noise"),
        ("alarm", "Alarm / Alert"),
        ("other", "Other"),
    ]
    recording = models.ForeignKey(
        Recording, on_delete=models.CASCADE, related_name="anomalies"
    )
    start_offset = models.FloatField(help_text="Seconds from recording start_time")
    end_offset = models.FloatField(help_text="Seconds from recording start_time")
    anomaly_type = models.CharField(
        max_length=10, choices=ANOMALY_TYPE_CHOICES, default="other"
    )
    audio_level_db = models.FloatField(default=0.0)
    transcript = models.TextField(blank=True, default="")
    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["recording", "start_offset"]

    def __str__(self):
        return f"{self.anomaly_type} at {self.start_offset:.1f}s in {self.recording}"


class RadioUser(AbstractUser):
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)


class EmailVerificationToken(models.Model):
    user = models.ForeignKey(RadioUser, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
