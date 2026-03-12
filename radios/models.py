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


PIPELINE_STAGES = ["recording", "segmentation", "fingerprinting", "transcription", "summarization", "daily_summarization"]


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
    enable_daily_summarization = models.BooleanField(
        default=False,
        help_text="Globally enable/disable daily broadcast summarization.",
    )
    proxy_url = models.URLField(
        max_length=500,
        blank=True,
        default="",
        help_text="Global proxy URL for stream recording (e.g. http://proxy:8080 or socks5://proxy:1080). "
                  "Used by sources with proxy_mode='global'.",
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
    PROXY_MODE_CHOICES = [
        ("none", "No Proxy"),
        ("global", "Use Global Proxy"),
        ("custom", "Custom Proxy"),
    ]
    proxy_mode = models.CharField(
        max_length=10,
        choices=PROXY_MODE_CHOICES,
        default="none",
        help_text="'none' = direct connection, 'global' = use the global proxy, "
                  "'custom' = use this source's own proxy_url.",
    )
    proxy_url = models.URLField(
        max_length=500,
        blank=True,
        default="",
        help_text="Proxy URL for this source (only used when proxy_mode='custom'). "
                  "e.g. http://proxy:8080 or socks5://proxy:1080",
    )
    stream_codec = models.CharField(
        max_length=20,
        blank=True,
        default="",
        help_text="Audio codec of the stream (e.g. 'mp3', 'aac', 'ogg'). "
                  "Auto-detected on first recording if left blank.",
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

    def get_effective_proxy_url(self) -> str:
        """
        Return the proxy URL to use for recording, or empty string for direct connection.
        """
        if self.proxy_mode == "global":
            return GlobalPipelineSettings.get_settings().proxy_url
        elif self.proxy_mode == "custom":
            return self.proxy_url
        return ""

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

    # --- Recording status (managed by record_streams command) ---
    RECORDING_STATUS_CHOICES = [
        ("idle", "Idle"),
        ("recording", "Recording"),
        ("error", "Error"),
    ]
    recording_status = models.CharField(
        max_length=10, choices=RECORDING_STATUS_CHOICES, default="idle", db_index=True,
    )
    recording_error = models.TextField(blank=True, default="")
    recording_started_at = models.DateTimeField(null=True, blank=True)

    # --- Pipeline stage enables (admin-controlled) ---
    enable_recording = models.BooleanField(default=True)
    enable_segmentation = models.BooleanField(default=True)
    enable_fingerprinting = models.BooleanField(default=True)
    enable_transcription = models.BooleanField(default=True)
    enable_summarization = models.BooleanField(default=True)
    enable_daily_summarization = models.BooleanField(default=False)

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
    daily_summarization_owner_visible = models.BooleanField(default=True)
    daily_summarization_public_visible = models.BooleanField(default=False)

    _STAGE_DEPENDENCIES = {
        "recording": None,
        "segmentation": "recording",
        "fingerprinting": "segmentation",
        "transcription": "segmentation",
        "summarization": "transcription",
        "daily_summarization": "transcription",
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
        return f"{self.radio.name} - {self.name}"


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


STAGE_STATUS_CHOICES = [
    ("pending", "Pending"),
    ("running", "Running"),
    ("done", "Done"),
    ("failed", "Failed"),
    ("skipped", "Skipped"),
]


class Recording(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    stream = models.ForeignKey("Stream", on_delete=models.CASCADE, related_name="recordings")
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    file = models.FileField(upload_to=recording_upload_path)

    # Per-stage status tracking
    segmentation_status = models.CharField(
        max_length=10, choices=STAGE_STATUS_CHOICES, default="pending", db_index=True,
    )
    segmentation_error = models.TextField(blank=True, default="")
    fingerprinting_status = models.CharField(
        max_length=10, choices=STAGE_STATUS_CHOICES, default="pending", db_index=True,
    )
    fingerprinting_error = models.TextField(blank=True, default="")
    transcription_status = models.CharField(
        max_length=10, choices=STAGE_STATUS_CHOICES, default="pending", db_index=True,
    )
    transcription_error = models.TextField(blank=True, default="")
    summarization_status = models.CharField(
        max_length=10, choices=STAGE_STATUS_CHOICES, default="pending", db_index=True,
    )
    summarization_error = models.TextField(blank=True, default="")

    analysis_started_at = models.DateTimeField(null=True, blank=True)
    analysis_completed_at = models.DateTimeField(null=True, blank=True)

    @property
    def analysis_status(self):
        """Computed overall status from per-stage statuses."""
        stages = [
            self.segmentation_status, self.fingerprinting_status,
            self.transcription_status, self.summarization_status,
        ]
        if any(s == "failed" for s in stages):
            return "failed"
        if all(s in ("done", "skipped") for s in stages):
            return "done"
        if any(s == "running" for s in stages):
            return "running"
        return "pending"

    @property
    def analysis_error(self):
        """Aggregate error text from all failed stages."""
        errors = filter(None, [
            self.segmentation_error, self.fingerprinting_error,
            self.transcription_error, self.summarization_error,
        ])
        return "\n---\n".join(errors)

    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    class Meta:
        ordering = ["-start_time"]
        indexes = [
            models.Index(fields=["start_time"], name="idx_recording_start_time"),
            models.Index(fields=["stream", "start_time"], name="idx_recording_stream_start"),
        ]

    def __str__(self):
        return f"{self.stream.source} [{self.start_time} - {self.end_time}]"


class Artist(models.Model):
    """A music artist, deduplicated by Shazam ID or case-insensitive name."""
    name = models.CharField(max_length=255, db_index=True)
    shazam_id = models.CharField(max_length=50, unique=True, null=True, blank=True)
    musicbrainz_id = models.CharField(max_length=36, unique=True, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    @classmethod
    def get_or_create_by_name(cls, name):
        """Case-insensitive lookup, create if missing."""
        try:
            return cls.objects.get(name__iexact=name), False
        except cls.DoesNotExist:
            return cls.objects.create(name=name), True
        except cls.MultipleObjectsReturned:
            return cls.objects.filter(name__iexact=name).first(), False


class Genre(models.Model):
    """A music genre, normalized like Tag."""
    name = models.CharField(max_length=100, unique=True, db_index=True)
    slug = models.SlugField(max_length=100, unique=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def save(self, *args, **kwargs):
        self.name = self.name.lower().strip()
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    @classmethod
    def get_or_create_normalized(cls, name: str):
        """Return (genre, created) for a normalized genre name."""
        normalized = name.lower().strip()
        slug = slugify(normalized)
        return cls.objects.get_or_create(slug=slug, defaults={"name": normalized})


class Song(models.Model):
    """
    A music track identified via Shazam fingerprinting.
    Shared across all SongOccurrence instances of the same track.
    """
    title = models.CharField(max_length=255, db_index=True)
    artist = models.CharField(max_length=255, blank=True, db_index=True,
        help_text="Denormalized artist display name from Shazam.")
    artist_ref = models.ForeignKey(
        Artist, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="songs",
        help_text="Normalized artist reference.",
    )
    shazam_key = models.CharField(
        max_length=50, unique=True, null=True, blank=True,
        help_text="Shazam track key — canonical dedup key.",
    )
    musicbrainz_id = models.CharField(
        max_length=36, unique=True, null=True, blank=True,
        help_text="MusicBrainz Recording ID (for future enrichment).",
    )
    genres = models.ManyToManyField(Genre, related_name="songs", blank=True)
    album_name = models.CharField(max_length=255, blank=True)
    album_cover_url = models.URLField(max_length=500, blank=True)
    release_year = models.PositiveSmallIntegerField(null=True, blank=True)
    isrc = models.CharField(max_length=15, blank=True,
        help_text="International Standard Recording Code (future MusicBrainz).")
    duration_seconds = models.PositiveIntegerField(null=True, blank=True,
        help_text="Track duration in seconds (helps sliding window step).")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["title"]

    def __str__(self):
        return f"{self.artist} — {self.title}" if self.artist else self.title

    @classmethod
    def get_or_create_from_fingerprint(cls, result):
        """
        Resolve a FingerprintResult to a Song row.
        Prefers shazam_key as the canonical key; falls back to (title, artist).
        Updates metadata on existing rows when Shazam provides new info.
        """
        if result.shazam_key:
            song, created = cls.objects.get_or_create(
                shazam_key=result.shazam_key,
                defaults={"title": result.title, "artist": result.artist},
            )
        else:
            song, created = cls.objects.get_or_create(
                title=result.title,
                artist=result.artist,
                shazam_key=None,
            )

        # Update metadata if we have new info
        updated_fields = []
        if result.title and song.title != result.title:
            song.title = result.title
            updated_fields.append("title")
        if result.artist and song.artist != result.artist:
            song.artist = result.artist
            updated_fields.append("artist")
        if hasattr(result, "album_name") and result.album_name and not song.album_name:
            song.album_name = result.album_name
            updated_fields.append("album_name")
        if hasattr(result, "album_cover_url") and result.album_cover_url and not song.album_cover_url:
            song.album_cover_url = result.album_cover_url
            updated_fields.append("album_cover_url")
        if hasattr(result, "release_year") and result.release_year and not song.release_year:
            song.release_year = result.release_year
            updated_fields.append("release_year")
        if updated_fields:
            song.save(update_fields=updated_fields)

        # Link artist
        if result.artist and not song.artist_ref:
            artist_obj, _ = Artist.get_or_create_by_name(result.artist)
            song.artist_ref = artist_obj
            song.save(update_fields=["artist_ref"])

        # Link genres
        if hasattr(result, "genres") and result.genres:
            for genre_name in result.genres:
                if genre_name:
                    genre_obj, _ = Genre.get_or_create_normalized(genre_name)
                    song.genres.add(genre_obj)

        return song


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
    text_original = models.TextField(blank=True, default="",
        help_text="Raw transcription before LLM correction; empty if correction was not applied.")
    text_english = models.TextField(blank=True, default="",
        help_text="English translation when original text is non-English; empty if already English.")
    confidence = models.FloatField(default=0.0)
    language = models.CharField(max_length=10, blank=True, default="")
    song = models.ForeignKey(
        Song, null=True, blank=True, on_delete=models.SET_NULL, related_name="occurrences",
        help_text="Deprecated: use SongOccurrence instead. Kept for data migration.",
    )

    class Meta:
        ordering = ["recording", "start_offset"]
        indexes = [
            models.Index(fields=["recording", "start_offset"]),
        ]

    def __str__(self):
        return f"{self.segment_type} [{self.start_offset:.1f}-{self.end_offset:.1f}s]"


_DEFAULT_CHUNK_PROMPT = """\
You are analysing a radio broadcast segment. Below are transcripts of speech from a single recording chunk.{language_hint}

Return ONLY valid JSON with these fields:
- "summary": 2-4 sentences summarising what was talked about
- "tags": list of up to 15 lowercase keyword tags (topics, people, places, events, themes — no duplicates)

Transcripts:
{content}

Respond with ONLY the JSON object, no markdown fences or explanation."""

_DEFAULT_DAILY_PROMPT = """\
You are analysing a full day of radio broadcasts. Below are summaries of individual recording chunks from the same radio station.

Return ONLY valid JSON with these fields:
- "summary": 3-5 sentences summarising the main themes and events across the day
- "tags": list of up to 15 lowercase keyword tags (most important topics, people, places, events — no duplicates)

Chunk summaries:
{content}

Respond with ONLY the JSON object, no markdown fences or explanation."""


_DEFAULT_CORRECTION_PROMPT = """\
You are correcting speech-to-text transcription errors from a radio broadcast.

Radio: {radio_name}
Location: {radio_location}
Language: {radio_language}

Below are numbered transcript segments. Fix transcription errors (misspelled names,
incorrect words, garbled text) while preserving the original meaning and language.
Do NOT summarize. Only correct obvious errors.

After correcting each segment, also provide an English translation of the corrected text.
If the original text is already in English, the translation should be identical to the
corrected text.

Segments:
{segments}

Return ONLY a valid JSON array where each element has:
- "index": the segment number
- "text": the corrected text in the original language
- "text_english": English translation of the corrected text

If a segment needs no correction, return the original text unchanged.
No markdown fences or explanation."""


class TranscriptionSettings(models.Model):
    """
    Singleton (pk=1). Controls the transcription pipeline stage:
    backend selection and model parameters.
    API keys are read from environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY).
    """
    BACKEND_CHOICES = [
        ("local", "Local (faster-whisper)"),
        ("openai", "OpenAI Whisper API"),
        ("anthropic", "Anthropic (Claude)"),
        ("ollama", "Ollama (local or cloud)"),
    ]
    MODEL_SIZE_CHOICES = [
        ("tiny", "Tiny — fastest, least accurate"),
        ("base", "Base"),
        ("small", "Small"),
        ("medium", "Medium (recommended)"),
        ("large-v2", "Large v2"),
        ("large-v3", "Large v3 — slowest, most accurate"),
    ]
    DEVICE_CHOICES = [
        ("cpu", "CPU"),
        ("cuda", "CUDA (GPU)"),
        ("auto", "Auto-detect"),
    ]
    COMPUTE_TYPE_CHOICES = [
        ("int8", "int8 — fastest, good for CPU"),
        ("float16", "float16 — recommended for GPU"),
        ("float32", "float32 — highest accuracy"),
    ]

    backend = models.CharField(
        max_length=20, choices=BACKEND_CHOICES, default="local",
        help_text="Which backend to use for speech transcription.",
    )

    # --- Local (faster-whisper) ---
    local_model_size = models.CharField(
        max_length=20, choices=MODEL_SIZE_CHOICES, default="medium",
        help_text="Whisper model size. Larger models are more accurate but require more RAM and time.",
    )
    local_device = models.CharField(
        max_length=10, choices=DEVICE_CHOICES, default="cpu",
        help_text="Device to run faster-whisper on.",
    )
    local_compute_type = models.CharField(
        max_length=10, choices=COMPUTE_TYPE_CHOICES, default="int8",
        help_text="Numeric precision for faster-whisper inference.",
    )

    # --- OpenAI Whisper API ---
    openai_model = models.CharField(
        max_length=100, default="whisper-1",
        help_text="OpenAI Whisper model name. API key must be set in the OPENAI_API_KEY environment variable.",
    )

    # --- Anthropic ---
    anthropic_model = models.CharField(
        max_length=100, default="claude-sonnet-4-20250514",
        help_text="Claude model ID for audio transcription. API key must be set in the ANTHROPIC_API_KEY environment variable.",
    )

    # --- Ollama ---
    ollama_model = models.CharField(
        max_length=100, default="whisper",
        help_text=(
            "Ollama model name for transcription (e.g. 'whisper'). "
            "For local instances no key is needed; for ollama.com cloud set OLLAMA_API_KEY."
        ),
    )
    ollama_base_url = models.URLField(
        max_length=500, default="http://localhost:11434",
        help_text=(
            "Base URL of the Ollama server for audio transcription. "
            "Use 'http://localhost:11434' for a local instance. "
            "Note: Ollama cloud (ollama.com) does not support audio transcription."
        ),
    )

    # --- Transcription correction (LLM post-processing) ---
    CORRECTION_BACKEND_CHOICES = [
        ("local_ollama", "Local Ollama"),
        ("cloud_ollama", "Cloud Ollama"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic (Claude)"),
    ]

    enable_correction = models.BooleanField(
        default=False,
        help_text="Enable LLM post-processing to fix transcription errors and produce English translation from corrected text.",
    )
    correction_backend = models.CharField(
        max_length=20, choices=CORRECTION_BACKEND_CHOICES, default="local_ollama",
        help_text="Which LLM backend to use for transcription correction.",
    )
    correction_local_ollama_model = models.CharField(
        max_length=100, default="llama3.2",
        help_text="Ollama model name for local correction (e.g. 'llama3.2', 'mistral').",
    )
    correction_local_ollama_url = models.URLField(
        max_length=500, default="http://localhost:11434",
        help_text="Base URL of the local Ollama server for correction.",
    )
    correction_cloud_ollama_model = models.CharField(
        max_length=100, default="llama3.2",
        help_text="Ollama model name for cloud correction. Set OLLAMA_API_KEY.",
    )
    correction_cloud_ollama_url = models.URLField(
        max_length=500, default="https://ollama.com",
        help_text="Base URL of the Ollama cloud API for correction.",
    )
    correction_openai_model = models.CharField(
        max_length=100, default="gpt-4o-mini",
        help_text="OpenAI model name for correction. API key must be set in OPENAI_API_KEY.",
    )
    correction_anthropic_model = models.CharField(
        max_length=100, default="claude-haiku-4-5-20251001",
        help_text="Claude model ID for correction. API key must be set in ANTHROPIC_API_KEY.",
    )
    correction_prompt = models.TextField(
        default=_DEFAULT_CORRECTION_PROMPT,
        help_text=(
            "Prompt template for transcription correction. "
            "Placeholders: {segments}, {radio_name}, {radio_location}, {radio_language}."
        ),
    )

    class Meta:
        verbose_name = "Transcription Settings"
        verbose_name_plural = "Transcription Settings"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        pass

    @classmethod
    def get_settings(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return "Transcription Settings"


class SummarizationSettings(models.Model):
    """
    Singleton (pk=1). Controls the summarization pipeline stage:
    backend selection, model parameters, and editable prompts.
    API keys are read from environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_API_KEY).

    Prompt templates support these placeholders:
      {content}        — transcript text (chunk prompt) or chunk summaries (daily prompt)
      {language_hint}  — language sentence inserted when a language hint is available, else empty
    """
    BACKEND_CHOICES = [
        ("local_ollama", "Local Ollama"),
        ("cloud_ollama", "Cloud Ollama"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic (Claude)"),
    ]

    backend = models.CharField(
        max_length=20, choices=BACKEND_CHOICES, default="local_ollama",
        help_text="Which LLM backend to use for summarization.",
    )

    # --- Local Ollama (Python client, no API key needed) ---
    local_ollama_model = models.CharField(
        max_length=100, default="llama3.2",
        help_text="Ollama model name, e.g. 'llama3.2', 'mistral', 'phi3'.",
    )
    local_ollama_url = models.URLField(
        max_length=500, default="http://localhost:11434",
        help_text="Base URL of the local Ollama server.",
    )

    # --- Cloud Ollama (OpenAI-compatible API, OLLAMA_API_KEY required) ---
    cloud_ollama_model = models.CharField(
        max_length=100, default="llama3.2",
        help_text="Ollama model name for the cloud API (e.g. 'llama3.2', 'mistral'). Set OLLAMA_API_KEY.",
    )
    cloud_ollama_url = models.URLField(
        max_length=500, default="https://ollama.com",
        help_text="Base URL of the Ollama cloud API (https://ollama.com).",
    )

    # --- OpenAI ---
    openai_model = models.CharField(
        max_length=100, default="gpt-4o-mini",
        help_text="OpenAI model name. API key must be set in the OPENAI_API_KEY environment variable.",
    )

    # --- Anthropic ---
    anthropic_model = models.CharField(
        max_length=100, default="claude-haiku-4-5-20251001",
        help_text="Claude model ID. API key must be set in the ANTHROPIC_API_KEY environment variable.",
    )

    # --- Prompts ---
    prompt_chunk = models.TextField(
        default=_DEFAULT_CHUNK_PROMPT,
        help_text=(
            "Prompt template for per-chunk summaries. "
            "Placeholders: {content} (transcripts), {language_hint} (language sentence or empty)."
        ),
    )
    prompt_daily = models.TextField(
        default=_DEFAULT_DAILY_PROMPT,
        help_text=(
            "Prompt template for daily summaries. "
            "Placeholders: {content} (list of chunk summaries)."
        ),
    )

    class Meta:
        verbose_name = "Summarization Settings"
        verbose_name_plural = "Summarization Settings"

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
        return "Summarization Settings"


class Tag(models.Model):
    """
    A normalized keyword/topic tag extracted from summarized recordings.
    Tags are always lowercase and slugified for consistent matching.
    """
    name = models.CharField(max_length=100, unique=True, db_index=True)
    slug = models.SlugField(max_length=100, unique=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def save(self, *args, **kwargs):
        self.name = self.name.lower().strip()
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

    @classmethod
    def get_or_create_normalized(cls, name: str):
        """Return (tag, created) for a normalized tag name."""
        normalized = name.lower().strip()
        slug = slugify(normalized)
        return cls.objects.get_or_create(slug=slug, defaults={"name": normalized})


class ChunkSummary(models.Model):
    """LLM-generated summary of a single recording chunk, with extracted tags."""
    recording = models.OneToOneField(
        Recording, on_delete=models.CASCADE, related_name="chunk_summary"
    )
    summary_text = models.TextField()
    tags = models.ManyToManyField(Tag, related_name="chunk_summaries", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Summary for {self.recording}"


class DailySummary(models.Model):
    """Aggregated LLM summary for a radio on a given day, with extracted tags."""
    radio = models.ForeignKey(
        Radio, on_delete=models.CASCADE, related_name="daily_summaries"
    )
    date = models.DateField()
    summary_text = models.TextField()
    chunk_count = models.PositiveIntegerField(default=0)
    tags = models.ManyToManyField(Tag, related_name="daily_summaries", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("radio", "date")
        ordering = ["-date"]

    def __str__(self):
        return f"{self.radio.name} - {self.date}"


class SongOccurrence(models.Model):
    """
    A single song identified within a music segment.
    Long segments (e.g. DJ mixtapes) may contain multiple songs;
    each gets its own SongOccurrence with absolute offsets.
    """
    segment = models.ForeignKey(
        TranscriptionSegment, on_delete=models.CASCADE, related_name="song_occurrences",
    )
    song = models.ForeignKey(
        Song, on_delete=models.CASCADE, related_name="segment_occurrences",
    )
    start_offset = models.FloatField(help_text="Absolute seconds from recording start")
    end_offset = models.FloatField(help_text="Estimated absolute seconds from recording start")
    confidence = models.FloatField(default=1.0)

    class Meta:
        ordering = ["start_offset"]
        indexes = [models.Index(fields=["segment", "start_offset"])]

    def __str__(self):
        return f"{self.song} [{self.start_offset:.1f}-{self.end_offset:.1f}s]"


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


_DEFAULT_BROADCAST_DAY_PROMPT = """\
You are analysing a full day of radio broadcasts. Below is a chronological timeline of everything that aired on {radio_name} ({radio_location}) on {date} ({timezone}).
The radio's language is: {radio_language}.

Your task is to reconstruct the broadcast day: identify individual shows/programmes, summarise each one, and provide an overview of the entire day.

Return ONLY valid JSON with these fields:
- "overview": 3-5 sentences summarising the day's broadcast
- "tags": list of up to 15 lowercase keyword tags (main topics, people, events — no duplicates)
- "shows": array of objects, each with:
  - "name": show/programme name (infer from context if not stated explicitly)
  - "type": one of: music, news, talk, sports, cultural, religious, spot, mixed, unknown
  - "start_time": "HH:MM" (local time)
  - "end_time": "HH:MM" (local time)
  - "summary": 2-4 sentences describing what happened in this show
  - "tags": list of up to 10 lowercase keyword tags for this show
  - "songs": list of "Title - Artist" strings for songs played during this show

Timeline:
{timeline}

Respond with ONLY the JSON object, no markdown fences or explanation."""


class DailySummarizationSettings(models.Model):
    """
    Singleton (pk=1). Controls the daily broadcast summarization pipeline stage.
    API keys are read from environment variables.
    """
    BACKEND_CHOICES = [
        ("local_ollama", "Local Ollama"),
        ("cloud_ollama", "Cloud Ollama"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic (Claude)"),
    ]

    backend = models.CharField(
        max_length=20, choices=BACKEND_CHOICES, default="local_ollama",
        help_text="Which LLM backend to use for daily broadcast summarization.",
    )

    # --- Local Ollama ---
    local_ollama_model = models.CharField(
        max_length=100, default="llama3.2",
        help_text="Ollama model name, e.g. 'llama3.2', 'mistral'.",
    )
    local_ollama_url = models.URLField(
        max_length=500, default="http://localhost:11434",
        help_text="Base URL of the local Ollama server.",
    )

    # --- Cloud Ollama ---
    cloud_ollama_model = models.CharField(
        max_length=100, default="llama3.2",
        help_text="Ollama model name for cloud API. Set OLLAMA_API_KEY.",
    )
    cloud_ollama_url = models.URLField(
        max_length=500, default="https://ollama.com",
        help_text="Base URL of the Ollama cloud API.",
    )

    # --- OpenAI ---
    openai_model = models.CharField(
        max_length=100, default="gpt-4o-mini",
        help_text="OpenAI model name. API key from OPENAI_API_KEY env var.",
    )

    # --- Anthropic ---
    anthropic_model = models.CharField(
        max_length=100, default="claude-haiku-4-5-20251001",
        help_text="Claude model ID. API key from ANTHROPIC_API_KEY env var.",
    )

    # --- Prompt ---
    prompt_broadcast_day = models.TextField(
        default=_DEFAULT_BROADCAST_DAY_PROMPT,
        help_text=(
            "Prompt template for daily broadcast summarization. "
            "Placeholders: {radio_name}, {radio_location}, {radio_language}, "
            "{date}, {timezone}, {timeline}."
        ),
    )

    enable_web_scraping = models.BooleanField(
        default=False,
        help_text="Placeholder for future web scraping of radio schedules.",
    )

    class Meta:
        verbose_name = "Daily Summarization Settings"
        verbose_name_plural = "Daily Summarization Settings"

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        pass

    @classmethod
    def get_settings(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return "Daily Summarization Settings"


class BroadcastDaySummary(models.Model):
    """A reconstructed broadcast day for a radio station on a given date."""
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("done", "Done"),
        ("failed", "Failed"),
    ]

    radio = models.ForeignKey(
        Radio, on_delete=models.CASCADE, related_name="broadcast_days",
    )
    date = models.DateField()
    status = models.CharField(
        max_length=10, choices=STATUS_CHOICES, default="pending", db_index=True,
    )
    error = models.TextField(blank=True, default="")
    overview = models.TextField(blank=True, default="")
    tags = models.ManyToManyField(Tag, related_name="broadcast_days", blank=True)
    recording_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("radio", "date")
        ordering = ["-date"]

    def __str__(self):
        return f"{self.radio.name} — {self.date}"


class ShowBlock(models.Model):
    """An individual show within a broadcast day."""
    SHOW_TYPE_CHOICES = [
        ("music", "Music"),
        ("news", "News"),
        ("talk", "Talk"),
        ("sports", "Sports"),
        ("cultural", "Cultural"),
        ("religious", "Religious"),
        ("spot", "Spot / Advertisement"),
        ("mixed", "Mixed"),
        ("unknown", "Unknown"),
    ]

    broadcast_day = models.ForeignKey(
        BroadcastDaySummary, on_delete=models.CASCADE, related_name="shows",
    )
    name = models.CharField(max_length=255)
    show_type = models.CharField(
        max_length=10, choices=SHOW_TYPE_CHOICES, default="unknown",
    )
    start_time = models.DateTimeField(help_text="Absolute start time in radio's local timezone.")
    end_time = models.DateTimeField(help_text="Absolute end time in radio's local timezone.")
    summary = models.TextField(blank=True, default="")
    show_url = models.URLField(blank=True, default="", help_text="For future web scraping.")
    tags = models.ManyToManyField(Tag, related_name="show_blocks", blank=True)
    songs = models.ManyToManyField(
        SongOccurrence, related_name="show_blocks", blank=True,
    )
    order = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ["order", "start_time"]

    def __str__(self):
        return f"{self.name} ({self.start_time:%H:%M}–{self.end_time:%H:%M})"


class RadioUser(AbstractUser):
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)


class EmailVerificationToken(models.Model):
    user = models.ForeignKey(RadioUser, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
