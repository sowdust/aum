from django.db import models
from django.utils import timezone
from django.core.validators import URLValidator, MinValueValidator, MaxValueValidator
from django_countries.fields import CountryField

BROADCAST_TYPE_CHOICES = [
    ("fm", "FM"),
    ("am", "AM"),
    ("sw", "Shortwave"),
    ("web", "Online"),
    ("dab", "Digital Audio Broadcasting"),
]


class Radio(models.Model):
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True)
    description = models.TextField(blank=True)
    country = CountryField(blank=True, null=True)
    city = models.CharField(max_length=100, blank=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    languages = models.CharField(max_length=100, blank=True)
    """    from django.contrib.postgres.fields import ArrayField

        broadcast_types = ArrayField(
            models.CharField(max_length=20, choices=BROADCAST_TYPE_CHOICES),
            default=list,
            blank=True,
        )
    """
    frequencies = models.CharField(max_length=200, blank=True)
    website = models.URLField(blank=True, validators=[URLValidator()])
    contact_email = models.EmailField(blank=True)
    logo = models.ImageField(upload_to="logos/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    verified = models.BooleanField(default=False)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class AudioStream(models.Model):
    name = models.CharField(max_length=200)
    models.ForeignKey(Radio, on_delete=models.SET_NULL, related_name="streams")
    streaming_url = models.URLField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


import uuid
import os
from django.db import models
from django.conf import settings
from django.utils.text import slugify


def safe_stream_folder(stream_name: str) -> str:
    """
    Produces a stable directory name from the stream name.
    Example: "Radio Rock 101.5" -> "radio-rock-101-5"
    """
    return slugify(stream_name) or "stream"


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


class StreamRecording(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    stream = models.ForeignKey("AudioStream", on_delete=models.CASCADE, related_name="recordings")
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    file = models.FileField(upload_to=recording_upload_path)

    class Meta:
        ordering = ["-start_time"]

    def __str__(self):
        return f"{self.stream.radio} [{self.start_time} - {self.end_time}]"









"""
class StreamEndpoint(models.Model):
    station = models.ForeignKey(Radio, on_delete=models.CASCADE, related_name="streams")
    name = models.CharField(max_length=255, blank=True)
    url = models.URLField(validators=[URLValidator()])
    is_primary = models.BooleanField(default=False)
    bitrate = models.PositiveIntegerField(null=True, blank=True, help_text="kbps")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("station", "url")

    def __str__(self):
        return f"{self.station.name} — {self.name or self.url}"


class Recording(models.Model):
    station = models.ForeignKey(Radio, on_delete=models.CASCADE, related_name="recordings")
    stream = models.ForeignKey(StreamEndpoint, on_delete=models.SET_NULL, null=True, blank=True)
    file = models.FileField(upload_to="recordings/%Y/%m/%d/")
    started_at = models.DateTimeField()
    duration_seconds = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    size_bytes = models.BigIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    checksum = models.CharField(max_length=128, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        started = timezone.localtime(self.started_at).strftime("%Y-%m-%d %H:%M")
        return f"{self.station.name} — {started} ({self.duration_seconds}s)"

"""
