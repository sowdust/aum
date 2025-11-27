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
    name = models.CharField(max_length=180)
    slug = models.SlugField(max_length=220, unique=True)
    motto = models.CharField(max_length=255, blank=True)
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


class Stream(models.Model):
    radio = models.ForeignKey(Radio, on_delete=models.CASCADE, related_name="streams")
    name = models.CharField(max_length=200)
    url = models.URLField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


import uuid
from django.db import models
from django.utils.text import slugify


def safe_stream_folder(stream_name: str) -> str:
    """
    Produces a stable directory name from the stream name.
    Example: "Radio Rock 101.5" -> "radio-rock-101-5"
    """
    return slugify(stream_name) or uuid.uuid4


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


class Recording(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    stream = models.ForeignKey("Stream", on_delete=models.CASCADE, related_name="recordings")
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    file = models.FileField(upload_to=recording_upload_path)

    class Meta:
        ordering = ["-start_time"]

    def __str__(self):
        return f"{self.stream.radio} [{self.start_time} - {self.end_time}]"

