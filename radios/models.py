from django.conf import settings
from django.db import models
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
    logo = models.ImageField(upload_to="logos/", null=True, blank=True)
    motto = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    is_fm = models.BooleanField(default=False)
    is_am = models.BooleanField(default=False)
    is_dab = models.BooleanField(default=False)
    is_sw = models.BooleanField(default=False)
    is_web = models.BooleanField(default=False)
    frequencies = models.CharField(max_length=200, blank=True)
    website = models.URLField(blank=True, validators=[URLValidator()])
    country = CountryField(blank=True, null=True)
    city = models.CharField(max_length=100, blank=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    languages = models.CharField(max_length=255, blank=True)
    contact_email = models.EmailField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="managed_radios"
    )

    permissions = (
        ('create_radio', 'Create Radio'),
        ('edit_radio', 'Edit Radio'),
        ('delete_radio', 'Delete Radio'),
    )

    # assign_perm('create_radio', user, radio)

    def is_owned(self):
        return self.owner is not None

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

