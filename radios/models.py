from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.core.validators import URLValidator
from django_countries.fields import CountryField
from django.utils.text import slugify
from django.db import models
import uuid

BROADCAST_TYPE_CHOICES = [
    ("fm", "FM"),
    ("am", "AM"),
    ("sw", "Shortwave"),
    ("web", "Online"),
    ("dab", "Digital Audio Broadcasting"),
]

class Radio(models.Model):
    name = models.CharField(max_length=180)
    slug = models.SlugField(primary_key=True, unique=True, blank=True)
    logo = models.ImageField(upload_to="logos/", null=True, blank=True)
    motto = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    since = models.DateField(blank=True, null=True, help_text="Radio creation date")
    until = models.DateField(blank=True, null=True, help_text="Radio end date")
    is_fm = models.BooleanField(default=False, help_text="Does the radio broadcast via FM?")
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
    show_archive = models.BooleanField(default=False)

    members = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        through="RadioMembership",
        related_name="radios",
        blank = True,
    )

    def _generate_unique_slug(self):
        """
        Generate a slug from name. If the slug is taken, add city, country,
        frequency in that order. If still not unique, append -2, -3, ...
        """

        base_slug = slugify(self.name)
        slug_candidate = base_slug

        def is_unique(slug_to_test):
            qs = Radio.objects.filter(slug=slug_to_test)
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
            slug_candidate = slugify(
                f"{self.name}-{self.city}-{self.country}-{self.frequency}"
            )
            if is_unique(slug_candidate):
                return slug_candidate

        counter = 2
        while True:
            slug_candidate = f"{base_slug}-{counter}"
            if is_unique(slug_candidate):
                return slug_candidate
            counter += 1



    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = self._generate_unique_slug()

        super().save(*args, **kwargs)



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
    radio = models.ForeignKey(Radio, on_delete=models.CASCADE, related_name="streams")
    name = models.CharField(max_length=200)
    url = models.URLField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name


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


class RadioUser(AbstractUser):
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)

class EmailVerificationToken(models.Model):
    user = models.ForeignKey(RadioUser, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)