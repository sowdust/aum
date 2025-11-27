from django.db import models
# Admin configuration
from django.contrib import admin
#from django.contrib.postgres.fields import ArrayField
from django import forms
from .models import Radio, Recording, AudioStream

admin.site.register(AudioStream)
admin.site.register(Recording)


class RadioAdminForm(forms.ModelForm):
    class Meta:
        model = Radio
        fields = "__all__"
#        widgets = {
#            "broadcast_types": forms.SelectMultiple(choices=BROADCAST_TYPE_CHOICES),
#        }

@admin.register(Radio)
class RadioAdmin(admin.ModelAdmin):
    form = RadioAdminForm
    list_display = ("name", "country", "city", "verified")
    list_filter = ("country", "verified")
    search_fields = ("name", "city", "description")
    prepopulated_fields = {"slug": ("name",)}

"""
@admin.register(StreamEndpoint)
class StreamEndpointAdmin(admin.ModelAdmin):
    list_display = ("station", "name", "url", "bitrate", "is_primary")
    search_fields = ("url", "station__name")

@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    list_display = ("station", "started_at", "duration_seconds", "file")
    list_filter = ("station",)

"""