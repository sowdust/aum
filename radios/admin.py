from django.db import models
# Admin configuration
from django.contrib import admin
#from django.contrib.postgres.fields import ArrayField
from django import forms
from .models import (
    Radio, AudioFeed, Recording, Stream, RadioUser,
    TranscriptionSegment, ChunkSummary, DailySummary, FeedAnomaly,
    GlobalPipelineSettings,
)

admin.site.register(Recording)
admin.site.register(RadioUser)


class StreamInline(admin.TabularInline):
    model = Stream
    extra = 0
    fields = [
        "name", "url", "is_active",
        "enable_recording", "enable_segmentation", "enable_fingerprinting",
        "enable_transcription", "enable_summarization",
        "recording_owner_visible", "segmentation_owner_visible",
        "fingerprinting_owner_visible", "transcription_owner_visible",
        "summarization_owner_visible",
    ]


@admin.register(Stream)
class StreamAdmin(admin.ModelAdmin):
    list_display = ("name", "radio", "audio_feed", "is_active", "enable_recording")
    list_filter = ("is_active", "radio", "audio_feed")
    search_fields = ("name", "url")
    fieldsets = [
        (None, {"fields": ["radio", "audio_feed", "name", "url", "is_active"]}),
        ("Pipeline Controls", {
            "description": "Admin-controlled per-stream enables. Still obey global settings and upstream dependencies.",
            "fields": [
                "enable_recording", "enable_segmentation", "enable_fingerprinting",
                "enable_transcription", "enable_summarization",
            ],
        }),
        ("Visibility Settings", {
            "description": "What results the radio owner sees for this stream.",
            "fields": [
                "recording_owner_visible", "segmentation_owner_visible",
                "fingerprinting_owner_visible", "transcription_owner_visible",
                "summarization_owner_visible",
            ],
        }),
        ("Advanced / Public Visibility", {
            "classes": ("collapse",),
            "description": "Public visibility — not yet exposed in owner UI. Default False.",
            "fields": [
                "recording_public_visible", "segmentation_public_visible",
                "fingerprinting_public_visible", "transcription_public_visible",
                "summarization_public_visible",
            ],
        }),
    ]


@admin.register(GlobalPipelineSettings)
class GlobalPipelineSettingsAdmin(admin.ModelAdmin):
    fieldsets = [
        ("Pipeline Stage Switches", {
            "fields": [
                "enable_recording",
                "enable_segmentation",
                "enable_fingerprinting",
                "enable_transcription",
                "enable_summarization",
            ],
            "description": (
                "Global kill switches. Disabling a stage here overrides all per-source "
                "settings and disables all downstream stages too."
            ),
        }),
        ("Proxy", {
            "fields": ["proxy_url"],
            "description": (
                "Global proxy for stream recording. Sources with proxy_mode='global' will use this URL. "
                "Example: http://proxy:8080 or socks5://proxy:1080"
            ),
        }),
    ]

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def changelist_view(self, request, extra_context=None):
        from django.shortcuts import redirect
        from django.urls import reverse
        GlobalPipelineSettings.get_settings()  # ensure row exists
        return redirect(reverse("admin:radios_globalpipelinesettings_change", args=[1]))


@admin.register(AudioFeed)
class AudioFeedAdmin(admin.ModelAdmin):
    list_display = ("name", "feed_type", "country", "city", "timezone", "proxy_mode")
    search_fields = ("name", "city", "description")
    prepopulated_fields = {"slug": ("name",)}
    list_filter = ("feed_type", "country", "timezone", "proxy_mode")
    inlines = [StreamInline]
    fieldsets = [
        (None, {"fields": [
            "name", "slug", "description", "feed_type",
            "country", "city", "latitude", "longitude", "languages", "website",
            "frequencies", "show_archive", "timezone",
            "recording_start_hour", "recording_end_hour",
        ]}),
        ("Proxy Settings", {
            "fields": ["proxy_mode", "proxy_url"],
            "description": (
                "'No Proxy' = direct connection. 'Use Global Proxy' = use the URL from Global Pipeline Settings. "
                "'Custom Proxy' = use the proxy URL specified below."
            ),
        }),
    ]


@admin.register(TranscriptionSegment)
class TranscriptionSegmentAdmin(admin.ModelAdmin):
    list_display = (
        "recording", "segment_type", "start_offset", "end_offset",
        "language", "has_transcription", "song_artist", "song_title",
    )
    list_filter = ("segment_type", "language")
    search_fields = ("text", "text_english", "song_title", "song_artist")
    readonly_fields = ("recording", "segment_type", "start_offset", "end_offset")

    @admin.display(boolean=True, description="Transcribed")
    def has_transcription(self, obj):
        return bool(obj.text)


@admin.register(ChunkSummary)
class ChunkSummaryAdmin(admin.ModelAdmin):
    list_display = ("recording", "created_at")
    search_fields = ("summary_text",)


@admin.register(DailySummary)
class DailySummaryAdmin(admin.ModelAdmin):
    list_display = ("radio", "date", "chunk_count", "created_at")
    list_filter = ("radio", "date")
    search_fields = ("summary_text",)


@admin.register(FeedAnomaly)
class FeedAnomalyAdmin(admin.ModelAdmin):
    list_display = ("recording", "anomaly_type", "start_offset", "end_offset", "audio_level_db", "detected_at")
    list_filter = ("anomaly_type",)
    search_fields = ("transcript",)


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
    list_display = ("name", "country", "city", "timezone", "proxy_mode", "recording_start_hour", "recording_end_hour")
    search_fields = ("name", "city", "description")
    prepopulated_fields = {"slug": ("name",)}
    list_filter = ("country", "timezone", "proxy_mode")
    inlines = [StreamInline]
    fieldsets = [
        (None, {"fields": [
            "name", "slug", "description", "logo", "motto", "since", "until",
            "country", "city", "latitude", "longitude", "languages", "website",
            "frequencies", "show_archive", "timezone",
            "recording_start_hour", "recording_end_hour",
            "is_fm", "is_am", "is_dab", "is_sw", "is_web",
            "contact_email",
        ]}),
        ("Proxy Settings", {
            "fields": ["proxy_mode", "proxy_url"],
            "description": (
                "'No Proxy' = direct connection. 'Use Global Proxy' = use the URL from Global Pipeline Settings. "
                "'Custom Proxy' = use the proxy URL specified below."
            ),
        }),
    ]

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