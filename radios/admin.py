from django.db import models
# Admin configuration
from django.contrib import admin
#from django.contrib.postgres.fields import ArrayField
from django import forms
from .models import (
    Radio, AudioFeed, Recording, Stream, RadioUser,
    TranscriptionSegment, ChunkSummary, DailySummary, FeedAnomaly,
    GlobalPipelineSettings, TranscriptionSettings, SummarizationSettings,
    DailySummarizationSettings, BroadcastDaySummary, ShowBlock,
    Song, SongOccurrence, Artist, Genre,
)

@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    list_display = (
        "__str__", "segmentation_status", "fingerprinting_status",
        "transcription_status", "summarization_status",
        "analysis_started_at", "analysis_completed_at",
    )
    list_filter = (
        "segmentation_status", "fingerprinting_status",
        "transcription_status", "summarization_status",
        "stream",
    )
    search_fields = ("id", "stream__name")
    readonly_fields = (
        "id", "analysis_started_at", "analysis_completed_at",
        "analysis_status_display", "analysis_error_display",
    )
    fieldsets = [
        (None, {"fields": ["id", "stream", "start_time", "end_time", "file"]}),
        ("Analysis Status", {
            "fields": [
                "analysis_status_display",
                "segmentation_status", "segmentation_error",
                "fingerprinting_status", "fingerprinting_error",
                "transcription_status", "transcription_error",
                "summarization_status", "summarization_error",
                "analysis_started_at", "analysis_completed_at",
            ],
        }),
    ]
    actions = ["retry_failed_stages"]

    @admin.display(description="Overall status")
    def analysis_status_display(self, obj):
        return obj.analysis_status

    @admin.display(description="Errors")
    def analysis_error_display(self, obj):
        return obj.analysis_error or "(none)"

    @admin.action(description="Retry failed stages (reset to pending)")
    def retry_failed_stages(self, request, queryset):
        count = 0
        for stage in ("segmentation", "fingerprinting", "transcription", "summarization"):
            field = f"{stage}_status"
            error_field = f"{stage}_error"
            count += queryset.filter(**{field: "failed"}).update(
                **{field: "pending", error_field: ""}
            )
        self.message_user(request, f"Reset {count} failed stage(s) to pending.")
admin.site.register(RadioUser)


class StreamInline(admin.TabularInline):
    model = Stream
    extra = 0
    fields = [
        "name", "url", "is_active",
        "enable_recording", "enable_segmentation", "enable_fingerprinting",
        "enable_transcription", "enable_summarization", "enable_daily_summarization",
        "recording_owner_visible", "segmentation_owner_visible",
        "fingerprinting_owner_visible", "transcription_owner_visible",
        "summarization_owner_visible", "daily_summarization_owner_visible",
    ]


@admin.register(Stream)
class StreamAdmin(admin.ModelAdmin):
    list_display = (
        "name", "radio", "audio_feed", "is_active",
        "enable_recording", "recording_status",
    )
    list_filter = ("is_active", "recording_status", "radio", "audio_feed")
    search_fields = ("name", "url")
    readonly_fields = ("recording_status", "recording_error", "recording_started_at")
    fieldsets = [
        (None, {"fields": ["radio", "audio_feed", "name", "url", "is_active"]}),
        ("Recording Status", {
            "fields": ["recording_status", "recording_error", "recording_started_at"],
            "description": "Live recording state — managed by the record_streams service.",
        }),
        ("Pipeline Controls", {
            "description": "Admin-controlled per-stream enables. Still obey global settings and upstream dependencies.",
            "fields": [
                "enable_recording", "enable_segmentation", "enable_fingerprinting",
                "enable_transcription", "enable_summarization", "enable_daily_summarization",
            ],
        }),
        ("Visibility Settings", {
            "description": "What results the radio owner sees for this stream.",
            "fields": [
                "recording_owner_visible", "segmentation_owner_visible",
                "fingerprinting_owner_visible", "transcription_owner_visible",
                "summarization_owner_visible", "daily_summarization_owner_visible",
            ],
        }),
        ("Advanced / Public Visibility", {
            "classes": ("collapse",),
            "description": "Public visibility — not yet exposed in owner UI. Default False.",
            "fields": [
                "recording_public_visible", "segmentation_public_visible",
                "fingerprinting_public_visible", "transcription_public_visible",
                "summarization_public_visible", "daily_summarization_public_visible",
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
                "enable_daily_summarization",
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


@admin.register(TranscriptionSettings)
class TranscriptionSettingsAdmin(admin.ModelAdmin):
    fieldsets = [
        ("Backend", {
            "fields": ["backend"],
            "description": "Select which backend to use for speech transcription.",
        }),
        ("Local Backend (faster-whisper)", {
            "fields": ["local_model_size", "local_device", "local_compute_type"],
            "description": (
                "Used when backend is 'Local'. faster-whisper runs entirely on this machine. "
                "Larger models are more accurate but slower and require more RAM."
            ),
        }),
        ("OpenAI Whisper API", {
            "classes": ("collapse",),
            "fields": ["openai_model"],
            "description": "API key must be set in the OPENAI_API_KEY environment variable.",
        }),
        ("Anthropic (Claude)", {
            "classes": ("collapse",),
            "fields": ["anthropic_model"],
            "description": "API key must be set in the ANTHROPIC_API_KEY environment variable.",
        }),
        ("Ollama (local or cloud)", {
            "classes": ("collapse",),
            "fields": ["ollama_model", "ollama_base_url"],
            "description": (
                "Ollama's OpenAI-compatible transcription API. "
                "For a local instance leave the URL as http://localhost:11434 — no API key needed. "
                "For ollama.com cloud, set the URL to https://api.ollama.com and set the "
                "OLLAMA_API_KEY environment variable."
            ),
        }),
        ("RunPod (faster-whisper serverless)", {
            "classes": ("collapse",),
            "fields": ["runpod_endpoint_id", "runpod_model", "runpod_translate"],
            "description": (
                "Serverless faster-whisper on RunPod. "
                "Set your endpoint ID (runpod.io/console/serverless) here and the "
                "RUNPOD_API_KEY environment variable on the server. "
                "Audio is sent as base64. Multiple segments are submitted in parallel "
                "(up to 10 concurrent jobs). "
                "Enable 'Translate' to get English translations for non-English audio "
                "(costs one extra API call per non-English segment)."
            ),
        }),
        ("Transcription Correction (LLM)", {
            "classes": ("collapse",),
            "fields": [
                "enable_correction", "correction_backend",
                "correction_local_ollama_model", "correction_local_ollama_url",
                "correction_cloud_ollama_model", "correction_cloud_ollama_url",
                "correction_openai_model", "correction_anthropic_model",
                "correction_prompt",
            ],
            "description": (
                "Optional LLM post-processing to fix transcription errors and produce "
                "English translation from corrected text. Runs after speech-to-text, "
                "before summarization."
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
        TranscriptionSettings.get_settings()
        return redirect(reverse("admin:radios_transcriptionsettings_change", args=[1]))


@admin.register(SummarizationSettings)
class SummarizationSettingsAdmin(admin.ModelAdmin):
    fieldsets = [
        ("Backend", {
            "fields": ["backend"],
            "description": "Select which LLM backend to use for summarization.",
        }),
        ("Local Ollama", {
            "fields": ["local_ollama_model", "local_ollama_url"],
            "description": "Ollama running locally. No API key needed.",
        }),
        ("Cloud Ollama", {
            "classes": ("collapse",),
            "fields": ["cloud_ollama_model", "cloud_ollama_url"],
            "description": "ollama.com cloud API. Set OLLAMA_API_KEY environment variable.",
        }),
        ("OpenAI", {
            "classes": ("collapse",),
            "fields": ["openai_model"],
            "description": "API key must be set in the OPENAI_API_KEY environment variable.",
        }),
        ("Anthropic (Claude)", {
            "classes": ("collapse",),
            "fields": ["anthropic_model"],
            "description": "API key must be set in the ANTHROPIC_API_KEY environment variable.",
        }),
        ("Prompts", {
            "fields": ["prompt_chunk", "prompt_daily"],
            "description": (
                "Edit the prompts sent to the LLM. "
                "Placeholders — chunk prompt: {content}, {language_hint}; "
                "daily prompt: {content}."
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
        SummarizationSettings.get_settings()
        return redirect(reverse("admin:radios_summarizationsettings_change", args=[1]))


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
        ("Stream Codec", {
            "classes": ("collapse",),
            "fields": ["stream_codec"],
            "description": "Audio codec of the stream. Auto-detected on first recording if left blank.",
        }),
    ]


@admin.register(Artist)
class ArtistAdmin(admin.ModelAdmin):
    list_display = ("name", "shazam_id", "musicbrainz_id", "created_at")
    search_fields = ("name", "shazam_id", "musicbrainz_id")
    readonly_fields = ("created_at",)


@admin.register(Genre)
class GenreAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "created_at")
    search_fields = ("name",)
    prepopulated_fields = {"slug": ("name",)}
    readonly_fields = ("created_at",)


class SongOccurrenceInline(admin.TabularInline):
    model = SongOccurrence
    extra = 0
    readonly_fields = ("segment", "start_offset", "end_offset", "confidence")


@admin.register(Song)
class SongAdmin(admin.ModelAdmin):
    list_display = ("title", "artist", "shazam_key", "album_name", "release_year", "created_at")
    search_fields = ("title", "artist", "shazam_key", "album_name")
    list_filter = ("genres",)
    readonly_fields = ("created_at",)
    inlines = [SongOccurrenceInline]
    filter_horizontal = ("genres",)


@admin.register(SongOccurrence)
class SongOccurrenceAdmin(admin.ModelAdmin):
    list_display = ("song", "segment", "start_offset", "end_offset", "confidence")
    search_fields = ("song__title", "song__artist")
    list_filter = ("confidence",)
    readonly_fields = ("segment", "song", "start_offset", "end_offset", "confidence")


class SegmentSongOccurrenceInline(admin.TabularInline):
    model = SongOccurrence
    extra = 0
    readonly_fields = ("song", "start_offset", "end_offset", "confidence")


@admin.register(TranscriptionSegment)
class TranscriptionSegmentAdmin(admin.ModelAdmin):
    list_display = (
        "recording", "segment_type", "start_offset", "end_offset",
        "language", "has_transcription", "get_songs",
    )
    list_filter = ("segment_type", "language")
    search_fields = ("text", "text_english")
    readonly_fields = ("recording", "segment_type", "start_offset", "end_offset")
    inlines = [SegmentSongOccurrenceInline]

    @admin.display(boolean=True, description="Transcribed")
    def has_transcription(self, obj):
        return bool(obj.text)

    @admin.display(description="Songs")
    def get_songs(self, obj):
        occurrences = obj.song_occurrences.select_related("song").all()
        if not occurrences:
            return ""
        return ", ".join(str(occ.song) for occ in occurrences)


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


@admin.register(DailySummarizationSettings)
class DailySummarizationSettingsAdmin(admin.ModelAdmin):
    fieldsets = [
        ("Backend", {
            "fields": ["backend"],
            "description": "Select which LLM backend to use for daily broadcast summarization.",
        }),
        ("Local Ollama", {
            "fields": ["local_ollama_model", "local_ollama_url"],
            "description": "Ollama running locally. No API key needed.",
        }),
        ("Cloud Ollama", {
            "classes": ("collapse",),
            "fields": ["cloud_ollama_model", "cloud_ollama_url"],
            "description": "ollama.com cloud API. Set OLLAMA_API_KEY environment variable.",
        }),
        ("OpenAI", {
            "classes": ("collapse",),
            "fields": ["openai_model"],
            "description": "API key must be set in the OPENAI_API_KEY environment variable.",
        }),
        ("Anthropic (Claude)", {
            "classes": ("collapse",),
            "fields": ["anthropic_model"],
            "description": "API key must be set in the ANTHROPIC_API_KEY environment variable.",
        }),
        ("Prompt", {
            "fields": ["prompt_broadcast_day"],
            "description": (
                "Edit the prompt sent to the LLM. "
                "Placeholders: {radio_name}, {radio_location}, {radio_language}, "
                "{date}, {timezone}, {timeline}."
            ),
        }),
        ("Future Features", {
            "classes": ("collapse",),
            "fields": ["enable_web_scraping"],
        }),
    ]

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def changelist_view(self, request, extra_context=None):
        from django.shortcuts import redirect
        from django.urls import reverse
        DailySummarizationSettings.get_settings()
        return redirect(reverse("admin:radios_dailysummarizationsettings_change", args=[1]))


class ShowBlockInline(admin.TabularInline):
    model = ShowBlock
    extra = 0
    readonly_fields = ("name", "show_type", "start_time", "end_time", "order")
    fields = ("order", "name", "show_type", "start_time", "end_time", "summary")


@admin.register(BroadcastDaySummary)
class BroadcastDaySummaryAdmin(admin.ModelAdmin):
    list_display = ("radio", "date", "status", "show_count", "recording_count", "updated_at")
    list_filter = ("status", "radio")
    search_fields = ("overview", "radio__name")
    readonly_fields = ("created_at", "updated_at")
    inlines = [ShowBlockInline]

    @admin.display(description="Shows")
    def show_count(self, obj):
        return obj.shows.count()


@admin.register(ShowBlock)
class ShowBlockAdmin(admin.ModelAdmin):
    list_display = ("name", "broadcast_day", "show_type", "start_time", "end_time", "order")
    list_filter = ("show_type",)
    search_fields = ("name", "summary")
    filter_horizontal = ("tags", "songs")


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
        ("Stream Codec", {
            "classes": ("collapse",),
            "fields": ["stream_codec"],
            "description": "Audio codec of the stream. Auto-detected on first recording if left blank.",
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