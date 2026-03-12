from rest_framework import serializers

from radios.models import (
    Radio, Recording, Stream, TranscriptionSegment,
    Tag, ChunkSummary, DailySummary, Song, SongOccurrence,
    Artist, Genre,
)


class RadioSerializer(serializers.ModelSerializer):
    country_name = serializers.CharField(source="country.name", read_only=True, default="")

    class Meta:
        model = Radio
        fields = [
            "slug", "name", "description", "country", "country_name",
            "city", "languages", "website", "frequencies", "motto",
            "is_fm", "is_am", "is_dab", "is_sw", "is_web",
            "since", "until", "created_at",
        ]


class StreamBriefSerializer(serializers.ModelSerializer):
    radio_slug = serializers.CharField(source="radio.slug", read_only=True, default=None)
    radio_name = serializers.CharField(source="radio.name", read_only=True, default=None)

    class Meta:
        model = Stream
        fields = ["id", "name", "radio_slug", "radio_name"]


class RecordingBriefSerializer(serializers.ModelSerializer):
    stream = StreamBriefSerializer(read_only=True)

    class Meta:
        model = Recording
        fields = ["id", "start_time", "end_time", "stream"]


class TranscriptSearchResultSerializer(serializers.ModelSerializer):
    recording_id = serializers.UUIDField(source="recording.id", read_only=True)
    recording_start = serializers.DateTimeField(source="recording.start_time", read_only=True)
    radio_slug = serializers.CharField(source="recording.stream.radio.slug", read_only=True, default=None)
    radio_name = serializers.CharField(source="recording.stream.radio.name", read_only=True, default=None)
    snippet = serializers.SerializerMethodField()

    class Meta:
        model = TranscriptionSegment
        fields = [
            "id", "segment_type", "start_offset", "end_offset",
            "text", "text_english", "language", "confidence",
            "recording_id", "recording_start",
            "radio_slug", "radio_name",
            "snippet",
        ]

    def get_snippet(self, obj):
        query = self.context.get("query", "")
        if not query:
            return ""
        text = obj.text or obj.text_english or ""
        # Simple snippet: find query in text and return surrounding context
        lower_text = text.lower()
        lower_query = query.lower()
        idx = lower_text.find(lower_query)
        if idx == -1:
            return text[:200]
        start = max(0, idx - 80)
        end = min(len(text), idx + len(query) + 80)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet


class ArtistSerializer(serializers.ModelSerializer):
    class Meta:
        model = Artist
        fields = ["id", "name", "shazam_id", "musicbrainz_id"]


class GenreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Genre
        fields = ["id", "name", "slug"]


class SongSerializer(serializers.ModelSerializer):
    artist_ref = ArtistSerializer(read_only=True)
    genres = GenreSerializer(many=True, read_only=True)

    class Meta:
        model = Song
        fields = [
            "id", "title", "artist", "artist_ref", "shazam_key",
            "musicbrainz_id", "genres", "album_name", "album_cover_url",
            "release_year", "isrc", "duration_seconds",
        ]


class SongOccurrenceSearchResultSerializer(serializers.ModelSerializer):
    """Serializer for song search results based on SongOccurrence."""
    song = SongSerializer(read_only=True)
    recording_id = serializers.UUIDField(source="segment.recording.id", read_only=True)
    recording_start = serializers.DateTimeField(source="segment.recording.start_time", read_only=True)
    radio_slug = serializers.CharField(
        source="segment.recording.stream.radio.slug", read_only=True, default=None,
    )
    radio_name = serializers.CharField(
        source="segment.recording.stream.radio.name", read_only=True, default=None,
    )

    class Meta:
        model = SongOccurrence
        fields = [
            "id", "song", "start_offset", "end_offset", "confidence",
            "recording_id", "recording_start",
            "radio_slug", "radio_name",
        ]


class SongAggregateSerializer(serializers.Serializer):
    song_title = serializers.CharField()
    song_artist = serializers.CharField()
    shazam_key = serializers.CharField(allow_null=True)
    play_count = serializers.IntegerField()


class TagSerializer(serializers.ModelSerializer):
    summary_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model = Tag
        fields = ["id", "name", "slug", "summary_count"]


class ChunkSummarySerializer(serializers.ModelSerializer):
    recording = RecordingBriefSerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)

    class Meta:
        model = ChunkSummary
        fields = ["id", "summary_text", "tags", "recording", "created_at"]


class DailySummarySerializer(serializers.ModelSerializer):
    radio_slug = serializers.CharField(source="radio.slug", read_only=True)
    radio_name = serializers.CharField(source="radio.name", read_only=True)
    tags = TagSerializer(many=True, read_only=True)

    class Meta:
        model = DailySummary
        fields = [
            "id", "radio_slug", "radio_name", "date",
            "summary_text", "chunk_count", "tags", "created_at",
        ]


class SummarySearchResultSerializer(serializers.Serializer):
    """Unified serializer for mixed chunk + daily summary results."""
    id = serializers.IntegerField()
    summary_type = serializers.CharField()
    summary_text = serializers.CharField()
    radio_slug = serializers.CharField(allow_null=True)
    radio_name = serializers.CharField(allow_null=True)
    date = serializers.DateField(allow_null=True)
    tags = TagSerializer(many=True)
    created_at = serializers.DateTimeField()
