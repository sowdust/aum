from django.db.models import Q, Count
from rest_framework import viewsets, generics
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from radios.models import (
    Radio, TranscriptionSegment, Tag, ChunkSummary, DailySummary,
    SongOccurrence,
)
from radios.api.serializers import (
    RadioSerializer,
    TranscriptSearchResultSerializer,
    SongOccurrenceSearchResultSerializer,
    SongAggregateSerializer,
    TagSerializer,
    ChunkSummarySerializer,
    DailySummarySerializer,
    SummarySearchResultSerializer,
)
from radios.api.filters import RadioFilter
from radios.api.permissions import get_visible_stream_ids
from radios.api.fts import search_transcription_fts, search_summary_fts


class RadioViewSet(viewsets.ReadOnlyModelViewSet):
    """List and retrieve radios with search/filter support."""
    queryset = Radio.objects.all()
    serializer_class = RadioSerializer
    filterset_class = RadioFilter
    permission_classes = [AllowAny]
    lookup_field = "slug"


class TranscriptSearchView(generics.ListAPIView):
    """Full-text search across transcription segments using FTS5."""
    serializer_class = TranscriptSearchResultSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        query = self.request.query_params.get("q", "").strip()
        if not query:
            return TranscriptionSegment.objects.none()

        # FTS5 search for candidate IDs
        fts_results = search_transcription_fts(query)
        if not fts_results:
            return TranscriptionSegment.objects.none()

        segment_ids = [row[0] for row in fts_results]
        # Preserve FTS5 rank ordering
        id_to_rank = {row[0]: idx for idx, row in enumerate(fts_results)}

        # Visibility enforcement
        visible_stream_ids = get_visible_stream_ids(
            self.request.user, "transcription"
        )

        qs = (
            TranscriptionSegment.objects
            .filter(id__in=segment_ids, recording__stream_id__in=visible_stream_ids)
            .select_related("recording", "recording__stream", "recording__stream__radio")
        )

        # Additional filters
        radio = self.request.query_params.get("radio")
        if radio:
            qs = qs.filter(recording__stream__radio__slug=radio)

        date_from = self.request.query_params.get("date_from")
        if date_from:
            qs = qs.filter(recording__start_time__date__gte=date_from)

        date_to = self.request.query_params.get("date_to")
        if date_to:
            qs = qs.filter(recording__start_time__date__lte=date_to)

        language = self.request.query_params.get("language")
        if language:
            qs = qs.filter(language=language)

        segment_type = self.request.query_params.get("segment_type")
        if segment_type:
            qs = qs.filter(segment_type=segment_type)

        # Sort by FTS5 rank
        results = list(qs)
        results.sort(key=lambda s: id_to_rank.get(s.id, 9999))
        return results

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        ctx["query"] = self.request.query_params.get("q", "")
        return ctx


class SongSearchView(generics.ListAPIView):
    """Search for songs identified via fingerprinting (uses SongOccurrence)."""
    permission_classes = [AllowAny]

    def get_serializer_class(self):
        if self.request.query_params.get("group") == "true":
            return SongAggregateSerializer
        return SongOccurrenceSearchResultSerializer

    def get_queryset(self):
        visible_stream_ids = get_visible_stream_ids(
            self.request.user, "fingerprinting"
        )

        qs = (
            SongOccurrence.objects
            .filter(
                segment__recording__stream_id__in=visible_stream_ids,
            )
            .select_related(
                "song", "song__artist_ref",
                "segment", "segment__recording",
                "segment__recording__stream",
                "segment__recording__stream__radio",
            )
            .prefetch_related("song__genres")
        )

        query = self.request.query_params.get("q", "").strip()
        if query:
            qs = qs.filter(
                Q(song__title__icontains=query) | Q(song__artist__icontains=query)
            )

        title = self.request.query_params.get("title", "").strip()
        if title:
            qs = qs.filter(song__title__icontains=title)

        artist = self.request.query_params.get("artist", "").strip()
        if artist:
            qs = qs.filter(song__artist__icontains=artist)

        radio = self.request.query_params.get("radio")
        if radio:
            qs = qs.filter(segment__recording__stream__radio__slug=radio)

        date_from = self.request.query_params.get("date_from")
        if date_from:
            qs = qs.filter(segment__recording__start_time__date__gte=date_from)

        date_to = self.request.query_params.get("date_to")
        if date_to:
            qs = qs.filter(segment__recording__start_time__date__lte=date_to)

        return qs

    def list(self, request, *args, **kwargs):
        if request.query_params.get("group") == "true":
            qs = self.get_queryset()
            aggregated = (
                qs.values("song__title", "song__artist", "song__shazam_key")
                .annotate(play_count=Count("id"))
                .order_by("-play_count")
            )
            results = [
                {
                    "song_title": r["song__title"],
                    "song_artist": r["song__artist"],
                    "shazam_key": r["song__shazam_key"],
                    "play_count": r["play_count"],
                }
                for r in aggregated
            ]
            page = self.paginate_queryset(results)
            if page is not None:
                serializer = SongAggregateSerializer(page, many=True)
                return self.get_paginated_response(serializer.data)
            serializer = SongAggregateSerializer(results, many=True)
            return Response(serializer.data)
        return super().list(request, *args, **kwargs)


class TagSearchView(generics.ListAPIView):
    """Search and browse tags, scoped by visibility."""
    serializer_class = TagSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        visible_stream_ids = get_visible_stream_ids(
            self.request.user, "summarization"
        )

        # Only return tags that appear in visible summaries
        qs = Tag.objects.filter(
            Q(chunk_summaries__recording__stream_id__in=visible_stream_ids)
            | Q(daily_summaries__radio__streams__id__in=visible_stream_ids)
        ).distinct()

        query = self.request.query_params.get("q", "").strip()
        if query:
            qs = qs.filter(name__icontains=query)

        radio = self.request.query_params.get("radio")
        if radio:
            qs = qs.filter(
                Q(chunk_summaries__recording__stream__radio__slug=radio)
                | Q(daily_summaries__radio__slug=radio)
            ).distinct()

        date_from = self.request.query_params.get("date_from")
        if date_from:
            qs = qs.filter(
                Q(chunk_summaries__recording__start_time__date__gte=date_from)
                | Q(daily_summaries__date__gte=date_from)
            )

        date_to = self.request.query_params.get("date_to")
        if date_to:
            qs = qs.filter(
                Q(chunk_summaries__recording__start_time__date__lte=date_to)
                | Q(daily_summaries__date__lte=date_to)
            )

        qs = qs.annotate(
            summary_count=Count("chunk_summaries", distinct=True)
            + Count("daily_summaries", distinct=True)
        ).order_by("-summary_count")

        return qs


class TagDetailView(generics.GenericAPIView):
    """Retrieve a tag with its associated summaries."""
    serializer_class = TagSerializer
    permission_classes = [AllowAny]
    lookup_field = "slug"
    queryset = Tag.objects.all()

    def get(self, request, *args, **kwargs):
        tag = self.get_object()
        visible_stream_ids = get_visible_stream_ids(
            request.user, "summarization"
        )

        chunk_summaries = (
            ChunkSummary.objects
            .filter(tags=tag, recording__stream_id__in=visible_stream_ids)
            .select_related("recording", "recording__stream", "recording__stream__radio")
            .prefetch_related("tags")
            .order_by("-recording__start_time")
        )

        daily_summaries = (
            DailySummary.objects
            .filter(tags=tag, radio__streams__id__in=visible_stream_ids)
            .select_related("radio")
            .prefetch_related("tags")
            .distinct()
            .order_by("-date")
        )

        data = {
            "tag": TagSerializer(tag).data,
            "chunk_summaries": ChunkSummarySerializer(chunk_summaries, many=True).data,
            "daily_summaries": DailySummarySerializer(daily_summaries, many=True).data,
        }
        return Response(data)


class SummarySearchView(generics.ListAPIView):
    """Full-text search across chunk and daily summaries using FTS5."""
    serializer_class = SummarySearchResultSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        query = self.request.query_params.get("q", "").strip()
        if not query:
            return []

        summary_type = self.request.query_params.get("type")
        if summary_type == "both":
            summary_type = None

        fts_results = search_summary_fts(query, summary_type=summary_type)
        if not fts_results:
            return []

        visible_stream_ids = get_visible_stream_ids(
            self.request.user, "summarization"
        )

        # Separate chunk and daily IDs
        chunk_ids = [r[0] for r in fts_results if r[1] == "chunk"]
        daily_ids = [r[0] for r in fts_results if r[1] == "daily"]

        results = []

        # Fetch chunk summaries
        if chunk_ids:
            chunk_qs = (
                ChunkSummary.objects
                .filter(id__in=chunk_ids, recording__stream_id__in=visible_stream_ids)
                .select_related("recording", "recording__stream", "recording__stream__radio")
                .prefetch_related("tags")
            )
            radio = self.request.query_params.get("radio")
            if radio:
                chunk_qs = chunk_qs.filter(recording__stream__radio__slug=radio)

            date_from = self.request.query_params.get("date_from")
            if date_from:
                chunk_qs = chunk_qs.filter(recording__start_time__date__gte=date_from)

            date_to = self.request.query_params.get("date_to")
            if date_to:
                chunk_qs = chunk_qs.filter(recording__start_time__date__lte=date_to)

            tags_param = self.request.query_params.get("tags")
            if tags_param:
                for tag_slug in tags_param.split(","):
                    tag_slug = tag_slug.strip()
                    if tag_slug:
                        chunk_qs = chunk_qs.filter(tags__slug=tag_slug)

            for cs in chunk_qs:
                results.append({
                    "id": cs.id,
                    "summary_type": "chunk",
                    "summary_text": cs.summary_text,
                    "radio_slug": getattr(cs.recording.stream.radio, "slug", None),
                    "radio_name": getattr(cs.recording.stream.radio, "name", None),
                    "date": cs.recording.start_time.date() if cs.recording.start_time else None,
                    "tags": cs.tags.all(),
                    "created_at": cs.created_at,
                })

        # Fetch daily summaries
        if daily_ids:
            daily_qs = (
                DailySummary.objects
                .filter(id__in=daily_ids, radio__streams__id__in=visible_stream_ids)
                .select_related("radio")
                .prefetch_related("tags")
                .distinct()
            )
            radio = self.request.query_params.get("radio")
            if radio:
                daily_qs = daily_qs.filter(radio__slug=radio)

            date_param = self.request.query_params.get("date")
            if date_param:
                daily_qs = daily_qs.filter(date=date_param)

            date_from = self.request.query_params.get("date_from")
            if date_from:
                daily_qs = daily_qs.filter(date__gte=date_from)

            date_to = self.request.query_params.get("date_to")
            if date_to:
                daily_qs = daily_qs.filter(date__lte=date_to)

            tags_param = self.request.query_params.get("tags")
            if tags_param:
                for tag_slug in tags_param.split(","):
                    tag_slug = tag_slug.strip()
                    if tag_slug:
                        daily_qs = daily_qs.filter(tags__slug=tag_slug)

            for ds in daily_qs:
                results.append({
                    "id": ds.id,
                    "summary_type": "daily",
                    "summary_text": ds.summary_text,
                    "radio_slug": ds.radio.slug,
                    "radio_name": ds.radio.name,
                    "date": ds.date,
                    "tags": ds.tags.all(),
                    "created_at": ds.created_at,
                })

        # Sort by FTS rank (preserve order from FTS results)
        id_type_to_rank = {(r[0], r[1]): idx for idx, r in enumerate(fts_results)}
        results.sort(key=lambda r: id_type_to_rank.get((r["id"], r["summary_type"]), 9999))

        return results

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
