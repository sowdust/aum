from django.urls import path, include
from rest_framework.routers import DefaultRouter

from radios.api.views import (
    RadioViewSet,
    TranscriptSearchView,
    SongSearchView,
    TagSearchView,
    TagDetailView,
    SummarySearchView,
)

router = DefaultRouter()
router.register(r"radios", RadioViewSet, basename="radio")

urlpatterns = [
    path("", include(router.urls)),
    path("search/transcripts/", TranscriptSearchView.as_view(), name="search-transcripts"),
    path("search/songs/", SongSearchView.as_view(), name="search-songs"),
    path("search/tags/", TagSearchView.as_view(), name="search-tags"),
    path("search/summaries/", SummarySearchView.as_view(), name="search-summaries"),
    path("tags/<slug:slug>/", TagDetailView.as_view(), name="tag-detail"),
]
