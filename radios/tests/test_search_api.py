"""
Tests for the Search API endpoints.

Covers:
  - Radio search and filtering
  - Transcript FTS5 search with visibility enforcement
  - Song search (ORM-based)
  - Tag search and detail
  - Summary FTS5 search
  - FTS5 query sanitization (injection prevention)
  - Pagination

Run with:
    python manage.py test radios.tests.test_search_api
"""
import datetime

from django.test import TestCase, override_settings
from django.utils import timezone
from django.db import connection
from rest_framework.test import APIClient

from radios.models import (
    Radio, RadioMembership, RadioUser, Stream, Recording,
    TranscriptionSegment, Tag, ChunkSummary, DailySummary, Song,
)
from radios.api.fts import sanitize_fts_query


def _create_fts_tables():
    """Ensure FTS5 virtual tables exist in the test database."""
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS radios_transcription_fts
            USING fts5(
                segment_id UNINDEXED,
                text,
                text_english,
                tokenize='unicode61 remove_diacritics 2'
            )
        """)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS radios_summary_fts
            USING fts5(
                summary_id UNINDEXED,
                summary_type UNINDEXED,
                summary_text,
                tokenize='unicode61 remove_diacritics 2'
            )
        """)


class FTSSanitizationTest(TestCase):
    """Test that FTS5 query sanitization prevents injection."""

    def test_strips_special_characters(self):
        result = sanitize_fts_query('test" OR 1=1 --')
        # OR must be quoted (become "OR") so it's not a bare FTS5 boolean operator
        self.assertIn('"OR"', result)
        # No unquoted bare OR operator
        self.assertNotIn(' OR ', result)
        self.assertNotIn("--", result)

    def test_empty_query_returns_none(self):
        self.assertIsNone(sanitize_fts_query(""))
        self.assertIsNone(sanitize_fts_query("   "))
        self.assertIsNone(sanitize_fts_query("***"))

    def test_prefix_matching_appended(self):
        result = sanitize_fts_query("hello world")
        # Last token should have * for prefix matching
        self.assertTrue(result.endswith('"*'))

    def test_single_token(self):
        result = sanitize_fts_query("politica")
        self.assertEqual(result, '"politica"*')

    def test_diacritics_passed_through(self):
        result = sanitize_fts_query("café")
        self.assertIn("café", result)

    def test_fts5_operators_neutralized(self):
        result = sanitize_fts_query("NOT hello AND world OR test")
        # NOT/AND/OR are plain words, should be quoted
        self.assertIn('"NOT"', result)
        self.assertIn('"AND"', result)


@override_settings(REST_FRAMEWORK={
    'DEFAULT_THROTTLE_CLASSES': [],
    'DEFAULT_THROTTLE_RATES': {},
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
})
class RadioSearchAPITest(TestCase):

    @classmethod
    def setUpTestData(cls):
        Radio.objects.all().delete()
        cls.radio1 = Radio.objects.create(
            name="Radio Rock", city="Rome", country="IT",
            is_fm=True, is_web=True, languages="Italian",
            motto="Rock your world", frequencies="101.5 FM",
        )
        cls.radio2 = Radio.objects.create(
            name="Jazz FM", city="London", country="GB",
            is_fm=True, languages="English",
            motto="Smooth jazz", frequencies="99.9 FM",
        )

    def setUp(self):
        self.client = APIClient()

    def test_list_radios(self):
        resp = self.client.get("/api/v1/radios/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 2)

    def test_search_by_q(self):
        resp = self.client.get("/api/v1/radios/", {"q": "rock"})
        self.assertEqual(resp.data["count"], 1)
        self.assertEqual(resp.data["results"][0]["slug"], self.radio1.slug)

    def test_filter_by_country(self):
        resp = self.client.get("/api/v1/radios/", {"country": "IT"})
        self.assertEqual(resp.data["count"], 1)

    def test_filter_by_broadcast_type(self):
        resp = self.client.get("/api/v1/radios/", {"broadcast_type": "web"})
        self.assertEqual(resp.data["count"], 1)
        self.assertEqual(resp.data["results"][0]["name"], "Radio Rock")

    def test_filter_by_language(self):
        resp = self.client.get("/api/v1/radios/", {"language": "Italian"})
        self.assertEqual(resp.data["count"], 1)

    def test_detail_by_slug(self):
        resp = self.client.get(f"/api/v1/radios/{self.radio1.slug}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["name"], "Radio Rock")


@override_settings(REST_FRAMEWORK={
    'DEFAULT_THROTTLE_CLASSES': [],
    'DEFAULT_THROTTLE_RATES': {},
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
})
class VisibilityEnforcementTest(TestCase):
    """Test that anonymous users cannot see owner-only data."""

    @classmethod
    def setUpTestData(cls):
        cls.owner = RadioUser.objects.create_user(
            username="owner1", password="testpass123"
        )
        cls.other_user = RadioUser.objects.create_user(
            username="other1", password="testpass123"
        )
        cls.admin_user = RadioUser.objects.create_user(
            username="admin1", password="testpass123", is_staff=True
        )

        cls.radio = Radio.objects.create(name="Secret Radio", city="Berlin", country="DE")
        RadioMembership.objects.create(
            user=cls.owner, radio=cls.radio, role="owner", verified=True
        )

        # Stream with transcription visible to owner only
        cls.stream = Stream.objects.create(
            radio=cls.radio, name="Secret Stream", url="http://example.com/stream",
            transcription_owner_visible=True,
            transcription_public_visible=False,
            fingerprinting_owner_visible=True,
            fingerprinting_public_visible=False,
            summarization_owner_visible=True,
            summarization_public_visible=False,
        )

        now = timezone.now()
        cls.recording = Recording.objects.create(
            stream=cls.stream,
            start_time=now - datetime.timedelta(minutes=20),
            end_time=now,
            file="test.mp3",
        )
        cls.segment = TranscriptionSegment.objects.create(
            recording=cls.recording,
            segment_type="speech",
            start_offset=0, end_offset=60,
            text="This is a secret broadcast about politics",
            language="en",
        )

    def setUp(self):
        self.client = APIClient()
        _create_fts_tables()

    def test_anonymous_cannot_see_owner_only_transcripts(self):
        resp = self.client.get("/api/v1/search/transcripts/", {"q": "secret"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 0)

    def test_owner_can_see_own_transcripts(self):
        self.client.force_authenticate(user=self.owner)
        resp = self.client.get("/api/v1/search/transcripts/", {"q": "secret"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)

    def test_other_user_cannot_see_owner_only_transcripts(self):
        self.client.force_authenticate(user=self.other_user)
        resp = self.client.get("/api/v1/search/transcripts/", {"q": "secret"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 0)

    def test_admin_can_see_everything(self):
        self.client.force_authenticate(user=self.admin_user)
        resp = self.client.get("/api/v1/search/transcripts/", {"q": "secret"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)

    def test_anonymous_cannot_see_owner_only_songs(self):
        # Create a music segment linked to a Song row
        song = Song.objects.create(title="Secret Song", artist="Hidden Artist")
        TranscriptionSegment.objects.create(
            recording=self.recording,
            segment_type="music",
            start_offset=60, end_offset=120,
            song=song,
        )
        resp = self.client.get("/api/v1/search/songs/", {"q": "Secret Song"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 0)


@override_settings(REST_FRAMEWORK={
    'DEFAULT_THROTTLE_CLASSES': [],
    'DEFAULT_THROTTLE_RATES': {},
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
})
class TranscriptSearchAPITest(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.radio = Radio.objects.create(name="Public Radio", city="Milan", country="IT")
        cls.stream = Stream.objects.create(
            radio=cls.radio, name="Public Stream", url="http://example.com/s",
            transcription_public_visible=True,
        )
        now = timezone.now()
        cls.recording = Recording.objects.create(
            stream=cls.stream,
            start_time=now - datetime.timedelta(minutes=20),
            end_time=now,
            file="test.mp3",
        )
        cls.seg1 = TranscriptionSegment.objects.create(
            recording=cls.recording,
            segment_type="speech", start_offset=0, end_offset=30,
            text="Il sindaco ha parlato di politica economica",
            text_english="The mayor spoke about economic policy",
            language="it",
        )
        cls.seg2 = TranscriptionSegment.objects.create(
            recording=cls.recording,
            segment_type="speech", start_offset=30, end_offset=60,
            text="Weather forecast for tomorrow is sunny",
            language="en",
        )

    def setUp(self):
        self.client = APIClient()
        _create_fts_tables()

    def test_search_returns_matching_segments(self):
        resp = self.client.get("/api/v1/search/transcripts/", {"q": "politica"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)
        self.assertEqual(resp.data["results"][0]["id"], self.seg1.id)

    def test_search_english_translation(self):
        resp = self.client.get("/api/v1/search/transcripts/", {"q": "mayor"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)

    def test_prefix_matching(self):
        resp = self.client.get("/api/v1/search/transcripts/", {"q": "politic"})
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.data["count"], 1)

    def test_empty_query_returns_empty(self):
        resp = self.client.get("/api/v1/search/transcripts/", {"q": ""})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 0)

    def test_filter_by_radio(self):
        resp = self.client.get("/api/v1/search/transcripts/", {
            "q": "weather", "radio": self.radio.slug,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)

    def test_filter_by_language(self):
        resp = self.client.get("/api/v1/search/transcripts/", {
            "q": "politica", "language": "it",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 1)

    def test_snippet_included(self):
        resp = self.client.get("/api/v1/search/transcripts/", {"q": "politica"})
        self.assertIn("snippet", resp.data["results"][0])
        self.assertIn("politica", resp.data["results"][0]["snippet"])


@override_settings(REST_FRAMEWORK={
    'DEFAULT_THROTTLE_CLASSES': [],
    'DEFAULT_THROTTLE_RATES': {},
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
})
class SongSearchAPITest(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.radio = Radio.objects.create(name="Music Radio", city="Paris", country="FR")
        cls.stream = Stream.objects.create(
            radio=cls.radio, name="Music Stream", url="http://example.com/m",
            fingerprinting_public_visible=True,
        )
        now = timezone.now()
        cls.recording = Recording.objects.create(
            stream=cls.stream,
            start_time=now - datetime.timedelta(minutes=20),
            end_time=now,
            file="test.mp3",
        )
        cls.queen_song = Song.objects.create(title="Bohemian Rhapsody", artist="Queen")
        cls.lennon_song = Song.objects.create(title="Imagine", artist="John Lennon")
        cls.song1 = TranscriptionSegment.objects.create(
            recording=cls.recording,
            segment_type="music", start_offset=0, end_offset=180,
            song=cls.queen_song,
        )
        cls.song2 = TranscriptionSegment.objects.create(
            recording=cls.recording,
            segment_type="music", start_offset=180, end_offset=360,
            song=cls.lennon_song,
        )

    def setUp(self):
        self.client = APIClient()

    def test_search_by_artist(self):
        resp = self.client.get("/api/v1/search/songs/", {"artist": "queen"})
        self.assertEqual(resp.data["count"], 1)
        self.assertEqual(resp.data["results"][0]["song_title"], "Bohemian Rhapsody")

    def test_search_by_title(self):
        resp = self.client.get("/api/v1/search/songs/", {"title": "imagine"})
        self.assertEqual(resp.data["count"], 1)

    def test_search_by_q(self):
        resp = self.client.get("/api/v1/search/songs/", {"q": "queen"})
        self.assertEqual(resp.data["count"], 1)

    def test_grouped_results(self):
        # Add another play of the same song (reuse existing Song row)
        TranscriptionSegment.objects.create(
            recording=self.recording,
            segment_type="music", start_offset=360, end_offset=540,
            song=self.queen_song,
        )
        resp = self.client.get("/api/v1/search/songs/", {"q": "queen", "group": "true"})
        self.assertEqual(resp.data["count"], 1)
        self.assertEqual(resp.data["results"][0]["play_count"], 2)


@override_settings(REST_FRAMEWORK={
    'DEFAULT_THROTTLE_CLASSES': [],
    'DEFAULT_THROTTLE_RATES': {},
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
})
class TagSearchAPITest(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.radio = Radio.objects.create(name="Tag Radio", city="Oslo", country="NO")
        cls.stream = Stream.objects.create(
            radio=cls.radio, name="Tag Stream", url="http://example.com/t",
            summarization_public_visible=True,
        )
        now = timezone.now()
        cls.recording = Recording.objects.create(
            stream=cls.stream,
            start_time=now - datetime.timedelta(minutes=20),
            end_time=now,
            file="test.mp3",
        )
        cls.tag1 = Tag.objects.create(name="politics", slug="politics")
        cls.tag2 = Tag.objects.create(name="economy", slug="economy")
        cls.tag3 = Tag.objects.create(name="sports", slug="sports")

        cls.chunk = ChunkSummary.objects.create(
            recording=cls.recording,
            summary_text="Discussion about politics and economy",
        )
        cls.chunk.tags.add(cls.tag1, cls.tag2)

        cls.daily = DailySummary.objects.create(
            radio=cls.radio,
            date=now.date(),
            summary_text="A day of politics coverage",
            chunk_count=1,
        )
        cls.daily.tags.add(cls.tag1)

    def setUp(self):
        self.client = APIClient()

    def test_list_tags(self):
        resp = self.client.get("/api/v1/search/tags/")
        self.assertEqual(resp.status_code, 200)
        # politics appears in both chunk and daily = count 2
        names = [t["name"] for t in resp.data["results"]]
        self.assertIn("politics", names)
        self.assertIn("economy", names)

    def test_search_tags_by_q(self):
        resp = self.client.get("/api/v1/search/tags/", {"q": "pol"})
        self.assertEqual(resp.data["count"], 1)
        self.assertEqual(resp.data["results"][0]["name"], "politics")

    def test_tag_detail(self):
        resp = self.client.get(f"/api/v1/tags/{self.tag1.slug}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["tag"]["name"], "politics")
        self.assertGreaterEqual(len(resp.data["chunk_summaries"]), 1)
        self.assertGreaterEqual(len(resp.data["daily_summaries"]), 1)


@override_settings(REST_FRAMEWORK={
    'DEFAULT_THROTTLE_CLASSES': [],
    'DEFAULT_THROTTLE_RATES': {},
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
})
class SummarySearchAPITest(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.radio = Radio.objects.create(name="Summary Radio", city="Lisbon", country="PT")
        cls.stream = Stream.objects.create(
            radio=cls.radio, name="Summary Stream", url="http://example.com/s",
            summarization_public_visible=True,
        )
        now = timezone.now()
        cls.recording = Recording.objects.create(
            stream=cls.stream,
            start_time=now - datetime.timedelta(minutes=20),
            end_time=now,
            file="test.mp3",
        )
        cls.chunk = ChunkSummary.objects.create(
            recording=cls.recording,
            summary_text="The broadcast covered elections and healthcare reform",
        )
        cls.daily = DailySummary.objects.create(
            radio=cls.radio,
            date=now.date(),
            summary_text="An important day with elections coverage",
            chunk_count=1,
        )

    def setUp(self):
        self.client = APIClient()
        _create_fts_tables()

    def test_search_summaries(self):
        resp = self.client.get("/api/v1/search/summaries/", {"q": "elections"})
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.data["count"], 1)

    def test_filter_by_type_chunk(self):
        resp = self.client.get("/api/v1/search/summaries/", {"q": "elections", "type": "chunk"})
        self.assertEqual(resp.status_code, 200)
        for r in resp.data["results"]:
            self.assertEqual(r["summary_type"], "chunk")

    def test_empty_query_returns_empty(self):
        resp = self.client.get("/api/v1/search/summaries/", {"q": ""})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["count"], 0)


@override_settings(REST_FRAMEWORK={
    'DEFAULT_THROTTLE_CLASSES': [],
    'DEFAULT_THROTTLE_RATES': {},
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
})
class PaginationTest(TestCase):
    """
    Tests that the radio list is paginated.  Uses 30 radios so that the first
    page (PAGE_SIZE=25) is full and a second page exists, without needing to
    override PAGE_SIZE (DRF bakes page_size into the paginator class at import
    time, so @override_settings('PAGE_SIZE') has no effect on it).
    """

    @classmethod
    def setUpTestData(cls):
        Radio.objects.all().delete()
        for i in range(30):
            Radio.objects.create(
                name=f"Radio {i}", city="Testville", country="US",
            )

    def setUp(self):
        self.client = APIClient()

    def test_paginated_radio_list(self):
        resp = self.client.get("/api/v1/radios/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data["results"]), 25)
        self.assertEqual(resp.data["count"], 30)
        self.assertIsNotNone(resp.data["next"])

    def test_page_2(self):
        resp = self.client.get("/api/v1/radios/", {"page": 2})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data["results"]), 5)
