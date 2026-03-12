"""
Integration tests for music fingerprinting.

Run with:
    python manage.py test radios.tests.test_fingerprinting

    # Use a different recording or label file:
    TEST_MP3=/path/to/recording.mp3 TEST_LABELS=/path/to/labels.txt \
        python manage.py test radios.tests.test_fingerprinting

    # Fingerprint a single audio file (no labels needed):
    TEST_MP3=/path/to/song.mp3 \
        python manage.py test \
        radios.tests.test_fingerprinting.FingerprintSingleFileTest
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import django.test
from django.test import tag, TestCase
from django.utils import timezone

from radios.tests import fmt_dur, fmt_time, print_test_db_location

_DEFAULT_MP3    = Path(__file__).parent / "test_files" / "test_1.mp3"
_DEFAULT_LABELS = Path(__file__).parent / "test_files" / "test_1.txt"

TEST_MP3    = Path(os.environ.get("TEST_MP3",    str(_DEFAULT_MP3)))
TEST_LABELS = Path(os.environ.get("TEST_LABELS", str(_DEFAULT_LABELS)))


def _load_ground_truth():
    from radios.analysis.audacity_to_labels import parse_audacity_labels
    return parse_audacity_labels(str(TEST_LABELS))


def _make_shazam_response(title, artist, key, genre=None, album=None, year=None, cover=None):
    """Build a mock Shazam response dict."""
    track = {
        "title": title,
        "subtitle": artist,
        "key": key,
    }
    if genre:
        track["genres"] = {"primary": genre}
    sections = []
    if album or year:
        metadata = []
        if album:
            metadata.append({"title": "Album", "text": album})
        if year:
            metadata.append({"title": "Released", "text": str(year)})
        sections.append({"type": "SONG", "metadata": metadata})
    if sections:
        track["sections"] = sections
    if cover:
        track["images"] = {"coverart": cover}
    return {"track": track}


@tag("slow", "integration")
class FingerprintingTest(django.test.SimpleTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def _skip_if_no_file(self):
        if not TEST_MP3.exists():
            self.skipTest(f"Test fixture not found: {TEST_MP3}")

    def test_fingerprint_ground_truth_music_segments(self):
        self._skip_if_no_file()
        if not TEST_LABELS.exists():
            self.skipTest(f"Ground-truth labels not found: {TEST_LABELS}")

        from radios.analysis.fingerprinter import fingerprint_segment

        gt = _load_ground_truth()
        music_segs = [s for s in gt if s["type"] == "music"]

        if not music_segs:
            self.skipTest("No music segments in ground-truth labels")

        print(f"\n=== Fingerprinting music segments (ground truth) ===")
        header = f"{'#':>3}  {'start':>7}  {'end':>7}  {'duration':>8}  {'result'}"
        print(header)
        print("-" * 70)

        matched = 0
        raised = False
        for i, seg in enumerate(music_segs):
            start = seg["start"]
            end   = seg["end"]
            dur   = end - start
            try:
                result = fingerprint_segment(str(TEST_MP3), start, end)
            except Exception as exc:
                raised = True
                print(f"{i:>3}  {fmt_time(start):>7}  {fmt_time(end):>7}  {fmt_dur(dur):>8}  ERROR: {exc}")
                continue

            if result:
                matched += 1
                label = f"{result.artist} - {result.title} (score: {result.score:.2f})"
            else:
                label = "No match"
            print(f"{i:>3}  {fmt_time(start):>7}  {fmt_time(end):>7}  {fmt_dur(dur):>8}  {label}")

        print(f"\nMatched: {matched}/{len(music_segs)} segments")
        self.assertFalse(raised, "fingerprint_segment() raised an unexpected exception")


@tag("slow", "integration")
class FingerprintSingleFileTest(django.test.SimpleTestCase):
    """Fingerprint an entire audio file as a single segment."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def _skip_if_no_file(self):
        if not TEST_MP3.exists():
            self.skipTest(f"Test fixture not found: {TEST_MP3}")

    def test_fingerprint_whole_file(self):
        self._skip_if_no_file()

        import subprocess
        from radios.analysis.fingerprinter import fingerprint_segment, _BOUNDARY_TRIM

        # Get file duration via ffprobe
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(TEST_MP3),
            ],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(probe.returncode, 0, f"ffprobe failed: {probe.stderr}")
        file_duration = float(probe.stdout.strip())

        print(f"\n=== Fingerprinting single file ===")
        print(f"File:     {TEST_MP3}")
        print(f"Duration: {fmt_dur(file_duration)}")

        result = fingerprint_segment(
            str(TEST_MP3),
            start=-_BOUNDARY_TRIM,
            end=file_duration + _BOUNDARY_TRIM,
        )

        if result:
            print(f"Artist:   {result.artist}")
            print(f"Title:    {result.title}")
            print(f"Score:    {result.score:.2f}")
            print(f"Key:      {result.shazam_key}")
        else:
            print("Result:   No match (song may not be in the Shazam database)")


class SlidingWindowMockTest(django.test.SimpleTestCase):
    """Test sliding window logic with mock Shazam responses."""

    @patch("radios.analysis.fingerprinter._extract_and_recognize")
    def test_multiple_songs_detected(self, mock_recognize):
        from radios.analysis.fingerprinter import fingerprint_segment_sliding

        responses = [
            # First call at pos ~10 -> song A
            MagicMock(
                title="Song A", artist="Artist A", score=1.0,
                shazam_key="key_a", genres=["rock"], album_name="Album A",
                release_year=2020, album_cover_url="http://example.com/a.jpg",
                estimated_start=0.0, estimated_end=0.0,
            ),
            # Second call (after skipping 240s) -> song B
            MagicMock(
                title="Song B", artist="Artist B", score=1.0,
                shazam_key="key_b", genres=["pop"], album_name="Album B",
                release_year=2021, album_cover_url="http://example.com/b.jpg",
                estimated_start=0.0, estimated_end=0.0,
            ),
        ]
        mock_recognize.side_effect = responses

        # Segment of 600s (10 min) with 10s boundary trim on each side
        results = fingerprint_segment_sliding("/fake/path.mp3", 0.0, 620.0)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Song A")
        self.assertEqual(results[0].shazam_key, "key_a")
        self.assertEqual(results[1].title, "Song B")
        self.assertEqual(results[1].shazam_key, "key_b")

    @patch("radios.analysis.fingerprinter._extract_and_recognize")
    def test_duplicate_shazam_key_skipped(self, mock_recognize):
        from radios.analysis.fingerprinter import fingerprint_segment_sliding

        same_song = MagicMock(
            title="Same Song", artist="Same Artist", score=1.0,
            shazam_key="same_key", genres=[], album_name="",
            release_year=None, album_cover_url="",
            estimated_start=0.0, estimated_end=0.0,
        )
        mock_recognize.return_value = same_song

        results = fingerprint_segment_sliding("/fake/path.mp3", 0.0, 620.0)

        # Should only appear once despite multiple calls
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shazam_key, "same_key")

    @patch("radios.analysis.fingerprinter._extract_and_recognize")
    def test_no_matches(self, mock_recognize):
        from radios.analysis.fingerprinter import fingerprint_segment_sliding

        mock_recognize.return_value = None

        results = fingerprint_segment_sliding("/fake/path.mp3", 0.0, 620.0)
        self.assertEqual(len(results), 0)

    def test_segment_too_short(self):
        from radios.analysis.fingerprinter import fingerprint_segment_sliding

        results = fingerprint_segment_sliding("/fake/path.mp3", 0.0, 40.0)
        self.assertEqual(len(results), 0)


class FingerprintResultParsingTest(django.test.SimpleTestCase):
    """Test Shazam response parsing in _recognize()."""

    def test_parse_full_response(self):
        import asyncio
        from radios.analysis.fingerprinter import _recognize

        response = _make_shazam_response(
            title="Bohemian Rhapsody",
            artist="Queen",
            key="12345",
            genre="Rock",
            album="A Night at the Opera",
            year=1975,
            cover="http://example.com/cover.jpg",
        )

        mock_shazam_cls = MagicMock()
        instance = MagicMock()
        instance.recognize = AsyncMock(return_value=response)
        mock_shazam_cls.return_value = instance

        result = asyncio.run(_recognize(mock_shazam_cls, "/fake/audio.wav"))

        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Bohemian Rhapsody")
        self.assertEqual(result.artist, "Queen")
        self.assertEqual(result.shazam_key, "12345")
        self.assertEqual(result.genres, ["Rock"])
        self.assertEqual(result.album_name, "A Night at the Opera")
        self.assertEqual(result.release_year, 1975)
        self.assertEqual(result.album_cover_url, "http://example.com/cover.jpg")

    def test_parse_minimal_response(self):
        import asyncio
        from radios.analysis.fingerprinter import _recognize

        response = {"track": {"title": "Unknown", "subtitle": "", "key": "99"}}

        mock_shazam_cls = MagicMock()
        instance = MagicMock()
        instance.recognize = AsyncMock(return_value=response)
        mock_shazam_cls.return_value = instance

        result = asyncio.run(_recognize(mock_shazam_cls, "/fake/audio.wav"))

        self.assertIsNotNone(result)
        self.assertEqual(result.title, "Unknown")
        self.assertEqual(result.genres, [])
        self.assertEqual(result.album_name, "")
        self.assertIsNone(result.release_year)

    def test_no_track_returns_none(self):
        import asyncio
        from radios.analysis.fingerprinter import _recognize

        response = {}

        mock_shazam_cls = MagicMock()
        instance = MagicMock()
        instance.recognize = AsyncMock(return_value=response)
        mock_shazam_cls.return_value = instance

        result = asyncio.run(_recognize(mock_shazam_cls, "/fake/audio.wav"))
        self.assertIsNone(result)


class SongModelTest(TestCase):
    """Test Song.get_or_create_from_fingerprint with new fields."""

    def test_creates_song_with_metadata(self):
        from radios.models import Song, Artist, Genre
        from radios.analysis.fingerprinter import FingerprintResult

        result = FingerprintResult(
            title="Test Song",
            artist="Test Artist",
            score=1.0,
            shazam_key="shazam_123",
            genres=["Rock", "Alternative"],
            album_name="Test Album",
            release_year=2023,
            album_cover_url="http://example.com/cover.jpg",
            estimated_start=10.0,
            estimated_end=250.0,
        )

        song = Song.get_or_create_from_fingerprint(result)

        self.assertEqual(song.title, "Test Song")
        self.assertEqual(song.artist, "Test Artist")
        self.assertEqual(song.shazam_key, "shazam_123")
        self.assertEqual(song.album_name, "Test Album")
        self.assertEqual(song.release_year, 2023)
        self.assertEqual(song.album_cover_url, "http://example.com/cover.jpg")
        self.assertIsNotNone(song.artist_ref)
        self.assertEqual(song.artist_ref.name, "Test Artist")
        self.assertEqual(song.genres.count(), 2)

    def test_deduplicates_by_shazam_key(self):
        from radios.models import Song
        from radios.analysis.fingerprinter import FingerprintResult

        result = FingerprintResult(
            title="Song", artist="Artist", score=1.0,
            shazam_key="dedup_key", genres=[], album_name="",
            release_year=None, album_cover_url="",
            estimated_start=0, estimated_end=0,
        )

        song1 = Song.get_or_create_from_fingerprint(result)
        song2 = Song.get_or_create_from_fingerprint(result)
        self.assertEqual(song1.pk, song2.pk)
        self.assertEqual(Song.objects.filter(shazam_key="dedup_key").count(), 1)

    def test_creates_song_occurrence(self):
        import datetime
        from radios.models import Song, SongOccurrence, Radio, Stream, Recording, TranscriptionSegment
        from radios.analysis.fingerprinter import FingerprintResult

        radio = Radio.objects.create(name="Test Radio", city="Test")
        stream = Stream.objects.create(radio=radio, name="Stream", url="http://example.com")
        now = timezone.now()
        recording = Recording.objects.create(
            stream=stream,
            start_time=now - datetime.timedelta(minutes=20),
            end_time=now,
            file="test.mp3",
        )
        seg = TranscriptionSegment.objects.create(
            recording=recording, segment_type="music",
            start_offset=0, end_offset=300,
        )

        result = FingerprintResult(
            title="Occurrence Song", artist="Artist", score=0.9,
            shazam_key="occ_key", genres=["Pop"], album_name="Album",
            release_year=2024, album_cover_url="",
            estimated_start=10.0, estimated_end=250.0,
        )

        song = Song.get_or_create_from_fingerprint(result)
        occ = SongOccurrence.objects.create(
            segment=seg, song=song,
            start_offset=result.estimated_start,
            end_offset=result.estimated_end,
            confidence=result.score,
        )

        self.assertEqual(seg.song_occurrences.count(), 1)
        self.assertEqual(occ.song.title, "Occurrence Song")
        self.assertAlmostEqual(occ.start_offset, 10.0)
        self.assertAlmostEqual(occ.end_offset, 250.0)
