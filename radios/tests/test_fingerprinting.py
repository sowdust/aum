"""
Integration tests for music fingerprinting.

Run with:
    python manage.py test radios.tests.test_fingerprinting
    ACOUSTID_API_KEY=your_key python manage.py test radios.tests.test_fingerprinting

    # Use a different recording or label file:
    TEST_MP3=/path/to/recording.mp3 TEST_LABELS=/path/to/labels.txt \\
        ACOUSTID_API_KEY=your_key python manage.py test radios.tests.test_fingerprinting

    # Fingerprint a single audio file (no labels needed):
    TEST_MP3=/path/to/song.mp3 \\
        ACOUSTID_API_KEY=your_key python manage.py test \\
        radios.tests.test_fingerprinting.FingerprintSingleFileTest
"""

import os
from pathlib import Path

import django.test
from django.conf import settings
from django.test import tag

from radios.tests import fmt_dur, fmt_time, print_test_db_location

_DEFAULT_MP3    = Path(__file__).parent / "test_files" / "test_1.mp3"
_DEFAULT_LABELS = Path(__file__).parent / "test_files" / "test_1.txt"

TEST_MP3    = Path(os.environ.get("TEST_MP3",    str(_DEFAULT_MP3)))
TEST_LABELS = Path(os.environ.get("TEST_LABELS", str(_DEFAULT_LABELS)))


def _load_ground_truth():
    from radios.analysis.audacity_to_labels import parse_audacity_labels
    return parse_audacity_labels(str(TEST_LABELS))


@tag("slow", "integration")
class FingerprintingTest(django.test.SimpleTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def _skip_if_no_file(self):
        if not TEST_MP3.exists():
            self.skipTest(f"Test fixture not found: {TEST_MP3}")

    def _skip_if_no_api_key(self):
        if not getattr(settings, "ACOUSTID_API_KEY", ""):
            self.skipTest(
                "ACOUSTID_API_KEY is not set — skipping fingerprinting test. "
                "Set ACOUSTID_API_KEY=your_key to run."
            )

    def test_fingerprint_ground_truth_music_segments(self):
        self._skip_if_no_file()
        self._skip_if_no_api_key()
        if not TEST_LABELS.exists():
            self.skipTest(f"Ground-truth labels not found: {TEST_LABELS}")

        from radios.analysis.fingerprinter import fingerprint_segment

        api_key = settings.ACOUSTID_API_KEY
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
    """Fingerprint an entire audio file as a single segment.

    Usage:
        TEST_MP3=/path/to/song.mp3 ACOUSTID_API_KEY=your_key \\
            python manage.py test radios.tests.test_fingerprinting.FingerprintSingleFileTest
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def _skip_if_no_file(self):
        if not TEST_MP3.exists():
            self.skipTest(f"Test fixture not found: {TEST_MP3}")

    def _skip_if_no_api_key(self):
        if not getattr(settings, "ACOUSTID_API_KEY", ""):
            self.skipTest(
                "ACOUSTID_API_KEY is not set — skipping fingerprinting test. "
                "Set ACOUSTID_API_KEY=your_key to run."
            )

    def test_fingerprint_whole_file(self):
        self._skip_if_no_file()
        self._skip_if_no_api_key()

        import subprocess
        from radios.analysis.fingerprinter import fingerprint_segment, _BOUNDARY_TRIM

        api_key = settings.ACOUSTID_API_KEY

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

        # fingerprint_segment trims _BOUNDARY_TRIM seconds from each edge
        # (designed for segment boundaries within a longer recording).  For a
        # standalone file we pre-expand the window so the trim lands at 0 and
        # file_duration, fingerprinting the whole content.
        result = fingerprint_segment(
            str(TEST_MP3),
            start=-_BOUNDARY_TRIM,
            end=file_duration + _BOUNDARY_TRIM,
        )

        if result:
            print(f"Artist:   {result.artist}")
            print(f"Title:    {result.title}")
            print(f"Score:    {result.score:.2f}")
            print(f"MBID:     {result.mbid}")
        else:
            print("Result:   No match (song may not be in the AcoustID database)")

        # Not asserting a match — whether a song is in AcoustID is outside our
        # control.  The test passes as long as no exception was raised.
