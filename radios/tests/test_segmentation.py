"""
Integration tests for audio segmentation.

Run with:
    python manage.py test radios.tests.test_segmentation
    python manage.py test radios --tag=integration

    # Use a different recording or label file:
    TEST_MP3=/path/to/recording.mp3 python manage.py test radios.tests.test_segmentation
    TEST_LABELS=/path/to/labels.txt python manage.py test radios.tests.test_segmentation

    # Also split and save the segments to media/segments/:
    SAVE_SEGMENTS=1 python manage.py test radios.tests.test_segmentation

Excluded from fast CI runs via:
    python manage.py test radios --exclude-tag=slow
"""

import os
from pathlib import Path

import django.test
from django.conf import settings
from django.test import tag

from radios.tests import fmt_dur, fmt_time

_DEFAULT_MP3    = Path(__file__).parent / "test_files" / "test_1.mp3"
_DEFAULT_LABELS = Path(__file__).parent / "test_files" / "test_1.txt"

TEST_MP3    = Path(os.environ.get("TEST_MP3",    str(_DEFAULT_MP3)))
TEST_LABELS = Path(os.environ.get("TEST_LABELS", str(_DEFAULT_LABELS)))
# Set SAVE_SEGMENTS=1 to write each segment as a separate MP3 under media/segments/
SAVE_SEGMENTS = bool(os.environ.get("SAVE_SEGMENTS", ""))


def _load_ground_truth():
    from radios.analysis.audacity_to_labels import parse_audacity_labels
    return parse_audacity_labels(str(TEST_LABELS))


def _segments_save_dir() -> str:
    """Return the directory where segments should be saved."""
    return str(Path(settings.MEDIA_ROOT) / "segments")


@tag("slow", "integration")
class SegmentationTest(django.test.SimpleTestCase):

    _segments = None  # cached across test methods

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if TEST_MP3.exists():
            from radios.analysis.segmenter import segment_audio
            save_dir = _segments_save_dir() if SAVE_SEGMENTS else None
            if save_dir:
                print(f"\nRunning inaSpeechSegmenter on {TEST_MP3.name} — saving segments to {save_dir} ...")
            else:
                print(f"\nRunning inaSpeechSegmenter on {TEST_MP3.name} — this may take several minutes...")
            cls._segments = segment_audio(str(TEST_MP3), save_dir=save_dir)

    def _skip_if_no_file(self):
        if not TEST_MP3.exists():
            self.skipTest(f"Test fixture not found: {TEST_MP3}")

    def test_segments_produced(self):
        self._skip_if_no_file()
        segments = self._segments
        self.assertGreater(len(segments), 0, "segment_audio() returned no segments")

        saved_count = sum(1 for s in segments if s.file_path)

        print(f"\n=== Segmentation: {TEST_MP3.name} ===")
        has_paths = saved_count > 0
        if has_paths:
            header = f"{'#':>3}  {'type':<20}  {'start':>7}  {'end':>7}  {'duration':>8}  {'saved file'}"
        else:
            header = f"{'#':>3}  {'type':<20}  {'start':>7}  {'end':>7}  {'duration':>8}"
        print(header)
        print("-" * len(header))

        for i, s in enumerate(segments):
            dur = s.end - s.start
            row = f"{i:>3}  {s.segment_type:<20}  {fmt_time(s.start):>7}  {fmt_time(s.end):>7}  {fmt_dur(dur):>8}"
            if has_paths:
                file_label = os.path.basename(s.file_path) if s.file_path else "(not saved)"
                row += f"  {file_label}"
            print(row)

        totals = {}
        for s in segments:
            totals[s.segment_type] = totals.get(s.segment_type, 0.0) + (s.end - s.start)
        parts = " | ".join(
            f"{t}: {fmt_dur(totals[t])}"
            for t in ("speech", "music", "noise", "noEnergy")
            if t in totals
        )
        print(f"Summary: {len(segments)} segments — {parts}")
        if has_paths:
            print(f"Saved:   {saved_count}/{len(segments)} segments written to {_segments_save_dir()}/{TEST_MP3.stem}/")

        if SAVE_SEGMENTS:
            self.assertEqual(
                saved_count, len(segments),
                f"Only {saved_count}/{len(segments)} segments were saved to disk",
            )

    def test_compare_ground_truth(self):
        self._skip_if_no_file()
        if not TEST_LABELS.exists():
            self.skipTest(f"Ground-truth labels not found: {TEST_LABELS}")

        gt = _load_ground_truth()
        segments = self._segments

        print(f"\n=== Ground Truth vs Predicted ===")
        header = f"{'GT type':<10}  {'GT start':>7}  {'GT end':>7}  {'GT dur':>7}  {'Best match':<18}  {'Overlap':>7}"
        print(header)
        print("-" * len(header))

        correct = 0
        for gt_seg in gt:
            gt_start = gt_seg["start"]
            gt_end   = gt_seg["end"]
            gt_type  = gt_seg["type"]
            gt_dur   = gt_end - gt_start

            best_overlap = 0.0
            best_seg = None
            for s in segments:
                overlap = max(0.0, min(gt_end, s.end) - max(gt_start, s.start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_seg = s

            overlap_pct = (best_overlap / gt_dur * 100) if gt_dur > 0 else 0.0
            match_type  = best_seg.segment_type if best_seg else "—"
            match_mark  = "*" if match_type == gt_type else " "
            print(
                f"{gt_type:<10}  {fmt_time(gt_start):>7}  {fmt_time(gt_end):>7}  {fmt_dur(gt_dur):>7}  "
                f"{match_type:<18}  {overlap_pct:>6.0f}%  {match_mark}"
            )
            if match_type == gt_type:
                correct += 1

        accuracy = correct / len(gt) if gt else 0.0
        print(f"\nCorrect type: {correct}/{len(gt)} ({accuracy:.0%})")
        self.assertGreaterEqual(
            accuracy, 0.70,
            f"Less than 70% of ground-truth segments matched the correct type ({accuracy:.0%})"
        )
