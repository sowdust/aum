"""
End-to-end pipeline test: segmentation -> fingerprinting.

Run with:
    python manage.py test radios.tests.test_pipeline
    ACOUSTID_API_KEY=your_key python manage.py test radios.tests.test_pipeline

    # Use a different recording or label file:
    TEST_MP3=/path/to/recording.mp3 TEST_LABELS=/path/to/labels.txt \\
        ACOUSTID_API_KEY=your_key python manage.py test radios.tests.test_pipeline
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


def _load_ground_truth():
    from radios.analysis.audacity_to_labels import parse_audacity_labels
    return parse_audacity_labels(str(TEST_LABELS))


@tag("slow", "integration")
class PipelineTest(django.test.SimpleTestCase):

    _segments = None  # cached for the class

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if TEST_MP3.exists():
            from radios.analysis.segmenter import segment_audio
            print(f"\nRunning inaSpeechSegmenter on {TEST_MP3.name} — this may take several minutes...")
            cls._segments = segment_audio(str(TEST_MP3))

    def test_segment_then_fingerprint(self):
        if not TEST_MP3.exists():
            self.skipTest(f"Test fixture not found: {TEST_MP3}")

        segments = self._segments
        self.assertGreater(len(segments), 0, "segment_audio() returned no segments")

        api_key        = getattr(settings, "ACOUSTID_API_KEY", "")
        do_fingerprint = bool(api_key)

        if do_fingerprint:
            from radios.analysis.fingerprinter import fingerprint_segment

        # ── Segmentation + fingerprinting output ──────────────────────────────
        print(f"\n=== Pipeline: {TEST_MP3.name} ===")
        header = f"{'#':>3}  {'type':<8}  {'start':>7}  {'end':>7}  {'duration':>8}  {'fingerprint'}"
        print(header)
        print("-" * 80)

        fp_results = {}  # index -> result for music segments

        for i, s in enumerate(segments):
            dur      = s.end - s.start
            fp_label = "—"

            if s.segment_type == "music" and do_fingerprint:
                try:
                    result = fingerprint_segment(str(TEST_MP3), s.start, s.end, api_key)
                    fp_results[i] = result
                    if result:
                        fp_label = f"{result.artist} - {result.title} (score: {result.score:.2f})"
                    else:
                        fp_label = "No match"
                except Exception as exc:
                    fp_results[i] = None
                    fp_label = f"ERROR: {exc}"
            elif s.segment_type == "music":
                fp_label = "(no API key)"

            print(
                f"{i:>3}  {s.segment_type:<8}  {fmt_time(s.start):>7}  {fmt_time(s.end):>7}  "
                f"{fmt_dur(dur):>8}  {fp_label}"
            )

        # ── Segmentation totals ───────────────────────────────────────────────
        totals = {}
        for s in segments:
            totals[s.segment_type] = totals.get(s.segment_type, 0.0) + (s.end - s.start)
        parts = " | ".join(
            f"{t}: {fmt_dur(totals[t])}"
            for t in ("speech", "music", "noise", "noEnergy")
            if t in totals
        )
        print(f"\nSegments: {len(segments)} — {parts}")

        if do_fingerprint:
            matched = sum(1 for r in fp_results.values() if r)
            print(f"Fingerprinted: {matched}/{len(fp_results)} music segments identified")

        # ── Ground-truth comparison ───────────────────────────────────────────
        if not TEST_LABELS.exists():
            print(f"\n(No ground-truth labels found at {TEST_LABELS} — skipping comparison)")
            return

        gt = _load_ground_truth()

        print(f"\n=== Segmentation vs Ground Truth ===")
        header = f"{'GT type':<10}  {'GT start':>7}  {'GT end':>7}  {'GT dur':>7}  {'Best match':<18}  {'Overlap':>7}"
        print(header)
        print("-" * len(header))

        correct    = 0
        type_stats = {}

        for gt_seg in gt:
            gt_start = gt_seg["start"]
            gt_end   = gt_seg["end"]
            gt_type  = gt_seg["type"]
            gt_dur   = gt_end - gt_start

            best_overlap = 0.0
            best_seg     = None
            for s in segments:
                overlap = max(0.0, min(gt_end, s.end) - max(gt_start, s.start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_seg     = s

            overlap_pct = (best_overlap / gt_dur * 100) if gt_dur > 0 else 0.0
            match_type  = best_seg.segment_type if best_seg else "—"
            is_correct  = match_type == gt_type
            match_mark  = "*" if is_correct else " "

            print(
                f"{gt_type:<10}  {fmt_time(gt_start):>7}  {fmt_time(gt_end):>7}  {fmt_dur(gt_dur):>7}  "
                f"{match_type:<18}  {overlap_pct:>6.0f}%  {match_mark}"
            )

            if is_correct:
                correct += 1
            stats = type_stats.setdefault(gt_type, {"total": 0, "correct": 0})
            stats["total"]   += 1
            stats["correct"] += int(is_correct)

        accuracy = correct / len(gt) if gt else 0.0
        print(f"\nSegmentation accuracy: {correct}/{len(gt)} ({accuracy:.0%}) correct vs ground truth")

        print("\nPer-type accuracy:")
        for t, stats in sorted(type_stats.items()):
            pct = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
            print(f"  {t:<12}  {stats['correct']}/{stats['total']} ({pct:.0f}%)")
