"""
End-to-end pipeline test: segmentation -> fingerprinting -> transcription -> summarization.

Environment variables
---------------------
  TEST_MP3               Path to the audio file to test (default: test_files/test_1.mp3)
  TEST_LABELS            Path to an Audacity label export (tab-separated .txt):
                             <start_sec>\t<end_sec>\t<type>
                         e.g.:
                             0.000000\t187.000000\tspeech
                             187.000000\t534.500000\tmusic
                         Used in test_segment_then_fingerprint for ground-truth comparison.
                         Used in test_transcribe_then_summarize to pick speech segments —
                         when present, ground-truth boundaries are used instead of the
                         segmenter output, giving more precise and reproducible results.
                         Default: radios/tests/test_files/test_1.txt
  ACOUSTID_API_KEY   AcoustID API key; fingerprinting is skipped when not set.
  OPENAI_API_KEY     Required when the DB TranscriptionSettings or SummarizationSettings
                     backend is set to "openai".
  ANTHROPIC_API_KEY  Required when the DB TranscriptionSettings or SummarizationSettings
                     backend is set to "anthropic".
  OLLAMA_API_KEY     Required only for ollama.com cloud; leave unset for a local Ollama instance.

  Transcription and summarization backends and model parameters are configured
  via Django admin (Transcription Settings / Summarization Settings).

Run with:
    python manage.py test radios.tests.test_pipeline

    # Full pipeline with API keys for cloud backends:
    TEST_MP3=/path/to/recording.mp3 TEST_LABELS=/path/to/labels.txt \\
        ACOUSTID_API_KEY=your_key \\
        OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \\
        python manage.py test radios.tests.test_pipeline
"""

import os
from pathlib import Path

import django.test
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
class PipelineTest(django.test.TestCase):
    """
    Full pipeline integration test.
    Uses TestCase (not SimpleTestCase) to allow access to TranscriptionSettings
    and SummarizationSettings database singletons.
    """

    _segments = None  # cached segmentation result shared across tests

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()
        if TEST_MP3.exists():
            from radios.analysis.segmenter import segment_audio
            print(f"\nRunning inaSpeechSegmenter on {TEST_MP3.name} — this may take several minutes...")
            cls._segments = segment_audio(str(TEST_MP3))

    def _load_and_print_transcription_config(self):
        """
        Load TranscriptionSettings from the database, print all parameters,
        and skip the test if a required API key is missing.
        Returns the settings object.
        """
        from radios.models import TranscriptionSettings
        cfg = TranscriptionSettings.get_settings()

        print("\nTranscription config:")
        print(f"  Backend:      {cfg.get_backend_display()}")
        if cfg.backend == "local":
            print(f"  Model size:   {cfg.get_local_model_size_display()}")
            print(f"  Device:       {cfg.get_local_device_display()}")
            print(f"  Compute type: {cfg.get_local_compute_type_display()}")
        elif cfg.backend == "openai":
            print(f"  Model:        {cfg.openai_model}")
            key_status = "set" if os.environ.get("OPENAI_API_KEY") else "NOT SET"
            print(f"  API key:      {key_status}")
            if not os.environ.get("OPENAI_API_KEY"):
                self.skipTest("OPENAI_API_KEY not set — skipping transcription with openai backend.")
        elif cfg.backend == "anthropic":
            print(f"  Model:        {cfg.anthropic_model}")
            key_status = "set" if os.environ.get("ANTHROPIC_API_KEY") else "NOT SET"
            print(f"  API key:      {key_status}")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self.skipTest("ANTHROPIC_API_KEY not set — skipping transcription with anthropic backend.")
        elif cfg.backend == "ollama":
            print(f"  Model:        {cfg.ollama_model}")
            print(f"  URL:          {cfg.ollama_base_url}")
            is_cloud = "localhost" not in cfg.ollama_base_url and "127.0.0.1" not in cfg.ollama_base_url
            key_present = bool(os.environ.get("OLLAMA_API_KEY"))
            key_status = "set" if key_present else ("NOT SET" if is_cloud else "not set (OK for local)")
            print(f"  API key:      {key_status}")
            if is_cloud and not key_present:
                self.skipTest("OLLAMA_API_KEY not set — skipping transcription with ollama backend (cloud URL configured).")
        return cfg

    def _load_and_print_summarization_config(self):
        """
        Load SummarizationSettings from the database, print all parameters,
        and skip the test if a required API key is missing.
        Returns the settings object.
        """
        from radios.models import SummarizationSettings
        cfg = SummarizationSettings.get_settings()

        print("\nSummarization config:")
        print(f"  Backend: {cfg.get_backend_display()}")
        if cfg.backend == "local_ollama":
            print(f"  Model:   {cfg.local_ollama_model}")
            print(f"  URL:     {cfg.local_ollama_url}")
        elif cfg.backend == "cloud_ollama":
            print(f"  Model:   {cfg.cloud_ollama_model}")
            print(f"  URL:     {cfg.cloud_ollama_url}")
            key_status = "set" if os.environ.get("OLLAMA_API_KEY") else "NOT SET"
            print(f"  API key: {key_status}")
            if not os.environ.get("OLLAMA_API_KEY"):
                self.skipTest("OLLAMA_API_KEY not set — skipping summarization with cloud_ollama backend.")
        elif cfg.backend == "openai":
            print(f"  Model:   {cfg.openai_model}")
            key_status = "set" if os.environ.get("OPENAI_API_KEY") else "NOT SET"
            print(f"  API key: {key_status}")
            if not os.environ.get("OPENAI_API_KEY"):
                self.skipTest("OPENAI_API_KEY not set — skipping summarization with openai backend.")
        elif cfg.backend == "anthropic":
            print(f"  Model:   {cfg.anthropic_model}")
            key_status = "set" if os.environ.get("ANTHROPIC_API_KEY") else "NOT SET"
            print(f"  API key: {key_status}")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self.skipTest("ANTHROPIC_API_KEY not set — skipping summarization with anthropic backend.")
        return cfg

    # -------------------------------------------------------------------------
    # Stage 1+2: Segmentation + fingerprinting (original test, unchanged)
    # -------------------------------------------------------------------------

    def test_segment_then_fingerprint(self):
        if not TEST_MP3.exists():
            self.skipTest(f"Test fixture not found: {TEST_MP3}")

        segments = self._segments
        self.assertGreater(len(segments), 0, "segment_audio() returned no segments")

        api_key        = os.environ.get("ACOUSTID_API_KEY", "")
        do_fingerprint = bool(api_key)

        print(f"\nFingerprinting config:")
        print(f"  ACOUSTID_API_KEY: {'set' if api_key else 'NOT SET (fingerprinting skipped)'}")

        if do_fingerprint:
            from radios.analysis.fingerprinter import fingerprint_segment

        print(f"\n=== Pipeline: {TEST_MP3.name} ===")
        header = f"{'#':>3}  {'type':<8}  {'start':>7}  {'end':>7}  {'duration':>8}  {'fingerprint'}"
        print(header)
        print("-" * 80)

        fp_results = {}

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

    # -------------------------------------------------------------------------
    # Stage 3+4: Transcription + summarization
    # -------------------------------------------------------------------------

    def test_transcribe_then_summarize(self):
        if not TEST_MP3.exists():
            self.skipTest(f"Test fixture not found: {TEST_MP3}")

        segments = self._segments
        self.assertGreater(len(segments), 0, "segment_audio() returned no segments")

        self._load_and_print_transcription_config()
        self._load_and_print_summarization_config()

        from radios.analysis.transcriber import transcribe_segment
        from radios.analysis.summarizer import summarize_texts

        # If ground-truth labels are available, prefer them over the segmenter
        # output — they give precise, manually verified speech boundaries.
        if TEST_LABELS.exists():
            gt = _load_ground_truth()
            speech_segments = [
                type("Seg", (), {"start": s["start"], "end": s["end"],
                                 "segment_type": s["type"]})()
                for s in gt if s["type"] in ("speech", "speech_over_music")
            ]
            segments_source = f"ground-truth labels ({TEST_LABELS.name})"
        else:
            speech_segments = [
                s for s in segments if s.segment_type in ("speech", "speech_over_music")
            ]
            segments_source = "segmenter output (no TEST_LABELS provided)"

        if not speech_segments:
            self.skipTest("No speech segments found — cannot test transcription/summarization.")

        # ── Transcription ─────────────────────────────────────────────────────
        print(f"\n=== Transcription — {len(speech_segments)} speech segment(s) from {segments_source} ===")

        transcribed_texts = []
        for i, s in enumerate(speech_segments):
            dur = s.end - s.start
            print(f"\n--- Segment {i+1}: {fmt_time(s.start)} → {fmt_time(s.end)} ({fmt_dur(dur)}) ---")
            try:
                result = transcribe_segment(str(TEST_MP3), s.start, s.end)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                continue

            if result and result.text.strip():
                transcribed_texts.append(result.text_english or result.text)
                print(f"  Language: {result.language}")
                print(f"  Text:\n{result.text}")
                if result.text_english:
                    print(f"  English:\n{result.text_english}")
            else:
                print("  (no result)")

        print(f"\nTranscribed: {len(transcribed_texts)}/{len(speech_segments)} segments")

        if not transcribed_texts:
            print("No transcribed text available — skipping summarization.")
            return

        # ── Summarization ─────────────────────────────────────────────────────
        print(f"\n=== Summarization ===")

        try:
            summary = summarize_texts(transcribed_texts)
        except Exception as exc:
            self.fail(f"summarize_texts() raised an exception: {exc}")

        if summary:
            print(f"\n  Summary:\n{summary.summary_text}")
            print(f"\n  Tags: {', '.join(summary.tags) if summary.tags else '(none)'}")
        else:
            print("\n  (no summary returned)")
