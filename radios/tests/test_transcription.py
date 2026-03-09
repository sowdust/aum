"""
Integration tests for speech transcription.

The backend and model are read from the TranscriptionSettings database record
(configurable via Django admin). API keys are supplied as environment variables.

When no test MP3 is available, each test falls back to summarizing a hardcoded
example radio transcript so the summarization pipeline is exercised regardless.

Run with:
    python manage.py test radios.tests.test_transcription

    # Supply API keys for non-local backends:
    OPENAI_API_KEY=sk-...        python manage.py test radios.tests.test_transcription
    ANTHROPIC_API_KEY=sk-ant-... python manage.py test radios.tests.test_transcription
    OLLAMA_API_KEY=...           python manage.py test radios.tests.test_transcription  # cloud only

    # Use a different recording or label file:
    TEST_MP3=/path/to/recording.mp3 TEST_LABELS=/path/to/labels.txt \\
        python manage.py test radios.tests.test_transcription
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

# Hardcoded fallback transcript used when no test MP3 is present.
# Represents a realistic radio broadcast excerpt.
_FALLBACK_TRANSCRIPTS = [
    (
        "Morning news segment",
        "Buongiorno, sono le otto e questo è il giornale radio. "
        "Il governo ha approvato ieri sera un nuovo pacchetto di misure economiche "
        "del valore di venti miliardi di euro, destinato alle piccole e medie imprese "
        "e ai progetti di energia rinnovabile. "
        "Il ministro dell'economia ha dichiarato che il piano creerà circa trecentomila "
        "posti di lavoro nei prossimi tre anni. "
        "L'opposizione ha accolto la notizia con scetticismo, chiedendo maggiori dettagli "
        "sui meccanismi di finanziamento e sui criteri di accesso ai fondi.",
    ),
    (
        "Sports segment",
        "Nel calcio, la Juventus ha battuto l'Inter per due a uno nel derby di ieri sera. "
        "Le reti di Vlahovic e Kostic hanno deciso l'incontro, riportando i bianconeri "
        "in vetta alla classifica di Serie A a quattro punti dal Milan. "
        "L'allenatore ha elogiato la prestazione difensiva della squadra ma ha sottolineato "
        "la necessità di concretizzare meglio le occasioni create nel primo tempo.",
    ),
]


def _load_ground_truth():
    from radios.analysis.audacity_to_labels import parse_audacity_labels
    return parse_audacity_labels(str(TEST_LABELS))


def _load_and_print_transcription_config(test_case):
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
            test_case.skipTest("OPENAI_API_KEY not set — skipping openai transcription test.")
    elif cfg.backend == "anthropic":
        print(f"  Model:        {cfg.anthropic_model}")
        key_status = "set" if os.environ.get("ANTHROPIC_API_KEY") else "NOT SET"
        print(f"  API key:      {key_status}")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            test_case.skipTest("ANTHROPIC_API_KEY not set — skipping anthropic transcription test.")
    elif cfg.backend == "ollama":
        print(f"  Model:        {cfg.ollama_model}")
        print(f"  URL:          {cfg.ollama_base_url}")
        is_cloud = "localhost" not in cfg.ollama_base_url and "127.0.0.1" not in cfg.ollama_base_url
        key_present = bool(os.environ.get("OLLAMA_API_KEY"))
        key_status = "set" if key_present else ("NOT SET" if is_cloud else "not set (OK for local)")
        print(f"  API key:      {key_status}")
        if is_cloud and not key_present:
            test_case.skipTest("OLLAMA_API_KEY not set — skipping ollama transcription test (cloud URL configured).")

    return cfg


def _run_summarization_fallback():
    """
    Summarize hardcoded example transcripts and print results.
    Used when no test MP3 is available so the summarization pipeline
    is exercised regardless.
    """
    from radios.analysis.summarizer import summarize_texts

    print("\n  [No test MP3 found — running summarization fallback on hardcoded transcripts]")
    print(f"  Segments: {len(_FALLBACK_TRANSCRIPTS)}")

    all_ok = True
    for label, text in _FALLBACK_TRANSCRIPTS:
        print(f"\n  --- {label} ---")
        preview = text[:100].replace("\n", " ")
        print(f"  Input: {preview}...")
        try:
            result = summarize_texts([text])
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_ok = False
            continue

        if result:
            print(f"  Summary: {result.summary_text}")
            print(f"  Tags:    {', '.join(result.tags) if result.tags else '(none)'}")
        else:
            print("  (no result returned)")
            all_ok = False

    return all_ok


@tag("slow", "integration")
class TranscriptionTest(django.test.TestCase):
    """Transcribe speech segments from ground-truth labels."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def test_transcribe_ground_truth_speech_segments(self):
        if not TEST_MP3.exists():
            print(f"\n  Test MP3 not found: {TEST_MP3}")
            from radios.models import SummarizationSettings
            cfg = SummarizationSettings.get_settings()
            print(f"  Summarization backend: {cfg.get_backend_display()}")
            ok = _run_summarization_fallback()
            self.assertTrue(ok, "Summarization fallback raised errors — check logs above.")
            return

        if not TEST_LABELS.exists():
            self.skipTest(f"Ground-truth labels not found: {TEST_LABELS}")

        _load_and_print_transcription_config(self)
        print(f"  File:         {TEST_MP3.name}")
        print(f"  Labels:       {TEST_LABELS.name}")

        from radios.analysis.transcriber import transcribe_segment

        gt = _load_ground_truth()
        speech_segs = [s for s in gt if s["type"] in ("speech", "speech_over_music")]

        if not speech_segs:
            self.skipTest("No speech segments in ground-truth labels")

        print(f"\n=== Transcription of {len(speech_segs)} speech segment(s) ===")

        transcribed = 0
        raised = False
        for i, seg in enumerate(speech_segs):
            start, end, dur = seg["start"], seg["end"], seg["end"] - seg["start"]
            print(f"\n--- Segment {i+1}: {fmt_time(start)} → {fmt_time(end)} ({fmt_dur(dur)}) ---")
            try:
                result = transcribe_segment(str(TEST_MP3), start, end)
            except Exception as exc:
                raised = True
                print(f"  ERROR: {exc}")
                continue

            if result:
                transcribed += 1
                print(f"  Language: {result.language}")
                print(f"  Text:\n{result.text}")
                if result.text_english:
                    print(f"  English:\n{result.text_english}")
            else:
                print("  (no result)")

        print(f"\nTranscribed: {transcribed}/{len(speech_segs)} segments")
        self.assertFalse(raised, "transcribe_segment() raised an unexpected exception")


@tag("slow", "integration")
class TranscriptionSingleSegmentTest(django.test.TestCase):
    """Transcribe the first 60s of a recording as a quick sanity check."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def test_transcribe_first_60s(self):
        if not TEST_MP3.exists():
            print(f"\n  Test MP3 not found: {TEST_MP3}")
            from radios.models import SummarizationSettings
            cfg = SummarizationSettings.get_settings()
            print(f"  Summarization backend: {cfg.get_backend_display()}")
            ok = _run_summarization_fallback()
            self.assertTrue(ok, "Summarization fallback raised errors — check logs above.")
            return

        _load_and_print_transcription_config(self)

        import subprocess
        from radios.analysis.transcriber import transcribe_segment

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(TEST_MP3)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(probe.returncode, 0, f"ffprobe failed: {probe.stderr}")
        file_duration = float(probe.stdout.strip())
        end = min(60.0, file_duration)

        print(f"  File:     {TEST_MP3.name}")
        print(f"  Duration: {fmt_dur(file_duration)}")
        print(f"  Testing:  first {fmt_dur(end)}")

        print("\n=== Transcription result ===")
        result = transcribe_segment(str(TEST_MP3), 0.0, end)

        if result:
            print(f"  Language:   {result.language}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Text:\n{result.text}")
            if result.text_english:
                print(f"  English:\n{result.text_english}")
        else:
            print("  (no transcription returned)")
