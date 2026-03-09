"""
Integration tests for LLM summarization and tag extraction.

The backend and model are read from the SummarizationSettings database record
(configurable via Django admin). API keys are supplied as environment variables.

Run with:
    python manage.py test radios.tests.test_summarization

    # Supply API keys for non-local backends:
    OPENAI_API_KEY=sk-...        python manage.py test radios.tests.test_summarization
    ANTHROPIC_API_KEY=sk-ant-... python manage.py test radios.tests.test_summarization
    OLLAMA_API_KEY=...           python manage.py test radios.tests.test_summarization  # cloud only

    # Supply a custom transcript file (plain text):
    SAMPLE_TRANSCRIPT=/path/to/transcript.txt \
        python manage.py test radios.tests.test_summarization
"""

import os
from pathlib import Path

import django.test
from django.test import tag

from radios.tests import print_test_db_location

# ---------------------------------------------------------------------------
# Sample transcripts (used when SAMPLE_TRANSCRIPT env var is not set)
# ---------------------------------------------------------------------------

_SAMPLE_CHUNKS = [
    (
        "Chunk 1 — Morning news",
        [
            "Good morning. Today's top story: the city council voted last night to approve "
            "the new public transport plan, which includes three new metro lines and expanded "
            "bus routes across the northern districts.",
            "The mayor confirmed the project will begin construction in spring, with an "
            "estimated completion date of 2028. Critics have raised concerns about noise "
            "levels near residential areas during the construction phase.",
        ],
    ),
    (
        "Chunk 2 — Sports",
        [
            "In football, Juventus defeated Inter Milan 2-1 in last night's derby. "
            "Goals from Vlahovic and Kostic secured the victory, putting Juventus back "
            "in first place in Serie A.",
            "The coach praised the team's defensive performance but acknowledged they "
            "need to work on converting chances in the first half.",
        ],
    ),
    (
        "Chunk 3 — Weather",
        [
            "The weather forecast for the coming week: expect cloudy skies with occasional "
            "showers through Wednesday. Temperatures will remain between 10 and 14 degrees. "
            "From Thursday onward, a high-pressure system moving in from the Atlantic will "
            "bring clearer skies and temperatures rising back toward seasonal averages of 17 degrees.",
        ],
    ),
]


def _load_transcript_file():
    """Load transcript from SAMPLE_TRANSCRIPT env var if set."""
    path = os.environ.get("SAMPLE_TRANSCRIPT")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip()


def _load_and_print_summarization_config(test_case):
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
            test_case.skipTest("OLLAMA_API_KEY not set — skipping cloud_ollama summarization test.")
    elif cfg.backend == "openai":
        print(f"  Model:   {cfg.openai_model}")
        key_status = "set" if os.environ.get("OPENAI_API_KEY") else "NOT SET"
        print(f"  API key: {key_status}")
        if not os.environ.get("OPENAI_API_KEY"):
            test_case.skipTest("OPENAI_API_KEY not set — skipping openai summarization test.")
    elif cfg.backend == "anthropic":
        print(f"  Model:   {cfg.anthropic_model}")
        key_status = "set" if os.environ.get("ANTHROPIC_API_KEY") else "NOT SET"
        print(f"  API key: {key_status}")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            test_case.skipTest("ANTHROPIC_API_KEY not set — skipping anthropic summarization test.")

    return cfg


def _print_result(result):
    """Print summary and tags in a consistent format."""
    if result is None:
        print("  (no result returned)")
        return
    print(f"  Summary:\n{result.summary_text}")
    print(f"\n  Tags: {', '.join(result.tags) if result.tags else '(none)'}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@tag("slow", "integration")
class SummarizationChunkTest(django.test.TestCase):
    """Test per-chunk summarization using sample radio transcripts."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def test_summarize_sample_chunks(self):
        _load_and_print_summarization_config(self)

        from radios.analysis.summarizer import summarize_texts

        custom = _load_transcript_file()
        if custom:
            chunks = [("Custom transcript", [custom])]
            print(f"  Input:   SAMPLE_TRANSCRIPT ({Path(os.environ['SAMPLE_TRANSCRIPT']).name})")
        else:
            chunks = _SAMPLE_CHUNKS
            print(f"  Input:   {len(chunks)} built-in sample chunks")

        print(f"\n=== Chunk Summarization ===")

        all_ok = True
        for label, texts in chunks:
            print(f"\n--- {label} ---")
            for i, t in enumerate(texts):
                preview = t[:80].replace("\n", " ")
                print(f"  Input [{i+1}]: {preview}{'...' if len(t) > 80 else ''}")

            try:
                result = summarize_texts(texts)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                all_ok = False
                continue

            _print_result(result)

        self.assertTrue(all_ok, "One or more summarize_texts() calls raised an exception")


@tag("slow", "integration")
class SummarizationDailyTest(django.test.TestCase):
    """Test daily aggregation by chaining chunk summaries into a daily summary."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def test_summarize_daily(self):
        _load_and_print_summarization_config(self)

        from radios.analysis.summarizer import summarize_texts, summarize_daily_texts

        print(f"\n=== Daily Summarization ===")

        # Step 1: generate chunk summaries from sample data
        chunk_summaries = []
        for label, texts in _SAMPLE_CHUNKS:
            try:
                result = summarize_texts(texts)
                if result:
                    chunk_summaries.append(result.summary_text)
                    print(f"\n  Chunk ({label}): {result.summary_text[:100]}...")
                else:
                    print(f"\n  Chunk ({label}): (no result)")
            except Exception as exc:
                self.fail(f"summarize_texts() raised during chunk phase: {exc}")

        if not chunk_summaries:
            self.skipTest("No chunk summaries generated — cannot test daily aggregation.")

        # Step 2: aggregate into a daily summary
        print(f"\n--- Daily aggregate from {len(chunk_summaries)} chunk(s) ---")
        try:
            daily = summarize_daily_texts(chunk_summaries)
        except Exception as exc:
            self.fail(f"summarize_daily_texts() raised: {exc}")

        _print_result(daily)
        self.assertIsNotNone(daily, "summarize_daily_texts() returned None")
        self.assertTrue(daily.summary_text, "Daily summary text is empty")


@tag("slow", "integration")
class SummarizationSanityTest(django.test.TestCase):
    """Quick sanity check: summarize a single short text and assert non-empty output."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print_test_db_location()

    def test_single_text_sanity(self):
        _load_and_print_summarization_config(self)

        from radios.analysis.summarizer import summarize_texts

        text = (
            "The Italian government today announced a new economic stimulus package "
            "worth 20 billion euros, targeting small businesses and green energy projects. "
            "The finance minister said the plan will create approximately 300,000 jobs "
            "over the next three years."
        )

        print(f"\n=== Summarization sanity check ===")
        print(f"  Input: {text[:100]}...")

        result = summarize_texts([text])
        _print_result(result)

        self.assertIsNotNone(result, "summarize_texts() returned None for a non-empty input")
        self.assertTrue(result.summary_text, "summary_text is empty")
        self.assertIsInstance(result.tags, list, "tags is not a list")
