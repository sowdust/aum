"""
LLM-based summarization and tag extraction for radio recordings.

Supports four backends (configured via the admin Summarization Settings page):
- "local_ollama"  — Ollama running locally (no API key needed; requires Ollama running)
- "cloud_ollama"  — ollama.com cloud API (OLLAMA_API_KEY environment variable required)
- "openai"        — OpenAI Chat API (OPENAI_API_KEY environment variable required)
- "anthropic"     — Claude (ANTHROPIC_API_KEY environment variable required)

All backend parameters (model, URL, API keys) and the prompt templates are
stored in the SummarizationSettings database model and editable in the admin.

Usage
-----
    from radios.analysis.summarizer import summarize_texts, summarize_daily_texts

    result = summarize_texts(["text1", "text2", ...])
    if result:
        print(result.summary_text)
        print(result.tags)  # ["news", "sports", "milan", ...]

    result = summarize_daily_texts(["chunk summary 1", "chunk summary 2", ...])
"""

import dataclasses
import json
import logging
from typing import Optional

from radios.analysis._llm_backends import call_llm

logger = logging.getLogger("broadcast_analysis")

# Trim oversized inputs to avoid API token limits.
_MAX_INPUT_CHARS = 40_000

# Maximum number of tags to keep per summary.
_MAX_TAGS = 15


@dataclasses.dataclass
class SummaryResult:
    summary_text: str   # 2-4 sentence summary of content
    tags: list          # Normalized lowercase keyword tags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_texts(
    texts: list,
    language_hint: str = "",
) -> Optional[SummaryResult]:
    """
    Summarize a list of speech transcript texts from a single recording chunk.

    texts         — list of transcript strings from speech segments
    language_hint — ISO 639-1 code(s) if known, e.g. "it,en"

    Returns SummaryResult or None if texts are empty or the backend fails.
    """
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return None

    combined = "\n\n".join(texts)
    if len(combined) > _MAX_INPUT_CHARS:
        combined = combined[:_MAX_INPUT_CHARS]
        logger.debug("Input truncated to %d chars for summarization.", _MAX_INPUT_CHARS)

    cfg = _get_settings()
    language_hint_sentence = (
        f" The content is likely in: {language_hint}." if language_hint else ""
    )
    prompt = cfg.prompt_chunk.format(
        content=combined,
        language_hint=language_hint_sentence,
    )

    response_text = call_llm(prompt, cfg, label="Summarization")
    if not response_text:
        return None
    return _parse_response(response_text)


def summarize_daily_texts(
    chunk_summaries: list,
) -> Optional[SummaryResult]:
    """
    Aggregate a list of per-chunk summaries into a single daily summary.

    chunk_summaries — list of summary strings (one per recording chunk)

    Returns SummaryResult or None on failure.
    """
    chunk_summaries = [s.strip() for s in chunk_summaries if s and s.strip()]
    if not chunk_summaries:
        return None

    combined = "\n\n".join(
        f"Chunk {i + 1}: {s}" for i, s in enumerate(chunk_summaries)
    )
    if len(combined) > _MAX_INPUT_CHARS:
        combined = combined[:_MAX_INPUT_CHARS]

    cfg = _get_settings()
    prompt = cfg.prompt_daily.format(content=combined)

    response_text = call_llm(prompt, cfg, label="Daily Summarization")
    if not response_text:
        return None
    return _parse_response(response_text)


# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------

def _get_settings():
    """Load SummarizationSettings singleton from the database."""
    from radios.models import SummarizationSettings
    return SummarizationSettings.get_settings()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(response_text: str) -> Optional[SummaryResult]:
    """Parse JSON response from the LLM into a SummaryResult."""
    text = response_text.strip()

    # Strip markdown fences in case the model ignored the instruction.
    if text.startswith("```"):
        lines = [line for line in text.split("\n")
                 if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not parse JSON from LLM response: %s — raw: %r",
            exc, text[:300],
        )
        return None

    summary = data.get("summary", "").strip()
    if not summary:
        logger.warning("LLM returned empty summary field.")
        return None

    raw_tags = data.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []

    # Normalize: lowercase, strip whitespace, deduplicate, cap count.
    tags = list(dict.fromkeys(
        t.lower().strip() for t in raw_tags
        if isinstance(t, str) and t.strip()
    ))[:_MAX_TAGS]

    return SummaryResult(summary_text=summary, tags=tags)
