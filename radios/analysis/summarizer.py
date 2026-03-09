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
import time
from typing import Optional

from django.conf import settings

logger = logging.getLogger("broadcast_analysis")

# Retry settings for API backends.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry

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
    return _call_backend(prompt, cfg)


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
    return _call_backend(prompt, cfg)


# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------

def _get_settings():
    """Load SummarizationSettings singleton from the database."""
    from radios.models import SummarizationSettings
    return SummarizationSettings.get_settings()


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------

_BACKENDS = {
    "local_ollama": "_call_local_ollama",
    "cloud_ollama": "_call_cloud_ollama",
    "openai": "_call_openai",
    "anthropic": "_call_anthropic",
}


def _call_backend(prompt: str, cfg) -> Optional[SummaryResult]:
    """Route the prompt to the configured backend."""
    handler_name = _BACKENDS.get(cfg.backend)
    if handler_name is None:
        logger.error("Unknown summarization backend: %r", cfg.backend)
        return None
    handler = globals()[handler_name]
    return handler(prompt, cfg)


# ---------------------------------------------------------------------------
# Response parser (shared by all backends)
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


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _retry_with_backoff(func, backend_label):
    """
    Call func() up to _MAX_RETRIES times with exponential backoff.

    func must return an Optional[SummaryResult].
    Non-retryable errors should be raised as _PermanentError.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return func()
        except _PermanentError as exc:
            logger.error("%s: %s", backend_label, exc)
            return None
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "%s error (attempt %d/%d), retrying in %.1fs: %s",
                    backend_label, attempt + 1, _MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d retries: %s",
                    backend_label, _MAX_RETRIES, exc,
                )
                return None
    return None


class _PermanentError(Exception):
    """Raised inside a retry loop for errors that should not be retried."""


# ---------------------------------------------------------------------------
# Backend: Ollama (shared by local and cloud)
# ---------------------------------------------------------------------------

def _call_ollama(prompt, model, host, headers=None, label="Ollama"):
    """
    Shared Ollama implementation for both local and cloud backends.

    Uses the ollama Python client (https://github.com/ollama/ollama-python).
    The cloud API is identical to the local API — just a different host with
    a Bearer token.  See https://docs.ollama.com/cloud
    """
    try:
        import ollama as ollama_lib
    except ImportError:
        logger.error("ollama package is not installed — run: pip install ollama")
        return None

    logger.info(
        "%s summarization: host=%r model=%r",
        label, host, model,
    )

    client = ollama_lib.Client(host=host, headers=headers or {})

    def _do_request():
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.3},
            )
        except ollama_lib.ResponseError as exc:
            status = getattr(exc, "status_code", None)
            if status == 401:
                raise _PermanentError(
                    f"Authentication failed (HTTP 401) at {host!r}. "
                    f"Verify your OLLAMA_API_KEY is valid "
                    f"(create one at https://ollama.com/settings/keys). "
                    f"Detail: {exc}"
                ) from exc
            if status == 404:
                raise _PermanentError(
                    f"Endpoint not found (HTTP 404) at {host!r}. "
                    f"Verify the host URL is correct and the model {model!r} exists. "
                    f"The Ollama cloud host is https://ollama.com (not api.ollama.com). "
                    f"Detail: {exc}"
                ) from exc
            # Other HTTP errors — let the retry loop handle them.
            raise

        result_text = response.message.content or ""
        logger.debug(
            "%s response: %d chars from %r", label, len(result_text), host,
        )
        return _parse_response(result_text)

    return _retry_with_backoff(_do_request, label)


def _call_local_ollama(prompt: str, cfg) -> Optional[SummaryResult]:
    """Summarize using a locally running Ollama instance."""
    return _call_ollama(
        prompt,
        model=cfg.local_ollama_model,
        host=cfg.local_ollama_url,
        label="Local Ollama",
    )


def _call_cloud_ollama(prompt: str, cfg) -> Optional[SummaryResult]:
    """
    Summarize using the ollama.com cloud API.

    Requires OLLAMA_API_KEY to be set as an environment variable.
    The correct cloud host is https://ollama.com (not api.ollama.com).
    See https://docs.ollama.com/cloud
    """
    api_key = settings.OLLAMA_API_KEY
    if not api_key:
        logger.error(
            "OLLAMA_API_KEY is not set — cannot use cloud_ollama backend. "
            "Create a key at https://ollama.com/settings/keys"
        )
        return None

    host = cfg.cloud_ollama_url.rstrip("/")

    # Guard against the most common misconfiguration.
    if "api.ollama.com" in host:
        logger.error(
            "cloud_ollama_url is set to %r — this is wrong. "
            "The Ollama cloud API lives at https://ollama.com (not api.ollama.com). "
            "Update SummarizationSettings.cloud_ollama_url in the Django admin. "
            "See https://docs.ollama.com/cloud",
            host,
        )
        return None

    return _call_ollama(
        prompt,
        model=cfg.cloud_ollama_model,
        host=host,
        headers={"Authorization": f"Bearer {api_key}"},
        label="Cloud Ollama",
    )


# ---------------------------------------------------------------------------
# Backend: OpenAI
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, cfg) -> Optional[SummaryResult]:
    """Summarize using the OpenAI Chat API."""
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.error(
            "OPENAI_API_KEY is not set — cannot use openai backend"
        )
        return None

    try:
        import openai
    except ImportError:
        logger.error("openai package is not installed — run: pip install openai")
        return None

    client = openai.OpenAI(api_key=api_key)

    def _do_request():
        response = client.chat.completions.create(
            model=cfg.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
        )
        result_text = response.choices[0].message.content or ""
        return _parse_response(result_text)

    return _retry_with_backoff(_do_request, "OpenAI")


# ---------------------------------------------------------------------------
# Backend: Anthropic (Claude)
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, cfg) -> Optional[SummaryResult]:
    """Summarize using the Anthropic Claude API."""
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        logger.error(
            "ANTHROPIC_API_KEY is not set — cannot use anthropic backend"
        )
        return None

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package is not installed — run: pip install anthropic")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    def _do_request():
        response = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text
        return _parse_response(result_text)

    return _retry_with_backoff(_do_request, "Anthropic")
