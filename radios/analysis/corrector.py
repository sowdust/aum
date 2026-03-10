"""
LLM-based transcription correction and translation.

Fixes speech-to-text errors (misspelled names, garbled words) and produces
English translations from the corrected text. Runs after transcription,
before summarization.

Supports four backends (configured via the admin Transcription Settings page):
- "local_ollama"  -- Ollama running locally (no API key needed)
- "cloud_ollama"  -- ollama.com cloud API (OLLAMA_API_KEY env var required)
- "openai"        -- OpenAI Chat API (OPENAI_API_KEY env var required)
- "anthropic"     -- Claude (ANTHROPIC_API_KEY env var required)

Usage
-----
    from radios.analysis.corrector import correct_transcription

    corrections = correct_transcription(
        segments_data=[{"index": 0, "text": "some garbled text"}, ...],
        radio_name="Radio Example",
        radio_location="Rome, Italy",
        radio_language="it",
    )
    if corrections:
        for c in corrections:
            print(c["index"], c["text"], c["text_english"])
"""

import json
import logging
import time
from typing import Optional

from django.conf import settings

logger = logging.getLogger("broadcast_analysis")

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

_MAX_INPUT_CHARS = 40_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_transcription(
    segments_data: list,
    radio_name: str = "",
    radio_location: str = "",
    radio_language: str = "",
) -> Optional[list]:
    """
    Correct transcription errors and produce English translations.

    segments_data -- list of {"index": N, "text": "raw transcription"}
    radio_name    -- name of the radio station (context for the LLM)
    radio_location -- city/country of the radio (context for the LLM)
    radio_language -- language code(s) of the radio (context for the LLM)

    Returns list of {"index": N, "text": "corrected", "text_english": "translation"}
    or None on failure.
    """
    if not segments_data:
        return None

    cfg = _get_settings()

    segments_text = "\n".join(
        f"{s['index']}. {s['text']}" for s in segments_data
    )
    if len(segments_text) > _MAX_INPUT_CHARS:
        segments_text = segments_text[:_MAX_INPUT_CHARS]
        logger.debug("Input truncated to %d chars for correction.", _MAX_INPUT_CHARS)

    prompt = cfg.correction_prompt.format(
        segments=segments_text,
        radio_name=radio_name,
        radio_location=radio_location,
        radio_language=radio_language,
    )

    return _call_backend(prompt, cfg)


# ---------------------------------------------------------------------------
# Settings loader
# ---------------------------------------------------------------------------

def _get_settings():
    from radios.models import TranscriptionSettings
    return TranscriptionSettings.get_settings()


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------

_BACKENDS = {
    "local_ollama": "_call_local_ollama",
    "cloud_ollama": "_call_cloud_ollama",
    "openai": "_call_openai",
    "anthropic": "_call_anthropic",
}


def _call_backend(prompt: str, cfg) -> Optional[list]:
    handler_name = _BACKENDS.get(cfg.correction_backend)
    if handler_name is None:
        logger.error("Unknown correction backend: %r", cfg.correction_backend)
        return None
    handler = globals()[handler_name]
    return handler(prompt, cfg)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(response_text: str) -> Optional[list]:
    """Parse JSON array response from the LLM into a list of corrections."""
    text = response_text.strip()

    if text.startswith("```"):
        lines = [line for line in text.split("\n")
                 if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not parse JSON from correction LLM response: %s -- raw: %r",
            exc, text[:300],
        )
        return None

    if not isinstance(data, list):
        logger.warning("Correction LLM returned non-array JSON: %s", type(data).__name__)
        return None

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "index" not in item or "text" not in item or "text_english" not in item:
            logger.warning("Correction item missing required fields: %r", item)
            continue
        results.append({
            "index": item["index"],
            "text": str(item["text"]),
            "text_english": str(item["text_english"]),
        })

    if not results:
        logger.warning("Correction LLM returned no valid items.")
        return None

    return results


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

class _PermanentError(Exception):
    """Raised inside a retry loop for errors that should not be retried."""


def _retry_with_backoff(func, backend_label):
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


# ---------------------------------------------------------------------------
# Backend: Ollama (shared by local and cloud)
# ---------------------------------------------------------------------------

def _call_ollama(prompt, model, host, headers=None, label="Ollama"):
    try:
        import ollama as ollama_lib
    except ImportError:
        logger.error("ollama package is not installed -- run: pip install ollama")
        return None

    logger.info("%s correction: host=%r model=%r", label, host, model)

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
                    f"Verify your OLLAMA_API_KEY is valid."
                ) from exc
            if status == 404:
                raise _PermanentError(
                    f"Endpoint not found (HTTP 404) at {host!r}. "
                    f"Verify the host URL is correct and the model {model!r} exists."
                ) from exc
            raise

        result_text = response.message.content or ""
        logger.debug("%s response: %d chars from %r", label, len(result_text), host)
        return _parse_response(result_text)

    return _retry_with_backoff(_do_request, label)


def _call_local_ollama(prompt: str, cfg) -> Optional[list]:
    return _call_ollama(
        prompt,
        model=cfg.correction_local_ollama_model,
        host=cfg.correction_local_ollama_url,
        label="Local Ollama (correction)",
    )


def _call_cloud_ollama(prompt: str, cfg) -> Optional[list]:
    api_key = settings.OLLAMA_API_KEY
    if not api_key:
        logger.error(
            "OLLAMA_API_KEY is not set -- cannot use cloud_ollama correction backend."
        )
        return None

    host = cfg.correction_cloud_ollama_url.rstrip("/")

    if "api.ollama.com" in host:
        logger.error(
            "correction_cloud_ollama_url is set to %r -- this is wrong. "
            "The Ollama cloud API lives at https://ollama.com (not api.ollama.com).",
            host,
        )
        return None

    return _call_ollama(
        prompt,
        model=cfg.correction_cloud_ollama_model,
        host=host,
        headers={"Authorization": f"Bearer {api_key}"},
        label="Cloud Ollama (correction)",
    )


# ---------------------------------------------------------------------------
# Backend: OpenAI
# ---------------------------------------------------------------------------

def _call_openai(prompt: str, cfg) -> Optional[list]:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.error("OPENAI_API_KEY is not set -- cannot use openai correction backend")
        return None

    try:
        import openai
    except ImportError:
        logger.error("openai package is not installed -- run: pip install openai")
        return None

    client = openai.OpenAI(api_key=api_key)

    def _do_request():
        response = client.chat.completions.create(
            model=cfg.correction_openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.3,
        )
        result_text = response.choices[0].message.content or ""
        return _parse_response(result_text)

    return _retry_with_backoff(_do_request, "OpenAI (correction)")


# ---------------------------------------------------------------------------
# Backend: Anthropic (Claude)
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, cfg) -> Optional[list]:
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        logger.error("ANTHROPIC_API_KEY is not set -- cannot use anthropic correction backend")
        return None

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package is not installed -- run: pip install anthropic")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    def _do_request():
        response = client.messages.create(
            model=cfg.correction_anthropic_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text
        return _parse_response(result_text)

    return _retry_with_backoff(_do_request, "Anthropic (correction)")
