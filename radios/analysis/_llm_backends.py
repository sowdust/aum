"""
Shared LLM backend dispatch for all analysis modules.

Supports four backends (configured via a settings object):
- "local_ollama"  — Ollama running locally (no API key needed)
- "cloud_ollama"  — ollama.com cloud API (OLLAMA_API_KEY env var required)
- "openai"        — OpenAI Chat API (OPENAI_API_KEY env var required)
- "anthropic"     — Claude (ANTHROPIC_API_KEY env var required)

Usage
-----
    from radios.analysis._llm_backends import call_llm

    # settings_obj must have: backend, local_ollama_model, local_ollama_url,
    # cloud_ollama_model, cloud_ollama_url, openai_model, anthropic_model
    text = call_llm("your prompt", settings_obj)
"""

import logging
import time
from typing import Optional

from django.conf import settings

logger = logging.getLogger("broadcast_analysis")

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0


class _PermanentError(Exception):
    """Raised inside a retry loop for errors that should not be retried."""


def call_llm(prompt: str, settings_obj, label: str = "LLM") -> Optional[str]:
    """
    Send a prompt to the configured LLM backend and return the raw text response.

    settings_obj is duck-typed: must have `backend`, `local_ollama_model`,
    `local_ollama_url`, `cloud_ollama_model`, `cloud_ollama_url`,
    `openai_model`, `anthropic_model`.

    Returns the raw text response or None on failure.
    """
    backend = settings_obj.backend
    handlers = {
        "local_ollama": _call_local_ollama,
        "cloud_ollama": _call_cloud_ollama,
        "openai": _call_openai,
        "anthropic": _call_anthropic,
    }
    handler = handlers.get(backend)
    if handler is None:
        logger.error("%s: unknown backend %r", label, backend)
        return None
    return handler(prompt, settings_obj, label)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _retry_with_backoff(func, backend_label):
    """
    Call func() up to _MAX_RETRIES times with exponential backoff.
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


# ---------------------------------------------------------------------------
# Backend: Ollama (shared by local and cloud)
# ---------------------------------------------------------------------------

def _call_ollama_raw(prompt, model, host, headers=None, label="Ollama"):
    """Shared Ollama implementation returning raw text."""
    try:
        import ollama as ollama_lib
    except ImportError:
        logger.error("ollama package is not installed — run: pip install ollama")
        return None

    logger.info("%s: host=%r model=%r", label, host, model)
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
        logger.debug("%s response: %d chars", label, len(result_text))
        return result_text

    return _retry_with_backoff(_do_request, label)


def _call_local_ollama(prompt, cfg, label):
    return _call_ollama_raw(
        prompt,
        model=cfg.local_ollama_model,
        host=cfg.local_ollama_url,
        label=f"{label} (Local Ollama)",
    )


def _call_cloud_ollama(prompt, cfg, label):
    api_key = settings.OLLAMA_API_KEY
    if not api_key:
        logger.error(
            "OLLAMA_API_KEY is not set — cannot use cloud_ollama backend."
        )
        return None

    host = cfg.cloud_ollama_url.rstrip("/")
    if "api.ollama.com" in host:
        logger.error(
            "cloud_ollama_url is set to %r — this is wrong. "
            "The Ollama cloud API lives at https://ollama.com (not api.ollama.com).",
            host,
        )
        return None

    return _call_ollama_raw(
        prompt,
        model=cfg.cloud_ollama_model,
        host=host,
        headers={"Authorization": f"Bearer {api_key}"},
        label=f"{label} (Cloud Ollama)",
    )


# ---------------------------------------------------------------------------
# Backend: OpenAI
# ---------------------------------------------------------------------------

def _call_openai(prompt, cfg, label):
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.error("OPENAI_API_KEY is not set — cannot use openai backend")
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
            max_tokens=4096,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    return _retry_with_backoff(_do_request, f"{label} (OpenAI)")


# ---------------------------------------------------------------------------
# Backend: Anthropic (Claude)
# ---------------------------------------------------------------------------

def _call_anthropic(prompt, cfg, label):
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        logger.error("ANTHROPIC_API_KEY is not set — cannot use anthropic backend")
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
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return _retry_with_backoff(_do_request, f"{label} (Anthropic)")
