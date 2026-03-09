"""
Speech transcription with language detection and English translation.

Supports three backends (configured via the admin Transcription Settings page):
- "local"     — faster-whisper (runs on CPU/GPU, no API key needed)
- "openai"    — OpenAI Whisper API (OPENAI_API_KEY environment variable required)
- "anthropic" — Claude audio input (ANTHROPIC_API_KEY environment variable required)

Backend selection and model parameters are stored in the TranscriptionSettings
database model and configurable from the Django admin. API keys are never stored
in the database — set them as environment variables.

Requirements
------------
- ffmpeg on $PATH (for audio extraction)
- faster-whisper Python package (for local backend)
- openai Python package (for openai backend)
- anthropic Python package (for anthropic backend)

Usage
-----
    from radios.analysis.transcriber import transcribe_segment

    result = transcribe_segment("recording.mp3", 120.5, 185.3)
    if result:
        print(result.text, result.language, result.text_english)
"""

import base64
import dataclasses
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Optional

from django.conf import settings

logger = logging.getLogger("broadcast_analysis")

# Skip segments shorter than this — too short for reliable transcription.
_MIN_DURATION = 2.0

# Max audio per API call (seconds). Longer segments are split into chunks.
_MAX_CLIP = 600.0

# Trim segment edges to avoid bleed-over from adjacent segments.
_BOUNDARY_TRIM = 2.0

# Overlap between sub-chunks when splitting long segments (seconds).
_CHUNK_OVERLAP = 5.0

# Retry settings for API backends.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry


@dataclasses.dataclass
class TranscriptionResult:
    text: str           # Original language transcript
    text_english: str   # English translation (empty if already English)
    language: str       # Detected language code (ISO 639-1)
    confidence: float   # 0.0-1.0


def transcribe_segment(
    source_path: str,
    start: float,
    end: float,
    backend: str = "",
    language_hint: str = "",
) -> Optional[TranscriptionResult]:
    """
    Transcribe the speech in [start, end) seconds of source_path.

    backend — override the backend for this call; empty string (default) reads
              the active backend from TranscriptionSettings in the database.

    Returns a TranscriptionResult or None if the segment is too short,
    the audio cannot be extracted, or transcription fails.
    """
    if not backend:
        backend = _get_transcription_settings().backend
    if not os.path.isfile(source_path):
        logger.error("Source file does not exist: %s", source_path)
        return None

    # Validate language_hint (alphanumeric + comma only)
    if language_hint and not re.match(r'^[a-zA-Z0-9,]+$', language_hint):
        logger.warning("Invalid language_hint %r, ignoring", language_hint)
        language_hint = ""

    # Apply boundary trim
    trimmed_start = start + _BOUNDARY_TRIM
    trimmed_end = end - _BOUNDARY_TRIM
    duration = trimmed_end - trimmed_start

    if duration < _MIN_DURATION:
        logger.debug(
            "Segment too short to transcribe after boundary trim (%.1fs)", duration
        )
        return None

    if duration > _MAX_CLIP:
        return _split_and_transcribe(
            source_path, trimmed_start, trimmed_end, backend, language_hint
        )

    return _transcribe_slice(
        source_path, trimmed_start, trimmed_end, backend, language_hint
    )


def _transcribe_slice(
    source_path: str,
    start: float,
    end: float,
    backend: str,
    language_hint: str,
) -> Optional[TranscriptionResult]:
    """Extract audio slice and transcribe with the chosen backend."""
    fmt = "wav" if backend == "local" else "mp3"
    audio_path = _extract_audio_slice(source_path, start, end, fmt=fmt)
    if audio_path is None:
        return None

    try:
        if backend == "local":
            return _transcribe_local(audio_path, language_hint)
        elif backend == "openai":
            return _transcribe_openai(audio_path, language_hint)
        elif backend == "anthropic":
            return _transcribe_anthropic(audio_path, language_hint)
        elif backend == "ollama":
            return _transcribe_ollama(audio_path, language_hint)
        else:
            logger.error("Unknown transcription backend: %s", backend)
            return None
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def _extract_audio_slice(
    source_path: str, start: float, end: float, fmt: str = "wav"
) -> Optional[str]:
    """
    Extract [start, end) from source_path to a temp file via ffmpeg.
    Returns the temp file path, or None on failure.
    """
    suffix = f".{fmt}"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)

    duration = end - start
    codec_args = ["-ar", "16000", "-ac", "1"]
    if fmt == "wav":
        codec_args += ["-f", "wav"]
    else:
        codec_args += ["-codec:a", "libmp3lame", "-q:a", "4"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", source_path,
    ] + codec_args + [tmp_path]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            logger.error(
                "ffmpeg extraction failed for %s [%.1f-%.1f]: %s",
                source_path, start, end,
                proc.stderr.decode(errors="replace")[-500:],
            )
            os.unlink(tmp_path)
            return None
        return tmp_path
    except Exception as exc:
        logger.error("ffmpeg extraction error: %s", exc)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None


def _split_and_transcribe(
    source_path: str,
    start: float,
    end: float,
    backend: str,
    language_hint: str,
) -> Optional[TranscriptionResult]:
    """
    Split a long segment into sub-chunks of at most _MAX_CLIP seconds,
    transcribe each, and concatenate results.
    """
    chunks = []
    pos = start
    while pos < end:
        chunk_end = min(pos + _MAX_CLIP, end)
        chunks.append((pos, chunk_end))
        pos = chunk_end - _CHUNK_OVERLAP if chunk_end < end else end

    texts = []
    texts_english = []
    language = ""
    total_confidence = 0.0
    count = 0

    for chunk_start, chunk_end in chunks:
        result = _transcribe_slice(
            source_path, chunk_start, chunk_end, backend, language_hint
        )
        if result and result.text.strip():
            texts.append(result.text.strip())
            if result.text_english:
                texts_english.append(result.text_english.strip())
            if not language:
                language = result.language
            total_confidence += result.confidence
            count += 1

    if not texts:
        return None

    return TranscriptionResult(
        text=" ".join(texts),
        text_english=" ".join(texts_english) if texts_english else "",
        language=language,
        confidence=total_confidence / count if count else 0.0,
    )


# ---------------------------------------------------------------------------
# Backend: local (faster-whisper)
# ---------------------------------------------------------------------------

_local_model = None


def _get_transcription_settings():
    """Load TranscriptionSettings singleton from the database."""
    from radios.models import TranscriptionSettings
    return TranscriptionSettings.get_settings()


def _get_local_model():
    """Lazy-load the faster-whisper model as a singleton."""
    global _local_model
    if _local_model is None:
        from faster_whisper import WhisperModel

        cfg = _get_transcription_settings()
        logger.info(
            "Loading faster-whisper model: %s (device=%s, compute=%s)",
            cfg.local_model_size, cfg.local_device, cfg.local_compute_type,
        )
        _local_model = WhisperModel(
            cfg.local_model_size,
            device=cfg.local_device,
            compute_type=cfg.local_compute_type,
        )
    return _local_model


def _transcribe_local(audio_path: str, language_hint: str) -> Optional[TranscriptionResult]:
    """Transcribe using faster-whisper (local)."""
    try:
        model = _get_local_model()
    except Exception as exc:
        logger.error("Failed to load faster-whisper model: %s", exc)
        return None

    try:
        # First language from hint, if any
        lang = language_hint.split(",")[0].strip().lower() if language_hint else None
        # faster-whisper expects None for auto-detection
        if lang and len(lang) != 2:
            lang = None

        segments, info = model.transcribe(
            audio_path,
            language=lang if lang else None,
            beam_size=5,
            task="transcribe",
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts)
        if not text.strip():
            return None

        detected_lang = info.language or ""
        confidence = info.language_probability or 0.0

        # If not English, get translation
        text_english = ""
        if detected_lang and detected_lang != "en":
            try:
                translate_segments, _ = model.transcribe(
                    audio_path,
                    language=detected_lang,
                    beam_size=5,
                    task="translate",
                )
                en_parts = []
                for segment in translate_segments:
                    en_parts.append(segment.text.strip())
                text_english = " ".join(en_parts)
            except Exception as exc:
                logger.warning("Translation to English failed: %s", exc)

        return TranscriptionResult(
            text=text,
            text_english=text_english,
            language=detected_lang,
            confidence=confidence,
        )
    except Exception as exc:
        logger.error("faster-whisper transcription failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Backend: openai (OpenAI Whisper API)
# ---------------------------------------------------------------------------

def _transcribe_openai(audio_path: str, language_hint: str) -> Optional[TranscriptionResult]:
    """Transcribe using the OpenAI Whisper API."""
    import os
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set — cannot use openai backend")
        return None

    cfg = _get_transcription_settings()

    try:
        import openai
    except ImportError:
        logger.error("openai package is not installed")
        return None

    client = openai.OpenAI(api_key=api_key)

    for attempt in range(_MAX_RETRIES):
        try:
            # Transcribe in original language
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model=cfg.openai_model,
                    file=f,
                    response_format="verbose_json",
                    **({"language": language_hint.split(",")[0].strip()[:2]}
                       if language_hint else {}),
                )

            text = transcript.text or ""
            detected_lang = getattr(transcript, "language", "") or ""

            if not text.strip():
                return None

            # Translate to English if not already English
            text_english = ""
            if detected_lang and detected_lang != "en":
                with open(audio_path, "rb") as f:
                    translation = client.audio.translations.create(
                        model=cfg.openai_model,
                        file=f,
                    )
                text_english = translation.text or ""

            return TranscriptionResult(
                text=text,
                text_english=text_english,
                language=detected_lang,
                confidence=1.0,  # OpenAI API doesn't return confidence
            )

        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.error("OpenAI API failed after %d retries: %s", _MAX_RETRIES, exc)
                return None
    return None


# ---------------------------------------------------------------------------
# Backend: ollama (OpenAI-compatible — local or ollama.com cloud)
# ---------------------------------------------------------------------------

def _transcribe_ollama(audio_path: str, language_hint: str) -> Optional[TranscriptionResult]:
    """
    Transcribe using Ollama's OpenAI-compatible audio transcription endpoint.

    Works with both a local Ollama instance (no key needed) and the ollama.com
    cloud (set OLLAMA_API_KEY and point ollama_base_url to https://api.ollama.com).
    """
    try:
        import openai
    except ImportError:
        logger.error("openai package is not installed")
        return None

    api_key = settings.OLLAMA_API_KEY
    cfg = _get_transcription_settings()

    # Avoid double-appending /v1 if the admin already included it in the URL
    raw_url = cfg.ollama_base_url.rstrip("/")
    base_url = raw_url if raw_url.endswith("/v1") else raw_url + "/v1"

    logger.info(
        "Ollama transcription: base_url=%r model=%r api_key=%s",
        base_url,
        cfg.ollama_model,
        f"{api_key[:8]}…" if api_key and len(api_key) > 8 else ("(set, short)" if api_key else "(not set — using 'ollama')"),
    )

    # Local Ollama ignores the api_key value but the openai client requires a non-empty string.
    client = openai.OpenAI(api_key=api_key or "ollama", base_url=base_url)

    for attempt in range(_MAX_RETRIES):
        try:
            kwargs = {
                "model": cfg.ollama_model,
                "response_format": "verbose_json",
            }
            if language_hint:
                lang = language_hint.split(",")[0].strip()[:2].lower()
                if lang:
                    kwargs["language"] = lang

            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(file=f, **kwargs)

            text = transcript.text or ""
            if not text.strip():
                return None

            detected_lang = getattr(transcript, "language", "") or ""

            return TranscriptionResult(
                text=text,
                text_english="",  # Ollama transcription does not auto-translate
                language=detected_lang,
                confidence=1.0,
            )

        except Exception as exc:
            exc_str = str(exc)
            if "redirect" in exc_str.lower() or "301" in exc_str or "302" in exc_str:
                logger.error(
                    "Ollama transcription: server redirected the request — the configured URL %r "
                    "is likely wrong. Check TranscriptionSettings.ollama_base_url in admin. "
                    "Error: %s",
                    base_url, exc,
                )
                return None
            if "401" in exc_str or "unauthorized" in exc_str.lower() or "authentication" in exc_str.lower():
                logger.error(
                    "Ollama transcription: authentication failed (HTTP 401). "
                    "Verify OLLAMA_API_KEY is correct and the configured URL %r is right. "
                    "Error: %s",
                    base_url, exc,
                )
                return None
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Ollama transcription error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.error("Ollama transcription failed after %d retries: %s", _MAX_RETRIES, exc)
                return None
    return None


# ---------------------------------------------------------------------------
# Backend: anthropic (Claude audio input)
# ---------------------------------------------------------------------------

def _transcribe_anthropic(audio_path: str, language_hint: str) -> Optional[TranscriptionResult]:
    """
    Transcribe using Claude's audio input capability.
    Sends audio as base64 and asks for JSON with text, translation, and language.
    """
    import os
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is not set — cannot use anthropic backend")
        return None

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package is not installed")
        return None

    model = _get_transcription_settings().anthropic_model

    # Read and encode audio
    try:
        with open(audio_path, "rb") as f:
            audio_data = base64.standard_b64encode(f.read()).decode("ascii")
    except Exception as exc:
        logger.error("Failed to read audio file %s: %s", audio_path, exc)
        return None

    # Determine media type from extension
    ext = os.path.splitext(audio_path)[1].lower()
    media_type_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
    }
    media_type = media_type_map.get(ext, "audio/mpeg")

    hint_text = f" The audio is likely in: {language_hint}." if language_hint else ""
    prompt = (
        "Transcribe this audio clip. Return ONLY valid JSON with these fields:\n"
        '- "text": the verbatim transcript in the original language\n'
        '- "text_english": English translation if the original is not English, otherwise empty string\n'
        '- "language": ISO 639-1 language code (e.g. "en", "it", "es")\n'
        '- "confidence": your confidence in the transcription accuracy, 0.0 to 1.0\n'
        f"{hint_text}\n"
        "Respond with ONLY the JSON object, no markdown fences or explanation."
    )

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": audio_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            response_text = response.content[0].text.strip()

            # Strip markdown fences if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json) and last line (```)
                lines = [l for l in lines if not l.strip().startswith("```")]
                response_text = "\n".join(lines).strip()

            data = json.loads(response_text)

            text = data.get("text", "")
            if not text.strip():
                return None

            return TranscriptionResult(
                text=text,
                text_english=data.get("text_english", ""),
                language=data.get("language", ""),
                confidence=float(data.get("confidence", 0.8)),
            )

        except json.JSONDecodeError as exc:
            logger.warning(
                "Claude returned non-JSON response (attempt %d/%d): %s",
                attempt + 1, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
            else:
                logger.error("Claude JSON parsing failed after %d retries", _MAX_RETRIES)
                return None
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Anthropic API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.error("Anthropic API failed after %d retries: %s", _MAX_RETRIES, exc)
                return None
    return None
