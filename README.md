# AUM — Radio Archive

AUM is a Django web application for managing radio stations, recording live audio streams, and archiving recordings with automatic content analysis (speech/music segmentation, song fingerprinting, transcription).

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the app](#running-the-app)
5. [Stream recording](#stream-recording)
6. [Analysis pipeline](#analysis-pipeline)
7. [URL structure](#url-structure)
8. [Testing](#testing)
9. [Project layout](#project-layout)

---

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on `$PATH`
- SQLite3 (bundled with Python)

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd aum

# Create and activate a virtual environment
python3.9 -m venv env
source env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Apply database migrations
python manage.py migrate

# Create a superuser (admin account)
python manage.py createsuperuser
```

---

## Configuration

All settings are in `aum/settings.py`. Key values to review before deploying:

| Setting | Default | Description |
|---------|---------|-------------|
| `SECRET_KEY` | (insecure default) | **Change this in production.** |
| `DEBUG` | `True` | Set to `False` in production. |
| `ALLOWED_HOSTS` | `[]` | Add your domain/IP for production. |
| `MEDIA_ROOT` | `<project>/media/` | Where uploaded files and recordings are stored. |
| `CHUNK_SIZE` | `1200` (20 min) | Recording chunk duration in seconds. |
| `ACOUSTID_API_KEY` | `""` | Free API key from [acoustid.org](https://acoustid.org/) for song identification. |

### Analysis pipeline settings

```python
# Minimum segment duration (seconds) — shorter segments are merged into neighbours.
# Higher = fewer, longer blocks. Lower = finer granularity. Recommended: 10–30.
SEGMENT_MIN_DURATION = 15.0

# Directory where segment MP3 files are saved when save_dir is requested.
# If unset, segments are not written to disk unless explicitly passed to segment_audio().
SEGMENT_SAVE_DIR = ""   # e.g. BASE_DIR / "media" / "segments"

# Transcription backend: "local" (faster-whisper) or "api" (OpenAI Whisper)
TRANSCRIPTION_BACKEND = "local"
WHISPER_MODEL_SIZE = "medium"   # tiny | base | small | medium | large-v3
WHISPER_DEVICE = "cpu"          # cpu | cuda

# LLM summarisation
LLM_PROVIDER = "openai"         # openai | anthropic
LLM_MODEL = "gpt-4o-mini"
LLM_API_KEY = ""                # or set via environment variable
```

API keys can be passed via environment variables instead of hardcoding them:

```bash
export ACOUSTID_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

### Email

The default email backend prints to the console (development only).  For
production, configure a real SMTP backend in `settings.py`.

---

## Running the app

```bash
source env/bin/activate

# Development server (http://127.0.0.1:8000/)
python manage.py runserver

# Open the admin panel at http://127.0.0.1:8000/admin/
```

---

## Stream recording

The stream recorder is a long-running management command that captures all
active `Stream` objects with `ffmpeg`, writing MP3 files in chunks
(`CHUNK_SIZE` seconds, default 20 minutes):

```bash
source env/bin/activate
python manage.py record_streams
```

- Handles `SIGINT` / `SIGTERM` for graceful shutdown (waits for current
  chunks to finish before exiting).
- Recorded files are saved under `media/recordings/<stream-name>/<YYYY>/<MM>/<DD>/`.
- Logs via the `stream_recorder` logger (stdout by default).

To run continuously in the background (e.g. on a server):

```bash
nohup python manage.py record_streams &
```

---

## Analysis pipeline

After recording, the analysis pipeline processes each `Recording` through
up to five stages:

| Stage | Tool | Output |
|-------|------|--------|
| **record** | ffmpeg | MP3 chunks |
| **segment** | inaSpeechSegmenter CNN | speech / music / noise / silence blocks |
| **fingerprint** | AcoustID + fpcalc | artist + title for music segments |
| **transcribe** | faster-whisper *(planned)* | text for speech segments |
| **summarize** | LLM *(planned)* | summary of transcribed text |

Run the analysis command to process pending recordings:

```bash
python manage.py analyze_recordings
```

For full documentation of the segmentation stage — tuning parameters,
boundary refinement, Audacity ground-truth labelling workflow, and the
full test suite — see [`radios/analysis/README.md`](radios/analysis/README.md).

---

## URL structure

| URL | View | Description |
|-----|------|-------------|
| `/` | `radio_list` | Home — list of all radios |
| `/radios/` | `radio_list` | Same as above |
| `/radios/<slug>/` | `radio_detail` | Radio detail page |
| `/radios/<slug>/recordings` | `recording_archive` | Recording archive with date filter |
| `/register/` | `register` | New user registration |
| `/login/` | `login` | Login |
| `/logout/` | `logout` | Logout |
| `/verify/<token>/` | `verify_email` | Email verification |
| `/password-reset/` | — | Standard Django password reset flow |
| `/admin/` | Django admin | Site administration |

---

## Testing

```bash
source env/bin/activate

# Fast unit tests only (skips slow integration tests)
python manage.py test radios --exclude-tag=slow

# All integration / analysis tests (slow — requires test_1.mp3 fixture)
python manage.py test radios --tag=integration

# Individual test modules
python manage.py test radios.tests.test_segmentation
python manage.py test radios.tests.test_fingerprinting
python manage.py test radios.tests.test_pipeline

# With fingerprinting (requires AcoustID key)
ACOUSTID_API_KEY=your_key python manage.py test radios.tests.test_pipeline

# Save each detected segment as a separate MP3 (written to media/segments/)
SAVE_SEGMENTS=1 python manage.py test radios.tests.test_segmentation
```

Test fixtures live in `radios/tests/test_files/`.  Override them with:

```bash
TEST_MP3=/path/to/recording.mp3 \
TEST_LABELS=/path/to/labels.txt \
    python manage.py test radios.tests.test_segmentation
```

---

## Project layout

```
aum/                        Django project config (settings, URLs, WSGI)
radios/                     Single application
  models.py                 RadioUser, Radio, Stream, Recording, ...
  views.py                  All views
  urls.py                   URL routing
  forms.py                  Registration, radio creation, stream formset
  admin.py                  Django admin registration
  management/
    commands/
      record_streams.py     Stream recording daemon
      analyze_recordings.py Analysis pipeline entry point
  analysis/
    segmenter.py            Audio segmentation (inaSpeechSegmenter CNN)
    fingerprinter.py        Song identification (AcoustID + fpcalc)
    audacity_to_labels.py   Convert Audacity exports to ground-truth JSON
    tune.py                 Boundary accuracy evaluation + grid search
    README.md               Full analysis pipeline documentation
  tests/
    test_segmentation.py    Segmentation integration tests
    test_fingerprinting.py  Fingerprinting integration tests
    test_pipeline.py        End-to-end pipeline tests
    test_files/             Test fixtures (MP3 + Audacity label files)
  templates/                HTML templates
media/                      Uploaded files, recordings, saved segments
static/                     Static assets (CSS, JS, images)
env/                        Python virtual environment (not committed)
```
