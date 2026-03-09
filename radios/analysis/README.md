# Broadcast Analysis

Audio segmentation and analysis pipeline for AUM radio recordings.

## Overview

The segmenter classifies audio into content types so downstream tools know
what to do with each piece:

| Segment type | Meaning                  | Downstream action           |
|--------------|--------------------------|------------------------------|
| `speech`     | Human talking            | Transcribe with Whisper      |
| `music`      | Instrumental or vocal    | Fingerprint with AcoustID    |
| `noise`      | Non-speech, non-music    | Skip or flag                 |
| `noEnergy`   | Dead air / silence       | Skip                         |

The engine is **inaSpeechSegmenter** -- a CNN trained on broadcast radio/TV
audio by INA (French National Audiovisual Institute).

## How it works

1. **CNN classification** -- inaSpeechSegmenter processes the audio and
   produces frame-level labels (~0.02 s resolution).  This creates many
   tiny fragments, especially near content transitions.

2. **Two-pass consolidation** -- Short fragments are merged into
   broadcast-scale blocks (minutes of speech, full songs):
   - **Pass 1 (< 3 s)** -- removes CNN classification jitter (1-3 s label
     flickers at boundaries), sharpening transition zones.
   - **Pass 2 (< SEGMENT_MIN_DURATION)** -- absorbs remaining short
     segments to produce broadcast-scale blocks.
   - Both passes process the **shortest** segment first (not left-to-right)
     to avoid systematically shifting boundaries forward in time.
   - Prefers type-matching neighbours to preserve real transition positions.

3. **Boundary refinement (proximity-weighted spectral flux)** -- After
   consolidation, each boundary is fine-tuned using spectral flux (how much
   the audio's frequency content changes between consecutive frames).  The
   flux is smoothed over 2 s and searched within +/- SEGMENT_MIN_DURATION
   seconds.  Peak selection uses a Gaussian proximity weight (sigma = 5 s)
   centred on the consolidated boundary, so a moderate transition peak 2 s
   away easily beats a loud drum hit 10 s away.  This prevents the
   refinement from jumping to beats/onsets within a music segment.

4. **Energy tagging** -- Each segment gets an RMS energy value in dB,
   useful for downstream silence detection or volume normalisation.

## Django settings

Add these to `aum/settings.py` to tune segmentation behaviour:

```python
# Minimum segment duration in seconds (default: 15.0).
# Segments shorter than this are absorbed into neighbours.
# Higher = fewer, longer blocks (cleaner for radio shows).
# Lower = finer granularity for detailed analysis.
# Recommended range: 10-30.
SEGMENT_MIN_DURATION = 15.0

# Directory where segment MP3 files are written when saving is enabled.
# If unset (or empty string), segments are not written to disk unless
# save_dir is passed explicitly to segment_audio().
# Example: SEGMENT_SAVE_DIR = BASE_DIR / "media" / "segments"
SEGMENT_SAVE_DIR = ""
```

The legacy webrtcvad settings (`SILENCE_THRESHOLD_DB`, `VAD_AGGRESSIVENESS`)
are only used if you roll back to `segmenter_webrtcvad.py`.

## Install

### Prerequisites

- Python 3.9+
- ffmpeg and ffprobe on `$PATH`
- ~2 GB disk for TensorFlow + model weights

### Steps

```bash
# Activate the project virtualenv
source env/bin/activate

# Set TMPDIR if the default /tmp partition is too small
export TMPDIR=/home/user/tmp
mkdir -p "$TMPDIR"

# 1. Install TensorFlow CPU (must be <2.16 for Python 3.9 compatibility)
pip install 'tensorflow-cpu<2.16'

# 2. Install inaSpeechSegmenter
#    It will pull in numpy, scipy, soundfile, pandas, etc.
pip install inaSpeechSegmenter

# 3. Verify the install
python -c "from inaSpeechSegmenter import Segmenter; print('OK')"
```

**Notes:**

- On Python 3.9, TensorFlow >= 2.16 has a `typing` incompatibility.
  Pin to `tensorflow-cpu<2.16` (installs 2.15.x).
- We use `tensorflow-cpu` instead of `tensorflow` to avoid pulling in
  ~4 GB of CUDA/cuDNN packages. The segmenter runs fine on CPU for
  batch processing (roughly 10x real-time on a modern server).
- The first call to `segment_audio()` downloads the CNN model weights
  (~3 MB) from GitHub and caches them locally.
- If `pip install` fails with "No space left on device", clear the pip
  cache first: `pip cache purge`.

### GPU support (optional)

If you have an NVIDIA GPU and want faster processing:

```bash
# Replace tensorflow-cpu with full tensorflow (includes CUDA bindings)
pip install 'tensorflow<2.16'
```

## Usage

```python
from radios.analysis.segmenter import segment_audio

# Basic usage — segments are not written to disk
segments = segment_audio("path/to/recording.mp3")

# Save each segment as a separate MP3 under /path/to/output/<stem>/
segments = segment_audio("path/to/recording.mp3", save_dir="/path/to/output")

for s in segments:
    dur = s.end - s.start
    saved = f"  -> {s.file_path}" if s.file_path else ""
    print(f"{s.segment_type:10s}  {s.start:8.1f}s - {s.end:8.1f}s  ({dur:.0f}s)  {s.energy_db:.1f} dB{saved}")
```

The function returns a list of `AudioSegment` dataclasses:

```python
@dataclasses.dataclass
class AudioSegment:
    start: float                    # seconds from file start
    end: float                      # seconds from file start
    segment_type: str               # speech | music | noise | noEnergy
    energy_db: float                # average RMS energy in dB
    file_path: Optional[str]        # absolute path to saved MP3, or None
```

### Saving segments to disk

When `save_dir` is provided (or `SEGMENT_SAVE_DIR` is set in Django
settings), each final segment is extracted from the source file with
`ffmpeg` (stream copy — no re-encoding) and written as:

```
<save_dir>/<source_stem>/<NNN>_<type>_<MMmSSs>-<MMmSSs>.mp3
```

For example:
```
media/segments/test_1/000_speech_00m00s-01m18s.mp3
media/segments/test_1/001_music_01m18s-06m23s.mp3
media/segments/test_1/002_speech_06m23s-09m41s.mp3
```

The `file_path` field on each returned `AudioSegment` holds the absolute
path to the file, or `None` if saving failed or was not requested.

## Quick test

To test the segmenter on any audio file (MP3, WAV, etc.) from the project root:

```bash
source env/bin/activate

python manage.py shell -c "
from radios.analysis.segmenter import segment_audio

segments = segment_audio('path/to/your/recording.mp3')

def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f'{m:02d}:{s:02d}'

print(f'{'#':>3}  {'Type':<12} {'Start':>7} {'End':>7} {'Dur':>6}  {'Energy':>8}')
print('-' * 52)
for i, s in enumerate(segments):
    dur = s.end - s.start
    m, sec = divmod(int(dur), 60)
    print(f'{i:>3}  {s.segment_type:<12} {fmt(s.start):>7} {fmt(s.end):>7}  {m:>2d}m{sec:02d}s  {s.energy_db:>7.1f} dB')

print()
total = segments[-1].end if segments else 0
for stype in ('speech', 'music', 'noise', 'noEnergy'):
    t = sum(s.end - s.start for s in segments if s.segment_type == stype)
    if t > 0:
        m, sec = divmod(int(t), 60)
        print(f'{stype:<12}  {m}m {sec:02d}s  ({100*t/total:.0f}%)')
"
```

To test on actual recordings already in the database:

```bash
python manage.py shell -c "
from radios.models import Recording
from radios.analysis.segmenter import segment_audio

rec = Recording.objects.order_by('-start_time').first()
print(f'Analysing: {rec.file.path}')
segments = segment_audio(rec.file.path)

for s in segments:
    dur = s.end - s.start
    print(f'{s.segment_type:<20} {s.start:7.1f}s - {s.end:7.1f}s  ({dur:.0f}s)  {s.energy_db:.1f}dB')
"
```

The first call is slow (~10 s) because it loads the CNN weights. Subsequent
calls on the same process are fast (the model is cached as a singleton).

## Tuning boundary accuracy

The segmenter has several parameters that control how precisely it places
segment boundaries.  They are all collected at the top of `segmenter.py`
under the `# Tuning parameters` block:

| Parameter | Default | Effect |
|---|---|---|
| `JITTER_THRESHOLD` | 3.0 s | Pass 1: absorb CNN label flicker shorter than this |
| `SEGMENT_MIN_DURATION_DEFAULT` | 15.0 s | Pass 2: minimum block length (overridden by Django `SEGMENT_MIN_DURATION`) |
| `FLUX_HOP_SEC` | 0.25 s | FFT frame step — smaller = finer time resolution |
| `FLUX_WIN_SEC` | 0.5 s | FFT window — larger = better frequency resolution, worse time precision |
| `FLUX_SMOOTH_SEC` | 2.0 s | Smoothing window — smaller = less beat suppression, sharper boundary location |
| `REFINE_SEARCH_RADIUS` | 15.0 s | Max distance the refinement can move a boundary from the CNN position |
| `PROXIMITY_SIGMA` | 5.0 s | Gaussian sigma — smaller = refinement stays closer to CNN boundary |

To improve accuracy systematically, use the tuning workflow described below.

---

### Step 1 — Label ground truth in Audacity

You need at least 2-3 recordings with manually verified boundaries.
Even 2-3 transitions per file is enough to detect systematic errors.

**Install Audacity** (free, open-source): https://www.audacityteam.org/

**Labeling procedure:**

1. Open your recording in Audacity (`File > Open`).

2. Add a label track: `Tracks > Add New > Label Track`.
   A thin empty track appears below the waveform.

3. For each segment in the recording:
   - Click at the start of the segment in the waveform and drag to the end
     (the selection turns grey).
   - Press **Ctrl+B** (Windows/Linux) or **Cmd+B** (Mac).
     A region label appears with the selected time range.
   - Type the content type into the label box:
     `speech`, `music`, `noise`, or `noEnergy`.
   - Press Enter to confirm.

4. Repeat for every segment from start to finish.  Make sure the regions
   cover the entire file with no gaps.

5. Export the labels: `File > Export > Export Labels`.
   Save the file alongside your recording, e.g. `recording.txt`.

**Tips for accurate labeling:**
- Zoom in with Ctrl+scroll (or View > Zoom In) near each transition
  before placing the boundary — Audacity shows the waveform at high
  resolution, making the speech/music switch easy to spot visually.
- The boundary should be at the exact moment the content changes,
  not at the nearest word or beat.
- For gradual fades, pick the midpoint of the fade.
- You can adjust label boundaries after placing them by dragging the
  edge of the region label.

---

### Step 2 — Convert Audacity labels to JSON

Audacity exports a tab-separated text file like this:

```
0.000000    187.000000    speech
187.000000  534.500000    music
534.500000  721.000000    speech
```

Convert it to the format `tune.py` expects:

```bash
source env/bin/activate
cd /path/to/aum

python radios/analysis/audacity_to_labels.py \
    /path/to/recording.txt \
    /path/to/recording.mp3 \
    labels.json
```

Running the command again for a second recording **appends** to the same
`labels.json` rather than overwriting it, so you can build up a dataset
one file at a time:

```bash
python radios/analysis/audacity_to_labels.py \
    /path/to/recording2.txt \
    /path/to/recording2.mp3 \
    labels.json
```

The resulting `labels.json` looks like this:

```json
[
  {
    "file": "/absolute/path/to/recording.mp3",
    "segments": [
      {"start": 0.0,   "end": 187.0, "type": "speech"},
      {"start": 187.0, "end": 534.5, "type": "music"},
      {"start": 534.5, "end": 721.0, "type": "speech"}
    ]
  },
  {
    "file": "/absolute/path/to/recording2.mp3",
    "segments": [...]
  }
]
```

You can also write or edit `labels.json` by hand if you prefer.

---

### Step 3 — Evaluate current parameters

Run a single evaluation pass to see how accurate the current settings are:

```bash
source env/bin/activate

python radios/analysis/tune.py labels.json
```

Output example:

```
Loaded 2 labeled file(s).

Evaluating current parameters...

  recording.mp3 ... boundaries: 4, mean=3.2s, max=6.1s
  recording2.mp3 ... boundaries: 3, mean=1.8s, max=3.4s

Results across 7 boundaries:
  mean error:   2.6 s
  median error: 2.1 s
  max error:    6.1 s
```

**What to aim for:** mean < 3 s, max < 8 s is good for broadcast radio.

---

### Step 4 — Grid search (optional)

If the single-evaluation results are not good enough, run a grid search
to find the best parameter combination automatically:

```bash
python radios/analysis/tune.py labels.json --grid
```

The grid is defined in the `PARAM_GRID` dict at the top of `tune.py`.
Edit it to narrow or widen the search ranges before running.  Default
grid has ~243 combinations; with 2 files it takes roughly 30-60 minutes
(the CNN is the bottleneck — it reruns per combination).

**Speeding up the grid search:**
- Label shorter test clips (a few minutes each with 2-3 transitions).
- Reduce the number of values per parameter before running.
- Run with only the parameters you actually want to test — comment out
  the others in `PARAM_GRID`.

After the grid search finishes, it prints the top 5 parameter sets by
mean boundary error.  Copy the winning values into the `# Tuning
parameters` block in `segmenter.py`.

---

## Running the integration tests

The test suite lives in `radios/tests/` and exercises segmentation,
fingerprinting, and the combined pipeline end-to-end.  Test fixtures
(`test_1.mp3` and `test_1.txt`) live in `radios/tests/test_files/`.

All tests are tagged `slow` and `integration` because `test_1.mp3` is a
~60-minute recording and segmentation takes several minutes.

```bash
source env/bin/activate

# Run all integration/analysis tests
python manage.py test radios --tag=integration

# Run each module individually
python manage.py test radios.tests.test_segmentation
python manage.py test radios.tests.test_fingerprinting
python manage.py test radios.tests.test_pipeline

# Exclude slow tests from a fast CI run (runs 0 of these)
python manage.py test radios --exclude-tag=slow

# Fingerprinting requires an AcoustID API key
ACOUSTID_API_KEY=your_key python manage.py test radios.tests.test_fingerprinting
ACOUSTID_API_KEY=your_key python manage.py test radios.tests.test_pipeline

# Fingerprint a single audio file with no labels (whole file as one segment)
TEST_MP3=/path/to/song.mp3 ACOUSTID_API_KEY=your_key \
    python manage.py test radios.tests.test_fingerprinting.FingerprintSingleFileTest
```

### Using a different recording or label file

The test files default to `radios/tests/test_files/test_1.mp3` and
`test_1.txt`, but any recording can be passed via environment variables:

| Variable | Used by | Description |
|---|---|---|
| `TEST_MP3`          | all three test modules                      | Path to the audio file to analyse |
| `TEST_LABELS`       | `test_segmentation`, `test_fingerprinting`, `test_pipeline` | Path to the Audacity label export (tab-separated `.txt`) |
| `ACOUSTID_API_KEY`  | `test_fingerprinting`, `test_pipeline`      | AcoustID API key (fingerprinting is skipped if unset) |
| `SAVE_SEGMENTS`     | `test_segmentation`                         | Set to `1` to write each segment as an MP3 to `media/segments/` |

`TEST_LABELS` is optional for `test_fingerprinting`: if omitted, only
`FingerprintSingleFileTest` runs (fingerprints `TEST_MP3` as one segment).

```bash
# Segmentation only — custom recording
TEST_MP3=/path/to/other.mp3 \
    python manage.py test radios.tests.test_segmentation

# Segmentation + compare against custom ground-truth labels
TEST_MP3=/path/to/other.mp3 TEST_LABELS=/path/to/other.txt \
    python manage.py test radios.tests.test_segmentation

# Segmentation + save each detected segment as a separate MP3
SAVE_SEGMENTS=1 python manage.py test radios.tests.test_segmentation

# Full pipeline on a custom file with fingerprinting
TEST_MP3=/path/to/other.mp3 ACOUSTID_API_KEY=your_key \
    python manage.py test radios.tests.test_pipeline

# Everything at once
TEST_MP3=/path/to/other.mp3 TEST_LABELS=/path/to/other.txt \
    ACOUSTID_API_KEY=your_key \
    python manage.py test radios --tag=integration
```

If `TEST_MP3` does not exist on disk the tests skip automatically with a
clear message rather than failing.

### What each test file does

| Test file | Class | What it checks |
|---|---|---|
| `test_segmentation.py` | `SegmentationTest` | Runs `segment_audio()` and prints a table of segments with start/end/duration (human-readable). Compares against `TEST_LABELS` ground-truth and asserts ≥ 70% of segments match the correct type. With `SAVE_SEGMENTS=1`, saves each segment to `media/segments/` and asserts all files were written. |
| `test_fingerprinting.py` | `FingerprintingTest` | Calls `fingerprint_segment()` on each music segment from the ground-truth labels. Prints artist/title/score for each. Skipped if `ACOUSTID_API_KEY` or `TEST_LABELS` is not set. |
| `test_fingerprinting.py` | `FingerprintSingleFileTest` | Fingerprints `TEST_MP3` as a single whole-file segment — no labels needed. Useful for quickly checking whether a known song is identified. Prints artist/title/score; does not fail if there is no match (the song may not be in the AcoustID database). |
| `test_pipeline.py` | `PipelineTest` | Runs segmentation then fingerprints every music segment. Prints a unified segment table, then a ground-truth comparison table and per-type accuracy breakdown. |

### Output example

```
=== Segmentation: test_1.mp3 ===
  #  type                  start      end  duration
---  --------------------  -------  -------  --------
  0  speech                00:00    01:18      1m 18s
  1  music                 01:18    06:23      5m 05s
...
Summary: 12 segments — speech: 54m 00s | music: 30m 20s

=== Ground Truth vs Predicted ===
GT type     GT start  GT end   GT dur   Best match          Overlap
----------  -------  -------  -------  ------------------  -------
speech      00:00    01:18     1m 18s  speech                  99%  *
music       01:18    06:23     5m 05s  music                   94%  *
...
Correct type: 11/12 (92%)

=== Pipeline: test_1.mp3 ===
  #  type      start    end    duration  fingerprint
---  --------  -------  -------  --------  ------------------------------
  0  speech    00:00    01:18    1m 18s    —
  1  music     01:18    06:23    5m 05s    Radiohead - Creep (score: 0.92)
...
Segments: 12 — speech: 54m 00s | music: 30m 20s
Fingerprinted: 4/6 music segments identified

=== Segmentation vs Ground Truth ===
...
Segmentation accuracy: 11/12 (92%) correct vs ground truth

Per-type accuracy:
  music         4/6 (67%)
  speech        7/6 (100%)
```

---

## Transcription

The transcriber converts speech segments to text. It supports three backends:

| Backend | Engine | Requires | Best for |
|---------|--------|----------|----------|
| `local` | faster-whisper | No API key | Development, offline, privacy-sensitive |
| `openai` | OpenAI Whisper API | `OPENAI_API_KEY` | High accuracy, cloud OK |
| `anthropic` | Claude audio input | `ANTHROPIC_API_KEY` | Multilingual, includes translation |

### Settings

```python
# In aum/settings.py:
TRANSCRIPTION_BACKEND = "local"           # "local", "openai", or "anthropic"
WHISPER_MODEL_SIZE = "medium"             # faster-whisper model (local backend)
WHISPER_DEVICE = "cpu"                    # "cpu" or "cuda" (local backend)
WHISPER_COMPUTE_TYPE = "int8"             # "int8", "float16", "float32"
OPENAI_API_KEY = ""                       # For openai backend
ANTHROPIC_API_KEY = ""                    # For anthropic backend
TRANSCRIPTION_LLM_MODEL = "claude-sonnet-4-20250514"  # Claude model (anthropic backend)
```

### How it works

1. Speech and speech-over-music segments (from the segmenter) are fed to the transcriber.
2. Segment edges are trimmed by 2s to avoid bleed-over from adjacent segments.
3. Segments longer than 10 minutes are split into overlapping sub-chunks.
4. The chosen backend transcribes the audio and detects the language.
5. If the detected language is not English, a translation is generated.
6. Results are stored on `TranscriptionSegment` (`text`, `text_english`, `language`, `confidence`).

### Running

Transcription runs as part of the `analyze_recordings` pipeline:

```bash
python manage.py analyze_recordings --once
```

It activates when the `transcription` stage is enabled (both globally and per-stream).

### Testing

```bash
# Local backend (default, no API key needed)
python manage.py test radios.tests.test_transcription

# Anthropic backend
TRANSCRIPTION_BACKEND=anthropic ANTHROPIC_API_KEY=key \
    python manage.py test radios.tests.test_transcription

# OpenAI backend
TRANSCRIPTION_BACKEND=openai OPENAI_API_KEY=key \
    python manage.py test radios.tests.test_transcription
```

---

## Files

| File                        | Purpose                                                          |
|-----------------------------|------------------------------------------------------------------|
| `segmenter.py`              | Current segmenter (inaSpeechSegmenter CNN)                       |
| `transcriber.py`            | Speech transcription (local/OpenAI/Anthropic backends)           |
| `fingerprinter.py`          | Music identification via AcoustID/Chromaprint                    |
| `tune.py`                   | Evaluate boundary accuracy + grid search over tuning parameters  |
| `audacity_to_labels.py`     | Convert Audacity label export to `labels.json` for `tune.py`     |
| `segmenter_webrtcvad.py`    | Previous segmenter (webrtcvad + 4 Hz modulation). Rollback only. |
| `__init__.py`               | Package init                                                     |

## Rollback to webrtcvad

If you need to revert to the old segmenter:

```bash
cd radios/analysis/
cp segmenter_webrtcvad.py segmenter.py

# Re-enable webrtcvad in requirements.txt:
#   uncomment "webrtcvad" line
#   comment out or remove "inaSpeechSegmenter" line

pip install webrtcvad
```

The old segmenter has no TensorFlow dependency -- only `numpy` and `webrtcvad`.
