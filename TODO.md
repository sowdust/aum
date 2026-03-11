# AUM — Project TODO

- [ ] extract show name and songs from transcript
- [ ] understand why fingerprinting songs doesnt fucking work

## Pipeline Stages
- [x] if key is not set or anything else, do not mark the process as completed.
- 
- [x] record — `record_streams.py` (ffmpeg MP3 chunks)
- [x] segment — `analysis/segmenter.py` (inaSpeechSegmenter CNN)
- [x] fingerprint — `analysis/fingerprinter.py` (pyacoustid + fpcalc)
  - [ ] fingerprint more than one song if longer than 6 minutes
- [x] transcribe — `analysis/transcriber.py` (3 backends: local/openai/anthropic)
- [x] **correct** — `analysis/corrector.py` (LLM post-processing: fix transcription errors + re-translate)
  - [x] 4 backends: local_ollama/cloud_ollama/openai/anthropic
  - [x] Configurable via admin (Transcription Settings → Correction fieldset)
  - [x] Preserves raw text in `text_original` field
  - [ ] If `text_original` storage becomes a problem, consider periodic cleanup or making the field optional
- [ ] **summarize** — `analysis/summarizer.py` (LLM: openai/anthropic)
  - [ ] ChunkSummary per recording (summary text + tags)
  - [ ] DailySummary per radio per day (aggregated summary + tags)
  - [ ] Tag model with full-text search

## REST API
- [ ] Install and configure Django REST Framework
- [ ] Design endpoints: radios, recordings, segments (transcripts), summaries, tags
- [ ] Auth: token or JWT-based
- [ ] Permissions: respect owner/public visibility flags
- [ ] Rate limiting

## Search
- [ ] Full-text search on TranscriptionSegment.text / text_english
- [ ] Tag search across recordings and daily summaries
- [ ] Search API endpoint

## Storage 
- [ ] Decide what to do with mp3 file
- [ ] set on_delete policy to delete them?
- [ ] set some process to archive/convert/remove them

## Cleanup / Commits
- [ ] Review and commit fingerprinter.py changes
- [ ] Commit transcriber.py and test_transcription.py
- [ ] Remove test.py and toremove.py if no longer needed
