# AUM — Project TODO

- [x] extract show name and songs from transcript (via BroadcastDaySummary / ShowBlock)

## Segmentation / Fingerprinting Quality
- [ ] **Segment boundary accuracy**: identified songs often include adjacent speech/noise — the player exposes this clearly. Need to investigate whether the issue is in inaSpeechSegmenter's output boundaries (possibly off by a few seconds) or in how fingerprinting maps back to those boundaries. Goal: when a song is played via the Songs Played player, only music should be heard — no presenter talk bleeding in.

## Pipeline Stages
- [ ] if key is not set or anything else, do not mark the process as completed.
- 
- [x] record — `record_streams.py` (ffmpeg MP3 chunks)
- [x] segment — `analysis/segmenter.py` (inaSpeechSegmenter CNN)
- [x] fingerprint — `analysis/fingerprinter.py` (pyacoustid + fpcalc)
  - [x] fingerprint more than one song if longer than 6 minutes
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

## Broadcast Day Summarization (future)
- [ ] REST API: `BroadcastDaySummarySerializer`, `ShowBlockSerializer`, ViewSet at `/api/v1/radios/<slug>/broadcast-days/`
- [ ] Web scraping: fetch radio website schedule to cross-reference show names (controlled by `DailySummarizationSettings.enable_web_scraping`)
- [ ] FTS integration: add broadcast day summaries to full-text search index

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


## Production

 - [ ] Batch scheduling of text to speech via GPU
 - [ ] Text to speech BEFORE encoding?
 - [ ] Compile tensor flow according to the machine