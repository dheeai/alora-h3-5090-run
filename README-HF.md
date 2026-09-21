---
language:
- kn
license:
- cc-by-3.0
- cc-by-4.0
- cc-by-sa-4.0
configs:
- config_name: default
  data_files:
  - split: train
    path: train-*.parquet
  - split: validation
    path: validation-*.parquet
  - split: test
    path: test-*.parquet
task_categories:
- automatic-speech-recognition
- text-to-video
---

# Alora Kannada Video Dataset — 1000 Clips

A curated collection of **1000 short Kannada video clips with synced audio and captions**,
built for audiovisual generative-model training and benchmarking (MiniMax H3 LoRA).

## Dataset Summary

| | |
|---|---|
| Clips | **1000** (train 900 / val 50 / test 50) |
| Language | Kannada (`kn`) |
| Video | **124 frames, 24 fps, 5.1667 s**, H.264 (height ≤ 720, no upscaling) |
| Audio | Synced AAC embedded + PCM16 mono **48 kHz** WAV (248000 samples) |
| Captions | Per-clip `.txt`: scene phrase + Kannada dialogue (machine-generated, unverified) |
| Sources | CC-BY YouTube and Wikimedia Commons recordings across 10 collections |
| Splits | Grouped 90/5/5 by source recording — no recording spans two splits |

Every MP4 is verified to decode to exactly 124 frames with embedded audio ≥ 248000 samples;
every WAV holds exactly 248000 samples. A language-identification gate removed 59
confidently non-Kannada clips (listed in `dropped-non-kannada.jsonl`).

## Collections

| Collection | Clips |
|---|---:|
| diverse-5s | 69 |
| generation-5s-v2 | 58 |
| video-5s-v3 | 17 |
| video-5s-v4 | 28 |
| video-5s-v5 | 35 |
| video-5s-v6 | 96 |
| video-5s-v7 | 195 |
| video-5s-v8 | 190 |
| video-5s-v9 | 253 |
| video-5s-v10 | 59 |

Sources span movies, songs, folk and Yakshagana performance, interviews, podcasts, TV,
documentaries, speeches, stand-up, news, kids' storytelling, vlogs, and educational content.

## Dataset Structure

Each collection directory contains `<id>.mp4`, `<id>.wav`, `<id>.txt` plus `metadata.jsonl`
and `sources.json`. The repository root holds the unified `metadata.jsonl` and the split
lists `train.jsonl`, `val.jsonl`, `test.jsonl` with fields
`id, collection, video_file, audio_file, caption, source_id`.

Record fields include fps/frames/resolution, caption, transcript, category, source
title/URL/creator, license, leakage group, split, and SHA-256 hashes. Keep `test.jsonl`
untouched until checkpoints are chosen.

## Usage

The video files are fetched directly (they are not part of the Parquet config):

```python
from huggingface_hub import snapshot_download
path = snapshot_download('aaron1z/alora-kannada-fleurs-v1')  # read token required
```

H3 training uses frame counts of 17n+5: clips are natively **124 frames**; benchmark at
**107 frames** (crop first 107 frames + first 214000 audio samples — video and audio
together, never separately).

```python
from datasets import load_dataset
ds = load_dataset('aaron1z/alora-kannada-fleurs-v1')  # legacy 114-pair ASR seed (Parquet)
```

## Considerations and Limitations

- Captions are machine-generated (faster-whisper large-v3-turbo, Kannada forced) and
  **not human verified**; treat dialogue text as approximate.
- `generation_training_ready=false` on every record. Language, active speaker, lip
  synchronization, music/overlap, utterance boundaries, demographics, and generation
  permissions all remain pending human review.
- Source copyright licenses (CC-BY variants, recorded per clip) do **not** by themselves
  grant permission to build identifiable voice or likeness models.
- No claim of dialect balance or speaker representativeness; child and grandparent
  voice coverage remain gaps. Different recordings may feature the same unknown speaker.

## Provenance

Excerpts from CC-BY YouTube and Wikimedia Commons recordings, trimmed to uniform
windows, re-encoded, and audio-resampled. Retain source credits and license links;
no endorsement is implied. Evaluation expansions were capped at two clips per recording
(older collections) with speech-coverage and scene-cut filters; see per-collection
validation reports where present.

## Legacy ASR seed

The repository also hosts the original 114-pair Kannada audio/transcript seed
(`train-*.parquet`, `statistics.json`) used for speech-recognition work; it is unchanged
and separate from the video dataset above.
