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

# Alora Kannada Video Dataset — 839 Clips

A curated collection of **839 short Kannada video clips with synced audio and captions**,
built for audiovisual generative-model training and benchmarking (MiniMax H3 LoRA).

## Dataset Summary

| | |
|---|---|
| Clips | **839** (train **745** / val **47** / test **47**) |
| Language | Kannada (`kn`) |
| Video | **124 frames, 24 fps, 5.1667 s**, H.264 (height ≤ 720, no upscaling) |
| Audio | Synced AAC embedded + PCM16 mono **48 kHz** WAV (248000 samples) |
| Captions | Per-clip `.txt`: scene phrase + Kannada dialogue (machine-generated, unverified) |
| Sources | 497 source recordings across 180 publishers (CC-BY YouTube and Wikimedia Commons) |
| Splits | Grouped 90/5/5 by source recording — no recording spans two splits |

Every MP4 is verified to decode to exactly **124 frames @ 24 fps** with embedded audio
≥ 248000 samples; every WAV holds exactly **248000 samples** mono 48 kHz. Files are
SHA-256 verified, and the split assignment is grouped by `source_id` to avoid leakage.

## Collections

| Collection | Clips | Focus |
|---|---:|---|
| diverse-5s | 37 | Wikimedia/Wikipedia voices, music and film excerpts |
| generation-5s-v2 | 31 | Kundapura comedy, interviews, studio conversation |
| video-5s-v3 | 5 | early five-second pilot |
| video-5s-v4 | 17 | regional comedy, short film, documentary |
| video-5s-v5 | 17 | theatre, Yakshagana, community interviews |
| video-5s-v6 | 93 | mixed interviews, speeches, news |
| video-5s-v7 | 179 | broad search expansion |
| video-5s-v8 | 172 | broad search expansion |
| video-5s-v9 | 226 | largest expansion batch |
| video-5s-v10 | 54 | documentary / farming media |
| video-5s-v11 | 8 | newest batch, strict audit filter |

Sources span movies, songs, folk and Yakshagana performance, interviews, podcasts, TV,
documentaries, speeches, stand-up, news, kids' storytelling, vlogs, and educational content.

## How it was built and audited

Every clip must pass automated gates before it enters the dataset; every record keeps
`generation_training_ready = false` until a human signs off.

- **Build** — CC-BY sources only; a window qualifies with ≥ 80 % speech coverage, < 0.1 %
  clipped samples, reasonable level, and no detected scene cut. Output is normalized to
  124 frames @ 24 fps (H.264 CRF20, height ≤ 720) with a paired 248000-sample 48 kHz WAV.
- **Media integrity** — full decode: frame count, sample count, SHA-256 hashes.
- **Language ID** — SpeechBrain `lang-id-voxlingua107-ecapa`, calibrated on Kannada + English
  references.
- **Utterance boundary** — Silero VAD (via faster-whisper) flags speech within 100 ms of a clip edge.
- **Lip-sync estimate** — SyncNet v2 + S3FD face tracks with a PySceneDetect scene-cut gate.
- **Captions** — faster-whisper large-v3-turbo, Kannada forced, with a Kannada-script check.

**Selection history**

| Stage | Clips |
|---|---:|
| Documented total across collections | 1000 |
| − earlier exclusion pass (bad/missing script, multi-flag, unverified language, empty caption) | −169 |
| = curated baseline | 831 |
| + `video-5s-v11` appended (316 built, 29 non-Kannada dropped) | +287 |
| − v11 audit filter (language-uncertain, boundary **and** sync flagged, no Kannada script) | −279 |
| **= final dataset** | **839** |

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
- Automated language / lip-sync / boundary results are screening estimates, not
  certification. Short or accented speech and small or profile faces reduce reliability.
- Source copyright licenses (CC-BY variants, recorded per clip) do **not** by themselves
  grant permission to build identifiable voice or likeness models.
- No claim of dialect balance or speaker representativeness; child and grandparent
  voice coverage remain gaps. Different recordings may feature the same unknown speaker.

## Provenance

Excerpts from CC-BY YouTube and Wikimedia Commons recordings, trimmed to uniform
windows, re-encoded, and audio-resampled. Retain source credits and license links;
no endorsement is implied. Evaluation expansions used speech-coverage and scene-cut
filters; see per-collection validation reports where present.

## Legacy ASR seed

The repository also hosts the original 114-pair Kannada audio/transcript seed
(`train-*.parquet`, `statistics.json`) used for speech-recognition work; it is unchanged
and separate from the video dataset above.