# Alora Kannada — MiniMax H3 on RTX 5090

Run-kit for the MiniMax **H3** audiovisual benchmark and capped LoRA pilot on a single
**RTX 5090 (32 GB)**, using the curated **Kannada** video dataset hosted on Hugging Face.

It is deliberately small and explicit: download → verify → benchmark → record → go / no-go.
Nothing here trains a production model by default, and nothing here modifies the dataset.

- **Dataset (private):** [`aaron1z/alora-kannada-fleurs-v1`](https://huggingface.co/datasets/aaron1z/alora-kannada-fleurs-v1)
- **Trainer:** AI Toolkit pinned at `f56b5a1d405f819c74724228564e99982624c186` (SimpleTuner 32 GB preset included as an alternate path)
- **Agents:** follow [`AGENT.md`](./AGENT.md) end to end. Humans: read on.

---

## Dataset at a glance

| | |
|---|---|
| **Clips** | **839** |
| **Splits** (grouped 90/5/5 by source recording) | train **745** · val **47** · test **47** |
| **Language** | Kannada (`kn`) |
| **Video** | **124 frames · 24 fps · 5.1667 s**, H.264, height ≤ 720 (no upscaling) |
| **Audio** | AAC embedded in the MP4 **+** PCM16 mono **48 kHz** WAV (248000 samples) |
| **Captions** | Per-clip `.txt` — scene phrase + machine Kannada dialogue |
| **Coverage** | 497 source recordings across 180 publishers |
| **Licenses** | CC-BY-3.0 (827), CC-BY-4.0 (9), CC-BY-SA-4.0 (3) |

### Collections

| Collection | Clips | Focus |
|---|---:|---|
| `diverse-5s` | 37 | Wikimedia/Wikipedia voices + music and film excerpts |
| `generation-5s-v2` | 31 | Kundapura comedy, interviews, studio conversation |
| `video-5s-v3` | 5 | early five-second pilot |
| `video-5s-v4` | 17 | regional comedy, short film, documentary |
| `video-5s-v5` | 17 | theatre, Yakshagana, community interviews |
| `video-5s-v6` | 93 | mixed interviews, speeches, news |
| `video-5s-v7` | 179 | broad search expansion |
| `video-5s-v8` | 172 | broad search expansion |
| `video-5s-v9` | 226 | largest expansion batch |
| `video-5s-v10` | 54 | documentary / farming media |
| `video-5s-v11` | 8 | newest batch, strict audit filter |

Each collection directory holds `<id>.mp4`, `<id>.wav`, `<id>.txt`, plus `metadata.jsonl`
(and, where produced, `sources.json`, `build-report.json`, `audit-summary.json`). The
repository root holds the unified `metadata.jsonl` and the split lists
`train.jsonl` / `val.jsonl` / `test.jsonl`.

---

## How the dataset was built and audited

Every clip passes automated gates before it can enter the dataset. All of it is
reproducible from the `work/` pipeline, and every record keeps
`generation_training_ready = false` until a human signs off.

**Build (`build.py`)**
- Source must be CC-BY. Only windows with **≥ 80 % speech coverage**, < 0.1 % clipped
  samples, reasonable level, and **no detected scene cut** are eligible.
- Output is normalized: 124 frames @ 24 fps, H.264 CRF20, height capped at 720 without
  upscaling, paired 248000-sample 48 kHz WAV (also muxed as the MP4 audio track).
- Recorded per clip: source URL/creator/license, category, timestamps, speech stats,
  and SHA-256 hashes of both files.
- Splits are grouped **90/5/5 by `source_id`**, so no recording spans two splits.

**Audits (per clip)**
| Check | Method | Output |
|---|---|---|
| Media integrity | PyAV full decode | `nb_frames == 124`, embedded audio ≥ 248000 samples |
| Language ID | SpeechBrain `lang-id-voxlingua107-ecapa` (calibrated on Kannada + English refs) | top language + score |
| Utterance boundary | Silero VAD via faster-whisper | speech in first/last 100 ms flags |
| Lip-sync estimate | SyncNet v2 + S3FD face tracks, PySceneDetect cut gate | best-track confidence + offset |
| Caption | faster-whisper large-v3-turbo (Kannada forced, VAD on) | `transcript`, `caption`, Kannada-script ratio |
| File integrity | SHA-256 | hash match on every file |

**Selection history**

| Stage | Clips | Notes |
|---|---:|---|
| Documented total across collections | **1000** | aggregate before quality filtering |
| − Earlier exclusion pass | **−169** | bad/missing Kannada script (82), unresolved multi-flag (73), unverified language (12), empty caption (2) |
| = Curated baseline | **831** | |
| + `video-5s-v11` appended | **+287** | 316 built, 29 dropped as confidently non-Kannada |
| − v11 audit filter | **−279** | language-uncertain 137, boundary **and** sync flagged 268, no Kannada script 59 (union) |
| **= Final dataset** | **839** | train 745 · val 47 · test 47 |

The v11 yield was intentionally strict: only clips that passed **all** language, boundary,
sync and script gates were kept. Failed clips and their reasons are recorded per collection,
not silently discarded.

> Note on repo contents: the collection directories now contain **exactly the 839 indexed
> clips** — every `<id>.mp4` / `<id>.wav` / `<id>.txt` is referenced by `metadata.jsonl` and a
> split list. The Hugging Face repo additionally hosts a separate **legacy 114-pair Kannada
> ASR seed** (`train`/`validation`/`test-*.parquet`, `statistics.json`, `manifest.json`,
> `sources.json`) that is unrelated to the video dataset. The root also carries
> `checksums.json` (SHA-256 for every file) and `exclusions.jsonl` (every removed clip with
> its reason and stage). Release tagged **`v839-2026-09-21`** on GitHub and Hugging Face.

**What automated checks do *not* prove**
- Captions are approximate machine output, not verified dialogue.
- Language / lip-sync / boundary results are screening estimates, not certification; short
  or accented speech and small/profile faces reduce reliability.
- Source copyright (CC-BY) does not by itself grant permission for identifiable voice or
  likeness models. Human review of language, active speaker, lip sync, boundaries, music,
  demographics, and generation permissions all remain **pending**.

---

## Quickstart

```bash
export HF_TOKEN=<read-scope token for aaron1z/alora-kannada-fleurs-v1>
pip install -r requirements.txt

python dataset/download_dataset.py            # -> ./data/alora-kannada-fleurs-v1/
python dataset/make_107_crops.py --n 50       # -> ./data/benchmark-107/
bash scripts/run_benchmark.sh                 # 39-frame smoke, then 107-frame 100 steps
```

Recorded measurements land in `RESULTS.json` (template provided) plus `monitor_*.csv`.

### Pre-flight (must all pass)

- every MP4 decodes to exactly **124 frames @ 24 fps** with embedded audio ≥ 248000 samples;
- every WAV is 248000 samples, mono, 48 kHz;
- every training clip has a non-empty Kannada-script `.txt` caption;
- `train` / `val` / `test` share no `source_id`.

H3 frame counts follow **17n+5**. Clips are natively **124 frames**; the benchmark runs at
**107 frames** (first 107 frames + first 214000 audio samples, cropped together — never
audio or video alone).

---

## What success means

100 optimizer steps at **107 frames with audio enabled**, no OOM or sustained swapping,
validation generation and checkpoint reload succeed, and the **audio loss mask is nonzero**.
Anything else is a fail — report it as one.

Only on an explicit pass, run the capped **Phase 2** pilot on the full dataset
(`train.jsonl`, `frames: 124`, `max_train_steps: 2000`), keeping `test.jsonl`
untouched until checkpoints are chosen. Full instructions and thresholds are in
[`AGENT.md`](./AGENT.md).

---

## Repository layout

| Path | Purpose |
|---|---|
| `README.md` | this human-facing overview |
| `README-HF.md` | dataset card copy for the Hugging Face repo |
| `AGENT.md` | end-to-end agent runbook and rules |
| `dataset/download_dataset.py` | pull the private dataset from HF |
| `dataset/make_107_crops.py` | build 39 / 73 / 107-frame benchmark crops + captions |
| `dataset/loader_config.json` | dataset layout spec for the trainer |
| `dataset/eval_prompts.kn.txt` | 50 unseen Kannada prompts (from the held-out test split) |
| `scripts/run_benchmark.sh` | end-to-end orchestration |
| `scripts/monitor.py` | VRAM / RAM sampler |
| `configs/` | AI Toolkit job templates + SimpleTuner 32 GB preset (alternate path) |
| `docs/` | full benchmark instructions, SimpleTuner H3 guide, license notes |
| `LICENSE` | MIT for this run-kit code (dataset media is **not** covered) |
| `CITATION.cff` | citation metadata for the dataset / run-kit |

Data, checkpoints and run outputs are gitignored.

---

## Provenance & license

Excerpts are taken from CC-BY YouTube and Wikimedia Commons recordings, trimmed to uniform
five-second windows, re-encoded, and audio-resampled. All clips carry their source URL,
creator and license; retain attribution and license links. No endorsement is implied, and
**nothing in this repository re-licenses the dataset**. The approval of a source license
does not extend to identifiable voice or likeness generation.