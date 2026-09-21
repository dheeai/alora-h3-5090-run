# AGENT.md — Run the MiniMax H3 Kannada benchmark on the local RTX 5090

You are an autonomous agent. Your job: download the prepared Kannada dataset, verify it loads,
run the H3 audiovisual benchmark on this machine, record measurements, and report go / no-go.
**Benchmark only — do not start long training runs, do not rent cloud GPUs, do not modify the dataset.**

## 0. Machine and access

- Hardware: RTX 5090 (32 GB VRAM) + 64 GB system RAM. Native Linux preferred; WSL2 acceptable only
  if its memory allocation is verified.
- Dataset: **private HF dataset `aaron1z/alora-kannada-fleurs-v1`** (839 Kannada clips).
  Authenticate with a read-scope HF token: `export HF_TOKEN=...` (ask the owner; the token is NOT in this repo).
- Trainer: **AI Toolkit, pinned revision `f56b5a1d405f819c74724228564e99982624c186`** (2026-09-10).
  Use that revision's dependency stack; do not copy a CUDA recipe from another trainer.
- Model files: MiniMax H3 default FL2VA pruned partition, Comfy-Org **INT8 ConvRot** transformer +
  **NVFP4** text encoder (~43 GB on disk — file sizes, not peak RAM).

## 1. Get the dataset

```bash
python dataset/download_dataset.py          # -> ./data/alora-kannada-fleurs-v1/
```

What you get (per collection `diverse-5s`, `generation-5s-v2`, `video-5s-v3` … `video-5s-v11`):
- `<id>.mp4` — **124 frames, 24 fps, 5.1667 s, H.264, synced AAC audio embedded** (audio track is what the loader reads)
- `<id>.wav` — PCM16 mono 48 kHz archival copy
- `<id>.txt` — caption: scene phrase + **exact Kannada dialogue**
- `metadata.jsonl`, `train.jsonl` / `val.jsonl` / `test.jsonl` (grouped 90/5/5 split, no source overlap)

Pre-flight checks (must all pass before training):
- every MP4 decodes to exactly **124 frames @ 24 fps** and its embedded audio is >= 248000 samples @ 48 kHz;
- every training clip has a non-empty `.txt` caption in Kannada script;
- `train/val/test` share no `source_id`.

## 2. Prepare benchmark crops

H3 uses frame counts of **17n+5**. Training clips are 124 frames. The benchmark runs at **107**:

```bash
python dataset/make_107_crops.py --n 50      # -> ./data/benchmark-107/  (50 clips, 107 frames + captions)
```

- 107 frames = 4.458 s fits inside the 124-frame clips; the script crops video + audio together (214000-sample WAV).
- Keep audio enabled throughout. **A run with audio disabled does not count.**

## 3. Run the benchmark

1. **39-frame smoke test** (from the same script, `--frames 39`): loading, caching, audio loss nonzero.
2. **73-frame check.**
3. **107-frame, 100 optimizer steps** — this is the number that matters. Separate warmup from steady state.
4. Validation generation + **checkpoint save/reload**.

Suggested starting settings (record whatever you actually use):
rank 16 LoRA (keep the H3 `adaln_proj` exclusion), batch 1, grad-accum 1, gradient checkpointing on,
LR 6e-5, `do_audio: true` (audio loss multiplier 1), low-VRAM mode + disk caches + layer offloading,
~512 resolution bucket, keep AI Toolkit's training-adapter + contrastive-guidance defaults
(do **not** disable both). Turn **automatic frame counting off**; use explicit synchronized crops;
disable whole-video shrinking.

## 4. Record (RESULTS.json)

For each stage (39 / 73 / 107) and for loading / caching / training / validation separately:
peak VRAM, peak system RAM, steady seconds/step, cache time, sample time, checkpoint save+reload
success, any OOM / Triton error / swapping, and confirmation that **audio loss is nonzero**.
`scripts/monitor.py` polls nvidia-smi + /proc/meminfo into a CSV — run it alongside training.

## 5. Pass / fail (all must hold)

1. Real audio targets; nonzero audio mask on every sample.
2. 100 steps at 107 frames without OOM or sustained swapping.
3. Validation generation + checkpoint reload succeed.
4. Losses finite; adapter effect reproduces after reload.

Stop and report failure immediately if loading fails, audio training is zero, full-window validation
fails, or adapter reload fails. Do not paper over these.

## 6. After the benchmark

- Pass → run the Phase 2 capped pilot below. Only run it on an explicit **pass**.
- Fail → report which resource bound it, and what to change first (spatial resolution, then caching/offload).
- Deliver: RESULTS.json, config used, monitor CSVs, raw logs, and a one-paragraph go/no-go.

## 7. Phase 2 — capped pilot on the full 839 clips

Run **only after** the 107-frame benchmark passes. No separate download needed — train on
`data/alora-kannada-fleurs-v1/` using `train.jsonl` (745 clips), validate on `val.jsonl` (47),
and keep `test.jsonl` (47) **completely untouched** until checkpoints are chosen.

- Same settings as the benchmark, except `frames: 124` (the clips' native grid) and
  `max_train_steps: 2000`.
- Save a checkpoint **every 100 steps**; generate and review samples **every 250 steps**.
- At ~745 training clips and batch size 1, 2000 steps ≈ 2.7 passes. This is a budget, not a convergence guarantee.
- Compare checkpoints against each other and the base model on `dataset/eval_prompts.kn.txt`
  (50 held-out Kannada prompts, 2 seeds each): exact dialogue, intelligibility, pronunciation,
  lip sync, visual quality, unwanted repetition. Pick the best checkpoint — never assume the last one is best.
- Then test the chosen adapter in the intended H3 inference runtime before calling it done.

Do not exceed 2000 steps, do not touch the test split early, and do not start this phase on a failed benchmark.

## Do-not list

- Do not rent cloud GPUs or book multi-day jobs.
- Do not retrain/modify/re-upload the dataset.
- Do not disable audio to fit memory and report success.
- Do not present assumed step times as measured.
- Do not extrapolate 107-frame cost from 39-frame results.

## Files in this repo

| Path | Purpose |
|---|---|
| `README.md` | human-facing overview |
| `README-HF.md` | dataset card copy for the Hugging Face repo |
| `dataset/download_dataset.py` | pull the private dataset from HF |
| `dataset/make_107_crops.py` | build 39/73/107-frame benchmark crops + captions |
| `dataset/loader_config.json` | dataset layout spec for the trainer |
| `dataset/eval_prompts.kn.txt` | 50 unseen Kannada prompts (from the held-out test split) |
| `scripts/run_benchmark.sh` | end-to-end orchestration |
| `scripts/monitor.py` | VRAM/RAM sampler |
| `configs/` | AI Toolkit job templates + inspected SimpleTuner 32 GB preset (alternate path) |
| `docs/` | full benchmark instructions, SimpleTuner H3 guide, license notes |

