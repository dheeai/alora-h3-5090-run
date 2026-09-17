# Agent Instructions — MiniMax H3 Kannada LoRA Benchmark on a Local RTX 5090 (AI Toolkit)

Purpose: run a **local feasibility / speed benchmark** of MiniMax H3 joint video+audio LoRA training on an existing **RTX 5090 (32 GB VRAM, 64 GB system RAM)** using **AI Toolkit**, before any cloud rental or dataset scale-up.

Scope: **benchmark only.** Do not start a long training run, do not rent cloud GPUs, do not expand the dataset, and do not claim Kannada quality. Produce measured numbers and a go / no-go decision.

Derived from:
- `MiniMax-H3-Kannada-Training-Plan.txt`
- `AI-Toolkit-RTX-5090-Evaluation.md`
- (background) `MiniMax-H3-Kannada-training-report.md`

---

## 0. Hard success definition

The benchmark **passes only if all of the following hold** on the intended audiovisual configuration:

1. Data loads with **real audio targets**, and the **audio loss mask is nonzero** for every training sample.
2. Training runs a full **100 optimizer steps** at the **intended final frame count (107)** without OOM, crash, or sustained system swapping.
3. Validation generation runs and a **checkpoint saves and reloads** successfully.
4. Losses are finite, and the saved adapter has a reproducible effect after reload.

A generation-only run, an image training run, or any run with **audio disabled does NOT count as success.**

Stop and report failure immediately if: loading fails, audio training is zero, full-window validation fails, or adapter reload fails. Do not paper over these.

---

## 1. Environment preconditions (verify before training)

- GPU: RTX 5090, 32 GB VRAM. System RAM: 64 GB. These are separate; 64 GB RAM does **not** add VRAM.
- Prefer **native Linux**. WSL2 is acceptable only if its memory allocation is checked explicitly.
- AI Toolkit pinned revision: **`f56b5a1d405f819c74724228564e99982624c186`** (committed 2026-09-10). Use the dependency versions of that revision; do not copy a CUDA recipe from another trainer.
- Expected model files: ~**43 GB** total (~21 GB quantized transformer, ~16 GB text encoder). These are file sizes, **not** peak RAM/VRAM.
- Known risk: an open issue (**#1017**) reports 5090 text-embedding prep exceeding 120 GB RAM with a Triton compile error in one configuration. Treat memory behavior as unverified; measure it.

Record: OS, exact AI Toolkit commit, CUDA/driver, Python, torch, GPU driver, and the effective WSL2 RAM limit if applicable.

---

## 2. Dataset preparation (20–50 clips only)

Use **20 to 50 reviewed existing clips**. Keep separate unseen clips for evaluation. Do not audit or upload the full private collection as part of this benchmark.

- Provide MP4 files that **contain the synchronized audio track** plus matching **caption text files**. A separate WAV may remain archival, but the inspected loader extracts audio **from the video** — a silent MP4 with a side WAV does not demonstrate audio training.
- H3 uses **24 fps** and frame counts of **17n + 5**: `39` (1.625 s), `73` (3.042 s), `107` (4.458 s), `124` (5.167 s).
- The 5-second / 120-frame clips **do not sit on the H3 grid**. Either use a reviewed **107-frame** window that contains the target phrase, or re-extract a genuine **124-frame** segment from the original. Do not slow, stretch, or pad speech to force a duration.
- For a controlled benchmark: **turn automatic frame counting off** and use **explicit synchronized crops**. **Disable whole-video shrinking** (the loader can otherwise spread frames across the whole source and retime audio; tail-trim preserves real time but still verify).
- Captions: describe scene/action, include the **exact Kannada dialogue in original script**, plus relevant speaker and sound/music info.
- Split by source recording and known speaker to reduce leakage. Keep any evaluation clips strictly separate.

Pre-flight validation (must pass before training): every sample has embedded audio, a nonzero audio mask, finite caption text, and correct frame count for its target length.

---

## 3. Training configuration (starting point — test, do not assume)

| Setting | Starting choice |
|---|---|
| Model | MiniMax H3 default **FL2VA** pruned partition, text-to-video training mode |
| Weights | Default Comfy-Org **INT8 ConvRot** transformer and **NVFP4** text encoder |
| Adapter | **Rank 16** LoRA; retain H3 exclusion for `adaln_proj` |
| Batch | **1**; gradient accumulation 1; gradient checkpointing **on** |
| Learning rate | **0.00006** |
| Audio | `do_audio: true`, audio loss multiplier 1; **confirm real audio targets and logged audio loss** |
| Memory | Low-VRAM mode, disk caches, layer offloading; tune offload vs. RAM and throughput |
| Frames | **39** first (feasibility), then **73**, then intended **107** |
| Resolution | Start near the **512** bucket, preserving aspect ratio; check crops |
| Evaluation | One small sample + checkpoint reload; avoid large default previews during memory testing |
| Distillation | Retain AI Toolkit training adapter **and** contrastive guidance default. Do not disable both just to make speed look better. |

If you change offloading, distillation handling, resolution, or frame count, **record that alongside every timing result** — it changes the workload.

---

## 4. Benchmark procedure

1. **39-frame integration check.** Confirm load, cache, audio training, loss sanity.
2. **73-frame check.** Intermediate memory/throughput point.
3. **107-frame benchmark — 100 optimizer steps.** This is the number that matters.
   - Separate **warmup** steps from **steady-state** throughput.
   - Keep **audio enabled** throughout.
4. **Validation generation** at the benchmark length.
5. **Checkpoint save and reload**; verify the adapter effect reproduces.

Do not extrapolate a 107-frame cost from a 39-frame result. Benchmark the intended final frame count.

---

## 5. Measurements to record (required)

Per stage (39, 73, 107) and for caching / training / validation separately:

- Peak **VRAM**
- Peak **system RAM**
- **Steady-state seconds per optimizer step** (after warmup) and warmup step times
- **Cache time** (text/latent)
- **Sample/generation time**
- **Checkpoint save time** and **reload success/failure**
- Any OOM, Triton/compile error, or sustained swapping event, with context
- Confirmation of **nonzero audio loss** and audio-parameter participation
- Files: raw logs, config used, and dataset list/hashes

Reference arithmetic only (not a forecast): 100 steps = 25 min @15 s/step, 50 min @30 s, 1 h 40 m @60 s, 3 h 20 m @120 s. Caching, validation, and checkpointing are additional.

Local electricity (optional): `kW (whole system) × hours × tariff/kWh`. Assumed values (0.7 kW, ₹10/kWh) are illustrations, not measurements. Add prep/eval energy; exclude hardware cost.

---

## 6. Decision rules

- **Proceed locally** only if the 107-frame audiovisual run completes training, saving, and sampling checks without memory failure or sustained swapping.
- If it does not fit: first reduce **spatial resolution**, then optimize **staged caching / offloading**. Shortening every dialogue is a compromise; cloud training remains the fallback if longer speech must be preserved.
- The 39-frame check **cannot** certify the 107-frame setting.
- Only after a passing benchmark, consider a capped follow-on run (max 2,000 steps, save every 100, review samples every 250). If nothing changes, the first 100 steps may count toward 1,000; if frames/dataset/objective change, treat it as a new pilot.
- The artifact is an **H3 LoRA**; validate it in a compatible H3 inference runtime before calling deployment done. It is not portable to other architectures.

---

## 7. Explicit do-not list

- Do not rent cloud GPUs or book multi-day jobs.
- Do not expand the dataset or upload the full collection.
- Do not start the 2,000-step run before the benchmark passes.
- Do not disable audio to fit memory and still report success.
- Do not count inference-only speedups as training timings.
- Do not present assumed step times as measured results.
- Do not normalize the original Kannada script away in captions.

---

## 8. Deliverables (single report back)

1. Environment + revision summary.
2. Dataset summary: clip count, frame lengths used, audio/mask verification.
3. Configuration actually used (full table, flags included).
4. Measurement tables for 39 / 73 / 107 frames.
5. Pass/fail per Section 0, with raw evidence.
6. Go / no-go recommendation and, if go, a costed estimate using **measured** step time.

---

## Sources

- `outputs/h3-training-report/MiniMax-H3-Kannada-Training-Plan.txt` — benchmark at 39 then 107 frames, 100 steps, memory/caching/validation/checkpoint/reload, $50 feasibility cap, 5090 not verified for 5 s audiovisual.
- `outputs/h3-training-report/AI-Toolkit-RTX-5090-Evaluation.md` — AI Toolkit revision, 100-step local pilot, dataset rules (20–50 clips, embedded audio + captions, 24 fps / 17n+5), auto-frame-count off, disable whole-video shrinking, metrics, decision rules.
- `outputs/h3-training-report/MiniMax-H3-Kannada-training-report.md` — 32 GB preset context, 90/5/5 splits, evaluation protocol, stop rules.
