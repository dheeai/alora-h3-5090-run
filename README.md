# Alora Kannada — MiniMax H3 RTX 5090 run-kit

Everything needed to run the H3 audiovisual benchmark (and, if it passes, the capped pilot)
on a local RTX 5090 against the prepared 1000-clip Kannada dataset.

- **Dataset (private):** `aaron1z/alora-kannada-fleurs-v1` on Hugging Face — 1000 clips,
  124 frames / 24 fps / 5.1667 s, synced audio, Kannada captions, grouped 90/5/5 split.
- **Trainer:** AI Toolkit @ `f56b5a1d405f819c74724228564e99982624c186` (SimpleTuner 32 GB preset included as an alternate path).
- **Instructions:** agents follow `AGENT.md` start to finish. Humans: the short version below.

## Quickstart

```bash
export HF_TOKEN=<read-scope token for aaron1z/alora-kannada-fleurs-v1>
pip install -r requirements.txt
python dataset/download_dataset.py            # -> ./data/alora-kannada-fleurs-v1/
python dataset/make_107_crops.py --n 50       # -> ./data/benchmark-107/
bash scripts/run_benchmark.sh                 # 39-frame smoke, then 107-frame 100 steps
```

Recorded measurements land in `RESULTS.json` (template provided) plus `monitor_*.csv`.

## What success means

100 optimizer steps at 107 frames with **audio enabled**, no OOM/swapping, validation and
checkpoint reload OK, and a nonzero audio loss mask. Anything else is a fail — report it as such.

## Repo layout

See `AGENT.md` for the full file map and rules. Data and checkpoints are gitignored; nothing in
this repo re-licenses the dataset (CC-BY sources; attribution lives in the HF repo).
