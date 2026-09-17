#!/usr/bin/env bash
# End-to-end benchmark orchestration. Assumes AGENT.md preconditions are met.
set -euo pipefail
cd "$(dirname "$0")/.."

export AI_TOOLKIT_DIR="${AI_TOOLKIT_DIR:-$HOME/ai-toolkit}"
[ -d "$AI_TOOLKIT_DIR" ] || { git clone https://github.com/ostris/ai-toolkit "$AI_TOOLKIT_DIR"; }
( cd "$AI_TOOLKIT_DIR" && git checkout f56b5a1d405f819c74724228564e99982624c186 )

python dataset/download_dataset.py

# Stage 1: 39-frame smoke (5 clips)  | Stage 2: 73-frame (10) | Stage 3: 107-frame benchmark (50)
for spec in "39 5" "73 10" "107 50"; do
  set -- $spec
  python dataset/make_107_crops.py --frames "$1" --n "$2"
  python scripts/monitor.py --out "monitor_$1.csv" &
  MON=$!
  echo "=== RUN: $1 frames — configure the AI Toolkit job from configs/aitoolkit-job-$1.json ==="
  echo "    (launch the job here; 100 optimizer steps at 107; audio enabled; auto frame count OFF)"
  # <launch AI Toolkit job with the dataset dir data/benchmark-$1>
  kill $MON || true
done

echo "Fill RESULTS.json from the monitor CSVs and trainer logs, then report go/no-go per AGENT.md section 5."