#!/usr/bin/env bash
# Table 4.16 (0.5B rows) and the 0.5B direct-answer scores of Table 4.13, evaluation only: the
# untrained bridge (theta_0) and the 0.5B AO bridges trained on the laptop (4k items, seeds
# 0-2: results/runs/laptop/mac_extra/B_answer_only_n4000_s*_mps; projector.pt is a release
# asset) on MMStar (letter scoring) and A-OKVQA direct answers (score min(1, n/3)), real and
# swapped images. The DA-target rows come from run_mac_AB.sh (job da_seeds), the 7B rows from
# run_mac_final.sh. Each job writes results/runs/laptop/mac_transfer/<job>/. Run from the repo
# root:
#     bash experiments/laptop/run_mac_transfer.sh
# The first run downloads MMStar; finished jobs are skipped on a re-run.
# Progress: tail -f results/runs/laptop/mac_transfer/queue.log
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH=src
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${MPS_HIGH:-0.7}" PYTORCH_MPS_LOW_WATERMARK_RATIO="${MPS_LOW:-0.5}"
OUT=results/runs/laptop/mac_transfer; mkdir -p "$OUT"
LOG="$OUT/queue.log"
BRIDGE=configs/ov_qwen2_05b_frozen.json
"$PY" -c "import datasets" 2>/dev/null || "$PY" -m pip install -q datasets

run() {  # run <name> <benchmark> [extra args...]
  local name=$1 bench=$2; shift 2
  local dir="$OUT/$name"
  if [ -f "$dir/summary.md" ]; then echo "skip $name (done)" | tee -a "$LOG"; return; fi
  echo "=== $(date -u +%FT%TZ) $name" | tee -a "$LOG"
  "$PY" -m cmr.runners.transfer --benchmark "$bench" --model "bridge:$BRIDGE" \
    --conditions real,swapped --rotations 1 --device mps --dtype float32 --threads 4 \
    --out "$dir" "$@" >> "$LOG" 2>&1 || echo "FAILED $name (see $LOG)" | tee -a "$LOG"
}

DA=(--aokvqa-dir data/raw/aokvqa --coco-dir data/raw/coco --split val --max-new-tokens 10)
# MMStar first (needs the download), then direct answer
run mmstar_theta0 mmstar
for s in 0 1 2; do
  run "mmstar_ao_s$s" mmstar --projector "results/runs/laptop/mac_extra/B_answer_only_n4000_s${s}_mps/projector.pt"
done
run da_theta0 aokvqa-da "${DA[@]}"
for s in 0 1 2; do
  run "da_ao_s$s" aokvqa-da "${DA[@]}" --projector "results/runs/laptop/mac_extra/B_answer_only_n4000_s${s}_mps/projector.pt"
done
echo "=== $(date -u +%FT%TZ) all done" | tee -a "$LOG"
grep -h -A4 "^| image" "$OUT"/*/summary.md 2>/dev/null | head -60
