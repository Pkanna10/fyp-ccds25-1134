#!/usr/bin/env bash
# Laptop queue, evaluation only (no training): Table 4.8 (A->R rows), Table 4.9 and the 7B
# rows of Tables 4.13 (direct answer) and 4.16. Each job writes
# results/runs/laptop/mac_final/<job>/. Run from the repo root:
#     caffeinate -i bash experiments/laptop/run_mac_final.sh
# Needs data/raw/{aokvqa,coco}, the cloud bridges results/runs/cloud/<run>/projector.pt and the
# R->A predictions.jsonl (release assets); fetch() below downloads a missing bridge from the
# Modal volume fyp-runs instead. Run times: results/analysis/laptop_run_times.txt.
# Progress: tail -f results/runs/laptop/mac_final/queue.log
# Finished jobs are skipped on a re-run; a failed job is logged and the queue moves on.
#
# Part 1 (0.5B, fp32; Tables 4.8 and 4.9), with experiments/laptop/explanation_checks.py
#   atr_*    A->R bridges (cloud, full data, seeds 0-2) generate explanations, real and swapped
#   atrbal_* the same for the balanced-loss A->R bridges (lambda = 0.3), if they can be fetched
#   steer_*  AO bridges answer with the same-seed R->A explanation as context
# Part 2 (7B, bf16: fp32 does not fit in 24 GB), only if a short timing test passes:
#   MMStar and A-OKVQA direct answers for the untrained 7B bridge and the 7B AO bridge
#   (all data, seed 0: G7_B_answer_only_full_s0; if it cannot be fetched, the 4k seed-0
#   bridge G7_B_answer_only_n4000_s0).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
MODAL="${MODAL:-modal}"
export PYTHONPATH=src
OUT=results/runs/laptop/mac_final; mkdir -p "$OUT"
LOG="$OUT/queue.log"
say() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

job() {  # job <name> <command...>: skipped when <name>/summary.md exists
  local name=$1; shift
  if [ -f "$OUT/$name/summary.md" ]; then say "skip $name (done)"; return 0; fi
  say "start $name"
  if "$@" >> "$LOG" 2>&1; then say "end $name ok"; else say "FAILED $name (see above)"; fi
}

# fetch <run name>: projector.pt from the Modal volume into results/runs/cloud/<run>/ (nothing
# to do when it is already there, e.g. from the release assets)
fetch() {
  local run=$1 dst="results/runs/cloud/$1/projector.pt"
  [ -f "$dst" ] && return 0
  mkdir -p "results/runs/cloud/$run"
  "$MODAL" volume get fyp-runs "$run/projector.pt" "$dst" >> "$LOG" 2>&1 && [ -f "$dst" ]
}

# Download Qwen2-7B-Instruct (~15 GB, once) in the background while part 1 runs.
"$PY" -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2-7B-Instruct', allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*'])" \
  > "$OUT/download_7b.log" 2>&1 &
DL_PID=$!
say "queue started; Qwen2-7B download running in the background (pid $DL_PID)"

# ------------------------------------------------------------------ part 1: 0.5B
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
EC=("$PY" experiments/laptop/explanation_checks.py)
for s in 0 1 2; do
  job "atr_full_s$s" "${EC[@]}" generate --out "$OUT/atr_full_s$s" \
    --projector "results/runs/cloud/G05_B_answer_then_rationale_full_s$s/projector.pt"
done
for s in 0 1 2; do
  job "steer_full_s$s" "${EC[@]}" steer --out "$OUT/steer_full_s$s" \
    --projector "results/runs/cloud/G05_B_answer_only_full_s$s/projector.pt" \
    --explanations "results/runs/cloud/G05_B_rationale_then_answer_full_s$s/eval_val/predictions.jsonl"
done
for s in 0 1 2; do
  run="G05_BAL_answer_then_rationale_full_rw0.3_s$s"
  if fetch "$run"; then
    job "atrbal_full_s$s" "${EC[@]}" generate --out "$OUT/atrbal_full_s$s" \
      --projector "results/runs/cloud/$run/projector.pt"
  else
    say "skip atrbal_full_s$s (could not download $run from Modal)"
  fi
done

# ------------------------------------------------------------------ part 2: 7B
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # no PyTorch cap; the 7B weights alone are ~15 GB
B7=configs/ov_qwen2_7b_frozen.json
T7=("$PY" -m cmr.runners.transfer --model "bridge:$B7" --conditions real,swapped --rotations 1
    --device mps --dtype bfloat16 --threads 4)
DA=(--aokvqa-dir data/raw/aokvqa --coco-dir data/raw/coco --split val --max-new-tokens 10)
P7=results/runs/cloud/G7_B_answer_only_n4000_s0/projector.pt; P7NAME=ao4k_s0
if fetch G7_B_answer_only_full_s0; then P7=results/runs/cloud/G7_B_answer_only_full_s0/projector.pt; P7NAME=aofull_s0; fi
say "7B AO bridge for part 2: $P7"

say "waiting for the Qwen2-7B-Instruct download (started at the beginning)"
wait "$DL_PID" || say "download of Qwen2-7B-Instruct failed (see $OUT/download_7b.log)"
rm -rf "$OUT/7b_timing"
say "start 7b_timing (MMStar, 8 items x real/swapped)"
t0=$(date +%s)
"${T7[@]}" --benchmark mmstar --limit 8 --out "$OUT/7b_timing" >> "$LOG" 2>&1
rc=$?; dt=$(( $(date +%s) - t0 ))
say "end 7b_timing rc=$rc seconds=$dt (model loading + 16 scored items)"
if [ $rc -ne 0 ]; then
  say "7B does not run on this laptop (see the log above); part 2 skipped"
elif [ $dt -gt 600 ]; then
  say "7B timing test took $dt s, too slow for the remaining time; part 2 skipped"
else
  job "7b_mmstar_theta0" "${T7[@]}" --benchmark mmstar --out "$OUT/7b_mmstar_theta0"
  job "7b_mmstar_$P7NAME" "${T7[@]}" --benchmark mmstar --projector "$P7" --out "$OUT/7b_mmstar_$P7NAME"
  job "7b_da_theta0" "${T7[@]}" --benchmark aokvqa-da "${DA[@]}" --out "$OUT/7b_da_theta0"
  job "7b_da_$P7NAME" "${T7[@]}" --benchmark aokvqa-da "${DA[@]}" --projector "$P7" --out "$OUT/7b_da_$P7NAME"
fi
say "all done"
grep -h -A14 "^| measure\|^| image" "$OUT"/*/summary.md 2>/dev/null | head -120
