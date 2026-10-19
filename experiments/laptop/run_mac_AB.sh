#!/usr/bin/env bash
# Laptop runs (0.5B, Apple-silicon GPU) for the proposals of Table 6.1: option-shuffled
# training (A, Table 4.11), SigLIP option scores for the fusion (B, Table 4.10), fewer image
# tokens with the training speed-ups (E1, Table 4.12) and the direct-answer (DA) target
# (Section 4.11, Table 4.16). Each job writes results/runs/laptop/mac_AB/<run>/. Needs
# data/raw/{aokvqa,coco} (scripts/download_data.sh --train) and the predictions.jsonl of
# results/runs/laptop/pilot_a1v2_instrument (release asset; it defines the subset P).
# Run from the repo root:
#     bash experiments/laptop/run_mac_AB.sh            # the default jobs, in order
#     JOBS="ao_s1" bash experiments/laptop/run_mac_AB.sh   # just some jobs
# The report's runs: the default jobs; tok196_s0; tok196_pilot_s0 and tok81_pilot_s0 with
# CPU_SHARE=5 (their folders, AO_shuf_tok{196,81}_pilot_n4000_s0, carry no "k5");
# tok196_pilot_s0, e1_seeds and da_seeds with CPU_SHARE=auto. Run names: shuf = options shuffled in training; tok196 / tok81 = image
# tokens; pilot = the two later speed-ups of Section 3.8.2 (torch.compile and CPU-GPU batch
# splitting); k<share> = how many of the 16 micro-batches of a step run on the CPU (kauto =
# set from measured step times). Run times: results/analysis/laptop_run_times.txt.
# Finished jobs are skipped on a re-run; training resumes from its last checkpoint.
# Progress: tail -f results/runs/laptop/mac_AB/queue.log
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
export PYTHONPATH=src
# Cap the MPS pool below PyTorch's defaults (1.7 / 1.4), which let it grow into swap.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${MPS_HIGH:-0.7}" PYTORCH_MPS_LOW_WATERMARK_RATIO="${MPS_LOW:-0.5}"
OUT=results/runs/laptop/mac_AB; mkdir -p "$OUT"
LOG="$OUT/queue.log"
# memory log every 30 s: swap, macOS free-memory %, and the resident size of each python process
( while true; do
    echo "$(date -u +%FT%TZ) $(sysctl -n vm.swapusage | awk '{print "swap_used="$6}') " \
         "free=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/percentage/{print $2}') " \
         "py_rss_mb=$(ps -axo rss=,comm= | awk '/[Pp]ython/{s+=$1} END{printf "%d", s/1024}')"
    sleep 30
  done ) >> "$OUT/mem.log" 2>&1 &
MON=$!
# system metrics every 15 s (no sudo): CPU, GPU, free memory, swap, compressed memory,
# python RSS, thermal state
( while true; do
    cpu=$(top -l 2 -n 0 -s 1 | awk -F'[:,%]' '/CPU usage/{u=$2+0; s=$4+0; i=$6+0} END{printf "user=%s sys=%s idle=%s", u, s, i}')
    gpu=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
    therm=$(pmset -g therm 2>/dev/null | grep -Eo 'CPU_Speed_Limit[^0-9]*[0-9]+' | tr -d ' \t' | head -1)
    therm=${therm:-no_warning}  # pmset prints a speed limit only when macOS is throttling
    comp=$(vm_stat | awk '/occupied by compressor/{printf "%.1f", $NF*16384/2^30}')
    echo "$(date -u +%FT%TZ) cpu_$cpu gpu_util=${gpu:-na} free=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/percentage/{print $2}') " \
         "$(sysctl -n vm.swapusage | awk '{print "swap_used="$6}') compressed_gb=${comp:-na} " \
         "py_rss_mb=$(ps -axo rss=,comm= | awk '/[Pp]ython/{s+=$1} END{printf "%d", s/1024}') therm=${therm:-none}"
    sleep 15
  done ) >> "$OUT/metrics.log" 2>&1 &
MET=$!
trap 'kill $MON $MET 2>/dev/null' EXIT
WAKE=(); command -v caffeinate >/dev/null && WAKE=(caffeinate -i)   # no sleep mid-run
COMMON=(--bridge configs/ov_qwen2_05b_frozen.json --aokvqa-dir data/raw/aokvqa
        --coco-dir data/raw/coco --prior-run results/runs/laptop/pilot_a1v2_instrument
        --conditions real,swapped --device mps --dtype bfloat16 --eval-dtype float32
        --threads 4 --select last --eval-every 1000000 --batch-size 1 --grad-accum 16
        --checkpoint-every 10 --resume-if-present --option-shuffle --dev-size 100 --block-attention
        --feature-cache cache/features)
# frozen SigLIP features are cached per image, for the train and val questions used
N4K=(--train-limit 4000 --feature-cache-splits train,val)
CPU_SHARE="${CPU_SHARE:-auto}"   # micro-batches of 16 on the CPU copy; auto = balanced from step times
PILOT=(--compile --pad-multiple 32 --cpu-share "$CPU_SHARE" --cpu-threads "${CPU_THREADS:-6}")
# DA target: COMMON minus --option-shuffle and --prior-run, which stage_b refuses for it
DA_COMMON=(); skip_next=0
for a in "${COMMON[@]}"; do
  if [ "$skip_next" = 1 ]; then skip_next=0; continue; fi
  case $a in
    --option-shuffle) continue ;;
    --prior-run) skip_next=1; continue ;;
  esac
  DA_COMMON+=("$a")
done

run_job() {  # name, then every stage_b argument
  local name=$1; shift
  if [ -s "$OUT/$name/eval_val/summary.md" ]; then echo "skip $name (done)" | tee -a "$LOG"; return; fi
  echo "=== $(date -u +%FT%TZ) start $name" | tee -a "$LOG"
  "${WAKE[@]}" "$PY" -m cmr.runners.stage_b "$@" --out "$OUT/$name" >>"$LOG" 2>&1
  echo "=== $(date -u +%FT%TZ) end $name exit=$?" | tee -a "$LOG"
}
job() { local name=$1; shift; run_job "$name" "${COMMON[@]}" "$@"; }  # name, then stage_b arguments
da_job() { local name=$1; shift; run_job "$name" "${DA_COMMON[@]}" "$@"; }  # DA target

for j in ${JOBS:-siglip ao_s0 ao_s1 atr_bal_s0}; do
  case $j in
    # B: SigLIP image-option scores on val, fused with saved predictions afterwards (fuse_siglip.py)
    siglip) [ -s results/runs/laptop/siglip_val/scores.jsonl ] && echo "skip siglip (done)" | tee -a "$LOG" || {
              echo "=== $(date -u +%FT%TZ) start siglip" | tee -a "$LOG"
              "${WAKE[@]}" "$PY" experiments/laptop/siglip_option_scores.py --aokvqa-dir data/raw/aokvqa \
                --coco-dir data/raw/coco --out results/runs/laptop/siglip_val 2>&1 | tee -a "$LOG" | tail -3; } ;;
    # A: shuffled options, 4k items
    ao_s0) job AO_shuf_n4000_s0 --target answer_only --rotations 4 --seed 0 "${N4K[@]}" ;;
    ao_s1) job AO_shuf_n4000_s1 --target answer_only --rotations 4 --seed 1 "${N4K[@]}" ;;
    # A + balanced loss for answer-then-rationale
    atr_bal_s0) job ATR_shuf_rw0.3_n4000_s0 --target answer_then_rationale --rotations 4 \
                  --seed 0 --rationale-weight 0.3 "${N4K[@]}" ;;
    # E1: fewer image tokens (SigLIP grid pooled 2x2 -> 196; 3x3 -> 81 below), AO, 4k items
    tok196_s0) job AO_shuf_tok196_n4000_s0 --bridge configs/ov_qwen2_05b_pool2.json \
                 --target answer_only --rotations 4 --seed 0 "${N4K[@]}" ;;
    # pilot: torch.compile + CPU-GPU batch splitting (CPU_SHARE of the 16 micro-batches on an
    # fp32 CPU copy); any seed: tok196_pilot_s1, tok81_pilot_s2, ...
    tok196_pilot_s[0-9]*) s=${j##*_s}
      job AO_shuf_tok196_pilot_k${CPU_SHARE}_n4000_s$s --bridge configs/ov_qwen2_05b_pool2.json \
        --target answer_only --rotations 4 --seed "$s" "${N4K[@]}" "${PILOT[@]}" ;;
    tok81_pilot_s[0-9]*) s=${j##*_s}
      job AO_shuf_tok81_pilot_k${CPU_SHARE}_n4000_s$s --bridge configs/ov_qwen2_05b_pool3.json \
        --target answer_only --rotations 4 --seed "$s" "${N4K[@]}" "${PILOT[@]}" ;;
    # E1 extra seeds (1-2) at both token counts, with the pilot speed-ups
    e1_seeds) for s in 1 2; do for t in 196 81; do
        pool=$([ "$t" = 196 ] && echo 2 || echo 3)
        job AO_shuf_tok${t}_pilot_k${CPU_SHARE}_n4000_s$s --bridge configs/ov_qwen2_05b_pool$pool.json \
          --target answer_only --rotations 4 --seed "$s" "${N4K[@]}" "${PILOT[@]}"
      done; done ;;
    # DA target, seeds 0-2: trained on the most frequent direct answer, evaluated by
    # greedy generation on val (real, swapped); 729 tokens, 4k items, pilot speed-ups
    da_seeds) for s in 0 1 2; do
        da_job B_direct_answer_n4000_s$s --target direct_answer --seed "$s" "${N4K[@]}" "${PILOT[@]}"
      done ;;
    *) echo "unknown job $j" | tee -a "$LOG" ;;
  esac
done
echo "queue finished $(date -u +%FT%TZ)" | tee -a "$LOG"
