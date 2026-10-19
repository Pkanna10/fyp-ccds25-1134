#!/usr/bin/env bash
# Launch one phase of the cloud study (report Table 3.5; jobs in experiments/cloud/fyp_modal.py)
# on Modal. Run from the repo root, with the modal client installed (pip install modal) and
# logged in (modal setup):
#     bash experiments/cloud/launch.sh             # phase 1: 0.5B main grid + first 7B runs
#     PHASE=2   bash experiments/cloud/launch.sh   # phase 2: the rest of the 7B grid
#     PHASE=hp  bash experiments/cloud/launch.sh   # 0.5B hyperparameter study (Table 4.5)
#     PHASE=bal bash experiments/cloud/launch.sh   # balanced loss, lambda = 0.3 (Table 4.6)
#     modal run experiments/cloud/fyp_modal.py::main --phase 1 --dry-run   # list the jobs only
# Needs data/raw/aokvqa and data/raw/coco (scripts/download_data.sh --train) and the
# predictions.jsonl of results/runs/laptop/pilot_a1v2_instrument (release asset,
# scripts/download_release.sh). The jobs run detached, so the local machine can sleep.
# Fetch the results with
#     modal volume get fyp-runs / results/runs/cloud/
set -euo pipefail
cd "$(dirname "$0")/../.."
PHASE="${PHASE:-1}"
MODAL="${MODAL:-modal}"
command -v "$MODAL" >/dev/null || { echo "modal client not found: pip install modal && modal setup"; exit 1; }

# 1) data volume: upload only what is missing
"$MODAL" volume create fyp-data >/dev/null 2>&1 || true
have() { "$MODAL" volume ls fyp-data "$1" >/dev/null 2>&1; }
have /aokvqa || "$MODAL" volume put fyp-data data/raw/aokvqa /aokvqa
have /coco   || "$MODAL" volume put fyp-data data/raw/coco /coco
# the text-only scores of the frozen 0.5B LLM, which define the prior-wrong subset P
have /prior/pilot_a1v2_instrument || "$MODAL" volume put fyp-data results/runs/laptop/pilot_a1v2_instrument /prior/pilot_a1v2_instrument
echo "data volume:"; "$MODAL" volume ls fyp-data /

# 2) launch detached
"$MODAL" run --detach experiments/cloud/fyp_modal.py::main --phase "$PHASE"
