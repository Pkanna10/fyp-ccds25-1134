#!/usr/bin/env bash
# Download the release assets (trained bridges and full run outputs) and unpack them into
# results/runs/, next to the small summaries that are in git. Works from any directory; a re-run
# reuses the archives already downloaded to .release/.
#     bash scripts/download_release.sh                 # predictions + logs, and all weights
#     bash scripts/download_release.sh runs            # only predictions + logs (~60 MB)
#     bash scripts/download_release.sh runs weights-0.5b   # any list of the assets below
# Assets (each a .tar.gz, checked against SHA256SUMS.txt from the same release):
#     runs             predictions.jsonl, train_log.jsonl and queue logs of every reported run
#     weights-0.5b     projector.pt of every reported 0.5B run (cloud and laptop)
#     weights-0.5b-hp  projector.pt of the 0.5B hyperparameter study (46 runs)
#     weights-7b       projector.pt of every 7B run
# REPO and TAG select another repository or release (default Pkanna10/fyp-ccds25-1134, v1.0).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${REPO:-Pkanna10/fyp-ccds25-1134}"
TAG="${TAG:-v1.0}"
ASSETS=("$@"); [ ${#ASSETS[@]} -gt 0 ] || ASSETS=(runs weights-0.5b weights-0.5b-hp weights-7b)
BASE="https://github.com/$REPO/releases/download/$TAG"
mkdir -p .release && cd .release
curl -fsSL -o SHA256SUMS.txt "$BASE/SHA256SUMS.txt"
for a in "${ASSETS[@]}"; do
  f="$a.tar.gz"
  [ -s "$f" ] || { echo "get $f"; curl -fL --retry 3 -o "$f.part" "$BASE/$f" && mv "$f.part" "$f"; }
  grep " $f\$" SHA256SUMS.txt | shasum -a 256 -c -
  tar -xzf "$f" -C ..
done
echo "unpacked into results/runs/ (see results/README.md)"
