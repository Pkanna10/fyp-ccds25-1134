#!/usr/bin/env bash
# Download A-OKVQA and the COCO images it uses into data/raw (idempotent: skips finished parts).
# The data are not redistributed by this repository; see THIRD_PARTY_NOTICES.md for their terms.
# Set DATA_ROOT to use another folder.
#
#   bash scripts/download_data.sh            # enough to evaluate and draw the example figures
#   bash scripts/download_data.sh --train    # also the training images
#
#   data/raw/aokvqa/aokvqa_v1p0_{train,val,test}.json   A-OKVQA questions + rationales (~25 MB)
#   data/raw/coco/val2017/*.jpg                        COCO val2017 (A-OKVQA val)    (~1 GB)
#   data/raw/coco/annotations/captions_val2017.json    COCO captions (Table 4.7)  (~240 MB zip)
#   data/raw/coco/train2017/*.jpg                      only with --train (training)  (~18 GB)
set -euo pipefail
ROOT="${DATA_ROOT:-data/raw}"
mkdir -p "$ROOT/aokvqa" "$ROOT/coco"

fetch() {  # fetch <url> <dest>
  [ -s "$2" ] && { echo "have $2"; return; }
  echo "get  $1"; curl -fL --retry 4 --retry-delay 2 -o "$2.part" "$1" && mv "$2.part" "$2"
}

fetch https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz "$ROOT/aokvqa.tar.gz"
[ -s "$ROOT/aokvqa/aokvqa_v1p0_val.json" ] || tar -xzf "$ROOT/aokvqa.tar.gz" -C "$ROOT/aokvqa"
# The annotation files used by every run (also recorded in each run's manifest).
(cd "$ROOT/aokvqa" && sha256sum -c - <<'SUMS'
94820c4452d29102365647598c54608d4f48f56bee01b514c996a09cc0972029  aokvqa_v1p0_train.json
fd90b05dec8d78d341b69d9516c54a9b30dd4c12c3545599dab5d6f725aba6b4  aokvqa_v1p0_val.json
SUMS
)

# COCO 2017 from the official links given on cocodataset.org.

fetch http://images.cocodataset.org/zips/val2017.zip "$ROOT/coco/val2017.zip"
[ -d "$ROOT/coco/val2017" ] || unzip -q "$ROOT/coco/val2017.zip" -d "$ROOT/coco"

fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip "$ROOT/coco/ann.zip"
[ -s "$ROOT/coco/annotations/captions_val2017.json" ] || unzip -q "$ROOT/coco/ann.zip" -d "$ROOT/coco"

if [[ "${1:-}" == "--train" ]]; then
  fetch http://images.cocodataset.org/zips/train2017.zip "$ROOT/coco/train2017.zip"
  [ -d "$ROOT/coco/train2017" ] || unzip -q "$ROOT/coco/train2017.zip" -d "$ROOT/coco"
fi
echo "done: $(ls "$ROOT/aokvqa" | tr '\n' ' ')"
