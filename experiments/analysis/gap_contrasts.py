# ruff: noqa: RUF001, RUF002
"""Section 4.5: intervals for differences in the image-reliance gap between targets.

The gap of a run is real-image accuracy minus swapped-image accuracy (rotation 0, val). Each
contrast is gap(target) − gap(AO) with the plain percentile bootstrap of Section 3.7 (items
paired, seeds resampled within each target). The gap divided by real-image accuracy is also
given, since a less accurate system has less accuracy to lose.
Needs the cloud predictions (release asset runs.tar.gz). Output: printed, and written to
results/analysis/gap_contrasts.txt unless --no-save.

    python experiments/analysis/gap_contrasts.py [--cloud-runs results/runs/cloud] [--no-save]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cmr.eval.stats import hierarchical_bootstrap  # noqa: E402

SETTINGS = [
    ("0.5B 4k", "G05_B_{}_n4000"),
    ("0.5B 16k", "G05_B_{}_full"),
    ("7B 4k", "G7_B_{}_n4000"),
]
AO, AR, RA = "answer_only", "answer_then_rationale", "rationale_then_answer"


def correct(run_dir: Path, cond: str) -> dict[str, float]:
    out = {}
    with open(run_dir / "eval_val" / "predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("context", "none"), r.get("rotation", 0), r["condition"]) == (
                "none",
                0,
                cond,
            ):
                out[r["question_id"]] = float(r["pred_idx"] == r["correct_idx"])
    return out


def gaps(root: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """(n_seeds, n_items) real-image correctness and real − swapped per item."""
    seeds = sorted(int(p.name.rsplit("_s", 1)[1]) for p in root.glob(f"{prefix}_s?"))
    real, gap = [], []
    for s in seeds:
        r, w = correct(root / f"{prefix}_s{s}", "real"), correct(root / f"{prefix}_s{s}", "swapped")
        q = sorted(r)
        real.append([r[i] for i in q])
        gap.append([r[i] - w[i] for i in q])
    return np.array(real), np.array(gap)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cloud-runs", type=Path, default=ROOT / "results/runs/cloud")
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args()
    lines = [
        "Gap = real − swapped accuracy (pp); gap(target) − gap(AO) [95% plain bootstrap]",
        "",
    ]
    for label, pat in SETTINGS:
        ao_real, ao_gap = gaps(args.cloud_runs, pat.format(AO))
        rel = ao_gap.mean() / ao_real.mean()
        lines.append(f"{label}: AO gap {100 * ao_gap.mean():.1f}, gap / real {rel:.2f}")
        for tgt, name in ((AR, "A→R"), (RA, "R→A")):
            real, gap = gaps(args.cloud_runs, pat.format(tgt))
            r = hierarchical_bootstrap(
                gap, ao_gap, n_boot=4000, seed=0, small_seed_correction=False
            )
            rel = gap.mean() / real.mean()
            lines.append(
                f"{label}: {name} gap {100 * gap.mean():.1f}, gap / real {rel:.2f};"
                f" {name} − AO {100 * r.mean_diff:+.1f} [{100 * r.low:+.1f}, {100 * r.high:+.1f}]"
            )
    text = "\n".join(lines)
    print(text)
    if not args.no_save:
        (ROOT / "results/analysis/gap_contrasts.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
