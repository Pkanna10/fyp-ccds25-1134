# ruff: noqa: RUF001, RUF002
"""Section 5.1: costs of the rationale targets before and after averaging over option orders.

For each setting, the A→R − AO and R→A − AO contrasts on real images (val) are computed from
the original order alone and from the 4-order ensemble (option log-probabilities averaged
over the four rotations, as in Table 4.10), with the plain percentile bootstrap of Section 3.7.
R→A is scored in the original order only, so its ensemble value is its order-0 value.
Also prints the Table 4.6 R→A default − AO contrast on the matched seeds 0–2.
Needs the cloud predictions (release asset runs.tar.gz). Output: printed, and written to
results/analysis/ensemble_costs.txt unless --no-save.

    python experiments/analysis/ensemble_costs.py [--cloud-runs results/runs/cloud] [--no-save]
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
    ("0.5B 4k", "G05_B_{}_n4000", None),
    ("0.5B 16k", "G05_B_{}_full", None),
    ("7B 4k", "G7_B_{}_n4000", (0, 1, 2)),
    ("7B 16k", "G7_B_{}_full", (0, 1, 2)),
]
AO, AR, RA = "answer_only", "answer_then_rationale", "rationale_then_answer"


def correct(run_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """{question_id: correct} at order 0 and under the 4-order ensemble (real image)."""
    lp: dict[str, dict[int, np.ndarray]] = {}
    gold: dict[str, int] = {}
    with open(run_dir / "eval_val" / "predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("context", "none") != "none" or r["condition"] != "real":
                continue
            lp.setdefault(r["question_id"], {})[r.get("rotation", 0)] = np.array(
                r["option_logprobs"]
            )
            gold[r["question_id"]] = r["correct_idx"]
    rot0 = {q: float(np.argmax(v[0]) == gold[q]) for q, v in lp.items()}
    ens = {q: float(np.argmax(np.mean(list(v.values()), 0)) == gold[q]) for q, v in lp.items()}
    return rot0, ens


def matrix(root: Path, prefix: str, seeds: tuple[int, ...] | None) -> tuple[np.ndarray, ...]:
    if seeds is None:
        seeds = tuple(sorted(int(p.name.rsplit("_s", 1)[1]) for p in root.glob(f"{prefix}_s?")))
    rows = [correct(root / f"{prefix}_s{s}") for s in seeds]
    qids = sorted(rows[0][0])
    return tuple(np.array([[r[i][q] for q in qids] for r in rows]) for i in (0, 1))


def fmt(x: np.ndarray, y: np.ndarray) -> str:
    r = hierarchical_bootstrap(x, y, n_boot=4000, seed=0, small_seed_correction=False)
    return f"{100 * r.mean_diff:+5.1f} [{100 * r.low:+5.1f}, {100 * r.high:+5.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cloud-runs", type=Path, default=ROOT / "results/runs/cloud")
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args()
    root = args.cloud_runs
    head = "Contrast on real images (pp) [95% plain bootstrap]   order 0   |   4-order ensemble"
    lines = [head, ""]
    for label, pat, seeds in SETTINGS:
        ao0, aoe = matrix(root, pat.format(AO), seeds)
        for tgt, name in ((AR, "A→R − AO"), (RA, "R→A − AO")):
            if not any(root.glob(pat.format(tgt) + "_s?")):
                continue
            x0, xe = matrix(root, pat.format(tgt), seeds)
            if tgt == RA:
                xe = x0  # R→A is scored in the original order only
            lines.append(f"{label}: {name:9s} {fmt(x0, ao0)}   |   {fmt(xe, aoe)}")
    s3 = (0, 1, 2)
    ra, _ = matrix(root, f"G05_B_{RA}_full", s3)
    ao, _ = matrix(root, f"G05_B_{AO}_full", s3)
    lines += ["", f"Table 4.6, 0.5B 16k: R→A default − AO, seeds 0–2: {fmt(ra, ao)}"]
    text = "\n".join(lines)
    print(text)
    if not args.no_save:
        (ROOT / "results/analysis/ensemble_costs.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
