"""Collect per-seed val metrics of the cloud runs into results/run_metrics.json.

results/run_metrics.json is the input of make_report_tables.py and make_figures.py; it is
already in the repository, so this script is needed only to rebuild it. It covers the run
folders G05_* / G7_* named B (main), HP (hyperparameter study), BAL (balanced loss) and theta0
(untrained bridge θ₀); G7_A1_instrument and G05_gate64 have no eval_val/ and are skipped.

Reads each run's eval_val/summary.json (in the repository) and eval_val/predictions.jsonl
(release asset runs.tar.gz, scripts/download_release.sh) and records, for the real image:
accuracy (real), swapped-image accuracy (swapped), accuracy on the prior-wrong subset P (accP),
permutation accuracy (perm: argmax of the probabilities averaged over option orders),
circular accuracy (circ: right under every order), accuracy averaged over the option orders
(mean_rot) and the number of orders scored (n_rot: 4, or 1 for R→A).

    python scripts/collect_runs.py            # results/runs/cloud -> results/run_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"(G05|G7)_(B|HP|BAL|theta0)_(.*?)(?:_s(\d))?$")


def metrics(run: Path) -> dict[str, float | int | None]:
    s = json.loads((run / "eval_val/summary.json").read_text())
    rows = {r["condition"]: r for r in s["rows"] if r["context"] == "none"}
    real: dict[str, dict[int, dict]] = defaultdict(dict)
    with (run / "eval_val/predictions.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["context"] == "none" and r["condition"] == "real":
                real[r["question_id"]][r["rotation"]] = r
    rots = sorted(next(iter(real.values())))
    mean_rot = np.mean(
        [[real[q][k]["pred_idx"] == real[q][k]["correct_idx"] for k in rots] for q in real]
    )
    sw = rows.get("swapped")
    return {
        "real": 100 * rows["real"]["acc"],
        "swapped": 100 * sw["acc"] if sw else None,
        "accP": 100 * rows["real"].get("acc_prior_wrong", float("nan")),
        "perm": 100 * rows["real"]["perm_acc"],
        "circ": 100 * rows["real"]["circ_acc"],
        "mean_rot": 100 * float(mean_rot),
        "n_rot": len(rots),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--runs", type=Path, default=ROOT / "results/runs/cloud", help="folder of cloud runs"
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "results/run_metrics.json", help="JSON to write"
    )
    args = ap.parse_args()
    cells: dict[str, dict[str, dict]] = {}
    missing = []
    for d in sorted(args.runs.glob("G*_*")):
        m = NAME.match(d.name)
        if not m or not (d / "eval_val/summary.json").exists():
            continue
        if not (d / "eval_val/predictions.jsonl").exists():
            missing.append(d.name)
            continue
        cells.setdefault(f"{m.group(1)}_{m.group(2)}_{m.group(3)}", {})[m.group(4) or "-"] = (
            metrics(d)
        )
    if missing:
        raise SystemExit(
            f"{len(missing)} runs have no predictions.jsonl (download runs.tar.gz), "
            f"e.g. {missing[0]}"
        )
    args.out.write_text(json.dumps(cells, indent=1) + "\n")
    print(f"wrote {args.out} ({len(cells)} settings)")


if __name__ == "__main__":
    main()
