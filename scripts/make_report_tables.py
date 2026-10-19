"""Rebuild the seed-averaged report tables from results/run_metrics.json.

Needs only files in the repository (no release assets, no GPU); CI checks that the output
matches the committed CSVs. Writes one CSV per table to results/tables/ and prints them:
    table_4_1_theta0.csv      Table 4.1   untrained bridge θ₀ under each target's instruction
    table_4_3_main.csv        Table 4.3   target x LLM x training-set size, mean ± SD over seeds
    table_4_6_balanced.csv    Table 4.6   balanced loss (λ = 0.3) vs default loss and AO
    table_f_1_seeds.csv       Table F.1   every seed of every main and balanced-loss setting

Table 4.2 (CPU/Mac against cloud) and the paired-bootstrap contrasts of Tables 4.4-4.6 come from
experiments/cloud/analyze_*.py.

    python scripts/make_report_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# the three training targets: (label in the report, name in the run folders)
ARMS = [("AO", "answer_only"), ("A→R", "answer_then_rationale"), ("R→A", "rationale_then_answer")]


def sd(v: list[float]) -> float:
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def ms(v: list[float], d: int = 1) -> str:
    return f"{np.mean(v):.{d}f} ± {sd(v):.{d}f}" if len(v) > 1 else f"{np.mean(v):.{d}f}"


TARGETS = tuple(name for _, name in ARMS)


def build(C: dict) -> dict[str, tuple[list[str], list[list[str]]]]:
    main = []
    for g, size in [("G05", "0.5B"), ("G7", "7B")]:
        for d, dl in [("n4000", "4k"), ("full", "16k")]:
            for an, ak in ARMS:
                k = f"{g}_B_{ak}_{d}"
                if k not in C:
                    continue
                v = list(C[k].values())
                real = [x["real"] for x in v]
                sw = [x["swapped"] for x in v]
                gap = [a - b for a, b in zip(real, sw)]
                P = [x["accP"] for x in v]
                rot = [x["mean_rot"] for x in v]
                main.append(
                    [
                        size,
                        dl,
                        an,
                        str(len(v)),
                        ms(real),
                        ms(sw),
                        ms(gap),
                        ms(P) if g == "G05" else "–",
                        ms(rot) if v[0]["n_rot"] == 4 else "–",
                    ]
                )
    theta = []
    for g, size in [("G05", "0.5B"), ("G7", "7B")]:
        for an, ak in ARMS:
            x = C[f"{g}_theta0_{ak}"]["-"]
            theta.append(
                [
                    size,
                    an,
                    f"{x['real']:.1f}",
                    f"{x['swapped']:.1f}",
                    f"{x['real'] - x['swapped']:.1f}",
                    f"{x['mean_rot']:.1f}" if x["n_rot"] == 4 else "–",
                ]
            )

    def g1_order(k: str) -> tuple:
        """LLM size, then data, then target, with each balanced row after its default row."""
        tgt = next(i for i, t in enumerate(TARGETS) if t in k)
        return (k.startswith("G7"), "_full" in k, tgt, "_BAL_" in k)

    seeds = []
    for k in sorted(C, key=g1_order):
        if k.startswith(("G05_B", "G7_B", "G05_BAL", "G7_BAL")):
            v = C[k]
            ss = sorted(v)
            name = (
                k.replace("G05_", "0.5B ")
                .replace("G7_", "7B ")
                .replace("B_", "")
                .replace("BAL_", "balanced ")
                .replace("answer_only", "AO")
                .replace("answer_then_rationale", "A→R")
                .replace("rationale_then_answer", "R→A")
                .replace("_n4000", " 4k")
                .replace("_full", " 16k")
                .replace("_rw0.3", " λ=0.3")
            )
            r = [v[s]["real"] for s in ss]
            seeds.append(
                [name]
                + [f"{v[s]['real']:.1f}" if s in v else "" for s in "01234"]
                + [f"{np.mean(r):.1f}", f"{sd(r):.1f}", f"{max(r) - np.mean(r):+.1f}"]
            )
    bal = []
    for lab, k in [
        ("0.5B 16k: AO", "G05_B_answer_only_full"),
        ("0.5B 16k: A→R default", "G05_B_answer_then_rationale_full"),
        ("0.5B 16k: A→R balanced", "G05_BAL_answer_then_rationale_full_rw0.3"),
        ("0.5B 16k: R→A default", "G05_B_rationale_then_answer_full"),
        ("0.5B 16k: R→A balanced", "G05_BAL_rationale_then_answer_full_rw0.3"),
        ("7B 4k: AO", "G7_B_answer_only_n4000"),
        ("7B 4k: A→R default", "G7_B_answer_then_rationale_n4000"),
        ("7B 4k: A→R balanced", "G7_BAL_answer_then_rationale_n4000_rw0.3"),
        ("7B 4k: R→A default", "G7_B_rationale_then_answer_n4000"),
        ("7B 4k: R→A balanced", "G7_BAL_rationale_then_answer_n4000_rw0.3"),
    ]:
        # matched seeds: only those the balanced-loss runs also have
        keep = ["0", "1", "2"] if lab.startswith("0.5B") else ["0", "1"]
        v = [C[k][s] for s in keep if s in C[k]]
        real = [x["real"] for x in v]
        sw = [x["swapped"] for x in v]
        bal.append([lab, str(len(v)), ms(real), ms(sw), ms([a - b for a, b in zip(real, sw)])])
    return {
        "table_4_1_theta0": (
            ["LLM", "Target prompt", "Real", "Swapped", "Gap", "4-order average"],
            theta,
        ),
        "table_4_3_main": (
            [
                "LLM",
                "Data",
                "Target",
                "Seeds",
                "Real",
                "Swapped",
                "Gap",
                "Acc. on P",
                "4-order average",
            ],
            main,
        ),
        "table_4_6_balanced": (["Setting", "Seeds", "Real", "Swapped", "Gap"], bal),
        "table_f_1_seeds": (
            ["Setting", "s0", "s1", "s2", "s3", "s4", "Mean", "SD", "Best − mean"],
            seeds,
        ),
    }


def main() -> None:
    C = json.loads((ROOT / "results/run_metrics.json").read_text())
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, (head, rows) in build(C).items():
        with (out / f"{name}.csv").open("w", newline="") as f:
            csv.writer(f).writerows([head, *rows])
        print(f"\n{name}\n" + "\n".join(" | ".join(r) for r in [head, *rows]))


if __name__ == "__main__":
    main()
