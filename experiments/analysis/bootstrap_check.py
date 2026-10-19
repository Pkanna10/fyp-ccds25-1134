# ruff: noqa: RUF001
"""Section 3.7: plain percentile bootstrap vs the small-seed-corrected version.

Recomputes every two-target contrast of Tables 4.4 and 4.6 and Section 4.3 (rotation 0, val)
with ``cmr.eval.stats.hierarchical_bootstrap``, with and without the small-seed corrections,
next to the interval printed in the report (from experiments/cloud/analyze_*.py). The last
column resamples seed s of both targets together (the matched design). Section 3.7 quotes the
two summary lines (corrected: narrower for 33 of 38; paired seeds: widths change <= 0.5 pp).
Needs the cloud predictions (release asset runs.tar.gz). Output: printed, and written to
results/analysis/bootstrap_check.txt unless --no-save.

    python experiments/analysis/bootstrap_check.py [--cloud-runs results/runs/cloud] [--no-save]
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

S3, S2 = (0, 1, 2), (0, 1)
B05, B7 = "G05_B_", "G7_B_"
AO, AR, RA = "answer_only", "answer_then_rationale", "rationale_then_answer"
BAL_AR05 = "G05_BAL_answer_then_rationale_full_rw0.3"
BAL_RA05 = "G05_BAL_rationale_then_answer_full_rw0.3"
BAL_AR7 = "G7_BAL_answer_then_rationale_n4000_rw0.3"
BAL_RA7 = "G7_BAL_rationale_then_answer_n4000_rw0.3"

# (label, run a, seeds a, run b, seeds b, report interval(s): "real" or "real | swapped")
CONTRASTS = [
    (
        "0.5B 4k: A→R − AO",
        f"{B05}{AR}_n4000",
        None,
        f"{B05}{AO}_n4000",
        None,
        "−9.3 [−11.9, −6.9] | −1.4 [−3.7, +0.9]",
    ),
    (
        "0.5B 4k: R→A − AO",
        f"{B05}{RA}_n4000",
        None,
        f"{B05}{AO}_n4000",
        None,
        "−6.4 [−9.1, −3.8] | −0.8 [−3.7, +2.1]",
    ),
    ("0.5B 4k: R→A − A→R", f"{B05}{RA}_n4000", None, f"{B05}{AR}_n4000", None, "+2.9 [+0.1, +5.8]"),
    (
        "0.5B 16k: A→R − AO",
        f"{B05}{AR}_full",
        None,
        f"{B05}{AO}_full",
        None,
        "−5.8 [−7.9, −3.7] | −1.4 [−3.6, +0.9]",
    ),
    (
        "0.5B 16k: R→A − AO",
        f"{B05}{RA}_full",
        None,
        f"{B05}{AO}_full",
        None,
        "−6.1 [−8.7, −3.5] | −1.0 [−4.0, +2.1]",
    ),
    ("0.5B 16k: R→A − A→R", f"{B05}{RA}_full", None, f"{B05}{AR}_full", None, "−0.3 [−3.0, +2.3]"),
    (
        "7B 4k: A→R − AO",
        f"{B7}{AR}_n4000",
        S3,
        f"{B7}{AO}_n4000",
        S3,
        "−1.6 [−3.1, −0.0] | −2.2 [−4.2, −0.0]",
    ),
    (
        "7B 4k: R→A − AO",
        f"{B7}{RA}_n4000",
        S3,
        f"{B7}{AO}_n4000",
        S3,
        "−4.1 [−6.1, −2.1] | −2.6 [−5.4, +0.3]",
    ),
    (
        "7B 16k: A→R − AO",
        f"{B7}{AR}_full",
        S3,
        f"{B7}{AO}_full",
        S3,
        "−1.8 [−3.6, +0.0] | −1.2 [−3.2, +0.8]",
    ),
    (
        "Data 0.5B (16k − 4k): AO",
        f"{B05}{AO}_full",
        None,
        f"{B05}{AO}_n4000",
        None,
        "+5.9 [+4.2, +7.7] | +1.7 [+0.1, +3.4]",
    ),
    (
        "Data 0.5B (16k − 4k): A→R",
        f"{B05}{AR}_full",
        None,
        f"{B05}{AR}_n4000",
        None,
        "+9.5 [+7.1, +11.8] | +1.7 [−0.5, +4.0]",
    ),
    (
        "Data 0.5B (16k − 4k): R→A",
        f"{B05}{RA}_full",
        None,
        f"{B05}{RA}_n4000",
        None,
        "+6.3 [+4.0, +8.5] | +1.5 [−0.8, +3.9]",
    ),
    ("Data 7B (16k − 4k): AO", f"{B7}{AO}_full", S3, f"{B7}{AO}_n4000", S3, "+1.4 [−0.2, +3.1]"),
    ("Data 7B (16k − 4k): A→R", f"{B7}{AR}_full", S3, f"{B7}{AR}_n4000", S3, "+1.3 [−0.3, +2.8]"),
    (
        "Scale 4k (7B − 0.5B): AO",
        f"{B7}{AO}_n4000",
        S3,
        f"{B05}{AO}_n4000",
        S3,
        "+20.5 [+17.7, +23.3]",
    ),
    (
        "Scale 4k (7B − 0.5B): A→R",
        f"{B7}{AR}_n4000",
        S3,
        f"{B05}{AR}_n4000",
        S3,
        "+27.7 [+24.6, +30.7]",
    ),
    (
        "Scale 4k (7B − 0.5B): R→A",
        f"{B7}{RA}_n4000",
        S3,
        f"{B05}{RA}_n4000",
        S3,
        "+22.4 [+19.3, +25.3]",
    ),
    (
        "Table 4.6, 0.5B: A→R default − AO",
        f"{B05}{AR}_full",
        S3,
        f"{B05}{AO}_full",
        S3,
        "−6.0 [−8.2, −3.7]",
    ),
    (
        "Table 4.6, 0.5B: A→R balanced − AO",
        BAL_AR05,
        S3,
        f"{B05}{AO}_full",
        S3,
        "−1.1 [−2.7, +0.5]",
    ),
    (
        "Table 4.6, 0.5B: A→R balanced − default",
        BAL_AR05,
        S3,
        f"{B05}{AR}_full",
        S3,
        "+4.9 [+2.9, +6.9]",
    ),
    (
        "Table 4.6, 0.5B: R→A balanced − AO",
        BAL_RA05,
        S3,
        f"{B05}{AO}_full",
        S3,
        "−6.3 [−8.8, −3.8]",
    ),
    (
        "Table 4.6, 0.5B: R→A balanced − default",
        BAL_RA05,
        S3,
        f"{B05}{RA}_full",
        S3,
        "+0.1 [−2.3, +2.4]",
    ),
    ("Table 4.6, 7B: A→R balanced − AO", BAL_AR7, S2, f"{B7}{AO}_n4000", S2, "+0.0 [−1.1, +1.2]"),
    (
        "Table 4.6, 7B: A→R balanced − default",
        BAL_AR7,
        S2,
        f"{B7}{AR}_n4000",
        S2,
        "+1.6 [+0.1, +3.1]",
    ),
    ("Table 4.6, 7B: R→A balanced − AO", BAL_RA7, S2, f"{B7}{AO}_n4000", S2, "−2.3 [−4.4, −0.2]"),
    (
        "Table 4.6, 7B: R→A balanced − default",
        BAL_RA7,
        S2,
        f"{B7}{RA}_n4000",
        S2,
        "+1.7 [−0.2, +3.5]",
    ),
    (
        "Section 4.3: tuned A→R (λ = 0) − tuned AO (4 epochs)",
        "G05_HP_answer_then_rationale_rw0",
        S2,
        "G05_HP_answer_only_ep4",
        S2,
        "−2.3 [−4.5, −0.2]",
    ),
    (
        "Section 4.3: tuned R→A (lr 3e-4) − tuned AO (4 epochs)",
        "G05_HP_rationale_then_answer_lr3e-4",
        S2,
        "G05_HP_answer_only_ep4",
        S2,
        "−6.3 [−9.2, −3.3]",
    ),
]


def load(run_dir: Path, cond: str) -> dict[str, float]:
    """{question_id: correct} at rotation 0, no context."""
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


def matrix(root: Path, prefix: str, seeds: tuple[int, ...] | None, cond: str) -> np.ndarray:
    """(n_seeds, n_items) correctness; ``seeds=None`` takes every seed present."""
    if seeds is None:
        seeds = tuple(sorted(int(p.name.rsplit("_s", 1)[1]) for p in root.glob(f"{prefix}_s?")))
    rows = [load(root / f"{prefix}_s{s}", cond) for s in seeds]
    qids = sorted(rows[0])
    return np.array([[r[q] for q in qids] for r in rows])


def paired_seeds(x: np.ndarray, y: np.ndarray, n_boot: int = 4000) -> tuple[float, float]:
    """Percentile interval when seed s of both targets is resampled together (matched design)."""
    rng = np.random.default_rng(0)
    out = np.empty(n_boot)
    for i in range(n_boot):
        it = rng.integers(0, x.shape[1], x.shape[1])
        sd = rng.integers(0, x.shape[0], x.shape[0])
        out[i] = x[sd][:, it].mean() - y[sd][:, it].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(r: object, lo: float, hi: float) -> str:
    return f"{100 * r.mean_diff:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"  # type: ignore[attr-defined]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cloud-runs", type=Path, default=ROOT / "results/runs/cloud")
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args()
    lines = [
        "Contrast (pp)  |  report (plain, analyze_*.py)  |  plain  |  corrected (t, df)"
        "  |  plain with seeds resampled in pairs (same seed ids only)",
        "",
    ]
    widths, pair_gain = [], []
    for label, a, sa, b, sb, report in CONTRASTS:
        for cond, rep in zip(("real", "swapped"), report.split(" | "), strict=False):
            x, y = matrix(args.cloud_runs, a, sa, cond), matrix(args.cloud_runs, b, sb, cond)
            plain = hierarchical_bootstrap(x, y, n_boot=4000, seed=0, small_seed_correction=False)
            corr = hierarchical_bootstrap(x, y, n_boot=4000, seed=0)
            widths.append((corr.high - corr.low) - (plain.high - plain.low))
            pair = ""
            if sa == sb:
                lo, hi = paired_seeds(x, y)
                pair_gain.append((plain.high - plain.low) - (hi - lo))
                pair = f"   paired {fmt(plain, lo, hi)}"
            lines.append(
                f"{label} ({cond}, n = {len(x)}, {len(y)})\n    report {rep}   plain "
                f"{fmt(plain, plain.low, plain.high)}   corrected {fmt(corr, corr.low, corr.high)}"
                f" (df {corr.df:.0f}){pair}"
            )
    narrower = sum(w < 0 for w in widths)
    lines += [
        "",
        f"corrected interval narrower than plain for {narrower} of {len(widths)} contrasts",
        f"paired-seed interval narrower than plain for {sum(g > 0 for g in pair_gain)} of "
        f"{len(pair_gain)} contrasts; width change {100 * min(pair_gain):+.1f} to "
        f"{100 * max(pair_gain):+.1f} pp",
    ]
    text = "\n".join(lines)
    print(text)
    if not args.no_save:
        (ROOT / "results/analysis/bootstrap_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
