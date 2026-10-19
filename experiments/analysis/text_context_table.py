"""Table 4.7 and Section 4.6: the frozen LLM answering from text context alone.

Rotation-0 accuracy (Wilson 95% CI) of the frozen 0.5B and 7B LLMs, no image, with context
none / COCO caption / masked rationale / rationale, and the rationale rows split by the
``rationale_leak`` flag (the correct option's text appears in at least one human rationale).
With the A-OKVQA val file, the no-leak questions are split once more by whether a rationale
contains a content-word stem of the correct option. Recomputed from predictions.jsonl
(release asset) and checked against summary.json; without predictions the table comes from
summary.json and the split is skipped. The two runs are the text-only check runs: 0.5B on CPU
(pilot_a1v2_instrument, which also defines the prior-wrong subset P) and 7B in the cloud
(G7_A1_instrument).

    python experiments/analysis/text_context_table.py
        [--run-05b results/runs/laptop/pilot_a1v2_instrument]
        [--run-7b results/runs/cloud/G7_A1_instrument] [--aokvqa-dir data/raw/aokvqa] [--no-save]

Output: printed, and saved to results/analysis/text_context_table.txt unless --no-save.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmr.analysis.reliance import summarize  # noqa: E402
from cmr.data.aokvqa import load_aokvqa  # noqa: E402
from cmr.eval.prompts import MASK_VERSION, join_rationales, leaks  # noqa: E402
from cmr.fsutil import read_jsonl  # noqa: E402

CONTEXTS = (
    ("none", "none"),
    ("caption", "COCO caption"),
    ("rationale_masked", "human rationale, answer words masked"),
    ("rationale", "human rationale"),
)


def context_rows(run: Path, records: list[dict[str, Any]]) -> tuple[dict[str, dict], list[str]]:
    """Per-context rows (recomputed when records exist, else summary.json) and check notes."""
    saved = {r["context"]: r for r in json.loads((run / "summary.json").read_text())["rows"]}
    if not records:
        return saved, ["predictions.jsonl not found: accuracies read from summary.json"]
    rows = {r["context"]: r for r in summarize(records, reference=("none", "none"))["rows"]}
    notes = []
    for ctx, _ in CONTEXTS:
        same = rows[ctx]["n"] == saved[ctx]["n"] and math.isclose(
            rows[ctx]["acc"], saved[ctx]["acc"], abs_tol=1e-12
        )
        if not same:
            notes.append(f"MISMATCH with summary.json for context {ctx}")
    if not notes:
        notes.append("recomputed accuracies equal summary.json for all four contexts")
    versions = {r.get("mask_version") for r in records if r["context"] == "rationale_masked"}
    notes.append(
        f"masking version of the rationale_masked records: {sorted(versions)} "
        f"(library: {MASK_VERSION})"
    )
    return rows, notes


def leak_split(
    records: list[dict[str, Any]], stem_leak: set[str] | None = None
) -> list[tuple[str, int, float, float]]:
    """``(label, n, acc full, acc masked)`` for the leak and no-leak subsets (rotation 0).

    With ``stem_leak`` (question ids whose rationales contain a content-word stem of the
    correct option, the masker's matcher), the no-leak subset is split once more.
    """
    by: dict[str, dict[str, dict[str, Any]]] = {"rationale": {}, "rationale_masked": {}}
    flag: dict[str, set[bool]] = {}
    for r in records:
        flag.setdefault(r["question_id"], set()).add(bool(r["rationale_leak"]))
        if r["rotation"] == 0 and r["condition"] == "none" and r["context"] in by:
            by[r["context"]][r["question_id"]] = r
    if any(len(v) != 1 for v in flag.values()):
        raise ValueError("rationale_leak differs between records of the same question")
    common = sorted(set(by["rationale"]) & set(by["rationale_masked"]))
    groups = [
        ("correct option's text in a rationale", [q for q in common if flag[q] == {True}]),
        ("not in any", [q for q in common if flag[q] == {False}]),
    ]
    if stem_leak is not None:
        groups += [
            (
                "  of which: a stem of it in a rationale",
                [q for q in groups[1][1] if q in stem_leak],
            ),
            ("  of which: no stem either", [q for q in groups[1][1] if q not in stem_leak]),
        ]
    out = []
    for label, qs in groups:
        acc = [
            sum(by[c][q]["pred_idx"] == by[c][q]["correct_idx"] for q in qs) / len(qs)
            for c in ("rationale", "rationale_masked")
        ]
        out.append((label, len(qs), acc[0], acc[1]))
    return out


def pct_ci(row: dict[str, Any]) -> str:
    """'55.9 [53.0, 58.7]' from a summary row (acc and its Wilson interval, as fractions)."""
    lo, hi = row["acc_ci"]
    return f"{100 * row['acc']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def report(runs: dict[str, Path], stem_leak: set[str] | None = None) -> str:
    """The printed report, as one string."""
    records = {k: read_jsonl(p / "predictions.jsonl") for k, p in runs.items()}
    rows, notes = {}, []
    for k, p in runs.items():
        rows[k], run_notes = context_rows(p, records[k])
        notes += [f"{k} ({p.name}): {m}" for m in run_notes]

    out = [
        "Table 4.7: frozen LLM with text context only (no image, no training)",
        "accuracy % at rotation 0 [Wilson 95% CI]",
        "",
    ]
    out.append(f"  {'context given as text':38s}" + "".join(f"{k:>22s}" for k in runs))
    for ctx, label in CONTEXTS:
        cells = [pct_ci(rows[k][ctx]) for k in runs]
        out.append(f"  {label:38s}" + "".join(f"{c:>22s}" for c in cells))
    n = {k: rows[k]["none"]["n"] for k in runs}
    out.append(f"  {'questions (n)':38s}" + "".join(f"{n[k]:>22d}" for k in runs))

    out += ["", "Section 4.6: full vs masked rationale, split by the rationale_leak flag"]
    for k in runs:
        if not records[k]:
            out.append(f"  {k}: skipped (no predictions.jsonl)")
            continue
        for label, m, full, masked in leak_split(records[k], stem_leak):
            out.append(
                f"  {k:5s} {label:40s} n = {m:4d}   full {100 * full:5.1f}%   "
                f"masked {100 * masked:5.1f}%   drop {100 * (full - masked):5.1f} pp"
            )
    out += ["", "checks:"] + [f"  {m}" for m in notes]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-05b",
        type=Path,
        default=ROOT / "results/runs/laptop/pilot_a1v2_instrument",
        help="text-only run of the 0.5B LLM",
    )
    ap.add_argument(
        "--run-7b",
        type=Path,
        default=ROOT / "results/runs/cloud/G7_A1_instrument",
        help="text-only run of the 7B LLM",
    )
    ap.add_argument(
        "--aokvqa-dir",
        type=Path,
        default=ROOT / "data/raw/aokvqa",
        help="A-OKVQA folder; without it the stem split is skipped",
    )
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args(argv)

    stem_leak = None
    if (args.aokvqa_dir / "aokvqa_v1p0_val.json").exists():
        stem_leak = {
            it.question_id
            for it in load_aokvqa(args.aokvqa_dir, "val")
            if leaks(join_rationales(it.rationales), [it.correct_choice])
        }
    text = report({"0.5B": args.run_05b, "7B": args.run_7b}, stem_leak)
    print(text)
    if not args.no_save:
        out = ROOT / "results/analysis/text_context_table.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print(f"\nsaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
