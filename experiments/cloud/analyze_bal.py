"""Balanced-loss phase (lambda = 0.3): the cells of Table 4.6 and its bootstrap contrasts.

Val, original option order (rotation 0), matched seeds (0.5B full data: 0-2; 7B 4k: 0-1).
Prints each cell's real, swapped and gap accuracy as mean +/- 95% t-interval over seeds (the
SDs of Table 4.6 come from scripts/make_report_tables.py), then the "vs AO" and "vs default"
contrasts with the paired bootstrap of Section 3.7, using the code of analyze_phase2_hp.py.
Inputs: the cloud run folders with predictions.jsonl (release asset).
Output: printed; results/analysis/analyze_bal.txt is this output.

    python experiments/cloud/analyze_bal.py results/runs/cloud
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase2_hp import boot, get, row  # noqa: E402  (loads the runs under sys.argv[1])

print(f"{'cell':34s} n  real          swapped       real−swapped")
for lab, pre, seeds in [
    ("0.5B full AO", "G05_B_answer_only_full", (0, 1, 2)),
    ("0.5B full A→R default", "G05_B_answer_then_rationale_full", (0, 1, 2)),
    ("0.5B full A→R balanced λ=0.3", "G05_BAL_answer_then_rationale_full_rw0.3", (0, 1, 2)),
    ("0.5B full R→A default", "G05_B_rationale_then_answer_full", (0, 1, 2)),
    ("0.5B full R→A balanced λ=0.3", "G05_BAL_rationale_then_answer_full_rw0.3", (0, 1, 2)),
    ("7B 4k AO", "G7_B_answer_only_n4000", (0, 1)),
    ("7B 4k A→R default", "G7_B_answer_then_rationale_n4000", (0, 1)),
    ("7B 4k A→R balanced λ=0.3", "G7_BAL_answer_then_rationale_n4000_rw0.3", (0, 1)),
    ("7B 4k R→A default", "G7_B_rationale_then_answer_n4000", (0, 1)),
    ("7B 4k R→A balanced λ=0.3", "G7_BAL_rationale_then_answer_n4000_rw0.3", (0, 1)),
]:
    print(row(lab, get(pre, seeds)))
print("\nContrasts (bootstrap over items and seeds, matched seeds)")
for lab, a, b, seeds in [
    (
        "0.5B full: A→R balanced − AO",
        "G05_BAL_answer_then_rationale_full_rw0.3",
        "G05_B_answer_only_full",
        (0, 1, 2),
    ),
    (
        "0.5B full: A→R balanced − A→R default",
        "G05_BAL_answer_then_rationale_full_rw0.3",
        "G05_B_answer_then_rationale_full",
        (0, 1, 2),
    ),
    (
        "0.5B full: R→A balanced − AO",
        "G05_BAL_rationale_then_answer_full_rw0.3",
        "G05_B_answer_only_full",
        (0, 1, 2),
    ),
    (
        "0.5B full: R→A balanced − R→A default",
        "G05_BAL_rationale_then_answer_full_rw0.3",
        "G05_B_rationale_then_answer_full",
        (0, 1, 2),
    ),
    (
        "7B 4k: A→R balanced − AO",
        "G7_BAL_answer_then_rationale_n4000_rw0.3",
        "G7_B_answer_only_n4000",
        (0, 1),
    ),
    (
        "7B 4k: A→R balanced − A→R default",
        "G7_BAL_answer_then_rationale_n4000_rw0.3",
        "G7_B_answer_then_rationale_n4000",
        (0, 1),
    ),
    (
        "7B 4k: R→A balanced − AO",
        "G7_BAL_rationale_then_answer_n4000_rw0.3",
        "G7_B_answer_only_n4000",
        (0, 1),
    ),
    (
        "7B 4k: R→A balanced − R→A default",
        "G7_BAL_rationale_then_answer_n4000_rw0.3",
        "G7_B_rationale_then_answer_n4000",
        (0, 1),
    ),
]:
    print(
        f"{lab:42s} real {boot(get(a, seeds), get(b, seeds))}   swapped {boot(get(a, seeds), get(b, seeds), 'swapped')}"
    )
