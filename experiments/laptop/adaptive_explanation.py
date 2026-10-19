"""Combined answer rule of Sections 3.6 and 4.6, evaluated on saved predictions.

For each val question (real image, original option order) the rule uses the R->A answer when
the R->A explanation names exactly one option (whole-word match), and otherwise the AO answer
of the same seed. Prints, for 0.5B full data and 4k (seeds 0-4) and 7B 4k (seeds 0-2), means
over seeds of: AO accuracy, R->A accuracy, the rule's accuracy ("adaptive"; Section 4.6 quotes
65.4% against 67.4% for AO) and the share of questions where it uses R->A ("named-share").
Inputs: predictions.jsonl of the cloud G*_B_* runs (release assets) and the A-OKVQA val file.
Output: printed; results/analysis/adaptive_explanation.txt is this output.

    python experiments/laptop/adaptive_explanation.py results/runs/cloud \\
        data/raw/aokvqa/aokvqa_v1p0_val.json
"""

import json, sys, re
import numpy as np

root, valp = sys.argv[1], sys.argv[2]
val = {q["question_id"]: q for q in json.load(open(valp))}


def preds(d, cond="real"):
    out = {}
    for l in open(f"{d}/eval_val/predictions.jsonl"):
        r = json.loads(l)
        if (
            r["condition"] == cond
            and r.get("rotation", 0) == 0
            and r.get("context", "none") == "none"
        ):
            out[r["question_id"]] = r
    return out


def named(gen, choices):
    g = " " + re.sub(r"[^a-z0-9 ]", " ", gen.lower()) + " "
    return [
        i
        for i, c in enumerate(choices)
        if " " + re.sub(r"[^a-z0-9 ]", " ", c.lower()).strip() + " " in g
    ]


for size, data, seeds in (
    ("G05", "full", range(5)),
    ("G05", "n4000", range(5)),
    ("G7", "n4000", range(3)),
):
    accs = {"AO": [], "R→A": [], "adaptive": [], "named-share": []}
    for s in seeds:
        ra = preds(f"{root}/{size}_B_rationale_then_answer_{data}_s{s}")
        ao = preds(f"{root}/{size}_B_answer_only_{data}_s{s}")
        ok_ao, ok_ra, ok_ad, nm = [], [], [], []
        for q, r in ra.items():
            n = named(r.get("generation", ""), val[q]["choices"])
            use_ra = len(n) == 1
            nm.append(use_ra)
            ok_ra.append(r["pred_idx"] == r["correct_idx"])
            ok_ao.append(ao[q]["pred_idx"] == ao[q]["correct_idx"])
            ok_ad.append(ok_ra[-1] if use_ra else ok_ao[-1])
        for k, v in (("AO", ok_ao), ("R→A", ok_ra), ("adaptive", ok_ad), ("named-share", nm)):
            accs[k].append(np.mean(v))
    print(f"{size} {data:5s}: " + "  ".join(f"{k} {100 * np.mean(v):.1f}" for k, v in accs.items()))
