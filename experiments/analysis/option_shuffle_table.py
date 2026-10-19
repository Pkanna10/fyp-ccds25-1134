"""Table 4.11: option-shuffled training against the same setting trained in the original order.

Option-shuffled training is proposal A of Table 6.1 (Section 4.7). Compares the laptop runs
mac_AB/AO_shuf_n4000_s{0,1} with the cloud runs G05_B_answer_only_n4000_s{0,1}, and
mac_AB/ATR_shuf_rw0.3_n4000_s0 with the cloud hyperparameter-study run
G05_HP_answer_then_rationale_rw0.3_s0 (A→R, balanced loss, lambda = 0.3). Measures, with an
item bootstrap of each difference: rot0 = accuracy in the original order, mean_rot = mean
accuracy over the 4 orders, perm = the 4-order ensemble, circ = correct in all 4 orders, flip
= share of questions whose answer changes between orders. The last block (AO, seed-averaged
per-question scores) gives the AO rows of Table 4.11; the rot0 line of the A→R pair its last.
Inputs: predictions.jsonl of these runs (release assets). Output: printed;
results/analysis/option_shuffle_table.txt is this output.

    python experiments/analysis/option_shuffle_table.py [CLOUD_RUNS] [MAC_AB_RUNS]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(p):
    R = defaultdict(dict)  # condition -> question_id -> {rotation: prediction record}
    with open(p) as f:
        lines = f.readlines()
    for line in lines:
        r = json.loads(line)
        if r["context"] != "none":
            continue
        R[r["condition"]].setdefault(r["question_id"], {})[r["rotation"]] = r
    return R


def metrics(R, cond="real"):
    d = R[cond]
    q = sorted(d)
    rot0 = np.array([d[k][0]["pred_idx"] == d[k][0]["correct_idx"] for k in q], float)
    rots = sorted(d[q[0]])
    allr = np.array(
        [[d[k][r]["pred_idx"] == d[k][r]["correct_idx"] for r in rots] for k in q], float
    )
    # perm: argmax of the option log-probs averaged over rotations (4-order ensemble);
    # flip: the predicted option differs between rotations
    perm = []
    flip = []
    for k in q:
        lp = np.mean([d[k][r]["option_logprobs"] for r in rots], 0)
        perm.append(np.argmax(lp) == d[k][0]["correct_idx"])
        preds = {d[k][r]["pred_idx"] for r in rots}
        flip.append(len(preds) > 1)
    return q, dict(
        rot0=rot0,
        mean_rot=allr.mean(1),
        circ=allr.min(1),
        perm=np.array(perm, float),
        flip=np.array(flip, float),
    )


def boot(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(a)
    d = a - b
    s = [d[rng.integers(0, N, N)].mean() for _ in range(n)]
    return 100 * d.mean(), 100 * np.percentile(s, 2.5), 100 * np.percentile(s, 97.5)


ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument(
    "cloud", nargs="?", default=str(ROOT / "results/runs/cloud"), help="cloud run folders"
)
ap.add_argument(
    "laptop",
    nargs="?",
    default=str(ROOT / "results/runs/laptop/mac_AB"),
    help="laptop run folders of experiments/laptop/run_mac_AB.sh",
)
args = ap.parse_args()
B, A = args.cloud + "/", args.laptop + "/"
pairs = [
    ("AO 4k s0", A + "AO_shuf_n4000_s0", B + "G05_B_answer_only_n4000_s0"),
    ("AO 4k s1", A + "AO_shuf_n4000_s1", B + "G05_B_answer_only_n4000_s1"),
    ("A→R λ0.3 4k s0", A + "ATR_shuf_rw0.3_n4000_s0", B + "G05_HP_answer_then_rationale_rw0.3_s0"),
]
pooled = defaultdict(lambda: [[], []])
for name, m, b in pairs:
    Rm = load(m + "/eval_val/predictions.jsonl")
    Rb = load(b + "/eval_val/predictions.jsonl")
    qm, M = metrics(Rm)
    qb, Bm = metrics(Rb)
    assert qm == qb
    _, Ms = metrics(Rm, "swapped")
    _, Bs = metrics(Rb, "swapped")
    print(f"\n{name}:   shuffled vs original order")
    for k in ["rot0", "mean_rot", "perm", "circ", "flip"]:
        d, lo, hi = boot(M[k], Bm[k])
        means = f"{100 * M[k].mean():5.1f} vs {100 * Bm[k].mean():5.1f}"
        print(f"  {k:9s} {means}   diff {d:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
    print(f"  swapped   {100 * Ms['rot0'].mean():5.1f} vs {100 * Bs['rot0'].mean():5.1f}")
    if name.startswith("AO"):
        for k in M:
            pooled[k][0].append(M[k])
            pooled[k][1].append(Bm[k])
print("\nAO 4k, mean of seeds 0-1 (item bootstrap on the seed-averaged per-item score):")
for k in ["rot0", "mean_rot", "perm", "circ", "flip"]:
    a = np.mean(pooled[k][0], 0)
    b = np.mean(pooled[k][1], 0)
    m, lo, hi = boot(a, b)
    means = f"{100 * a.mean():5.1f} vs {100 * b.mean():5.1f}"
    print(f"  {k:9s} {means}   diff {m:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
