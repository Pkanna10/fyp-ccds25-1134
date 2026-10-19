"""Training-free decision rules on saved predictions (Table 4.10 and Section 4.7).

Columns (real images, means over seeds): rot0 = accuracy in the original option order;
4-rot-ens = the 4-order ensemble (option log-probabilities averaged over the four orders, the
"4-order ensemble" of Table 4.10); prior-sub(cv) = the original-order score minus alpha times
the swapped-image score, an estimate of the language prior; both(cv) = the same with both
scores averaged over the four orders; alpha = mean fitted weight (0 to 1 in steps of 0.05, by
two-fold cross-fitting on val as in Section 3.7). The prior subtraction did not help
consistently (Section 4.7) and was not adopted. R->A is scored in the original order only, so
its ensemble equals rot0.
Inputs: predictions.jsonl of the cloud G*_B_* runs (release assets). Output: printed;
results/analysis/order_average.txt is this output.

    python experiments/laptop/order_average.py results/runs/cloud
"""

import json, glob, sys, collections
import numpy as np

root = sys.argv[1]
rng = np.random.default_rng(0)


def load(d):
    L = collections.defaultdict(lambda: collections.defaultdict(dict))
    gold = {}
    for l in open(f"{d}/eval_val/predictions.jsonl"):
        r = json.loads(l)
        if r.get("context", "none") != "none":
            continue
        L[r["condition"]][r["question_id"]][r["rotation"]] = np.array(r["option_logprobs"])
        gold[r["question_id"]] = r["correct_idx"]
    return L, gold


def acc(score, gold, qs):
    return np.mean([np.argmax(score[q]) == gold[q] for q in qs])


groups = collections.defaultdict(list)
for d in sorted(glob.glob(f"{root}/G*_B_*_s[0-9]")):
    groups[d.split("/")[-1][:-3]].append(d)
print(f"{'cell':40s} n  rot0   4-rot-ens   prior-sub(cv)  both(cv)  alpha")
for g, ds in groups.items():
    res = collections.defaultdict(list)
    alphas = []
    for d in ds:
        L, gold = load(d)
        qs = sorted(gold)
        r0 = {q: L["real"][q][0] for q in qs}
        ens = {q: np.mean([L["real"][q][k] for k in L["real"][q]], 0) for q in qs}
        sw0 = {q: L["swapped"][q][0] for q in qs}
        swe = {q: np.mean([L["swapped"][q][k] for k in L["swapped"][q]], 0) for q in qs}
        res["rot0"].append(acc(r0, gold, qs))
        res["ens"].append(acc(ens, gold, qs))
        # 2-fold cross-fitting for alpha: fit on one half, test on the other
        perm = rng.permutation(len(qs))
        halves = [[qs[i] for i in perm[: len(qs) // 2]], [qs[i] for i in perm[len(qs) // 2 :]]]
        grid = np.linspace(0, 1, 21)
        for base, prior, key in ((r0, sw0, "sub"), (ens, swe, "both")):
            tot = 0
            for a, b in ((0, 1), (1, 0)):
                best = max(
                    grid,
                    key=lambda al: acc(
                        {q: base[q] - al * prior[q] for q in halves[a]}, gold, halves[a]
                    ),
                )
                alphas.append(best)
                tot += acc(
                    {q: base[q] - best * prior[q] for q in halves[b]}, gold, halves[b]
                ) * len(halves[b])
            res[key].append(tot / len(qs))
    m = {k: 100 * np.mean(v) for k, v in res.items()}
    print(
        f"{g:40s} {len(ds)}  {m['rot0']:5.1f}  {m['ens']:5.1f}       {m['sub']:5.1f}          {m['both']:5.1f}    {np.mean(alphas):.2f}"
    )
