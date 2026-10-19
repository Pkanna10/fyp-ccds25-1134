"""SigLIP fusion (Table 4.10, Figure 4.9): late fusion of letter log-probs with SigLIP scores.

score(option) = log p_LLM(option) + w * log-softmax(SigLIP logits)(option). w is chosen from 0
to 3 in steps of 0.1 by two-fold cross-fitting on val (Section 3.7): fitted on one random half
of val (the same halves for every run), applied to the other, and the halves swapped; w is
fitted per run, and accuracies are averaged over seeds. The LLM score is the original-order
log-probability ("LLM") or the 4-order ensemble, log-probabilities averaged over the four
option orders ("LLM-4ord"); each is shown alone and "+SigLIP". w = mean fitted weight; the
last column is the fusion gain in the original order with the paired bootstrap of Section 3.7.

    python experiments/laptop/fuse_siglip.py results/runs/cloud \\
        results/runs/laptop/siglip_val/scores.jsonl [option,photo,qa]

Inputs (release assets): predictions.jsonl of the cloud G*_B_* runs and the SigLIP scores
written by experiments/laptop/siglip_option_scores.py. The text template "qa" ("<question>
<option>") was the most accurate of the three on val (SigLIP alone) and is the default; the
optional third argument repeats the fusion with the others. Output: printed;
results/analysis/fuse_siglip.txt is the default output and fuse_siglip_templates.txt the
output with "option,photo,qa".
"""

import json, glob, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from cmr.eval.stats import hierarchical_bootstrap  # noqa: E402

root, sig = sys.argv[1], sys.argv[2]
templates = sys.argv[3].split(",") if len(sys.argv) > 3 else ["qa"]
S = {}
for l in open(sig):
    r = json.loads(l)
    S[r["question_id"]] = r


def lsm(x):
    x = np.array(x, float)
    return x - np.log(np.exp(x - x.max()).sum()) - x.max()


rng = np.random.default_rng(0)
qs = sorted(S)
perm = rng.permutation(len(qs))
halves = [set(qs[i] for i in perm[: len(qs) // 2]), set(qs[i] for i in perm[len(qs) // 2 :])]
grid = np.linspace(0, 3, 31)


def correct(base, sg, gold, w, sub):
    return [float(np.argmax(base[q] + w * sg[q]) == gold[q]) for q in sub]


def acc(base, sg, gold, w, sub):
    return np.mean(correct(base, sg, gold, w, sub))


for tpl in templates:
    sg = {q: lsm(S[q][tpl]) for q in qs}
    print(
        f"SigLIP alone ({tpl}): {100 * np.mean([np.argmax(sg[q]) == S[q]['correct_idx'] for q in qs]):.1f}"
    )
    groups = collections.defaultdict(list)
    for d in sorted(glob.glob(f"{root}/G*_B_*_s[0-9]")):
        groups[d.split("/")[-1][:-3]].append(d)
    print(f"{'cell':40s} n  LLM    +SigLIP   LLM-4ord  +SigLIP   w     gain at order 0 [95% CI]")
    for g, ds in groups.items():
        res = collections.defaultdict(list)
        ws = []
        items = collections.defaultdict(list)  # per-seed correctness on qs, for the interval
        for d in ds:
            L = collections.defaultdict(dict)
            gold = {}
            for l in open(f"{d}/eval_val/predictions.jsonl"):
                r = json.loads(l)
                if r["condition"] == "real" and r.get("context", "none") == "none":
                    L[r["question_id"]][r["rotation"]] = np.array(r["option_logprobs"])
                    gold[r["question_id"]] = r["correct_idx"]
            r0 = {q: L[q][0] for q in qs}
            ens = {q: np.mean(list(L[q].values()), 0) for q in qs}
            for key, base in (("r0", r0), ("ens", ens)):
                res[key].append(np.mean([np.argmax(base[q]) == gold[q] for q in qs]))
                fused = {}
                for a, b in ((0, 1), (1, 0)):
                    w = max(grid, key=lambda w: acc(base, sg, gold, w, halves[a]))
                    ws.append(w)
                    fused.update(zip(sorted(halves[b]), correct(base, sg, gold, w, sorted(halves[b]))))
                res[key + "+s"].append(np.mean(list(fused.values())))
                if key == "r0":
                    items["base"].append([float(np.argmax(base[q]) == gold[q]) for q in qs])
                    items["fused"].append([fused[q] for q in qs])
        m = {k: 100 * np.mean(v) for k, v in res.items()}
        # paired percentile bootstrap over items and seeds, as for the target contrasts
        hb = hierarchical_bootstrap(
            items["fused"], items["base"], n_boot=4000, seed=0, small_seed_correction=False
        )
        print(
            f"{g:40s} {len(ds)}  {m['r0']:5.1f}  {m['r0+s']:5.1f}     {m['ens']:5.1f}    {m['ens+s']:5.1f}    {np.mean(ws):.2f}"
            f"  {100 * hb.mean_diff:+.1f} [{100 * hb.low:+.1f}, {100 * hb.high:+.1f}]"
        )
