"""7B and hyperparameter-study analysis: Table 4.4 (7B and scale rows) and Table 4.5.

All accuracies are on val in the original option order (rotation 0). Prints, in order:
  - 7B theta_0 and every main cell on seeds 0-2 (mean +/- 95% t-interval over seeds);
  - with the paired bootstrap of Section 3.7: the 7B target contrasts, next to the same 0.5B
    contrasts on seeds 0-2, the 7B data-size contrasts and the 7B - 0.5B scale contrasts;
  - the hyperparameter study (Table 4.5, mean of seeds 0-1): dev accuracy from the
    end-of-training dev check in train_log.jsonl, val real / swapped / gap per setting
    ("base" = the main 4k runs), the setting chosen on dev per target, and the tuned contrasts.
Setting tags: lr<x> = learning rate x, bs64 = batch 64 with learning rate 4e-4, ep4 = 4
epochs, rw<x> = balanced loss with lambda = x.
Inputs: the cloud run folders with predictions.jsonl and train_log.jsonl (release assets).
Output: printed; results/analysis/analyze_phase2_hp.txt is this output. analyze_bal.py imports
this module, which loads the runs under sys.argv[1] on import.

    python experiments/cloud/analyze_phase2_hp.py results/runs/cloud
"""

import json, glob, os, re, sys
import numpy as np
from math import sqrt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/runs/cloud"
ARMS = {"answer_only": "AO", "answer_then_rationale": "A→R", "rationale_then_answer": "R→A"}
TQ = {2: 12.706, 3: 4.303, 5: 2.776}  # t(0.975, n - 1) for n seeds
rng = np.random.default_rng(0)


def load(d):
    p = f"{d}/eval_val/predictions.jsonl"
    rot0 = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("context", "none") == "none" and r.get("rotation", 0) == 0:
            rot0.setdefault(r["condition"], {})[r["question_id"]] = int(
                r["pred_idx"] == r["correct_idx"]
            )
    dev = None
    tl = f"{d}/train_log.jsonl"
    if os.path.exists(tl):
        for l in open(tl):
            j = json.loads(l)
            if "dev_acc" in j:
                dev = j["dev_acc"]
    sj = json.load(open(f"{d}/eval_val/summary.json"))
    accP = [r for r in sj["rows"] if r["condition"] == "real"][0].get("acc_prior_wrong")
    return dict(rot0=rot0, dev=dev, accP=accP)


R = {
    os.path.basename(d): load(d)
    for d in sorted(glob.glob(f"{ROOT}/G*"))
    if os.path.exists(f"{d}/eval_val/predictions.jsonl")
}
qids = sorted(R["G05_B_answer_only_n4000_s0"]["rot0"]["real"])


def vec(r, c="real"):
    return np.array([r["rot0"][c][q] for q in qids], float)


def get(prefix, seeds):
    return [R[f"{prefix}_s{s}"] for s in seeds if f"{prefix}_s{s}" in R]


def ms(x):
    x = np.array(x, float)
    if len(x) == 1:
        return f"{100 * x[0]:5.1f}       "
    h = TQ[len(x)] * x.std(ddof=1) / sqrt(len(x))
    return f"{100 * x.mean():5.1f} ±{100 * h:4.1f}"


def boot(A, B, c="real", n=4000):
    """Paired bootstrap (Section 3.7): questions paired across A and B, seeds within each."""
    a = np.stack([vec(r, c) for r in A])
    b = np.stack([vec(r, c) for r in B])
    est = a.mean() - b.mean()
    out = np.empty(n)
    nI = a.shape[1]
    for i in range(n):
        it = rng.integers(0, nI, nI)
        out[i] = (
            a[rng.integers(0, len(a), len(a))][:, it].mean()
            - b[rng.integers(0, len(b), len(b))][:, it].mean()
        )
    lo, hi = np.percentile(out, [2.5, 97.5])
    return f"{100 * est:+5.1f} pp [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"


def row(label, rs):
    real = [vec(r).mean() for r in rs]
    sw = [vec(r, "swapped").mean() for r in rs]
    return f"{label:26s} {len(rs)}  {ms(real)}  {ms(sw)}  {ms([a - b for a, b in zip(real, sw)])}"


def main():
    print("## 7B vs 0.5B, seeds 0-2 (± = 95% t-interval over seeds)")
    print(f"{'cell':26s} n  real          swapped       real−swapped")
    for k in ("answer_only", "answer_then_rationale", "rationale_then_answer"):
        t = R.get(f"G7_theta0_{k}")
        if t:
            print(
                f"{'7B θ0 ' + ARMS[k]:26s} -  {100 * vec(t).mean():5.1f}         {100 * vec(t, 'swapped').mean():5.1f}"
                if "swapped" in t["rot0"]
                else f"{'7B θ0 ' + ARMS[k]:26s} -  {100 * vec(t).mean():5.1f}"
            )
    for size, data in (("G05", "n4000"), ("G7", "n4000"), ("G05", "full"), ("G7", "full")):
        for k in ARMS:
            rs = get(f"{size}_B_{k}_{data}", (0, 1, 2))
            if rs:
                print(row(f"{'0.5B' if size == 'G05' else '7B'} {data} {ARMS[k]}", rs))
    print()
    for data in ("n4000", "full"):
        ao = get(f"G7_B_answer_only_{data}", (0, 1, 2))
        for k in ("answer_then_rationale", "rationale_then_answer"):
            x = get(f"G7_B_{k}_{data}", (0, 1, 2))
            if x:
                print(
                    f"7B {data:5s} {ARMS[k]} − AO: real {boot(x, ao)}   swapped {boot(x, ao, 'swapped')}"
                )
                y5 = get(f"G05_B_{k}_{data}", (0, 1, 2))
                ao5 = get(f"G05_B_answer_only_{data}", (0, 1, 2))
                print(f"   0.5B same seeds 0-2: {boot(y5, ao5)}")
    for k in ("answer_only", "answer_then_rationale"):
        print(
            f"7B full − n4000 {ARMS[k]}: real {boot(get(f'G7_B_{k}_full', (0, 1, 2)), get(f'G7_B_{k}_n4000', (0, 1, 2)))}"
        )
    for k in ARMS:
        print(
            f"7B − 0.5B at n4000 {ARMS[k]}: {boot(get(f'G7_B_{k}_n4000', (0, 1, 2)), get(f'G05_B_{k}_n4000', (0, 1, 2)))}"
        )

    print("\n## Hyperparameters: 0.5B, 4k items, seeds 0-1 (baseline lr1e-4 = phase-1 runs s0-1)")
    print(f"{'arm':4s} {'setting':9s}  dev acc   val real   val swapped  real−sw")
    best = {}
    for k, a in ARMS.items():
        sets = [("base", f"G05_B_{k}_n4000")] + [
            (
                re.sub(rf"G05_HP_{k}_|_s0$", "", os.path.basename(p)),
                p.rstrip("/").split("/")[-1][:-3],
            )
            for p in sorted(glob.glob(f"{ROOT}/G05_HP_{k}_*_s0"))
        ]
        for tag, pre in sets:
            rs = get(pre, (0, 1))
            dev = np.mean([r["dev"] for r in rs])
            real = [vec(r).mean() for r in rs]
            sw = [vec(r, "swapped").mean() for r in rs]
            print(
                f"{a:4s} {tag:9s}  {100 * dev:5.1f}    {100 * np.mean(real):5.1f}      {100 * np.mean(sw):5.1f}       {100 * (np.mean(real) - np.mean(sw)):5.1f}"
            )
            if k not in best or dev > best[k][1]:
                best[k] = (tag, dev, rs)
        print()
    print("Best setting per arm, chosen on dev:", {ARMS[k]: v[0] for k, v in best.items()})
    for k in ("answer_then_rationale", "rationale_then_answer"):
        print(f"tuned {ARMS[k]} − tuned AO (val real): {boot(best[k][2], best['answer_only'][2])}")


if __name__ == "__main__":
    main()
