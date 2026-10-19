"""Main-run analysis: accuracy by LLM size, training-set size and target (Tables 4.1-4.4).

All accuracies are on val in the original option order (rotation 0), with real and swapped
images. Prints, in order:
  - theta_0 (the untrained bridge) under each target's instruction (Table 4.1);
  - every trained cell: mean +/- 95% t-interval over seeds of real, swapped, gap, accuracy on
    the prior-wrong subset P and the 4-order mean (Table 4.3 gives SDs instead, from
    scripts/make_report_tables.py); the t-interval assumes 5 seeds, so the 3-seed 7B cells
    print "nan" (their intervals come from analyze_phase2_hp.py);
  - target, data-size and interaction contrasts with the paired bootstrap of Section 3.7
    (Table 4.4, 0.5B rows; the 7B rows come from analyze_phase2_hp.py), and a superseded
    scale contrast on 0.5B seed 0 only, kept so that the later random draws are unchanged;
  - the earlier CPU / laptop runs results/runs/laptop/B_*_n4000_s* (Table 4.2). Without their
    predictions.jsonl this block prints n=0, as in results/analysis/analyze_phase1.txt; their
    accuracies are also in each run's eval_val/summary.json.
Inputs: the cloud run folders with predictions.jsonl and train_log.jsonl (release assets).
Output: printed; results/analysis/analyze_phase1.txt is this output.

    python experiments/cloud/analyze_phase1.py results/runs/cloud
"""

import json, glob, os, re, sys
import numpy as np
from math import sqrt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/runs/cloud"
ARMS = {"answer_only": "AO", "answer_then_rationale": "A→R", "rationale_then_answer": "R→A"}
T4 = 2.776  # t(0.975, df=4): 95% t-interval over 5 seeds


def load_preds(path):
    """({cond: {qid: correct}} at rotation 0, {cond: rotation-averaged acc}), context none."""
    rot0, allrot = {}, {}
    for l in open(path):
        r = json.loads(l)
        if r.get("context", "none") != "none":
            continue
        ok = int(r["pred_idx"] == r["correct_idx"])
        allrot.setdefault(r["condition"], []).append(ok)
        if r.get("rotation", 0) == 0:
            rot0.setdefault(r["condition"], {})[r["question_id"]] = ok
    return rot0, {c: float(np.mean(v)) for c, v in allrot.items()}


def pfile(d):
    for p in (f"{d}/eval_val/predictions.jsonl", f"{d}/predictions.jsonl"):
        if os.path.exists(p):
            return p


runs = {}
for d in sorted(glob.glob(f"{ROOT}/G*_B_*")):
    m = re.match(r".*/(G05|G7)_B_(\w+?)_(n4000|full)_s(\d)$", d)
    size, arm, data, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
    rot0, avg = load_preds(pfile(d))
    sj = json.load(open(f"{d}/eval_val/summary.json"))
    real = [r for r in sj["rows"] if r["condition"] == "real"][0]
    runs[(size, data, ARMS[arm], seed)] = dict(
        rot0=rot0,
        avg=avg,
        accP=real.get("acc_prior_wrong"),
        steps=sum(1 for _ in open(f"{d}/train_log.jsonl")),
    )
theta = {}
for d in sorted(glob.glob(f"{ROOT}/G*_theta0_*")):
    m = re.match(r".*/(G05|G7)_theta0_(\w+)$", d)
    rot0, avg = load_preds(pfile(d))
    theta[(m.group(1), ARMS[m.group(2)])] = dict(rot0=rot0, avg=avg)

qids = sorted(next(iter(runs.values()))["rot0"]["real"].keys())


def vec(run, cond):
    return np.array([run["rot0"][cond][q] for q in qids], float)


def cell(size, data, arm):
    seeds = sorted(s for (z, dd, a, s) in runs if (z, dd, a) == (size, data, arm))
    return [runs[(size, data, arm, s)] for s in seeds], seeds


def msd(x):
    x = np.array(x, float)
    if len(x) < 2:
        return f"{100 * x.mean():5.1f}      "
    h = T4 * x.std(ddof=1) / sqrt(len(x)) if len(x) == 5 else float("nan")
    return f"{100 * x.mean():5.1f} ±{100 * h:4.1f}"


rng = np.random.default_rng(0)


def boot_diff(A, B, cond="real", n=4000):
    """Paired bootstrap (Section 3.7): questions paired across targets, seeds within each."""
    a = np.stack([vec(r, cond) for r in A])
    b = np.stack([vec(r, cond) for r in B])
    est = a.mean() - b.mean()
    nI = a.shape[1]
    out = np.empty(n)
    for i in range(n):
        it = rng.integers(0, nI, nI)
        sa = rng.integers(0, len(a), len(a))
        sb = rng.integers(0, len(b), len(b))
        out[i] = a[sa][:, it].mean() - b[sb][:, it].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return f"{100 * est:+5.1f} pp [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"


print("## θ0 (untrained bridge), rotation 0")
for (size, arm), t in sorted(theta.items()):
    print(
        f"{size:4s} {arm:4s} real {100 * np.mean(list(t['rot0']['real'].values())):5.1f}  "
        + "  ".join(
            f"{c} {100 * np.mean(list(v.values())):5.1f}"
            for c, v in t["rot0"].items()
            if c != "real"
        )
    )

print("\n## Trained cells, rotation 0, mean ± 95% t-CI over seeds (n=5 unless shown)")
print(
    "size data   arm   n  real          swapped       real−swapped  Acc|P(real)  rot-avg real  steps"
)
for size in ("G05", "G7"):
    for data in ("n4000", "full"):
        for arm in ("AO", "A→R", "R→A"):
            rs, seeds = cell(size, data, arm)
            if not rs:
                continue
            real = [vec(r, "real").mean() for r in rs]
            sw = [vec(r, "swapped").mean() for r in rs]
            gap = [a - b for a, b in zip(real, sw)]
            accp = [r["accP"] for r in rs if r["accP"] is not None]
            ravg = [r["avg"]["real"] for r in rs]
            print(
                f"{size:4s} {data:6s} {arm:4s} {len(rs)}  {msd(real)}  {msd(sw)}  {msd(gap)}  "
                f"{msd(accp) if accp else '   -        '}  {msd(ravg)}  {rs[0]['steps']}"
            )

print("\n## Arm contrasts on real-image accuracy (hierarchical bootstrap over items and seeds)")
for size in ("G05", "G7"):
    for data in ("n4000", "full"):
        ao, _ = cell(size, data, "AO")
        for arm in ("A→R", "R→A"):
            x, _ = cell(size, data, arm)
            if ao and x:
                print(
                    f"{size:4s} {data:6s} {arm} − AO : {boot_diff(x, ao)}   (swapped: {boot_diff(x, ao, 'swapped')})"
                )
        ar, _ = cell(size, data, "A→R")
        ra, _ = cell(size, data, "R→A")
        if ar and ra:
            print(f"{size:4s} {data:6s} R→A − A→R : {boot_diff(ra, ar)}")

print("\n## Data dose: full − n4000 (0.5B)")
for arm in ("AO", "A→R", "R→A"):
    print(
        f"{arm:4s}: real {boot_diff(cell('G05', 'full', arm)[0], cell('G05', 'n4000', arm)[0])}   "
        f"swapped {boot_diff(cell('G05', 'full', arm)[0], cell('G05', 'n4000', arm)[0], 'swapped')}"
    )

# Kept so that the random draws of the later blocks are unchanged; the report's scale
# contrasts use matched seeds 0-2 (analyze_phase2_hp.py).
print("\n## Scale: 7B − 0.5B at n4000, 0.5B seed 0 only (superseded; see analyze_phase2_hp.py)")
for arm in ("AO", "A→R"):
    print(f"{arm:4s}: {boot_diff(cell('G7', 'n4000', arm)[0], [runs[('G05', 'n4000', arm, 0)]])}")
print(
    "θ0 7B−0.5B AO:",
    f"{100 * (np.mean(list(theta[('G7', 'AO')]['rot0']['real'].values())) - np.mean(list(theta[('G05', 'AO')]['rot0']['real'].values()))):+.1f} pp",
)

print("\n## Interaction: (A→R − AO) at full minus at n4000, 0.5B")


def contrast_boot(n=4000):
    cells = {
        k: np.stack([vec(r, "real") for r in cell("G05", *k)[0]])
        for k in [
            ("n4000", "AO"),
            ("n4000", "A→R"),
            ("full", "AO"),
            ("full", "A→R"),
            ("n4000", "R→A"),
            ("full", "R→A"),
        ]
    }

    def stat(c, arm):
        return (c[("full", arm)].mean() - c[("full", "AO")].mean()) - (
            c[("n4000", arm)].mean() - c[("n4000", "AO")].mean()
        )

    for arm in ("A→R", "R→A"):
        est = stat(cells, arm)
        out = []
        nI = len(qids)
        for _ in range(n):
            it = rng.integers(0, nI, nI)
            out.append(
                stat({k: v[rng.integers(0, len(v), len(v))][:, it] for k, v in cells.items()}, arm)
            )
        lo, hi = np.percentile(out, [2.5, 97.5])
        print(f"{arm}: {100 * est:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]")


contrast_boot()

print("\n## Cross-check: earlier CPU runs (results/runs/laptop/B_*_n4000), rotation 0")
for arm, a in (
    ("answer_only", "AO"),
    ("answer_then_rationale", "A→R"),
    ("rationale_then_answer", "R→A"),
):
    xs = []
    for d in sorted(
        glob.glob(os.path.join(os.path.dirname(ROOT.rstrip("/")), "laptop", f"B_{arm}_n4000_s*"))
    ):
        p = pfile(d)
        if p:
            xs.append(np.mean(list(load_preds(p)[0]["real"].values())))
    print(
        f"{a:4s} CPU n={len(xs)} real {msd(xs) if len(xs) == 5 else ', '.join(f'{100 * x:.1f}' for x in xs)}"
    )
