"""Is the 4k training set (first 4,000 questions of the training pool) representative?

Report Section 3.2. Compares the 4k set with 1,000 random 4,000-question draws from the pool
on 11 statistics against the rest of the pool: standardised mean differences of 8 per-question
features, total-variation distance of correct-option position and question type, and the
Jensen-Shannon divergence of the correct options. A pooled test counts how many statistics
fall outside the draws' central 95%.

    python experiments/analysis/prefix_representativeness.py [--aokvqa-dir data/raw/aokvqa]

Inputs: aokvqa_v1p0_{train,val}.json in --aokvqa-dir. Output: printed, and written to
results/analysis/prefix_representativeness.txt unless --no-save.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import operator
import random
import string
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmr.data.aokvqa import AOKVQAItem, dev_split, load_aokvqa  # noqa: E402
from cmr.eval.prompts import mentions  # noqa: E402

N_4K, DEV_FRACTION, DEV_SEED = 4000, 0.06, 0
N_DRAWS, DRAW_SEED = 1000, 0
TAIL = 0.025
SCALARS = (
    "question length",
    "mean rationale length",
    "correct-option length",
    "mean option length",
    "yes/no question",
    "answer stated in a rationale",
    "difficult_direct_answer",
    "direct answers = correct option",
)
CATEGORICAL = ("correct-option position", "question type")
STATS = (*(f"SMD {s}" for s in SCALARS), *(f"TV {c}" for c in CATEGORICAL), "JS correct options")
QTYPES = ("what", "which", "why", "how", "who", "where", "when")
AUXILIARY = frozenset(
    ("is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should",
     "has", "have")
)  # fmt: skip


def key(text: str) -> str:
    return " ".join(text.lower().split())


def question_type(question: str) -> str:
    tokens = question.lower().split()
    tok = tokens[0].strip(string.punctuation) if tokens else ""
    for apostrophe in ("'", "\u2019"):  # ASCII and typographic apostrophe
        tok = tok.split(apostrophe, 1)[0]
    if tok in QTYPES:
        return tok
    return "auxiliary" if tok in AUXILIARY else "other"


def ntok(text: str) -> int:
    return len(text.split())


def scalar_features(it: AOKVQAItem) -> tuple[float, ...]:
    gold = it.choices[it.correct_choice_idx]
    gkey = key(gold)
    da = it.direct_answers
    return (
        ntok(it.question),
        sum(ntok(r) for r in it.rationales) / len(it.rationales),
        ntok(gold),
        sum(ntok(c) for c in it.choices) / len(it.choices),
        int(gkey in ("yes", "no")),
        int(any(mentions(r, gold) for r in it.rationales)),
        int(bool(it.difficult_direct_answer)),
        sum(1 for a in da if key(a) == gkey) / len(da) if da else 0.0,
    )


@dataclass
class Population:
    """Per-question feature columns and their totals."""

    n: int
    cols: list[list[float]]
    sq: list[list[float]]
    cats: list[list[Any]]
    gold: list[str]
    tot_s: list[float]
    tot_q: list[float]
    tot_c: list[Counter]
    tot_g: Counter


def features(items: Sequence[AOKVQAItem]) -> Population:
    rows = [scalar_features(it) for it in items]
    cols = [[r[f] for r in rows] for f in range(len(SCALARS))]
    sq = [[v * v for v in c] for c in cols]
    cats = [[it.correct_choice_idx for it in items], [question_type(it.question) for it in items]]
    gold = [key(it.choices[it.correct_choice_idx]) for it in items]
    return Population(
        len(items), cols, sq, cats, gold,
        [sum(c) for c in cols], [sum(c) for c in sq], [Counter(c) for c in cats], Counter(gold),
    )  # fmt: skip


# ----------------------------------------------------------------------------- statistics


def smd(s1: float, q1: float, n1: int, s2: float, q2: float, n2: int) -> float:
    m1, m2 = s1 / n1, s2 / n2
    v1 = (q1 - s1 * s1 / n1) / (n1 - 1)
    v2 = (q2 - s2 * s2 / n2) / (n2 - 1)
    den = math.sqrt(max(0.0, (v1 + v2) / 2))
    if den == 0.0:
        return 0.0 if m1 == m2 else math.copysign(math.inf, m1 - m2)
    return (m1 - m2) / den


def total_variation(c1: Counter, n1: int, tot: Counter, n2: int) -> float:
    return 0.5 * sum(abs(c1[k] / n1 - (t - c1[k]) / n2) for k, t in tot.items())


def js_divergence(c1: Counter, n1: int, tot: Counter, n2: int) -> float:
    js = 0.0
    for k, t in tot.items():
        a, b = c1[k] / n1, (t - c1[k]) / n2
        m = (a + b) / 2
        if a:
            js += a * math.log2(a / m)
        if b:
            js += b * math.log2(b / m)
    return js / 2


def stats_for(pop: Population, subset: Sequence[int]) -> tuple[list[float], list[float]]:
    """(the 11 statistics of ``subset`` against the rest of ``pop``, its feature means)."""
    n1, n2 = len(subset), pop.n - len(subset)
    get = operator.itemgetter(*subset)
    out, means = [], []
    for f in range(len(SCALARS)):
        s1, q1 = sum(get(pop.cols[f])), sum(get(pop.sq[f]))
        out.append(smd(s1, q1, n1, pop.tot_s[f] - s1, pop.tot_q[f] - q1, n2))
        means.append(s1 / n1)
    for c in range(len(CATEGORICAL)):
        out.append(total_variation(Counter(get(pop.cats[c])), n1, pop.tot_c[c], n2))
    out.append(js_divergence(Counter(get(pop.gold)), n1, pop.tot_g, n2))
    return out, means


def cut_ranks(n_ref: int) -> tuple[int, int]:
    """(k, n + 1 - k), k = ceil(0.025 n): for n = 1,000 the 25th and 976th values."""
    k = math.ceil(round(TAIL * n_ref, 9))
    return k, n_ref + 1 - k


def is_outside(x: float, ref: Sequence[float]) -> bool:
    lo, hi = cut_ranks(len(ref))
    srt = sorted(ref)
    return x < srt[lo - 1] or x > srt[hi - 1]


def pooled_k(table: Sequence[Sequence[float]]) -> list[int]:
    """K of every pool member, each against the other members."""
    ks = [0] * len(table)
    for s in range(len(table[0])):
        col = [row[s] for row in table]
        for j, x in enumerate(col):
            ks[j] += is_outside(x, col[:j] + col[j + 1 :])
    return ks


# ----------------------------------------------------------------------------- report


def analyse(train_all: list[AOKVQAItem], val_all: list[AOKVQAItem]) -> list[str]:
    pool_items, _dev = dev_split(train_all, dev_fraction=DEV_FRACTION, seed=DEV_SEED)
    set_4k = pool_items[:N_4K]
    val = [it for it in val_all if it.has_labels]
    for it in (*pool_items, *val):
        if len(it.choices) != 4 or len(it.rationales) != 3:
            raise ValueError(f"{it.question_id}: expected 4 options and 3 rationales")

    pool = features(pool_items)
    rng = random.Random(DRAW_SEED)
    draws = [rng.sample(range(pool.n), N_4K) for _ in range(N_DRAWS)]
    results = [stats_for(pool, range(N_4K)), *(stats_for(pool, s) for s in draws)]
    table = [r[0] for r in results]
    draw_means = [sum(r[1][f] for r in results[1:]) / N_DRAWS for f in range(len(SCALARS))]
    draw_shares = [Counter() for _ in CATEGORICAL]  # summed category counts over the draws
    for draw in draws:
        for c, col in enumerate(pool.cats):
            draw_shares[c].update(col[i] for i in draw)
    ks = pooled_k(table)
    k0 = ks[0]
    n_ge = sum(1 for k in ks if k >= k0)
    lo_r, hi_r = cut_ranks(N_DRAWS)

    rows, n_beyond_all = [], 0
    for s, name in enumerate(STATS):
        x = table[0][s]
        null = sorted(row[s] for row in table[1:])
        pctl = 100 * (sum(v < x for v in null) + sum(v == x for v in null) / 2) / N_DRAWS
        out = x < null[lo_r - 1] or x > null[hi_r - 1]
        beyond = x < null[0] or x > null[-1]
        n_beyond_all += beyond
        rows.append(
            f"  {name:38s} {x:+.4f}  draws {lo_r}th-{hi_r}th [{null[lo_r - 1]:+.4f}, "
            f"{null[hi_r - 1]:+.4f}]  min-max [{null[0]:+.4f}, {null[-1]:+.4f}]  "
            f"percentile {pctl:6.2f}  {'outside' if out else 'inside'}"
            f"{' (beyond all draws)' if beyond else ''}"
        )
    dist = Counter(ks[1:])

    pops = {"4k": features(set_4k), "rest": features(pool_items[N_4K:]), "val": features(val)}
    side = [f"  {'':38s} {'4k set':>8s} {'rest':>8s} {'draws':>8s} {'val':>8s}"]
    for f, name in enumerate(SCALARS):
        cells = [pops[p].tot_s[f] / pops[p].n for p in ("4k", "rest")] + [draw_means[f]]
        cells.append(pops["val"].tot_s[f] / pops["val"].n)
        side.append(f"  {name:38s} " + " ".join(f"{v:8.4f}" for v in cells))
    for c, name in enumerate(CATEGORICAL):
        cats = sorted({k for p in pops.values() for k in p.tot_c[c]}, key=str)
        for cat in cats:
            share = {lab: f"{p.tot_c[c][cat] / p.n:8.4f}" for lab, p in pops.items()}
            label = f"{name} = {cat}"
            side.append(
                f"  {label:38s} {share['4k']} {share['rest']} "
                f"{draw_shares[c][cat] / (N_DRAWS * N_4K):8.4f} {share['val']}"
            )
    top = [
        f"  {lab}: " + ", ".join(f"{k!r} {c}" for k, c in pops[lab].tot_g.most_common(10))
        for lab in ("4k", "rest")
    ]

    return [
        "Representativeness of the 4k training set",
        f"training pool {pool.n:,} questions (official train {len(train_all):,} minus the dev "
        f"split); 4k set = pool[:{N_4K}]: {len(set_4k):,} questions, "
        f"{len({it.image_id for it in set_4k}):,} images; rest {pool.n - N_4K:,}; "
        f"val {len(val):,}",
        f"draws: {N_DRAWS:,} x rng.sample(range({pool.n}), {N_4K}), random.Random({DRAW_SEED}); "
        f"each statistic compares a set with the rest of the pool",
        "",
        f"4k set outside the central 95% of the draws on {k0} of {len(STATS)} statistics; "
        f"p = {n_ge}/{N_DRAWS + 1} = {n_ge / (N_DRAWS + 1):.4f} (pooled test); "
        f"beyond the min-max of all {N_DRAWS:,} draws on {n_beyond_all} of {len(STATS)}",
        "Per statistic (4k set; the draws' 25th-976th values and min-max):",
        *rows,
        "Number of statistics outside, over the draws: "
        + ", ".join(f"K={k}: {dist[k]}" for k in range(max(dist) + 1)),
        "",
        "Means (shares for yes/no, stated answer, flags and categories); 'rest' = the "
        f"{pool.n - N_4K:,} pool questions after the 4k set, 'draws' = average of the "
        f"{N_DRAWS:,} draw means (shares):",
        *side,
        "Ten most frequent correct options:",
        *top,
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--aokvqa-dir", type=Path, default=ROOT / "data/raw/aokvqa")
    ap.add_argument("--no-save", action="store_true", help="print only")
    args = ap.parse_args(argv)
    files = [args.aokvqa_dir / f"aokvqa_v1p0_{s}.json" for s in ("train", "val")]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        print(f"missing in --aokvqa-dir: {', '.join(missing)}", file=sys.stderr)
        return 2
    lines = [f"sha256 {p.name} {sha256(p)}" for p in files] + [""]
    lines += analyse(load_aokvqa(args.aokvqa_dir, "train"), load_aokvqa(args.aokvqa_dir, "val"))
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if not args.no_save:
        out = ROOT / "results/analysis/prefix_representativeness.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
