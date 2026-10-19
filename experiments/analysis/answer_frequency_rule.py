"""Answer-frequency rule: how far do A-OKVQA's option strings alone predict the answer?

Report Sections 4.5 and 5.4 (38.4% [35.7, 41.1] with counts from all training questions, 29.7%
with counts from other val questions). With no image and no question, each
option scores s = g / (g + d/3) (Kafle and Kanan, Eq. 2), where g / d count how often its
lower-cased string is the correct / a wrong option in the count source (unseen: 1/2); ties at
the maximum share the credit. Count sources: all training questions, and other val folds
(5 folds grouped by image, with a label-randomisation test). 95% CI: question bootstrap.

    python experiments/analysis/answer_frequency_rule.py [--aokvqa-dir data/raw/aokvqa]

Inputs: aokvqa_v1p0_{train,val}.json in --aokvqa-dir. Output: printed, and written to
results/analysis/answer_frequency_rule.txt unless --no-save.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import math
import random
import sys
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmr.data.aokvqa import AOKVQAItem, dev_split, load_aokvqa  # noqa: E402

K = 3  # wrong options per question
UNSEEN = Fraction(1, 2)
N_BOOT, BOOT_SEED = 10_000, 0
N_FOLDS, FOLD_SEED = 5, 0
N_NULL, NULL_SEED = 1000, 2
N_4K, DEV_FRACTION, DEV_SEED = 4000, 0.06, 0
MOST_COMMON_PUBLISHED = 30.70  # A-OKVQA val "Most Common" baseline as published

Keys = list[list[str]]
Scores = list[list[Fraction]]


# ----------------------------------------------------------------------------- the rule


def key(choice: str) -> str:
    return " ".join(choice.lower().split())


def count_table(keyed: Sequence[Sequence[str]], gold: Sequence[int]) -> tuple[Counter, Counter]:
    """(g, d): times each key is the correct option / a wrong option, per occurrence."""
    g: Counter[str] = Counter()
    d: Counter[str] = Counter()
    for keys, gi in zip(keyed, gold, strict=True):
        for i, k in enumerate(keys):
            (g if i == gi else d)[k] += 1
    return g, d


@functools.cache
def score(g: int, d: int) -> Fraction:
    return UNSEEN if g + d == 0 else Fraction(g) / (g + Fraction(d, K))


def item_scores(keys: Sequence[str], g: Counter, d: Counter) -> list[Fraction]:
    return [score(g[k], d[k]) for k in keys]


def tie_credit(scores: Sequence[Fraction | int], gold: int) -> Fraction:
    top = max(scores)
    m = sum(1 for s in scores if s == top)
    return Fraction(1, m) if scores[gold] == top else Fraction(0)


def n_tied_at_max(scores: Scores) -> int:
    return sum(1 for s in scores if s.count(max(s)) > 1)


def mean(values: Sequence[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


# ----------------------------------------------------------------------------- bootstrap


def quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default ``quantile``)."""
    h = (len(sorted_vals) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


def bootstrap_ci(values: Sequence[Fraction]) -> tuple[float, float]:
    """95% percentile CI of the mean; values share an integer denominator so sums are exact."""
    den = math.lcm(*(v.denominator for v in values))
    ints = [int(v * den) for v in values]
    n = len(ints)
    rng = random.Random(BOOT_SEED)
    means = sorted(sum(rng.choices(ints, k=n)) / (den * n) for _ in range(N_BOOT))
    return quantile(means, 0.025), quantile(means, 0.975)


def pct(x: Fraction | float) -> str:
    return f"{100 * float(x):.2f}%"


def acc_ci(credits: Sequence[Fraction]) -> str:
    lo, hi = bootstrap_ci(credits)
    return f"{pct(mean(credits))} [{pct(lo)}, {pct(hi)}]"


# ----------------------------------------------------------------------------- within val


def image_folds(image_ids: Sequence[int]) -> list[int]:
    """Fold of each question: sorted unique images, Random(0).shuffle, round-robin."""
    images = sorted(set(image_ids))
    random.Random(FOLD_SEED).shuffle(images)
    fold_of = {img: i % N_FOLDS for i, img in enumerate(images)}
    return [fold_of[i] for i in image_ids]


def cv_scores(keyed: Keys, labels: Sequence[int], folds: Sequence[int]) -> Scores:
    """Each question's four scores, from the counts of the other folds only."""
    scores: Scores = [[] for _ in keyed]
    for f in range(N_FOLDS):
        rest = [i for i, fi in enumerate(folds) if fi != f]
        g, d = count_table([keyed[i] for i in rest], [labels[i] for i in rest])
        for i, fi in enumerate(folds):
            if fi == f:
                scores[i] = item_scores(keyed[i], g, d)
    return scores


def cv_credits(keyed: Keys, labels: Sequence[int], folds: Sequence[int]) -> list[Fraction]:
    scores = cv_scores(keyed, labels, folds)
    return [tie_credit(s, gi) for s, gi in zip(scores, labels, strict=True)]


def check_held_out_labels_unused(keyed: Keys, labels: Sequence[int], folds: Sequence[int]) -> None:
    base = cv_scores(keyed, labels, folds)
    for f in range(N_FOLDS):
        shifted = [(lab + 1) % 4 if folds[i] == f else lab for i, lab in enumerate(labels)]
        again = cv_scores(keyed, shifted, folds)
        if any(again[i] != base[i] for i, fi in enumerate(folds) if fi == f):
            raise RuntimeError(f"fold {f}: held-out scores depend on held-out labels")


def randomisation_null(keyed: Keys, folds: Sequence[int]) -> list[Fraction]:
    rng = random.Random(NULL_SEED)
    out = []
    for _ in range(N_NULL):
        labels = [rng.randrange(len(ks)) for ks in keyed]
        out.append(mean(cv_credits(keyed, labels, folds)))
    return out


# ----------------------------------------------------------------------------- report


def analyse(train_all: list[AOKVQAItem], val_all: list[AOKVQAItem]) -> list[str]:
    val = [it for it in val_all if it.has_labels]
    n = len(val)
    va_keys = [[key(c) for c in it.choices] for it in val]
    va_gold = [it.correct_choice_idx for it in val]

    def score_val(source: Sequence[AOKVQAItem]) -> tuple[Scores, list[Fraction]]:
        g, d = count_table(
            [[key(c) for c in it.choices] for it in source],
            [it.correct_choice_idx for it in source],
        )
        scores = [item_scores(ks, g, d) for ks in va_keys]
        return scores, [tie_credit(s, gi) for s, gi in zip(scores, va_gold, strict=True)]

    # 1. counts from all training questions
    scores, credits = score_val(train_all)
    acc = mean(credits)
    unique = sum(1 for c in credits if c == 1)
    g, d = count_table(
        [[key(c) for c in it.choices] for it in train_all],
        [it.correct_choice_idx for it in train_all],
    )
    gold_unseen = sum(
        1 for ks, gi in zip(va_keys, va_gold, strict=True) if g[ks[gi]] + d[ks[gi]] == 0
    )
    mc = mean([tie_credit([g[k] for k in ks], gi) for ks, gi in zip(va_keys, va_gold, strict=True)])

    # 4k-set counts (descriptive)
    pool, _dev = dev_split(train_all, dev_fraction=DEV_FRACTION, seed=DEV_SEED)
    _, credits_4k = score_val(pool[:N_4K])

    # 2. counts from the other val folds
    ids = [it.image_id for it in val]
    folds = image_folds(ids)
    check_held_out_labels_unused(va_keys, va_gold, folds)
    cv_scr = cv_scores(va_keys, va_gold, folds)
    cv_cred = [tie_credit(s, gi) for s, gi in zip(cv_scr, va_gold, strict=True)]
    acc_cv = mean(cv_cred)
    null = randomisation_null(va_keys, folds)
    n_ge = sum(1 for v in null if v >= acc_cv)
    null_f = sorted(float(v) for v in null)
    fold_n = Counter(folds)
    diffs = [a - b for a, b in zip(cv_cred, credits, strict=True)]
    d_lo, d_hi = bootstrap_ci(diffs)

    return [
        "Answer-frequency rule on A-OKVQA val (no image, no question; chance 25%)",
        f"val questions with labels {n}; score s = g / (g + d/3), unseen keys 1/2; "
        "ties share the credit",
        f"CI: question bootstrap, {N_BOOT:,} draws, random.Random({BOOT_SEED}), 95% percentile",
        "",
        f"1. Counts from all {len(train_all):,} training questions",
        f"   accuracy {acc_ci(credits)}   (exact {acc.numerator}/{acc.denominator})",
        f"   correct option the unique maximum: {pct(Fraction(unique, n))} ({unique} of {n}); "
        f"questions with a tie at the maximum: {n_tied_at_max(scores)}",
        f"   correct option never seen in training: {gold_unseen} of {n} "
        f"({pct(Fraction(gold_unseen, n))})",
        "",
        "2. Counts from the other val folds (5 folds grouped by image)",
        f"   accuracy {acc_ci(cv_cred)}   (exact {acc_cv.numerator}/{acc_cv.denominator}; "
        "the bootstrap CI ignores the dependence between folds)",
        f"   label-randomisation test ({N_NULL:,} relabellings, random.Random({NULL_SEED})): "
        f"p = {1 + n_ge}/{N_NULL + 1} = {(1 + n_ge) / (N_NULL + 1):.4f}; null mean "
        f"{pct(sum(null_f) / N_NULL)}, 2.5-97.5th percentiles "
        f"[{pct(quantile(null_f, 0.025))}, {pct(quantile(null_f, 0.975))}]",
        "   held-out labels never reach their own scores (checked): yes",
        f"   correct option the unique maximum: "
        f"{pct(Fraction(sum(1 for c in cv_cred if c == 1), n))}; ties at the maximum: "
        f"{n_tied_at_max(cv_scr)}",
        f"   images {len(set(ids))}; questions per fold "
        + ", ".join(str(fold_n[f]) for f in range(N_FOLDS))
        + f"; at most {max(Counter(ids).values())} val questions share an image",
        f"   difference from rule 1: {100 * float(mean(diffs)):+.2f} pp, paired bootstrap "
        f"[{100 * d_lo:+.2f}, {100 * d_hi:+.2f}]",
        "",
        "Reference rows (descriptive)",
        f"   most common answer (largest g only, same tie credit): {pct(mc)} "
        f"(published A-OKVQA val baseline {MOST_COMMON_PUBLISHED:.2f}%; its tie and "
        "normalisation rules are not stated)",
        f"   counts from the 4k training set only ({N_4K:,} of {len(pool):,} pool questions): "
        f"{pct(mean(credits_4k))}",
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
        out = ROOT / "results/analysis/answer_frequency_rule.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
