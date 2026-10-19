"""Paired statistics for systems scored on the same questions (numpy only).

Item-level tests (Wilson interval, exact McNemar test, paired bootstrap) compare two fixed
checkpoints; claims about a training setting run with several seeds also need the variance
between seeds, which :func:`hierarchical_bootstrap` adds (report Section 3.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided interval ``[low, high]`` (proportions in 0..1)."""

    estimate: float
    low: float
    high: float


def wilson_interval(successes: float, n: int, z: float = 1.959964) -> Interval:
    """Wilson score interval (95% by default) for ``successes / n``; bounds clipped to [0, 1].

    >>> ci = wilson_interval(6, 10)
    >>> ci.estimate, round(ci.low, 3), round(ci.high, 3)
    (0.6, 0.313, 0.832)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if successes <= 0 else max(0.0, centre - half)
    high = 1.0 if successes >= n else min(1.0, centre + half)
    return Interval(p, low, high)


@dataclass(frozen=True)
class McNemarResult:
    """Discordant-pair counts and the exact two-sided McNemar p-value."""

    b: int  # A correct, B wrong
    c: int  # A wrong, B correct
    p_value: float


def mcnemar_exact(correct_a: ArrayLike, correct_b: ArrayLike) -> McNemarResult:
    """Exact two-sided McNemar test on per-question correctness (p = 1 if no discordant pair).

    >>> mcnemar_exact([1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0])
    McNemarResult(b=5, c=0, p_value=0.0625)
    """
    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    if a.shape != b_arr.shape:
        raise ValueError("inputs must have the same shape")
    b = int(np.sum(a & ~b_arr))
    c = int(np.sum(~a & b_arr))

    n = b + c
    if n == 0:
        return McNemarResult(b, c, 1.0)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return McNemarResult(b, c, min(1.0, 2 * tail))


@dataclass(frozen=True)
class BootstrapResult:
    """Observed mean difference, its 95% percentile interval and a two-sided p-value."""

    mean_diff: float
    low: float
    high: float
    p_value: float


def paired_bootstrap(
    scores_a: ArrayLike, scores_b: ArrayLike, n_boot: int = 10_000, seed: int = 0
) -> BootstrapResult:
    """Bootstrap mean(a - b) over questions: 95% percentile CI and re-centred two-sided p.

    >>> r = paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 0], n_boot=1000)
    >>> r.mean_diff, r.low <= r.mean_diff <= r.high
    (0.5, True)
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("inputs must be equal-length non-empty 1-D arrays")
    d = a - b

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)

    observed = float(d.mean())
    low, high = np.quantile(means, [0.025, 0.975])
    centred = means - observed
    p = float((np.sum(np.abs(centred) >= abs(observed)) + 1) / (n_boot + 1))
    return BootstrapResult(observed, float(low), float(high), p)


# ---------------------------------------------------------------------------
# Student-t helpers (numpy only: scipy is not a runtime dependency)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularised incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            step = d * c
            h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T_df| >= |t|) via the incomplete beta; df = inf gives the normal tail.

    >>> t_two_sided_p(0.0, 5)
    1.0
    >>> round(t_two_sided_p(1.959964, float("inf")), 4), round(t_two_sided_p(2.0, 10), 4)
    (0.05, 0.0734)
    """
    if df <= 0:
        raise ValueError("df must be positive")
    t = abs(t)
    if math.isinf(df):
        return math.erfc(t / math.sqrt(2.0))
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_critical(level: float, df: float) -> float:
    """Two-sided critical value c with P(|T_df| >= c) = 1 - level (bisection).

    >>> round(t_critical(0.95, float("inf")), 2), round(t_critical(0.95, 2), 3)
    (1.96, 4.303)
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    alpha = 1.0 - level
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > alpha:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Seed-aware comparisons (seeds x questions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TResult:
    """A difference with a t-reference interval: CI = mean_diff +/- t_crit(df) * se."""

    mean_diff: float
    se: float
    df: float
    low: float
    high: float
    p_value: float


@dataclass(frozen=True)
class HierarchicalResult(TResult):
    """:class:`TResult` of :func:`hierarchical_bootstrap` plus descriptive extras."""

    pct_low: float  # percentile interval of the bootstrap draws (reported, not decisive)
    pct_high: float
    seed_sd_a: float  # SD of the per-seed mean score
    seed_sd_b: float  # 0.0 for a one-sample test
    n_seeds_a: int
    n_seeds_b: int  # 0 for a one-sample test
    n_items: int


def _seed_item(x: ArrayLike, name: str) -> np.ndarray:
    """``x`` as a finite, non-empty float (n_seeds, n_items) array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array of shape (n_seeds, n_items)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _t_result(observed: float, var_terms: list[tuple[float, float]], level: float) -> TResult:
    """t-reference CI and p with Satterthwaite df over (variance, df) terms.

    Zero total variance gives se = 0, a point CI and p = 0 (p = 1 if ``observed`` is 0).
    """
    total = sum(v for v, _ in var_terms)
    if total <= 0.0:
        p = 1.0 if observed == 0.0 else 0.0
        return TResult(observed, 0.0, math.inf, observed, observed, p)
    denom = sum(v * v / d for v, d in var_terms if v > 0.0 and d > 0)
    df = total * total / denom if denom > 0.0 else math.inf
    se = math.sqrt(total)
    half = t_critical(level, df) * se
    return TResult(
        observed, se, df, observed - half, observed + half, t_two_sided_p(observed / se, df)
    )


def _churn_excess(x: np.ndarray) -> float:
    """Variance the two-way bootstrap adds by counting the seed x item interaction twice.

    Equals 2 (N-1)/N^2 * sigma_e^2 / S for N items, S seeds and interaction mean square
    sigma_e^2 (the residual of ``x`` after removing seed and item means).
    """
    n_seeds, n_items = x.shape
    if n_items < 2:
        return 0.0
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + x.mean()
    mse = float(np.sum(resid**2)) / ((n_seeds - 1) * (n_items - 1))
    return float(2.0 * (n_items - 1) / n_items**2 * mse / n_seeds)


def hierarchical_bootstrap(
    scores_a: ArrayLike,
    scores_b: ArrayLike | None = None,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
    small_seed_correction: bool = True,
) -> HierarchicalResult:
    """Seed-aware bootstrap of mean(scores_a) - mean(scores_b) over seeds and items.

    Inputs are (n_seeds, n_items) on the same items; ``scores_b=None`` tests mean(scores_a)
    against 0. Each draw resamples items once (shared, keeping the pairing) and seeds within
    each system. ``small_seed_correction`` (default) applies three fixes for few seeds:

    1. scale seed deviations by sqrt(S/(S-1)) so a seed mean's variance is unbiased;
    2. subtract the seed x item interaction variance that resampling both axes counts twice;
    3. use a t reference with Satterthwaite df instead of bootstrap percentiles
       (still returned as ``pct_low`` / ``pct_high``).

    Without it, the plain percentile CI and shift p-value are returned (not calibrated at
    3 seeds); that plain version is the interval the report uses (Section 3.7), and
    experiments/analysis/bootstrap_check.py compares the two. Seeds are assumed independent
    between the two systems. Zero variance gives se = 0 and a degenerate p, so check se > 0
    before quoting p.

    Raises:
        ValueError: fewer than 2 seeds per system, or ``n_boot`` < 2.

    >>> a = [[1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
    >>> b = [[0, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1]]
    >>> r = hierarchical_bootstrap(a, b, n_boot=2000)
    >>> round(r.mean_diff, 4), r.n_seeds_a, r.n_seeds_b, r.n_items, r.low < 0 < r.high
    (0.3333, 3, 3, 4, True)
    """
    a = _seed_item(scores_a, "scores_a")
    arms = [a]
    if scores_b is not None:
        b = _seed_item(scores_b, "scores_b")
        if b.shape[1] != a.shape[1]:
            raise ValueError("both systems must be scored on the same items")
        arms.append(b)
    if any(x.shape[0] < 2 for x in arms):
        raise ValueError(
            "a seed-aware test needs >= 2 seeds per system; with 1 seed use paired_bootstrap "
            "and scope the claim to 'these checkpoints'"
        )
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")

    n_items = arms[0].shape[1]
    signs = (1.0, -1.0)
    observed = float(sum(s * x.mean() for s, x in zip(signs, arms, strict=False)))

    # Correction 1: inflate each system's seed deviations by sqrt(S/(S-1)).
    work = arms
    if small_seed_correction:
        work = [
            x.mean(axis=0) + math.sqrt(x.shape[0] / (x.shape[0] - 1)) * (x - x.mean(axis=0))
            for x in arms
        ]

    # Bootstrap in chunks of ~1e6 weights: items shared, seeds resampled per system.
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot)
    chunk = max(1, min(n_boot, 1_000_000 // n_items))
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        flat = rng.integers(0, n_items, size=(m, n_items)) + (np.arange(m) * n_items)[:, None]
        weights = np.bincount(flat.ravel(), minlength=m * n_items).reshape(m, n_items)
        weights = weights / n_items
        for s, x in zip(signs, work, strict=False):
            per_seed = weights @ x.T  # (m, S): each seed's mean over the resampled items
            pick = rng.integers(0, x.shape[0], size=(m, x.shape[0]))
            boot[start : start + m] += s * np.take_along_axis(per_seed, pick, axis=1).mean(axis=1)

    # Variance split into seed terms + item term; correction 2 removes the double-counted
    # seed x item interaction (_churn_excess).
    seed_means = [x.mean(axis=1) for x in arms]
    seed_terms = [(float(sm.var(ddof=1)) / sm.size, float(sm.size - 1)) for sm in seed_means]
    se_boot = float(boot.std(ddof=1))
    v_seed = sum(v for v, _ in seed_terms)
    excess = sum(_churn_excess(x) for x in arms) if small_seed_correction else 0.0
    v_item = max(se_boot**2 - excess - v_seed, 0.0)
    terms = [*seed_terms, (v_item, float(n_items - 1))]

    # Correction 3: t reference with Satterthwaite df.
    if small_seed_correction:
        res = _t_result(observed, terms, level)
    else:  # plain percentile CI and shift p-value
        lo_q, hi_q = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
        centred = boot - boot.mean()
        p = float((np.sum(np.abs(centred) >= abs(observed)) + 1) / (n_boot + 1))
        res = TResult(observed, se_boot, math.inf, float(lo_q), float(hi_q), p)

    pct = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    sd = [float(sm.std(ddof=1)) for sm in seed_means] + [0.0]
    return HierarchicalResult(
        mean_diff=res.mean_diff,
        se=res.se,
        df=res.df,
        low=res.low,
        high=res.high,
        p_value=res.p_value,
        pct_low=float(pct[0]),
        pct_high=float(pct[1]),
        seed_sd_a=sd[0],
        seed_sd_b=sd[1],
        n_seeds_a=arms[0].shape[0],
        n_seeds_b=arms[1].shape[0] if len(arms) > 1 else 0,
        n_items=n_items,
    )
