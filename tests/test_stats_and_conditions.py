"""Statistics (cmr.eval.stats; Section 3.7) and the swapped-image assignment (Section 3.5).

Checked against scipy where possible: Wilson intervals, the exact McNemar test, the paired
bootstrap over questions and the seed-aware (hierarchical) bootstrap with the small-seed
correction of Section 3.7. The swapped image is a fixed derangement: no question keeps its own
image.
"""

import numpy as np
import pytest
from scipy import stats as sps

from cmr.eval.conditions import swap_assignment
from cmr.eval.stats import (
    hierarchical_bootstrap,
    mcnemar_exact,
    paired_bootstrap,
    t_critical,
    t_two_sided_p,
    wilson_interval,
)


def test_wilson_matches_scipy() -> None:
    ours = wilson_interval(70, 100)
    ref = sps.binomtest(70, 100).proportion_ci(method="wilson")
    assert ours.low == pytest.approx(ref.low, abs=1e-6)
    assert ours.high == pytest.approx(ref.high, abs=1e-6)


def test_wilson_edges() -> None:
    assert wilson_interval(0, 10).low == 0.0
    assert wilson_interval(10, 10).high == 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_mcnemar_matches_binomtest() -> None:
    a = np.array([1] * 30 + [0] * 10 + [1] * 50 + [0] * 10, bool)
    b = np.array([0] * 30 + [1] * 10 + [1] * 50 + [0] * 10, bool)
    res = mcnemar_exact(a, b)
    assert (res.b, res.c) == (30, 10)
    assert res.p_value == pytest.approx(sps.binomtest(10, 40, 0.5).pvalue)


def test_mcnemar_no_discordance() -> None:
    assert mcnemar_exact([1, 0], [1, 0]).p_value == 1.0


def test_paired_bootstrap_detects_and_rejects() -> None:
    rng = np.random.default_rng(0)
    base = rng.random(500)
    better = paired_bootstrap(base + 0.1, base)
    assert better.mean_diff == pytest.approx(0.1)
    assert better.low > 0 and better.p_value < 0.01
    null = paired_bootstrap(base, base[rng.permutation(500)])
    assert null.low < 0 < null.high


def test_paired_bootstrap_validates() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


@pytest.mark.parametrize("seed", range(5))
def test_swap_assignment_properties(seed: int) -> None:
    ids = [i // 3 for i in range(30)]  # 3 questions per image
    perm = swap_assignment(ids, seed=seed)
    assert sorted(perm) == list(range(30))  # a true permutation
    assert all(ids[p] != ids[i] for i, p in enumerate(perm))  # never the same image
    assert perm == swap_assignment(ids, seed=seed)  # deterministic


def test_swap_assignment_impossible_cases() -> None:
    with pytest.raises(ValueError):
        swap_assignment([1, 1, 1])
    with pytest.raises(ValueError):
        swap_assignment([1, 1, 1, 2])


# --- seed-aware recipe statistics -----------------------------------------------------


@pytest.mark.parametrize("df", [1, 2, 3.7, 4, 30, 1e4])
def test_t_helpers_match_scipy(df: float) -> None:
    for t in (0.0, 0.3, 1.0, 2.5, 8.0):
        assert t_two_sided_p(t, df) == pytest.approx(2 * sps.t.sf(t, df), abs=1e-10)
    assert t_critical(0.95, df) == pytest.approx(sps.t.ppf(0.975, df), rel=1e-8)
    assert t_two_sided_p(1.959964, float("inf")) == pytest.approx(0.05, abs=1e-6)


def test_hierarchical_exact_two_seed_case() -> None:
    # 2 seeds x 1 item: per-seed means 0 and 1 -> unbiased Var(seed mean) = s^2/S = 0.5/2.
    # The sqrt(S/(S-1)) inflation makes the bootstrap SE match it (0.5), with df = S-1 = 1.
    res = hierarchical_bootstrap([[0.0], [1.0]], [[0.0], [0.0]], n_boot=40_000)
    assert res.mean_diff == pytest.approx(0.5)
    assert res.se == pytest.approx(0.5, rel=0.02)
    assert res.df == pytest.approx(1.0)
    assert res.high - res.mean_diff == pytest.approx(sps.t.ppf(0.975, 1) * res.se)
    assert res.p_value == pytest.approx(2 * sps.t.sf(0.5 / res.se, 1))


def test_hierarchical_correction_restores_unbiased_seed_variance() -> None:
    # Seed-only variance (every item equal within a seed): plain bootstrap gives the
    # plug-in variance (S-1)/S * s^2/S; the corrected one gives s^2/S.
    a = np.repeat(np.array([[0.60], [0.62], [0.65]]), 50, axis=1)
    b = np.full((3, 50), 0.6)
    plain = hierarchical_bootstrap(a, b, n_boot=40_000, small_seed_correction=False)
    corr = hierarchical_bootstrap(a, b, n_boot=40_000)
    s2_over_s = np.var([0.60, 0.62, 0.65], ddof=1) / 3
    assert corr.se**2 == pytest.approx(s2_over_s, rel=0.03)
    assert plain.se**2 == pytest.approx(s2_over_s * 2 / 3, rel=0.03)
    assert corr.df == pytest.approx(2.0, rel=0.01)  # ~ seed term only: S_a - 1


def test_hierarchical_keeps_item_pairing_and_one_sample() -> None:
    rng = np.random.default_rng(1)
    a = np.tile(rng.random(200), (3, 1))  # identical checkpoints: no seed variance
    res = hierarchical_bootstrap(a, a - 0.1, n_boot=500)
    assert res.mean_diff == pytest.approx(0.1)
    assert res.low == pytest.approx(0.1) and res.high == pytest.approx(0.1)
    one = hierarchical_bootstrap(a, None, n_boot=500)
    assert one.n_seeds_b == 0 and one.mean_diff == pytest.approx(a.mean())


def test_hierarchical_validates() -> None:
    with pytest.raises(ValueError, match="2 seeds"):
        hierarchical_bootstrap([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="same items"):
        hierarchical_bootstrap(np.zeros((2, 3)), np.zeros((2, 4)))
    # zero variance everywhere: point CI; p = 0 for a nonzero difference
    flat = hierarchical_bootstrap(np.ones((2, 3)), np.zeros((2, 3)), n_boot=50)
    assert (flat.se, flat.low, flat.high, flat.p_value) == (0.0, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        hierarchical_bootstrap([1.0, 0.0], [1.0, 0.0])  # 1-D is ambiguous: reject


def _simulate_arm(rng: np.random.Generator, latent: np.ndarray, seed_sd_logit: float) -> np.ndarray:
    """3 checkpoints of one setting: shared item latent + seed shift + per-item noise."""
    u = rng.normal(0, seed_sd_logit, size=(3, 1))
    return (latent + u + 0.6 * rng.normal(size=(3, latent.size)) > 0).astype(float)


def test_identical_settings_false_positive_rate() -> None:
    """Identical settings, 3 seeds: item-level tests over-reject; seed-aware ones stay near 5%."""
    rng = np.random.default_rng(12345)
    n, sims = 1145, 120
    latent = rng.normal(0.6, 1.6, n) + rng.logistic(0, 1, n)
    naive = corrected = 0
    seed_sds = []
    for k in range(sims):
        a = _simulate_arm(rng, latent, 0.088)
        b = _simulate_arm(rng, latent, 0.088)
        seed_sds.append(a.mean(1).std(ddof=1))
        d = a.mean(0) - b.mean(0)
        naive += t_two_sided_p(d.mean() / (d.std(ddof=1) / np.sqrt(n)), n - 1) < 0.05
        hb = hierarchical_bootstrap(a, b, n_boot=200, seed=k)
        corrected += hb.p_value < 0.05
    assert 0.012 < float(np.sqrt(np.mean(np.square(seed_sds)))) < 0.022
    assert naive / sims > 0.12  # item-level tests ignore seed variance
    assert corrected / sims <= 0.09
