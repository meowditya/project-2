"""Posteriors: conjugacy by brute force, sufficiency, Bayes estimators, intervals, differences."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import integrate, stats

from _common import DATA, assert_close, assert_raises
from treatment_bayes import PRIORS, Prior, compare, exact, load_trial, posterior
from treatment_bayes.data import Arm

ARMS = load_trial()
POST = {k: posterior(a, PRIORS["uniform"]) for k, a in ARMS.items()}


def test_csv_matches_the_project_sheet():
    assert {k: (a.n, a.s) for k, a in ARMS.items()} == DATA


def test_conjugate_update_equals_numerical_bayes_on_a_grid():
    """Posterior from Bayes' theorem by brute force: prior x likelihood, normalised numerically."""
    t = np.linspace(1e-6, 1 - 1e-6, 200_001)
    for pr in PRIORS.values():
        for k, (n, s) in DATA.items():
            unnorm = stats.beta(pr.alpha, pr.beta).pdf(t) * stats.binom(n, t).pmf(s)
            dens = unnorm / integrate.simpson(unnorm, x=t)
            p = posterior(ARMS[k], pr)
            assert_close(dens, p.pdf(t), atol=2e-6 * dens.max(), what=f"{pr.name} {k}")


def test_sufficiency_order_does_not_matter():
    """Only (n, s) enters: sequential one-patient updates in any order give the batch posterior."""
    rng = np.random.default_rng(3)
    for k, (n, s) in DATA.items():
        outcomes = np.array([1] * s + [0] * (n - s))
        for _ in range(5):
            rng.shuffle(outcomes)
            a, b = 1, 1
            for y in outcomes:
                a, b = a + int(y), b + 1 - int(y)
            assert (a, b) == (POST[k].alpha, POST[k].beta)


def test_bayes_estimators_minimise_their_losses():
    """Module 4 Theorems 4.1-4.3, checked numerically on the B posterior."""
    p = POST["B"]
    draws = stats.beta(p.alpha, p.beta).rvs(size=400_000, random_state=np.random.default_rng(11))
    grid = np.linspace(0.65, 0.74, 361)
    sq = [np.mean((draws - a) ** 2) for a in grid]
    ab = [np.mean(np.abs(draws - a)) for a in grid]
    step = grid[1] - grid[0]
    assert abs(grid[int(np.argmin(sq))] - p.mean) <= 2 * step
    assert abs(grid[int(np.argmin(ab))] - p.median) <= 2 * step
    dens = p.pdf(grid)
    assert abs(grid[int(np.argmax(dens))] - p.mode) <= step


def test_mean_median_mode_ordering_and_mle():
    for k, (n, s) in DATA.items():
        p = POST[k]
        assert p.mode == s / n, "under the uniform prior the MAP equals the MLE"
        assert p.mean < p.median < p.mode, "left-skewed Beta (alpha > beta): mean < median < mode"


def test_median_and_credible_interval_by_exact_cdf():
    for k, p in POST.items():
        e = exact.BetaExact(int(p.alpha), int(p.beta))
        assert_close(float(e.cdf_at(Fraction(p.median))), 0.5, atol=1e-12, what=f"{k} median")
        lo, hi = p.interval(0.95)
        assert_close(float(e.cdf_at(Fraction(lo))), 0.025, atol=1e-12, what=f"{k} 2.5%")
        assert_close(float(e.cdf_at(Fraction(hi))), 0.975, atol=1e-12, what=f"{k} 97.5%")


def test_difference_interval_endpoints_by_exact_probability():
    for i, j in [("B", "A"), ("B", "C"), ("C", "A")]:
        lo, hi = compare.diff_interval(POST[i], POST[j], 0.95)
        ei = exact.BetaExact(int(POST[i].alpha), int(POST[i].beta))
        ej = exact.BetaExact(int(POST[j].alpha), int(POST[j].beta))
        assert_close(float(exact.prob_diff_greater(ei, ej, Fraction(lo))), 0.975, atol=1e-10, what=f"{i}-{j} low")
        assert_close(float(exact.prob_diff_greater(ei, ej, Fraction(hi))), 0.025, atol=1e-10, what=f"{i}-{j} high")


def test_difference_moments_and_monte_carlo_quantiles():
    rng = np.random.default_rng(5)
    n = 1_000_000
    for i, j in [("B", "A"), ("B", "C"), ("C", "A")]:
        d = rng.beta(POST[i].alpha, POST[i].beta, n) - rng.beta(POST[j].alpha, POST[j].beta, n)
        assert_close(d.mean(), POST[i].mean - POST[j].mean, atol=4 * d.std() / np.sqrt(n), what="mean")
        lo, hi = compare.diff_interval(POST[i], POST[j], 0.95)
        assert_close(np.quantile(d, [0.025, 0.975]), [lo, hi], atol=1.5e-3, what=f"{i}-{j} quantiles")


def test_quadrature_comparisons_match_exact_for_integer_priors():
    for key in ("uniform", "beta22", "beta1010"):
        pr = PRIORS[key]
        ps = {k: posterior(a, pr) for k, a in ARMS.items()}
        es = {k: exact.BetaExact(int(p.alpha), int(p.beta)) for k, p in ps.items()}
        for k in ps:
            assert_close(compare.prob_best(ps, k), float(exact.prob_best(es, k)), atol=1e-12, what=f"{key} best {k}")
            assert_close(compare.expected_loss(ps, k), float(exact.expected_loss(es, k)), atol=1e-12, what=f"{key} loss {k}")
        assert_close(compare.prob_greater(ps["B"], ps["A"]), float(exact.prob_greater(es["B"], es["A"])), atol=1e-12,
                     what=f"{key} B>A")


def test_jeffreys_prior_against_monte_carlo():
    ps = {k: posterior(a, PRIORS["jeffreys"]) for k, a in ARMS.items()}
    rng = np.random.default_rng(17)
    n = 2_000_000
    dr = {k: rng.beta(p.alpha, p.beta, n) for k, p in ps.items()}
    win = (dr["B"] > dr["A"]).mean()
    assert abs(win - compare.prob_greater(ps["B"], ps["A"])) <= 4 * np.sqrt(win * (1 - win) / n)
    best = (np.argmax(np.vstack([dr["A"], dr["B"], dr["C"]]), axis=0) == 1).mean()
    assert abs(best - compare.prob_best(ps, "B")) <= 4 * np.sqrt(best * (1 - best) / n)


def test_input_validation():
    assert_raises(ValueError, Arm, "X", 10, 11, match="successes")
    assert_raises(ValueError, Arm, "X", 0, 0, match="at least one")
    assert_raises(ValueError, Prior, "bad", 0, 1, match="alpha")
    assert_raises(ValueError, POST["A"].interval, 1.5, match="level")
    assert_raises(ValueError, posterior(Arm("X", 3, 0), PRIORS["uniform"]).__getattribute__, "mode", match="interior")
