"""Posterior predictive (beta-binomial, exact rationals) and classical summaries (by hand)."""

from __future__ import annotations

import math
from fractions import Fraction
from math import comb

from _common import DATA, assert_close, beta_fn
from treatment_bayes import PRIORS, classical, load_trial, posterior, predictive

ARMS = load_trial()


def betabinom_pmf(m: int, a: int, b: int) -> list[Fraction]:
    """P(Y = y) = C(m, y) B(a + y, b + m - y) / B(a, b), exactly."""
    return [comb(m, y) * beta_fn(a + y, b + m - y) / beta_fn(a, b) for y in range(m + 1)]


def test_predictive_against_exact_beta_binomial():
    for k, (n, s) in DATA.items():
        a, b = s + 1, n - s + 1
        pmf = betabinom_pmf(100, a, b)
        assert sum(pmf) == 1
        mean = sum(y * p for y, p in enumerate(pmf))
        assert mean == Fraction(100 * a, a + b)
        var = sum((y - mean) ** 2 * p for y, p in enumerate(pmf))
        pr = predictive.predictive(posterior(ARMS[k], PRIORS["uniform"]), 100)
        assert_close(pr.mean, float(mean), rtol=1e-12, what=f"{k} mean")
        assert_close(pr.sd, math.sqrt(var), rtol=1e-10, what=f"{k} sd")
        cdf = [sum(pmf[: y + 1]) for y in range(101)]
        assert cdf[pr.low] >= Fraction(1, 40) and (pr.low == 0 or cdf[pr.low - 1] < Fraction(1, 40))
        assert cdf[pr.high] >= Fraction(39, 40) and cdf[pr.high - 1] < Fraction(39, 40)
        coverage = cdf[pr.high] - (cdf[pr.low - 1] if pr.low > 0 else 0)
        assert_close(pr.coverage, float(coverage), atol=1e-12, what=f"{k} coverage")
        assert coverage >= Fraction(95, 100)


def test_next_patient_probability_is_the_posterior_mean():
    for k, (n, s) in DATA.items():
        assert betabinom_pmf(1, s + 1, n - s + 1)[1] == Fraction(s + 1, n + 2)


def test_wald_intervals_by_hand():
    z = 1.959963984540054
    for k, (n, s) in DATA.items():
        p = s / n
        se = math.sqrt(p * (1 - p) / n)
        c = classical.proportion(ARMS[k])
        assert_close([c.mle, c.se, c.wald_low, c.wald_high], [p, se, p - z * se, p + z * se], rtol=1e-14, what=k)


def test_difference_z_test_by_hand():
    d = classical.difference(ARMS["B"], ARMS["A"])
    assert_close(d.estimate, 0.08, atol=1e-15, what="estimate")
    pooled = 132 / 200
    z = 0.08 / math.sqrt(pooled * (1 - pooled) * (2 / 100))
    assert_close(d.z_pooled, z, rtol=1e-14, what="z")
    assert_close(d.p_one_sided, 0.5 * math.erfc(z / math.sqrt(2)), rtol=1e-12, what="one-sided p")
    assert_close(d.se, math.sqrt(0.7 * 0.3 / 100 + 0.62 * 0.38 / 100), rtol=1e-14, what="unpooled se")
