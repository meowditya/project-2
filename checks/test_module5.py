"""Module 5 approximations: the course's own numbers, closed forms, and exact conjugate answers."""

from __future__ import annotations

import math

from _common import DATA, assert_close, assert_raises
from treatment_bayes import approx

# Module 5 slides 10-19: Normal likelihood (sigma^2 = 1), Cauchy-type prior (mu = 0, tau = 1)
X = [1.440, 1.770, 3.559, 2.071, 2.129, 3.715, 2.461, 0.735, 1.313, 1.554]


def test_reproduces_slide_12_lindley():
    n, xbar = len(X), sum(X) / len(X)
    v = approx.lindley(xbar, L2=lambda t: -n, L3=lambda t: 0.0, u=lambda t: t, u1=lambda t: 1.0, u2=lambda t: 0.0,
                       rho1=lambda t: -2 * t / (1 + t * t))
    assert round(xbar, 4) == 2.0747
    assert round(v, 4) == 1.9965
    assert_close(v, xbar - 2 * xbar / (n * (1 + xbar**2)), rtol=1e-15, what="slide 11 closed form")


def test_reproduces_slides_18_19_tierney_kadane():
    n, xbar = len(X), sum(X) / len(X)
    L0 = lambda t: -(t - xbar) ** 2 / 2 - math.log(1 + t * t) / n
    L0p = lambda t: -(t - xbar) - (2 / n) * t / (1 + t * t)
    L0pp = lambda t: -1 - (2 / n) * (1 - t * t) / (1 + t * t) ** 2
    r = approx.tierney_kadane(n, L0, L0p, L0pp, lambda t: L0(t) + math.log(t) / n, lambda t: L0p(t) + 1 / (n * t),
                              lambda t: L0pp(t) - 1 / (n * t * t), (0.5, 3.5))
    assert [round(x, 4) for x in (r.theta0, r.sigma0, r.theta_star, r.sigma_star, r.L0_at_theta0, r.Lstar_at_theta_star)] \
        == [1.9946, 1.0122, 2.0447, 0.9999, -0.1637, -0.0934]
    assert round(r.value, 6) == 1.995138, "slide 19 R output"


def test_lindley_closed_forms_for_the_binomial():
    """Uniform prior: Lindley mean = s/n + (n - 2s)/n^2, Lindley odds = (s + 1)/(n - s) exactly."""
    for k, (n, s) in DATA.items():
        assert_close(approx.binomial_lindley(n, s, "mean"), s / n + (n - 2 * s) / n**2, rtol=1e-14, what=f"{k} mean")
        assert_close(approx.binomial_lindley(n, s, "odds"), (s + 1) / (n - s), rtol=1e-13, what=f"{k} odds")


def test_tierney_kadane_maximisers_are_the_known_modes():
    for k, (n, s) in DATA.items():
        r = approx.binomial_tierney_kadane(n, s, "mean")
        assert_close(r.theta0, s / n, rtol=1e-12, what="theta0 = posterior mode")
        assert_close(r.theta_star, (s + 1) / (n + 1), rtol=1e-12, what="theta* for u = theta")
        r = approx.binomial_tierney_kadane(n, s, "odds")
        assert_close(r.theta_star, (s + 1) / n, rtol=1e-12, what="theta* for u = odds")


def test_approximation_errors_against_exact_conjugate_values():
    """Both methods carry O(n^-2) error (slide 9): below 1/n^2 relative to the size of E[u]."""
    for k, (n, s) in DATA.items():
        a, b = s + 1, n - s + 1
        exact_mean, exact_odds = a / (a + b), a / (b - 1)
        for which, ex in (("mean", exact_mean), ("odds", exact_odds)):
            li = approx.binomial_lindley(n, s, which)
            tk = approx.binomial_tierney_kadane(n, s, which).value
            scale = max(ex, 1.0)  # absolute for the mean (< 1), relative for the odds (> 1)
            assert abs(li - ex) < scale / n**2 and abs(tk - ex) < scale / n**2, (k, which, li, tk, ex)
            assert li != ex or which == "odds", "Lindley is exact only for the odds"


def test_jeffreys_prior_exercises_the_rho_term():
    """With Beta(1/2,1/2) the prior derivative rho1 is non-zero; approximations still track the exact mean."""
    for k, (n, s) in DATA.items():
        ex = (s + 0.5) / (n + 1)
        li = approx.binomial_lindley(n, s, "mean", 0.5, 0.5)
        tk = approx.binomial_tierney_kadane(n, s, "mean", 0.5, 0.5).value
        assert abs(li - ex) < 1e-4 and abs(tk - ex) < 1e-4
        assert li != approx.binomial_lindley(n, s, "mean"), "the prior must change the Lindley value"


def test_rejects_boundary_data():
    assert_raises(ValueError, approx.binomial_lindley, 10, 0, match="0 < s < n")
    assert_raises(ValueError, approx.binomial_tierney_kadane, 10, 10, match="0 < s < n")
