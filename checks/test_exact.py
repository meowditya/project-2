"""Exact rational machinery: densities, CDFs, comparisons, all against independent routes."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
from scipy import integrate, stats

from _common import DATA, assert_close, assert_raises, beta_cdf_binomial_tail, beta_fn, prob_greater_closed_form
from treatment_bayes import exact

POST = {k: exact.BetaExact(s + 1, n - s + 1) for k, (n, s) in DATA.items()}  # uniform prior


def test_posterior_parameters_from_the_project_table():
    assert (POST["A"].a, POST["A"].b) == (63, 39)
    assert (POST["B"].a, POST["B"].b) == (71, 31)
    assert (POST["C"].a, POST["C"].b) == (67, 35)


def test_density_integrates_to_one_and_moments_are_exact():
    for d in POST.values():
        assert exact.poly_integral(d.pdf) == 1
        t = [Fraction(0), Fraction(1)]  # the polynomial "t"
        assert exact.poly_integral(exact.poly_mul(d.pdf, t)) == d.mean
        second = exact.poly_integral(exact.poly_mul(d.pdf, [Fraction(0), Fraction(0), Fraction(1)]))
        assert second - d.mean**2 == d.var
        assert d.beta_function == beta_fn(d.a, d.b)


def test_cdf_equals_binomial_tail_identity():
    for d in POST.values():
        for x in (Fraction(1, 10), Fraction(1, 2), Fraction(5, 8), Fraction(7, 10), Fraction(99, 100)):
            assert d.cdf_at(x) == beta_cdf_binomial_tail(d.a, d.b, x)
        assert d.cdf_at(0) == 0 and d.cdf_at(1) == 1 and d.cdf_at(-1) == 0 and d.cdf_at(2) == 1


def test_cdf_matches_scipy_in_floating_point():
    xs = np.linspace(0.01, 0.99, 99)
    for d in POST.values():
        got = [float(d.cdf_at(Fraction(float(x)))) for x in xs]
        assert_close(got, stats.beta(d.a, d.b).cdf(xs), atol=1e-13, what=f"Beta({d.a},{d.b}) cdf")


def test_mode_is_the_density_maximum():
    for d in POST.values():
        m = d.mode
        deriv = [c * k for k, c in enumerate(d.pdf)][1:]
        assert exact.poly_eval(deriv, m) == 0, "f'(mode) must be exactly 0"
        assert d.mode == Fraction(d.a - 1, d.a + d.b - 2)


def test_prob_greater_equals_closed_form_sum_exactly():
    for i, j in [("B", "A"), ("B", "C"), ("C", "A"), ("A", "B")]:
        x, y = POST[i], POST[j]
        assert exact.prob_greater(x, y) == prob_greater_closed_form(x.a, x.b, y.a, y.b), (i, j)


def test_prob_greater_complement_and_quadrature():
    for i, j in combinations(POST, 2):
        x, y = POST[i], POST[j]
        assert exact.prob_greater(x, y) + exact.prob_greater(y, x) == 1
        q, _ = integrate.quad(lambda t: stats.beta(x.a, x.b).pdf(t) * stats.beta(y.a, y.b).cdf(t), 0, 1,
                              epsabs=1e-14, epsrel=1e-12, limit=400)
        assert_close(float(exact.prob_greater(x, y)), q, atol=1e-12, what=f"P({i}>{j}) vs quad")


def test_prob_best_sums_to_one_and_respects_bounds():
    pb = {k: exact.prob_best(POST, k) for k in POST}
    assert sum(pb.values()) == 1
    for k in POST:
        others = [j for j in POST if j != k]
        pg = [exact.prob_greater(POST[k], POST[j]) for j in others]
        assert pb[k] <= min(pg), "P(best) cannot exceed any pairwise win probability"
        assert pb[k] >= pg[0] + pg[1] - 1, "Bonferroni lower bound"


def test_prob_best_by_quadrature_with_scipy_directly():
    for k in POST:
        others = [POST[j] for j in POST if j != k]
        f = stats.beta(POST[k].a, POST[k].b).pdf
        Fs = [stats.beta(o.a, o.b).cdf for o in others]
        q, _ = integrate.quad(lambda t: f(t) * Fs[0](t) * Fs[1](t), 0, 1, epsabs=1e-14, epsrel=1e-12, limit=400)
        assert_close(float(exact.prob_best(POST, k)), q, atol=1e-12, what=f"P({k} best)")


def test_expected_max_three_ways():
    em = exact.expected_max(POST)
    # (1) E[max] = int_0^1 P(max > t) dt, by quadrature with scipy
    q, _ = integrate.quad(lambda t: 1 - np.prod([stats.beta(d.a, d.b).cdf(t) for d in POST.values()]), 0, 1,
                          epsabs=1e-14, epsrel=1e-12, limit=400)
    assert_close(float(em), q, atol=1e-12, what="E[max] vs quad")
    # (2) E[max] = sum_k E[p_k 1{k best}] = sum_k int t f_k(t) prod_{j!=k} F_j(t) dt (exact)
    total = Fraction(0)
    for k in POST:
        integrand = exact.poly_mul([Fraction(0), Fraction(1)], POST[k].pdf)
        for j, d in POST.items():
            if j != k:
                integrand = exact.poly_mul(integrand, d.cdf)
        total += exact.poly_integral(integrand)
    assert total == em, "two exact routes to E[max] must agree exactly"
    assert em > max(d.mean for d in POST.values())


def test_expected_loss_choice_equals_largest_posterior_mean():
    loss = {k: exact.expected_loss(POST, k) for k in POST}
    assert all(v > 0 for v in loss.values())
    assert min(loss, key=loss.get) == max(POST, key=lambda k: POST[k].mean) == "B"
    for i, j in combinations(POST, 2):  # loss differences are exactly mean differences
        assert loss[i] - loss[j] == POST[j].mean - POST[i].mean


def test_prob_diff_greater_identities_and_quadrature():
    B, A = POST["B"], POST["A"]
    assert exact.prob_diff_greater(B, A, 0) == exact.prob_greater(B, A)
    for d in (Fraction(1, 20), Fraction(1, 10), Fraction(-1, 20), Fraction(3, 10)):
        p = exact.prob_diff_greater(B, A, d)
        assert p + exact.prob_diff_greater(A, B, -d) == 1, "P(B-A > d) + P(A-B > -d) = 1"
        fb, Fa = stats.beta(B.a, B.b).pdf, stats.beta(A.a, A.b).cdf
        q, _ = integrate.quad(lambda t: fb(t) * Fa(t - float(d)), max(0.0, float(d)), 1, epsabs=1e-14, epsrel=1e-12, limit=400)
        assert_close(float(p), q, atol=1e-11, what=f"P(B-A > {d})")
    assert exact.prob_diff_greater(B, A, 1) == 0 and exact.prob_diff_greater(B, A, -1) == 1
    margins = [exact.prob_diff_greater(B, A, Fraction(k, 100)) for k in range(-20, 21)]
    assert all(x > y for x, y in zip(margins, margins[1:])), "P(B - A > d) must fall as d rises"


def test_bad_parameters_rejected():
    assert_raises(ValueError, exact.BetaExact, 0, 3, match="integers")
    assert_raises(ValueError, exact.BetaExact, 2.5, 3, match="integers")
    assert_raises(ValueError, lambda: exact.BetaExact(1, 5).mode, match="interior")
