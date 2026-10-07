"""Monte Carlo agreement judged by standard errors; the decision rules; the more-data scenario."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from _common import DATA, prob_greater_closed_form
from treatment_bayes import PRIORS, exact, load_trial, montecarlo, posterior
from treatment_bayes.decision import decide

ARMS = load_trial()
POST = {k: posterior(a, PRIORS["uniform"]) for k, a in ARMS.items()}
EX = {k: exact.BetaExact(int(p.alpha), int(p.beta)) for k, p in POST.items()}


def test_monte_carlo_within_four_standard_errors():
    rng = np.random.default_rng(424242)  # a different seed from the analysis script
    dr = montecarlo.draw(POST, 1_000_000, rng)
    for i, j in [("B", "A"), ("B", "C"), ("C", "A")]:
        est = montecarlo.mc_prob_greater(dr, i, j)
        assert abs(est.z_against(float(exact.prob_greater(EX[i], EX[j])))) < 4, (i, j, est)
    for k, est in montecarlo.mc_prob_best(dr).items():
        assert abs(est.z_against(float(exact.prob_best(EX, k)))) < 4, (k, est)
    em = montecarlo.mc_expected_max(dr)
    assert abs(em.z_against(float(exact.expected_max(EX)))) < 4


def test_standard_error_formula():
    est = montecarlo.mc_prob_greater({"x": np.array([1.0, 0.0, 1.0, 1.0]), "y": np.zeros(4)}, "x", "y")
    assert est.value == 0.75 and abs(est.se - np.sqrt(0.75 * 0.25 * 4 / 3 / 4)) < 1e-15


def test_both_loss_functions_choose_b():
    loss = {k: float(exact.expected_loss(EX, k)) for k in EX}
    best = {k: float(exact.prob_best(EX, k)) for k in EX}
    d = decide(loss, best)
    assert d.by_expected_loss == "B" and d.by_prob_best == "B" and d.agree


def test_decide_rejects_mismatched_inputs():
    try:
        decide({"A": 0.1}, {"B": 0.9})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_more_data_scenario_is_monotone_and_matches_closed_form():
    prev = 0.0
    for N in (100, 200, 300, 500):
        sx = {k: Fraction(s, n) * N for k, (n, s) in DATA.items()}
        es = {k: exact.BetaExact(int(v) + 1, N - int(v) + 1) for k, v in sx.items()}
        pb = float(exact.prob_best(es, "B"))
        assert pb > prev
        prev = pb
        assert exact.prob_greater(es["B"], es["A"]) == prob_greater_closed_form(es["B"].a, es["B"].b, es["A"].a, es["A"].b)
