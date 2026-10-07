"""Classical (frequentist) summaries, for comparison with the Bayesian answers.

MLE of a Bernoulli probability is the sample proportion s/n (Module 3,
MoM-versus-MLE table, p. 14: Bernoulli(theta) -> X-bar). The Wald intervals
use the normal approximation to the binomial (assignment problems 1.7-1.8,
Central Limit Theorem); hypothesis tests are outside the course syllabus and
are shown only to contrast "p-value" with "posterior probability".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from .data import Arm

__all__ = ["Z975", "ProportionEstimate", "proportion", "DifferenceEstimate", "difference"]

#: Phi^{-1}(0.975), the 95% two-sided normal quantile.
Z975 = float(stats.norm.ppf(0.975))


@dataclass(frozen=True)
class ProportionEstimate:
    mle: float
    se: float
    wald_low: float
    wald_high: float


def proportion(arm: Arm, z: float = Z975) -> ProportionEstimate:
    p = arm.s / arm.n
    se = math.sqrt(p * (1.0 - p) / arm.n)
    return ProportionEstimate(p, se, p - z * se, p + z * se)


@dataclass(frozen=True)
class DifferenceEstimate:
    estimate: float
    se: float
    wald_low: float
    wald_high: float
    z_pooled: float
    p_one_sided: float


def difference(x: Arm, y: Arm, z: float = Z975) -> DifferenceEstimate:
    """p_x - p_y: Wald interval (unpooled SE) and a pooled one-sided z-test of p_x > p_y."""
    px, py = x.s / x.n, y.s / y.n
    est = px - py
    se = math.sqrt(px * (1 - px) / x.n + py * (1 - py) / y.n)
    pooled = (x.s + y.s) / (x.n + y.n)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / x.n + 1 / y.n))
    zstat = est / se0
    return DifferenceEstimate(est, se, est - z * se, est + z * se, zstat, float(stats.norm.sf(zstat)))
