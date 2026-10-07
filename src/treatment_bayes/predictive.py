"""Posterior predictive for future patients (beta-binomial).

If theta | data ~ Beta(a, b), the number of successes Y among m future
patients has the beta-binomial distribution

    P(Y = y) = C(m, y) B(a + y, b + m - y) / B(a, b),   y = 0..m,

with mean m a / (a + b). For m = 1 this is P(next patient succeeds) =
posterior mean (assignment problems 1.2(5) and 1.11(6)).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

from .posterior import BetaPosterior

__all__ = ["Predictive", "predictive"]


@dataclass(frozen=True)
class Predictive:
    m: int
    mean: float
    sd: float
    low: int
    high: int
    coverage: float


def predictive(post: BetaPosterior, m: int, level: float = 0.95) -> Predictive:
    """Beta-binomial predictive of successes in m future patients.

    The interval is equal-tailed on the integer scale: ``low`` is the
    (1-level)/2 quantile and ``high`` the (1+level)/2 quantile, so its actual
    coverage P(low <= Y <= high) is at least ``level`` and is reported.
    """
    if m < 1:
        raise ValueError("m must be a positive number of future patients")
    dist = stats.betabinom(m, post.alpha, post.beta)
    tail = (1.0 - level) / 2.0
    low, high = int(dist.ppf(tail)), int(dist.ppf(1.0 - tail))
    coverage = float(dist.cdf(high) - dist.cdf(low - 1))
    return Predictive(m, float(dist.mean()), float(dist.std()), low, high, coverage)
