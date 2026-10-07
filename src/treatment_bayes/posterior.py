"""Beta-binomial posteriors for any real prior parameters (floating point).

Model (MA 232 Module 4, Example 4.6): each patient on a treatment succeeds
independently with probability theta. With s successes in n patients the
likelihood is theta^s (1 - theta)^(n - s), and a Beta(alpha, beta) prior gives

    theta | data ~ Beta(alpha + s, beta + n - s).

Point estimates are the three Bayes estimators of Module 4, Theorems 4.1-4.3:
posterior mean (squared-error loss), posterior median (absolute-error loss),
posterior mode / MAP (zero-one loss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from .data import Arm

__all__ = ["Prior", "BetaPosterior", "posterior", "PRIORS"]


@dataclass(frozen=True)
class Prior:
    """Beta(alpha, beta) prior on a success probability."""

    name: str
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        for v, nm in ((self.alpha, "alpha"), (self.beta, "beta")):
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"prior {self.name!r}: {nm} must be finite and > 0, got {v!r}")


#: Main prior and the sensitivity set. Beta(1,1) is uniform ("no preference");
#: Beta(1/2,1/2) is the Jeffreys prior for a Bernoulli probability (Module 4,
#: Section 5.3.3); Beta(2,2) and Beta(10,10) are the priors of assignment
#: problems 1.2 and 1.12, mildly and strongly centred on 1/2.
PRIORS: dict[str, Prior] = {
    "uniform": Prior("Uniform Beta(1,1)", 1, 1),
    "jeffreys": Prior("Jeffreys Beta(1/2,1/2)", 0.5, 0.5),
    "beta22": Prior("Beta(2,2)", 2, 2),
    "beta1010": Prior("Beta(10,10)", 10, 10),
}


@dataclass(frozen=True)
class BetaPosterior:
    """Beta(alpha, beta) posterior of one treatment's success probability."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not (self.alpha > 0 and self.beta > 0):
            raise ValueError(f"Beta parameters must be > 0, got ({self.alpha}, {self.beta})")

    @property
    def dist(self):
        return stats.beta(self.alpha, self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)

    @property
    def mode(self) -> float:
        """MAP estimate (alpha - 1) / (alpha + beta - 2); interior only if alpha, beta > 1."""
        if not (self.alpha > 1 and self.beta > 1):
            raise ValueError("the posterior mode is interior only when alpha > 1 and beta > 1")
        return (self.alpha - 1) / (self.alpha + self.beta - 2)

    @property
    def median(self) -> float:
        """Posterior median: the root of F(m) = 1/2 (regularised incomplete beta)."""
        return float(self.dist.ppf(0.5))

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile level must be in (0, 1), got {q}")
        return float(self.dist.ppf(q))

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval: ((1-level)/2, (1+level)/2) quantiles."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"credible level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        return self.quantile(tail), self.quantile(1.0 - tail)

    def pdf(self, x):
        """Beta density; 0 outside (0, 1). Uses scipy.special directly (fast inside quad)."""
        x = np.asarray(x, dtype=float)
        inside = (x > 0.0) & (x < 1.0)
        xs = np.where(inside, x, 0.5)
        logf = (self.alpha - 1.0) * np.log(xs) + (self.beta - 1.0) * np.log1p(-xs) - special.betaln(self.alpha, self.beta)
        out = np.where(inside, np.exp(logf), 0.0)
        return float(out) if out.ndim == 0 else out

    def cdf(self, x):
        """Regularised incomplete beta I_x(alpha, beta), clipped to [0, 1] outside the support."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        out = special.betainc(self.alpha, self.beta, x)
        return float(out) if np.ndim(out) == 0 else out


def posterior(arm: Arm, prior: Prior) -> BetaPosterior:
    """Conjugate update: Beta(alpha + s, beta + n - s)."""
    return BetaPosterior(prior.alpha + arm.s, prior.beta + arm.failures)
