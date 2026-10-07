"""Seeded Monte Carlo from the independent Beta posteriors.

The simulation route of assignment problem 1.10(4): draw each p_k from its
posterior and average. These are plain i.i.d. draws from known Beta
posteriors (Module 5, slide 20, before MCMC is introduced); no Markov chain
is involved. Every estimate comes with its Monte Carlo standard error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .posterior import BetaPosterior

__all__ = ["MCEstimate", "draw", "mc_prob_greater", "mc_prob_best", "mc_expected_max", "mc_diff_quantiles"]


@dataclass(frozen=True)
class MCEstimate:
    value: float
    se: float
    n_draws: int

    def z_against(self, exact: float) -> float:
        """(estimate - exact) / standard error."""
        return (self.value - exact) / self.se if self.se > 0 else float("nan")


def draw(posts: Mapping[str, BetaPosterior], n_draws: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """n_draws independent draws from each posterior, in the dict's order."""
    if n_draws < 2:
        raise ValueError("need at least two draws")
    return {k: rng.beta(p.alpha, p.beta, size=n_draws) for k, p in posts.items()}


def _mean_se(x: np.ndarray) -> MCEstimate:
    return MCEstimate(float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size)), int(x.size))


def mc_prob_greater(draws: Mapping[str, np.ndarray], i: str, j: str) -> MCEstimate:
    return _mean_se((draws[i] > draws[j]).astype(float))


def mc_prob_best(draws: Mapping[str, np.ndarray]) -> dict[str, MCEstimate]:
    names = list(draws)
    stacked = np.vstack([draws[k] for k in names])
    winner = np.argmax(stacked, axis=0)
    return {k: _mean_se((winner == i).astype(float)) for i, k in enumerate(names)}


def mc_expected_max(draws: Mapping[str, np.ndarray]) -> MCEstimate:
    return _mean_se(np.max(np.vstack(list(draws.values())), axis=0))


def mc_diff_quantiles(draws: Mapping[str, np.ndarray], i: str, j: str, qs=(0.025, 0.5, 0.975)) -> list[float]:
    return [float(v) for v in np.quantile(draws[i] - draws[j], qs)]
