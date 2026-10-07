"""Comparisons between independent Beta posteriors by one-dimensional quadrature.

These work for any real Beta parameters (for example the Jeffreys prior,
where the exact rational route in ``exact.py`` does not apply). With integer
parameters they must agree with ``exact.py``; the checks confirm that.

Independence: the three arms have separate patients and independent priors,
so their posteriors are independent and joint probabilities factorise.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import integrate, optimize

from .posterior import BetaPosterior

__all__ = [
    "QUAD_OPTS",
    "prob_greater",
    "prob_best",
    "expected_max",
    "expected_loss",
    "diff_cdf",
    "diff_quantile",
    "diff_interval",
]

QUAD_OPTS = dict(epsabs=1e-14, epsrel=1e-12, limit=500)


def _quad(f, lo: float = 0.0, hi: float = 1.0, points=None) -> float:
    val, _err = integrate.quad(f, lo, hi, points=points, **QUAD_OPTS)
    return float(val)


def _peaks(*posts: BetaPosterior) -> list[float]:
    """Posterior means as quad breakpoints: keeps the adaptive rule on the mass."""
    return sorted({min(max(p.mean, 1e-9), 1 - 1e-9) for p in posts})


def prob_greater(x: BetaPosterior, y: BetaPosterior) -> float:
    """P(X > Y) = int_0^1 f_X(t) F_Y(t) dt."""
    return _quad(lambda t: x.pdf(t) * y.cdf(t), points=_peaks(x, y))


def prob_best(posts: Mapping[str, BetaPosterior], k: str) -> float:
    """P(k is best) = int_0^1 f_k(t) prod_{j != k} F_j(t) dt."""
    others = [p for j, p in posts.items() if j != k]

    def g(t):
        v = posts[k].pdf(t)
        for p in others:
            v *= p.cdf(t)
        return v

    return _quad(g, points=_peaks(*posts.values()))


def expected_max(posts: Mapping[str, BetaPosterior]) -> float:
    """E[max_j p_j] = int_0^1 (1 - prod_j F_j(t)) dt."""

    def g(t):
        v = 1.0
        for p in posts.values():
            v *= p.cdf(t)
        return 1.0 - v

    return _quad(g, points=_peaks(*posts.values()))


def expected_loss(posts: Mapping[str, BetaPosterior], k: str) -> float:
    """E[max_j p_j - p_k]: expected opportunity loss of choosing k."""
    return expected_max(posts) - posts[k].mean


def diff_cdf(x: BetaPosterior, y: BetaPosterior, d: float) -> float:
    """P(X - Y <= d) = int_0^1 f_X(t) [1 - F_Y(t - d)] dt."""
    if d <= -1.0:
        return 0.0
    if d >= 1.0:
        return 1.0
    lo, hi = max(0.0, d), 1.0  # outside [max(0,d), 1] the integrand is f_X(t) (F_Y(t-d) = 0)
    tail = _quad(lambda t: x.pdf(t) * (1.0 - y.cdf(t - d)), lo, hi, points=[p for p in _peaks(x) if lo < p < hi] or None)
    head = float(x.cdf(lo))  # t < d: F_Y(t - d) = 0, integrand = f_X(t)
    return min(1.0, max(0.0, head + tail))


def diff_quantile(x: BetaPosterior, y: BetaPosterior, q: float) -> float:
    """Quantile of X - Y: the root of diff_cdf(d) = q (Brent's method)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile level must be in (0, 1), got {q}")
    return float(optimize.brentq(lambda d: diff_cdf(x, y, d) - q, -1.0 + 1e-12, 1.0 - 1e-12, xtol=1e-13, rtol=1e-13))


def diff_interval(x: BetaPosterior, y: BetaPosterior, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return diff_quantile(x, y, tail), diff_quantile(x, y, 1.0 - tail)


def diff_pdf(x: BetaPosterior, y: BetaPosterior, d) -> np.ndarray:
    """Density of X - Y: int f_X(t) f_Y(t - d) dt, for plotting."""
    d = np.atleast_1d(np.asarray(d, dtype=float))
    out = np.empty_like(d)
    for i, di in enumerate(d):
        lo, hi = max(0.0, di), min(1.0, 1.0 + di)
        out[i] = _quad(lambda t: x.pdf(t) * y.pdf(t - di), lo, hi) if hi > lo else 0.0
    return out
