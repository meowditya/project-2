"""Exact rational arithmetic for Beta posteriors with integer parameters.

With a Beta(1, 1) prior and binomial data, every posterior is Beta(a, b) with
integer a, b >= 1. Its density

    f(t) = t^(a-1) (1 - t)^(b-1) / B(a, b),   B(a, b) = (a-1)! (b-1)! / (a+b-1)!

is a polynomial in t with rational coefficients, and so is its CDF. Products
of densities and CDFs are polynomials too, so every quantity the project
needs is the integral of a polynomial over [0, 1] and has an *exact rational
value*:

    P(X > Y)            = int_0^1 f_X(t) F_Y(t) dt
    P(k is best)        = int_0^1 f_k(t) prod_{j != k} F_j(t) dt
    E[max_j p_j]        = 1 - int_0^1 prod_j F_j(t) dt
    P(X - Y > d), d>=0  = int_d^1 f_X(t) F_Y(t - d) dt

Polynomials are lists of ``Fraction`` coefficients, index = power of t.
Nothing here uses floating point, so there is no rounding to analyse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, factorial
from typing import Iterable, Mapping

__all__ = [
    "Poly",
    "BetaExact",
    "poly_eval",
    "poly_integral",
    "prob_greater",
    "prob_best",
    "expected_max",
    "expected_loss",
    "prob_diff_greater",
]

Poly = list  # list[Fraction]; index k holds the coefficient of t**k


# ---------------------------------------------------------------------------
# Polynomial helpers
# ---------------------------------------------------------------------------


def _trim(p: Poly) -> Poly:
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def poly_mul(p: Poly, q: Poly) -> Poly:
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            out[i + j] += a * b
    return _trim(out)


def poly_eval(p: Poly, t: Fraction | int) -> Fraction:
    """Horner evaluation, exact for rational t."""
    t = Fraction(t)
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * t + c
    return acc


def poly_antiderivative(p: Poly) -> Poly:
    """P with P' = p and P(0) = 0."""
    return [Fraction(0)] + [c / (k + 1) for k, c in enumerate(p)]


def poly_integral(p: Poly, lo: Fraction | int = 0, hi: Fraction | int = 1) -> Fraction:
    P = poly_antiderivative(p)
    return poly_eval(P, hi) - poly_eval(P, lo)


def poly_shift(p: Poly, d: Fraction) -> Poly:
    """Coefficients of q(t) = p(t - d)."""
    d = Fraction(d)
    out = [Fraction(0)] * len(p)
    for k, c in enumerate(p):
        if c == 0:
            continue
        # (t - d)^k = sum_j C(k, j) t^j (-d)^(k-j)
        for j in range(k + 1):
            out[j] += c * comb(k, j) * (-d) ** (k - j)
    return _trim(out)


def _one_minus_t_pow(m: int) -> Poly:
    return [Fraction(comb(m, j) * (-1) ** j) for j in range(m + 1)]


def _product(polys: Iterable[Poly]) -> Poly:
    out: Poly = [Fraction(1)]
    for p in polys:
        out = poly_mul(out, p)
    return out


# ---------------------------------------------------------------------------
# Beta distribution with integer parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaExact:
    """Beta(a, b) with integer a, b >= 1, held exactly."""

    a: int
    b: int
    pdf: Poly = field(init=False, repr=False, compare=False)
    cdf: Poly = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (isinstance(self.a, int) and isinstance(self.b, int)) or self.a < 1 or self.b < 1:
            raise ValueError(f"BetaExact needs integers a, b >= 1; got a={self.a!r}, b={self.b!r}")
        kernel = [Fraction(0)] * (self.a - 1) + _one_minus_t_pow(self.b - 1)  # t^(a-1) (1-t)^(b-1)
        inv_B = 1 / self.beta_function
        pdf = [c * inv_B for c in kernel]
        cdf = poly_antiderivative(pdf)
        if poly_eval(cdf, 1) != 1:  # exact normalisation check
            raise ArithmeticError(f"Beta({self.a},{self.b}) CDF(1) = {poly_eval(cdf, 1)} != 1")
        object.__setattr__(self, "pdf", pdf)
        object.__setattr__(self, "cdf", cdf)

    @property
    def beta_function(self) -> Fraction:
        """B(a, b) = (a-1)! (b-1)! / (a+b-1)! exactly."""
        return Fraction(factorial(self.a - 1) * factorial(self.b - 1), factorial(self.a + self.b - 1))

    @property
    def mean(self) -> Fraction:
        return Fraction(self.a, self.a + self.b)

    @property
    def var(self) -> Fraction:
        s = self.a + self.b
        return Fraction(self.a * self.b, s * s * (s + 1))

    @property
    def mode(self) -> Fraction:
        if self.a <= 1 or self.b <= 1:
            raise ValueError("interior mode needs a > 1 and b > 1")
        return Fraction(self.a - 1, self.a + self.b - 2)

    def cdf_at(self, t: Fraction | int) -> Fraction:
        t = Fraction(t)
        if t <= 0:
            return Fraction(0)
        if t >= 1:
            return Fraction(1)
        return poly_eval(self.cdf, t)

    def odds_mean(self) -> Fraction:
        """E[theta / (1 - theta)] = a / (b - 1), finite when b > 1."""
        if self.b <= 1:
            raise ValueError("E[odds] is infinite when b <= 1")
        return Fraction(self.a, self.b - 1)


# ---------------------------------------------------------------------------
# Comparisons between independent posteriors
# ---------------------------------------------------------------------------


def prob_greater(x: BetaExact, y: BetaExact) -> Fraction:
    """P(X > Y) for independent X, Y: int_0^1 f_X(t) F_Y(t) dt."""
    return poly_integral(poly_mul(x.pdf, y.cdf))


def prob_best(dists: Mapping[str, BetaExact], k: str) -> Fraction:
    """P(p_k > p_j for every j != k): int_0^1 f_k(t) prod_{j != k} F_j(t) dt."""
    if k not in dists:
        raise KeyError(k)
    integrand = _product([dists[k].pdf] + [d.cdf for j, d in dists.items() if j != k])
    return poly_integral(integrand)


def expected_max(dists: Mapping[str, BetaExact]) -> Fraction:
    """E[max_j p_j] = int_0^1 (1 - prod_j F_j(t)) dt = 1 - int_0^1 prod_j F_j(t) dt."""
    return 1 - poly_integral(_product(d.cdf for d in dists.values()))


def expected_loss(dists: Mapping[str, BetaExact], k: str) -> Fraction:
    """Posterior expected opportunity loss of choosing k: E[max_j p_j - p_k]."""
    return expected_max(dists) - dists[k].mean


def prob_diff_greater(x: BetaExact, y: BetaExact, d: Fraction | int) -> Fraction:
    """P(X - Y > d) for independent X, Y and rational d.

    For 0 <= d < 1: int_d^1 f_X(t) F_Y(t - d) dt (t - d stays inside [0, 1]).
    For -1 < d < 0: 1 - P(Y - X > -d) (continuous, so > and >= agree).
    """
    d = Fraction(d)
    if d >= 1:
        return Fraction(0)
    if d <= -1:
        return Fraction(1)
    if d < 0:
        return 1 - prob_diff_greater(y, x, -d)
    integrand = poly_mul(x.pdf, poly_shift(y.cdf, d))
    return poly_integral(integrand, d, 1)
