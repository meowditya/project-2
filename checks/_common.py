"""Shared helpers for the checks (plain Python, no pytest needed)."""

from __future__ import annotations

import re
import sys
from fractions import Fraction
from math import comb, factorial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA = {"A": (100, 62), "B": (100, 70), "C": (100, 66)}  # the project sheet's table, typed independently


def assert_close(actual, expected, *, rtol: float = 0.0, atol: float = 0.0, what: str = "") -> None:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    if a.shape != e.shape:
        raise AssertionError(f"{what}: shape {a.shape} != {e.shape}")
    err = np.abs(a - e)
    lim = atol + rtol * np.abs(e)
    if not np.all(err <= lim):
        i = int(np.argmax(err - lim))
        raise AssertionError(f"{what}: {a.flat[i]!r} vs {e.flat[i]!r} (|diff| {err.flat[i]:.3g} > {lim.flat[i]:.3g})")


def assert_raises(exc, fn, *args, match: str | None = None, **kw) -> None:
    try:
        fn(*args, **kw)
    except exc as e:
        if match and not re.search(match, str(e)):
            raise AssertionError(f"{exc.__name__} raised, but message {str(e)!r} lacks {match!r}")
        return
    raise AssertionError(f"expected {exc.__name__}")


# --- independent exact oracles (no code shared with src/) ---------------------

def beta_fn(a: int, b: int) -> Fraction:
    return Fraction(factorial(a - 1) * factorial(b - 1), factorial(a + b - 1))


def beta_cdf_binomial_tail(a: int, b: int, x: Fraction) -> Fraction:
    """I_x(a, b) = P(Binomial(a+b-1, x) >= a), valid for integer a, b >= 1."""
    m = a + b - 1
    return sum((comb(m, j) * x**j * (1 - x) ** (m - j) for j in range(a, m + 1)), Fraction(0))


def prob_greater_closed_form(ax: int, bx: int, ay: int, by: int) -> Fraction:
    """P(X > Y), X ~ Beta(ax, bx), Y ~ Beta(ay, by), integers: a finite closed-form sum.

    P(X > Y) = sum_{i=0}^{ax-1} B(ay + i, by + bx) / ((bx + i) B(1 + i, bx) B(ay, by)),
    obtained by expanding the Beta CDF of X as a binomial tail and integrating
    term by term against the density of Y.
    """
    return sum((beta_fn(ay + i, by + bx) / ((bx + i) * beta_fn(1 + i, bx) * beta_fn(ay, by)) for i in range(ax)),
               Fraction(0))
