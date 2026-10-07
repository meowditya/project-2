"""Module 5 Bayesian approximations, single parameter: Lindley and Tierney-Kadane.

Both approximate a posterior expectation

    E[u(theta) | x] = int u(theta) exp{L(theta)} pi(theta) dtheta / int exp{L(theta)} pi(theta) dtheta

as in MA 232 Module 5, slides 7-19, using the course's notation:

Lindley (slides 8-11), expanding around the MLE theta_hat:
    sigma^2 = -1 / L2(theta_hat)
    E[u | x] ~= u(theta_hat) + (sigma^2 / 2) (u2 + 2 u1 rho1) + (sigma^4 / 2) L3 u1
with L2, L3 the 2nd and 3rd derivatives of the log-likelihood, u1, u2 the
derivatives of u, rho1 the derivative of the log-prior, all at theta_hat.

Tierney-Kadane (slides 14-15, 19), for u > 0:
    L0(theta) = [L(theta) + log pi(theta)] / n,   L*(theta) = L0(theta) + log u(theta) / n
    theta0 = argmax L0, theta* = argmax L*, sigma0^2 = -1 / L0''(theta0), sigma*^2 = -1 / L*''(theta*)
    E[u | x] ~= (sigma* / sigma0) exp{ n [L*(theta*) - L0(theta0)] }

The generic functions take the needed derivatives as callables, exactly as
the course's R code does; ``binomial_*`` specialise them to the treatment
problem, where the conjugate posterior gives the exact answer to compare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from scipy import optimize

__all__ = [
    "lindley",
    "tierney_kadane",
    "TKResult",
    "binomial_lindley",
    "binomial_tierney_kadane",
    "U_FUNCTIONS",
]

Fn = Callable[[float], float]


def lindley(theta_hat: float, L2: Fn, L3: Fn, u: Fn, u1: Fn, u2: Fn, rho1: Fn) -> float:
    """Single-parameter Lindley approximation (Module 5, slide 11)."""
    l2 = L2(theta_hat)
    if not l2 < 0:
        raise ValueError(f"L2(theta_hat) = {l2} must be negative at the MLE")
    s2 = -1.0 / l2
    return (u(theta_hat) + 0.5 * s2 * (u2(theta_hat) + 2.0 * u1(theta_hat) * rho1(theta_hat))
            + 0.5 * s2 * s2 * L3(theta_hat) * u1(theta_hat))


@dataclass(frozen=True)
class TKResult:
    value: float
    theta0: float
    sigma0: float
    theta_star: float
    sigma_star: float
    L0_at_theta0: float
    Lstar_at_theta_star: float


def tierney_kadane(n: int, L0: Fn, L0p: Fn, L0pp: Fn, Ls: Fn, Lsp: Fn, Lspp: Fn,
                   bracket: tuple[float, float]) -> TKResult:
    """Single-parameter T-K (Module 5, slides 14-15), roots found as in the slide-19 R code."""
    theta0 = optimize.brentq(L0p, *bracket, xtol=1e-14, rtol=1e-14)
    theta_s = optimize.brentq(Lsp, *bracket, xtol=1e-14, rtol=1e-14)
    c0, cs = L0pp(theta0), Lspp(theta_s)
    if not (c0 < 0 and cs < 0):
        raise ValueError("L0 and L* must be maximised (negative second derivative) at their roots")
    sigma0, sigma_s = math.sqrt(-1.0 / c0), math.sqrt(-1.0 / cs)
    l0, ls = L0(theta0), Ls(theta_s)
    value = (sigma_s / sigma0) * math.exp(n * (ls - l0))
    return TKResult(value, theta0, sigma0, theta_s, sigma_s, l0, ls)


# ---------------------------------------------------------------------------
# Binomial likelihood, Beta(a0, b0) prior
# ---------------------------------------------------------------------------

#: u(theta) with its first two derivatives. "odds" = theta / (1 - theta).
U_FUNCTIONS: dict[str, tuple[Fn, Fn, Fn]] = {
    "mean": (lambda t: t, lambda t: 1.0, lambda t: 0.0),
    "odds": (lambda t: t / (1 - t), lambda t: 1.0 / (1 - t) ** 2, lambda t: 2.0 / (1 - t) ** 3),
}


def _check(n: int, s: int) -> None:
    if not 0 < s < n:
        raise ValueError("Lindley/T-K here need 0 < s < n so the MLE is interior")


def binomial_lindley(n: int, s: int, which: str = "mean", a0: float = 1.0, b0: float = 1.0) -> float:
    """Lindley approximation to E[u(theta) | s of n] under a Beta(a0, b0) prior.

    L(t) = s log t + (n - s) log(1 - t); MLE t_hat = s / n.
    L2 = -s/t^2 - (n-s)/(1-t)^2,  L3 = 2s/t^3 - 2(n-s)/(1-t)^3,
    rho1 = (a0 - 1)/t - (b0 - 1)/(1 - t).
    """
    _check(n, s)
    f = n - s
    u, u1, u2 = U_FUNCTIONS[which]
    return lindley(
        s / n,
        L2=lambda t: -s / t**2 - f / (1 - t) ** 2,
        L3=lambda t: 2 * s / t**3 - 2 * f / (1 - t) ** 3,
        u=u, u1=u1, u2=u2,
        rho1=lambda t: (a0 - 1) / t - (b0 - 1) / (1 - t),
    )


def binomial_tierney_kadane(n: int, s: int, which: str = "mean", a0: float = 1.0, b0: float = 1.0) -> TKResult:
    """T-K approximation to E[u(theta) | s of n] under a Beta(a0, b0) prior.

    n L0(t) = (s + a0 - 1) log t + (n - s + b0 - 1) log(1 - t) (constants cancel in the ratio).
    """
    _check(n, s)
    p, q = s + a0 - 1, n - s + b0 - 1  # exponents of t and (1 - t) in the posterior kernel
    L0 = lambda t: (p * math.log(t) + q * math.log(1 - t)) / n
    L0p = lambda t: (p / t - q / (1 - t)) / n
    L0pp = lambda t: (-p / t**2 - q / (1 - t) ** 2) / n
    if which == "mean":  # log u = log t
        Ls = lambda t: L0(t) + math.log(t) / n
        Lsp = lambda t: L0p(t) + 1.0 / (n * t)
        Lspp = lambda t: L0pp(t) - 1.0 / (n * t**2)
    elif which == "odds":  # log u = log t - log(1 - t)
        Ls = lambda t: L0(t) + (math.log(t) - math.log(1 - t)) / n
        Lsp = lambda t: L0p(t) + (1.0 / t + 1.0 / (1 - t)) / n
        Lspp = lambda t: L0pp(t) + (-1.0 / t**2 + 1.0 / (1 - t) ** 2) / n
    else:
        raise ValueError(f"unknown u: {which!r}")
    eps = 1e-9
    return tierney_kadane(n, L0, L0p, L0pp, Ls, Lsp, Lspp, bracket=(eps, 1 - eps))
