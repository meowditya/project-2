"""Bayesian comparison of treatments from binomial trial data (MA 232 Group Project 2)."""

from .data import Arm, load_trial
from .posterior import PRIORS, BetaPosterior, Prior, posterior

__all__ = ["Arm", "load_trial", "PRIORS", "BetaPosterior", "Prior", "posterior"]
