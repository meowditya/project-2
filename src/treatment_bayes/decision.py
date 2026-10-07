"""Choosing a treatment as a Bayes decision.

Module 4 (Definition 4.3) defines the Bayes action as the one that minimises
posterior expected loss. Here the action is a choice k in {A, B, C}.

* Opportunity loss L(p, k) = max_j p_j - p_k (success rate given up by not
  choosing the best). Its posterior expectation is E[max_j p_j] - E[p_k];
  the first term does not depend on k, so the Bayes action is the treatment
  with the largest posterior mean.
* Zero-one loss L(p, k) = 1 if k is not the best, else 0. Its posterior
  expectation is 1 - P(k is best), so the Bayes action is the treatment
  most likely to be best.

These mirror the course's squared-error and zero-one losses: the first
rewards being close, the second rewards being exactly right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = ["Decision", "decide"]


@dataclass(frozen=True)
class Decision:
    by_expected_loss: str
    by_prob_best: str
    expected_loss: dict[str, float]
    prob_best: dict[str, float]

    @property
    def agree(self) -> bool:
        return self.by_expected_loss == self.by_prob_best


def decide(expected_loss: Mapping[str, float], prob_best: Mapping[str, float]) -> Decision:
    if set(expected_loss) != set(prob_best):
        raise ValueError("expected_loss and prob_best must cover the same treatments")
    by_loss = min(expected_loss, key=expected_loss.__getitem__)
    by_best = max(prob_best, key=prob_best.__getitem__)
    return Decision(by_loss, by_best, dict(expected_loss), dict(prob_best))
