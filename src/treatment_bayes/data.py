"""Trial data: treatment -> (patients n, successes s).

The project sheet (MA 232 Group Project 2) gives simulated clinical-trial
data: A 62/100, B 70/100, C 66/100. ``load_trial`` reads the same table from
data/trial.csv so the numbers live in one place.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Arm", "load_trial", "DEFAULT_CSV"]

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "trial.csv"


@dataclass(frozen=True)
class Arm:
    """One treatment arm: ``n`` patients, ``s`` successes, ``n - s`` failures."""

    name: str
    n: int
    s: int

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or not isinstance(self.s, int):
            raise ValueError(f"{self.name}: patients and successes must be integers")
        if self.n < 1:
            raise ValueError(f"{self.name}: need at least one patient, got n={self.n}")
        if not 0 <= self.s <= self.n:
            raise ValueError(f"{self.name}: successes must lie in [0, n]; got s={self.s}, n={self.n}")

    @property
    def failures(self) -> int:
        return self.n - self.s


def load_trial(path: str | Path = DEFAULT_CSV) -> dict[str, Arm]:
    """Read ``treatment,patients,successes`` rows into an ordered dict of arms."""
    arms: dict[str, Arm] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        expected = {"treatment", "patients", "successes"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path}: columns must be {sorted(expected)}, got {reader.fieldnames}")
        for row in reader:
            name = row["treatment"].strip()
            if name in arms:
                raise ValueError(f"{path}: treatment {name!r} appears twice")
            arms[name] = Arm(name, int(row["patients"]), int(row["successes"]))
    if len(arms) < 2:
        raise ValueError(f"{path}: need at least two treatments to compare")
    return arms
