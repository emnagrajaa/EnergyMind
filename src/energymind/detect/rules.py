"""Physics + coherence rules.

Two exact-ish signals:

* a **physics violation** - a naive injection breaks one of the electrical
  identities (finding F1), so ``max |r_*|`` jumps above the tolerance;
* a **baseline deviation** - a consistent injection keeps the identities but
  moves the load away from the household's hour-of-week baseline.

Cross-household coherence (finding F10) is exposed as ``neigh_share`` and is used
by the origin classifier rather than the detection score: a single-meter fault
has a *low* neighbourhood share, a genuine grid event a *high* one.
"""

from __future__ import annotations

import pandas as pd

from energymind.detect.base import PHYSICS_TOL, physics_violation, to_score


class Rules:
    """Composite rule score: identity violation dominates, baseline z backs it up."""

    name = "rules"

    def __init__(self, tolerance: float = PHYSICS_TOL) -> None:
        self.tolerance = tolerance

    def fit(self, frame: pd.DataFrame) -> Rules:  # noqa: ARG002 - rules need no fitting
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        physics = physics_violation(frame)
        z = frame["p_resid_z"].abs().fillna(0.0).to_numpy()
        violation = (physics > self.tolerance).to_numpy()
        # A broken identity is a near-certain anomaly; otherwise fall back to the
        # baseline deviation so consistent injections are still ranked.
        combined = z + violation * (100.0 + physics.to_numpy() * 1e3)
        return to_score(combined, frame.index)
