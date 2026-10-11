"""Detector interface and shared helpers (blueprint Step 8).

Every detector consumes the Step 6 feature table (past-only features, physics
residuals, coherence, NILM residual) and returns a **higher-is-more-anomalous**
score per (household, interval). Detectors are fit on the training rows only and
scored on validation/test.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from energymind import io

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL
INTERVAL = pd.Timedelta(minutes=30)

#: Physics-identity residual columns and the tolerance used to call a violation.
RESIDUAL_COLUMNS = ["r_S", "r_VI", "r_E", "r_PF"]
PHYSICS_TOL = 1e-2

#: Window features the tabular detectors (Isolation Forest, classifier) consume.
WINDOW_FEATURES = [
    "p_roll_mean_2h",
    "p_roll_mean_24h",
    "p_roll_mean_7d",
    "p_roll_std_24h",
    "base_load_24h",
    "load_factor_24h",
    "p_ramp",
    "p_resid_z",
    "unmetered_share",
    "sub1_share",
    "sub2_share",
    "sub3_share",
    "sub4_share",
    "pv_offset_ratio",
    "neigh_share",
    "i_roll_mean_2h",
    "v_roll_mean_2h",
    "v_roll_mean_7d",
]


@runtime_checkable
class Detector(Protocol):
    """Minimal detector protocol (``fit`` may be a no-op for rule-based models)."""

    name: str

    def fit(self, frame: pd.DataFrame) -> Detector: ...

    def score(self, frame: pd.DataFrame) -> pd.Series: ...


def physics_violation(frame: pd.DataFrame) -> pd.Series:
    """Max absolute identity residual (NaN -> 0)."""
    return frame[RESIDUAL_COLUMNS].abs().max(axis=1).fillna(0.0)


def design(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Numeric design matrix with NaNs replaced by 0 (tree models tolerate gaps)."""
    return frame[features].astype(float).fillna(0.0)


def to_score(values, index) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=index, name="score")
