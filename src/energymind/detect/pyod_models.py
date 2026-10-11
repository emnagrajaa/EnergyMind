"""ECOD on window features (parameter-free tabular detector, PyOD)."""

from __future__ import annotations

import pandas as pd

from energymind.detect.base import WINDOW_FEATURES, design, to_score
from energymind.detect.iforest import ecod_score


class ECODDetector:
    """PyOD's ECOD empirical-tail detector; falls back to a numpy ECOD if absent."""

    name = "ecod"

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = features or WINDOW_FEATURES
        self.model = None

    def fit(self, frame: pd.DataFrame) -> ECODDetector:
        try:
            from pyod.models.ecod import ECOD

            self.model = ECOD().fit(design(frame, self.features))
        except Exception:
            self.model = None
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return ecod_score(frame, self.features)
        x = design(frame, self.features)
        return to_score(self.model.decision_function(x), frame.index)
