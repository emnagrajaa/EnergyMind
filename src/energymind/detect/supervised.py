"""Supervised interval detector (the labelled upper bound).

We *have* labels in the benchmark, so a supervised model is legitimate and is
reported separately from the unsupervised detectors. It predicts, per
(household, interval), the probability that the interval falls inside an injected
event, using the same past-only features plus the physics residuals and the NILM
residual. Trained on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energymind import io
from energymind.detect.base import RESIDUAL_COLUMNS, WINDOW_FEATURES, design, to_score
from energymind.evaluate import point_labels

HOUSEHOLD_COL = io.HOUSEHOLD_COL

#: Interval features the supervised detector consumes.
SUPERVISED_FEATURES = [
    *WINDOW_FEATURES,
    *RESIDUAL_COLUMNS,
    "nilm_resid_total",
    "r_sub",
]


class SupervisedDetector:
    """LightGBM binary classifier over interval features, trained on injected events."""

    name = "supervised"

    def __init__(
        self,
        events: pd.DataFrame,
        tol_intervals: int = 1,
        features: list[str] | None = None,
        negative_ratio: float = 4.0,
        random_state: int = 0,
    ) -> None:
        self.events = events
        self.tol_intervals = tol_intervals
        self.features = features or SUPERVISED_FEATURES
        self.negative_ratio = negative_ratio
        self.random_state = random_state
        self.model = None

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.features if c in frame.columns]
        return design(frame, cols)

    def fit(self, frame: pd.DataFrame) -> SupervisedDetector:
        from lightgbm import LGBMClassifier

        tol = pd.Timedelta(minutes=30 * self.tol_intervals)
        y = point_labels(frame, self.events, tol).to_numpy()
        x = self._features(frame)

        pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        if len(pos) == 0:
            self.model = None
            return self
        rng = np.random.default_rng(self.random_state)
        n_neg = min(len(neg), int(len(pos) * self.negative_ratio))
        neg = rng.choice(neg, size=n_neg, replace=False)
        idx = np.concatenate([pos, neg])
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            verbose=-1,
            random_state=self.random_state,
        )
        model.fit(x.iloc[idx], y[idx])
        self.model = model
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return to_score(np.zeros(len(frame)), frame.index)
        proba = self.model.predict_proba(self._features(frame))[:, 1]
        return to_score(proba, frame.index)
