"""Isolation Forest on past-only window features (classical unsupervised ML)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from energymind.detect.base import WINDOW_FEATURES, design, to_score


class IsolationForestDetector:
    """Isolation Forest over the window-feature vector.

    Fitted on the training rows only; the score is ``-score_samples`` so that
    larger values mean *more* anomalous.
    """

    name = "iforest"

    def __init__(
        self,
        features: list[str] | None = None,
        n_estimators: int = 200,
        max_samples: str | int = "auto",
        random_state: int = 0,
    ) -> None:
        self.features = features or WINDOW_FEATURES
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model: IsolationForest | None = None

    def fit(self, frame: pd.DataFrame) -> IsolationForestDetector:
        x = design(frame, self.features)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(x)
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("IsolationForestDetector.fit must be called first")
        x = design(frame, self.features)
        return to_score(-self.model.score_samples(x), frame.index)


def ecod_score(frame: pd.DataFrame, features: list[str] | None = None) -> pd.Series:
    """Cheap parameter-free alternative (ECOD-like empirical tail score).

    Used when PyOD is unavailable. For each feature it is the negative log of the
    two-sided empirical tail probability, summed across features. Constant
    features carry no tail information and are skipped; ties get their average
    rank, so an all-equal column cannot manufacture a spurious edge score.
    """
    from scipy.stats import rankdata

    features = features or WINDOW_FEATURES
    x = design(frame, features)
    n = len(x)
    total = np.zeros(n)
    for column in features:
        v = x[column].to_numpy()
        if not np.isfinite(v).any() or np.nanstd(v) == 0:
            continue
        rank = rankdata(v, method="average")
        tail = np.minimum(rank, n - rank + 1) / n
        total += -np.log(np.clip(tail, 1e-6, 1.0))
    return to_score(total, frame.index)
