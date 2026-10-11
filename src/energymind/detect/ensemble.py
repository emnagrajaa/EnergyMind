"""Soft-vote ensemble over calibrated members (blueprint Section 8).

Combines detector scores by standardising each per household (robust median/MAD)
and averaging, so members with different scales contribute comparably. Only the
unsupervised members are combined: stacking the supervised detector would hide the
unsupervised signal that the paper reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energymind import io
from energymind.detect.base import to_score

HOUSEHOLD_COL = io.HOUSEHOLD_COL


def _household_z(frame: pd.DataFrame, scores: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=scores.index, dtype=float)
    for _household, group in frame.groupby(HOUSEHOLD_COL, sort=False):
        v = scores.loc[group.index].to_numpy(dtype=float)
        med = np.nanmedian(v)
        mad = np.nanmedian(np.abs(v - med))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            scale = np.nanstd(v) or 1.0
        out.loc[group.index] = (v - med) / scale
    return out.fillna(0.0)


class EnsembleDetector:
    """Mean of per-household standardised member scores."""

    name = "ensemble"

    def __init__(self, members: list | None = None) -> None:
        if members is None:
            from energymind.detect.iforest import IsolationForestDetector
            from energymind.detect.matrix_profile import MatrixProfileDetector
            from energymind.detect.pyod_models import ECODDetector
            from energymind.detect.rules import Rules

            members = [Rules(), ECODDetector(), MatrixProfileDetector(), IsolationForestDetector()]
        self.members = members

    def fit(self, frame: pd.DataFrame) -> EnsembleDetector:
        for member in self.members:
            member.fit(frame)
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        total = pd.Series(0.0, index=frame.index, dtype=float)
        for member in self.members:
            total += _household_z(frame, member.score(frame))
        return to_score(total.to_numpy(), frame.index)
