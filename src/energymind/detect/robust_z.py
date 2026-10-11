"""Robust z versus the household hour-of-week baseline (the bar to beat)."""

from __future__ import annotations

import pandas as pd

from energymind.detect.base import to_score


class RobustZ:
    """Absolute robust residual ``|P - median(how)| / (1.4826 * MAD(how))``.

    The per-household hour-of-week median and MAD live in the feature table
    (``p_resid_z``), fitted on the training window; this detector is a no-op to
    fit and simply exposes that score.
    """

    name = "robust_z"

    def fit(self, frame: pd.DataFrame) -> RobustZ:  # noqa: ARG002 - nothing to fit
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        return to_score(frame["p_resid_z"].abs(), frame.index)
