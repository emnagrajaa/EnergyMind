"""Matrix profile: discords (STUMPY) plus a dedicated replay/motif check.

* **Discords** find the most unusual subsequence per household (spikes, stuck-at,
  dropouts).
* **Replay check** looks for a window that is a near-exact copy of the same window
  ``lag`` intervals earlier (7 days by default). A trivial copy has a *very small*
  distance to its earlier self -- which a discord detector cannot see, because a
  copied normal day looks perfectly normal on its own.

The detector returns the sum of a standardised discord score and a standardised
replay score, so either signal can push a row over the alert threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energymind import io
from energymind.detect.base import to_score

HOUSEHOLD_COL = io.HOUSEHOLD_COL
DAY = 48  # intervals per day
EPS = 1e-3


def _fill(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    s = s.interpolate(limit_direction="both")
    return s.to_numpy()


def _discord(values: np.ndarray, window: int) -> np.ndarray:
    """Matrix-profile discord score aligned to the window's last row (length n)."""
    n = len(values)
    out = np.full(n, np.nan)
    if n < window + 2:
        return out
    try:
        import stumpy

        mp = np.asarray(stumpy.stump(values, m=window)[:, 0], dtype=float)
    except Exception:
        mp = _naive_mp(values, window)
    out[window - 1 : window - 1 + len(mp)] = mp
    return out


def _naive_mp(values: np.ndarray, window: int) -> np.ndarray:
    """Small numpy fallback for the matrix profile (O(n^2))."""
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(values, window)
    m = len(windows)
    profile = np.empty(m)
    for i in range(m):
        d = np.sqrt(((windows - windows[i]) ** 2).mean(axis=1))
        d[i] = np.inf
        profile[i] = d.min()
    return profile


def _replay(values: np.ndarray, lag: int, window: int) -> np.ndarray:
    """``-log`` distance to the window ``lag`` intervals earlier, aligned to end rows."""
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(values)
    out = np.full(n, np.nan)
    if n < lag + window + 1:
        return out
    windows = sliding_window_view(values, window)  # (n-window+1, window)
    dist = np.sqrt(((windows[lag:] - windows[:-lag]) ** 2).mean(axis=1))
    end_rows = (window - 1) + lag + np.arange(len(dist))
    out[end_rows] = -np.log(dist + EPS)
    return out


class MatrixProfileDetector:
    """Univariate P: discord score + 7-day replay score, standardised per household."""

    name = "matrix_profile"

    def __init__(self, window: int = DAY, lag: int = 7 * DAY, n_jobs: int | None = None) -> None:
        self.window = window
        self.lag = lag
        self.n_jobs = n_jobs

    def fit(self, frame: pd.DataFrame) -> MatrixProfileDetector:  # noqa: ARG002 - unsupervised
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0.0, index=frame.index)
        for _household, group in frame.groupby(HOUSEHOLD_COL, sort=False):
            values = _fill(group["active_power_kw"].to_numpy())
            discord = _discord(values, self.window)
            replay = _replay(values, self.lag, self.window)
            scores.loc[group.index] = _z(discord) + _z(replay)
        return to_score(scores.to_numpy(), frame.index)


def _z(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(v)
    if not np.isfinite(scale) or scale <= 0:
        return np.nan_to_num(v, nan=0.0)
    return np.nan_to_num((v - med) / scale, nan=0.0)
