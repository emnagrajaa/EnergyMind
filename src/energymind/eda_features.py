"""EDA-level feature engineering - now a thin re-export of :mod:`energymind.features`.

This module used to hold a small, self-contained prototype of the Step 6
pipeline. To remove the risk of the EDA features and the production features
diverging, the implementation moved to :mod:`energymind.features` (canonical) and
this module simply re-exports the names the EDA code and `02_eda.ipynb` rely on.

New code should import from :mod:`energymind.features`.
"""

from __future__ import annotations

from energymind.features import (
    CORR_FEATURES,
    CURRENT_VALUE,
    FEATURE_TABLE,
    HOUSEHOLD_COL,
    INTERVALS_PER_HOUR,
    LOCAL_OFFSET_H,
    MIN_E_KWH,
    UTC_COL,
    WINDOWS,
    add_local_time,
    engineer_features,
    fit_baseline,
)

__all__ = [
    "CORR_FEATURES",
    "CURRENT_VALUE",
    "FEATURE_TABLE",
    "HOUSEHOLD_COL",
    "INTERVALS_PER_HOUR",
    "LOCAL_OFFSET_H",
    "MIN_E_KWH",
    "UTC_COL",
    "WINDOWS",
    "add_local_time",
    "engineer_features",
    "fit_baseline",
]
