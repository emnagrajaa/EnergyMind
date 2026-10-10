"""EDA-level feature engineering (blueprint Section 6, prototype for Step 6).

These are the features the exploratory analysis needs in order to *look* at the
data the way the detectors will later see it. ``features.py`` (Step 6) remains the
production implementation; this module is intentionally small, self-contained and
**past-only by construction** so the EDA figures can be reproduced without it.

Conventions
-----------
* Everything is computed on the clean baseline (``clean.parquet``), per household,
  on the local-time calendar (UTC-6, no DST).
* **Past-only** features use ``shift(1)`` *before* rolling, so a change of the value
  at ``t`` cannot alter them at ``t``. Features that deliberately use the current
  value (the residual, the ramp, the shares) are listed in :data:`CURRENT_VALUE`.
* Anything fitted (hour-of-week medians, MAD scale, "on" thresholds) is fitted on
  the **training window only** (``train_end``) to avoid leakage into val/test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energymind import io
from energymind.validate import SUB_COLS

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL
LOCAL_OFFSET_H = 6  # file time is UTC-6, no DST

INTERVALS_PER_HOUR = 2
#: Shares are undefined when total energy is ~0 (3 % of rows have E < 0.1 kWh and the
#: ratio explodes), so they are only computed above this floor.
MIN_E_KWH = 0.1
WINDOWS = {"2h": 4, "24h": 48, "7d": 336}  # in 30-minute intervals

#: Features that use the value at ``t`` on purpose (everything else is past-only).
CURRENT_VALUE = [
    "p_resid",
    "p_resid_z",
    "p_ramp",
    "unmetered_kwh",
    "unmetered_share",
    "sub1_share",
    "sub2_share",
    "sub3_share",
    "sub4_share",
    "pv_offset_ratio",
    "neigh_share",
]

#: Short, paper-ready description of each engineered feature.
FEATURE_TABLE = [
    ("p_roll_mean_{2h,24h,7d}", "past-only rolling mean of active power", "past-only"),
    ("p_roll_std_{2h,24h,7d}", "past-only rolling std of active power", "past-only"),
    ("base_load_24h", "past-only 24 h rolling minimum of P (night base load)", "past-only"),
    ("load_factor_24h", "past-only 24 h mean / max of P", "past-only"),
    ("p_ramp", "P(t) - P(t-1) (ramp rate, kW per interval)", "current value"),
    ("p_resid", "P minus the household's hour-of-week median (train-fitted)", "current value"),
    (
        "p_resid_z",
        "p_resid / (1.4826 * MAD of the same hour-of-week, train-fitted)",
        "current value",
    ),
    (
        "unmetered_kwh",
        "E - sum(sub1..sub4) - 0.5 * EV (energy not seen by sub-meters)",
        "current value",
    ),
    (
        "sub{1..4}_share",
        "sub-meter k energy / total energy (only where E >= 0.1 kWh)",
        "current value",
    ),
    ("pv_offset_ratio", "PV / (P + PV): how much of the gross load PV covers", "current value"),
    (
        "neigh_share",
        "share of households with |p_resid_z| > 2 at the same timestamp",
        "current value",
    ),
    (
        "ev_session_len",
        "consecutive 30-min intervals with EV load > 0 around t (charging burst length)",
        "derived",
    ),
]


# --------------------------------------------------------------------------- #
# Local-time helpers
# --------------------------------------------------------------------------- #


def add_local_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add local-calendar columns (no mutation of the input)."""
    out = df.copy()
    local = out[UTC_COL].dt.tz_convert(None) - pd.Timedelta(hours=LOCAL_OFFSET_H)
    out["ts_local"] = local
    out["date_local"] = local.dt.normalize()
    out["hour_local"] = local.dt.hour
    out["slot"] = local.dt.hour * 2 + local.dt.minute // 30  # 0..47
    out["dow"] = local.dt.dayofweek
    out["month"] = local.dt.month
    out["how"] = out["dow"] * 48 + out["slot"]  # hour-of-week slot, 0..335
    out["season"] = out["month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def _past_roll(s: pd.Series, window: int, how: str) -> pd.Series:
    """``shift(1)`` first, then roll: the value at t never enters its own window."""
    r = s.shift(1).rolling(window, min_periods=max(2, window // 2))
    return getattr(r, how)()


def fit_baseline(df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """Per-household hour-of-week median and MAD fitted on the training window."""
    tr = df[df["ts_local"] < train_end]
    g = tr.groupby([HOUSEHOLD_COL, "how"])["active_power_kw"]
    base = pd.DataFrame(
        {
            "how_median": g.median(),
            "how_mad": g.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))),
        }
    ).reset_index()
    return base


def engineer_features(df: pd.DataFrame, train_end: str | pd.Timestamp) -> pd.DataFrame:
    """Return ``df`` (with local-time columns) plus the engineered EDA features.

    ``df`` is ``clean.parquet``. ``train_end`` is the exclusive end of the training
    window (``2020-09-01`` in ``configs/data.yaml``).
    """
    train_end = pd.Timestamp(train_end)
    out = add_local_time(df).sort_values([HOUSEHOLD_COL, UTC_COL]).reset_index(drop=True)
    p = out["active_power_kw"]
    grp = out.groupby(HOUSEHOLD_COL, sort=False)["active_power_kw"]

    # --- past-only rolling statistics ---------------------------------------
    for name, w in WINDOWS.items():
        out[f"p_roll_mean_{name}"] = grp.transform(lambda s, w=w: _past_roll(s, w, "mean"))
        out[f"p_roll_std_{name}"] = grp.transform(lambda s, w=w: _past_roll(s, w, "std"))
    out["base_load_24h"] = grp.transform(lambda s: _past_roll(s, WINDOWS["24h"], "min"))
    roll_max = grp.transform(lambda s: _past_roll(s, WINDOWS["24h"], "max"))
    out["load_factor_24h"] = out["p_roll_mean_24h"] / roll_max.replace(0, np.nan)

    # --- ramp and baseline residual (current-value features) -----------------
    out["p_ramp"] = grp.diff()
    base = fit_baseline(out, train_end)
    out = out.merge(base, on=[HOUSEHOLD_COL, "how"], how="left")
    out["p_resid"] = p.to_numpy() - out["how_median"]
    scale = (1.4826 * out["how_mad"]).replace(0, np.nan)
    out["p_resid_z"] = out["p_resid"] / scale

    # --- energy balance and sub-meter shares ---------------------------------
    subs = out[SUB_COLS].sum(axis=1)
    out["unmetered_kwh"] = out["total_energy_kwh"] - subs - 0.5 * out["ev_load_kw"]
    e = out["total_energy_kwh"].where(out["total_energy_kwh"] >= MIN_E_KWH)
    out["unmetered_share"] = out["unmetered_kwh"] / e
    for k, col in enumerate(SUB_COLS, start=1):
        out[f"sub{k}_share"] = out[col] / e

    # --- DER ------------------------------------------------------------------
    gross = out["active_power_kw"] + out["solar_generation_kw"]
    out["pv_offset_ratio"] = np.where(
        out["has_solar"].eq(1) & (gross > 0), out["solar_generation_kw"] / gross, np.nan
    )
    ev_on = out["ev_load_kw"].fillna(0).gt(0)
    new_run = ev_on.ne(ev_on.shift()) | out[HOUSEHOLD_COL].ne(out[HOUSEHOLD_COL].shift())
    run_id = new_run.cumsum()
    run_len = ev_on.groupby(run_id).transform("size")
    out["ev_session_len"] = np.where(ev_on, run_len, 0)

    # --- neighbourhood coherence ---------------------------------------------
    flag = (out["p_resid_z"].abs() > 2).astype(float).where(out["p_resid_z"].notna())
    out["neigh_share"] = flag.groupby(out[UTC_COL]).transform("mean")
    return out


#: Columns used by the feature-correlation figure, in display order.
CORR_FEATURES = [
    "active_power_kw",
    "p_ramp",
    "p_roll_mean_2h",
    "p_roll_mean_24h",
    "p_roll_mean_7d",
    "p_roll_std_24h",
    "base_load_24h",
    "load_factor_24h",
    "p_resid_z",
    "unmetered_share",
    "sub1_share",
    "sub4_share",
    "neigh_share",
]
