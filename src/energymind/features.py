"""Step 6 - canonical past-only feature pipeline (blueprint Section 6 / Step 6).

This is the production feature builder. It runs on the **injected** dataset (never
on the clean baseline) so that detectors never reuse a feature that was computed
before an anomaly was introduced. ``eda_features.py`` is now a thin re-export of
this module, so the EDA figures and this pipeline can never diverge.

Design rules
------------
* Everything is computed per household, on the local-time calendar (UTC-6, no DST).
* **Past-only** features use ``shift(1)`` *before* rolling, so a change of the
  value at ``t`` cannot alter them at ``t``. Features that deliberately use the
  current value (the residual, the ramp, the shares) are listed in
  :data:`CURRENT_VALUE`.
* Anything fitted (hour-of-week medians, MAD scale, NILM regressors) is fitted on
  the **training window / training households only** to avoid leakage into
  val/test.

Build everything with::

    python -m energymind.features               # features.parquet + nilm_predictions.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from energymind import io, nilm
from energymind.config import Settings, load_settings
from energymind.validate import SUB_COLS, flat_run_length, physics_residuals

HOUSEHOLD_COL = io.HOUSEHOLD_COL
UTC_COL = io.UTC_COL
LOCAL_OFFSET_H = 6  # file time is UTC-6, no DST

INTERVALS_PER_HOUR = 2
#: Shares are undefined when total energy is ~0, so they are computed above this floor.
MIN_E_KWH = 0.1
WINDOWS = {"2h": 4, "24h": 48, "7d": 336}  # in 30-minute intervals
#: Physics-residual columns (identity checks).
RESIDUAL_COLUMNS = ["r_S", "r_VI", "r_E", "r_PF", "r_sub"]

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
    ("i_roll_mean_{2h,24h,7d}", "past-only rolling mean of current", "past-only"),
    ("v_roll_mean_{2h,24h,7d}", "past-only rolling mean of voltage", "past-only"),
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
    ("sub{1..4}_share", "sub-meter k energy / total energy (where E >= 0.1 kWh)", "current value"),
    ("pv_offset_ratio", "PV / (P + PV): how much of the gross load PV covers", "current value"),
    (
        "neigh_share",
        "share of households with |p_resid_z| > 2 at the same timestamp",
        "current value",
    ),
    ("r_S, r_VI, r_E, r_PF, r_sub", "relative electrical-identity residuals", "current value"),
    ("flat_run_p", "length of the flat run of P ending at t", "current value"),
    ("nilm_resid_sub{1..4}", "measured sub-meter minus NILM estimate", "current value"),
    (
        "ev_session_len",
        "consecutive 30-min intervals with EV load > 0 around t (charging burst length)",
        "derived",
    ),
]

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
# Past-only rolling and the fitted baseline
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
    """Return ``df`` (with local-time columns) plus the past-only core features.

    ``df`` is ``clean.parquet`` or ``injected.parquet``. ``train_end`` is the
    exclusive end of the training window (``2020-09-01`` in ``configs/data.yaml``).
    """
    train_end = pd.Timestamp(train_end)
    out = add_local_time(df).sort_values([HOUSEHOLD_COL, UTC_COL]).reset_index(drop=True)
    p = out["active_power_kw"]
    grp = out.groupby(HOUSEHOLD_COL, sort=False)["active_power_kw"]

    # --- past-only rolling statistics (P) -----------------------------------
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
    # Floor the per-slot MAD by a fraction of the household's global robust scale:
    # constant-load slots (deep night) have MAD = 0, which would otherwise make
    # p_resid_z explode.
    train_rows = out["ts_local"] < train_end
    global_scale = (
        out[train_rows]
        .groupby(HOUSEHOLD_COL)["active_power_kw"]
        .apply(lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))
    )
    floor = (0.1 * global_scale).clip(lower=1e-3)
    scale = np.maximum(1.4826 * out["how_mad"], out[HOUSEHOLD_COL].map(floor))
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


def add_rolling_current_voltage(out: pd.DataFrame) -> pd.DataFrame:
    """Add past-only rolling means of current and voltage (Step 6 channels)."""
    for col, tag in (("current_a", "i"), ("voltage_v", "v")):
        grp = out.groupby(HOUSEHOLD_COL, sort=False)[col]
        for name, w in WINDOWS.items():
            out[f"{tag}_roll_mean_{name}"] = grp.transform(lambda s, w=w: _past_roll(s, w, "mean"))
    return out


def add_physics_residuals(out: pd.DataFrame) -> pd.DataFrame:
    """Attach the relative electrical-identity residuals (recomputed)."""
    stale = [c for c in RESIDUAL_COLUMNS if c in out.columns]
    if stale:
        out = out.drop(columns=stale)
    return pd.concat([out, physics_residuals(out)], axis=1)


def add_bounds_flags(out: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Recompute the plausibility flags on the (injected) values."""
    from energymind.validate import bounds_flags

    stale = [c for c in out.columns if c.startswith("flag_")]
    if stale:
        out = out.drop(columns=stale)
    return pd.concat([out, bounds_flags(out, settings.data)], axis=1)


def add_flat_run(out: pd.DataFrame) -> pd.DataFrame:
    """Attach the active-power flat-run length."""
    out["flat_run_p"] = out.groupby(HOUSEHOLD_COL, sort=False)["active_power_kw"].transform(
        lambda s: flat_run_length(s.reset_index(drop=True)).to_numpy()
    )
    return out


# --------------------------------------------------------------------------- #
# Full Step 6 pipeline
# --------------------------------------------------------------------------- #


def train_end_of(settings: Settings) -> pd.Timestamp:
    """The exclusive end of the training window, as a timestamp."""
    return pd.Timestamp(settings.data.splits.temporal.train.end)


def build_features(
    df: pd.DataFrame,
    settings: Settings,
    *,
    nilm_model: nilm.NilmModel | None = None,
) -> pd.DataFrame:
    """Build the full Step 6 feature table for ``df`` (usually injected data).

    Adds the past-only core, current/voltage rolling, physics residuals, the
    flat-run length and, when a fitted ``nilm_model`` is given, the NILM
    estimates and residuals (``provenance = inferred``).
    """
    out = engineer_features(df, train_end_of(settings))
    out = add_rolling_current_voltage(out)
    out = add_physics_residuals(out)
    out = add_bounds_flags(out, settings)
    out = add_flat_run(out)
    if nilm_model is not None:
        out = nilm.add_nilm(out, nilm_model)
    return out


#: Columns the pipeline adds on top of the input frame (for the leakage test).
def added_columns(base: pd.DataFrame, built: pd.DataFrame) -> list[str]:
    return [c for c in built.columns if c not in base.columns]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def prepare(settings: Settings | None = None) -> dict:
    """Fit the NILM on the clean training split and build features for one run.

    Writes ``data/features/features.parquet`` (injected run) and
    ``data/features/nilm_predictions.parquet`` (clean hold-out evaluation).
    """
    settings = settings or load_settings()
    from energymind import clean, inject

    clean_df = clean.load_baseline(settings)
    engineered_clean = engineer_features(clean_df, train_end_of(settings))
    nilm_model, nilm_report = nilm.fit_and_evaluate(engineered_clean, settings)

    injected, labels = inject.inject_all(clean_df, settings, 0.05, 1)
    feature_frame = build_features(injected, settings, nilm_model=nilm_model)

    out_dir = io.resolve(settings.data.paths.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.parquet"
    feature_frame.to_parquet(features_path, index=False)

    predictions = engineered_clean[[HOUSEHOLD_COL, UTC_COL]].join(
        nilm.predict_nilm(nilm_model, engineered_clean)
    )
    pred_path = out_dir / "nilm_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)
    return {
        "features_path": features_path,
        "predictions_path": pred_path,
        "nilm": nilm_report,
        "rows": len(feature_frame),
        "columns": feature_frame.shape[1],
        "events": len(labels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 6: past-only features + NILM residual.")
    parser.add_argument("--configs-dir", default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    result = prepare(settings)

    print("Step 6 complete")
    print(f"  features rows     : {result['rows']}")
    print(f"  features columns  : {result['columns']}")
    print(f"  injected events   : {result['events']}")
    print(f"  NILM hold-out MAE : {result['nilm']['mae_total']:.4f} kWh/interval")
    for name, target in result["nilm"]["mae"].items():
        print(f"      {name}: {target:.4f}")
    print(f"  wrote             : {result['features_path']}")
    print(f"                      {result['predictions_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
