"""Tests for Step 6/7: past-only features, physics residuals and NILM residual."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energymind import features, nilm
from energymind.config import REPO_ROOT, load_settings
from energymind.validate import SUB_COLS

RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"
FAR_FUTURE = pd.Timestamp("2099-01-01")


def synthetic(n: int = 400, n_hh: int = 2) -> pd.DataFrame:
    """A small, identity-consistent frame with the columns the pipeline needs."""
    rows = []
    t0 = pd.Timestamp("2020-03-01 06:00", tz="UTC")
    for h in range(n_hh):
        hh = f"HH_{h:05d}_AUS"
        for i in range(n):
            p = 1.5 + 0.6 * np.sin(i / 5.0) + 0.1 * h
            v = 230.0 + 0.8 * np.sin(i / 7.0)
            q = p * 0.2
            s = float(np.sqrt(p**2 + q**2))
            rows.append(
                {
                    "household_id": hh,
                    "ts_utc": t0 + pd.Timedelta(minutes=30 * i),
                    "active_power_kw": p,
                    "reactive_power_kvar": q,
                    "apparent_power_kva": s,
                    "voltage_v": v,
                    "current_a": s * 1000.0 / v,
                    "power_factor": p / s,
                    "frequency_hz": 50.0,
                    "total_energy_kwh": 0.5 * p,
                    **dict.fromkeys(SUB_COLS, 0.125 * p),
                    "ev_load_kw": 0.0,
                    "solar_generation_kw": 0.3 if h == 0 else 0.0,
                    "has_solar": 1.0 if h == 0 else 0.0,
                    "temperature_c": 15.0 + 3.0 * np.sin(i / 50.0),
                    "occupancy_probability": 0.5 + 0.2 * np.sin(i / 20.0),
                    "quality_mask": 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


# --------------------------------------------------------------------------- #
# No leakage: past-only features do not react to the current value
# --------------------------------------------------------------------------- #


def test_past_only_features_do_not_leak():
    base = synthetic(n=400, n_hh=1)
    # the split point: rows >= 200 are outside the fitted training window.
    # train_end must be naive local time (as in configs/data.yaml: 2020-09-01).
    local = (base["ts_utc"] - pd.Timedelta(hours=6)).dt.tz_localize(None)
    train_end = local.iloc[200]
    fx = features.engineer_features(base, train_end)

    t = 300
    modified = base.copy()
    modified.loc[t, "active_power_kw"] *= 5.0
    fx2 = features.engineer_features(modified, train_end)

    numeric = [
        c for c in fx.columns if c not in base.columns and pd.api.types.is_numeric_dtype(fx[c])
    ]
    changed = {c for c in numeric if not np.isclose(fx[c].iloc[t], fx2[c].iloc[t], equal_nan=True)}
    assert changed, "the modified value must change at least the current-value features"
    assert changed <= set(features.CURRENT_VALUE), sorted(changed - set(features.CURRENT_VALUE))


def test_past_only_window_changes_only_in_the_future():
    base = synthetic(n=300, n_hh=1)
    fx = features.engineer_features(base, train_end=FAR_FUTURE)
    modified = base.copy()
    modified.loc[200, "active_power_kw"] *= 5.0
    fx2 = features.engineer_features(modified, train_end=FAR_FUTURE)
    # a past-only rolling mean at t uses rows < t, so it is unchanged at t...
    assert np.isclose(fx["p_roll_mean_2h"].iloc[200], fx2["p_roll_mean_2h"].iloc[200])
    # ...but the value propagates into the next interval's window.
    assert not np.isclose(fx["p_roll_mean_2h"].iloc[201], fx2["p_roll_mean_2h"].iloc[201])


# --------------------------------------------------------------------------- #
# Full Step 6 pipeline
# --------------------------------------------------------------------------- #


def test_build_features_adds_residuals_and_flat_run(settings):
    frame = features.build_features(synthetic(n=300, n_hh=2), settings)
    for column in ["r_S", "r_VI", "r_E", "r_PF", "r_sub", "flat_run_p"]:
        assert column in frame.columns
    for window in features.WINDOWS:
        assert f"i_roll_mean_{window}" in frame.columns
        assert f"v_roll_mean_{window}" in frame.columns


def test_physics_residuals_small_on_clean_frame(settings):
    frame = features.build_features(synthetic(n=200, n_hh=1), settings)
    assert frame[["r_S", "r_VI", "r_E", "r_PF"]].abs().max().max() < 1e-2


# --------------------------------------------------------------------------- #
# NILM
# --------------------------------------------------------------------------- #


def test_nilm_fit_predict_and_residual(settings):
    frame = synthetic(n=600, n_hh=2)
    engineered = features.engineer_features(frame, FAR_FUTURE)
    model = nilm.fit_nilm(engineered, settings)
    assert model.provenance == "inferred"
    assert set(model.targets) == set(SUB_COLS)

    out = nilm.add_nilm(engineered, model)
    for target in SUB_COLS:
        assert f"nilm_{target}" in out.columns
        assert f"nilm_resid_{target}" in out.columns
    assert "nilm_resid_total" in out.columns
    assert np.isfinite(out["nilm_resid_total"].dropna()).all()


def test_nilm_residual_detects_submeter_bias(settings):
    frame = synthetic(n=600, n_hh=1)
    engineered = features.engineer_features(frame, FAR_FUTURE)
    model = nilm.fit_nilm(engineered, settings)

    # a biased cooling sub-meter must show up in the residual
    before = nilm.add_nilm(engineered, model)
    biased = engineered.copy()
    biased.loc[300:400, "sub2_cooling_kwh"] *= 1.6
    after = nilm.add_nilm(biased, model)
    delta = (after["nilm_resid_sub2_cooling_kwh"] - before["nilm_resid_sub2_cooling_kwh"]).abs()
    assert delta.iloc[300:400].mean() > 0.01


# --------------------------------------------------------------------------- #
# Real dataset (skipped in CI)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_real_features_pipeline(settings):
    from energymind import clean, inject

    clean_df = clean.load_baseline(settings)
    injected, _ = inject.inject_all(clean_df, settings, 0.05, 1)
    frame = features.build_features(injected, settings)
    assert len(frame) == 13 * 17_520
    for column in ["r_S", "flat_run_p", "p_resid_z", "neigh_share"]:
        assert column in frame.columns
    assert frame["p_roll_mean_2h"].notna().sum() > 0
