"""Tests for Step 2 cleaning: grid, masking, imputation and quality mask."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energymind import clean
from energymind.config import REPO_ROOT, load_settings
from energymind.validate import SUB_COLS, physics_residuals

RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"
HOUSEHOLDS = ["HH_A", "HH_B"]


def _measured(n: int = 5) -> pd.DataFrame:
    """A physically consistent measured frame (identities hold exactly)."""
    p = np.linspace(1.0, 5.0, n)
    q = p * 0.2
    s = np.sqrt(p**2 + q**2)
    v = np.full(n, 230.0)
    return pd.DataFrame(
        {
            "active_power_kw": p,
            "reactive_power_kvar": q,
            "apparent_power_kva": s,
            "voltage_v": v,
            "current_a": s * 1000.0 / v,
            "power_factor": p / s,
            "frequency_hz": np.full(n, 50.0),
            "total_energy_kwh": 0.5 * p,
            **{c: np.full(n, 0.1) for c in SUB_COLS},
            "ev_load_kw": np.zeros(n),
        }
    )


# --------------------------------------------------------------------------- #
# to_grid
# --------------------------------------------------------------------------- #


def test_to_grid_flags_missing_records():
    cfg = load_settings().data
    idx = pd.date_range("2020-01-01 06:00", periods=6, freq="30min", tz="UTC")
    rows = [
        {"household_id": hh, "ts_utc": t, "active_power_kw": 1.0} for hh in HOUSEHOLDS for t in idx
    ]
    df = pd.DataFrame(rows)
    hole = idx[2]
    df = df[~((df["household_id"] == "HH_A") & (df["ts_utc"] == hole))]

    grid = clean.to_grid(df, cfg)

    assert len(grid) == len(HOUSEHOLDS) * len(idx)
    assert grid.groupby("household_id").size().eq(len(idx)).all()
    a = grid[grid["household_id"] == "HH_A"]
    assert int(a["is_missing_record"].sum()) == 1
    assert bool(a.loc[a["ts_utc"] == hole, "is_missing_record"].iloc[0])


# --------------------------------------------------------------------------- #
# mask_provided_labels
# --------------------------------------------------------------------------- #


def test_mask_provided_labels_blanks_electrical_channels():
    df = _measured(4)
    df.insert(0, "household_id", "HH_A")
    df.insert(1, "ts_utc", pd.date_range("2020-01-01 06:00", periods=4, freq="30min", tz="UTC"))
    labels = pd.DataFrame(
        {"household_id": ["HH_A"], "ts_utc": [df["ts_utc"].iloc[1]], "anomaly_type": ["spike"]}
    )

    masked = clean.mask_provided_labels(df, labels)

    assert masked["provided_anomaly"].tolist() == [0, 1, 0, 0]
    assert np.isnan(masked.loc[1, "active_power_kw"])
    assert np.isnan(masked.loc[1, "voltage_v"])
    assert not np.isnan(masked.loc[0, "active_power_kw"])


# --------------------------------------------------------------------------- #
# impute_single_slots
# --------------------------------------------------------------------------- #


def _marked_frame(n: int = 5) -> pd.DataFrame:
    df = _measured(n)
    df.insert(0, "household_id", "HH_A")
    df.insert(1, "ts_utc", pd.date_range("2020-01-01 06:00", periods=n, freq="30min", tz="UTC"))
    df["is_missing_record"] = False
    df["provided_anomaly"] = np.int8(0)
    return df


def test_impute_fills_isolated_single_slot():
    df = _marked_frame(5)
    df.loc[2, clean.INTERP_COLUMNS] = np.nan
    df.loc[2, "is_missing_record"] = True

    out = clean.impute_single_slots(df)

    assert out.loc[2, "is_imputed"] == 1
    # linear interpolation of P between neighbours (P = 1..5 -> 2 and 4)
    assert out.loc[2, "active_power_kw"] == pytest.approx(3.0)
    # dependent channels rebuilt, identities hold on the filled row
    residuals = physics_residuals(out).loc[2]
    assert residuals[["r_S", "r_VI", "r_E", "r_PF"]].abs().max() < 1e-9


def test_impute_leaves_multi_slot_gaps_nan():
    df = _marked_frame(5)
    df.loc[[1, 2], clean.INTERP_COLUMNS] = np.nan
    df.loc[[1, 2], "is_missing_record"] = True

    out = clean.impute_single_slots(df)

    assert out.loc[1, "is_imputed"] == 0
    assert out.loc[2, "is_imputed"] == 0
    assert np.isnan(out.loc[1, "active_power_kw"])
    assert np.isnan(out.loc[2, "active_power_kw"])


# --------------------------------------------------------------------------- #
# quality_mask
# --------------------------------------------------------------------------- #


def test_quality_mask_excludes_flags_imputation_and_labels():
    cfg = load_settings().data
    df = _measured(4)
    df = pd.concat([df, physics_residuals(df)], axis=1)
    df["flag_voltage"] = False
    df["flag_frequency"] = False
    df["provided_anomaly"] = np.int8(0)
    df["is_imputed"] = np.int8(0)
    df["is_missing_record"] = False

    assert clean.quality_mask(df, cfg).all()

    df.loc[0, "flag_voltage"] = True  # impossible value
    df.loc[1, "provided_anomaly"] = np.int8(1)  # provided anomaly
    df.loc[2, "is_imputed"] = np.int8(1)  # imputed slot
    df.loc[3, "r_S"] = 0.5  # identity broken
    mask = clean.quality_mask(df, cfg)
    assert mask.tolist() == [False, False, False, False]


def test_quality_mask_excludes_missing_records():
    cfg = load_settings().data
    df = _measured(3)
    df = pd.concat([df, physics_residuals(df)], axis=1)
    for column in ["flag_voltage", "flag_frequency"]:
        df[column] = False
    df["provided_anomaly"] = np.int8(0)
    df["is_imputed"] = np.int8(0)
    df["is_missing_record"] = [False, True, False]
    assert clean.quality_mask(df, cfg).tolist() == [True, False, True]


# --------------------------------------------------------------------------- #
# Real dataset (skipped in CI)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_real_baseline():
    settings = load_settings()
    df = clean.build_baseline(settings)

    assert len(df) == 13 * 17_520  # full grid: raw 227,752 + 8 missing records
    assert df.groupby("household_id").size().eq(17_520).all()
    assert int(df["is_missing_record"].sum()) == 8
    assert int(df["provided_anomaly"].sum()) == 698

    quality = df[df["quality_mask"] == 1]
    assert len(quality) > 0
    assert quality[clean.IDENTITY_RESIDUALS].abs().max().max() < 1e-2
