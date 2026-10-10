"""Tests for Step 2 physics: residuals, dependent recomputation, bounds, flat runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energymind.config import load_settings
from energymind.validate import (
    SUB_COLS,
    bounds_flags,
    flat_run_length,
    physics_residuals,
    recompute_dependents,
)


def _consistent(n: int = 4) -> pd.DataFrame:
    p = np.linspace(1.0, 4.0, n)
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


def test_residuals_are_zero_when_consistent():
    residuals = physics_residuals(_consistent())
    for column in ["r_S", "r_VI", "r_E", "r_PF"]:
        assert residuals[column].abs().max() < 1e-9


def test_residuals_detect_a_single_channel_edit():
    df = _consistent()
    df["active_power_kw"] = df["active_power_kw"] * 1.1  # naive edit
    residuals = physics_residuals(df)
    assert residuals["r_S"].abs().max() > 0.01
    assert residuals["r_E"].abs().max() > 0.01


def test_recompute_dependents_restores_identities():
    df = _consistent(2)
    recompute_dependents(df, [0], p=10.0, q=2.0, v=230.0)
    s = np.sqrt(10.0**2 + 2.0**2)
    assert df.loc[0, "apparent_power_kva"] == pytest.approx(s)
    assert df.loc[0, "current_a"] == pytest.approx(s * 1000.0 / 230.0)
    assert df.loc[0, "power_factor"] == pytest.approx(10.0 / s)
    assert df.loc[0, "total_energy_kwh"] == pytest.approx(5.0)
    # restored row is consistent again
    assert physics_residuals(df).loc[0, "r_S"] == pytest.approx(0.0, abs=1e-9)


def test_bounds_flags_impossible_values():
    cfg = load_settings().data
    df = _consistent(3)
    df.loc[0, "voltage_v"] = 200.0
    df.loc[1, "frequency_hz"] = 60.0
    df.loc[2, "active_power_kw"] = -1.0
    flags = bounds_flags(df, cfg)
    assert bool(flags.loc[0, "flag_voltage"])
    assert bool(flags.loc[1, "flag_frequency"])
    assert bool(flags.loc[2, "flag_active_power_negative"])
    assert not bool(flags.loc[0, "flag_frequency"])


def test_submeter_over_total_flag():
    cfg = load_settings().data
    df = _consistent(1)
    for column in SUB_COLS:
        df.loc[0, column] = 1.0  # 4.0 kWh vs E = 0.5 kWh
    df.loc[0, "ev_load_kw"] = 0.0
    assert bool(bounds_flags(df, cfg).loc[0, "flag_submeter_over_total"])


def test_flat_run_length():
    series = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0, 1.0])
    assert flat_run_length(series).tolist() == [1, 2, 3, 1, 2, 1]
