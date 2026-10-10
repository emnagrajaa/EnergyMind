"""Tests for Step 5 (inject/): propagation, determinism, zone and scope."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energymind import clean
from energymind.config import REPO_ROOT, load_settings
from energymind.inject import adversarial as adv
from energymind.inject import data_quality as dq
from energymind.inject import inject_all
from energymind.inject.base import build_context, parse_duration
from energymind.validate import SUB_COLS, physics_residuals

RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"
IDENTITY = ["r_S", "r_VI", "r_E", "r_PF"]


def synthetic_baseline(n: int = 2000, n_hh: int = 2) -> pd.DataFrame:
    """A small, identity-consistent baseline on the real 30-minute grid."""
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
                    **dict.fromkeys(SUB_COLS, 0.125 * p),  # sum to E (balanced)
                    "ev_load_kw": 0.0,
                    "solar_generation_kw": 0.0,
                    "battery_charge_kw": 0.0,
                    "battery_discharge_kw": 0.0,
                    "quality_mask": 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _ctx(df, settings):
    return build_context(df, settings)


def _max_identity(df, pos) -> float:
    return float(physics_residuals(df).loc[pos, IDENTITY].abs().max().max())


# --------------------------------------------------------------------------- #
# Consistent variants keep identities
# --------------------------------------------------------------------------- #


def test_additive_bias_consistent_vs_naive(settings):
    pos = np.arange(10, 40)
    df = synthetic_baseline()
    ctx = _ctx(df, settings)
    dq.additive_bias(
        df, pos, {"k": 2.0}, "consistent", np.random.default_rng(0), ctx, "HH_00000_AUS"
    )
    assert _max_identity(df, pos) < 1e-2

    df2 = synthetic_baseline()
    ctx2 = _ctx(df2, settings)
    dq.additive_bias(df2, pos, {"k": 2.0}, "naive", np.random.default_rng(0), ctx2, "HH_00000_AUS")
    assert _max_identity(df2, pos) > 1e-2


def test_gain_error_consistent_vs_naive(settings):
    pos = np.arange(5, 25)
    df = synthetic_baseline()
    ctx = _ctx(df, settings)
    dq.gain_error(df, pos, {"g": 0.25}, "consistent", np.random.default_rng(0), ctx, "HH_00000_AUS")
    assert _max_identity(df, pos) < 1e-2

    df2 = synthetic_baseline()
    dq.gain_error(
        df2,
        pos,
        {"g": 0.25},
        "naive",
        np.random.default_rng(0),
        _ctx(df2, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df2, pos) > 1e-2


def test_voltage_drift_consistent_vs_naive(settings):
    pos = np.arange(0, 40)
    df = synthetic_baseline()
    dq.voltage_drift(
        df,
        pos,
        {"d_v": 8.0},
        "consistent",
        np.random.default_rng(0),
        _ctx(df, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df, pos) < 1e-2

    df2 = synthetic_baseline()
    dq.voltage_drift(
        df2,
        pos,
        {"d_v": 8.0},
        "naive",
        np.random.default_rng(0),
        _ctx(df2, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df2, pos) > 1e-2


def test_noise_burst_consistent_vs_naive(settings):
    pos = np.arange(10, 28)
    df = synthetic_baseline()
    dq.noise_burst(
        df,
        pos,
        {"alpha": 2.0},
        "consistent",
        np.random.default_rng(1),
        _ctx(df, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df, pos) < 1e-2

    df2 = synthetic_baseline()
    dq.noise_burst(
        df2,
        pos,
        {"alpha": 2.0},
        "naive",
        np.random.default_rng(1),
        _ctx(df2, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df2, pos) > 1e-2


def test_stuck_at_and_replay_are_consistent(settings):
    df = synthetic_baseline()
    pos = np.arange(20, 44)
    dq.stuck_at(
        df, pos, {}, "consistent", np.random.default_rng(0), _ctx(df, settings), "HH_00000_AUS"
    )
    assert _max_identity(df, pos) < 1e-2

    df2 = synthetic_baseline()
    src = np.arange(100, 124)  # a valid source window: 100 - 48 >= 0
    adv.replay(
        df2,
        src,
        {"lag": "1d"},
        "consistent",
        np.random.default_rng(0),
        _ctx(df2, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df2, src) < 1e-2


def test_mimicry_and_under_reporting_are_consistent(settings):
    df = synthetic_baseline()
    pos = np.arange(50, 80)
    adv.mimicry(
        df,
        pos,
        {"noise": 0.1},
        "consistent",
        np.random.default_rng(0),
        _ctx(df, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df, pos) < 1e-2

    df2 = synthetic_baseline()
    adv.under_reporting(
        df2,
        pos,
        {"r": 0.5},
        "consistent",
        np.random.default_rng(0),
        _ctx(df2, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df2, pos) < 1e-2


def test_sub_meter_bias_breaks_balance_not_identities(settings):
    df = synthetic_baseline()
    pos = np.arange(5, 15)
    dq.sub_meter_bias(
        df,
        pos,
        {"g": 0.6},
        "consistent",
        np.random.default_rng(0),
        _ctx(df, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df, pos) < 1e-2  # electrical identities still hold
    assert physics_residuals(df).loc[pos, "r_sub"].abs().max() > 0.1  # balance broken


def test_dropout_blanks_channels(settings):
    df = synthetic_baseline()
    pos = np.arange(3, 8)
    dq.dropout(
        df, pos, {}, "consistent", np.random.default_rng(0), _ctx(df, settings), "HH_00000_AUS"
    )
    assert df.loc[pos, "active_power_kw"].isna().all()
    assert df.loc[pos, "voltage_v"].isna().all()


def test_unit_error_and_dead_sensor_break_identities(settings):
    pos = np.arange(4, 10)
    df = synthetic_baseline()
    dq.unit_error(
        df,
        pos,
        {"factor": 1000.0},
        "naive",
        np.random.default_rng(0),
        _ctx(df, settings),
        "HH_00000_AUS",
    )
    assert _max_identity(df, pos) > 1e-2

    df2 = synthetic_baseline()
    dq.dead_sensor(
        df2, pos, {}, "naive", np.random.default_rng(0), _ctx(df2, settings), "HH_00000_AUS"
    )
    assert _max_identity(df2, pos) > 1e-2


# --------------------------------------------------------------------------- #
# inject_all: determinism, zone, non-overlap, scope
# --------------------------------------------------------------------------- #

SYNTH_TYPES = [
    "additive_bias",
    "gain_error",
    "stuck_at",
    "dropout",
    "dead_sensor",
    "noise_burst",
    "unit_error",
    "under_reporting",
    "sub_meter_bias",
    "coherent_peak",
    "neighborhood_outage",
    "voltage_sag",
]


def test_inject_all_is_deterministic(settings):
    df = synthetic_baseline()
    a, la = inject_all(df, settings, 0.02, 7, types=SYNTH_TYPES)
    b, lb = inject_all(df, settings, 0.02, 7, types=SYNTH_TYPES)
    assert la.equals(lb)
    assert np.allclose(
        a["active_power_kw"].to_numpy(), b["active_power_kw"].to_numpy(), equal_nan=True
    )


def test_inject_all_zone_and_non_overlap(settings):
    df = synthetic_baseline(n_hh=3)
    _, labels = inject_all(df, settings, 0.03, 3, types=SYNTH_TYPES)
    assert len(labels) > 0
    # every event sits on quality rows (synthetic baseline is all-quality)
    assert (labels["t_end"] > labels["t_start"]).all()
    # no overlap within a household
    for _, group in labels.groupby("household_id"):
        spans = sorted(zip(group["t_start"], group["t_end"], strict=True))
        for earlier, later in zip(spans, spans[1:], strict=False):
            assert later[0] >= earlier[1]


def test_physical_events_span_households(settings):
    df = synthetic_baseline(n_hh=3)
    _, labels = inject_all(
        df, settings, 0.05, 1, types=["coherent_peak", "neighborhood_outage", "voltage_sag"]
    )
    physical = labels[labels["origin"] == "physical_event"]
    assert len(physical) > 0
    assert (physical["scope"] == "neighborhood").all()
    assert physical["household_id"].str.contains(",").all()


def test_parse_duration():
    assert parse_duration("30min") == 1
    assert parse_duration("12h") == 24
    assert parse_duration("7d") == 336


# --------------------------------------------------------------------------- #
# Real dataset (skipped in CI)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_real_injection_guarantees(settings):
    df = clean.load_baseline(settings)
    injected, labels = inject_all(df, settings, 0.05, 1)
    residuals = physics_residuals(injected)[IDENTITY].abs().max(axis=1)

    # Consistent events keep the identities; single-channel "naive" edits break them.
    break_types = {
        "additive_bias",
        "gain_error",
        "noise_burst",
        "dead_sensor",
        "unit_error",
        "voltage_drift",
    }
    broke = 0
    for _, event in labels.iterrows():
        mask = (
            (injected["household_id"] == event["household_id"])
            & (injected["ts_utc"] >= event["t_start"])
            & (injected["ts_utc"] < event["t_end"])
        ).to_numpy()
        worst = float(residuals[mask].max())
        blank = bool(injected.loc[mask, "active_power_kw"].isna().all())
        if event["variant"] == "consistent":
            assert blank or worst < 1e-2, (event["type"], worst)
        elif event["type"] in break_types:
            assert blank or worst > 1e-2, (event["type"], worst)
            broke += 1
    assert broke > 0  # at least one naive single-channel edit was produced

    # replay and mimicry are withheld from train/val: only in test.
    withheld = labels[labels["type"].isin(["replay", "mimicry"])]
    if len(withheld):
        assert (withheld["t_start"] >= pd.Timestamp("2020-11-01 06:00", tz="UTC")).all()

    # events never touch provided anomalies, missing records or imputed rows
    for _, event in labels.iterrows():
        mask = (
            (injected["household_id"] == event["household_id"])
            & (injected["ts_utc"] >= event["t_start"])
            & (injected["ts_utc"] < event["t_end"])
        ).to_numpy()
        assert (injected.loc[mask, "quality_mask"] == 1).all()
