"""Tests for Step 8: event-level evaluation, protocol fixes and the detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energymind import evaluate
from energymind.detect.base import WINDOW_FEATURES
from energymind.detect.matrix_profile import MatrixProfileDetector
from energymind.detect.pyod_models import ECODDetector
from energymind.detect.robust_z import RobustZ
from energymind.detect.rules import Rules
from energymind.detect.supervised import SupervisedDetector

INTERVAL = pd.Timedelta(minutes=30)
FEATURE_COLUMNS = [
    *WINDOW_FEATURES,
    "r_S",
    "r_VI",
    "r_E",
    "r_PF",
    "r_sub",
    "nilm_resid_total",
]


def synthetic(n: int = 1200, household: str = "HH_00000_AUS") -> pd.DataFrame:
    t0 = pd.Timestamp("2020-03-01 06:00", tz="UTC")
    p = 2.0 + 0.5 * np.sin(np.arange(n) / 5.0)
    frame = pd.DataFrame(
        {
            "household_id": household,
            "ts_utc": [t0 + INTERVAL * i for i in range(n)],
            "active_power_kw": p,
        }
    )
    for column in FEATURE_COLUMNS:
        frame[column] = 0.0
    frame["p_resid_z"] = 0.5
    return frame


def labels_for(frame: pd.DataFrame, start_i: int, end_i: int, **extra) -> pd.DataFrame:
    row = {
        "anomaly_id": "a1",
        "household_id": frame["household_id"].iloc[0],
        "t_start": frame["ts_utc"].iloc[start_i],
        "t_end": frame["ts_utc"].iloc[end_i],
        "type": "additive_bias",
        "origin": "data_quality",
        "severity": "medium",
        "variant": "consistent",
        "scope": "single",
    }
    row.update(extra)
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# Event handling and protocol
# --------------------------------------------------------------------------- #


def test_expand_events_splits_neighbourhood():
    labels = pd.DataFrame(
        [
            {
                "anomaly_id": "n1",
                "household_id": "A,B,C",
                "t_start": pd.Timestamp("2020-03-01 06:00", tz="UTC"),
                "t_end": pd.Timestamp("2020-03-01 07:00", tz="UTC"),
                "type": "coherent_peak",
                "origin": "physical_event",
                "severity": "high",
                "variant": "consistent",
                "scope": "neighborhood",
            }
        ]
    )
    events = evaluate.expand_events(labels)
    assert len(events) == 3
    assert events["anomaly_id"].nunique() == 1


def test_point_labels_marks_window_plus_tolerance():
    frame = synthetic(50)
    events = evaluate.expand_events(labels_for(frame, 10, 15))
    flag = evaluate.point_labels(frame, events, tol=INTERVAL)
    assert flag.iloc[9] and flag.iloc[10] and flag.iloc[14] and flag.iloc[15] and flag.iloc[16]
    assert not flag.iloc[8] and not flag.iloc[17]


def test_sweep_threshold_respects_budget_and_oracle_is_upper_bound():
    frame = synthetic(1000)
    events = evaluate.expand_events(labels_for(frame, 400, 430))
    scores = pd.Series(0.1, index=frame.index)
    scores.iloc[400:430] = 5.0  # clear anomaly
    budget = 1.0

    constrained = evaluate.sweep_thresholds(frame, scores, events, INTERVAL, budget=budget)
    oracle = evaluate.sweep_thresholds(frame, scores, events, INTERVAL, budget=None)

    assert constrained["recall"] == pytest.approx(1.0)
    assert oracle["f1"] >= constrained["f1"] - 1e-9
    allowed = evaluate._budget_allowed(frame, budget)
    assert constrained["fa"] <= allowed


def test_oracle_threshold_detects_when_budget_would_not():
    frame = synthetic(500)
    events = evaluate.expand_events(labels_for(frame, 250, 260))
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.uniform(0.0, 1.0, size=len(frame)), index=frame.index)
    scores.iloc[250:260] = 0.5  # a modest signal buried in noise
    # a very tight budget forces a threshold above the signal -> no detection
    strict = evaluate.select_threshold(frame, scores, events, 0.001, 1)
    assert strict > 0.5
    # oracle still finds the operating point
    oracle = evaluate.oracle_threshold(frame, scores, events, 1)
    m, _ = evaluate.evaluate(frame, scores, events, oracle, 1.0, 1)
    assert m["event_recall"] == pytest.approx(1.0)


def test_expected_calibration_error_perfect_is_zero():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert evaluate.expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #


def test_robust_z_and_rules_score_raise_on_anomaly():
    frame = synthetic(200)
    frame.loc[100:110, "p_resid_z"] = 25.0
    frame.loc[150:155, "r_S"] = 0.5  # physics violation
    z = RobustZ().fit(frame).score(frame)
    rules = Rules().fit(frame).score(frame)
    assert z.iloc[100:111].mean() > z.iloc[:50].mean()
    assert rules.iloc[150:156].mean() > rules.iloc[:50].mean()


def test_ecod_scores_are_finite_and_aligned():
    frame = synthetic(300)
    frame.loc[100:120, "p_resid_z"] = 20.0
    scores = ECODDetector().fit(frame).score(frame)
    assert len(scores) == len(frame)
    assert np.isfinite(scores.to_numpy()).all()
    assert scores.iloc[100:121].mean() > scores.iloc[:50].mean()


def test_matrix_profile_flags_a_replayed_block():
    frame = synthetic(1000)
    lag = 7 * 48
    p = frame["active_power_kw"].to_numpy().copy()
    p[800:900] = p[800 - lag : 900 - lag]  # exact 7-day replay
    frame["active_power_kw"] = p
    scores = MatrixProfileDetector().fit(frame).score(frame)
    assert scores.iloc[800:900].mean() > scores.iloc[300:400].mean()


def test_supervised_detector_ranks_events():
    pytest.importorskip("lightgbm")
    frame = synthetic(600)
    frame.loc[400:430, "p_resid_z"] = 30.0
    events = evaluate.expand_events(labels_for(frame, 400, 430))
    detector = SupervisedDetector(events=events).fit(frame)
    scores = detector.score(frame)
    assert scores.iloc[400:431].mean() > scores.iloc[100:200].mean()
    assert scores.notna().all()
