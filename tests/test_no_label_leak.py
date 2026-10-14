"""Fix 1: the decision record must never use injected labels as predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energymind.agent_pipeline import _build_record


def _frame(n: int = 6) -> pd.DataFrame:
    t0 = pd.Timestamp("2020-06-01 12:00", tz="UTC")
    rows = []
    for i in range(n):
        p = 2.0 + 0.3 * i
        v = 230.0
        rows.append(
            {
                "household_id": "HH_A",
                "ts_utc": t0 + pd.Timedelta(minutes=30 * i),
                "active_power_kw": p,
                "voltage_v": v,
                "current_a": p * 1000.0 / v,
                "reactive_power_kvar": p * 0.2,
                "apparent_power_kva": p * 1.02,
                "power_factor": 0.98,
                "total_energy_kwh": 0.5 * p,
                "sub1_heating_kwh": 0.1,
                "sub2_cooling_kwh": 0.1,
                "sub3_lighting_kwh": 0.1,
                "sub4_appliance_kwh": 0.1,
                "ev_load_kw": 0.0,
                "how_median": 2.2,
                "p_resid_z": 1.5 + i,
                "r_S": 0.0,
                "r_VI": 0.0,
                "r_E": 0.0,
                "r_PF": 0.0,
                "neigh_share": 0.1,
                "flat_run_p": 1,
                "sub2_share": 0.15,
                "nilm_resid_sub1_heating_kwh": 0.01,
                "nilm_resid_sub2_cooling_kwh": 0.02,
                "nilm_resid_sub3_lighting_kwh": 0.0,
                "nilm_resid_sub4_appliance_kwh": 0.01,
                "nilm_resid_total": 0.04,
            }
        )
    return pd.DataFrame(rows)


def _events(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "anomaly_id": "evt-1",
                "household_id": "HH_A",
                "t_start": frame["ts_utc"].iloc[0],
                "t_end": frame["ts_utc"].iloc[-1] + pd.Timedelta(minutes=30),
                # ground-truth label (physical_event) - must NOT become the prediction
                "type": "coherent_peak",
                "origin": "physical_event",
                "severity": "high",
                "variant": "consistent",
                "scope": "single",
            }
        ]
    )


def _walk(obj):
    """Yield every scalar/string reachable in the record."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from _walk(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk(item)
    else:
        yield str(obj)


def test_prediction_comes_from_classifier_not_label():
    frame = _frame()
    events = _events(frame)
    detector_scores = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
    classifier = {
        "type_probs": {"additive_bias": 0.8, "coherent_peak": 0.2},
        "origin_probs": {"data_quality": 0.9, "physical_event": 0.1},
    }

    record = _build_record(frame, events, "evt-1", detector_scores, classification=classifier)

    # the label is physical_event, but the classifier says data_quality
    assert record["origin_pred"] == "data_quality"
    assert record["type_pred"] == "additive_bias"
    assert record["classification"]["model"] == "lightgbm"
    assert record["classification"]["origin_probs"] == classifier["origin_probs"]


def test_record_contains_no_label_leakage_markers():
    frame = _frame()
    events = _events(frame)
    detector_scores = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
    record = _build_record(
        frame,
        events,
        "evt-1",
        detector_scores,
        classification={"type_probs": {"x": 0.6}, "origin_probs": {"data_quality": 0.6}},
    )

    tokens = list(_walk(record))
    assert not any("labels(sim)" in token for token in tokens)
    assert not any(token.startswith("true_") for token in tokens)


def test_no_classifier_gives_null_predictions_not_labels():
    frame = _frame()
    events = _events(frame)
    detector_scores = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)

    record = _build_record(frame, events, "evt-1", detector_scores, classification=None)

    assert record["type_pred"] is None
    assert record["origin_pred"] is None
    assert record["classification"] == {}
    # severity is recomputed from the window, not taken from the label
    assert record["severity"] in {"low", "medium", "high"}
