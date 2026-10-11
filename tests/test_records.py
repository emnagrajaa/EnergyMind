"""Tests for Step 9: signatures, records and template text."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energymind import records


def window(n: int = 20) -> pd.DataFrame:
    t0 = pd.Timestamp("2020-11-15 00:00", tz="UTC")
    p = 4.0 + 0.5 * np.sin(np.arange(n))
    return pd.DataFrame(
        {
            "ts_utc": [t0 + pd.Timedelta(minutes=30 * i) for i in range(n)],
            "active_power_kw": p,
            "voltage_v": 230.0,
            "current_a": p * 1000.0 / 230.0,
            "how_median": 2.5,
            "r_S": 0.0,
            "r_VI": 0.0,
            "r_E": 0.0,
            "r_PF": 0.0,
            "neigh_share": 0.08,
            "nilm_resid_sub1_heating_kwh": 0.01,
            "nilm_resid_sub2_cooling_kwh": 0.31,
            "nilm_resid_sub3_lighting_kwh": 0.0,
            "nilm_resid_sub4_appliance_kwh": 0.02,
            "nilm_resid_total": 0.34,
        }
    )


def test_signature_is_64_dim_and_finite():
    w = window()
    sig = records.build_signature(w, w["ts_utc"].iloc[0])
    assert sig.shape == (records.SIGNATURE_DIM,)
    assert np.isfinite(sig).all()


def test_cause_hypotheses_uses_evidence():
    evidence = [
        {"check": "current_vs_power_identity", "rel_residual": 0.16, "status": "broken"},
        {"check": "neighborhood_share", "value": 0.08, "status": "isolated"},
        {"check": "baseline_deviation", "observed_kw": 5.1, "expected_kw": 3.9, "z": 3.4},
    ]
    hypotheses = records.cause_hypotheses("data_quality", evidence)
    assert any("identity is broken" in h for h in hypotheses)


def test_narrative_dropout_and_corrupt_sentences():
    dropout = records.narrative(
        {
            "zero_reading": True,
            "zero_frac": 1.0,
            "expected_kw": 3.6,
            "share": 0.25,
            "coherent": False,
        },
        "data_quality",
        "dropout",
    )
    text = " ".join(dropout)
    assert "exactly zero" in text and "sensor outage" in text

    corrupt = records.narrative(
        {
            "observed_kw": 6.37,
            "expected_kw": 3.0,
            "neighbor_median_kw": 2.0,
            "local_before_kw": 1.9,
            "coherent": False,
            "identity_broken": False,
        },
        "data_quality",
        "unit_error",
    )
    text2 = " ".join(corrupt)
    assert "inconsistent with neighbouring intervals" in text2 and "falsified" in text2


def test_narrative_varies_with_the_data():
    cold = " ".join(
        records.narrative(
            {"observed_kw": 0.5, "expected_kw": 3.0, "z": -6.0, "share": 0.0, "coherent": False},
            "data_quality",
            "additive_bias",
        )
    )
    event = " ".join(
        records.narrative(
            {"observed_kw": 5.0, "expected_kw": 3.0, "z": 4.0, "share": 0.8, "coherent": True},
            "physical_event",
            "coherent_peak",
        )
    )
    assert cold != event
    assert "genuine network event" in event


def test_build_record_and_render_text():
    w = window()
    sig = records.build_signature(w, w["ts_utc"].iloc[0])
    evidence = [
        {"check": "baseline_deviation", "observed_kw": 5.1, "expected_kw": 3.9, "z": 3.4},
        {"check": "current_vs_power_identity", "rel_residual": 0.0, "status": "holds"},
        {"check": "neighborhood_share", "value": 0.08, "status": "isolated"},
    ]
    record = records.build_record(
        anomaly_id="uuid",
        household_id="HH_00072_AUS",
        t_start=w["ts_utc"].iloc[0],
        t_end=w["ts_utc"].iloc[-1],
        detection={"model": "rules", "version": "0.1.0", "score": 12.3, "p_anomaly": 0.91},
        classification={
            "model": "lightgbm",
            "version": "0.1.0",
            "type_probs": {"gain_error": 0.64},
            "origin_probs": {"data_quality": 0.81},
            "type_pred": "gain_error",
            "origin_pred": "data_quality",
        },
        evidence=evidence,
        explanation={
            "shap_top": [["p_resid_z", 0.4]],
            "lime_top": ["p_resid_z"],
            "reliability": 0.6,
        },
        signature=sig,
    )
    assert record["signature"] is not None
    assert len(record["signature"]) == records.SIGNATURE_DIM
    # the template text only quotes the evidence numbers
    assert "5.1" in record["text"] and "3.9" in record["text"]
    assert "8%" in record["text"]  # neighbourhood share
    assert record["cause_hypotheses"]
