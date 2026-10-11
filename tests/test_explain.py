"""Tests for Step 9 explanations: SHAP, LIME reliability, attribution, evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energymind import explain


def window(n: int = 30) -> pd.DataFrame:
    t0 = pd.Timestamp("2020-11-15 00:00", tz="UTC")
    p = 4.0 + 0.5 * np.sin(np.arange(n))
    return pd.DataFrame(
        {
            "ts_utc": [t0 + pd.Timedelta(minutes=30 * i) for i in range(n)],
            "active_power_kw": p,
            "voltage_v": 230.0,
            "current_a": p * 1000.0 / 230.0,
            "how_median": 2.5,
            "p_resid_z": 6.0,
            "r_S": 0.0,
            "r_VI": 0.0,
            "r_E": 0.0,
            "r_PF": 0.0,
            "neigh_share": 0.5,
            "nilm_resid_total": 0.3,
            "sub2_share": 0.4,
        }
    )


# --------------------------------------------------------------------------- #
# reliability
# --------------------------------------------------------------------------- #


def test_mean_jaccard_identical_and_disjoint():
    assert explain.mean_jaccard(["a", "b", "c"], [["a", "b", "c"], ["a", "b", "c"]]) == 1.0
    assert explain.mean_jaccard(["a", "b", "c"], [["x", "y", "z"]]) == 0.0
    # partial overlap: |{a}| / |{a,b,c,d}| = 0.25
    assert explain.mean_jaccard(["a", "b", "c"], [["a", "d"]]) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# reconstruction attribution
# --------------------------------------------------------------------------- #


def test_reconstruction_attribution_ranks_the_worst_channel_and_time():
    channels = ["P", "V", "I"]
    window_array = np.zeros((10, 3))
    recon = np.zeros((10, 3))
    recon[4, 1] = 3.0  # channel V, time 4 is wrong
    attribution = explain.reconstruction_attribution(window_array, recon, channels)
    assert attribution["top_channel"] == "V"
    assert attribution["top_time_index"] == 4
    assert list(attribution["per_channel"])[0] == "V"


# --------------------------------------------------------------------------- #
# evidence checks
# --------------------------------------------------------------------------- #


def test_evidence_checks_flag_broken_identity_and_coherence():
    w = window()
    w.loc[5:10, "r_S"] = 0.2  # break an identity
    checks = {c["check"]: c for c in explain.evidence_checks(w)}
    assert checks["current_vs_power_identity"]["status"] == "broken"
    assert checks["neighborhood_share"]["status"] == "coherent"
    assert checks["baseline_deviation"]["z"] == 6.0


def test_evidence_checks_hold_when_consistent():
    checks = {c["check"]: c for c in explain.evidence_checks(window())}
    assert checks["current_vs_power_identity"]["status"] == "holds"


def test_sanity_check_sub_meter_feature_in_top_k():
    explanation = {
        "shap_top": [("sub2_share_mean", 0.4), ("p_resid_z_max", 0.2), ("duration_h", 0.1)]
    }
    assert explain.sanity_check(explanation, ["sub1_", "sub2_", "sub3_", "sub4_"], k=3)
    assert not explain.sanity_check(explanation, ["nilm_resid_total"], k=3)


# --------------------------------------------------------------------------- #
# TreeSHAP (needs shap)
# --------------------------------------------------------------------------- #


def test_tree_shap_top_returns_k_signed_features():
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    y = (x["a"] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(x, y)
    top = explain.tree_shap_top(model, x.iloc[0], list(x.columns), class_index=1, k=3)
    assert len(top) == 3
    assert all(isinstance(name, str) and isinstance(value, float) for name, value in top)
    # the informative feature must appear in the top 3
    assert "a" in [name for name, _ in top]
