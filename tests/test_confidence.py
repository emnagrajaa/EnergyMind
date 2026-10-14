"""Tests for M8 confidence: features, calibration, logistic f and ECE."""

from __future__ import annotations

import numpy as np
import pytest

from energymind.confidence import (
    FEATURE_NAMES,
    AccuracyCalibrator,
    ConfidenceModel,
    Signal,
    build_matrix,
    convergence_L,
    expected_calibration_error,
    history_H,
    reliability_curve,
    risk_coverage_summary,
)


def test_convergence_L_definition():
    assert convergence_L(True, 1, 3) == 1.0
    assert convergence_L(True, 2, 3) == pytest.approx(2 / 3)
    assert convergence_L(True, 3, 3) == pytest.approx(1 / 3)
    assert convergence_L(False, 3, 3) == 0.0


def test_history_H_shrinks_toward_prior():
    # one agreeing verified neighbour at similarity 1
    assert history_H([(1.0, "data_quality")], "data_quality", prior=0.5, m=2.0) > 0.5
    # no history -> prior
    assert history_H([], "data_quality", prior=0.4, m=2.0) == 0.4
    # disagreeing neighbour pulls below prior
    assert history_H([(1.0, "adversarial")], "data_quality", prior=0.5, m=2.0) < 0.5


def test_accuracy_calibrator_is_monotone():
    conf = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    correct = np.array([0, 0, 0, 1, 1, 1])
    cal = AccuracyCalibrator().fit(conf, correct)
    low = cal.transform(0.15)
    high = cal.transform(0.85)
    assert 0.0 <= low <= high <= 1.0


def test_confidence_model_fits_and_predicts_in_unit_interval():
    rng = np.random.default_rng(0)
    signals = []
    y = []
    for _ in range(120):
        converged = bool(rng.random() < 0.5)
        model_conf = float(rng.uniform(0.3, 0.99))
        correct = int((model_conf > 0.6) == (rng.random() < 0.85))
        signals.append(
            Signal(
                model_conf=model_conf,
                converged=converged,
                rounds_used=1 if converged else 3,
                r_max=3,
                proposed_origin="data_quality",
                histories=[(0.95, "data_quality")] if correct else [(0.95, "adversarial")],
                checks=[{"check": "current_vs_power_identity", "status": "holds"}],
                completeness=0.98,
                seen_before=False,
            )
        )
        y.append(correct)

    model = ConfidenceModel().fit(signals, np.array(y, dtype=float))
    f = model.predict(signals)
    assert f.shape == (len(signals),)
    assert np.all((f >= 0.0) & (f <= 1.0))
    # a high-A, high-H, converged case should score above a low-confidence opponent
    high = Signal(0.95, True, 1, 3, "data_quality", [(0.98, "data_quality")], [], 1.0, False)
    low = Signal(0.3, False, 3, 3, "adversarial", [], [], 0.5, False)
    fh = model.predict([high])[0]
    fl = model.predict([low])[0]
    assert fh > fl


def test_build_matrix_shape_and_names():
    signal = Signal(0.7, True, 1, 3, "data_quality", [], [], 1.0, False)
    matrix = build_matrix([signal], AccuracyCalibrator())
    assert matrix.shape == (1, len(FEATURE_NAMES))


def test_ece_zero_for_perfect_calibration():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y, p) == 0.0
    centers, accs = reliability_curve(y, p, n_bins=10)
    assert len(centers) == len(accs) > 0


def test_risk_coverage_is_better_for_f_than_overconfident_model():
    # f ranks correct cases first; model_conf is flat/high, so it cannot separate
    cases = [{"f": 0.1 * (i + 1), "model_conf": 0.9, "correct": i >= 5} for i in range(10)]
    summary = risk_coverage_summary(cases)
    # lower area under the risk curve is better
    assert summary["auc_risk_f"] < summary["auc_risk_model_conf"]


def test_risk_coverage_is_empty_safe():
    summary = risk_coverage_summary([])
    assert summary["auc_risk_f"] == 0.0
    assert summary["auc_risk_model_conf"] == 0.0
