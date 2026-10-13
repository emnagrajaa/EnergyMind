"""Tests for M8 simulated operators and risk-coverage."""

from __future__ import annotations

from energymind import simulate_hil


def _cases() -> list[dict]:
    # 10 cases: half correct with high f, half wrong with low f
    cases = []
    for i in range(10):
        correct = i < 5
        cases.append(
            {
                "f": 0.95 if correct else 0.2,
                "model_conf": 0.9 if correct else 0.8,  # model overconfident on wrong ones
                "action": "flag_data_suspect",
                "origin": "data_quality",
                "converged": True,
                "correct": correct,
            }
        )
    return cases


def test_operators_are_seed_deterministic():
    cases = _cases()
    rng_rows_a = simulate_hil.simulate(
        cases, [3] * len(cases), simulate_hil.rubber_stamper(), seed=1
    )
    rng_rows_b = simulate_hil.simulate(
        cases, [3] * len(cases), simulate_hil.rubber_stamper(), seed=1
    )
    assert rng_rows_a == rng_rows_b


def test_calibrated_operator_beats_rubber_stamper_when_proposals_are_wrong():
    cases = _cases()
    tiers = [3] * len(cases)  # everything escalated
    calibrated = simulate_hil.simulate(cases, tiers, simulate_hil.calibrated_operator(), seed=0)
    rubber = simulate_hil.simulate(cases, tiers, simulate_hil.rubber_stamper(), seed=0)
    # rubber-stamper accepts the wrong proposals -> higher error, more automation bias
    assert rubber["final_error"] >= calibrated["final_error"]
    assert rubber["automation_bias_rate"] >= calibrated["automation_bias_rate"]


def test_always_auto_has_no_operator_load():
    cases = _cases()
    rows = simulate_hil.simulate(cases, [1] * len(cases), simulate_hil.rubber_stamper())
    assert rows["operator_load"] == 0
    assert rows["coverage_auto"] == 1.0


def test_risk_coverage_is_better_for_f_than_overconfident_model():
    # f ranks correct cases first; model_conf is flat/high, so it cannot separate
    cases = [{"f": 0.1 * (i + 1), "model_conf": 0.9, "correct": i >= 5} for i in range(10)]
    summary = simulate_hil.risk_coverage_summary(cases)
    # lower area under the risk curve is better
    assert summary["auc_risk_f"] < summary["auc_risk_model_conf"]


def test_run_simulation_covers_strategies_and_operators():
    from energymind.config import load_settings

    cases = _cases()
    settings = load_settings()
    thresholds = {"tau_auto": 0.9, "tau_review": 0.5}
    rows = simulate_hil.run_simulation(cases, settings.policy, thresholds)
    strategies = {r["strategy"] for r in rows}
    assert strategies == {"always_auto", "single_threshold", "tiered_f"}
    assert any(r["operator"] == "calibrated" for r in rows)
