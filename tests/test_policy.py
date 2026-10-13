"""Tests for M8 routing: three tiers, never-automate, threshold selection."""

from __future__ import annotations

from energymind.config import load_settings
from energymind.policy import RoutingCase, choose_thresholds, classify_tier, route_all


def _policy():
    return load_settings().policy


def test_tier1_requires_data_quality_low_risk_and_convergence():
    policy = _policy()
    # low-risk, reversible, converged, data_quality, high f -> tier 1
    assert (
        classify_tier(
            0.97, "flag_data_suspect", "data_quality", True, policy, tau_auto=0.9, tau_review=0.6
        )
        == 1
    )


def test_physical_event_is_not_autonomous():
    policy = _policy()
    # same high confidence but a physical_event with a medium-risk action -> tier 2, not 1
    assert (
        classify_tier(
            0.97,
            "open_maintenance_ticket",
            "physical_event",
            True,
            policy,
            tau_auto=0.9,
            tau_review=0.6,
        )
        == 2
    )
    # the high-risk physical action is human-led
    assert classify_tier(0.97, "coordinate_with_grid_operator", "physical_event", True, policy) == 3


def test_adversarial_and_unknown_are_human_led():
    policy = _policy()
    assert classify_tier(0.99, "investigate_tampering", "adversarial", True, policy) == 3
    assert classify_tier(0.99, "flag_data_suspect", "unknown", True, policy) == 3


def test_high_risk_and_non_convergence_are_human_led():
    policy = _policy()
    assert classify_tier(0.99, "coordinate_with_grid_operator", "data_quality", True, policy) == 3
    assert classify_tier(0.99, "flag_data_suspect", "data_quality", False, policy) == 3


def test_never_automate_action_is_human_led():
    policy = _policy()
    assert classify_tier(0.99, "remote_disconnect", "data_quality", True, policy) == 3


def test_low_confidence_is_human_led():
    policy = _policy()
    assert (
        classify_tier(0.45, "flag_data_suspect", "data_quality", True, policy, tau_review=0.6) == 3
    )


def test_choose_thresholds_meets_error_target():
    policy = _policy()
    # 20 cases: high-f ones are always correct, so a low tau_auto should meet the target
    cases = [
        RoutingCase(
            f=0.95, action="flag_data_suspect", origin="data_quality", converged=True, correct=True
        )
        for _ in range(10)
    ] + [
        RoutingCase(
            f=0.4, action="flag_data_suspect", origin="data_quality", converged=True, correct=False
        )
        for _ in range(10)
    ]
    thresholds = choose_thresholds(cases, policy, error_target=0.05)
    assert thresholds["tau_auto"] <= 0.95
    tiers = route_all(cases, policy, thresholds)
    # the confident, correct cases are autonomous; the rest are not
    assert tiers[0] == 1
    assert tiers[-1] in (2, 3)
