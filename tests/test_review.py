"""Human-in-the-loop review: tier-aware cards, prompts, decisions, write-back."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from energymind.config import REPO_ROOT, load_settings
from energymind.review import (
    ORIGIN_NAMES,
    CustomDecision,
    HumanDecision,
    allowed_action_names,
    apply_custom_decision,
    apply_human_decision,
    coerce_approval,
    coerce_custom,
    parse_action,
    parse_approval,
    parse_origin,
    prompt_for_approval,
    prompt_for_custom_action,
    review_card,
)

RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"


def _case(**overrides):
    case = {
        "anomaly_id": "a1",
        "origin": "data_quality",
        "action": "quarantine_and_estimate",
        "f": 0.615,
        "proposal_source": "analyst_fallback",
        "correct": True,
        "true_origin": "data_quality",
        "record": {
            "evidence": [
                {"check": "baseline_deviation", "status": "anomalous"},
                {"check": "current_vs_power_identity", "status": "broken"},
            ]
        },
        "outcome": SimpleNamespace(
            decision={"analyst": {"rationale": "Strong data-quality failure indication."}}
        ),
    }
    case.update(overrides)
    return case


def _reader(*answers):
    """An input function that returns pre-scripted lines in order."""
    queue = list(answers)

    def _input(_prompt=""):
        return queue.pop(0)

    return _input


# --------------------------------------------------------------------------- #
# Cards: tier-2 and tier-3 wording must differ clearly
# --------------------------------------------------------------------------- #


def test_tier2_card_asks_to_approve_the_proposal():
    card = review_card(_case(), tier=2)
    assert "The system detected a possible data-quality anomaly." in card
    assert "Proposed action: Quarantine and estimate" in card
    assert "Confidence: 61.5%" in card
    assert "Reason: Strong data-quality failure indication." in card
    assert "baseline deviation: anomalous" in card
    assert "Do you approve the proposed decision?" in card
    assert "[Approve] [Reject]" in card
    assert "Request more information" not in card
    assert "TIER 3" not in card


def test_tier3_card_warns_and_asks_for_the_humans_own_decision():
    card = review_card(
        _case(origin="adversarial", action="investigate_tampering"), tier=3, risk="high"
    )
    assert "possible possible tampering anomaly" in card
    assert "TIER 3 (human-led, high-risk)" in card
    assert "FOR REFERENCE ONLY" in card
    assert "Proposed action (reference only): Investigate tampering" in card
    assert "risk: high" in card
    assert "Enter your own final action and origin." in card
    # tier 3 does NOT ask to approve/reject the agents' proposal
    assert "Do you approve" not in card


# --------------------------------------------------------------------------- #
# Tier-2 prompt
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("1", HumanDecision.APPROVE),
        ("2", HumanDecision.REJECT),
        ("approve", HumanDecision.APPROVE),
        ("Reject", HumanDecision.REJECT),
        ("  approve  ", HumanDecision.APPROVE),
    ],
)
def test_prompt_for_approval_accepts_valid_input(answer, expected):
    out: list[str] = []
    decision = prompt_for_approval("CARD", input_fn=_reader(answer), output_fn=out.append)
    assert decision is expected
    text = "\n".join(out)
    assert "1. Approve the proposed decision" in text
    assert "2. Reject the proposed decision" in text
    assert "Request more information" not in text


def test_prompt_for_approval_reprompts_on_invalid_then_accepts_valid():
    out: list[str] = []
    decision = prompt_for_approval(
        "CARD", input_fn=_reader("3", "", "maybe", "2"), output_fn=out.append
    )
    assert decision is HumanDecision.REJECT
    text = "\n".join(out)
    assert text.count("Invalid choice") == 3  # "3", "", "maybe"


def test_prompt_for_approval_raises_when_no_input_available():
    def eof(_prompt=""):
        raise EOFError

    with pytest.raises(RuntimeError):
        prompt_for_approval("CARD", input_fn=eof, output_fn=lambda *_: None)


def test_parse_approval_rejects_non_decisions():
    assert parse_approval("1") is HumanDecision.APPROVE
    assert parse_approval("approve") is HumanDecision.APPROVE
    assert parse_approval("request_more_info") is None  # not a tier-2 option
    assert parse_approval("") is None
    assert parse_approval("4") is None


# --------------------------------------------------------------------------- #
# Tier-3 prompt: enter the human's own action/origin
# --------------------------------------------------------------------------- #


def test_prompt_for_custom_action_accepts_valid_action_and_origin():
    out: list[str] = []
    allowed = allowed_action_names(load_settings().policy)
    decision = prompt_for_custom_action(
        "CARD",
        allowed_actions=allowed,
        input_fn=_reader("flag_data_suspect", "data_quality"),
        output_fn=out.append,
    )
    assert decision == CustomDecision(action="flag_data_suspect", origin="data_quality")
    assert "Allowed actions:" in "\n".join(out)


def test_prompt_for_custom_action_reprompts_invalid_action_then_accepts_valid():
    out: list[str] = []
    allowed = allowed_action_names(load_settings().policy)
    decision = prompt_for_custom_action(
        "CARD",
        allowed_actions=allowed,
        input_fn=_reader("bogus_action", "not_an_action", "quarantine_and_estimate", "adversarial"),
        output_fn=out.append,
    )
    assert decision == CustomDecision(action="quarantine_and_estimate", origin="adversarial")
    assert "\n".join(out).count("Invalid action") == 2


def test_prompt_for_custom_action_reprompts_invalid_origin():
    allowed = allowed_action_names(load_settings().policy)
    decision = prompt_for_custom_action(
        "CARD",
        allowed_actions=allowed,
        input_fn=_reader("no_action", "nonsense", "unknown"),
        output_fn=lambda *_: None,
    )
    assert decision == CustomDecision(action="no_action", origin="unknown")


def test_parse_action_and_origin_validate_against_allowed_sets():
    allowed = allowed_action_names(load_settings().policy)
    assert "no_action" in allowed  # explicit no-action is allowed when in the set
    assert parse_action("Flag Data Suspect", allowed) == "flag_data_suspect"
    assert parse_action("bogus", allowed) is None
    assert parse_origin("Data Quality") == "data_quality"
    assert parse_origin("bogus") is None
    assert set(ORIGIN_NAMES) == {"data_quality", "physical_event", "adversarial", "unknown"}


# --------------------------------------------------------------------------- #
# Programmatic coercion + application helpers
# --------------------------------------------------------------------------- #


def test_coerce_approval_and_custom():
    assert coerce_approval("approve") is HumanDecision.APPROVE
    assert coerce_approval("reject") is HumanDecision.REJECT
    assert coerce_approval("request_more_info") is None
    assert coerce_approval(None) is None
    assert coerce_custom({"action": "no_action", "origin": "unknown"}) == CustomDecision(
        "no_action", "unknown"
    )
    assert coerce_custom(("flag_data_suspect", "data_quality")) == CustomDecision(
        "flag_data_suspect", "data_quality"
    )
    assert coerce_custom({"action": "flag_data_suspect"}) is None  # origin required
    assert coerce_custom("approve") is None  # an approve answer is not a custom action


def test_apply_human_decision_approve_and_reject():
    approved = apply_human_decision(_case(correct=True), HumanDecision.APPROVE)
    assert approved["resolution"] == "approved"
    assert approved["final_action"] == "quarantine_and_estimate"
    assert approved["reusable"] is True

    wrong = apply_human_decision(_case(correct=False), HumanDecision.APPROVE)
    assert wrong["reusable"] is False

    rejected = apply_human_decision(_case(), HumanDecision.REJECT)
    assert rejected["resolution"] == "rejected"
    assert rejected["final_action"] is None and rejected["reusable"] is False


def test_apply_custom_decision_uses_the_humans_own_choice():
    applied = apply_custom_decision(_case(), {"action": "flag_data_suspect", "origin": "unknown"})
    assert applied["resolution"] == "custom"
    assert applied["final_action"] == "flag_data_suspect"
    assert applied["final_origin"] == "unknown"
    assert applied["reusable"] is False  # reusability is decided by correctness rules

    with pytest.raises(ValueError):
        apply_custom_decision(_case(), {"action": "flag_data_suspect"})  # missing origin


def test_unknown_human_answer_is_rejected():
    with pytest.raises(ValueError):
        apply_human_decision(_case(), "maybe")


# --------------------------------------------------------------------------- #
# End-to-end: tier 1 / tier 2 / tier 3
# --------------------------------------------------------------------------- #


def _tiers(report, tier):
    return [c for c in report["cases"] if c["split"] == "test" and c["tier"] == tier]


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_tier2_approve_only_affects_tier2_and_tier3_waits():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=20, n_test=16, human_default="approve")
    human = report["human_review"]
    tier2 = _tiers(report, 2)
    tier3 = _tiers(report, 3)
    assert tier2 and tier3, "expected both tiers in the run"
    assert human["approved"] == len(tier2)
    assert human["pending_human"] == len(tier3)  # tier 3 not answered -> still pending
    for case in tier2:
        assert case["resolution"] == "approved"
        assert case["final_action"] == case["action"]
    for case in tier3:
        assert case["resolution"] == "pending_human"
        assert case["final_action"] is None and case["final_origin"] is None


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_tier2_reject_takes_no_action():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=20, n_test=16, human_default="reject")
    human = report["human_review"]
    tier2 = _tiers(report, 2)
    assert human["rejected"] == len(tier2) > 0
    for case in tier2:
        assert case["resolution"] == "rejected"
        assert case["final_action"] is None


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_tier3_custom_human_action_is_recorded_and_stored():
    from energymind import pipeline

    report = pipeline.run(
        load_settings(),
        n_val=20,
        n_test=16,
        human_action_default={"action": "flag_data_suspect", "origin": "data_quality"},
    )
    human = report["human_review"]
    tier3 = _tiers(report, 3)
    assert tier3, "expected tier-3 cases"
    assert human["custom"] == len(tier3)
    for case in tier3:
        assert case["resolution"] == "custom"
        assert case["final_action"] == "flag_data_suspect"
        assert case["final_origin"] == "data_quality"
        # correctness is scored on the human's origin, not the agents' proposal
        assert case["human_correct"] == (case["true_origin"] == "data_quality")


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_tier3_audit_events_and_write_back_status():
    from energymind import pipeline

    report = pipeline.run(
        load_settings(),
        n_val=20,
        n_test=16,
        human_action_default={"action": "flag_data_suspect", "origin": "data_quality"},
    )
    tier3 = _tiers(report, 3)
    tier3_ids = {c["anomaly_id"] for c in tier3}

    events = report["audit_events"]
    # every escalated case has a request, and every answered tier-3 case a custom decision
    requested = {e["anomaly_id"] for e in events if e["event_type"] == "human_review_requested"}
    assert tier3_ids <= requested
    custom = [e for e in events if e["event_type"] == "human_custom_decision"]
    assert {e["anomaly_id"] for e in custom} == tier3_ids
    for event in custom:
        assert event["actor"] == "human"  # the human actor logs the decision
        assert event["payload"]["action"] == "flag_data_suspect"
        assert event["payload"]["origin"] == "data_quality"
    # no tier-3 case is ever logged as approved/rejected like a tier-2 proposal
    tier2_decisions = [
        e for e in events if e["event_type"] == "human_decision" and e["anomaly_id"] in tier3_ids
    ]
    assert tier2_decisions == []

    # write-back status follows the correctness rules on the human's decision
    stored = {
        e["anomaly_id"]: e["payload"]["status"]
        for e in events
        if e["event_type"] == "decision" and e["anomaly_id"] in tier3_ids
    }
    for case in tier3:
        expected = "verified" if case["human_correct"] else "rejected"
        assert stored[case["anomaly_id"]] == expected


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_tier3_missing_input_stays_pending():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=20, n_test=16)
    human = report["human_review"]
    assert human["custom"] == 0
    assert human["pending_human"] == human["requested"] > 0
    assert all(c["final_action"] is None for c in _tiers(report, 3))


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_pipeline_interactive_prompts_per_tier():
    import itertools

    from energymind import pipeline

    allowed = allowed_action_names(load_settings().policy)
    tier2_answers = itertools.cycle([HumanDecision.APPROVE, HumanDecision.REJECT])
    prompted: list[tuple[int, str]] = []

    def reader(card, *, tier, allowed_actions):
        prompted.append((tier, card))
        assert allowed_actions == allowed
        if tier == 2:
            return next(tier2_answers)
        return CustomDecision(action="flag_data_suspect", origin="data_quality")

    report = pipeline.run(
        load_settings(), n_val=20, n_test=16, interactive=True, human_prompt=reader
    )
    human = report["human_review"]
    assert human["pending_human"] == 0  # interactive answered everything
    assert human["approved"] + human["rejected"] == len(_tiers(report, 2))
    assert human["custom"] == len(_tiers(report, 3))
    assert len(prompted) == human["requested"]
    # tier-3 cards carry the high-risk warning; tier-2 do not
    for tier, card in prompted:
        assert ("TIER 3 (human-led, high-risk)" in card) == (tier == 3)
    assert all("Do you approve" not in card for tier, card in prompted if tier == 3)


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_non_interactive_without_answers_stays_pending():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=20, n_test=16)
    events = {e["event_type"] for e in report["audit_events"]}
    assert "human_decision" not in events
    assert "human_custom_decision" not in events
    assert report["human_review"]["pending_human"] == report["human_review"]["requested"] > 0


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_tier1_autonomous_behavior_unchanged(monkeypatch):
    from energymind import pipeline

    # force every test case through the autonomous tier
    monkeypatch.setattr(
        pipeline.policy, "route_all", lambda cases, policy, thresholds: [1] * len(cases)
    )
    report = pipeline.run(load_settings(), n_val=20, n_test=16)
    test_cases = [c for c in report["cases"] if c["split"] == "test"]
    assert test_cases and all(c["tier"] == 1 for c in test_cases)
    for case in test_cases:
        assert case["resolution"] == "auto"
        assert case["final_action"] == case["action"]  # the agents' proposal is used
        assert case["anomaly_id"] not in report["review_cards"]  # no human card
    assert report["review_cards"] == {}
    assert report["human_review"]["approved"] == 0
    events = {e["event_type"] for e in report["audit_events"]}
    assert "human_review_requested" not in events
    assert "human_decision" not in events
    assert "human_custom_decision" not in events


# --------------------------------------------------------------------------- #
# The same pipeline must run with the cloud/LLM agents (injected, not hard-coded)
# --------------------------------------------------------------------------- #


class _CountingAnalyst:
    """Stand-in for a cloud Analyst: counts calls, delegates to the rule agent."""

    def __init__(self):
        from energymind.agents import RuleBasedAnalyst

        self._inner = RuleBasedAnalyst()
        self.calls = 0

    @property
    def role(self):
        return self._inner.role

    def propose(self, *args, **kwargs):
        self.calls += 1
        return self._inner.propose(*args, **kwargs)


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_pipeline_runs_with_injected_agents():
    from energymind import pipeline
    from energymind.agents import RuleBasedValidator

    analyst = _CountingAnalyst()
    pipeline.run(
        load_settings(),
        n_val=8,
        n_test=6,
        analyst=analyst,
        validator=RuleBasedValidator(),
    )
    assert analyst.calls > 0  # the injected Analyst reached the deliberation
