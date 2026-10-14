"""Human-in-the-loop: the review card and the human's approve/reject decision."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from energymind.config import REPO_ROOT, load_settings
from energymind.review import (
    HumanDecision,
    apply_human_decision,
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


# --------------------------------------------------------------------------- #
# The card
# --------------------------------------------------------------------------- #


def test_review_card_has_the_expected_fields():
    card = review_card(_case())
    assert "The system detected a possible data-quality anomaly." in card
    assert "Proposed action: Quarantine and estimate" in card
    assert "Confidence: 61.5%" in card
    assert "Reason: Strong data-quality failure indication." in card
    assert "baseline deviation: anomalous" in card
    assert "current vs power identity: broken" in card
    assert "Do you approve this action?" in card
    assert "[Approve] [Reject] [Request more information]" in card


def test_review_card_handles_unknown_origin_and_missing_evidence():
    card = review_card(_case(origin=None, action=None, f=None, record={}, outcome=None))
    assert "possible unclassified anomaly" in card
    assert "Proposed action: no action chosen" in card
    assert "Confidence: n/a" in card
    assert "Evidence: none" in card


# --------------------------------------------------------------------------- #
# The human's decision
# --------------------------------------------------------------------------- #


def test_approve_keeps_the_proposed_decision_and_marks_it_reusable():
    applied = apply_human_decision(_case(correct=True), HumanDecision.APPROVE)
    assert applied["resolution"] == "approved"
    assert applied["final_action"] == "quarantine_and_estimate"
    assert applied["final_origin"] == "data_quality"
    assert applied["reusable"] is True


def test_approve_of_a_wrong_proposal_is_not_reusable():
    applied = apply_human_decision(_case(correct=False), HumanDecision.APPROVE)
    assert applied["resolution"] == "approved"
    assert applied["reusable"] is False


def test_reject_takes_no_action_and_is_not_reusable():
    applied = apply_human_decision(_case(correct=True), HumanDecision.REJECT)
    assert applied["resolution"] == "rejected"
    assert applied["final_action"] is None
    assert applied["final_origin"] is None
    assert applied["reusable"] is False


def test_request_more_info_is_unresolved():
    applied = apply_human_decision(_case(), HumanDecision.REQUEST_MORE_INFO)
    assert applied["resolution"] == "more_info_requested"
    assert applied["final_action"] is None
    assert applied["reusable"] is False


def test_unknown_human_answer_is_rejected():
    with pytest.raises(ValueError):
        apply_human_decision(_case(), "maybe")


# --------------------------------------------------------------------------- #
# End-to-end: escalated cases wait for the human, then are stored
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_pipeline_escalated_cases_wait_for_the_human():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=20, n_test=16)
    human = report["human_review"]
    escalated = human["requested"]
    assert escalated == human["pending_human"] > 0
    assert human["approved"] == 0
    assert len(report["review_cards"]) == escalated
    assert "hil" not in report  # no simulated operators any more
    assert all("Do you approve this action?" in card for card in report["review_cards"].values())


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_pipeline_applies_human_approval_and_stores_it():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=20, n_test=16, human_default="approve")
    human = report["human_review"]
    assert human["approved"] == human["requested"] > 0
    assert human["pending_human"] == 0
    # every escalated case now carries a final decision
    escalated = [c for c in report["cases"] if c["split"] == "test" and c["tier"] >= 2]
    assert all(c["resolution"] == "approved" for c in escalated)
    assert all(c["final_action"] for c in escalated)
    # the human's answers are in the hash-chained audit log
    events = {e["event_type"] for e in report["audit_events"]}
    assert "human_decision" in events


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_pipeline_rejection_takes_no_action():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=20, n_test=16, human_default="reject")
    human = report["human_review"]
    assert human["rejected"] == human["requested"] > 0
    escalated = [c for c in report["cases"] if c["split"] == "test" and c["tier"] >= 2]
    assert all(c["resolution"] == "rejected" and c["final_action"] is None for c in escalated)


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
