"""Fix 4: the proposal is separated from routing; non-convergence is not an error."""

from __future__ import annotations

import pytest

from energymind.agent_pipeline import run_case
from energymind.agents import Action, AgentRole, DecisionOrigin, Proposal
from energymind.config import REPO_ROOT, load_settings
from energymind.pipeline import _correct
from energymind.policy import classify_tier
from energymind.retrieval import InMemoryRetriever
from energymind.store import AuditLog, MemoryStore

TAU = 0.9
RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"


def _record(signature, *, origin_pred="data_quality", severity="medium"):
    return {
        "anomaly_id": "current-0001",
        "household_id": "HH_00072_AUS",
        "t_start": "2020-11-14T18:00:00Z",
        "t_end": "2020-11-15T06:00:00Z",
        "duration_h": 12.0,
        "severity": severity,
        "type_pred": "gain_error",
        "origin_pred": origin_pred,
        "signature": list(signature),
        "detection": {"model": "rules", "score": 12.0, "p_anomaly": 0.9},
        "classification": {
            "model": "lightgbm",
            "type_probs": {"gain_error": 0.7},
            "origin_probs": {origin_pred: 0.7},
        },
        "explanation": {"shap_top": [["p_resid_z_max", 0.4]], "lime_top": [], "reliability": 0.5},
        "evidence": [
            {
                "check": "baseline_deviation",
                "observed_kw": 5.1,
                "expected_kw": 3.9,
                "z": 3.4,
                "status": "anomalous",
            }
        ],
    }


def _vec(i: int = 0) -> list[float]:
    v = [0.0] * 64
    v[i % 64] = 1.0
    return v


class ScriptedAgent:
    """An agent that always returns the same proposal (to force (non-)convergence)."""

    def __init__(self, role, origin, action):
        self._role = role
        self.origin = origin
        self.action = action

    @property
    def role(self):
        return self._role

    def propose(self, view, round, other_proposal=None, own_previous_proposal=None):
        evidence_ids = list(view.evidence.ids)[:1] if view is not None else []
        if not evidence_ids:
            evidence_ids = ["model.p_anomaly"]
        return Proposal(
            agent=self._role,
            round=round,
            action=self.action,
            origin=self.origin,
            support=0.8,
            evidence_ids=evidence_ids,
            rationale="scripted",
            backend="scripted",
            model="scripted-v1",
            prompt_hash="hash",
            latency_ms=1.0,
        )


def _store():
    return MemoryStore(InMemoryRetriever(TAU), AuditLog(sqlite_path=":memory:"))


def _disagreeing_agents():
    analyst = ScriptedAgent(
        AgentRole.ANALYST, DecisionOrigin.DATA_QUALITY, Action.FLAG_DATA_SUSPECT
    )
    validator = ScriptedAgent(
        AgentRole.VALIDATOR, DecisionOrigin.PHYSICAL_EVENT, Action.COORDINATE_WITH_GRID_OPERATOR
    )
    return analyst, validator


def test_non_converged_case_falls_back_to_analyst_proposal():
    analyst, validator = _disagreeing_agents()
    outcome = run_case(
        _record(_vec(0)),
        _store(),
        settings=load_settings(),
        tau_sim=TAU,
        analyst=analyst,
        validator=validator,
    )
    assert outcome.deliberation is not None
    assert outcome.deliberation.converged is False
    assert outcome.proposal_source == "analyst_fallback"
    assert outcome.proposal_origin == "data_quality"  # the Analyst's proposal
    assert outcome.proposal_action == "flag_data_suspect"
    # Fix 4 acceptance: a non-converged case can still be correct.
    assert _correct(outcome, "data_quality") is True


def test_non_converged_proposal_is_still_tier_3():
    analyst, validator = _disagreeing_agents()
    settings = load_settings()
    outcome = run_case(
        _record(_vec(0)),
        _store(),
        settings=settings,
        tau_sim=TAU,
        analyst=analyst,
        validator=validator,
    )
    tier = classify_tier(
        1.0,
        outcome.proposal_action,
        outcome.proposal_origin,
        converged=False,
        policy=settings.policy,
        tau_auto=0.5,
        tau_review=0.2,
    )
    assert tier == 3  # non-convergence forces human-led routing


def test_unknown_analyst_origin_is_abstention_and_incorrect():
    analyst = ScriptedAgent(AgentRole.ANALYST, DecisionOrigin.UNKNOWN, Action.FLAG_DATA_SUSPECT)
    validator = ScriptedAgent(
        AgentRole.VALIDATOR, DecisionOrigin.PHYSICAL_EVENT, Action.COORDINATE_WITH_GRID_OPERATOR
    )
    settings = load_settings()
    outcome = run_case(
        _record(_vec(0)),
        _store(),
        settings=settings,
        tau_sim=TAU,
        analyst=analyst,
        validator=validator,
    )
    assert outcome.proposal_origin == "unknown"
    assert _correct(outcome, "data_quality") is False
    tier = classify_tier(
        1.0,
        outcome.proposal_action,
        outcome.proposal_origin,
        converged=False,
        policy=settings.policy,
        tau_auto=0.5,
        tau_review=0.2,
    )
    assert tier == 3


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_pipeline_nonconverged_cases_are_not_all_incorrect():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=20, n_test=16)
    # Fix 4 acceptance: fewer than 100% of non-converged cases are incorrect.
    assert report["nonconverged_correct_fraction"] is not None
    assert report["nonconverged_correct_fraction"] > 0.0
    assert report["abstention_rate"] is not None
    assert report["leakage_violations"] == 0
