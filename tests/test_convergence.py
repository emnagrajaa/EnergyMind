"""Fix 5: agents must be able to converge (origin + action risk class)."""

from __future__ import annotations

import pytest

from energymind.agent_pipeline import run_case
from energymind.agents import (
    Action,
    AgentRole,
    DecisionOrigin,
    Proposal,
    proposals_converge,
)
from energymind.config import REPO_ROOT, load_settings
from energymind.policy import action_risk, lower_risk_action
from energymind.retrieval import InMemoryRetriever
from energymind.store import AuditLog, MemoryStore

TAU = 0.9
RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"


def _policy():
    return load_settings().policy


class ScriptedAgent:
    def __init__(self, role, origin, action):
        self._role = role
        self.origin = origin
        self.action = action

    @property
    def role(self):
        return self._role

    def propose(self, view, round, other_proposal=None, own_previous_proposal=None):
        ids = list(view.evidence.ids)[:1] if view is not None else []
        return Proposal(
            agent=self._role,
            round=round,
            action=self.action,
            origin=self.origin,
            support=0.8,
            evidence_ids=ids or ["model.p_anomaly"],
            rationale="scripted",
            backend="scripted",
            model="scripted-v1",
            prompt_hash="hash",
            latency_ms=1.0,
        )


def _record():
    return {
        "anomaly_id": "current-0001",
        "household_id": "HH_00072_AUS",
        "t_start": "2020-11-14T18:00:00Z",
        "t_end": "2020-11-15T06:00:00Z",
        "duration_h": 12.0,
        "severity": "medium",
        "type_pred": "gain_error",
        "origin_pred": "data_quality",
        "signature": [1.0] + [0.0] * 63,
        "detection": {"model": "rules", "score": 12.0, "p_anomaly": 0.9},
        "classification": {
            "model": "lightgbm",
            "type_probs": {"gain_error": 0.7},
            "origin_probs": {"data_quality": 0.7},
        },
        "explanation": {"shap_top": [], "lime_top": [], "reliability": 0.5},
        "evidence": [
            {"check": "baseline_deviation", "z": 3.4, "status": "anomalous"},
        ],
    }


def _store():
    return MemoryStore(InMemoryRetriever(TAU), AuditLog(sqlite_path=":memory:"))


def test_action_risk_and_lower_risk_action():
    policy = _policy()
    assert action_risk(Action.FLAG_DATA_SUSPECT, policy) == "low"
    assert action_risk(Action.OPEN_MAINTENANCE_TICKET, policy) == "medium"
    assert lower_risk_action(Action.OPEN_MAINTENANCE_TICKET, Action.FLAG_DATA_SUSPECT, policy) == (
        Action.FLAG_DATA_SUSPECT
    )


def test_proposals_converge_on_same_origin_and_risk_class():
    policy = _policy()
    common = {
        "round": 1,
        "support": 0.8,
        "rationale": "x",
        "backend": "scripted",
        "model": "scripted-v1",
        "prompt_hash": "hash",
        "latency_ms": 1.0,
    }
    analyst = Proposal(
        agent=AgentRole.ANALYST,
        action=Action.FLAG_DATA_SUSPECT,  # low
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        **common,
    )
    validator = Proposal(
        agent=AgentRole.VALIDATOR,
        action=Action.REQUEST_METER_REREAD,  # also low
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["sensor.flat_run_length"],
        **common,
    )
    risk_of = lambda action: action_risk(action, policy)  # noqa: E731
    assert proposals_converge(analyst, validator, risk_of) is True

    differing_risk = validator.model_copy(update={"action": Action.OPEN_MAINTENANCE_TICKET})
    assert proposals_converge(analyst, differing_risk, risk_of) is False

    differing_origin = validator.model_copy(update={"origin": DecisionOrigin.PHYSICAL_EVENT})
    assert proposals_converge(analyst, differing_origin, risk_of) is False


def test_deliberation_converges_and_takes_the_lower_risk_action():
    analyst = ScriptedAgent(
        AgentRole.ANALYST, DecisionOrigin.DATA_QUALITY, Action.REQUEST_METER_REREAD
    )
    validator = ScriptedAgent(
        AgentRole.VALIDATOR, DecisionOrigin.DATA_QUALITY, Action.FLAG_DATA_SUSPECT
    )
    outcome = run_case(
        _record(),
        _store(),
        settings=load_settings(),
        tau_sim=TAU,
        analyst=analyst,
        validator=validator,
    )
    assert outcome.deliberation is not None
    assert outcome.deliberation.converged is True
    assert outcome.proposal_origin == "data_quality"
    # both are low-risk; the agreed action is the deterministic lower one
    assert outcome.proposal_action == "flag_data_suspect"
    assert outcome.proposal_source == "converged"


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_pipeline_reports_positive_convergence_rate():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=30, n_test=20)
    assert report["convergence_rate"] > 0.0
    by_origin = report["convergence_rate_by_origin"]
    assert by_origin, "convergence must be broken down per origin"
    assert all({"n", "converged", "rate"} <= set(v) for v in by_origin.values())
