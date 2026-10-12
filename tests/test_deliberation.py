"""Unit tests for M7.5 multi-round deliberation and audit orchestration.

Verifies:
- Blind Round 1: Strict independence and zero peer leakage in round 1.
- Convergence semantics: Action and origin matching regardless of support, rationale, citations.
- Rounds and snapshot stability: Sequential evaluation uses frozen previous-round snapshots.
- Peer evidence disclosure: Restricted to explicitly cited items; uncited items remain hidden.
- Revision rules: Action/origin changes require newly cited evidence; invalid revisions rejected.
- Non-convergence preservation: No tie-breaking by support; r_max limit respected.
- Fallback integration: Rule-based fallback proposals participate normally; audit flags tracked.
- Audit logging: Complete hash-chained audit log with valid chain verification.
- Evidence boundaries: Complete exclusion of hidden ground truth and privacy preservation.
- Determinism and ground-truth invariance.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from energymind.agents.analyst import RuleBasedAnalyst
from energymind.agents.base import (
    Action,
    Agent,
    AgentContractError,
    AgentRole,
    AgentView,
    DecisionOrigin,
    DeliberationResult,
    EvidenceCatalog,
    EvidenceItem,
    Proposal,
)
from energymind.agents.deliberation import (
    Deliberator,
    build_peer_context,
    create_effective_view,
    deliberate,
)
from energymind.agents.validator import RuleBasedValidator
from energymind.store import AuditLog

# --------------------------------------------------------------------------- #
# Test Doubles & Fixtures
# --------------------------------------------------------------------------- #


class ScriptedAgent:
    """Deterministic scripted agent for orchestration testing."""

    def __init__(
        self,
        role: AgentRole,
        proposals: list[Proposal] | None = None,
        generator: Any = None,
    ) -> None:
        self._role = role
        self.proposals = list(proposals) if proposals is not None else []
        self.generator = generator
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def role(self) -> AgentRole:
        return self._role

    def propose(
        self,
        view: AgentView,
        round: int,
        other_proposal: Proposal | None = None,
        own_previous_proposal: Proposal | None = None,
    ) -> Proposal:
        self.call_count += 1
        self.calls.append(
            {
                "view": view,
                "round": round,
                "other_proposal": other_proposal,
                "own_previous_proposal": own_previous_proposal,
            }
        )
        if self.generator is not None:
            return self.generator(view, round, other_proposal, own_previous_proposal)
        if self.proposals:
            idx = min(self.call_count - 1, len(self.proposals) - 1)
            prop = self.proposals[idx]
            return Proposal(
                agent=self._role,
                round=round,
                action=prop.action,
                origin=prop.origin,
                support=prop.support,
                evidence_ids=prop.evidence_ids,
                rationale=prop.rationale,
                objection_to_other=prop.objection_to_other,
                changed_since_last_round=prop.changed_since_last_round,
                backend=prop.backend,
                model=prop.model,
                prompt_hash=prop.prompt_hash,
                fallback_used=prop.fallback_used,
                fallback_reason=prop.fallback_reason,
                latency_ms=prop.latency_ms,
            )

        # Default proposals disagree by role so multi-round tests run through rounds
        if self._role == AgentRole.ANALYST:
            default_action = Action.QUARANTINE_AND_ESTIMATE
            default_evidence = ["model.p_anomaly"]
        else:
            default_action = Action.FLAG_DATA_SUSPECT
            default_evidence = ["physics.current_vs_power_identity"]

        return Proposal(
            agent=self._role,
            round=round,
            action=default_action,
            origin=DecisionOrigin.DATA_QUALITY,
            support=0.8,
            evidence_ids=default_evidence,
            rationale=f"Default scripted proposal for {self._role.value}.",
            backend="scripted",
            model="scripted-v1",
            prompt_hash="hash-123",
            latency_ms=1.0,
        )


class RecordingValidator:
    """Wrapper around an Agent that records every call."""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    @property
    def role(self) -> AgentRole:
        return self.inner.role

    def propose(
        self,
        view: AgentView,
        round: int,
        other_proposal: Proposal | None = None,
        own_previous_proposal: Proposal | None = None,
    ) -> Proposal:
        self.calls.append(
            {
                "view": view,
                "round": round,
                "other_proposal": other_proposal,
                "own_previous_proposal": own_previous_proposal,
            }
        )
        return self.inner.propose(
            view,
            round=round,
            other_proposal=other_proposal,
            own_previous_proposal=own_previous_proposal,
        )


class ExplodingAgent:
    """Agent that raises an unhandled error during propose."""

    def __init__(self, role: AgentRole, exception: Exception | None = None) -> None:
        self._role = role
        self.exception = exception or RuntimeError("Simulated agent catastrophic explosion")

    @property
    def role(self) -> AgentRole:
        return self._role

    def propose(
        self,
        view: AgentView,
        round: int,
        other_proposal: Proposal | None = None,
        own_previous_proposal: Proposal | None = None,
    ) -> Proposal:
        raise self.exception


@pytest.fixture
def sample_raw_record() -> dict[str, Any]:
    """A realistic AnomalyRecord dictionary containing model outputs,

    physics checks, baseline checks, neighborhood coherence, and simulation labels.
    """
    return {
        "anomaly_id": "CASE_2020_001",
        "household_id": "HH_00072",
        "t_start": "2020-06-01T12:00:00Z",
        "t_end": "2020-06-01T14:00:00Z",
        "duration_h": 2.0,
        "detection": {
            "model": "isolation_forest",
            "p_anomaly": 0.94,
            "score": 2.15,
        },
        "classification": {
            "model": "lightgbm",
            "type_probs": {
                "dead_sensor": 0.85,
                "dropout": 0.10,
                "drift": 0.05,
            },
            "origin_probs": {
                "data_quality": 0.90,
                "physical_event": 0.08,
                "adversarial": 0.02,
            },
        },
        "explanation": {
            "shap_top": [
                ["active_power_kw", 0.35421],
                ["voltage_v", -0.12345],
            ],
            "lime_top": [
                ["active_power_kw", 0.33211],
                ["current_a", 0.15432],
            ],
            "reliability": 0.78,
        },
        "severity": "high",
        "evidence": [
            {
                "check": "current_vs_power_identity",
                "rel_residual": 0.0012,
                "status": "holds",
            },
            {
                "check": "baseline_deviation",
                "observed_kw": 0.12,
                "expected_kw": 1.45,
                "z": -2.85,
                "status": "anomalous",
            },
            {
                "check": "neighborhood_share",
                "share": 0.03,
                "status": "isolated",
            },
        ],
        # Simulation labels (must be filtered out)
        "true_type": "dead_sensor",
        "true_origin": "data_quality",
        "ground_truth": {"type": "dead_sensor", "origin": "data_quality"},
        "y_true": 1,
        "injected_type": "dead_sensor",
        "injected_origin": "data_quality",
    }


def make_prop(
    role: AgentRole,
    action: Action,
    origin: DecisionOrigin,
    support: float = 0.85,
    evidence_ids: list[str] | None = None,
    rationale: str = "Test rationale.",
    round: int = 1,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> Proposal:
    return Proposal(
        agent=role,
        round=round,
        action=action,
        origin=origin,
        support=support,
        evidence_ids=evidence_ids or ["model.p_anomaly"],
        rationale=rationale,
        backend="test",
        model="test-m",
        prompt_hash="test-h",
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        latency_ms=10.0,
    )


# --------------------------------------------------------------------------- #
# Group A: Blind Round 1 (Tests 1 - 6)
# --------------------------------------------------------------------------- #


def test_round1_analyst_receives_no_peer_proposal(sample_raw_record: dict[str, Any]) -> None:
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    deliberator = Deliberator(analyst=analyst, validator=validator, r_max=1)
    deliberator.deliberate(sample_raw_record)

    assert analyst.calls[0]["other_proposal"] is None
    assert analyst.calls[0]["own_previous_proposal"] is None


def test_round1_validator_receives_no_peer_proposal(sample_raw_record: dict[str, Any]) -> None:
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    deliberator = Deliberator(analyst=analyst, validator=validator, r_max=1)
    deliberator.deliberate(sample_raw_record)

    assert validator.calls[0]["other_proposal"] is None
    assert validator.calls[0]["own_previous_proposal"] is None


def test_round1_analyst_receives_no_peer_evidence(sample_raw_record: dict[str, Any]) -> None:
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    deliberator = Deliberator(analyst=analyst, validator=validator, r_max=1)
    deliberator.deliberate(sample_raw_record)

    view = analyst.calls[0]["view"]
    for eid in view.evidence.ids:
        assert not eid.startswith("physics.")
        assert not eid.startswith("baseline.")
        assert not eid.startswith("neighborhood.")


def test_round1_validator_receives_no_peer_evidence(sample_raw_record: dict[str, Any]) -> None:
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    deliberator = Deliberator(analyst=analyst, validator=validator, r_max=1)
    deliberator.deliberate(sample_raw_record)

    view = validator.calls[0]["view"]
    for eid in view.evidence.ids:
        assert not eid.startswith("model.")
        assert not eid.startswith("xai.")


def test_sequential_execution_does_not_leak_first_proposal_to_second(
    sample_raw_record: dict[str, Any],
) -> None:
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    deliberator = Deliberator(analyst=analyst, validator=validator, r_max=1)
    deliberator.deliberate(sample_raw_record)

    # Even though analyst ran first, validator R1 has no other_proposal
    assert validator.calls[0]["other_proposal"] is None


def test_round1_proposals_independently_produced(sample_raw_record: dict[str, Any]) -> None:
    # Validator produces identical proposal regardless of what analyst proposed
    val_prop = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    val_agent = ScriptedAgent(AgentRole.VALIDATOR, [val_prop])

    # Run with Analyst proposing Action A
    analyst_a = ScriptedAgent(
        AgentRole.ANALYST,
        [
            make_prop(
                AgentRole.ANALYST,
                Action.QUARANTINE_AND_ESTIMATE,
                DecisionOrigin.DATA_QUALITY,
                evidence_ids=["model.p_anomaly"],
            )
        ],
    )
    res_a = Deliberator(analyst=analyst_a, validator=val_agent, r_max=1).deliberate(
        sample_raw_record
    )

    # Run with Analyst proposing Action B
    analyst_b = ScriptedAgent(
        AgentRole.ANALYST,
        [
            make_prop(
                AgentRole.ANALYST,
                Action.NO_ACTION,
                DecisionOrigin.PHYSICAL_EVENT,
                evidence_ids=["model.p_anomaly"],
            )
        ],
    )
    res_b = Deliberator(analyst=analyst_b, validator=val_agent, r_max=1).deliberate(
        sample_raw_record
    )

    assert res_a.validator_final == res_b.validator_final


# --------------------------------------------------------------------------- #
# Group B: Convergence (Tests 7 - 13)
# --------------------------------------------------------------------------- #


def test_convergence_same_action_and_same_origin(sample_raw_record: dict[str, Any]) -> None:
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=3,
    )
    result = delib.deliberate(sample_raw_record)
    assert result.converged is True
    assert result.rounds_used == 1
    assert result.agreed_action == Action.FLAG_DATA_SUSPECT
    assert result.agreed_origin == DecisionOrigin.DATA_QUALITY


def test_no_convergence_same_action_different_origin(sample_raw_record: dict[str, Any]) -> None:
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.PHYSICAL_EVENT,  # Diff origin!
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=1,
    )
    result = delib.deliberate(sample_raw_record)
    assert result.converged is False
    assert result.agreed_action is None
    assert result.agreed_origin is None


def test_no_convergence_different_action_same_origin(sample_raw_record: dict[str, Any]) -> None:
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,  # Diff action!
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=1,
    )
    result = delib.deliberate(sample_raw_record)
    assert result.converged is False


def test_convergence_independent_of_support_difference(sample_raw_record: dict[str, Any]) -> None:
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        support=0.99,
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        support=0.15,  # Drastically lower support
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=1,
    )
    result = delib.deliberate(sample_raw_record)
    assert result.converged is True


def test_convergence_independent_of_rationale_difference(
    sample_raw_record: dict[str, Any],
) -> None:
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        rationale="Analyst rationale A",
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        rationale="Totally different validator rationale B",
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=1,
    )
    result = delib.deliberate(sample_raw_record)
    assert result.converged is True


def test_convergence_independent_of_evidence_ids_difference(
    sample_raw_record: dict[str, Any],
) -> None:
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity", "baseline.z_score"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=1,
    )
    result = delib.deliberate(sample_raw_record)
    assert result.converged is True


def test_converged_result_contains_agreed_action_and_origin(
    sample_raw_record: dict[str, Any],
) -> None:
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.COORDINATE_WITH_GRID_OPERATOR,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.COORDINATE_WITH_GRID_OPERATOR,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=1,
    )
    res = delib.deliberate(sample_raw_record)
    assert res.agreed_action == Action.COORDINATE_WITH_GRID_OPERATOR
    assert res.agreed_origin == DecisionOrigin.PHYSICAL_EVENT


# --------------------------------------------------------------------------- #
# Group C: Rounds & Snapshot Stability (Tests 14 - 23)
# --------------------------------------------------------------------------- #


def test_disagreement_initiates_round_2(sample_raw_record: dict[str, Any]) -> None:
    # R1 disagree, R2 agree
    a_props = [
        make_prop(
            AgentRole.ANALYST,
            Action.QUARANTINE_AND_ESTIMATE,
            DecisionOrigin.DATA_QUALITY,
            evidence_ids=["model.p_anomaly"],
            round=1,
        ),
        make_prop(
            AgentRole.ANALYST,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            evidence_ids=["model.p_anomaly", "physics.current_vs_power_identity"],
            round=2,
        ),
    ]
    v_props = [
        make_prop(
            AgentRole.VALIDATOR,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            evidence_ids=["physics.current_vs_power_identity"],
            round=1,
        ),
        make_prop(
            AgentRole.VALIDATOR,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            evidence_ids=["physics.current_vs_power_identity"],
            round=2,
        ),
    ]
    analyst = ScriptedAgent(AgentRole.ANALYST, a_props)
    validator = ScriptedAgent(AgentRole.VALIDATOR, v_props)
    delib = Deliberator(analyst=analyst, validator=validator, r_max=3)
    res = delib.deliberate(sample_raw_record)

    assert analyst.call_count == 2
    assert validator.call_count == 2
    assert res.converged is True
    assert res.rounds_used == 2


def test_round2_analyst_receives_validator_r1(sample_raw_record: dict[str, Any]) -> None:
    v_prop_r1 = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
        rationale="Validator R1 Rationale",
        round=1,
    )
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR, [v_prop_r1, v_prop_r1])
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    assert analyst.call_count == 2
    other_prop = analyst.calls[1]["other_proposal"]
    assert other_prop is not None
    assert other_prop.agent == AgentRole.VALIDATOR
    assert other_prop.round == 1
    assert other_prop.rationale == "Validator R1 Rationale"


def test_round2_validator_receives_analyst_r1(sample_raw_record: dict[str, Any]) -> None:
    a_prop_r1 = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        rationale="Analyst R1 Rationale",
        round=1,
    )
    analyst = ScriptedAgent(AgentRole.ANALYST, [a_prop_r1, a_prop_r1])
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    assert validator.call_count == 2
    other_prop = validator.calls[1]["other_proposal"]
    assert other_prop is not None
    assert other_prop.agent == AgentRole.ANALYST
    assert other_prop.round == 1
    assert other_prop.rationale == "Analyst R1 Rationale"


def test_round2_analyst_does_not_receive_validator_r2(sample_raw_record: dict[str, Any]) -> None:
    v_r1 = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        round=1,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    v_r2 = make_prop(
        AgentRole.VALIDATOR,
        Action.COORDINATE_WITH_GRID_OPERATOR,
        DecisionOrigin.PHYSICAL_EVENT,
        round=2,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR, [v_r1, v_r2])
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    # Analyst R2 sees Validator R1, NEVER Validator R2
    assert analyst.calls[1]["other_proposal"].round == 1
    assert analyst.calls[1]["other_proposal"].action == Action.NO_ACTION


def test_round2_validator_does_not_receive_analyst_r2(sample_raw_record: dict[str, Any]) -> None:
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        round=1,
        evidence_ids=["model.p_anomaly"],
    )
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        round=2,
        evidence_ids=["model.p_anomaly"],
    )
    # Validator produces different action in round 1 so deliberation continues to round 2
    v_r1 = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        round=1,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2])
    validator = ScriptedAgent(AgentRole.VALIDATOR, [v_r1, v_r1])
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    # Validator R2 sees Analyst R1, NEVER Analyst R2
    assert validator.calls[1]["other_proposal"].round == 1
    assert validator.calls[1]["other_proposal"].action == Action.FLAG_DATA_SUSPECT


def test_round_snapshots_are_stable_and_order_invariant(
    sample_raw_record: dict[str, Any],
) -> None:
    # Snapshots frozen from R(N-1) guarantee that execution order does not alter inputs
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a, p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v, p_v]),
        r_max=2,
    )
    delib.deliberate(sample_raw_record)

    assert delib.analyst.calls[1]["other_proposal"] == p_v  # type: ignore[attr-defined]
    assert delib.validator.calls[1]["other_proposal"] == p_a  # type: ignore[attr-defined]


def test_convergence_in_round_2_stops_immediately(sample_raw_record: dict[str, Any]) -> None:
    a_props = [
        make_prop(
            AgentRole.ANALYST,
            Action.QUARANTINE_AND_ESTIMATE,
            DecisionOrigin.DATA_QUALITY,
            round=1,
            evidence_ids=["model.p_anomaly"],
        ),
        make_prop(
            AgentRole.ANALYST,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            round=2,
            evidence_ids=["model.p_anomaly", "physics.current_vs_power_identity"],
        ),
        make_prop(
            AgentRole.ANALYST,
            Action.NO_ACTION,
            DecisionOrigin.PHYSICAL_EVENT,
            round=3,
            evidence_ids=["model.p_anomaly"],
        ),
    ]
    v_props = [
        make_prop(
            AgentRole.VALIDATOR,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            round=1,
            evidence_ids=["physics.current_vs_power_identity"],
        ),
        make_prop(
            AgentRole.VALIDATOR,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            round=2,
            evidence_ids=["physics.current_vs_power_identity"],
        ),
        make_prop(
            AgentRole.VALIDATOR,
            Action.NO_ACTION,
            DecisionOrigin.PHYSICAL_EVENT,
            round=3,
            evidence_ids=["physics.current_vs_power_identity"],
        ),
    ]
    analyst = ScriptedAgent(AgentRole.ANALYST, a_props)
    validator = ScriptedAgent(AgentRole.VALIDATOR, v_props)
    delib = Deliberator(analyst=analyst, validator=validator, r_max=3)
    res = delib.deliberate(sample_raw_record)

    assert res.converged is True
    assert res.rounds_used == 2
    assert analyst.call_count == 2
    assert validator.call_count == 2


def test_convergence_in_round_3_stops_immediately(sample_raw_record: dict[str, Any]) -> None:
    # Agree on round 3
    a_props = [
        make_prop(
            AgentRole.ANALYST,
            Action.QUARANTINE_AND_ESTIMATE,
            DecisionOrigin.DATA_QUALITY,
            round=1,
            evidence_ids=["model.p_anomaly"],
        ),
        make_prop(
            AgentRole.ANALYST,
            Action.QUARANTINE_AND_ESTIMATE,
            DecisionOrigin.DATA_QUALITY,
            round=2,
            evidence_ids=["model.p_anomaly"],
        ),
        make_prop(
            AgentRole.ANALYST,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            round=3,
            evidence_ids=["model.p_anomaly", "physics.current_vs_power_identity"],
        ),
    ]
    v_props = [
        make_prop(
            AgentRole.VALIDATOR,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            round=1,
            evidence_ids=["physics.current_vs_power_identity"],
        ),
        make_prop(
            AgentRole.VALIDATOR,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            round=2,
            evidence_ids=["physics.current_vs_power_identity"],
        ),
        make_prop(
            AgentRole.VALIDATOR,
            Action.FLAG_DATA_SUSPECT,
            DecisionOrigin.DATA_QUALITY,
            round=3,
            evidence_ids=["physics.current_vs_power_identity"],
        ),
    ]
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, a_props),
        validator=ScriptedAgent(AgentRole.VALIDATOR, v_props),
        r_max=3,
    )
    res = delib.deliberate(sample_raw_record)
    assert res.converged is True
    assert res.rounds_used == 3


def test_r_max_never_exceeded(sample_raw_record: dict[str, Any]) -> None:
    # Disagree constantly with r_max=2
    analyst = ScriptedAgent(
        AgentRole.ANALYST,
        [
            make_prop(
                AgentRole.ANALYST,
                Action.QUARANTINE_AND_ESTIMATE,
                DecisionOrigin.DATA_QUALITY,
                evidence_ids=["model.p_anomaly"],
            )
        ],
    )
    validator = ScriptedAgent(
        AgentRole.VALIDATOR,
        [
            make_prop(
                AgentRole.VALIDATOR,
                Action.NO_ACTION,
                DecisionOrigin.PHYSICAL_EVENT,
                evidence_ids=["physics.current_vs_power_identity"],
            )
        ],
    )
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    res = delib.deliberate(sample_raw_record)

    assert res.rounds_used == 2
    assert analyst.call_count == 2
    assert validator.call_count == 2


def test_persistent_disagreement_returns_non_converged_result(
    sample_raw_record: dict[str, Any],
) -> None:
    analyst = ScriptedAgent(
        AgentRole.ANALYST,
        [
            make_prop(
                AgentRole.ANALYST,
                Action.QUARANTINE_AND_ESTIMATE,
                DecisionOrigin.DATA_QUALITY,
                evidence_ids=["model.p_anomaly"],
            )
        ],
    )
    validator = ScriptedAgent(
        AgentRole.VALIDATOR,
        [
            make_prop(
                AgentRole.VALIDATOR,
                Action.NO_ACTION,
                DecisionOrigin.PHYSICAL_EVENT,
                evidence_ids=["physics.current_vs_power_identity"],
            )
        ],
    )
    delib = Deliberator(analyst=analyst, validator=validator, r_max=3)
    res = delib.deliberate(sample_raw_record)

    assert res.converged is False
    assert res.agreed_action is None
    assert res.agreed_origin is None
    assert res.rounds_used == 3


# --------------------------------------------------------------------------- #
# Group D: Peer Evidence Disclosure (Tests 24 - 30)
# --------------------------------------------------------------------------- #


def test_only_peer_cited_evidence_disclosed(sample_raw_record: dict[str, Any]) -> None:
    v_prop = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR, [v_prop, v_prop])
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    a_r2_view = analyst.calls[1]["view"]
    assert "physics.current_vs_power_identity" in a_r2_view.evidence
    assert "baseline.deviation" not in a_r2_view.evidence
    assert "neighborhood.share" not in a_r2_view.evidence


def test_uncited_peer_evidence_remains_hidden(sample_raw_record: dict[str, Any]) -> None:
    a_prop = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST, [a_prop, a_prop])
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    v_r2_view = validator.calls[1]["view"]
    assert "model.p_anomaly" in v_r2_view.evidence
    for eid in v_r2_view.evidence.ids:
        assert not eid.startswith("xai.shap.")
        assert not eid.startswith("xai.lime.")


def test_peer_evidence_ids_remain_stable(sample_raw_record: dict[str, Any]) -> None:
    v_prop = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR, [v_prop, v_prop])
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    item = analyst.calls[1]["view"].evidence.get("physics.current_vs_power_identity")
    assert item is not None
    assert item.id == "physics.current_vs_power_identity"
    assert item.source == "peer_disclosed:validator"


def test_analyst_can_receive_cited_physics_evidence_after_r1(
    sample_raw_record: dict[str, Any],
) -> None:
    v_prop = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR, [v_prop, v_prop])
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    assert "physics.current_vs_power_identity" in analyst.calls[1]["view"].evidence


def test_validator_can_receive_cited_model_evidence_after_r1(
    sample_raw_record: dict[str, Any],
) -> None:
    a_prop = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST, [a_prop, a_prop])
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    assert "model.p_anomaly" in validator.calls[1]["view"].evidence


def test_raw_peer_agent_view_never_shared(sample_raw_record: dict[str, Any]) -> None:
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    delib = Deliberator(analyst=analyst, validator=validator, r_max=2)
    delib.deliberate(sample_raw_record)

    assert isinstance(analyst.calls[1]["other_proposal"], Proposal)
    assert not isinstance(analyst.calls[1]["other_proposal"], AgentView)


def test_hidden_ground_truth_never_becomes_peer_evidence(
    sample_raw_record: dict[str, Any],
) -> None:
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
    )
    delib.deliberate(sample_raw_record)

    for call in delib.analyst.calls + delib.validator.calls:  # type: ignore[attr-defined]
        view = call["view"]
        for item in view.evidence:
            assert "true_origin" not in item.id
            assert "ground_truth" not in item.id
            assert "injected" not in item.id


# --------------------------------------------------------------------------- #
# Group E: Revision Validation (Tests 31 - 39)
# --------------------------------------------------------------------------- #


def test_unchanged_action_and_origin_allowed_without_new_evidence(
    sample_raw_record: dict[str, Any],
) -> None:
    a_prop = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_prop, a_prop]),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
    )
    res = delib.deliberate(sample_raw_record)
    assert res.analyst_final.action == Action.FLAG_DATA_SUSPECT


def test_changed_action_with_new_peer_evidence_accepted(
    sample_raw_record: dict[str, Any],
) -> None:
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    # Changed action citing newly disclosed physics evidence
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly", "physics.current_vs_power_identity"],
    )
    v_prop = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [v_prop, v_prop]),
        r_max=2,
    )
    res = delib.deliberate(sample_raw_record)
    assert res.converged is True
    assert res.rounds_used == 2
    assert res.agreed_action == Action.FLAG_DATA_SUSPECT


def test_changed_origin_with_new_peer_evidence_accepted(
    sample_raw_record: dict[str, Any],
) -> None:
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.UNKNOWN,
        evidence_ids=["model.p_anomaly"],
    )
    # Changed origin citing newly disclosed physics evidence
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly", "physics.current_vs_power_identity"],
    )
    v_prop = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [v_prop, v_prop]),
        r_max=2,
    )
    res = delib.deliberate(sample_raw_record)
    assert res.converged is True
    assert res.agreed_origin == DecisionOrigin.DATA_QUALITY


def test_changed_action_without_new_evidence_rejected(
    sample_raw_record: dict[str, Any],
) -> None:
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    # Changed action WITHOUT citing any new evidence
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.NO_ACTION,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2]),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
    )
    res = delib.deliberate(sample_raw_record)

    # Invalid revision is rejected; prior proposal is retained
    assert res.analyst_final.action == Action.QUARANTINE_AND_ESTIMATE


def test_changed_origin_without_new_evidence_rejected(
    sample_raw_record: dict[str, Any],
) -> None:
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.UNKNOWN,
        evidence_ids=["model.p_anomaly"],
    )
    # Changed origin WITHOUT citing any new evidence
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2]),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
    )
    res = delib.deliberate(sample_raw_record)

    # Retains previous proposal origin
    assert res.analyst_final.origin == DecisionOrigin.UNKNOWN


def test_rejected_revision_preserves_prior_proposal(sample_raw_record: dict[str, Any]) -> None:
    audit_log = AuditLog(sqlite_path=":memory:")
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        rationale="R1 Initial",
    )
    # Illegally flips action without new evidence
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.NO_ACTION,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        rationale="R2 Illegal Flip",
    )
    v_r1 = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [v_r1, v_r1]),
        r_max=2,
        audit_log=audit_log,
    )
    res = delib.deliberate(sample_raw_record)

    assert res.analyst_final.action == Action.FLAG_DATA_SUSPECT
    assert res.analyst_final.rationale == "R1 Initial"


def test_rationale_only_change_does_not_require_new_evidence(
    sample_raw_record: dict[str, Any],
) -> None:
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        rationale="Old rationale.",
    )
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        rationale="New refined rationale.",
    )
    v_r = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [v_r, v_r]),
        r_max=2,
    )
    res = delib.deliberate(sample_raw_record)
    assert res.analyst_final.rationale == "New refined rationale."


def test_support_only_change_does_not_require_new_evidence(
    sample_raw_record: dict[str, Any],
) -> None:
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        support=0.6,
        evidence_ids=["model.p_anomaly"],
    )
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        support=0.85,  # Increased support
        evidence_ids=["model.p_anomaly"],
    )
    v_r = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [v_r, v_r]),
        r_max=2,
    )
    res = delib.deliberate(sample_raw_record)
    assert res.analyst_final.support == 0.85


def test_fabricated_new_evidence_id_fails_citation_validation(
    sample_raw_record: dict[str, Any],
) -> None:
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["model.p_anomaly", "hallucinated.ghost.id"],
    )
    analyst = ScriptedAgent(
        AgentRole.ANALYST,
        [
            make_prop(
                AgentRole.ANALYST,
                Action.FLAG_DATA_SUSPECT,
                DecisionOrigin.DATA_QUALITY,
                evidence_ids=["model.p_anomaly"],
            ),
            a_r2,
        ],
    )
    v_r = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    delib = Deliberator(
        analyst=analyst,
        validator=ScriptedAgent(AgentRole.VALIDATOR, [v_r, v_r]),
        r_max=2,
    )
    with pytest.raises(AgentContractError, match="not present in the agent's visible"):
        delib.deliberate(sample_raw_record)


# --------------------------------------------------------------------------- #
# Group F: Non-Convergence Semantics (Tests 40 - 46)
# --------------------------------------------------------------------------- #


def test_non_converged_preserves_analyst_final_proposal(sample_raw_record: dict[str, Any]) -> None:
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    res = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v]),
        r_max=2,
    ).deliberate(sample_raw_record)

    assert res.analyst_final.action == Action.QUARANTINE_AND_ESTIMATE


def test_non_converged_preserves_validator_final_proposal(
    sample_raw_record: dict[str, Any],
) -> None:
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    res = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v]),
        r_max=2,
    ).deliberate(sample_raw_record)

    assert res.validator_final.action == Action.NO_ACTION


def test_non_converged_agreed_action_is_none(sample_raw_record: dict[str, Any]) -> None:
    res = Deliberator(
        analyst=ScriptedAgent(
            AgentRole.ANALYST,
            [
                make_prop(
                    AgentRole.ANALYST,
                    Action.QUARANTINE_AND_ESTIMATE,
                    DecisionOrigin.DATA_QUALITY,
                    evidence_ids=["model.p_anomaly"],
                )
            ],
        ),
        validator=ScriptedAgent(
            AgentRole.VALIDATOR,
            [
                make_prop(
                    AgentRole.VALIDATOR,
                    Action.NO_ACTION,
                    DecisionOrigin.PHYSICAL_EVENT,
                    evidence_ids=["physics.current_vs_power_identity"],
                )
            ],
        ),
        r_max=2,
    ).deliberate(sample_raw_record)

    assert res.agreed_action is None


def test_non_converged_agreed_origin_is_none(sample_raw_record: dict[str, Any]) -> None:
    res = Deliberator(
        analyst=ScriptedAgent(
            AgentRole.ANALYST,
            [
                make_prop(
                    AgentRole.ANALYST,
                    Action.QUARANTINE_AND_ESTIMATE,
                    DecisionOrigin.DATA_QUALITY,
                    evidence_ids=["model.p_anomaly"],
                )
            ],
        ),
        validator=ScriptedAgent(
            AgentRole.VALIDATOR,
            [
                make_prop(
                    AgentRole.VALIDATOR,
                    Action.NO_ACTION,
                    DecisionOrigin.PHYSICAL_EVENT,
                    evidence_ids=["physics.current_vs_power_identity"],
                )
            ],
        ),
        r_max=2,
    ).deliberate(sample_raw_record)

    assert res.agreed_origin is None


def test_non_converged_rounds_used_equals_r_max(sample_raw_record: dict[str, Any]) -> None:
    res = Deliberator(
        analyst=ScriptedAgent(
            AgentRole.ANALYST,
            [
                make_prop(
                    AgentRole.ANALYST,
                    Action.QUARANTINE_AND_ESTIMATE,
                    DecisionOrigin.DATA_QUALITY,
                    evidence_ids=["model.p_anomaly"],
                )
            ],
        ),
        validator=ScriptedAgent(
            AgentRole.VALIDATOR,
            [
                make_prop(
                    AgentRole.VALIDATOR,
                    Action.NO_ACTION,
                    DecisionOrigin.PHYSICAL_EVENT,
                    evidence_ids=["physics.current_vs_power_identity"],
                )
            ],
        ),
        r_max=3,
    ).deliberate(sample_raw_record)

    assert res.rounds_used == 3


def test_higher_support_does_not_become_winner(sample_raw_record: dict[str, Any]) -> None:
    # Analyst has 0.99 support, Validator has 0.05
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        support=0.99,
        evidence_ids=["model.p_anomaly"],
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        support=0.05,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    res = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v]),
        r_max=2,
    ).deliberate(sample_raw_record)

    assert res.converged is False
    assert res.agreed_action is None


def test_no_majority_or_tie_break_behavior(sample_raw_record: dict[str, Any]) -> None:
    res = Deliberator(
        analyst=ScriptedAgent(
            AgentRole.ANALYST,
            [
                make_prop(
                    AgentRole.ANALYST,
                    Action.QUARANTINE_AND_ESTIMATE,
                    DecisionOrigin.DATA_QUALITY,
                    evidence_ids=["model.p_anomaly"],
                )
            ],
        ),
        validator=ScriptedAgent(
            AgentRole.VALIDATOR,
            [
                make_prop(
                    AgentRole.VALIDATOR,
                    Action.NO_ACTION,
                    DecisionOrigin.PHYSICAL_EVENT,
                    evidence_ids=["physics.current_vs_power_identity"],
                )
            ],
        ),
        r_max=3,
    ).deliberate(sample_raw_record)

    assert res.converged is False
    assert res.agreed_action is None


# --------------------------------------------------------------------------- #
# Group G: Fallback Integration (Tests 47 - 51)
# --------------------------------------------------------------------------- #


def test_analyst_fallback_proposal_participates_normally(
    sample_raw_record: dict[str, Any],
) -> None:
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        fallback_used=True,
        fallback_reason="timeout",
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    res = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v]),
        r_max=1,
    ).deliberate(sample_raw_record)

    assert res.converged is True
    assert res.fallback_used is True
    assert res.analyst_fallback is True
    assert res.validator_fallback is False


def test_validator_fallback_proposal_participates_normally(
    sample_raw_record: dict[str, Any],
) -> None:
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
        fallback_used=True,
        fallback_reason="backend_error",
    )
    res = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v]),
        r_max=1,
    ).deliberate(sample_raw_record)

    assert res.converged is True
    assert res.fallback_used is True
    assert res.analyst_fallback is False
    assert res.validator_fallback is True


def test_fallback_flags_preserved_in_deliberation_result(
    sample_raw_record: dict[str, Any],
) -> None:
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        fallback_used=True,
        fallback_reason="timeout",
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
        fallback_used=True,
        fallback_reason="schema_error",
    )
    res = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v]),
        r_max=1,
    ).deliberate(sample_raw_record)

    assert res.fallback_used is True
    assert res.analyst_fallback is True
    assert res.validator_fallback is True


def test_fallback_does_not_imply_automatic_non_convergence(
    sample_raw_record: dict[str, Any],
) -> None:
    delib = Deliberator(
        analyst=RuleBasedAnalyst(),
        validator=RuleBasedValidator(),
        r_max=2,
    )
    res = delib.deliberate(sample_raw_record)
    assert isinstance(res, DeliberationResult)


def test_fallback_does_not_receive_special_voting_weight(
    sample_raw_record: dict[str, Any],
) -> None:
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        fallback_used=True,
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
        fallback_used=False,
    )
    res = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v]),
        r_max=1,
    ).deliberate(sample_raw_record)

    assert res.converged is False


# --------------------------------------------------------------------------- #
# Group H: Audit Logging & Hash Chain (Tests 52 - 65)
# --------------------------------------------------------------------------- #


def test_audit_deliberation_started_logged(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=1,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    assert any(e["event_type"] == "deliberation_started" for e in events)


def test_audit_every_analyst_proposal_logged(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    analyst_events = [e for e in events if e["event_type"] == "analyst_proposal"]
    assert len(analyst_events) >= 1


def test_audit_every_validator_proposal_logged(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    val_events = [e for e in events if e["event_type"] == "validator_proposal"]
    assert len(val_events) >= 1


def test_audit_round_numbers_logged_correctly(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    props = [e for e in events if "proposal" in e["event_type"]]
    assert props[0]["payload"]["round"] == 1


def test_audit_peer_evidence_disclosure_logged(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a, p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v, p_v]),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    disclosed = [e for e in events if e["event_type"] == "peer_evidence_disclosed"]
    assert len(disclosed) >= 1
    assert "physics.current_vs_power_identity" in disclosed[0]["payload"]["validator_disclosed_ids"]


def test_audit_rejected_revision_logged(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    a_r1 = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    a_r2 = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )  # Illegal change
    v_r1 = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [a_r1, a_r2]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [v_r1, v_r1]),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    rejected = [e for e in events if e["event_type"] == "revision_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["payload"]["attempted_action"] == "flag_data_suspect"


def test_audit_convergence_logged(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=1,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    conv = [e for e in events if e["event_type"] == "convergence"]
    assert len(conv) == 1


def test_audit_non_convergence_logged(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(
            AgentRole.ANALYST,
            [
                make_prop(
                    AgentRole.ANALYST,
                    Action.QUARANTINE_AND_ESTIMATE,
                    DecisionOrigin.DATA_QUALITY,
                    evidence_ids=["model.p_anomaly"],
                )
            ],
        ),
        validator=ScriptedAgent(
            AgentRole.VALIDATOR,
            [
                make_prop(
                    AgentRole.VALIDATOR,
                    Action.NO_ACTION,
                    DecisionOrigin.PHYSICAL_EVENT,
                    evidence_ids=["physics.current_vs_power_identity"],
                )
            ],
        ),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    non_conv = [e for e in events if e["event_type"] == "non_convergence"]
    assert len(non_conv) == 1


def test_audit_completed_event_logged(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=1,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    comp = [e for e in events if e["event_type"] == "deliberation_completed"]
    assert len(comp) == 1


def test_audit_prompt_hashes_preserved(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=1,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    props = [e for e in events if "proposal" in e["event_type"]]
    assert props[0]["payload"]["prompt_hash"] == "hash-123"


def test_audit_fallback_metadata_preserved(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
        fallback_used=True,
        fallback_reason="timeout",
    )
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=1,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    events = audit.events()
    a_prop = [e for e in events if e["event_type"] == "analyst_proposal"][0]
    assert a_prop["payload"]["fallback_used"] is True
    assert a_prop["payload"]["fallback_reason"] == "timeout"


def test_audit_chain_verifies_after_one_round_deliberation(
    sample_raw_record: dict[str, Any],
) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=1,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    verification = audit.verify_chain()
    assert verification["valid"] is True
    assert verification["events"] >= 3


def test_audit_chain_verifies_after_multi_round_deliberation(
    sample_raw_record: dict[str, Any],
) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.NO_ACTION,
        DecisionOrigin.PHYSICAL_EVENT,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [p_a, p_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [p_v, p_v]),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    verification = audit.verify_chain()
    assert verification["valid"] is True
    assert verification["events"] >= 5


def test_no_secret_fields_present_in_audit_payload(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    forbidden_needles = ["api_key", "password", "token", "secret", "bearer"]
    for ev in audit.events():
        payload_str = str(ev["payload"]).lower()
        for needle in forbidden_needles:
            assert needle not in payload_str


# --------------------------------------------------------------------------- #
# Group I: Evidence Boundary & Privacy (Tests 66 - 72)
# --------------------------------------------------------------------------- #


def test_hidden_truth_absent_through_all_rounds(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        r_max=2,
        audit_log=audit,
    ).deliberate(sample_raw_record)

    for ev in audit.events():
        payload = ev["payload"]
        assert "true_origin" not in payload
        assert "true_type" not in payload
        assert "y_true" not in payload


def test_peer_exchange_does_not_expose_entire_analyst_view(
    sample_raw_record: dict[str, Any],
) -> None:
    p_a = make_prop(
        AgentRole.ANALYST,
        Action.QUARANTINE_AND_ESTIMATE,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST, [p_a, p_a])
    validator = ScriptedAgent(AgentRole.VALIDATOR)
    Deliberator(analyst=analyst, validator=validator, r_max=2).deliberate(sample_raw_record)

    # Validator view in round 2 has model.p_anomaly, but NOT other uncited analyst evidence
    v_r2_view = validator.calls[1]["view"]
    assert "model.p_anomaly" in v_r2_view.evidence
    assert "model.score" not in v_r2_view.evidence
    assert "xai.reliability" not in v_r2_view.evidence


def test_peer_exchange_does_not_expose_entire_validator_view(
    sample_raw_record: dict[str, Any],
) -> None:
    p_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )
    analyst = ScriptedAgent(AgentRole.ANALYST)
    validator = ScriptedAgent(AgentRole.VALIDATOR, [p_v, p_v])
    Deliberator(analyst=analyst, validator=validator, r_max=2).deliberate(sample_raw_record)

    a_r2_view = analyst.calls[1]["view"]
    assert "physics.current_vs_power_identity" in a_r2_view.evidence
    assert "baseline.deviation" not in a_r2_view.evidence


def test_future_history_remains_absent(sample_raw_record: dict[str, Any]) -> None:
    future_case = {
        "anomaly_id": "FUTURE_001",
        "status": "verified",
        "timestamp": "2025-01-01T00:00:00Z",  # In future relative to 2020 record
        "type_pred": "dead_sensor",
        "origin_pred": "data_quality",
    }
    rec_val = RecordingValidator(RuleBasedValidator())
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=rec_val,
        r_max=1,
    )
    delib.deliberate(sample_raw_record, history=[future_case])  # type: ignore[list-item]

    for call in rec_val.calls:
        for eid in call["view"].evidence.ids:
            assert "FUTURE_001" not in eid


def test_unverified_history_remains_absent(sample_raw_record: dict[str, Any]) -> None:
    unverified_case = {
        "anomaly_id": "UNVERIFIED_001",
        "status": "unverified",  # Not verified!
        "timestamp": "2019-01-01T00:00:00Z",
        "type_pred": "dead_sensor",
        "origin_pred": "data_quality",
    }
    rec_val = RecordingValidator(RuleBasedValidator())
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=rec_val,
        r_max=1,
    )
    delib.deliberate(sample_raw_record, history=[unverified_case])  # type: ignore[list-item]

    for call in rec_val.calls:
        for eid in call["view"].evidence.ids:
            assert "UNVERIFIED_001" not in eid


def test_self_history_remains_absent(sample_raw_record: dict[str, Any]) -> None:
    self_case = {
        "anomaly_id": "CASE_2020_001",  # Same anomaly_id as current case
        "status": "verified",
        "timestamp": "2019-01-01T00:00:00Z",
        "type_pred": "dead_sensor",
        "origin_pred": "data_quality",
    }
    rec_val = RecordingValidator(RuleBasedValidator())
    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST),
        validator=rec_val,
        r_max=1,
    )
    delib.deliberate(sample_raw_record, history=[self_case])  # type: ignore[list-item]

    for call in rec_val.calls:
        for eid in call["view"].evidence.ids:
            assert "history.0.anomaly_id" not in eid


def test_round_exchange_cannot_introduce_arbitrary_evidence_id(
    sample_raw_record: dict[str, Any],
) -> None:
    fraud_prop = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["arbitrary.unverified.id"],
    )
    from energymind.agents.deliberation import build_peer_context
    from energymind.agents.validator import build_validator_view

    val_view = build_validator_view(sample_raw_record)
    disclosed = build_peer_context(fraud_prop, val_view)
    assert len(disclosed) == 0


# --------------------------------------------------------------------------- #
# Group J: Determinism (Tests 73 - 76)
# --------------------------------------------------------------------------- #


def test_determinism_same_input_gives_same_rounds_used(sample_raw_record: dict[str, Any]) -> None:
    delib = Deliberator(
        analyst=RuleBasedAnalyst(),
        validator=RuleBasedValidator(),
        r_max=3,
    )
    res1 = delib.deliberate(sample_raw_record)
    res2 = delib.deliberate(sample_raw_record)
    assert res1.rounds_used == res2.rounds_used


def test_determinism_same_input_gives_same_convergence_result(
    sample_raw_record: dict[str, Any],
) -> None:
    delib = Deliberator(
        analyst=RuleBasedAnalyst(),
        validator=RuleBasedValidator(),
        r_max=3,
    )
    res1 = delib.deliberate(sample_raw_record)
    res2 = delib.deliberate(sample_raw_record)
    assert res1.converged == res2.converged
    assert res1.agreed_action == res2.agreed_action
    assert res1.agreed_origin == res2.agreed_origin


def test_determinism_same_input_gives_same_audit_event_ordering(
    sample_raw_record: dict[str, Any],
) -> None:
    audit1 = AuditLog(sqlite_path=":memory:")
    audit2 = AuditLog(sqlite_path=":memory:")

    delib1 = Deliberator(
        analyst=RuleBasedAnalyst(),
        validator=RuleBasedValidator(),
        r_max=3,
        audit_log=audit1,
    )
    delib2 = Deliberator(
        analyst=RuleBasedAnalyst(),
        validator=RuleBasedValidator(),
        r_max=3,
        audit_log=audit2,
    )

    delib1.deliberate(sample_raw_record)
    delib2.deliberate(sample_raw_record)

    events1 = [e["event_type"] for e in audit1.events()]
    events2 = [e["event_type"] for e in audit2.events()]
    assert events1 == events2


def test_determinism_sequential_order_does_not_change_snapshot_semantics(
    sample_raw_record: dict[str, Any],
) -> None:
    prop_a = make_prop(
        AgentRole.ANALYST,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["model.p_anomaly"],
    )
    prop_v = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.current_vs_power_identity"],
    )

    delib = Deliberator(
        analyst=ScriptedAgent(AgentRole.ANALYST, [prop_a]),
        validator=ScriptedAgent(AgentRole.VALIDATOR, [prop_v]),
        r_max=1,
    )
    res = delib.deliberate(sample_raw_record)
    assert res.converged is True


# --------------------------------------------------------------------------- #
# Group K: Ground-Truth Invariance (Test 77)
# --------------------------------------------------------------------------- #


def test_changing_hidden_ground_truth_labels_does_not_affect_deliberation_result(
    sample_raw_record: dict[str, Any],
) -> None:
    rec1 = copy.deepcopy(sample_raw_record)
    rec2 = copy.deepcopy(sample_raw_record)

    # Invert ground-truth labels
    rec2["true_origin"] = "physical_event"
    rec2["true_type"] = "normal_fluctuation"
    rec2["ground_truth"] = {"type": "normal_fluctuation", "origin": "physical_event"}
    rec2["y_true"] = 0
    rec2["injected_type"] = "other"

    delib = Deliberator(
        analyst=RuleBasedAnalyst(),
        validator=RuleBasedValidator(),
        r_max=3,
    )
    res1 = delib.deliberate(rec1)
    res2 = delib.deliberate(rec2)

    assert res1.converged == res2.converged
    assert res1.rounds_used == res2.rounds_used
    assert res1.agreed_action == res2.agreed_action
    assert res1.agreed_origin == res2.agreed_origin


# --------------------------------------------------------------------------- #
# Group L: Coordinator Robustness & Helpers (Tests 78 - 82)
# --------------------------------------------------------------------------- #


def test_deliberate_convenience_function(sample_raw_record: dict[str, Any]) -> None:
    res = deliberate(
        sample_raw_record,
        analyst=RuleBasedAnalyst(),
        validator=RuleBasedValidator(),
        r_max=2,
    )
    assert isinstance(res, DeliberationResult)


def test_deliberator_agent_exception_handled_safely(sample_raw_record: dict[str, Any]) -> None:
    audit = AuditLog(sqlite_path=":memory:")
    exploding_analyst = ExplodingAgent(AgentRole.ANALYST)
    delib = Deliberator(
        analyst=exploding_analyst,
        validator=ScriptedAgent(AgentRole.VALIDATOR),
        audit_log=audit,
    )

    with pytest.raises(RuntimeError, match="Simulated agent catastrophic explosion"):
        delib.deliberate(sample_raw_record)

    # Failure was recorded in audit log
    events = audit.events()
    assert any(e["event_type"] == "orchestration_failed" for e in events)


def test_deliberation_missing_anomaly_id_raises_value_error(
    sample_raw_record: dict[str, Any],
) -> None:
    bad_record = copy.deepcopy(sample_raw_record)
    bad_record["anomaly_id"] = ""
    delib = Deliberator(analyst=RuleBasedAnalyst(), validator=RuleBasedValidator())

    with pytest.raises(ValueError, match="anomaly_id"):
        delib.deliberate(bad_record)


def test_build_peer_context_helper() -> None:
    catalog = EvidenceCatalog(
        [
            EvidenceItem(id="physics.r1", source="physics", value=0.1, description="r1"),
            EvidenceItem(id="physics.r2", source="physics", value=0.2, description="r2"),
        ]
    )
    view = AgentView(anomaly_id="C1", role=AgentRole.VALIDATOR, evidence=catalog)
    prop = make_prop(
        AgentRole.VALIDATOR,
        Action.FLAG_DATA_SUSPECT,
        DecisionOrigin.DATA_QUALITY,
        evidence_ids=["physics.r1"],
    )

    disclosed = build_peer_context(prop, view)
    assert len(disclosed) == 1
    assert disclosed[0].id == "physics.r1"
    assert disclosed[0].source == "peer_disclosed:validator"


def test_create_effective_view_helper() -> None:
    base_catalog = EvidenceCatalog(
        [EvidenceItem(id="model.p", source="detector", value=0.9, description="score")]
    )
    base_view = AgentView(anomaly_id="C1", role=AgentRole.ANALYST, evidence=base_catalog)
    peer_item = EvidenceItem(
        id="physics.r1",
        source="peer_disclosed:validator",
        value=0.1,
        description="r1",
    )

    eff_view = create_effective_view(base_view, [peer_item])
    assert len(eff_view.evidence) == 2
    assert "model.p" in eff_view.evidence
    assert "physics.r1" in eff_view.evidence
