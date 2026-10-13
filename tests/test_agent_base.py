"""Unit tests for M7.1 agent contract layer (src/energymind/agents/base.py).

Tests run completely offline without dataset, LLM, or database dependencies.
"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from energymind.agents.base import (
    Action,
    Agent,
    AgentContractError,
    AgentRole,
    AgentView,
    CaseMetadata,
    DecisionOrigin,
    DeliberationResult,
    EvidenceCatalog,
    EvidenceItem,
    InvalidEvidenceError,
    InvalidRevisionError,
    Proposal,
    ProposalOut,
    UnknownEvidenceCitationError,
    prompt_hash,
    proposals_agree,
    validate_citations,
    validate_revision,
)
from energymind.config import Origin as InjectionOrigin


def test_exception_hierarchy():
    """Verify that domain exceptions subclass AgentContractError and ValueError."""
    assert issubclass(AgentContractError, ValueError)
    assert issubclass(InvalidEvidenceError, AgentContractError)
    assert issubclass(UnknownEvidenceCitationError, AgentContractError)
    assert issubclass(InvalidRevisionError, AgentContractError)


# --------------------------------------------------------------------------- #
# Fixtures & Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_evidence_items() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="model.p_anomaly",
            source="detector.ensemble",
            description="Ensemble anomaly probability",
            value=0.94,
        ),
        EvidenceItem(
            id="physics.current_vs_power_identity",
            source="physics.residuals",
            description="V*I vs S electrical identity residual",
            value=0.0012,
            unit="relative",
            status="valid",
        ),
        EvidenceItem(
            id="neighborhood.share",
            source="neighborhood.aggregator",
            description="Share of neighboring meters seeing similar excursion",
            value=0.08,
            unit="ratio",
        ),
    ]


@pytest.fixture
def sample_catalog(sample_evidence_items: list[EvidenceItem]) -> EvidenceCatalog:
    return EvidenceCatalog(sample_evidence_items)


@pytest.fixture
def sample_agent_view(sample_catalog: EvidenceCatalog) -> AgentView:
    return AgentView(
        anomaly_id="CASE_2020_001",
        role=AgentRole.ANALYST,
        evidence=sample_catalog,
        case_metadata=CaseMetadata(
            household_id="HH_00072",
            t_start="2020-06-01T12:00:00Z",
            t_end="2020-06-01T14:00:00Z",
            duration_h=2.0,
        ),
    )


def make_proposal_out(
    action: Action = Action.FLAG_DATA_SUSPECT,
    origin: DecisionOrigin = DecisionOrigin.DATA_QUALITY,
    support: float = 0.85,
    evidence_ids: list[str] | None = None,
    rationale: str = "High anomaly score and single-household excursion.",
    objection_to_other: str | None = None,
    changed_since_last_round: bool = False,
) -> ProposalOut:
    return ProposalOut(
        action=action,
        origin=origin,
        support=support,
        evidence_ids=["model.p_anomaly"] if evidence_ids is None else evidence_ids,
        rationale=rationale,
        objection_to_other=objection_to_other,
        changed_since_last_round=changed_since_last_round,
    )


def make_proposal(
    agent: AgentRole = AgentRole.ANALYST,
    round_num: int = 1,
    action: Action = Action.FLAG_DATA_SUSPECT,
    origin: DecisionOrigin = DecisionOrigin.DATA_QUALITY,
    support: float = 0.85,
    evidence_ids: list[str] | None = None,
    rationale: str = "High anomaly score and single-household excursion.",
    objection_to_other: str | None = None,
    changed_since_last_round: bool = False,
    backend: str = "ollama",
    model: str = "qwen3.5:9b",
    prompt_hash_val: str = "a" * 64,
    latency_ms: float = 120.0,
    fallback_used: bool = False,
) -> Proposal:
    return Proposal(
        agent=agent,
        round=round_num,
        action=action,
        origin=origin,
        support=support,
        evidence_ids=["model.p_anomaly"] if evidence_ids is None else evidence_ids,
        rationale=rationale,
        objection_to_other=objection_to_other,
        changed_since_last_round=changed_since_last_round,
        backend=backend,
        model=model,
        prompt_hash=prompt_hash_val,
        latency_ms=latency_ms,
        fallback_used=fallback_used,
    )


# --------------------------------------------------------------------------- #
# Test 1 & 2: DecisionOrigin
# --------------------------------------------------------------------------- #


def test_01_valid_decision_origin_accepts_all_four_values():
    """1. valid DecisionOrigin accepts all four values."""
    expected = {"data_quality", "physical_event", "adversarial", "unknown"}
    actual = {item.value for item in DecisionOrigin}
    assert actual == expected

    for val in ("data_quality", "physical_event", "adversarial", "unknown"):
        origin = DecisionOrigin(val)
        assert origin == val
        assert origin.value == val


def test_02_invalid_decision_origin_rejected():
    """2. invalid DecisionOrigin rejected."""
    with pytest.raises(ValueError):
        DecisionOrigin("sensor_defect")

    with pytest.raises(ValidationError):
        ProposalOut(
            action=Action.NO_ACTION,
            origin="invalid_origin",  # type: ignore[arg-type]
            support=0.5,
            evidence_ids=["id1"],
            rationale="Test rationale",
        )


# --------------------------------------------------------------------------- #
# Test 3 & 4: EvidenceItem
# --------------------------------------------------------------------------- #


def test_03_evidence_item_blank_id_rejected():
    """3. EvidenceItem blank ID rejected."""
    with pytest.raises(ValidationError):
        EvidenceItem(id="", source="src", description="desc", value=1.0)

    with pytest.raises(ValidationError):
        EvidenceItem(id="   ", source="src", description="desc", value=1.0)


def test_04_evidence_item_blank_source_rejected():
    """4. EvidenceItem blank source rejected."""
    with pytest.raises(ValidationError):
        EvidenceItem(id="item.1", source="", description="desc", value=1.0)

    with pytest.raises(ValidationError):
        EvidenceItem(id="item.1", source="   ", description="desc", value=1.0)


def test_evidence_item_blank_description_rejected():
    """EvidenceItem blank description rejected."""
    with pytest.raises(ValidationError):
        EvidenceItem(id="item.1", source="src", description="", value=1.0)


def test_evidence_item_non_serializable_value_rejected():
    """EvidenceItem rejects arbitrary non-serializable objects."""

    class CustomObject:
        pass

    with pytest.raises(ValidationError):
        EvidenceItem(id="item.1", source="src", description="desc", value=CustomObject())


# --------------------------------------------------------------------------- #
# Test 5 & 6: EvidenceCatalog
# --------------------------------------------------------------------------- #


def test_05_duplicate_evidence_catalog_ids_rejected(sample_evidence_items: list[EvidenceItem]):
    """5. duplicate EvidenceCatalog IDs rejected."""
    dup_item = EvidenceItem(
        id="model.p_anomaly",
        source="other_source",
        description="Duplicate ID with different content",
        value=0.5,
    )
    items_with_dup = [*sample_evidence_items, dup_item]

    with pytest.raises((InvalidEvidenceError, ValidationError)):
        EvidenceCatalog(items_with_dup)


def test_06_evidence_lookup_works(sample_catalog: EvidenceCatalog):
    """6. evidence lookup works."""
    # Membership test by string
    assert "model.p_anomaly" in sample_catalog
    assert "physics.current_vs_power_identity" in sample_catalog
    assert "unknown.id" not in sample_catalog

    # Retrieval by index / key
    item = sample_catalog["model.p_anomaly"]
    assert item.id == "model.p_anomaly"
    assert item.value == 0.94

    with pytest.raises(KeyError):
        _ = sample_catalog["nonexistent.id"]

    # Retrieval by .get()
    assert sample_catalog.get("neighborhood.share") is not None
    assert sample_catalog.get("nonexistent.id") is None
    assert sample_catalog.get("nonexistent.id", default=42) == 42

    # Length and IDs set
    assert len(sample_catalog) == 3
    assert "neighborhood.share" in sample_catalog.ids


# --------------------------------------------------------------------------- #
# Test 7: AgentView
# --------------------------------------------------------------------------- #


def test_07_agent_view_can_contain_only_controlled_evidence(sample_catalog: EvidenceCatalog):
    """7. AgentView can contain only controlled evidence."""
    view = AgentView(
        anomaly_id="CASE_001",
        role=AgentRole.VALIDATOR,
        evidence=sample_catalog,
    )
    assert view.anomaly_id == "CASE_001"
    assert view.role == AgentRole.VALIDATOR
    assert len(view.evidence) == 3

    # Attempting to assign forbidden or arbitrary attributes fails (frozen + extra forbid)
    with pytest.raises(ValidationError):
        AgentView(
            anomaly_id="CASE_001",
            role=AgentRole.VALIDATOR,
            evidence=sample_catalog,
            record={"some": "record"},  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------- #
# Test 8, 9, 10, 11, 12, 13: ProposalOut
# --------------------------------------------------------------------------- #


def test_08_proposal_out_support_less_than_zero_rejected():
    """8. ProposalOut support < 0 rejected."""
    with pytest.raises(ValidationError):
        make_proposal_out(support=-0.01)


def test_09_proposal_out_support_greater_than_one_rejected():
    """9. ProposalOut support > 1 rejected."""
    with pytest.raises(ValidationError):
        make_proposal_out(support=1.01)


def test_10_proposal_out_empty_evidence_list_rejected():
    """10. ProposalOut empty evidence list rejected."""
    with pytest.raises((ValidationError, InvalidEvidenceError)):
        make_proposal_out(evidence_ids=[])


def test_11_invalid_action_rejected():
    """11. invalid action rejected."""
    with pytest.raises(ValidationError):
        make_proposal_out(action="remote_disconnect")  # type: ignore[arg-type]


def test_12_invalid_origin_rejected():
    """12. invalid origin rejected."""
    with pytest.raises(ValidationError):
        make_proposal_out(origin="unspecified")  # type: ignore[arg-type]


def test_13_proposal_does_not_trust_llm_runtime_metadata():
    """13. Proposal does not trust LLM runtime metadata because ProposalOut cannot provide it."""
    # ProposalOut cannot carry runtime metadata fields
    with pytest.raises(ValidationError):
        ProposalOut(
            action=Action.NO_ACTION,
            origin=DecisionOrigin.DATA_QUALITY,
            support=0.5,
            evidence_ids=["ev1"],
            rationale="Rationale",
            backend="ollama",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        ProposalOut(
            action=Action.NO_ACTION,
            origin=DecisionOrigin.DATA_QUALITY,
            support=0.5,
            evidence_ids=["ev1"],
            rationale="Rationale",
            round=1,  # type: ignore[call-arg]
        )

    # But Proposal requires authoritative metadata
    p_out = make_proposal_out(support=0.8)
    trusted_p = Proposal.from_proposal_out(
        p_out,
        agent=AgentRole.ANALYST,
        round=1,
        backend="ollama",
        model="qwen3.5:9b",
        prompt_hash="f" * 64,
        latency_ms=88.5,
    )
    assert trusted_p.agent == AgentRole.ANALYST
    assert trusted_p.round == 1
    assert trusted_p.backend == "ollama"
    assert trusted_p.latency_ms == 88.5


# --------------------------------------------------------------------------- #
# Test 14 & 15: Proposal Validation
# --------------------------------------------------------------------------- #


def test_14_proposal_round_zero_rejected():
    """14. Proposal round 0 rejected."""
    with pytest.raises(ValidationError):
        make_proposal(round_num=0)


def test_15_proposal_negative_latency_rejected():
    """15. Proposal negative latency rejected."""
    with pytest.raises(ValidationError):
        make_proposal(latency_ms=-1.0)


# --------------------------------------------------------------------------- #
# Test 16, 17, 18, 19: validate_citations
# --------------------------------------------------------------------------- #


def test_16_validate_citations_accepts_known_ids(
    sample_agent_view: AgentView, sample_catalog: EvidenceCatalog
):
    """16. validate_citations accepts known IDs."""
    prop = make_proposal(evidence_ids=["model.p_anomaly", "neighborhood.share"])
    # Validate against AgentView
    validate_citations(prop, sample_agent_view)
    # Validate against EvidenceCatalog directly
    validate_citations(prop, sample_catalog)


def test_17_validate_citations_rejects_unknown_ids(sample_agent_view: AgentView):
    """17. validate_citations rejects unknown IDs."""
    prop = make_proposal(evidence_ids=["model.p_anomaly", "hallucinated.id"])
    with pytest.raises(UnknownEvidenceCitationError):
        validate_citations(prop, sample_agent_view)


def test_18_validate_citations_rejects_empty_ids(sample_agent_view: AgentView):
    """18. validate_citations rejects empty IDs."""
    # ProposalOut construction itself blocks empty, but test the citation validator specifically
    p_out = make_proposal_out(evidence_ids=["model.p_anomaly"])
    object.__setattr__(p_out, "evidence_ids", [])
    with pytest.raises(InvalidEvidenceError):
        validate_citations(p_out, sample_agent_view)


def test_19_validate_citations_handles_duplicate_citations(sample_agent_view: AgentView):
    """19. validate_citations handles duplicate citations according to chosen policy (rejects)."""
    p_out = make_proposal_out(evidence_ids=["model.p_anomaly"])
    object.__setattr__(p_out, "evidence_ids", ["model.p_anomaly", "model.p_anomaly"])
    with pytest.raises(InvalidEvidenceError):
        validate_citations(p_out, sample_agent_view)


# --------------------------------------------------------------------------- #
# Test 20, 21, 22: proposals_agree
# --------------------------------------------------------------------------- #


def test_20_proposals_agree_true_when_action_and_origin_match():
    """20. proposals_agree true when action+origin match."""
    p1 = make_proposal(
        agent=AgentRole.ANALYST,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
        support=0.9,
    )
    p2 = make_proposal(
        agent=AgentRole.VALIDATOR,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
        support=0.6,  # support can differ
    )
    assert proposals_agree(p1, p2) is True


def test_21_proposals_agree_false_when_action_differs():
    """21. proposals_agree false when action differs."""
    p1 = make_proposal(
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
    )
    p2 = make_proposal(
        action=Action.QUARANTINE_AND_ESTIMATE,
        origin=DecisionOrigin.DATA_QUALITY,
    )
    assert proposals_agree(p1, p2) is False


def test_22_proposals_agree_false_when_origin_differs():
    """22. proposals_agree false when origin differs."""
    p1 = make_proposal(
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
    )
    p2 = make_proposal(
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.PHYSICAL_EVENT,
    )
    assert proposals_agree(p1, p2) is False


# --------------------------------------------------------------------------- #
# Test 23, 24, 25, 26, 27: validate_revision
# --------------------------------------------------------------------------- #


def test_23_validate_revision_accepts_initial_proposal():
    """23. validate_revision accepts initial proposal."""
    p1 = make_proposal(round_num=1)
    validate_revision(p1, previous_proposal=None)


def test_24_validate_revision_accepts_unchanged_decision():
    """24. validate_revision accepts unchanged decision."""
    p1 = make_proposal(
        round_num=1,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["id1"],
    )
    p2 = make_proposal(
        round_num=2,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["id1"],  # unchanged decision needs no new evidence
    )
    validate_revision(p2, p1)


def test_25_validate_revision_accepts_changed_decision_with_genuinely_new_evidence():
    """25. validate_revision accepts changed decision with genuinely new evidence."""
    p1 = make_proposal(
        round_num=1,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["id1"],
    )
    p2 = make_proposal(
        round_num=2,
        action=Action.QUARANTINE_AND_ESTIMATE,
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["id1", "id2"],  # "id2" is genuinely new
    )
    validate_revision(p2, p1)


def test_26_validate_revision_rejects_changed_action_with_no_new_evidence():
    """26. validate_revision rejects changed action with no new evidence."""
    p1 = make_proposal(
        round_num=1,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["id1", "id2"],
    )
    p2 = make_proposal(
        round_num=2,
        action=Action.QUARANTINE_AND_ESTIMATE,
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["id1"],  # subset of previous, no new evidence
    )
    with pytest.raises(InvalidRevisionError):
        validate_revision(p2, p1)


def test_27_validate_revision_rejects_changed_origin_with_no_new_evidence():
    """27. validate_revision rejects changed origin with no new evidence."""
    p1 = make_proposal(
        round_num=1,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
        evidence_ids=["id1"],
    )
    p2 = make_proposal(
        round_num=2,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.ADVERSARIAL,
        evidence_ids=["id1"],  # same evidence
    )
    with pytest.raises(InvalidRevisionError):
        validate_revision(p2, p1)


# --------------------------------------------------------------------------- #
# Test 28 & 29: prompt_hash
# --------------------------------------------------------------------------- #


def test_28_prompt_hash_deterministic():
    """28. prompt_hash deterministic."""
    prompt = "You are the Analyst agent. Analyze anomaly record #42."
    h1 = prompt_hash(prompt)
    h2 = prompt_hash(prompt)
    assert h1 == h2
    assert len(h1) == 64


def test_29_prompt_hash_changes_when_prompt_changes():
    """29. prompt_hash changes when prompt changes."""
    p1 = "You are the Analyst agent. Analyze anomaly record #42."
    p2 = "You are the Analyst agent. Analyze anomaly record #43."
    assert prompt_hash(p1) != prompt_hash(p2)


# --------------------------------------------------------------------------- #
# Test 30 & 31: DeliberationResult
# --------------------------------------------------------------------------- #


def test_30_deliberation_result_converged_consistency_enforced():
    """30. DeliberationResult converged consistency enforced."""
    p_analyst = make_proposal(
        agent=AgentRole.ANALYST,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
    )
    p_validator = make_proposal(
        agent=AgentRole.VALIDATOR,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
    )

    # Valid converged result
    res = DeliberationResult(
        anomaly_id="CASE_1",
        converged=True,
        rounds_used=1,
        analyst_final=p_analyst,
        validator_final=p_validator,
        agreed_action=Action.FLAG_DATA_SUSPECT,
        agreed_origin=DecisionOrigin.DATA_QUALITY,
    )
    assert res.converged is True

    # Missing agreed_action when converged=True
    with pytest.raises(ValidationError):
        DeliberationResult(
            anomaly_id="CASE_1",
            converged=True,
            rounds_used=1,
            analyst_final=p_analyst,
            validator_final=p_validator,
            agreed_action=None,
            agreed_origin=DecisionOrigin.DATA_QUALITY,
        )

    # Mismatched action when converged=True
    with pytest.raises(ValidationError):
        DeliberationResult(
            anomaly_id="CASE_1",
            converged=True,
            rounds_used=1,
            analyst_final=p_analyst,
            validator_final=p_validator,
            agreed_action=Action.NO_ACTION,
            agreed_origin=DecisionOrigin.DATA_QUALITY,
        )


def test_31_deliberation_result_non_converged_consistency_enforced():
    """31. DeliberationResult non-converged consistency enforced."""
    p_analyst = make_proposal(
        agent=AgentRole.ANALYST,
        action=Action.FLAG_DATA_SUSPECT,
        origin=DecisionOrigin.DATA_QUALITY,
    )
    p_validator = make_proposal(
        agent=AgentRole.VALIDATOR,
        action=Action.QUARANTINE_AND_ESTIMATE,
        origin=DecisionOrigin.ADVERSARIAL,
    )

    # Valid non-converged result
    res = DeliberationResult(
        anomaly_id="CASE_2",
        converged=False,
        rounds_used=3,
        analyst_final=p_analyst,
        validator_final=p_validator,
        agreed_action=None,
        agreed_origin=None,
    )
    assert res.converged is False
    assert res.agreed_action is None
    assert res.agreed_origin is None

    # Invalid: non-converged cannot have agreed_action
    with pytest.raises(ValidationError):
        DeliberationResult(
            anomaly_id="CASE_2",
            converged=False,
            rounds_used=3,
            analyst_final=p_analyst,
            validator_final=p_validator,
            agreed_action=Action.FLAG_DATA_SUSPECT,
            agreed_origin=None,
        )


# --------------------------------------------------------------------------- #
# Test 32: DecisionOrigin vs Injection Origin
# --------------------------------------------------------------------------- #


def test_32_decision_origin_unknown_works_without_altering_injection_origin():
    """32. DecisionOrigin unknown works without altering the repository injection Origin."""
    # DecisionOrigin includes unknown
    assert DecisionOrigin.UNKNOWN.value == "unknown"
    assert "unknown" in {d.value for d in DecisionOrigin}

    # Repository config.py Origin does NOT include unknown (closed to 3 types)
    import typing

    injection_allowed = set(typing.get_args(InjectionOrigin))
    assert injection_allowed == {"data_quality", "adversarial", "physical_event"}
    assert "unknown" not in injection_allowed


# --------------------------------------------------------------------------- #
# Test 33 & Strong Leakage Test: Ground Truth Boundary
# --------------------------------------------------------------------------- #


def test_33_agent_view_serialization_contains_no_known_ground_truth_fields(
    sample_agent_view: AgentView,
):
    """33. AgentView serialization contains no known ground-truth fields."""
    forbidden_terms = {
        "true_type",
        "true_origin",
        "ground_truth",
        "injected_type",
        "injected_origin",
        "injection_label",
        "target_label",
        "y_true",
    }

    dumped = sample_agent_view.model_dump()
    dumped_json = json.dumps(dumped)

    for term in forbidden_terms:
        assert term not in dumped, f"Forbidden term '{term}' found in model_dump"
        assert term not in dumped_json, f"Forbidden term '{term}' found in serialized JSON"


def test_strong_leakage_rejection(sample_catalog: EvidenceCatalog):
    """Strong leakage test: attempt to inject ground truth into AgentView through various paths."""
    forbidden_payload: dict[str, Any] = {
        "true_type": "additive_bias",
        "true_origin": "data_quality",
        "ground_truth": {"label": 1},
        "injected_type": "voltage_drift",
        "injection_label": "fault",
        "target_label": "tamper",
        "y_true": 1,
    }

    # 1. Direct pass to AgentView constructor
    for key, val in forbidden_payload.items():
        with pytest.raises(ValidationError):
            AgentView(
                anomaly_id="CASE_LEAK",
                role=AgentRole.ANALYST,
                evidence=sample_catalog,
                **{key: val},  # type: ignore[arg-type]
            )

    # 2. Direct pass to CaseMetadata
    for key, val in forbidden_payload.items():
        with pytest.raises(ValidationError):
            CaseMetadata(**{key: val})

    # 3. No unrestricted 'record' dict allowed in AgentView
    with pytest.raises(ValidationError):
        AgentView(
            anomaly_id="CASE_LEAK",
            role=AgentRole.ANALYST,
            evidence=sample_catalog,
            record={"anomaly_type": "dropout", **forbidden_payload},  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------- #
# Protocol & Additional Contract Tests
# --------------------------------------------------------------------------- #


def test_agent_protocol_conformance(sample_agent_view: AgentView):
    """Verify that a compliant mock Agent conforms to the Agent Protocol."""

    class MockAnalyst:
        @property
        def role(self) -> AgentRole:
            return AgentRole.ANALYST

        def propose(
            self,
            view: AgentView,
            round: int,
            other_proposal: Proposal | None = None,
            own_previous_proposal: Proposal | None = None,
        ) -> Proposal:
            return make_proposal(agent=self.role, round_num=round)

    agent = MockAnalyst()
    assert isinstance(agent, Agent)
    p = agent.propose(sample_agent_view, round=1)
    assert p.agent == AgentRole.ANALYST
    assert p.round == 1
