"""Unit tests for M7.2 Analyst agent and model-side evidence boundary.

Verifies:
- build_analyst_view creates strict model-side views from anomaly records.
- Complete exclusion of hidden ground truth and Validator-side evidence.
- RuleBasedAnalyst deterministic baseline decision logic.
- Citation validity and non-empty rationale.
- Action enum consistency against configs/policy.yaml.
- Behavioral tests (hidden label invariance, validator evidence invariance,
  model evidence sensitivity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from energymind.agents.analyst import (
    FORBIDDEN_ANALYST_KEYS,
    P_ANOMALY_LOW_THRESHOLD,
    P_ANOMALY_QUARANTINE_THRESHOLD,
    RuleBasedAnalyst,
    assert_no_leakage,
    build_analyst_view,
)
from energymind.agents.base import (
    Action,
    Agent,
    AgentContractError,
    AgentRole,
    AgentView,
    CaseMetadata,
    DecisionOrigin,
    EvidenceCatalog,
    EvidenceItem,
    validate_citations,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_raw_record() -> dict[str, Any]:
    """A realistic AnomalyRecord dictionary containing model outputs,

    ground truth simulation labels, and Validator-side evidence.
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
        # Ground truth / simulation labels (must be filtered out)
        "true_type": "dead_sensor",
        "true_origin": "data_quality",
        "ground_truth": {"type": "dead_sensor", "origin": "data_quality"},
        "y_true": 1,
        "target_label": "dead_sensor",
        "injected_type": "dead_sensor",
        "injected_origin": "data_quality",
        "injection_label": "dead_sensor",
        "injected_variant": "zero_power",
        "injected_severity": "high",
        "injected_params": {"factor": 0.0},
        # Validator-side fields (must be filtered out)
        "neighborhood_share": 0.05,
        "neigh_share": 0.05,
        "physics_identity_residual": 0.001,
        "current_vs_power_identity": {"status": "valid", "rel_residual": 0.001},
        "r_S": 0.001,
        "r_VI": 0.002,
        "r_E": 0.0,
        "r_PF": 0.0,
        "nilm_resid_total": 0.012,
        "nilm_residual_total": 0.012,
        "verified_history": [],
        "historical_outcome": "maintenance_scheduled",
        "operator_resolution": "repaired",
        "previous_decision": "flag_data_suspect",
        "human_decision": "approved",
        "qdrant_matches": [],
        "seen_before": False,
    }


# --------------------------------------------------------------------------- #
# 7.1 Required Analyst View Tests
# --------------------------------------------------------------------------- #


def test_view_normal_valid_record(sample_raw_record: dict[str, Any]):
    """1. Normal valid record creates an Analyst view."""
    view = build_analyst_view(sample_raw_record)
    assert isinstance(view, AgentView)


def test_view_preserves_anomaly_id(sample_raw_record: dict[str, Any]):
    """2. Anomaly ID is preserved exactly."""
    view = build_analyst_view(sample_raw_record)
    assert view.anomaly_id == "CASE_2020_001"


def test_view_role_is_analyst(sample_raw_record: dict[str, Any]):
    """3. Role is AgentRole.ANALYST."""
    view = build_analyst_view(sample_raw_record)
    assert view.role == AgentRole.ANALYST


def test_view_p_anomaly_stable_id(sample_raw_record: dict[str, Any]):
    """4. p_anomaly becomes stable evidence ID model.p_anomaly."""
    view = build_analyst_view(sample_raw_record)
    item = view.evidence.get("model.p_anomaly")
    assert item is not None
    assert item.value == 0.94
    assert item.source == "detector.isolation_forest"


def test_view_detector_score_included(sample_raw_record: dict[str, Any]):
    """5. Detector score included when available with stable ID."""
    view = build_analyst_view(sample_raw_record)
    item = view.evidence.get("model.detector_score")
    assert item is not None
    assert item.value == 2.15


def test_view_type_probs_deterministic_ids(sample_raw_record: dict[str, Any]):
    """6. Type probabilities generate deterministic evidence IDs."""
    view = build_analyst_view(sample_raw_record)
    assert view.evidence.get("model.type.dead_sensor") is not None
    assert view.evidence.get("model.type.dead_sensor").value == 0.85
    assert view.evidence.get("model.type.dropout") is not None
    assert view.evidence.get("model.type.drift") is not None


def test_view_origin_probs_deterministic_ids(sample_raw_record: dict[str, Any]):
    """7. Origin probabilities generate deterministic evidence IDs."""
    view = build_analyst_view(sample_raw_record)
    assert view.evidence.get("model.origin.data_quality") is not None
    assert view.evidence.get("model.origin.data_quality").value == 0.90
    assert view.evidence.get("model.origin.physical_event") is not None
    assert view.evidence.get("model.origin.adversarial") is not None


def test_view_shap_evidence_included(sample_raw_record: dict[str, Any]):
    """8. SHAP evidence is included with xai.shap prefix."""
    view = build_analyst_view(sample_raw_record)
    item = view.evidence.get("xai.shap.active_power_kw")
    assert item is not None
    assert item.source == "xai.shap"
    assert item.value == 0.35421


def test_view_lime_evidence_included(sample_raw_record: dict[str, Any]):
    """9. LIME evidence is included with xai.lime prefix."""
    view = build_analyst_view(sample_raw_record)
    item = view.evidence.get("xai.lime.active_power_kw")
    assert item is not None
    assert item.source == "xai.lime"
    assert item.value == 0.33211


def test_view_reliability_included(sample_raw_record: dict[str, Any]):
    """10. XAI reliability is included with stable ID xai.reliability."""
    view = build_analyst_view(sample_raw_record)
    item = view.evidence.get("xai.reliability")
    assert item is not None
    assert item.value == 0.78


def test_view_missing_shap_handled(sample_raw_record: dict[str, Any]):
    """11. Missing SHAP is handled without error."""
    sample_raw_record["explanation"]["shap_top"] = None
    view = build_analyst_view(sample_raw_record)
    assert view.evidence.get("xai.shap.active_power_kw") is None
    assert view.evidence.get("xai.reliability") is not None


def test_view_missing_lime_handled(sample_raw_record: dict[str, Any]):
    """12. Missing LIME is handled without error."""
    sample_raw_record["explanation"]["lime_top"] = []
    view = build_analyst_view(sample_raw_record)
    assert view.evidence.get("xai.lime.active_power_kw") is None


def test_view_probability_ordering_invariance(sample_raw_record: dict[str, Any]):
    """13. Probability insertion ordering does not affect evidence IDs or order."""
    record1 = dict(sample_raw_record)
    record1["classification"] = {
        "model": "clf",
        "origin_probs": {"physical_event": 0.08, "data_quality": 0.90, "adversarial": 0.02},
    }

    record2 = dict(sample_raw_record)
    record2["classification"] = {
        "model": "clf",
        "origin_probs": {"adversarial": 0.02, "data_quality": 0.90, "physical_event": 0.08},
    }

    view1 = build_analyst_view(record1)
    view2 = build_analyst_view(record2)

    origin_ids_1 = [item.id for item in view1.evidence if item.id.startswith("model.origin.")]
    origin_ids_2 = [item.id for item in view2.evidence if item.id.startswith("model.origin.")]
    assert origin_ids_1 == origin_ids_2


def test_view_no_duplicate_evidence_ids(sample_raw_record: dict[str, Any]):
    """14. Duplicate evidence IDs cannot occur."""
    view = build_analyst_view(sample_raw_record)
    ids = list(view.evidence.ids)
    assert len(ids) == len(set(ids))


def test_view_serializable(sample_raw_record: dict[str, Any]):
    """15. AgentView remains fully JSON serializable."""
    view = build_analyst_view(sample_raw_record)
    dumped = view.model_dump()
    serialized = json.dumps(dumped)
    assert "CASE_2020_001" in serialized
    assert view.model_dump_json() is not None


def test_view_case_metadata_extracted(sample_raw_record: dict[str, Any]):
    """Safe case metadata is preserved in AgentView."""
    view = build_analyst_view(sample_raw_record)
    assert view.case_metadata.household_id == "HH_00072"
    assert view.case_metadata.t_start == "2020-06-01T12:00:00Z"
    assert view.case_metadata.t_end == "2020-06-01T14:00:00Z"
    assert view.case_metadata.duration_h == 2.0


def test_view_empty_anomaly_id_rejected():
    """Empty anomaly_id raises ValueError."""
    with pytest.raises(ValueError, match="anomaly_id"):
        build_analyst_view({"anomaly_id": ""})


# --------------------------------------------------------------------------- #
# 7.2 Leakage Tests
# --------------------------------------------------------------------------- #


def test_leakage_ground_truth_excluded(sample_raw_record: dict[str, Any]):
    """AgentView contains zero ground-truth keys."""
    view = build_analyst_view(sample_raw_record)
    dumped = view.model_dump()

    ground_truth_keys = [
        "true_type",
        "true_origin",
        "ground_truth",
        "y_true",
        "target_label",
        "injected_type",
        "injected_origin",
        "injection_label",
        "injected_variant",
        "injected_severity",
        "injected_params",
    ]

    def _find_keys(obj: Any) -> set[str]:
        keys = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(k)
                keys |= _find_keys(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                keys |= _find_keys(item)
        return keys

    found_keys = _find_keys(dumped)
    for gtk in ground_truth_keys:
        assert gtk not in found_keys, f"Found ground truth key '{gtk}' in AgentView"


def test_leakage_validator_evidence_excluded(sample_raw_record: dict[str, Any]):
    """AgentView contains zero Validator-side evidence keys."""
    view = build_analyst_view(sample_raw_record)
    dumped = view.model_dump()

    validator_keys = [
        "neighborhood_share",
        "neigh_share",
        "physics_identity_residual",
        "verified_history",
        "historical_outcome",
        "current_vs_power_identity",
        "qdrant_matches",
        "seen_before",
        "r_S",
        "r_VI",
        "r_E",
        "r_PF",
        "nilm_resid_total",
        "nilm_residual_total",
    ]

    def _find_keys(obj: Any) -> set[str]:
        keys = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                keys.add(k)
                keys |= _find_keys(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                keys |= _find_keys(item)
        return keys

    found_keys = _find_keys(dumped)
    for vk in validator_keys:
        assert vk not in found_keys, f"Found validator key '{vk}' in AgentView"


def test_assert_no_leakage_passes_on_valid_view(sample_raw_record: dict[str, Any]):
    """assert_no_leakage runs cleanly on build_analyst_view output."""
    view = build_analyst_view(sample_raw_record)
    assert_no_leakage(view)


def test_assert_no_leakage_catches_injected_forbidden_key():
    """assert_no_leakage detects any forbidden key if injected."""
    bad_view = AgentView(
        anomaly_id="CASE_TEST",
        role=AgentRole.ANALYST,
        evidence=EvidenceCatalog(
            [
                EvidenceItem(
                    id="model.p_anomaly",
                    source="detector",
                    description="p",
                    value=0.9,
                )
            ]
        ),
        case_metadata=CaseMetadata(household_id="HH1"),
    )
    # Valid view passes
    assert_no_leakage(bad_view)

    # Now create view with a forbidden key in extra metadata or evidence
    leaked_item = EvidenceItem(
        id="model.p_anomaly",
        source="detector",
        description="p",
        value={"true_origin": "data_quality"},
    )
    leaked_view = AgentView(
        anomaly_id="CASE_TEST",
        role=AgentRole.ANALYST,
        evidence=EvidenceCatalog([leaked_item]),
    )
    with pytest.raises(AgentContractError, match="Forbidden key 'true_origin'"):
        assert_no_leakage(leaked_view)


def test_forbidden_keys_constant():
    """Verify core forbidden ground truth and validator keys are in FORBIDDEN_ANALYST_KEYS."""
    assert "true_origin" in FORBIDDEN_ANALYST_KEYS
    assert "true_type" in FORBIDDEN_ANALYST_KEYS
    assert "ground_truth" in FORBIDDEN_ANALYST_KEYS
    assert "neighborhood_share" in FORBIDDEN_ANALYST_KEYS
    assert "physics_identity_residual" in FORBIDDEN_ANALYST_KEYS


def test_baseline_threshold_constants():
    """Verify M7.2 deterministic baseline thresholds."""
    assert P_ANOMALY_LOW_THRESHOLD == 0.50
    assert P_ANOMALY_QUARANTINE_THRESHOLD == 0.85


# --------------------------------------------------------------------------- #
# 7.3 Rule Analyst Tests
# --------------------------------------------------------------------------- #


def test_rule_analyst_satisfies_agent_protocol():
    """RuleBasedAnalyst implements the Agent protocol."""
    analyst = RuleBasedAnalyst()
    assert isinstance(analyst, Agent)
    assert analyst.role == AgentRole.ANALYST


def test_rule_analyst_low_anomaly_conservative():
    """16. Low anomaly evidence -> conservative NO_ACTION result."""
    view = AgentView(
        anomaly_id="CASE_LOW",
        role=AgentRole.ANALYST,
        evidence=EvidenceCatalog(
            [
                EvidenceItem(
                    id="model.p_anomaly",
                    source="detector",
                    description="p",
                    value=0.30,
                )
            ]
        ),
    )
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.action == Action.NO_ACTION
    assert proposal.origin == DecisionOrigin.UNKNOWN
    assert proposal.support == 0.70  # 1.0 - 0.30
    assert proposal.evidence_ids == ["model.p_anomaly"]


def test_rule_analyst_clear_data_quality_normal(sample_raw_record: dict[str, Any]):
    """17. Clear data-quality origin -> FLAG_DATA_SUSPECT."""
    sample_raw_record["severity"] = "low"
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.DATA_QUALITY
    assert proposal.action == Action.FLAG_DATA_SUSPECT
    assert "model.origin.data_quality" in proposal.evidence_ids


def test_rule_analyst_clear_data_quality_severe(sample_raw_record: dict[str, Any]):
    """Clear data-quality with severe flag -> QUARANTINE_AND_ESTIMATE."""
    sample_raw_record["severity"] = "critical"
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.DATA_QUALITY
    assert proposal.action == Action.QUARANTINE_AND_ESTIMATE
    assert "model.severity" in proposal.evidence_ids


def test_rule_analyst_clear_physical_event(sample_raw_record: dict[str, Any]):
    """18. Clear physical-event origin -> COORDINATE_WITH_GRID_OPERATOR."""
    sample_raw_record["classification"]["origin_probs"] = {
        "physical_event": 0.88,
        "data_quality": 0.08,
        "adversarial": 0.04,
    }
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.PHYSICAL_EVENT
    assert proposal.action == Action.COORDINATE_WITH_GRID_OPERATOR
    assert "model.origin.physical_event" in proposal.evidence_ids


def test_rule_analyst_clear_adversarial(sample_raw_record: dict[str, Any]):
    """19. Clear adversarial origin -> INVESTIGATE_TAMPERING."""
    sample_raw_record["classification"]["origin_probs"] = {
        "adversarial": 0.82,
        "data_quality": 0.12,
        "physical_event": 0.06,
    }
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.ADVERSARIAL
    assert proposal.action == Action.INVESTIGATE_TAMPERING
    assert "model.origin.adversarial" in proposal.evidence_ids


def test_rule_analyst_ambiguous_origin_low_top_prob(sample_raw_record: dict[str, Any]):
    """20. Ambiguous origin (low top probability) -> UNKNOWN."""
    sample_raw_record["classification"]["origin_probs"] = {
        "data_quality": 0.45,
        "physical_event": 0.35,
        "adversarial": 0.20,
    }
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.UNKNOWN
    assert proposal.action == Action.FLAG_DATA_SUSPECT


def test_rule_analyst_close_top_two_probabilities(sample_raw_record: dict[str, Any]):
    """21. Close top-two probabilities -> UNKNOWN conservative."""
    sample_raw_record["classification"]["origin_probs"] = {
        "data_quality": 0.52,
        "physical_event": 0.48,
        "adversarial": 0.0,
    }
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.UNKNOWN
    assert proposal.action == Action.FLAG_DATA_SUSPECT


def test_rule_analyst_missing_origin_probs(sample_raw_record: dict[str, Any]):
    """Missing origin probs -> conservative UNKNOWN proposal."""
    sample_raw_record["classification"]["origin_probs"] = {}
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.UNKNOWN
    assert proposal.action == Action.FLAG_DATA_SUSPECT
    assert "model.p_anomaly" in proposal.evidence_ids


def test_rule_analyst_cites_corresponding_evidence(sample_raw_record: dict[str, Any]):
    """22. Strong origin proposal cites corresponding evidence."""
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert "model.origin.data_quality" in proposal.evidence_ids
    assert "model.p_anomaly" in proposal.evidence_ids


def test_rule_analyst_no_empty_evidence(sample_raw_record: dict[str, Any]):
    """23. No proposal has empty evidence citations."""
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert len(proposal.evidence_ids) > 0


def test_rule_analyst_action_in_action_enum(sample_raw_record: dict[str, Any]):
    """24. Returned action is an Action enum member."""
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert isinstance(proposal.action, Action)


def test_rule_analyst_origin_in_decision_origin(sample_raw_record: dict[str, Any]):
    """25. Returned origin is a DecisionOrigin enum member."""
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert isinstance(proposal.origin, DecisionOrigin)


def test_rule_analyst_support_range(sample_raw_record: dict[str, Any]):
    """26. Support is bounded in [0.0, 1.0]."""
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert 0.0 <= proposal.support <= 1.0


def test_rule_analyst_rationale_non_empty(sample_raw_record: dict[str, Any]):
    """27. Rationale is non-empty string."""
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert isinstance(proposal.rationale, str)
    assert len(proposal.rationale) > 10


def test_rule_analyst_passes_validate_citations(sample_raw_record: dict[str, Any]):
    """28. Proposal passes validate_citations against view."""
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    # If this does not raise, citation validation passed
    validate_citations(proposal, view)


def test_rule_analyst_rejects_raw_dict(sample_raw_record: dict[str, Any]):
    """29. Raw record is never consumed directly by RuleBasedAnalyst."""
    analyst = RuleBasedAnalyst()
    with pytest.raises((AttributeError, AgentContractError)):
        analyst.propose(sample_raw_record, round=1)  # type: ignore[arg-type]


def test_rule_analyst_rejects_wrong_role_view():
    """AgentView with role != ANALYST is rejected."""
    wrong_view = AgentView(
        anomaly_id="CASE_TEST",
        role=AgentRole.VALIDATOR,
        evidence=EvidenceCatalog(
            [
                EvidenceItem(
                    id="model.p_anomaly",
                    source="detector",
                    description="p",
                    value=0.9,
                )
            ]
        ),
    )
    analyst = RuleBasedAnalyst()
    with pytest.raises(AgentContractError, match="role=ANALYST"):
        analyst.propose(wrong_view, round=1)


def test_rule_analyst_rejects_empty_evidence_view():
    """AgentView with zero evidence items is rejected."""
    empty_view = AgentView(
        anomaly_id="CASE_EMPTY",
        role=AgentRole.ANALYST,
        evidence=EvidenceCatalog([]),
    )
    analyst = RuleBasedAnalyst()
    with pytest.raises(AgentContractError, match="without evidence"):
        analyst.propose(empty_view, round=1)


# --------------------------------------------------------------------------- #
# Behavioral Invariance Tests (Tests 30 - 32)
# --------------------------------------------------------------------------- #


def test_behavioral_hidden_label_invariance(sample_raw_record: dict[str, Any]):
    """30. Changing hidden true label does NOT change the proposal."""
    view_a = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal_a = analyst.propose(view_a, round=1)

    # Mutate all hidden simulation/ground-truth fields to adversarial/physical
    mutated_record = dict(sample_raw_record)
    mutated_record["true_type"] = "tampering"
    mutated_record["true_origin"] = "adversarial"
    mutated_record["ground_truth"] = {"type": "theft", "origin": "adversarial"}
    mutated_record["y_true"] = 0
    mutated_record["target_label"] = "theft"
    mutated_record["injected_type"] = "theft"
    mutated_record["injected_origin"] = "adversarial"
    mutated_record["injection_label"] = "theft"

    view_b = build_analyst_view(mutated_record)
    proposal_b = analyst.propose(view_b, round=1)

    assert proposal_a.action == proposal_b.action
    assert proposal_a.origin == proposal_b.origin
    assert proposal_a.support == proposal_b.support
    assert proposal_a.evidence_ids == proposal_b.evidence_ids
    assert proposal_a.rationale == proposal_b.rationale


def test_behavioral_validator_evidence_invariance(sample_raw_record: dict[str, Any]):
    """31. Changing Validator-only evidence does NOT change the proposal."""
    view_a = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal_a = analyst.propose(view_a, round=1)

    # Mutate Validator-side fields to extreme values
    mutated_record = dict(sample_raw_record)
    mutated_record["neighborhood_share"] = 0.99
    mutated_record["neigh_share"] = 0.99
    mutated_record["physics_identity_residual"] = 999.0
    mutated_record["current_vs_power_identity"] = {"status": "broken", "rel_residual": 50.0}
    mutated_record["verified_history"] = [{"resolution": "false_alarm"}]
    mutated_record["historical_outcome"] = "tampering_confirmed"

    view_b = build_analyst_view(mutated_record)
    proposal_b = analyst.propose(view_b, round=1)

    assert proposal_a.action == proposal_b.action
    assert proposal_a.origin == proposal_b.origin
    assert proposal_a.support == proposal_b.support
    assert proposal_a.evidence_ids == proposal_b.evidence_ids
    assert proposal_a.rationale == proposal_b.rationale


def test_behavioral_model_evidence_sensitivity(sample_raw_record: dict[str, Any]):
    """32. Changing model-side evidence CAN and DOES change the proposal."""
    view_a = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal_a = analyst.propose(view_a, round=1)

    # Mutate model-side evidence: change classification to adversarial
    mutated_record = dict(sample_raw_record)
    mutated_record["classification"] = {
        "model": "lightgbm",
        "origin_probs": {
            "adversarial": 0.92,
            "data_quality": 0.05,
            "physical_event": 0.03,
        },
    }

    view_b = build_analyst_view(mutated_record)
    proposal_b = analyst.propose(view_b, round=1)

    # Must change!
    assert proposal_a.action != proposal_b.action
    assert proposal_a.origin != proposal_b.origin
    assert proposal_b.action == Action.INVESTIGATE_TAMPERING
    assert proposal_b.origin == DecisionOrigin.ADVERSARIAL


# --------------------------------------------------------------------------- #
# 8. Parameterized Testing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("origin_key", "expected_origin", "expected_action"),
    [
        ("physical_event", DecisionOrigin.PHYSICAL_EVENT, Action.COORDINATE_WITH_GRID_OPERATOR),
        ("adversarial", DecisionOrigin.ADVERSARIAL, Action.INVESTIGATE_TAMPERING),
    ],
)
def test_parametrized_origin_mappings(
    sample_raw_record: dict[str, Any],
    origin_key: str,
    expected_origin: DecisionOrigin,
    expected_action: Action,
):
    """Parametrized test for high-confidence origin mapping."""
    sample_raw_record["classification"]["origin_probs"] = {
        origin_key: 0.90,
        "data_quality": 0.10,
    }
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.origin == expected_origin
    assert proposal.action == expected_action


@pytest.mark.parametrize(
    ("p_val", "expected_action"),
    [
        (0.10, Action.NO_ACTION),
        (0.35, Action.NO_ACTION),
        (0.49, Action.NO_ACTION),
        (0.51, Action.FLAG_DATA_SUSPECT),
        (0.95, Action.QUARANTINE_AND_ESTIMATE),
    ],
)
def test_parametrized_anomaly_probability_thresholds(
    sample_raw_record: dict[str, Any],
    p_val: float,
    expected_action: Action,
):
    """Parametrized test across anomaly detector probability thresholds."""
    sample_raw_record["severity"] = None
    sample_raw_record["detection"]["p_anomaly"] = p_val
    view = build_analyst_view(sample_raw_record)
    analyst = RuleBasedAnalyst()
    proposal = analyst.propose(view, round=1)

    assert proposal.action == expected_action


# --------------------------------------------------------------------------- #
# 9. Action Enum Consistency Check
# --------------------------------------------------------------------------- #


def test_action_enum_matches_policy_yaml():
    """Verify that Action enum members exactly match the policy actions in configs/policy.yaml."""
    policy_path = Path(__file__).resolve().parent.parent / "configs" / "policy.yaml"
    assert policy_path.exists(), f"policy.yaml not found at {policy_path}"

    with open(policy_path, encoding="utf-8") as f:
        policy_cfg = yaml.safe_load(f)

    yaml_actions = set(policy_cfg.get("actions", {}).keys())
    enum_actions = {a.value for a in Action}

    assert enum_actions == yaml_actions, (
        f"Drift between Action enum and configs/policy.yaml!\n"
        f"Enum: {enum_actions}\n"
        f"YAML: {yaml_actions}"
    )
