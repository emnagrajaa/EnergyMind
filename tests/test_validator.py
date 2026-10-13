"""Unit tests for M7.3 Validator agent, physical/historical evidence boundary, and memory safety.

Verifies:
- build_validator_view creates strict physical/contextual views from anomaly records.
- Complete exclusion of model-side classifier/XAI outputs and hidden ground truth.
- Historical memory eligibility (unverified exclusion, self-match exclusion,
  temporal future exclusion).
- RuleBasedValidator deterministic baseline decision logic.
- Citation validity and non-empty rationale.
- Behavioral invariance (hidden-label, model-evidence, unverified-history, future-history).
- Analyst vs Validator evidence asymmetry.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from energymind.agents.analyst import build_analyst_view
from energymind.agents.base import (
    Action,
    Agent,
    AgentContractError,
    AgentRole,
    AgentView,
    DecisionOrigin,
    EvidenceCatalog,
    EvidenceItem,
    validate_citations,
)
from energymind.agents.validator import (
    FORBIDDEN_VALIDATOR_KEYS,
    NEIGHBORHOOD_COHERENT_THRESHOLD,
    PHYSICS_IDENTITY_BROKEN_THRESHOLD,
    PHYSICS_IDENTITY_SEVERE_THRESHOLD,
    TAU_SIM_DEFAULT,
    HistoricalCaseSummary,
    RuleBasedValidator,
    assert_no_validator_leakage,
    build_validator_view,
    eligible_history,
    hydrate_historical_case,
)
from energymind.retrieval import Case, Match
from energymind.store import AuditLog

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


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
        # Model-side outputs (must be excluded from Validator)
        "detection": {
            "model": "isolation_forest",
            "p_anomaly": 0.94,
            "score": 2.15,
        },
        "classification": {
            "model": "lightgbm",
            "type_probs": {"dead_sensor": 0.85, "dropout": 0.10, "drift": 0.05},
            "origin_probs": {"data_quality": 0.90, "physical_event": 0.08, "adversarial": 0.02},
        },
        "explanation": {
            "shap_top": [["active_power_kw", 0.35421], ["voltage_v", -0.12345]],
            "lime_top": [["active_power_kw", 0.33211], ["current_a", 0.15432]],
            "reliability": 0.78,
        },
        "severity": "high",
        # Simulation / ground-truth labels (must be excluded from Validator)
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
        # Validator-side physical, baseline, neighborhood, sensor evidence
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
                "value": 0.08,
                "status": "isolated",
            },
            {
                "check": "nilm_residual_total",
                "value": 0.015,
            },
            {
                "check": "sub_meter_share_cooling",
                "value": 0.32,
            },
        ],
        "facts": {
            "observed_kw": 0.12,
            "expected_kw": 1.45,
            "z": -2.85,
            "share": 0.08,
            "coherent": False,
            "identity_rel": 0.0012,
            "identity_broken": False,
            "zero_frac": 0.0,
            "zero_reading": False,
            "flat_run": 0,
            "voltage_dev": 1.2,
        },
    }


# --------------------------------------------------------------------------- #
# 19.1 View Tests
# --------------------------------------------------------------------------- #


def test_view_normal_valid_record(sample_raw_record: dict[str, Any]):
    """1. Normal valid record creates a Validator view."""
    view = build_validator_view(sample_raw_record)
    assert isinstance(view, AgentView)


def test_view_preserves_anomaly_id(sample_raw_record: dict[str, Any]):
    """2. Anomaly ID is preserved exactly."""
    view = build_validator_view(sample_raw_record)
    assert view.anomaly_id == "CASE_2020_001"


def test_view_role_is_validator(sample_raw_record: dict[str, Any]):
    """3. Role is AgentRole.VALIDATOR."""
    view = build_validator_view(sample_raw_record)
    assert view.role == AgentRole.VALIDATOR


def test_view_physics_evidence_stable_ids(sample_raw_record: dict[str, Any]):
    """4. Physics evidence converted to stable IDs."""
    view = build_validator_view(sample_raw_record)
    item = view.evidence.get("physics.current_vs_power_identity")
    assert item is not None
    assert item.value == 0.0012
    assert item.status == "holds"
    assert item.unit == "relative"


def test_view_baseline_evidence_included(sample_raw_record: dict[str, Any]):
    """5. Baseline evidence included with stable IDs."""
    view = build_validator_view(sample_raw_record)
    assert view.evidence.get("baseline.z_score") is not None
    assert view.evidence.get("baseline.z_score").value == -2.85
    assert view.evidence.get("baseline.observed_power") is not None
    assert view.evidence.get("baseline.observed_power").value == 0.12
    assert view.evidence.get("baseline.expected_power") is not None
    assert view.evidence.get("baseline.expected_power").value == 1.45


def test_view_neighborhood_evidence_included(sample_raw_record: dict[str, Any]):
    """6. Neighborhood evidence included with stable IDs."""
    view = build_validator_view(sample_raw_record)
    item = view.evidence.get("neighborhood.share")
    assert item is not None
    assert item.value == 0.08
    assert item.status == "isolated"


def test_view_nilm_evidence_included(sample_raw_record: dict[str, Any]):
    """7. NILM evidence included when available."""
    view = build_validator_view(sample_raw_record)
    item = view.evidence.get("nilm.residual_total")
    assert item is not None
    assert item.value == 0.015
    assert item.unit == "kWh"


def test_view_sensor_diagnostics_included(sample_raw_record: dict[str, Any]):
    """8. Sensor diagnostics evidence included."""
    sample_raw_record["facts"]["flat_run"] = 4
    sample_raw_record["facts"]["voltage_dev"] = 3.5
    view = build_validator_view(sample_raw_record)
    assert view.evidence.get("sensor.flat_run_length") is not None
    assert view.evidence.get("sensor.flat_run_length").value == 4
    assert view.evidence.get("sensor.voltage_deviation") is not None
    assert view.evidence.get("sensor.voltage_deviation").value == 3.5


def test_view_optional_evidence_absent_safely(sample_raw_record: dict[str, Any]):
    """9. Optional evidence can be safely absent."""
    del sample_raw_record["evidence"]
    del sample_raw_record["facts"]
    view = build_validator_view(sample_raw_record)
    assert isinstance(view, AgentView)
    assert view.evidence.get("nilm.residual_total") is None


def test_view_history_summaries_stable_ids(sample_raw_record: dict[str, Any]):
    """10. History summaries produce deterministic IDs."""
    hist_cases = [
        HistoricalCaseSummary(
            anomaly_id="HIST_001",
            similarity=0.95,
            type="dead_sensor",
            origin="data_quality",
            status="verified",
            severity="high",
            resolution="sensor_replaced",
            timestamp="2020-05-15T10:00:00Z",
        )
    ]
    view = build_validator_view(sample_raw_record, history=hist_cases)
    assert view.evidence.get("history.0.similarity") is not None
    assert view.evidence.get("history.0.similarity").value == 0.95
    assert view.evidence.get("history.0.type").value == "dead_sensor"
    assert view.evidence.get("history.0.origin").value == "data_quality"
    assert view.evidence.get("history.0.resolution").value == "sensor_replaced"


def test_view_history_ordering_deterministic(sample_raw_record: dict[str, Any]):
    """11. History ordering is deterministic (sorted by descending similarity)."""
    hist_cases = [
        HistoricalCaseSummary(
            anomaly_id="HIST_A",
            similarity=0.91,
            type="dropout",
            origin="data_quality",
            status="verified",
            timestamp="2020-05-01T10:00:00Z",
        ),
        HistoricalCaseSummary(
            anomaly_id="HIST_B",
            similarity=0.98,
            type="drift",
            origin="data_quality",
            status="verified",
            timestamp="2020-05-02T10:00:00Z",
        ),
    ]
    view = build_validator_view(sample_raw_record, history=hist_cases)
    # HIST_B has higher similarity (0.98), so it must be history.0
    assert view.evidence.get("history.0.similarity").value == 0.98
    assert view.evidence.get("history.0.type").value == "drift"
    assert view.evidence.get("history.1.similarity").value == 0.91
    assert view.evidence.get("history.1.type").value == "dropout"


def test_view_no_duplicate_evidence_ids(sample_raw_record: dict[str, Any]):
    """12. Duplicate evidence IDs cannot occur."""
    view = build_validator_view(sample_raw_record)
    ids = list(view.evidence.ids)
    assert len(ids) == len(set(ids))


def test_view_serializable(sample_raw_record: dict[str, Any]):
    """13. Validator AgentView remains fully JSON serializable."""
    view = build_validator_view(sample_raw_record)
    dumped = view.model_dump()
    serialized = json.dumps(dumped)
    assert "CASE_2020_001" in serialized
    assert view.model_dump_json() is not None


def test_view_empty_anomaly_id_rejected():
    """14. Empty anomaly_id raises ValueError."""
    with pytest.raises(ValueError, match="anomaly_id"):
        build_validator_view({"anomaly_id": ""})


# --------------------------------------------------------------------------- #
# 19.2 Model-Side Leakage Tests
# --------------------------------------------------------------------------- #


def test_leakage_model_interpretation_excluded(sample_raw_record: dict[str, Any]):
    """15. Ensure Validator view contains zero model-side classifier or XAI fields."""
    view = build_validator_view(sample_raw_record)
    dumped = view.model_dump()

    forbidden_model_keys = [
        "type_probs",
        "origin_probs",
        "p_anomaly",
        "detector_score",
        "shap_top",
        "lime_top",
        "reliability",
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

    found = _find_keys(dumped)
    for fmk in forbidden_model_keys:
        assert fmk not in found, f"Found forbidden model key '{fmk}' in Validator view"

    # Also check evidence IDs
    for item in view.evidence:
        assert not item.id.startswith("model.type.")
        assert not item.id.startswith("model.origin.")
        assert not item.id.startswith("model.p_anomaly")
        assert not item.id.startswith("xai.")


def test_leakage_analyst_opinion_excluded(sample_raw_record: dict[str, Any]):
    """16. Ensure Validator view contains zero Analyst opinions in blind mode."""
    sample_raw_record["analyst_proposal"] = {"action": "flag_data_suspect"}
    sample_raw_record["analyst_rationale"] = "Model strongly predicts dead sensor"
    sample_raw_record["analyst_support"] = 0.95

    view = build_validator_view(sample_raw_record)
    dumped = view.model_dump()

    for item in view.evidence:
        assert not item.id.startswith("analyst.")
    assert "analyst_proposal" not in dumped
    assert "analyst_rationale" not in dumped


# --------------------------------------------------------------------------- #
# 19.3 Hidden-Truth Leakage Tests
# --------------------------------------------------------------------------- #


def test_leakage_ground_truth_excluded(sample_raw_record: dict[str, Any]):
    """17. Ensure Validator view contains zero simulation / ground truth fields."""
    view = build_validator_view(sample_raw_record)
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

    found = _find_keys(dumped)
    for gtk in ground_truth_keys:
        assert gtk not in found, f"Found ground truth key '{gtk}' in Validator view"


def test_assert_no_validator_leakage_passes_on_valid_view(sample_raw_record: dict[str, Any]):
    """18. assert_no_validator_leakage passes on clean view."""
    view = build_validator_view(sample_raw_record)
    assert_no_validator_leakage(view)


def test_assert_no_validator_leakage_catches_injected_forbidden_key():
    """19. assert_no_validator_leakage catches injected ground truth keys."""
    bad_view = AgentView(
        anomaly_id="CASE_TEST",
        role=AgentRole.VALIDATOR,
        evidence=EvidenceCatalog(
            [
                EvidenceItem(
                    id="physics.current_vs_power_identity",
                    source="physics.residuals",
                    description="test",
                    value={"true_origin": "data_quality"},
                )
            ]
        ),
    )
    with pytest.raises(AgentContractError, match="Forbidden key 'true_origin'"):
        assert_no_validator_leakage(bad_view)


def test_assert_no_validator_leakage_catches_model_side_prefix():
    """20. assert_no_validator_leakage catches model-side evidence IDs."""
    bad_view = AgentView(
        anomaly_id="CASE_TEST",
        role=AgentRole.VALIDATOR,
        evidence=EvidenceCatalog(
            [
                EvidenceItem(
                    id="model.origin.data_quality",
                    source="classifier",
                    description="test",
                    value=0.9,
                )
            ]
        ),
    )
    with pytest.raises(AgentContractError, match="Forbidden model-side evidence ID"):
        assert_no_validator_leakage(bad_view)


# --------------------------------------------------------------------------- #
# 19.4 History Safety Tests
# --------------------------------------------------------------------------- #


def test_history_unverified_excluded(sample_raw_record: dict[str, Any]):
    """21. Unverified history is excluded."""
    unverified_case = Case(
        anomaly_id="HIST_UNVERIFIED",
        signature=[0.1] * 64,
        type_pred="dead_sensor",
        origin_pred="data_quality",
        status="proposed",  # Not verified!
        timestamp="2020-05-01T00:00:00Z",
    )
    assert not eligible_history(unverified_case, current_anomaly_id="CASE_2020_001")

    view = build_validator_view(sample_raw_record, history=[unverified_case])
    assert view.evidence.get("history.0.similarity") is None


def test_history_current_anomaly_self_match_excluded(sample_raw_record: dict[str, Any]):
    """22. Current anomaly cannot match itself even if verified."""
    self_match = Case(
        anomaly_id="CASE_2020_001",  # Same ID!
        signature=[0.5] * 64,
        type_pred="dead_sensor",
        origin_pred="data_quality",
        status="verified",
        timestamp="2020-05-01T00:00:00Z",
    )
    assert not eligible_history(self_match, current_anomaly_id="CASE_2020_001")

    view = build_validator_view(sample_raw_record, history=[self_match])
    assert view.evidence.get("history.0.similarity") is None


def test_history_future_event_excluded(sample_raw_record: dict[str, Any]):
    """23. Future historical events are strictly excluded."""
    future_case = Case(
        anomaly_id="HIST_FUTURE",
        signature=[0.5] * 64,
        type_pred="dead_sensor",
        origin_pred="data_quality",
        status="verified",
        timestamp="2020-07-01T12:00:00Z",  # AFTER t_start 2020-06-01T12:00:00Z
    )
    assert not eligible_history(
        future_case,
        current_anomaly_id="CASE_2020_001",
        current_timestamp="2020-06-01T12:00:00Z",
    )

    view = build_validator_view(sample_raw_record, history=[future_case])
    assert view.evidence.get("history.0.similarity") is None


def test_history_verified_past_event_included(sample_raw_record: dict[str, Any]):
    """24. Verified past event is included."""
    past_case = Match(
        case=Case(
            anomaly_id="HIST_PAST",
            signature=[0.5] * 64,
            type_pred="dead_sensor",
            origin_pred="data_quality",
            status="verified",
            timestamp="2020-05-01T12:00:00Z",  # BEFORE current
        ),
        similarity=0.96,
    )
    assert eligible_history(
        past_case,
        current_anomaly_id="CASE_2020_001",
        current_timestamp="2020-06-01T12:00:00Z",
    )

    view = build_validator_view(sample_raw_record, history=[past_case])
    assert view.evidence.get("history.0.similarity") is not None
    assert view.evidence.get("history.0.similarity").value == 0.96


def test_history_low_similarity_excluded(sample_raw_record: dict[str, Any]):
    """25. Low-similarity event (< tau_sim) is excluded by eligibility policy."""
    low_sim_match = Match(
        case=Case(
            anomaly_id="HIST_LOW_SIM",
            signature=[0.5] * 64,
            type_pred="dead_sensor",
            origin_pred="data_quality",
            status="verified",
            timestamp="2020-05-01T12:00:00Z",
        ),
        similarity=0.82,  # < TAU_SIM_DEFAULT 0.90
    )
    assert not eligible_history(
        low_sim_match,
        current_anomaly_id="CASE_2020_001",
        current_timestamp="2020-06-01T12:00:00Z",
        tau_sim=0.90,
    )

    view = build_validator_view(sample_raw_record, history=[low_sim_match], tau_sim=0.90)
    assert view.evidence.get("history.0.similarity") is None


def test_history_raw_payload_not_exposed(sample_raw_record: dict[str, Any]):
    """26. Raw Qdrant payload/embeddings are never exposed in AgentView."""
    case_with_payload = Match(
        case=Case(
            anomaly_id="HIST_PAYLOAD",
            signature=[0.123] * 64,  # Raw 64-dim float array
            type_pred="gain_error",
            origin_pred="data_quality",
            status="verified",
            timestamp="2020-05-01T12:00:00Z",
            record={"secret_db_info": "internal_uri_123"},
        ),
        similarity=0.95,
    )
    view = build_validator_view(sample_raw_record, history=[case_with_payload])
    serialized = view.model_dump_json()

    assert "secret_db_info" not in serialized
    assert "0.123" not in serialized


def test_history_hidden_labels_not_exposed(sample_raw_record: dict[str, Any]):
    """27. Historical hidden injection labels are not exposed."""
    case_with_labels = Match(
        case=Case(
            anomaly_id="HIST_LABELS",
            signature=[0.1] * 64,
            type_pred="gain_error",
            origin_pred="data_quality",
            status="verified",
            timestamp="2020-05-01T12:00:00Z",
            record={"true_origin": "data_quality", "y_true": 1},
        ),
        similarity=0.95,
    )
    view = build_validator_view(sample_raw_record, history=[case_with_labels])
    serialized = view.model_dump_json()

    assert "true_origin" not in serialized
    assert "y_true" not in serialized


def test_history_changing_unverified_does_not_affect_proposal(sample_raw_record: dict[str, Any]):
    """28. Changing unverified historical cases does not alter proposal."""
    validator = RuleBasedValidator()
    view_1 = build_validator_view(sample_raw_record, history=[])
    prop_1 = validator.propose(view_1, round=1)

    unverified = [
        Case(
            anomaly_id=f"UNV_{i}",
            signature=[0.1] * 64,
            type_pred="tampering",
            origin_pred="adversarial",
            status="proposed",
            timestamp="2020-05-01T00:00:00Z",
        )
        for i in range(5)
    ]
    view_2 = build_validator_view(sample_raw_record, history=unverified)
    prop_2 = validator.propose(view_2, round=1)

    assert prop_1.action == prop_2.action
    assert prop_1.origin == prop_2.origin
    assert prop_1.support == prop_2.support
    assert prop_1.evidence_ids == prop_2.evidence_ids


def test_history_changing_future_does_not_affect_proposal(sample_raw_record: dict[str, Any]):
    """29. Changing future history does not alter proposal."""
    validator = RuleBasedValidator()
    view_1 = build_validator_view(sample_raw_record, history=[])
    prop_1 = validator.propose(view_1, round=1)

    future_match = [
        Match(
            case=Case(
                anomaly_id="FUTURE_1",
                signature=[0.1] * 64,
                type_pred="tampering",
                origin_pred="adversarial",
                status="verified",
                timestamp="2025-01-01T00:00:00Z",  # In the future!
            ),
            similarity=1.0,
        )
    ]
    view_2 = build_validator_view(sample_raw_record, history=future_match)
    prop_2 = validator.propose(view_2, round=1)

    assert prop_1.action == prop_2.action
    assert prop_1.origin == prop_2.origin
    assert prop_1.support == prop_2.support


# --------------------------------------------------------------------------- #
# 19.5 Rule Validator Tests
# --------------------------------------------------------------------------- #


def test_rule_validator_satisfies_agent_protocol():
    """30. RuleBasedValidator implements the Agent protocol."""
    validator = RuleBasedValidator()
    assert isinstance(validator, Agent)
    assert validator.role == AgentRole.VALIDATOR


def test_rule_validator_broken_identity(sample_raw_record: dict[str, Any]):
    """31. Broken electrical identity -> REQUEST_METER_REREAD / DATA_QUALITY."""
    sample_raw_record["evidence"][0]["status"] = "broken"
    sample_raw_record["evidence"][0]["rel_residual"] = 0.04
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.DATA_QUALITY
    assert proposal.action == Action.REQUEST_METER_REREAD
    assert "physics.current_vs_power_identity" in proposal.evidence_ids


def test_rule_validator_severe_broken_identity(sample_raw_record: dict[str, Any]):
    """32. Severe broken electrical identity -> QUARANTINE_AND_ESTIMATE."""
    sample_raw_record["evidence"][0]["status"] = "broken"
    sample_raw_record["evidence"][0]["rel_residual"] = 0.25
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.DATA_QUALITY
    assert proposal.action == Action.QUARANTINE_AND_ESTIMATE
    assert "physics.current_vs_power_identity" in proposal.evidence_ids


def test_rule_validator_dropout_reading(sample_raw_record: dict[str, Any]):
    """33. Dropout reading -> REQUEST_METER_REREAD."""
    sample_raw_record["facts"]["zero_reading"] = True
    sample_raw_record["facts"]["zero_frac"] = 0.95
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.DATA_QUALITY
    assert proposal.action == Action.REQUEST_METER_REREAD
    assert "sensor.zero_reading" in proposal.evidence_ids


def test_rule_validator_flat_run(sample_raw_record: dict[str, Any]):
    """34. Long flat run -> OPEN_MAINTENANCE_TICKET."""
    sample_raw_record["facts"]["flat_run"] = 12
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.DATA_QUALITY
    assert proposal.action == Action.OPEN_MAINTENANCE_TICKET
    assert "sensor.flat_run_length" in proposal.evidence_ids


def test_rule_validator_coherent_neighborhood_event(sample_raw_record: dict[str, Any]):
    """35. Coherent neighborhood event -> COORDINATE_WITH_GRID_OPERATOR."""
    sample_raw_record["evidence"][2]["value"] = 0.85
    sample_raw_record["evidence"][2]["status"] = "coherent"
    sample_raw_record["facts"]["share"] = 0.85
    sample_raw_record["facts"]["coherent"] = True
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.PHYSICAL_EVENT
    assert proposal.action == Action.COORDINATE_WITH_GRID_OPERATOR
    assert "neighborhood.share" in proposal.evidence_ids


def test_rule_validator_tampering_depressed_load(sample_raw_record: dict[str, Any]):
    """36. Tampering (depressed load + identities hold + normal neighbors)."""
    # identities hold, neighbors isolated, z = -3.2

    sample_raw_record["evidence"][1]["z"] = -3.2
    sample_raw_record["facts"]["z"] = -3.2
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.ADVERSARIAL
    assert proposal.action == Action.INVESTIGATE_TAMPERING
    assert "baseline.z_score" in proposal.evidence_ids


def test_rule_validator_ambiguous_evidence(sample_raw_record: dict[str, Any]):
    """37. Ambiguous or normal evidence -> FLAG_DATA_SUSPECT / UNKNOWN."""
    sample_raw_record["evidence"][1]["z"] = -0.5
    sample_raw_record["facts"]["z"] = -0.5
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert proposal.origin == DecisionOrigin.UNKNOWN
    assert proposal.action == Action.FLAG_DATA_SUSPECT


def test_rule_validator_citations_are_valid(sample_raw_record: dict[str, Any]):
    """38. All cited evidence IDs exist in view."""
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    validate_citations(proposal, view)


def test_rule_validator_citations_non_empty(sample_raw_record: dict[str, Any]):
    """39. Proposal evidence_ids is non-empty."""
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert len(proposal.evidence_ids) > 0


def test_rule_validator_action_in_action_enum(sample_raw_record: dict[str, Any]):
    """40. Action belongs to closed Action enum."""
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert isinstance(proposal.action, Action)


def test_rule_validator_origin_in_decision_origin(sample_raw_record: dict[str, Any]):
    """41. Origin belongs to DecisionOrigin enum."""
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert isinstance(proposal.origin, DecisionOrigin)


def test_rule_validator_support_bounded(sample_raw_record: dict[str, Any]):
    """42. Support is strictly bounded in [0, 1]."""
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert 0.0 <= proposal.support <= 1.0


def test_rule_validator_rationale_non_empty(sample_raw_record: dict[str, Any]):
    """43. Rationale is non-empty string."""
    view = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal = validator.propose(view, round=1)

    assert isinstance(proposal.rationale, str)
    assert len(proposal.rationale) > 10


def test_rule_validator_rejects_wrong_role_view():
    """44. AgentView with role != VALIDATOR is rejected."""
    analyst_view = AgentView(
        anomaly_id="CASE_TEST",
        role=AgentRole.ANALYST,
        evidence=EvidenceCatalog(
            [
                EvidenceItem(
                    id="physics.current_vs_power_identity",
                    source="physics",
                    description="p",
                    value=0.001,
                )
            ]
        ),
    )
    validator = RuleBasedValidator()
    with pytest.raises(AgentContractError, match="role=VALIDATOR"):
        validator.propose(analyst_view, round=1)


def test_rule_validator_rejects_empty_evidence_view():
    """45. AgentView with empty evidence is rejected."""
    empty_view = AgentView(
        anomaly_id="CASE_TEST",
        role=AgentRole.VALIDATOR,
        evidence=EvidenceCatalog([]),
    )
    validator = RuleBasedValidator()
    with pytest.raises(AgentContractError, match="without evidence"):
        validator.propose(empty_view, round=1)


# --------------------------------------------------------------------------- #
# 19.6 Behavioral Isolation Tests
# --------------------------------------------------------------------------- #


def test_behavioral_hidden_label_invariance(sample_raw_record: dict[str, Any]):
    """46. Changing hidden true labels does NOT change the Validator proposal."""
    view_a = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal_a = validator.propose(view_a, round=1)

    mutated = dict(sample_raw_record)
    mutated["true_type"] = "physical_collapse"
    mutated["true_origin"] = "physical_event"
    mutated["ground_truth"] = {"type": "grid_loss"}
    mutated["y_true"] = 0
    mutated["target_label"] = "grid_loss"
    mutated["injected_type"] = "grid_loss"
    mutated["injected_origin"] = "physical_event"

    view_b = build_validator_view(mutated)
    proposal_b = validator.propose(view_b, round=1)

    assert proposal_a.action == proposal_b.action
    assert proposal_a.origin == proposal_b.origin
    assert proposal_a.support == proposal_b.support
    assert proposal_a.evidence_ids == proposal_b.evidence_ids
    assert proposal_a.rationale == proposal_b.rationale


def test_behavioral_model_evidence_invariance(sample_raw_record: dict[str, Any]):
    """47. Changing model-side classifier/XAI outputs does NOT change the Validator proposal."""
    view_a = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal_a = validator.propose(view_a, round=1)

    mutated = dict(sample_raw_record)
    mutated["detection"] = {"p_anomaly": 0.05, "score": -10.0}
    mutated["classification"] = {
        "origin_probs": {"physical_event": 0.99, "data_quality": 0.01},
        "type_probs": {"coherent_peak": 0.99},
    }
    mutated["explanation"] = {"shap_top": [], "lime_top": [], "reliability": 0.01}

    view_b = build_validator_view(mutated)
    proposal_b = validator.propose(view_b, round=1)

    assert proposal_a.action == proposal_b.action
    assert proposal_a.origin == proposal_b.origin
    assert proposal_a.support == proposal_b.support
    assert proposal_a.evidence_ids == proposal_b.evidence_ids


def test_behavioral_validator_evidence_sensitivity(sample_raw_record: dict[str, Any]):
    """48. Changing Validator-side physical evidence DOES change the proposal."""
    view_a = build_validator_view(sample_raw_record)
    validator = RuleBasedValidator()
    proposal_a = validator.propose(view_a, round=1)

    # Mutate neighborhood to coherent grid event
    mutated = dict(sample_raw_record)
    mutated["evidence"] = [
        {"check": "current_vs_power_identity", "rel_residual": 0.001, "status": "holds"},
        {"check": "neighborhood_share", "value": 0.90, "status": "coherent"},
        {"check": "baseline_deviation", "z": 4.5, "status": "anomalous"},
    ]
    mutated["facts"] = {"share": 0.90, "coherent": True, "identity_broken": False, "z": 4.5}

    view_b = build_validator_view(mutated)
    proposal_b = validator.propose(view_b, round=1)

    assert proposal_a.action != proposal_b.action
    assert proposal_a.origin != proposal_b.origin
    assert proposal_b.action == Action.COORDINATE_WITH_GRID_OPERATOR
    assert proposal_b.origin == DecisionOrigin.PHYSICAL_EVENT


def test_behavioral_unverified_history_invariance(sample_raw_record: dict[str, Any]):
    """49. Changing unverified historical cases does NOT alter the proposal."""
    validator = RuleBasedValidator()
    view_a = build_validator_view(sample_raw_record, history=[])
    prop_a = validator.propose(view_a, round=1)

    unverified_pool = [
        Case(
            anomaly_id=f"UNV_{i}",
            signature=[0.5] * 64,
            type_pred="tampering",
            origin_pred="adversarial",
            status="human_labelled",  # Not verified
            timestamp="2020-05-01T00:00:00Z",
        )
        for i in range(10)
    ]

    view_b = build_validator_view(sample_raw_record, history=unverified_pool)
    prop_b = validator.propose(view_b, round=1)

    assert prop_a.action == prop_b.action
    assert prop_a.origin == prop_b.origin
    assert prop_a.support == prop_b.support


def test_behavioral_future_history_invariance(sample_raw_record: dict[str, Any]):
    """50. Adding future verified matches does NOT alter the proposal."""
    validator = RuleBasedValidator()
    view_a = build_validator_view(sample_raw_record, history=[])
    prop_a = validator.propose(view_a, round=1)

    future_match = [
        Match(
            case=Case(
                anomaly_id="FUT_1",
                signature=[0.5] * 64,
                type_pred="dead_sensor",
                origin_pred="data_quality",
                status="verified",
                timestamp="2026-12-31T23:59:59Z",
            ),
            similarity=0.99,
        )
    ]

    view_b = build_validator_view(sample_raw_record, history=future_match)
    prop_b = validator.propose(view_b, round=1)

    assert prop_a.action == prop_b.action
    assert prop_a.origin == prop_b.origin
    assert prop_a.support == prop_b.support


# --------------------------------------------------------------------------- #
# 20. Analyst / Validator Asymmetry Test
# --------------------------------------------------------------------------- #


def test_analyst_validator_evidence_asymmetry(sample_raw_record: dict[str, Any]):
    """51. Analyst and Validator AgentViews do NOT expose the same primary evidence.

    Asymmetry invariant:
    - Analyst exposes model-side predictions (model.*, xai.*).
    - Validator exposes physical, contextual, and historical checks
      (physics.*, baseline.*, neighborhood.*).
    - Their primary evidence sets are completely disjoint.
    - Overlap is strictly restricted to non-causal case metadata
      (anomaly_id, household_id, timestamps).
    """
    analyst_view = build_analyst_view(sample_raw_record)

    validator_view = build_validator_view(sample_raw_record)

    analyst_ids = set(analyst_view.evidence.ids)
    validator_ids = set(validator_view.evidence.ids)

    # Primary evidence IDs must be completely disjoint
    overlap = analyst_ids.intersection(validator_ids)
    assert overlap == set(), f"Found unexpected evidence ID overlap: {overlap}"

    # Analyst must contain model-side evidence
    assert any(eid.startswith("model.origin.") for eid in analyst_ids)
    assert any(eid.startswith("xai.") for eid in analyst_ids)

    # Validator must contain physical/contextual evidence
    assert any(eid.startswith("physics.") for eid in validator_ids)
    assert any(eid.startswith("neighborhood.") for eid in validator_ids)
    assert any(eid.startswith("baseline.") for eid in validator_ids)

    # Metadata consistency
    assert analyst_view.anomaly_id == validator_view.anomaly_id
    assert analyst_view.case_metadata.household_id == validator_view.case_metadata.household_id
    assert analyst_view.case_metadata.t_start == validator_view.case_metadata.t_start


# --------------------------------------------------------------------------- #
# Constants & Helper Tests
# --------------------------------------------------------------------------- #


def test_validator_constants():
    """52. Verify core forbidden keys and named baseline thresholds."""
    assert "true_origin" in FORBIDDEN_VALIDATOR_KEYS
    assert "type_probs" in FORBIDDEN_VALIDATOR_KEYS
    assert "shap_top" in FORBIDDEN_VALIDATOR_KEYS
    assert TAU_SIM_DEFAULT == 0.90
    assert NEIGHBORHOOD_COHERENT_THRESHOLD == 0.40
    assert PHYSICS_IDENTITY_BROKEN_THRESHOLD == 0.01
    assert PHYSICS_IDENTITY_SEVERE_THRESHOLD == 0.10


def test_hydrate_historical_case_with_audit_log(tmp_path):
    """53. Hydrate historical summary looks up resolution/outcome in AuditLog."""
    log = AuditLog(sqlite_path=tmp_path / "audit.db")
    log.append_event("detector", "proposed", {"type": "gain_error"}, anomaly_id="A100")
    log.append_event("agent_1", "decision", {"action": "request_meter_reread"}, anomaly_id="A100")
    log.append_event(
        "operator",
        "outcome",
        {"status": "verified", "resolution": "meter_recalibrated"},
        anomaly_id="A100",
    )

    case = Case(
        anomaly_id="A100",
        signature=[0.1] * 64,
        type_pred="gain_error",
        origin_pred="data_quality",
        status="verified",
        timestamp="2020-04-01T00:00:00Z",
    )
    summary = hydrate_historical_case(case, audit_log=log)

    assert summary is not None
    assert summary.anomaly_id == "A100"
    assert summary.resolution == "meter_recalibrated"
    assert summary.outcome == "verified"
