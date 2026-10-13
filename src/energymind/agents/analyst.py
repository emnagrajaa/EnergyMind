"""M7.2 - Analyst agent and model-side evidence boundary.

The Analyst agent represents the model side of the EnergyMind deliberation architecture:
it reasons strictly from detector outputs, classifier probability distributions,
explanation attributions (SHAP/LIME), and model-derived severity.

Design Principles:
- MODEL-SIDE ONLY: The Analyst answers "What does the model layer think happened?".
  It does NOT validate physical truth (which is the Validator's role in M7.3).

- STRICT EVIDENCE SEPARATION: It never receives hidden ground truth (true_type, true_origin),
  Validator-side physical residuals, neighborhood coherence, or historical memory.
- NO RAW RECORD LEAKAGE: AgentView exposes only controlled EvidenceItem objects via EvidenceCatalog.
- SELF-SUPPORT != SYSTEM CONFIDENCE: Proposal.support reflects agent self-assessed support [0, 1],
  distinct from M8 calibrated system confidence.
- CONSERVATIVE UNCERTAINTY: Ambiguous distributions or low anomaly probability resolve to
  DecisionOrigin.UNKNOWN rather than guessing.
"""

from __future__ import annotations

import builtins
from typing import Any

from energymind.agents.base import (
    Action,
    AgentContractError,
    AgentRole,
    AgentView,
    CaseMetadata,
    DecisionOrigin,
    EvidenceCatalog,
    EvidenceItem,
    Proposal,
    prompt_hash,
    validate_citations,
)

# --------------------------------------------------------------------------- #
# Security Boundary: Forbidden Analyst Keys
# --------------------------------------------------------------------------- #

FORBIDDEN_ANALYST_KEYS: frozenset[str] = frozenset(
    {
        # Ground truth / simulation labels
        "true_type",
        "true_origin",
        "ground_truth",
        "y_true",
        "target",
        "target_label",
        "injected_type",
        "injected_origin",
        "injection_label",
        "injected_variant",
        "injected_severity",
        "injected_params",
        # Validator-side physical consistency / residuals
        "current_vs_power_identity",
        "baseline_deviation",
        "neighborhood_share",
        "neigh_share",
        "nilm_residual_total",
        "nilm_resid_total",
        "sub_meter_share_cooling",
        "physics_identity_residual",
        "r_S",
        "r_VI",
        "r_E",
        "r_PF",
        "r_sub",
        # Validator-side memory / historical outcomes / human decisions
        "verified_history",
        "historical_outcome",
        "operator_resolution",
        "previous_decision",
        "human_decision",
        "qdrant_matches",
        "seen_before",
    }
)


def assert_no_leakage(view: AgentView) -> None:
    """Defensive verification that an AgentView contains zero forbidden keys."""
    dumped = view.model_dump()
    _check_recursive_keys(dumped)


def _check_recursive_keys(obj: Any) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_ANALYST_KEYS:
                raise AgentContractError(f"Forbidden key '{k}' found in AgentView")
            _check_recursive_keys(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _check_recursive_keys(item)


# --------------------------------------------------------------------------- #
# Thresholds for Deterministic Baseline Analyst Logic
# --------------------------------------------------------------------------- #

#: Anomaly detector probability threshold below which no anomaly action is taken.
P_ANOMALY_LOW_THRESHOLD: float = 0.50

#: Minimum top-class probability required to treat an origin as strongly supported.
ORIGIN_STRONG_PROB_THRESHOLD: float = 0.60

#: Minimum margin between top-1 and top-2 origin probabilities required for confidence.
ORIGIN_STRONG_MARGIN_THRESHOLD: float = 0.20

#: Anomaly probability threshold for escalating data quality failure to quarantine.
P_ANOMALY_QUARANTINE_THRESHOLD: float = 0.85

#: Origin probability threshold for escalating data quality failure to quarantine.
ORIGIN_QUARANTINE_THRESHOLD: float = 0.80


# --------------------------------------------------------------------------- #
# Analyst View Builder
# --------------------------------------------------------------------------- #


def build_analyst_view(record: dict[str, Any]) -> AgentView:
    """Build a strictly filtered, model-side AgentView from an anomaly record.

    Extracts ONLY:
    - detection: p_anomaly, score
    - classification: type_probs, origin_probs
    - explanation: shap_top, lime_top, reliability
    - model-side severity
    - safe case metadata (household_id, timestamps, duration)

    Excludes all ground truth, physical residuals, neighborhood coherence, and
    historical outcomes.
    """
    anomaly_id = str(record.get("anomaly_id", "")).strip()
    if not anomaly_id:
        raise ValueError("record must contain a non-empty 'anomaly_id'")

    # Safe case metadata
    case_meta = CaseMetadata(
        household_id=record.get("household_id"),
        t_start=str(record.get("t_start")) if record.get("t_start") is not None else None,
        t_end=str(record.get("t_end")) if record.get("t_end") is not None else None,
        duration_h=float(record["duration_h"]) if record.get("duration_h") is not None else None,
    )

    items: list[EvidenceItem] = []
    seen_ids: set[str] = set()

    # 1. Detection evidence
    detection = record.get("detection") or {}
    det_model = str(detection.get("model", "detector"))
    if "p_anomaly" in detection and detection["p_anomaly"] is not None:
        p_val = float(detection["p_anomaly"])
        eid = "model.p_anomaly"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source=f"detector.{det_model}",
                description="Anomaly detector probability",
                value=round(p_val, 4),
            )
        )
    if "score" in detection and detection["score"] is not None:
        score_val = float(detection["score"])
        eid = "model.detector_score"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source=f"detector.{det_model}",
                description="Raw anomaly detector score",
                value=round(score_val, 4),
            )
        )

    # 2. Classification evidence (sorted for determinism)
    classification = record.get("classification") or {}
    clf_model = str(classification.get("model", "classifier"))

    type_probs = classification.get("type_probs") or {}
    for t_name, t_prob in sorted(type_probs.items()):
        if t_prob is not None:
            eid = f"model.type.{t_name}"
            if eid not in seen_ids:
                seen_ids.add(eid)
                items.append(
                    EvidenceItem(
                        id=eid,
                        source=f"classifier.{clf_model}",
                        description=f"Predicted probability of anomaly type '{t_name}'",
                        value=round(float(t_prob), 4),
                    )
                )

    origin_probs = classification.get("origin_probs") or {}
    for o_name, o_prob in sorted(origin_probs.items()):
        if o_prob is not None:
            eid = f"model.origin.{o_name}"
            if eid not in seen_ids:
                seen_ids.add(eid)
                items.append(
                    EvidenceItem(
                        id=eid,
                        source=f"classifier.{clf_model}",
                        description=f"Predicted probability of decision origin '{o_name}'",
                        value=round(float(o_prob), 4),
                    )
                )

    # 3. Explainability evidence (SHAP / LIME / reliability)
    explanation = record.get("explanation") or {}

    shap_top = explanation.get("shap_top") or []
    for entry in shap_top:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            feat_name, feat_val = entry[0], entry[1]
            eid = f"xai.shap.{feat_name}"
            if eid not in seen_ids:
                seen_ids.add(eid)
                items.append(
                    EvidenceItem(
                        id=eid,
                        source="xai.shap",
                        description=f"SHAP feature attribution for '{feat_name}'",
                        value=round(float(feat_val), 5),
                    )
                )

    lime_top = explanation.get("lime_top") or []
    for idx, entry in enumerate(lime_top):
        if isinstance(entry, str):
            feat_name = entry
            feat_val: Any = idx + 1
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            feat_name = entry[0]
            feat_val = round(float(entry[1]), 5)
        else:
            continue
        eid = f"xai.lime.{feat_name}"
        if eid not in seen_ids:
            seen_ids.add(eid)
            items.append(
                EvidenceItem(
                    id=eid,
                    source="xai.lime",
                    description=f"LIME salient feature '{feat_name}'",
                    value=feat_val,
                )
            )

    reliability = explanation.get("reliability")
    if reliability is not None:
        eid = "xai.reliability"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="xai.agreement",
                description=(
                    "Explanation reliability: mean Jaccard agreement "
                    "between top SHAP and LIME features"
                ),
                value=round(float(reliability), 4),
            )
        )

    # 4. Model-derived severity
    severity = record.get("severity")
    if severity is not None:
        eid = "model.severity"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="model.assessment",
                description="Predicted anomaly severity level",
                value=str(severity),
            )
        )

    view = AgentView(
        anomaly_id=anomaly_id,
        role=AgentRole.ANALYST,
        evidence=EvidenceCatalog(items),
        case_metadata=case_meta,
    )

    # Defensive check
    assert_no_leakage(view)
    return view


# --------------------------------------------------------------------------- #
# Rule-Based Analyst Agent
# --------------------------------------------------------------------------- #


class RuleBasedAnalyst:
    """Deterministic, rule-based baseline implementation of Agent for the Analyst role.

    Acts as an offline baseline, ablation model, and fallback executor.
    Reasons strictly from visible AgentView evidence.
    """

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
        """Formulate a decision proposal using deterministic rules over model evidence."""
        if view.role != AgentRole.ANALYST:
            raise AgentContractError(
                f"RuleBasedAnalyst requires an AgentView with role=ANALYST, got {view.role}"
            )

        # 1. Extract p_anomaly if present
        p_item = view.evidence.get("model.p_anomaly")
        p_anomaly = float(p_item.value) if p_item is not None and p_item.value is not None else None

        # 2. Extract origin probabilities
        origin_prefix = "model.origin."
        origin_probs: dict[str, float] = {}
        for item in view.evidence:
            if item.id.startswith(origin_prefix) and item.value is not None:
                orig_name = item.id[len(origin_prefix) :]
                origin_probs[orig_name] = float(item.value)

        # Branch 1: Low anomaly evidence -> conservative NO_ACTION
        if p_anomaly is not None and p_anomaly < P_ANOMALY_LOW_THRESHOLD:
            action = Action.NO_ACTION
            origin = DecisionOrigin.UNKNOWN
            evidence_ids = ["model.p_anomaly"]
            support = builtins.round(1.0 - p_anomaly, 4)
            rationale = (
                f"Anomaly detector probability ({p_anomaly:.2f}) is below detection "
                f"threshold ({P_ANOMALY_LOW_THRESHOLD}); no action needed."
            )

        # Branch 2: Origin probabilities missing or empty
        elif not origin_probs:
            action = Action.FLAG_DATA_SUSPECT
            origin = DecisionOrigin.UNKNOWN
            evidence_ids = ["model.p_anomaly"] if p_item else list(view.evidence.ids)[:1]
            if not evidence_ids:
                raise AgentContractError("Analyst cannot formulate a proposal without evidence")
            support = builtins.round(p_anomaly * 0.5, 4) if p_anomaly is not None else 0.5
            rationale = (
                "Anomaly detected but origin classification probabilities are unavailable; "
                "flagging data as suspect."
            )

        # Branch 3: Analyze origin distribution
        else:
            sorted_origins = sorted(origin_probs.items(), key=lambda kv: kv[1], reverse=True)
            top_orig, top_prob = sorted_origins[0]
            second_prob = sorted_origins[1][1] if len(sorted_origins) > 1 else 0.0
            margin = top_prob - second_prob

            is_ambiguous = (
                top_prob < ORIGIN_STRONG_PROB_THRESHOLD or margin < ORIGIN_STRONG_MARGIN_THRESHOLD
            )

            if is_ambiguous:
                action = Action.FLAG_DATA_SUSPECT
                origin = DecisionOrigin.UNKNOWN
                evidence_ids = []
                if p_item:
                    evidence_ids.append("model.p_anomaly")
                evidence_ids.append(f"model.origin.{top_orig}")
                if len(sorted_origins) > 1:
                    evidence_ids.append(f"model.origin.{sorted_origins[1][0]}")
                support = builtins.round((p_anomaly if p_anomaly is not None else 0.7) * 0.5, 4)
                rationale = (
                    f"Anomaly detected but origin distribution is ambiguous between top candidates "
                    f"({top_orig}={top_prob:.2f}, margin={margin:.2f}); flagging as suspect."
                )

            elif top_orig == "data_quality":
                origin = DecisionOrigin.DATA_QUALITY
                sev_item = view.evidence.get("model.severity")
                sev_str = (
                    str(sev_item.value).lower()
                    if sev_item is not None and sev_item.value is not None
                    else None
                )

                if sev_str in ("high", "critical"):
                    is_severe = True
                elif sev_str in ("low", "medium", "info"):
                    is_severe = False
                else:
                    is_severe = (
                        p_anomaly is not None
                        and p_anomaly >= P_ANOMALY_QUARANTINE_THRESHOLD
                        and top_prob >= ORIGIN_QUARANTINE_THRESHOLD
                    )

                if is_severe:
                    action = Action.QUARANTINE_AND_ESTIMATE
                    evidence_ids = [f"model.origin.{top_orig}"]
                    if p_item:
                        evidence_ids.insert(0, "model.p_anomaly")
                    if sev_item:
                        evidence_ids.append("model.severity")
                    rationale = (
                        f"Strong data-quality failure indication ({top_orig}={top_prob:.2f}) "
                        "with high severity; recommending quarantine and estimation."
                    )
                else:
                    action = Action.FLAG_DATA_SUSPECT
                    evidence_ids = [f"model.origin.{top_orig}"]
                    if p_item:
                        evidence_ids.insert(0, "model.p_anomaly")
                    if sev_item:
                        evidence_ids.append("model.severity")
                    rationale = (
                        f"Strong data-quality failure indication ({top_orig}={top_prob:.2f}); "
                        "flagging data as suspect."
                    )

                p_factor = p_anomaly if p_anomaly is not None else 1.0
                support = builtins.round(top_prob * p_factor, 4)

            elif top_orig == "physical_event":
                origin = DecisionOrigin.PHYSICAL_EVENT
                action = Action.COORDINATE_WITH_GRID_OPERATOR
                evidence_ids = [f"model.origin.{top_orig}"]
                if p_item:
                    evidence_ids.insert(0, "model.p_anomaly")
                p_factor = p_anomaly if p_anomaly is not None else 1.0
                support = builtins.round(top_prob * p_factor, 4)
                rationale = (
                    f"Model strongly predicts physical grid event ({top_orig}={top_prob:.2f}); "
                    "coordinating with grid operator."
                )

            elif top_orig == "adversarial":
                origin = DecisionOrigin.ADVERSARIAL
                action = Action.INVESTIGATE_TAMPERING
                evidence_ids = [f"model.origin.{top_orig}"]
                if p_item:
                    evidence_ids.insert(0, "model.p_anomaly")
                p_factor = p_anomaly if p_anomaly is not None else 1.0
                support = builtins.round(top_prob * p_factor, 4)
                rationale = (
                    f"Model predicts potential adversarial manipulation or tampering "
                    f"({top_orig}={top_prob:.2f}); initiating investigation."
                )

            else:
                action = Action.FLAG_DATA_SUSPECT
                origin = DecisionOrigin.UNKNOWN
                evidence_ids = [f"model.origin.{top_orig}"]
                if p_item:
                    evidence_ids.insert(0, "model.p_anomaly")
                support = builtins.round(top_prob * 0.5, 4)
                rationale = f"Unmapped origin '{top_orig}'; flagging as suspect."

        # Guard support range
        support = max(0.0, min(1.0, support))

        proposal = Proposal(
            agent=AgentRole.ANALYST,
            round=round,
            action=action,
            origin=origin,
            support=support,
            evidence_ids=evidence_ids,
            rationale=rationale,
            backend="rule_based",
            model="rules:analyst_v1",
            prompt_hash=prompt_hash("rule_based_analyst"),
            latency_ms=0.0,
            fallback_used=False,
        )

        validate_citations(proposal, view)
        return proposal
