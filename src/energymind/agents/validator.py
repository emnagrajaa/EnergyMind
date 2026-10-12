"""M7.3 - Validator agent and independent physical/historical evidence boundary.

The Validator agent represents the independent physical, contextual, and historical
trust layer in the EnergyMind deliberation architecture:
- It answers: "Does independent physical, contextual, and historical evidence support
  the interpretation of this anomaly?"
- It is NOT a second classifier. It does not repeat detector scores or classifier
  probabilities.
- Asymmetry is strictly preserved: the Validator examines physical residuals (e.g. V*I vs S),
  baseline departures, neighborhood coherence, NILM residuals, sensor diagnostics, and
  verified historical precedents.
- Strict isolation from hidden ground truth (true_type, true_origin, y_true, etc.).
- Strict temporal integrity: future events and unverified cases are excluded from memory;
  current incident cannot match itself.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from datetime import datetime
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
from energymind.retrieval import Case, Match

# --------------------------------------------------------------------------- #
# Security Boundary: Forbidden Validator Keys
# --------------------------------------------------------------------------- #

FORBIDDEN_VALIDATOR_KEYS: frozenset[str] = frozenset(
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
        # Model-side interpretation (Analyst-side in blind round 1)
        "type_probs",
        "origin_probs",
        "p_anomaly",
        "detector_score",
        "shap_top",
        "lime_top",
        "reliability",
        "model.type",
        "model.origin",
        "xai.shap",
        "xai.lime",
        "xai.reliability",
        # Analyst opinion / proposals
        "analyst_proposal",
        "analyst_rationale",
        "analyst_support",
        "analyst_action",
        "analyst_origin",
        "other_proposal",
    }
)

FORBIDDEN_VALIDATOR_ID_PREFIXES: tuple[str, ...] = (
    "model.type.",
    "model.origin.",
    "model.p_anomaly",
    "model.detector_score",
    "xai.shap.",
    "xai.lime.",
    "xai.reliability",
    "analyst.",
)


def assert_no_validator_leakage(view: AgentView) -> None:
    """Defensive verification that a Validator AgentView contains zero forbidden keys."""
    dumped = view.model_dump()
    _check_recursive_keys(dumped)

    # Check evidence IDs and sources
    for item in view.evidence:
        for prefix in FORBIDDEN_VALIDATOR_ID_PREFIXES:
            if item.id.startswith(prefix):
                raise AgentContractError(
                    f"Forbidden model-side evidence ID '{item.id}' found in Validator view"
                )
        if (
            item.source.startswith("detector.")
            or item.source.startswith("classifier.")
            or item.source.startswith("xai.")
        ):
            raise AgentContractError(
                f"Forbidden model-side source '{item.source}' found in Validator view"
            )


def _check_recursive_keys(obj: Any) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_VALIDATOR_KEYS:
                raise AgentContractError(f"Forbidden key '{k}' found in Validator AgentView")
            _check_recursive_keys(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _check_recursive_keys(item)


# --------------------------------------------------------------------------- #
# Named Thresholds for Deterministic Baseline Validator Logic
# --------------------------------------------------------------------------- #

#: Default similarity threshold for precedent retrieval (matches configs/store.yaml).
TAU_SIM_DEFAULT: float = 0.90

#: Neighborhood affected fraction required to indicate coherent grid-level event.
NEIGHBORHOOD_COHERENT_THRESHOLD: float = 0.40

#: Relative electrical identity residual threshold above which identity is broken.
PHYSICS_IDENTITY_BROKEN_THRESHOLD: float = 0.01

#: Severe electrical identity residual threshold triggering quarantine.
PHYSICS_IDENTITY_SEVERE_THRESHOLD: float = 0.10

#: Baseline z-score magnitude required to indicate anomalous consumption.
BASELINE_ANOMALOUS_Z_THRESHOLD: float = 2.0

#: Fraction of zero readings in window indicating sensor dropout.
SENSOR_ZERO_FRACTION_THRESHOLD: float = 0.50

#: Minimum consecutive flat active power intervals indicating stuck sensor.
SENSOR_FLAT_RUN_THRESHOLD: int = 8


# --------------------------------------------------------------------------- #
# Historical Case Summary & Memory Adapter
# --------------------------------------------------------------------------- #


@dataclass
class HistoricalCaseSummary:
    """Safe, compact summary of a verified historical precedent.

    Excludes raw vector payloads, embeddings, and unverified predictions.
    """

    anomaly_id: str
    similarity: float
    type: str
    origin: str
    status: str = "verified"
    severity: str | None = None
    resolution: str | None = None
    outcome: str | None = None
    timestamp: str | None = None


def eligible_history(
    case: Case | Match | HistoricalCaseSummary | dict,
    current_anomaly_id: str,
    current_timestamp: str | datetime | None = None,
    tau_sim: float = TAU_SIM_DEFAULT,
) -> bool:
    """Evaluate whether a historical candidate satisfies memory safety invariants.

    Rules:
    1. Status must be 'verified'. Unverified cases are excluded.
    2. Anomaly ID must not equal current_anomaly_id (prevents test case self-matching).
    3. If historical timestamp and current timestamp exist, historical must precede current.
    4. Similarity (if present) must be >= tau_sim.
    """
    # 1. Extract status
    status = getattr(case, "status", None)
    if status is None and isinstance(case, Match):
        status = getattr(case.case, "status", None)
    elif status is None and isinstance(case, dict):
        status = case.get("status")

    if status != "verified":
        return False

    # 2. Check anomaly_id
    anomaly_id = getattr(case, "anomaly_id", None)
    if anomaly_id is None and isinstance(case, Match):
        anomaly_id = getattr(case.case, "anomaly_id", None)
    elif anomaly_id is None and isinstance(case, dict):
        anomaly_id = case.get("anomaly_id")

    if str(anomaly_id).strip() == str(current_anomaly_id).strip():
        return False

    # 3. Check similarity
    sim = getattr(case, "similarity", None)
    if sim is None and isinstance(case, dict):
        sim = case.get("similarity")
    if sim is not None and float(sim) < tau_sim:
        return False

    # 4. Check temporal ordering
    hist_ts = _extract_timestamp(case)
    curr_ts = _parse_timestamp(current_timestamp)
    return not (hist_ts is not None and curr_ts is not None and hist_ts >= curr_ts)


def _extract_timestamp(case: Any) -> datetime | None:
    if isinstance(case, Match):
        c = case.case
        ts_val = getattr(c, "timestamp", None) or (
            c.record.get("t_start") if isinstance(c.record, dict) else None
        )
        return _parse_timestamp(ts_val)
    if isinstance(case, (Case, HistoricalCaseSummary)):
        ts_val = getattr(case, "timestamp", None)
        if not ts_val and hasattr(case, "record") and isinstance(case.record, dict):
            ts_val = case.record.get("t_start")
        return _parse_timestamp(ts_val)
    if isinstance(case, dict):
        ts_val = case.get("timestamp") or case.get("t_start")
        return _parse_timestamp(ts_val)
    return None


def _parse_timestamp(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        s = str(val).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def hydrate_historical_case(
    item: Case | Match | HistoricalCaseSummary | dict,
    audit_log: Any = None,
) -> HistoricalCaseSummary | None:
    """Hydrate a safe HistoricalCaseSummary from a Case or Match without exposing raw payloads."""
    if isinstance(item, HistoricalCaseSummary):
        return item

    if isinstance(item, Match):
        case = item.case
        sim = float(item.similarity)
    elif isinstance(item, Case):
        case = item
        sim = 1.0
    elif isinstance(item, dict):
        return HistoricalCaseSummary(
            anomaly_id=str(item.get("anomaly_id", "")),
            similarity=float(item.get("similarity", 1.0)),
            type=str(item.get("type") or item.get("type_pred", "")),
            origin=str(item.get("origin") or item.get("origin_pred", "")),
            status=str(item.get("status", "verified")),
            severity=str(item["severity"]) if item.get("severity") else None,
            resolution=str(item["resolution"]) if item.get("resolution") else None,
            outcome=str(item["outcome"]) if item.get("outcome") else None,
            timestamp=str(item["timestamp"])
            if item.get("timestamp")
            else str(item["t_start"])
            if item.get("t_start")
            else None,
        )
    else:
        return None

    # Lookup audit log if available
    resolution: str | None = None
    outcome: str | None = None
    if audit_log is not None and hasattr(audit_log, "events"):
        for ev in audit_log.events():
            if ev.get("anomaly_id") == case.anomaly_id:
                if ev.get("event_type") == "outcome":
                    payload = ev.get("payload", {})
                    outcome = payload.get("status", "verified")
                    if "resolution" in payload:
                        resolution = payload["resolution"]
                elif ev.get("event_type") == "decision":
                    payload = ev.get("payload", {})
                    resolution = payload.get("action")

    ts_str = getattr(case, "timestamp", None)
    if not ts_str and isinstance(case.record, dict):
        ts_str = case.record.get("t_start")

    return HistoricalCaseSummary(
        anomaly_id=str(case.anomaly_id),
        similarity=sim,
        type=str(case.type_pred),
        origin=str(case.origin_pred),
        status=str(case.status),
        severity=str(case.severity) if case.severity else None,
        resolution=resolution,
        outcome=outcome,
        timestamp=str(ts_str) if ts_str else None,
    )


# --------------------------------------------------------------------------- #
# Validator View Builder
# --------------------------------------------------------------------------- #


def build_validator_view(
    record: dict[str, Any],
    history: list[Case | Match | HistoricalCaseSummary | dict] | None = None,
    tau_sim: float = TAU_SIM_DEFAULT,
    current_timestamp: str | datetime | None = None,
    audit_log: Any = None,
) -> AgentView:
    """Build a strictly filtered, independent AgentView for the Validator agent.

    Extracts:
    - Physics checks: current_vs_power_identity, apparent/voltage/energy identity residuals
    - Baseline checks: baseline.z_score, observed_power, expected_power
    - Neighborhood coherence: neighborhood.share
    - NILM checks: nilm.residual_total, nilm.sub_meter_share_cooling
    - Sensor diagnostics: sensor.zero_reading, sensor.zero_fraction, sensor.flat_run_length,
      sensor.voltage_deviation
    - Eligible verified historical precedents (history.0.*)
    - Safe case metadata (household_id, timestamps, duration)

    Excludes all classifier probabilities, detector scores, SHAP/LIME attributions,
    and hidden ground truth.
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

    # Helpers to extract checks from record["evidence"] or top-level/facts
    raw_evidence_list = record.get("evidence") or []
    checks_by_name: dict[str, dict] = {}
    if isinstance(raw_evidence_list, list):
        for entry in raw_evidence_list:
            if isinstance(entry, dict) and "check" in entry:
                checks_by_name[entry["check"]] = entry

    facts = record.get("facts") or {}

    # 1. Physics Evidence: Current vs Power electrical identity
    ident_entry = checks_by_name.get("current_vs_power_identity") or {}
    ident_top = record.get("current_vs_power_identity")
    if isinstance(ident_top, dict):
        ident_entry = {**ident_entry, **ident_top}

    rel_res = (
        ident_entry.get("rel_residual")
        if "rel_residual" in ident_entry
        else facts.get("identity_rel")
    )
    if rel_res is None and isinstance(ident_top, (int, float)):
        rel_res = float(ident_top)

    ident_status = ident_entry.get("status")
    if ident_status is None:
        if facts.get("identity_broken"):
            ident_status = "broken"
        elif rel_res is not None:
            ident_status = (
                "broken" if float(rel_res) > PHYSICS_IDENTITY_BROKEN_THRESHOLD else "holds"
            )

    if rel_res is not None or ident_status is not None:
        eid = "physics.current_vs_power_identity"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="physics.residuals",
                description="V*I vs S electrical conservation identity check",
                value=builtins.round(float(rel_res), 4) if rel_res is not None else 0.0,
                unit="relative",
                status=str(ident_status) if ident_status else "holds",
            )
        )

    # Secondary physics residuals if present
    for r_key in ("r_S", "r_VI", "r_E", "r_PF", "r_sub", "physics_identity_residual"):
        val = record.get(r_key)
        if val is not None:
            eid = f"physics.{r_key}"
            if eid not in seen_ids:
                seen_ids.add(eid)
                items.append(
                    EvidenceItem(
                        id=eid,
                        source="physics.residuals",
                        description=f"Electrical identity residual '{r_key}'",
                        value=builtins.round(float(val), 4),
                        unit="relative",
                    )
                )

    # 2. Baseline Evidence
    base_entry = checks_by_name.get("baseline_deviation") or {}
    z_val = base_entry.get("z") if "z" in base_entry else facts.get("z")
    if z_val is None and "p_resid_z" in record:
        z_val = record["p_resid_z"]

    if z_val is not None:
        eid = "baseline.z_score"
        seen_ids.add(eid)
        z_status = base_entry.get("status") or (
            "anomalous" if abs(float(z_val)) >= BASELINE_ANOMALOUS_Z_THRESHOLD else "normal"
        )
        items.append(
            EvidenceItem(
                id=eid,
                source="baseline.model",
                description="Departure from hour-of-week baseline (z-score)",
                value=builtins.round(float(z_val), 2),
                status=str(z_status),
            )
        )

    obs_val = base_entry.get("observed_kw") or facts.get("observed_kw")
    if obs_val is not None:
        eid = "baseline.observed_power"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="baseline.model",
                description="Observed active power median",
                value=builtins.round(float(obs_val), 3),
                unit="kW",
            )
        )

    exp_val = base_entry.get("expected_kw") or facts.get("expected_kw")
    if exp_val is not None:
        eid = "baseline.expected_power"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="baseline.model",
                description="Expected active power baseline",
                value=builtins.round(float(exp_val), 3),
                unit="kW",
            )
        )

    # 3. Neighborhood Coherence
    neigh_entry = checks_by_name.get("neighborhood_share") or {}
    share_val = (
        neigh_entry.get("value")
        if "value" in neigh_entry
        else record.get("neighborhood_share")
        if record.get("neighborhood_share") is not None
        else record.get("neigh_share")
        if record.get("neigh_share") is not None
        else facts.get("share")
    )
    if share_val is not None:
        eid = "neighborhood.share"
        seen_ids.add(eid)
        share_f = float(share_val)
        status_str = neigh_entry.get("status") or (
            "coherent" if share_f >= NEIGHBORHOOD_COHERENT_THRESHOLD else "isolated"
        )
        items.append(
            EvidenceItem(
                id=eid,
                source="neighborhood.aggregator",
                description="Share of neighboring households with concurrent excursion",
                value=builtins.round(share_f, 3),
                unit="ratio",
                status=str(status_str),
            )
        )

    # 4. NILM Evidence
    nilm_res = (
        checks_by_name.get("nilm_residual_total", {}).get("value")
        or record.get("nilm_residual_total")
        or record.get("nilm_resid_total")
        or facts.get("nilm_resid_total")
    )
    if nilm_res is not None:
        eid = "nilm.residual_total"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="nilm.residual",
                description="Total NILM equipment residual",
                value=builtins.round(float(nilm_res), 4),
                unit="kWh",
            )
        )

    cooling_share = checks_by_name.get("sub_meter_share_cooling", {}).get("value") or record.get(
        "sub_meter_share_cooling"
    )
    if cooling_share is not None:
        eid = "nilm.sub_meter_share_cooling"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="nilm.residual",
                description="Cooling equipment sub-meter share",
                value=builtins.round(float(cooling_share), 4),
                unit="ratio",
            )
        )

    # 5. Sensor Diagnostics
    zero_reading = facts.get("zero_reading") or record.get("zero_reading")
    if zero_reading is not None:
        eid = "sensor.zero_reading"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="sensor.diagnostics",
                description="Meter reports zero power while expected load is positive",
                value=bool(zero_reading),
            )
        )

    zero_frac = facts.get("zero_frac") or record.get("zero_frac")
    if zero_frac is not None:
        eid = "sensor.zero_fraction"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="sensor.diagnostics",
                description="Fraction of zero readings in window",
                value=builtins.round(float(zero_frac), 4),
                unit="ratio",
            )
        )

    flat_run = facts.get("flat_run") or record.get("flat_run")
    if flat_run is not None:
        eid = "sensor.flat_run_length"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="sensor.diagnostics",
                description="Consecutive intervals with constant active power",
                value=int(flat_run),
                unit="intervals",
            )
        )

    v_dev = facts.get("voltage_dev") or record.get("voltage_dev")
    if v_dev is not None:
        eid = "sensor.voltage_deviation"
        seen_ids.add(eid)
        items.append(
            EvidenceItem(
                id=eid,
                source="sensor.diagnostics",
                description="Voltage deviation from nominal 230V",
                value=builtins.round(float(v_dev), 2),
                unit="V",
            )
        )

    # 6. Verified Historical Precedents
    if history:
        curr_ts = current_timestamp or case_meta.t_start
        eligible_summaries: list[HistoricalCaseSummary] = []
        for cand in history:
            if eligible_history(cand, anomaly_id, current_timestamp=curr_ts, tau_sim=tau_sim):
                summary = hydrate_historical_case(cand, audit_log=audit_log)
                if summary:
                    eligible_summaries.append(summary)

        # Sort eligible cases deterministically by descending similarity
        eligible_summaries.sort(key=lambda s: s.similarity, reverse=True)

        for idx, h_case in enumerate(eligible_summaries):
            sim_eid = f"history.{idx}.similarity"
            seen_ids.add(sim_eid)
            items.append(
                EvidenceItem(
                    id=sim_eid,
                    source="memory.retriever",
                    description=f"Cosine similarity of verified precedent #{idx + 1}",
                    value=builtins.round(float(h_case.similarity), 4),
                )
            )

            type_eid = f"history.{idx}.type"
            seen_ids.add(type_eid)
            items.append(
                EvidenceItem(
                    id=type_eid,
                    source="memory.retriever",
                    description=f"Verified anomaly type of precedent #{idx + 1}",
                    value=str(h_case.type),
                )
            )

            orig_eid = f"history.{idx}.origin"
            seen_ids.add(orig_eid)
            items.append(
                EvidenceItem(
                    id=orig_eid,
                    source="memory.retriever",
                    description=f"Verified origin of precedent #{idx + 1}",
                    value=str(h_case.origin),
                )
            )

            if h_case.severity:
                sev_eid = f"history.{idx}.severity"
                seen_ids.add(sev_eid)
                items.append(
                    EvidenceItem(
                        id=sev_eid,
                        source="memory.retriever",
                        description=f"Severity of precedent #{idx + 1}",
                        value=str(h_case.severity),
                    )
                )

            if h_case.resolution:
                res_eid = f"history.{idx}.resolution"
                seen_ids.add(res_eid)
                items.append(
                    EvidenceItem(
                        id=res_eid,
                        source="memory.audit",
                        description=f"Historical resolution of precedent #{idx + 1}",
                        value=str(h_case.resolution),
                    )
                )

            if h_case.outcome:
                out_eid = f"history.{idx}.outcome"
                seen_ids.add(out_eid)
                items.append(
                    EvidenceItem(
                        id=out_eid,
                        source="memory.audit",
                        description=f"Historical outcome of precedent #{idx + 1}",
                        value=str(h_case.outcome),
                    )
                )

    view = AgentView(
        anomaly_id=anomaly_id,
        role=AgentRole.VALIDATOR,
        evidence=EvidenceCatalog(items),
        case_metadata=case_meta,
    )

    # Defensive check
    assert_no_validator_leakage(view)
    return view


# --------------------------------------------------------------------------- #
# Rule-Based Validator Agent
# --------------------------------------------------------------------------- #


class RuleBasedValidator:
    """Deterministic baseline rule-based implementation of Agent for the Validator role.

    Evaluates independent physical conservation laws, baseline consistency,
    neighborhood coherence, sensor diagnostics, and verified historical precedents.
    Does NOT use or see classifier probabilities or model-side XAI attributions.
    """

    @property
    def role(self) -> AgentRole:
        return AgentRole.VALIDATOR

    def propose(
        self,
        view: AgentView,
        round: int,
        other_proposal: Proposal | None = None,
        own_previous_proposal: Proposal | None = None,
    ) -> Proposal:
        """Formulate a decision proposal using deterministic rules over independent evidence."""
        if view.role != AgentRole.VALIDATOR:
            raise AgentContractError(
                f"RuleBasedValidator requires an AgentView with role=VALIDATOR, got {view.role}"
            )

        if not list(view.evidence.ids):
            raise AgentContractError("Validator cannot formulate a proposal without evidence")

        # Extract visible evidence
        ident_item = view.evidence.get("physics.current_vs_power_identity")
        ident_val = (
            float(ident_item.value)
            if ident_item is not None and ident_item.value is not None
            else 0.0
        )
        ident_status = ident_item.status if ident_item else "holds"
        is_ident_broken = ident_status == "broken" or ident_val > PHYSICS_IDENTITY_BROKEN_THRESHOLD

        neigh_item = view.evidence.get("neighborhood.share")
        neigh_val = (
            float(neigh_item.value)
            if neigh_item is not None and neigh_item.value is not None
            else 0.0
        )
        is_coherent = (neigh_item is not None and neigh_item.status == "coherent") or (
            neigh_val >= NEIGHBORHOOD_COHERENT_THRESHOLD
        )

        zero_reading_item = view.evidence.get("sensor.zero_reading")
        zero_frac_item = view.evidence.get("sensor.zero_fraction")
        is_zero_dropout = (zero_reading_item is not None and bool(zero_reading_item.value)) or (
            zero_frac_item is not None
            and float(zero_frac_item.value) > SENSOR_ZERO_FRACTION_THRESHOLD
        )

        flat_run_item = view.evidence.get("sensor.flat_run_length")
        is_flat_run = (
            flat_run_item is not None and int(flat_run_item.value) >= SENSOR_FLAT_RUN_THRESHOLD
        )

        z_item = view.evidence.get("baseline.z_score")
        z_val = float(z_item.value) if z_item is not None and z_item.value is not None else None

        hist_sim_item = view.evidence.get("history.0.similarity")
        hist_orig_item = view.evidence.get("history.0.origin")
        hist_sim = (
            float(hist_sim_item.value)
            if hist_sim_item is not None and hist_sim_item.value is not None
            else None
        )
        hist_orig = (
            str(hist_orig_item.value)
            if hist_orig_item is not None and hist_orig_item.value is not None
            else None
        )

        # ------------------------------------------------------------------- #
        # Rule Branching
        # ------------------------------------------------------------------- #

        # Branch 1: Long flat run -> open maintenance ticket
        if is_flat_run:
            action = Action.OPEN_MAINTENANCE_TICKET
            origin = DecisionOrigin.DATA_QUALITY
            evidence_ids = ["sensor.flat_run_length"]
            support = 0.88
            rationale = (
                f"Sensor power reading is constant for {flat_run_item.value} intervals, "
                "indicating a frozen/stuck meter sensor."
            )

        # Branch 2: Zero reading / dropout with positive context
        elif is_zero_dropout:
            action = Action.REQUEST_METER_REREAD
            origin = DecisionOrigin.DATA_QUALITY
            evidence_ids = []
            if zero_reading_item:
                evidence_ids.append("sensor.zero_reading")
            if zero_frac_item:
                evidence_ids.append("sensor.zero_fraction")
            exp_item = view.evidence.get("baseline.expected_power")
            if exp_item:
                evidence_ids.append("baseline.expected_power")
            support = 0.85
            rationale = (
                "Meter reported zero consumption while baseline expects active load; "
                "indicating an instrument dropout."
            )

        # Branch 3: Broken electrical identity
        elif is_ident_broken:
            origin = DecisionOrigin.DATA_QUALITY
            evidence_ids = ["physics.current_vs_power_identity"]
            if neigh_item:
                evidence_ids.append("neighborhood.share")

            if ident_val >= PHYSICS_IDENTITY_SEVERE_THRESHOLD:
                action = Action.QUARANTINE_AND_ESTIMATE
                support = 0.92
                rationale = (
                    f"Severe electrical identity residual ({ident_val:.4f} > "
                    f"{PHYSICS_IDENTITY_SEVERE_THRESHOLD}); quarantining corrupted measurement."
                )
            else:
                action = Action.REQUEST_METER_REREAD
                support = 0.85
                rationale = (
                    f"Electrical conservation identity broken (rel. residual {ident_val:.4f}); "
                    "requesting meter reread."
                )

        # Branch 4: Coherent neighborhood grid event
        elif is_coherent and not is_ident_broken:
            origin = DecisionOrigin.PHYSICAL_EVENT
            action = Action.COORDINATE_WITH_GRID_OPERATOR
            evidence_ids = ["neighborhood.share"]
            if z_item:
                evidence_ids.append("baseline.z_score")
            if ident_item:
                evidence_ids.append("physics.current_vs_power_identity")
            support = builtins.round(min(1.0, neigh_val + 0.10), 4)
            rationale = (
                f"Deviation is shared across neighboring meters ({neigh_val:.0%}) while "
                "electrical identities hold; indicates genuine physical grid event."
            )

        # Branch 5: Suspicious load suppression (adversarial)
        elif (
            not is_ident_broken
            and not is_coherent
            and z_val is not None
            and z_val <= -BASELINE_ANOMALOUS_Z_THRESHOLD
        ):
            origin = DecisionOrigin.ADVERSARIAL
            action = Action.INVESTIGATE_TAMPERING
            evidence_ids = ["baseline.z_score"]
            if neigh_item:
                evidence_ids.append("neighborhood.share")
            if ident_item:
                evidence_ids.append("physics.current_vs_power_identity")
            support = 0.80
            rationale = (
                f"Active load is significantly depressed (z={z_val:.2f}) while electrical "
                "identity holds and neighbors remain unaffected; investigating tampering."
            )

        # Branch 6: Incomplete, ambiguous, or unconfirmed physical pattern
        else:
            action = Action.FLAG_DATA_SUSPECT
            origin = DecisionOrigin.UNKNOWN
            evidence_ids = list(view.evidence.ids)[:2]
            support = 0.50
            rationale = (
                "Physical and contextual evidence is ambiguous or insufficient to confirm "
                "a physical event or sensor fault; flagging data as suspect."
            )

        # Precedent reinforcement: history enhances confidence if aligned
        if (
            hist_sim is not None
            and hist_sim >= TAU_SIM_DEFAULT
            and hist_orig is not None
            and origin != DecisionOrigin.UNKNOWN
            and hist_orig == origin.value
            and "history.0.similarity" in view.evidence
            and "history.0.origin" in view.evidence
        ):
            evidence_ids.extend(["history.0.similarity", "history.0.origin"])
            support = builtins.round(min(1.0, support + 0.05), 4)
            rationale += (
                f" Verified historical precedent confirms {hist_orig} (similarity={hist_sim:.2f})."
            )

        # Guard support range
        support = max(0.0, min(1.0, support))

        proposal = Proposal(
            agent=AgentRole.VALIDATOR,
            round=round,
            action=action,
            origin=origin,
            support=support,
            evidence_ids=evidence_ids,
            rationale=rationale,
            backend="rule_based",
            model="rules:validator_v1",
            prompt_hash=prompt_hash("rule_based_validator"),
            latency_ms=0.0,
            fallback_used=False,
        )

        validate_citations(proposal, view)
        return proposal
