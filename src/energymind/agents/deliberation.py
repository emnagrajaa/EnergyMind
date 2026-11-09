"""M7.5 - Multi-round deliberation, peer evidence exchange, and audit orchestration.

Coordinates independent Analyst and Validator agents through bounded deliberation:
- Blind Round 1: Agents formulate proposals independently with zero peer leakage.
- Subsequent Rounds: Discloses only explicitly cited peer evidence across boundaries.
- Round Snapshots: Agents evaluate based on previous-round frozen snapshots.
- Revision Policy: Enforces that action/origin changes must cite novel evidence.
- Max Rounds: Strictly respects r_max limit (default 3); preserves non-convergence.
- Audit Logging: Cryptographically hash-chained logging of all deliberation events.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from energymind.agents.analyst import RuleBasedAnalyst, build_analyst_view
from energymind.agents.base import (
    Action,
    Agent,
    AgentContractError,
    AgentRole,
    AgentView,
    DeliberationResult,
    EvidenceCatalog,
    EvidenceItem,
    InvalidRevisionError,
    Proposal,
    proposals_converge,
    validate_citations,
    validate_revision,
)
from energymind.agents.llm_agent import create_analyst_agent, create_validator_agent
from energymind.agents.validator import (
    FORBIDDEN_VALIDATOR_ID_PREFIXES,
    HistoricalCaseSummary,
    RuleBasedValidator,
    build_validator_view,
)
from energymind.config import Settings, load_settings
from energymind.policy import action_risk, lower_risk_action
from energymind.retrieval import Case, Match
from energymind.store import AuditLog

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Peer Evidence Disclosure Helpers
# --------------------------------------------------------------------------- #


def build_peer_context(
    proposal: Proposal,
    source_view: AgentView,
) -> tuple[EvidenceItem, ...]:
    """Extract and isolate evidence items explicitly cited in a peer proposal.

    Invariants:
    1. Only evidence IDs present in proposal.evidence_ids are extracted.
    2. Every cited ID must exist in source_view.evidence; uncited or unverified
       items are never disclosed.
    3. Items are marked with source='peer_disclosed:<agent>' to distinguish
       them from an agent's own evidence.
    4. Never touches or accesses the raw record or hidden ground truth.
    """
    disclosed: list[EvidenceItem] = []
    for eid in proposal.evidence_ids:
        if eid in source_view.evidence:
            source_item = source_view.evidence.get(eid)
            disclosed_item = EvidenceItem(
                id=source_item.id,
                source=f"peer_disclosed:{proposal.agent.value}",
                description=source_item.description,
                value=source_item.value,
                unit=source_item.unit,
                status=source_item.status,
            )
            disclosed.append(disclosed_item)
        else:
            logger.warning(
                "Cited evidence ID '%s' in %s proposal not found in source view; "
                "skipping disclosure",
                eid,
                proposal.agent.value,
            )
    return tuple(disclosed)


def create_effective_view(
    base_view: AgentView,
    disclosed_peer_items: Sequence[EvidenceItem],
) -> AgentView:
    """Create an augmented AgentView combining own evidence with peer-disclosed items.

    Own evidence takes precedence if an ID collision occurs. Peer items are
    clearly isolated and appended to the evidence catalog.
    """
    combined: list[EvidenceItem] = list(base_view.evidence.items)
    existing_ids = {it.id for it in combined}

    for peer_item in disclosed_peer_items:
        # The Validator must never receive Analyst-side model/XAI evidence,
        # even when it was explicitly cited in a peer proposal.  Peer exchange
        # may disclose conclusions, but it cannot bypass the role boundary.
        if base_view.role == AgentRole.VALIDATOR and any(
            peer_item.id.startswith(prefix) for prefix in FORBIDDEN_VALIDATOR_ID_PREFIXES
        ):
            continue
        if peer_item.id not in existing_ids:
            combined.append(peer_item)
            existing_ids.add(peer_item.id)

    return AgentView(
        anomaly_id=base_view.anomaly_id,
        role=base_view.role,
        evidence=EvidenceCatalog(items=combined),
        allowed_actions=base_view.allowed_actions,
        case_metadata=base_view.case_metadata,
    )


def _proposal_to_audit_payload(proposal: Proposal, anomaly_id: str | None = None) -> dict[str, Any]:
    """Serialize safe proposal metadata for append-only audit logging."""
    return {
        "anomaly_id": anomaly_id,
        "role": proposal.agent.value,
        "round": proposal.round,
        "action": proposal.action.value,
        "origin": proposal.origin.value,
        "support": proposal.support,
        "evidence_ids": list(proposal.evidence_ids),
        "rationale": proposal.rationale,
        "objection_to_other": proposal.objection_to_other,
        "changed_since_last_round": proposal.changed_since_last_round,
        "backend": proposal.backend,
        "model": proposal.model,
        "prompt_hash": proposal.prompt_hash,
        "fallback_used": proposal.fallback_used,
        "fallback_reason": proposal.fallback_reason,
        "latency_ms": proposal.latency_ms,
    }


# --------------------------------------------------------------------------- #
# Deliberator Coordinator
# --------------------------------------------------------------------------- #


class Deliberator:
    """Bounded, two-agent deliberation and audit log coordinator.

    Coordinates an Analyst and a Validator:
    - Round 1 is strictly blind and independent.
    - Subsequent rounds operate on frozen snapshots from the previous round.
    - Restricts peer evidence to explicitly cited items from the other agent.
    - Rejects invalid revisions (changes in action/origin without new evidence citations).
    - Stops at r_max rounds; preserves non-convergence without arbitrary tie-breaking.
    - Records an append-only, hash-chained audit log of all deliberation events.
    """

    def __init__(
        self,
        analyst: Agent | None = None,
        validator: Agent | None = None,
        r_max: int = 3,
        audit_log: AuditLog | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings
        self.r_max = r_max
        self.audit_log = audit_log

        if analyst is not None:
            self.analyst = analyst
        else:
            try:
                self.analyst = create_analyst_agent(settings=settings)
            except Exception as exc:
                logger.info(
                    "Default LLM analyst initialization failed (%s); using RuleBasedAnalyst",
                    exc,
                )
                self.analyst = RuleBasedAnalyst()

        if validator is not None:
            self.validator = validator
        else:
            try:
                self.validator = create_validator_agent(settings=settings)
            except Exception as exc:
                logger.info(
                    "Default LLM validator initialization failed (%s); using RuleBasedValidator",
                    exc,
                )
                self.validator = RuleBasedValidator()

    def deliberate(
        self,
        record: dict[str, Any],
        *,
        history: list[Case | Match | HistoricalCaseSummary | dict] | None = None,
        audit_log: AuditLog | None = None,
    ) -> DeliberationResult:
        """Run bounded deliberation on an anomaly record."""
        anomaly_id = str(record.get("anomaly_id", "")).strip()
        if not anomaly_id:
            raise ValueError("record must contain a non-empty 'anomaly_id'")

        log = audit_log or self.audit_log or AuditLog(sqlite_path=":memory:")

        # Fix 5a: convergence compares origins and action *risk classes* (from the policy).
        policy = (self.settings or load_settings()).policy
        risk_of = lambda action: action_risk(action, policy)  # noqa: E731

        start_event = log.append_event(
            actor="deliberator",
            event_type="deliberation_started",
            payload={
                "anomaly_id": anomaly_id,
                "r_max": self.r_max,
                "analyst_role": self.analyst.role.value,
                "validator_role": self.validator.role.value,
            },
            anomaly_id=anomaly_id,
        )
        audit_seq_start = start_event["seq"]

        # 1. Build initial base views (strict evidence boundaries)
        analyst_base_view = build_analyst_view(record)
        validator_base_view = build_validator_view(record, history=history, audit_log=log)

        if analyst_base_view.role != AgentRole.ANALYST:
            raise AgentContractError(
                f"Analyst view role must be ANALYST, got {analyst_base_view.role}"
            )
        if validator_base_view.role != AgentRole.VALIDATOR:
            raise AgentContractError(
                f"Validator view role must be VALIDATOR, got {validator_base_view.role}"
            )

        # 2. Multi-round state tracking
        cumulative_analyst_disclosed: dict[str, EvidenceItem] = {}
        cumulative_validator_disclosed: dict[str, EvidenceItem] = {}

        analyst_proposals: list[Proposal] = []
        validator_proposals: list[Proposal] = []

        analyst_fallback_ever: bool = False
        validator_fallback_ever: bool = False

        final_round: int = 1
        converged: bool = False
        agreed_action: Action | None = None
        agreed_origin: Any = None

        try:
            for current_round in range(1, self.r_max + 1):
                final_round = current_round

                if current_round == 1:
                    # BLIND ROUND 1: Neither agent receives any peer proposal or evidence
                    prop_a = self.analyst.propose(
                        analyst_base_view,
                        round=1,
                        other_proposal=None,
                        own_previous_proposal=None,
                    )
                    validate_citations(prop_a, analyst_base_view)

                    prop_v = self.validator.propose(
                        validator_base_view,
                        round=1,
                        other_proposal=None,
                        own_previous_proposal=None,
                    )
                    validate_citations(prop_v, validator_base_view)
                else:
                    # ROUND N > 1: Snapshot previous round
                    snapshot_a = analyst_proposals[-1]
                    snapshot_v = validator_proposals[-1]

                    # Disclose peer evidence from round N - 1
                    new_v_items = build_peer_context(snapshot_v, validator_base_view)
                    for item in new_v_items:
                        cumulative_validator_disclosed[item.id] = item

                    new_a_items = build_peer_context(snapshot_a, analyst_base_view)
                    for item in new_a_items:
                        cumulative_analyst_disclosed[item.id] = item

                    log.append_event(
                        actor="deliberator",
                        event_type="peer_evidence_disclosed",
                        payload={
                            "round_from": current_round - 1,
                            "round_to": current_round,
                            "validator_disclosed_ids": [it.id for it in new_v_items],
                            "analyst_disclosed_ids": [it.id for it in new_a_items],
                        },
                        anomaly_id=anomaly_id,
                    )

                    effective_a_view = create_effective_view(
                        analyst_base_view, list(cumulative_validator_disclosed.values())
                    )
                    effective_v_view = create_effective_view(
                        validator_base_view, list(cumulative_analyst_disclosed.values())
                    )

                    # Propose with frozen previous-round snapshots
                    raw_prop_a = self.analyst.propose(
                        effective_a_view,
                        round=current_round,
                        other_proposal=snapshot_v,
                        own_previous_proposal=snapshot_a,
                    )
                    validate_citations(raw_prop_a, effective_a_view)

                    raw_prop_v = self.validator.propose(
                        effective_v_view,
                        round=current_round,
                        other_proposal=snapshot_a,
                        own_previous_proposal=snapshot_v,
                    )
                    validate_citations(raw_prop_v, effective_v_view)

                    # Validate revision for Analyst
                    try:
                        validate_revision(raw_prop_a, snapshot_a)
                        prop_a = raw_prop_a
                    except InvalidRevisionError as exc:
                        log.append_event(
                            actor="deliberator",
                            event_type="revision_rejected",
                            payload={
                                "agent": AgentRole.ANALYST.value,
                                "round": current_round,
                                "attempted_action": raw_prop_a.action.value,
                                "attempted_origin": raw_prop_a.origin.value,
                                "retained_action": snapshot_a.action.value,
                                "retained_origin": snapshot_a.origin.value,
                                "reason": str(exc),
                            },
                            anomaly_id=anomaly_id,
                        )
                        prop_a = snapshot_a

                    # Validate revision for Validator
                    try:
                        validate_revision(raw_prop_v, snapshot_v)
                        prop_v = raw_prop_v
                    except InvalidRevisionError as exc:
                        log.append_event(
                            actor="deliberator",
                            event_type="revision_rejected",
                            payload={
                                "agent": AgentRole.VALIDATOR.value,
                                "round": current_round,
                                "attempted_action": raw_prop_v.action.value,
                                "attempted_origin": raw_prop_v.origin.value,
                                "retained_action": snapshot_v.action.value,
                                "retained_origin": snapshot_v.origin.value,
                                "reason": str(exc),
                            },
                            anomaly_id=anomaly_id,
                        )
                        prop_v = snapshot_v

                # Track fallbacks
                if prop_a.fallback_used:
                    analyst_fallback_ever = True
                if prop_v.fallback_used:
                    validator_fallback_ever = True

                analyst_proposals.append(prop_a)
                validator_proposals.append(prop_v)

                log.append_event(
                    actor="analyst",
                    event_type="analyst_proposal",
                    payload=_proposal_to_audit_payload(prop_a, anomaly_id),
                    anomaly_id=anomaly_id,
                )
                log.append_event(
                    actor="validator",
                    event_type="validator_proposal",
                    payload=_proposal_to_audit_payload(prop_v, anomaly_id),
                    anomaly_id=anomaly_id,
                )

                # Check convergence (same origin AND same action risk class)
                if proposals_converge(prop_a, prop_v, risk_of):
                    converged = True
                    agreed_origin = prop_a.origin
                    agreed_action = lower_risk_action(prop_a.action, prop_v.action, policy)
                    log.append_event(
                        actor="deliberator",
                        event_type="convergence",
                        payload={
                            "round": current_round,
                            "agreed_action": agreed_action.value,
                            "agreed_origin": agreed_origin.value,
                            "analyst_action": prop_a.action.value,
                            "validator_action": prop_v.action.value,
                        },
                        anomaly_id=anomaly_id,
                    )
                    break

            if not converged:
                log.append_event(
                    actor="deliberator",
                    event_type="non_convergence",
                    payload={
                        "rounds_used": final_round,
                        "analyst_final_action": analyst_proposals[-1].action.value,
                        "validator_final_action": validator_proposals[-1].action.value,
                        "reason": "max_rounds_reached",
                    },
                    anomaly_id=anomaly_id,
                )

            comp_event = log.append_event(
                actor="deliberator",
                event_type="deliberation_completed",
                payload={
                    "converged": converged,
                    "rounds_used": final_round,
                    "agreed_action": agreed_action.value if agreed_action else None,
                    "agreed_origin": agreed_origin.value if agreed_origin else None,
                    "fallback_used": analyst_fallback_ever or validator_fallback_ever,
                },
                anomaly_id=anomaly_id,
            )
            audit_seq_end = comp_event["seq"]

            return DeliberationResult(
                anomaly_id=anomaly_id,
                converged=converged,
                rounds_used=final_round,
                analyst_final=analyst_proposals[-1],
                validator_final=validator_proposals[-1],
                agreed_action=agreed_action,
                agreed_origin=agreed_origin,
                fallback_used=analyst_fallback_ever or validator_fallback_ever,
                analyst_fallback=analyst_fallback_ever,
                validator_fallback=validator_fallback_ever,
                audit_seq_start=audit_seq_start,
                audit_seq_end=audit_seq_end,
            )

        except Exception as exc:
            log.append_event(
                actor="deliberator",
                event_type="orchestration_failed",
                payload={
                    "round": final_round,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                anomaly_id=anomaly_id,
            )
            raise


# --------------------------------------------------------------------------- #
# Convenience Function
# --------------------------------------------------------------------------- #


def deliberate(
    record: dict[str, Any],
    *,
    analyst: Agent | None = None,
    validator: Agent | None = None,
    history: list[Case | Match | HistoricalCaseSummary | dict] | None = None,
    r_max: int | None = None,
    audit_log: AuditLog | None = None,
    settings: Settings | None = None,
) -> DeliberationResult:
    """Convenience entry point for running bounded deliberation on a record."""
    cfg = settings or load_settings()
    max_rounds = r_max if r_max is not None else cfg.agents.r_max
    deliberator = Deliberator(
        analyst=analyst,
        validator=validator,
        r_max=max_rounds,
        audit_log=audit_log,
        settings=cfg,
    )
    return deliberator.deliberate(record, history=history, audit_log=audit_log)
