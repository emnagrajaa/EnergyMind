"""M7.1 - Contract layer for the two-agent deliberation system.

This module defines the strict contracts and validation primitives used by Agent 1
(Analyst) and Agent 2 (Validator), their deliberation loop, and downstream M8 confidence
and human-in-the-loop routing.

Design Principles:
- RULE 1: Ground truth must never enter an AgentView. No unrestricted record dicts.
- RULE 2: DecisionOrigin represents an agent's conclusion (including 'unknown'), which is
  distinct from the injection-origin taxonomy (data_quality, adversarial, physical_event).
- RULE 3: Closed action set matching configs/policy.yaml (8 canonical actions).
- RULE 4: Every proposal must cite at least one valid evidence ID visible to that agent.
- RULE 5: Proposal 'support' reflects agent self-assessed support [0, 1], NOT calibrated
  system confidence (which is computed separately in M8).
- RULE 6: Authoritative metadata (agent, round, model, backend, prompt hash, latency) must
  NOT come from the LLM (ProposalOut); it is added by trusted runtime code (Proposal).
- RULE 7: All evidence and contract structures are deterministic and JSON-serializable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Domain Exceptions
# --------------------------------------------------------------------------- #


class AgentContractError(ValueError):
    """Base exception for all agent contract violations."""


class InvalidEvidenceError(AgentContractError):
    """Raised when an evidence item, catalog, or citation list is invalid."""


class UnknownEvidenceCitationError(AgentContractError):
    """Raised when a proposal cites an evidence ID not present in the agent's catalog."""


class InvalidRevisionError(AgentContractError):
    """Raised when an agent revises its decision without citing novel evidence."""


# --------------------------------------------------------------------------- #
# Closed Action Set & Decision Taxonomy
# --------------------------------------------------------------------------- #


class Action(StrEnum):
    """Closed action set matching configs/policy.yaml.

    Agents cannot invent new actions. Each action carries predefined risk and
    reversibility properties utilized by policy routing in M8.
    """

    NO_ACTION = "no_action"
    FLAG_DATA_SUSPECT = "flag_data_suspect"
    QUARANTINE_AND_ESTIMATE = "quarantine_and_estimate"
    REQUEST_METER_REREAD = "request_meter_reread"
    OPEN_MAINTENANCE_TICKET = "open_maintenance_ticket"
    NOTIFY_CUSTOMER = "notify_customer"
    INVESTIGATE_TAMPERING = "investigate_tampering"
    COORDINATE_WITH_GRID_OPERATOR = "coordinate_with_grid_operator"


ActionLiteral = Literal[
    "no_action",
    "flag_data_suspect",
    "quarantine_and_estimate",
    "request_meter_reread",
    "open_maintenance_ticket",
    "notify_customer",
    "investigate_tampering",
    "coordinate_with_grid_operator",
]

DEFAULT_ACTIONS: tuple[Action, ...] = tuple(Action)


class DecisionOrigin(StrEnum):
    """Decision origin proposed by an agent.

    DecisionOrigin.unknown is a valid decision state indicating inconclusive or
    ambiguous evidence. This is strictly distinct from the dataset injection
    Origin (data_quality, adversarial, physical_event), which is a simulation label.
    """

    DATA_QUALITY = "data_quality"
    PHYSICAL_EVENT = "physical_event"
    ADVERSARIAL = "adversarial"
    UNKNOWN = "unknown"


DecisionOriginLiteral = Literal[
    "data_quality",
    "physical_event",
    "adversarial",
    "unknown",
]


class AgentRole(StrEnum):
    """Role of an agent in the two-agent architecture."""

    ANALYST = "analyst"
    VALIDATOR = "validator"


AgentRoleLiteral = Literal["analyst", "validator"]


# --------------------------------------------------------------------------- #
# Evidence Contracts
# --------------------------------------------------------------------------- #


def _to_json_compatible(v: Any) -> Any:
    """Coerce numpy scalar or array types to standard Python equivalents."""
    if isinstance(v, (np.generic, np.ndarray)):
        return v.tolist()
    return v


class EvidenceItem(BaseModel):
    """An immutable, strongly validated evidence item presented to an agent.

    Contains verified observations, model outputs, or physical checks with stable IDs.
    Does not allow secret-bearing or arbitrary non-serializable Python objects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    source: str
    description: str
    value: Any = None
    unit: str | None = None
    status: str | None = None

    @field_validator("id", "source", "description")
    @classmethod
    def _validate_non_blank(cls, v: str, info) -> str:
        if not isinstance(v, str) or not v.strip():
            raise InvalidEvidenceError(f"Evidence {info.field_name} cannot be blank")
        return v.strip()

    @field_validator("value")
    @classmethod
    def _validate_json_serializable(cls, v: Any) -> Any:
        v_clean = _to_json_compatible(v)
        try:
            json.dumps(v_clean)
        except (TypeError, OverflowError) as exc:
            raise InvalidEvidenceError(
                f"Evidence value must be JSON-serializable, got {type(v).__name__}: {exc}"
            ) from exc
        return v_clean


class EvidenceCatalog(BaseModel):
    """Controlled collection of EvidenceItem objects for an agent.

    Preserves insertion order, rejects duplicate evidence IDs, supports
    membership tests and retrieval by ID.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[EvidenceItem, ...] = Field(default_factory=tuple)

    def __init__(
        self,
        items: Sequence[EvidenceItem | dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if items is not None:
            if "items" in kwargs:
                raise ValueError("Cannot pass 'items' both positionally and as keyword argument")
            kwargs["items"] = tuple(items)
        super().__init__(**kwargs)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, v: Any) -> tuple[EvidenceItem, ...]:
        if isinstance(v, (list, tuple, set)):
            seen_ids: set[str] = set()
            validated_items: list[EvidenceItem] = []
            for item in v:
                if not isinstance(item, EvidenceItem):
                    item = EvidenceItem.model_validate(item)
                if item.id in seen_ids:
                    raise InvalidEvidenceError(f"Duplicate evidence ID in catalog: '{item.id}'")
                seen_ids.add(item.id)
                validated_items.append(item)
            return tuple(validated_items)
        raise ValueError(f"items must be a sequence, got {type(v).__name__}")

    @property
    def ids(self) -> frozenset[str]:
        """Expose the valid evidence ID set."""
        return frozenset(item.id for item in self.items)

    def __contains__(self, key: Any) -> bool:
        """Support membership lookup by evidence ID string or EvidenceItem."""
        if isinstance(key, str):
            return key in self.ids
        if isinstance(key, EvidenceItem):
            return key.id in self.ids and self.get(key.id) == key
        return False

    def __getitem__(self, item_id: str) -> EvidenceItem:
        """Retrieve an EvidenceItem by its ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(f"Evidence ID '{item_id}' not found in catalog")

    def get(self, item_id: str, default: Any = None) -> EvidenceItem | Any:
        """Retrieve by ID with an optional default."""
        for item in self.items:
            if item.id == item_id:
                return item
        return default

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


# --------------------------------------------------------------------------- #
# Agent View (Evidence Boundary)
# --------------------------------------------------------------------------- #


class CaseMetadata(BaseModel):
    """Optional safe case metadata that cannot reveal truth labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    household_id: str | None = None
    t_start: str | None = None
    t_end: str | None = None
    duration_h: float | None = None


class AgentView(BaseModel):
    """Information presented to one agent.

    This is a strict evidence boundary. Ground truth (true_type, true_origin,
    ground_truth, injected_type, injection_label, target_label, y_true) must
    NEVER be included. No unrestricted record dictionaries are allowed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anomaly_id: str
    role: AgentRole
    evidence: EvidenceCatalog
    allowed_actions: tuple[Action, ...] = DEFAULT_ACTIONS
    case_metadata: CaseMetadata | None = None

    @field_validator("anomaly_id")
    @classmethod
    def _validate_anomaly_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("anomaly_id cannot be blank")
        return v.strip()


# --------------------------------------------------------------------------- #
# Proposals (Untrusted LLM Output vs Trusted Runtime Representation)
# --------------------------------------------------------------------------- #


class ProposalOut(BaseModel):
    """Untrusted structured output from a language model or agent proposer.

    Must NOT contain authoritative runtime metadata (agent identity, round number,
    model name, backend, prompt hash, latency, fallback state).

    The 'support' field indicates the agent's self-assessed support for its proposal [0, 1].
    It is NOT system confidence. Calibrated system confidence f is computed in M8.
    """

    model_config = ConfigDict(extra="forbid")

    action: Action
    origin: DecisionOrigin
    support: float = Field(
        ge=0.0,
        le=1.0,
        description="Agent self-support [0, 1]. NOT calibrated system confidence.",
    )
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=1, max_length=600)
    objection_to_other: str | None = Field(default=None, max_length=400)
    changed_since_last_round: bool = False

    @field_validator("evidence_ids")
    @classmethod
    def _validate_evidence_ids(cls, v: list[str]) -> list[str]:
        if not v:
            raise InvalidEvidenceError("evidence_ids cannot be empty")
        for eid in v:
            if not isinstance(eid, str) or not eid.strip():
                raise InvalidEvidenceError("evidence_ids cannot contain blank IDs")
        return v

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale cannot be blank")
        return v

    @field_validator("objection_to_other")
    @classmethod
    def _validate_objection(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v


class Proposal(BaseModel):
    """Trusted internal proposal representation enriched with runtime metadata.

    The 'support' field represents agent self-support [0, 1], NOT calibrated
    system confidence. System confidence f is computed in M8.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: AgentRole
    round: int = Field(ge=1)
    action: Action
    origin: DecisionOrigin
    support: float = Field(
        ge=0.0,
        le=1.0,
        description="Agent self-support [0, 1]. NOT calibrated system confidence.",
    )
    evidence_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    objection_to_other: str | None = None
    changed_since_last_round: bool = False

    # Trusted runtime metadata
    backend: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_digest: str | None = None
    prompt_hash: str = Field(min_length=1)
    fallback_used: bool = False
    fallback_reason: str | None = None
    latency_ms: float = Field(ge=0.0)

    @field_validator("evidence_ids")
    @classmethod
    def _validate_evidence_ids(cls, v: list[str]) -> list[str]:
        if not v:
            raise InvalidEvidenceError("evidence_ids cannot be empty")
        for eid in v:
            if not isinstance(eid, str) or not eid.strip():
                raise InvalidEvidenceError("evidence_ids cannot contain blank IDs")
        return v

    @classmethod
    def from_proposal_out(
        cls,
        proposal_out: ProposalOut,
        *,
        agent: AgentRole,
        round: int,
        backend: str,
        model: str,
        prompt_hash: str,
        latency_ms: float,
        model_digest: str | None = None,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> Proposal:
        """Create a trusted Proposal from untrusted ProposalOut and authoritative metadata."""
        return cls(
            agent=agent,
            round=round,
            action=proposal_out.action,
            origin=proposal_out.origin,
            support=proposal_out.support,
            evidence_ids=proposal_out.evidence_ids,
            rationale=proposal_out.rationale,
            objection_to_other=proposal_out.objection_to_other,
            changed_since_last_round=proposal_out.changed_since_last_round,
            backend=backend,
            model=model,
            model_digest=model_digest,
            prompt_hash=prompt_hash,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
        )


# --------------------------------------------------------------------------- #
# Agent Protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class Agent(Protocol):
    """Protocol for M7 agents (Analyst and Validator)."""

    @property
    def role(self) -> AgentRole: ...

    def propose(
        self,
        view: AgentView,
        round: int,
        other_proposal: Proposal | None = None,
        own_previous_proposal: Proposal | None = None,
    ) -> Proposal: ...


# --------------------------------------------------------------------------- #
# Deliberation Result
# --------------------------------------------------------------------------- #


class DeliberationResult(BaseModel):
    """Final output contract of M7 deliberation for downstream M8 confidence and routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anomaly_id: str
    converged: bool
    rounds_used: int = Field(ge=1)
    analyst_final: Proposal
    validator_final: Proposal
    agreed_action: Action | None = None
    agreed_origin: DecisionOrigin | None = None
    fallback_used: bool = False
    analyst_fallback: bool = False
    validator_fallback: bool = False
    audit_seq_start: int | None = None
    audit_seq_end: int | None = None

    @model_validator(mode="after")
    def _validate_consistency(self) -> DeliberationResult:
        if self.converged:
            if self.agreed_action is None:
                raise ValueError("agreed_action must exist when converged is True")
            if self.agreed_origin is None:
                raise ValueError("agreed_origin must exist when converged is True")
            if self.agreed_action not in (
                self.analyst_final.action,
                self.validator_final.action,
            ):
                # Fix 5a: agents may agree on origin + risk class while proposing
                # different actions; the agreed action is then the lower-risk one.
                raise ValueError(
                    f"agreed_action ({self.agreed_action}) must be one of the agents' "
                    f"proposed actions (Analyst={self.analyst_final.action}, "
                    f"Validator={self.validator_final.action}) when converged is True"
                )
            if (
                self.analyst_final.origin != self.agreed_origin
                or self.validator_final.origin != self.agreed_origin
            ):
                raise ValueError(
                    f"Analyst origin ({self.analyst_final.origin}) and Validator origin "
                    f"({self.validator_final.origin}) must match "
                    f"agreed_origin ({self.agreed_origin}) when converged is True"
                )
        else:
            if self.agreed_action is not None:
                raise ValueError(
                    f"agreed_action must be None when converged is False, got {self.agreed_action}"
                )
            if self.agreed_origin is not None:
                raise ValueError(
                    f"agreed_origin must be None when converged is False, got {self.agreed_origin}"
                )
        return self


# --------------------------------------------------------------------------- #
# Validation & Helper Functions
# --------------------------------------------------------------------------- #


def validate_citations(
    proposal: Proposal | ProposalOut,
    evidence_source: AgentView | EvidenceCatalog,
) -> None:
    """Validate that all evidence IDs cited in a proposal exist in the given evidence source.

    Raises:
        InvalidEvidenceError: if citations are empty, blank, or duplicate.
        UnknownEvidenceCitationError: if any cited evidence ID is not in the catalog.
    """
    if isinstance(evidence_source, AgentView):
        catalog = evidence_source.evidence
    elif isinstance(evidence_source, EvidenceCatalog):
        catalog = evidence_source
    else:
        raise TypeError(
            f"evidence_source must be AgentView or EvidenceCatalog, got "
            f"{type(evidence_source).__name__}"
        )

    evidence_ids = proposal.evidence_ids
    if not evidence_ids:
        raise InvalidEvidenceError("Proposal evidence citation list cannot be empty")

    seen: set[str] = set()
    for eid in evidence_ids:
        if not isinstance(eid, str) or not eid.strip():
            raise InvalidEvidenceError("Evidence citation ID cannot be blank")
        if eid in seen:
            raise InvalidEvidenceError(f"Duplicate evidence citation ID: '{eid}'")
        seen.add(eid)

        if eid not in catalog:
            raise UnknownEvidenceCitationError(
                f"Evidence ID '{eid}' is not present in the agent's visible evidence catalog"
            )


def validate_revision(
    new_proposal: Proposal | ProposalOut,
    previous_proposal: Proposal | ProposalOut | None,
) -> None:
    """Validate that an agent only revises its action or origin when citing new evidence.

    Raises:
        InvalidRevisionError: if action or origin changed without citing novel evidence IDs.
    """
    if previous_proposal is None:
        return

    action_changed = new_proposal.action != previous_proposal.action
    origin_changed = new_proposal.origin != previous_proposal.origin

    if action_changed or origin_changed:
        prev_ids = set(previous_proposal.evidence_ids)
        new_ids = set(new_proposal.evidence_ids)
        novel_ids = new_ids - prev_ids
        if not novel_ids:
            changes = []
            if action_changed:
                changes.append(
                    f"action changed ({previous_proposal.action} -> {new_proposal.action})"
                )
            if origin_changed:
                changes.append(
                    f"origin changed ({previous_proposal.origin} -> {new_proposal.origin})"
                )
            raise InvalidRevisionError(
                f"Agent changed decision ({', '.join(changes)}) without citing new evidence IDs."
            )


def proposals_agree(p1: Proposal | ProposalOut, p2: Proposal | ProposalOut) -> bool:
    """Check whether two proposals agree.

    Agreement requires identical action AND identical origin.
    Does not require identical support, rationale, or citations.
    """
    return p1.action == p2.action and p1.origin == p2.origin


def proposals_converge(p1: Proposal | ProposalOut, p2: Proposal | ProposalOut, risk_of) -> bool:
    """Fix 5a convergence: same origin **and** same action risk class.

    ``risk_of`` maps an action to its risk class (``low`` / ``medium`` / ``high``). Two
    agents that agree on the origin and propose equally risky actions have converged even
    if the action *strings* differ; the caller then takes the lower-risk action.
    """
    if p1.origin != p2.origin:
        return False
    ra, rb = risk_of(p1.action), risk_of(p2.action)
    return ra is not None and ra == rb


def prompt_hash(prompt: str | bytes) -> str:
    """Compute deterministic SHA-256 hash of prompt text or bytes."""
    if isinstance(prompt, str):
        data = prompt.encode("utf-8")
    elif isinstance(prompt, (bytes, bytearray)):
        data = bytes(prompt)
    else:
        raise TypeError(f"Prompt must be str or bytes, got {type(prompt).__name__}")
    return hashlib.sha256(data).hexdigest()
