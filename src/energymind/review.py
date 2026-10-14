"""M8 - the human-in-the-loop review step.

There are exactly **two agents**, the Analyst and the Validator. They propose a decision
and the pipeline computes a calibrated confidence ``f``. Routing then decides who acts:

* **tier 1** - autonomous (unchanged);
* **tier 2** - *uncertain*: the human **approves or rejects the agents' proposal**;
* **tier 3** - *high-risk*: the human **enters their own final action and origin** (the
  agents' proposal is shown for reference only and is never auto-approved or reused).

Tier 2 card::

    The system detected a possible data-quality anomaly.

    Proposed action: Quarantine and estimate
    Confidence: 61.5%
    Reason: Strong data-quality failure indication ...
    Evidence: baseline deviation: anomalous; current vs power identity: broken

    Do you approve the proposed decision?
    [Approve] [Reject]

Tier 3 card::

    The system detected a possible possible tampering anomaly.
    TIER 3 (human-led, high-risk): decide the final action yourself.
    The agents' proposal below is shown for reference only and is NOT approved.

    Proposed action (reference only): Investigate tampering
    Predicted origin: possible tampering
    ...

    Enter your own final action and origin.

The human is a real person (the user of the system), not a simulated component. This
module renders the cards, reads the human's answer from the terminal, and applies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: Human-readable origin labels (the decision-origin vocabulary).
ORIGIN_LABELS = {
    "data_quality": "data-quality",
    "physical_event": "physical grid",
    "adversarial": "possible tampering",
    "unknown": "unclassified",
}

#: The closed origin set the human may choose from at tier 3.
ORIGIN_NAMES: tuple[str, ...] = tuple(ORIGIN_LABELS)

#: Human-readable action labels.
ACTION_LABELS = {
    "no_action": "No action",
    "flag_data_suspect": "Flag data as suspect",
    "quarantine_and_estimate": "Quarantine and estimate",
    "request_meter_reread": "Request a meter reread",
    "open_maintenance_ticket": "Open a maintenance ticket",
    "notify_customer": "Notify the customer",
    "investigate_tampering": "Investigate tampering",
    "coordinate_with_grid_operator": "Coordinate with the grid operator",
}

#: Tier-2 answers.
RESPONSES = ("Approve", "Reject")


class HumanDecision(StrEnum):
    """A tier-2 answer (approve/reject) to a review card."""

    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_MORE_INFO = "request_more_info"


#: Human decision -> the resolution recorded on the case.
RESOLUTIONS = {
    HumanDecision.APPROVE: "approved",
    HumanDecision.REJECT: "rejected",
    HumanDecision.REQUEST_MORE_INFO: "more_info_requested",
}

#: Tier-2 interactive menu (approve/reject only).
APPROVAL_MENU = (
    ("1", HumanDecision.APPROVE, "Approve the proposed decision"),
    ("2", HumanDecision.REJECT, "Reject the proposed decision"),
)


@dataclass(frozen=True)
class CustomDecision:
    """A tier-3 decision entered by the human (their own action and origin)."""

    action: str
    origin: str


def allowed_action_names(policy) -> list[str]:
    """The policy's allowed action set (sorted), including ``no_action`` when present."""
    return sorted(policy.actions)


# --------------------------------------------------------------------------- #
# Card rendering
# --------------------------------------------------------------------------- #


def _label(mapping: dict[str, str], value: str | None, default: str) -> str:
    if not value:
        return default
    return mapping.get(value, str(value).replace("_", " "))


def _confidence_text(f) -> str:
    if f is None:
        return "n/a"
    return f"{100.0 * float(f):.1f}%"


def _reason(case: dict) -> str:
    outcome = case.get("outcome")
    decision = (getattr(outcome, "decision", None) or {}) if outcome is not None else {}
    for role in ("analyst", "validator"):
        rationale = (decision.get(role) or {}).get("rationale")
        if rationale:
            return str(rationale)
    if case.get("proposal_source") == "seen_before":
        reused = decision.get("reused_from") or {}
        return f"Reused a verified precedent (similarity {reused.get('similarity')})."
    return "No rationale available."


def _evidence(case: dict, max_items: int) -> list[str]:
    record = case.get("record") or {}
    items: list[str] = []
    for check in (record.get("evidence") or [])[:max_items]:
        name = str(check.get("check", "check")).replace("_", " ")
        status = check.get("status")
        items.append(f"{name}: {status}" if status else name)
    return items


def review_card(
    case: dict,
    *,
    tier: int = 2,
    confidence: float | None = None,
    risk: str | None = None,
    max_evidence: int = 4,
) -> str:
    """Render the human-facing card for one escalated case (tier 2 or tier 3)."""
    f = confidence if confidence is not None else case.get("f")
    origin = _label(ORIGIN_LABELS, case.get("origin"), "unclassified")
    action = _label(ACTION_LABELS, case.get("action"), "no action chosen")
    evidence = "; ".join(_evidence(case, max_evidence)) or "none"

    if tier >= 3:
        risk_text = f" (risk: {risk})" if risk else ""
        return "\n".join(
            [
                f"The system detected a possible {origin} anomaly.",
                "TIER 3 (human-led, high-risk): decide the final action yourself.",
                "The agents' proposal below is shown FOR REFERENCE ONLY and is NOT approved.",
                "",
                f"Proposed action (reference only): {action}",
                f"Predicted origin: {origin}",
                f"Confidence: {_confidence_text(f)}{risk_text}",
                f"Reason: {_reason(case)}",
                f"Evidence: {evidence}",
                "",
                "Enter your own final action and origin.",
            ]
        )

    return "\n".join(
        [
            f"The system detected a possible {origin} anomaly.",
            "",
            f"Proposed action: {action}",
            f"Confidence: {_confidence_text(f)}",
            f"Reason: {_reason(case)}",
            f"Evidence: {evidence}",
            "",
            "Do you approve the proposed decision?",
            "[" + "] [".join(RESPONSES) + "]",
        ]
    )


# --------------------------------------------------------------------------- #
# Parsing / coercion (shared by the terminal prompt and programmatic answers)
# --------------------------------------------------------------------------- #


def _read(input_fn, prompt: str, output_fn) -> str:
    reader = input_fn or input  # resolve builtins.input at call time
    try:
        return reader(prompt)
    except EOFError as exc:  # no terminal/input available: fail loudly, never guess
        raise RuntimeError(
            "no input available for interactive human review (pass --human-answer / "
            "--human-action, or run without --interactive-human-review)"
        ) from exc


def parse_approval(raw) -> HumanDecision | None:
    """Parse a tier-2 answer (``1``/``2``/``approve``/``reject``); ``None`` if invalid."""
    text = str(raw).strip().lower()
    for key, decision, _ in APPROVAL_MENU:
        if text == key:
            return decision
    return {"approve": HumanDecision.APPROVE, "reject": HumanDecision.REJECT}.get(text)


def parse_action(raw, allowed_actions) -> str | None:
    """Parse a tier-3 action against the policy's allowed set; ``None`` if invalid."""
    text = str(raw).strip().lower().replace(" ", "_")
    return text if text in set(allowed_actions) else None


def parse_origin(raw) -> str | None:
    """Parse a tier-3 origin against the closed origin set; ``None`` if invalid."""
    text = str(raw).strip().lower().replace(" ", "_")
    return text if text in ORIGIN_NAMES else None


def prompt_for_approval(card: str, *, input_fn=None, output_fn=print) -> HumanDecision:
    """Tier 2: show the card and read exactly one of Approve / Reject (re-prompt)."""
    output_fn(card)
    output_fn("")
    output_fn("Choose one option:")
    for key, _, label in APPROVAL_MENU:
        output_fn(f"  {key}. {label}")
    while True:
        raw = _read(input_fn, "Your choice [1-2]: ", output_fn)
        decision = parse_approval(raw)
        if decision is not None:
            return decision
        output_fn(f"Invalid choice {raw!r}: please enter 1 (approve) or 2 (reject).")


def prompt_for_custom_action(
    card: str,
    *,
    allowed_actions,
    origins=ORIGIN_NAMES,
    input_fn=None,
    output_fn=print,
) -> CustomDecision:
    """Tier 3: show the card and read the human's own action and origin (re-prompt)."""
    output_fn(card)
    output_fn("")
    output_fn("Allowed actions: " + ", ".join(allowed_actions))
    while True:
        raw = _read(input_fn, "Your action: ", output_fn)
        action = parse_action(raw, allowed_actions)
        if action is not None:
            break
        output_fn(f"Invalid action {raw!r}: choose one of {', '.join(allowed_actions)}.")
    while True:
        raw = _read(input_fn, "Your origin: ", output_fn)
        origin = parse_origin(raw)
        if origin is not None:
            break
        output_fn(f"Invalid origin {raw!r}: choose one of {', '.join(origins)}.")
    return CustomDecision(action=action, origin=origin)


def prompt_human(
    card: str,
    *,
    tier: int,
    allowed_actions,
    origins=ORIGIN_NAMES,
    input_fn=None,
    output_fn=print,
) -> HumanDecision | CustomDecision:
    """Dispatch to the tier-appropriate terminal prompt."""
    if tier >= 3:
        return prompt_for_custom_action(
            card,
            allowed_actions=allowed_actions,
            origins=origins,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    return prompt_for_approval(card, input_fn=input_fn, output_fn=output_fn)


def coerce_approval(value) -> HumanDecision | None:
    """Coerce a programmatic tier-2 answer to ``APPROVE``/``REJECT`` (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, HumanDecision):
        return value if value in (HumanDecision.APPROVE, HumanDecision.REJECT) else None
    return parse_approval(value)


def coerce_custom(value) -> CustomDecision | None:
    """Coerce a programmatic tier-3 answer to a ``CustomDecision`` (or ``None``).

    Accepts a ``CustomDecision``, a ``{"action": ..., "origin": ...}`` mapping, or an
    ``(action, origin)`` pair. A missing/unknown action or origin yields ``None`` (the
    case then stays ``pending_human`` rather than being silently approved).
    """
    if value is None:
        return None
    if isinstance(value, CustomDecision):
        action, origin = value.action, value.origin
    elif isinstance(value, dict):
        action, origin = value.get("action"), value.get("origin")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        action, origin = value
    else:
        return None
    if not action or not origin:
        return None
    return CustomDecision(action=str(action), origin=str(origin))


# --------------------------------------------------------------------------- #
# Applying the human's decision
# --------------------------------------------------------------------------- #


def apply_human_decision(case: dict, decision: str | HumanDecision, *, note=None) -> dict:
    """Apply a tier-2 approve/reject answer.

    * ``approve`` -> the proposed action/origin become final (reusable iff correct);
    * ``reject`` -> no action is taken and the proposal is not reused.
    """
    answer = HumanDecision(decision)
    if answer is HumanDecision.APPROVE:
        return {
            "anomaly_id": case.get("anomaly_id"),
            "human_decision": answer.value,
            "resolution": RESOLUTIONS[answer],
            "final_action": case.get("action"),
            "final_origin": case.get("origin"),
            "reusable": bool(case.get("correct")),
            "note": note,
        }
    return {
        "anomaly_id": case.get("anomaly_id"),
        "human_decision": answer.value,
        "resolution": RESOLUTIONS[answer],
        "final_action": None,
        "final_origin": None,
        "reusable": False,
        "note": note,
    }


def apply_custom_decision(case: dict, custom: CustomDecision | dict, *, note=None) -> dict:
    """Apply a tier-3 human decision (their own action/origin).

    The agents' proposed action is **not** used: only the human's action/origin become the
    final decision. Reusability is decided by the pipeline's correctness rules, never by
    the mere presence of a human answer.
    """
    decision = custom if isinstance(custom, CustomDecision) else coerce_custom(custom)
    if decision is None:
        raise ValueError(f"invalid tier-3 human decision: {custom!r}")
    return {
        "anomaly_id": case.get("anomaly_id"),
        "human_decision": "custom",
        "resolution": "custom",
        "final_action": decision.action,
        "final_origin": decision.origin,
        "reusable": False,
        "note": note,
    }
