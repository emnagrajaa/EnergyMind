"""M8 - the human-in-the-loop approval step.

There are exactly **two agents**, the Analyst and the Validator. They propose a decision
and the pipeline computes a calibrated confidence ``f``. When the routing says a case is
uncertain or high-risk, the system puts it to the **human user / reviewer** as a short
card and waits for a decision; only then is the final decision stored::

    The system detected a possible data-quality anomaly.

    Proposed action: Quarantine and estimate
    Confidence: 61.5%
    Reason: Strong data-quality failure indication ...
    Evidence: baseline deviation: anomalous; current vs power identity: broken

    Do you approve this action?
    [Approve] [Reject] [Request more information]

The human is a real person (the user of the system), not a simulated component. This
module only renders the card and applies the human's answer.
"""

from __future__ import annotations

from enum import StrEnum

#: Human-readable origin labels (the decision-origin vocabulary).
ORIGIN_LABELS = {
    "data_quality": "data-quality",
    "physical_event": "physical grid",
    "adversarial": "possible tampering",
    "unknown": "unclassified",
}

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

#: The three answers the human can give on a review card.
RESPONSES = ("Approve", "Reject", "Request more information")


class HumanDecision(StrEnum):
    """The human's answer to a review card."""

    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_MORE_INFO = "request_more_info"


#: Human decision -> the resolution recorded on the case.
RESOLUTIONS = {
    HumanDecision.APPROVE: "approved",
    HumanDecision.REJECT: "rejected",
    HumanDecision.REQUEST_MORE_INFO: "more_info_requested",
}


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


def review_card(case: dict, *, confidence: float | None = None, max_evidence: int = 4) -> str:
    """Render the human-facing approval card for one escalated case.

    ``case`` is a pipeline case record (``origin`` / ``action`` / ``f`` /
    ``proposal_source`` / ``record`` / ``outcome``); ``confidence`` overrides ``case['f']``.
    """
    f = confidence if confidence is not None else case.get("f")
    origin = _label(ORIGIN_LABELS, case.get("origin"), "unclassified")
    action = _label(ACTION_LABELS, case.get("action"), "no action chosen")
    evidence = _evidence(case, max_evidence)
    evidence_text = "; ".join(evidence) if evidence else "none"

    lines = [
        f"The system detected a possible {origin} anomaly.",
        "",
        f"Proposed action: {action}",
        f"Confidence: {_confidence_text(f)}",
        f"Reason: {_reason(case)}",
        f"Evidence: {evidence_text}",
        "",
        "Do you approve this action?",
        "[" + "] [".join(RESPONSES) + "]",
    ]
    return "\n".join(lines)


def apply_human_decision(case: dict, decision: str | HumanDecision, *, note=None) -> dict:
    """Apply the human's answer to an escalated case.

    Returns the final decision for the case:

    * ``approve`` -> the proposed action/origin become final (and are reusable);
    * ``reject`` -> no action is taken and the proposal is not reused;
    * ``request_more_info`` -> still unresolved; the case stays open for human-led work.
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
