"""M8 - tiered routing (blueprint Sections 12-13).

The confidence ``f`` alone is not enough: a 0.9-confident wrong
``investigate_tampering`` costs more than a 0.6-confident wrong
``flag_data_suspect``. Routing uses both ``f`` and the action's risk.

Three tiers:

* **1 Autonomous** - ``f >= tau_auto`` **and** low-risk & reversible **and**
  converged **and** origin = ``data_quality``;
* **2 Human approval** - ``f >= tau_review``, or any medium-risk action;
* **3 Human-led** - low ``f``, non-converged, unseen type, any high-risk action,
  ``adversarial`` or ``unknown`` origin, or anything on the never-automate list.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Action names the scenario never automates (from configs/policy.yaml).
NEVER_AUTOMATE_DEFAULT = [
    "remote_disconnect",
    "switching",
    "setpoint_change",
    "load_control",
    "tampering_accusation",
    "field_crew_dispatch",
]

TIER_NAMES = {1: "autonomous", 2: "human_approval", 3: "human_led"}


def _value(x) -> str | None:
    if x is None:
        return None
    return getattr(x, "value", x)


#: Ordering of the action risk classes (Fix 5a).
RISK_RANK = {"low": 0, "medium": 1, "high": 2}


def action_spec(action, policy):
    """The policy entry for an action (or ``None``)."""
    value = _value(action)
    return policy.actions.get(value) if value else None


def action_risk(action, policy) -> str | None:
    """The risk class (``low`` / ``medium`` / ``high``) of an action, or ``None``."""
    spec = action_spec(action, policy)
    return spec.risk if spec else None


def same_risk_class(a, b, policy) -> bool:
    """True when two actions share the same risk class (unknown actions are not equal)."""
    ra, rb = action_risk(a, policy), action_risk(b, policy)
    return ra is not None and ra == rb


def action_sort_key(action, policy) -> tuple[int, int, str]:
    """Sort key: risk class, then reversibility (reversible first), then name."""
    spec = action_spec(action, policy)
    risk = RISK_RANK.get(spec.risk, 2) if spec else 2
    reversible = 0 if (spec and spec.reversible) else 1
    return (risk, reversible, _value(action) or "")


def lower_risk_action(a, b, policy):
    """The lower-risk of two actions (ties broken by reversibility, then name)."""
    return min((a, b), key=lambda action: action_sort_key(action, policy))


@dataclass
class RoutingCase:
    """Minimal per-case inputs for routing."""

    f: float
    action: str | None
    origin: str | None
    converged: bool
    correct: bool = True
    novelty: bool = False


def classify_tier(
    f: float,
    action: str | None,
    origin: str | None,
    converged: bool,
    policy,
    *,
    tau_auto: float | None = None,
    tau_review: float | None = None,
    novelty: bool = False,
) -> int:
    """Return 1, 2 or 3 for a case (see module docstring)."""
    tau_auto = policy.tiers.tau_auto if tau_auto is None else tau_auto
    tau_review = policy.tiers.tau_review if tau_review is None else tau_review
    action_value = _value(action)
    origin_value = _value(origin)

    spec = policy.actions.get(action_value) if action_value else None
    risk = spec.risk if spec else "high"
    reversible = spec.reversible if spec else False
    never = set(policy.never_automate) | set(NEVER_AUTOMATE_DEFAULT)

    # Tier 3 (human-led) — get this out of the way first.
    if (
        action_value is None
        or action_value in never
        or not converged
        or novelty
        or origin_value in ("adversarial", "unknown")
        or risk == "high"
        or f is None
        or f < tau_review
    ):
        return 3

    # Tier 1 (autonomous) — the strict bar.
    if (
        f >= tau_auto
        and risk == "low"
        and reversible
        and converged
        and origin_value == "data_quality"
    ):
        return 1

    return 2


def choose_thresholds(cases: list[RoutingCase], policy, error_target: float | None = None) -> dict:
    """Pick ``tau_auto`` (lowest meeting the tier-1 error target) and ``tau_review``.

    ``tau_auto`` is the lowest threshold for which autonomous decisions on
    validation have error <= ``error_target`` (maximising coverage). ``tau_review``
    is the lowest ``f`` at which decisions are better than chance (calibration
    crossing 0.5), capped at ``tau_auto``.
    """
    error_target = policy.error_target if error_target is None else error_target
    fs = sorted({c.f for c in cases})
    if not fs:
        return {"tau_auto": policy.tiers.tau_auto, "tau_review": policy.tiers.tau_review}

    tau_auto = fs[-1]
    for tau in fs:  # ascending: first (lowest) that meets the target
        tier1 = [
            c
            for c in cases
            if classify_tier(
                c.f, c.action, c.origin, c.converged, policy, tau_auto=tau, tau_review=0.0
            )
            == 1
        ]
        if not tier1:
            continue
        error = 1.0 - sum(c.correct for c in tier1) / len(tier1)
        if error <= error_target:
            tau_auto = tau
            break

    # tau_review: lowest f whose calibration accuracy (in a local window) >= 0.5.
    tau_review = 0.5
    for tau in fs:
        near = [c for c in cases if c.f >= tau]
        if near and sum(c.correct for c in near) / len(near) >= 0.5 and tau <= tau_auto:
            tau_review = tau
            break
    return {"tau_auto": float(tau_auto), "tau_review": float(min(tau_review, tau_auto))}


def route_all(cases: list[RoutingCase], policy, thresholds: dict) -> list[int]:
    return [
        classify_tier(
            c.f,
            c.action,
            c.origin,
            c.converged,
            policy,
            tau_auto=thresholds["tau_auto"],
            tau_review=thresholds["tau_review"],
            novelty=c.novelty,
        )
        for c in cases
    ]
