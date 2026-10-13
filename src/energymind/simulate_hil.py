"""M8 - human-in-the-loop simulation and trust metrics (blueprint Section 13).

Three simulated operators act on the cases that routing escalates:

* **rubber-stamper** - approves almost everything (over-trust);
* **skeptic** - rejects a fixed fraction at random (under-trust);
* **calibrated** - approves with probability equal to the confidence ``f``.

Each routing strategy (always-auto, a single threshold on the raw model
probability, and the tiered ``f`` routing) is scored on final error and operator
load, and the **risk-coverage curve** compares ``f`` against the raw probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from energymind.policy import route_all


@dataclass
class Operator:
    """A simulated operator that accepts or rejects the AI proposal."""

    name: str
    approve: object  # callable(case, rng) -> bool

    def decide(self, case: dict, rng) -> bool:
        return bool(self.approve(case, rng))


def rubber_stamper(p: float = 0.95) -> Operator:
    return Operator("rubber_stamper", lambda case, rng: rng.random() < p)


def skeptic(reject_p: float = 0.4) -> Operator:
    return Operator("skeptic", lambda case, rng: rng.random() >= reject_p)


def calibrated_operator() -> Operator:
    return Operator("calibrated", lambda case, rng: rng.random() < float(case.get("f", 0.5)))


HUMAN_RESOLVE_ACCURACY = 0.85  # a human, once involved, resolves correctly with this prob


# --------------------------------------------------------------------------- #
# Routing strategies
# --------------------------------------------------------------------------- #


def route_always_auto(cases: list[dict]) -> list[int]:
    return [1] * len(cases)


def route_single_threshold(cases: list[dict], tau: float) -> list[int]:
    """One threshold on the raw model probability (no tiers, no risk)."""
    return [1 if c.get("model_conf", 0.0) >= tau else 3 for c in cases]


def route_tiered(cases: list[dict], policy, thresholds: dict) -> list[int]:
    from energymind.policy import RoutingCase

    rcases = [
        RoutingCase(
            f=c["f"],
            action=c["action"],
            origin=c["origin"],
            converged=c["converged"],
            correct=c["correct"],
            novelty=c.get("novelty", False),
        )
        for c in cases
    ]
    return route_all(rcases, policy, thresholds)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def simulate(cases: list[dict], tiers: list[int], operator: Operator, seed: int = 0) -> dict:
    """Run one strategy+operator over the cases and return trust metrics."""
    rng = np.random.default_rng(seed)
    approved_wrong = approved = rejected_right = rejected = 0
    final_correct = 0
    escalating = 0
    escalating_ai_right_confident = 0

    for case, tier in zip(cases, tiers, strict=True):
        proposal_correct = bool(case["correct"])
        f = float(case.get("f", 0.5))
        if tier == 1:
            final_correct += int(proposal_correct)
            continue

        escalating += 1
        if proposal_correct and f >= 0.8:
            escalating_ai_right_confident += 1
        accept = operator.decide({**case, "f": f}, rng)
        if accept:
            approved += 1
            if not proposal_correct:
                approved_wrong += 1
            final_correct += int(proposal_correct)
        else:
            rejected += 1
            if proposal_correct:
                rejected_right += 1
            # a human resolves it
            final_correct += int(rng.random() < HUMAN_RESOLVE_ACCURACY)

    n = len(cases)
    return {
        "operator": operator.name,
        "n": n,
        "coverage_auto": round(sum(t == 1 for t in tiers) / n, 4) if n else 0.0,
        "final_error": round(1 - final_correct / n, 4) if n else 0.0,
        "automation_bias_rate": round(approved_wrong / approved, 4) if approved else 0.0,
        "unjustified_rejection_rate": round(rejected_right / rejected, 4) if rejected else 0.0,
        "appropriate_reliance": round(final_correct / n, 4) if n else 0.0,
        "unnecessary_escalation": round(escalating_ai_right_confident / n, 4) if n else 0.0,
        "operator_load": escalating,
    }


def run_simulation(
    cases: list[dict],
    policy,
    thresholds: dict,
    *,
    operators: list[Operator] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Score every (strategy, operator) combination."""
    operators = operators or [rubber_stamper(), skeptic(), calibrated_operator()]
    strategies = {
        "always_auto": route_always_auto(cases),
        "single_threshold": route_single_threshold(cases, tau=0.8),
        "tiered_f": route_tiered(cases, policy, thresholds),
    }
    rows = []
    for strategy_name, tiers in strategies.items():
        for operator in operators:
            metrics = simulate(cases, tiers, operator, seed=seed)
            rows.append({"strategy": strategy_name, **metrics})
    return rows


# --------------------------------------------------------------------------- #
# Risk-coverage curve
# --------------------------------------------------------------------------- #


def risk_coverage(cases: list[dict], score_key: str = "f") -> dict:
    """Coverage vs error among the most-confident cases, for a score.

    Returns ``{"coverage": [...], "risk": [...]}`` where ``risk`` is the error
    among the top-``coverage`` fraction of cases by ``score_key``.
    """
    if not cases:
        return {"coverage": [], "risk": []}
    order = sorted(cases, key=lambda c: c.get(score_key, 0.0), reverse=True)
    correct = np.array([bool(c["correct"]) for c in order], dtype=float)
    coverage, risk = [], []
    for k in range(1, len(correct) + 1):
        coverage.append(k / len(correct))
        risk.append(1.0 - correct[:k].mean())
    return {"coverage": coverage, "risk": risk}


def risk_coverage_summary(cases: list[dict]) -> dict:
    """Compare the area under the risk-coverage curve for f vs model_conf."""

    def area(points: dict) -> float:
        if not points["coverage"]:
            return 0.0
        return float(np.trapezoid(points["risk"], points["coverage"]))

    f_curve = risk_coverage(cases, "f")
    p_curve = risk_coverage(cases, "model_conf")
    return {
        "auc_risk_f": round(area(f_curve), 4),
        "auc_risk_model_conf": round(area(p_curve), 4),
        "f_curve": f_curve,
        "model_conf_curve": p_curve,
    }
