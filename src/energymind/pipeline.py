"""M8 - end-to-end decision pipeline (blueprint Step 12).

    case -> "seen before?" -> (agents | reuse) -> confidence f -> tier -> audit log

The pipeline connects the M6 memory, the M7 two-agent deliberation, the M8
confidence model and the tiered router, and writes every step to the append-only
audit log. It also produces the calibration report, the risk-coverage curve and
the simulated-operator table.

    python -m energymind.pipeline --demo --n-val 30 --n-test 20
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from energymind import evaluate, io, policy
from energymind.agent_pipeline import _build_record, leakage_violations, run_case
from energymind.agents import RuleBasedAnalyst, RuleBasedValidator
from energymind.agents.base import DecisionOrigin
from energymind.confidence import (
    HISTORY_PRIOR,
    ConfidenceModel,
    Signal,
    expected_calibration_error,
    reliability_curve,
    risk_coverage_summary,
    variance_inflation_factors,
)
from energymind.config import Settings, load_settings
from energymind.detect.rules import Rules
from energymind.benchmark import DETECTOR_FACTORIES
from energymind.splits import assign_temporal
from energymind.policy import RoutingCase, action_risk, choose_thresholds
from energymind.retrieval import Case, InMemoryRetriever
from energymind.review import (
    ORIGIN_NAMES,
    allowed_action_names,
    apply_custom_decision,
    apply_human_decision,
    coerce_approval,
    coerce_custom,
    prompt_human,
    review_card,
)
from energymind.store import AuditLog, MemoryStore

logger = logging.getLogger(__name__)

#: Map from the injected origin to the action the label implies (for scoring).
TRUE_ACTION = {
    "data_quality": "flag_data_suspect",
    "physical_event": "coordinate_with_grid_operator",
    "adversarial": "investigate_tampering",
}


def _model_conf(record: dict) -> float:
    cls = record.get("classification") or {}
    probs = cls.get("origin_probs") or cls.get("type_probs") or {}
    if probs:
        return float(max(probs.values()))
    det = record.get("detection") or {}
    return float(det.get("p_anomaly", 0.5))


def _proposal_prob(record: dict, origin: str | None) -> float | None:
    """Classifier probability of the *proposed origin* (for the A feature)."""
    probs = (record.get("classification") or {}).get("origin_probs") or {}
    if origin and origin in probs:
        return float(probs[origin])
    return None


def _signal(record: dict, outcome, r_max: int) -> Signal:
    result = outcome.deliberation
    skipped = outcome.path == "seen_before"
    return Signal(
        model_conf=_model_conf(record),
        # On the reuse path deliberation did not run: convergence is 0, and D=1 (below)
        # already accounts for it -- so L_eff is 0 rather than an artificial 1.
        converged=bool(result.converged) if result else False,
        rounds_used=result.rounds_used if result else 0,
        r_max=r_max,
        proposed_origin=outcome.proposal_origin,
        histories=outcome.history,
        checks=record.get("evidence") or [],
        completeness=float(record.get("completeness", 1.0)),
        deliberation_skipped=skipped,
        proposal_prob=_proposal_prob(record, outcome.proposal_origin),
    )


def _correct(outcome, true_origin: str) -> bool:
    """Fix 4: correctness is the *proposed* origin vs the truth, not convergence."""
    return outcome.proposal_origin is not None and outcome.proposal_origin == true_origin


# --------------------------------------------------------------------------- #
# Case assembly
# --------------------------------------------------------------------------- #


def _operating_detector(settings, frame, events):
    """Fit the configured unsupervised detector on train and threshold on val."""
    name = str(settings.models.decision.get("operating_detector", "ensemble+ph"))
    detector_name = name.removesuffix("+ph")
    if detector_name == "supervised" or detector_name not in DETECTOR_FACTORIES:
        raise ValueError(
            f"decision.operating_detector must be an unsupervised detector: {name}"
        )
    parts = {
        split: frame[assign_temporal(frame, settings.data) == split]
        for split in ("train", "val", "test")
    }
    detector = DETECTOR_FACTORIES[detector_name](settings)
    detector.fit(parts["train"])
    scores = detector.score(frame)
    budget = settings.models.evaluation.false_alarm_budget_per_household_week
    tol = settings.models.evaluation.tolerance_intervals
    threshold = evaluate.select_threshold_per_household(
        parts["val"], scores.loc[parts["val"].index], events, budget, tol
    )
    return name, scores, threshold


def _detected_event_ids(frame, events, scores, threshold, tol_intervals):
    """Return event IDs with at least one operating-detector alert."""
    tol = tol_intervals * evaluate.INTERVAL
    detected = set()
    for event in events.itertuples(index=False):
        mask = (frame[io.HOUSEHOLD_COL] == event.household_id) & frame[io.UTC_COL].between(
            event.t_start - tol, event.t_end + tol, inclusive="both"
        )
        if not mask.any():
            continue
        values = scores.loc[frame.index[mask]]
        limits = frame.loc[mask, io.HOUSEHOLD_COL].map(threshold)
        if bool((values.to_numpy(dtype=float) > limits.to_numpy(dtype=float)).any()):
            detected.add(str(event.anomaly_id))
    return detected


def build_cases(
    settings: Settings,
    store: MemoryStore,
    *,
    n_val: int,
    n_test: int,
    r_max: int,
    classifier=None,
    detector_scores=None,
    frame=None,
    events=None,
    analyst=None,
    validator=None,
) -> tuple[list[dict], int]:
    """Run the pipeline over val/test events (chronologically).

    ``analyst`` / ``validator`` override the two agents (default: the rule-based
    agents). Pass cloud/LLM agents here to run the same pipeline with them.

    Returns ``(cases, leakage_violations)``; the latter counts retrieved precedents
    that are not strictly earlier than their query (must be 0).
    """
    frame = (
        frame
        if frame is not None
        else pd.read_parquet(io.resolve("data/features/features.parquet"))
    )
    events = (
        events
        if events is not None
        else evaluate.expand_events(pd.read_parquet(io.resolve("data/injected/labels.parquet")))
    )
    operating_name = "provided"
    operating_threshold = None
    if detector_scores is None:
        operating_name, detector_scores, operating_threshold = _operating_detector(
            settings, frame, events
        )

    t = settings.data.splits.temporal
    off = settings.data.time.assume_utc_offset_h
    val_start = pd.Timestamp(t.val.start).tz_localize("UTC") - pd.Timedelta(hours=off)
    test_start = pd.Timestamp(t.test.start).tz_localize("UTC") - pd.Timedelta(hours=off)

    # classify one event at a time via the (real) classifier if provided
    proba = {}
    if classifier is not None:
        from energymind.classify import build_event_dataset

        x, meta = build_event_dataset(frame, events, {"rules": detector_scores})
        pred = classifier.predict(x)
        for i, anomaly_id in enumerate(meta["anomaly_id"]):
            proba[anomaly_id] = {
                "type_probs": {
                    classifier.classes_type[j]: float(pred["type_proba"][i][j])
                    for j in range(len(classifier.classes_type))
                },
                "origin_probs": {
                    classifier.classes_origin[j]: float(pred["origin_proba"][i][j])
                    for j in range(len(classifier.classes_origin))
                },
            }

    cases: list[dict] = []
    total_leakage = 0
    detected = _detected_event_ids(
        frame,
        events,
        detector_scores,
        operating_threshold
        if operating_threshold is not None
        else pd.Series(float("-inf"), index=frame[io.HOUSEHOLD_COL].unique()),
        settings.models.evaluation.tolerance_intervals,
    ) if operating_threshold is not None else set(events["anomaly_id"].astype(str))
    missed = {"val": 0, "test": 0}
    for split_name, start, n in (("val", val_start, n_val), ("test", test_start, n_test)):
        subset = events[events["t_start"] >= start]
        # chronological order: later cases may learn from earlier ones (causal memory)
        ordered = subset.sort_values("t_start")
        all_ids = ordered["anomaly_id"].drop_duplicates().tolist()
        ids = [str(value) for value in all_ids if str(value) in detected]
        missed[split_name] = len(all_ids) - len(ids)
        for anomaly_id in ids:
            # Predictions come from the classifier (never the labels); the label is
            # kept only for scoring, outside the record.
            record = _build_record(
                frame, events, anomaly_id, detector_scores, classification=proba.get(anomaly_id)
            )
            row = events.loc[events.anomaly_id == anomaly_id].iloc[0]
            true_origin = str(row["origin"])
            true_type = str(row["type"])
            outcome = run_case(
                record,
                store,
                settings=settings,
                r_max=r_max,
                analyst=analyst or RuleBasedAnalyst(),
                validator=validator or RuleBasedValidator(),
            )
            correct = _correct(outcome, true_origin)
            total_leakage += leakage_violations(outcome.history, record.get("t_start"))

            # write the decided case back so later cases can reuse it (causal memory).
            # Fix 4: the *proposed* decision is what is recorded; it is reusable only
            # when it matches the observed outcome.
            if outcome.proposal_origin and record.get("signature"):
                store.remember_case(
                    Case(
                        anomaly_id=str(anomaly_id),
                        signature=[float(x) for x in record["signature"]],
                        type_pred=record.get("type_pred") or "unknown",
                        origin_pred=outcome.proposal_origin,
                        status="verified" if correct else "rejected",
                        severity=record.get("severity"),
                        decision_action=outcome.proposal_action,
                        t_start=record.get("t_start"),
                        t_end=record.get("t_end"),
                        source="decision",
                    ),
                    event_type="decision_writeback",
                )

            signal = _signal(record, outcome, r_max)
            cases.append(
                {
                    "anomaly_id": str(anomaly_id),
                    "split": split_name,
                    "true_origin": true_origin,
                    "true_type": true_type,
                    "true_action": TRUE_ACTION.get(true_origin),
                    "record": record,
                    "outcome": outcome,
                    "signal": signal,
                    "model_conf": _model_conf(record),
                    "correct": correct,
                    "action": outcome.proposal_action,
                    "origin": outcome.proposal_origin,
                    "proposal_source": outcome.proposal_source,
                    "converged": bool(outcome.deliberation.converged)
                    if outcome.deliberation
                    else True,
                }
            )
    return cases, total_leakage, {"detector": operating_name, "threshold": operating_threshold, "missed_by_detector": missed}


# --------------------------------------------------------------------------- #
# End-to-end run
# --------------------------------------------------------------------------- #


def run(
    settings: Settings | None = None,
    *,
    n_val: int = 40,
    n_test: int = 20,
    r_max: int | None = None,
    qdrant: bool = False,
    reset: bool = False,
    analyst=None,
    validator=None,
    human_decisions: dict[str, object] | None = None,
    human_default: str | None = None,
    human_action_default: dict | None = None,
    interactive: bool = False,
    human_prompt=None,
) -> dict:
    """Fit f on validation, route the test cases, and log every decision.

    ``analyst`` / ``validator`` override the two agents (default: rule-based). To use the
    cloud/LLM agents, pass the objects built from ``configs/agents.yaml`` (see
    ``energymind.agents.llm_agent``).

    Tier 2 (uncertain) is answered with ``approve`` / ``reject``; tier 3 (high-risk) is
    answered with the human's own action and origin. For an escalated case the answer is
    resolved in this order:

    1. ``human_decisions[anomaly_id]`` -- explicit programmatic answer (tier 2: a string;
       tier 3: a ``{"action", "origin"}`` mapping or ``(action, origin)`` pair);
    2. if ``interactive`` -- prompt the real terminal (``human_prompt`` overrides the
       reader, e.g. for tests);
    3. ``human_default`` (tier 2) / ``human_action_default`` (tier 3) -- a batch answer;
    4. otherwise the case stays ``pending_human`` (never silently approved/rejected).
    """
    from energymind.store import ingest_events

    settings = settings or load_settings()
    r_max = r_max or settings.agents.r_max

    retriever = None
    if qdrant:
        from energymind.retrieval import LocalQdrantRetriever

        retriever = LocalQdrantRetriever(settings.store.qdrant)
        if retriever.count() > 0 and not reset:
            raise RuntimeError(
                "the Qdrant collection is not empty; pass --reset to avoid reusing a "
                "collection that may hold labels from another run"
            )
        if reset:
            retriever.delete_collection()
        retriever.ensure_collection()
    else:
        retriever = InMemoryRetriever(settings.store.retrieval.tau_sim)
    audit = AuditLog(sqlite_path=":memory:")
    store = MemoryStore(retriever, audit)
    if retriever.count() == 0:
        ingest_events(settings, store)  # train-only, time-causal history

    # a real classifier over the events (used for A and the classifier evidence)
    frame = pd.read_parquet(io.resolve("data/features/features.parquet"))
    events = evaluate.expand_events(pd.read_parquet(io.resolve("data/injected/labels.parquet")))
    operating_name, detector_scores, operating_threshold = _operating_detector(
        settings, frame, events
    )
    classifier = None
    try:
        from energymind.classify import EventClassifier, assign_event_split, build_event_dataset

        x, meta = build_event_dataset(frame, events, {"rules": detector_scores})
        meta = meta.assign(split=assign_event_split(meta, settings, frame))
        train = meta["split"] == "train"
        if train.sum() > 20:
            classifier = EventClassifier.fit(
                x[train], meta.loc[train, "type"], meta.loc[train, "origin"], "lightgbm"
            )
    except Exception:  # lightgbm optional
        classifier = None

    cases, leakage_count, detector_report = build_cases(
        settings,
        store,
        n_val=n_val,
        n_test=n_test,
        r_max=r_max,
        classifier=classifier,
        detector_scores=None,
        frame=frame,
        events=events,
        analyst=analyst,
        validator=validator,
    )
    val = [c for c in cases if c["split"] == "val"]
    test = [c for c in cases if c["split"] == "test"]

    # Fix 6.4: fit A (isotonic) on a *separate* split from beta. val is chronological,
    # so the earlier half calibrates A and the later half fits beta / the base rate.
    # (Fix 8 refines these bounds to the val-A/val-B calendar windows.)
    half = max(1, len(val) // 2)
    val_a = val[:half]
    val_b = val[half:] or val
    beta_signals = [c["signal"] for c in val_b]
    beta_y = np.array([c["correct"] for c in val_b], dtype=float)
    a_signals = [c["signal"] for c in val_a]
    a_y = np.array([c["correct"] for c in val_a], dtype=float)

    # Fix 6.5: H shrinks toward the validation base rate p0 (accuracy on the beta split).
    base_rate = float(beta_y.mean()) if len(beta_y) else HISTORY_PRIOR
    for c in cases:
        c["signal"].base_rate = base_rate

    model = ConfidenceModel().fit(beta_signals, beta_y, a_signals=a_signals, a_y=a_y)
    for c in cases:
        c["f"] = float(model.predict([c["signal"]])[0])

    coefficients_ci = model.coefficient_cis(beta_signals, beta_y)
    vif = variance_inflation_factors(beta_signals, model.calibrator)
    high_vif = {name: value for name, value in vif.items() if value > 5}
    if high_vif:
        logger.warning("high variance-inflation factors (> 5): %s", high_vif)

    thresholds = choose_thresholds(
        [
            RoutingCase(
                f=c["f"],
                action=c["action"],
                origin=c["origin"],
                converged=c["converged"],
                correct=c["correct"],
            )
            for c in val
        ],
        settings.policy,
    )
    test_tiers = policy.route_all(
        [
            RoutingCase(
                f=c["f"],
                action=c["action"],
                origin=c["origin"],
                converged=c["converged"],
                correct=c["correct"],
            )
            for c in test
        ],
        settings.policy,
        thresholds,
    )
    written_back = 0
    review_cards: dict[str, str] = {}
    allowed_actions = allowed_action_names(settings.policy)
    for case, tier in zip(test, test_tiers, strict=True):
        case["tier"] = tier
        auto = tier == 1
        case["requires_human"] = not auto
        case["human_correct"] = None

        if auto:
            # The two agents agreed on a low-risk, reversible data-quality decision.
            case["resolution"] = "auto"
            case["final_action"] = case["action"]
            case["final_origin"] = case["origin"]
        else:
            # Tier 2 (uncertain) asks approve/reject; tier 3 (high-risk) asks the human
            # for their own action/origin. The agents' proposal is never auto-approved.
            risk = action_risk(case.get("action"), settings.policy)
            card = review_card(case, tier=tier, confidence=case["f"], risk=risk)
            case["review_card"] = card
            review_cards[case["anomaly_id"]] = card
            # The request is logged *before* the answer is obtained (interactive or not).
            store.audit.append_event(
                actor="human",
                event_type="human_review_requested",
                payload={"tier": tier, "card": card},
                anomaly_id=case["anomaly_id"],
            )

            given = (human_decisions or {}).get(case["anomaly_id"])
            reader = human_prompt or prompt_human
            if tier == 2:
                answer = coerce_approval(given)
                if answer is None and interactive:
                    answer = coerce_approval(reader(card, tier=2, allowed_actions=allowed_actions))
                if answer is None:
                    answer = coerce_approval(human_default)
            else:
                answer = coerce_custom(given)
                if answer is None and interactive:
                    answer = coerce_custom(reader(card, tier=3, allowed_actions=allowed_actions))
                if answer is None:
                    answer = coerce_custom(human_action_default)

            if tier == 3 and answer is not None:
                # Validate the human's own action/origin against the policy's sets.
                if answer.action not in allowed_actions:
                    raise ValueError(
                        f"tier-3 action {answer.action!r} is not in the allowed action set"
                    )
                if answer.origin not in ORIGIN_NAMES:
                    raise ValueError(f"tier-3 origin {answer.origin!r} is not a valid origin")

            if answer is None:
                # Non-interactive / programmatic mode, or no valid tier-3 action: wait.
                case["resolution"] = "pending_human"
                case["final_action"] = None
                case["final_origin"] = None
            elif tier == 2:
                applied = apply_human_decision(case, answer)
                case["resolution"] = applied["resolution"]
                case["final_action"] = applied["final_action"]
                case["final_origin"] = applied["final_origin"]
                store.audit.append_event(
                    actor="human",
                    event_type="human_decision",
                    payload={
                        "human_decision": applied["human_decision"],
                        "resolution": applied["resolution"],
                        "note": applied["note"],
                    },
                    anomaly_id=case["anomaly_id"],
                )
            else:
                applied = apply_custom_decision(case, answer)
                case["resolution"] = applied["resolution"]
                case["final_action"] = applied["final_action"]
                case["final_origin"] = applied["final_origin"]
                # Correctness of the *human's* decision (not the agents' proposal).
                case["human_correct"] = applied["final_origin"] == case["true_origin"]
                store.audit.append_event(
                    actor="human",
                    event_type="human_custom_decision",
                    payload={
                        "action": applied["final_action"],
                        "origin": applied["final_origin"],
                        "resolution": applied["resolution"],
                        "correct": case["human_correct"],
                        "note": applied["note"],
                    },
                    anomaly_id=case["anomaly_id"],
                )

        # Memory hygiene: only a decision that matches the observed outcome is reusable.
        if case["resolution"] in ("auto", "approved"):
            status = "verified" if case["correct"] else "rejected"
        elif case["resolution"] == "custom":
            # The tier-3 human decision is reusable only if it matches the outcome.
            status = "verified" if case["human_correct"] else "rejected"
        else:
            status = "proposed"
        store.record_decision(
            case["anomaly_id"],
            action=case["final_action"],
            origin=case["final_origin"],
            confidence=case["f"],
            tier=tier,
            status=status,
        )
        written_back += 1

    y_test = np.array([c["correct"] for c in test], dtype=float)
    f_test = np.array([c["f"] for c in test], dtype=float)
    centers, accs = reliability_curve(y_test, f_test)

    rc = risk_coverage_summary(test)

    # The human-in-the-loop step: what was escalated and how it was resolved.
    human_review = {
        "requested": len(review_cards),
        "auto": sum(1 for c in test if c["resolution"] == "auto"),
        "approved": sum(1 for c in test if c["resolution"] == "approved"),
        "rejected": sum(1 for c in test if c["resolution"] == "rejected"),
        "custom": sum(1 for c in test if c["resolution"] == "custom"),
        "more_info_requested": sum(1 for c in test if c["resolution"] == "more_info_requested"),
        "pending_human": sum(1 for c in test if c["resolution"] == "pending_human"),
    }

    # Fix 4: abstentions (Analyst gave no usable origin) and the fate of non-converged
    # cases, which are no longer correct-by-construction.
    abstention_rate = (
        round(sum(1 for c in test if c["origin"] in (None, "unknown")) / len(test), 4)
        if test
        else None
    )
    nonconverged = [c for c in test if not c["converged"]]
    nonconverged_correct_fraction = (
        round(sum(1 for c in nonconverged if c["correct"]) / len(nonconverged), 4)
        if nonconverged
        else None
    )
    proposal_source_counts = {
        src: int(sum(1 for c in test if c["proposal_source"] == src))
        for src in ("seen_before", "converged", "analyst_fallback")
    }

    # Fix 5: convergence rate, overall and per (true) origin, over val + test.
    combined = val + test
    convergence_rate = (
        round(sum(1 for c in combined if c["converged"]) / len(combined), 4) if combined else 0.0
    )
    convergence_rate_by_origin = {}
    for origin in sorted({c["true_origin"] for c in combined}):
        group = [c for c in combined if c["true_origin"] == origin]
        n_conv = sum(1 for c in group if c["converged"])
        convergence_rate_by_origin[origin] = {
            "n": len(group),
            "converged": n_conv,
            "rate": round(n_conv / len(group), 4) if group else None,
        }

    return {
        "operating_detector": detector_report["detector"],
        "operating_threshold": (
            {str(k): float(v) for k, v in detector_report["threshold"].items()}
            if isinstance(detector_report["threshold"], pd.Series)
            else detector_report["threshold"]
        ),
        "missed_by_detector": detector_report["missed_by_detector"],
        "false_alarm_cases_included": False,
        "thresholds": thresholds,
        "coefficients": model.coefficients,
        "coefficients_ci": coefficients_ci,
        "vif": vif,
        "n_val": len(val),
        "n_val_a": len(val_a),
        "n_val_b": len(val_b),
        "base_rate": round(base_rate, 4),
        "n_test": len(test),
        "val_error": round(1 - float(np.mean([c["correct"] for c in val])), 4) if val else None,
        "test_error": round(1 - float(np.mean(y_test)), 4) if test else None,
        "ece_test": round(expected_calibration_error(y_test, f_test), 4) if test else None,
        "abstention_rate": abstention_rate,
        "nonconverged_correct_fraction": nonconverged_correct_fraction,
        "proposal_source_counts": proposal_source_counts,
        "convergence_rate": convergence_rate,
        "convergence_rate_by_origin": convergence_rate_by_origin,
        "review_cards": review_cards,
        "human_review": human_review,
        "reliability": {"centers": centers.tolist(), "accuracy": accs.tolist()},
        "tier_counts": {t: int(sum(1 for c in test if c["tier"] == t)) for t in (1, 2, 3)},
        "risk_coverage": rc,
        "audit": audit.verify_chain(),
        "written_back": written_back,
        "leakage_violations": int(leakage_count),
        "audit_events": [
            {
                "seq": e["seq"],
                "anomaly_id": e["anomaly_id"],
                "actor": e["actor"],
                "event_type": e["event_type"],
                "payload": e["payload"],
            }
            for e in audit.events()
        ],
        "cases": [
            {
                "anomaly_id": c["anomaly_id"],
                "split": c["split"],
                "f": round(c["f"], 4),
                "tier": c.get("tier"),
                "action": c["action"],
                "origin": c["origin"],
                "true_origin": c["true_origin"],
                "converged": c["converged"],
                "correct": c["correct"],
                "final_action": c.get("final_action"),
                "final_origin": c.get("final_origin"),
                "resolution": c.get("resolution"),
                "human_correct": c.get("human_correct"),
            }
            for c in cases
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M8 end-to-end decision pipeline.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--qdrant", action="store_true")
    parser.add_argument("--reset", action="store_true", help="recreate the Qdrant collection")
    parser.add_argument("--demo", action="store_true")
    human_mode = parser.add_mutually_exclusive_group()
    human_mode.add_argument(
        "--interactive-human-review",
        action="store_true",
        help="prompt the real terminal for each escalated (tier 2/3) case",
    )
    human_mode.add_argument(
        "--human-answer",
        choices=["none", "approve", "reject", "request_more_info"],
        default="none",
        help="apply one tier-2 answer (approve/reject) to every tier-2 card",
    )
    parser.add_argument(
        "--human-action",
        default=None,
        help="tier-3 batch answer: the human's own action (from the policy action set)",
    )
    parser.add_argument(
        "--human-origin",
        default=None,
        help="tier-3 batch answer: the human's own origin",
    )
    parser.add_argument("--output-json", default="results/m8_decision.json")
    args = parser.parse_args(argv)

    io.load_env()
    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    if bool(args.human_action) != bool(args.human_origin):
        parser.error("--human-action and --human-origin must be given together")
    human_default = None if args.human_answer == "none" else args.human_answer
    human_action_default = (
        {"action": args.human_action, "origin": args.human_origin} if args.human_action else None
    )
    report = run(
        settings,
        n_val=args.n_val,
        n_test=args.n_test,
        qdrant=args.qdrant,
        reset=args.reset,
        human_default=human_default,
        human_action_default=human_action_default,
        interactive=args.interactive_human_review,
    )

    print("M8 decision pipeline")
    print(f"  cases            : val={report['n_val']} test={report['n_test']}")
    print(f"  thresholds       : {report['thresholds']}")
    print(f"  test error       : {report['test_error']}  ECE={report['ece_test']}")
    print(f"  tier counts      : {report['tier_counts']}")
    print(f"  coefficients     : {report['coefficients']}")
    print(f"  coefficients CI  : {report['coefficients_ci']}")
    print(f"  VIF              : {report['vif']} (warn if > 5)")
    print(f"  leakage violations: {report['leakage_violations']} (must be 0)")
    print(f"  abstention rate  : {report['abstention_rate']}")
    print(f"  convergence rate : {report['convergence_rate']} (must be > 0)")
    print(
        f"  risk-coverage    : f AUC={report['risk_coverage']['auc_risk_f']} "
        f"vs model_conf AUC={report['risk_coverage']['auc_risk_model_conf']}"
    )
    print(f"  audit chain      : {report['audit']['valid']} ({report['audit']['events']} events)")
    print(f"  human review     : {report['human_review']}")
    if report["review_cards"]:
        first = next(iter(report["review_cards"].values()))
        print("\n  example review card sent to the human:")
        for line in first.splitlines():
            print(f"    {line}")
    out = io.resolve(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {out}")
    _ = DecisionOrigin, json
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
