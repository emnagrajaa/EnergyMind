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

import numpy as np
import pandas as pd

from energymind import evaluate, io, policy
from energymind.agent_pipeline import _build_record, leakage_violations, run_case
from energymind.agents import RuleBasedAnalyst, RuleBasedValidator
from energymind.agents.base import DecisionOrigin
from energymind.confidence import (
    ConfidenceModel,
    Signal,
    expected_calibration_error,
    reliability_curve,
    risk_coverage_summary,
)
from energymind.config import Settings, load_settings
from energymind.detect.rules import Rules
from energymind.policy import RoutingCase, choose_thresholds
from energymind.retrieval import Case, InMemoryRetriever
from energymind.review import apply_human_decision, review_card
from energymind.store import AuditLog, MemoryStore

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


def _signal(record: dict, outcome, r_max: int) -> Signal:
    result = outcome.deliberation
    return Signal(
        model_conf=_model_conf(record),
        converged=bool(result.converged) if result else True,  # reuse path = converged by gate
        rounds_used=result.rounds_used if result else 1,
        r_max=r_max,
        proposed_origin=outcome.proposal_origin,
        histories=outcome.history,
        checks=record.get("evidence") or [],
        completeness=float(record.get("completeness", 1.0)),
        seen_before=outcome.path == "seen_before",
    )


def _correct(outcome, true_origin: str) -> bool:
    """Fix 4: correctness is the *proposed* origin vs the truth, not convergence."""
    return outcome.proposal_origin is not None and outcome.proposal_origin == true_origin


# --------------------------------------------------------------------------- #
# Case assembly
# --------------------------------------------------------------------------- #


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
    detector_scores = (
        detector_scores if detector_scores is not None else Rules().fit(frame).score(frame)
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
    for split_name, start, n in (("val", val_start, n_val), ("test", test_start, n_test)):
        subset = events[events["t_start"] >= start]
        # chronological order: later cases may learn from earlier ones (causal memory)
        ordered = subset.sort_values("t_start")
        ids = ordered["anomaly_id"].drop_duplicates().tolist()[:n]
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
    return cases, total_leakage


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
    human_decisions: dict[str, str] | None = None,
    human_default: str | None = None,
) -> dict:
    """Fit f on validation, route the test cases, and log every decision.

    ``analyst`` / ``validator`` override the two agents (default: rule-based). To use the
    cloud/LLM agents, pass the objects built from ``configs/agents.yaml`` (see
    ``energymind.agents.llm_agent``).

    ``human_decisions`` maps ``anomaly_id -> "approve" | "reject" | "request_more_info"``
    for cases escalated to the human. Without an entry, an escalated case stays
    ``pending_human`` (the system waits for the real reviewer rather than guessing).
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
    detector_scores = Rules().fit(frame).score(frame)
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

    cases, leakage_count = build_cases(
        settings,
        store,
        n_val=n_val,
        n_test=n_test,
        r_max=r_max,
        classifier=classifier,
        detector_scores=detector_scores,
        frame=frame,
        events=events,
        analyst=analyst,
        validator=validator,
    )
    val = [c for c in cases if c["split"] == "val"]
    test = [c for c in cases if c["split"] == "test"]

    model = ConfidenceModel().fit([c["signal"] for c in val], [c["correct"] for c in val])
    for c in cases:
        c["f"] = float(model.predict([c["signal"]])[0])

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
    for case, tier in zip(test, test_tiers, strict=True):
        case["tier"] = tier
        auto = tier == 1
        case["requires_human"] = not auto

        if auto:
            # The two agents agreed on a low-risk, reversible data-quality decision.
            case["resolution"] = "auto"
            case["final_action"] = case["action"]
            case["final_origin"] = case["origin"]
        else:
            # Uncertain or high-risk: ask the human (the real user), show the card, wait.
            card = review_card(case, confidence=case["f"])
            case["review_card"] = card
            review_cards[case["anomaly_id"]] = card
            answer = (human_decisions or {}).get(case["anomaly_id"], human_default)
            if answer is None:
                case["resolution"] = "pending_human"
                case["final_action"] = None
                case["final_origin"] = None
                store.audit.append_event(
                    actor="human",
                    event_type="human_review_requested",
                    payload={"tier": tier, "card": card},
                    anomaly_id=case["anomaly_id"],
                )
            else:
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

        # Memory hygiene: only a decision that matches the observed outcome is reusable.
        if case["resolution"] in ("auto", "approved"):
            status = "verified" if case["correct"] else "rejected"
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
        "thresholds": thresholds,
        "coefficients": model.coefficients,
        "n_val": len(val),
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
                "resolution": c.get("resolution"),
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
    parser.add_argument(
        "--human-answer",
        choices=["none", "approve", "reject", "request_more_info"],
        default="none",
        help="simulate the human's answer to every escalated review card (default: wait)",
    )
    parser.add_argument("--output-json", default="results/m8_decision.json")
    args = parser.parse_args(argv)

    io.load_env()
    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    human_default = None if args.human_answer == "none" else args.human_answer
    report = run(
        settings,
        n_val=args.n_val,
        n_test=args.n_test,
        qdrant=args.qdrant,
        reset=args.reset,
        human_default=human_default,
    )

    print("M8 decision pipeline")
    print(f"  cases            : val={report['n_val']} test={report['n_test']}")
    print(f"  thresholds       : {report['thresholds']}")
    print(f"  test error       : {report['test_error']}  ECE={report['ece_test']}")
    print(f"  tier counts      : {report['tier_counts']}")
    print(f"  coefficients     : {report['coefficients']}")
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
