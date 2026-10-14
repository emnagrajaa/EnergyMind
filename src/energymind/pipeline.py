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

from energymind import evaluate, io, policy, simulate_hil
from energymind.agent_pipeline import _build_record, run_case
from energymind.agents import RuleBasedAnalyst, RuleBasedValidator
from energymind.agents.base import DecisionOrigin
from energymind.confidence import (
    ConfidenceModel,
    Signal,
    expected_calibration_error,
    reliability_curve,
)
from energymind.config import Settings, load_settings
from energymind.detect.rules import Rules
from energymind.policy import RoutingCase, choose_thresholds
from energymind.retrieval import InMemoryRetriever
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
        proposed_origin=outcome.origin.value if outcome.origin else None,
        histories=outcome.history,
        checks=record.get("evidence") or [],
        completeness=float(record.get("completeness", 1.0)),
        seen_before=outcome.path == "seen_before",
    )


def _correct(outcome, true_origin: str) -> bool:
    return outcome.origin is not None and outcome.origin.value == true_origin


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
) -> list[dict]:
    """Run the pipeline over val/test events and return per-case records."""
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
    for split_name, start, n in (("val", val_start, n_val), ("test", test_start, n_test)):
        ids = events.loc[events["t_start"] >= start, "anomaly_id"].unique()[:n]
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
                analyst=RuleBasedAnalyst(),
                validator=RuleBasedValidator(),
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
                    "correct": _correct(outcome, true_origin),
                    "action": outcome.action.value if outcome.action else None,
                    "origin": outcome.origin.value if outcome.origin else None,
                    "converged": bool(outcome.deliberation.converged)
                    if outcome.deliberation
                    else True,
                }
            )
    return cases


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
) -> dict:
    """Fit f on validation, route the test cases, and log every decision."""
    from energymind.store import ingest_events

    settings = settings or load_settings()
    r_max = r_max or settings.agents.r_max

    retriever = None
    if qdrant:
        from energymind.retrieval import LocalQdrantRetriever

        retriever = LocalQdrantRetriever(settings.store.qdrant)
        retriever.ensure_collection()
    else:
        retriever = InMemoryRetriever(settings.store.retrieval.tau_sim)
    audit = AuditLog(sqlite_path=":memory:")
    store = MemoryStore(retriever, audit)
    if retriever.count() == 0:
        ingest_events(settings, store)

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

    cases = build_cases(
        settings,
        store,
        n_val=n_val,
        n_test=n_test,
        r_max=r_max,
        classifier=classifier,
        detector_scores=detector_scores,
        frame=frame,
        events=events,
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
    for case, tier in zip(test, test_tiers, strict=True):
        case["tier"] = tier
        # Memory hygiene: only a decision consistent with the observed outcome becomes
        # reusable; an incorrect decision is recorded as rejected and not retrieved.
        status = "verified" if case["correct"] else "rejected"
        store.record_decision(
            case["anomaly_id"],
            action=case["action"],
            origin=case["origin"],
            confidence=case["f"],
            tier=tier,
            status=status,
        )
        written_back += 1

    y_test = np.array([c["correct"] for c in test], dtype=float)
    f_test = np.array([c["f"] for c in test], dtype=float)
    centers, accs = reliability_curve(y_test, f_test)

    hil_rows = simulate_hil.run_simulation(test, settings.policy, thresholds)
    rc = simulate_hil.risk_coverage_summary(test)

    return {
        "thresholds": thresholds,
        "coefficients": model.coefficients,
        "n_val": len(val),
        "n_test": len(test),
        "val_error": round(1 - float(np.mean([c["correct"] for c in val])), 4) if val else None,
        "test_error": round(1 - float(np.mean(y_test)), 4) if test else None,
        "ece_test": round(expected_calibration_error(y_test, f_test), 4) if test else None,
        "reliability": {"centers": centers.tolist(), "accuracy": accs.tolist()},
        "tier_counts": {t: int(sum(1 for c in test if c["tier"] == t)) for t in (1, 2, 3)},
        "hil": hil_rows,
        "risk_coverage": rc,
        "audit": audit.verify_chain(),
        "written_back": written_back,
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
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output-json", default="results/m8_decision.json")
    args = parser.parse_args(argv)

    io.load_env()
    settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    report = run(settings, n_val=args.n_val, n_test=args.n_test, qdrant=args.qdrant)

    print("M8 decision pipeline")
    print(f"  cases          : val={report['n_val']} test={report['n_test']}")
    print(f"  thresholds     : {report['thresholds']}")
    print(f"  test error     : {report['test_error']}  ECE={report['ece_test']}")
    print(f"  tier counts    : {report['tier_counts']}")
    print(f"  coefficients   : {report['coefficients']}")
    print(
        f"  risk-coverage  : f AUC={report['risk_coverage']['auc_risk_f']} "
        f"vs model_conf AUC={report['risk_coverage']['auc_risk_model_conf']}"
    )
    print(f"  audit chain    : {report['audit']['valid']} ({report['audit']['events']} events)")
    print("\n  simulated operators:")
    for row in report["hil"]:
        print(
            f"    {row['strategy']:16s} {row['operator']:15s} coverage={row['coverage_auto']:.2f} "
            f"error={row['final_error']:.2f} bias={row['automation_bias_rate']:.2f} "
            f"reject={row['unjustified_rejection_rate']:.2f} rely={row['appropriate_reliance']:.2f}"
        )
    out = io.resolve(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {out}")
    _ = DecisionOrigin, json
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
