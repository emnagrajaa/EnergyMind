"""Qdrant -> deliberation pipeline (blueprint Sections 10-11).

This wires the three pieces that were built separately:

    anomaly record (M9)
        -> retrieve verified precedents from the vector memory (M6, Qdrant)
        -> "seen before?" gate
              yes -> reuse the historical decision (agents skipped)
              no  -> two-agent deliberation (M7) with the precedents as evidence
        -> append-only hash-chained audit log (M6)

Every step is appended to the same ``AuditLog``, so the retrieval, the gate
decision and the deliberation form one auditable chain.

    python -m energymind.agent_pipeline --demo            # offline (in-memory memory)
    python -m energymind.agent_pipeline --qdrant --reset  # live Qdrant cloud/local
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime

from energymind import io
from energymind.agents import (
    Action,
    Agent,
    DecisionOrigin,
    DeliberationResult,
    RuleBasedAnalyst,
    RuleBasedValidator,
    deliberate,
)
from energymind.config import Settings, load_settings
from energymind.retrieval import InMemoryRetriever, LocalQdrantRetriever, Match
from energymind.store import AuditLog, MemoryStore

#: Severity order used by the "seen before" gate (new severity must not exceed the match).
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class CaseOutcome:
    """Result of routing one case through memory + deliberation."""

    anomaly_id: str
    path: str  # "seen_before" | "deliberated"
    action: Action | None
    origin: DecisionOrigin | None
    reused_from: Match | None = None
    deliberation: DeliberationResult | None = None
    history: list[Match] = field(default_factory=list)

    @property
    def history_ids(self) -> list[str]:
        return [m.case.anomaly_id for m in self.history]


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def retrieve_history(
    store: MemoryStore,
    record: dict,
    *,
    k: int = 5,
    match_type: bool = True,
    match_origin: bool = True,
    exclude_id: str | None = None,
) -> list[Match]:
    """Filtered k-NN over the verified memory using the record's 64-dim signature.

    ``exclude_id`` (the current anomaly_id) is always removed, so a case never
    matches itself.
    """
    signature = record.get("signature")
    if not signature:
        return []
    matches = store.nearest_verified(
        signature,
        type_pred=record.get("type_pred") if match_type else None,
        origin_pred=record.get("origin_pred") if match_origin else None,
        k=k + 1 if exclude_id else k,
    )
    if exclude_id:
        matches = [m for m in matches if m.case.anomaly_id != exclude_id][:k]
    return matches


def _expired(valid_until: str | None) -> bool:
    if not valid_until:
        return False
    try:
        moment = datetime.fromisoformat(str(valid_until).replace("Z", "+00:00"))
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment < datetime.now(UTC)


def seen_before(matches: list[Match], severity: str | None, tau_sim: float) -> Match | None:
    """Return the best reusable precedent, or ``None`` to run the agent loop.

    A match qualifies only if: verified, not expired, cosine similarity >=
    ``tau_sim``, and the new case is no more severe than the matched one (blueprint
    Section 10). Type/origin equality is enforced by the retrieval filters.
    """
    new_rank = _SEVERITY_RANK.get(str(severity).lower(), -1)
    for match in matches:  # sorted by similarity, descending
        if match.similarity < tau_sim:
            continue
        if match.case.status != "verified":
            continue
        if _expired(match.case.valid_until):
            continue
        if _SEVERITY_RANK.get(str(match.case.severity).lower(), -1) < new_rank:
            continue
        return match
    return None


def _as_origin(value: str | None) -> DecisionOrigin | None:
    if value in DecisionOrigin._value2member_map_:
        return DecisionOrigin(value)
    return None


def _as_action(value: str | None) -> Action | None:
    if value in Action._value2member_map_:
        return Action(value)
    return None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_case(
    record: dict,
    store: MemoryStore,
    *,
    settings: Settings | None = None,
    r_max: int | None = None,
    k: int | None = None,
    match_type: bool = True,
    match_origin: bool = True,
    tau_sim: float | None = None,
    analyst: Agent | None = None,
    validator: Agent | None = None,
) -> CaseOutcome:
    """Route one anomaly record through memory -> (gate | agents) -> audit log."""
    settings = settings or load_settings()
    anomaly_id = str(record.get("anomaly_id", "")).strip()
    if not anomaly_id:
        raise ValueError("record must contain a non-empty 'anomaly_id'")

    k = k or settings.store.retrieval.k
    tau = tau_sim if tau_sim is not None else settings.store.retrieval.tau_sim

    history = retrieve_history(
        store,
        record,
        k=k,
        match_type=match_type,
        match_origin=match_origin,
        exclude_id=anomaly_id,
    )
    store.audit.append_event(
        actor="retriever",
        event_type="history_retrieved",
        payload={
            "k": k,
            "n": len(history),
            "ids": [m.case.anomaly_id for m in history],
            "similarities": [round(m.similarity, 4) for m in history],
            "tau_sim": tau,
        },
        anomaly_id=anomaly_id,
    )

    match = seen_before(history, record.get("severity"), tau)
    if match is not None:
        store.audit.append_event(
            actor="pipeline",
            event_type="seen_before_reuse",
            payload={
                "reused_from": match.case.anomaly_id,
                "similarity": round(match.similarity, 4),
                "action": match.case.decision_action,
                "origin": match.case.origin_pred,
                "agents_skipped": True,
            },
            anomaly_id=anomaly_id,
        )
        return CaseOutcome(
            anomaly_id=anomaly_id,
            path="seen_before",
            action=_as_action(match.case.decision_action),
            origin=_as_origin(match.case.origin_pred),
            reused_from=match,
            deliberation=None,
            history=history,
        )

    result = deliberate(
        record,
        analyst=analyst or RuleBasedAnalyst(),
        validator=validator or RuleBasedValidator(),
        history=history,
        r_max=r_max,
        audit_log=store.audit,
        settings=settings,
    )
    return CaseOutcome(
        anomaly_id=anomaly_id,
        path="deliberated",
        action=result.agreed_action,
        origin=result.agreed_origin,
        reused_from=None,
        deliberation=result,
        history=history,
    )


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #


def _build_record(frame, events, anomaly_id: str, detector_scores) -> dict:
    """Assemble a minimal §9 record for one event (real evidence, real detector score)."""
    import pandas as pd

    from energymind import explain, records

    group = events[events.anomaly_id == anomaly_id]
    info = group.iloc[0]
    window = pd.concat(
        [
            frame[
                (frame.household_id == r.household_id)
                & frame.ts_utc.between(r.t_start, r.t_end, inclusive="left")
            ]
            for r in group.itertuples()
        ]
    )
    signature = records.build_signature(window, window["ts_utc"].iloc[0])
    mask = (frame.household_id == info["household_id"]) & frame.ts_utc.between(
        window["ts_utc"].iloc[0], window["ts_utc"].iloc[-1], inclusive="left"
    )
    scores = detector_scores[mask.to_numpy()]
    duration_h = float((window["ts_utc"].max() - window["ts_utc"].min()).total_seconds() / 3600)

    return {
        "anomaly_id": str(anomaly_id),
        "household_id": str(info["household_id"]),
        "t_start": str(window["ts_utc"].iloc[0]),
        "t_end": str(window["ts_utc"].iloc[-1]),
        "duration_h": duration_h,
        "severity": str(info["severity"]),
        # In simulation the injected label supplies the "model" prediction.
        "type_pred": str(info["type"]),
        "origin_pred": str(info["origin"]),
        "signature": [float(v) for v in signature],
        "detection": {
            "model": "rules",
            "score": round(float(scores.max()), 4),
            "p_anomaly": round(float(scores.mean()), 4),
        },
        "classification": {
            "model": "labels(sim)",
            "type_probs": {str(info["type"]): 0.7},
            "origin_probs": {str(info["origin"]): 0.7},
        },
        "explanation": {"shap_top": [], "lime_top": [], "reliability": None},
        "evidence": explain.evidence_checks(window),
        "facts": explain.window_facts(
            window, frame, info["household_id"], window["ts_utc"].iloc[0], window["ts_utc"].iloc[-1]
        ),
    }


def _main_demo(args) -> int:
    import pandas as pd

    from energymind import evaluate
    from energymind.detect.rules import Rules
    from energymind.store import ingest_events

    settings: Settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()

    retriever = InMemoryRetriever(settings.store.retrieval.tau_sim)
    backend = "in-memory"
    if args.qdrant:
        retriever = LocalQdrantRetriever(settings.store.qdrant)
        if args.reset:
            retriever.delete_collection()
        retriever.ensure_collection()
        backend = f"qdrant:{settings.store.qdrant.collection}"
    store = MemoryStore(retriever, AuditLog(settings.store.audit.backend, args.sqlite))

    frame = pd.read_parquet(io.resolve(args.feature_file))
    labels = pd.read_parquet(io.resolve(args.label_file))
    events = evaluate.expand_events(labels)
    if retriever.count() == 0:
        ingest_events(settings, store, feature_file=args.feature_file, label_file=args.label_file)

    detector = Rules().fit(frame)
    detector_scores = detector.score(frame)

    # Pick test-split events as the "current" cases.
    t = settings.data.splits.temporal
    off = settings.data.time.assume_utc_offset_h
    test_start = pd.Timestamp(t.test.start).tz_localize("UTC") - pd.Timedelta(hours=off)
    test_ids = events.loc[events["t_start"] >= test_start, "anomaly_id"].unique()

    print(f"backend        : {backend}")
    print(f"cases in memory: {retriever.count()}")
    print(f"\nrouting {min(args.cases, len(test_ids))} test cases through memory -> deliberation:")
    for anomaly_id in test_ids[: args.cases]:
        record = _build_record(frame, events, anomaly_id, detector_scores)
        outcome = run_case(record, store, settings=settings, r_max=args.r_max)
        if outcome.path == "seen_before":
            detail = (
                f"reused {outcome.reused_from.case.anomaly_id[:12]} "
                f"(sim={outcome.reused_from.similarity:.3f}) "
                f"action={outcome.action} origin={outcome.origin}"
            )
        else:
            result = outcome.deliberation
            detail = (
                f"converged={result.converged} rounds={result.rounds_used} "
                f"action={outcome.action} origin={outcome.origin} history={outcome.history_ids}"
            )
        print(f"  {str(anomaly_id)[:12]}  {outcome.path:11s}  {detail}")

    print(f"\naudit chain : {store.audit.verify_chain()}")
    print("event types :", sorted({e["event_type"] for e in store.audit.events()}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qdrant -> deliberation pipeline.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--qdrant", action="store_true", help="use the configured Qdrant cluster")
    parser.add_argument("--reset", action="store_true", help="recreate the Qdrant collection")
    parser.add_argument("--cases", type=int, default=5, help="number of cases to route")
    parser.add_argument("--r-max", type=int, default=3)
    parser.add_argument("--feature-file", default="data/features/features.parquet")
    parser.add_argument("--label-file", default="data/injected/labels.parquet")
    parser.add_argument("--sqlite", default="data/memory/audit.db")
    parser.add_argument(
        "--demo", action="store_true", help="run the seeded pipeline demo (default)"
    )
    args = parser.parse_args(argv)

    io.load_env()
    return _main_demo(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
