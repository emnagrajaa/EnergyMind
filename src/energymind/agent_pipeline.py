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
import json
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
    decision: dict = field(default_factory=dict)
    #: Fix 4 - the proposed decision, always defined even when the agents did not
    #: converge (in which case it is the Analyst's last proposal). ``proposal_source``
    #: is one of ``seen_before`` / ``converged`` / ``analyst_fallback``.
    proposal_origin: str | None = None
    proposal_action: str | None = None
    proposal_source: str = ""

    @property
    def history_ids(self) -> list[str]:
        return [m.case.anomaly_id for m in self.history]


def _proposal_summary(proposal) -> dict:
    if proposal is None:
        return {}
    return {
        "action": proposal.action.value,
        "origin": proposal.origin.value,
        "support": proposal.support,
        "evidence_ids": list(proposal.evidence_ids),
        "rationale": proposal.rationale,
    }


def _requires_human(settings: Settings, converged: bool, action: Action | None, origin) -> bool:
    """Provisional human-review flag (NOT the M8 tier; M8 owns the final routing).

    A case may be handled without a human only when the agents converged on a
    low-risk, reversible action for a ``data_quality`` origin. Everything else
    (non-convergence, adversarial/unknown origin, medium/high risk, no action)
    requires a human.
    """
    if not converged or action is None:
        return True
    spec = settings.policy.actions.get(action.value)
    if spec is None:
        return True
    is_data_quality = origin == DecisionOrigin.DATA_QUALITY
    return not (is_data_quality and spec.risk == "low" and spec.reversible)


def build_decision(
    settings: Settings,
    *,
    path: str,
    converged: bool,
    action: Action | None,
    origin: DecisionOrigin | None,
    reused_from: Match | None = None,
    analyst=None,
    validator=None,
) -> dict:
    """Assemble the human-readable decision summary.

    The *anomaly decision* (what to do + origin) comes from M7 (memory reuse or the
    agents). ``requires_human`` is provisional routing; calibrated confidence ``f``
    and the final tiers are produced by M8.
    """
    return {
        "path": path,
        "converged": converged,
        "action": action.value if action else None,
        "origin": origin.value if origin else None,
        "requires_human": _requires_human(settings, converged, action, origin),
        "reused_from": (
            {
                "anomaly_id": reused_from.case.anomaly_id,
                "similarity": round(reused_from.similarity, 4),
            }
            if reused_from is not None
            else None
        ),
        "analyst": _proposal_summary(analyst),
        "validator": _proposal_summary(validator),
    }


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def retrieve_history(
    store: MemoryStore,
    record: dict,
    *,
    k: int = 5,
    match_type: bool = False,
    match_origin: bool = False,
    exclude_id: str | None = None,
    before: str | None = None,
) -> list[Match]:
    """k-NN over the verified memory using the record's 64-dim signature.

    By default retrieval filters **only** on ``status == 'verified'``, not expired and
    strictly before ``before`` (the query's ``t_start``) -- it does **not** filter on
    the predicted type/origin (that would turn a wrong prediction into wrong
    precedents and make the history term circular). The type/origin filters stay
    available but are off by default. ``exclude_id`` (the current anomaly_id) is always
    removed, so a case never matches itself.
    """
    signature = record.get("signature")
    if not signature:
        return []
    before = before if before is not None else record.get("t_start")
    matches = store.nearest_verified(
        signature,
        type_pred=record.get("type_pred") if match_type else None,
        origin_pred=record.get("origin_pred") if match_origin else None,
        k=k,
        before=before,
    )
    matches = [m for m in matches if not _expired(m.case.valid_until)]
    if exclude_id:
        matches = [m for m in matches if m.case.anomaly_id != exclude_id]
    return matches[:k]


def leakage_violations(history: list[Match], t_start: str | None) -> int:
    """Count retrieved precedents whose ``t_end`` is not strictly before ``t_start``."""
    from energymind.retrieval import _parse_ts

    cutoff = _parse_ts(t_start)
    violations = 0
    for match in history:
        end = _parse_ts(match.case.t_end)
        if cutoff is None or end is None or end >= cutoff:
            violations += 1
    return violations


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


def _weighted_majority_origin(matches: list[Match]) -> tuple[str | None, float]:
    """Similarity-weighted majority origin of the precedents and its share."""
    totals: dict[str, float] = {}
    grand = 0.0
    for match in matches:
        weight = float(match.similarity)
        origin = match.case.origin_pred
        totals[origin] = totals.get(origin, 0.0) + weight
        grand += weight
    if grand <= 0.0:
        return None, 0.0
    origin = max(totals, key=lambda key: totals[key])
    return origin, totals[origin] / grand


def seen_before(
    matches: list[Match],
    proposed_origin: str | None,
    severity: str | None,
    tau_sim: float,
    majority_threshold: float = 0.7,
) -> Match | None:
    """Return the best reusable precedent, or ``None`` to run the agent loop.

    Retrieval no longer filters on the predicted type/origin, so this gate enforces
    them: the **similarity-weighted majority origin of the top-k precedents must
    equal** ``proposed_origin`` with a weighted share ``>= majority_threshold``
    (otherwise the agents run). It then keeps the original checks -- the returned
    precedent is verified, not expired, has cosine similarity ``>= tau_sim``, shares
    the proposed origin, and the new case is no more severe than the matched one.
    """
    if not matches:
        return None
    majority_origin, share = _weighted_majority_origin(matches)
    if majority_origin is None or majority_origin != proposed_origin or share < majority_threshold:
        return None

    new_rank = _SEVERITY_RANK.get(str(severity).lower(), -1)
    for match in matches:  # sorted by similarity, descending
        if match.similarity < tau_sim:
            continue
        if match.case.status != "verified":
            continue
        if _expired(match.case.valid_until):
            continue
        if match.case.origin_pred != proposed_origin:
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


def _proposal_of(origin, action) -> tuple[str, str | None]:
    """Normalise a proposal to ``(origin_value, action_value)``.

    A missing origin becomes ``"unknown"`` (Fix 4.4); a missing action stays ``None``.
    """
    origin_value = getattr(origin, "value", origin) or "unknown"
    action_value = getattr(action, "value", action)
    return str(origin_value), (str(action_value) if action_value else None)


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
    match_type: bool = False,
    match_origin: bool = False,
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
            "leakage_violations": leakage_violations(history, record.get("t_start")),
        },
        anomaly_id=anomaly_id,
    )

    match = seen_before(history, record.get("origin_pred"), record.get("severity"), tau)
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
        action = _as_action(match.case.decision_action)
        origin = _as_origin(match.case.origin_pred)
        decision = build_decision(
            settings,
            path="seen_before",
            converged=True,
            action=action,
            origin=origin,
            reused_from=match,
        )
        proposal_origin, proposal_action = _proposal_of(origin, action)
        return CaseOutcome(
            anomaly_id=anomaly_id,
            path="seen_before",
            action=action,
            origin=origin,
            reused_from=match,
            deliberation=None,
            history=history,
            decision=decision,
            proposal_origin=proposal_origin,
            proposal_action=proposal_action,
            proposal_source="seen_before",
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
    decision = build_decision(
        settings,
        path="deliberated",
        converged=result.converged,
        action=result.agreed_action,
        origin=result.agreed_origin,
        analyst=result.analyst_final,
        validator=result.validator_final,
    )
    store.audit.append_event(
        actor="pipeline",
        event_type="decision_summary",
        payload=decision,
        anomaly_id=anomaly_id,
    )
    # Fix 4: converged -> the agreed decision; otherwise the Analyst's last proposal.
    if result.converged:
        proposal_origin, proposal_action = _proposal_of(result.agreed_origin, result.agreed_action)
        proposal_source = "converged"
    else:
        proposal_origin, proposal_action = _proposal_of(
            result.analyst_final.origin, result.analyst_final.action
        )
        proposal_source = "analyst_fallback"
    return CaseOutcome(
        anomaly_id=anomaly_id,
        path="deliberated",
        action=result.agreed_action,
        origin=result.agreed_origin,
        reused_from=None,
        deliberation=result,
        history=history,
        decision=decision,
        proposal_origin=proposal_origin,
        proposal_action=proposal_action,
        proposal_source=proposal_source,
    )


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #


def _observed_severity(window) -> str:
    """Label-free severity from the window's robust z-score (2σ / 5σ buckets)."""
    if "p_resid_z" not in window.columns or window["p_resid_z"].dropna().empty:
        return "low"
    z = float(window["p_resid_z"].abs().max())
    if z > 5:
        return "high"
    if z > 2:
        return "medium"
    return "low"


def _argmax(probs: dict | None) -> str | None:
    if not probs:
        return None
    return str(max(probs, key=lambda key: probs[key]))


def _build_record(
    frame,
    events,
    anomaly_id: str,
    detector_scores,
    classification: dict | None = None,
) -> dict:
    """Assemble a §9 record from the window, the detector and the classifier.

    Predictions come **only** from the classifier output (``classification``);
    injected labels are never read here. With no classifier output,
    ``type_pred``/``origin_pred`` are ``None`` and ``classification`` is empty.
    """
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
    completeness = float(1.0 - window["active_power_kw"].isna().mean())

    type_probs = (classification or {}).get("type_probs", {})
    origin_probs = (classification or {}).get("origin_probs", {})
    classification_block = (
        {"model": "lightgbm", "type_probs": type_probs, "origin_probs": origin_probs}
        if classification
        else {}
    )

    return {
        "anomaly_id": str(anomaly_id),
        "household_id": str(info["household_id"]),
        "t_start": str(window["ts_utc"].iloc[0]),
        "t_end": str(window["ts_utc"].iloc[-1]),
        "duration_h": duration_h,
        "severity": _observed_severity(window),
        "completeness": completeness,
        "type_pred": _argmax(type_probs),
        "origin_pred": _argmax(origin_probs),
        "signature": [float(v) for v in signature],
        "detection": {
            "model": "rules",
            "score": round(float(scores.max()), 4),
            "p_anomaly": round(float(scores.mean()), 4),
        },
        "classification": classification_block,
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
        d = outcome.decision
        where = "human" if d["requires_human"] else "auto-tier1(prov.)"
        if outcome.path == "seen_before":
            rf = d["reused_from"]
            detail = f"reused {rf['anomaly_id'][:12]} (sim={rf['similarity']:.3f})"
        else:
            result = outcome.deliberation
            detail = (
                f"rounds={result.rounds_used} "
                f"analyst={d['analyst'].get('action')} validator={d['validator'].get('action')}"
            )
        print(
            f"  {str(anomaly_id)[:12]}  {outcome.path:11s}  action={str(d['action']):26s} "
            f"origin={str(d['origin']):12s} converged={str(d['converged']):5s} "
            f"-> {where:16s} {detail}"
        )

    print(f"\naudit chain : {store.audit.verify_chain()}")
    print("event types :", sorted({e["event_type"] for e in store.audit.events()}))
    print("\nexample decision summary (last case):")
    print(json.dumps(outcome.decision, indent=2))
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
