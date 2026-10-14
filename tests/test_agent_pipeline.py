"""Tests for the Qdrant -> deliberation pipeline (agent_pipeline.py)."""

from __future__ import annotations

from energymind.agent_pipeline import run_case, seen_before
from energymind.agents import RuleBasedValidator
from energymind.config import load_settings
from energymind.retrieval import Case, InMemoryRetriever, Match
from energymind.store import AuditLog, MemoryStore

TAU = 0.9


def _record(signature, *, severity="medium", type_pred="gain_error", origin_pred="data_quality"):
    return {
        "anomaly_id": "current-0001",
        "household_id": "HH_00072_AUS",
        "t_start": "2020-11-14T18:00:00Z",
        "t_end": "2020-11-15T06:00:00Z",
        "duration_h": 12.0,
        "severity": severity,
        "type_pred": type_pred,
        "origin_pred": origin_pred,
        "signature": list(signature),
        "detection": {"model": "rules", "score": 12.0, "p_anomaly": 0.9},
        "classification": {
            "model": "lightgbm",
            "type_probs": {type_pred: 0.7},
            "origin_probs": {origin_pred: 0.7},
        },
        "explanation": {
            "shap_top": [["p_resid_z_max", 0.4]],
            "lime_top": ["p_resid_z_max"],
            "reliability": 0.5,
        },
        "evidence": [
            {
                "check": "baseline_deviation",
                "observed_kw": 5.1,
                "expected_kw": 3.9,
                "z": 3.4,
                "status": "anomalous",
            },
            {"check": "current_vs_power_identity", "rel_residual": 0.0, "status": "holds"},
            {"check": "neighborhood_share", "value": 0.08, "status": "isolated"},
        ],
    }


def _vec(i: int) -> list[float]:
    v = [0.0] * 64
    v[i % 64] = 1.0
    return v


def _case(i: int, **kw) -> Case:
    defaults = {
        "anomaly_id": f"case-{i:04d}",
        "signature": _vec(i),
        "type_pred": "gain_error",
        "origin_pred": "data_quality",
        "status": "verified",
        "severity": "medium",
        "decision_action": "flag_data_suspect",
        # Fix 3: verified history must be time-causal, i.e. strictly earlier than the
        # queried case (whose window starts 2020-11-14).
        "t_start": "2020-08-01T00:00:00+00:00",
        "t_end": "2020-08-01T02:00:00+00:00",
    }
    defaults.update(kw)
    return Case(**defaults)


def _store(cases: list[Case]) -> MemoryStore:
    retriever = InMemoryRetriever(TAU)
    retriever.upsert(cases)
    return MemoryStore(retriever, AuditLog(sqlite_path=":memory:"))


# --------------------------------------------------------------------------- #
# seen_before gate
# --------------------------------------------------------------------------- #


def test_seen_before_requires_similarity_severity_and_verified():
    good = _case(0, severity="high")
    near = Match(good, 0.95)
    assert seen_before([near], "data_quality", "medium", TAU) is near

    assert seen_before([Match(good, 0.5)], "data_quality", "medium", TAU) is None  # below tau
    assert (
        seen_before([Match(_case(1, status="proposed"), 0.99)], "data_quality", "medium", TAU)
        is None
    )
    assert seen_before([Match(_case(2, severity="low"), 0.99)], "data_quality", "high", TAU) is None
    assert (
        seen_before(
            [Match(_case(3, valid_until="2000-01-01T00:00:00Z"), 0.99)],
            "data_quality",
            "low",
            TAU,
        )
        is None
    )


def test_seen_before_rejects_origin_majority_mismatch():
    """Fix 2 acceptance: 5 verified 'adversarial' precedents must not be reused for
    a query whose predicted origin is 'data_quality'."""
    matches = [Match(_case(i, origin_pred="adversarial", severity="high"), 0.99) for i in range(5)]
    assert seen_before(matches, "data_quality", "medium", TAU) is None
    # the same precedents are reused for a matching predicted origin
    assert seen_before(matches, "adversarial", "medium", TAU) is not None


def test_seen_before_requires_majority_share_at_least_point_seven():
    # 4 data_quality + 1 adversarial at equal similarity -> share 0.8 (> 0.7) -> reuse
    balanced = [
        Match(_case(i, origin_pred="data_quality", severity="high"), 0.99) for i in range(4)
    ]
    balanced.append(Match(_case(9, origin_pred="adversarial", severity="high"), 0.99))
    assert seen_before(balanced, "data_quality", "medium", TAU) is not None
    # 3 data_quality + 2 adversarial -> share 0.6 (< 0.7) -> no reuse
    mixed = [Match(_case(i, origin_pred="data_quality", severity="high"), 0.99) for i in range(3)]
    mixed += [Match(_case(9, origin_pred="adversarial", severity="high"), 0.99) for _ in range(2)]
    assert seen_before(mixed, "data_quality", "medium", TAU) is None


# --------------------------------------------------------------------------- #
# run_case routing
# --------------------------------------------------------------------------- #


def test_run_case_reuses_verified_match_and_skips_agents():
    signature = _vec(0)
    store = _store([_case(0, severity="high", decision_action="flag_data_suspect")])

    class ExplodingAgent:
        role = "analyst"

        def propose(self, *a, **k):
            raise AssertionError("agents must be skipped on the seen-before path")

    outcome = run_case(
        _record(signature, severity="medium"),
        store,
        settings=load_settings(),
        tau_sim=TAU,
        analyst=ExplodingAgent(),
        validator=ExplodingAgent(),
    )
    assert outcome.path == "seen_before"
    assert outcome.reused_from is not None
    assert outcome.action is not None and outcome.action.value == "flag_data_suspect"
    event_types = [e["event_type"] for e in store.audit.events()]
    assert "history_retrieved" in event_types and "seen_before_reuse" in event_types
    assert store.audit.verify_chain()["valid"]


def test_run_case_deliberates_when_no_match():
    store = _store([])
    outcome = run_case(_record(_vec(5)), store, settings=load_settings(), tau_sim=TAU, r_max=2)
    assert outcome.path == "deliberated"
    assert outcome.deliberation is not None
    assert outcome.deliberation.rounds_used >= 1
    event_types = [e["event_type"] for e in store.audit.events()]
    assert "history_retrieved" in event_types
    assert "deliberation_completed" in event_types
    assert store.audit.verify_chain()["valid"]


def test_run_case_passes_history_to_the_validator():
    # sim >= tau (eligible as history) but severity blocks reuse -> agents run and
    # the Validator receives history.0.* evidence.
    signature = _vec(0)
    store = _store([_case(0, severity="low")])

    seen: dict[str, list[str]] = {}

    class SpyValidator(RuleBasedValidator):
        def propose(self, view, round, other_proposal=None, own_previous_proposal=None):
            seen["ids"] = [item.id for item in view.evidence]
            return super().propose(view, round, other_proposal, own_previous_proposal)

    outcome = run_case(
        _record(signature, severity="high"),
        store,
        settings=load_settings(),
        tau_sim=TAU,
        r_max=1,
        validator=SpyValidator(),
    )
    assert outcome.path == "deliberated"
    assert outcome.history, "the verified precedent should be in the history list"
    assert any(eid.startswith("history.") for eid in seen["ids"]), seen["ids"]


def test_retrieve_history_defaults_do_not_filter_type_or_origin():
    from energymind.agent_pipeline import retrieve_history

    store = _store(
        [
            _case(0, type_pred="gain_error", origin_pred="data_quality"),
            _case(1, type_pred="dropout", origin_pred="data_quality"),
            _case(2, type_pred="gain_error", origin_pred="adversarial"),
        ]
    )
    record = _record(_vec(0), type_pred="gain_error", origin_pred="data_quality")
    # Fix 2: retrieval filters only on status/expiry -> all three verified cases return
    ids = {m.case.anomaly_id for m in retrieve_history(store, record, k=5)}
    assert ids == {"case-0000", "case-0001", "case-0002"}
    # the optional type/origin filters still work when explicitly requested
    filtered = [
        m.case.anomaly_id
        for m in retrieve_history(store, record, k=5, match_type=True, match_origin=True)
    ]
    assert filtered == ["case-0000"]


# --------------------------------------------------------------------------- #
# Decision summary
# --------------------------------------------------------------------------- #


def test_decision_summary_seen_before_low_risk_is_autonomous():
    # A verified, converged low-risk data_quality reuse -> provisional no-human.
    store = _store([_case(0, severity="high", decision_action="flag_data_suspect")])
    outcome = run_case(
        _record(_vec(0), severity="medium"), store, settings=load_settings(), tau_sim=TAU
    )
    d = outcome.decision
    assert d["path"] == "seen_before"
    assert d["converged"] is True
    assert d["action"] == "flag_data_suspect"
    assert d["origin"] == "data_quality"
    assert d["requires_human"] is False
    assert d["reused_from"]["anomaly_id"] == "case-0000"


def test_decision_summary_high_risk_or_adversarial_requires_human():
    from energymind.agent_pipeline import build_decision
    from energymind.agents import Action, DecisionOrigin

    settings = load_settings()
    high = build_decision(
        settings,
        path="deliberated",
        converged=True,
        action=Action.INVESTIGATE_TAMPERING,
        origin=DecisionOrigin.ADVERSARIAL,
    )
    assert high["requires_human"] is True

    nonconv = build_decision(
        settings,
        path="deliberated",
        converged=False,
        action=None,
        origin=None,
    )
    assert nonconv["requires_human"] is True
    assert nonconv["analyst"] == {} and nonconv["validator"] == {}


def test_run_case_records_decision_summary_in_audit():
    store = _store([])
    run_case(_record(_vec(9)), store, settings=load_settings(), tau_sim=TAU, r_max=1)
    events = store.audit.events()
    assert any(e["event_type"] == "decision_summary" for e in events)
    summary = next(e for e in events if e["event_type"] == "decision_summary")
    assert set(summary["payload"]) >= {"path", "converged", "action", "origin", "requires_human"}
    assert store.audit.verify_chain()["valid"]
