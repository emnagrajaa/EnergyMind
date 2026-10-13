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
    assert seen_before([near], "medium", TAU) is near

    assert seen_before([Match(good, 0.5)], "medium", TAU) is None  # below tau
    assert seen_before([Match(_case(1, status="proposed"), 0.99)], "medium", TAU) is None
    assert seen_before([Match(_case(2, severity="low"), 0.99)], "high", TAU) is None
    assert (
        seen_before([Match(_case(3, valid_until="2000-01-01T00:00:00Z"), 0.99)], "low", TAU) is None
    )


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


def test_retrieve_history_filters_by_type_and_origin():
    from energymind.agent_pipeline import retrieve_history

    store = _store(
        [
            _case(0, type_pred="gain_error", origin_pred="data_quality"),
            _case(1, type_pred="dropout", origin_pred="data_quality"),
            _case(2, type_pred="gain_error", origin_pred="adversarial"),
        ]
    )
    record = _record(_vec(0), type_pred="gain_error", origin_pred="data_quality")
    ids = [m.case.anomaly_id for m in retrieve_history(store, record, k=5)]
    assert ids == ["case-0000"]
