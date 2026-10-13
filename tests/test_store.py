"""Tests for Step 10 store: append-only hash-chained audit log and MemoryStore."""

from __future__ import annotations

import sqlite3

import pytest

from energymind.retrieval import Case, InMemoryRetriever
from energymind.store import GENESIS_HASH, AuditLog, MemoryStore, canonical_json


def _case(i: int = 0, **kw) -> Case:
    vec = [0.0] * 64
    vec[i % 64] = 1.0
    defaults = {
        "anomaly_id": f"00000000-0000-0000-0000-{i:012d}",
        "signature": vec,
        "type_pred": "gain_error",
        "origin_pred": "data_quality",
        "status": "proposed",
    }
    defaults.update(kw)
    return Case(**defaults)


def test_canonical_json_is_deterministic():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_json({"a": 2, "b": 1}) == '{"a":2,"b":1}'


def test_audit_chain_links_and_verifies(tmp_path):
    log = AuditLog(sqlite_path=tmp_path / "audit.db")
    first = log.append_event("detector", "proposed", {"type": "gain_error"}, anomaly_id="a1")
    second = log.append_event(
        "agent_1", "decision", {"action": "flag_data_suspect"}, anomaly_id="a1"
    )
    assert first["prev_hash"] == GENESIS_HASH
    assert second["prev_hash"] == first["row_hash"]
    report = log.verify_chain()
    assert report["valid"] is True
    assert report["events"] == 2
    assert report["head"] == second["row_hash"]


def test_audit_is_append_only(tmp_path):
    log = AuditLog(sqlite_path=tmp_path / "audit.db")
    log.append_event("detector", "proposed", {"x": 1}, anomaly_id="a1")
    with pytest.raises(sqlite3.Error):
        log.conn.execute("UPDATE audit_event SET payload='{}' WHERE seq=1")
    with pytest.raises(sqlite3.Error):
        log.conn.execute("DELETE FROM audit_event WHERE seq=1")


def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(sqlite_path=tmp_path / "audit.db")
    log.append_event("detector", "proposed", {"x": 1}, anomaly_id="a1")
    log.append_event("agent_1", "decision", {"x": 2}, anomaly_id="a1")
    # bypass the append-only trigger and edit a row
    log.conn.execute("DROP TRIGGER audit_event_no_update")
    log.conn.execute("UPDATE audit_event SET payload='{\"x\":99}' WHERE seq=1")
    log.conn.commit()
    report = log.verify_chain()
    assert report["valid"] is False
    assert report["broken_at"] == 1


def test_memory_store_insert_verify_and_retrieve(tmp_path):
    audit = AuditLog(sqlite_path=tmp_path / "audit.db")
    store = MemoryStore(InMemoryRetriever(), audit)
    case = _case(0)
    store.insert_case(case)
    assert len(audit.events()) == 1  # proposed
    assert (
        store.nearest_verified(
            case.signature, type_pred="gain_error", origin_pred="data_quality", k=3
        )
        == []
    )

    store.decide(case.anomaly_id, action="flag_data_suspect")
    store.verify(case.anomaly_id)
    assert len(audit.events()) == 3  # proposed, decision, outcome
    assert audit.verify_chain()["valid"]

    hits = store.nearest_verified(
        case.signature, type_pred="gain_error", origin_pred="data_quality", k=3
    )
    assert len(hits) == 1
    assert hits[0].similarity == pytest.approx(1.0)


def test_record_decision_writes_action_and_confidence_back(tmp_path):
    audit = AuditLog(sqlite_path=tmp_path / "audit.db")
    store = MemoryStore(InMemoryRetriever(), audit)
    case = _case(0, status="proposed")
    store.insert_case(case)

    payload = store.record_decision(
        case.anomaly_id,
        action="flag_data_suspect",
        origin="data_quality",
        confidence=0.87,
        tier=1,
        status="verified",
    )
    assert payload["decision_action"] == "flag_data_suspect"
    assert payload["confidence"] == 0.87 and payload["tier"] == 1

    # the decision is now retrievable as a verified precedent that carries an action
    hits = store.nearest_verified(
        case.signature, type_pred="gain_error", origin_pred="data_quality", k=3
    )
    assert len(hits) == 1
    reused = hits[0].case
    assert reused.status == "verified"
    assert reused.decision_action == "flag_data_suspect"
    assert reused.confidence == 0.87 and reused.tier == 1
    assert audit.verify_chain()["valid"]
