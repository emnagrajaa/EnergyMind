"""Fix 3: memory must be time-causal and must not contain test labels."""

from __future__ import annotations

import pytest

from energymind.agent_pipeline import leakage_violations, retrieve_history
from energymind.config import REPO_ROOT, load_settings
from energymind.retrieval import Case, InMemoryRetriever, LocalQdrantRetriever
from energymind.store import AuditLog, MemoryStore

RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"


def _vec(i: int = 0) -> list[float]:
    v = [0.0] * 64
    v[i % 64] = 1.0
    return v


def _case(i: int, t_start: str, t_end: str, **kw) -> Case:
    defaults = {
        "anomaly_id": f"00000000-0000-0000-0000-{i:012d}",
        "signature": _vec(0),
        "type_pred": "gain_error",
        "origin_pred": "data_quality",
        "status": "verified",
        "t_start": t_start,
        "t_end": t_end,
    }
    defaults.update(kw)
    return Case(**defaults)


CASE0 = "00000000-0000-0000-0000-000000000000"
CASE1 = "00000000-0000-0000-0000-000000000001"


TRAIN = ("2020-08-01T00:00:00+00:00", "2020-08-01T02:00:00+00:00")
TEST = ("2020-12-01T00:00:00+00:00", "2020-12-01T02:00:00+00:00")
QUERY_START = "2020-11-01T00:00:00+00:00"


def test_inmemory_search_is_time_causal():
    retriever = InMemoryRetriever(tau_sim=0.0)
    retriever.upsert([_case(0, *TRAIN), _case(1, *TEST)])
    hits = retriever.search(_vec(0), status="verified", k=5, before=QUERY_START)
    assert [m.case.anomaly_id for m in hits] == [CASE0]  # only the train case


def test_qdrant_search_is_time_causal(tmp_path):
    pytest.importorskip("qdrant_client")
    settings = load_settings()
    config = settings.store.qdrant.model_copy(
        update={"local_path": str(tmp_path), "collection": "t_causal"}
    )
    retriever = LocalQdrantRetriever(config)
    retriever.upsert([_case(0, *TRAIN), _case(1, *TEST)])
    hits = retriever.search(_vec(0), status="verified", k=5, before=QUERY_START)
    assert [m.case.anomaly_id for m in hits] == [CASE0]
    retriever._client.close()


def test_retrieve_history_excludes_future_and_reports_no_violations():
    store = MemoryStore(InMemoryRetriever(tau_sim=0.0), AuditLog(sqlite_path=":memory:"))
    store.insert_case(_case(0, *TRAIN))
    store.insert_case(_case(1, *TEST))  # a future case (Dec)
    record = {"anomaly_id": "q", "signature": _vec(0), "t_start": QUERY_START}

    history = retrieve_history(store, record, k=5)
    assert [m.case.anomaly_id for m in history] == [CASE0]
    assert leakage_violations(history, QUERY_START) == 0


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_pipeline_has_no_time_leakage():
    from energymind import pipeline

    report = pipeline.run(load_settings(), n_val=10, n_test=8)
    assert report["leakage_violations"] == 0
    assert report["audit"]["valid"] is True


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_ingest_events_is_train_only():
    from energymind.store import ingest_events

    settings = load_settings()
    store = MemoryStore(InMemoryRetriever(), AuditLog(sqlite_path=":memory:"))
    cases = ingest_events(settings, store)
    assert len(cases) > 0
    assert all(c.t_end is not None and c.status == "verified" for c in cases)
    assert all(c.source == "train_history" for c in cases)
    # nothing from the test period (Nov-Dec) is ingested
    assert all(c.t_end < "2020-11-01" for c in cases)
