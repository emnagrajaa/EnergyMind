"""Tests for Step 10 retrieval: cosine, filters, in-memory and Qdrant (embedded)."""

from __future__ import annotations

import pytest

from energymind.config import load_settings
from energymind.retrieval import Case, InMemoryRetriever, LocalQdrantRetriever, cosine


def _case(i: int, **kw) -> Case:
    vec = [0.0] * 64
    vec[i % 64] = 1.0
    vec[(i + 1) % 64] = 0.5
    defaults = {
        "anomaly_id": f"00000000-0000-0000-0000-{i:012d}",
        "signature": vec,
        "type_pred": "gain_error",
        "origin_pred": "data_quality",
        "status": "verified",
        "severity": "medium",
    }
    defaults.update(kw)
    return Case(**defaults)


def test_cosine_bounds_and_orthogonality():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def test_inmemory_filters_and_ranking():
    retriever = InMemoryRetriever()
    retriever.upsert(
        [
            _case(0),
            _case(1, status="proposed"),
            _case(2, type_pred="dropout"),
            _case(3, origin_pred="adversarial"),
        ]
    )
    query = _case(0).signature
    # only verified + same type + same origin; case 0 itself should rank first
    hits = retriever.search(
        query, type_pred="gain_error", origin_pred="data_quality", status="verified", k=5
    )
    ids = [m.case.anomaly_id for m in hits]
    assert ids == ["00000000-0000-0000-0000-000000000000"]
    assert hits[0].similarity > 0.9


def test_inmemory_status_none_returns_all_types():
    retriever = InMemoryRetriever()
    retriever.upsert([_case(0), _case(2, type_pred="dropout")])
    hits = retriever.search(_case(0).signature, type_pred=None, origin_pred=None, status=None, k=10)
    assert len(hits) == 2


def test_qdrant_embedded_roundtrip(tmp_path):
    pytest.importorskip("qdrant_client")
    settings = load_settings()
    config = settings.store.qdrant.model_copy(
        update={"local_path": str(tmp_path), "collection": "t_case"}
    )
    retriever = LocalQdrantRetriever(config)
    retriever.upsert([_case(0), _case(1, status="proposed"), _case(2, type_pred="dropout")])
    assert retriever.count() == 3

    hits = retriever.search(
        _case(0).signature,
        type_pred="gain_error",
        origin_pred="data_quality",
        status="verified",
        k=5,
    )
    # only case 0 is verified + gain_error + data_quality (case 1 is proposed, case 2 dropout)
    assert len(hits) == 1
    assert hits[0].similarity > 0.99

    retriever.set_status(_case(1).anomaly_id, "verified")
    hits2 = retriever.search(
        _case(1).signature,
        type_pred="gain_error",
        origin_pred="data_quality",
        status="verified",
        k=5,
    )
    assert len(hits2) == 2
    assert hits2[0].similarity >= hits2[-1].similarity
    retriever._client.close()
