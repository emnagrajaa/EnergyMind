"""Step 10 - case memory and retrieval.

Cases are stored and retrieved by **filtered vector similarity search**, not RAG:
a fixed 64-dim physics signature (``records.build_signature``) is compared with
cosine similarity, and SQL/payload filters (``type_pred``, ``origin_pred``,
``status``) stop cross-type matches.

Two retriever implementations behind one interface:

* :class:`LocalQdrantRetriever` - a Qdrant collection (cloud, server or embedded);
* :class:`InMemoryRetriever` - a numpy fallback for tests and offline runs.

Keys are compared on physics, so text embeddings stay out of the vector space.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import numpy as np

from energymind import io

#: Statuses a case can hold (blueprint Section 10).
STATUSES = ("proposed", "decided", "human_labelled", "verified", "rejected", "expired")


@dataclass
class Case:
    anomaly_id: str
    signature: list[float]
    type_pred: str
    origin_pred: str
    status: str = "proposed"
    severity: str | None = None
    model_version: str = ""
    valid_until: str | None = None
    record: dict = field(default_factory=dict)
    timestamp: str | None = None
    #: The action this case was decided with (carried so "seen before" reuse is meaningful).
    decision_action: str | None = None
    #: Calibrated confidence and routing tier once the case has been decided.
    confidence: float | None = None
    tier: int | None = None
    #: Event window (ISO-8601 UTC) used to keep memory time-causal.
    t_start: str | None = None
    t_end: str | None = None
    #: Provenance of the case ('train_history' for ingested train events, 'decision' for
    #: cases written back after being decided).
    source: str | None = None


@dataclass
class Match:
    case: Case
    similarity: float


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cosine(a, b) -> float:
    """Cosine similarity of two numeric vectors (0 if either is zero)."""
    a = _unit(np.asarray(a, dtype=float))
    b = _unit(np.asarray(b, dtype=float))
    return float(np.dot(a, b))


@runtime_checkable
class Retriever(Protocol):
    def upsert(self, cases: list[Case]) -> None: ...

    def search(
        self,
        signature,
        *,
        type_pred: str | None = None,
        origin_pred: str | None = None,
        status: str | None = "verified",
        k: int = 5,
        before: str | None = None,
    ) -> list[Match]: ...

    def count(self) -> int: ...


def _passes(case: Case, type_pred, origin_pred, status) -> bool:
    return (
        (status is None or case.status == status)
        and (type_pred is None or case.type_pred == type_pred)
        and (origin_pred is None or case.origin_pred == origin_pred)
    )


def _parse_ts(value: str | None):
    """Parse an ISO-8601 timestamp (``Z`` or offset) to an aware datetime, or None."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment


def _is_causal(case: Case, before: str | None) -> bool:
    """True unless the case is not strictly before ``before`` (time-causal memory)."""
    if before is None:
        return True
    moment = _parse_ts(case.t_end)
    cutoff = _parse_ts(before)
    if moment is None or cutoff is None:
        return False  # unproven causality is treated as a violation
    return moment < cutoff


# --------------------------------------------------------------------------- #
# In-memory retriever (tests / offline)
# --------------------------------------------------------------------------- #


class InMemoryRetriever:
    """A numpy k-NN over the signatures, with the same filters as Qdrant."""

    def __init__(self, tau_sim: float = 0.9) -> None:
        self.tau_sim = tau_sim
        self.cases: dict[str, Case] = {}

    def upsert(self, cases: list[Case]) -> None:
        for case in cases:
            self.cases[case.anomaly_id] = case

    def search(
        self,
        signature,
        *,
        type_pred: str | None = None,
        origin_pred: str | None = None,
        status: str | None = "verified",
        k: int = 5,
        before: str | None = None,
    ) -> list[Match]:
        query = np.asarray(signature, dtype=float)
        matches = [
            Match(case, cosine(query, case.signature))
            for case in self.cases.values()
            if _passes(case, type_pred, origin_pred, status) and _is_causal(case, before)
        ]
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:k]

    def count(self) -> int:
        return len(self.cases)

    def update_payload(self, anomaly_id: str, payload: dict) -> None:
        case = self.cases.get(anomaly_id)
        if case is None:
            return
        for key, value in payload.items():
            if hasattr(case, key):
                setattr(case, key, value)


# --------------------------------------------------------------------------- #
# Qdrant retriever
# --------------------------------------------------------------------------- #
_DISTANCE = {"cosine": "Cosine", "dot": "Dot", "euclid": "Euclid"}


class LocalQdrantRetriever:
    """Filtered k-NN over a Qdrant collection.

    The client is created lazily. Precedence for the endpoint:
    ``config.url`` -> ``QDRANT_URL`` in the environment; ``config.local_path``
    runs an embedded Qdrant instead. The API key is read from ``config.api_key_env``.
    """

    def __init__(self, config, *, client=None) -> None:
        self.config = config
        self._client = client

    # -- client --------------------------------------------------------------
    def _connect(self):
        if self._client is not None:
            return self._client
        io.load_env()
        from qdrant_client import QdrantClient

        local_path = getattr(self.config, "local_path", "") or ""
        if local_path:
            path = io.resolve(local_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
        else:
            url = getattr(self.config, "url", "") or os.environ.get("QDRANT_URL", "")
            api_key = os.environ.get(self.config.api_key_env, "") or None
            if not url:
                raise RuntimeError(
                    "no Qdrant endpoint: set QDRANT_URL (or qdrant.url / qdrant.local_path)"
                )
            self._client = QdrantClient(url=url, api_key=api_key, timeout=30)
        return self._client

    @property
    def client(self):
        return self._connect()

    # -- schema --------------------------------------------------------------
    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        client = self._connect()
        name = self.config.collection
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=getattr(Distance, _DISTANCE[self.config.distance].upper()),
                ),
            )
        # Index the payload fields we filter on (required by the server).
        with contextlib.suppress(Exception):  # indexes may already exist
            for field_name in ("type_pred", "origin_pred", "status"):
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        with contextlib.suppress(Exception):  # time-causal range filter
            client.create_payload_index(
                collection_name=name, field_name="t_end", field_schema=PayloadSchemaType.DATETIME
            )

    # -- writes --------------------------------------------------------------
    def upsert(self, cases: list[Case]) -> None:
        from qdrant_client.models import PointStruct

        self.ensure_collection()
        points = [
            PointStruct(
                id=case.anomaly_id,
                vector=[float(x) for x in case.signature],
                payload={
                    "type_pred": case.type_pred,
                    "origin_pred": case.origin_pred,
                    "status": case.status,
                    "severity": case.severity,
                    "model_version": case.model_version,
                    "valid_until": case.valid_until,
                    "decision_action": case.decision_action,
                    "confidence": case.confidence,
                    "tier": case.tier,
                    "t_start": case.t_start,
                    "t_end": case.t_end,
                    "source": case.source,
                },
            )
            for case in cases
        ]
        self._connect().upsert(collection_name=self.config.collection, points=points)

    def set_status(self, anomaly_id: str, status: str) -> None:
        self.update_payload(anomaly_id, {"status": status})

    def update_payload(self, anomaly_id: str, payload: dict) -> None:
        """Merge fields into an existing point's payload (e.g. record a decision)."""
        self._connect().set_payload(
            collection_name=self.config.collection,
            payload=payload,
            points=[anomaly_id],
        )

    # -- reads ---------------------------------------------------------------
    def search(
        self,
        signature,
        *,
        type_pred: str | None = None,
        origin_pred: str | None = None,
        status: str | None = "verified",
        k: int = 5,
        before: str | None = None,
    ) -> list[Match]:
        from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchValue

        must = []
        if type_pred is not None:
            must.append(FieldCondition(key="type_pred", match=MatchValue(value=type_pred)))
        if origin_pred is not None:
            must.append(FieldCondition(key="origin_pred", match=MatchValue(value=origin_pred)))
        if status is not None:
            must.append(FieldCondition(key="status", match=MatchValue(value=status)))
        if before is not None:
            # server-side time-causal filter (best effort); re-checked client-side below
            with contextlib.suppress(Exception):
                must.append(FieldCondition(key="t_end", range=DatetimeRange(lt=before)))
        query_filter = Filter(must=must) if must else None

        response = self._connect().query_points(
            collection_name=self.config.collection,
            query=[float(x) for x in signature],
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )
        matches: list[Match] = []
        for point in response.points:
            payload = point.payload or {}
            case = Case(
                anomaly_id=str(point.id),
                signature=[],
                type_pred=payload.get("type_pred", ""),
                origin_pred=payload.get("origin_pred", ""),
                status=payload.get("status", ""),
                severity=payload.get("severity"),
                model_version=payload.get("model_version", ""),
                valid_until=payload.get("valid_until"),
                decision_action=payload.get("decision_action"),
                confidence=payload.get("confidence"),
                tier=payload.get("tier"),
                t_start=payload.get("t_start"),
                t_end=payload.get("t_end"),
                source=payload.get("source"),
            )
            if not _is_causal(case, before):
                continue
            matches.append(Match(case=case, similarity=float(point.score)))
        return matches

    def count(self) -> int:
        return int(self._connect().count(self.config.collection, exact=True).count)

    def delete_collection(self) -> None:
        client = self._connect()
        if client.collection_exists(self.config.collection):
            client.delete_collection(self.config.collection)
