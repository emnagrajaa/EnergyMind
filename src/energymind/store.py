"""Step 10 - append-only, hash-chained audit log and the memory facade.

The audit log is a **real append-only log**: sequential rows, each storing
``row_hash = sha256(prev_hash || canonical_json(row))``, so any edit or deletion
breaks the chain and is detectable. On SQLite the immutability is enforced with
triggers (``UPDATE``/``DELETE`` raise); on PostgreSQL the same schema is paired
with ``REVOKE UPDATE, DELETE`` (``sql/schema.sql``).

Vectors live in Qdrant (``retrieval.py``); the log lives here. The two are joined
by ``anomaly_id``.

    python -m energymind.store --demo      # offline demo (SQLite + in-memory)
    python -m energymind.store --qdrant    # use the configured Qdrant cluster
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from energymind import io
from energymind.config import Settings, load_settings
from energymind.retrieval import Case, InMemoryRetriever, LocalQdrantRetriever, Match

#: The hash that precedes the first row.
GENESIS_HASH = hashlib.sha256(b"energymind:genesis").hexdigest()

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS audit_event (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    anomaly_id  TEXT,
    ts          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    row_hash    TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_event_no_update
BEFORE UPDATE ON audit_event BEGIN
    SELECT RAISE(ABORT, 'audit_event is append-only (no UPDATE)');
END;
CREATE TRIGGER IF NOT EXISTS audit_event_no_delete
BEFORE DELETE ON audit_event BEGIN
    SELECT RAISE(ABORT, 'audit_event is append-only (no DELETE)');
END;
"""


def canonical_json(obj: object) -> str:
    """Deterministic JSON (sorted keys, no whitespace) for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _row_hash(prev_hash: str, content: dict) -> str:
    return hashlib.sha256((prev_hash + canonical_json(content)).encode()).hexdigest()


class AuditLog:
    """Append-only hash-chained event log (SQLite by default, Postgres optional)."""

    def __init__(
        self,
        backend: str = "sqlite",
        sqlite_path: str | Path = "data/memory/audit.db",
        dsn: str | None = None,
    ) -> None:
        self.backend = backend
        if backend == "sqlite":
            path = io.resolve(sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path))
            self.conn.executescript(SQLITE_DDL)
            self.conn.commit()
        elif backend == "postgres":
            import psycopg  # optional

            if not dsn:
                raise ValueError("postgres backend needs a DSN")
            self.conn = psycopg.connect(dsn, autocommit=True)
        else:
            raise ValueError(f"unknown audit backend: {backend}")
        self._placeholder = "?" if backend == "sqlite" else "%s"

    # -- writes --------------------------------------------------------------
    def _last_hash(self) -> str:
        cur = self.conn.execute("SELECT row_hash FROM audit_event ORDER BY seq DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else GENESIS_HASH

    def append_event(
        self, actor: str, event_type: str, payload: dict, anomaly_id: str | None = None
    ) -> dict:
        """Append one event and return the stored row (with its hash)."""
        ts = datetime.now(UTC).isoformat()
        content = {
            "anomaly_id": anomaly_id,
            "ts": ts,
            "actor": actor,
            "event_type": event_type,
            "payload": payload,
        }
        prev_hash = self._last_hash()
        row_hash = _row_hash(prev_hash, content)
        p = self._placeholder
        if self.backend == "sqlite":
            cur = self.conn.execute(
                "INSERT INTO audit_event (anomaly_id, ts, actor, event_type, payload, prev_hash, "
                f"row_hash) VALUES ({p},{p},{p},{p},{p},{p},{p})",
                (anomaly_id, ts, actor, event_type, canonical_json(payload), prev_hash, row_hash),
            )
            self.conn.commit()
            seq = cur.lastrowid
        else:
            cur = self.conn.execute(
                "INSERT INTO audit_event (anomaly_id, ts, actor, event_type, payload, prev_hash, "
                f"row_hash) VALUES ({p},{p},{p},{p},{p},{p},{p}) RETURNING seq",
                (anomaly_id, ts, actor, event_type, json.dumps(payload), prev_hash, row_hash),
            )
            seq = cur.fetchone()[0]
        return {"seq": seq, **content, "prev_hash": prev_hash, "row_hash": row_hash}

    # -- reads ---------------------------------------------------------------
    def events(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT seq, anomaly_id, ts, actor, event_type, payload, prev_hash, row_hash "
            "FROM audit_event ORDER BY seq"
        )
        out = []
        for seq, anomaly_id, ts, actor, event_type, payload, prev_hash, row_hash in cur.fetchall():
            out.append(
                {
                    "seq": seq,
                    "anomaly_id": anomaly_id,
                    "ts": ts,
                    "actor": actor,
                    "event_type": event_type,
                    "payload": json.loads(payload) if isinstance(payload, str) else payload,
                    "prev_hash": prev_hash,
                    "row_hash": row_hash,
                }
            )
        return out

    def verify_chain(self) -> dict:
        """Recompute every hash; return whether the chain is intact."""
        prev = GENESIS_HASH
        events = self.events()
        for event in events:
            content = {
                "anomaly_id": event["anomaly_id"],
                "ts": event["ts"],
                "actor": event["actor"],
                "event_type": event["event_type"],
                "payload": event["payload"],
            }
            expected = _row_hash(prev, content)
            if event["prev_hash"] != prev or event["row_hash"] != expected:
                return {"valid": False, "events": len(events), "broken_at": event["seq"]}
            prev = event["row_hash"]
        return {"valid": True, "events": len(events), "broken_at": None, "head": prev}


class MemoryStore:
    """The vector memory (Qdrant/in-memory) plus its audit log, joined by anomaly_id."""

    def __init__(self, retriever=None, audit: AuditLog | None = None) -> None:
        self.retriever = retriever or InMemoryRetriever()
        self.audit = audit or AuditLog()

    # -- cases ---------------------------------------------------------------
    def insert_case(self, case: Case) -> Case:
        self.retriever.upsert([case])
        self.audit.append_event(
            actor="detector",
            event_type="proposed",
            payload={
                "type_pred": case.type_pred,
                "origin_pred": case.origin_pred,
                "severity": case.severity,
                "status": case.status,
            },
            anomaly_id=case.anomaly_id,
        )
        return case

    def decide(self, anomaly_id: str, action: str, actor: str = "agent_1") -> dict:
        return self.audit.append_event(
            actor=actor, event_type="decision", payload={"action": action}, anomaly_id=anomaly_id
        )

    def verify(self, anomaly_id: str, actor: str = "operator:sim") -> None:
        """Mark a case verified and record the outcome (only verified cases are reused)."""
        if isinstance(self.retriever, LocalQdrantRetriever):
            self.retriever.set_status(anomaly_id, "verified")
        elif anomaly_id in getattr(self.retriever, "cases", {}):
            self.retriever.cases[anomaly_id].status = "verified"
        self.audit.append_event(
            actor=actor, event_type="outcome", payload={"status": "verified"}, anomaly_id=anomaly_id
        )

    def nearest_verified(
        self, signature, *, type_pred=None, origin_pred=None, k: int = 5
    ) -> list[Match]:
        return self.retriever.search(
            signature, type_pred=type_pred, origin_pred=origin_pred, status="verified", k=k
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _demo_cases(n: int = 6, dim: int = 64) -> list[Case]:
    import numpy as np

    rng = np.random.default_rng(0)
    types = ["gain_error", "additive_bias", "dropout", "coherent_peak"]
    origins = ["data_quality", "data_quality", "data_quality", "physical_event"]
    cases = []
    for i in range(n):
        signature = rng.normal(size=dim).astype(float)
        cases.append(
            Case(
                anomaly_id=f"00000000-0000-0000-0000-{i:012d}",
                signature=signature.tolist(),
                type_pred=types[i % len(types)],
                origin_pred=origins[i % len(origins)],
                status="verified" if i % 2 == 0 else "proposed",
                severity=["low", "medium", "high"][i % 3],
                model_version="0.1.0",
            )
        )
    return cases


def ingest_events(
    settings: Settings,
    store: MemoryStore,
    *,
    feature_file: str = "data/features/features.parquet",
    label_file: str = "data/injected/labels.parquet",
    status: str = "verified",
) -> list[Case]:
    """Build one Case per injected event (from its 64-dim signature) and store them.

    In simulation the injected labels are the ground truth, so events can be marked
    ``verified`` (only verified cases are reused by the "seen before?" rule). Each
    case also writes a ``proposed`` event to the audit log.
    """
    import pandas as pd

    from energymind import evaluate, records

    frame = pd.read_parquet(io.resolve(feature_file))
    labels = pd.read_parquet(io.resolve(label_file))
    events = evaluate.expand_events(labels)

    cases: list[Case] = []
    for anomaly_id, group in events.groupby("anomaly_id", sort=False):
        parts = [
            frame[
                (frame["household_id"] == row.household_id)
                & frame["ts_utc"].between(row.t_start, row.t_end, inclusive="left")
            ]
            for row in group.itertuples()
        ]
        window = pd.concat(parts) if parts else frame.iloc[0:0]
        if window.empty:
            continue
        signature = records.build_signature(window, window["ts_utc"].iloc[0])
        info = group.iloc[0]
        case = Case(
            anomaly_id=str(anomaly_id),
            signature=[float(x) for x in signature],
            type_pred=str(info["type"]),
            origin_pred=str(info["origin"]),
            status=status,
            severity=str(info["severity"]),
            model_version="0.1.0",
        )
        cases.append(case)
        store.insert_case(case)
    return cases


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description="Step 10: Qdrant memory + append-only audit log.")
    parser.add_argument("--configs-dir", default=None)
    parser.add_argument("--qdrant", action="store_true", help="use the configured Qdrant cluster")
    parser.add_argument("--demo", action="store_true", help="run the seeded offline demo (default)")
    parser.add_argument("--reset", action="store_true", help="recreate the Qdrant collection")
    parser.add_argument("--verify", action="store_true", help="only verify the audit chain")
    parser.add_argument(
        "--ingest", action="store_true", help="ingest real injected events as cases"
    )
    parser.add_argument("--feature-file", default="data/features/features.parquet")
    parser.add_argument("--label-file", default="data/injected/labels.parquet")
    parser.add_argument("--sqlite", default="data/memory/audit.db")
    args = parser.parse_args(argv)

    io.load_env()
    settings: Settings = load_settings(args.configs_dir) if args.configs_dir else load_settings()
    audit = AuditLog(settings.store.audit.backend, args.sqlite)

    if args.verify:
        report = audit.verify_chain()
        print("audit chain:", report)
        return 0 if report["valid"] else 1

    retriever = InMemoryRetriever(settings.store.retrieval.tau_sim)
    backend = "in-memory"
    if args.qdrant:
        retriever = LocalQdrantRetriever(settings.store.qdrant)
        if args.reset:
            retriever.delete_collection()
        retriever.ensure_collection()
        backend = f"qdrant:{settings.store.qdrant.collection}"
    store = MemoryStore(retriever, audit)

    if args.ingest:
        cases = ingest_events(
            settings, store, feature_file=args.feature_file, label_file=args.label_file
        )
    else:
        cases = _demo_cases()
        for case in cases:
            store.insert_case(case)
        # promote one case and record a decision -> the audit trail
        store.decide(cases[0].anomaly_id, action="flag_data_suspect")
        store.verify(cases[0].anomaly_id)

    if not cases:
        print("no cases to store (run `python -m energymind.inject` then `energymind.features`)")
        return 1

    anchor = cases[0]
    query = np.asarray(anchor.signature, dtype=float) + np.random.default_rng(1).normal(0, 0.05, 64)
    matches = store.nearest_verified(
        query, type_pred=anchor.type_pred, origin_pred=anchor.origin_pred, k=3
    )

    print(f"backend        : {backend}")
    print(f"cases stored   : {retriever.count()}")
    print(f"audit events   : {len(audit.events())}")
    print(f"chain valid    : {audit.verify_chain()}")
    print("\nfiltered k-NN (verified, same type/origin):")
    for match in matches:
        print(
            f"  {match.case.anomaly_id}  sim={match.similarity:.4f}  "
            f"type={match.case.type_pred}  status={match.case.status}"
        )
    print(f"\ntau_sim = {settings.store.retrieval.tau_sim} -> 'seen before' if sim >= tau")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
