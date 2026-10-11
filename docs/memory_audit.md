# Memory and audit log (Step 10)

> Implementation: `src/energymind/store.py`, `src/energymind/retrieval.py`,
> `src/energymind/config.py` (`StoreConfig`), `configs/store.yaml`, `sql/schema.sql`,
> `docker-compose.yml`. Tests: `tests/test_store.py`, `tests/test_retrieval.py`.

This document describes how EnergyMind remembers past cases ("seen before?") and
how it keeps a tamper-evident log of every decision.

---

## 1. Design: two stores, one key

| Concern | Store | Why |
| --- | --- | --- |
| **Vector memory** — find similar *verified* past cases | **Qdrant** | purpose-built filtered k-NN over the 64-dim signature |
| **Audit log** — append-only, tamper-evident history | **SQLite** (default) / **PostgreSQL** | ACID, SQL, and database-level immutability |

Both are joined by `anomaly_id`.

**Why not Qdrant for the audit log.** A vector database has no transactions, no
SQL ordering/joins, and no permission model to forbid `update`/`delete` per
collection, so the hash-chain / tamper-evidence requirement (blueprint §10)
cannot be enforced there. Qdrant is therefore used **for vectors only**; the log
lives in a real database. This is exactly what the blueprint recommends.

```
 case ──► Qdrant collection `anomaly_case`      (64-dim signature, cosine, payload filters)
   │
   └────► audit_event table (hash chain)         detector | agent_1 | agent_2 | confidence | operator
```

## 2. What a memory point is (and how many to expect)

**One Qdrant point = one anomaly *case* (one event)**, never one meter reading or
one window. Raw intervals are deliberately **not** embedded (redundant, and it
would expose raw consumption data that Phase 1 forbids sharing).

| Level | Reusable points | Notes |
| --- | --- | --- |
| Phase 1 demo (5%, seed 1) | **290** | all events of one run, verified in simulation |
| Phase 1 realistic (1 neighbourhood-year) | hundreds – few thousand | blueprint's own figure; only *verified* cases kept |
| Phase 2 (federated, N nodes) | ~1k × N nodes | 50 neighbourhoods ≈ 50k cases |

Storage: 64 × float32 = **256 B/vector** + ~200 B payload ≈ **0.5 KB/case**, so
1M cases ≈ **0.5 GB** — comfortably inside Qdrant's "millions of vectors per node".
The vector count is *small by design*; it scales with anomalies seen, not with the
sampling rate.

Two things called "snapshots" exist and are **not** the same as cases:
Qdrant **collection snapshots** (ops backup/restore) and Phase-2
**experience snapshots** (`experience_id = month`, model-training streams).

**Memory hygiene (blueprint §10):** only `status = 'verified'` cases are returned
by search; `proposed` / `human_labelled` / `rejected` / `expired` are excluded, so
unconfirmed feedback never leaks into later decisions. Duplicates collapse via the
filters (same predicted `type` and `origin`).

## 3. Retrieval (not RAG)

Cases are retrieved by **filtered vector similarity**, not RAG: the 64-dim numeric
signature (physics, `records.build_signature`) is compared with cosine similarity,
and payload filters (`type_pred`, `origin_pred`, `status`) stop cross-type matches.

```
Retriever.search(signature, type_pred=..., origin_pred=..., status="verified", k=5) -> [Match]
```

Two implementations behind one interface:

* `LocalQdrantRetriever` — cloud / server / embedded Qdrant;
* `InMemoryRetriever` — numpy fallback for tests and offline runs.

A "seen before" match requires cosine similarity ≥ `tau_sim` (default 0.9,
chosen on validation) plus matching `type`/`origin` and `status = verified`.

## 4. The append-only hash chain

Each audit row stores `row_hash = SHA-256(prev_hash ‖ canonical_json(row))`, with
`prev_hash` equal to the previous row's `row_hash` (the first row chains to a
fixed `GENESIS_HASH`). Any edit or deletion breaks the chain, and
`verify_chain()` recomputes every hash to detect it.

```sql
CREATE TABLE audit_event (
    seq BIGSERIAL PRIMARY KEY, anomaly_id TEXT, ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor TEXT NOT NULL, event_type TEXT NOT NULL, payload JSONB NOT NULL,
    prev_hash BYTEA, row_hash BYTEA NOT NULL
);
```

* **SQLite** enforces immutability with `BEFORE UPDATE`/`BEFORE DELETE` triggers
  that `RAISE(ABORT)`.
* **PostgreSQL** uses the same trigger plus `REVOKE UPDATE, DELETE ON audit_event`.

The full SQL is in `sql/schema.sql`; the matching SQLite DDL lives in
`store.SQLITE_DDL`.

## 5. Configuration

`configs/store.yaml`:

```yaml
qdrant:
  url: ""                     # empty -> QDRANT_URL from the environment
  api_key_env: QDRANT_API_KEY
  collection: anomaly_case
  vector_size: 64
  distance: cosine
  local_path: ""              # set to run an embedded Qdrant (offline)
audit:
  backend: sqlite             # sqlite (default) | postgres
  sqlite_path: data/memory/audit.db
retrieval:
  k: 5
  tau_sim: 0.9
```

`.env` (git-ignored; `.env.example` is committed):

```
QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io:6333
QDRANT_API_KEY=<key>
PG_DSN=postgresql://energymind:energymind@localhost:5432/energymind   # optional
```

A single `.env` value is enough for the whole app; `io.load_env()` injects it into
the process environment. The secret is **never** written to a committed file.

## 6. Usage

```bash
# offline: in-memory vectors + SQLite log, seeded demo cases
python -m energymind.store --demo

# ingest the real injected events of the built run (290 cases) -> offline
python -m energymind.store --demo --ingest

# live cloud Qdrant
python -m energymind.store --qdrant --reset --ingest

# ingest every (prevalence, seed) and print the memory growth curve
python -m energymind.store --all-runs                 # uses configs' prevalences/seeds
python -m energymind.store --all-runs --seeds 1 2 3
python -m energymind.store --qdrant --reset --all-runs

# verify the append-only hash chain
python -m energymind.store --verify

# local services instead of the cloud
docker compose up -d qdrant      # then QDRANT_URL=http://localhost:6333
```

Programmatic API:

```python
from energymind.store import MemoryStore, AuditLog, ingest_all_runs
from energymind.retrieval import LocalQdrantRetriever

store = MemoryStore(LocalQdrantRetriever(settings.store.qdrant), AuditLog())
store.insert_case(case)                     # upsert + "proposed" audit event
store.decide(case.anomaly_id, "flag_data_suspect")
store.verify(case.anomaly_id)               # only verified cases are reusable
store.nearest_verified(sig, type_pred=..., origin_pred=..., k=5)
store.audit.verify_chain()                  # -> {"valid": True, ...}
```

## 7. Test evidence

### 7.1 Unit tests (offline, no network)

`tests/test_store.py` (5):

* `canonical_json` is order-independent;
* the chain links (`second.prev_hash == first.row_hash`) and `verify_chain` returns
  `{valid: True}`;
* **append-only**: `UPDATE`/`DELETE` on `audit_event` raise `sqlite3.Error`;
* **tamper detection**: after dropping the trigger and editing a row,
  `verify_chain` → `{valid: False, broken_at: 1}`;
* `MemoryStore` writes `proposed → decision → outcome` and
  `nearest_verified` returns the case with similarity ≈ 1.0.

`tests/test_retrieval.py` (9): cosine bounds, filter correctness, ranking, and a
full **embedded-Qdrant** upsert/filter/`set_status` round-trip.

Full suite: **92 tests pass** locally; the CI-mirror venv (no Qdrant/shap/torch)
passes with 2 skips.

### 7.2 Live cloud Qdrant (`--qdrant --reset --ingest`)

```
backend        : qdrant:anomaly_case
cases stored   : 290                    # every injected event of the 5%/seed-1 run
audit events   : 359
chain valid    : {'valid': True, 'events': 359, 'broken_at': None, 'head': 'e1daad8afd8b…'}
filtered k-NN  : 24f694ea-… sim=0.9989  type=additive_bias  status=verified   ← exact match
```

### 7.3 Memory growth curve (`--all-runs`, seed 1)

```
prevalence=0.01  seed=1  cases=59    total=59     audit_events=59
prevalence=0.05  seed=1  cases=290   total=349    audit_events=349
prevalence=0.10  seed=1  cases=601   total=950    audit_events=950
chain valid    : {'valid': True, 'events': 950, ...}
```

Cases grow with prevalence (59 → 290 → 601) and accumulate monotonically; the
audit log grows one event per case. With all 3 seeds the grid is ~2,850 cases —
still tiny for Qdrant.

### 7.4 Bugs found and fixed by this step

* **Cloud Qdrant requires payload indexes** on the filtered fields; `ensure_collection`
  now creates `KEYWORD` indexes on `type_pred`, `origin_pred`, `status`.
* **NaN signatures** (events whose window had missing sub-meters) were rejected by
  Qdrant (`VectorStruct` 400); `records.build_signature` now ends with
  `np.nan_to_num`, so signatures are always finite (and an all-NaN window returns
  0 rather than emitting a warning).

## 8. Limitations

* Qdrant has no TTL enforcement here; `valid_until` is stored on the point but
  expiry is applied by the caller (Phase 2 uses a scheduled sweep).
* A **neighbourhood event** is stored as **one** point, not one per household.
  Switch the grouping in `store._cases_from_frame` (`anomaly_id` →
  `(anomaly_id, household_id)`) if per-household granularity is wanted.
* The audit `actor`/`event_type` values are written by the demo and (in Step 11)
  by the deliberation loop; the schema already covers `detector | agent_1 |
  agent_2 | confidence | operator:<id>` and `detected | proposal | feedback |
  decision | approval | override | outcome`.
* The Postgres path requires `docker compose up -d postgres` (or a managed DSN);
  SQLite is the zero-setup default.

## 9. Security

The Qdrant API key is read from the git-ignored `.env` only. **Rotate the key**
after sharing it, then update `.env`. Verify it stays out of git with
`git check-ignore -v .env`.
