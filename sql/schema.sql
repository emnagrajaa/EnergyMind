-- EnergyMind memory + audit log (Step 10).
--
-- Vector storage lives in Qdrant (collection `anomaly_case`, 64-dim signature,
-- cosine + payload filters on type_pred/origin_pred/status). This schema defines
-- the part Qdrant cannot provide: an append-only, tamper-evident audit log.
--
--   psql "$PG_DSN" -f sql/schema.sql
--
-- The application inserts only; UPDATE/DELETE are revoked and also blocked by a
-- trigger, and each row chains to the previous one via SHA-256.

CREATE TABLE IF NOT EXISTS audit_event (
    seq         BIGSERIAL PRIMARY KEY,
    anomaly_id  TEXT,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor       TEXT NOT NULL,      -- detector | agent_1 | agent_2 | confidence | operator:<id>
    event_type  TEXT NOT NULL,      -- detected | proposal | feedback | decision | approval | override | outcome
    payload     JSONB NOT NULL,
    prev_hash   BYTEA,
    row_hash    BYTEA NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_event_anomaly_idx ON audit_event (anomaly_id, seq);

-- Enforce append-only at the database level.
CREATE OR REPLACE FUNCTION audit_event_immutable() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_event is append-only (no % allowed)', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_event_no_update ON audit_event;
CREATE TRIGGER audit_event_no_update
    BEFORE UPDATE OR DELETE ON audit_event
    FOR EACH ROW EXECUTE FUNCTION audit_event_immutable();

-- If a dedicated application role exists, revoke writes beyond INSERT.
-- REVOKE UPDATE, DELETE ON audit_event FROM app_role;

-- ---------------------------------------------------------------------------
-- Case mirror (optional). The canonical vectors are in Qdrant; this table is a
-- convenience join for SQL reporting and can be left empty.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anomaly_case (
    anomaly_id     TEXT PRIMARY KEY,
    household_id   TEXT NOT NULL,
    t_start        TIMESTAMPTZ NOT NULL,
    t_end          TIMESTAMPTZ NOT NULL,
    type_pred      TEXT,
    origin_pred    TEXT,
    severity       TEXT,
    status         TEXT NOT NULL DEFAULT 'proposed',
    record         JSONB NOT NULL,
    model_version  TEXT NOT NULL
);
