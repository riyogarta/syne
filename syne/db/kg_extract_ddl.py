"""KG ingestion DDL (Riyo, 18 Nov 2026) - KG-RAG stage 2.

Single source for migration 53. schema.sql carries the identical text;
tests/test_kg_extract.py asserts the two stay byte-identical.

  * kg_entities.norm_name - normalised name (syne/memory/kg_extract.py
    normalize_name) used by entity resolution; filled for old entities by
    the KG worker at start.
  * kg_extract_log        - one row per processed chunk (memory row or chat
    message): done / skipped / error, model, counts, attempts.
  * kg.extract_era_start  - only chunks created at or after this moment are
    extracted (old chunks are never processed again, Riyo, memory #57758).
    Set once, at the moment this runs; ON CONFLICT keeps the first value.
"""

KG_EXTRACT_SCHEMA_SQL = r"""
-- ============================================================
-- KG INGESTION (Riyo, 18 Nov 2026) - KG-RAG stage 2.
-- Mirrors migration 53 (syne/db/kg_extract_ddl.py, kept byte-identical).
-- ============================================================
ALTER TABLE kg_entities ADD COLUMN IF NOT EXISTS norm_name TEXT;
CREATE INDEX IF NOT EXISTS idx_kg_entities_norm_name ON kg_entities (norm_name)
    WHERE norm_name IS NOT NULL;

CREATE TABLE IF NOT EXISTS kg_extract_log (
    source_type VARCHAR(20) NOT NULL CHECK (source_type IN ('memory', 'message')),
    source_id   INT NOT NULL,
    layer       SMALLINT,                 -- 0 memory, 1 user message, 2 assistant message
    session_id  INT,
    status      VARCHAR(10) NOT NULL CHECK (status IN ('done', 'skipped', 'error')),
    reason      TEXT,
    model       TEXT,
    n_entities  INT NOT NULL DEFAULT 0,
    n_relations INT NOT NULL DEFAULT 0,
    latency_ms  INT,
    attempts    INT NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_type, source_id)
);

INSERT INTO config (key, value, description) VALUES
    ('kg.extract_era_start',
     to_jsonb(to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"')),
     'KG ingestion era: only memories and chat messages created at or after this moment (ISO, UTC) are extracted into the KG. Set once when migration 53 ran; older chunks are never extracted again. Moving it earlier makes the worker process older rows.')
ON CONFLICT (key) DO NOTHING;
""".strip()
