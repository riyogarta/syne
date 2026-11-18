"""KG prune on delete (Riyo, 18 Nov 2026) - v1.23.04.

Single source for migration 56. schema.sql carries the identical text;
tests/test_kg_prune.py asserts the two stay byte-identical.

Before this, deleting a memory left its graph behind: kg_relation_sources
has no foreign key (a source is a memory OR a message), kg_relations only
set source_memory_id to NULL, chunk entities waited for a manual prune, and
entities stayed with no relation. Every place that deletes memories or
messages (memory_delete, dedup, CLI, /new, refused turns, migrations) now
goes through the triggers below, so none of them needs its own cleanup.

  * kg_prune_sources(type, ids) - the cleanup for deleted sources of one
    type ('memory' | 'message'). Reference entities (the ARRAY list, kept
    equal to syne/memory/kg_extract.py REFERENCE_TYPES by a test) are never
    removed for being alone; chunk entities go only with their own memory.
  * trg_memory_kg_prune / trg_messages_kg_prune - statement-level AFTER
    DELETE triggers (one call per DELETE, however many rows), wrapped so a
    failure is a WARNING and never blocks the delete.
"""

KG_PRUNE_SCHEMA_SQL = r"""
-- ============================================================
-- KG PRUNE ON DELETE (Riyo, 18 Nov 2026).
-- Mirrors migration 56 (syne/db/kg_prune_ddl.py, kept byte-identical).
-- Deleting a memory or chat message removes what the knowledge graph holds
-- from it: its kg_relation_sources and kg_extract_log rows, the relations
-- left with no source, the chunk entity memory:<id>, and the entities those
-- relations touched that are left with no relation (reference types kept).
-- A relation that keeps another source points source_memory_id at a living
-- memory again. A failure here is logged as a WARNING and never blocks the
-- delete itself.
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_kg_entities_chunk_description
    ON kg_entities (description) WHERE entity_type = 'chunk';

CREATE OR REPLACE FUNCTION kg_prune_sources(p_type TEXT, p_ids INT[]) RETURNS void AS $$
DECLARE
    rels INT[];
    ents INT[];
BEGIN
    IF p_ids IS NULL OR cardinality(p_ids) = 0 THEN
        RETURN;
    END IF;
    DELETE FROM kg_extract_log WHERE source_type = p_type AND source_id = ANY(p_ids);
    WITH d AS (
        DELETE FROM kg_relation_sources
         WHERE source_type = p_type AND source_id = ANY(p_ids)
        RETURNING relation_id
    )
    SELECT array_agg(DISTINCT relation_id) INTO rels FROM d;
    IF rels IS NOT NULL THEN
        WITH d AS (
            DELETE FROM kg_relations r
             WHERE r.id = ANY(rels)
               AND NOT EXISTS (SELECT 1 FROM kg_relation_sources s WHERE s.relation_id = r.id)
            RETURNING r.subject_id, r.object_id
        )
        SELECT array_agg(DISTINCT u.e) INTO ents
          FROM (SELECT subject_id AS e FROM d UNION SELECT object_id FROM d) u;
        UPDATE kg_relations r
           SET source_memory_id = (
                   SELECT max(s.source_id) FROM kg_relation_sources s
                    WHERE s.relation_id = r.id AND s.source_type = 'memory'
                      AND EXISTS (SELECT 1 FROM memory m WHERE m.id = s.source_id)),
               updated_at = NOW()
         WHERE r.id = ANY(rels)
           AND (r.source_memory_id IS NULL
                OR NOT EXISTS (SELECT 1 FROM memory m WHERE m.id = r.source_memory_id))
           AND EXISTS (SELECT 1 FROM kg_relation_sources s JOIN memory m ON m.id = s.source_id
                        WHERE s.relation_id = r.id AND s.source_type = 'memory');
    END IF;
    IF p_type = 'memory' THEN
        DELETE FROM kg_entities
         WHERE entity_type = 'chunk'
           AND description = ANY(ARRAY(SELECT 'memory:' || x FROM unnest(p_ids) AS t(x)));
    END IF;
    IF ents IS NOT NULL THEN
        DELETE FROM kg_entities e
         WHERE e.id = ANY(ents)
           AND COALESCE(e.entity_type, '') <> ALL(ARRAY['chunk', 'ayah', 'quran_ayah_ref', 'quran_chunk', 'quran_surah', 'kaidah_entry']::text[])
           AND NOT EXISTS (SELECT 1 FROM kg_relations r
                            WHERE r.subject_id = e.id OR r.object_id = e.id);
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION kg_prune_on_delete() RETURNS trigger AS $$
BEGIN
    BEGIN
        PERFORM kg_prune_sources(TG_ARGV[0], ARRAY(SELECT id FROM gone));
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'kg_prune_on_delete(%): %', TG_ARGV[0], SQLERRM;
    END;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger
                    WHERE tgname = 'trg_memory_kg_prune' AND tgrelid = 'memory'::regclass) THEN
        CREATE TRIGGER trg_memory_kg_prune AFTER DELETE ON memory
            REFERENCING OLD TABLE AS gone
            FOR EACH STATEMENT EXECUTE FUNCTION kg_prune_on_delete('memory');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger
                    WHERE tgname = 'trg_messages_kg_prune' AND tgrelid = 'messages'::regclass) THEN
        CREATE TRIGGER trg_messages_kg_prune AFTER DELETE ON messages
            REFERENCING OLD TABLE AS gone
            FOR EACH STATEMENT EXECUTE FUNCTION kg_prune_on_delete('message');
    END IF;
END $$;
""".strip()


async def prune_orphan_sources(conn) -> dict:
    """Prune sources deleted before the triggers existed. Idempotent.

    A memory id counts when it is a kg_relation_sources / kg_extract_log
    source or a chunk entity memory:<id> and the memory row is gone; a
    message id the same against messages. Returns {type: ids pruned}.
    """
    out = {}
    for st, table in (("memory", "memory"), ("message", "messages")):
        extra = ""
        if st == "memory":
            extra = (" UNION SELECT substring(description from 8)::int FROM kg_entities"
                     " WHERE entity_type = 'chunk' AND description ~ '^memory:[0-9]{1,9}$'")
        ids = await conn.fetchval(
            "SELECT array_agg(s.id) FROM (SELECT source_id AS id FROM kg_relation_sources"
            " WHERE source_type = $1 UNION SELECT source_id FROM kg_extract_log"
            f" WHERE source_type = $1{extra}) s"
            f" WHERE NOT EXISTS (SELECT 1 FROM {table} x WHERE x.id = s.id)", st)
        ids = list(ids or [])
        if ids:
            await conn.execute("SELECT kg_prune_sources($1, $2::int[])", st, ids)
        out[st] = len(ids)
    return out
