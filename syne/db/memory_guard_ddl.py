"""Memory guard (Riyo, 18 Nov 2026) - v1.23.07.

Single source for migration 57. schema.sql carries the identical text;
tests/test_memory_guard.py asserts it.

A chunk that another chunk refers to cannot be deleted. To delete it,
delete the chunks that refer to it first, or delete them together in one
statement. "Refers to" follows the knowledge graph:

  * directly: a relation from a living chunk to this chunk (next_part:
    part i refers to part i+1);
  * through what this chunk contains: this chunk --contains--> ayah, and a
    living chunk --explains/cites/asbab_of/...--> that ayah.

Memories without a chunk, or chunks nobody refers to, delete as before.
There is no bypass. The trigger is AFTER DELETE FOR EACH STATEMENT and its
name sorts before trg_memory_kg_prune, so it runs first: a refused delete
is rolled back whole and the graph is never pruned.
"""

MEMORY_GUARD_SCHEMA_SQL = r"""
-- ============================================================
-- MEMORY GUARD (Riyo, 18 Nov 2026).
-- Mirrors migration 57 (syne/db/memory_guard_ddl.py, kept identical).
-- A chunk referenced by another living chunk cannot be deleted - directly
-- (next_part) or through an ayah it contains (explains, cites, asbab_of).
-- Delete the referencing chunks first, or together in one statement.
-- Runs before trg_memory_kg_prune (triggers fire in name order).
-- ============================================================
CREATE OR REPLACE FUNCTION memory_guard_on_delete() RETURNS trigger AS $$
DECLARE
    n INT;
    detail TEXT;
BEGIN
    WITH g AS (
        SELECT e.id AS ent, o.id AS mem
          FROM gone o
          JOIN kg_entities e
            ON e.entity_type = 'chunk' AND e.description = 'memory:' || o.id
    ), targets AS (
        SELECT g.mem, g.ent AS target FROM g
        UNION
        SELECT g.mem, r.object_id FROM g
          JOIN kg_relations r ON r.subject_id = g.ent AND r.predicate = 'contains'
    ), refs AS (
        SELECT DISTINCT t.mem, m.id AS by_mem
          FROM targets t
          JOIN kg_relations r ON r.object_id = t.target
          JOIN kg_entities s ON s.id = r.subject_id AND s.entity_type = 'chunk'
          JOIN memory m ON m.id = CASE WHEN s.description ~ '^memory:[0-9]{1,9}$'
                                       THEN substring(s.description from 8)::int END
    )
    SELECT count(*), string_agg(line, '; ') INTO n, detail FROM (
        SELECT format('#%s is referenced by %s other memories (e.g. #%s)',
                      mem, count(*), min(by_mem)) AS line
          FROM refs GROUP BY mem ORDER BY mem
    ) x;
    IF n > 0 THEN
        RAISE EXCEPTION USING
            ERRCODE = 'restrict_violation',
            MESSAGE = format('memory guard: %s memories are referenced by other chunks and cannot be deleted: %s',
                             n, left(detail, 900)),
            HINT = 'Delete the chunks that reference them first, or delete them together in one statement.';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger
                    WHERE tgname = 'trg_memory_guard' AND tgrelid = 'memory'::regclass) THEN
        CREATE TRIGGER trg_memory_guard AFTER DELETE ON memory
            REFERENCING OLD TABLE AS gone
            FOR EACH STATEMENT EXECUTE FUNCTION memory_guard_on_delete();
    END IF;
END $$;
""".strip()

GUARD_TRIGGER = "trg_memory_guard"


def guard_refusal(exc: BaseException):
    """Return the guard's message when exc is a memory-guard refusal, else None."""
    if getattr(exc, "sqlstate", None) != "23001":
        return None
    msg = str(exc)
    if "memory guard" not in msg:
        return None
    hint = getattr(exc, "hint", None)
    return f"{msg} {hint}" if hint else msg
