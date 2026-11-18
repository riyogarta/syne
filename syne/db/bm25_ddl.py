"""BM25 keyword index DDL (Riyo, 18 Nov 2026) - KG-RAG stage 1, step 1.

Single source for migration 51. schema.sql carries the identical text
(fresh-install path and the idempotent re-run at every start);
tests/test_bm25_schema.py asserts the two stay byte-identical.

What it builds:
  * memory.tsv / messages.tsv  - the word index of each row (tsvector), and
    tsv_len - the row's length in tokens (BM25 needs document length).
  * syne_tsv(text)             - how text becomes words: Arabic harakat and
    tatweel are removed first, so "بسم الله" matches "بِسْمِ اللَّهِ"; then the
    'simple' parser (lower-case, no stemming: content mixes Indonesian,
    Arabic and English and Postgres has no Indonesian stemmer).
  * bm25_corpus / bm25_terms   - BM25 statistics per corpus: number of
    documents, total length, and document frequency per word. Corpora:
    'memory', 'messages:user', 'messages:assistant' (history_search level
    1 / 2). Rows with zero tokens are not counted.
  * triggers                   - fill tsv on INSERT and on UPDATE of content
    and keep the statistics in step on INSERT / UPDATE / DELETE. Updates of
    other columns (access_count, embedding, status) do not fire them.
  * syne_bm25_rebuild()        - recompute the statistics from the rows, in
    case they drift (a statistics error never blocks a write; it is only
    logged as a WARNING).

Old rows keep tsv NULL until the background backfill (step 2) indexes them;
the *_tsv_pending indexes let it page through them.

The function blocks are written for syne.cli.helpers._split_sql_statements:
plpgsql bodies end with "$$ LANGUAGE plpgsql;", sql bodies and DO blocks
with "$$;".
"""

BM25_SCHEMA_SQL = r"""
-- ============================================================
-- BM25 KEYWORD INDEX (Riyo, 18 Nov 2026) - KG-RAG stage 1.
-- Mirrors migration 51 (syne/db/bm25_ddl.py, kept byte-identical).
-- tsv = word index of the row, tsv_len = its length in tokens.
-- bm25_corpus / bm25_terms = BM25 statistics, kept in step by triggers.
-- Old rows stay tsv NULL until the background backfill indexes them.
-- ============================================================
ALTER TABLE memory ADD COLUMN IF NOT EXISTS tsv tsvector;
ALTER TABLE memory ADD COLUMN IF NOT EXISTS tsv_len INT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tsv tsvector;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tsv_len INT;

CREATE TABLE IF NOT EXISTS bm25_corpus (
    corpus    TEXT PRIMARY KEY,           -- 'memory' | 'messages:user' | 'messages:assistant'
    n_docs    BIGINT NOT NULL DEFAULT 0,  -- rows with at least one token
    total_len BIGINT NOT NULL DEFAULT 0   -- sum of tsv_len (avg doc length = total_len / n_docs)
);

CREATE TABLE IF NOT EXISTS bm25_terms (
    corpus TEXT NOT NULL,
    term   TEXT NOT NULL,
    df     INT NOT NULL DEFAULT 0,        -- documents of the corpus containing the term
    PRIMARY KEY (corpus, term)
);

CREATE INDEX IF NOT EXISTS idx_memory_tsv ON memory USING gin (tsv);
CREATE INDEX IF NOT EXISTS idx_messages_tsv ON messages USING gin (tsv);
CREATE INDEX IF NOT EXISTS idx_memory_tsv_pending ON memory (id) WHERE tsv IS NULL;
CREATE INDEX IF NOT EXISTS idx_messages_tsv_pending ON messages (id)
    WHERE role IN ('user', 'assistant') AND tsv IS NULL;

-- Text -> words. Harakat (U+064B-U+065F, U+0670) and tatweel (U+0640) are
-- removed so vowelled and unvowelled Arabic match. Used for rows AND queries.
CREATE OR REPLACE FUNCTION syne_tsv(p_text TEXT) RETURNS tsvector
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT to_tsvector('simple'::regconfig,
        regexp_replace(coalesce(p_text, ''), '[\u064B-\u065F\u0670\u0640]', '', 'g'))
$$;

-- Length in tokens (positions; Postgres keeps at most 256 per word).
CREATE OR REPLACE FUNCTION syne_tsv_len(p_tsv tsvector) RETURNS INT
    LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT coalesce(sum(coalesce(array_length(u.positions, 1), 1)), 0)::int
      FROM unnest(p_tsv) AS u
$$;

-- Add (p_sign = 1) or remove (p_sign = -1) one document from the statistics.
-- Words are upserted in sorted order so concurrent writers lock in the same
-- order. Any error is downgraded to a WARNING: a write must never fail
-- because of statistics; syne_bm25_rebuild() repairs drift.
CREATE OR REPLACE FUNCTION syne_bm25_apply(p_corpus TEXT, p_tsv tsvector, p_len INT, p_sign INT) RETURNS void AS $$
BEGIN
    IF p_corpus IS NULL OR p_tsv IS NULL OR coalesce(p_len, 0) = 0 THEN
        RETURN;
    END IF;
    BEGIN
        INSERT INTO bm25_corpus AS c (corpus, n_docs, total_len)
        VALUES (p_corpus, p_sign, p_sign::bigint * p_len)
        ON CONFLICT (corpus) DO UPDATE
            SET n_docs = c.n_docs + EXCLUDED.n_docs,
                total_len = c.total_len + EXCLUDED.total_len;
        INSERT INTO bm25_terms AS t (corpus, term, df)
        SELECT p_corpus, lx, p_sign
          FROM unnest(tsvector_to_array(p_tsv)) AS lx
         ORDER BY lx
        ON CONFLICT (corpus, term) DO UPDATE SET df = t.df + EXCLUDED.df;
        IF p_sign < 0 THEN
            DELETE FROM bm25_terms
             WHERE corpus = p_corpus AND df <= 0
               AND term = ANY (tsvector_to_array(p_tsv));
        END IF;
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'bm25 statistics not updated (%): %', p_corpus, SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION syne_memory_tsv_trg() RETURNS trigger AS $$
BEGIN
    IF TG_OP IN ('UPDATE', 'DELETE') THEN
        PERFORM syne_bm25_apply('memory', OLD.tsv, OLD.tsv_len, -1);
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    BEGIN
        NEW.tsv := syne_tsv(NEW.content);
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'memory % not keyword-indexed: %', NEW.id, SQLERRM;
        NEW.tsv := ''::tsvector;
    END;
    NEW.tsv_len := syne_tsv_len(NEW.tsv);
    PERFORM syne_bm25_apply('memory', NEW.tsv, NEW.tsv_len, 1);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Only user and assistant rows are indexed (history_search level 1 / 2).
CREATE OR REPLACE FUNCTION syne_messages_tsv_trg() RETURNS trigger AS $$
BEGIN
    IF TG_OP IN ('UPDATE', 'DELETE') AND OLD.role IN ('user', 'assistant') THEN
        PERFORM syne_bm25_apply('messages:' || OLD.role, OLD.tsv, OLD.tsv_len, -1);
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    IF NEW.role IN ('user', 'assistant') THEN
        BEGIN
            NEW.tsv := syne_tsv(NEW.content);
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'message % not keyword-indexed: %', NEW.id, SQLERRM;
            NEW.tsv := ''::tsvector;
        END;
        NEW.tsv_len := syne_tsv_len(NEW.tsv);
        PERFORM syne_bm25_apply('messages:' || NEW.role, NEW.tsv, NEW.tsv_len, 1);
    ELSE
        NEW.tsv := NULL;
        NEW.tsv_len := NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger
                    WHERE tgname = 'trg_memory_tsv' AND tgrelid = 'memory'::regclass) THEN
        CREATE TRIGGER trg_memory_tsv
            BEFORE INSERT OR UPDATE OF content, tsv OR DELETE ON memory
            FOR EACH ROW EXECUTE FUNCTION syne_memory_tsv_trg();
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger
                    WHERE tgname = 'trg_messages_tsv' AND tgrelid = 'messages'::regclass) THEN
        CREATE TRIGGER trg_messages_tsv
            BEFORE INSERT OR UPDATE OF content, role, tsv OR DELETE ON messages
            FOR EACH ROW EXECUTE FUNCTION syne_messages_tsv_trg();
    END IF;
END $$;

-- Recompute the statistics from the indexed rows. Locks the two statistics
-- tables, so writes of memory/messages wait until it finishes. Run rarely.
CREATE OR REPLACE FUNCTION syne_bm25_rebuild() RETURNS void AS $$
BEGIN
    LOCK TABLE bm25_corpus, bm25_terms IN EXCLUSIVE MODE;
    DELETE FROM bm25_terms;
    DELETE FROM bm25_corpus;
    INSERT INTO bm25_corpus (corpus, n_docs, total_len)
    SELECT 'memory', count(*), coalesce(sum(tsv_len), 0)
      FROM memory WHERE tsv_len > 0;
    INSERT INTO bm25_corpus (corpus, n_docs, total_len)
    SELECT 'messages:' || role, count(*), coalesce(sum(tsv_len), 0)
      FROM messages WHERE role IN ('user', 'assistant') AND tsv_len > 0
     GROUP BY role;
    INSERT INTO bm25_terms (corpus, term, df)
    SELECT 'memory', lx, count(*)
      FROM memory, unnest(tsvector_to_array(tsv)) AS lx
     WHERE tsv_len > 0
     GROUP BY lx;
    INSERT INTO bm25_terms (corpus, term, df)
    SELECT 'messages:' || role, lx, count(*)
      FROM messages, unnest(tsvector_to_array(tsv)) AS lx
     WHERE role IN ('user', 'assistant') AND tsv_len > 0
     GROUP BY role, lx;
END;
$$ LANGUAGE plpgsql;

-- Background backfill of old rows (syne/memory/bm25_backfill.py).
INSERT INTO config (key, value, description) VALUES
    ('bm25.backfill_enabled', 'true', 'Index old memory and chat rows for BM25 keyword search in the background after start (rows written before migration 51 have no keyword index yet). false = do not run. Checked before every batch, so setting false stops a running backfill.'),
    ('bm25.backfill_batch', '50', 'Rows indexed per transaction by the BM25 background backfill. Clamped 10-2000. Smaller = shorter waits for new messages and memories, which wait on the same BM25 statistics rows (measured: 200 rows = new writes waited up to ~4 s, 50 rows = ~0.2 s).'),
    ('bm25.backfill_pause', '0.5', 'Seconds the BM25 background backfill waits between batches. Clamped 0-60.')
ON CONFLICT (key) DO NOTHING;
""".strip()
