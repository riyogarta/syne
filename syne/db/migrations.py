"""Versioned schema migrations.

Complements the idempotent `schema.sql` (fresh-install path) with a
single integer `schema_version` in the `config` table plus an ordered
list of migration steps for data transformations that idempotent rerun
can't safely express: backfill, rename, type change, ADD NOT NULL to
populated tables, etc.

Architecture:
    schema.sql       — fresh install builds full current schema
    MIGRATIONS list  — ordered (target_version, migration_fn, mode) tuples
    runner           — at startup, advisory-locked, runs migrations whose
                       target_version > current; bumps version after each

Modes:
    "transactional"     — migration_fn runs inside conn.transaction().
                          DDL + small data ops. Crash → atomic rollback.
                          version only bumped on commit.
    "non_transactional" — for CREATE INDEX CONCURRENTLY, large batched
                          backfills, etc. MUST be self-resumable (use
                          marker columns or WHERE-clause filters that
                          skip already-migrated rows). Caller responsible
                          for atomicity.

CRITICAL invariant — dual-path parity:
    schema.sql (current) MUST produce a DB equivalent to:
        schema.sql (release N) + all MIGRATIONS where target_version > N
    A CI parity test enforces this. When you add a migration here, you
    MUST also update schema.sql to reflect the new state.

Adding a migration:
    1. Append a tuple to MIGRATIONS with the next sequential version
    2. Update schema.sql to include the new structure (fresh-install path)
    3. Run the parity test to confirm both paths converge
    4. Bump CURRENT_SCHEMA_VERSION below
"""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

logger = logging.getLogger("syne.db.migrations")


# Advisory lock key — stable constant. Multiple Syne processes share this
# so concurrent boots don't race on the migration runner. Pick any int.
ADVISORY_LOCK_KEY = 0x53594E_45  # 'SYNE' in hex


# ─────────────────────────────────────────────────────────────────────────
# Migration functions
# ─────────────────────────────────────────────────────────────────────────
# Each takes an asyncpg connection. Transactional ones can assume they
# run inside a BEGIN; non-transactional ones manage their own atomicity.


async def _m1_messages_status(conn) -> None:
    """Add messages.status column + index for soft-archive compaction.

    Compaction no longer DELETEs old messages — it marks them
    status='compacted' (retained in DB for semantic search, reversible).
    Existing rows default to 'active'. Idempotent.
    """
    await conn.execute(
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active'"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_status "
        "ON messages (session_id, status, created_at)"
    )


async def _m2_drop_legacy_compaction_config(conn) -> None:
    """Remove dead compaction config keys from live DB.

    `session.compaction_threshold` (legacy char-based, superseded by
    `compaction.trigger_percent`) and `session.max_messages` are no
    longer read by any code path. Fresh installs already omit them
    (schema.sql seed cleaned in v1.16.9); this migration purges them
    from existing installs so live == seed. Idempotent.
    """
    await conn.execute(
        "DELETE FROM config WHERE key IN "
        "('session.compaction_threshold', 'session.max_messages')"
    )

async def _m3_set_keep_recent_40(conn) -> None:
    """Normalize session.compaction_keep_recent to 40 on live DB.

    Older installs seeded this at 200 (> history_limit, and effectively
    overridden by the dynamic `_keep` computation in conversation.py, so
    behaviorally inert). Fresh installs already seed 40 (schema.sql,
    v1.16.9). This migration aligns existing installs so live == seed.
    Only touches the row if it still holds the legacy 200. Idempotent.
    """
    await conn.execute(
        "UPDATE config SET value = '40'::jsonb, updated_at = NOW() "
        "WHERE key = 'session.compaction_keep_recent' "
        "AND value = '200'::jsonb"
    )


async def _m4_seed_db_as_files_rule(conn) -> None:
    """Add the generic DB_AS_FILES meta-rule to existing installs.

    Fresh installs get it via schema.sql (v1.16.12). This backfills
    installs created before that. Generic — no user-specific content.
    Idempotent via ON CONFLICT (code) DO NOTHING.
    """
    await conn.execute(
        "INSERT INTO rules (code, name, description, severity) VALUES ("
        "'DB_AS_FILES', 'Database Is Source Of Truth', "
        "'Syne stores all operational configuration in the DATABASE, not in files. "
        "The tables replace the .md files other agents rely on: `rules` = rules.md, "
        "`identity` = persona.md, `soul` = behavior/style, `memory` = notes/user.md, "
        "`config` = settings. Before acting, READ the relevant table as the source of "
        "truth \u2014 do not rely on memory. When unsure about a rule, identity, persona, "
        "or configuration, query the table first.', "
        "'hard') ON CONFLICT (code) DO NOTHING"
    )

async def _m5_seed_compaction_overlap(conn) -> None:
    """Add session.compaction_overlap_percent to existing installs.

    Fresh installs get it via schema.sql (v1.16.14). This backfills older
    installs. Default 15 = keep 15% of the summarized batch as a raw
    verbatim bridge for smooth summary→recent transition. Char-budget
    guarded in compact_session. Idempotent via ON CONFLICT DO NOTHING.
    """
    await conn.execute(
        "INSERT INTO config (key, value, description) VALUES ("
        "'session.compaction_overlap_percent', '15', "
        "'Compaction overlap: keep this % of the summarized batch ALSO as raw "
        "messages (a verbatim bridge between summary and recent tail) for smooth "
        "transition. 0 = disabled. Char-budget guarded.') "
        "ON CONFLICT (key) DO NOTHING"
    )

async def _m6_seed_subagent_config(conn) -> None:
    """Seed subagents.* tuning keys into config for existing installs.

    Fresh installs get these via schema.sql. This backfills older installs
    so sub-agent tuning is fully DB-driven (no hidden hardcoded fallbacks).
    max_rounds default is 100 (raised from the old code fallback of 30) to
    let heavier tasks run longer. Idempotent via ON CONFLICT DO NOTHING.
    """
    await conn.execute(
        "INSERT INTO config (key, value, description) VALUES "
        "('subagents.enabled', 'true', "
        "'Enable/disable sub-agent spawning. true = allowed.'), "
        "('subagents.max_concurrent', '2', "
        "'Max concurrent sub-agents running at once.'), "
        "('subagents.max_rounds', '100', "
        "'Max tool-call rounds per sub-agent before forced stop (prevents runaway loops).'), "
        "('subagents.round_delay', '2.0', "
        "'Delay in seconds between sub-agent tool-call rounds (throttle).') "
        "ON CONFLICT (key) DO NOTHING"
    )


# ─────────────────────────────────────────────────────────────────────────
# Ordered migration list
# ─────────────────────────────────────────────────────────────────────────
# Format: (target_version, migration_fn, mode)
#   mode ∈ {"transactional", "non_transactional"}

async def _m7_seed_cap002_rule(conn) -> None:
    """Add the CAP002 'Trust System Evidence' meta-rule to existing installs.

    Fresh installs get it via schema.sql (v1.16.20). This backfills
    installs created before that. Generic — no user-specific content.
    Idempotent via ON CONFLICT (code) DO NOTHING.
    """
    await conn.execute(
        "INSERT INTO rules (code, name, description, severity) VALUES ("
        "'CAP002', 'Trust System Evidence', "
        "'System-generated context is authoritative evidence of what actually "
        "happened. Tool-result footers, run IDs, execution reports, and tool-call "
        "counts are real — do NOT claim you faked, hallucinated, or did not "
        "actually do an action when system evidence shows it ran. A tool being "
        "absent from the current turn tool list does NOT mean you lack the "
        "capability — tools may be context-gated per turn. Before concluding "
        "anything negative about your own actions, verify against the system "
        "context. When in doubt, trust the footer over your gut.', "
        "'hard') ON CONFLICT (code) DO NOTHING"
    )


async def _m8_seed_sub001_rule(conn) -> None:
    """Add SUB001 'Delegate Multi-Step Work To Sub-Agents' rule to existing installs.

    Fresh installs get it via schema.sql. Sub-agent spawning is now Molt's own
    judgment call (tool is always-on, not keyword-gated). This rule tells Molt
    WHEN to spawn. Generic — no user-specific content. Idempotent.
    """
    await conn.execute(
        "INSERT INTO rules (code, name, description, severity) VALUES ("
        "'SUB001', 'Delegate Multi-Step Work To Sub-Agents', "
        "'Spawning a sub-agent is YOUR judgment call from the task itself — never "
        "wait for the user to type keywords like \"delegate\" or \"sub-agent\". "
        "SPAWN when ANY applies: (1) the task needs 2+ dependent tool rounds where "
        "one step''s output determines the next; (2) 3+ independent subtasks that "
        "can run in parallel; (3) long-running work that would block the chat; "
        "(4) heavy research/file reading that would pollute the main context. "
        "DO NOT spawn for 1-2 direct tool calls (just run exec/db_query yourself), "
        "pure conversation, or tasks needing mid-way user decisions (a sub-agent "
        "runs blind, cannot ask). Each sub-agent has its own round budget "
        "(subagents.max_rounds).', "
        "'soft') ON CONFLICT (code) DO NOTHING"
    )


async def _m9_add_memory_tainted(conn) -> None:
    """Add memory.tainted BOOLEAN for indirect-prompt-injection defense.

    A memory is "tainted" if any of the content it was derived from came
    from an external, untrusted source (fetched web page, uploaded doc,
    image analysis output, etc). At recall time, if ANY tainted memory
    surfaces in a session, the session is marked tainted too and the
    owner-DM exec bypass is disabled until the owner confirms.

    Backfill: legacy rows default to FALSE (treated as clean). This is
    permissive but matches historical behavior; new memories stored after
    the app has upgraded will carry the correct taint bit from their
    session context.
    """
    await conn.execute(
        "ALTER TABLE memory ADD COLUMN IF NOT EXISTS tainted BOOLEAN DEFAULT false"
    )


async def _m10_add_sessions_tainted(conn) -> None:
    """Add sessions.tainted + sessions.taint_reason for persistent taint.

    Taint (indirect-prompt-injection defense) previously lived only in the
    in-memory Conversation object and was lost on restart/evict. Persist it
    on the session row so it survives restarts. Monotonic: only ever set to
    TRUE by automatic taint paths; the ONLY reset path is the owner-only
    /untaint slash command. Legacy rows default to FALSE (clean). Idempotent.
    """
    await conn.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tainted BOOLEAN DEFAULT false"
    )
    await conn.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS taint_reason TEXT"
    )


async def _m11_add_kg_entities_tainted(conn) -> None:
    """Add kg_entities.tainted for taint propagation through the graph.

    Entities extracted from a tainted memory must carry the taint marker so
    that when they surface in a clean session's Knowledge Graph block, the
    session is re-tainted (prevents laundering external content via the KG).
    Legacy rows default to FALSE. Idempotent.
    """
    await conn.execute(
        "ALTER TABLE kg_entities ADD COLUMN IF NOT EXISTS tainted BOOLEAN DEFAULT false"
    )


async def _m12_add_kg_relations_tainted(conn) -> None:
    """Add kg_relations.tainted for taint propagation through the graph.

    Relations derived from a tainted memory carry the taint marker so that
    recalling them into a clean session re-taints that session. Legacy rows
    default to FALSE. Idempotent.
    """
    await conn.execute(
        "ALTER TABLE kg_relations ADD COLUMN IF NOT EXISTS tainted BOOLEAN DEFAULT false"
    )


async def _m14_drop_tainted_columns(conn) -> None:
    """Retire the session/memory-taint mechanism (superseded by consent gate).

    The taint mechanism marked sessions/memories/graph rows that had touched
    external/untrusted content, then blocked the owner-DM exec bypass while
    the flag was set. The consent gate (v1.17.x) replaces this with a
    per-action ya/yes prompt tied to the tool's op=x classification — a
    finer-grained defense that doesn't require session-wide state.

    Drop the four columns added by _m9/_m10/_m11/_m12. IF EXISTS makes this
    idempotent and safe on installs that never had the columns (schema.sql
    of the same release omits them from the fresh-install path).
    """
    for table, col in (
        ("memory", "tainted"),
        ("sessions", "tainted"),
        ("kg_entities", "tainted"),
        ("kg_relations", "tainted"),
    ):
        await conn.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {col}")


async def _m13_seed_consent_config(conn) -> None:
    """Seed the security.consent_* keys so upgraded installs get them visibly.

    Runtime always falls back to DEFAULT_CONSENT_ENABLED=True when the key
    is absent, so the functional default doesn't change. This migration
    matters because:
      - `syne config` (v1.17.20) only shows rows present in the table, so
        without seeds the operator can't discover the flag exists or read
        its current value.
      - Fresh installs (schema.sql) seed these; the migration keeps
        upgraded installs in parity with fresh ones (dual-path invariant).
    ON CONFLICT (key) DO NOTHING preserves any operator-customized value.
    """
    await conn.execute("""
        INSERT INTO config (key, value, description) VALUES
            ('security.consent_enabled', 'true',
             'Master switch for the consent gate on op=x tool calls. When true, destructive tools (exec, file_write, memory_delete, update_*, etc.) prompt "balas ya" before running. Set false to disable the whole gate.'),
            ('security.consent_ttl_seconds', '600',
             'How long an approved consent stays cached (seconds). Default 600 = 10 min. The next identical call within this window skips the prompt.'),
            ('security.consent_mode', '"sliding"',
             'TTL mode: "sliding" refreshes the clock on every reuse (active work keeps its grant alive), "fixed" expires from the original grant time regardless of reuse.')
        ON CONFLICT (key) DO NOTHING
    """)


async def _m15_shell_allowlist_tables(conn) -> None:
    """Create shell_allowlist + shell_allowlist_candidates for the shell_guard
    security parser (design: memory 54510).

    shell_allowlist: binaries the OWNER has explicitly approved for shell
    execution. This is the ONLY thing that opens the default-deny gate at
    runtime (the hardcoded DEFAULT_ALLOWLIST in shell_guard.py is the release
    floor). Rows are added via /allowlist add (owner-only Telegram command).
    `hits` counts real uses so proven entries can later be promoted into the
    hardcoded floor in a future release.

    shell_allowlist_candidates: unknown binaries that got HARD_DENY'd, parked
    here for the owner to review and promote. Deduplicated by binary; each
    sighting bumps seen_count + last_seen_at and keeps a sample command for
    context so the owner isn't approving blind.

    Both tables live only in migrations (schema.sql of a later release will
    carry them for fresh installs — dual-path invariant). IF NOT EXISTS keeps
    this idempotent.
    """
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS shell_allowlist (
            bin_name    TEXT PRIMARY KEY,
            added_by    TEXT,
            added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            note        TEXT,
            hits        BIGINT NOT NULL DEFAULT 0
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS shell_allowlist_candidates (
            bin_name       TEXT PRIMARY KEY,
            first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            seen_count     BIGINT NOT NULL DEFAULT 1,
            sample_command TEXT,
            context        TEXT,
            status         TEXT NOT NULL DEFAULT 'pending'
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_shell_cand_status "
        "ON shell_allowlist_candidates (status, last_seen_at DESC)"
    )


async def _m16_shell_denylist_table(conn) -> None:
    """Create shell_denylist for the shell_guard runtime denylist (memory 54511
    'update with the times': hard-deny a newly-discovered dangerous command
    without waiting for a release).

    Entries are checked BEFORE the allowlist, so a denylist row can never be
    overridden by an allowlist entry. Two kinds:
      * kind='binary'  — exact binary-name match (safe; e.g. deny 'nmap').
      * kind='pattern' — substring match on the normalized command (powerful;
                         e.g. deny '--no-preserve-root'). Owner-authored.
    """
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS shell_denylist (
            entry     TEXT PRIMARY KEY,
            kind      TEXT NOT NULL DEFAULT 'binary',
            added_by  TEXT,
            added_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            note      TEXT
        )
    """)


async def _m17_decay_v2_config(conn) -> None:
    """Decay v2 config-only migration (memory 54522 blueprint, approved Riyo
    9 Jul 2026). NO column alters — existing memory rows are left to decay
    naturally (option a).

    Converges existing installs to schema.sql v2 seeds:
      * initial_recall_count 5 -> 1  (new memories start low)
      * promotion_threshold 10 -> 1000 (auto-promote dormant, high threshold)
      * seed max_records = 50 (cap for LRU/LFU eviction in run_decay)

    decay_interval is kept (it now paces the cap-check cadence). decay_amount
    is dead under v2 but harmless, so it is left untouched.
    """
    # Lower starting recall for new memories (only if still at old default).
    await conn.execute("""
        UPDATE config SET value = '1'
        WHERE key = 'memory.initial_recall_count' AND value = '5'
    """)
    # Raise promotion threshold so auto-promote is effectively dormant.
    await conn.execute("""
        UPDATE config SET value = '1000'
        WHERE key = 'memory.promotion_threshold' AND value = '10'
    """)
    # Seed the eviction cap if it does not exist yet.
    await conn.execute("""
        INSERT INTO config (key, value, description)
        VALUES ('memory.max_records', '50',
                'Decay v2 cap: max non-permanent memories; LRU/LFU eviction above this')
        ON CONFLICT (key) DO NOTHING
    """)


async def _m18_decay_v2_initial_count_force(conn) -> None:
    """Force memory.initial_recall_count to 1 on ALL installs (decay v2 fix).

    v17 only updated the value when it equalled the assumed old default '5'.
    Installs that had customised it (e.g. prod at 10) were skipped and stayed
    on the pre-v2 starting count. This migration converges unconditionally to
    the JSON number 1, matching the schema.sql seed. Also normalises a
    jsonb-string \"1\" into a jsonb-number 1 for type parity.
    """
    await conn.execute("""
        UPDATE config
        SET value = '1'::jsonb
        WHERE key = 'memory.initial_recall_count'
          AND value IS DISTINCT FROM '1'::jsonb
    """)


async def _m19_decay_v2_cap_1000(conn) -> None:
    """Raise memory.max_records (decay v2 eviction cap) to 1000 on ALL installs.

    Evaluation-window setting (Riyo, 10 Jul 2026): with the cap at 1000 and
    promotion_threshold already at 1000, decay v2 accumulates recall signal
    without evicting or auto-promoting anything. Final values will be tuned
    once enough real recall data exists. Unconditional converge to JSON
    number 1000.
    """
    await conn.execute("""
        UPDATE config
        SET value = '1000'::jsonb
        WHERE key = 'memory.max_records'
          AND value IS DISTINCT FROM '1000'::jsonb
    """)


async def _m20_history_search_embedding(conn) -> None:
    """Semantic search across all-time message history — schema + function seeds.

    Design (Riyo, 13 Jul 2026): embed ONLY user messages as anchors. Assistant
    and tool rows are the derived context you scroll around each anchor. This
    keeps the semantic space clean (one canonical anchor per topic), shrinks
    storage 3-5x versus embedding every row, and mirrors how a human recalls a
    past conversation: 'when did I ASK about X?' → jump to that user message →
    read the surrounding turns.

    This migration is idempotent — safe to re-run:
      * ADD COLUMN IF NOT EXISTS: nullable embedding vector on messages
      * CREATE OR REPLACE FUNCTION ensure_messages_hnsw_index(): mirrors the
        memory HNSW pattern; called AFTER enough embedded rows exist so the
        column can be typed to the correct dimension and indexed.
      * Config seeds (ON CONFLICT DO NOTHING preserves any operator override).

    HNSW build is deferred to first backfill / CLI trigger — pgvector cannot
    build the index on an empty column, and doing it inline here would fail
    on fresh installs.
    """
    await conn.execute("""
        ALTER TABLE messages ADD COLUMN IF NOT EXISTS embedding vector
    """)
    # Not-null lookup index — the backfill CLI needs to page fast through
    # rows that still need embedding.
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_embedding_pending
        ON messages (session_id, id)
        WHERE role = 'user' AND embedding IS NULL
    """)
    # HNSW builder — dimension picked up from the first embedded row.
    await conn.execute("""
        CREATE OR REPLACE FUNCTION ensure_messages_hnsw_index() RETURNS void AS $fn$
        DECLARE
            dim INT;
        BEGIN
            SELECT vector_dims(embedding) INTO dim
              FROM messages WHERE embedding IS NOT NULL LIMIT 1;
            IF dim IS NULL THEN RETURN; END IF;
            EXECUTE format('ALTER TABLE messages ALTER COLUMN embedding TYPE vector(%s)', dim);
            DROP INDEX IF EXISTS idx_messages_embedding_hnsw;
            EXECUTE 'CREATE INDEX idx_messages_embedding_hnsw '
                    'ON messages USING hnsw (embedding vector_cosine_ops) '
                    'WITH (m = 24, ef_construction = 200)';
        END;
        $fn$ LANGUAGE plpgsql
    """)
    # Config seeds. INSERT ON CONFLICT DO NOTHING so an operator who tuned
    # the value already isn't clobbered.
    await conn.execute("""
        INSERT INTO config (key, value, description)
        VALUES
          ('messages.embedding_enabled', 'true'::jsonb,
           'When true, save_message fires a background embed for role=user rows '
           'so history_search can retrieve them semantically. Safe to toggle off '
           'in emergencies — content is always kept, only the embedding stops.'),
          ('history_search.default_limit', '10'::jsonb,
           'Default number of anchor previews returned per history_search call.'),
          ('history_search.context_before', '2'::jsonb,
           'Default turns to include BEFORE each anchor in history_expand.'),
          ('history_search.context_after', '5'::jsonb,
           'Default turns AFTER each anchor. Larger than before because you want '
           'to see the assistant response + follow-up.'),
          ('history_search.max_content_chars', '4000'::jsonb,
           'Truncate user message content to this many chars before embedding. '
           'Long messages average their semantic signal into noise otherwise.')
        ON CONFLICT (key) DO NOTHING
    """)


async def _m21_smart_hnsw_guard(conn) -> None:
    """Make ensure_*_hnsw_index() skip the expensive DROP+CREATE when the
    index already exists and the column is already typed to the right
    dimension. Without this, the auto-build hook (helpers._ensure_vector_index)
    rebuilds the HNSW index on EVERY `syne update` — a ~90s lock on a large
    embedding table. Idempotent CREATE OR REPLACE; safe to re-run.

    Guard mechanism: pgvector stores the vector dimension directly in the
    column's atttypmod, so `atttypmod = dim` means the column is already the
    correct vector(dim) type. Combined with an index-existence check, the
    function returns early when nothing needs rebuilding.
    """
    await conn.execute("""CREATE OR REPLACE FUNCTION ensure_memory_hnsw_index() RETURNS void AS $$
DECLARE
    dim INT;
    cur_typmod INT;
    idx_exists BOOL;
BEGIN
    SELECT vector_dims(embedding) INTO dim FROM memory WHERE embedding IS NOT NULL LIMIT 1;
    IF dim IS NULL THEN RETURN; END IF;
    -- Smart guard: pgvector stores dimension directly in atttypmod. If the
    -- column is already typed vector(dim) AND the HNSW index exists, skip the
    -- expensive DROP+CREATE (rebuilding a large index can take ~90s).
    SELECT a.atttypmod INTO cur_typmod
      FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
     WHERE c.relname = 'memory' AND a.attname = 'embedding';
    SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'idx_memory_embedding_hnsw') INTO idx_exists;
    IF cur_typmod = dim AND idx_exists THEN RETURN; END IF;
    EXECUTE format('ALTER TABLE memory ALTER COLUMN embedding TYPE vector(%s)', dim);
    DROP INDEX IF EXISTS idx_memory_embedding_hnsw;
    EXECUTE 'CREATE INDEX idx_memory_embedding_hnsw '
            'ON memory USING hnsw (embedding vector_cosine_ops) '
            'WITH (m = 24, ef_construction = 200)';
END;
$$ LANGUAGE plpgsql;""")
    await conn.execute("""CREATE OR REPLACE FUNCTION ensure_messages_hnsw_index() RETURNS void AS $$
DECLARE
    dim INT;
    cur_typmod INT;
    idx_exists BOOL;
BEGIN
    SELECT vector_dims(embedding) INTO dim FROM messages WHERE embedding IS NOT NULL LIMIT 1;
    IF dim IS NULL THEN RETURN; END IF;
    -- Smart guard: pgvector stores dimension directly in atttypmod. If the
    -- column is already typed vector(dim) AND the HNSW index exists, skip the
    -- expensive DROP+CREATE (rebuilding a large index can take ~90s).
    SELECT a.atttypmod INTO cur_typmod
      FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
     WHERE c.relname = 'messages' AND a.attname = 'embedding';
    SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'idx_messages_embedding_hnsw') INTO idx_exists;
    IF cur_typmod = dim AND idx_exists THEN RETURN; END IF;
    EXECUTE format('ALTER TABLE messages ALTER COLUMN embedding TYPE vector(%s)', dim);
    DROP INDEX IF EXISTS idx_messages_embedding_hnsw;
    EXECUTE 'CREATE INDEX idx_messages_embedding_hnsw '
            'ON messages USING hnsw (embedding vector_cosine_ops) '
            'WITH (m = 24, ef_construction = 200)';
END;
$$ LANGUAGE plpgsql;""")


async def _m22_fk_index_and_autovacuum(conn) -> None:
    """Persist two production DB tweaks into the migration path so BOTH fresh
    installs and existing DBs get them (previously applied ad-hoc to the live
    DB only, so they were not install defaults).

    1. FK index on kg_relations.source_memory_id — the only foreign key that
       lacked a backing index. Without it, DELETE/UPDATE on memory seq-scans
       the entire kg_relations table (100k+ rows) for the referential check.
    2. Aggressive per-table autovacuum for high-churn tables (memory,
       kg_relations, kg_entities). The global scale_factor=0.2 default is too
       lax — dead tuples + stale planner stats accumulated (memory once
       reported 2 live rows when it actually held 31k, breaking query plans).

    All statements idempotent. CREATE INDEX (not CONCURRENTLY) because this
    migration runs inside a transaction; IF NOT EXISTS makes it a no-op where
    the index is already present.
    """
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_kg_relations_source_memory "
        "ON kg_relations (source_memory_id)"
    )
    await conn.execute(
        "ALTER TABLE memory SET ("
        "autovacuum_vacuum_scale_factor = 0.05, "
        "autovacuum_analyze_scale_factor = 0.02, "
        "autovacuum_vacuum_threshold = 50)"
    )
    await conn.execute(
        "ALTER TABLE kg_relations SET ("
        "autovacuum_vacuum_scale_factor = 0.05, "
        "autovacuum_analyze_scale_factor = 0.02)"
    )
    await conn.execute(
        "ALTER TABLE kg_entities SET ("
        "autovacuum_vacuum_scale_factor = 0.05, "
        "autovacuum_analyze_scale_factor = 0.02)"
    )


async def _m23_fetch_url_config(conn) -> None:
    """web_fetch retired -> fetch_url promoted to core tool (Riyo, 14 Jul 2026).

    Converges existing DBs to the new config surface:
      * seed fetch_url.timeout = 15 (core tool reads this key; default 15)
      * DROP the now-dead web_fetch.timeout key (the web_fetch tool no longer
        exists, so this config variable is unused — remove it per owner request)
    Fresh installs get fetch_url.timeout directly from schema.sql instead.
    """
    await conn.execute("""
        INSERT INTO config (key, value, description)
        VALUES ('fetch_url.timeout', '15',
                'Fetch URL (core tool) timeout in seconds')
        ON CONFLICT (key) DO NOTHING
    """)
    await conn.execute("DELETE FROM config WHERE key = 'web_fetch.timeout'")


async def _m24_rule_checker_timeout(conn) -> None:
    """Seed security.rule_checker_timeout (Riyo, 16 Aug 2026).

    Pairs with the /checker 'off' option. Both checker drivers now run under
    a per-call ceiling so a hung Ollama or a stalled provider API cannot hold
    a user's reply; on timeout the checker fails open with a warning.

    Seed-only (ON CONFLICT DO NOTHING) — this key is NEW, so there is no
    prior value to converge and no assumed-old-default to guess wrong about.
    Fresh installs get the same key from schema.sql. Owner retunes it at
    runtime via /checker timeout <sec>, which is why this only ever seeds.
    """
    await conn.execute("""
        INSERT INTO config (key, value, description)
        VALUES ('security.rule_checker_timeout', '120',
                'Max seconds the rule checker may spend judging one draft '
                'response, applied to both drivers (Ollama evaluator and '
                'main provider). Read at check time and clamped to 5-120. '
                'On timeout the checker returns ERROR and fails open — the '
                'reply is sent with a warning tag rather than held. '
                'Default 120; owner tunes it via /checker timeout <sec>.')
        ON CONFLICT (key) DO NOTHING
    """)


async def _m25_drop_dead_config(conn) -> None:
    """Delete config keys that no code path reads (Riyo, 4 Nov 2026).

    Audited by literal match across all 171 repo files, not just .py - config
    in Syne is also read from inside schema.sql by PL/pgSQL, which an earlier
    .py-only pass got wrong.

      memory.decay_amount   - decay v2 is cap-based. run_decay() reads
                              decay_interval, max_records and
                              promotion_threshold; the per-cycle subtraction
                              this key configured no longer exists.
      time.format.date      - conversation.py loads exactly one template,
      time.format.time        time.format.full (_TIME_CFG_KEYS). The three
      time.format.day         granular ones were never consulted.

    Unconditional DELETE: no value to converge, nothing to guess wrong about.
    Fresh installs no longer seed them. Idempotent.
    """
    await conn.execute("""
        DELETE FROM config WHERE key IN (
            'memory.decay_amount',
            'time.format.date',
            'time.format.time',
            'time.format.day'
        )
    """)


async def _m26_trusted_group_ids_description(conn) -> None:
    """Refresh history.trusted_group_ids description (Riyo, 4 Nov 2026).

    Two problems. On this install the description is NULL, so `syne config
    list` says nothing about the only privacy control the recall feature has.
    And the text that fresh installs got describes the superseded rule - it
    speaks of groups "treated as private like a DM" and of rows being limited
    to "that same group", which was the destination-tier model, not the
    family-only one that replaced it.

    Description-only: value is never touched, so an owner who already listed
    their family group keeps it. schema.sql seeds with ON CONFLICT DO NOTHING
    and therefore cannot refresh prose on an existing row - hence this.
    """
    await conn.execute(
        "UPDATE config SET description = $1 WHERE key = 'history.trusted_group_ids'",
        "JSON array of Telegram chat ids that count as family for history recall. Recall requires BOTH ends to be family: the sender must be owner or family, and every chat involved must be on this list (a DM always qualifies as its own chat). Inside that circle recall flows in every direction. A group left off this list neither receives recalled rows nor contributes its own. Empty = DM-only, the fail-closed default. Telegram only: platform ids are not unified across channels.",
    )


async def _m27_dm_policy_description(conn) -> None:
    """Refresh telegram.dm_policy description (Riyo, 4 Nov 2026).

    Live DBs carry prose from an older release: "DM policy: open (accept all)
    or registered (only known users)". "registered" is not a value any code
    path recognises, and the text omits "approval" - which is both the
    default and, since the key was wired up in 1.20.97, the branch every
    unrecognised value now falls through to. So `syne config list` was
    offering an option that does not exist while hiding the real one.

    Description-only: the value is never touched, so an owner running "open"
    on purpose stays open. schema.sql seeds with ON CONFLICT DO NOTHING and
    cannot refresh prose on a row that already exists - hence this migration.
    Text is kept byte-identical to the schema.sql seed so fresh and migrated
    installs read the same.
    """
    await conn.execute(
        "UPDATE config SET description = $1 WHERE key = 'telegram.dm_policy'",
        'DM policy for a Telegram user who has never messaged before. "open" admits them immediately; ANY other value, including a typo or unparseable one, falls through to "approval", where the owner must approve via the inline buttons before the user can chat. Read at the DM gate on every first contact, so a change takes effect without a restart. Default "approval".',
    )


async def _m28_seed_discord_config(conn) -> None:
    """Seed discord.* channel config (Riyo, 4 Nov 2026).

    Discord joins Telegram as a core channel, living in
    syne/communication/discord.py rather than being bolted on as an ability.

    All five keys are NEW, so there is no prior value to converge and no
    assumed-old-default to guess wrong about - seed-only via ON CONFLICT DO
    NOTHING, which also makes this idempotent. Fresh installs get identical
    rows from schema.sql; the prose below is copied from that same seed so a
    migrated DB and a fresh one read the same text in `syne config list`.

    Defaults are deliberately fail-closed. discord.enabled is false, so
    applying this migration starts nothing and changes nothing about the
    Telegram or WhatsApp paths - the owner opts in by storing a bot token and
    flipping the switch. dm_policy and guild_policy default to approval and
    allowlist, mirroring the Telegram gates they are modelled on.
    """
    rows = [
        ('discord.enabled', 'false',
         'Master switch for the Discord channel. false = the channel is never started even if a bot token is stored. Read once at boot, so changing it needs a restart. Default false so adding the dependency alone changes nothing.'),
        ('discord.dm_policy', '"approval"',
         'DM policy for a Discord user who has never messaged before. "open" admits them immediately; ANY other value, including a typo or unparseable one, falls through to "approval", where the owner must approve via the inline buttons on Telegram before the user can chat. Read at the DM gate on every first contact, so a change takes effect without a restart. Default "approval". Note: Discord only delivers a DM to the bot if the sender shares a server with it.'),
        ('discord.guild_policy', '"allowlist"',
         'Guild (server) policy: allowlist = the bot stays silent in a server until the owner approves it, open = any server it is added to is accepted. Approval is per server, not per channel, so a new channel inside an approved server needs no second approval. Default "allowlist".'),
        ('discord.require_mention', 'true',
         'Default require_mention for newly approved Discord servers. Kept true because without the privileged Message Content intent Discord only sends message text when the bot is mentioned, replied to, or DMed — so true also matches what the bot can actually read.'),
        ('discord.members', '[]',
         'JSON array of per-entry Discord settings managed by /discordmembers on Telegram. Each entry: {"id", "name", "kind": "dm"|"guild", "model", "blocked"}. model null = use the globally active model. Mirrors the shape whatsapp uses for its own allowlist.'),
    ]
    for key, value, description in rows:
        await conn.execute(
            """
            INSERT INTO config (key, value, description)
            VALUES ($1, $2::text::jsonb, $3)
            ON CONFLICT (key) DO NOTHING
            """,
            key, value, description,
        )


async def _m29_whatsapp_channel_config(conn) -> None:
    """Move WhatsApp from ability to core channel (Riyo, 5 Nov 2026).

    WhatsApp now lives in syne/communication/whatsapp.py next to Telegram and
    Discord, driving neonize (embedded whatsmeow) in-process. The old
    syne/abilities/whatsapp.py drove the `wacli` binary as a subprocess and is
    deleted in the same change, so every wacli-era leftover goes with it.

    Four groups, in order:

    1. SEED the three keys the new channel actually reads. Seed-only via
       ON CONFLICT DO NOTHING, so an existing whatsapp.allowed_chat_jids with
       real members and an existing whatsapp.trigger_name keep their values —
       there is nothing to converge and nothing to guess wrong about. Prose is
       copied verbatim from schema.sql so a migrated DB and a fresh one read
       the same text in `syne config list`.

       whatsapp.enabled is NEW and seeds to false, mirroring discord.enabled
       (migration 28). That is deliberate: applying this migration must not
       silently start a channel. The owner opts in by flipping the switch,
       which also means WhatsApp stays down across the deploy until they do.

    2. DROP keys no code path reads any more, audited by literal grep across
       the repo before writing this:
         whatsapp.wacli_path     - the wacli binary is gone; neonize needs no
                                   external executable.
         whatsapp.owner_only     - zero readers even BEFORE this change. The
                                   old bridge gated on allowed_chat_jids and
                                   the trigger word, never on this key.
         whatsapp.require_trigger - same: zero readers. On the new channel the
                                   trigger is unconditional in groups AND DMs
                                   (a reply returns as from_me, so an
                                   untriggered bot would answer itself), so a
                                   switch to disable it must not even exist.

    3. DELETE the stale `abilities` row. On this install that row was
       source=self_created, NOT bundled -- and the loader's dynamic pass
       selects exactly that set (WHERE source != 'bundled'), so with the file
       deleted the loader would try to import syne.abilities.whatsapp on every
       boot. It fails soft rather than fatally: _resolve_module_to_filepath
       returns None for a missing file, validation is skipped, and the ability
       is passed over with a WARNING. The leftover row therefore costs a log
       warning per boot plus a phantom entry in `syne abilities` -- not a
       crash. The DELETE matches on name, so it clears the row whatever its
       source column happens to say.

    Idempotent: seeds skip on conflict, deletes are unconditional.
    """
    rows = [
        ('whatsapp.enabled', 'false',
         'Master switch for the WhatsApp channel. false = the channel is never started even if a session is already paired. Read once at boot, so changing it needs a restart. Default false so adding the dependency alone changes nothing.'),
        ('whatsapp.trigger_name', 'null',
         'Trigger name the bot answers to on WhatsApp. null = fall back to telegram.bot_trigger_name, then identity.name. The trigger is MANDATORY on WhatsApp in BOTH groups and DMs, including from the owner - there is no require_mention escape hatch, because a WhatsApp reply arrives back as from_me and an untriggered bot would answer itself.'),
        ('whatsapp.allowed_chat_jids', '[]',
         'JSON array of per-entry WhatsApp settings managed by /wamembers on Telegram. Each entry: {"jid", "name", "model", "blocked"}. model null = use the globally active model. Access level is NOT stored here: a DM takes it from users.access_level and a group member from groups.settings[\'members\'], absent meaning "public". Mirrors the shape discord.members uses.'),
    ]
    for key, value, description in rows:
        await conn.execute(
            """
            INSERT INTO config (key, value, description)
            VALUES ($1, $2::text::jsonb, $3)
            ON CONFLICT (key) DO NOTHING
            """,
            key, value, description,
        )

    await conn.execute("""
        DELETE FROM config WHERE key IN (
            'whatsapp.wacli_path',
            'whatsapp.owner_only',
            'whatsapp.require_trigger'
        )
    """)

    await conn.execute("DELETE FROM abilities WHERE name = 'whatsapp'")



async def _m30_whatsapp_reply_prefix(conn) -> None:
    """Seed whatsapp.reply_prefix (Riyo, 6 Nov 2026).

    Syne runs on the owner's own WhatsApp number, so its replies and the
    owner's own messages come from the same account. The prefix marks every
    message Syne sends. Seed-only (ON CONFLICT DO NOTHING) so a value the
    owner already set is kept; description copied verbatim from schema.sql.
    """
    await conn.execute(
        """
        INSERT INTO config (key, value, description)
        VALUES ($1, $2::text::jsonb, $3)
        ON CONFLICT (key) DO NOTHING
        """,
        "whatsapp.reply_prefix",
        '"\U0001F40D"',
        'Marker put on its own line, followed by a blank line, at the top of every WhatsApp message Syne sends, so a reply from the bot is told apart from the owner typing on the same number. Default "🐍". Empty string = no marker. Read on every send, so a change takes effect without a restart.',
    )


async def _m31_history_expand_window(conn) -> None:
    """history_expand window defaults (Riyo, 7 Nov 2026).

    context_before / context_after now count conversation messages only and
    default to 10 / 10 (was 2 / 5 counting every row incl. tool results);
    tool results are no longer returned. Value is only moved when it is
    still the old default, so an operator who tuned it is kept.
    Descriptions are refreshed to match schema.sql.
    """
    await conn.execute(
        "UPDATE config SET value = '10'::jsonb "
        "WHERE key = 'history_search.context_before' AND value = '2'::jsonb"
    )
    await conn.execute(
        "UPDATE config SET value = '10'::jsonb "
        "WHERE key = 'history_search.context_after' AND value = '5'::jsonb"
    )
    for key, desc in (
        ("history_search.context_before", 'Default number of conversation messages (user + assistant text) to include BEFORE each anchor in history_expand. Tool results are not included. Re-evaluate against real use (Riyo, 7 Nov 2026).'),
        ("history_search.context_after", 'Default number of conversation messages (user + assistant text) to include AFTER each anchor in history_expand. Tool results are not included.'),
    ):
        await conn.execute(
            "UPDATE config SET description = $2 WHERE key = $1", key, desc
        )



async def _m32_drop_compaction_config(conn) -> None:
    """Delete compaction config keys (Riyo, 7 Nov 2026).

    Compaction was removed entirely: syne/compaction.py, run_compact() and
    the /compact command are gone, so nothing reads these keys anymore.
    Legacy compaction_summary rows and messages.status are left untouched.
    """
    await conn.execute(
        "DELETE FROM config WHERE key IN ("
        "'compaction.trigger_percent', "
        "'session.compaction_keep_recent', "
        "'session.compaction_overlap_percent')"
    )


async def _m33_history_expand_adaptive(conn) -> None:
    """Seed history_expand dedup + adaptive radius config (Riyo, 10 Nov 2026).

    Both on by default; switched off only for experiment conditions.
    """
    await conn.execute(
        """
        INSERT INTO config (key, value, description) VALUES
    ('history_search.dedup', 'true', 'history_expand merges overlapping windows of several anchors into one block and flags turns already in the live context window instead of repeating them. On by default; switch off only for experiments.'),
    ('history_search.adaptive_radius', 'true', 'history_expand treats context_before/context_after as the maximum and stops the window at a topic shift (cosine between blocks of user-message embeddings below history_search.adaptive_threshold) or a time gap longer than history_search.adaptive_gap_minutes. The anchor exchange is always kept. On by default; off = fixed radius (experiment condition K3).'),
    ('history_search.adaptive_threshold', '0.5', 'Adaptive radius: cosine floor between neighbouring blocks of user-message embeddings. Below it the window stops (topic shift). Tune on development data.'),
    ('history_search.adaptive_gap_minutes', '120', 'Adaptive radius: a gap longer than this (minutes) between two exchanges stops the window. 0 disables the time rule.'),
    ('history_search.adaptive_block', '2', 'Adaptive radius: number of user messages averaged per block when comparing topics (TextTiling-style).')
        ON CONFLICT (key) DO NOTHING
        """
    )


async def _m34_ratification_config(conn) -> None:
    """Seed user ratification switches (Riyo, 10 Nov 2026). On by default."""
    await conn.execute(
        """
        INSERT INTO config (key, value, description) VALUES
    ('history_search.ratification', 'true', 'User ratification: when a user message approves the previous assistant answer (text) or reacts to it with an approving emoji, that user row is embedded together with the answer so history_search can find facts only the assistant wrote. 👎 marks the answer disputed; a reaction taken back revokes it. Master switch; on by default, off only for experiments.'),
    ('history_search.ratification_text', 'true', 'Ratification from text replies (lexicon, ambiguous ok/ya decided by memory.evaluator_model). Needs history_search.ratification.'),
    ('history_search.ratification_reaction', 'true', 'Ratification from emoji reactions (Telegram). Needs history_search.ratification.')
        ON CONFLICT (key) DO NOTHING
        """
    )


async def _m35_memory_budget_config(conn) -> None:
    """Seed per-turn memory budget config (Riyo, 12 Nov 2026).

    Ships in log mode with every budget 0 (no limit): turn_stats collects
    real X/R/M sizes first, budgets are set from that data, then enforce.
    """
    await conn.execute(
        """
        INSERT INTO config (key, value, description) VALUES
    ('memory_budget.mode', '"log"', 'Per-turn memory budget mode. log = measure and record violations in metadata.turn_stats only, change nothing. enforce = also fit X/R/M to their budgets before every model call (X drops oldest messages in one chunk to 70% of budget, R drops weakest recall items, M skips a history result whole with a notice; never truncates). Start in log, set budgets from turn_stats data, then switch to enforce.'),
    ('memory_budget.x_tokens', '0', 'Budget X (estimated tokens): session window, the non-system messages present when the turn starts. 0 = no limit.'),
    ('memory_budget.r_tokens', '0', 'Budget R (estimated tokens): injected recall, i.e. memories + cross-session history + knowledge graph. 0 = no limit.'),
    ('memory_budget.m_tokens', '0', 'Budget M (estimated tokens): history_search + history_expand results produced within one turn. 0 = no limit.')
        ON CONFLICT (key) DO NOTHING
        """
    )


async def _m36_auto_recall_as_history_tools(conn) -> None:
    """history.auto_recall = history_search + history_expand (Riyo, 12 Nov 2026).

    Its own limit / similarity / length knobs are gone: it reads the
    history_search.* settings exactly like the tools. Idempotent.
    """
    await conn.execute("""
        DELETE FROM config WHERE key IN (
            'history.auto_recall_limit',
            'history.auto_recall_min_similarity',
            'history.auto_recall_min_chars'
        )
    """)
    await conn.execute(
        "UPDATE config SET description = $1 WHERE key = 'history.auto_recall'",
        "Automatic history_search + history_expand before every answer, for owner/family only: searches the sender's other chats (current session excluded) with the same scope and the same history_search.* settings as the tools, and injects the expanded blocks into context.",
    )


async def _m37_remove_auto_recall(conn) -> None:
    """history.auto_recall removed (Riyo, 12 Nov 2026).

    Past chats are reached only through history_search + history_expand,
    called when the topic is not in the context window. Idempotent.
    """
    await conn.execute("DELETE FROM config WHERE key = 'history.auto_recall'")
    await conn.execute(
        "UPDATE config SET description = $1 WHERE key = 'memory_budget.r_tokens'",
        'Budget R (estimated tokens): injected recall, i.e. memories + knowledge graph. 0 = no limit.',
    )


MIGRATIONS: list[tuple[int, Callable[..., Awaitable[None]], str]] = [
    (1, _m1_messages_status, "transactional"),
    (2, _m2_drop_legacy_compaction_config, "transactional"),
    (3, _m3_set_keep_recent_40, "transactional"),
    (4, _m4_seed_db_as_files_rule, "transactional"),
    (5, _m5_seed_compaction_overlap, "transactional"),
    (6, _m6_seed_subagent_config, "transactional"),
    (7, _m7_seed_cap002_rule, "transactional"),
    (8, _m8_seed_sub001_rule, "transactional"),
    (9, _m9_add_memory_tainted, "transactional"),
    (10, _m10_add_sessions_tainted, "transactional"),
    (11, _m11_add_kg_entities_tainted, "transactional"),
    (12, _m12_add_kg_relations_tainted, "transactional"),
    (13, _m13_seed_consent_config, "transactional"),
    (14, _m14_drop_tainted_columns, "transactional"),
    (15, _m15_shell_allowlist_tables, "transactional"),
    (16, _m16_shell_denylist_table, "transactional"),
    (17, _m17_decay_v2_config, "transactional"),
    (18, _m18_decay_v2_initial_count_force, "transactional"),
    (19, _m19_decay_v2_cap_1000, "transactional"),
    (20, _m20_history_search_embedding, "transactional"),
    (21, _m21_smart_hnsw_guard, "transactional"),
    (22, _m22_fk_index_and_autovacuum, "transactional"),
    (23, _m23_fetch_url_config, "transactional"),
    (24, _m24_rule_checker_timeout, "transactional"),
    (25, _m25_drop_dead_config, "transactional"),
    (26, _m26_trusted_group_ids_description, "transactional"),
    (27, _m27_dm_policy_description, "transactional"),
    (28, _m28_seed_discord_config, "transactional"),
    (29, _m29_whatsapp_channel_config, "transactional"),
    (30, _m30_whatsapp_reply_prefix, "transactional"),
    (31, _m31_history_expand_window, "transactional"),
    (32, _m32_drop_compaction_config, "transactional"),
    (33, _m33_history_expand_adaptive, "transactional"),
    (34, _m34_ratification_config, "transactional"),
    (35, _m35_memory_budget_config, "transactional"),
    (36, _m36_auto_recall_as_history_tools, "transactional"),
    (37, _m37_remove_auto_recall, "transactional"),
]


# Current schema version. Equal to the last entry in MIGRATIONS (or 0
# if no migrations exist yet). Fresh installs are seeded to this value
# so they skip the entire MIGRATIONS list — they already match the
# current state via schema.sql.
CURRENT_SCHEMA_VERSION = MIGRATIONS[-1][0] if MIGRATIONS else 0


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────


async def run_migrations(conn) -> dict:
    """Run pending migrations against the live database.

    Args:
        conn: asyncpg connection (single, not pool — we hold the advisory
              lock for the duration of the run).

    Returns:
        dict with 'applied' (list of versions) and 'final_version' (int).

    Behavior:
        - Acquires session-scoped advisory lock so concurrent boots wait.
        - Reads `schema_version` from config table (default 0).
        - For each (target, fn, mode) where target > current:
            * "transactional" → run fn + version bump in one transaction
            * "non_transactional" → run fn directly, bump version after
        - Releases lock at end.

    On a fresh install where `schema_version` doesn't exist yet, we
    seed it to CURRENT_SCHEMA_VERSION (skipping all migrations) because
    schema.sql already produced the latest state. This is detected by
    checking if the row was just inserted vs read.
    """
    # Acquire lock first — any other process trying to migrate waits here.
    await conn.execute("SELECT pg_advisory_lock($1)", ADVISORY_LOCK_KEY)
    applied: list[int] = []
    try:
        # Read or initialize schema_version
        current = await _get_or_init_schema_version(conn)

        if not MIGRATIONS:
            logger.debug(f"No migrations defined. Current version: {current}.")
            return {"applied": [], "final_version": current}

        for target_version, migration_fn, mode in MIGRATIONS:
            if target_version <= current:
                continue

            logger.info(f"Running migration → v{target_version} (mode={mode})")

            if mode == "transactional":
                async with conn.transaction():
                    await migration_fn(conn)
                    await conn.execute(
                        "UPDATE config SET value = $1::text::jsonb, updated_at = NOW() "
                        "WHERE key = 'schema_version'",
                        str(target_version),
                    )
            elif mode == "non_transactional":
                # Migration manages its own atomicity. We only bump version
                # after it returns successfully. If it crashes mid-way, the
                # next boot re-runs it — it MUST be idempotent/resumable.
                await migration_fn(conn)
                await conn.execute(
                    "UPDATE config SET value = $1::text::jsonb, updated_at = NOW() "
                    "WHERE key = 'schema_version'",
                    str(target_version),
                )
            else:
                raise ValueError(f"Unknown migration mode: {mode!r}")

            applied.append(target_version)
            current = target_version
            logger.info(f"Migration to v{target_version} complete.")

        return {"applied": applied, "final_version": current}
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", ADVISORY_LOCK_KEY)


async def _get_or_init_schema_version(conn) -> int:
    """Read schema_version from config; seed it on fresh installs.

    A fresh install just ran schema.sql which already produced the latest
    state. So if no schema_version row exists, seed it to CURRENT — never
    re-run all historical migrations on a fresh DB.
    """
    row = await conn.fetchrow(
        "SELECT value FROM config WHERE key = 'schema_version'"
    )
    if row is None:
        # Fresh install (or pre-versioning DB). Seed to current.
        # Use INSERT ... ON CONFLICT in case another process beat us
        # between the SELECT and INSERT (rare, but advisory lock should
        # prevent it — defensive anyway).
        await conn.execute(
            """
            INSERT INTO config (key, value, description)
            VALUES ('schema_version', $1::text::jsonb,
                    'Schema migration version — bumped by syne/db/migrations.py')
            ON CONFLICT (key) DO NOTHING
            """,
            str(CURRENT_SCHEMA_VERSION),
        )
        logger.info(f"Seeded schema_version = {CURRENT_SCHEMA_VERSION} (fresh install)")
        return CURRENT_SCHEMA_VERSION

    # Parse — value is jsonb, asyncpg returns str
    import json
    raw = row["value"]
    try:
        return int(json.loads(raw)) if isinstance(raw, str) else int(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning(f"Unparseable schema_version: {raw!r}, treating as 0")
        return 0
