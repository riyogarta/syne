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
