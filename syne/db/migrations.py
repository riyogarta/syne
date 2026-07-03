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

# ─────────────────────────────────────────────────────────────────────────
# Ordered migration list
# ─────────────────────────────────────────────────────────────────────────
# Format: (target_version, migration_fn, mode)
#   mode ∈ {"transactional", "non_transactional"}

MIGRATIONS: list[tuple[int, Callable[..., Awaitable[None]], str]] = [
    (1, _m1_messages_status, "transactional"),
    (2, _m2_drop_legacy_compaction_config, "transactional"),
    (3, _m3_set_keep_recent_40, "transactional"),
    (4, _m4_seed_db_as_files_rule, "transactional"),
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
