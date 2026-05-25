"""Parity test: schema.sql (fresh install) ≡ schema.sql @ last release + migrations.

Catches dual-path drift — the bug class where a column added to schema.sql
without a corresponding migration step (or vice versa) silently produces
different DB structures for fresh installs vs upgraded installs.

Strategy:
    db1: apply current schema.sql to a fresh DB → represents fresh install
    db2: apply schema.sql FROM LAST RELEASE TAG, then run all migrations
         whose target_version > LAST_RELEASE_VERSION → represents upgrade path
    Compare information_schema dumps. Must be identical (modulo ignorable
    diffs like default object names).

Requires:
    - Docker available (testcontainers spins up an ephemeral Postgres)
    - Internet access on first run (pulls pgvector/postgres image)

If testcontainers or docker is unavailable, the test SKIPs gracefully —
it's a pre-merge guardrail, not a blocker for local dev without docker.

To bump LAST_RELEASE_TAG: after each release, update this constant to
the tag of the rilis (e.g. "v1.13.6") and the corresponding
LAST_RELEASE_SCHEMA_VERSION. Migrations newer than that version are
the only ones exercised in db2.
"""

from __future__ import annotations

import asyncio
import os
import subprocess

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Update these after each release tag
# ─────────────────────────────────────────────────────────────────────────
LAST_RELEASE_TAG = "main"  # Initially "main" — bump to actual tag once
                           # versioned migrations are added.
LAST_RELEASE_SCHEMA_VERSION = 0  # Bump in lockstep with MIGRATIONS in
                                  # syne/db/migrations.py


# Note: do NOT set a module-level pytestmark.asyncio — that would mark the
# sync metadata test as async too and produce a warning. Async tests below
# are async-def, so pytest-asyncio's auto mode picks them up automatically.


def _has_docker() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_testcontainers() -> bool:
    try:
        import testcontainers.postgres  # noqa: F401
        return True
    except ImportError:
        return False


def _git_show_schema_at_tag(tag: str) -> str | None:
    """Return contents of syne/db/schema.sql at the given git tag, or None."""
    try:
        r = subprocess.run(
            ["git", "show", f"{tag}:syne/db/schema.sql"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        return r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


async def _dump_information_schema(conn) -> dict:
    """Dump structural metadata for parity comparison.

    Returns dict with sorted lists per category for deterministic compare.
    """
    out: dict = {}

    out["columns"] = [
        dict(r) for r in await conn.fetch("""
            SELECT table_schema, table_name, column_name, data_type,
                   is_nullable, column_default, udt_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
    ]
    out["constraints"] = [
        dict(r) for r in await conn.fetch("""
            SELECT tc.table_name, tc.constraint_name, tc.constraint_type,
                   tc.is_deferrable, tc.initially_deferred
            FROM information_schema.table_constraints tc
            WHERE tc.table_schema = 'public'
              -- Exclude auto-generated constraint names that vary between
              -- fresh install vs upgrade. The constraint TYPES still match.
              AND tc.constraint_name NOT LIKE '%_pkey'
              AND tc.constraint_name NOT LIKE '%_check'
            ORDER BY tc.table_name, tc.constraint_name
        """)
    ]
    out["indexes"] = [
        dict(r) for r in await conn.fetch("""
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname
        """)
    ]
    return out


async def _apply_schema(conn, schema_sql: str) -> None:
    """Run a schema.sql blob, splitting on DO $$ / CREATE FUNCTION boundaries."""
    from syne.cli.helpers import _split_sql_statements
    for stmt in _split_sql_statements(schema_sql):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("--"):
            continue
        await conn.execute(stmt)


@pytest.mark.skipif(
    not (_has_docker() and _has_testcontainers()),
    reason="requires docker + testcontainers (pip install testcontainers[postgres])",
)
async def test_schema_parity_fresh_vs_upgraded():
    """Fresh-install schema must equal upgrade-path schema."""
    from testcontainers.postgres import PostgresContainer
    import asyncpg

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_schema_path = os.path.join(repo_root, "syne", "db", "schema.sql")
    with open(current_schema_path) as f:
        current_schema = f.read()

    # Anchor for upgrade path
    last_schema = _git_show_schema_at_tag(LAST_RELEASE_TAG) or current_schema
    # If LAST_RELEASE_TAG = "main" (first setup) the anchor equals current —
    # the test then just exercises the "no migrations needed" case.

    # Use pgvector-enabled image for tests
    image = os.environ.get("SYNE_TEST_PG_IMAGE", "pgvector/pgvector:pg16")

    with PostgresContainer(image) as pg1, PostgresContainer(image) as pg2:
        # db1: fresh install
        conn1 = await asyncpg.connect(pg1.get_connection_url(driver="asyncpg"))
        try:
            await _apply_schema(conn1, current_schema)
            from syne.db.migrations import run_migrations
            await run_migrations(conn1)
            dump1 = await _dump_information_schema(conn1)
        finally:
            await conn1.close()

        # db2: last release + migrations since
        conn2 = await asyncpg.connect(pg2.get_connection_url(driver="asyncpg"))
        try:
            await _apply_schema(conn2, last_schema)
            # Seed schema_version = LAST_RELEASE_SCHEMA_VERSION so only newer
            # migrations run
            await conn2.execute(
                """
                INSERT INTO config (key, value, description)
                VALUES ('schema_version', $1::text::jsonb,
                        'Schema migration version')
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                str(LAST_RELEASE_SCHEMA_VERSION),
            )
            await run_migrations(conn2)
            dump2 = await _dump_information_schema(conn2)
        finally:
            await conn2.close()

    assert dump1 == dump2, _diff_message(dump1, dump2)


def _diff_message(d1: dict, d2: dict) -> str:
    """Produce a readable diff for assert failure."""
    lines = ["Schema parity FAILURE — fresh-install ≠ upgrade-path"]
    for key in set(d1) | set(d2):
        a = d1.get(key, [])
        b = d2.get(key, [])
        if a == b:
            continue
        lines.append(f"\n[{key}] diff:")
        # Show only-in-fresh and only-in-upgrade
        af = {str(r) for r in a}
        bf = {str(r) for r in b}
        only_fresh = af - bf
        only_upg = bf - af
        for x in sorted(only_fresh)[:10]:
            lines.append(f"  + (only in fresh): {x}")
        for x in sorted(only_upg)[:10]:
            lines.append(f"  - (only in upgrade): {x}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Smoke test that doesn't need Docker — verifies the runner is import-clean
# and CURRENT_SCHEMA_VERSION is consistent with MIGRATIONS.
# ─────────────────────────────────────────────────────────────────────────


def test_migrations_metadata_consistent():
    """CURRENT_SCHEMA_VERSION must equal last MIGRATIONS entry's target."""
    from syne.db.migrations import MIGRATIONS, CURRENT_SCHEMA_VERSION

    if not MIGRATIONS:
        assert CURRENT_SCHEMA_VERSION == 0
        return

    targets = [t for t, _, _ in MIGRATIONS]
    # Sequential, strictly increasing
    assert targets == sorted(set(targets)), "MIGRATIONS targets must be strictly increasing unique ints"
    assert CURRENT_SCHEMA_VERSION == targets[-1], (
        "CURRENT_SCHEMA_VERSION must equal last MIGRATIONS target"
    )
    # Validate mode values
    for target, _, mode in MIGRATIONS:
        assert mode in ("transactional", "non_transactional"), (
            f"Migration v{target} has invalid mode: {mode!r}"
        )
