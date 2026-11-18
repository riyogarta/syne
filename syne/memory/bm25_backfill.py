"""Background BM25 backfill (Riyo, 18 Nov 2026) - KG-RAG stage 1, step 2.

Migration 51 adds the keyword index (tsv / tsv_len) and its triggers, but
rows written before it keep tsv NULL. This task indexes them in small
batches after Syne has started, so startup is never held up.

How a row is indexed: ``UPDATE ... SET tsv = NULL`` on a pending row fires
the BEFORE UPDATE OF tsv trigger, which computes tsv / tsv_len from the
content and adds the row to the BM25 statistics. The row-to-words rule
therefore lives in one place (syne_tsv() in syne/db/bm25_ddl.py) for new
rows and old rows alike. A row whose content cannot be indexed gets an
empty tsv, so it never stays pending.

Batches are short transactions with a pause between them, because every
indexed row also updates the shared statistics rows (bm25_terms) that a
new memory or message waits on. Rows locked by another writer are skipped
(SKIP LOCKED) and picked up by a later batch.

Config (DB):
  bm25.backfill_enabled  true   - false = do not run
  bm25.backfill_batch    50     - rows per transaction (clamped 10-2000)
  bm25.backfill_pause    0.5    - seconds between batches (clamped 0-60)

The task ends when nothing is pending; every start checks again (an index
lookup, cheap when there is nothing to do).
"""

import asyncio
import logging
import time

from .. import maintenance as _maint
from ..db.connection import get_connection
from ..db.models import get_config

logger = logging.getLogger("syne.memory.bm25_backfill")

# table -> WHERE clause selecting rows still waiting for their keyword index.
# Each matches a partial index from bm25_ddl (idx_*_tsv_pending).
PENDING = {
    "memory": "tsv IS NULL",
    "messages": "role IN ('user', 'assistant') AND tsv IS NULL",
}

_MAX_FAILURES = 5
_FAILURE_BACKOFF = 60.0
_LOG_EVERY = 50  # batches


def _clamp(value, lo, hi, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


async def _settings() -> tuple[bool, int, float]:
    enabled = await get_config("bm25.backfill_enabled", True)
    batch = int(_clamp(await get_config("bm25.backfill_batch", 50), 10, 2000, 50))
    pause = _clamp(await get_config("bm25.backfill_pause", 0.5), 0, 60, 0.5)
    return enabled is not False, batch, pause


async def pending_count(table: str) -> int:
    """Rows of ``table`` that have no keyword index yet."""
    async with get_connection() as conn:
        return int(await conn.fetchval(
            f"SELECT count(*) FROM {table} WHERE {PENDING[table]}"))


async def backfill_batch(table: str, batch: int) -> int:
    """Index up to ``batch`` pending rows of ``table``. Returns rows indexed."""
    where = PENDING[table]
    sql = (
        f"UPDATE {table} SET tsv = NULL WHERE id IN ("
        f"SELECT id FROM {table} WHERE {where} "
        f"ORDER BY id LIMIT $1 FOR UPDATE SKIP LOCKED)"
    )
    async with get_connection() as conn:
        status = await conn.execute(sql, batch)
    # asyncpg returns the command tag, e.g. "UPDATE 200".
    try:
        return int(str(status).split()[-1])
    except (ValueError, IndexError):
        return 0


async def run_backfill() -> dict:
    """Index every pending row, table by table. Returns rows indexed per table.

    Never raises: errors are logged, retried with a backoff, and after
    _MAX_FAILURES consecutive failures the task stops until the next start.
    """
    done = {t: 0 for t in PENDING}
    try:
        enabled, batch, pause = await _settings()
        if not enabled:
            logger.info("BM25 backfill disabled (bm25.backfill_enabled = false)")
            return done
        remaining = {t: await pending_count(t) for t in PENDING}
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"BM25 backfill not started: {e}")
        return done

    if not any(remaining.values()):
        logger.debug("BM25 backfill: nothing pending")
        return done
    logger.info(
        "BM25 backfill started: "
        + ", ".join(f"{t} {n}" for t, n in remaining.items())
        + f" rows pending (batch {batch}, pause {pause}s)"
    )
    started = time.monotonic()

    for table in PENDING:
        if not remaining[table]:
            continue
        failures = 0
        batches = 0
        while True:
            if _maint.active():
                # /restore is replacing the database: wait, then re-check.
                await asyncio.sleep(5)
                continue
            try:
                enabled, batch, pause = await _settings()
                if not enabled:
                    logger.info("BM25 backfill stopped: bm25.backfill_enabled = false")
                    return done
                n = await backfill_batch(table, batch)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                failures += 1
                logger.warning(
                    f"BM25 backfill batch failed on {table} "
                    f"({failures}/{_MAX_FAILURES}): {e}")
                if failures >= _MAX_FAILURES:
                    logger.error(
                        f"BM25 backfill gave up on {table} after {failures} "
                        "failures; it retries at the next start")
                    break
                await asyncio.sleep(_FAILURE_BACKOFF)
                continue
            failures = 0
            if n == 0:
                break
            done[table] += n
            batches += 1
            if batches % _LOG_EVERY == 0:
                logger.info(f"BM25 backfill {table}: {done[table]}/{remaining[table]} rows")
            if pause:
                await asyncio.sleep(pause)

    left = {}
    try:
        left = {t: await pending_count(t) for t in PENDING}
    except Exception:
        pass
    logger.info(
        f"BM25 backfill finished in {time.monotonic() - started:.0f}s: "
        + ", ".join(f"{t} {n} indexed" for t, n in done.items())
        + (" | still pending: " + ", ".join(f"{t} {n}" for t, n in left.items())
           if any(left.values()) else "")
    )
    return done
