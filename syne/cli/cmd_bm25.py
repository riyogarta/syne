"""BM25 keyword index commands.

Usage:
    syne bm25 rebuild
"""

import asyncio
import time

import click

from . import cli
from .shared import console


@cli.group()
def bm25():
    """BM25 keyword index."""


@bm25.command("rebuild")
def bm25_rebuild():
    """Recompute the BM25 statistics from the indexed rows (repairs drift)."""
    async def _run():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db

        pool = await init_db(load_settings().database_url)
        try:
            async with pool.acquire() as conn:
                q = "SELECT corpus, n_docs, total_len FROM bm25_corpus ORDER BY corpus"
                before = {r["corpus"]: r for r in await conn.fetch(q)}
                t0 = time.monotonic()
                await conn.execute("SELECT syne_bm25_rebuild()")
                after = await conn.fetch(q)
                terms = await conn.fetchval("SELECT count(*) FROM bm25_terms")
            console.print(f"[green]Rebuilt in {time.monotonic() - t0:.1f}s[/green] ({terms} terms)")
            for r in after:
                b = before.get(r["corpus"])
                was = f"{b['n_docs']} docs / {b['total_len']} words" if b else "missing"
                console.print(f"  {r['corpus']:<20} {r['n_docs']} docs / {r['total_len']} words"
                              f"  [dim](was {was})[/dim]")
        finally:
            await close_db()

    asyncio.run(_run())
