"""Memory management commands."""

import asyncio
import click

from . import cli
from .shared import console, _get_provider_async

from rich.table import Table


@cli.group()
def memory():
    """Memory management commands."""
    pass


@memory.command("stats")
def memory_stats():
    """Show memory statistics."""
    async def _stats():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db

        settings = load_settings()
        pool = await init_db(settings.database_url)

        async with pool.acquire() as conn:
            total = await conn.fetchrow("SELECT COUNT(*) as c FROM memory")
            by_cat = await conn.fetch(
                "SELECT category, COUNT(*) as c FROM memory GROUP BY category ORDER BY c DESC"
            )
            by_source = await conn.fetch(
                "SELECT source, COUNT(*) as c FROM memory GROUP BY source ORDER BY c DESC"
            )
            most_accessed = await conn.fetch(
                "SELECT content, access_count, category FROM memory ORDER BY access_count DESC LIMIT 5"
            )

        console.print(f"\n[bold]Total memories: {total['c']}[/bold]\n")

        if by_cat:
            t = Table(title="By Category")
            t.add_column("Category")
            t.add_column("Count", justify="right")
            for r in by_cat:
                t.add_row(r["category"] or "none", str(r["c"]))
            console.print(t)

        if by_source:
            t = Table(title="By Source")
            t.add_column("Source")
            t.add_column("Count", justify="right")
            for r in by_source:
                t.add_row(r["source"] or "none", str(r["c"]))
            console.print(t)

        if most_accessed:
            t = Table(title="Most Accessed")
            t.add_column("Memory")
            t.add_column("Category")
            t.add_column("Access Count", justify="right")
            for r in most_accessed:
                content = r["content"][:60] + "..." if len(r["content"]) > 60 else r["content"]
                t.add_row(content, r["category"] or "", str(r["access_count"]))
            console.print(t)

        await close_db()

    asyncio.run(_stats())


@memory.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Max results")
def memory_search(query, limit):
    """Search memories by semantic similarity."""
    async def _search():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.memory.engine import MemoryEngine

        settings = load_settings()
        await init_db(settings.database_url)

        # Need a provider for embedding
        provider = await _get_provider_async()
        engine = MemoryEngine(provider)

        results = await engine.recall(query, limit=limit)

        if not results:
            console.print("[yellow]No matching memories found.[/yellow]")
        else:
            t = Table(title=f"Search: '{query}'")
            t.add_column("ID", justify="right")
            t.add_column("Category")
            t.add_column("Content")
            t.add_column("Similarity", justify="right")
            for r in results:
                content = r["content"][:80] + "..." if len(r["content"]) > 80 else r["content"]
                t.add_row(str(r["id"]), r["category"] or "", content, f"{r['similarity']:.0%}")
            console.print(t)

        await close_db()

    asyncio.run(_search())


@memory.command("add")
@click.argument("content")
@click.option("--category", "-c", default="fact", help="Category: fact, preference, event, lesson, decision, health")
def memory_add(content, category):
    """Manually add a memory."""
    async def _add():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.memory.engine import MemoryEngine

        settings = load_settings()
        await init_db(settings.database_url)

        provider = await _get_provider_async()
        engine = MemoryEngine(provider)

        mem_id = await engine.store_if_new(content=content, category=category)
        if mem_id:
            console.print(f"[green]✓ Memory stored (id: {mem_id})[/green]")
        else:
            console.print("[yellow]Similar memory already exists. Skipped.[/yellow]")

        await close_db()

    asyncio.run(_add())
