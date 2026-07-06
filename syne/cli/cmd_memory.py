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


@memory.command("delete")
@click.argument("memory_id", type=int)
def memory_delete(memory_id):
    """Delete a single memory by ID."""
    async def _delete():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db

        settings = load_settings()
        pool = await init_db(settings.database_url)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, category, content FROM memory WHERE id = $1",
                memory_id,
            )
            if not row:
                console.print(f"[yellow]Memory #{memory_id} not found.[/yellow]")
                await close_db()
                return

            preview = row["content"][:120] + ("..." if len(row["content"]) > 120 else "")
            console.print(f"\n[bold]Memory #{row['id']}[/bold] [{row['category'] or 'none'}]")
            console.print(f"  {preview}\n")

            deleted = await conn.execute("DELETE FROM memory WHERE id = $1", memory_id)
            console.print(f"[green]✓ Deleted memory #{memory_id}[/green] ({deleted})")

        await close_db()

    asyncio.run(_delete())


@memory.command("prune")
@click.argument("pattern")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def memory_prune(pattern, yes):
    """Bulk-delete memories whose content matches a case-insensitive SQL LIKE pattern.

    Use '%' as a wildcard. Examples:
      syne memory prune '%consent%bug%'
      syne memory prune '%nihil%'
      syne memory prune '%output tidak sampai%'

    Intended for cleaning up session-hallucination artifacts (bug narratives,
    complaints, "putus"/"kejegal" entries) that pollute future recall. This
    matches the DO-NOT-STORE rules the evaluator now enforces going forward.
    """
    async def _prune():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db

        settings = load_settings()
        pool = await init_db(settings.database_url)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, category, content FROM memory WHERE content ILIKE $1 ORDER BY id",
                pattern,
            )
            if not rows:
                console.print(f"[yellow]No memories match pattern: {pattern!r}[/yellow]")
                await close_db()
                return

            t = Table(title=f"Match: {pattern!r} ({len(rows)} memories)")
            t.add_column("ID", justify="right")
            t.add_column("Category")
            t.add_column("Content")
            for r in rows:
                preview = r["content"][:100] + ("..." if len(r["content"]) > 100 else "")
                t.add_row(str(r["id"]), r["category"] or "", preview)
            console.print(t)

            if not yes:
                confirm = click.confirm(
                    f"\nDelete these {len(rows)} memories? This cannot be undone.",
                    default=False,
                )
                if not confirm:
                    console.print("[yellow]Aborted — no memories deleted.[/yellow]")
                    await close_db()
                    return

            ids = [r["id"] for r in rows]
            await conn.execute("DELETE FROM memory WHERE id = ANY($1::bigint[])", ids)
            console.print(f"[green]✓ Deleted {len(ids)} memories.[/green]")

        await close_db()

    asyncio.run(_prune())
