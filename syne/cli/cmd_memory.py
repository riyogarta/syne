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


@memory.command("reembed-history")
@click.option("--batch", "-b", default=50, help="Rows to embed per batch (default 50)")
@click.option("--limit", "-l", default=None, type=int, help="Cap total rows processed (default: no cap — process until done)")
@click.option("--sleep", "-s", default=0.0, type=float, help="Seconds to sleep between batches (throttle if Ollama shared)")
def memory_reembed_history(batch, limit, sleep):
    """Backfill embeddings for user messages that don't have one yet.

    Only user-role rows are embedded (assistant/tool/system are anchors' context,
    not anchors themselves). Rows already embedded are skipped. Resumable: safe
    to Ctrl-C and re-run — the pending index picks up where you stopped.

    After the first batch succeeds, HNSW index is created automatically
    (ensure_messages_hnsw_index()). Rebuilds are idempotent.
    """
    async def _run():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.models import get_config
        from syne.llm.drivers import create_hybrid_provider, get_model_from_list

        settings = load_settings()
        pool = await init_db(settings.database_url)

        # Build a provider so we can embed. Same path as agent boot.
        models = await get_config("provider.models", None)
        active_key = await get_config("provider.active_model", None)
        if not (models and active_key):
            console.print("[red]No active chat model configured — cannot embed.[/red]")
            await close_db()
            return
        entry = get_model_from_list(models, active_key)
        if not entry:
            console.print(f"[red]Active model '{active_key}' not in registry.[/red]")
            await close_db()
            return
        provider = await create_hybrid_provider(entry)
        if provider is None:
            console.print("[red]Failed to build hybrid provider.[/red]")
            await close_db()
            return

        max_chars = int(await get_config("history_search.max_content_chars", 4000))

        # Count pending so the progress bar is meaningful.
        async with pool.acquire() as conn:
            total_pending = await conn.fetchval(
                "SELECT COUNT(*) FROM messages "
                "WHERE role = 'user' AND embedding IS NULL AND content <> ''"
            )
        if not total_pending:
            console.print("[green]Nothing to backfill — every user message already has an embedding.[/green]")
            await close_db()
            return

        target = min(int(total_pending), int(limit)) if limit else int(total_pending)
        console.print(
            f"[cyan]Backfill target: {target} user messages "
            f"(batch={batch}, sleep={sleep}s between batches).[/cyan]"
        )

        done = 0
        failed = 0
        with console.status(f"Embedding batch...") as status:
            while done < target:
                remaining = target - done
                take = min(int(batch), remaining)

                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT id, content FROM messages "
                        "WHERE role = 'user' AND embedding IS NULL AND content <> '' "
                        "ORDER BY id ASC LIMIT $1",
                        take,
                    )
                if not rows:
                    break

                # Embed serially — Ollama semaphore keeps parallelism sane
                # inside the driver already, and small batch sizes make
                # concurrent DB writes noisy for tiny wins.
                import asyncio as _asyncio
                for r in rows:
                    try:
                        body = r["content"]
                        if len(body) > max_chars:
                            body = body[:max_chars]
                        resp = await provider.embed(body)
                        vec = getattr(resp, "vector", None)
                        if not vec:
                            failed += 1
                            continue
                        async with pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE messages SET embedding = $1::vector WHERE id = $2",
                                str(vec), r["id"],
                            )
                        done += 1
                    except Exception as e:
                        failed += 1
                        console.print(f"[yellow]row id={r['id']} failed: {e}[/yellow]")

                status.update(f"Embedded {done}/{target} (failed {failed})")

                if sleep > 0:
                    await _asyncio.sleep(sleep)

        # Try to build HNSW after backfill so subsequent searches are fast.
        # ensure_messages_hnsw_index() no-ops if no embeddings exist yet, so
        # this is safe even when everything failed.
        try:
            async with pool.acquire() as conn:
                await conn.execute("SELECT ensure_messages_hnsw_index()")
            console.print("[green]✓ HNSW index built / refreshed.[/green]")
        except Exception as e:
            console.print(f"[yellow]HNSW build skipped: {e}[/yellow]")

        console.print(
            f"[green]✓ Backfill done. Embedded: {done}. Failed: {failed}.[/green]"
        )
        await close_db()

    asyncio.run(_run())
