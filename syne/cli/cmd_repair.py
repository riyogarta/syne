"""Repair command — diagnose and fix Syne installation."""

import asyncio
import os
import subprocess
import click

from . import cli
from .shared import console
from .helpers import _ensure_vector_index

from rich.panel import Panel


@cli.command()
@click.option("--fix", is_flag=True, help="Attempt to auto-fix issues found")
def repair(fix):
    """Diagnose and repair Syne installation. Run without --fix to check only."""
    async def _repair():
        from syne.config import load_settings
        issues = []
        fixed = []

        console.print(Panel("[bold]Syne Repair[/bold]", style="blue"))
        console.print()

        settings = load_settings()

        # -- 1. Database connectivity --
        console.print("[bold]1. Database[/bold]")
        try:
            from syne.db.connection import init_db, close_db
            pool = await init_db(settings.database_url)
            async with pool.acquire() as conn:
                ver = await conn.fetchval("SELECT version()")
            console.print(f"   [green]✓ Connected[/green] — {ver[:40]}...")

            # Check pgvector
            async with pool.acquire() as conn:
                ext = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            if ext:
                console.print(f"   [green]✓ pgvector {ext}[/green]")
            else:
                issues.append("pgvector extension not installed")
                console.print("   [red]✗ pgvector not installed[/red]")
                if fix:
                    try:
                        async with pool.acquire() as conn:
                            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        fixed.append("Installed pgvector extension")
                        console.print("   [green]  → Fixed: pgvector installed[/green]")
                    except Exception as e:
                        console.print(f"   [red]  → Failed to install: {e}[/red]")

            # Check tables
            async with pool.acquire() as conn:
                tables = await conn.fetch("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public' ORDER BY tablename
                """)
            table_names = [t["tablename"] for t in tables]
            expected = ["abilities", "capabilities", "config", "identity", "memory",
                       "messages", "rules", "scheduled_tasks", "sessions", "soul",
                       "subagent_runs", "users"]
            missing = [t for t in expected if t not in table_names]

            if missing:
                issues.append(f"Missing tables: {', '.join(missing)}")
                console.print(f"   [red]✗ Missing tables: {', '.join(missing)}[/red]")
                if fix:
                    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "schema.sql")
                    if os.path.exists(schema_path):
                        with open(schema_path) as f:
                            schema = f.read()
                        async with pool.acquire() as conn:
                            await conn.execute(schema)
                        fixed.append("Created missing tables")
                        console.print("   [green]  → Fixed: tables created[/green]")
            else:
                console.print(f"   [green]✓ All {len(expected)} tables present[/green]")

            # Check seed data
            async with pool.acquire() as conn:
                identity_count = await conn.fetchval("SELECT COUNT(*) FROM identity")
                config_count = await conn.fetchval("SELECT COUNT(*) FROM config")
                rules_count = await conn.fetchval("SELECT COUNT(*) FROM rules")
                soul_count = await conn.fetchval("SELECT COUNT(*) FROM soul")

            if identity_count == 0:
                issues.append("Identity table empty (no seed data)")
                console.print("   [yellow]⚠ Identity table empty[/yellow]")
                if fix:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO identity (key, value) VALUES
                            ('name', 'Syne'),
                            ('motto', 'I remember, therefore I am'),
                            ('personality', 'Helpful, direct, resourceful. Has opinions. Not a corporate drone.')
                            ON CONFLICT DO NOTHING
                        """)
                    fixed.append("Seeded identity table")
                    console.print("   [green]  → Fixed: identity seeded[/green]")
            else:
                console.print(f"   [green]✓ Identity: {identity_count} entries[/green]")

            if config_count == 0:
                issues.append("Config table empty")
                console.print("   [yellow]⚠ Config table empty[/yellow]")
                if fix:
                    from syne.db.models import set_config
                    defaults = {
                        "memory.auto_capture": False,
                        "memory.auto_evaluate": True,
                        "memory.recall_limit": 10,
                        "memory.max_importance": 1.0,
                        "provider.primary": {"name": "google", "auth": "oauth"},
                        "provider.chat_model": "gemini-2.5-pro",
                        "provider.embedding_model": "text-embedding-004",
                        "provider.embedding_dimensions": 768,
                        "session.compaction_threshold": 150000,
                        "session.max_messages": 100,
                        "subagents.enabled": True,
                        "subagents.max_concurrent": 2,
                        "subagents.timeout_seconds": 300,
                        "exec.timeout_max": 300,
                        "exec.output_max_chars": 4000,
                    }
                    for k, v in defaults.items():
                        await set_config(k, v)
                    fixed.append("Seeded config defaults")
                    console.print("   [green]  → Fixed: config seeded[/green]")
            else:
                console.print(f"   [green]✓ Config: {config_count} entries[/green]")

            console.print(f"   [dim]Rules: {rules_count} | Soul: {soul_count}[/dim]")

            await close_db()

        except Exception as e:
            issues.append(f"Database connection failed: {e}")
            console.print(f"   [red]✗ Connection failed: {e}[/red]")
            console.print(f"   [dim]URL: {settings.database_url[:30]}...[/dim]")

        # -- 2. Google OAuth --
        console.print("\n[bold]2. Google OAuth[/bold]")
        try:
            from syne.auth.google_oauth import get_credentials
            creds = await get_credentials(auto_refresh=False)
            if creds:
                token = await creds.get_token()
                console.print(f"   [green]✓ Authenticated as {creds.email}[/green]")
                console.print(f"   [dim]Token valid (first 8): {token[:8]}...[/dim]")
            else:
                issues.append("No Google OAuth credentials found")
                console.print("   [yellow]⚠ No credentials found[/yellow]")
                console.print("   [dim]Run 'syne init' to authenticate[/dim]")
        except Exception as e:
            issues.append(f"OAuth token refresh failed: {e}")
            console.print(f"   [red]✗ Token refresh failed: {e}[/red]")
            console.print("   [dim]Run 'syne init' to re-authenticate[/dim]")

        # -- 3. Telegram Bot --
        console.print("\n[bold]3. Telegram Bot[/bold]")
        if settings.telegram_bot_token:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe"
                    )
                    data = resp.json()
                if data.get("ok"):
                    bot = data["result"]
                    console.print(f"   [green]✓ @{bot['username']} ({bot['first_name']})[/green]")
                else:
                    issues.append("Telegram bot token invalid")
                    console.print(f"   [red]✗ Invalid token: {data.get('description')}[/red]")
            except Exception as e:
                issues.append(f"Telegram check failed: {e}")
                console.print(f"   [red]✗ Error: {e}[/red]")
        else:
            console.print("   [yellow]⚠ Not configured (SYNE_TELEGRAM_BOT_TOKEN)[/yellow]")

        # -- 4. Abilities --
        console.print("\n[bold]4. Abilities[/bold]")
        try:
            from syne.abilities import AbilityRegistry
            from syne.abilities.loader import load_all_abilities
            from syne.db.connection import init_db, close_db
            await init_db(settings.database_url)

            registry = AbilityRegistry()
            count = await load_all_abilities(registry)
            console.print(f"   [green]✓ {count} abilities loaded[/green]")

            for ab in registry.list_all():
                has_config = bool(ab.config and any(ab.config.values()))
                ab_status = "[green]ready[/green]" if has_config else "[yellow]needs config[/yellow]"
                enabled = "✓" if ab.enabled else "✗"
                console.print(f"   {enabled} {ab.name} ({ab.source}) — {ab_status}")

            await close_db()
        except Exception as e:
            console.print(f"   [red]✗ Error loading abilities: {e}[/red]")

        # -- 5. Process --
        console.print("\n[bold]5. Process[/bold]")
        result = subprocess.run(["pgrep", "-f", "syne.main"], capture_output=True)
        if result.returncode == 0:
            pids = result.stdout.decode().strip().split("\n")
            console.print(f"   [green]✓ Running (PID: {', '.join(pids)})[/green]")
        else:
            console.print("   [yellow]⚠ Not running[/yellow]")

        # Check systemd service
        result = subprocess.run(
            ["systemctl", "--user", "is-enabled", "syne"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            console.print("   [green]✓ Autostart enabled (systemd)[/green]")
        else:
            console.print("   [dim]Autostart not configured (see 'syne autostart')[/dim]")

        # -- 6. Docker DB --
        console.print("\n[bold]6. Docker (syne-db)[/bold]")
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", "syne-db"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            db_status = result.stdout.strip()
            color = "green" if db_status == "running" else "red"
            console.print(f"   [{color}]{db_status}[/{color}]")
            if db_status != "running" and fix:
                subprocess.run(["docker", "start", "syne-db"])
                fixed.append("Started syne-db container")
                console.print("   [green]  → Fixed: container started[/green]")
        else:
            console.print("   [yellow]⚠ Container not found[/yellow]")

        # -- 7. .env security --
        console.print("\n[bold]7. Security[/bold]")
        syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(syne_dir, ".env")
        if os.path.exists(env_path):
            env_stat = os.stat(env_path)
            env_mode = oct(env_stat.st_mode)[-3:]
            if env_mode == "600":
                console.print(f"   [green]✓ .env permissions: {env_mode} (owner-only)[/green]")
            else:
                issues.append(f".env has insecure permissions: {env_mode} (should be 600)")
                console.print(f"   [red]✗ .env permissions: {env_mode} (should be 600)[/red]")
                if fix:
                    os.chmod(env_path, 0o600)
                    fixed.append("Fixed .env permissions to 600")
                    console.print("   [green]  → Fixed: chmod 600[/green]")
        else:
            console.print("   [yellow]⚠ .env not found[/yellow]")

        # Check .gitignore includes .env
        gitignore_path = os.path.join(syne_dir, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path) as f:
                gitignore = f.read()
            if ".env" in gitignore:
                console.print("   [green]✓ .env in .gitignore[/green]")
            else:
                issues.append(".env not in .gitignore — risk of committing secrets")
                console.print("   [red]✗ .env not in .gitignore[/red]")
                if fix:
                    with open(gitignore_path, "a") as f:
                        f.write("\n.env\n")
                    fixed.append("Added .env to .gitignore")
                    console.print("   [green]  → Fixed: added to .gitignore[/green]")

        # -- 8. Vector Index --
        console.print("\n[bold]8. Vector Index[/bold]")
        try:
            from syne.db.connection import init_db, close_db
            await init_db(settings.database_url)
            from syne.db.connection import get_connection
            async with get_connection() as conn:
                idx = await conn.fetchval(
                    "SELECT indexname FROM pg_indexes WHERE indexname = 'idx_memory_embedding_hnsw'"
                )
            if idx:
                console.print("   [green]✓ HNSW index exists[/green]")
            else:
                issues.append("HNSW vector index missing on memory.embedding")
                console.print("   [yellow]⚠ HNSW index not found[/yellow]")
                if fix:
                    try:
                        async with get_connection() as conn:
                            await conn.execute("SELECT ensure_memory_hnsw_index()")
                        # Verify it was created
                        async with get_connection() as conn:
                            idx2 = await conn.fetchval(
                                "SELECT indexname FROM pg_indexes WHERE indexname = 'idx_memory_embedding_hnsw'"
                            )
                        if idx2:
                            fixed.append("Created HNSW vector index")
                            console.print("   [green]  → Fixed: HNSW index created[/green]")
                        else:
                            console.print("   [dim]  → No embeddings yet — index will be created when data exists[/dim]")
                    except Exception as e:
                        console.print(f"   [yellow]  → Could not create index: {e}[/yellow]")
            await close_db()
        except Exception as e:
            console.print(f"   [dim]  → Skipped: {e}[/dim]")

        # -- Summary --
        console.print()
        if not issues:
            console.print("[bold green]All checks passed![/bold green]")
        else:
            console.print(f"[bold yellow]⚠ {len(issues)} issue(s) found[/bold yellow]")
            for issue in issues:
                console.print(f"  • {issue}")
            if fixed:
                console.print(f"\n[bold green]{len(fixed)} issue(s) fixed[/bold green]")
                for f_ in fixed:
                    console.print(f"  ✓ {f_}")
            elif not fix:
                console.print("\n[dim]Run 'syne repair --fix' to attempt auto-repair[/dim]")

    asyncio.run(_repair())
