"""Syne CLI — command line interface."""

import asyncio
import os
import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="syne")
def cli():
    """Syne — AI Agent Framework with Unlimited Memory 🧠"""
    pass


# ── Init ─────────────────────────────────────────────────────

@cli.command()
def init():
    """Initialize Syne: authenticate, setup database, configure."""
    console.print(Panel("[bold]Welcome to Syne 🧠[/bold]\nAI Agent Framework with Unlimited Memory", style="blue"))
    console.print()

    # 1. Provider selection
    console.print("[bold]Step 1: Choose your chat AI provider[/bold]")
    console.print()
    console.print("  [bold cyan]OAuth (free, rate limited):[/bold cyan]")
    console.print("  1. Google Gemini [green](recommended — free via OAuth)[/green]")
    console.print("  2. ChatGPT / GPT [green](free via Codex OAuth)[/green]")
    console.print("  3. Claude [green](free via claude.ai OAuth)[/green]")
    console.print()
    console.print("  [bold yellow]API Key (paid per token):[/bold yellow]")
    console.print("  4. OpenAI [yellow](API key, paid)[/yellow]")
    console.print("  5. Anthropic Claude [yellow](API key, paid)[/yellow]")
    console.print("  6. Together AI [yellow](API key, paid)[/yellow]")
    console.print()

    choice = click.prompt("Select provider", type=click.IntRange(1, 6), default=1)

    env_lines = []
    provider_config = None  # Will be saved to DB after schema init

    if choice == 1:
        console.print("\n[bold green]✓ Google Gemini selected (OAuth)[/bold green]")

        # Check for existing OpenClaw credentials
        from .auth.google_oauth import detect_openclaw_credentials, login_google
        existing = detect_openclaw_credentials()
        if existing:
            console.print(f"  Found existing OpenClaw credentials for [bold]{existing.email}[/bold]")
            if click.confirm("  Reuse these credentials?", default=True):
                from pathlib import Path
                creds_path = Path.home() / ".syne" / "google_credentials.json"
                existing.save(creds_path)
                console.print(f"  [green]✓ Credentials saved to {creds_path}[/green]")
            else:
                console.print("  Starting fresh OAuth login...")
                creds = asyncio.run(login_google())
                from pathlib import Path
                creds_path = Path.home() / ".syne" / "google_credentials.json"
                creds.save(creds_path)
        else:
            console.print("  No existing credentials found. Starting OAuth login...")
            creds = asyncio.run(login_google())
            from pathlib import Path
            creds_path = Path.home() / ".syne" / "google_credentials.json"
            creds.save(creds_path)

        env_lines.append("SYNE_PROVIDER=google")
        provider_config = {"driver": "google_cca", "model": "gemini-2.5-pro", "auth": "oauth"}

    elif choice == 2:
        console.print("\n[bold green]✓ ChatGPT selected (Codex OAuth)[/bold green]")
        console.print("  [dim]Requires OpenAI/ChatGPT login via Codex CLI.[/dim]")
        console.print("  [dim]Auth tokens read from ~/.openclaw or Codex config.[/dim]")
        env_lines.append("SYNE_PROVIDER=codex")
        provider_config = {"driver": "codex", "model": "gpt-5.2", "auth": "oauth"}

    elif choice == 3:
        console.print("\n[bold green]✓ Claude selected (OAuth)[/bold green]")
        console.print("  [dim]Requires claude.ai subscription + Claude CLI login.[/dim]")
        console.print("  [dim]Auth tokens read from ~/.claude/.credentials.json[/dim]")
        env_lines.append("SYNE_PROVIDER=anthropic")
        provider_config = {"driver": "anthropic", "model": "claude-sonnet-4-20250514", "auth": "oauth"}

    elif choice == 4:
        console.print("\n[bold green]✓ OpenAI selected (API key)[/bold green]")
        api_key = click.prompt("OpenAI API key", hide_input=True)
        env_lines.append(f"SYNE_OPENAI_API_KEY={api_key}")
        env_lines.append("SYNE_PROVIDER=openai")
        provider_config = {"driver": "openai_compat", "model": "gpt-4o", "auth": "api_key"}

    elif choice == 5:
        console.print("\n[bold green]✓ Anthropic Claude selected (API key)[/bold green]")
        api_key = click.prompt("Anthropic API key", hide_input=True)
        env_lines.append(f"SYNE_ANTHROPIC_API_KEY={api_key}")
        env_lines.append("SYNE_PROVIDER=anthropic")
        provider_config = {"driver": "anthropic", "model": "claude-sonnet-4-20250514", "auth": "api_key"}

    elif choice == 6:
        console.print("\n[bold green]✓ Together AI selected (API key)[/bold green]")
        api_key = click.prompt("Together API key", hide_input=True)
        env_lines.append(f"SYNE_TOGETHER_API_KEY={api_key}")
        env_lines.append("SYNE_PROVIDER=together")
        provider_config = {"driver": "openai_compat", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "auth": "api_key"}

    # 2. Telegram bot (optional)
    console.print("\n[bold]Step 2: Telegram Bot (optional)[/bold]")
    telegram_token = None
    if click.confirm("Setup Telegram bot?", default=True):
        console.print("Get a bot token from @BotFather on Telegram")
        telegram_token = click.prompt("Telegram bot token")
        # Token goes to DB, not .env (per credential policy)
        console.print("[dim]Token will be saved to database after schema init.[/dim]")

    # 3. Database (Docker only — no external DB option)
    console.print("\n[bold]Step 3: Database[/bold]")
    console.print("  Syne uses its own isolated PostgreSQL + pgvector container.")
    console.print("  [dim]This ensures your API keys, OAuth tokens, and memories stay private.[/dim]")
    console.print("  [dim]Docker is required — PostgreSQL is bundled, not optional.[/dim]")
    db_url = "postgresql://syne:syne@localhost:5433/syne"
    env_lines.append(f"SYNE_DATABASE_URL={db_url}")

    # 4. Write .env
    env_content = "\n".join(env_lines) + "\n"
    env_path = os.path.join(os.getcwd(), ".env")

    if os.path.exists(env_path):
        if not click.confirm(f"\n.env already exists. Overwrite?"):
            console.print("[yellow]Keeping existing .env[/yellow]")
        else:
            with open(env_path, "w") as f:
                f.write(env_content)
            console.print(f"[green]✓ .env written[/green]")
    else:
        with open(env_path, "w") as f:
            f.write(env_content)
        console.print(f"[green]✓ .env written[/green]")

    # 5. Start DB (Docker — always)
    console.print("\n[bold]Starting database...[/bold]")
    console.print("[dim]Run: docker compose up -d db[/dim]")
    result = os.system("docker compose up -d db")
    if result != 0:
        console.print("[red]Failed to start database. Make sure Docker is installed and running.[/red]")
        console.print("[dim]Install Docker: https://docs.docker.com/get-docker/[/dim]")
        return

    # Wait for DB
    console.print("Waiting for PostgreSQL to be ready...")
    import time
    for _ in range(15):
        result = os.system("docker compose exec -T db pg_isready -U syne > /dev/null 2>&1")
        if result == 0:
            break
        time.sleep(2)
    else:
        console.print("[red]Database didn't start in time. Run 'docker compose up -d db' manually.[/red]")
        return

    console.print("[green]✓ Database ready[/green]")

    # 6. Initialize schema + Agent identity
    console.print("\n[bold]Step 4: Initialize database schema...[/bold]")
    schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
    with open(schema_path) as f:
        schema = f.read()

    async def _init_schema():
        from .db.connection import init_db, close_db
        pool = await init_db(db_url)
        async with pool.acquire() as conn:
            await conn.execute(schema)
        await close_db()

    asyncio.run(_init_schema())
    console.print("[green]✓ Schema initialized[/green]")

    console.print("\n[bold]Step 5: Agent Identity[/bold]")
    name = click.prompt("Agent name", default="Syne")
    motto = click.prompt("Motto (optional)", default="I remember, therefore I am")

    # Save identity + credentials to DB
    async def _save_identity():
        from .db.connection import init_db, close_db
        from .db.models import set_identity
        pool = await init_db(db_url)
        await set_identity("name", name)
        await set_identity("motto", motto)
        # Save Telegram token to DB if provided
        if telegram_token:
            from .db.credentials import set_telegram_bot_token
            await set_telegram_bot_token(telegram_token)
            console.print("[green]✓ Telegram bot token saved to database[/green]")
        await close_db()

    asyncio.run(_save_identity())

    console.print()
    console.print(Panel(
        f"[bold]{name}[/bold]\n[italic]{motto}[/italic]",
        title="Your Agent",
        style="green",
    ))

    # 7. Summary
    console.print("\n[bold green]✅ Setup complete![/bold green]")
    console.print()
    console.print("Next steps:")
    console.print("  [bold]syne start[/bold]               # Start agent")
    console.print("  [bold]syne status[/bold]              # Check status")
    console.print("  [bold]syne repair[/bold]              # Diagnose issues")
    console.print()


# ── Start ────────────────────────────────────────────────────

@cli.command()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def start(debug):
    """Start Syne agent."""
    if debug:
        import logging
        logging.getLogger("syne").setLevel(logging.DEBUG)

    from .main import run
    console.print("[bold blue]Starting Syne...[/bold blue]")
    asyncio.run(run())


# ── Status ───────────────────────────────────────────────────

@cli.command()
def status():
    """Show Syne status."""
    async def _status():
        from .config import load_settings
        from .db.connection import init_db, close_db

        settings = load_settings()

        table = Table(title="🧠 Syne Status", show_header=False, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Version", "0.1.0")

        # DB connection
        try:
            pool = await init_db(settings.database_url)
            async with pool.acquire() as conn:
                # Counts
                mem = await conn.fetchrow("SELECT COUNT(*) as c FROM memory")
                users = await conn.fetchrow("SELECT COUNT(*) as c FROM users")
                sessions = await conn.fetchrow("SELECT COUNT(*) as c FROM sessions WHERE status = 'active'")
                msgs = await conn.fetchrow("SELECT COUNT(*) as c FROM messages")

                # Identity
                identity = await conn.fetchrow("SELECT value FROM identity WHERE key = 'name'")
                agent_name = identity["value"] if identity else "Syne"

            table.add_row("Database", "[green]Connected[/green]")
            table.add_row("Agent", agent_name)
            table.add_row("Memories", str(mem["c"]))
            table.add_row("Users", str(users["c"]))
            table.add_row("Active Sessions", str(sessions["c"]))
            table.add_row("Total Messages", str(msgs["c"]))

            await close_db()
        except Exception as e:
            table.add_row("Database", f"[red]Error: {e}[/red]")

        # Provider
        provider = os.environ.get("SYNE_PROVIDER", "not configured")
        table.add_row("Provider", provider)

        # Telegram
        tg = "✓ Configured" if settings.telegram_bot_token else "Not configured"
        table.add_row("Telegram", tg)

        console.print(table)

    asyncio.run(_status())


# ── Database ─────────────────────────────────────────────────

@cli.group()
def db():
    """Database management commands."""
    pass


@db.command("init")
def db_init():
    """Initialize database schema."""
    async def _init():
        from .config import load_settings
        from .db.connection import init_db, close_db

        settings = load_settings()
        pool = await init_db(settings.database_url)

        schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
        with open(schema_path) as f:
            schema = f.read()

        async with pool.acquire() as conn:
            await conn.execute(schema)

        console.print("[green]✓ Database schema initialized[/green]")
        await close_db()

    asyncio.run(_init())


@db.command("reset")
@click.confirmation_option(prompt="This will DELETE ALL DATA. Are you sure?")
def db_reset():
    """Reset database (DROP ALL + re-init)."""
    async def _reset():
        from .config import load_settings
        from .db.connection import init_db, close_db

        settings = load_settings()
        pool = await init_db(settings.database_url)

        async with pool.acquire() as conn:
            await conn.execute("""
                DROP TABLE IF EXISTS messages CASCADE;
                DROP TABLE IF EXISTS subagent_runs CASCADE;
                DROP TABLE IF EXISTS sessions CASCADE;
                DROP TABLE IF EXISTS memory CASCADE;
                DROP TABLE IF EXISTS abilities CASCADE;
                DROP TABLE IF EXISTS capabilities CASCADE;
                DROP TABLE IF EXISTS groups CASCADE;
                DROP TABLE IF EXISTS users CASCADE;
                DROP TABLE IF EXISTS rules CASCADE;
                DROP TABLE IF EXISTS soul CASCADE;
                DROP TABLE IF EXISTS identity CASCADE;
                DROP TABLE IF EXISTS config CASCADE;
                DROP EXTENSION IF EXISTS vector;
            """)

        console.print("[yellow]Tables dropped.[/yellow]")

        schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
        with open(schema_path) as f:
            schema = f.read()

        async with pool.acquire() as conn:
            await conn.execute(schema)

        console.print("[green]✓ Database re-initialized[/green]")
        await close_db()

    asyncio.run(_reset())


# ── Identity ─────────────────────────────────────────────────

@cli.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
def identity(key, value):
    """View or set identity values. Usage: syne identity [key] [value]"""
    async def _identity():
        from .config import load_settings
        from .db.connection import init_db, close_db
        from .db.models import get_identity, set_identity

        settings = load_settings()
        await init_db(settings.database_url)

        if key and value:
            await set_identity(key, value)
            console.print(f"[green]✓ {key} = {value}[/green]")
        else:
            data = await get_identity()
            table = Table(title="🪪 Identity", show_header=True)
            table.add_column("Key", style="bold")
            table.add_column("Value")
            for k, v in data.items():
                table.add_row(k, v)
            console.print(table)

        await close_db()

    asyncio.run(_identity())


# ── Prompt ───────────────────────────────────────────────────

@cli.command()
def prompt():
    """Show the current system prompt."""
    async def _prompt():
        from .config import load_settings
        from .db.connection import init_db, close_db
        from .boot import build_system_prompt

        settings = load_settings()
        await init_db(settings.database_url)

        p = await build_system_prompt()
        console.print(Panel(Syntax(p, "markdown"), title="System Prompt", style="blue"))

        await close_db()

    asyncio.run(_prompt())


# ── Memory ───────────────────────────────────────────────────

@cli.group()
def memory():
    """Memory management commands."""
    pass


@memory.command("stats")
def memory_stats():
    """Show memory statistics."""
    async def _stats():
        from .config import load_settings
        from .db.connection import init_db, close_db

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

        console.print(f"\n🧠 [bold]Total memories: {total['c']}[/bold]\n")

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
        from .config import load_settings
        from .db.connection import init_db, close_db
        from .memory.engine import MemoryEngine

        settings = load_settings()
        await init_db(settings.database_url)

        # Need a provider for embedding
        provider = _get_provider(settings)
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
        from .config import load_settings
        from .db.connection import init_db, close_db
        from .memory.engine import MemoryEngine

        settings = load_settings()
        await init_db(settings.database_url)

        provider = _get_provider(settings)
        engine = MemoryEngine(provider)

        mem_id = await engine.store_if_new(content=content, category=category)
        if mem_id:
            console.print(f"[green]✓ Memory stored (id: {mem_id})[/green]")
        else:
            console.print("[yellow]Similar memory already exists. Skipped.[/yellow]")

        await close_db()

    asyncio.run(_add())


# ── Repair ───────────────────────────────────────────────────

@cli.command()
@click.option("--fix", is_flag=True, help="Attempt to auto-fix issues found")
def repair(fix):
    """Diagnose and repair Syne installation. Run without --fix to check only."""
    async def _repair():
        from .config import load_settings
        issues = []
        fixed = []

        console.print(Panel("[bold]🔧 Syne Repair[/bold]", style="blue"))
        console.print()

        settings = load_settings()

        # ── 1. Database connectivity ──
        console.print("[bold]1. Database[/bold]")
        try:
            from .db.connection import init_db, close_db
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
                       "messages", "rules", "sessions", "soul", "subagent_runs", "users"]
            missing = [t for t in expected if t not in table_names]

            if missing:
                issues.append(f"Missing tables: {', '.join(missing)}")
                console.print(f"   [red]✗ Missing tables: {', '.join(missing)}[/red]")
                if fix:
                    schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
                    if os.path.exists(schema_path):
                        with open(schema_path) as f:
                            schema = f.read()
                        async with pool.acquire() as conn:
                            await conn.execute(schema)
                        fixed.append(f"Created missing tables")
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
                    from .db.models import set_config
                    defaults = {
                        "memory.auto_capture": False,
                        "memory.auto_evaluate": True,
                        "memory.recall_limit": 10,
                        "memory.max_importance": 1.0,
                        "provider.primary": {"name": "google", "auth": "oauth"},
                        "provider.chat_model": "gemini-2.5-pro",
                        "provider.embedding_model": "text-embedding-004",
                        "provider.embedding_dimensions": 768,
                        "session.compaction_threshold": 80000,
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

        # ── 2. Google OAuth ──
        console.print("\n[bold]2. Google OAuth[/bold]")
        from pathlib import Path
        creds_path = Path.home() / ".syne" / "google_credentials.json"
        if creds_path.exists():
            try:
                from .auth.google_oauth import get_credentials
                creds = await get_credentials()
                token = await creds.get_token()
                console.print(f"   [green]✓ Authenticated as {creds.email}[/green]")
                console.print(f"   [dim]Token valid (first 8): {token[:8]}...[/dim]")
            except Exception as e:
                issues.append(f"OAuth token refresh failed: {e}")
                console.print(f"   [red]✗ Token refresh failed: {e}[/red]")
                console.print("   [dim]Run 'syne init' to re-authenticate[/dim]")
        else:
            issues.append("No Google credentials found")
            console.print("   [yellow]⚠ No credentials at {creds_path}[/yellow]")
            console.print("   [dim]Run 'syne init' to set up OAuth[/dim]")

        # ── 3. Telegram Bot ──
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

        # ── 4. Abilities ──
        console.print("\n[bold]4. Abilities[/bold]")
        try:
            from .abilities import AbilityRegistry
            from .abilities.loader import load_all_abilities
            from .db.connection import init_db, close_db
            await init_db(settings.database_url)

            registry = AbilityRegistry()
            count = await load_all_abilities(registry)
            console.print(f"   [green]✓ {count} abilities loaded[/green]")

            for ab in registry.list_all():
                has_config = bool(ab.config and any(ab.config.values()))
                status = "[green]ready[/green]" if has_config else "[yellow]needs config[/yellow]"
                enabled = "✓" if ab.enabled else "✗"
                console.print(f"   {enabled} {ab.name} ({ab.source}) — {status}")

            await close_db()
        except Exception as e:
            console.print(f"   [red]✗ Error loading abilities: {e}[/red]")

        # ── 5. Process ──
        console.print("\n[bold]5. Process[/bold]")
        import subprocess
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
            console.print(f"   [green]✓ Autostart enabled (systemd)[/green]")
        else:
            console.print("   [dim]Autostart not configured (see 'syne autostart')[/dim]")

        # ── 6. Docker DB ──
        console.print("\n[bold]6. Docker (syne-db)[/bold]")
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", "syne-db"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            status = result.stdout.strip()
            color = "green" if status == "running" else "red"
            console.print(f"   [{color}]{status}[/{color}]")
            if status != "running" and fix:
                os.system("docker start syne-db")
                fixed.append("Started syne-db container")
                console.print("   [green]  → Fixed: container started[/green]")
        else:
            console.print("   [yellow]⚠ Container not found[/yellow]")

        # ── Summary ──
        console.print()
        if not issues:
            console.print("[bold green]✅ All checks passed![/bold green]")
        else:
            console.print(f"[bold yellow]⚠ {len(issues)} issue(s) found[/bold yellow]")
            for issue in issues:
                console.print(f"  • {issue}")
            if fixed:
                console.print(f"\n[bold green]🔧 {len(fixed)} issue(s) fixed[/bold green]")
                for f_ in fixed:
                    console.print(f"  ✓ {f_}")
            elif not fix:
                console.print("\n[dim]Run 'syne repair --fix' to attempt auto-repair[/dim]")

    asyncio.run(_repair())


# ── Autostart ────────────────────────────────────────────────

@cli.command()
@click.option("--enable/--disable", default=True, help="Enable or disable autostart")
def autostart(enable):
    """Configure systemd autostart for Syne."""
    import subprocess
    from pathlib import Path

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "syne.service"
    syne_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(syne_dir, ".venv", "bin", "python")

    if not enable:
        subprocess.run(["systemctl", "--user", "stop", "syne"], capture_output=True)
        subprocess.run(["systemctl", "--user", "disable", "syne"], capture_output=True)
        if service_file.exists():
            service_file.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        console.print("[yellow]Autostart disabled.[/yellow]")
        return

    # Create service file
    service_dir.mkdir(parents=True, exist_ok=True)

    env_file = os.path.join(syne_dir, ".env")

    service_content = f"""[Unit]
Description=Syne AI Agent
After=network.target docker.service

[Service]
Type=simple
WorkingDirectory={syne_dir}
EnvironmentFile={env_file}
ExecStart={venv_python} -m syne.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

    service_file.write_text(service_content)
    console.print(f"[green]✓ Service file written: {service_file}[/green]")

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    subprocess.run(["systemctl", "--user", "enable", "syne"], capture_output=True)
    console.print("[green]✓ Autostart enabled[/green]")

    # Enable linger so service runs without login
    import getpass
    username = getpass.getuser()
    result = subprocess.run(["loginctl", "enable-linger", username], capture_output=True)
    if result.returncode == 0:
        console.print(f"[green]✓ Linger enabled for {username}[/green]")

    console.print()
    console.print("[bold]Commands:[/bold]")
    console.print("  systemctl --user start syne    # Start now")
    console.print("  systemctl --user stop syne     # Stop")
    console.print("  systemctl --user restart syne  # Restart")
    console.print("  systemctl --user status syne   # Status")
    console.print("  journalctl --user -u syne -f   # Logs")


# ── Stop ─────────────────────────────────────────────────────

@cli.command()
def stop():
    """Stop running Syne process."""
    import subprocess
    
    # Try systemd first
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "syne"],
        capture_output=True, text=True,
    )
    if result.stdout.strip() == "active":
        subprocess.run(["systemctl", "--user", "stop", "syne"])
        console.print("[green]✓ Syne stopped (systemd)[/green]")
        return

    # Fall back to pkill
    result = subprocess.run(["pkill", "-f", "syne.main"], capture_output=True)
    if result.returncode == 0:
        console.print("[green]✓ Syne process killed[/green]")
    else:
        console.print("[yellow]Syne is not running.[/yellow]")


# ── Restart ──────────────────────────────────────────────────

@cli.command()
def restart():
    """Restart Syne (stop + start)."""
    import subprocess

    # Try systemd
    result = subprocess.run(
        ["systemctl", "--user", "is-enabled", "syne"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        subprocess.run(["systemctl", "--user", "restart", "syne"])
        console.print("[green]✓ Syne restarted (systemd)[/green]")
        return

    # Manual: kill + start
    subprocess.run(["pkill", "-f", "syne.main"], capture_output=True)
    import time
    time.sleep(2)

    syne_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(syne_dir, ".venv", "bin", "python")
    subprocess.Popen(
        [venv_python, "-m", "syne.main"],
        cwd=syne_dir,
        stdout=open("/tmp/syne.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    console.print("[green]✓ Syne restarted (PID in background)[/green]")
    console.print("[dim]Logs: tail -f /tmp/syne.log[/dim]")


# ── Helpers ──────────────────────────────────────────────────

def _get_provider(settings):
    """Quick provider init for CLI commands."""
    provider_name = os.environ.get("SYNE_PROVIDER", "google")

    if provider_name == "google":
        from .llm.google import GoogleProvider
        from .auth.google_oauth import get_credentials

        async def _make():
            creds = await get_credentials()
            return GoogleProvider(credentials=creds)

        return asyncio.run(_make())
    elif provider_name == "openai":
        from .llm.openai import OpenAIProvider
        if not settings.openai_api_key:
            console.print("[red]SYNE_OPENAI_API_KEY not set[/red]")
            sys.exit(1)
        return OpenAIProvider(api_key=settings.openai_api_key)
    else:
        console.print(f"[red]Unknown provider: {provider_name}[/red]")
        sys.exit(1)


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
