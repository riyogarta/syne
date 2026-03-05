"""Init command — first-time setup wizard."""

import asyncio
import os
import secrets
import subprocess
import sys
import time
import click

from . import cli
from .shared import console, _strip_env_quotes
from .helpers import (
    _ensure_system_deps, _ensure_docker, _ensure_ollama,
    _ensure_evaluator_model, _setup_service, _create_symlink,
    _setup_update_check,
)

from rich.panel import Panel


def _detect_server_tier(sys_cpu, sys_ram_gb):
    """Detect server tier based on hardware resources.

    Returns: (tier_name, embed_model, embed_dims, embed_size, eval_model, eval_size)
    """
    if sys_cpu >= 4 and sys_ram_gb >= 16:
        return ("beast", "qwen3-embedding:8b", 4096, "~4.7GB", "qwen3:4b", "~2.5GB")
    elif sys_cpu >= 4 and sys_ram_gb >= 8:
        return ("strong", "qwen3-embedding:4b", 2560, "~2.5GB", "qwen3:1.7b", "~1.3GB")
    elif sys_ram_gb >= 4:
        return ("moderate", "qwen3-embedding:0.6b", 1024, "~700MB", "qwen3:1.7b", "~1.3GB")
    elif sys_ram_gb >= 2 and sys_cpu >= 2:
        return ("minimal", "qwen3-embedding:0.6b", 1024, "~700MB", "qwen3:0.6b", "~400MB")
    else:
        return ("cloud", None, None, None, None, None)


@cli.command()
def init():
    """Initialize Syne: authenticate, setup database, configure."""
    console.print(Panel("[bold]Welcome to Syne[/bold]\nAI Agent Framework with Unlimited Memory", style="blue"))
    console.print()

    # Pre-check: system dependencies (pip, venv)
    _ensure_system_deps()

    # Pre-check: Docker (install + start + determine sudo prefix)
    docker_prefix = _ensure_docker()

    # 1. Provider selection
    console.print("[bold]Step 1: Choose your chat AI provider[/bold]")
    console.print()
    console.print("  [bold cyan]OAuth (free, rate limited):[/bold cyan]")
    console.print("  1. Google Gemini [green](recommended — free via OAuth)[/green]")
    console.print("  2. ChatGPT / GPT [green](free via Codex OAuth)[/green]")
    console.print("  3. Claude [green](free via claude.ai OAuth)[/green]")
    console.print("  [dim]     Google/GPT OAuth need SSH port forwarding on headless servers.[/dim]")
    console.print("  [dim]       Claude works everywhere — no forwarding needed![/dim]")
    console.print()
    console.print("  [bold yellow]API Key (paid per token):[/bold yellow]")
    console.print("  4. Google Gemini [yellow](API key, free tier available)[/yellow]")
    console.print("  5. Google Vertex AI [yellow](API key, GCP)[/yellow]")
    console.print("  6. OpenAI [yellow](API key, paid)[/yellow]")
    console.print("  7. Anthropic Claude [yellow](API key, paid)[/yellow]")
    console.print("  8. Together AI [yellow](API key, paid)[/yellow]")
    console.print("  9. Groq [yellow](API key, free tier available)[/yellow]")
    console.print()

    choice = click.prompt("Select provider", type=click.IntRange(1, 9), default=1)

    env_lines = []
    provider_config = None  # Will be saved to DB after schema init
    google_creds = None  # Only set for Google OAuth provider
    codex_creds = None  # Only set for Codex OAuth provider

    if choice == 1:
        console.print("\n[bold green]✓ Google Gemini selected (OAuth)[/bold green]")
        console.print("  [dim]Sign in with your Google account to get free Gemini access.[/dim]")

        from syne.auth.google_oauth import login_google
        google_creds = asyncio.run(login_google())
        # Credentials saved to DB after schema init (Step 6)

        provider_config = {"driver": "google_cca", "model": "gemini-2.5-pro", "auth": "oauth"}

    elif choice == 2:
        console.print("\n[bold green]✓ ChatGPT selected (Codex OAuth)[/bold green]")
        console.print("  [dim]Requires ChatGPT Plus/Pro/Team subscription.[/dim]")

        from syne.auth.codex_oauth import login_codex
        codex_creds = asyncio.run(login_codex())
        # Credentials saved to DB after schema init (Step 6)

        provider_config = {"driver": "codex", "model": "gpt-5.2", "auth": "oauth"}

    elif choice == 3:
        console.print("\n[bold green]✓ Claude selected (OAuth)[/bold green]")
        console.print("  [dim]Requires claude.ai Pro/Max subscription.[/dim]")

        from syne.auth.claude_oauth import login_claude
        claude_creds = asyncio.run(login_claude())

        provider_config = {
            "driver": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "auth": "oauth",
            "_claude_creds": claude_creds.to_dict(),
        }

    elif choice == 4:
        console.print("\n[bold green]✓ Google Gemini selected (API key)[/bold green]")
        console.print("  [dim]Get your key at aistudio.google.com/apikey — free tier available.[/dim]")
        api_key = click.prompt("Google AI API key")
        provider_config = {"driver": "google_cca", "model": "gemini-2.5-pro", "auth": "api_key", "_api_key": api_key, "_credential_key": "credential.google_cca_api_key"}

    elif choice == 5:
        console.print("\n[bold green]✓ Google Vertex AI selected (API key)[/bold green]")
        console.print("  [dim]Get your key at console.cloud.google.com (APIs & Services > Credentials).[/dim]")
        api_key = click.prompt("Google Cloud API key")
        provider_config = {"driver": "vertex", "model": "gemini-2.5-pro", "auth": "api_key", "_api_key": api_key, "_credential_key": "credential.vertex_ai_api_key"}

    elif choice == 6:
        console.print("\n[bold green]✓ OpenAI selected (API key)[/bold green]")
        api_key = click.prompt("OpenAI API key")
        provider_config = {"driver": "openai_compat", "model": "gpt-4o", "auth": "api_key", "_api_key": api_key, "_base_url": "https://api.openai.com/v1"}

    elif choice == 7:
        console.print("\n[bold green]✓ Anthropic Claude selected (API key)[/bold green]")
        api_key = click.prompt("Anthropic API key")
        provider_config = {"driver": "anthropic", "model": "claude-sonnet-4-20250514", "auth": "api_key", "_api_key": api_key}

    elif choice == 8:
        console.print("\n[bold green]✓ Together AI selected (API key)[/bold green]")
        api_key = click.prompt("Together API key")
        provider_config = {"driver": "openai_compat", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "auth": "api_key", "_api_key": api_key, "_base_url": "https://api.together.xyz/v1"}

    elif choice == 9:
        console.print("\n[bold green]✓ Groq selected (API key)[/bold green]")
        console.print("  [dim]Get your key at console.groq.com[/dim]")
        api_key = click.prompt("Groq API key")
        provider_config = {"driver": "openai_compat", "model": "llama-3.3-70b-versatile", "auth": "api_key", "_api_key": api_key, "_base_url": "https://api.groq.com/openai/v1"}

    # 2. Embedding provider (for memory)
    console.print("\n[bold]Step 2: Choose embedding provider (for memory)[/bold]")
    console.print()

    # Detect system resources
    sys_cpu = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    sys_ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
            else:
                sys_ram_gb = 0
    except Exception:
        sys_ram_gb = 0

    # Detect tier
    tier_name, tier_embed, tier_dims, tier_embed_size, tier_eval, tier_eval_size = _detect_server_tier(sys_cpu, sys_ram_gb)
    ollama_available = tier_name != "cloud"

    # Permanent warning
    console.print("  [bold yellow]Warning:[/bold yellow] Your embedding model choice is [bold]permanent[/bold] for this installation.")
    console.print("  [dim]Changing it later requires resetting ALL memories (re-embedding is not yet supported).[/dim]")
    console.print("  [dim]Choose based on your server's long-term capacity.[/dim]")
    console.print()

    # Cloud options
    console.print("  [bold cyan]Cloud providers:[/bold cyan]")
    console.print("  1. Together AI [dim](~$0.008/1M tokens)[/dim]")
    console.print("  2. OpenAI [dim](~$0.02/1M tokens)[/dim]")
    console.print()

    # Ollama options with tier-based availability
    console.print("  [bold cyan]Local (Ollama — FREE, no API costs):[/bold cyan]")

    # Option 3: qwen3-embedding:0.6b — min 2 CPU, 2GB RAM
    if sys_cpu >= 2 and sys_ram_gb >= 2:
        console.print("  3. qwen3-embedding:0.6b  [dim](1024d, ~700MB)       min 2 CPU, 2GB RAM[/dim]")
    else:
        console.print("  3. qwen3-embedding:0.6b  [red dim]\\[unavailable — need 2 CPU, 2GB RAM][/red dim]")

    # Option 4: qwen3-embedding:4b — min 4 CPU, 8GB RAM
    if sys_cpu >= 4 and sys_ram_gb >= 8:
        console.print("  4. qwen3-embedding:4b    [dim](2560d, ~2.5GB)       min 4 CPU, 8GB RAM[/dim]")
    else:
        console.print("  4. qwen3-embedding:4b    [red dim]\\[unavailable — need 4 CPU, 8GB RAM][/red dim]")

    # Option 5: qwen3-embedding:8b — min 4 CPU, 16GB RAM
    if sys_cpu >= 4 and sys_ram_gb >= 16:
        console.print("  5. qwen3-embedding:8b    [dim](4096d, ~4.7GB)       min 4 CPU, 16GB RAM[/dim]")
    else:
        console.print("  5. qwen3-embedding:8b    [red dim]\\[unavailable — need 4 CPU, 16GB RAM][/red dim]")

    console.print()

    # Determine default and show recommendation
    # Map tier to recommended option
    tier_defaults = {"beast": 5, "strong": 4, "moderate": 3, "minimal": 3, "cloud": 1}
    default_embed = tier_defaults.get(tier_name, 1)
    console.print(f"  [dim]Your server: {sys_cpu} CPU, {sys_ram_gb:.0f}GB RAM[/dim]", end="")
    if ollama_available:
        console.print(f" [dim]-> recommended: option {default_embed}[/dim]")
    else:
        console.print(f" [dim]-> Ollama unavailable, recommended: option 1[/dim]")
    console.print()

    embed_choice = click.prompt("Select embedding provider", type=click.IntRange(1, 5), default=default_embed)
    embedding_config = None

    # Validate Ollama selection against system capacity
    ollama_reqs = {
        3: (2, 2, "qwen3-embedding:0.6b"),
        4: (4, 8, "qwen3-embedding:4b"),
        5: (4, 16, "qwen3-embedding:8b"),
    }
    if embed_choice in ollama_reqs:
        req_cpu, req_ram, model_label = ollama_reqs[embed_choice]
        if sys_cpu < req_cpu or sys_ram_gb < req_ram:
            console.print(f"\n[red]{model_label} not available — need {req_cpu} CPU, {req_ram}GB RAM (you have {sys_cpu} CPU, {sys_ram_gb:.0f}GB RAM)[/red]")
            console.print("[dim]Please select a different option.[/dim]")
            embed_choice = click.prompt("Select embedding provider", type=click.IntRange(1, 5), default=default_embed)
            # Re-validate once more
            if embed_choice in ollama_reqs:
                req_cpu, req_ram, model_label = ollama_reqs[embed_choice]
                if sys_cpu < req_cpu or sys_ram_gb < req_ram:
                    console.print(f"[red]{model_label} still not available. Falling back to Together AI.[/red]")
                    embed_choice = 1

    # Ollama model configs
    ollama_models = {
        3: ("qwen3-embedding:0.6b", 1024),
        4: ("qwen3-embedding:4b", 2560),
        5: ("qwen3-embedding:8b", 4096),
    }

    if embed_choice == 1:
        console.print("\n[bold green]✓ Together AI selected for embeddings[/bold green]")
        console.print("  [dim]Sign up at together.ai — $5 free credit included.[/dim]")
        embed_api_key = click.prompt("Together AI API key")
        embedding_config = {
            "driver": "together",
            "model": "BAAI/bge-base-en-v1.5",
            "dimensions": 768,
            "_api_key": embed_api_key,
            "_credential_key": "credential.together_api_key",
        }
    elif embed_choice == 2:
        console.print("\n[bold green]✓ OpenAI selected for embeddings[/bold green]")
        embed_api_key = click.prompt("OpenAI API key")
        embedding_config = {
            "driver": "openai_compat",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "_api_key": embed_api_key,
            "_credential_key": "credential.openai_compat_api_key",
        }
    elif embed_choice in ollama_models:
        selected_embed_model, selected_embed_dims = ollama_models[embed_choice]
        console.print(f"\n[bold green]✓ Ollama selected for embeddings — {selected_embed_model} (FREE, local)[/bold green]")
        _ensure_ollama(embed_model=selected_embed_model)
        embedding_config = {
            "driver": "ollama",
            "model": selected_embed_model,
            "dimensions": selected_embed_dims,
            "_ollama": True,
        }

    # 3. Telegram bot
    console.print("\n[bold]Step 3: Telegram Bot[/bold]")
    console.print("  A Telegram bot is [bold]required[/bold] — it's how Syne talks to you.")
    console.print()
    console.print("  [bold cyan]How to create a bot (takes 1 minute):[/bold cyan]")
    console.print("  1. Open Telegram → search [bold]@BotFather[/bold] → tap Start")
    console.print("  2. Send [bold]/newbot[/bold]")
    console.print("  3. Choose a display name (e.g. \"My Syne\")")
    console.print("  4. Choose a username ending in 'bot' (e.g. \"my_syne_bot\")")
    console.print("  5. BotFather gives you a token like: [dim]1234567890:ABCdefGHIjklMNOpqrsTUVwxyz[/dim]")
    console.print()
    console.print("  [dim]Tip: Also send /setprivacy → Disable so the bot can read group messages.[/dim]")
    console.print()
    while True:
        telegram_token = click.prompt("Paste your bot token here")
        telegram_token = telegram_token.strip()
        # Basic format validation: digits:alphanumeric
        if ":" in telegram_token and len(telegram_token) > 20:
            break
        console.print("  [red]That doesn't look like a valid bot token.[/red]")
        console.print("  [dim]Expected format: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz[/dim]")

    # 3b. Web Search (optional)
    console.print("\n[bold]Step 3b: Web Search (optional)[/bold]")
    console.print("  Lets Syne search the web for you.")
    console.print()
    console.print("  1. Tavily [green](recommended — 1,000 free searches/month, no credit card)[/green]")
    console.print("  2. Brave Search [dim](paid)[/dim]")
    console.print("  3. Skip [dim](can be added later via chat)[/dim]")
    console.print()

    web_search_choice = click.prompt("Select web search provider", type=click.IntRange(1, 3), default=1)
    web_search_api_key = ""
    web_search_driver = ""

    if web_search_choice == 1:
        console.print()
        console.print("  [bold cyan]How to get a free Tavily API key:[/bold cyan]")
        console.print("  1. Go to [link]https://app.tavily.com[/link]")
        console.print("  2. Sign up (free, no credit card)")
        console.print("  3. Copy your API key (starts with tvly-)")
        console.print()
        web_search_api_key = click.prompt("Tavily API key", default="", show_default=False).strip()
        web_search_driver = "tavily"
        if web_search_api_key:
            console.print("[green]✓ Tavily API key saved[/green]")
        else:
            console.print("[dim]  Skipped — you can set it up later by giving Syne your API key via chat.[/dim]")
    elif web_search_choice == 2:
        console.print()
        console.print("  [bold cyan]How to get a Brave Search API key:[/bold cyan]")
        console.print("  1. Go to [link]https://brave.com/search/api/[/link]")
        console.print("  2. Sign up and choose a plan")
        console.print("  3. Create an app → copy the API key")
        console.print()
        web_search_api_key = click.prompt("Brave Search API key", default="", show_default=False).strip()
        web_search_driver = "brave"
        if web_search_api_key:
            console.print("[green]✓ Brave Search API key saved[/green]")
        else:
            console.print("[dim]  Skipped — you can set it up later by giving Syne your API key via chat.[/dim]")
    else:
        console.print("[dim]  Skipped — you can set it up later by giving Syne your API key via chat.[/dim]")

    # 3c. Auto-capture memory
    # Auto-select evaluator model based on tier
    selected_eval_model = tier_eval or "qwen3:0.6b"
    selected_eval_size = tier_eval_size or "~400MB"

    console.print("\n[bold]Step 3c: Auto-capture Memory[/bold]")
    console.print("  When enabled, Syne automatically remembers important facts from conversations")
    console.print("  (e.g. \"I live in Jakarta\", \"My wife's name is Yuli\").")
    console.print()
    if ollama_available:
        console.print(f"  Uses a local AI model ([bold]{selected_eval_model}[/bold], {selected_eval_size}) via Ollama —")
        console.print("  no extra API costs, no rate limit impact.")
        console.print(f"  [dim]Evaluator model auto-selected for your server ({tier_name} tier).[/dim]")
    else:
        console.print("  Uses a local AI model via Ollama — no extra API costs.")
    console.print()
    can_run_ollama = ollama_available or embed_choice in (3, 4, 5)
    if can_run_ollama:
        console.print("  1. ON  [green](recommended)[/green]")
        console.print("  2. OFF [dim](memory only stored on explicit request)[/dim]")
    else:
        console.print("  1. ON  [dim](requires Ollama — your server may be too small)[/dim]")
        console.print("  2. OFF [green](recommended for your server)[/green]")
        console.print()
        console.print("  [yellow]Your system has limited resources for local AI.[/yellow]")
        console.print("  [dim]Auto-capture requires Ollama (~1.3GB RAM when active).[/dim]")
    console.print()

    auto_capture_choice = click.prompt("Enable auto-capture?", type=click.IntRange(1, 2), default=1 if can_run_ollama else 2)
    auto_capture_enabled = auto_capture_choice == 1

    if auto_capture_enabled:
        # Ensure Ollama is installed (may already be if embedding uses Ollama)
        if embed_choice not in (3, 4, 5):
            _ensure_ollama()
        _ensure_evaluator_model(eval_model=selected_eval_model)
        console.print("[green]✓ Auto-capture enabled[/green]")
    else:
        console.print("[dim]  Auto-capture disabled — you can enable it later via /autocapture or chat.[/dim]")

    # 4. Database (Docker only — no external DB option)
    console.print("\n[bold]Step 4: Database[/bold]")
    console.print("  Syne uses its own isolated PostgreSQL + pgvector container.")
    console.print("  [dim]This ensures your API keys, OAuth tokens, and memories stay private.[/dim]")
    console.print("  [dim]Docker is required — PostgreSQL is bundled, not optional.[/dim]")

    # Reuse existing credentials if .env exists, otherwise generate new ones
    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(syne_dir, ".env")
    existing_env = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    existing_env[k.strip()] = _strip_env_quotes(v)

    if existing_env.get("SYNE_DB_USER") and existing_env.get("SYNE_DB_PASSWORD"):
        db_user = existing_env["SYNE_DB_USER"]
        db_password = existing_env["SYNE_DB_PASSWORD"]
        db_name = existing_env.get("SYNE_DB_NAME", "mnemosyne")
        console.print(f"  [dim]Reusing existing DB credentials from .env ({db_user})[/dim]")
    elif existing_env.get("SYNE_DATABASE_URL"):
        # Parse credentials from DATABASE_URL (legacy .env without separate DB_USER/PASSWORD)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(existing_env["SYNE_DATABASE_URL"])
            db_user = parsed.username or f"syne_{secrets.token_hex(4)}"
            db_password = parsed.password or secrets.token_hex(16)
            db_name = (parsed.path or "/mnemosyne").lstrip("/") or "mnemosyne"
            console.print(f"  [dim]Reusing DB credentials from DATABASE_URL ({db_user})[/dim]")
        except Exception:
            db_suffix = secrets.token_hex(4)
            db_user = f"syne_{db_suffix}"
            db_password = secrets.token_hex(16)
            db_name = "mnemosyne"
    else:
        db_suffix = secrets.token_hex(4)
        db_user = f"syne_{db_suffix}"
        db_password = secrets.token_hex(16)
        db_name = "mnemosyne"

    db_url = f"postgresql://{db_user}:{db_password}@localhost:5433/{db_name}"

    env_lines.append(f"SYNE_DB_USER={db_user}")
    env_lines.append(f"SYNE_DB_PASSWORD={db_password}")
    env_lines.append(f"SYNE_DB_NAME={db_name}")
    env_lines.append(f"SYNE_DATABASE_URL={db_url}")
    console.print("  [green]✓ Database credentials generated (unique to this install)[/green]")

    # 4. Write .env (always in project root, where docker-compose.yml lives)
    env_content = "\n".join(env_lines) + "\n"
    env_path = os.path.join(syne_dir, ".env")

    with open(env_path, "w") as f:
        f.write(env_content)
    os.chmod(env_path, 0o600)
    # Also set in current process environment so subprocesses (_setup_update_check) can use them
    for line in env_lines:
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v
    console.print("[green]✓ .env written (chmod 600)[/green]")

    # 5. Start DB (Docker — uses prefix from _ensure_docker)
    docker_cmd = f"{docker_prefix}docker compose"

    # PostgreSQL bakes credentials into the volume at FIRST creation.
    # Subsequent starts with different POSTGRES_USER/PASSWORD are IGNORED.
    # So we must always clean up stale volumes before starting.

    # Check for existing volume (docker volume, not docker compose)
    vol_cmd = f"{docker_prefix}docker volume ls -q"
    volume_check = subprocess.run(
        vol_cmd, shell=True, capture_output=True, text=True,
    )
    volume_exists = "syne_syne_pgdata" in (volume_check.stdout or "")

    if volume_exists:
        console.print("[yellow]Found existing Syne DB volume — removing to apply fresh credentials...[/yellow]")
        # Stop container + remove volume (credentials are baked in, can't change)
        subprocess.run(f"{docker_cmd} down -v", shell=True, cwd=syne_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    console.print("\n[bold]Starting database...[/bold]")
    result = subprocess.run(f"{docker_cmd} up -d db", shell=True, cwd=syne_dir).returncode
    if result != 0:
        console.print("[red]Failed to start database.[/red]")
        return

    # Wait for DB
    console.print("Waiting for PostgreSQL to be ready...")
    for _ in range(20):
        result = subprocess.run(f"{docker_cmd} exec -T db pg_isready -U {db_user}", shell=True, cwd=syne_dir, capture_output=True).returncode
        if result == 0:
            break
        time.sleep(2)
    else:
        console.print("[red]Database didn't start in time.[/red]")
        return

    console.print("[green]✓ Database ready[/green]")

    # 6. Initialize schema + Agent identity
    console.print("\n[bold]Step 5: Initialize database & agent identity...[/bold]")
    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "schema.sql")
    with open(schema_path) as f:
        schema = f.read()

    async def _init_schema():
        from syne.db.connection import init_db, close_db
        pool = await init_db(db_url)
        async with pool.acquire() as conn:
            await conn.execute(schema)
        await close_db()

    asyncio.run(_init_schema())
    console.print("[green]✓ Schema initialized[/green]")

    console.print()  # Agent identity (part of Step 5)
    name = click.prompt("Agent name", default="Syne")
    motto = click.prompt("Motto (optional)", default="I remember, therefore I am")

    # Save identity + credentials to DB
    async def _save_identity():
        from syne.db.connection import init_db, close_db
        from syne.db.models import set_identity, set_config
        pool = await init_db(db_url)
        await set_identity("name", name)
        await set_identity("motto", motto)
        # Save Telegram token to DB if provided
        if telegram_token:
            from syne.db.credentials import set_telegram_bot_token
            await set_telegram_bot_token(telegram_token)
            console.print("[green]✓ Telegram bot token saved to database[/green]")
        # Save Brave Search API key to DB if provided
        if web_search_api_key:
            await set_config("web_search.api_key", web_search_api_key)
            if web_search_driver:
                await set_config("web_search.driver", web_search_driver)
            console.print("[green]✓ Web search API key saved to database[/green]")
        # Save Google OAuth credentials to DB if collected
        if google_creds:
            await google_creds.save_to_db()
            console.print(f"[green]✓ Google OAuth credentials saved to database ({google_creds.email})[/green]")
        # Save Codex OAuth credentials to DB if collected
        if codex_creds:
            await set_config("credential.codex_access_token", codex_creds["access_token"])
            await set_config("credential.codex_refresh_token", codex_creds["refresh_token"])
            await set_config("credential.codex_expires_at", codex_creds["expires_at"])
            console.print("[green]✓ ChatGPT OAuth credentials saved to database[/green]")
        # Save Claude OAuth credentials to DB if collected
        if provider_config and provider_config.get("_claude_creds"):
            from syne.db.credentials import set_claude_oauth_credentials
            cdata = provider_config["_claude_creds"]
            await set_claude_oauth_credentials(
                access_token=cdata["access_token"],
                refresh_token=cdata["refresh_token"],
                expires_at=cdata["expires_at"],
                email=cdata.get("email"),
            )
            console.print(f"[green]✓ Claude OAuth credentials saved to database ({cdata.get('email', 'unknown')})[/green]")
        # Save API key to DB if provided (not in .env!)
        if provider_config and provider_config.get("_api_key"):
            cred_key = provider_config.get("_credential_key", f"credential.{provider_config['driver']}_api_key")
            await set_config(cred_key, provider_config["_api_key"])
            console.print("[green]✓ API key saved to database[/green]")
        # Save provider config to DB
        if provider_config:
            # Strip internal _api_key before saving
            db_config = {k: v for k, v in provider_config.items() if not k.startswith("_")}
            await set_config("provider.primary", db_config)

            # Build model registry from selected provider
            driver = provider_config["driver"]
            auth = provider_config["auth"]
            model_id = provider_config["model"]
            models_registry = []

            if driver == "vertex":
                cred_key = provider_config.get("_credential_key", "credential.vertex_ai_api_key")
                # Auto-detect region
                api_key = provider_config.get("_api_key", "")
                try:
                    import asyncio as _aio
                    from syne.llm.vertex import detect_vertex_region
                    region = _aio.get_event_loop().run_until_complete(detect_vertex_region(api_key))
                    console.print(f"  [green]Auto-detected region: {region}[/green]")
                except Exception:
                    region = "us-central1"
                    console.print(f"  [yellow]Region detection failed, using {region}[/yellow]")
                base_entry = {"key": "", "label": "", "driver": "vertex", "model_id": "", "auth": "api_key", "credential_key": cred_key, "context_window": 1048576, "region": region}
                models_registry = [
                    {**base_entry, "key": "gemini-pro", "label": "Gemini 2.5 Pro", "model_id": "gemini-2.5-pro"},
                    {**base_entry, "key": "gemini-flash", "label": "Gemini 2.5 Flash", "model_id": "gemini-2.5-flash"},
                ]
                active_key = "gemini-pro" if "pro" in model_id else "gemini-flash"
            elif driver == "google_cca":
                if auth == "api_key":
                    cred_key = provider_config.get("_credential_key", "credential.google_cca_api_key")
                    base_entry = {"key": "", "label": "", "driver": "google_cca", "model_id": "", "auth": "api_key", "credential_key": cred_key, "context_window": 1048576}
                    models_registry = [
                        {**base_entry, "key": "gemini-pro", "label": "Gemini 2.5 Pro", "model_id": "gemini-2.5-pro"},
                        {**base_entry, "key": "gemini-flash", "label": "Gemini 2.5 Flash", "model_id": "gemini-2.5-flash"},
                    ]
                else:
                    models_registry = [
                        {"key": "gemini-pro", "label": "Gemini 2.5 Pro", "driver": "google_cca", "model_id": "gemini-2.5-pro", "auth": "oauth", "context_window": 1048576},
                        {"key": "gemini-flash", "label": "Gemini 2.5 Flash", "driver": "google_cca", "model_id": "gemini-2.5-flash", "auth": "oauth", "context_window": 1048576},
                    ]
                active_key = "gemini-pro" if "pro" in model_id else "gemini-flash"
            elif driver == "codex":
                models_registry = [
                    {"key": "gpt-5.2", "label": "GPT-5.2", "driver": "codex", "model_id": "gpt-5.2", "auth": "oauth", "context_window": 1048576},
                ]
                active_key = "gpt-5.2"
            elif driver == "anthropic":
                models_registry = [
                    {"key": "claude-sonnet", "label": "Claude Sonnet 4", "driver": "anthropic", "model_id": "claude-sonnet-4-20250514", "auth": auth, "context_window": 200000},
                    {"key": "claude-opus", "label": "Claude Opus 4", "driver": "anthropic", "model_id": "claude-opus-4-20250514", "auth": auth, "context_window": 200000},
                ]
                active_key = "claude-sonnet" if "sonnet" in model_id else "claude-opus"
            elif driver == "openai_compat":
                active_key = model_id.split("/")[-1].replace(".", "-")
                base_url = provider_config.get("_base_url", "https://api.openai.com/v1")
                cred_key = "credential.openai_compat_api_key"
                models_registry = [
                    {"key": active_key, "label": model_id, "driver": "openai_compat", "model_id": model_id, "auth": auth, "base_url": base_url, "credential_key": cred_key},
                ]
            else:
                active_key = model_id
                models_registry = [
                    {"key": active_key, "label": model_id, "driver": driver, "model_id": model_id, "auth": auth},
                ]

            await set_config("provider.models", models_registry)
            await set_config("provider.active_model", active_key)
            console.print("[green]✓ Provider config saved to database[/green]")
        # Save embedding config to DB
        if embedding_config:
            await set_config("provider.embedding_model", embedding_config["model"])
            await set_config("provider.embedding_dimensions", embedding_config["dimensions"])
            await set_config("provider.embedding_driver", embedding_config["driver"])
            # Save embedding API key (not needed for Ollama)
            if embedding_config.get("_api_key"):
                cred_key = embedding_config["_credential_key"]
                await set_config(cred_key, embedding_config["_api_key"])

            # Build and save embedding model registry entry
            driver = embedding_config["driver"]
            if driver == "ollama":
                # Build registry key from model name (e.g. "qwen3-embedding:4b" → "ollama-qwen3-4b")
                model_id = embedding_config["model"]
                ollama_key_map = {
                    "qwen3-embedding:0.6b": "ollama-qwen3",
                    "qwen3-embedding:4b": "ollama-qwen3-4b",
                    "qwen3-embedding:8b": "ollama-qwen3-8b",
                }
                active_key = ollama_key_map.get(model_id, "ollama-qwen3")
                embed_entry = {
                    "key": active_key,
                    "label": f"Ollama — {model_id} (local, FREE)",
                    "driver": "ollama",
                    "model_id": model_id,
                    "auth": "none",
                    "base_url": "http://localhost:11434",
                    "dimensions": embedding_config["dimensions"],
                    "cost": "FREE (local CPU)",
                }
            elif driver == "together":
                embed_entry = {
                    "key": "together-bge",
                    "label": "Together AI — bge-base-en-v1.5",
                    "driver": "together",
                    "model_id": embedding_config["model"],
                    "auth": "api_key",
                    "credential_key": "credential.together_api_key",
                    "dimensions": embedding_config["dimensions"],
                    "cost": "~$0.008/1M tokens",
                }
                active_key = "together-bge"
            else:
                embed_entry = {
                    "key": "openai-small",
                    "label": "OpenAI — text-embedding-3-small",
                    "driver": "openai_compat",
                    "model_id": embedding_config["model"],
                    "auth": "api_key",
                    "credential_key": "credential.openai_compat_api_key",
                    "dimensions": embedding_config["dimensions"],
                    "cost": "$0.02/1M tokens",
                }
                active_key = "openai-small"
            await set_config("provider.embedding_models", [embed_entry])
            await set_config("provider.active_embedding", active_key)
            console.print("[green]✓ Embedding config saved to database[/green]")
        # Save auto_capture setting
        await set_config("memory.auto_capture", auto_capture_enabled)
        if auto_capture_enabled:
            await set_config("memory.evaluator_driver", "ollama")
            await set_config("memory.evaluator_model", selected_eval_model)
        await close_db()

    asyncio.run(_save_identity())

    console.print()
    console.print(Panel(
        f"[bold]{name}[/bold]\n[italic]{motto}[/italic]",
        title="Your Agent",
        style="green",
    ))

    # 7. Setup systemd service + start
    console.print("\n[bold]Setting up systemd service...[/bold]")
    _setup_service()

    # 8. Create symlink so `syne` works globally without activating venv
    _create_symlink()

    # 9. Setup weekly update check
    _setup_update_check()

    # Check if service actually started OK
    time.sleep(3)
    _svc_check = subprocess.run(
        ["systemctl", "--user", "is-active", "syne"],
        capture_output=True, text=True,
    )
    _svc_ok = _svc_check.stdout.strip() == "active"

    docker_group_added = os.environ.get("_SYNE_DOCKER_GROUP_ADDED") == "1"

    if _svc_ok:
        console.print("\n[bold green]Setup complete! Syne is running.[/bold green]")
        if docker_group_added:
            console.print()
            console.print(Panel(
                "[bold]Docker group was just added to your user.[/bold]\n"
                "Service works now via workaround, but for reliability:\n\n"
                "  [bold]1.[/bold] Log out (exit SSH / close terminal)\n"
                "  [bold]2.[/bold] Log back in\n"
                "  [bold]3.[/bold] [bold]syne restart[/bold]",
                title="ACTION REQUIRED",
                style="bold yellow",
            ))
    else:
        if docker_group_added:
            console.print("\n[bold yellow]Setup complete, but the service needs a session restart.[/bold yellow]")
            console.print()
            console.print(Panel(
                "[bold]Docker group was just added to your user.[/bold]\n"
                "The systemd service can't access Docker until you log out and back in.\n\n"
                "  [bold]1.[/bold] Log out (exit SSH / close terminal)\n"
                "  [bold]2.[/bold] Log back in\n"
                "  [bold]3.[/bold] [bold]syne restart[/bold]",
                title="ACTION REQUIRED",
                style="bold red",
            ))
        else:
            _journal = subprocess.run(
                ["journalctl", "--user", "-u", "syne", "-n", "10", "--no-pager", "-o", "cat"],
                capture_output=True, text=True,
            )
            _jout = (_journal.stdout or "").lower()
            _is_docker_perm = ("permission denied" in _jout or "connect: no such file" in _jout) and "docker" in _jout

            if _is_docker_perm:
                console.print("\n[bold yellow]Setup complete, but the service can't access Docker.[/bold yellow]")
                console.print()
                console.print(Panel(
                    "Try logging out and back in, then:\n\n"
                    "  [bold]syne restart[/bold]",
                    title="ACTION REQUIRED",
                    style="bold yellow",
                ))
            else:
                console.print("\n[bold yellow]Setup complete, but the service failed to start.[/bold yellow]")
                console.print("[dim]Check logs: journalctl --user -u syne -n 20 --no-pager[/dim]")
                console.print()

    console.print()
    console.print("[dim]Run [bold]syne --help[/bold] to see all available commands.[/dim]")
    console.print()
