"""Syne CLI — command line interface."""

import asyncio
import os
import secrets
import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()


@click.group()
@click.version_option(version="0.3.0", prog_name="syne")
def cli():
    """Syne — AI Agent Framework with Unlimited Memory 🧠"""
    pass


# ── Init ─────────────────────────────────────────────────────

def _docker_ok(use_sudo=False) -> bool:
    """Check if Docker daemon is reachable."""
    import subprocess
    cmd = ["sudo", "docker", "info"] if use_sudo else ["docker", "info"]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ensure_ollama():
    """Ensure Ollama is installed and qwen3-embedding model is pulled.

    Skips download if already present.
    """
    import shutil
    import subprocess
    import time

    model_name = "qwen3-embedding:0.6b"

    # 1. Install Ollama if missing
    if not shutil.which("ollama"):
        console.print("[bold yellow]Ollama is not installed — installing now...[/bold yellow]")
        ret = os.system("curl -fsSL https://ollama.com/install.sh | sh")
        if ret != 0:
            console.print("[red]Ollama installation failed.[/red]")
            console.print("[dim]Install manually: https://ollama.com/download[/dim]")
            raise SystemExit(1)
        console.print("[green]✓ Ollama installed[/green]")

    # 2. Ensure Ollama server is running
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5,
        )
        server_running = result.returncode == 0
    except Exception:
        server_running = False

    if not server_running:
        console.print("[dim]Starting Ollama server...[/dim]")
        # Try systemd (with sudo fallback), then direct serve
        ret = os.system("systemctl start ollama 2>/dev/null || sudo systemctl start ollama 2>/dev/null || (ollama serve &>/dev/null &)")
        time.sleep(3)

    # 3. Check if model already exists
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        if model_name.split(":")[0] in result.stdout:
            console.print(f"[green]✓ Ollama model {model_name} already available[/green]")
            return
    except Exception:
        pass

    # 4. Pull model
    console.print(f"[bold]Downloading {model_name} (~700MB)...[/bold]")
    console.print("[dim]This may take a few minutes depending on your connection.[/dim]")
    ret = os.system(f"ollama pull {model_name}")
    if ret != 0:
        console.print(f"[red]Failed to pull {model_name}.[/red]")
        console.print("[dim]Try manually: ollama pull qwen3-embedding:0.6b[/dim]")
        raise SystemExit(1)
    console.print(f"[green]✓ Model {model_name} ready[/green]")


def _ensure_docker() -> str:
    """Ensure Docker is installed, running, and accessible.

    Returns the docker command prefix: '' or 'sudo ' depending on permissions.
    """
    import shutil
    import time
    import getpass

    # 1. Install if missing
    if not shutil.which("docker"):
        console.print("[bold yellow]Docker is not installed — installing now...[/bold yellow]")
        ret = os.system("curl -fsSL https://get.docker.com | sh")
        if ret != 0:
            console.print("[red]Docker installation failed.[/red]")
            console.print("[dim]Install manually: https://docs.docker.com/get-docker/[/dim]")
            raise SystemExit(1)
        # Add user to docker group, start Docker
        current_user = getpass.getuser()
        os.system(f"sudo usermod -aG docker {current_user}")
        os.system("sudo systemctl daemon-reload")
        os.system("sudo systemctl start docker")
        os.system("sudo systemctl enable docker > /dev/null 2>&1")
        console.print(f"[green]✓ Docker installed, {current_user} added to docker group[/green]")
        # Re-exec syne init with docker group active
        console.print("[dim]Activating docker group and restarting init...[/dim]\n")
        os.execvp("sg", ["sg", "docker", "-c", "syne init"])
        # execvp replaces current process — code below won't run

    # 2. Start daemon if not running
    if not _docker_ok() and not _docker_ok(use_sudo=True):
        console.print("[dim]Starting Docker daemon...[/dim]")
        os.system("sudo systemctl daemon-reload")
        os.system("sudo systemctl start docker")
        os.system("sudo systemctl enable docker > /dev/null 2>&1")
        for _ in range(20):
            time.sleep(2)
            if _docker_ok() or _docker_ok(use_sudo=True):
                break
        else:
            console.print("[bold red]Docker daemon failed to start.[/bold red]")
            console.print("[dim]Run: sudo systemctl status docker[/dim]")
            raise SystemExit(1)

    # 3. Determine if we need sudo prefix
    if _docker_ok():
        prefix = ""
    elif _docker_ok(use_sudo=True):
        prefix = "sudo "
    else:
        console.print("[bold red]Cannot connect to Docker.[/bold red]")
        raise SystemExit(1)

    console.print("[green]✓ Docker ready[/green]")
    if prefix:
        console.print("[dim]  (using sudo — after reboot 'sudo' won't be needed)[/dim]")
    console.print()
    return prefix


def _create_symlink():
    """Create ~/.local/bin/syne symlink so `syne` works without activating venv."""
    syne_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "syne")

    if not os.path.exists(syne_bin):
        # Fallback: find syne in the current venv
        import sys
        venv_bin = os.path.join(os.path.dirname(sys.executable), "syne")
        if os.path.exists(venv_bin):
            syne_bin = venv_bin
        else:
            console.print("[yellow]⚠ Could not find syne binary to create symlink[/yellow]")
            return

    local_bin = os.path.expanduser("~/.local/bin")
    os.makedirs(local_bin, exist_ok=True)
    target = os.path.join(local_bin, "syne")

    # Check if symlink already exists and points to correct target
    if os.path.islink(target) and os.readlink(target) == syne_bin:
        pass  # Already correct
    else:
        # Remove old symlink/file if exists
        if os.path.exists(target) or os.path.islink(target):
            os.remove(target)
        os.symlink(syne_bin, target)
        console.print(f"[green]✓[/green] Created symlink: {target} → {syne_bin}")

    # Ensure ~/.local/bin is in PATH
    path_line = 'export PATH="$HOME/.local/bin:$PATH"'
    shell_rc = os.path.expanduser("~/.bashrc")
    if os.path.exists(os.path.expanduser("~/.zshrc")):
        shell_rc = os.path.expanduser("~/.zshrc")

    try:
        with open(shell_rc, "r") as f:
            content = f.read()
        if path_line not in content and ".local/bin" not in content:
            with open(shell_rc, "a") as f:
                f.write(f"\n# Syne CLI\n{path_line}\n")
            console.print(f"[green]✓[/green] Added ~/.local/bin to PATH in {os.path.basename(shell_rc)}")
    except Exception:
        console.print(f"[yellow]⚠ Add to your shell rc: {path_line}[/yellow]")


def _setup_service():
    """Setup systemd user service, enable, and start Syne."""
    import subprocess
    import getpass
    from pathlib import Path

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "syne.service"
    syne_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(syne_dir, ".venv", "bin", "python")
    env_file = os.path.join(syne_dir, ".env")

    service_dir.mkdir(parents=True, exist_ok=True)

    # DB container has restart: unless-stopped — Docker auto-starts it.
    # No ExecStartPre needed (avoids sudo/docker-group issues on fresh install).
    service_content = f"""[Unit]
Description=Syne AI Agent
After=network.target docker.service

[Service]
Type=simple
WorkingDirectory={syne_dir}
EnvironmentFile={env_file}
ExecStart={venv_python} -m syne.main
Restart=on-failure
RestartSec=10
StartLimitIntervalSec=120
StartLimitBurst=5

[Install]
WantedBy=default.target
"""

    service_file.write_text(service_content)
    console.print(f"  [green]✓ Service file: {service_file}[/green]")

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    subprocess.run(["systemctl", "--user", "enable", "syne"], capture_output=True)
    console.print("  [green]✓ Autostart enabled[/green]")

    # Enable linger so service runs without active login session
    username = getpass.getuser()
    result = subprocess.run(["loginctl", "enable-linger", username], capture_output=True)
    if result.returncode == 0:
        console.print(f"  [green]✓ Linger enabled for {username}[/green]")

    # Start the service
    console.print("  Starting Syne...")
    result = subprocess.run(["systemctl", "--user", "start", "syne"], capture_output=True, text=True)
    if result.returncode == 0:
        import time
        time.sleep(2)
        # Check if actually running
        check = subprocess.run(["systemctl", "--user", "is-active", "syne"], capture_output=True, text=True)
        if check.stdout.strip() == "active":
            console.print("  [green]✓ Syne is running![/green]")
        else:
            console.print("  [yellow]⚠ Service started but may have exited. Check logs:[/yellow]")
            console.print("    journalctl --user -u syne --no-pager -n 20")
    else:
        console.print(f"  [yellow]⚠ Could not start service: {result.stderr.strip()}[/yellow]")
        console.print("    Try: systemctl --user start syne")


@cli.command()
def init():
    """Initialize Syne: authenticate, setup database, configure."""
    console.print(Panel("[bold]Welcome to Syne 🧠[/bold]\nAI Agent Framework with Unlimited Memory", style="blue"))
    console.print()

    # Pre-check: Docker (install + start + determine sudo prefix)
    docker_prefix = _ensure_docker()

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
    console.print("  7. Groq [yellow](API key, free tier available)[/yellow]")
    console.print()

    choice = click.prompt("Select provider", type=click.IntRange(1, 7), default=1)

    env_lines = []
    provider_config = None  # Will be saved to DB after schema init
    google_creds = None  # Only set for Google OAuth provider
    codex_creds = None  # Only set for Codex OAuth provider

    if choice == 1:
        console.print("\n[bold green]✓ Google Gemini selected (OAuth)[/bold green]")
        console.print("  [dim]Sign in with your Google account to get free Gemini access.[/dim]")

        from .auth.google_oauth import login_google
        google_creds = asyncio.run(login_google())
        # Credentials saved to DB after schema init (Step 6)

        provider_config = {"driver": "google_cca", "model": "gemini-2.5-pro", "auth": "oauth"}

    elif choice == 2:
        console.print("\n[bold green]✓ ChatGPT selected (Codex OAuth)[/bold green]")
        console.print("  [dim]Requires ChatGPT Plus/Pro/Team subscription.[/dim]")

        from .auth.codex_oauth import login_codex
        codex_creds = asyncio.run(login_codex())
        # Credentials saved to DB after schema init (Step 6)

        provider_config = {"driver": "codex", "model": "gpt-5.2", "auth": "oauth"}

    elif choice == 3:
        console.print("\n[bold green]✓ Claude selected (OAuth)[/bold green]")
        console.print("  [dim]Requires claude.ai Pro/Max subscription.[/dim]")
        
        # Check if Claude CLI credentials exist
        cred_path = os.path.expanduser("~/.claude/.credentials.json")
        if os.path.exists(cred_path):
            console.print(f"  [green]✓ Found credentials at {cred_path}[/green]")
        else:
            console.print("\n  [yellow]⚠ Claude credentials not found.[/yellow]")
            console.print("  [dim]You need to install Claude CLI and login first:[/dim]")
            console.print("  [bold]1.[/bold] npm install -g @anthropic-ai/claude-code")
            console.print("  [bold]2.[/bold] claude login")
            console.print(f"  [dim]Syne reads tokens from {cred_path}[/dim]")
            
            proceed = click.confirm("\n  Have you completed claude login?", default=False)
            if not proceed:
                console.print("  [dim]Run 'claude login' first, then re-run 'syne init'.[/dim]")
                raise SystemExit(1)
            
            # Re-check after user says yes
            if not os.path.exists(cred_path):
                console.print(f"  [red]✗ Still not found: {cred_path}[/red]")
                console.print("  [dim]Please run 'claude login' and try again.[/dim]")
                raise SystemExit(1)
            
            console.print(f"  [green]✓ Found credentials at {cred_path}[/green]")
        
        provider_config = {"driver": "anthropic", "model": "claude-sonnet-4-20250514", "auth": "oauth"}

    elif choice == 4:
        console.print("\n[bold green]✓ OpenAI selected (API key)[/bold green]")
        api_key = click.prompt("OpenAI API key")
        provider_config = {"driver": "openai_compat", "model": "gpt-4o", "auth": "api_key", "_api_key": api_key}

    elif choice == 5:
        console.print("\n[bold green]✓ Anthropic Claude selected (API key)[/bold green]")
        api_key = click.prompt("Anthropic API key")
        provider_config = {"driver": "anthropic", "model": "claude-sonnet-4-20250514", "auth": "api_key", "_api_key": api_key}

    elif choice == 6:
        console.print("\n[bold green]✓ Together AI selected (API key)[/bold green]")
        api_key = click.prompt("Together API key")
        provider_config = {"driver": "openai_compat", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "auth": "api_key", "_api_key": api_key}

    elif choice == 7:
        console.print("\n[bold green]✓ Groq selected (API key)[/bold green]")
        console.print("  [dim]Get your key at console.groq.com[/dim]")
        api_key = click.prompt("Groq API key")
        provider_config = {"driver": "openai_compat", "model": "llama-3.3-70b-versatile", "auth": "api_key", "_api_key": api_key}

    # 2. Embedding provider (for memory)
    console.print("\n[bold]Step 2: Choose embedding provider (for memory)[/bold]")
    console.print()
    # Detect system resources for Ollama eligibility
    ollama_available = True
    ollama_reason = ""
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
    sys_disk_gb = 0
    try:
        import shutil as _shutil
        disk = _shutil.disk_usage("/")
        sys_disk_gb = disk.free / (1024 ** 3)
    except Exception:
        pass

    min_cpu, min_ram, min_disk = 2, 2.0, 2.0
    reasons = []
    if sys_cpu < min_cpu:
        reasons.append(f"{sys_cpu} CPU (need {min_cpu}+)")
    if sys_ram_gb < min_ram:
        reasons.append(f"{sys_ram_gb:.1f}GB RAM (need {min_ram:.0f}GB+)")
    if sys_disk_gb < min_disk:
        reasons.append(f"{sys_disk_gb:.1f}GB disk (need {min_disk:.0f}GB+)")
    if reasons:
        ollama_available = False
        ollama_reason = ", ".join(reasons)

    # Show options — Ollama recommended if system supports it
    if ollama_available:
        console.print("  1. Together AI [dim](~$0.008/1M tokens)[/dim]")
        console.print("  2. OpenAI [dim](~$0.02/1M tokens)[/dim]")
        console.print("  3. Ollama [green](recommended — FREE, local, qwen3-embedding:0.6b)[/green]")
        console.print("     [dim]Trade-off: ~1.3GB RAM when active, CPU burst during embedding[/dim]")
        default_embed = 3
    else:
        console.print("  1. Together AI [green](recommended — ~$0.008/1M tokens)[/green]")
        console.print("  2. OpenAI [yellow](~$0.02/1M tokens)[/yellow]")
        console.print(f"  3. Ollama (FREE — local)  [red dim]\\[unavailable][/red dim]")
        console.print(f"     [dim]⚠️  Your system: {ollama_reason}[/dim]")
        console.print(f"     [dim]Minimum: {min_cpu} CPU, {min_ram:.0f}GB RAM, {min_disk:.0f}GB disk[/dim]")
        default_embed = 1
    console.print()

    embed_choice = click.prompt("Select embedding provider", type=click.IntRange(1, 3), default=default_embed)
    embedding_config = None

    # Block unavailable Ollama selection
    if embed_choice == 3 and not ollama_available:
        console.print(f"\n[red]Ollama not available on this system: {ollama_reason}[/red]")
        console.print("[dim]Please select option 1 or 2.[/dim]")
        embed_choice = click.prompt("Select embedding provider", type=click.IntRange(1, 2), default=1)

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
    elif embed_choice == 3:
        console.print("\n[bold green]✓ Ollama selected for embeddings (FREE, local)[/bold green]")
        _ensure_ollama()
        embedding_config = {
            "driver": "ollama",
            "model": "qwen3-embedding:0.6b",
            "dimensions": 1024,
            "_ollama": True,  # Flag: no API key needed
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
    console.print("  Brave Search API lets Syne search the web. Free tier: 2,000 queries/month.")
    console.print()
    console.print("  [bold cyan]How to get a free API key:[/bold cyan]")
    console.print("  1. Go to [link]https://brave.com/search/api/[/link]")
    console.print("  2. Click \"Get Started\" → sign up (free)")
    console.print("  3. Create an app → copy the API key")
    console.print()
    console.print("  [dim]Press Enter to skip — you can add it later via chat.[/dim]")
    console.print()
    brave_api_key = click.prompt("Brave Search API key (optional)", default="", show_default=False)
    brave_api_key = brave_api_key.strip()
    if brave_api_key:
        console.print("[green]✓ Brave Search API key saved[/green]")
    else:
        console.print("[dim]  Skipped — tell Syne to \"enable web search\" later to configure.[/dim]")

    # 4. Database (Docker only — no external DB option)
    console.print("\n[bold]Step 4: Database[/bold]")
    console.print("  Syne uses its own isolated PostgreSQL + pgvector container.")
    console.print("  [dim]This ensures your API keys, OAuth tokens, and memories stay private.[/dim]")
    console.print("  [dim]Docker is required — PostgreSQL is bundled, not optional.[/dim]")

    # Reuse existing credentials if .env exists, otherwise generate new ones
    syne_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    env_path = os.path.join(syne_dir, ".env")
    existing_env = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    existing_env[k.strip()] = v.strip()

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
    console.print(f"  [green]✓ Database credentials generated (unique to this install)[/green]")

    # 4. Write .env (always in project root, where docker-compose.yml lives)
    env_content = "\n".join(env_lines) + "\n"
    env_path = os.path.join(syne_dir, ".env")

    with open(env_path, "w") as f:
        f.write(env_content)
    os.chmod(env_path, 0o600)
    console.print(f"[green]✓ .env written (chmod 600)[/green]")

    # 5. Start DB (Docker — uses prefix from _ensure_docker)
    docker_cmd = f"{docker_prefix}docker compose"

    # PostgreSQL bakes credentials into the volume at FIRST creation.
    # Subsequent starts with different POSTGRES_USER/PASSWORD are IGNORED.
    # So we must always clean up stale volumes before starting.
    import subprocess
    
    # Check for existing volume (docker volume, not docker compose)
    vol_cmd = f"{docker_prefix}docker volume ls -q"
    volume_check = subprocess.run(
        vol_cmd, shell=True, capture_output=True, text=True,
    )
    volume_exists = "syne_syne_pgdata" in (volume_check.stdout or "")

    if volume_exists:
        console.print("[yellow]⚠️  Found existing Syne DB volume — removing to apply fresh credentials...[/yellow]")
        # Stop container + remove volume (credentials are baked in, can't change)
        os.system(f"cd {syne_dir} && {docker_cmd} down -v 2>/dev/null")

    console.print("\n[bold]Starting database...[/bold]")
    result = os.system(f"cd {syne_dir} && {docker_cmd} up -d db")
    if result != 0:
        console.print("[red]Failed to start database.[/red]")
        return

    # Wait for DB
    console.print("Waiting for PostgreSQL to be ready...")
    import time
    for _ in range(20):
        result = os.system(f"cd {syne_dir} && {docker_cmd} exec -T db pg_isready -U {db_user} > /dev/null 2>&1")
        if result == 0:
            break
        time.sleep(2)
    else:
        console.print("[red]Database didn't start in time.[/red]")
        return

    console.print("[green]✓ Database ready[/green]")

    # 6. Initialize schema + Agent identity
    console.print("\n[bold]Step 5: Initialize database & agent identity...[/bold]")
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

    console.print()  # Agent identity (part of Step 5)
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
        # Save Brave Search API key to DB if provided
        if brave_api_key:
            from .db.models import set_config
            await set_config("web_search.api_key", brave_api_key)
            console.print("[green]✓ Brave Search API key saved to database[/green]")
        # Save Google OAuth credentials to DB if collected
        if google_creds:
            await google_creds.save_to_db()
            console.print(f"[green]✓ Google OAuth credentials saved to database ({google_creds.email})[/green]")
        # Save Codex OAuth credentials to DB if collected
        if codex_creds:
            from .db.models import set_config
            await set_config("credential.codex_access_token", codex_creds["access_token"])
            await set_config("credential.codex_refresh_token", codex_creds["refresh_token"])
            await set_config("credential.codex_expires_at", codex_creds["expires_at"])
            console.print("[green]✓ ChatGPT OAuth credentials saved to database[/green]")
        # Save API key to DB if provided (not in .env!)
        if provider_config and provider_config.get("_api_key"):
            from .db.models import set_config
            driver = provider_config["driver"]
            await set_config(f"credential.{driver}_api_key", provider_config["_api_key"])
            console.print(f"[green]✓ API key saved to database[/green]")
        # Save provider config to DB
        if provider_config:
            from .db.models import set_config
            # Strip internal _api_key before saving
            db_config = {k: v for k, v in provider_config.items() if not k.startswith("_")}
            await set_config("provider.primary", db_config)

            # Build model registry from selected provider
            driver = provider_config["driver"]
            auth = provider_config["auth"]
            model_id = provider_config["model"]
            models_registry = []

            if driver == "google_cca":
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
                # OpenAI API key, Together AI, or Groq — needs base_url + credential_key
                active_key = model_id.split("/")[-1].replace(".", "-")  # e.g. "gpt-4o", "llama-3-3-70b-versatile"
                # Determine base_url and credential_key from model/provider context
                if "groq" in model_id or "groq" in provider_config.get("_api_key", ""):
                    base_url = "https://api.groq.com/openai/v1"
                    cred_key = "credential.openai_compat_api_key"
                elif "together" in model_id.lower() or "llama" in model_id.lower() or "meta-llama" in model_id:
                    base_url = "https://api.together.xyz/v1"
                    cred_key = "credential.openai_compat_api_key"
                else:
                    base_url = "https://api.openai.com/v1"
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
            console.print(f"[green]✓ Provider config saved to database[/green]")
        # Save embedding config to DB
        if embedding_config:
            from .db.models import set_config
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
                embed_entry = {
                    "key": "ollama-qwen3",
                    "label": "Ollama — qwen3-embedding:0.6b (local, FREE)",
                    "driver": "ollama",
                    "model_id": embedding_config["model"],
                    "auth": "none",
                    "base_url": "http://localhost:11434",
                    "dimensions": embedding_config["dimensions"],
                    "cost": "FREE (local CPU)",
                }
                active_key = "ollama-qwen3"
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
            console.print(f"[green]✓ Embedding config saved to database[/green]")
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

    console.print("\n[bold green]✅ Setup complete! Syne is running.[/bold green]")
    console.print()
    console.print("Commands:")
    console.print("  [bold]syne cli[/bold]                        # Interactive chat")
    console.print("  [bold]syne status[/bold]                     # Check agent status")
    console.print("  [bold]systemctl --user status syne[/bold]    # Service status")
    console.print("  [bold]systemctl --user restart syne[/bold]   # Restart")
    console.print("  [bold]journalctl --user -u syne -f[/bold]   # Logs")
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


@cli.command(name="cli")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli_mode(debug):
    """Interactive CLI chat (like Claude Code)."""
    from .channels.cli_channel import run_cli
    asyncio.run(run_cli(debug=debug))


# ── Status ───────────────────────────────────────────────────

@cli.command()
def status():
    """Show Syne status."""
    async def _status():
        from .config import load_settings
        from .db.connection import init_db, close_db

        settings = load_settings()

        from . import __version__ as syne_version

        table = Table(title=f"🧠 Syne Status · v{syne_version}", show_header=False, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Version", syne_version)

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

        # Provider (from DB)
        try:
            from .db.models import get_config
            async def _get_prov():
                p = await get_config("provider.primary")
                return p if p else "not configured"
            provider = asyncio.run(_get_prov())
            if isinstance(provider, dict):
                table.add_row("Provider", f"{provider.get('driver', '?')} ({provider.get('auth', '?')})")
            else:
                table.add_row("Provider", str(provider))
        except Exception:
            table.add_row("Provider", "not configured")

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
        try:
            from .auth.google_oauth import get_credentials
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

    if not enable:
        service_dir = Path.home() / ".config" / "systemd" / "user"
        service_file = service_dir / "syne.service"
        subprocess.run(["systemctl", "--user", "stop", "syne"], capture_output=True)
        subprocess.run(["systemctl", "--user", "disable", "syne"], capture_output=True)
        if service_file.exists():
            service_file.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        console.print("[yellow]Autostart disabled.[/yellow]")
        return

    _setup_service()
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
    """Quick provider init for CLI commands. Reads config from DB."""
    from .llm.drivers import create_chat_provider

    async def _make():
        return await create_chat_provider()

    try:
        return asyncio.run(_make())
    except Exception as e:
        console.print(f"[red]Failed to initialize provider: {e}[/red]")
        console.print("[dim]Run 'syne init' to configure a provider.[/dim]")
        sys.exit(1)


@cli.command()
def update():
    """Pull latest code and reinstall."""
    import subprocess

    syne_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_dir = os.path.join(syne_dir, ".venv")
    venv_pip = os.path.join(venv_dir, "bin", "pip")
    venv_syne = os.path.join(venv_dir, "bin", "syne")

    console.print("📥 Pulling latest code...")
    result = subprocess.run(["git", "pull"], cwd=syne_dir, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Git pull failed: {result.stderr}[/red]")
        return
    console.print(f"[dim]{result.stdout.strip()}[/dim]")

    console.print("🔧 Setting up virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], cwd=syne_dir)

    console.print("📦 Installing...")
    result = subprocess.run([venv_pip, "install", "-e", ".", "-q"], cwd=syne_dir, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Install failed: {result.stderr}[/red]")
        return

    # Ensure syne is callable from anywhere
    local_bin = os.path.expanduser("~/.local/bin")
    os.makedirs(local_bin, exist_ok=True)
    target = os.path.join(local_bin, "syne")
    if os.path.exists(venv_syne):
        if os.path.exists(target) or os.path.islink(target):
            os.remove(target)
        os.symlink(venv_syne, target)

    # Ensure ~/.local/bin is in PATH
    path_line = 'export PATH="$HOME/.local/bin:$PATH"'
    shell_rc = os.path.expanduser("~/.bashrc")
    if os.path.exists(os.path.expanduser("~/.zshrc")):
        shell_rc = os.path.expanduser("~/.zshrc")
    try:
        with open(shell_rc, "r") as f:
            content = f.read()
        if path_line not in content and ".local/bin" not in content:
            with open(shell_rc, "a") as f:
                f.write(f"\n# Syne CLI\n{path_line}\n")
    except Exception:
        pass

    console.print("[green]✅ Syne updated! Run: syne cli[/green]")


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
