"""Helper functions for Syne CLI — installation, Docker, Ollama, service management."""

import os
import subprocess
import sys

from .shared import console, _strip_env_quotes, _get_db_url


def _docker_ok(use_sudo=False) -> bool:
    """Check if Docker daemon is reachable."""
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
    import time

    model_name = "qwen3-embedding:0.6b"

    # 1. Install Ollama if missing
    if not shutil.which("ollama"):
        console.print("[bold yellow]Ollama is not installed — installing now...[/bold yellow]")
        console.print("[dim]This downloads ~100MB and requires sudo for installation.[/dim]")
        sys.stdout.flush()
        ret = subprocess.run("curl -fL https://ollama.com/install.sh | sh", shell=True).returncode
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
        sys.stdout.flush()

        # Try multiple strategies in order
        strategies = [
            # 1. systemctl without sudo (user may have permissions)
            (["systemctl", "start", "ollama"], False),
            # 2. sudo systemctl (ollama official install = root systemd service)
            (["sudo", "systemctl", "start", "ollama"], True),
        ]

        for cmd, needs_sudo in strategies:
            try:
                if needs_sudo:
                    console.print("[dim]  Trying sudo (you may be prompted for password)...[/dim]")
                    sys.stdout.flush()
                r = subprocess.run(cmd, timeout=15)
                if r.returncode == 0:
                    time.sleep(2)
                    break
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue

        # 3. Final fallback: ollama serve directly in background
        # Check if server is up after systemctl attempts
        try:
            r = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                server_running = True
        except Exception:
            pass

        if not server_running:
            # Direct serve as background process
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait until server is actually ready (up to 20s)
        if not server_running:
            for i in range(20):
                time.sleep(1)
                try:
                    r = subprocess.run(
                        ["ollama", "list"], capture_output=True, text=True, timeout=5,
                    )
                    if r.returncode == 0:
                        server_running = True
                        break
                except Exception:
                    pass

        if not server_running:
            console.print("[red]Could not start Ollama server.[/red]")
            console.print("[dim]Try: sudo systemctl start ollama[/dim]")
            raise SystemExit(1)

    # 3. Check if model already exists
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        if any(line.startswith(model_name) for line in result.stdout.splitlines()):
            console.print(f"[green]✓ Ollama model {model_name} already available[/green]")
            return
    except Exception:
        pass

    # 4. Pull model
    console.print(f"\n[bold]Downloading {model_name} (~700MB)...[/bold]")
    console.print("[dim]This may take a few minutes depending on your connection.[/dim]")
    sys.stdout.flush()
    sys.stderr.flush()
    ret = subprocess.run(["ollama", "pull", model_name]).returncode
    if ret != 0:
        console.print(f"[red]Failed to pull {model_name}.[/red]")
        console.print("[dim]Try manually: ollama pull qwen3-embedding:0.6b[/dim]")
        raise SystemExit(1)
    console.print(f"[green]✓ Model {model_name} ready[/green]")


def _ensure_evaluator_model():
    """Ensure the Ollama evaluator model (qwen3:0.6b) is available.

    Non-fatal: if Ollama is not installed or pull fails, just warn.
    The evaluator model is small (~500MB) and used for memory auto_capture.
    """
    import shutil

    model_name = "qwen3:0.6b"

    if not shutil.which("ollama"):
        console.print("[dim]⏭ Ollama not installed — skipping evaluator model[/dim]")
        return

    # Check if model already exists
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        if any(line.startswith(model_name) for line in result.stdout.splitlines()):
            console.print(f"[green]✓ Evaluator model {model_name} already available[/green]")
            return
    except Exception:
        pass

    # Pull model
    console.print(f"\n[bold]Downloading evaluator model {model_name} (~500MB)...[/bold]")
    console.print("[dim]Used for memory auto-capture (local, no API costs).[/dim]")
    sys.stdout.flush()
    sys.stderr.flush()
    ret = subprocess.run(["ollama", "pull", model_name]).returncode
    if ret != 0:
        console.print(f"[yellow]⚠️ Failed to pull {model_name} — auto_capture will need manual setup.[/yellow]")
        console.print(f"[dim]Try manually: ollama pull {model_name}[/dim]")
        return
    console.print(f"[green]✓ Evaluator model {model_name} ready[/green]")


def _ensure_evaluator_if_enabled(syne_dir: str):
    """Check DB if auto_capture is ON, and if so ensure Ollama + evaluator model.

    Called during `syne update` / `syne updatedev`. Reads the config table
    directly via asyncpg to avoid importing the full Syne stack.
    Non-fatal: if DB is unreachable or auto_capture is OFF, silently skip.
    """
    import asyncio

    db_url = _get_db_url(syne_dir)
    if not db_url:
        return

    async def _check():
        import asyncpg
        conn = await asyncpg.connect(db_url)
        try:
            row = await conn.fetchrow(
                "SELECT value FROM config WHERE key = 'memory.auto_capture'"
            )
            if row:
                import json
                val = json.loads(row["value"])
                return bool(val)
            return False
        finally:
            await conn.close()

    try:
        enabled = asyncio.run(_check())
    except Exception:
        return

    if enabled:
        console.print("[dim]Auto-capture is enabled — ensuring Ollama + evaluator model...[/dim]")
        try:
            _ensure_ollama()
        except SystemExit:
            console.print("[yellow]⚠️ Ollama setup failed — auto_capture may not work until fixed.[/yellow]")
            return
        _ensure_evaluator_model()
    else:
        console.print("[dim]⏭ Auto-capture disabled — skipping evaluator model[/dim]")


def _ensure_system_deps():
    """Ensure python3-venv and pip are available. Install via apt if missing."""
    import shutil

    missing = []

    # Check venv module
    try:
        import venv  # noqa: F401
    except ImportError:
        missing.append("python3-venv")

    # Check pip
    if not shutil.which("pip") and not shutil.which("pip3"):
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True, timeout=5,
            )
        except Exception:
            missing.append("python3-pip")

    if not missing:
        return

    pkg_list = " ".join(missing)
    console.print(f"[bold yellow]Missing system packages: {pkg_list}[/bold yellow]")
    console.print("[dim]Installing via apt (sudo required)...[/dim]")

    sys.stdout.flush()
    ret = subprocess.run(f"sudo apt-get update -q && sudo apt-get install -y -q {pkg_list}", shell=True).returncode
    if ret != 0:
        console.print(f"[red]Failed to install {pkg_list}.[/red]")
        console.print(f"[dim]Install manually: sudo apt install {pkg_list}[/dim]")
        raise SystemExit(1)

    console.print(f"[green]✓ Installed {pkg_list}[/green]")
    console.print()


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
        console.print("[dim]This downloads Docker packages — may take a few minutes.[/dim]")
        sys.stdout.flush()
        ret = subprocess.run("curl -fL https://get.docker.com | sh", shell=True).returncode
        if ret != 0:
            console.print("[red]Docker installation failed.[/red]")
            console.print("[dim]Install manually: https://docs.docker.com/get-docker/[/dim]")
            raise SystemExit(1)
        # Add user to docker group, start Docker
        current_user = getpass.getuser()
        subprocess.run(["sudo", "usermod", "-aG", "docker", current_user])
        subprocess.run(["sudo", "systemctl", "daemon-reload"])
        subprocess.run(["sudo", "systemctl", "start", "docker"])
        subprocess.run(["sudo", "systemctl", "enable", "docker"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print(f"[green]✓ Docker installed, {current_user} added to docker group[/green]")
        # Re-exec syne init with docker group active
        console.print("[dim]Activating docker group and restarting init...[/dim]\n")
        os.environ["_SYNE_DOCKER_GROUP_ADDED"] = "1"
        os.execvp("sg", ["sg", "docker", "-c", "_SYNE_DOCKER_GROUP_ADDED=1 syne init"])
        # execvp replaces current process — code below won't run

    # 1b. Docker exists but user not in docker group
    current_user = getpass.getuser()
    import grp
    try:
        docker_members = grp.getgrnam("docker").gr_mem
    except KeyError:
        docker_members = []
    if current_user not in docker_members and not _docker_ok():
        console.print(f"[dim]Adding {current_user} to docker group...[/dim]")
        subprocess.run(["sudo", "usermod", "-aG", "docker", current_user])
        console.print("[dim]Activating docker group...[/dim]\n")
        os.environ["_SYNE_DOCKER_GROUP_ADDED"] = "1"
        os.execvp("sg", ["sg", "docker", "-c", "_SYNE_DOCKER_GROUP_ADDED=1 syne init"])

    # 2. Start daemon if not running
    if not _docker_ok() and not _docker_ok(use_sudo=True):
        console.print("[dim]Starting Docker daemon...[/dim]")
        subprocess.run(["sudo", "systemctl", "daemon-reload"])
        subprocess.run(["sudo", "systemctl", "start", "docker"])
        subprocess.run(["sudo", "systemctl", "enable", "docker"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    syne_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".venv", "bin", "syne")

    if not os.path.exists(syne_bin):
        # Fallback: find syne in the current venv
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


def _setup_update_check():
    """Create weekly update check scheduled task in DB."""
    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_python = os.path.join(syne_dir, ".venv", "bin", "python")

    # Run a small Python script to create the task via DB
    script = """
import asyncio, os
async def setup():
    from syne.db.connection import init_db, close_db
    db_url = os.environ.get("SYNE_DATABASE_URL")
    if not db_url:
        return  # No DB config — skip silently
    await init_db(db_url)

    # Check if task already exists
    from syne.scheduler import list_tasks
    tasks = await list_tasks()
    for t in tasks:
        if t.get("name") == "_syne_update_check":
            await close_db()
            return  # Already exists

    # Get owner user ID for notifications
    from syne.db.connection import get_connection
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT platform_id FROM users WHERE access_level = 'owner' AND platform = 'telegram' LIMIT 1"
        )
    created_by = int(row["platform_id"]) if row else None

    # Create weekly cron task (every Monday at 9:00 AM)
    from syne.scheduler import create_task
    result = await create_task(
        name="_syne_update_check",
        schedule_type="cron",
        schedule_value="0 9 * * 1",
        payload="__syne_update_check__",
        created_by=created_by,
    )
    await close_db()

asyncio.run(setup())
"""
    result = subprocess.run(
        [venv_python, "-c", script],
        cwd=syne_dir, capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("[green]✓[/green] Weekly update check scheduled")
    else:
        console.print(f"[yellow]⚠ Could not setup update check: {result.stderr.strip()[:100]}[/yellow]")


def _setup_service():
    """Setup systemd user service, enable, and start Syne."""
    import getpass
    from pathlib import Path

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "syne.service"
    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_python = os.path.join(syne_dir, ".venv", "bin", "python")
    env_file = os.path.join(syne_dir, ".env")

    service_dir.mkdir(parents=True, exist_ok=True)

    # Find docker binary path (may be /usr/bin/docker or /usr/local/bin/docker)
    import shutil as _shutil
    docker_bin = _shutil.which("docker") or "/usr/bin/docker"

    # Create a helper script for ExecStartPre
    helper_script = os.path.join(syne_dir, "start-db.sh")
    helper_content = f"""#!/bin/bash
# Start DB container for Syne service
COMPOSE="{docker_bin} compose -f {syne_dir}/docker-compose.yml"

# 1. Try direct (works if systemd session has docker group)
$COMPOSE up -d db 2>/dev/null && exit 0

# 2. Try via sg (activates docker group for this subprocess)
sg docker -c "$COMPOSE up -d db" 2>/dev/null && exit 0

# 3. Check if DB is already running anyway
$COMPOSE ps db 2>/dev/null | grep -q running && exit 0
sg docker -c "$COMPOSE ps db" 2>/dev/null | grep -q running && exit 0

echo "ERROR: Cannot start DB container. Try: logout, login, then: systemctl --user restart syne" >&2
exit 1
"""
    with open(helper_script, "w") as f:
        f.write(helper_content)
    os.chmod(helper_script, 0o755)

    service_content = f"""[Unit]
Description=Syne AI Agent
After=network.target docker.service

[Service]
Type=simple
WorkingDirectory={syne_dir}
EnvironmentFile={env_file}
ExecStartPre=/bin/bash {helper_script}
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


def _run_schema_migration(syne_dir: str):
    """Run schema.sql to apply any new tables/columns.

    Safe to run repeatedly — uses CREATE TABLE IF NOT EXISTS
    and DO $$ ALTER TABLE ADD COLUMN IF NOT EXISTS patterns.
    """
    import asyncio

    schema_path = os.path.join(syne_dir, "syne", "db", "schema.sql")
    if not os.path.exists(schema_path):
        return

    db_url = _get_db_url(syne_dir)
    if not db_url:
        console.print("[dim]⏭ Schema migration skipped (no DB config)[/dim]")
        return

    with open(schema_path) as f:
        schema = f.read()

    async def _migrate():
        import asyncpg
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(schema)
        finally:
            await conn.close()

    try:
        asyncio.run(_migrate())
        console.print("[green]✅ Schema up to date[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Schema migration: {e}[/yellow]")

    # Ensure vector index after schema migration
    _ensure_vector_index(syne_dir)


def _restart_service():
    """Restart syne systemd service if it's running."""
    # Ensure XDG_RUNTIME_DIR is set for user-level systemd
    env = os.environ.copy()
    if "XDG_RUNTIME_DIR" not in env:
        uid = os.getuid()
        runtime_dir = f"/run/user/{uid}"
        if os.path.isdir(runtime_dir):
            env["XDG_RUNTIME_DIR"] = runtime_dir

    # Try user service first (no sudo needed)
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "syne"],
            capture_output=True, text=True, timeout=5, env=env,
        )
        if result.stdout.strip() == "active":
            console.print("🔄 Restarting syne service...")
            restart = subprocess.run(
                ["systemctl", "--user", "restart", "syne"],
                capture_output=True, text=True, timeout=15, env=env,
            )
            if restart.returncode == 0:
                console.print("[green]✅ Service restarted[/green]")
            else:
                console.print(f"[yellow]⚠️ Service restart failed: {restart.stderr.strip()}[/yellow]")
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check system service (without sudo — just check status)
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "syne"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip() == "active":
            console.print("[yellow]⚠️ Syne runs as system service. Restart manually:[/yellow]")
            console.print("[dim]  sudo systemctl restart syne[/dim]")
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _ensure_vector_index(syne_dir: str | None = None):
    """Create HNSW index on memory.embedding if embeddings exist."""
    import asyncio

    db_url = _get_db_url(syne_dir)
    if not db_url:
        return

    async def _create():
        import asyncpg
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute("SELECT ensure_memory_hnsw_index()")
        finally:
            await conn.close()

    try:
        asyncio.run(_create())
    except Exception:
        pass  # Non-fatal: function may not exist yet on first migration
