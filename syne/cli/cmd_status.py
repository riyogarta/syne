"""Status command."""

import asyncio
import os
import click

from . import cli
from .shared import console

from rich.table import Table


@cli.command()
def status():
    """Show Syne status."""
    async def _status():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db

        settings = load_settings()

        from syne import __version__ as syne_version

        table = Table(title=f"Syne Status v{syne_version}", show_header=False, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Version", syne_version)

        # Git commit hash
        try:
            import subprocess as _sp
            git_result = _sp.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
            )
            if git_result.returncode == 0:
                table.add_row("Commit", git_result.stdout.strip())
        except Exception:
            pass

        # Service uptime (systemd)
        try:
            import subprocess as _sp
            uptime_result = _sp.run(
                ["systemctl", "--user", "show", "syne", "--property=ActiveEnterTimestamp"],
                capture_output=True, text=True,
            )
            if uptime_result.returncode == 0 and "=" in uptime_result.stdout:
                ts_str = uptime_result.stdout.strip().split("=", 1)[1]
                if ts_str:
                    table.add_row("Service started", ts_str)
        except Exception:
            pass

        # DB connection — single pool for all DB sections
        db_ok = False
        try:
            await init_db(settings.database_url)
            db_ok = True
        except Exception as e:
            table.add_row("Database", f"[red]Error: {e}[/red]")

        if db_ok:
            try:
                from syne.db.connection import get_connection
                from syne.db.models import get_config

                # Counts
                async with get_connection() as conn:
                    mem = await conn.fetchrow("SELECT COUNT(*) as c FROM memory")
                    users = await conn.fetchrow("SELECT COUNT(*) as c FROM users")
                    sessions = await conn.fetchrow("SELECT COUNT(*) as c FROM sessions WHERE status = 'active'")
                    msgs = await conn.fetchrow("SELECT COUNT(*) as c FROM messages")
                    identity = await conn.fetchrow("SELECT value FROM identity WHERE key = 'name'")
                    agent_name = identity["value"] if identity else "Syne"

                table.add_row("Database", "[green]Connected[/green]")
                table.add_row("Agent", agent_name)
                table.add_row("Memories", str(mem["c"]))
                table.add_row("Users", str(users["c"]))
                table.add_row("Active Sessions", str(sessions["c"]))
                table.add_row("Total Messages", str(msgs["c"]))

                # Provider
                provider_cfg = await get_config("provider.primary")
                if isinstance(provider_cfg, dict):
                    table.add_row("Provider", f"{provider_cfg.get('driver', '?')} ({provider_cfg.get('auth', '?')})")
                elif provider_cfg:
                    table.add_row("Provider", str(provider_cfg))
                else:
                    table.add_row("Provider", "not configured")

                # OAuth token status
                import time as _time
                driver = provider_cfg.get("driver", "") if isinstance(provider_cfg, dict) else ""
                auth_type = provider_cfg.get("auth", "") if isinstance(provider_cfg, dict) else ""
                if auth_type == "oauth" and driver in ("codex", "google", "google_cca", "anthropic"):
                    if driver == "codex":
                        expires_at = await get_config("credential.codex_expires_at", 0)
                    elif driver in ("google", "google_cca"):
                        expires_at_str = await get_config("credential.google_token_expiry", "")
                        try:
                            from datetime import datetime
                            expires_at = datetime.fromisoformat(expires_at_str).timestamp() if expires_at_str else 0
                        except Exception:
                            expires_at = 0
                    else:
                        expires_at = 0

                    expires_at = float(expires_at or 0)
                    now = _time.time()
                    if expires_at <= 0:
                        table.add_row("OAuth Token", "[yellow]Unknown expiry[/yellow]")
                    elif now >= expires_at:
                        elapsed = int(now - expires_at)
                        table.add_row("OAuth Token", f"[red]EXPIRED ({elapsed}s ago)[/red]")
                    else:
                        remaining = int(expires_at - now)
                        hours = remaining // 3600
                        mins = (remaining % 3600) // 60
                        table.add_row("OAuth Token", f"[green]Valid ({hours}h {mins}m remaining)[/green]")

                # Embedding provider
                embed_cfg = await get_config("provider.embedding")
                if isinstance(embed_cfg, dict):
                    table.add_row("Embedding", f"{embed_cfg.get('driver', '?')} ({embed_cfg.get('model', '?')})")
                elif embed_cfg:
                    table.add_row("Embedding", str(embed_cfg))
                else:
                    table.add_row("Embedding", "same as chat provider")

                # Credentials summary
                try:
                    import time as _cred_time
                    from syne.db.credentials import get_google_oauth_credentials, get_credential
                    cred_parts = []
                    g_creds = await get_google_oauth_credentials()
                    if g_creds and g_creds.get("refresh_token"):
                        exp = g_creds.get("expires_at", 0)
                        status_str = "[green]OK[/green]" if _cred_time.time() < exp else "[yellow]expired[/yellow]"
                        cred_parts.append(f"Google {status_str}")
                    codex_token = await get_credential("credential.codex_access_token")
                    if codex_token:
                        exp = float(await get_credential("credential.codex_expires_at", 0) or 0)
                        status_str = "[green]OK[/green]" if _cred_time.time() < exp else "[yellow]expired[/yellow]"
                        cred_parts.append(f"Codex {status_str}")
                    together_key = await get_credential("credential.together_api_key")
                    if together_key:
                        masked = str(together_key)[:8] + "..."
                        cred_parts.append(f"Together [green]{masked}[/green]")
                    groq_key = await get_credential("credential.groq_api_key")
                    if groq_key:
                        masked = str(groq_key)[:8] + "..."
                        cred_parts.append(f"Groq [green]{masked}[/green]")
                    if cred_parts:
                        table.add_row("Credentials", ", ".join(cred_parts))
                    else:
                        table.add_row("Credentials", "[dim]none configured[/dim]")
                except Exception:
                    pass

                # Scheduler/cron
                async with get_connection() as conn:
                    sched_count = await conn.fetchrow("SELECT COUNT(*) as c FROM scheduled_tasks WHERE enabled = true")
                if sched_count and sched_count["c"] > 0:
                    table.add_row("Scheduled Jobs", f"{sched_count['c']} active")
                else:
                    table.add_row("Scheduled Jobs", "none")

                # Telegram (check DB first, then env)
                tg_status = "Not configured"
                try:
                    from syne.db.credentials import get_telegram_bot_token
                    db_token = await get_telegram_bot_token()
                    if db_token:
                        tg_status = "Configured"
                    elif settings.telegram_bot_token:
                        tg_status = "Configured (env)"
                except Exception:
                    if settings.telegram_bot_token:
                        tg_status = "Configured (env)"
                table.add_row("Telegram", tg_status)

            except Exception as e:
                table.add_row("Provider", f"[red]Error: {e}[/red]")
            finally:
                await close_db()
        else:
            # DB failed — check Telegram from env only
            if settings.telegram_bot_token:
                table.add_row("Telegram", "Configured (env)")
            else:
                table.add_row("Telegram", "Not configured")

        console.print(table)

    asyncio.run(_status())
