"""Shared utilities for Syne CLI commands."""

import os
import sys

from rich.console import Console

console = Console()


def _strip_env_quotes(value: str) -> str:
    """Strip surrounding quotes from .env values (handles both ' and \")."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _read_env_value(key: str, syne_dir: str | None = None) -> str | None:
    """Read a single value from .env file.

    Args:
        key: Environment variable name (e.g. 'SYNE_DB_USER')
        syne_dir: Project root directory. Auto-detected if None.

    Returns:
        The value string, or None if not found.
    """
    if syne_dir is None:
        syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(syne_dir, ".env")
    if not os.path.exists(env_path):
        return None
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{key}="):
                return _strip_env_quotes(line.split("=", 1)[1])
    return None


def _get_syne_dir() -> str:
    """Return the Syne project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_db_url(syne_dir: str | None = None) -> str | None:
    """Read database URL from .env, constructing from parts if needed."""
    if syne_dir is None:
        syne_dir = _get_syne_dir()

    db_url = _read_env_value("SYNE_DATABASE_URL", syne_dir)
    if db_url:
        return db_url

    db_user = _read_env_value("SYNE_DB_USER", syne_dir)
    db_pass = _read_env_value("SYNE_DB_PASSWORD", syne_dir)
    db_name = _read_env_value("SYNE_DB_NAME", syne_dir)
    if db_user and db_pass and db_name:
        return f"postgresql://{db_user}:{db_pass}@localhost:5433/{db_name}"
    return None


async def _get_provider_async():
    """Async provider init for CLI commands. Reads model registry from DB."""
    from syne.db.models import get_config
    from syne.llm.drivers import create_hybrid_provider, get_model_from_list

    models = await get_config("provider.models", None)
    active_key = await get_config("provider.active_model", None)

    if not models or not active_key:
        console.print("[red]No provider configured. Run 'syne init' first.[/red]")
        sys.exit(1)

    model_entry = get_model_from_list(models, active_key)
    if not model_entry:
        console.print(f"[red]Model '{active_key}' not found in registry.[/red]")
        sys.exit(1)

    return await create_hybrid_provider(model_entry)
