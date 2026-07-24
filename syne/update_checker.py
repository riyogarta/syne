"""Update checker — notify owner when a new minor version is available.

Checks GitHub tags weekly. Only notifies on minor version bumps (0.3→0.4),
not patch bumps (0.4.0→0.4.1).

Results stored in DB config:
- update_check.latest_version: latest version found
- update_check.notified_version: last version we notified about (avoid spam)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


from . import __version__

logger = logging.getLogger("syne.update_checker")

GITHUB_REPO = "riyogarta/syne"


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse version string to tuple. '0.4.1' → (0, 4, 1)"""
    v = v.lstrip("v")
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts)


def _is_minor_upgrade(current: str, latest: str) -> bool:
    """Check if latest is a minor version upgrade from current.
    
    0.3.4 → 0.4.0 = True  (minor bump)
    0.4.0 → 0.4.1 = False (patch bump)
    0.4.1 → 0.4.1 = False (same)
    0.4.0 → 0.3.0 = False (downgrade)
    """
    cur = _parse_version(current)
    lat = _parse_version(latest)
    
    if len(cur) < 2 or len(lat) < 2:
        return False
    
    # Minor bump: major same, minor increased
    if lat[0] == cur[0] and lat[1] > cur[1]:
        return True
    # Major bump
    if lat[0] > cur[0]:
        return True
    
    return False


def _is_newer(current: str, latest: str) -> bool:
    """True if latest is a strictly newer version than current (any bump:
    major, minor, or patch). Tuple compare handles 1.20.9 < 1.21.0 and
    1.20.9 < 1.20.10 correctly."""
    cur = _parse_version(current)
    lat = _parse_version(latest)
    if not cur or not lat:
        return False
    return lat > cur


async def check_latest_version() -> Optional[str]:
    """Read the latest version from origin/main via local git — the SAME method
    `syne update` uses (git fetch + git show origin/main:syne/__init__.py).

    No GitHub API, no tags, no network library — just git against the repo the
    install already tracks. Returns the version string, or None on error.
    """
    import asyncio as _asyncio
    import os as _os

    # Repo root = three levels up from this file (syne/update_checker.py).
    # syne/update_checker.py -> repo root is two levels up.
    repo_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    async def _git(*args) -> tuple[int, str]:
        proc = await _asyncio.create_subprocess_exec(
            "git", *args, cwd=repo_dir,
            stdout=_asyncio.subprocess.PIPE, stderr=_asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return proc.returncode, out.decode("utf-8", "replace")

    try:
        await _git("fetch", "origin")
        code, text = await _git("show", "origin/main:syne/__init__.py")
        if code != 0:
            return None
        for line in text.splitlines():
            if line.strip().startswith("__version__"):
                return line.split("=", 1)[1].strip().strip('"').strip("'").lstrip("v")
    except Exception as e:
        logger.debug(f"Update check failed: {e}")
    return None


async def check_and_notify() -> Optional[str]:
    """Check for updates and return notification message if minor update available.
    
    Returns:
        Notification message string, or None if no update needed.
    """
    from .db.models import get_config, set_config
    
    latest = await check_latest_version()
    if not latest:
        return None
    
    current = __version__
    
    # Save latest version to DB
    await set_config("update_check.latest_version", latest)
    
    # Notify on ANY newer version (minor or patch) — the user is entitled to
    # know an update exists whenever the version changes. De-dup below keeps
    # it to one notification per version.
    if not _is_newer(current, latest):
        return None
    
    # Check if we already notified about this version
    notified = await get_config("update_check.notified_version", "")
    if notified == latest:
        return None
    
    # Mark as notified
    await set_config("update_check.notified_version", latest)
    
    return (
        f"⬆️ **Syne v{latest}** is available! (current: v{current})\n"
        f"Run `syne update` to upgrade."
    )


async def get_pending_update_notice() -> Optional[str]:
    """Get pending update notice for CLI banner (if any).
    
    Checks DB for latest_version vs current. Returns message if minor update
    available, None otherwise. Does NOT make network requests.
    """
    from .db.models import get_config
    
    latest = await get_config("update_check.latest_version", "")
    if not latest:
        return None
    
    current = __version__
    if _is_newer(current, latest):
        return (
            f"⬆️ Syne v{latest} available! (current: v{current}) — "
            f"run `syne update` to upgrade"
        )
    return None


# Daily hour (local time) at which the update check runs.
_UPDATE_CHECK_HOUR = 21


async def _seconds_until_next_run(hour: int, tz: ZoneInfo) -> float:
    """Seconds from now until the next occurrence of `hour`:00 in tz."""
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return (target - now).total_seconds()


async def update_check_loop(send_dm) -> None:
    """Background task: once a day at _UPDATE_CHECK_HOUR local time, check for a
    newer version and DM the owner if one exists.

    Args:
        send_dm: async callable send_dm(chat_id:int, text:str) that delivers a
            Telegram DM. Provided by main.run() (wraps the Telegram bot).

    Self-contained and fail-safe: any error is logged and the loop continues;
    it never raises out, so it can't take the process down. Does NOT rely on
    the scheduler table — every running install gets the daily check.
    """
    from .db.models import get_config
    from .db.connection import get_connection

    # Resolve local timezone from config (fallback UTC).
    try:
        tz_name = await get_config("system.timezone", "UTC")
        tz = ZoneInfo(tz_name if isinstance(tz_name, str) and tz_name else "UTC")
    except Exception:
        tz = ZoneInfo("UTC")

    logger.info(f"Update-check loop started (daily at {_UPDATE_CHECK_HOUR}:00 {tz}).")

    while True:
        try:
            delay = await _seconds_until_next_run(_UPDATE_CHECK_HOUR, tz)
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            logger.info("Update-check loop cancelled.")
            return
        except Exception as e:
            logger.warning(f"Update-check loop sleep error: {e}; retrying in 1h")
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                return
            continue

        # Time to check.
        try:
            message = await check_and_notify()
            if not message:
                continue
            # Find owner Telegram ID and DM them.
            async with get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT platform_id FROM users "
                    "WHERE access_level = 'owner' AND platform = 'telegram' LIMIT 1"
                )
            if row and row["platform_id"]:
                await send_dm(int(row["platform_id"]), message)
                logger.info("Update-check loop: notified owner of new version.")
            else:
                logger.debug("Update-check loop: no owner to notify.")
        except Exception as e:
            logger.error(f"Update-check loop check failed: {e}", exc_info=True)
