"""Update checker — notify owner when a new minor version is available.

Checks GitHub tags weekly. Only notifies on minor version bumps (0.3→0.4),
not patch bumps (0.4.0→0.4.1).

Results stored in DB config:
- update_check.latest_version: latest version found
- update_check.notified_version: last version we notified about (avoid spam)
"""

import logging
from typing import Optional

import httpx

from . import __version__

logger = logging.getLogger("syne.update_checker")

GITHUB_REPO = "riyogarta/syne"
GITHUB_TAGS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/tags"


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


async def check_latest_version() -> Optional[str]:
    """Fetch latest version tag from GitHub.
    
    Returns:
        Latest version string (e.g., "0.4.1") or None on error.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                GITHUB_TAGS_URL,
                params={"per_page": 1},
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            resp.raise_for_status()
            tags = resp.json()
            if tags:
                return tags[0]["name"].lstrip("v")
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
    
    # Check if this is a minor upgrade
    if not _is_minor_upgrade(current, latest):
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
    if _is_minor_upgrade(current, latest):
        return (
            f"⬆️ Syne v{latest} available! (current: v{current}) — "
            f"run `syne update` to upgrade"
        )
    return None
