"""Config snapshots — back up and restore only the config, not the whole DB.

A full `syne restore` replaces every table, so recovering from a bad config
change also throws away memories and chat history written since the dump.
A snapshot holds just what decides how Syne runs:

- every row of the `config` table (models, active model, credentials, ...)
- per-user model overrides (`users.preferences.model`)

Snapshots are JSON files in ~/syne/backup/config/. They contain credentials,
so they are written with mode 0600.

Automatic snapshots (startup, before a tool changes config) are skipped when
nothing changed since the previous one. Retention matches `syne backup`: the
newest SNAPSHOT_KEEP files are kept and older ones deleted after each new
snapshot. Automatic and manual snapshots are counted separately, so a burst
of automatic ones can never push out a manual backup.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from .connection import get_connection

logger = logging.getLogger("syne.db.config_snapshot")

SNAPSHOT_FORMAT = 1
SNAPSHOT_KEEP = 10  # per kind, same as full DB backups
_AUTO_PREFIX = "auto-"


def snapshot_dir() -> str:
    """~/syne/backup/config — next to the full DB backups."""
    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(syne_dir, "backup", "config")


def list_snapshots() -> list[tuple[str, float]]:
    """Snapshot files, newest first, as (path, mtime)."""
    d = snapshot_dir()
    if not os.path.isdir(d):
        return []
    files = [
        (os.path.join(d, f), os.path.getmtime(os.path.join(d, f)))
        for f in os.listdir(d) if f.endswith(".json")
    ]
    files.sort(key=lambda x: x[1], reverse=True)
    return files


async def _collect() -> dict:
    async with get_connection() as conn:
        rows = await conn.fetch("SELECT key, value, description FROM config ORDER BY key")
        users = await conn.fetch(
            "SELECT platform, platform_id, preferences->>'model' AS model "
            "FROM users WHERE preferences ? 'model' ORDER BY platform, platform_id"
        )
    return {
        "config": [
            {
                "key": r["key"],
                "value": json.loads(r["value"]) if isinstance(r["value"], str) else r["value"],
                "description": r["description"],
            }
            for r in rows
        ],
        "user_models": [
            {"platform": u["platform"], "platform_id": u["platform_id"], "model": u["model"]}
            for u in users
        ],
    }


def _payload_of(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        return {"config": data.get("config", []), "user_models": data.get("user_models", [])}
    except Exception:
        return None


async def create_snapshot(reason: str = "manual", auto: bool = False) -> Optional[str]:
    """Write a snapshot file and return its path.

    For auto snapshots, returns None when the config is identical to the
    newest existing snapshot (nothing new to keep).
    """
    payload = await _collect()

    if auto:
        latest = list_snapshots()
        if latest and _payload_of(latest[0][0]) == payload:
            return None

    d = snapshot_dir()
    os.makedirs(d, mode=0o700, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    safe_reason = "".join(c if c.isalnum() or c in "-_" else "-" for c in reason)[:40]
    base = f"{_AUTO_PREFIX if auto else ''}config-{stamp}-{safe_reason}"
    name = f"{base}.json"
    n = 1
    while os.path.exists(os.path.join(d, name)):
        n += 1
        name = f"{base}-{n}.json"
    path = os.path.join(d, name)

    doc = {
        "format": SNAPSHOT_FORMAT,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        **payload,
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    same_kind = [
        p for p, _ in list_snapshots()
        if os.path.basename(p).startswith(_AUTO_PREFIX) == auto
    ]
    for old in same_kind[SNAPSHOT_KEEP:]:
        try:
            os.remove(old)
        except OSError:
            pass

    logger.info(f"Config snapshot written: {name}")
    return path


async def auto_snapshot(reason: str) -> Optional[str]:
    """Best-effort automatic snapshot. Never raises."""
    try:
        return await create_snapshot(reason, auto=True)
    except Exception as e:
        logger.warning(f"Config auto-snapshot failed: {e}")
        return None


def load_snapshot(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if data.get("format") != SNAPSHOT_FORMAT or "config" not in data:
        raise ValueError(f"{os.path.basename(path)} is not a Syne config snapshot")
    return data


async def diff_snapshot(data: dict, prefix: Optional[str] = None) -> dict:
    """Compare a snapshot with the live DB.

    Returns {"changed": [...], "added": [...], "extra": [...], "users": [...]}:
    changed — keys whose value differs; added — keys only in the snapshot;
    extra — keys only in the DB (restore leaves these alone);
    users — (platform, platform_id, now, then) override differences.
    """
    live = await _collect()
    live_cfg = {e["key"]: e["value"] for e in live["config"]}
    snap_cfg = {e["key"]: e["value"] for e in data["config"]}
    if prefix:
        live_cfg = {k: v for k, v in live_cfg.items() if k.startswith(prefix)}
        snap_cfg = {k: v for k, v in snap_cfg.items() if k.startswith(prefix)}

    live_users = {(u["platform"], u["platform_id"]): u["model"] for u in live["user_models"]}
    snap_users = {(u["platform"], u["platform_id"]): u["model"] for u in data.get("user_models", [])}
    users = [
        (p, pid, live_users.get((p, pid)), snap_users.get((p, pid)))
        for (p, pid) in sorted(set(live_users) | set(snap_users))
        if live_users.get((p, pid)) != snap_users.get((p, pid))
    ]

    return {
        "changed": sorted(k for k in snap_cfg if k in live_cfg and live_cfg[k] != snap_cfg[k]),
        "added": sorted(k for k in snap_cfg if k not in live_cfg),
        "extra": sorted(k for k in live_cfg if k not in snap_cfg),
        "users": users,
    }


async def restore_snapshot(data: dict, prefix: Optional[str] = None, users: bool = True) -> dict:
    """Write the snapshot's config back (upsert), plus user model overrides.

    Keys that exist only in the DB are left untouched. With users=True,
    every user's model override is set to exactly what the snapshot holds
    (overrides absent from the snapshot are removed). Runs in one
    transaction.
    """
    entries = [e for e in data["config"] if not prefix or e["key"].startswith(prefix)]
    snap_users = {(u["platform"], u["platform_id"]): u["model"] for u in data.get("user_models", [])}

    async with get_connection() as conn:
        async with conn.transaction():
            for e in entries:
                await conn.execute(
                    """
                    INSERT INTO config (key, value, description) VALUES ($1, $2::jsonb, $3)
                    ON CONFLICT (key) DO UPDATE
                    SET value = $2::jsonb,
                        description = COALESCE($3, config.description),
                        updated_at = NOW()
                    """,
                    e["key"], json.dumps(e["value"]), e.get("description"),
                )
            if users:
                await conn.execute(
                    "UPDATE users SET preferences = preferences - 'model' WHERE preferences ? 'model'"
                )
                for (platform, platform_id), model in snap_users.items():
                    await conn.execute(
                        """
                        UPDATE users
                        SET preferences = COALESCE(preferences, '{}'::jsonb) || jsonb_build_object('model', $3::text)
                        WHERE platform = $1 AND platform_id = $2
                        """,
                        platform, platform_id, model,
                    )
    return {"keys": len(entries), "users": len(snap_users) if users else 0}
