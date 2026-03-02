"""Database query helpers for Syne tables."""

from typing import Optional
from .connection import get_connection


# ============================================================
# IDENTITY
# ============================================================

async def get_identity() -> dict:
    """Load all active identity key-value pairs."""
    async with get_connection() as conn:
        rows = await conn.fetch("SELECT key, value FROM identity WHERE active = true")
        return {row["key"]: row["value"] for row in rows}


async def set_identity(key: str, value: str):
    """Set an identity value (upsert)."""
    async with get_connection() as conn:
        await conn.execute("""
            INSERT INTO identity (key, value) VALUES ($1, $2)
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = NOW()
        """, key, value)


# ============================================================
# SOUL
# ============================================================

async def get_soul() -> list[dict]:
    """Load all active soul entries, ordered by priority."""
    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT category, key, content, priority
            FROM soul WHERE active = true
            ORDER BY priority DESC
        """)
        return [dict(row) for row in rows]


# ============================================================
# RULES
# ============================================================

async def get_rules(enforced_only: bool = True) -> list[dict]:
    """Load rules."""
    async with get_connection() as conn:
        query = "SELECT code, name, description, severity FROM rules"
        if enforced_only:
            query += " WHERE enforced = true"
        query += " ORDER BY severity DESC, code"
        rows = await conn.fetch(query)
        return [dict(row) for row in rows]


# ============================================================
# USERS
# ============================================================

async def get_user(platform: str, platform_id: str) -> Optional[dict]:
    """Find a user by platform and platform ID."""
    import json
    async with get_connection() as conn:
        row = await conn.fetchrow("""
            SELECT id, name, display_name, platform, platform_id, access_level, preferences, aliases
            FROM users WHERE platform = $1 AND platform_id = $2 AND active = true
        """, platform, platform_id)
        if not row:
            return None
        result = dict(row)
        # Parse JSONB fields that may come as strings
        for field in ['preferences', 'aliases']:
            if field in result and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    result[field] = {}
        return result


async def create_user(
    name: str,
    platform: str,
    platform_id: str,
    display_name: Optional[str] = None,
    access_level: str = "public",
) -> dict:
    """Create a new user."""
    import json
    async with get_connection() as conn:
        row = await conn.fetchrow("""
            INSERT INTO users (name, display_name, platform, platform_id, access_level)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, name, display_name, platform, platform_id, access_level, preferences
        """, name, display_name, platform, platform_id, access_level)
        result = dict(row)
        # Parse JSONB fields that may come as strings (asyncpg may return json/jsonb as str)
        if "preferences" in result and isinstance(result["preferences"], str):
            try:
                result["preferences"] = json.loads(result["preferences"])
            except (json.JSONDecodeError, TypeError):
                result["preferences"] = {}
        return result


async def get_or_create_user(
    name: str,
    platform: str,
    platform_id: str,
    display_name: Optional[str] = None,
    is_dm: bool = False,
) -> dict:
    """Get existing user or create new one.
    
    If no owner exists yet, the first user to interact **via DM** becomes
    the owner.  Group interactions NEVER auto-promote to owner — otherwise
    a random group member who happens to chat first would hijack ownership.
    
    Args:
        is_dm: True if this interaction is a direct message (private chat).
               Only DM interactions can trigger auto-promote to owner.
    """
    user = await get_user(platform, platform_id)
    if user:
        # Auto-promote: ONLY from DM, and only if no owner exists yet.
        if is_dm and user.get("access_level") != "owner":
            async with get_connection() as conn:
                owner_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM users WHERE access_level = 'owner'"
                )
                if owner_count == 0:
                    await conn.execute(
                        "UPDATE users SET access_level = 'owner' WHERE id = $1",
                        user["id"]
                    )
                    user = dict(user)
                    user["access_level"] = "owner"
                    import logging
                    logging.getLogger("syne.db").info(
                        f"Auto-promoted user {user['name']} ({platform_id}) to owner (first DM, no owner existed)"
                    )
                else:
                    # Fresh install heuristic: if no owner on THIS platform
                    # and total users <= 3, likely same person from another channel
                    owner_on_platform = await conn.fetchval(
                        "SELECT COUNT(*) FROM users WHERE access_level = 'owner' AND platform = $1",
                        platform
                    )
                    total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
                    if owner_on_platform == 0 and total_users <= 3:
                        await conn.execute(
                            "UPDATE users SET access_level = 'owner' WHERE id = $1",
                            user["id"]
                        )
                        user = dict(user)
                        user["access_level"] = "owner"
                        import logging
                        logging.getLogger("syne.db").info(
                            f"Auto-promoted user {user['name']} ({platform_id}) to owner "
                            f"(first DM on platform '{platform}', fresh install)"
                        )
        return user
    
    # New user — auto-promote to owner ONLY from DM.
    access_level = "public"
    if is_dm:
        async with get_connection() as conn:
            owner_count = await conn.fetchval(
                "SELECT COUNT(*) FROM users WHERE access_level = 'owner'"
            )
            if owner_count == 0:
                access_level = "owner"
            else:
                owner_on_platform = await conn.fetchval(
                    "SELECT COUNT(*) FROM users WHERE access_level = 'owner' AND platform = $1",
                    platform
                )
                total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
                if owner_on_platform == 0 and total_users <= 3:
                    access_level = "owner"
                else:
                    dm_policy = await get_config(
                        f"{platform}.dm_policy",
                        await get_config("dm_policy", "approval"),
                    )
                    if dm_policy == "approval":
                        access_level = "pending"
    else:
        # Group interaction — return ephemeral user dict, don't persist.
        # Group-only users belong in group_members, not global /members.
        return {
            "id": None,
            "name": name,
            "display_name": display_name or name,
            "platform": platform,
            "platform_id": platform_id,
            "access_level": "public",
            "preferences": {},
            "active": True,
        }

    return await create_user(name, platform, platform_id, display_name, access_level)


async def update_user(
    platform: str,
    platform_id: str,
    display_name: Optional[str] = None,
    aliases: Optional[dict] = None,
    access_level: Optional[str] = None,
    preferences: Optional[dict] = None,
) -> Optional[dict]:
    """Update user fields. Returns updated user or None if not found."""
    import json
    async with get_connection() as conn:
        # Build dynamic update
        updates = []
        params = []
        param_idx = 1
        
        if display_name is not None:
            updates.append(f"display_name = ${param_idx}")
            params.append(display_name)
            param_idx += 1
        
        if aliases is not None:
            updates.append(f"aliases = ${param_idx}::jsonb")
            params.append(json.dumps(aliases))
            param_idx += 1
        
        if access_level is not None:
            updates.append(f"access_level = ${param_idx}")
            params.append(access_level)
            param_idx += 1

        if preferences is not None:
            updates.append(f"preferences = ${param_idx}::jsonb")
            params.append(json.dumps(preferences))
            param_idx += 1

        if not updates:
            return await get_user(platform, platform_id)
        
        updates.append("updated_at = NOW()")
        params.extend([platform, platform_id])
        
        query = f"""
            UPDATE users SET {', '.join(updates)}
            WHERE platform = ${param_idx} AND platform_id = ${param_idx + 1}
            RETURNING id, name, display_name, platform, platform_id, access_level, preferences, aliases
        """
        
        row = await conn.fetchrow(query, *params)
        if not row:
            return None
        result = dict(row)
        # Parse JSONB fields that may come as strings
        for field in ['preferences', 'aliases']:
            if field in result and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    result[field] = {}
        return result


async def delete_user(platform: str, platform_id: str) -> bool:
    """Delete a user and all related data from the database. Returns True if deleted."""
    async with get_connection() as conn:
        # Get user id first
        row = await conn.fetchrow(
            "SELECT id FROM users WHERE platform = $1 AND platform_id = $2",
            platform, platform_id,
        )
        if not row:
            return False
        user_id = row["id"]

        # Delete dependent rows (messages cascade from sessions)
        await conn.execute("DELETE FROM memory WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM sessions WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM users WHERE id = $1", user_id)
        return True


async def get_user_alias(user: dict, group_id: str = None) -> str:
    """Get the display name/alias for a user, optionally for a specific group."""
    aliases = user.get("aliases") or {}
    
    # Check group-specific alias first
    if group_id and isinstance(aliases, dict):
        groups = aliases.get("groups", {})
        if group_id in groups:
            return groups[group_id]
    
    # Fall back to default alias
    if isinstance(aliases, dict) and aliases.get("default"):
        return aliases["default"]
    
    # Fall back to display_name or name
    return user.get("display_name") or user.get("name", "User")


# ============================================================
# GROUPS
# ============================================================

async def get_group(platform: str, platform_group_id: str) -> Optional[dict]:
    """Get a group by platform and group ID."""
    async with get_connection() as conn:
        row = await conn.fetchrow("""
            SELECT id, platform, platform_group_id, name, enabled, require_mention, allow_from, settings
            FROM groups WHERE platform = $1 AND platform_group_id = $2
        """, platform, platform_group_id)
        if not row:
            return None
        result = dict(row)
        # Ensure settings is always a dict (asyncpg may return jsonb as str in some configs)
        if isinstance(result.get("settings"), str):
            import json
            try:
                result["settings"] = json.loads(result["settings"])
            except (ValueError, TypeError):
                result["settings"] = {}
        return result


async def create_group(
    platform: str,
    platform_group_id: str,
    name: Optional[str] = None,
    enabled: bool = True,
    require_mention: bool = True,
    allow_from: str = "all",
) -> dict:
    """Create a new group registration."""
    async with get_connection() as conn:
        row = await conn.fetchrow("""
            INSERT INTO groups (platform, platform_group_id, name, enabled, require_mention, allow_from)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, platform, platform_group_id, name, enabled, require_mention, allow_from, settings
        """, platform, platform_group_id, name, enabled, require_mention, allow_from)
        return dict(row)


async def update_group(
    platform: str,
    platform_group_id: str,
    name: Optional[str] = None,
    enabled: Optional[bool] = None,
    require_mention: Optional[bool] = None,
    allow_from: Optional[str] = None,
    settings: Optional[dict] = None,
) -> Optional[dict]:
    """Update a group's settings. Returns updated group or None if not found."""
    import json
    async with get_connection() as conn:
        # Build dynamic update
        updates = []
        params = []
        param_idx = 1
        
        if name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(name)
            param_idx += 1
        
        if enabled is not None:
            updates.append(f"enabled = ${param_idx}")
            params.append(enabled)
            param_idx += 1
        
        if require_mention is not None:
            updates.append(f"require_mention = ${param_idx}")
            params.append(require_mention)
            param_idx += 1
        
        if allow_from is not None:
            updates.append(f"allow_from = ${param_idx}")
            params.append(allow_from)
            param_idx += 1
        
        if settings is not None:
            updates.append(f"settings = ${param_idx}::jsonb")
            params.append(json.dumps(settings))
            param_idx += 1
        
        if not updates:
            return await get_group(platform, platform_group_id)
        
        updates.append("updated_at = NOW()")
        params.extend([platform, platform_group_id])
        
        query = f"""
            UPDATE groups SET {', '.join(updates)}
            WHERE platform = ${param_idx} AND platform_group_id = ${param_idx + 1}
            RETURNING id, platform, platform_group_id, name, enabled, require_mention, allow_from, settings
        """
        
        row = await conn.fetchrow(query, *params)
        return dict(row) if row else None


async def update_group_member(
    platform: str,
    platform_group_id: str,
    member_id: str,
    name: Optional[str] = None,
    username: Optional[str] = None,
    alias: Optional[str] = None,
    access: Optional[str] = None,
) -> Optional[dict]:
    """Upsert a member in group settings.members JSONB.

    Auto-collect: only updates name/username/seen (never overwrites alias/access).
    Manual: explicitly sets alias and/or access.
    Returns the updated member entry or None if group not found.
    """
    import json
    from datetime import datetime, timezone

    async with get_connection() as conn:
        # Get current settings
        row = await conn.fetchrow(
            "SELECT settings FROM groups WHERE platform = $1 AND platform_group_id = $2",
            platform, platform_group_id,
        )
        if not row:
            return None

        settings = row["settings"]
        if isinstance(settings, str):
            settings = json.loads(settings) if settings else {}
        settings = settings or {}

        members = settings.get("members", {})
        existing = members.get(member_id, {})

        # Update fields — auto-collect never overwrites alias/access
        if name is not None:
            existing["name"] = name
        if username is not None:
            existing["username"] = username
        if alias is not None:
            existing["alias"] = alias
        if access is not None:
            existing["access"] = access

        # Set defaults for new members
        if "access" not in existing:
            existing["access"] = "public"
        existing["seen"] = datetime.now(timezone.utc).isoformat()

        members[member_id] = existing
        settings["members"] = members

        await conn.execute(
            "UPDATE groups SET settings = $1::jsonb, updated_at = NOW() WHERE platform = $2 AND platform_group_id = $3",
            json.dumps(settings, ensure_ascii=False), platform, platform_group_id,
        )

        return existing


async def get_group_members(platform: str, platform_group_id: str) -> dict:
    """Get all members from group settings.members. Returns {id: {name, alias, access, ...}}."""
    import json
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT settings FROM groups WHERE platform = $1 AND platform_group_id = $2",
            platform, platform_group_id,
        )
        if not row:
            return {}
        settings = row["settings"]
        if isinstance(settings, str):
            settings = json.loads(settings) if settings else {}
        return (settings or {}).get("members", {})


async def delete_group(platform: str, platform_group_id: str) -> bool:
    """Delete a group registration. Returns True if deleted."""
    async with get_connection() as conn:
        result = await conn.execute("""
            DELETE FROM groups WHERE platform = $1 AND platform_group_id = $2
        """, platform, platform_group_id)
        return "DELETE 1" in str(result)


async def list_groups(platform: str = None, enabled_only: bool = True) -> list[dict]:
    """List all registered groups."""
    import json
    async with get_connection() as conn:
        query = "SELECT id, platform, platform_group_id, name, enabled, require_mention, allow_from, settings FROM groups"
        conditions = []
        params = []
        
        if platform:
            params.append(platform)
            conditions.append(f"platform = ${len(params)}")
        
        if enabled_only:
            conditions.append("enabled = true")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY name, platform_group_id"
        
        rows = await conn.fetch(query, *params)
        results = []
        for row in rows:
            result = dict(row)
            # Parse JSONB settings field if string
            if 'settings' in result and isinstance(result['settings'], str):
                try:
                    result['settings'] = json.loads(result['settings'])
                except (json.JSONDecodeError, TypeError):
                    result['settings'] = {}
            results.append(result)
        return results


async def list_users(platform: str = None, access_level: str = None) -> list[dict]:
    """List users, optionally filtered by platform or access level."""
    import json
    async with get_connection() as conn:
        query = "SELECT id, name, display_name, platform, platform_id, access_level, preferences, aliases FROM users WHERE active = true"
        params = []
        
        if platform:
            params.append(platform)
            query += f" AND platform = ${len(params)}"
        
        if access_level:
            params.append(access_level)
            query += f" AND access_level = ${len(params)}"
        
        query += " ORDER BY name"
        
        rows = await conn.fetch(query, *params)
        results = []
        for row in rows:
            result = dict(row)
            # Parse JSONB fields that may come as strings
            for field in ['preferences', 'aliases']:
                if field in result and isinstance(result[field], str):
                    try:
                        result[field] = json.loads(result[field])
                    except (json.JSONDecodeError, TypeError):
                        result[field] = {}
            results.append(result)
        return results


async def get_first_owner(platform: str = "telegram") -> Optional[dict]:
    """Get the first owner (by created_at) — this user is immutable via UI."""
    import json
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """SELECT id, name, display_name, platform, platform_id, access_level, preferences, aliases
               FROM users WHERE platform = $1 AND access_level = 'owner'
               ORDER BY created_at ASC LIMIT 1""",
            platform,
        )
        if not row:
            return None
        result = dict(row)
        for field in ['preferences', 'aliases']:
            if field in result and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    result[field] = {}
        return result


# ============================================================
# CONFIG
# ============================================================

async def get_config(key: str, default=None):
    """Get a config value by key."""
    async with get_connection() as conn:
        row = await conn.fetchrow("SELECT value FROM config WHERE key = $1", key)
        if row:
            import json
            return json.loads(row["value"])
        return default


async def set_config(key: str, value, description: Optional[str] = None):
    """Set a config value (upsert)."""
    import json
    json_value = json.dumps(value)
    async with get_connection() as conn:
        if description:
            await conn.execute("""
                INSERT INTO config (key, value, description) VALUES ($1, $2::jsonb, $3)
                ON CONFLICT (key) DO UPDATE SET value = $2::jsonb, description = $3, updated_at = NOW()
            """, key, json_value, description)
        else:
            await conn.execute("""
                INSERT INTO config (key, value) VALUES ($1, $2::jsonb)
                ON CONFLICT (key) DO UPDATE SET value = $2::jsonb, updated_at = NOW()
            """, key, json_value)
