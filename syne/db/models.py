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
    async with get_connection() as conn:
        row = await conn.fetchrow("""
            INSERT INTO users (name, display_name, platform, platform_id, access_level)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, name, display_name, platform, platform_id, access_level, preferences
        """, name, display_name, platform, platform_id, access_level)
        return dict(row)


async def get_or_create_user(
    name: str,
    platform: str,
    platform_id: str,
    display_name: Optional[str] = None,
) -> dict:
    """Get existing user or create new one.
    
    If no owner exists yet, the first user to interact becomes the owner.
    This handles both fresh installs (no users) and cases where users
    were created before the owner logic was added.
    """
    user = await get_user(platform, platform_id)
    if user:
        # Auto-promote: if no owner exists yet, promote this existing user.
        # Also: if this is a fresh install (few users) and no owner exists
        # on THIS platform yet, promote — likely same person from another channel.
        if user.get("access_level") != "owner":
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
                        f"Auto-promoted user {user['name']} ({platform_id}) to owner (no owner existed)"
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
                            f"(first on platform '{platform}', fresh install)"
                        )
        return user
    
    # New user — if no owner exists, this one becomes owner.
    # Also: fresh install heuristic for cross-platform owner.
    access_level = "public"
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
                # Check DM approval policy — new users start as 'pending'
                # if approval mode is enabled
                dm_policy = await get_config("telegram.dm_policy", "open")
                if dm_policy == "approval":
                    access_level = "pending"
    
    return await create_user(name, platform, platform_id, display_name, access_level)


async def update_user(
    platform: str,
    platform_id: str,
    display_name: Optional[str] = None,
    aliases: Optional[dict] = None,
    access_level: Optional[str] = None,
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
        return dict(row) if row else None


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
