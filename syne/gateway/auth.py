"""Authentication and pairing for Syne Gateway remote nodes."""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

from syne.db.connection import get_connection

logger = logging.getLogger("syne.gateway.auth")

# Pairing token TTL
PAIRING_TOKEN_TTL = timedelta(minutes=10)


async def ensure_paired_nodes_table():
    """Create the paired_nodes table if it doesn't exist."""
    async with get_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS paired_nodes (
                id SERIAL PRIMARY KEY,
                node_id VARCHAR(100) UNIQUE NOT NULL,
                display_name VARCHAR(100) NOT NULL,
                token_hash VARCHAR(128) NOT NULL,
                platform VARCHAR(30) DEFAULT 'linux',
                active BOOLEAN DEFAULT true,
                last_seen TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pairing_tokens (
                id SERIAL PRIMARY KEY,
                token_hash VARCHAR(128) UNIQUE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL,
                used BOOLEAN DEFAULT false
            )
        """)


def _hash_token(token: str) -> str:
    """Hash a token for storage. Never store raw tokens."""
    return hashlib.sha256(token.encode()).hexdigest()


async def generate_pairing_token() -> str:
    """Generate a one-time pairing token (5 min TTL).

    Returns the raw token string (display to user).
    Only the hash is stored in DB.
    """
    token = secrets.token_urlsafe(24)
    token_hash = _hash_token(token)
    expires_at = datetime.now(timezone.utc) + PAIRING_TOKEN_TTL

    async with get_connection() as conn:
        # Clean up expired tokens first
        await conn.execute(
            "DELETE FROM pairing_tokens WHERE expires_at < NOW() OR used = true"
        )
        await conn.execute(
            "INSERT INTO pairing_tokens (token_hash, expires_at) VALUES ($1, $2)",
            token_hash, expires_at,
        )

    return token


async def verify_pairing_token(token: str) -> bool:
    """Verify and consume a pairing token. Returns True if valid."""
    token_hash = _hash_token(token)

    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            UPDATE pairing_tokens
            SET used = true
            WHERE token_hash = $1 AND expires_at > NOW() AND used = false
            RETURNING id
            """,
            token_hash,
        )
        return row is not None


async def register_node(node_id: str, display_name: str, platform: str = "linux") -> str:
    """Register a new paired node. Returns the permanent token (raw).

    If node_id already exists, regenerates the token.
    """
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    now = datetime.now(timezone.utc)

    async with get_connection() as conn:
        await conn.execute(
            """
            INSERT INTO paired_nodes (node_id, display_name, token_hash, platform, last_seen, updated_at)
            VALUES ($1, $2, $3, $4, $5, $5)
            ON CONFLICT (node_id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                token_hash = EXCLUDED.token_hash,
                platform = EXCLUDED.platform,
                active = true,
                last_seen = EXCLUDED.last_seen,
                updated_at = EXCLUDED.updated_at
            """,
            node_id, display_name, token_hash, platform, now,
        )

    logger.info(f"Node registered: {node_id} ({display_name})")
    return token


async def verify_node_token(node_id: str, token: str) -> bool:
    """Verify a node's permanent token. Updates last_seen on success."""
    token_hash = _hash_token(token)

    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            UPDATE paired_nodes
            SET last_seen = NOW()
            WHERE node_id = $1 AND token_hash = $2 AND active = true
            RETURNING id
            """,
            node_id, token_hash,
        )
        return row is not None


async def list_nodes() -> list[dict]:
    """List all paired nodes."""
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT node_id, display_name, platform, active, last_seen, created_at
            FROM paired_nodes
            ORDER BY created_at
            """
        )
        return [dict(r) for r in rows]


async def revoke_node(node_id: str) -> bool:
    """Revoke a node's access (soft delete — set active=false)."""
    async with get_connection() as conn:
        result = await conn.execute(
            "UPDATE paired_nodes SET active = false, updated_at = NOW() WHERE node_id = $1",
            node_id,
        )
        return result != "UPDATE 0"


async def delete_node(node_id: str) -> bool:
    """Permanently delete a paired node."""
    async with get_connection() as conn:
        result = await conn.execute(
            "DELETE FROM paired_nodes WHERE node_id = $1",
            node_id,
        )
        return result != "DELETE 0"
