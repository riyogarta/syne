"""Credential management — store/retrieve credentials from DB.

All credentials are stored in the `config` table with `credential.` prefix.
This centralizes sensitive data in the encrypted PostgreSQL store instead of files.
"""

import logging
from typing import Optional, Any
from .models import get_config, set_config

logger = logging.getLogger("syne.db.credentials")

# ============================================================
# CREDENTIAL KEYS
# ============================================================

# Telegram
CRED_TELEGRAM_BOT_TOKEN = "credential.telegram_bot_token"

# Google OAuth
CRED_GOOGLE_ACCESS_TOKEN = "credential.google_oauth_access_token"
CRED_GOOGLE_REFRESH_TOKEN = "credential.google_oauth_refresh_token"
CRED_GOOGLE_EXPIRES_AT = "credential.google_oauth_expires_at"
CRED_GOOGLE_PROJECT_ID = "credential.google_oauth_project_id"
CRED_GOOGLE_EMAIL = "credential.google_oauth_email"

# Claude OAuth
CRED_CLAUDE_ACCESS_TOKEN = "credential.claude_oauth_access_token"
CRED_CLAUDE_REFRESH_TOKEN = "credential.claude_oauth_refresh_token"
CRED_CLAUDE_EXPIRES_AT = "credential.claude_oauth_expires_at"
CRED_CLAUDE_EMAIL = "credential.claude_oauth_email"


# ============================================================
# GENERIC CREDENTIAL FUNCTIONS
# ============================================================

async def get_credential(key: str, default: Any = None) -> Any:
    """Get a credential from the database."""
    return await get_config(key, default)


async def set_credential(key: str, value: Any, description: str = None) -> None:
    """Store a credential in the database."""
    await set_config(key, value, description)


async def has_credential(key: str) -> bool:
    """Check if a credential exists in the database."""
    val = await get_config(key)
    return val is not None


# ============================================================
# TELEGRAM BOT TOKEN
# ============================================================

async def get_telegram_bot_token() -> Optional[str]:
    """Get Telegram bot token from DB."""
    return await get_credential(CRED_TELEGRAM_BOT_TOKEN)


async def set_telegram_bot_token(token: str) -> None:
    """Store Telegram bot token in DB."""
    await set_credential(
        CRED_TELEGRAM_BOT_TOKEN,
        token,
        "Telegram bot token (migrated from .env)"
    )


# ============================================================
# GOOGLE OAUTH CREDENTIALS
# ============================================================

async def get_google_oauth_credentials() -> Optional[dict]:
    """Get Google OAuth credentials from DB.
    
    Returns dict with: access_token, refresh_token, expires_at, project_id, email
    or None if not found.
    """
    refresh = await get_credential(CRED_GOOGLE_REFRESH_TOKEN)
    if not refresh:
        return None
    
    return {
        "access_token": await get_credential(CRED_GOOGLE_ACCESS_TOKEN, ""),
        "refresh_token": refresh,
        "expires_at": await get_credential(CRED_GOOGLE_EXPIRES_AT, 0),
        "project_id": await get_credential(CRED_GOOGLE_PROJECT_ID, ""),
        "email": await get_credential(CRED_GOOGLE_EMAIL),
    }


async def set_google_oauth_credentials(
    access_token: str,
    refresh_token: str,
    expires_at: float,
    project_id: str,
    email: Optional[str] = None,
) -> None:
    """Store Google OAuth credentials in DB."""
    await set_credential(
        CRED_GOOGLE_ACCESS_TOKEN,
        access_token,
        "Google OAuth access token"
    )
    await set_credential(
        CRED_GOOGLE_REFRESH_TOKEN,
        refresh_token,
        "Google OAuth refresh token"
    )
    await set_credential(
        CRED_GOOGLE_EXPIRES_AT,
        expires_at,
        "Google OAuth token expiry (epoch seconds)"
    )
    await set_credential(
        CRED_GOOGLE_PROJECT_ID,
        project_id,
        "Google Cloud project ID"
    )
    if email:
        await set_credential(
            CRED_GOOGLE_EMAIL,
            email,
            "Google account email"
        )
    
    logger.info(f"Google OAuth credentials stored in DB for {email or 'unknown'}")


async def update_google_access_token(access_token: str, expires_at: float, refresh_token: str = None) -> None:
    """Update just the access token (after refresh)."""
    await set_credential(CRED_GOOGLE_ACCESS_TOKEN, access_token)
    await set_credential(CRED_GOOGLE_EXPIRES_AT, expires_at)
    if refresh_token:
        await set_credential(CRED_GOOGLE_REFRESH_TOKEN, refresh_token)
    logger.debug("Google access token updated in DB")


# ============================================================
# CLAUDE OAUTH CREDENTIALS
# ============================================================

async def get_claude_oauth_credentials() -> Optional[dict]:
    """Get Claude OAuth credentials from DB.
    
    Returns dict with: access_token, refresh_token, expires_at, email
    or None if not found.
    """
    refresh = await get_credential(CRED_CLAUDE_REFRESH_TOKEN)
    if not refresh:
        return None
    
    return {
        "access_token": await get_credential(CRED_CLAUDE_ACCESS_TOKEN, ""),
        "refresh_token": refresh,
        "expires_at": await get_credential(CRED_CLAUDE_EXPIRES_AT, 0),
        "email": await get_credential(CRED_CLAUDE_EMAIL),
    }


async def set_claude_oauth_credentials(
    access_token: str,
    refresh_token: str,
    expires_at: float,
    email: Optional[str] = None,
) -> None:
    """Store Claude OAuth credentials in DB."""
    await set_credential(
        CRED_CLAUDE_ACCESS_TOKEN,
        access_token,
        "Claude OAuth access token"
    )
    await set_credential(
        CRED_CLAUDE_REFRESH_TOKEN,
        refresh_token,
        "Claude OAuth refresh token"
    )
    await set_credential(
        CRED_CLAUDE_EXPIRES_AT,
        expires_at,
        "Claude OAuth token expiry (epoch seconds)"
    )
    if email:
        await set_credential(
            CRED_CLAUDE_EMAIL,
            email,
            "Claude account email"
        )
    
    logger.info(f"Claude OAuth credentials stored in DB for {email or 'unknown'}")
