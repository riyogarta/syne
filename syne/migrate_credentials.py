"""One-time credential migration: move credentials from files to PostgreSQL.

This script can be run manually to migrate credentials, or the auto-migration
functions are called during startup in agent.py.

Usage:
    python -m syne.migrate_credentials

After migration:
    - Remove SYNE_TELEGRAM_BOT_TOKEN from .env
    - Delete ~/.syne/google_credentials.json
"""

import asyncio
import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("syne.migrate")

# Legacy paths
LEGACY_OAUTH_PATH = Path.home() / ".syne" / "google_credentials.json"
ENV_FILE_PATH = Path(__file__).parent.parent / ".env"


async def migrate_telegram_bot_token() -> bool:
    """Migrate Telegram bot token from .env to database.
    
    Returns True if migration happened, False if already in DB or not found.
    """
    from .db.credentials import get_telegram_bot_token, set_telegram_bot_token, has_credential, CRED_TELEGRAM_BOT_TOKEN
    
    # Check if already in DB
    if await has_credential(CRED_TELEGRAM_BOT_TOKEN):
        logger.info("Telegram bot token already in DB — skipping migration.")
        return False
    
    # Check .env
    env_token = os.environ.get("SYNE_TELEGRAM_BOT_TOKEN")
    if not env_token:
        # Try reading from .env file directly
        if ENV_FILE_PATH.exists():
            for line in ENV_FILE_PATH.read_text().splitlines():
                if line.startswith("SYNE_TELEGRAM_BOT_TOKEN="):
                    env_token = line.split("=", 1)[1].strip().strip('"\'')
                    break
    
    if not env_token:
        logger.warning("No bot token found in .env or environment — skipping migration.")
        return False
    
    # Migrate to DB
    await set_telegram_bot_token(env_token)
    logger.warning(
        "✅ Bot token migrated from .env to DB.\n"
        "   You can now remove SYNE_TELEGRAM_BOT_TOKEN from .env"
    )
    return True


async def migrate_google_oauth() -> bool:
    """Migrate Google OAuth credentials from file to database.
    
    Returns True if migration happened, False if already in DB or not found.
    """
    from .db.credentials import (
        get_google_oauth_credentials,
        set_google_oauth_credentials,
    )
    
    # Check if already in DB
    existing = await get_google_oauth_credentials()
    if existing and existing.get("refresh_token"):
        logger.info("Google OAuth credentials already in DB — skipping migration.")
        return False
    
    # Check legacy file
    if not LEGACY_OAUTH_PATH.exists():
        logger.info(f"No legacy OAuth file at {LEGACY_OAUTH_PATH} — skipping migration.")
        return False
    
    try:
        data = json.loads(LEGACY_OAUTH_PATH.read_text())
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read legacy OAuth file: {e}")
        return False
    
    # Validate required fields
    refresh = data.get("refresh_token")
    if not refresh:
        logger.warning("Legacy OAuth file has no refresh_token — cannot migrate.")
        return False
    
    # Migrate to DB
    await set_google_oauth_credentials(
        access_token=data.get("access_token", ""),
        refresh_token=refresh,
        expires_at=data.get("expires_at", 0),
        project_id=data.get("project_id", ""),
        email=data.get("email"),
    )
    
    logger.warning(
        f"✅ Google OAuth credentials migrated to DB.\n"
        f"   You can safely delete {LEGACY_OAUTH_PATH}"
    )
    return True


async def migrate_all() -> dict:
    """Run all credential migrations.
    
    Returns dict with migration results.
    """
    from .db.connection import init_db, close_db
    
    # Initialize DB connection
    db_url = os.environ.get(
        "SYNE_DATABASE_URL",
        "postgresql://syne:syne@localhost:5433/syne"
    )
    await init_db(db_url)
    
    results = {
        "telegram_bot_token": False,
        "google_oauth": False,
    }
    
    try:
        # Migrate bot token
        results["telegram_bot_token"] = await migrate_telegram_bot_token()
        
        # Migrate Google OAuth
        results["google_oauth"] = await migrate_google_oauth()
        
        # Summary
        migrated = [k for k, v in results.items() if v]
        if migrated:
            logger.info(f"✅ Migration complete. Migrated: {', '.join(migrated)}")
            logger.info("\n📋 Cleanup suggestions:")
            if results["telegram_bot_token"]:
                logger.info("   - Remove SYNE_TELEGRAM_BOT_TOKEN from .env")
            if results["google_oauth"]:
                logger.info(f"   - Delete {LEGACY_OAUTH_PATH}")
        else:
            logger.info("No migrations needed — credentials already in DB or not found.")
        
        return results
        
    finally:
        await close_db()


async def check_credentials_status() -> dict:
    """Check the status of all credentials (DB vs file).
    
    Returns dict with status info.
    """
    from .db.connection import init_db, close_db
    from .db.credentials import (
        has_credential,
        CRED_TELEGRAM_BOT_TOKEN,
        CRED_GOOGLE_REFRESH_TOKEN,
    )
    
    db_url = os.environ.get(
        "SYNE_DATABASE_URL",
        "postgresql://syne:syne@localhost:5433/syne"
    )
    await init_db(db_url)
    
    try:
        status = {
            "telegram_bot_token": {
                "in_db": await has_credential(CRED_TELEGRAM_BOT_TOKEN),
                "in_env": bool(os.environ.get("SYNE_TELEGRAM_BOT_TOKEN")),
            },
            "google_oauth": {
                "in_db": await has_credential(CRED_GOOGLE_REFRESH_TOKEN),
                "in_file": LEGACY_OAUTH_PATH.exists(),
            },
        }
        
        print("\n📊 Credential Status:")
        print("=" * 50)
        
        print("\n🤖 Telegram Bot Token:")
        print(f"   DB:  {'✅' if status['telegram_bot_token']['in_db'] else '❌'}")
        print(f"   Env: {'⚠️ REDUNDANT' if status['telegram_bot_token']['in_env'] and status['telegram_bot_token']['in_db'] else '✅' if status['telegram_bot_token']['in_env'] else '—'}")
        
        print("\n🔐 Google OAuth:")
        print(f"   DB:   {'✅' if status['google_oauth']['in_db'] else '❌'}")
        print(f"   File: {'⚠️ REDUNDANT' if status['google_oauth']['in_file'] and status['google_oauth']['in_db'] else '✅' if status['google_oauth']['in_file'] else '—'}")
        
        if status['telegram_bot_token']['in_env'] and status['telegram_bot_token']['in_db']:
            print("\n💡 Tip: Remove SYNE_TELEGRAM_BOT_TOKEN from .env (now in DB)")
        
        if status['google_oauth']['in_file'] and status['google_oauth']['in_db']:
            print(f"\n💡 Tip: Delete {LEGACY_OAUTH_PATH} (now in DB)")
        
        print()
        return status
        
    finally:
        await close_db()


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        asyncio.run(check_credentials_status())
    else:
        asyncio.run(migrate_all())


if __name__ == "__main__":
    main()
