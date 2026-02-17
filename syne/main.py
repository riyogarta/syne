"""Syne — Main entry point."""

import asyncio
import logging
import os

from .config import load_settings
from .agent import SyneAgent
from .channels.telegram import TelegramChannel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("syne")


async def _get_telegram_bot_token(settings) -> str | None:
    """Get Telegram bot token from DB (primary) or .env (fallback + migrate).
    
    Priority:
    1. Database (credential.telegram_bot_token)
    2. Environment/settings (.env SYNE_TELEGRAM_BOT_TOKEN) — auto-migrate to DB
    """
    from .db.credentials import get_telegram_bot_token, set_telegram_bot_token, has_credential, CRED_TELEGRAM_BOT_TOKEN
    
    # 1. Try database first
    db_token = await get_telegram_bot_token()
    if db_token:
        logger.debug("Using Telegram bot token from database.")
        return db_token
    
    # 2. Fall back to .env and migrate
    env_token = settings.telegram_bot_token
    if env_token:
        # Auto-migrate to DB
        await set_telegram_bot_token(env_token)
        logger.warning(
            "Bot token migrated from .env to DB. "
            "You can now remove SYNE_TELEGRAM_BOT_TOKEN from .env"
        )
        return env_token
    
    # No token found
    return None


async def _auto_migrate_google_oauth():
    """Auto-migrate Google OAuth credentials from file to DB if needed."""
    from .migrate_credentials import migrate_google_oauth
    try:
        await migrate_google_oauth()
    except Exception as e:
        logger.debug(f"Google OAuth migration check: {e}")


async def run():
    """Main run loop."""
    settings = load_settings()
    agent = SyneAgent(settings)
    channels = []

    try:
        await agent.start()
        
        # Auto-migrate Google OAuth after DB is ready
        await _auto_migrate_google_oauth()

        # Get Telegram bot token (DB first, then .env with auto-migrate)
        bot_token = await _get_telegram_bot_token(settings)
        
        # Start Telegram if configured
        if bot_token:
            telegram = TelegramChannel(agent, bot_token)
            await telegram.start()
            channels.append(telegram)
            logger.info("Telegram channel active.")
        else:
            logger.warning(
                "No Telegram bot token configured. "
                "Set via DB (credential.telegram_bot_token) or .env (SYNE_TELEGRAM_BOT_TOKEN)."
            )

        # Keep alive
        logger.info("Syne is running. Press Ctrl+C to stop.")
        while agent._running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        # Stop channels
        for ch in channels:
            await ch.stop()
        await agent.stop()


def main():
    """Entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
