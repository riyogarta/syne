"""Syne configuration management."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class SyneSettings(BaseSettings):
    """Settings loaded from environment variables or .env file."""

    # Database — always Docker, always isolated
    database_url: str = Field(
        default="postgresql://syne:syne@localhost:5433/syne",
        description="PostgreSQL connection string (Docker container)",
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8200, description="API port")
    debug: bool = Field(default=False, description="Debug mode")

    # Auth — at least one provider needed
    # Google OAuth
    google_oauth_token: Optional[str] = Field(default=None, description="Google OAuth token path")

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

    # Together
    together_api_key: Optional[str] = Field(default=None, description="Together API key")

    # Groq
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")

    # Telegram
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram bot token")

    # Optional services
    google_maps_api_key: Optional[str] = Field(default=None, description="Google Maps API key")
    brave_api_key: Optional[str] = Field(default=None, description="Brave Search API key")
    elevenlabs_api_key: Optional[str] = Field(default=None, description="ElevenLabs TTS API key")

    model_config = {"env_prefix": "SYNE_", "env_file": ".env", "extra": "ignore"}


def load_settings() -> SyneSettings:
    """Load settings from environment."""
    settings = SyneSettings()

    # Security: warn if DB is not localhost (potential data exposure)
    import logging
    logger = logging.getLogger("syne.config")
    db = settings.database_url
    if db and "localhost" not in db and "127.0.0.1" not in db and "db:" not in db:
        logger.warning(
            "⚠️ DATABASE IS NOT LOCALHOST — API keys, OAuth tokens, and memories "
            "may be exposed if the database is publicly accessible. "
            "Syne is designed to use its own isolated Docker PostgreSQL container."
        )

    return settings
