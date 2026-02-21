"""Database connection management."""

import asyncio
import logging

import asyncpg
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger("syne.db")

_pool: Optional[asyncpg.Pool] = None


async def init_db(dsn: str, min_size: int = 2, max_size: int = 10) -> asyncpg.Pool:
    """Initialize the database connection pool with retry.
    
    Retries up to 5 times with exponential backoff (2, 4, 8, 8, 8 seconds).
    This handles the case where the DB container isn't ready yet at boot.
    """
    global _pool
    max_retries = 5
    delays = [2, 4, 8, 8, 8]

    for attempt in range(max_retries):
        try:
            _pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
            if attempt > 0:
                logger.info(f"Database connected after {attempt + 1} attempts")
            return _pool
        except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as e:
            if attempt < max_retries - 1:
                delay = delays[attempt]
                logger.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Database connection failed after {max_retries} attempts: {e}")
                raise


async def close_db():
    """Close the database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Get the current connection pool."""
    if _pool is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _pool


@asynccontextmanager
async def get_connection():
    """Get a database connection from the pool."""
    pool = get_pool()
    async with pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def get_transaction():
    """Get a database connection with an active transaction."""
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn
