"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import os


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_pool():
    """Initialize database connection pool for tests."""
    from syne.db.connection import init_db, close_db, get_connection
    
    # Use test database URL or default
    db_url = os.environ.get(
        "SYNE_TEST_DATABASE_URL",
        "postgresql://syne:syne@localhost:5433/syne"
    )
    
    await init_db(db_url)
    yield
    await close_db()


@pytest.fixture
async def clean_groups(db_pool):
    """Clean up groups table before/after test."""
    from syne.db.connection import get_connection
    
    async with get_connection() as conn:
        await conn.execute("DELETE FROM groups WHERE platform = 'telegram'")
    
    yield
    
    async with get_connection() as conn:
        await conn.execute("DELETE FROM groups WHERE platform = 'telegram'")


@pytest.fixture
async def clean_test_users(db_pool):
    """Clean up test users before/after test."""
    from syne.db.connection import get_connection
    
    # Only clean test users (platform_id starting with 'test_')
    async with get_connection() as conn:
        await conn.execute("DELETE FROM users WHERE platform_id LIKE 'test_%'")
    
    yield
    
    async with get_connection() as conn:
        await conn.execute("DELETE FROM users WHERE platform_id LIKE 'test_%'")
