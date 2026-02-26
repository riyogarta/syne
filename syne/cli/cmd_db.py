"""Database management commands."""

import asyncio
import os
import click

from . import cli
from .shared import console


@cli.group()
def db():
    """Database management commands."""
    pass


@db.command("init")
def db_init():
    """Initialize database schema."""
    async def _init():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db

        settings = load_settings()
        pool = await init_db(settings.database_url)

        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "schema.sql")
        with open(schema_path) as f:
            schema = f.read()

        async with pool.acquire() as conn:
            await conn.execute(schema)

        console.print("[green]✓ Database schema initialized[/green]")
        await close_db()

    asyncio.run(_init())


@db.command("reset")
@click.confirmation_option(prompt="This will DELETE ALL DATA. Are you sure?")
def db_reset():
    """Reset database (DROP ALL + re-init)."""
    async def _reset():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db

        settings = load_settings()
        pool = await init_db(settings.database_url)

        async with pool.acquire() as conn:
            await conn.execute("""
                DROP TABLE IF EXISTS messages CASCADE;
                DROP TABLE IF EXISTS subagent_runs CASCADE;
                DROP TABLE IF EXISTS scheduled_tasks CASCADE;
                DROP TABLE IF EXISTS sessions CASCADE;
                DROP TABLE IF EXISTS memory CASCADE;
                DROP TABLE IF EXISTS abilities CASCADE;
                DROP TABLE IF EXISTS capabilities CASCADE;
                DROP TABLE IF EXISTS groups CASCADE;
                DROP TABLE IF EXISTS users CASCADE;
                DROP TABLE IF EXISTS rules CASCADE;
                DROP TABLE IF EXISTS soul CASCADE;
                DROP TABLE IF EXISTS identity CASCADE;
                DROP TABLE IF EXISTS config CASCADE;
                DROP EXTENSION IF EXISTS vector;
            """)

        console.print("[yellow]Tables dropped.[/yellow]")

        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "schema.sql")
        with open(schema_path) as f:
            schema = f.read()

        async with pool.acquire() as conn:
            await conn.execute(schema)

        console.print("[green]✓ Database re-initialized[/green]")
        await close_db()

    asyncio.run(_reset())
