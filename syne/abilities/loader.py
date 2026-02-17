"""Ability Loader — discovers and registers abilities."""

import json
import logging
from typing import Type

from .base import Ability
from .registry import AbilityRegistry
from ..db.connection import get_connection

logger = logging.getLogger("syne.abilities.loader")


# Bundled ability classes — explicitly imported to avoid dynamic module loading issues
def get_bundled_ability_classes() -> list[Type[Ability]]:
    """Return list of bundled ability classes.
    
    We import explicitly rather than using introspection for:
    1. Clear, predictable loading order
    2. Better error messages on import failures
    3. No surprises from dynamic discovery
    
    NOTE: web_search was migrated to core tool (syne/tools/web_search.py)
    """
    from .image_gen import ImageGenAbility
    from .image_analysis import ImageAnalysisAbility
    from .maps import MapsAbility
    from .screenshot import ScreenshotAbility
    
    return [
        ImageGenAbility,
        ImageAnalysisAbility,
        MapsAbility,
        ScreenshotAbility,
    ]


def load_bundled_abilities(registry: AbilityRegistry) -> int:
    """Discover and register all bundled abilities.
    
    Args:
        registry: The AbilityRegistry to register abilities to
        
    Returns:
        Number of abilities registered
    """
    count = 0
    
    for ability_cls in get_bundled_ability_classes():
        try:
            # Instantiate the ability
            ability = ability_cls()
            
            # Register with registry
            registry.register(
                ability=ability,
                source="bundled",
                module_path=f"syne.abilities.{ability.name}",
                enabled=True,
                requires_access_level="family",  # Default for bundled
            )
            count += 1
            logger.debug(f"Loaded bundled ability: {ability.name}")
            
        except Exception as e:
            logger.error(f"Failed to load bundled ability {ability_cls.__name__}: {e}")
    
    logger.info(f"Loaded {count} bundled abilities")
    return count


async def sync_abilities_to_db(registry: AbilityRegistry) -> int:
    """Ensure all bundled abilities are registered in the database.
    
    This creates or updates DB records for bundled abilities, preserving
    user-modified config and enabled state for existing entries.
    
    Args:
        registry: The AbilityRegistry with loaded abilities
        
    Returns:
        Number of abilities synced
    """
    count = 0
    
    async with get_connection() as conn:
        for ability in registry.list_all():
            if ability.source != "bundled":
                continue  # Only sync bundled abilities
            
            # Check if ability exists in DB
            existing = await conn.fetchrow(
                "SELECT id, enabled, config, requires_access_level FROM abilities WHERE name = $1",
                ability.name,
            )
            
            if existing:
                # Update in-memory state from DB (DB is source of truth for user config)
                ability.db_id = existing["id"]
                ability.enabled = existing["enabled"]
                ability.config = json.loads(existing["config"]) if existing["config"] else {}
                ability.requires_access_level = existing["requires_access_level"]
                
                # Update description/version if changed
                await conn.execute("""
                    UPDATE abilities 
                    SET description = $1, version = $2, updated_at = NOW()
                    WHERE id = $3
                """, ability.description, ability.version, existing["id"])
                
            else:
                # Insert new ability
                row = await conn.fetchrow("""
                    INSERT INTO abilities (name, description, version, source, module_path, config, enabled, requires_access_level)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                    RETURNING id
                """,
                    ability.name,
                    ability.description,
                    ability.version,
                    ability.source,
                    ability.module_path,
                    json.dumps(ability.config),
                    ability.enabled,
                    ability.requires_access_level,
                )
                ability.db_id = row["id"]
                logger.info(f"Added ability to DB: {ability.name}")
            
            count += 1
    
    logger.info(f"Synced {count} abilities to database")
    return count


async def ensure_abilities_table():
    """Create the abilities table if it doesn't exist."""
    async with get_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS abilities (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                version VARCHAR(20) DEFAULT '1.0',
                source VARCHAR(20) NOT NULL,
                module_path TEXT NOT NULL,
                config JSONB DEFAULT '{}',
                enabled BOOLEAN DEFAULT true,
                requires_access_level VARCHAR(20) DEFAULT 'family',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_abilities_enabled 
            ON abilities(enabled) WHERE enabled = true
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_abilities_source 
            ON abilities(source)
        """)
        
        logger.info("Abilities table ensured")


async def load_all_abilities(registry: AbilityRegistry) -> int:
    """Full ability loading flow: ensure table, load bundled, sync to DB.
    
    Args:
        registry: The AbilityRegistry to populate
        
    Returns:
        Total number of abilities loaded
    """
    # 1. Ensure DB table exists
    await ensure_abilities_table()
    
    # 2. Load bundled abilities
    bundled_count = load_bundled_abilities(registry)
    
    # 3. Sync to DB and load user config
    await sync_abilities_to_db(registry)
    
    # 4. Load any installed/self_created abilities from DB
    # (not implemented yet — would dynamically import from module_path)
    
    return bundled_count
