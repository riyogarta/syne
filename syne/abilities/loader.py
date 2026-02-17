"""Ability Loader — discovers and registers abilities."""

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Type, Optional

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
    
    return [
        ImageGenAbility,
        ImageAnalysisAbility,
        MapsAbility,
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


def load_dynamic_ability(module_path: str) -> Optional[Ability]:
    """Dynamically load an Ability class from a module path.

    Supports both dotted import paths (``syne.abilities.screenshot``) and
    file paths (``syne/abilities/screenshot.py``).  The module must contain
    exactly one public subclass of :class:`Ability`.

    Args:
        module_path: Python dotted path or file path to the ability module.

    Returns:
        An instantiated Ability, or None on failure.
    """
    try:
        # Normalise file path → dotted module path
        if module_path.endswith(".py") or "/" in module_path:
            # e.g. "syne/abilities/screenshot.py" → "syne.abilities.screenshot"
            module_path = module_path.replace("/", ".").removesuffix(".py")

        # If already imported, reload to pick up changes
        if module_path in sys.modules:
            mod = importlib.reload(sys.modules[module_path])
        else:
            mod = importlib.import_module(module_path)

        # Find the Ability subclass in the module
        ability_cls = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Ability)
                and attr is not Ability
                and not attr_name.startswith("_")
            ):
                ability_cls = attr
                break

        if ability_cls is None:
            logger.error(f"No Ability subclass found in {module_path}")
            return None

        return ability_cls()

    except Exception as e:
        logger.error(f"Failed to load dynamic ability from {module_path}: {e}")
        return None


async def load_dynamic_abilities_from_db(registry: AbilityRegistry) -> int:
    """Load non-bundled abilities (installed / self_created) from DB.

    For each DB record whose ``source`` is *not* ``bundled`` and that is not
    already in the registry, attempt to dynamically import the module and
    register the ability.

    Args:
        registry: The AbilityRegistry to populate.

    Returns:
        Number of dynamic abilities loaded.
    """
    count = 0
    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT id, name, description, version, source, module_path,
                   config, enabled, requires_access_level
            FROM abilities
            WHERE source != 'bundled'
        """)

    for row in rows:
        name = row["name"]
        if registry.get(name):
            continue  # already loaded

        ability = load_dynamic_ability(row["module_path"])
        if ability is None:
            logger.warning(f"Skipping dynamic ability '{name}' — load failed")
            continue

        registry.register(
            ability=ability,
            source=row["source"],
            module_path=row["module_path"],
            config=json.loads(row["config"]) if row["config"] else {},
            enabled=row["enabled"],
            requires_access_level=row["requires_access_level"],
            db_id=row["id"],
        )
        count += 1
        logger.info(f"Loaded dynamic ability: {name} (source={row['source']})")

    if count:
        logger.info(f"Loaded {count} dynamic abilities from DB")
    return count


async def register_dynamic_ability(
    registry: AbilityRegistry,
    name: str,
    description: str,
    module_path: str,
    version: str = "1.0",
    config: Optional[dict] = None,
) -> Optional[str]:
    """Register a new self-created ability at runtime.

    Called by the ``update_ability`` tool when ``action=create``.
    Dynamically imports the module, inserts/updates the DB record,
    and registers the ability in the live registry.

    Args:
        registry: The live AbilityRegistry.
        name: Ability name (must match the class's ``name`` attribute).
        description: Human-readable description.
        module_path: Dotted Python path (e.g. ``syne.abilities.screenshot``).
        version: Version string.
        config: Optional initial config dict.

    Returns:
        None on success, or an error message string.
    """
    # 1. Try to load
    ability = load_dynamic_ability(module_path)
    if ability is None:
        return f"Failed to load ability from '{module_path}'. Ensure file exists and contains a class that extends Ability."

    # Override name/description if the class doesn't set them
    if not ability.name:
        ability.name = name
    if not ability.description:
        ability.description = description

    # 2. Upsert into DB
    async with get_connection() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM abilities WHERE name = $1", ability.name
        )
        if existing:
            await conn.execute("""
                UPDATE abilities
                SET description = $1, version = $2, module_path = $3,
                    source = 'self_created', config = $4::jsonb, enabled = true,
                    updated_at = NOW()
                WHERE name = $5
            """, description, version, module_path,
                json.dumps(config or {}), ability.name)
            db_id = existing["id"]
            logger.info(f"Updated dynamic ability in DB: {ability.name}")
        else:
            row = await conn.fetchrow("""
                INSERT INTO abilities (name, description, version, source, module_path, config, enabled)
                VALUES ($1, $2, $3, 'self_created', $4, $5::jsonb, true)
                RETURNING id
            """, ability.name, description, version, module_path,
                json.dumps(config or {}))
            db_id = row["id"]
            logger.info(f"Inserted dynamic ability to DB: {ability.name}")

    # 3. Register in live registry (replace if exists)
    if registry.get(ability.name):
        registry.unregister(ability.name)

    registry.register(
        ability=ability,
        source="self_created",
        module_path=module_path,
        config=config or {},
        enabled=True,
        requires_access_level="family",
        db_id=db_id,
    )

    return None  # success


async def load_all_abilities(registry: AbilityRegistry) -> int:
    """Full ability loading flow: ensure table, load bundled, sync to DB, load dynamic.
    
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
    
    # 4. Load dynamic abilities (installed / self_created) from DB
    dynamic_count = await load_dynamic_abilities_from_db(registry)
    
    return bundled_count + dynamic_count
