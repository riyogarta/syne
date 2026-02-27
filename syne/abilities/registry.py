"""Ability Registry — manages loading, enabling, and executing abilities."""

import asyncio
import json
import logging
from typing import Optional
from dataclasses import dataclass, field

from .base import Ability
from ..db.connection import get_connection

logger = logging.getLogger("syne.abilities.registry")

# Execution timeout (seconds) — prevents hung abilities from blocking the agent
EXECUTE_TIMEOUT = 120

# Auto-disable after this many consecutive failures
MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class RegisteredAbility:
    """Wrapper for a registered ability with metadata."""
    name: str
    description: str
    version: str
    source: str  # 'bundled', 'installed', 'self_created'
    module_path: str
    instance: Ability
    config: dict = field(default_factory=dict)
    enabled: bool = True
    requires_access_level: str = "family"
    db_id: Optional[int] = None
    consecutive_failures: int = 0


class AbilityRegistry:
    """Manages available abilities for the agent.
    
    Responsibilities:
    - Load abilities from bundled package and DB
    - Register/unregister abilities at runtime
    - Enable/disable abilities
    - Convert to OpenAI function calling schema
    - Execute abilities with access level checks
    """

    # Access level hierarchy (lower index = lower privilege)
    ACCESS_LEVELS = ["public", "friend", "family", "admin", "owner"]

    def __init__(self):
        self._abilities: dict[str, RegisteredAbility] = {}

    def register(
        self,
        ability: Ability,
        source: str = "bundled",
        module_path: str = "",
        config: Optional[dict] = None,
        enabled: bool = True,
        requires_access_level: str = "family",
        db_id: Optional[int] = None,
    ):
        """Register an ability instance.
        
        Args:
            ability: The Ability instance to register
            source: Where the ability comes from ('bundled', 'installed', 'self_created')
            module_path: Python import path or file path
            config: Ability-specific configuration
            enabled: Whether ability is enabled
            requires_access_level: Minimum access level to use this ability
            db_id: Database ID if loaded from DB
        """
        registered = RegisteredAbility(
            name=ability.name,
            description=ability.description,
            version=ability.version,
            source=source,
            module_path=module_path or f"syne.abilities.{ability.name}",
            instance=ability,
            config=config or {},
            enabled=enabled,
            requires_access_level=requires_access_level,
            db_id=db_id,
        )
        self._abilities[ability.name] = registered
        logger.debug(f"Registered ability: {ability.name} (source={source}, enabled={enabled})")

    def unregister(self, name: str):
        """Remove an ability from the registry."""
        if name in self._abilities:
            del self._abilities[name]
            logger.debug(f"Unregistered ability: {name}")

    def get(self, name: str) -> Optional[RegisteredAbility]:
        """Get a registered ability by name."""
        return self._abilities.get(name)

    def list_all(self) -> list[RegisteredAbility]:
        """List all registered abilities."""
        return list(self._abilities.values())

    def list_enabled(self, access_level: str = "public") -> list[RegisteredAbility]:
        """List enabled abilities accessible at the given access level.
        
        Args:
            access_level: User's access level
            
        Returns:
            List of enabled abilities the user can access
        """
        try:
            max_level = self.ACCESS_LEVELS.index(access_level)
        except ValueError:
            max_level = 0  # Default to public if invalid

        return [
            ability for ability in self._abilities.values()
            if ability.enabled and self._check_access(ability.requires_access_level, max_level)
        ]

    def _check_access(self, required_level: str, user_level_index: int) -> bool:
        """Check if user has sufficient access."""
        try:
            required_index = self.ACCESS_LEVELS.index(required_level)
        except ValueError:
            required_index = 2  # Default to family if invalid
        return user_level_index >= required_index

    def to_openai_schema(self, access_level: str = "public") -> list[dict]:
        """Convert enabled abilities to OpenAI function calling format.
        
        Validates every schema before including it. Abilities with malformed
        schemas are logged and skipped — they won't reach the LLM API.
        
        Args:
            access_level: User's access level for filtering
            
        Returns:
            List of validated function schemas for OpenAI API
        """
        from .validator import validate_tool_schema

        abilities = self.list_enabled(access_level)
        schemas = []
        for ability in abilities:
            try:
                schema = ability.instance.get_schema()
                # Normalize: ensure OpenAI function calling format
                # Accept both {"type":"function","function":{...}} and flat {"name":...,"parameters":...}
                if schema and "type" not in schema and "name" in schema:
                    schema = {"type": "function", "function": schema}

                # Validate schema before adding — reject malformed definitions
                ok, err = validate_tool_schema(schema, ability.name)
                if not ok:
                    logger.error(f"Skipping ability '{ability.name}': {err}")
                    continue

                schemas.append(schema)
            except Exception as e:
                logger.error(f"Failed to get schema for {ability.name}: {e}")
        return schemas

    async def execute(
        self,
        name: str,
        params: dict,
        context: dict,
    ) -> dict:
        """Execute an ability by name.
        
        Args:
            name: Ability name
            params: Parameters from LLM function call
            context: Execution context (user_id, session_id, access_level, etc.)
            
        Returns:
            Execution result dict with success, result, error, media keys
        """
        ability = self.get(name)
        if not ability:
            return {"success": False, "error": f"Ability '{name}' not found."}

        if not ability.enabled:
            return {"success": False, "error": f"Ability '{name}' is disabled."}

        # Check access level
        access_level = context.get("access_level", "public")
        try:
            user_level = self.ACCESS_LEVELS.index(access_level)
        except ValueError:
            user_level = 0

        if not self._check_access(ability.requires_access_level, user_level):
            return {
                "success": False,
                "error": f"Insufficient permissions for ability '{name}'. "
                         f"Requires {ability.requires_access_level} access.",
            }

        # Validate config
        is_valid, error = await ability.instance.validate_config(ability.config)
        if not is_valid:
            return {"success": False, "error": error}

        # Add config to context for ability use
        exec_context = {**context, "config": ability.config}

        try:
            result = await asyncio.wait_for(
                ability.instance.execute(params, exec_context),
                timeout=EXECUTE_TIMEOUT,
            )
            # Reset failure counter on success
            if result.get("success"):
                ability.consecutive_failures = 0
            else:
                ability.consecutive_failures += 1
            return result
        except asyncio.TimeoutError:
            ability.consecutive_failures += 1
            logger.error(f"Ability '{name}' timed out after {EXECUTE_TIMEOUT}s")
            await self._check_auto_disable(ability)
            return {"success": False, "error": f"Ability timed out after {EXECUTE_TIMEOUT}s."}
        except Exception as e:
            ability.consecutive_failures += 1
            logger.exception(f"Error executing ability '{name}'")
            await self._check_auto_disable(ability)
            return {"success": False, "error": f"Execution error: {str(e)}"}

    async def enable(self, name: str) -> tuple[bool, str]:
        """Enable an ability, installing dependencies if needed.

        Calls ``ensure_dependencies()`` before enabling. If dependency
        installation fails, the ability stays disabled.

        Args:
            name: Ability name

        Returns:
            (True, message) on success, (False, error) on failure
        """
        ability = self.get(name)
        if not ability:
            return False, f"Ability '{name}' not found."

        # Install dependencies first
        try:
            dep_ok, dep_msg = await ability.instance.ensure_dependencies()
        except Exception as e:
            logger.error(f"ensure_dependencies() failed for '{name}': {e}")
            return False, f"Dependency check failed: {e}"

        if not dep_ok:
            logger.error(f"Cannot enable '{name}': {dep_msg}")
            return False, dep_msg

        ability.enabled = True

        # Update DB if ability has DB ID
        if ability.db_id:
            await self._update_db_enabled(ability.db_id, True)

        status = f"Ability '{name}' enabled."
        if dep_msg:
            status += f" ({dep_msg})"
        logger.info(f"Enabled ability: {name}")
        return True, status

    async def disable(self, name: str) -> bool:
        """Disable an ability.
        
        Args:
            name: Ability name
            
        Returns:
            True if successful, False if ability not found
        """
        ability = self.get(name)
        if not ability:
            return False

        ability.enabled = False

        # Update DB if ability has DB ID
        if ability.db_id:
            await self._update_db_enabled(ability.db_id, False)

        logger.info(f"Disabled ability: {name}")
        return True

    async def _update_db_enabled(self, db_id: int, enabled: bool):
        """Update enabled status in database."""
        try:
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE abilities SET enabled = $1, updated_at = NOW() WHERE id = $2",
                    enabled,
                    db_id,
                )
        except Exception as e:
            logger.error(f"Failed to update ability enabled status in DB: {e}")

    async def _check_auto_disable(self, ability: RegisteredAbility):
        """Auto-disable an ability after too many consecutive failures."""
        if ability.consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            return
        if ability.source == "bundled":
            # Never auto-disable bundled abilities — only warn
            logger.warning(
                f"Bundled ability '{ability.name}' has failed "
                f"{ability.consecutive_failures} times consecutively"
            )
            return
        logger.error(
            f"Auto-disabling ability '{ability.name}' after "
            f"{ability.consecutive_failures} consecutive failures"
        )
        ability.enabled = False
        ability.consecutive_failures = 0
        if ability.db_id:
            await self._update_db_enabled(ability.db_id, False)

    async def update_config(self, name: str, config: dict) -> bool:
        """Update ability configuration.
        
        Args:
            name: Ability name
            config: New configuration dict
            
        Returns:
            True if successful, False if ability not found
        """
        ability = self.get(name)
        if not ability:
            return False

        ability.config = config

        # Update DB if ability has DB ID
        if ability.db_id:
            try:
                async with get_connection() as conn:
                    await conn.execute(
                        "UPDATE abilities SET config = $1::jsonb, updated_at = NOW() WHERE id = $2",
                        json.dumps(config),
                        ability.db_id,
                    )
            except Exception as e:
                logger.error(f"Failed to update ability config in DB: {e}")
                return False

        logger.info(f"Updated config for ability: {name}")
        return True

    async def load_from_db(self):
        """Load ability metadata from database and update registry.
        
        This merges DB state (enabled, config, access_level) with
        already-registered abilities.
        """
        try:
            async with get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT id, name, description, version, source, module_path,
                           config, enabled, requires_access_level
                    FROM abilities
                """)

            for row in rows:
                ability = self.get(row["name"])
                if ability:
                    # Update existing ability with DB state
                    ability.db_id = row["id"]
                    ability.enabled = row["enabled"]
                    ability.config = json.loads(row["config"]) if row["config"] else {}
                    ability.requires_access_level = row["requires_access_level"]
                else:
                    # Ability in DB but not registered (installed/self_created)
                    # TODO: Dynamic loading for non-bundled abilities
                    logger.warning(f"Ability '{row['name']}' in DB but not loaded")

            logger.info(f"Loaded {len(rows)} ability records from DB")
        except Exception as e:
            logger.error(f"Failed to load abilities from DB: {e}")
