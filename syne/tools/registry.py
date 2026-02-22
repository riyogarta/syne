"""Tool registry — register and manage agent capabilities."""

import logging
from typing import Callable, Optional
from dataclasses import dataclass, field

from ..security import check_rule_700, OWNER_ONLY_TOOLS

logger = logging.getLogger("syne.tools.registry")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                    # JSON Schema for parameters
    handler: Callable                   # async function to execute
    requires_access_level: str = "public"
    enabled: bool = True
    scrub_level: str = "aggressive"     # "aggressive" | "safe" | "none"
    # aggressive: full regex scrub (Cookie, PEM, querystring, etc.)
    # safe: high-confidence patterns only (JWT, sk-*, bot tokens, etc.)
    #        Won't corrupt regex/code in output
    # none: tool has its own dedicated scrubber (e.g. exec)


class ToolRegistry:
    """Manages available tools for the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._approval_callback = None  # Optional async callback for interactive approval

    def set_approval_callback(self, callback):
        """Set approval callback for tools that need user confirmation.
        
        callback(tool_name: str, args: dict) -> tuple[bool, str]
        Returns (approved, reason).
        """
        self._approval_callback = callback

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable,
        requires_access_level: str = "public",
        scrub_level: str = "aggressive",
    ):
        """Register a new tool.
        
        Args:
            scrub_level: Credential scrubbing level for tool output.
                "aggressive" — full regex scrub (default, safest)
                "safe" — high-confidence patterns only (for code/content output)
                "none" — tool has its own dedicated scrubber
        """
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            requires_access_level=requires_access_level,
            scrub_level=scrub_level,
        )

    def unregister(self, name: str):
        """Remove a tool."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, access_level: str = "public") -> list[Tool]:
        """List available tools for a given access level."""
        level_order = ["public", "friend", "family", "admin", "owner"]
        try:
            max_level = level_order.index(access_level)
        except ValueError:
            max_level = 0

        return [
            tool for tool in self._tools.values()
            if tool.enabled and level_order.index(tool.requires_access_level) <= max_level
        ]

    def to_openai_schema(self, access_level: str = "public") -> list[dict]:
        """Convert available tools to OpenAI function calling format."""
        tools = self.list_tools(access_level)
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    async def execute(self, name: str, arguments: dict, access_level: str = "public") -> str:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."

        if not tool.enabled:
            return f"Error: Tool '{name}' is disabled."

        # ═══════════════════════════════════════════════════════════════
        # RULE 700 CHECK — HARDCODED, FIRST LAYER OF DEFENSE
        # This runs BEFORE the normal access level check to provide
        # defense-in-depth. Even if access levels are somehow bypassed,
        # Rule 700 still blocks non-owners from sensitive tools.
        # ═══════════════════════════════════════════════════════════════
        allowed, reason = check_rule_700(name, access_level)
        if not allowed:
            logger.warning(f"Rule 700 blocked: tool={name}, access_level={access_level}")
            return f"Error: {reason}"

        # Check access level (second layer)
        level_order = ["public", "friend", "family", "admin", "owner"]
        try:
            required = level_order.index(tool.requires_access_level)
            current = level_order.index(access_level)
        except ValueError:
            return f"Error: Invalid access level."

        if current < required:
            return f"Error: Insufficient permissions for tool '{name}'."

        # Interactive approval (e.g., CLI file write confirmation)
        if self._approval_callback:
            try:
                approved, reason = await self._approval_callback(name, arguments)
                if not approved:
                    return f"Cancelled: {reason}" if reason else "Cancelled by user."
            except Exception as e:
                logger.error(f"Approval callback error: {e}")

        try:
            result = await tool.handler(**arguments)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            return f"Error executing '{name}': {e}"
