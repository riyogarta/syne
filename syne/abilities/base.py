"""Base Ability class — all abilities inherit from this."""

from abc import ABC, abstractmethod
from typing import Optional


class Ability(ABC):
    """Base class for all Syne abilities.
    
    Abilities are modular capabilities that extend Syne's functionality.
    Each ability implements a standard interface for:
    - Execution with parameters and context
    - Schema generation for LLM function calling
    - Configuration validation
    """
    
    name: str
    description: str
    version: str = "1.0"
    
    @abstractmethod
    async def execute(self, params: dict, context: dict) -> dict:
        """Execute the ability with given parameters.
        
        Args:
            params: Parameters from LLM function call
            context: Execution context containing:
                - user_id: Database user ID
                - session_id: Current session ID
                - access_level: User's access level (public/friend/family/admin/owner)
                - config: Ability-specific configuration from DB
        
        Returns:
            dict with keys:
                - success: bool - Whether execution succeeded
                - result: any - Ability-specific output (string, dict, etc.)
                - error: str - Error message if success=False
                - media: Optional[str] - File path for images/audio output
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> dict:
        """Return JSON schema for LLM function calling.
        
        Returns OpenAI-compatible function schema:
        {
            "type": "function",
            "function": {
                "name": "ability_name",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
        """
        pass
    
    def get_required_config(self) -> list[str]:
        """Return list of required config keys (e.g., API keys).
        
        These keys must be present in the ability's config for it to work.
        """
        return []
    
    async def validate_config(self, config: dict) -> tuple[bool, str]:
        """Validate that required config is present and valid.
        
        Args:
            config: Configuration dict from DB (abilities.config column)
        
        Returns:
            Tuple of (is_valid, error_message)
            - (True, "") if valid
            - (False, "Missing required config: XYZ") if invalid
        """
        for key in self.get_required_config():
            if key not in config or not config[key]:
                return False, f"Missing required config: {key}"
        return True, ""
