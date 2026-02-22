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
    - Pre-processing input (ability-first strategy)
    
    ## Ability-First Principle
    
    Syne always prefers abilities over native LLM capabilities.
    If an ability can handle an input type (image, audio, document, etc.),
    it runs BEFORE the LLM sees the raw input. The LLM only gets the
    ability's processed result as text context.
    
    To participate in pre-processing, override `handles_input_type()` and
    `pre_process()`. The engine calls these BEFORE building the LLM context.
    """
    
    name: str
    description: str
    version: str = "1.0"
    
    def handles_input_type(self, input_type: str) -> bool:
        """Check if this ability can pre-process a given input type.
        
        Override this to declare what input types this ability handles.
        The engine calls this during the ability-first dispatch phase.
        
        Args:
            input_type: Type of input (e.g. "image", "audio", "document")
            
        Returns:
            True if this ability can pre-process this input type
        """
        return False
    
    async def pre_process(self, input_type: str, input_data: dict, user_prompt: str) -> Optional[str]:
        """Pre-process an input before it reaches the LLM.
        
        Called by the engine when `handles_input_type()` returns True.
        The result (if any) is injected as text context, and the raw input
        is stripped from the LLM message to avoid double-processing.
        
        Args:
            input_type: Type of input (e.g. "image", "audio", "document")
            input_data: Raw input data dict. Contents depend on input_type:
                - image: {"base64": str, "mime_type": str}
                - audio: {"base64": str, "mime_type": str, "duration": float}
                - document: {"base64": str, "mime_type": str, "filename": str}
            user_prompt: The user's message text (for context-aware processing)
            
        Returns:
            Processed text result, or None if pre-processing failed.
            When None is returned, the engine falls back to native LLM capability.
        """
        return None
    
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
