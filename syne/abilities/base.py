"""Base Ability class — all abilities inherit from this."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


def _get_workspace_root() -> str:
    """Return the workspace root directory (project_root/workspace/)."""
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    return os.path.join(project_root, "workspace")


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
    
    ## File Output Convention
    
    All abilities MUST use `self.get_output_dir()` for file output.
    This returns `workspace/outputs/` — a centralized location that keeps
    generated files organized and out of the project root.
    
    Example in execute():
        outdir = self.get_output_dir()
        out_path = os.path.join(outdir, "result.pdf")
        
    For session-isolated output:
        outdir = self.get_output_dir(session_id=context.get("session_id"))
        # Returns workspace/outputs/session_<id>/
    """
    
    name: str
    description: str
    version: str = "1.0"
    
    # Set to False to opt-out of ability-first pre-processing.
    # Default True = this ability is always tried first before LLM native.
    # Override in subclass or set via chat to disable priority for specific abilities.
    priority: bool = True
    
    def get_output_dir(self, session_id: Optional[str] = None) -> str:
        """Return the output directory for this ability.
        
        All generated files (PDFs, images, screenshots, reports) go here.
        Path: workspace/outputs/ (or workspace/outputs/session_<id>/ if isolated)
        
        The directory is created automatically if it doesn't exist.
        
        Args:
            session_id: Optional session ID for per-session isolation
            
        Returns:
            Absolute path to the output directory
        """
        base = os.path.join(_get_workspace_root(), "outputs")
        if session_id:
            base = os.path.join(base, f"session_{session_id}")
        os.makedirs(base, exist_ok=True)
        return base

    @staticmethod
    def get_uploads_dir() -> str:
        """Return the uploads directory (files received from users).
        
        Path: workspace/uploads/
        """
        d = os.path.join(_get_workspace_root(), "uploads")
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def get_temp_dir() -> str:
        """Return the temp directory for scratch files.
        
        Path: workspace/temp/
        """
        d = os.path.join(_get_workspace_root(), "temp")
        os.makedirs(d, exist_ok=True)
        return d

    def handles_input_type(self, input_type: str) -> bool:
        """Check if this ability can pre-process a given input type.
        
        DEFAULT: False — abilities that don't handle raw input (like image_gen,
        maps, web_screenshot) don't need pre-processing. They're invoked
        via LLM tool calls instead.
        
        Override to return True for abilities that CAN process raw input
        data before LLM sees it (e.g. image_analysis, audio_transcription).
        
        The engine only calls this for abilities where priority=True.
        
        Args:
            input_type: Type of input (e.g. "image", "audio", "document")
            
        Returns:
            True if this ability can pre-process this input type
        """
        return False
    
    async def pre_process(
        self, input_type: str, input_data: dict, user_prompt: str,
        config: Optional[dict] = None
    ) -> Optional[str]:
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
            config: Ability config from DB (API keys, settings, etc.)
            
        Returns:
            Processed text result, or None if pre-processing failed.
            When None is returned, the engine falls back to native LLM capability.
        """
        return None
    
    async def call_ability(self, name: str, params: dict, context: dict) -> dict:
        """Call another ability by name from within this ability.
        
        This delegates to the AbilityRegistry so abilities can compose
        each other (e.g. schedule_daily_prayer calls prayer_times).
        
        Args:
            name: Target ability name
            params: Parameters for the target ability
            context: Execution context (passed through from this ability's execute())
            
        Returns:
            Result dict from the target ability
        """
        # Import here to avoid circular imports
        from .registry import AbilityRegistry
        
        # Access the global registry via the context
        registry: AbilityRegistry = context.get("_registry")
        if registry is None:
            return {
                "success": False,
                "error": f"Cannot call ability '{name}': no registry in context. "
                         "Ensure context['_registry'] is set by the engine.",
            }
        
        return await registry.execute(name, params, context)

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
