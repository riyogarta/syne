"""Syne Agent — ties everything together."""

import logging
from pathlib import Path
from typing import Optional

from .config import SyneSettings
from .db.connection import init_db, close_db
from .db.models import get_config, set_config, get_or_create_user
from .llm.provider import LLMProvider
from .llm.google import GoogleProvider
from .llm.openai import OpenAIProvider
from .llm.codex import CodexProvider
from .llm.together import TogetherProvider
from .llm.hybrid import HybridProvider
from .llm.drivers import (
    create_hybrid_provider,
    get_model_from_list,
)
from .auth.google_oauth import get_credentials
from .memory.engine import MemoryEngine
from .tools.registry import ToolRegistry
from .abilities import AbilityRegistry
from .abilities.loader import load_all_abilities
from .context import ContextManager
from .conversation import ConversationManager
from .subagent import SubAgentManager
from .security import check_command_safety, check_rule_removal
from .ratelimit import init_rate_limiter_from_config

logger = logging.getLogger("syne.agent")


class SyneAgent:
    """The main Syne agent. Initializes all components and handles messages."""

    def __init__(self, settings: SyneSettings):
        self.settings = settings
        self.provider: Optional[LLMProvider] = None
        self.memory: Optional[MemoryEngine] = None
        self.tools: ToolRegistry = ToolRegistry()
        self.abilities: AbilityRegistry = AbilityRegistry()
        self.context_mgr: Optional[ContextManager] = None
        self.conversations: Optional[ConversationManager] = None
        self.subagents: Optional[SubAgentManager] = None
        self._running = False
        self._pending_sudo_command: Optional[str] = None
        self._cli_cwd: Optional[str] = None  # Set by CLI channel to override exec cwd

    async def start(self):
        """Start the agent — initialize DB, provider, memory, tools."""
        logger.info("Starting Syne agent...")

        # 1. Database
        await init_db(self.settings.database_url)
        logger.info("Database connected.")

        # 2. LLM Provider
        self.provider = await self._init_provider()
        logger.info(f"LLM provider: {self.provider.name}")

        # 3. Memory
        self.memory = MemoryEngine(self.provider)
        logger.info("Memory engine ready.")

        # 4. Built-in Tools (memory, subagent)
        self._register_default_tools()
        logger.info(f"Built-in tools registered: {len(self.tools.list_tools('owner'))}")

        # 5. Abilities (modular capabilities)
        ability_count = await load_all_abilities(self.abilities)
        logger.info(f"Abilities loaded: {ability_count}")

        # 6. Context Manager
        max_context = await get_config("session.compaction_threshold", 80000)
        self.context_mgr = ContextManager(max_context_tokens=max_context)
        logger.info(f"Context window: {max_context} tokens")

        # 6.5. Rate Limiter
        await init_rate_limiter_from_config()
        logger.info("Rate limiter initialized.")

        # 7. Sub-agent Manager
        from .boot import get_full_prompt
        # Build prompt with tools and abilities for owner access level
        tool_schemas = self.tools.to_openai_schema("owner")
        ability_schemas = self.abilities.to_openai_schema("owner")
        system_prompt = await get_full_prompt(
            user={"access_level": "owner"},
            tools=tool_schemas,
            abilities=ability_schemas,
        )
        self.subagents = SubAgentManager(
            provider=self.provider,
            system_prompt=system_prompt,
        )
        self.subagents.tools = self.tools
        self.subagents.abilities = self.abilities
        logger.info("Sub-agent manager ready (with tool access).")

        # 8. Conversation Manager
        self.conversations = ConversationManager(
            provider=self.provider,
            memory=self.memory,
            tools=self.tools,
            abilities=self.abilities,
            context_mgr=self.context_mgr,
            subagents=self.subagents,
        )

        self._running = True
        logger.info("Syne agent started.")

    async def stop(self):
        """Stop the agent gracefully."""
        self._running = False
        await close_db()
        logger.info("Syne agent stopped.")

    async def reload_provider(self):
        """Hot-reload the LLM provider from DB config.
        
        Called after /model switch to apply the new provider without restart.
        Updates: self.provider, memory engine, conversation manager, sub-agents.
        """
        new_provider = await self._init_provider()
        self.provider = new_provider
        logger.info(f"Provider reloaded: {new_provider.name}")

        # Update memory engine
        self.memory = MemoryEngine(new_provider)

        # Update conversation manager's provider + memory
        if self.conversations:
            self.conversations.provider = new_provider
            self.conversations.memory = self.memory
            # Update all active conversations
            for conv in self.conversations._active.values():
                conv.provider = new_provider
                conv.memory = self.memory

        # Update sub-agent manager
        if self.subagents:
            self.subagents.provider = new_provider

    async def _init_provider(self) -> LLMProvider:
        """Initialize the LLM provider based on config.
        
        Uses the driver-based model registry system:
        1. Read provider.active_model (key into provider.models)
        2. Look up the model entry from provider.models
        3. Use create_hybrid_provider() to instantiate
        
        Backward compatibility:
        - If provider.primary exists (old format), migrate to new format
        """
        # ═══════════════════════════════════════════════════════════════
        # Check for new driver-based model system
        # ═══════════════════════════════════════════════════════════════
        models = await get_config("provider.models", None)
        active_model_key = await get_config("provider.active_model", None)
        
        if models and active_model_key:
            # New driver-based system
            model_entry = get_model_from_list(models, active_model_key)
            if model_entry:
                logger.info(f"Using driver-based model: {model_entry.get('label', active_model_key)}")
                return await create_hybrid_provider(model_entry)
            else:
                logger.warning(f"Model key '{active_model_key}' not found in registry, falling back to default")
        
        # ═══════════════════════════════════════════════════════════════
        # Backward compatibility: old provider.primary system
        # Migrate to new format if possible
        # ═══════════════════════════════════════════════════════════════
        provider_config = await get_config("provider.primary", {"name": "google", "auth": "oauth"})
        provider_name = provider_config.get("name", "google") if isinstance(provider_config, dict) else "google"
        chat_model = await get_config("provider.chat_model", "gemini-2.5-pro")
        
        # Try to migrate to new format
        await self._migrate_to_driver_system(provider_name, chat_model)
        
        # Now use driver system
        models = await get_config("provider.models", None)
        active_model_key = await get_config("provider.active_model", None)
        
        if models and active_model_key:
            model_entry = get_model_from_list(models, active_model_key)
            if model_entry:
                return await create_hybrid_provider(model_entry)
        
        # Final fallback: use old system directly
        logger.warning("Falling back to legacy provider initialization")
        return await self._init_provider_legacy(provider_name, chat_model)

    async def _migrate_to_driver_system(self, provider_name: str, chat_model: str):
        """Migrate from old provider.primary format to new driver-based system."""
        # Check if already migrated
        models = await get_config("provider.models", None)
        if models:
            return  # Already has model registry
        
        # Build model registry based on detected provider
        default_models = [
            {"key": "gemini-pro", "label": "Gemini 2.5 Pro", "driver": "google_cca", "model_id": "gemini-2.5-pro", "auth": "oauth", "context_window": 1048576},
            {"key": "gemini-flash", "label": "Gemini 2.5 Flash", "driver": "google_cca", "model_id": "gemini-2.5-flash", "auth": "oauth", "context_window": 1048576},
            {"key": "gpt-5.2", "label": "GPT-5.2", "driver": "codex", "model_id": "gpt-5.2", "auth": "oauth", "context_window": 1048576},
        ]

        # Check if provider.primary has driver info (new init format)
        provider_config = await get_config("provider.primary", None)
        if isinstance(provider_config, dict) and provider_config.get("driver") == "anthropic":
            auth = provider_config.get("auth", "oauth")
            default_models.extend([
                {"key": "claude-sonnet", "label": "Claude Sonnet 4", "driver": "anthropic", "model_id": "claude-sonnet-4-20250514", "auth": auth, "context_window": 200000},
                {"key": "claude-opus", "label": "Claude Opus 4", "driver": "anthropic", "model_id": "claude-opus-4-20250514", "auth": auth, "context_window": 200000},
            ])

        # Map old provider to new model key
        model_key_map = {
            ("google", "gemini-2.5-pro"): "gemini-pro",
            ("google", "gemini-2.5-flash"): "gemini-flash",
            ("codex", "gpt-5.2"): "gpt-5.2",
            ("codex", "gpt-4.1"): "gpt-5.2",  # Map old models
        }

        # Handle anthropic provider mapping
        if isinstance(provider_config, dict) and provider_config.get("driver") == "anthropic":
            model_id = provider_config.get("model", chat_model)
            if "opus" in model_id:
                active_key = "claude-opus"
            else:
                active_key = "claude-sonnet"
        else:
            active_key = model_key_map.get((provider_name, chat_model), "gemini-pro")
        
        # Save new format
        await set_config("provider.models", default_models)
        await set_config("provider.active_model", active_key)
        logger.info(f"Migrated to driver-based model system: {active_key}")

    async def _init_provider_legacy(self, provider_name: str, chat_model: str) -> LLMProvider:
        """Legacy provider initialization for backward compatibility."""
        import os
        import json
        
        embedding_model = await get_config("provider.embedding_model", "text-embedding-004")

        if provider_name == "google":
            creds = await get_credentials()
            if not creds:
                raise RuntimeError("No Google OAuth credentials found. Run 'syne init' to authenticate.")
            chat_provider = GoogleProvider(
                credentials=creds,
                chat_model=chat_model,
            )
            
            together_key = os.environ.get("TOGETHER_API_KEY")
            if not together_key:
                together_key = await get_config("credential.together_api_key", None)

            if together_key:
                embed_provider = TogetherProvider(
                    api_key=together_key,
                    embedding_model="BAAI/bge-base-en-v1.5",
                )
                return HybridProvider(chat_provider=chat_provider, embed_provider=embed_provider)
            return chat_provider

        elif provider_name == "codex":
            access_token = await get_config("credential.codex_access_token", "")
            refresh_token = await get_config("credential.codex_refresh_token", "")
            if not access_token:
                access_token = os.environ.get("CODEX_ACCESS_TOKEN", "")
                refresh_token = os.environ.get("CODEX_REFRESH_TOKEN", "")
            
            chat_provider = CodexProvider(
                access_token=access_token,
                refresh_token=refresh_token,
                chat_model=chat_model,
            )
            
            together_key = os.environ.get("TOGETHER_API_KEY")
            if not together_key:
                together_key = await get_config("credential.together_api_key", None)

            if together_key:
                embed_provider = TogetherProvider(
                    api_key=together_key,
                    embedding_model="BAAI/bge-base-en-v1.5",
                )
                return HybridProvider(chat_provider=chat_provider, embed_provider=embed_provider)
            return chat_provider

        elif provider_name == "groq":
            groq_key = await get_config("credential.groq_api_key", None)
            if not groq_key:
                groq_key = os.environ.get("GROQ_API_KEY")
            
            chat_provider = OpenAIProvider(
                api_key=groq_key,
                chat_model=chat_model,
                base_url="https://api.groq.com/openai/v1",
                provider_name="groq",
            )
            
            together_key = os.environ.get("TOGETHER_API_KEY")
            if not together_key:
                together_key = await get_config("credential.together_api_key", None)

            if together_key:
                embed_provider = TogetherProvider(
                    api_key=together_key,
                    embedding_model="BAAI/bge-base-en-v1.5",
                )
                return HybridProvider(chat_provider=chat_provider, embed_provider=embed_provider)
            return chat_provider

        else:
            raise RuntimeError(f"Unknown provider: {provider_name}")

    def _register_default_tools(self):
        """Register built-in tools."""
        # ── Web Search (Core) ──
        from .tools.web_search import WEB_SEARCH_TOOL
        self.tools.register(
            name=WEB_SEARCH_TOOL["name"],
            description=WEB_SEARCH_TOOL["description"],
            parameters=WEB_SEARCH_TOOL["parameters"],
            handler=WEB_SEARCH_TOOL["handler"],
            requires_access_level=WEB_SEARCH_TOOL["requires_access_level"],
        )

        # ── Web Fetch (Core) ──
        from .tools.web_fetch import WEB_FETCH_TOOL
        self.tools.register(
            name=WEB_FETCH_TOOL["name"],
            description=WEB_FETCH_TOOL["description"],
            parameters=WEB_FETCH_TOOL["parameters"],
            handler=WEB_FETCH_TOOL["handler"],
            requires_access_level=WEB_FETCH_TOOL["requires_access_level"],
        )

        # ── Read Source (Core) ──
        from .tools.read_source import READ_SOURCE_TOOL
        self.tools.register(
            name=READ_SOURCE_TOOL["name"],
            description=READ_SOURCE_TOOL["description"],
            parameters=READ_SOURCE_TOOL["parameters"],
            handler=READ_SOURCE_TOOL["handler"],
            requires_access_level=READ_SOURCE_TOOL["requires_access_level"],
        )

        # ── File Operations (Core) ──
        from .tools.file_ops import FILE_READ_TOOL, FILE_WRITE_TOOL
        self.tools.register(
            name=FILE_READ_TOOL["name"],
            description=FILE_READ_TOOL["description"],
            parameters=FILE_READ_TOOL["parameters"],
            handler=FILE_READ_TOOL["handler"],
            requires_access_level=FILE_READ_TOOL["requires_access_level"],
        )
        self.tools.register(
            name=FILE_WRITE_TOOL["name"],
            description=FILE_WRITE_TOOL["description"],
            parameters=FILE_WRITE_TOOL["parameters"],
            handler=FILE_WRITE_TOOL["handler"],
            requires_access_level=FILE_WRITE_TOOL["requires_access_level"],
        )

        # ── Send File (Core) ──
        from .tools.send_file import SEND_FILE_TOOL
        self.tools.register(
            name=SEND_FILE_TOOL["name"],
            description=SEND_FILE_TOOL["description"],
            parameters=SEND_FILE_TOOL["parameters"],
            handler=SEND_FILE_TOOL["handler"],
            requires_access_level=SEND_FILE_TOOL["requires_access_level"],
        )

        # ── Scheduler (Core) ──
        from .tools.scheduler import MANAGE_SCHEDULE_TOOL
        self.tools.register(
            name=MANAGE_SCHEDULE_TOOL["name"],
            description=MANAGE_SCHEDULE_TOOL["description"],
            parameters=MANAGE_SCHEDULE_TOOL["parameters"],
            handler=MANAGE_SCHEDULE_TOOL["handler"],
            requires_access_level=MANAGE_SCHEDULE_TOOL["requires_access_level"],
        )

        # ── Send Reaction (Core) ──
        from .tools.reactions import SEND_REACTION_TOOL
        self.tools.register(
            name=SEND_REACTION_TOOL["name"],
            description=SEND_REACTION_TOOL["description"],
            parameters=SEND_REACTION_TOOL["parameters"],
            handler=SEND_REACTION_TOOL["handler"],
            requires_access_level=SEND_REACTION_TOOL["requires_access_level"],
        )

        # ── Send Voice (Core) ──
        from .tools.voice import SEND_VOICE_TOOL
        self.tools.register(
            name=SEND_VOICE_TOOL["name"],
            description=SEND_VOICE_TOOL["description"],
            parameters=SEND_VOICE_TOOL["parameters"],
            handler=SEND_VOICE_TOOL["handler"],
            requires_access_level=SEND_VOICE_TOOL["requires_access_level"],
        )

        # ── Exec (Core) ──
        self.tools.register(
            name="exec",
            description="Execute a shell command on the host system. Returns stdout, stderr, and exit code. Use for file operations, system checks, installing packages, running scripts, or any system task. Commands run as the Syne process user.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (bash)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30, max 300)",
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory (default: home directory)",
                    },
                },
                "required": ["command"],
            },
            handler=self._tool_exec,
            requires_access_level="owner",
        )

        # Memory search tool
        self.tools.register(
            name="memory_search",
            description="Search through stored memories for relevant information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant memories",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            handler=self._tool_memory_search,
        )

        # Memory store tool
        self.tools.register(
            name="memory_store",
            description="Store important information as a long-term memory.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Information to remember",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["fact", "preference", "event", "lesson", "decision"],
                        "description": "Memory category",
                    },
                },
                "required": ["content", "category"],
            },
            handler=self._tool_memory_store,
            requires_access_level="family",
        )

        # Sub-agent spawn tool
        self.tools.register(
            name="spawn_subagent",
            description="Spawn a background sub-agent to work on a task in parallel. Use for heavy tasks like documentation, research, or analysis while continuing the conversation.",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Detailed task description for the sub-agent",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context from current conversation to help the sub-agent",
                    },
                },
                "required": ["task"],
            },
            handler=self._tool_spawn_subagent,
            requires_access_level="family",
        )

        # Sub-agent status tool
        self.tools.register(
            name="subagent_status",
            description="Check status of running sub-agents or get result of a completed sub-agent.",
            parameters={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Specific run ID to check. Omit to list all active.",
                    },
                },
                "required": [],
            },
            handler=self._tool_subagent_status,
        )

        # ── Group Management (owner only) ──
        self.tools.register(
            name="manage_group",
            description="Add, update, remove, or list Telegram groups. Use to control which groups the bot responds in.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "update", "remove", "list"],
                        "description": "Action to perform",
                    },
                    "group_id": {
                        "type": "string",
                        "description": "Telegram group ID (e.g. -949261612)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Friendly name for the group",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether bot responds in this group",
                    },
                    "require_mention": {
                        "type": "boolean",
                        "description": "Whether bot requires @mention or trigger name",
                    },
                    "allow_from": {
                        "type": "string",
                        "enum": ["all", "registered"],
                        "description": "'all' = anyone can trigger bot, 'registered' = only known users",
                    },
                },
                "required": ["action"],
            },
            handler=self._tool_manage_group,
            requires_access_level="owner",
        )

        # ── User Management ──
        self.tools.register(
            name="manage_user",
            description="View, update user settings including aliases/display names per group.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "update", "get"],
                        "description": "Action to perform",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User platform ID (e.g. Telegram user ID)",
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Default display name for the user",
                    },
                    "aliases": {
                        "type": "string",
                        "description": "JSON object with aliases: {\"default\": \"Name\", \"groups\": {\"-123\": \"Other Name\"}}",
                    },
                    "access_level": {
                        "type": "string",
                        "enum": ["owner", "admin", "family", "friend", "public"],
                        "description": "User access level",
                    },
                },
                "required": ["action"],
            },
            handler=self._tool_manage_user,
            requires_access_level="owner",  # Note: get action checks differently in handler
        )

        # ── Self-Configuration Tools (owner only) ──

        # Update config
        self.tools.register(
            name="update_config",
            description="Read or update a configuration setting in the database. Use action 'get' to read, 'set' to write, 'list' to show all settings.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "set", "list"],
                        "description": "Action to perform",
                    },
                    "key": {
                        "type": "string",
                        "description": "Config key (e.g. 'memory.auto_capture', 'provider.chat_model')",
                    },
                    "value": {
                        "type": "string",
                        "description": "New value (as JSON string). For set action only.",
                    },
                },
                "required": ["action"],
            },
            handler=self._tool_update_config,
            requires_access_level="owner",
        )

        # Update ability config
        self.tools.register(
            name="update_ability",
            description=(
                "Manage abilities. Actions: "
                "'list' — show all abilities; "
                "'create' — register a new self-created ability (provide name, description, module_path for the Python file you wrote to syne/abilities/); "
                "'enable'/'disable' — toggle an ability; "
                "'config' — update ability settings (JSON)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "enable", "disable", "config"],
                        "description": "Action to perform",
                    },
                    "name": {
                        "type": "string",
                        "description": "Ability name",
                    },
                    "description": {
                        "type": "string",
                        "description": "Ability description (for create action)",
                    },
                    "module_path": {
                        "type": "string",
                        "description": "Python module path e.g. 'syne.abilities.screenshot' (for create action)",
                    },
                    "config": {
                        "type": "string",
                        "description": "JSON config to merge into ability config (for config action)",
                    },
                },
                "required": ["action"],
            },
            handler=self._tool_update_ability,
            requires_access_level="owner",
        )

        # Update identity/soul/rules
        self.tools.register(
            name="update_soul",
            description="Update identity, soul entries, or rules. Use target 'identity' to set name/motto/personality, 'soul' to add/remove behavior entries, 'rules' to add/remove rules.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["identity", "soul", "rules"],
                        "description": "What to update",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["get", "set", "add", "remove"],
                        "description": "Action: get=read, set=update value, add=add entry, remove=delete entry",
                    },
                    "key": {
                        "type": "string",
                        "description": "Key name. For identity: name/motto/personality. For soul: category. For rules: code.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to set or content to add",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["hard", "soft"],
                        "description": "Rule severity (for rules add only)",
                    },
                },
                "required": ["target", "action"],
            },
            handler=self._tool_update_soul,
            requires_access_level="owner",
        )

    async def _tool_spawn_subagent(self, task: str, context: str = "") -> str:
        """Tool handler: spawn a sub-agent."""
        if not self.subagents:
            return "Sub-agent system not initialized."
        
        # Get parent session ID from the currently active conversation
        parent_session_id = 0
        conv = self._get_active_conversation()
        if conv and hasattr(conv, 'session_id'):
            parent_session_id = conv.session_id
        
        if parent_session_id == 0:
            return "Cannot spawn sub-agent: no active session found."
        
        result = await self.subagents.spawn(
            task=task,
            parent_session_id=parent_session_id,
            context=context or None,
        )
        
        if result["success"]:
            return f"Sub-agent spawned (run_id: {result['run_id']}). {result['message']}"
        return f"Failed to spawn: {result['error']}"

    async def _tool_subagent_status(self, run_id: str = "") -> str:
        """Tool handler: check sub-agent status."""
        if not self.subagents:
            return "Sub-agent system not initialized."
        
        if run_id:
            run = await self.subagents.get_run(run_id)
            if not run:
                return f"No sub-agent found with run_id: {run_id}"
            
            lines = [
                f"**Sub-agent {run['run_id'][:8]}**",
                f"Status: {run['status']}",
                f"Task: {run['task'][:100]}",
            ]
            if run['status'] == 'completed':
                result_preview = run['result'][:500] if run['result'] else 'No result'
                lines.append(f"Result: {result_preview}")
            elif run['status'] == 'failed':
                lines.append(f"Error: {run['error']}")
            
            tokens = (run.get('input_tokens', 0) or 0) + (run.get('output_tokens', 0) or 0)
            if tokens:
                lines.append(f"Tokens: {tokens}")
            
            return "\n".join(lines)
        
        # List all active
        active = await self.subagents.list_active()
        if not active:
            return "No active sub-agents."
        
        lines = [f"**Active sub-agents ({len(active)}):**"]
        for run in active:
            lines.append(f"- {run['run_id'][:8]}: {run['task'][:60]}")
        return "\n".join(lines)

    def _get_active_conversation(self):
        """Get the conversation that is currently processing a message."""
        if not self.conversations:
            return None
        # Return the most recently active conversation
        for conv in self.conversations._active.values():
            if hasattr(conv, '_processing') and conv._processing:
                return conv
        # Fallback: return any active conversation
        active = list(self.conversations._active.values())
        return active[-1] if active else None

    def _sudo_confirmed_recently(self, conv) -> bool:
        """Check if user explicitly confirmed sudo in recent messages."""
        if not hasattr(conv, '_message_cache') or not conv._message_cache:
            return False
        # Check last 3 user messages for confirmation
        user_msgs = [m for m in conv._message_cache[-6:] if m.role == "user"]
        for msg in reversed(user_msgs[-3:]):
            content = (msg.content or "").strip().lower()
            if content in ("ya", "yes", "ok", "oke", "lanjut", "proceed", "confirm"):
                return True
        return False

    async def _tool_exec(self, command: str, timeout: int = 30, workdir: str = "") -> str:
        """Tool handler: execute shell command."""
        import asyncio
        import os
        from .db.models import get_config

        # ═══════════════════════════════════════════════════════════════
        # SECURITY: Command Blacklist Check
        # Check for dangerous command patterns BEFORE execution
        # ═══════════════════════════════════════════════════════════════
        safe, reason = check_command_safety(command)
        if not safe:
            logger.warning(f"Blocked dangerous command: {command[:100]}")
            return f"Error: {reason}"

        # ═══════════════════════════════════════════════════════════════
        # SECURITY: Sudo Guard
        # sudo commands require:
        #   1. Owner access level
        #   2. Direct message (NOT group chat)
        #   3. Explicit user confirmation in the same conversation
        # The LLM should ask the user first; if it calls sudo directly,
        # this guard blocks it and returns a message to ask the user.
        # ═══════════════════════════════════════════════════════════════
        command_stripped = command.strip()
        is_sudo = command_stripped.startswith("sudo ") or "| sudo " in command_stripped or "&& sudo " in command_stripped

        if is_sudo:
            # Check which conversation is calling this
            active_conv = self._get_active_conversation()
            is_group = active_conv.is_group if active_conv else False
            access_level = active_conv.user.get("access_level", "public") if active_conv else "public"

            if access_level != "owner":
                logger.warning(f"Sudo blocked: non-owner attempted sudo: {command[:100]}")
                return "Error: sudo commands are only allowed for the owner."

            if is_group:
                logger.warning(f"Sudo blocked in group: {command[:100]}")
                return "Error: sudo commands are not allowed in group chats. Please use a direct message."

            # Check if user explicitly confirmed in recent messages
            if active_conv and not self._sudo_confirmed_recently(active_conv):
                logger.info(f"Sudo requires confirmation: {command[:100]}")
                self._pending_sudo_command = command
                return (
                    "⚠️ This command requires sudo (root access). "
                    "Please confirm you want to run this command by replying 'ya' or 'yes'.\n\n"
                    f"Command: `{command}`"
                )

        timeout_max = await get_config("exec.timeout_max", 300)
        output_max = await get_config("exec.output_max_chars", 4000)

        timeout = min(max(timeout, 1), timeout_max)
        # Default cwd:
        # - CLI mode: directory where user ran `syne cli`
        # - Telegram/other: project root (for ability self-edit)
        project_root = str(Path(__file__).resolve().parent.parent)
        if workdir:
            cwd = workdir
        elif self._cli_cwd:
            cwd = self._cli_cwd
        else:
            cwd = project_root

        logger.info(f"Executing command: {command[:100]}")

        # ── CLI PTY mode: interactive commands get a real terminal ──
        # When running from `syne cli`, commands that need user input
        # (sudo, ssh, etc.) run with PTY pass-through so the user can
        # type passwords/confirmations directly in their terminal.
        if self._cli_cwd and self._command_needs_interactive(command_stripped):
            return await self._exec_pty(command, cwd, timeout)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            parts = []
            if stdout:
                out = stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
                if out:
                    parts.append(f"stdout:\n{out[:output_max]}")
            if stderr:
                err = stderr.decode("utf-8", errors="replace").replace("\x00", "").strip()
                if err:
                    parts.append(f"stderr:\n{err[:output_max // 2]}")
            parts.append(f"exit_code: {proc.returncode}")

            return "\n".join(parts) if parts else f"exit_code: {proc.returncode}"

        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            logger.error(f"Exec error: {e}")
            return f"Error: {str(e)}"

    @staticmethod
    def _command_needs_interactive(command: str) -> bool:
        """Detect if a command likely needs interactive terminal input.
        
        This checks for commands that prompt for passwords or confirmations
        that can't work with piped stdin (e.g. sudo, ssh, passwd).
        """
        cmd = command.strip()
        # sudo at start or in a chain
        if cmd.startswith("sudo ") or "| sudo " in cmd or "&& sudo " in cmd:
            return True
        # ssh without -o BatchMode (interactive login)
        if cmd.startswith("ssh ") and "-o BatchMode" not in cmd:
            return True
        # Other known interactive commands
        for prefix in ("passwd", "su ", "su\n"):
            if cmd.startswith(prefix):
                return True
        return False

    async def _exec_pty(self, command: str, cwd: str, timeout: int) -> str:
        """Execute command with PTY pass-through for interactive input.
        
        Runs in a real pseudo-terminal so the user can type passwords,
        confirmations, etc. directly. Output is captured AND displayed
        live in the terminal.
        
        Only used in CLI mode.
        """
        import asyncio
        import os
        import pty
        import select
        import signal
        import sys

        logger.info(f"PTY exec (interactive): {command[:100]}")

        # Run the blocking PTY operation in a thread to not block the event loop
        def _pty_run() -> tuple[int, str]:
            """Spawn command in PTY, forward I/O, capture output."""
            output_chunks: list[bytes] = []

            # Create PTY pair
            parent_fd, child_fd = pty.openpty()

            pid = os.fork()
            if pid == 0:
                # ── Child process ──
                os.setsid()
                os.dup2(child_fd, 0)  # stdin
                os.dup2(child_fd, 1)  # stdout
                os.dup2(child_fd, 2)  # stderr
                os.close(parent_fd)
                os.close(child_fd)
                os.chdir(cwd)
                os.execvp("/bin/bash", ["/bin/bash", "-c", command])
                os._exit(1)  # shouldn't reach here
            else:
                # ── Parent process ──
                os.close(child_fd)
                
                # Print a subtle indicator that interactive mode is active
                sys.stdout.write("\033[2m── interactive ──\033[0m\n")
                sys.stdout.flush()

                try:
                    while True:
                        # Wait for data from PTY or stdin
                        rlist, _, _ = select.select(
                            [parent_fd, sys.stdin.fileno()], [], [], 1.0
                        )

                        if parent_fd in rlist:
                            try:
                                data = os.read(parent_fd, 4096)
                            except OSError:
                                break
                            if not data:
                                break
                            # Write to user's terminal AND capture
                            sys.stdout.buffer.write(data)
                            sys.stdout.flush()
                            output_chunks.append(data)

                        if sys.stdin.fileno() in rlist:
                            try:
                                user_data = os.read(sys.stdin.fileno(), 4096)
                            except OSError:
                                break
                            if not user_data:
                                break
                            # Forward user input to the PTY child
                            os.write(parent_fd, user_data)

                except KeyboardInterrupt:
                    # Ctrl+C → send SIGINT to child
                    try:
                        os.kill(pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass

                finally:
                    os.close(parent_fd)

                # Wait for child to finish
                _, status = os.waitpid(pid, 0)
                exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1

                sys.stdout.write("\033[2m── end ──\033[0m\n")
                sys.stdout.flush()

                captured = b"".join(output_chunks)
                text = captured.decode("utf-8", errors="replace").strip()
                return exit_code, text

        loop = asyncio.get_event_loop()
        try:
            exit_code, output = await asyncio.wait_for(
                loop.run_in_executor(None, _pty_run),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return f"Error: Interactive command timed out after {timeout}s"
        except Exception as e:
            logger.error(f"PTY exec error: {e}", exc_info=True)
            return f"Error: {str(e)}"

        from .db.models import get_config
        output_max = await get_config("exec.output_max_chars", 4000)

        parts = []
        if output:
            parts.append(f"stdout:\n{output[:output_max]}")
        parts.append(f"exit_code: {exit_code}")
        return "\n".join(parts)

    @staticmethod
    def _mask_sensitive_value(key: str, value) -> str:
        """Mask credential values to prevent prompt injection exposure.
        
        Keys containing these patterns get masked: api_key, token, secret, password, credential.
        Shows first 4 and last 4 chars: 'sk-abc...xyz9'
        """
        sensitive_patterns = ("api_key", "token", "secret", "password", "credential", "_key")
        key_lower = key.lower()
        if any(p in key_lower for p in sensitive_patterns):
            s = str(value)
            if len(s) > 12:
                return f"{s[:4]}...{s[-4:]}"
            elif len(s) > 4:
                return f"{s[:2]}...{s[-2:]}"
            else:
                return "***"
        return str(value)

    async def _tool_update_config(self, action: str, key: str = "", value: str = "") -> str:
        """Tool handler: read/write config."""
        from .db.models import get_config, set_config
        from .db.connection import get_connection

        if action == "list":
            async with get_connection() as conn:
                rows = await conn.fetch("SELECT key, value FROM config ORDER BY key")
            if not rows:
                return "No config entries."
            lines = ["**Current Configuration:**"]
            for row in rows:
                masked = self._mask_sensitive_value(row['key'], row['value'])
                lines.append(f"- `{row['key']}`: {masked}")
            return "\n".join(lines)

        if action == "get":
            if not key:
                return "Error: key is required for get action."
            val = await get_config(key)
            if val is None:
                return f"Config key '{key}' not found."
            masked = self._mask_sensitive_value(key, val)
            return f"`{key}` = {masked}"

        if action == "set":
            if not key or not value:
                return "Error: key and value are required for set action."
            import json
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value  # Store as string
            await set_config(key, parsed)
            
            # Auto-sync: if chat_model changes, find matching model in registry
            # and update active_model to keep them in sync
            if key == "provider.chat_model":
                models = await get_config("provider.models", [])
                for m in models:
                    if m.get("model_id") == parsed or m.get("key") == parsed:
                        await set_config("provider.active_model", m["key"])
                        break
            
            return f"Config updated: `{key}` = {parsed}"

        return f"Unknown action: {action}"

    async def _tool_update_ability(self, action: str, name: str = "", description: str = "", module_path: str = "", config: str = "") -> str:
        """Tool handler: manage abilities."""
        if action == "list":
            all_abilities = self.abilities.list_all()
            if not all_abilities:
                return "No abilities registered."
            lines = ["**Abilities:**"]
            for ab in all_abilities:
                status = "✅" if ab.enabled else "❌"
                cfg_preview = str(ab.config)[:80] if ab.config else "{}"
                lines.append(f"- {status} **{ab.name}** v{ab.version} ({ab.source}) — {ab.description}")
                if ab.config:
                    lines.append(f"  Config: `{cfg_preview}`")
            return "\n".join(lines)

        if not name:
            return "Error: ability name is required."

        if action == "create":
            if not module_path:
                module_path = f"syne.abilities.{name}"
            if not description:
                description = f"Self-created ability: {name}"
            
            from .abilities.loader import register_dynamic_ability
            error = await register_dynamic_ability(
                registry=self.abilities,
                name=name,
                description=description,
                module_path=module_path,
            )
            if error:
                return f"Error creating ability: {error}"
            
            # Rebuild system prompt to include new ability schema
            ability_schemas = self.abilities.to_openai_schema("owner")
            tool_schemas = self.tools.to_openai_schema("owner")
            from .boot import build_system_prompt
            self._system_prompt = await build_system_prompt(
                tools=tool_schemas,
                abilities=ability_schemas,
            )
            
            return f"✅ Ability '{name}' created and registered. It is now available as a tool."

        if action == "enable":
            ok = await self.abilities.enable(name)
            return f"Ability '{name}' enabled." if ok else f"Ability '{name}' not found."

        if action == "disable":
            ok = await self.abilities.disable(name)
            return f"Ability '{name}' disabled." if ok else f"Ability '{name}' not found."

        if action == "config":
            if not config:
                # Show current config
                ab = self.abilities.get(name)
                if not ab:
                    return f"Ability '{name}' not found."
                return f"Config for '{name}': {ab.config}"
            import json
            try:
                new_config = json.loads(config)
            except json.JSONDecodeError:
                return "Error: config must be valid JSON."
            ab = self.abilities.get(name)
            if not ab:
                return f"Ability '{name}' not found."
            merged = {**ab.config, **new_config}
            ok = await self.abilities.update_config(name, merged)
            return f"Ability '{name}' config updated: {merged}" if ok else "Failed to update config."

        return f"Unknown action: {action}"

    async def _tool_update_soul(self, target: str, action: str, key: str = "", value: str = "", severity: str = "soft") -> str:
        """Tool handler: update identity, soul, or rules."""
        from .db.connection import get_connection

        if target == "identity":
            if action == "get":
                from .db.models import get_identity
                identity = await get_identity()
                lines = ["**Identity:**"]
                for k, v in identity.items():
                    lines.append(f"- {k}: {v}")
                return "\n".join(lines)
            if action == "set":
                if not key or not value:
                    return "Error: key and value required. Keys: name, motto, personality"
                async with get_connection() as conn:
                    existing = await conn.fetchrow("SELECT id FROM identity WHERE key = $1", key)
                    if existing:
                        await conn.execute(
                            "UPDATE identity SET value = $1, updated_at = NOW() WHERE key = $2",
                            value, key,
                        )
                    else:
                        await conn.execute(
                            "INSERT INTO identity (key, value) VALUES ($1, $2)",
                            key, value,
                        )
                return f"Identity updated: {key} = {value}"

        elif target == "soul":
            if action == "get":
                from .db.models import get_soul
                soul = await get_soul()
                lines = ["**Soul entries:**"]
                for entry in soul:
                    lines.append(f"- [{entry['category']}] {entry['content']}")
                return "\n".join(lines)
            if action == "add":
                if not key or not value:
                    return "Error: key (category) and value (content) required."
                async with get_connection() as conn:
                    await conn.execute(
                        "INSERT INTO soul (category, key, content) VALUES ($1, $2, $3)",
                        key, key, value,
                    )
                return f"Soul entry added: [{key}] {value}"
            if action == "remove":
                if not value:
                    return "Error: value (content to remove) required."
                async with get_connection() as conn:
                    result = await conn.execute(
                        "DELETE FROM soul WHERE content = $1", value,
                    )
                return f"Soul entry removed." if "DELETE 1" in str(result) else "Entry not found."

        elif target == "rules":
            if action == "get":
                from .db.models import get_rules
                rules = await get_rules()
                lines = ["**Rules:**"]
                for rule in rules:
                    marker = "🔴" if rule["severity"] == "hard" else "🟡"
                    lines.append(f"- {marker} [{rule['code']}] {rule['name']}: {rule['description']}")
                return "\n".join(lines)
            if action == "add":
                if not key or not value:
                    return "Error: key (code like 'SEC003') and value (name: description) required."
                parts = value.split(":", 1)
                name = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else name
                async with get_connection() as conn:
                    await conn.execute(
                        "INSERT INTO rules (code, name, description, severity) VALUES ($1, $2, $3, $4)",
                        key, name, desc, severity,
                    )
                return f"Rule added: [{key}] {name}"
            if action == "remove":
                if not key:
                    return "Error: key (rule code) required."
                
                # ═══════════════════════════════════════════════════════
                # SECURITY: Protect core rules from removal
                # Rules with SEC, MEM, IDT prefixes are hardcoded security rules
                # ═══════════════════════════════════════════════════════
                allowed, reason = check_rule_removal(key)
                if not allowed:
                    logger.warning(f"Blocked removal of protected rule: {key}")
                    return f"Error: {reason}"
                
                async with get_connection() as conn:
                    result = await conn.execute("DELETE FROM rules WHERE code = $1", key)
                return f"Rule removed." if "DELETE 1" in str(result) else "Rule not found."

        return f"Unknown target/action: {target}/{action}"

    async def _tool_memory_search(self, query: str, limit: int = 5) -> str:
        """Tool handler: search memories."""
        results = await self.memory.recall(query, limit=limit)
        if not results:
            return "No relevant memories found."

        lines = []
        for mem in results:
            score = f"{mem['similarity']:.0%}"
            lines.append(f"- [{mem['category']}] {mem['content']} (relevance: {score})")
        return "\n".join(lines)

    async def _tool_memory_store(self, content: str, category: str = "fact") -> str:
        """Tool handler: store a memory."""
        mem_id = await self.memory.store_if_new(
            content=content,
            category=category,
            source="user_confirmed",
        )
        if mem_id:
            return f"Memory stored (id: {mem_id})"
        return "Similar memory already exists. Skipped."

    async def _tool_manage_group(
        self,
        action: str,
        group_id: str = "",
        name: str = "",
        enabled: bool = None,
        require_mention: bool = None,
        allow_from: str = "",
    ) -> str:
        """Tool handler: manage Telegram groups."""
        from .db.models import (
            get_group,
            create_group,
            update_group,
            delete_group,
            list_groups,
            get_config,
        )

        if action == "list":
            groups = await list_groups(platform="telegram", enabled_only=False)
            if not groups:
                return "No groups registered."
            
            lines = ["**Registered Groups:**"]
            for g in groups:
                status = "✅" if g["enabled"] else "❌"
                mention = "📢" if g["require_mention"] else "👂"
                from_policy = "👥 all" if g["allow_from"] == "all" else "🔒 registered"
                group_name = g["name"] or "Unnamed"
                lines.append(
                    f"- {status} **{group_name}** (`{g['platform_group_id']}`)\n"
                    f"  {mention} mention: {g['require_mention']} | {from_policy}"
                )
            return "\n".join(lines)

        if action == "add":
            if not group_id:
                return "Error: group_id is required for add action."
            
            existing = await get_group("telegram", group_id)
            if existing:
                return f"Group {group_id} already registered as '{existing.get('name', 'Unnamed')}'."
            
            # Get default require_mention from config
            default_mention = await get_config("telegram.require_mention", True)
            
            group = await create_group(
                platform="telegram",
                platform_group_id=group_id,
                name=name or None,
                enabled=True if enabled is None else enabled,
                require_mention=default_mention if require_mention is None else require_mention,
                allow_from=allow_from or "all",
            )
            return f"Group registered: {group['name'] or group_id} (mention: {group['require_mention']}, allow: {group['allow_from']})"

        if action == "update":
            if not group_id:
                return "Error: group_id is required for update action."
            
            existing = await get_group("telegram", group_id)
            if not existing:
                return f"Group {group_id} not found. Use action='add' first."
            
            updated = await update_group(
                platform="telegram",
                platform_group_id=group_id,
                name=name or None,
                enabled=enabled,
                require_mention=require_mention,
                allow_from=allow_from or None,
            )
            if updated:
                return f"Group updated: {updated['name'] or group_id} (enabled: {updated['enabled']}, mention: {updated['require_mention']}, allow: {updated['allow_from']})"
            return "No changes made."

        if action == "remove":
            if not group_id:
                return "Error: group_id is required for remove action."
            
            deleted = await delete_group("telegram", group_id)
            if deleted:
                return f"Group {group_id} removed."
            return f"Group {group_id} not found."

        return f"Unknown action: {action}"

    async def _tool_manage_user(
        self,
        action: str,
        user_id: str = "",
        display_name: str = "",
        aliases: str = "",
        access_level: str = "",
        # Context passed by conversation handler
        _caller_user: dict = None,
    ) -> str:
        """Tool handler: manage users."""
        from .db.models import get_user, update_user, list_users
        import json

        if action == "list":
            users = await list_users(platform="telegram")
            if not users:
                return "No users registered."
            
            lines = ["**Registered Users:**"]
            for u in users:
                display = u.get("display_name") or u["name"]
                level = u.get("access_level", "public")
                level_icon = {"owner": "👑", "admin": "⭐", "family": "💚", "friend": "💙"}.get(level, "👤")
                user_aliases = u.get("aliases") or {}
                alias_info = ""
                if user_aliases.get("default"):
                    alias_info = f" (alias: {user_aliases['default']})"
                lines.append(f"- {level_icon} **{display}** (`{u['platform_id']}`) — {level}{alias_info}")
            return "\n".join(lines)

        if action == "get":
            if not user_id:
                return "Error: user_id is required for get action."
            
            user = await get_user("telegram", user_id)
            if not user:
                return f"User {user_id} not found."
            
            lines = [f"**User: {user.get('display_name') or user['name']}**"]
            lines.append(f"- Platform ID: `{user['platform_id']}`")
            lines.append(f"- Access level: {user.get('access_level', 'public')}")
            
            user_aliases = user.get("aliases") or {}
            if user_aliases:
                lines.append(f"- Aliases: `{json.dumps(user_aliases)}`")
            
            prefs = user.get("preferences") or {}
            if prefs:
                lines.append(f"- Preferences: `{json.dumps(prefs)}`")
            
            return "\n".join(lines)

        if action == "update":
            if not user_id:
                return "Error: user_id is required for update action."
            
            user = await get_user("telegram", user_id)
            if not user:
                return f"User {user_id} not found."
            
            # Parse aliases JSON if provided
            parsed_aliases = None
            if aliases:
                try:
                    parsed_aliases = json.loads(aliases)
                except json.JSONDecodeError:
                    return "Error: aliases must be valid JSON"
            
            updated = await update_user(
                platform="telegram",
                platform_id=user_id,
                display_name=display_name or None,
                aliases=parsed_aliases,
                access_level=access_level or None,
            )
            
            if updated:
                return f"User updated: {updated.get('display_name') or updated['name']} (level: {updated.get('access_level')})"
            return "No changes made."

        return f"Unknown action: {action}"

    async def handle_message(
        self,
        platform: str,
        chat_id: str,
        user_name: str,
        user_platform_id: str,
        message: str,
        display_name: Optional[str] = None,
        is_group: bool = False,
        message_metadata: Optional[dict] = None,
    ) -> str:
        """Handle an incoming message from any channel.
        
        Args:
            platform: Platform identifier (telegram, etc.)
            chat_id: Chat/conversation ID
            user_name: User's name
            user_platform_id: User's platform-specific ID
            message: Message content
            display_name: Optional display name
            is_group: Whether this is a group chat (affects security restrictions)
            message_metadata: Optional metadata (e.g. image data for vision)
            
        Returns:
            Agent response string
        """
        # Get or create user
        user = await get_or_create_user(
            name=user_name,
            platform=platform,
            platform_id=user_platform_id,
            display_name=display_name,
        )

        # Route to conversation manager
        return await self.conversations.handle_message(
            platform=platform,
            chat_id=chat_id,
            user=user,
            message=message,
            is_group=is_group,
            message_metadata=message_metadata,
        )
