"""Tests for syne.boot — prompt building helpers and section getters."""

import pytest

from syne.boot import (
    _format_tools_section,
    _format_abilities_section,
    build_user_context,
    _get_soul_management_section,
    _get_communication_behavior_section,
    _get_propose_before_execute_section,
    _get_subagent_behavior_section,
    _get_self_healing_section,
    _get_memory_behavior_section,
    _get_function_calling_section,
    _get_workspace_section,
    _get_security_context_section,
)


# ---------------------------------------------------------------------------
# _format_tools_section
# ---------------------------------------------------------------------------

class TestFormatToolsSection:

    def test_empty_list(self):
        assert _format_tools_section([]) == ""

    def test_single_tool_with_parameters(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "Does something useful.",
                    "parameters": {
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        result = _format_tools_section(tools)
        assert "# Available Tools" in result
        assert "## my_tool" in result
        assert "Does something useful." in result
        assert "query (string, required)" in result
        assert "limit (integer, optional)" in result

    def test_tool_with_enum_parameter(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "action_tool",
                    "description": "Performs an action.",
                    "parameters": {
                        "properties": {
                            "mode": {
                                "type": "string",
                                "description": "Operation mode",
                                "enum": ["read", "write", "delete"],
                            },
                        },
                        "required": ["mode"],
                    },
                },
            }
        ]
        result = _format_tools_section(tools)
        assert "string: read/write/delete" in result

    def test_tool_without_function_key(self):
        tools = [{"type": "function"}]
        result = _format_tools_section(tools)
        # No "function" key means the tool is skipped entirely
        assert "## " not in result.replace("# Available Tools", "")

    def test_non_function_type_skipped(self):
        tools = [{"type": "other", "name": "something"}]
        result = _format_tools_section(tools)
        # Header is emitted but no tool entries appear
        assert "## " not in result

    def test_tool_without_parameters(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "description": "No params needed.",
                    "parameters": {},
                },
            }
        ]
        result = _format_tools_section(tools)
        assert "## simple_tool" in result
        assert "Parameters:" not in result

    def test_multiple_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "First tool.",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "description": "Second tool.",
                    "parameters": {},
                },
            },
        ]
        result = _format_tools_section(tools)
        assert "## tool_a" in result
        assert "## tool_b" in result


# ---------------------------------------------------------------------------
# _format_abilities_section
# ---------------------------------------------------------------------------

class TestFormatAbilitiesSection:

    def test_empty_list(self):
        assert _format_abilities_section([]) == ""

    def test_single_ability(self):
        abilities = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information.",
                    "parameters": {
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        result = _format_abilities_section(abilities)
        assert "# Enabled Abilities" in result
        assert "## web_search" in result
        assert "Search the web for information." in result
        assert "query (string, required)" in result

    def test_ability_without_function_key(self):
        abilities = [{"type": "function"}]
        result = _format_abilities_section(abilities)
        assert "## " not in result.replace("# Enabled Abilities", "")

    def test_non_function_type_skipped(self):
        abilities = [{"type": "other"}]
        result = _format_abilities_section(abilities)
        # Header is emitted but no ability entries appear
        assert "## " not in result


# ---------------------------------------------------------------------------
# build_user_context (async)
# ---------------------------------------------------------------------------

class TestBuildUserContext:

    @pytest.mark.asyncio
    async def test_user_with_all_fields(self):
        user = {
            "display_name": "Riyo",
            "name": "riyo_fallback",
            "access_level": "owner",
            "platform": "telegram",
        }
        result = await build_user_context(user)
        assert "# Current User" in result
        assert "Name: Riyo" in result
        assert "Access level: owner" in result
        assert "Platform: telegram" in result

    @pytest.mark.asyncio
    async def test_user_display_name_preferred_over_name(self):
        user = {"display_name": "Display", "name": "Fallback"}
        result = await build_user_context(user)
        assert "Name: Display" in result
        assert "Fallback" not in result

    @pytest.mark.asyncio
    async def test_user_falls_back_to_name(self):
        user = {"name": "Fallback"}
        result = await build_user_context(user)
        assert "Name: Fallback" in result

    @pytest.mark.asyncio
    async def test_user_with_missing_fields(self):
        user = {}
        result = await build_user_context(user)
        assert "Name: Unknown" in result
        assert "Access level: public" in result
        assert "Platform: unknown" in result

    @pytest.mark.asyncio
    async def test_user_with_preferences_dict(self):
        user = {
            "name": "TestUser",
            "preferences": {
                "language": "id",
                "timezone": "Asia/Jakarta",
            },
        }
        result = await build_user_context(user)
        assert "Preferences:" in result
        assert "language: id" in result
        assert "timezone: Asia/Jakarta" in result

    @pytest.mark.asyncio
    async def test_user_with_empty_preferences(self):
        user = {"name": "TestUser", "preferences": {}}
        result = await build_user_context(user)
        assert "Preferences:" not in result

    @pytest.mark.asyncio
    async def test_user_with_non_dict_preferences(self):
        user = {"name": "TestUser", "preferences": "not a dict"}
        result = await build_user_context(user)
        # Non-dict preferences should be ignored
        assert "Preferences:" not in result


# ---------------------------------------------------------------------------
# Section getters — verify they return non-empty strings
# ---------------------------------------------------------------------------

class TestSectionGetters:

    def test_soul_management_section(self):
        result = _get_soul_management_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "PostgreSQL" in result

    def test_communication_behavior_section(self):
        result = _get_communication_behavior_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Communication Style" in result

    def test_propose_before_execute_section(self):
        result = _get_propose_before_execute_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "file_read" in result

    def test_subagent_behavior_section(self):
        result = _get_subagent_behavior_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Sub-Agent" in result

    def test_self_healing_section(self):
        result = _get_self_healing_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Self-Healing" in result

    def test_memory_behavior_section(self):
        result = _get_memory_behavior_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Memory" in result

    def test_function_calling_section(self):
        result = _get_function_calling_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Function Calling" in result

    def test_workspace_section(self):
        result = _get_workspace_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "workspace" in result.lower()

    def test_security_context_section(self):
        result = _get_security_context_section()
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Security" in result
