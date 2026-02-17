"""Tests for tool registry."""

import pytest
from syne.tools.registry import ToolRegistry


async def mock_handler(text: str = "hello") -> str:
    return f"Result: {text}"


async def mock_admin_handler() -> str:
    return "Admin only result"


class TestToolRegistry:

    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=mock_handler,
        )
        tool = reg.get("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"

    def test_get_nonexistent(self):
        reg = ToolRegistry()
        assert reg.get("nope") is None

    def test_list_tools_public(self):
        reg = ToolRegistry()
        reg.register("pub", "Public tool", {}, mock_handler, requires_access_level="public")
        reg.register("admin", "Admin tool", {}, mock_admin_handler, requires_access_level="admin")

        public_tools = reg.list_tools("public")
        assert len(public_tools) == 1
        assert public_tools[0].name == "pub"

    def test_list_tools_owner(self):
        reg = ToolRegistry()
        reg.register("pub", "Public tool", {}, mock_handler, requires_access_level="public")
        reg.register("admin", "Admin tool", {}, mock_admin_handler, requires_access_level="admin")

        owner_tools = reg.list_tools("owner")
        assert len(owner_tools) == 2

    def test_openai_schema(self):
        reg = ToolRegistry()
        reg.register(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=mock_handler,
        )
        schema = reg.to_openai_schema()
        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        reg = ToolRegistry()
        reg.register("test", "Test", {}, mock_handler)
        result = await reg.execute("test", {"text": "world"}, "public")
        assert result == "Result: world"

    @pytest.mark.asyncio
    async def test_execute_insufficient_access(self):
        reg = ToolRegistry()
        reg.register("admin_tool", "Admin", {}, mock_admin_handler, requires_access_level="admin")
        result = await reg.execute("admin_tool", {}, "public")
        assert "Insufficient permissions" in result

    @pytest.mark.asyncio
    async def test_execute_nonexistent(self):
        reg = ToolRegistry()
        result = await reg.execute("nope", {}, "public")
        assert "not found" in result

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register("temp", "Temp", {}, mock_handler)
        assert reg.get("temp") is not None
        reg.unregister("temp")
        assert reg.get("temp") is None
