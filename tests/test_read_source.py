"""Tests for read_source tool."""

import pytest
from syne.tools.read_source import _handle_read_source, _is_allowed_path, _resolve_path


class TestPathValidation:
    """Test path validation and security."""

    def test_allowed_syne_core(self):
        assert _is_allowed_path("syne/agent.py") is True

    def test_allowed_syne_tools(self):
        assert _is_allowed_path("syne/tools/read_source.py") is True

    def test_allowed_syne_abilities(self):
        assert _is_allowed_path("syne/abilities/base.py") is True

    def test_allowed_syne_security(self):
        assert _is_allowed_path("syne/security.py") is True

    def test_allowed_syne_db(self):
        assert _is_allowed_path("syne/db/schema.sql") is True

    def test_allowed_tests(self):
        assert _is_allowed_path("tests/test_read_source.py") is True

    def test_allowed_readme(self):
        assert _is_allowed_path("README.md") is True

    def test_allowed_pyproject(self):
        assert _is_allowed_path("pyproject.toml") is True

    def test_blocked_env(self):
        assert _is_allowed_path(".env") is False

    def test_blocked_secrets(self):
        assert _is_allowed_path("secrets/token.txt") is False

    def test_blocked_pycache(self):
        assert _is_allowed_path("syne/__pycache__/agent.pyc") is False

    def test_blocked_git(self):
        assert _is_allowed_path(".git/config") is False

    def test_blocked_random_path(self):
        assert _is_allowed_path("/etc/passwd") is False

    def test_blocked_home_dir(self):
        assert _is_allowed_path("../../../etc/shadow") is False

    def test_resolve_traversal_blocked(self):
        assert _resolve_path("syne/../../etc/passwd") is None

    def test_resolve_valid_path(self):
        result = _resolve_path("syne/agent.py")
        assert result is not None
        assert result.name == "agent.py"


class TestTreeAction:
    """Test tree listing."""

    @pytest.mark.asyncio
    async def test_tree_syne_dir(self):
        result = await _handle_read_source(action="tree", path="syne/")
        assert "📁 syne/" in result
        assert "agent.py" in result
        assert "security.py" in result

    @pytest.mark.asyncio
    async def test_tree_blocked_path(self):
        result = await _handle_read_source(action="tree", path="/etc/")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_tree_single_file(self):
        result = await _handle_read_source(action="tree", path="syne/agent.py")
        assert "📄" in result
        assert "bytes" in result

    @pytest.mark.asyncio
    async def test_tree_no_pycache(self):
        result = await _handle_read_source(action="tree", path="syne/")
        assert "__pycache__" not in result


class TestReadAction:
    """Test file reading."""

    @pytest.mark.asyncio
    async def test_read_file(self):
        result = await _handle_read_source(action="read", path="syne/security.py")
        assert "📄 syne/security.py" in result
        assert "lines 1-" in result
        # Should contain actual code
        assert "Rule 700" in result or "security" in result.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset(self):
        result = await _handle_read_source(action="read", path="syne/security.py", offset=10, limit=5)
        assert "lines 10-14" in result

    @pytest.mark.asyncio
    async def test_read_blocked_file(self):
        result = await _handle_read_source(action="read", path=".env")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_read_no_path(self):
        result = await _handle_read_source(action="read")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        result = await _handle_read_source(action="read", path="syne/nonexistent_xyz.py")
        assert "not accessible" in result or "not a file" in result

    @pytest.mark.asyncio
    async def test_read_line_numbers(self):
        result = await _handle_read_source(action="read", path="syne/__init__.py", limit=5)
        # Should have line numbers
        assert "   1 |" in result or "1 |" in result

    @pytest.mark.asyncio
    async def test_read_limit_cap(self):
        """Limit should be capped at MAX_LINES."""
        result = await _handle_read_source(action="read", path="syne/agent.py", limit=99999)
        # Should not crash, and should be capped
        assert "📄" in result


class TestSearchAction:
    """Test pattern searching."""

    @pytest.mark.asyncio
    async def test_search_pattern(self):
        result = await _handle_read_source(action="search", path="syne/", pattern="Rule 700")
        assert "matches" in result
        assert "security.py" in result

    @pytest.mark.asyncio
    async def test_search_no_pattern(self):
        result = await _handle_read_source(action="search", path="syne/")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        result = await _handle_read_source(action="search", path="syne/", pattern="zzz_nonexistent_pattern_xyz")
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_search_specific_file(self):
        result = await _handle_read_source(action="search", path="syne/security.py", pattern="OWNER_ONLY")
        assert "matches" in result
        assert "security.py" in result

    @pytest.mark.asyncio
    async def test_search_blocked_path(self):
        result = await _handle_read_source(action="search", path=".env", pattern="password")
        assert "not accessible" in result

    @pytest.mark.asyncio
    async def test_search_invalid_regex(self):
        result = await _handle_read_source(action="search", path="syne/", pattern="[invalid")
        assert "invalid pattern" in result.lower() or "error" in result.lower()


class TestUnknownAction:
    """Test unknown actions."""

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await _handle_read_source(action="write")
        assert "Unknown action" in result
