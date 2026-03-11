"""Tests for syne.node.executor — local tool execution on remote nodes."""

import asyncio
import os
import tempfile
import pytest

from syne.node.executor import (
    execute_tool,
    _exec,
    _file_read,
    _file_write,
    _read_source,
    _walk_tree,
)

# Suppress subprocess cleanup warnings (BaseSubprocessTransport.__del__ on closed loop)
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


# ── execute_tool dispatcher ──────────────────────────────────────────


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        result, success = await execute_tool("r1", "unknown_tool", {})
        assert success is False
        assert "unknown" in result.lower()

    @pytest.mark.asyncio
    async def test_dispatches_exec(self):
        result, success = await execute_tool("r1", "exec", {"command": "echo hello"})
        assert success is True
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_dispatches_file_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            result, success = await execute_tool("r1", "file_read", {"path": path})
            assert success is True
            assert "test content" in result
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_dispatches_file_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.txt")
            result, success = await execute_tool("r1", "file_write", {
                "path": path,
                "content": "written",
            })
            assert success is True
            assert os.path.exists(path)
            with open(path) as f:
                assert f.read() == "written"


# ── exec ─────────────────────────────────────────────────────────────


class TestExec:
    @pytest.mark.asyncio
    async def test_simple_command(self):
        result, success = await _exec({"command": "echo hello"})
        assert success is True
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_empty_command(self):
        result, success = await _exec({"command": ""})
        assert success is False
        assert "no command" in result.lower()

    @pytest.mark.asyncio
    async def test_no_command_key(self):
        result, success = await _exec({})
        assert success is False

    @pytest.mark.asyncio
    async def test_failed_command_includes_exit_code(self):
        result, success = await _exec({"command": "false"})
        assert success is False
        assert "exit code" in result.lower()

    @pytest.mark.asyncio
    async def test_stderr_captured(self):
        result, success = await _exec({"command": "echo err >&2"})
        assert "STDERR" in result

    @pytest.mark.asyncio
    async def test_timeout(self):
        result, success = await _exec({"command": "sleep 10", "timeout": "1"})
        assert success is False
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_timeout_capped_at_300(self):
        # timeout=999 should be capped to 300
        result, success = await _exec({"command": "echo ok", "timeout": "999"})
        assert success is True


# ── file_read ────────────────────────────────────────────────────────


class TestFileRead:
    @pytest.mark.asyncio
    async def test_read_existing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            result, success = await _file_read({"path": path})
            assert success is True
            assert result == "hello world"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        result, success = await _file_read({"path": "/tmp/nonexistent_file_xyz.txt"})
        assert success is False
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_no_path(self):
        result, success = await _file_read({"path": ""})
        assert success is False

    @pytest.mark.asyncio
    async def test_tilde_expansion(self):
        # Just check it doesn't crash with tilde
        result, success = await _file_read({"path": "~/nonexistent_xyz"})
        assert success is False  # File doesn't exist


# ── file_write ───────────────────────────────────────────────────────


class TestFileWrite:
    @pytest.mark.asyncio
    async def test_write_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "new.txt")
            result, success = await _file_write({"path": path, "content": "data"})
            assert success is True
            with open(path) as f:
                assert f.read() == "data"

    @pytest.mark.asyncio
    async def test_write_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "file.txt")
            result, success = await _file_write({"path": path, "content": "deep"})
            assert success is True
            assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_write_no_path(self):
        result, success = await _file_write({"path": "", "content": "data"})
        assert success is False

    @pytest.mark.asyncio
    async def test_write_reports_char_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "count.txt")
            result, success = await _file_write({"path": path, "content": "hello"})
            assert "5 chars" in result


# ── read_source ──────────────────────────────────────────────────────


class TestReadSource:
    @pytest.mark.asyncio
    async def test_tree_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            open(os.path.join(tmpdir, "a.py"), "w").close()
            os.mkdir(os.path.join(tmpdir, "sub"))
            open(os.path.join(tmpdir, "sub", "b.py"), "w").close()

            result, success = await _read_source({"action": "tree", "path": tmpdir})
            assert success is True
            assert "a.py" in result
            assert "sub" in result

    @pytest.mark.asyncio
    async def test_read_action(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            path = f.name
        try:
            result, success = await _read_source({"action": "read", "path": path})
            assert success is True
            assert "print" in result
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_search_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("def unique_function_name(): pass\n")

            result, success = await _read_source({
                "action": "search",
                "path": tmpdir,
                "pattern": "unique_function_name",
            })
            assert success is True
            assert "test.py" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result, success = await _read_source({"action": "delete", "path": "."})
        assert success is False
        assert "unknown" in result.lower()


# ── _walk_tree ───────────────────────────────────────────────────────


class TestWalkTree:
    def test_skips_noise_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.mkdir(os.path.join(tmpdir, ".git"))
            os.mkdir(os.path.join(tmpdir, "__pycache__"))
            os.mkdir(os.path.join(tmpdir, "src"))
            open(os.path.join(tmpdir, "main.py"), "w").close()

            lines = []
            _walk_tree(tmpdir, lines, prefix="", depth=0, max_depth=3)
            text = "\n".join(lines)
            assert ".git" not in text
            assert "__pycache__" not in text
            assert "src" in text
            assert "main.py" in text

    def test_respects_max_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = os.path.join(tmpdir, "a", "b", "c", "d")
            os.makedirs(deep)
            open(os.path.join(deep, "deep.txt"), "w").close()

            lines = []
            _walk_tree(tmpdir, lines, prefix="", depth=0, max_depth=1)
            text = "\n".join(lines)
            # Should show 'a' dir but not go deeper
            assert "a" in text
            assert "deep.txt" not in text
