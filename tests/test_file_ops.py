"""Tests for file operations tools (file_read, file_write)."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path

from syne.tools.file_ops import (
    file_read_handler,
    file_write_handler,
    _check_write_allowed,
    _PROJECT_ROOT,
)


class TestFileRead:
    """Tests for file_read tool."""

    @pytest.mark.asyncio
    async def test_read_basic(self, tmp_path, db_pool):
        """Test basic file read."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        
        result = await file_read_handler(path=str(test_file))
        
        assert "Line 1" in result
        assert "Line 5" in result
        assert "lines 1-5 of 5" in result

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tmp_path, db_pool):
        """Test file read with offset."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        
        result = await file_read_handler(path=str(test_file), offset=3)
        
        assert "Line 1" not in result
        assert "Line 2" not in result
        assert "Line 3" in result
        assert "Line 4" in result
        assert "Line 5" in result
        assert "lines 3-5 of 5" in result

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tmp_path, db_pool):
        """Test file read with limit."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        
        result = await file_read_handler(path=str(test_file), limit=2)
        
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" not in result
        assert "lines 1-2 of 5" in result

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, tmp_path, db_pool):
        """Test file read with offset and limit."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
        
        result = await file_read_handler(path=str(test_file), offset=2, limit=2)
        
        assert "Line 1" not in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "Line 4" not in result
        assert "lines 2-3 of 5" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, db_pool):
        """Test reading a file that doesn't exist."""
        result = await file_read_handler(path="/nonexistent/path/to/file.txt")
        
        assert "Error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_read_empty_path(self):
        """Test reading with empty path."""
        result = await file_read_handler(path="")
        
        assert "Error" in result
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_read_empty_file(self, tmp_path, db_pool):
        """Test reading an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        
        result = await file_read_handler(path=str(test_file))
        
        assert "empty file" in result.lower()


class TestFileWrite:
    """Tests for file_write tool."""

    @pytest.mark.asyncio
    async def test_write_basic(self, tmp_path):
        """Test basic file write."""
        test_file = tmp_path / "new_file.txt"
        content = "Hello, World!"
        
        result = await file_write_handler(
            path=str(test_file),
            content=content,
            workdir=str(tmp_path),
        )
        
        assert "File written" in result
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path):
        """Test that file_write creates parent directories."""
        nested_file = tmp_path / "a" / "b" / "c" / "file.txt"
        content = "Nested content"
        
        result = await file_write_handler(
            path=str(nested_file),
            content=content,
            workdir=str(tmp_path),
        )
        
        assert "File written" in result
        assert nested_file.exists()
        assert nested_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_overwrite(self, tmp_path):
        """Test that file_write overwrites existing file."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("Old content")
        
        new_content = "New content"
        result = await file_write_handler(
            path=str(test_file),
            content=new_content,
            workdir=str(tmp_path),
        )
        
        assert "File written" in result
        assert test_file.read_text() == new_content

    @pytest.mark.asyncio
    async def test_write_blocks_syne_core(self):
        """Test that file_write blocks writing to syne/ core directories."""
        # Try to write to syne/tools (core directory)
        core_path = _PROJECT_ROOT / "syne" / "tools" / "malicious.py"
        
        result = await file_write_handler(
            path=str(core_path),
            content="# malicious code",
            workdir=str(_PROJECT_ROOT),
        )
        
        assert "Error" in result
        assert not core_path.exists()

    @pytest.mark.asyncio
    async def test_write_blocks_syne_engine(self):
        """Test that file_write blocks writing to syne/engine."""
        # Try to write to syne/engine (core directory)
        engine_path = _PROJECT_ROOT / "syne" / "engine" / "test.py"
        
        result = await file_write_handler(
            path=str(engine_path),
            content="# malicious code",
            workdir=str(_PROJECT_ROOT),
        )
        
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_write_allows_syne_abilities(self, tmp_path):
        """Test that file_write allows writing to syne/abilities/."""
        # Temporarily create abilities directory if it doesn't exist
        abilities_dir = _PROJECT_ROOT / "syne" / "abilities"
        test_file = abilities_dir / "test_write_ability.py"
        
        try:
            result = await file_write_handler(
                path=str(test_file),
                content="# test ability\nclass TestAbility:\n    pass\n",
                workdir=str(_PROJECT_ROOT),
            )
            
            assert "File written" in result
            assert test_file.exists()
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()

    @pytest.mark.asyncio
    async def test_write_empty_path(self, tmp_path):
        """Test writing with empty path."""
        result = await file_write_handler(
            path="",
            content="content",
            workdir=str(tmp_path),
        )
        
        assert "Error" in result
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_write_outside_cwd(self, tmp_path):
        """Test that file_write blocks writing outside CWD."""
        # Try to write outside the working directory
        other_tmp = tempfile.mkdtemp()
        try:
            result = await file_write_handler(
                path=os.path.join(other_tmp, "test.txt"),
                content="content",
                workdir=str(tmp_path),  # Different workdir
            )
            
            assert "Error" in result
        finally:
            shutil.rmtree(other_tmp)


class TestCheckWriteAllowed:
    """Tests for _check_write_allowed helper function."""

    def test_inside_cwd_allowed(self, tmp_path):
        """Test that paths inside CWD are allowed."""
        test_path = tmp_path / "test.txt"
        allowed, reason = _check_write_allowed(test_path, tmp_path)
        assert allowed
        assert reason == ""

    def test_nested_inside_cwd_allowed(self, tmp_path):
        """Test that nested paths inside CWD are allowed."""
        test_path = tmp_path / "a" / "b" / "test.txt"
        allowed, reason = _check_write_allowed(test_path, tmp_path)
        assert allowed
        assert reason == ""

    def test_syne_tools_blocked(self):
        """Test that syne/tools is blocked."""
        cwd = _PROJECT_ROOT
        test_path = _PROJECT_ROOT / "syne" / "tools" / "test.py"
        allowed, reason = _check_write_allowed(test_path, cwd)
        assert not allowed
        assert "core" in reason.lower() or "protected" in reason.lower()

    def test_syne_abilities_allowed(self):
        """Test that syne/abilities is allowed."""
        cwd = _PROJECT_ROOT
        test_path = _PROJECT_ROOT / "syne" / "abilities" / "test.py"
        allowed, reason = _check_write_allowed(test_path, cwd)
        assert allowed
        assert reason == ""

    def test_outside_cwd_and_abilities_blocked(self, tmp_path):
        """Test that paths outside CWD and abilities are blocked."""
        cwd = tmp_path
        test_path = Path("/etc/passwd")
        allowed, reason = _check_write_allowed(test_path, cwd)
        assert not allowed
