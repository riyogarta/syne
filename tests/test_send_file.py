"""Tests for send_file core tool."""

import os
import pytest
import tempfile

from syne.tools.send_file import send_file_handler, _format_size


@pytest.mark.asyncio
async def test_send_file_basic():
    """Test sending an existing file returns MEDIA: protocol."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("hello world")
        path = f.name
    
    try:
        result = await send_file_handler(path=path)
        assert "MEDIA: " in result
        assert path in result
        assert ".txt" in result
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_send_file_with_caption():
    """Test sending with a custom caption."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        f.write("<html>test</html>")
        path = f.name
    
    try:
        result = await send_file_handler(path=path, caption="Landing page")
        assert "MEDIA: " in result
        assert "Landing page" in result
        assert path in result
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_send_file_not_found():
    """Test error on non-existent file."""
    result = await send_file_handler(path="/tmp/nonexistent_file_xyz.txt")
    assert "Error" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_send_file_empty():
    """Test error on empty file."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    
    try:
        result = await send_file_handler(path=path)
        assert "Error" in result
        assert "empty" in result
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_send_file_no_path():
    """Test error when path is empty."""
    result = await send_file_handler(path="")
    assert "Error" in result
    assert "required" in result


@pytest.mark.asyncio
async def test_send_file_directory():
    """Test error when path is a directory."""
    result = await send_file_handler(path="/tmp")
    assert "Error" in result
    assert "Not a file" in result


@pytest.mark.asyncio
async def test_send_file_image():
    """Test sending an image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, mode="wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        path = f.name
    
    try:
        result = await send_file_handler(path=path)
        assert "MEDIA: " in result
        assert ".png" in result
    finally:
        os.unlink(path)


def test_format_size_bytes():
    assert _format_size(500) == "500 B"


def test_format_size_kb():
    assert _format_size(2048) == "2.0 KB"


def test_format_size_mb():
    assert _format_size(5 * 1024 * 1024) == "5.0 MB"
