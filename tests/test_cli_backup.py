"""Tests for syne.cli.cmd_backup module — _format_size()."""

import pytest

from syne.cli.cmd_backup import _format_size


class TestFormatSize:
    """Tests for _format_size()."""

    def test_bytes(self):
        assert _format_size(500) == "500 bytes"

    def test_zero_bytes(self):
        assert _format_size(0) == "0 bytes"

    def test_one_byte(self):
        assert _format_size(1) == "1 bytes"

    def test_kilobytes(self):
        result = _format_size(2048)
        assert "KB" in result
        assert "2.0" in result

    def test_kilobytes_boundary(self):
        """Just over 1024 bytes should show KB."""
        result = _format_size(1025)
        assert "KB" in result

    def test_megabytes(self):
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result
        assert "5.0" in result

    def test_megabytes_boundary(self):
        """Just over 1MB should show MB."""
        result = _format_size(1024 * 1024 + 1)
        assert "MB" in result

    def test_large_megabytes(self):
        result = _format_size(150 * 1024 * 1024)
        assert "MB" in result
        assert "150.0" in result

    def test_exact_1024_shows_kb(self):
        """Exactly 1024 bytes is not > 1024, so shows bytes."""
        result = _format_size(1024)
        assert "bytes" in result
