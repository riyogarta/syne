"""Tests for syne.cli.shared module."""

import pytest

from syne.cli.shared import _strip_env_quotes, _get_syne_dir


class TestStripEnvQuotes:
    """Tests for _strip_env_quotes()."""

    def test_double_quotes(self):
        assert _strip_env_quotes('"hello world"') == "hello world"

    def test_single_quotes(self):
        assert _strip_env_quotes("'hello world'") == "hello world"

    def test_no_quotes(self):
        assert _strip_env_quotes("hello world") == "hello world"

    def test_empty_string(self):
        assert _strip_env_quotes("") == ""

    def test_whitespace_stripped(self):
        assert _strip_env_quotes("  hello  ") == "hello"

    def test_mismatched_quotes_not_stripped(self):
        assert _strip_env_quotes("\"hello'") == "\"hello'"

    def test_single_char(self):
        assert _strip_env_quotes("x") == "x"

    def test_empty_quoted_string(self):
        assert _strip_env_quotes('""') == ""

    def test_nested_quotes_preserved(self):
        assert _strip_env_quotes('"it\'s fine"') == "it's fine"


class TestGetSyneDir:
    """Tests for _get_syne_dir()."""

    def test_returns_string(self):
        result = _get_syne_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_project_root(self):
        """The syne dir should end with the project directory name."""
        result = _get_syne_dir()
        # shared.py is at syne/cli/shared.py, so dirname x3 = project root
        # The result should be an absolute path
        assert result.startswith("/")
