"""Tests for syne.cli.cmd_init module — _detect_server_tier()."""

import pytest

from syne.cli.cmd_init import _detect_server_tier


class TestDetectServerTier:
    """Tests for _detect_server_tier()."""

    def test_low_tier_cloud(self):
        """1 CPU, 1GB RAM -> cloud tier, all models None."""
        result = _detect_server_tier(1, 1)
        tier_name, embed_model, embed_dims, embed_size, eval_model, eval_size = result
        assert tier_name == "cloud"
        assert embed_model is None
        assert embed_dims is None
        assert embed_size is None
        assert eval_model is None
        assert eval_size is None

    def test_mid_tier_moderate(self):
        """2 CPU, 4GB RAM -> moderate tier."""
        result = _detect_server_tier(2, 4)
        tier_name, embed_model, embed_dims, embed_size, eval_model, eval_size = result
        assert tier_name == "moderate"
        assert embed_model == "qwen3-embedding:0.6b"
        assert embed_dims == 1024
        assert eval_model == "qwen3:1.7b"

    def test_high_tier_strong(self):
        """4 CPU, 8GB RAM -> strong tier."""
        result = _detect_server_tier(4, 8)
        tier_name, embed_model, embed_dims, embed_size, eval_model, eval_size = result
        assert tier_name == "strong"
        assert embed_model == "qwen3-embedding:4b"
        assert embed_dims == 2560
        assert eval_model == "qwen3:1.7b"

    def test_beast_tier(self):
        """4 CPU, 16GB RAM -> beast tier."""
        result = _detect_server_tier(4, 16)
        tier_name, embed_model, embed_dims, embed_size, eval_model, eval_size = result
        assert tier_name == "beast"
        assert embed_model == "qwen3-embedding:8b"
        assert embed_dims == 4096
        assert eval_model == "qwen3:4b"

    def test_beast_tier_large_server(self):
        """8 CPU, 32GB RAM -> beast tier (exceeds threshold)."""
        result = _detect_server_tier(8, 32)
        tier_name = result[0]
        assert tier_name == "beast"

    def test_return_tuple_structure(self):
        """Return value is a 6-tuple."""
        result = _detect_server_tier(4, 16)
        assert isinstance(result, tuple)
        assert len(result) == 6
        tier_name, embed_model, embed_dims, embed_size, eval_model, eval_size = result
        assert isinstance(tier_name, str)
        assert isinstance(embed_model, str)
        assert isinstance(embed_dims, int)
        assert isinstance(embed_size, str)
        assert isinstance(eval_model, str)
        assert isinstance(eval_size, str)

    def test_minimal_tier(self):
        """2 CPU, 2GB RAM -> minimal tier."""
        result = _detect_server_tier(2, 2)
        tier_name, embed_model, embed_dims, embed_size, eval_model, eval_size = result
        assert tier_name == "minimal"
        assert embed_model == "qwen3-embedding:0.6b"
        assert embed_dims == 1024
        assert eval_model == "qwen3:0.6b"

    def test_boundary_1cpu_4gb(self):
        """1 CPU, 4GB RAM -> moderate (has enough RAM, CPU doesn't matter for moderate)."""
        result = _detect_server_tier(1, 4)
        tier_name = result[0]
        assert tier_name == "moderate"
