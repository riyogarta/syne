"""Tests for syne.gateway.auth — token hashing, pairing verification logic."""

import pytest
from unittest.mock import AsyncMock, patch

from syne.gateway.auth import _hash_token, PAIRING_TOKEN_TTL


# ── Token hashing ────────────────────────────────────────────────────


class TestHashToken:
    def test_returns_hex_string(self):
        result = _hash_token("test_token")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest

    def test_deterministic(self):
        assert _hash_token("abc") == _hash_token("abc")

    def test_different_inputs(self):
        assert _hash_token("token_a") != _hash_token("token_b")

    def test_empty_string(self):
        result = _hash_token("")
        assert isinstance(result, str)
        assert len(result) == 64


# ── Pairing TTL ──────────────────────────────────────────────────────


class TestPairingTTL:
    def test_ttl_is_5_minutes(self):
        assert PAIRING_TOKEN_TTL.total_seconds() == 300


# ── generate_pairing_token ───────────────────────────────────────────


class TestGeneratePairingToken:
    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_returns_token_string(self, mock_get_conn):
        conn = AsyncMock()
        conn.fetchrow.return_value = None  # No existing node with name
        conn.execute.return_value = None
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import generate_pairing_token
        token = await generate_pairing_token("testnode")
        assert isinstance(token, str)
        assert len(token) > 10

    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_cleans_expired_tokens(self, mock_get_conn):
        conn = AsyncMock()
        conn.fetchrow.return_value = None
        conn.execute.return_value = None
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import generate_pairing_token
        await generate_pairing_token()

        # Should have called execute for cleanup + insert
        assert conn.execute.call_count >= 2

    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_raises_if_name_taken(self, mock_get_conn):
        conn = AsyncMock()
        conn.fetchrow.return_value = {"id": 1}  # Name already in use
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import generate_pairing_token
        with pytest.raises(ValueError, match="already in use"):
            await generate_pairing_token("taken-name")


# ── verify_pairing_token ────────────────────────────────────────────


class TestVerifyPairingToken:
    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_valid_token(self, mock_get_conn):
        conn = AsyncMock()
        conn.fetchrow.return_value = {"id": 1, "node_name": "mypc"}
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import verify_pairing_token
        valid, name = await verify_pairing_token("some-token")
        assert valid is True
        assert name == "mypc"

    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_invalid_token(self, mock_get_conn):
        conn = AsyncMock()
        conn.fetchrow.return_value = None
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import verify_pairing_token
        valid, name = await verify_pairing_token("bad-token")
        assert valid is False
        assert name == ""


# ── verify_node_token ────────────────────────────────────────────────


class TestVerifyNodeToken:
    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_valid(self, mock_get_conn):
        conn = AsyncMock()
        conn.fetchrow.return_value = {"id": 1}
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import verify_node_token
        result = await verify_node_token("node-1", "valid-token")
        assert result is True

    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_invalid(self, mock_get_conn):
        conn = AsyncMock()
        conn.fetchrow.return_value = None
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import verify_node_token
        result = await verify_node_token("node-1", "bad-token")
        assert result is False


# ── register_node ────────────────────────────────────────────────────


class TestRegisterNode:
    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_returns_token(self, mock_get_conn):
        conn = AsyncMock()
        conn.execute.return_value = None
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import register_node
        token = await register_node("node-1", "laptop")
        assert isinstance(token, str)
        assert len(token) > 10


# ── revoke_node / delete_node ────────────────────────────────────────


class TestNodeManagement:
    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_revoke_success(self, mock_get_conn):
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 1"
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import revoke_node
        result = await revoke_node("node-1")
        assert result is True

    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_revoke_not_found(self, mock_get_conn):
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 0"
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import revoke_node
        result = await revoke_node("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_delete_success(self, mock_get_conn):
        conn = AsyncMock()
        conn.execute.return_value = "DELETE 1"
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import delete_node
        result = await delete_node("node-1")
        assert result is True

    @pytest.mark.asyncio
    @patch("syne.gateway.auth.get_connection")
    async def test_delete_not_found(self, mock_get_conn):
        conn = AsyncMock()
        conn.execute.return_value = "DELETE 0"
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        from syne.gateway.auth import delete_node
        result = await delete_node("nonexistent")
        assert result is False
