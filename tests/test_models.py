"""Tests for syne.db.models — get_user_alias."""

import pytest

from syne.db.models import get_user_alias


class TestGetUserAlias:
    """Tests for get_user_alias (async function)."""

    @pytest.mark.asyncio
    async def test_display_name_preferred(self):
        """display_name takes precedence over name when no aliases exist."""
        user = {"name": "john", "display_name": "John Doe", "aliases": {}}
        result = await get_user_alias(user)
        assert result == "John Doe"

    @pytest.mark.asyncio
    async def test_name_fallback(self):
        """Falls back to name when display_name is absent."""
        user = {"name": "john", "aliases": {}}
        result = await get_user_alias(user)
        assert result == "john"

    @pytest.mark.asyncio
    async def test_name_fallback_display_name_none(self):
        """Falls back to name when display_name is None."""
        user = {"name": "john", "display_name": None, "aliases": {}}
        result = await get_user_alias(user)
        assert result == "john"

    @pytest.mark.asyncio
    async def test_default_alias(self):
        """Default alias overrides display_name and name."""
        user = {
            "name": "john",
            "display_name": "John Doe",
            "aliases": {"default": "Johnny"},
        }
        result = await get_user_alias(user)
        assert result == "Johnny"

    @pytest.mark.asyncio
    async def test_group_alias(self):
        """Group-specific alias overrides default alias."""
        user = {
            "name": "john",
            "display_name": "John Doe",
            "aliases": {
                "default": "Johnny",
                "groups": {"group_123": "JD"},
            },
        }
        result = await get_user_alias(user, group_id="group_123")
        assert result == "JD"

    @pytest.mark.asyncio
    async def test_group_alias_falls_back_to_default(self):
        """When group_id is given but no group alias exists, fall back to default alias."""
        user = {
            "name": "john",
            "display_name": "John Doe",
            "aliases": {
                "default": "Johnny",
                "groups": {},
            },
        }
        result = await get_user_alias(user, group_id="group_999")
        assert result == "Johnny"

    @pytest.mark.asyncio
    async def test_group_alias_no_groups_key(self):
        """When group_id is given but aliases has no groups key, fall back to default."""
        user = {
            "name": "john",
            "aliases": {"default": "Johnny"},
        }
        result = await get_user_alias(user, group_id="group_123")
        assert result == "Johnny"

    @pytest.mark.asyncio
    async def test_empty_user_dict(self):
        """Empty dict should return the fallback 'User'."""
        result = await get_user_alias({})
        assert result == "User"

    @pytest.mark.asyncio
    async def test_aliases_none(self):
        """aliases=None should not crash, falls back to display_name/name."""
        user = {"name": "john", "display_name": "John", "aliases": None}
        result = await get_user_alias(user)
        assert result == "John"

    @pytest.mark.asyncio
    async def test_no_aliases_key(self):
        """Missing aliases key should not crash."""
        user = {"name": "john", "display_name": "John"}
        result = await get_user_alias(user)
        assert result == "John"

    @pytest.mark.asyncio
    async def test_empty_default_alias(self):
        """Empty string default alias should fall through to display_name."""
        user = {
            "name": "john",
            "display_name": "John Doe",
            "aliases": {"default": ""},
        }
        result = await get_user_alias(user)
        assert result == "John Doe"

    @pytest.mark.asyncio
    async def test_group_id_none_ignores_groups(self):
        """When group_id is None, group aliases are not checked."""
        user = {
            "name": "john",
            "aliases": {
                "default": "Johnny",
                "groups": {"group_123": "JD"},
            },
        }
        result = await get_user_alias(user, group_id=None)
        assert result == "Johnny"
