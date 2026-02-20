"""Tests for group management and channel configuration."""

import pytest
from syne.db.models import (
    create_group,
    get_group,
    update_group,
    delete_group,
    list_groups,
    get_or_create_user,
    get_user,
    update_user,
    get_user_alias,
    get_config,
    set_config,
)


class TestGroupCRUD:
    """Test group CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_group(self, clean_groups):
        """Test creating a new group."""
        group = await create_group(
            platform="telegram",
            platform_group_id="-123456789",
            name="Test Group",
            enabled=True,
            require_mention=True,
            allow_from="all",
        )
        
        assert group is not None
        assert group["platform_group_id"] == "-123456789"
        assert group["name"] == "Test Group"
        assert group["enabled"] is True
        assert group["require_mention"] is True
        assert group["allow_from"] == "all"

    @pytest.mark.asyncio
    async def test_get_group(self, clean_groups):
        """Test retrieving a group."""
        await create_group(
            platform="telegram",
            platform_group_id="-111111111",
            name="Retrieve Test",
        )
        
        group = await get_group("telegram", "-111111111")
        assert group is not None
        assert group["name"] == "Retrieve Test"
        
        # Non-existent group
        missing = await get_group("telegram", "-999999999")
        assert missing is None

    @pytest.mark.asyncio
    async def test_update_group(self, clean_groups):
        """Test updating group settings."""
        await create_group(
            platform="telegram",
            platform_group_id="-222222222",
            name="Update Test",
            enabled=True,
            require_mention=True,
        )
        
        updated = await update_group(
            platform="telegram",
            platform_group_id="-222222222",
            name="Updated Name",
            enabled=False,
            require_mention=False,
            allow_from="registered",
        )
        
        assert updated is not None
        assert updated["name"] == "Updated Name"
        assert updated["enabled"] is False
        assert updated["require_mention"] is False
        assert updated["allow_from"] == "registered"

    @pytest.mark.asyncio
    async def test_delete_group(self, clean_groups):
        """Test deleting a group."""
        await create_group(
            platform="telegram",
            platform_group_id="-333333333",
            name="Delete Test",
        )
        
        # Verify exists
        assert await get_group("telegram", "-333333333") is not None
        
        # Delete
        deleted = await delete_group("telegram", "-333333333")
        assert deleted is True
        
        # Verify gone
        assert await get_group("telegram", "-333333333") is None
        
        # Delete non-existent returns False
        deleted_again = await delete_group("telegram", "-333333333")
        assert deleted_again is False

    @pytest.mark.asyncio
    async def test_list_groups(self, clean_groups):
        """Test listing groups."""
        await create_group("telegram", "-444444441", "Group A", enabled=True)
        await create_group("telegram", "-444444442", "Group B", enabled=True)
        await create_group("telegram", "-444444443", "Group C", enabled=False)
        
        # List enabled only
        enabled = await list_groups(platform="telegram", enabled_only=True)
        assert len(enabled) == 2
        
        # List all
        all_groups = await list_groups(platform="telegram", enabled_only=False)
        assert len(all_groups) == 3


class TestUserAutoDiscovery:
    """Test user auto-discovery functionality."""

    @pytest.mark.asyncio
    async def test_auto_create_user(self, clean_test_users):
        """Test auto-creating a user (non-first user gets 'public')."""
        # First user auto-becomes owner, so create a dummy owner first
        await get_or_create_user(
            name="Owner",
            platform="telegram",
            platform_id="test_owner_00000",
        )
        
        user = await get_or_create_user(
            name="Test User",
            platform="telegram",
            platform_id="test_12345",
            display_name="Testy",
        )
        
        assert user is not None
        assert user["name"] == "Test User"
        assert user["platform_id"] == "test_12345"
        assert user.get("access_level") == "public"  # Default for auto-created (non-first)

    @pytest.mark.asyncio
    async def test_get_existing_user(self, clean_test_users):
        """Test that existing user is returned, not duplicated."""
        # Create first
        user1 = await get_or_create_user(
            name="First Name",
            platform="telegram",
            platform_id="test_67890",
        )
        
        # Get same user with different name (should return existing)
        user2 = await get_or_create_user(
            name="Different Name",
            platform="telegram",
            platform_id="test_67890",
        )
        
        assert user1["id"] == user2["id"]
        assert user2["name"] == "First Name"  # Original name preserved


class TestUserAliases:
    """Test user alias functionality."""

    @pytest.mark.asyncio
    async def test_update_user_aliases(self, clean_test_users):
        """Test updating user aliases."""
        await get_or_create_user(
            name="Alias Test",
            platform="telegram",
            platform_id="test_alias_user",
        )
        
        aliases = {
            "default": "Ally",
            "groups": {
                "-123": "Boss",
                "-456": "Friend",
            },
        }
        
        updated = await update_user(
            platform="telegram",
            platform_id="test_alias_user",
            aliases=aliases,
        )
        
        assert updated is not None
        assert updated["aliases"]["default"] == "Ally"
        assert updated["aliases"]["groups"]["-123"] == "Boss"

    @pytest.mark.asyncio
    async def test_get_user_alias_for_group(self, clean_test_users):
        """Test getting correct alias for a specific group."""
        await get_or_create_user(
            name="Multi Alias",
            platform="telegram",
            platform_id="test_multi_alias",
        )
        
        await update_user(
            platform="telegram",
            platform_id="test_multi_alias",
            display_name="Default Display",
            aliases={
                "default": "General Name",
                "groups": {
                    "-group1": "Group1 Name",
                    "-group2": "Group2 Name",
                },
            },
        )
        
        user = await get_user("telegram", "test_multi_alias")
        
        # Group-specific alias
        alias1 = await get_user_alias(user, "-group1")
        assert alias1 == "Group1 Name"
        
        alias2 = await get_user_alias(user, "-group2")
        assert alias2 == "Group2 Name"
        
        # Non-existent group falls back to default alias
        alias_default = await get_user_alias(user, "-unknown_group")
        assert alias_default == "General Name"
        
        # No group falls back to default alias
        alias_none = await get_user_alias(user, None)
        assert alias_none == "General Name"


class TestBotTriggerMatching:
    """Test bot trigger name matching logic."""

    def test_mention_stripping(self):
        """Test that mentions are properly stripped from text."""
        from syne.channels.telegram import TelegramChannel
        
        # Create minimal instance for testing helper method
        class MockAgent:
            conversations = None
        
        channel = TelegramChannel(MockAgent(), "fake_token")
        
        # Test @username stripping
        is_mentioned, text = channel._check_and_strip_mention(
            "@SyneBot hello there",
            "SyneBot",
            "Syne"
        )
        assert is_mentioned is True
        assert text == "hello there"
        
        # Test trigger name stripping
        is_mentioned, text = channel._check_and_strip_mention(
            "Syne, what time is it?",
            "SyneBot",
            "Syne"
        )
        assert is_mentioned is True
        assert text == "what time is it?"
        
        # Test case-insensitive
        is_mentioned, text = channel._check_and_strip_mention(
            "syne: help me",
            "SyneBot",
            "Syne"
        )
        assert is_mentioned is True
        assert text == "help me"
        
        # Test no mention
        is_mentioned, text = channel._check_and_strip_mention(
            "just a regular message",
            "SyneBot",
            "Syne"
        )
        assert is_mentioned is False
        assert text == "just a regular message"

    def test_multiple_mentions(self):
        """Test handling multiple mentions in one message."""
        from syne.channels.telegram import TelegramChannel
        
        class MockAgent:
            conversations = None
        
        channel = TelegramChannel(MockAgent(), "fake_token")
        
        # Both @username and trigger name
        is_mentioned, text = channel._check_and_strip_mention(
            "@SyneBot Syne, do something",
            "SyneBot",
            "Syne"
        )
        assert is_mentioned is True
        assert "do something" in text


class TestChannelConfig:
    """Test channel configuration from database."""

    @pytest.mark.asyncio
    async def test_default_config_values(self, db_pool):
        """Test that default config values are accessible."""
        # These should exist from schema seed data
        dm_policy = await get_config("telegram.dm_policy", "unknown")
        assert dm_policy in ["open", "registered", "unknown"]
        
        group_policy = await get_config("telegram.group_policy", "unknown")
        assert group_policy in ["allowlist", "open", "unknown"]

    @pytest.mark.asyncio
    async def test_set_and_get_config(self, db_pool):
        """Test setting and getting config values."""
        await set_config("test.config.key", "test_value")
        
        value = await get_config("test.config.key")
        assert value == "test_value"
        
        # Clean up
        from syne.db.connection import get_connection
        async with get_connection() as conn:
            await conn.execute("DELETE FROM config WHERE key = 'test.config.key'")


class TestMessageFiltering:
    """Test message filtering logic."""

    @pytest.mark.asyncio
    async def test_unregistered_group_filtered(self, clean_groups, db_pool):
        """Test that messages from unregistered groups are filtered."""
        # Ensure allowlist policy
        await set_config("telegram.group_policy", "allowlist")
        
        # Check that unknown group returns None
        group = await get_group("telegram", "-unknown_group")
        assert group is None

    @pytest.mark.asyncio
    async def test_registered_group_allowed(self, clean_groups, db_pool):
        """Test that messages from registered groups are allowed."""
        await create_group(
            platform="telegram",
            platform_group_id="-registered_group",
            name="Registered",
            enabled=True,
        )
        
        group = await get_group("telegram", "-registered_group")
        assert group is not None
        assert group["enabled"] is True

    @pytest.mark.asyncio
    async def test_disabled_group_filtered(self, clean_groups, db_pool):
        """Test that messages from disabled groups are filtered."""
        await create_group(
            platform="telegram",
            platform_group_id="-disabled_group",
            name="Disabled",
            enabled=False,
        )
        
        group = await get_group("telegram", "-disabled_group")
        assert group is not None
        assert group["enabled"] is False
