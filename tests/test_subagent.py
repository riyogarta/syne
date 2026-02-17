"""Tests for the sub-agent system."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from syne.subagent import SubAgentManager
from syne.llm.provider import ChatMessage, ChatResponse


class TestSubAgentManager:
    """Test SubAgentManager core functionality."""

    def _make_manager(self, enabled=True, max_concurrent=2, timeout=5):
        """Create a SubAgentManager with mocked provider and config."""
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=ChatResponse(
            content="Task completed successfully.",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
        ))

        manager = SubAgentManager(
            provider=provider,
            system_prompt="You are a test agent.",
        )
        return manager, provider

    @pytest.mark.asyncio
    async def test_init(self):
        """Test manager initialization."""
        manager, _ = self._make_manager()
        assert manager.active_count == 0
        assert manager._on_complete is None

    @pytest.mark.asyncio
    async def test_set_completion_callback(self):
        """Test setting completion callback."""
        manager, _ = self._make_manager()
        callback = AsyncMock()
        manager.set_completion_callback(callback)
        assert manager._on_complete == callback

    @pytest.mark.asyncio
    async def test_active_count_cleanup(self):
        """Test that active_count cleans up finished tasks."""
        manager, _ = self._make_manager()

        # Add a done task
        done_task = asyncio.Future()
        done_task.set_result(None)
        manager._active_runs["fake-id"] = done_task

        # active_count should clean it up
        assert manager.active_count == 0
        assert "fake-id" not in manager._active_runs

    @pytest.mark.asyncio
    async def test_active_count_running(self):
        """Test active_count with running tasks."""
        manager, _ = self._make_manager()

        # Add a running task (never-completing future)
        running_task = asyncio.ensure_future(asyncio.sleep(999))
        manager._active_runs["running-id"] = running_task

        assert manager.active_count == 1

        # Cleanup
        running_task.cancel()
        try:
            await running_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @patch("syne.subagent.get_config")
    async def test_spawn_disabled(self, mock_config):
        """Test spawn when sub-agents are disabled."""
        mock_config.return_value = False
        manager, _ = self._make_manager()

        result = await manager.spawn(
            task="test task",
            parent_session_id=1,
        )

        assert result["success"] is False
        assert "disabled" in result["error"]

    @pytest.mark.asyncio
    @patch("syne.subagent.get_config")
    async def test_spawn_max_concurrent(self, mock_config):
        """Test spawn when max concurrent reached."""
        # First call: enabled=True, second: max_concurrent=1
        mock_config.side_effect = [True, 1]
        manager, _ = self._make_manager()

        # Fake one running
        running_task = asyncio.ensure_future(asyncio.sleep(999))
        manager._active_runs["existing"] = running_task

        result = await manager.spawn(
            task="test task",
            parent_session_id=1,
        )

        assert result["success"] is False
        assert "Max concurrent" in result["error"]

        # Cleanup
        running_task.cancel()
        try:
            await running_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self):
        """Test cancelling a non-existent run."""
        manager, _ = self._make_manager()
        result = await manager.cancel("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_empty(self):
        """Test cancel_all with no active runs."""
        manager, _ = self._make_manager()
        await manager.cancel_all()  # Should not raise
        assert manager.active_count == 0


class TestSubAgentGuardRails:
    """Test sub-agent safety constraints."""

    @pytest.mark.asyncio
    @patch("syne.subagent.get_config")
    async def test_default_enabled(self, mock_config):
        """Test that sub-agents are enabled by default."""
        mock_config.return_value = True
        provider = AsyncMock()
        manager = SubAgentManager(provider=provider, system_prompt="test")
        assert await manager.is_enabled() is True

    @pytest.mark.asyncio
    @patch("syne.subagent.get_config")
    async def test_default_max_concurrent(self, mock_config):
        """Test default max concurrent is 2."""
        mock_config.return_value = 2
        provider = AsyncMock()
        manager = SubAgentManager(provider=provider, system_prompt="test")
        assert await manager.max_concurrent() == 2

    @pytest.mark.asyncio
    @patch("syne.subagent.get_config")
    async def test_default_timeout(self, mock_config):
        """Test default timeout is 300s."""
        mock_config.return_value = 300
        provider = AsyncMock()
        manager = SubAgentManager(provider=provider, system_prompt="test")
        assert await manager.timeout_seconds() == 300
