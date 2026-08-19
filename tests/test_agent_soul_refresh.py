"""Tests for update_soul system-prompt hot-reload gating.

A mutating write to identity/soul/rules must invalidate the cached system
prompt; a read or a failed write must not, or every no-op would trigger a
full rebuild across all active sessions.
"""

import pytest
from unittest.mock import AsyncMock

from syne.agent import SyneAgent


def _agent(impl_result: str):
    """Bare SyneAgent -- no __init__, no DB. Only the two collaborators the
    wrapper actually touches are stubbed."""
    agent = object.__new__(SyneAgent)
    agent._update_soul_impl = AsyncMock(return_value=impl_result)
    agent.conversations = AsyncMock()
    return agent


class TestSoulWriteMissed:
    """The pure predicate -- no async, no mocks."""

    def test_error_prefix_is_a_miss(self):
        assert SyneAgent._soul_write_missed("Error: key required.")

    def test_unknown_target_action_is_a_miss(self):
        # identity has no 'add', rules have no 'set' -- these land here.
        assert SyneAgent._soul_write_missed("Unknown target/action: rules/set")

    def test_not_found_suffix_is_a_miss(self):
        assert SyneAgent._soul_write_missed("Rule not found.")
        assert SyneAgent._soul_write_missed("Entry not found.")

    def test_success_is_not_a_miss(self):
        assert not SyneAgent._soul_write_missed("Rule added: [SEC009] No")
        assert not SyneAgent._soul_write_missed("Identity updated: name = Molt")


@pytest.mark.asyncio
class TestUpdateSoulRefresh:

    async def test_mutating_write_refreshes_prompt(self):
        agent = _agent("Rule added: [SEC009] Test rule")
        out = await agent._tool_update_soul("rules", "add", "SEC009", "Test rule: body")
        agent.conversations.refresh_system_prompts.assert_awaited_once()
        assert out == "Rule added: [SEC009] Test rule"

    async def test_get_does_not_refresh(self):
        agent = _agent("**Rules:**\n- x")
        await agent._tool_update_soul("rules", "get")
        agent.conversations.refresh_system_prompts.assert_not_awaited()

    async def test_error_return_does_not_refresh(self):
        agent = _agent("Error: key (rule code) required.")
        await agent._tool_update_soul("rules", "remove")
        agent.conversations.refresh_system_prompts.assert_not_awaited()

    async def test_not_found_does_not_refresh(self):
        agent = _agent("Rule not found.")
        await agent._tool_update_soul("rules", "remove", "NOPE")
        agent.conversations.refresh_system_prompts.assert_not_awaited()

    async def test_unknown_target_action_does_not_refresh(self):
        # 'set' is a mutating action but rules do not implement it, so the impl
        # falls through to "Unknown target/action". Nothing was written.
        agent = _agent("Unknown target/action: rules/set")
        await agent._tool_update_soul("rules", "set", "X", "y")
        agent.conversations.refresh_system_prompts.assert_not_awaited()

    async def test_refresh_failure_is_surfaced_not_swallowed(self):
        agent = _agent("Rule added: [SEC009] Test rule")
        agent.conversations.refresh_system_prompts.side_effect = RuntimeError("boom")
        out = await agent._tool_update_soul("rules", "add", "SEC009", "Test rule: body")
        # The DB write already committed -- caller must still see it succeeded,
        # plus a warning that a restart is needed.
        assert "Rule added: [SEC009] Test rule" in out
        assert "boom" in out
        assert "restart" in out.lower()
