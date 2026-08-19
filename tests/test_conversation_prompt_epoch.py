"""Mid-turn system-prompt hot reload (epoch versioning).

A turn snapshots its system message once in build_context(), then may run for
dozens of tool rounds. If update_soul fires refresh_system_prompts() during
that turn, the remainder of the turn used to keep running under the OLD rules.
These tests pin the epoch mechanism that closes that window.
"""

import pytest

from syne.conversation import Conversation
from syne.llm.provider import ChatMessage


def _conv(is_group=False, prompt="RULES v1"):
    """Bare Conversation — no DB, no provider, no __init__ side effects."""
    c = object.__new__(Conversation)
    c.session_id = 42
    c.system_prompt = prompt
    c.is_group = is_group
    c.inbound = None
    c.user = {"access_level": "owner"}
    c.provider = type("P", (), {"name": "anthropic"})()
    c._sys_epoch = 0
    c._ctx_sys_epoch = -1
    return c


def _ctx(content="RULES v1"):
    return [
        ChatMessage(role="system", content=content),
        ChatMessage(role="user", content="hi"),
    ]


# ── epoch bookkeeping ──────────────────────────────────────────────

def test_fresh_context_is_not_stale():
    c = _conv()
    ctx = _ctx()
    c._ctx_sys_epoch = c._sys_epoch
    assert c._sync_system_prompt(ctx, "owner") is False
    assert ctx[0].content == "RULES v1"


def test_epoch_bump_makes_context_stale_and_swaps():
    c = _conv()
    ctx = _ctx()
    c._ctx_sys_epoch = c._sys_epoch

    # update_soul lands: refresher rewrites prompt + bumps epoch
    c.system_prompt = "RULES v2"
    c._sys_epoch += 1

    assert c._sync_system_prompt(ctx, "owner") is True
    assert ctx[0].content == "RULES v2"
    assert ctx[0].role == "system"
    assert c._ctx_sys_epoch == c._sys_epoch


def test_sync_is_idempotent_within_same_epoch():
    """Called every tool round — must be a no-op when nothing changed."""
    c = _conv()
    ctx = _ctx()
    c._ctx_sys_epoch = c._sys_epoch
    c.system_prompt = "RULES v2"
    c._sys_epoch += 1

    assert c._sync_system_prompt(ctx, "owner") is True
    for _ in range(5):
        assert c._sync_system_prompt(ctx, "owner") is False
    assert ctx[0].content == "RULES v2"


def test_multiple_bumps_land_latest_only():
    c = _conv()
    ctx = _ctx()
    c._ctx_sys_epoch = c._sys_epoch
    for v in ("v2", "v3", "v4"):
        c.system_prompt = f"RULES {v}"
        c._sys_epoch += 1

    assert c._sync_system_prompt(ctx, "owner") is True
    assert ctx[0].content == "RULES v4"


def test_user_message_is_never_touched():
    c = _conv()
    ctx = _ctx()
    c._ctx_sys_epoch = c._sys_epoch
    c.system_prompt = "RULES v2"
    c._sys_epoch += 1
    c._sync_system_prompt(ctx, "owner")

    assert ctx[1].role == "user"
    assert ctx[1].content == "hi"
    assert len(ctx) == 2


# ── safety guards ──────────────────────────────────────────────────

def test_refuses_when_slot0_is_not_system():
    """Never rewrite a non-system slot, even when stale."""
    c = _conv()
    ctx = [ChatMessage(role="user", content="hi")]
    c._ctx_sys_epoch = c._sys_epoch
    c.system_prompt = "RULES v2"
    c._sys_epoch += 1

    assert c._sync_system_prompt(ctx, "owner") is False
    assert ctx[0].role == "user"
    assert ctx[0].content == "hi"


def test_refuses_on_empty_context():
    c = _conv()
    c._ctx_sys_epoch = c._sys_epoch
    c.system_prompt = "RULES v2"
    c._sys_epoch += 1
    assert c._sync_system_prompt([], "owner") is False


# ── decoration parity: the two paths must not drift ────────────────

def test_group_restrictions_survive_hot_reload():
    """Regression: a naive fix would drop group restrictions on reload."""
    c = _conv(is_group=True)
    base = c._decorate_system_prompt("owner")
    assert len(base) > len("RULES v1"), "group restrictions should be appended"

    ctx = _ctx(base)
    c._ctx_sys_epoch = c._sys_epoch
    c.system_prompt = "RULES v2"
    c._sys_epoch += 1
    c._sync_system_prompt(ctx, "owner")

    assert ctx[0].content.startswith("RULES v2")
    # the decoration tail must still be there
    assert ctx[0].content == c._decorate_system_prompt("owner")
    assert len(ctx[0].content) > len("RULES v2")


def test_non_anthropic_guardrails_survive_hot_reload():
    from syne.conversation import _NON_ANTHROPIC_GUARDRAILS

    c = _conv()
    c.provider = type("P", (), {"name": "google"})()
    ctx = _ctx(c._decorate_system_prompt("owner"))
    c._ctx_sys_epoch = c._sys_epoch
    c.system_prompt = "RULES v2"
    c._sys_epoch += 1
    c._sync_system_prompt(ctx, "owner")

    assert _NON_ANTHROPIC_GUARDRAILS in ctx[0].content


def test_decorate_is_pure_no_epoch_side_effect():
    c = _conv()
    before = (c._sys_epoch, c._ctx_sys_epoch)
    c._decorate_system_prompt("owner")
    c._decorate_system_prompt("owner")
    assert (c._sys_epoch, c._ctx_sys_epoch) == before
