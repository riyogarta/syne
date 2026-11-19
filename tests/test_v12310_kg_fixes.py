"""v1.23.10: guard ignores contains, range x range pairs, title_target, per-turn inbound."""
import asyncio
from types import SimpleNamespace

import pytest

from syne.db import memory_guard_ddl as G
from syne.db import migrations as M
from syne.memory import kg_build as B
from syne.memory import kg_extract as K


def test_rel_pairs_single_side_fans_out():
    assert K._rel_pairs([1], [10, 11, 12]) == [(1, 10), (1, 11), (1, 12)]
    assert K._rel_pairs([1, 2], [10]) == [(1, 10), (2, 10)]


def test_rel_pairs_range_to_range_is_item_by_item():
    assert K._rel_pairs([1, 2, 3], [10, 11, 12]) == [(1, 10), (2, 11), (3, 12)]


def test_rel_pairs_unequal_ranges_give_nothing():
    assert K._rel_pairs(list(range(50)), list(range(100, 150 + 1))) == []


def test_guard_ignores_contains():
    assert "AND r.predicate <> 'contains'" in G.MEMORY_GUARD_SCHEMA_SQL


def test_default_title_target():
    assert B.DEFAULT_REF_SOURCES["ibnukatsir"]["title_target"] == "alquran"
    assert B.DEFAULT_REF_SOURCES["asbabnuzul"]["title_target"] == "alquran"


def test_schema_sql_title_target():
    from pathlib import Path
    sql = (Path(__file__).resolve().parents[1] / "syne" / "db" / "schema.sql").read_text()
    assert sql.count('"title_target": "alquran"') == 2


def test_migration_58_registered():
    assert (58, M._m58_guard_contains_title_target, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 58


def test_extract_prompt_skips_examples():
    assert "hypothetical scenarios" in K.PROMPT


async def test_turn_keeps_its_own_inbound():
    """A turn reads the inbound it started with, not the latest one."""
    from syne.conversation import Conversation, CURRENT_CONVERSATION
    a = SimpleNamespace(sender_access="owner", sender_id="1")
    b = SimpleNamespace(sender_access="public", sender_id="2")
    conv = SimpleNamespace(inbound=a, _turn_inbound=None)
    conv._inb = lambda: Conversation._inb(conv)
    conv._turn_inbound = a
    conv.inbound = b  # next message arrived mid-turn
    assert conv._inb() is a
    conv._turn_inbound = None
    assert conv._inb() is b


async def test_current_conversation_is_per_task():
    from syne.conversation import CURRENT_CONVERSATION
    seen = {}

    async def turn(name):
        CURRENT_CONVERSATION.set(name)
        await asyncio.sleep(0.01)
        seen[name] = CURRENT_CONVERSATION.get()

    await asyncio.gather(turn("a"), turn("b"))
    assert seen == {"a": "a", "b": "b"}


async def test_agent_active_conversation_prefers_current():
    from syne.agent import SyneAgent
    from syne.conversation import CURRENT_CONVERSATION
    other = SimpleNamespace(_processing=True)
    me = SimpleNamespace(_processing=True)
    fake = SimpleNamespace(conversations=SimpleNamespace(_active={"x": other, "y": me}))
    tok = CURRENT_CONVERSATION.set(me)
    try:
        assert SyneAgent._get_active_conversation(fake) is me
    finally:
        CURRENT_CONVERSATION.reset(tok)
