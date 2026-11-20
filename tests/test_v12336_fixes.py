"""v1.23.36 (review E/F/G, Riyo #57776): a matching name is not an identity."""
import json
import os
import uuid

import pytest

from syne.memory import kg_extract as K


class _C:
    def __init__(self, directory, users=(), groups=None):
        self.directory, self.users = directory, list(users)
        self.groups = groups or {}

    async def fetchval(self, q, *a):
        if "family.directory" in q:
            return json.dumps(self.directory)
        if "identity" in q:
            return "Molt"
        return None

    async def fetch(self, q, *a):
        if "FROM groups" in q:
            return [{"platform": pf, "platform_group_id": gid,
                     "settings": json.dumps({"members": {m: {"name": m} for m in ms}})}
                    for (pf, gid), ms in self.groups.items()]
        return self.users


RIYO = {"names": ["Riyo", "Riyogarta", "Yahyo"], "access": "owner",
        "chat_id": "3478983", "platform": "telegram"}
AGHA = {"names": ["Agha", "Gamitra"], "access": "family", "chat_id": "41005565",
        "platform": "telegram"}
BUDI = {"names": ["Budi Santoso"], "access": "family", "chat_id": "2", "platform": "telegram"}


# ── E: a word counts only when it is unique ──────────────────────────────

async def test_word_of_a_name_needs_membership_and_uniqueness():
    c = _C([RIYO, BUDI], groups={("telegram", "-100"): ["3478983"],
                                 ("telegram", "-200"): ["2"]})
    known = await K.known_people(c)
    assert "budi" in known and "santoso" in known
    assert K.person_scope("Budi", "person", "chat:telegram:-100", known) == "chat:telegram:-100"
    assert K.person_scope("Budi", "person", "chat:telegram:-200", known) is None
    two = await K.known_people(_C([RIYO, BUDI, {"names": ["Budi Hartono"], "chat_id": "3"}]))
    assert "budi" not in two
    assert "budi santoso" in two and "budi hartono" in two


# ── F: in a group, only a member is the family person ────────────────────

async def test_namesake_in_a_group_without_the_person():
    c = _C([RIYO, AGHA], groups={("telegram", "-300"): ["3478983"],
                                 ("telegram", "-1001185869359"): ["3478983", "41005565"]})
    known = await K.known_people(c)
    assert K.person_scope("Gamitra", "person", "chat:telegram:-300", known) == "chat:telegram:-300"
    assert K.person_scope("Gamitra", "person", "chat:telegram:-1001185869359", known) is None
    assert K.person_scope("Yahyo", "person", "chat:telegram:-300", known) is None
    # a group reminder session belongs to its group
    sch = "chat:telegram:scheduled:-300:12"
    assert K.group_of(sch) == ("telegram", "-300")
    assert K.person_scope("Gamitra", "person", sch, known) == sch
    assert K.person_scope("Yahyo", "person", sch, known) is None
    # DMs carry no chat scope: shared, as before
    assert K.person_scope("Gamitra", "person", None, known) is None


async def test_assistant_name_shared_everywhere():
    known = await K.known_people(_C([RIYO]))
    assert K.person_scope("Molt", "person", "chat:telegram:-999", known) is None


async def test_whatsapp_member_matches_owner_user_row():
    users = [{"platform": "whatsapp", "platform_id": "628111681624", "access_level": "owner",
              "name": "Pak Riyo", "display_name": "Pak Riyo", "aliases": "{}"}]
    c = _C([RIYO], users, groups={("whatsapp", "x@g.us"): ["628111681624"]})
    known = await K.known_people(c)
    assert K.person_scope("Riyo", "person", "chat:whatsapp:x@g.us", known) is None


async def test_cli_account_is_not_a_name():
    users = [{"platform": "cli", "platform_id": "cli:syne", "access_level": "owner",
              "name": "syne", "display_name": "syne", "aliases": "{}"}]
    g = await K.person_groups(_C([RIYO], users), fresh=True)
    assert "syne" not in g and g["riyo"].canon == "Riyo"
    assert "syne" not in await K.known_people(_C([RIYO], users))


def test_group_of():
    assert K.group_of("chat:whatsapp:120@g.us") == ("whatsapp", "120@g.us")
    assert K.group_of("chat:whatsapp:scheduled:120@g.us:5") == ("whatsapp", "120@g.us")
    assert K.group_of(None) is None and K.group_of("x") is None


# ── G: entries sharing a name stay two people ────────────────────────────

MAMA = [{"names": ["Mama", "Sari"], "access": "family", "chat_id": "11"},
        {"names": ["Rina", "Mama"], "access": "family", "chat_id": "12"}]


async def test_shared_name_joins_nobody():
    g = await K.person_groups(_C(MAMA), fresh=True)
    assert "mama" not in g
    assert g["sari"] is not g["rina"]
    assert g["sari"] == {"sari"} and g["rina"] == {"rina"}
    assert g["sari"].canon is None          # "Mama" is ambiguous: no canon
    assert g["rina"].canon == "Rina"
    assert K.person_key("mama", g) is None
    known = await K.known_people(_C(MAMA))
    assert "mama" not in known


async def test_family_row_joins_only_by_chat_id():
    users = [{"platform": "whatsapp", "platform_id": "628", "access_level": "family",
              "name": "Gamitra", "display_name": "Gamitra", "aliases": "{}"},
             {"platform": "telegram", "platform_id": "41005565", "access_level": "family",
              "name": "Agha G", "display_name": None, "aliases": "{}"}]
    g = await K.person_groups(_C([AGHA], users), fresh=True)
    # the linked telegram row adds its names; the unlinked whatsapp row is
    # another person, so "gamitra" now belongs to nobody
    assert "agha g" in g["agha"] and "gamitra" not in g
    assert g["agha"].accounts == {("telegram", "41005565")}


_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


@pg
async def test_merge_keeps_two_mamas_apart_on_real_postgres():
    import asyncpg
    from tests.test_v12332_fixes import _BASE
    c = await asyncpg.connect(_DB)
    schema = "kgmama_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    try:
        await c.execute(_BASE)
        await c.execute("CREATE UNIQUE INDEX ON kg_entities (LOWER(name), entity_type, coalesce(scope, ''))")
        await c.execute("INSERT INTO config VALUES ('family.directory', $1::jsonb)", json.dumps(MAMA))
        ids = {}
        for n in ("Mama", "Sari", "Rina"):
            ids[n] = await c.fetchval("INSERT INTO kg_entities (name, entity_type, norm_name) "
                                      "VALUES ($1, 'person', $2) RETURNING id", n, K.person_norm(n))
        assert await K.merge_known_people(c) == {"merged": 0, "people": 0}
        assert await K.name_known_people(c) == 0
        names = sorted(r["name"] for r in await c.fetch("SELECT name FROM kg_entities"))
        assert names == ["Mama", "Rina", "Sari"]
    finally:
        await c.execute(f"DROP SCHEMA {schema} CASCADE")
        await c.close()
