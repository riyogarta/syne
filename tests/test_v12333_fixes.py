"""v1.23.33: a known person's entity is named after family.directory."""
import json
import os
import uuid

import pytest

import syne.db.migrations as M
from syne.memory import kg_extract as K
from tests.test_v12332_fixes import DIRECTORY, USERS, _BASE, _Conn


async def test_canon_is_first_directory_name():
    g = await K.person_groups(_Conn(), fresh=True)
    assert g["riyogarta pratikto"].canon == "Riyo"
    assert g["gamitra"].canon == "Agha"
    assert g["bunli"].canon == "Yuli"
    assert g["riyo"] is not g["yuli"] and g["yuli"] is not g["agha"]


async def test_user_without_entry_has_no_canon():
    class C(_Conn):
        async def fetch(self, q, *a):
            return USERS + [{"platform": "discord", "platform_id": "77", "name": "Budi",
                             "display_name": "Budi", "aliases": "{}"}]
    g = await K.person_groups(C(), fresh=True)
    assert g["budi"].canon is None and g["riyo"].canon == "Riyo"


def test_migration_66_registered():
    assert M.MIGRATIONS[-1][0] == 66 and M.CURRENT_SCHEMA_VERSION == 66


_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


@pg
async def test_rename_and_new_entity_on_real_postgres():
    import asyncpg
    c = await asyncpg.connect(_DB)
    schema = "kgname_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    try:
        await c.execute(_BASE)
        await c.execute("CREATE UNIQUE INDEX ON kg_entities (LOWER(name), entity_type, coalesce(scope, ''))")
        await c.execute("INSERT INTO config VALUES ('family.directory', $1::jsonb)", json.dumps(DIRECTORY))
        for u in USERS:
            await c.execute("INSERT INTO users (platform, platform_id, name, display_name, aliases, "
                            "access_level) VALUES ($1, $2, $3, $4, $5::jsonb, 'owner')",
                            u["platform"], u["platform_id"], u["name"], u["display_name"], u["aliases"])

        async def ent(name):
            return await c.fetchval(
                "INSERT INTO kg_entities (name, entity_type, norm_name) "
                "VALUES ($1, 'person', $2) RETURNING id", name, K.person_norm(name))

        full, riyo, gam, molt = (await ent("Riyogarta Pratikto"), await ent("Riyo"),
                                 await ent("Gamitra"), await ent("Molt"))
        await c.execute("INSERT INTO kg_relations (subject_id, predicate, object_id) "
                        "VALUES ($1, 'uses', $2)", full, molt)
        await K.merge_known_people(c)
        assert await K.name_known_people(c) == 2
        rows = {r["id"]: (r["name"], sorted(r["aliases"]), r["norm_name"])
                for r in await c.fetch("SELECT * FROM kg_entities")}
        assert rows[full] == ("Riyo", ["Riyogarta Pratikto"], "riyo")
        assert rows[gam] == ("Agha", ["Gamitra"], "agha")
        assert riyo not in rows
        assert await K.name_known_people(c) == 0          # idempotent

        # a person with no entity yet: created under the directory name
        g = await K.person_groups(c, fresh=True)
        s = {"_person_groups": g}
        eid, how = await K.resolve_entity(c, "Bunli", "person", None, s, None)
        assert how == "person"
        assert await c.fetchrow("SELECT name, aliases FROM kg_entities WHERE id = $1", eid) == ("Yuli", ["Bunli"])
        assert await K.resolve_entity(c, "Bu Yuliazmi", "person", None, s, None) == (eid, "person")
        assert await K.resolve_entity(c, "Yahyo", "person", None, s, None) == (full, "person")
    finally:
        await c.execute(f"DROP SCHEMA {schema} CASCADE")
        await c.close()
