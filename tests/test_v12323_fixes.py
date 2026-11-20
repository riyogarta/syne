"""v1.23.23: titles of address are not names (review v1.23.16-19 bug 5,
decision #57775 option B). v1.23.36: fakes read the directory and the
group members from the connection."""
import json

from syne.memory import kg_extract as K

G = "chat:telegram:-100"


class _Conn:
    def __init__(self, users, directory=(), members=()):
        self.users, self.directory, self.members = users, list(directory), members

    async def fetch(self, q, *a):
        if "FROM groups" in q:
            return [{"platform": "telegram", "platform_group_id": "-100",
                     "settings": {"members": {m: {} for m in self.members}}}]
        return self.users

    async def fetchval(self, q, *a):
        if "family.directory" in q:
            return json.dumps(self.directory)
        return "Molt"


async def _known(users, directory=(), members=()):
    return await K.known_people(_Conn(users, directory, members))


def _u(name, display=None, aliases="{}", pid="9", access="family"):
    return {"platform": "telegram", "platform_id": pid, "access_level": access,
            "name": name, "display_name": display, "aliases": aliases}


async def test_title_is_not_a_known_name():
    known = await _known([
        _u("Pak Riyo", "Pak Riyo", pid="1"),
        _u("Ibu Yuli", aliases='{"groups": {"-1": "Mbak Yuli"}}', pid="2"),
        _u("Ustadz Agha", pid="3")])
    assert {"riyo", "yuli", "agha", "molt"} <= known
    assert not (known & K.HONORIFICS)
    assert "pak riyo" not in known


async def test_other_people_with_a_title_stay_in_their_group():
    known = await _known([_u("Pak Riyo", pid="1")], members=["1"])
    for name in ("Pak Budi", "Bu Sari", "Mas Andi", "Pak Dr. Hasan", "Pak", "Prof Hari"):
        assert K.person_scope(name, "person", G, known) == G, name


async def test_owner_and_family_stay_shared_with_or_without_title():
    known = await _known(
        [_u("Riyogarta", "Riyogarta Pratikto", pid="1", access="owner")],
        [{"names": ["Riyo", "Yahyo"], "access": "owner", "chat_id": "1"},
         {"names": ["Yuli", "Bunli"], "access": "family", "chat_id": "2"}],
        members=["1", "2"])
    for name in ("Riyo", "Pak Riyo", "Yahyo", "Bu Yuli", "Mbak Bunli",
                 "Riyogarta Pratikto", "Pak Riyogarta", "Molt"):
        assert K.person_scope(name, "person", G, known) is None, name


def test_strip_honorifics():
    assert K.strip_honorifics(K.normalize_name("Pak Dr. Budi")) == "budi"
    assert K.strip_honorifics(K.normalize_name("Ustadzah Aisyah")) == "aishah"
    assert K.strip_honorifics("pak") == ""
    # a title inside the name is kept: only leading titles are dropped
    assert K.strip_honorifics("budi mas") == "budi mas"
