"""v1.23.23: titles of address are not names (review v1.23.16-19 bug 5,
decision #57775 option B)."""
from syne.memory import kg_extract as K

G = "chat:telegram:-100"


class _Conn:
    def __init__(self, users):
        self.users = users

    async def fetch(self, q, *a):
        return self.users

    async def fetchval(self, q, *a):
        return "Molt"


async def _known(monkeypatch, users, directory=()):
    async def cfg(key, default=None):
        return list(directory)
    monkeypatch.setattr(K, "get_config", cfg)
    return await K.known_people(_Conn(users))


def _u(name, display=None, aliases="{}"):
    return {"name": name, "display_name": display, "aliases": aliases}


async def test_title_is_not_a_known_name(monkeypatch):
    known = await _known(monkeypatch, [
        _u("Pak Riyo", "Pak Riyo"), _u("Ibu Yuli", aliases='{"groups": {"-1": "Mbak Yuli"}}'),
        _u("Ustadz Agha")])
    assert {"riyo", "yuli", "agha", "molt"} <= known
    assert not (known & K.HONORIFICS)
    assert "pak riyo" not in known


async def test_other_people_with_a_title_stay_in_their_group(monkeypatch):
    known = await _known(monkeypatch, [_u("Pak Riyo")])
    for name in ("Pak Budi", "Bu Sari", "Mas Andi", "Pak Dr. Hasan", "Pak", "Prof Hari"):
        assert K.person_scope(name, "person", G, known) == G, name


async def test_owner_and_family_stay_shared_with_or_without_title(monkeypatch):
    known = await _known(monkeypatch, [_u("Riyogarta", "Riyogarta Pratikto")],
                         [{"names": ["Riyo", "Yahyo"]}, {"names": ["Yuli", "Bunli"]}])
    for name in ("Riyo", "Pak Riyo", "Yahyo", "Bu Yuli", "Mbak Bunli",
                 "Riyogarta Pratikto", "Pak Riyogarta", "Molt"):
        assert K.person_scope(name, "person", G, known) is None, name


def test_strip_honorifics():
    assert K.strip_honorifics(K.normalize_name("Pak Dr. Budi")) == "budi"
    assert K.strip_honorifics(K.normalize_name("Ustadzah Aisyah")) == "aishah"
    assert K.strip_honorifics("pak") == ""
    # a title inside the name is kept: only leading titles are dropped
    assert K.strip_honorifics("budi mas") == "budi mas"
