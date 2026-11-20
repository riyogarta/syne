"""v1.23.29: KG worker follows the embedding provider; /embedding hints
(review v1.23.16-19 bug 8)."""
from pathlib import Path
from types import SimpleNamespace

from syne.memory import embed_dims as D
from syne.memory import kg_extract as K

ROOT = Path(__file__).resolve().parent.parent


class _Prov:
    def __init__(self, v):
        self.v = v

    async def embed(self, text):
        return SimpleNamespace(vector=[self.v])


async def test_embedder_follows_provider_switch():
    holder = {"p": _Prov(1.0)}
    embed = K.make_embedder(lambda: holder["p"])
    assert await embed("x") == [1.0]
    holder["p"] = _Prov(2.0)            # /embedding -> reload_provider
    assert await embed("x") == [2.0]
    holder["p"] = None
    assert await embed("x") is None
    assert await K.make_embedder(None)("x") is None


def test_agent_passes_a_getter():
    src = (ROOT / "syne/agent.py").read_text()
    assert "embed_provider_getter=lambda: self.memory.provider" in src
    assert "embed_provider=self.memory.provider" not in src


class _Conn:
    def __init__(self, counts):
        self.counts = counts

    async def fetchval(self, q, *a):
        return next(n for t, n in self.counts.items() if f"FROM {t} " in q)


async def test_hint_same_dims_lists_tables_with_vectors():
    h = await D.reembed_hint(_Conn({"memory": 9, "messages": 5}), {"blocked": [], "relax": []})
    assert D.REEMBED_CMD["memory"] in h and D.REEMBED_CMD["messages"] in h


async def test_hint_skips_empty_and_retyped_tables():
    assert await D.reembed_hint(_Conn({"memory": 0, "messages": 0}),
                                {"blocked": [], "relax": ["memory", "messages"]}) == ""
    h = await D.reembed_hint(_Conn({"memory": 9, "messages": 0}), {"blocked": [], "relax": []})
    assert D.REEMBED_CMD["memory"] in h and "reembed-history" not in h


def test_texts_updated():
    tg = (ROOT / "syne/communication/telegram.py").read_text()
    assert "afterwards so search keeps working" not in tg
    assert "Same-dimension switches keep them" not in tg
    assert "reembed_hint(conn, plan)" in tg
    assert "Clear history vectors and switch" in (ROOT / "syne/config_guide.py").read_text()
