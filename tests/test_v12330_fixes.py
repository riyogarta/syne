"""v1.23.30: review of v1.23.20-29."""
from pathlib import Path

import syne.memory.embed_dims as D
import syne.memory.kg_extract as K

ROOT = Path(__file__).resolve().parents[1]


def test_auth_on_one_chunk_while_others_answer_spends_an_attempt():
    assert not K.spends_attempt("auth", 10.0, 1, last_ok=50.0, now=60.0)
    assert K.spends_attempt("auth", 10.0, K.CHUNK_FAULT_STRIKES, last_ok=50.0, now=60.0)


def test_auth_with_no_answer_never_spends():
    far = 10.0 + K.PROVIDER_DOWN_GIVE_UP * 9
    assert not K.spends_attempt("auth", 10.0, 99, last_ok=5.0, now=far)


def test_worker_tracks_auth_failures_as_suspects():
    src = (ROOT / "syne/memory/kg_extract.py").read_text()
    assert 'if kind != "auth":' not in src
    assert "suspects[this] = (first, strikes)" in src


def test_history_hides_deleted_user_rows():
    src = (ROOT / "syne/tools/history.py").read_text()
    assert "NOT coalesce(m.metadata ? 'deleted_user', false)" in src


def test_history_logs_null_similarity():
    src = (ROOT / "syne/tools/history.py").read_text()
    assert 'float(r["similarity"] or 0.0)' not in src
    assert 'None if r["similarity"] is None else float(r["similarity"])' in src


class _Conn:
    def __init__(self, counts):
        self.counts = counts

    async def fetchval(self, q, *a):
        return next(n for t, n in self.counts.items() if f"FROM {t} " in q)


async def test_hint_names_era_and_no_command():
    h = await D.reembed_hint(_Conn({"memory": 7, "messages": 3}), {"blocked": [], "relax": []})
    assert D.ERA_KEY in h and "syne memory" not in h
    assert await D.reembed_hint(_Conn({"memory": 0, "messages": 0}),
                                {"blocked": [], "relax": []}) == ""


def test_switch_records_era_and_guide_drops_reembed():
    tg = (ROOT / "syne/communication/telegram.py").read_text()
    assert "ERA_KEY: datetime.now(timezone.utc).isoformat()" in tg
    guide = (ROOT / "syne/config_guide.py").read_text()
    assert "reembed-memory --force" not in guide
    assert "embedding.era_start" in guide
