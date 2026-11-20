"""v1.23.39: family.directory documented; the unlinked-accounts notice only
for family accounts that look like another person; a check made before the
Telegram channel is up is retried within a minute."""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from syne.memory import kg_extract as K
from tests.test_v12336_fixes import _C, RIYO
from tests.test_v12337_fixes import YULI
from tests.test_v12338_fixes import _u, OWNER_TG, OWNER_WA, CLI

ROOT = Path(__file__).resolve().parents[1]


async def test_unlinked_only_when_a_name_is_shared():
    fam = lambda pf, pid, name: _u(pf, pid, name, access="family")
    c = _C([RIYO, YULI], [OWNER_TG, OWNER_WA, CLI,
                          fam("whatsapp", "628111681524", "Yuli WA"),    # listed
                          fam("discord", "77", "Bunli"),                 # like Yuli
                          fam("telegram", "88", "Budi"),                 # nobody else
                          fam("whatsapp", "6281", "Budi"),               # like the Budi above
                          fam("discord", "99", "Riyo")])                 # like the owner
    rows = await K.unlinked_accounts(c)
    assert rows == [("discord", "77", "Bunli", "Yuli"), ("discord", "99", "Riyo", "Riyo"),
                    ("telegram", "88", "Budi", "Budi"), ("whatsapp", "6281", "Budi", "Budi")]
    t = K.unlinked_text(rows)
    assert "discord 77 (Bunli), looks like Yuli" in t and '"accounts"' in t
    # an owner on two platforms, a family member alone: nothing to tell
    assert await K.unlinked_accounts(_C([], [OWNER_TG, OWNER_WA, fam("telegram", "5", "Sari")])) == []


async def test_no_channel_yet_retries_soon(monkeypatch):
    import syne.communication.registry as REG
    conn = _C([RIYO, YULI], [OWNER_TG, _u("discord", "77", "Bunli", access="family")])

    @asynccontextmanager
    async def gconn(timeout=None):
        yield conn
    store: dict = {}

    async def gc(k, d=None):
        return store.get(k, d)

    async def sc(k, v, desc=None):
        store[k] = v
    monkeypatch.setattr(K, "get_connection", gconn)
    monkeypatch.setattr(K, "get_config", gc)
    monkeypatch.setattr(K, "set_config", sc)
    monkeypatch.setattr(REG, "get_channel", lambda name: None)
    monkeypatch.setattr(K, "_unlinked_retry_at", 0.0)
    assert await K.maybe_notify_unlinked() is False
    wait = K._unlinked_retry_at - time.monotonic()
    assert 0 < wait <= K.UNLINKED_RETRY and K.UNLINKED_KEY not in store


def test_guide_documents_family_directory():
    g = (ROOT / "syne/config_guide.py").read_text()
    seg = g[g.index("## Family Directory"):g.index("## Execution & Tools")]
    for word in ("`names`", "`chat_id`", "`platform`", "`access`", "`accounts`",
                 "@s.whatsapp.net", "Discord: the user id", "replaces the whole list",
                 "Not needed", "Needed"):
        assert word in seg, word
