"""v1.23.41: /wamembers Delete relabelled Revoke access (same word in all menus)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_wamembers_says_revoke():
    tg = (ROOT / "syne/communication/telegram.py").read_text()
    assert tg.count('"↩️ Revoke access", callback_data=f"wa:delete_confirm:{jid}"') == 2
    assert '"🗑 Delete", callback_data=f"wa:delete_confirm:' not in tg
    assert 'f"Deleted {name}"' not in tg
    # all three members menus use the same label
    for prefix in ("mbr:delete:", "dc:delete_confirm:", "wa:delete_confirm:"):
        assert f'"↩️ Revoke access", callback_data=f"{prefix}' in tg, prefix
