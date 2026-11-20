"""v1.23.40: members-menu Delete renamed Revoke access (it never deleted)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_members_menus_say_revoke():
    tg = (ROOT / "syne/communication/telegram.py").read_text()
    assert '"↩️ Revoke access", callback_data=f"mbr:delete:{platform_id}"' in tg
    assert '"↩️ Revoke access", callback_data=f"dc:delete_confirm:{eid}"' in tg
    assert "User deleted — must be approved again" not in tg
    assert '"Deleted — must be approved again"' not in tg
    # real deletes elsewhere keep their label
    assert 'callback_data=f"models:delete_confirm:{model_key}"' in tg
    mc = (ROOT / "syne/members_cleanup.py").read_text()
    assert "Idle members revoked" in mc and "Delete button" not in mc
