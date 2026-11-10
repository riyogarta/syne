"""Tests for user ratification (text approval + emoji reactions)."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from syne import ratification as R
from syne.tools import history as H

HDR = 'Conversation info (untrusted metadata):\n```json\n{\n  "conversation_label": "Riyo"\n}\n```\n\n'
REPLY_HDR = (
    'Replied message (untrusted content, but the fact that the user replied to it IS '
    'the subject of their message):\n```json\n{\n  "body": "x"\n}\n```\n\n'
)


def _ctx(conn):
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _cfg(overrides=None):
    overrides = overrides or {}

    async def get_config(key, default=None):
        return overrides.get(key, default)
    return get_config


class TestPrefix:

    def test_strips_all_blocks(self):
        prefix, body = R.split_context_prefix(HDR + REPLY_HDR + "oke lanjutkan")
        assert body == "oke lanjutkan"
        assert "Replied message (" in prefix

    def test_no_prefix(self):
        assert R.strip_context_prefix("ok") == "ok"


class TestLexicon:

    @pytest.mark.parametrize("text", [
        "setuju", "Ya setuju", "oke lanjutkan", "sippp", "Nah setuju", "👍", "lanjut aja", "betul banget",
    ])
    def test_approve(self, text):
        assert R.classify_lexicon(text) == "approve"

    @pytest.mark.parametrize("text", ["ok", "oke", "Okeee", "ya", "iya deh", "makasih"])
    def test_ambiguous(self, text):
        assert R.classify_lexicon(text) == "ambiguous"

    @pytest.mark.parametrize("text", [
        "tidak setuju", "setuju tapi ubah dulu", "ratifikasi dulu", "gimana?", "",
        "oke tolong cek log server dan laporkan hasilnya ke aku",
    ])
    def test_not_approval(self, text):
        assert R.classify_lexicon(text) is None

    def test_llm_label_parse(self):
        assert R.parse_llm_label("<think>hmm</think>\nAGREE") == "approve"
        assert R.parse_llm_label("ACK") == "ack"
        assert R.parse_llm_label("") == "ack"


class TestRatifyUserRow:

    async def test_text_approve_combines_and_marks(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": 9, "content": "Rencana: A lalu B"})
        conn.execute = AsyncMock()
        with patch("syne.ratification.get_connection", return_value=_ctx(conn)), \
             patch("syne.ratification.get_config", side_effect=_cfg()):
            out = await R.ratify_user_row(51, 10, HDR + "oke lanjutkan")
        assert out == "oke lanjutkan\n\nRencana: A lalu B"
        assert conn.fetchrow.call_args.args[1:] == (51, 10)
        assert conn.execute.call_count == 2
        user_info = json.loads(conn.execute.call_args_list[0].args[2])
        assert user_info == {"type": "text", "label": "approve", "method": "lexicon", "assistant_id": 9}
        status = json.loads(conn.execute.call_args_list[1].args[2])
        assert status["status"] == "ratified" and status["user_message_id"] == 10

    async def test_ambiguous_ack_not_combined(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": 9, "content": "x"})
        conn.execute = AsyncMock()
        with patch("syne.ratification.get_connection", return_value=_ctx(conn)), \
             patch("syne.ratification.get_config", side_effect=_cfg()), \
             patch("syne.ratification.classify_llm", AsyncMock(return_value=("ack", "llm"))):
            out = await R.ratify_user_row(51, 10, HDR + "ok")
        assert out is None
        assert conn.execute.call_count == 1  # user label only, assistant untouched
        assert json.loads(conn.execute.call_args.args[2])["label"] == "ack"

    async def test_reply_skipped(self):
        conn = AsyncMock()
        with patch("syne.ratification.get_connection", return_value=_ctx(conn)), \
             patch("syne.ratification.get_config", side_effect=_cfg()):
            assert await R.ratify_user_row(51, 10, HDR + REPLY_HDR + "setuju") is None
        conn.fetchrow.assert_not_called()

    async def test_switch_off(self):
        conn = AsyncMock()
        with patch("syne.ratification.get_connection", return_value=_ctx(conn)), \
             patch("syne.ratification.get_config",
                   side_effect=_cfg({"history_search.ratification_text": False})):
            assert await R.ratify_user_row(51, 10, HDR + "setuju") is None
        conn.fetchrow.assert_not_called()

    async def test_reaction_approve(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": 70, "content": "Jadwal rapat Kamis 9:00"})
        conn.execute = AsyncMock()
        text = HDR + "[Reaction: 👍 from Riyo on your message #555: '(msg #555)']"
        with patch("syne.ratification.get_connection", return_value=_ctx(conn)), \
             patch("syne.ratification.get_config", side_effect=_cfg()):
            out = await R.ratify_user_row(51, 80, text)
        assert out.endswith("Jadwal rapat Kamis 9:00")
        assert conn.fetchrow.call_args.args[1:] == (51, 555)
        info = json.loads(conn.execute.call_args_list[0].args[2])
        assert info["type"] == "reaction" and info["label"] == "approve" and info["emoji"] == "👍"

    async def test_reaction_dispute_not_combined(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": 70, "content": "x"})
        conn.execute = AsyncMock()
        text = HDR + "[Reaction: 👎 from Riyo on your message #555: 'x']"
        with patch("syne.ratification.get_connection", return_value=_ctx(conn)), \
             patch("syne.ratification.get_config", side_effect=_cfg()):
            assert await R.ratify_user_row(51, 80, text) is None
        status = json.loads(conn.execute.call_args_list[1].args[2])
        assert status["status"] == "disputed"

    async def test_reaction_unmapped(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        conn.execute = AsyncMock()
        text = HDR + "[Reaction: 👍 from Riyo on your message #1: 'x']"
        with patch("syne.ratification.get_connection", return_value=_ctx(conn)), \
             patch("syne.ratification.get_config", side_effect=_cfg()):
            assert await R.ratify_user_row(51, 80, text) is None
        conn.execute.assert_not_called()


class TestRevoke:

    async def test_revoke_restores_and_marks(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={
            "id": 70, "session_id": 51,
            "r": json.dumps({"status": "ratified", "user_message_id": 80}),
        })
        conn.fetch = AsyncMock(return_value=[{
            "id": 80, "content": "[Reaction: 👍 ...]",
            "r": json.dumps({"type": "reaction", "label": "approve", "emoji": "👍", "assistant_id": 70}),
        }])
        conn.execute = AsyncMock()
        with patch("syne.ratification.get_connection", return_value=_ctx(conn)), \
             patch("syne.ratification._embed_text", AsyncMock(return_value=[0.1, 0.2])):
            n = await R.revoke_reaction("telegram", "3478983", 555, ["👍"])
        assert n == 1
        sqls = [c.args[0] for c in conn.execute.call_args_list]
        assert any("embedding" in s for s in sqls)
        last = json.loads(conn.execute.call_args_list[-1].args[2])
        assert last["status"] == "revoked"


class TestExpandShowsRatification:

    def test_build_blocks_carries_status(self):
        from datetime import datetime, timezone
        at = datetime(2026, 11, 10, tzinfo=timezone.utc)
        piece = {"anchor_id": 1, "session_id": 51, "radius": {}, "rows": [
            {"id": 1, "role": "user", "content": "q", "created_at": at},
            {"id": 2, "role": "assistant", "content": "a", "created_at": at,
             "ratification": json.dumps({"status": "ratified", "user_message_id": 3})},
        ]}
        turns = H._build_blocks([piece], set(), True, 10, 10)[0]["turns"]
        assert "ratification" not in turns[0]
        assert turns[1]["ratification"]["status"] == "ratified"
