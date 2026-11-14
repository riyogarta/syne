"""history_judge: the model's verdict on one history_search log row."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from syne.security import TOOL_PERMISSIONS, needs_consent
from syne.tools import history as H

SCOPE = {"access": "owner", "platform": "telegram", "chat_id": "3478983",
         "is_group": False, "sender_pid": "3478983", "session_id": 1}


def _conn(row):
    cap = {"execute": []}
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=row)

    async def execute(sql, *params):
        cap["execute"].append((sql, params))

    conn.execute = AsyncMock(side_effect=execute)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, cap


async def _judge(row, **kw):
    ctx, cap = _conn(row)
    with patch("syne.tools.history.get_connection", return_value=ctx):
        res = await H.history_judge_handler(_scope=SCOPE, **kw)
    return res, cap


async def test_found_records_relevant_hits():
    res, cap = await _judge({"session_id": 1, "hit_ids": [10, 20, 30]},
                            log_id=5, found=True, relevant_ids=[20, 99])
    out = json.loads(res)
    assert out["found"] is True and out["relevant_ids"] == [20]
    assert out["ignored_ids"] == [99]
    sql, params = cap["execute"][0]
    assert "judged_found = $2" in sql and "judged_at = now()" in sql
    assert params == (5, True, [20])


async def test_not_found_records_empty():
    res, cap = await _judge({"session_id": 1, "hit_ids": [10]}, log_id=5, found=False)
    assert json.loads(res)["found"] is False
    assert cap["execute"][0][1] == (5, False, [])


async def test_string_booleans_accepted():
    res, _ = await _judge({"session_id": 1, "hit_ids": [10]}, log_id="5", found="false")
    assert json.loads(res)["found"] is False


async def test_contradictions_refused():
    res, cap = await _judge({"session_id": 1, "hit_ids": [10]}, log_id=5, found=False, relevant_ids=[10])
    assert res.startswith("Error:") and not cap["execute"]
    res, cap = await _judge({"session_id": 1, "hit_ids": [10]}, log_id=5, found=True)
    assert res.startswith("Error:") and not cap["execute"]
    res, cap = await _judge({"session_id": 1, "hit_ids": [10]}, log_id=5, found=True, relevant_ids=[77])
    assert res.startswith("Error:") and "[10]" in res and not cap["execute"]


async def test_other_session_refused():
    res, cap = await _judge({"session_id": 2, "hit_ids": [10]}, log_id=5, found=True, relevant_ids=[10])
    assert "another chat" in res and not cap["execute"]


async def test_missing_row_and_bad_args():
    res, _ = await _judge(None, log_id=5, found=False)
    assert res.startswith("Error:") and "5" in res
    assert (await H.history_judge_handler(_scope=SCOPE, log_id="x", found=True)).startswith("Error:")
    assert (await H.history_judge_handler(_scope=SCOPE, log_id=5, found="maybe")).startswith("Error:")
    assert (await H.history_judge_handler(log_id=5, found=False)) == H._NO_SCOPE_ERROR


def test_permission_is_open_without_gate():
    perm = TOOL_PERMISSIONS["history_judge"]
    assert perm == H.HISTORY_JUDGE_TOOL["permission"] == 0o666
    for level in ("owner", "family", "public"):
        assert not needs_consent(level, perm)


async def test_search_returns_log_id_and_judge_hint():
    prov = MagicMock()
    prov.embed = AsyncMock(return_value=MagicMock(vector=[0.1]))
    H.set_embedding_provider(prov)
    from datetime import datetime, timezone
    row = {"id": 101, "session_id": 1, "chat_id": "3478983", "platform": "telegram",
           "role": "user", "created_at": datetime.now(timezone.utc), "similarity": 0.7}
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[row])
    conn.fetchval = AsyncMock(return_value=42)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    expand = AsyncMock(return_value={"blocks": []})
    with patch("syne.tools.history.get_connection", return_value=ctx), \
            patch("syne.tools.history._expand", expand):
        sim = json.loads(await H.history_search_handler(_scope=SCOPE, query="kopi"))
        rec = json.loads(await H.history_search_handler(_scope=SCOPE, query="kopi", sort_by="recency"))
    assert sim["log_id"] == 42 and "log_id=42" in sim["judge"]
    # Recency searches are not used for the threshold: no judge request.
    assert rec["log_id"] == 42 and "judge" not in rec
