"""Memory guard (Riyo, 15 Nov 2026): a final draft is checked against stored
decisions / preferences / lessons before it is sent (syne/memory_guard.py)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from syne import memory_guard as G
import syne.rule_checker as RC


CANDS = [
    {"id": 57566, "category": "decision", "created_at": None, "updated_at": None,
     "similarity": 0.60,
     "content": "session.history_limit = 100 adalah keputusan final Riyo. Jangan diungkit lagi."},
    {"id": 57442, "category": "decision", "created_at": None, "updated_at": None,
     "similarity": 0.82,
     "content": "history_expand default 10 sebelum + 10 sesudah pesan acuan."},
]
CFG = {"enabled": True, "model": "", "categories": ["decision"], "limit": 5,
       "min_similarity": 0.5, "timeout": 30.0}
DRAFT = ("Batas context window di config sekarang session.history_limit = 100, "
         "padahal dulu 1000. Mau kutelusuri?")


def _embed():
    return AsyncMock(return_value=SimpleNamespace(vector=[0.1, 0.2]))


def _judge(reply="CLEAN", exc=None):
    return AsyncMock(return_value=reply, side_effect=exc)


async def _check(**kw):
    args = dict(draft=DRAFT, user_message="History search digabung context window?",
                embed=_embed(), judge=_judge(), exclude_ids=[1, 2],
                access_level="owner", cfg=CFG)
    args.update(kw)
    return await G.check_draft(**args)


# ── verdict parser ──────────────────────────────────────────────────────

def test_parse_clean_after_think_block():
    r = G.parse_verdict("<think>hmm</think>\nCLEAN", [1])
    assert r is not None and r.conflict is False


def test_parse_conflict():
    r = G.parse_verdict("CONFLICT|57566|re-opens a final decision", [57566, 57442])
    assert r.conflict and r.memory_ids == [57566] and "final" in r.reason


def test_parse_conflict_with_unknown_id_is_clean():
    assert G.parse_verdict("CONFLICT|999|x", [57566]).conflict is False


def test_parse_garbage_is_undecided():
    assert G.parse_verdict("I think it is fine", [1]) is None


def test_prompt_carries_memories_user_message_and_draft():
    p = G.build_prompt(DRAFT, "pesan owner", CANDS)
    assert "#57566 [decision]" in p and "history_limit = 100" in p
    assert "pesan owner" in p and "Mau kutelusuri?" in p


# ── check_draft ─────────────────────────────────────────────────────────

async def test_short_draft_is_not_checked():
    emb = _embed()
    r = await _check(draft="Iya.", embed=emb)
    assert r.conflict is False
    emb.assert_not_called()


async def test_no_candidates_means_no_judge_call(monkeypatch):
    monkeypatch.setattr(G, "find_candidates", AsyncMock(return_value=[]))
    judge = _judge()
    r = await _check(judge=judge)
    assert r.conflict is False
    judge.assert_not_called()


async def test_conflict_returns_only_named_memories(monkeypatch):
    fc = AsyncMock(return_value=CANDS)
    monkeypatch.setattr(G, "find_candidates", fc)
    judge = _judge("CONFLICT|57566|re-opens a final decision")
    r = await _check(judge=judge)
    assert r.conflict and [m["id"] for m in r.memories] == [57566]
    prompt, timeout = judge.call_args.args
    assert "#57566" in prompt and timeout == 30.0
    assert fc.call_args.kwargs["exclude_ids"] == [1, 2]
    nudge = G.build_nudge(r)
    assert "#57566" in nudge and "#57442" not in nudge


async def test_judge_error_fails_open(monkeypatch):
    monkeypatch.setattr(G, "find_candidates", AsyncMock(return_value=CANDS))
    assert await _check(judge=_judge(exc=TimeoutError())) is None


async def test_embed_error_fails_open():
    assert await _check(embed=AsyncMock(side_effect=RuntimeError("down"))) is None


async def test_load_config_parses_strings_and_clamps():
    vals = {"memory.guard_enabled": "false",
            "memory.guard_model": "  ollama-qwen  ",
            "memory.guard_categories": '["decision"]',
            "memory.guard_limit": "99",
            "memory.guard_timeout": "1"}

    async def gc(key, default=None):
        return vals.get(key, default)

    c = await G.load_config(gc)
    assert c["enabled"] is False and c["categories"] == ["decision"]
    assert c["model"] == "ollama-qwen"
    assert c["limit"] == 20 and c["timeout"] == 5.0 and c["min_similarity"] == 0.5


# ── judge selection (memory.guard_model) ────────────────────────────────

MODELS = [
    {"key": "ollama-qwen", "driver": "ollama", "model_id": "qwen3:4b",
     "base_url": "http://10.0.0.5:11434"},
    {"key": "flash", "driver": "google_cca", "model_id": "gemini-3.8-flash"},
]


def _gc(models=MODELS):
    async def gc(key, default=None):
        return models if key == "provider.models" else default
    return gc


async def test_judge_default_uses_default_provider(monkeypatch):
    cvp = AsyncMock(return_value="CLEAN")
    monkeypatch.setattr(RC, "_check_via_provider", cvp)
    j = await G.build_judge("", default_provider="DEF", get_config=_gc())
    assert await j("p", 12.0) == "CLEAN"
    assert cvp.call_args.args == ("p", "DEF") and cvp.call_args.kwargs["timeout"] == 12.0


async def test_judge_ollama_entry_calls_ollama_directly(monkeypatch):
    cvo = AsyncMock(return_value="CLEAN")
    monkeypatch.setattr(RC, "_check_via_ollama", cvo)
    j = await G.build_judge("ollama-qwen", default_provider="DEF", get_config=_gc())
    assert j.label == "ollama:qwen3:4b"
    await j("p", 9.0)
    kw = cvo.call_args.kwargs
    assert kw["model"] == "qwen3:4b" and kw["base_url"] == "http://10.0.0.5:11434"


async def test_judge_other_entry_builds_provider(monkeypatch):
    import syne.llm.drivers as D
    built = AsyncMock(return_value="PROV")
    monkeypatch.setattr(D, "create_provider", built)
    cvp = AsyncMock(return_value="CLEAN")
    monkeypatch.setattr(RC, "_check_via_provider", cvp)
    j = await G.build_judge("flash", default_provider="DEF", get_config=_gc())
    await j("p", 5.0)
    assert built.call_args.args[0]["key"] == "flash"
    assert cvp.call_args.args[1] == "PROV"


async def test_judge_missing_key_skips_without_substitute():
    assert await G.build_judge("gone", default_provider="DEF", get_config=_gc()) is None


async def test_judge_unbuildable_entry_skips(monkeypatch):
    import syne.llm.drivers as D
    monkeypatch.setattr(D, "create_provider", AsyncMock(side_effect=RuntimeError("no key")))
    assert await G.build_judge("flash", default_provider="DEF", get_config=_gc()) is None


# ── conversation wiring ─────────────────────────────────────────────────

def _fake_conv(new_content="Jawaban baru tanpa mengungkit history_limit."):
    from syne.llm.provider import ChatResponse
    fake = SimpleNamespace(
        _mgr=SimpleNamespace(provider="default-provider"),
        provider="chat-provider",
        memory=SimpleNamespace(provider=SimpleNamespace(embed=AsyncMock())),
        _last_recalled_memory_ids=[11, 12],
        _llm_call=AsyncMock(return_value=ChatResponse(content=new_content, model="m")),
        _handle_tool_calls=AsyncMock(),
    )
    return fake, ChatResponse(content=DRAFT, model="m")


async def _run_guard(fake, draft_resp):
    from syne.conversation import Conversation
    return await Conversation._memory_guard_and_maybe_rewrite(
        fake, response=draft_resp, context=[], tool_schemas=None,
        access_level="owner", chat_kwargs={}, user_message="pesan",
    )


async def test_conversation_rewrites_once_on_conflict(monkeypatch):
    monkeypatch.setattr(G, "load_config", AsyncMock(return_value=CFG))
    cd = AsyncMock(return_value=G.GuardResult(
        conflict=True, memory_ids=[57566], reason="re-opens", memories=[CANDS[0]]))
    monkeypatch.setattr(G, "check_draft", cd)
    bj = AsyncMock(return_value="JUDGE")
    monkeypatch.setattr(G, "build_judge", bj)
    fake, draft = _fake_conv()
    out = await _run_guard(fake, draft)
    assert out.content.startswith("Jawaban baru")
    assert cd.call_args.kwargs["exclude_ids"] == [11, 12]
    assert cd.call_args.kwargs["judge"] == "JUDGE"
    assert bj.call_args.args[0] == "" and bj.call_args.kwargs["default_provider"] == "default-provider"
    fake._llm_call.assert_awaited_once()
    msgs = fake._llm_call.call_args.kwargs["messages"]
    assert msgs[-2].content == DRAFT and "#57566" in msgs[-1].content


async def test_conversation_keeps_draft_when_judge_unavailable(monkeypatch):
    monkeypatch.setattr(G, "load_config", AsyncMock(return_value=dict(CFG, model="gone")))
    monkeypatch.setattr(G, "build_judge", AsyncMock(return_value=None))
    cd = AsyncMock()
    monkeypatch.setattr(G, "check_draft", cd)
    fake, draft = _fake_conv()
    assert await _run_guard(fake, draft) is draft
    cd.assert_not_called()


async def test_conversation_keeps_draft_when_clean(monkeypatch):
    monkeypatch.setattr(G, "load_config", AsyncMock(return_value=CFG))
    monkeypatch.setattr(G, "build_judge", AsyncMock(return_value="JUDGE"))
    monkeypatch.setattr(G, "check_draft", AsyncMock(return_value=G.GuardResult(conflict=False)))
    fake, draft = _fake_conv()
    assert await _run_guard(fake, draft) is draft
    fake._llm_call.assert_not_called()


async def test_conversation_keeps_draft_when_disabled(monkeypatch):
    monkeypatch.setattr(G, "load_config", AsyncMock(return_value=dict(CFG, enabled=False)))
    cd = AsyncMock()
    monkeypatch.setattr(G, "check_draft", cd)
    fake, draft = _fake_conv()
    assert await _run_guard(fake, draft) is draft
    cd.assert_not_called()


async def test_conversation_keeps_draft_when_rewrite_empty(monkeypatch):
    monkeypatch.setattr(G, "load_config", AsyncMock(return_value=CFG))
    monkeypatch.setattr(G, "build_judge", AsyncMock(return_value="JUDGE"))
    monkeypatch.setattr(G, "check_draft", AsyncMock(return_value=G.GuardResult(
        conflict=True, memory_ids=[57566], memories=[CANDS[0]])))
    fake, draft = _fake_conv(new_content="  ")
    assert await _run_guard(fake, draft) is draft


# ── config seeding ──────────────────────────────────────────────────────

async def test_migration_46_seeds_defaults_idempotently():
    from syne.db import migrations as M
    assert (46, M._m46_memory_guard_config, "transactional") in M.MIGRATIONS
    conn = MagicMock()
    conn.execute = AsyncMock()
    await M._m46_memory_guard_config(conn)
    calls = conn.execute.call_args_list
    assert [c.args[1] for c in calls] == list(G.DEFAULTS)
    assert all("ON CONFLICT (key) DO NOTHING" in c.args[0] for c in calls)


def test_schema_sql_seeds_every_key():
    sql = (Path(G.__file__).parent / "db" / "schema.sql").read_text()
    for key in G.DEFAULTS:
        assert f"('{key}'," in sql


# ── /guard command wiring ───────────────────────────────────────────────

def test_guard_command_registered():
    src = (Path(G.__file__).parent / "communication" / "telegram.py").read_text()
    assert 'CommandHandler("guard", self._cmd_guard)' in src
    assert 'BotCommand("guard",' in src
    assert 'data.startswith("guard:")' in src
    assert 'set_config("memory.guard_model"' in src
    assert 'set_config("memory.guard_enabled"' in src
