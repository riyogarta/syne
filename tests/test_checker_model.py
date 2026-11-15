"""/checker model picker (Riyo, 15 Nov 2026): the rule checker can run on the
default model, the evaluator, or any /models entry (security.rule_checker_model)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import syne.db.models as DBM
import syne.judge_model as JM
import syne.rule_checker as RC
from syne.rule_checker import CheckResult, VerdictState, check_response

RULES = [{"code": "X", "name": "n", "description": "d", "severity": "hard"}]
MODELS = [
    {"key": "ollama-qwen", "driver": "ollama", "model_id": "qwen3:4b",
     "base_url": "http://10.0.0.5:11434"},
    {"key": "flash", "driver": "google_cca", "model_id": "gemini-3.8-flash"},
]


def _gc(models=MODELS):
    async def gc(key, default=None):
        return models if key == "provider.models" else default
    return gc


# ── shared judge builder ────────────────────────────────────────────────

async def test_judge_default_uses_default_provider(monkeypatch):
    cvp = AsyncMock(return_value="CLEAN")
    monkeypatch.setattr(RC, "_check_via_provider", cvp)
    j = await JM.build_judge("", default_provider="DEF", get_config=_gc())
    assert await j("p", 7.0) == "CLEAN"
    assert cvp.call_args.args == ("p", "DEF")


async def test_judge_ollama_entry_direct(monkeypatch):
    cvo = AsyncMock(return_value="CLEAN")
    monkeypatch.setattr(RC, "_check_via_ollama", cvo)
    j = await JM.build_judge("ollama-qwen", default_provider="DEF", get_config=_gc())
    await j("p", 9.0)
    assert cvo.call_args.kwargs["base_url"] == "http://10.0.0.5:11434"
    assert j.label == "ollama:qwen3:4b"


async def test_judge_missing_and_unbuildable_return_none(monkeypatch):
    import syne.llm.drivers as D
    assert await JM.build_judge("gone", default_provider="DEF", get_config=_gc()) is None
    monkeypatch.setattr(D, "create_provider", AsyncMock(side_effect=RuntimeError("no key")))
    assert await JM.build_judge("flash", default_provider="DEF", get_config=_gc()) is None


# ── check_response with a judge ─────────────────────────────────────────

async def test_check_response_uses_judge_instead_of_drivers(monkeypatch):
    cvo = AsyncMock(return_value="VIOLATED|X|should not be used")
    monkeypatch.setattr(RC, "_check_via_ollama", cvo)
    judge = AsyncMock(return_value="CLEAN")
    r = await check_response(draft="halo semua", user_message="hi", hard_rules=RULES,
                             evaluator_driver="ollama", judge=judge, timeout=11.0)
    assert r.state == VerdictState.CLEAN
    cvo.assert_not_called()
    assert judge.call_args.args[1] == 11.0


async def test_check_response_judge_error_is_error():
    judge = AsyncMock(side_effect=RuntimeError("model 'gone' unavailable"))
    r = await check_response(draft="halo semua", user_message="hi", hard_rules=RULES,
                             evaluator_driver="provider", provider=object(), judge=judge)
    assert r.state == VerdictState.ERROR and "checker call failed" in r.reason


# ── conversation wiring ─────────────────────────────────────────────────

def _fake_conv():
    from syne.llm.provider import ChatResponse
    fake = SimpleNamespace(
        _mgr=SimpleNamespace(provider=SimpleNamespace(chat_model="default-model")),
        provider=SimpleNamespace(chat_model="chat-model"),
        _last_time_context="",
        _pending_consent_hash="",
        _llm_call=AsyncMock(),
        _handle_tool_calls=AsyncMock(),
    )
    return fake, ChatResponse(content="Draf jawaban yang cukup panjang.", model="m")


async def _run(monkeypatch, cfg, judge=None):
    async def gc(key, default=None):
        return cfg.get(key, default)

    monkeypatch.setattr(DBM, "get_rules", AsyncMock(return_value=RULES))
    monkeypatch.setattr(DBM, "get_config", gc)
    rc = AsyncMock(return_value=CheckResult(VerdictState.CLEAN))
    monkeypatch.setattr(RC, "check_response", rc)
    bj = AsyncMock(return_value=judge)
    monkeypatch.setattr(JM, "build_judge", bj)

    from syne.conversation import Conversation
    fake, resp = _fake_conv()
    out = await Conversation._rule_check_and_maybe_regenerate(
        fake, response=resp, context=[], tool_schemas=None, access_level="owner",
        chat_kwargs={}, user_message="pesan",
    )
    assert out is resp
    return rc.call_args.kwargs, bj, fake


async def test_default_driver_is_provider_with_default_model(monkeypatch):
    kw, bj, fake = await _run(monkeypatch, {})
    assert kw["evaluator_driver"] == "provider"
    assert kw["provider"] is fake._mgr.provider and kw["judge"] is None
    bj.assert_not_called()


async def test_picked_model_is_passed_as_judge(monkeypatch):
    judge = AsyncMock(return_value="CLEAN")
    judge.label = "google_cca:gemini-3.8-flash"
    kw, bj, _ = await _run(monkeypatch, {"security.rule_checker_driver": "provider",
                                         "security.rule_checker_model": " flash "}, judge)
    assert bj.call_args.args[0] == "flash"
    assert kw["judge"] is judge


async def test_unavailable_model_is_not_substituted(monkeypatch):
    kw, bj, _ = await _run(monkeypatch, {"security.rule_checker_driver": "provider",
                                         "security.rule_checker_model": "gone"}, None)
    judge = kw["judge"]
    assert judge is not None and judge.label == "unavailable:gone"
    try:
        await judge("p", 5.0)
    except RuntimeError as e:
        assert "gone" in str(e)
    else:
        raise AssertionError("unavailable judge must raise")


async def test_evaluator_driver_ignores_model_key(monkeypatch):
    kw, bj, _ = await _run(monkeypatch, {"security.rule_checker_driver": "evaluator",
                                         "security.rule_checker_model": "flash",
                                         "memory.evaluator_driver": "ollama"})
    assert kw["evaluator_driver"] == "ollama" and kw["judge"] is None
    bj.assert_not_called()


# ── config seeding ──────────────────────────────────────────────────────

async def test_migration_47_adds_key_and_keeps_driver_value():
    from syne.db import migrations as M
    assert M.MIGRATIONS[-1][0] == 47 and M.MIGRATIONS[-1][1] is M._m47_rule_checker_model
    assert M.CURRENT_SCHEMA_VERSION == 47
    conn = MagicMock()
    conn.execute = AsyncMock()
    await M._m47_rule_checker_model(conn)
    sqls = [c.args[0] for c in conn.execute.call_args_list]
    assert "ON CONFLICT (key) DO NOTHING" in sqls[0]
    assert conn.execute.call_args_list[0].args[1] == "security.rule_checker_model"
    assert all("SET value" not in s for s in sqls)


def test_schema_sql_defaults():
    sql = (Path(RC.__file__).parent / "db" / "schema.sql").read_text()
    assert "('security.rule_checker_model', '\"\"'," in sql
    assert sql.count("('security.rule_checker_driver', '\"provider\"',") == 2
    assert "('security.rule_checker_driver', '\"evaluator\"'" not in sql


def test_checker_menu_wired():
    src = (Path(RC.__file__).parent / "communication" / "telegram.py").read_text()
    assert 'data.startswith("checker_set:")' in src
    assert "self._handle_checker_callback(query, data)" in src
    assert 'f"m:{key}"' in src
    assert 'set_config("security.rule_checker_model", model_key)' in src
    assert 'BotCommand("checker",' in src
