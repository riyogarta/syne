"""Embedding health: refuse chat turns while the embedding service is down.

Recall needs a query vector. When the embedding service (normally local
Ollama) does not answer, a turn used to hang for minutes in retries and then
fail with a raw error. Running the turn without memory is worse: the answer
sounds normal but ignores stored decisions. Owner decision (Riyo, 17 Nov
2026): while embedding is down NO turn is processed.

- The recall embed is one bounded attempt (embedding.recall_timeout,
  default 10s, no retry).
- A refused connection or an HTTP error means down at once. A timeout or a
  429 refuses only that turn (no owner DM, no 60s window); a second
  consecutive one means down. Success resets the count.
- On failure the state is "down" for DOWN_TTL seconds: every turn, of every
  sender on every channel, is refused at once with a short English notice.
  Groups get the notice once per incident, then silence.
- After DOWN_TTL the next turn probes again; success clears the state.
- The owner gets one Telegram DM per incident with the real error and, when
  the embedding driver is Ollama, a "Restart Ollama" button.
- restart_ollama() backs /restartollama and that button.
"""

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional

from .llm.provider import EmbeddingUnavailableError

logger = logging.getLogger("syne.embed_health")

DOWN_TTL = 60.0
DEFAULT_RECALL_TIMEOUT = 10.0
SYSTEMCTL = "/usr/bin/systemctl"
SOFT_FAILS_TO_DOWN = 2
# Two soft failures count as consecutive only when the second embed
# started after the first failed (not two turns waiting side by side on
# the same slow spell) and within this many seconds (v1.23.14).
SOFT_FAIL_WINDOW = 600.0


def sudoers_hint() -> str:
    """One-time sudoers setup for the user Syne actually runs as (not
    necessarily a user named 'syne')."""
    import getpass
    try:
        user = getpass.getuser()
    except Exception:
        user = "syne"
    return (
        f"echo '{user} ALL=(root) NOPASSWD: {SYSTEMCTL} restart ollama' "
        "| sudo tee /etc/sudoers.d/syne-ollama && "
        "sudo chmod 440 /etc/sudoers.d/syne-ollama && sudo visudo -c"
    )

_since: Optional[float] = None      # wall clock of the first failure
_checked: float = 0.0               # monotonic time of the last failed probe
_error: str = ""
_driver: str = ""
_incident: int = 0
_notified_incident: int = 0
_groups_told: set = set()
_soft_fails: int = 0                # consecutive timeouts / 429 while up
_soft_fail_at: float = 0.0          # monotonic time of the last one


def is_flagged() -> bool:
    return _since is not None


def within_ttl() -> bool:
    return _since is not None and (time.monotonic() - _checked) < DOWN_TTL


def last_error() -> str:
    return _error


async def active_embedding_entry() -> dict:
    """The /embedding entry in use, or {} when none is configured."""
    from .db.models import get_config
    from .llm.drivers import get_model_from_list
    models = await get_config("provider.embedding_models", None) or []
    key = await get_config("provider.active_embedding", None)
    if not models or not key:
        return {}
    return get_model_from_list(models, key) or {}


async def recall_timeout() -> float:
    from .db.models import get_config
    try:
        v = float(await get_config("embedding.recall_timeout", DEFAULT_RECALL_TIMEOUT))
    except (TypeError, ValueError):
        v = DEFAULT_RECALL_TIMEOUT
    return v if v > 0 else DEFAULT_RECALL_TIMEOUT


def _describe(err: BaseException, timeout: Optional[float] = None) -> str:
    if isinstance(err, asyncio.TimeoutError):
        return f"no answer within {timeout:.0f}s" if timeout else "timed out"
    text = str(err).strip()
    return f"{type(err).__name__}: {text}" if text else type(err).__name__


async def mark_down(err: BaseException, timeout: Optional[float] = None) -> bool:
    """Record a failure. Returns True when this starts a new incident."""
    global _since, _checked, _error, _driver, _incident
    try:
        driver = (await active_embedding_entry()).get("driver", "") or ""
    except Exception:
        driver = ""
    new = _since is None
    if new:
        _since = time.time()
        _incident += 1
        _groups_told.clear()
    _checked = time.monotonic()
    _error = _describe(err, timeout)
    _driver = driver
    if new:
        logger.error(f"Embedding DOWN ({driver or 'unknown driver'}): {_error} - refusing turns")
    else:
        logger.warning(f"Embedding still down: {_error}")
    return new


def mark_up() -> bool:
    """Clear the down state. Returns True when it was down."""
    global _since, _error
    if _since is None:
        return False
    logger.info(f"Embedding recovered after {time.time() - _since:.0f}s")
    _since = None
    _error = ""
    _groups_told.clear()
    return True


def user_notice() -> str:
    svc = "Embedding service (Ollama)" if _driver == "ollama" else "Embedding service"
    return (
        f"⚠️ {svc} is not responding — your message was not processed. "
        "Please try again in a moment."
    )


def should_reply(is_group: bool, platform: str, chat_id) -> bool:
    """DMs always get the notice; a group only once per incident."""
    if not is_group:
        return True
    key = f"{platform}:{chat_id}"
    if key in _groups_told:
        return False
    _groups_told.add(key)
    return True


async def gate(provider) -> None:
    """Called before every turn. Healthy: no cost. Down and probed less
    than DOWN_TTL ago: refuse at once. Down and older: probe again (a
    1-word message would otherwise never embed and never notice recovery).
    """
    if _since is None:
        return
    if within_ttl():
        raise EmbeddingUnavailableError(_error)
    await quick_embed(provider, "health check")


def is_soft_failure(err: BaseException) -> bool:
    """Timeout or rate limit: the service may just be busy (cold model
    load, a long background embed, a 429), not dead."""
    import re
    if isinstance(err, (asyncio.TimeoutError, TimeoutError)):
        return True
    try:
        import httpx
        if isinstance(err, httpx.TimeoutException):
            return True
    except ImportError:
        pass
    status = getattr(err, "status_code", None) or getattr(
        getattr(err, "response", None), "status_code", None
    )
    if status == 429:
        return True
    text = str(err).lower()
    return bool(re.search(r"\b429\b", text)) or "too many requests" in text or "rate limit" in text


async def quick_embed(provider, text: str) -> list:
    """Embed for recall, bounded and without retry.

    Success clears a down state. Failure marks it, notifies the owner once
    per incident and raises EmbeddingUnavailableError.
    """
    timeout = await recall_timeout()
    t_start = time.monotonic()
    try:
        resp = await provider.embed_fast(text, timeout=timeout)
        vector = getattr(resp, "vector", None)
        if not vector:
            raise RuntimeError("embedding provider returned an empty vector")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        global _soft_fails, _soft_fail_at
        soft = _since is None and is_soft_failure(e)
        if soft and _soft_fails and time.monotonic() - _soft_fail_at > SOFT_FAIL_WINDOW:
            _soft_fails = 0
        # Started before the last soft failure was recorded: it waited on
        # the same slow spell, so it is not a second sign of a dead service.
        overlapped = soft and _soft_fails > 0 and t_start < _soft_fail_at
        if soft and (overlapped or _soft_fails < SOFT_FAILS_TO_DOWN - 1):
            # Busy, not dead: refuse only this turn, no incident yet.
            if not overlapped:
                _soft_fails += 1
                _soft_fail_at = time.monotonic()
            msg = _describe(e, timeout)
            logger.warning(
                f"Embedding slow/limited: {msg} - turn refused, not marked down "
                f"({_soft_fails}/{SOFT_FAILS_TO_DOWN})"
            )
            raise EmbeddingUnavailableError(msg) from e
        _soft_fails = 0
        await mark_down(e, timeout)
        schedule_owner_notice()
        raise EmbeddingUnavailableError(_error) from e
    if _soft_fails:
        _soft_fails = 0
        _groups_told.clear()
    mark_up()
    return vector


def schedule_owner_notice() -> None:
    try:
        asyncio.get_running_loop().create_task(_notify_owner())
    except RuntimeError:
        pass


async def _notify_owner() -> None:
    global _notified_incident
    if _since is None or _notified_incident == _incident:
        return
    _notified_incident = _incident
    try:
        from .communication.registry import get_channel, CHANNEL_TELEGRAM
        ch = get_channel(CHANNEL_TELEGRAM)
        if ch is None or not hasattr(ch, "notify_embedding_down"):
            logger.warning("Embedding down: no Telegram channel to notify the owner")
            return
        ollama = _driver == "ollama"
        text = (
            "⚠️ Embedding is down — incoming messages are being refused.\n\n"
            f"Error: {_error}\n\n"
            + ("Tap the button below or send /restartollama."
               if ollama else "Check the embedding model in /embedding.")
        )
        await ch.notify_embedding_down(text, with_restart_button=ollama)
    except Exception as e:
        logger.error(f"Embedding-down owner notice failed: {e}", exc_info=True)


# ── /restartollama ──────────────────────────────────────────────────────

ProgressFn = Optional[Callable[[str], Awaitable[None]]]


def _installed(model: str, names: set) -> bool:
    return model in names or (":" not in model and f"{model}:latest" in names)


async def _pull(base: str, model: str, say) -> str:
    """Pull ``model``; returns "" on success, else the error text."""
    import httpx
    last = 0.0
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=600.0)) as client:
            async with client.stream(
                "POST", f"{base}/api/pull", json={"model": model, "stream": True},
            ) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")[:300]
                    return f"HTTP {r.status_code}: {body}"
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if ev.get("error"):
                        return str(ev["error"])
                    status = ev.get("status", "")
                    if status == "success":
                        return ""
                    now = time.monotonic()
                    if now - last >= 3:
                        last = now
                        total, done = ev.get("total"), ev.get("completed")
                        pct = f" {done * 100 // total}%" if total and done else ""
                        await say(f"⬇️ Pulling {model}: {status}{pct}")
    except Exception as e:
        return _describe(e)
    return "pull ended without a success status"


async def restart_ollama(progress: ProgressFn = None) -> str:
    """Restart the ollama service, pull the embedding model only if it is
    missing, then run one test embed. Returns the final report line."""
    import httpx

    async def say(t: str) -> None:
        if progress:
            try:
                await progress(t)
            except Exception:
                pass

    entry = await active_embedding_entry()
    if entry.get("driver") != "ollama":
        return "⚠️ The active embedding model does not use Ollama — nothing to restart."
    base = (entry.get("base_url") or "http://localhost:11434").rstrip("/")
    model = entry.get("model_id") or ""

    await say("🔄 Restarting Ollama...")
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "-n", SYSTEMCTL, "restart", "ollama",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), 90)
    except asyncio.TimeoutError:
        return "❌ systemctl restart ollama did not finish within 90s."
    except Exception as e:
        return f"❌ Could not run sudo: {_describe(e)}"
    if proc.returncode != 0:
        msg = (err or out or b"").decode(errors="replace").strip()[:500]
        return (
            f"❌ Restart failed (exit {proc.returncode}): {msg}\n\n"
            f"One-time setup on the server:\n{sudoers_hint()}"
        )

    await say("⏳ Ollama restarted, waiting for it to answer...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            deadline = time.monotonic() + 30
            while True:
                try:
                    r = await client.get(f"{base}/api/version")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    return f"❌ Ollama was restarted but does not answer at {base} after 30s."
                await asyncio.sleep(1)
            r = await client.get(f"{base}/api/tags")
            names = {m.get("name", "") for m in (r.json().get("models") or [])}
    except Exception as e:
        return f"❌ Could not read the installed models: {_describe(e)}"

    pulled = False
    if not _installed(model, names):
        await say(f"⬇️ {model} is not installed — pulling it...")
        perr = await _pull(base, model, say)
        if perr:
            return f"❌ Pull of {model} failed: {perr}"
        pulled = True

    await say("🧪 Testing one embed...")
    try:
        from .llm.ollama import OllamaProvider
        resp = await OllamaProvider(embedding_model=model, base_url=base).embed_fast(
            "health check", timeout=120,
        )
        if not getattr(resp, "vector", None):
            raise RuntimeError("empty vector")
    except Exception as e:
        await mark_down(e, 120)
        return f"❌ Ollama restarted, but the test embed failed: {_describe(e, 120)}"
    mark_up()
    return f"✅ Ollama restarted{' (model pulled)' if pulled else ''}, embedding OK."
