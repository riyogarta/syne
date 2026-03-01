"""WhatsApp Ability — text-only DM bridge via wacli.

Alternative communication channel. No slash commands, no groups.
All inbound messages go directly to agent.handle_message().

Setup: tell Syne via Telegram to "enable whatsapp" — Syne sets
whatsapp.enabled=true in DB, restarts, and the bridge auto-starts.

Requires: wacli binary installed and authenticated (`wacli auth`).
"""

import asyncio
import json
import logging
import os
import platform
import shutil
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Optional

from .base import Ability
from ..communication.inbound import InboundContext
from ..communication.outbound import process_outbound, split_message

logger = logging.getLogger("syne.whatsapp")

# wacli install config
_WACLI_REPO = "steipete/wacli"
_WACLI_INSTALL_DIR = os.path.expanduser("~/.local/bin")

# Module-level reference so main.py can access the running bridge
_bridge_instance: Optional["WhatsAppAbility"] = None


class WhatsAppAbility(Ability):
    """WhatsApp bridge using wacli subprocess."""

    name = "whatsapp"
    description = "WhatsApp text messaging bridge via wacli"
    version = "1.0"
    priority = False  # Not a pre-processing ability

    def __init__(self):
        self._agent = None
        self._wacli_path = "wacli"
        self._wacli_db: Optional[str] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._last_rowid: int = 0
        self._running = False
        self._send_lock = asyncio.Lock()
        self._allowed_chat_jids = set()  # if non-empty, only reply to these chat JIDs
        self._allowlist_name_by_jid = {}  # jid -> friendly name (for greetings/logs)
        self._allowlist_model_by_jid = {}  # jid -> model key override (None = default)

    # ── Dependencies ────────────────────────────────────────────

    async def _resolve_wacli(self) -> Optional[str]:
        """Find wacli binary — check PATH, DB config, and Go binary dirs.

        Returns full path if found, None otherwise.
        """
        # 1. PATH
        found = shutil.which("wacli")
        if found:
            return found

        # 2. DB config (previously saved full path)
        try:
            from ..db.models import get_config
            saved = await get_config("whatsapp.wacli_path")
            if saved and os.path.isfile(saved) and os.access(saved, os.X_OK):
                return saved
        except Exception:
            pass

        # 3. Dynamically resolve Go binary path
        gopaths = set()

        # $GOPATH env var
        env_gopath = os.environ.get("GOPATH", "").strip()
        if env_gopath:
            gopaths.add(env_gopath)

        # Ask `go env GOPATH` (if go is in PATH)
        if shutil.which("go"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "go", "env", "GOPATH",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                val = out.decode().strip()
                if val:
                    gopaths.add(val)
            except Exception:
                pass

        # Go default: ~/go (used when GOPATH is not set at all)
        gopaths.add(os.path.expanduser("~/go"))

        candidates = [os.path.join(_WACLI_INSTALL_DIR, "wacli")]
        for gp in gopaths:
            candidates.append(os.path.join(gp, "bin", "wacli"))

        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        return None

    async def _save_wacli_path(self, path: str):
        """Persist resolved wacli path to DB so start_bridge() can find it."""
        try:
            from ..db.models import set_config
            await set_config("whatsapp.wacli_path", path)
            logger.info(f"Saved wacli path to DB: {path}")
        except Exception as e:
            logger.warning(f"Failed to save wacli path to DB: {e}")

    async def ensure_dependencies(self) -> tuple[bool, str]:
        """Check for wacli binary; install if missing.

        Install strategy (no sudo — never escalate privileges):
        1. Resolve via PATH, DB config, or Go binary dirs
        2. go install (if Go available)
        3. macOS only: download pre-built binary from GitHub releases
        4. Fail with instructions for the owner
        """
        found = await self._resolve_wacli()
        if found:
            self._wacli_path = found
            await self._save_wacli_path(found)
            return True, ""

        logger.info("wacli not found, attempting to install...")

        # 1. Try go install (works on any platform if Go is available)
        if shutil.which("go"):
            ok, msg = await self._install_via_go()
            if ok:
                return True, msg

        # 2. macOS: try pre-built binary from GitHub releases
        if platform.system().lower() == "darwin":
            ok, msg = await self._install_from_release()
            if ok:
                return True, msg

        # Can't auto-install — tell the owner what to do
        if not shutil.which("go"):
            return False, (
                "wacli requires Go to build from source (no Linux binary available).\n"
                "Owner harus install dulu di server:\n"
                "  sudo apt install -y golang-go\n"
                "Setelah itu retry enable WhatsApp — wacli akan di-build otomatis."
            )

        return False, (
            "wacli build failed. Owner harus install manual di server:\n"
            "  go install github.com/steipete/wacli@latest\n"
            "  wacli auth  (scan QR code)\n"
            "Setelah itu retry enable WhatsApp."
        )

    async def _install_via_go(self) -> tuple[bool, str]:
        """Try installing wacli via 'go install'."""
        if not shutil.which("go"):
            return False, "Go not available"

        try:
            os.makedirs(_WACLI_INSTALL_DIR, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                "go", "install", f"github.com/{_WACLI_REPO}@latest",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "GOBIN": _WACLI_INSTALL_DIR},
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
            except asyncio.TimeoutError:
                return False, "go install timed out after 30 minutes. Try manually:\n  go install github.com/steipete/wacli@latest"
            if proc.returncode == 0:
                wacli_path = os.path.join(_WACLI_INSTALL_DIR, "wacli")
                if os.path.isfile(wacli_path):
                    self._wacli_path = wacli_path
                    await self._save_wacli_path(wacli_path)
                    logger.info(f"wacli installed via go install → {wacli_path}")
                    return True, f"wacli installed to {wacli_path}"
            logger.warning(f"go install failed: {stderr.decode()[:300]}")
            return False, f"go install failed: {stderr.decode()[:200]}"
        except Exception as e:
            return False, f"go install error: {e}"

    async def _install_from_release(self) -> tuple[bool, str]:
        """macOS: download pre-built wacli binary from GitHub releases."""
        import urllib.request
        import tarfile

        # Only macOS has pre-built binaries
        asset_name = "wacli-macos-universal.tar.gz"

        api_url = f"https://api.github.com/repos/{_WACLI_REPO}/releases/latest"
        try:
            req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                release = json.loads(resp.read())

            download_url = None
            for asset in release.get("assets", []):
                if asset["name"] == asset_name:
                    download_url = asset["browser_download_url"]
                    break

            if not download_url:
                return False, f"No release asset '{asset_name}' found"

        except Exception as e:
            return False, f"Failed to query GitHub releases: {e}"

        try:
            os.makedirs(_WACLI_INSTALL_DIR, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = os.path.join(tmpdir, asset_name)
                logger.info(f"Downloading wacli from {download_url}")
                urllib.request.urlretrieve(download_url, archive_path)

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(tmpdir)

                extracted = Path(tmpdir)
                for candidate in extracted.rglob("wacli"):
                    if candidate.is_file():
                        dest = os.path.join(_WACLI_INSTALL_DIR, "wacli")
                        shutil.copy2(str(candidate), dest)
                        os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                        self._wacli_path = dest
                        await self._save_wacli_path(dest)
                        logger.info(f"wacli installed from release → {dest}")
                        return True, f"wacli downloaded and installed to {dest}"

            return False, "wacli binary not found in release archive"
        except Exception as e:
            return False, f"Failed to download/install wacli: {e}"

    # ── Ability interface ──────────────────────────────────────

    async def execute(self, params: dict, context: dict) -> dict:
        """Send a WhatsApp message (LLM tool call)."""
        to = params.get("to", "").strip()
        message = params.get("message", "").strip()

        if not to or not message:
            return {"success": False, "error": "Both 'to' and 'message' are required."}

        if not self._running:
            return {"success": False, "error": "WhatsApp bridge is not running."}

        # Normalize JID
        if not to.endswith("@s.whatsapp.net"):
            to = to.replace("+", "").replace(" ", "").replace("-", "")
            to = f"{to}@s.whatsapp.net"

        try:
            await self._send_text(to, message)
        except Exception as e:
            return {"success": False, "error": str(e)}
        return {"success": True, "result": f"Message sent to {to}."}

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "whatsapp",
                "description": "Send a WhatsApp text message to a phone number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Phone number (e.g. 6281234567890) or JID",
                        },
                        "message": {
                            "type": "string",
                            "description": "Text message to send",
                        },
                    },
                    "required": ["to", "message"],
                },
            },
        }

    def get_guide(self, enabled: bool, config: dict) -> str:
        if self._running:
            return (
                "- Status: **active** (wacli bridge running)\n"
                "- DM-only, text-only — no commands, no groups\n"
                "- Inbound messages handled same as Telegram DMs\n"
                "- Outbound: `whatsapp(to='6281234567890', message='...')`"
            )
        if enabled:
            return (
                "- Status: **enabled** (bridge starts on next restart)\n"
                "- DM-only, text-only — no commands, no groups\n"
                "- Requires: `wacli` binary installed + authenticated (`wacli auth`)"
            )
        return (
            "- Status: **disabled**\n"
            "- To enable: `update_ability(action='enable', name='whatsapp')`\n"
            "- Requires: `wacli` binary installed + authenticated (`wacli auth` for QR scan)\n"
            "- Dependencies auto-installed when enabled"
        )

    def get_required_config(self) -> list[str]:
        return []

    async def _start_sync(self) -> bool:
        """Start the long-running `wacli sync --follow` process.

        This keeps the wacli WebSocket connection alive and syncs new
        messages into the local SQLite DB.  Inbound messages are picked
        up by the separate _poll_loop (DB poller).
        """
        # Make sure old process is not running
        await self._stop_sync()

        try:
            self._process = await asyncio.create_subprocess_exec(
                self._wacli_path, 'sync', '--follow',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(f"Failed to start wacli sync: {e}")
            self._process = None
            return False

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        return True

    async def _stop_sync(self):
        """Stop the sync process (if any)."""
        # Stop monitor task first so it won't auto-restart on exit.
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._monitor_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
            self._process = None

    # ── Bridge lifecycle ───────────────────────────────────────

    async def start_bridge(self, agent, wacli_path: str = "wacli"):
        """Start the wacli background bridge.

        Called from main.py when whatsapp.enabled is true.
        """
        global _bridge_instance
        self._agent = agent

        # Resolve wacli — uses DB config, PATH, and Go binary dirs
        resolved = await self._resolve_wacli()
        if resolved:
            self._wacli_path = resolved
        else:
            logger.error(
                "wacli binary not found. Install: go install github.com/steipete/wacli@latest"
            )
            return False

        self._running = True

        # Load allowlist (optional): if set, Syne will only reply to these chat JIDs.
        # Config value supports:
        #   1) list[str]                 (legacy)
        #   2) list[{'jid': str, 'name': str}] (recommended)
        #   3) comma-separated string    (legacy)
        try:
            from ..db.models import get_config
            allowed = await get_config('whatsapp.allowed_chat_jids', default=[])

            self._allowed_chat_jids = set()
            self._allowlist_name_by_jid = {}
            self._allowlist_model_by_jid = {}

            # Normalize legacy string formats
            if isinstance(allowed, str):
                try:
                    allowed = json.loads(allowed)
                except Exception:
                    allowed = [x.strip() for x in allowed.split(',') if x.strip()]

            if isinstance(allowed, list):
                for item in allowed:
                    if isinstance(item, dict):
                        jid = str(item.get('jid') or '').strip()
                        name = str(item.get('name') or '').strip()
                        model = item.get('model') or None
                        if jid:
                            self._allowed_chat_jids.add(jid)
                            if name:
                                self._allowlist_name_by_jid[jid] = name
                            if model:
                                self._allowlist_model_by_jid[jid] = model
                    else:
                        jid = str(item).strip()
                        if jid:
                            self._allowed_chat_jids.add(jid)
            elif allowed:
                # Any other scalar type
                self._allowed_chat_jids = {str(allowed).strip()}

            if self._allowed_chat_jids:
                logger.info(f'WhatsApp allowlist enabled: {sorted(self._allowed_chat_jids)}')
        except Exception as e:
            logger.warning(f'Failed to load WhatsApp allowlist: {e}')

        # Resolve wacli DB path for the inbound poller
        self._wacli_db = self._resolve_wacli_db()
        if not self._wacli_db:
            logger.error("wacli database not found — inbound messages will not work")

        # Wire callbacks so sub-agent results and status messages reach WhatsApp
        if self._agent.conversations:
            self._agent.conversations.add_delivery_callback(self._deliver_subagent_result)
            self._agent.conversations.add_status_callback(self._send_status_message)

        ok = await self._start_sync()
        if not ok:
            self._running = False
            return False

        # Start DB poller for inbound messages (independent of sync process)
        if self._wacli_db:
            self._poll_task = asyncio.create_task(self._poll_loop())

        _bridge_instance = self
        logger.info("WhatsApp bridge started.")
        return True

    async def stop(self):
        """Stop the WhatsApp bridge."""
        global _bridge_instance
        self._running = False

        # Unregister callbacks
        if self._agent and self._agent.conversations:
            self._agent.conversations.remove_delivery_callback(self._deliver_subagent_result)
            self._agent.conversations.remove_status_callback(self._send_status_message)

        # Stop poller
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None

        await self._stop_sync()
        _bridge_instance = None
        logger.info("WhatsApp bridge stopped.")

    # ── Process monitor ──────────────────────────────────────

    async def _monitor_loop(self):
        """Watch wacli sync process — restart if it dies unexpectedly."""
        try:
            if self._process:
                await self._process.wait()
            if self._running:
                logger.warning("wacli process ended unexpectedly, restarting...")
                await self._restart_wacli()
        except asyncio.CancelledError:
            pass

    # ── Inbound DB poller ─────────────────────────────────────

    def _resolve_wacli_db(self) -> Optional[str]:
        """Find the wacli SQLite database file.

        wacli stores its DB at ~/.wacli/wacli.db relative to the user
        running the process.
        """
        default = os.path.expanduser("~/.wacli/wacli.db")
        if os.path.isfile(default):
            return default
        return None

    async def _poll_loop(self):
        """Poll wacli SQLite DB for new inbound messages.

        Runs independently of the sync process.  SQLite read is safe
        even while wacli sync writes (WAL mode / shared cache).
        """
        db_path = self._wacli_db
        if not db_path:
            return

        # Seed last_rowid to current max so we only process NEW messages
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            cur = conn.execute("SELECT MAX(rowid) FROM messages")
            self._last_rowid = cur.fetchone()[0] or 0
            conn.close()
            logger.info(f"WhatsApp poller started (last_rowid={self._last_rowid}, db={db_path})")
        except Exception as e:
            logger.error(f"Failed to read wacli DB: {e}")
            return

        while self._running:
            try:
                await asyncio.sleep(2)
                if not self._running:
                    break

                # Read new inbound messages (from_me=0, text not empty, DM only)
                conn = sqlite3.connect(db_path, timeout=5)
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT rowid, chat_jid, sender_jid, sender_name, chat_name, text, from_me
                    FROM messages
                    WHERE rowid > ?
                      AND text IS NOT NULL AND text != ''
                      AND chat_jid NOT LIKE '%@g.us'
                      AND chat_jid != 'status@broadcast'
                    ORDER BY rowid ASC
                """, (self._last_rowid,))
                rows = cur.fetchall()
                conn.close()

                for row in rows:
                    self._last_rowid = row["rowid"]
                    msg = {
                        "ChatJID": row["chat_jid"],
                        "SenderJID": row["sender_jid"] or row["chat_jid"],
                        "PushName": row["sender_name"] or "",
                        "ChatName": row["chat_name"] or "",
                        "Text": row["text"],
                        "FromMe": bool(row["from_me"]),
                    }
                    await self._handle_inbound(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WhatsApp poll loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _handle_inbound(self, msg: dict):
        """Process a single inbound wacli JSON message."""
        from_me = bool(msg.get("FromMe", False))

        text = msg.get("Text", "").strip()
        if not text:
            return

        chat_jid = msg.get("ChatJID", "")

        # Special-case: when wacli is logged into the same WhatsApp account as the user,
        # messages typed on the phone show up in the DB as from_me=1 (because they are
        # sent by the logged-in account). We still want to treat *self-chat* messages
        # as inbound so Syne can reply.
        #
        # Accept from_me=1 only when it's the self-chat (sender base JID == chat_jid).
        sender_jid_raw = msg.get("SenderJID") or ""
        if from_me:
            if not sender_jid_raw:
                # Likely sent by wacli itself (sender_jid NULL)
                return
            sender_base = sender_jid_raw.split(":", 1)[0]
            if sender_base != chat_jid:
                # User sending messages to other contacts — do not auto-reply
                return

        # DM only — skip groups
        if chat_jid.endswith("@g.us"):
            return

        # Allowlist: if configured, only reply to specific chat JIDs.
        # This prevents accidental replies to other contacts on the same WhatsApp account.
        if self._allowed_chat_jids:
            chat_base = chat_jid.split(':', 1)[0] if chat_jid else ''
            if chat_jid not in self._allowed_chat_jids and chat_base not in self._allowed_chat_jids:
                logger.info(f'Ignoring WhatsApp inbound from non-allowlisted chat: {chat_jid}')
                return

        sender_jid = sender_jid_raw or chat_jid
        # Friendly name override from allowlist (stable naming)
        chat_base = chat_jid.split(':', 1)[0] if chat_jid else ''
        name_override = (
            self._allowlist_name_by_jid.get(chat_jid)
            or (self._allowlist_name_by_jid.get(chat_base) if chat_base else None)
        )
        # Normalize device JID (e.g. 628xxx:51@s.whatsapp.net) to base number for user id
        sender_platform_id = sender_jid.split(":", 1)[0] if ":" in sender_jid else sender_jid
        chat_name = msg.get("ChatName", "")
        sender_name = name_override or msg.get("PushName", chat_name or sender_jid.split("@")[0])

        logger.info(f"[whatsapp] {sender_name}: {text[:100]}")

        inbound = InboundContext(
            channel="whatsapp",
            platform="whatsapp",
            chat_type="direct",
            conversation_label=sender_name,
            chat_id=chat_jid,
            sender_name=sender_name,
            sender_id=sender_platform_id,
        )

        # User context prefix (core utility — adds sender info for LLM)
        from ..communication.inbound import build_user_context_prefix
        original_text = text
        user_prefix = build_user_context_prefix(inbound)
        if user_prefix:
            text = f"{user_prefix}\n\n{text}"

        # Resolve per-member model override from allowlist
        model_override = (
            self._allowlist_model_by_jid.get(chat_jid)
            or self._allowlist_model_by_jid.get(chat_base)
        )

        metadata = {
            "chat_id": chat_jid,
            "inbound": inbound,
            "original_text": original_text,
            "wa_model_override": model_override,  # None = use default
        }

        try:
            response = await self._agent.handle_message(
                platform="whatsapp",
                chat_id=chat_jid,
                user_name=sender_name,
                user_platform_id=sender_platform_id,
                message=text,
                display_name=sender_name,
                message_metadata=metadata,
            )

            # None = lock timeout (another request in progress) — silently drop
            if response is None:
                return
            if not response:
                response = "LLM returned an empty response. Please try again."

            await self._send_text(chat_jid, response)
        except Exception as e:
            logger.error(f"Error handling WhatsApp message: {e}", exc_info=True)
            from ..communication.errors import classify_error
            await self._send_text(chat_jid, classify_error(e))

    # ── Outbound ───────────────────────────────────────────────

    async def _send_text(self, jid: str, text: str):
        """Send a text message via wacli.

        `wacli sync --follow` holds an exclusive lock on the store. Running
        `wacli send ...` concurrently fails with "store is locked".

        Workaround: pause sync, send, then resume sync.
        """
        text = process_outbound(text)
        if not text:
            return

        async with self._send_lock:
            was_syncing = self._process is not None
            if was_syncing:
                await self._stop_sync()

            try:
                for chunk in split_message(text, max_length=4096):
                    proc = await asyncio.create_subprocess_exec(
                        self._wacli_path, 'send', 'text',
                        '--to', jid,
                        '--message', chunk,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                    if proc.returncode != 0:
                        err = stderr.decode('utf-8', errors='replace') if stderr else ''
                        raise RuntimeError(f"wacli send failed (rc={proc.returncode}): {err[:200]}")
                    await asyncio.sleep(0.3)
            finally:
                if self._running and was_syncing:
                    ok = await self._start_sync()
                    if not ok:
                        logger.error('Failed to restart wacli sync after sending message')

    # ── Sub-agent callbacks ────────────────────────────────────

    async def _deliver_subagent_result(self, message: str, parent_session_id: int):
        """Deliver sub-agent results to the WhatsApp chat that spawned it."""
        for key, conv in self._agent.conversations._active.items():
            if conv.session_id == parent_session_id:
                parts = key.split(":", 1)
                if len(parts) == 2 and parts[0] == "whatsapp":
                    await self._send_text(parts[1], message)
                    return

    async def _send_status_message(self, session_id: int, message: str):
        """Send a status notification to the WhatsApp chat for a session."""
        for key, conv in self._agent.conversations._active.items():
            if conv.session_id == session_id:
                parts = key.split(":", 1)
                if len(parts) == 2 and parts[0] == "whatsapp":
                    await self._send_text(parts[1], message)
                    return

    # ── Internal ───────────────────────────────────────────────

    async def _restart_wacli(self):
        """Restart wacli process after unexpected termination."""
        await asyncio.sleep(5)
        if not self._running:
            return
        logger.info("Restarting wacli sync...")
        try:
            ok = await self._start_sync()
            if not ok:
                logger.error('Failed to restart wacli sync')
        except Exception as e:
            logger.error(f"Failed to restart wacli: {e}")
