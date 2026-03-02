"""WhatsApp Ability — text + image bridge via wacli.

Alternative communication channel supporting DMs and groups.
All inbound messages go directly to agent.handle_message().
Images are downloaded via wacli, base64-encoded, and routed
to ImageAnalysisAbility for vision processing.

Setup: tell Syne via Telegram to "enable whatsapp" — Syne sets
whatsapp.enabled=true in DB, restarts, and the bridge auto-starts.

Requires: wacli binary installed and authenticated (`wacli auth`).
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import stat
import tempfile
import time
from pathlib import Path
from typing import Optional

from .base import Ability
from ..communication.inbound import InboundContext
from ..communication.outbound import extract_media, process_outbound, split_message

logger = logging.getLogger("syne.whatsapp")

# wacli install config
_WACLI_REPO = "steipete/wacli"
_WACLI_INSTALL_DIR = os.path.expanduser("~/.local/bin")

# Reliability: dedup / echo / debounce
_DEDUP_TTL = 120        # 2 minutes
_ECHO_TTL = 20          # 20 seconds
_DEBOUNCE_DELAY = 2.0   # 2 seconds — batch rapid messages
_DEDUP_MAX = 5000       # max cache entries before prune

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
        self._session_db: Optional[str] = None
        self._lid_to_pn_cache = {}  # lid digits -> phone digits
        # Dedup / echo / debounce state
        self._seen_rowids: set[int] = set()
        self._seen_hashes: dict[str, float] = {}     # "chat_jid:md5" -> timestamp
        self._echo_hashes: dict[str, float] = {}     # "chat_jid:md5" -> timestamp
        self._debounce_buffers: dict[str, list] = {}  # chat_jid -> [msg, ...]
        self._debounce_tasks: dict[str, asyncio.Task] = {}
        self._last_prune: float = 0.0

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

        # Resolve session DB path (whatsmeow store; contains lid<->phone mapping)
        session_db = os.path.expanduser("~/.wacli/session.db")
        self._session_db = session_db if os.path.isfile(session_db) else None

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

                # Read new inbound messages (text or media)
                conn = sqlite3.connect(db_path, timeout=5)
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT rowid, chat_jid, sender_jid, sender_name, chat_name, text,
                           from_me, media_type, mime_type, media_caption, msg_id
                    FROM messages
                    WHERE rowid > ?
                      AND ((text IS NOT NULL AND text != '') OR media_type IS NOT NULL)
                      AND chat_jid != 'status@broadcast'
                    ORDER BY rowid ASC
                """, (self._last_rowid,))
                rows = cur.fetchall()
                conn.close()

                if rows:
                    logger.info(f'[whatsapp] poll: {len(rows)} new row(s) after rowid {self._last_rowid}')

                now = time.time()
                if now - self._last_prune > 60:
                    self._prune_caches()

                for row in rows:
                    rowid = row["rowid"]
                    self._last_rowid = rowid

                    # Dedup: skip already-seen rowids
                    if rowid in self._seen_rowids:
                        continue

                    text = (row["text"] or "").strip()
                    chat_jid = row["chat_jid"] or ""

                    # Dedup: skip duplicate content within TTL
                    if text and chat_jid:
                        h = self._content_hash(chat_jid, text)
                        if h in self._seen_hashes and (now - self._seen_hashes[h]) < _DEDUP_TTL:
                            self._seen_rowids.add(rowid)
                            continue
                        self._seen_hashes[h] = now

                    self._seen_rowids.add(rowid)

                    msg = {
                        "ChatJID": chat_jid,
                        "SenderJID": row["sender_jid"] or chat_jid,
                        "PushName": row["sender_name"] or "",
                        "ChatName": row["chat_name"] or "",
                        "Text": row["text"],
                        "FromMe": bool(row["from_me"]),
                        "MediaType": row["media_type"],
                        "MimeType": row["mime_type"],
                        "MediaCaption": row["media_caption"],
                        "MsgID": row["msg_id"],
                    }
                    self._enqueue_debounce(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WhatsApp poll loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    # ── Debounce ──────────────────────────────────────────────

    def _enqueue_debounce(self, msg: dict):
        """Buffer a message and (re)start a debounce timer for its chat."""
        chat_jid = msg.get("ChatJID", "")
        self._debounce_buffers.setdefault(chat_jid, []).append(msg)

        # Cancel existing timer for this chat
        existing = self._debounce_tasks.get(chat_jid)
        if existing and not existing.done():
            existing.cancel()

        self._debounce_tasks[chat_jid] = asyncio.create_task(self._debounce_fire(chat_jid))

    async def _debounce_fire(self, chat_jid: str):
        """Wait for debounce delay, then merge buffered messages and dispatch."""
        try:
            await asyncio.sleep(_DEBOUNCE_DELAY)
        except asyncio.CancelledError:
            return

        msgs = self._debounce_buffers.pop(chat_jid, [])
        self._debounce_tasks.pop(chat_jid, None)
        if not msgs:
            return

        try:
            if len(msgs) == 1:
                await self._handle_inbound(msgs[0])
            else:
                # Merge: combine texts, use first msg's metadata
                merged = dict(msgs[0])
                merged["Text"] = "\n".join((m.get("Text") or "").strip() for m in msgs if (m.get("Text") or "").strip())
                await self._handle_inbound(merged)
        except Exception as e:
            logger.error(f'[whatsapp] debounce dispatch error: {e}', exc_info=True)

    def _lookup_pn_from_lid(self, lid: str) -> str:
        """Map a LID (digits) to a phone number (digits) via session.db.

        Returns '' if unknown/unavailable. Uses an in-memory cache.
        """
        if not lid or (not lid.isdigit()) or (not self._session_db):
            return ''
        cached = self._lid_to_pn_cache.get(lid)
        if cached is not None:
            return cached or ''
        pn = ''
        try:
            con = sqlite3.connect(f'file:{self._session_db}?mode=ro', uri=True, timeout=1)
            cur = con.execute('SELECT pn FROM whatsmeow_lid_map WHERE lid=? LIMIT 1', (lid,))
            row = cur.fetchone()
            con.close()
            pn = row[0] if row and row[0] else ''
        except Exception:
            pn = ''
        self._lid_to_pn_cache[lid] = pn
        return pn

    # ── Member helpers (load / save / match) ────────────────────

    async def _wa_load_members(self) -> dict:
        """Load WA members from config. Returns {normalized_phone: entry}.

        Each entry: {'jid': str, 'name': str, 'model': str|None, 'blocked': bool}
        Legacy entries without 'blocked' default to False.
        """
        from ..db.models import get_config
        raw = await get_config('whatsapp.allowed_chat_jids', default=[])

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = [x.strip() for x in raw.split(',') if x.strip()]

        by_jid: dict[str, dict] = {}
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    jid = str(item.get('jid') or '').strip()
                    name = str(item.get('name') or '').strip()
                    model = item.get('model') or None
                    blocked = bool(item.get('blocked', False))
                    if jid:
                        by_jid[jid] = {'jid': jid, 'name': name, 'model': model, 'blocked': blocked}
                else:
                    jid = str(item).strip()
                    if jid:
                        by_jid[jid] = {'jid': jid, 'name': '', 'model': None, 'blocked': False}
        return by_jid

    async def _wa_save_members(self, by_jid: dict):
        """Save WA members to config."""
        from ..db.models import set_config
        await set_config('whatsapp.allowed_chat_jids', list(by_jid.values()))

    def _wa_match_jid(self, by_jid: dict, chat_jid: str) -> tuple:
        """Find member by chat_jid. Returns (key, entry) or (None, None).

        Tries: full JID, base JID (strip device), local phone number,
        and LID→phone mapping via session.db.
        """
        chat_base = chat_jid.split(':', 1)[0] if chat_jid else ''
        chat_local = chat_base.split('@', 1)[0] if '@' in chat_base else chat_base

        # Direct match: full JID, base JID, or phone number
        for candidate in (chat_jid, chat_base, chat_local):
            if candidate and candidate in by_jid:
                return candidate, by_jid[candidate]

        # LID→phone mapping
        if chat_base.endswith('@lid') and chat_local.isdigit() and self._session_db:
            pn = self._lookup_pn_from_lid(chat_local)
            if pn:
                for candidate in (pn, f"{pn}@s.whatsapp.net"):
                    if candidate in by_jid:
                        return candidate, by_jid[candidate]

        return None, None

    def _has_trigger_word(self, text: str, trigger: str) -> bool:
        if not trigger:
            return True
        try:
            return re.search(rf'\b{re.escape(trigger)}\b', text or '', re.IGNORECASE) is not None
        except Exception:
            return False

    def _strip_trigger_word(self, text: str, trigger: str) -> str:
        """Strip trigger word if it appears at the start (e.g. 'Molt:', 'molt,')."""
        if not trigger:
            return (text or '').strip()
        try:
            return re.sub(rf'^\s*{re.escape(trigger)}[,:]?\s*', '', text or '', flags=re.IGNORECASE).strip()
        except Exception:
            return (text or '').strip()

    def _content_hash(self, chat_jid: str, text: str) -> str:
        """Return a compact content hash for dedup/echo detection."""
        return f"{chat_jid}:{hashlib.md5(text.encode()).hexdigest()[:12]}"

    def _prune_caches(self):
        """Remove expired entries from dedup/echo caches."""
        now = time.time()
        self._seen_hashes = {k: v for k, v in self._seen_hashes.items() if (now - v) < _DEDUP_TTL}
        self._echo_hashes = {k: v for k, v in self._echo_hashes.items() if (now - v) < _ECHO_TTL}
        if len(self._seen_rowids) > _DEDUP_MAX:
            sorted_ids = sorted(self._seen_rowids)
            self._seen_rowids = set(sorted_ids[-_DEDUP_MAX:])
        self._last_prune = now

    async def _get_trigger_name(self) -> str:
        """Get the bot trigger name from config or identity (same as Telegram)."""
        from ..db.models import get_config, get_identity
        trigger = await get_config("telegram.bot_trigger_name", None)
        if trigger:
            return trigger
        identity = await get_identity()
        return identity.get("name", "Syne")

    async def _handle_inbound(self, msg: dict):
        """Process a single inbound wacli JSON message."""
        from_me = bool(msg.get("FromMe", False))

        text = msg.get("Text", "").strip()
        chat_jid = msg.get("ChatJID", "")
        media_type = msg.get("MediaType")
        is_image = media_type == "image"

        logger.info(f'[whatsapp] inbound: from_me={from_me} chat={chat_jid} media={media_type} text={text[:60]}')

        # Allow through if there's text OR an image
        if not text and not is_image:
            return

        # from_me handling:
        # - DM self-chat (sender == chat): allow (owner talking to themselves)
        # - DM to other contact: drop (outgoing message, not inbound)
        # - Group: always drop (prevents echo loop — Syne's own replies come back as from_me)
        sender_jid_raw = msg.get("SenderJID") or ""
        is_group = chat_jid.endswith("@g.us")
        if from_me:
            if is_group:
                logger.info(f'[whatsapp] drop: from_me in group ({chat_jid})')
                return
            if not sender_jid_raw:
                logger.info(f'[whatsapp] drop: from_me with no sender (wacli echo)')
                return
            # DM: only allow self-chat
            sender_local = sender_jid_raw.split(":", 1)[0].split("@", 1)[0]
            chat_local_fm = chat_jid.split(":", 1)[0].split("@", 1)[0]
            if sender_local != chat_local_fm:
                logger.info(f'[whatsapp] drop: from_me to other contact ({sender_local} != {chat_local_fm})')
                return

        # Echo detection: skip messages that match something we recently sent
        if text:
            h = self._content_hash(chat_jid, text)
            now = time.time()
            if h in self._echo_hashes and (now - self._echo_hashes[h]) < _ECHO_TTL:
                del self._echo_hashes[h]
                logger.debug(f'[whatsapp] echo suppressed: {text[:60]}')
                return

        sender_jid = sender_jid_raw or chat_jid
        # Normalize device JID (e.g. 628xxx:51@s.whatsapp.net) to base number for user id
        sender_platform_id = sender_jid.split(":", 1)[0] if ":" in sender_jid else sender_jid

        # Trigger word check
        # - Text messages: always require trigger
        # - Image with caption containing trigger: process (strip trigger from caption)
        # - Image in DM without trigger: still process (images don't need trigger in DMs)
        # - Image in group without trigger in caption: skip (groups always need trigger)
        trigger = await self._get_trigger_name()

        if is_image:
            caption = (msg.get("MediaCaption") or "").strip()
            if caption and trigger and self._has_trigger_word(caption, trigger):
                # Caption has trigger — use stripped caption as text
                text = self._strip_trigger_word(caption, trigger)
            elif caption and not is_group:
                # DM image with caption (no trigger needed)
                text = caption
            elif not caption and not is_group:
                # DM image without caption — use default prompt
                text = ""
            elif is_group:
                # Group: require trigger in caption
                if not caption or not trigger or not self._has_trigger_word(caption, trigger):
                    logger.info(f'[whatsapp] image in group without trigger, skipping')
                    return
                text = self._strip_trigger_word(caption, trigger)
        else:
            # Text-only: require trigger
            if trigger and not self._has_trigger_word(text, trigger):
                logger.info(f'[whatsapp] no trigger "{trigger}" in: {text[:80]}')
                return
            text = self._strip_trigger_word(text, trigger)
            if not text:
                return

        # Load members, check blocked, auto-register
        by_jid = await self._wa_load_members()

        if is_group:
            # Groups: match/register by chat_jid (the group itself)
            member_key, member = self._wa_match_jid(by_jid, chat_jid)
            group_name = msg.get("ChatName", "") or chat_jid.split("@")[0]

            if member and member.get('blocked'):
                logger.info(f'[whatsapp] blocked group, dropping: {chat_jid}')
                return

            if not member:
                reg_key = chat_jid
                member = {'jid': reg_key, 'name': group_name, 'model': None, 'blocked': False}
                by_jid[reg_key] = member
                member_key = reg_key
                await self._wa_save_members(by_jid)
                logger.info(f'[whatsapp] auto-registered group: {group_name} ({reg_key})')

            # In groups, sender_name comes from PushName (the person who typed)
            sender_name = msg.get("PushName", "") or sender_jid.split("@")[0]
            conversation_label = member.get('name') or group_name
        else:
            # DM: match/register by chat_jid (the contact)
            member_key, member = self._wa_match_jid(by_jid, chat_jid)
            chat_local = chat_jid.split(':', 1)[0].split('@', 1)[0] if chat_jid else ''
            push_name = msg.get("PushName", "") or msg.get("ChatName", "") or chat_local

            if member and member.get('blocked'):
                logger.info(f'[whatsapp] blocked member, dropping: {chat_jid}')
                return

            if not member:
                reg_key = chat_local or chat_jid
                member = {'jid': reg_key, 'name': push_name, 'model': None, 'blocked': False}
                by_jid[reg_key] = member
                member_key = reg_key
                await self._wa_save_members(by_jid)
                logger.info(f'[whatsapp] auto-registered new member: {push_name} ({reg_key})')

            sender_name = member.get('name') or push_name
            conversation_label = sender_name

        model_override = member.get('model') or None

        # ── Image handling ──
        image_metadata = None
        if is_image:
            msg_id = msg.get("MsgID")
            mime_type = msg.get("MimeType") or "image/jpeg"
            if msg_id:
                local_path = await self._download_media(chat_jid, msg_id)
                if local_path:
                    try:
                        with open(local_path, "rb") as f:
                            image_bytes = f.read()
                        if image_bytes:
                            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                            image_metadata = {"mime_type": mime_type, "base64": image_b64}
                            logger.info(f"[whatsapp] image downloaded: {len(image_bytes)} bytes, mime={mime_type}")
                        else:
                            logger.warning(f"[whatsapp] downloaded image is empty: {local_path}")
                    except Exception as e:
                        logger.error(f"[whatsapp] failed to read image file: {e}")
                else:
                    logger.warning(f"[whatsapp] media download returned no path for msg_id={msg_id}")
            else:
                logger.warning("[whatsapp] image message without msg_id, cannot download")

            # If download failed, fall back to text-only (caption) or skip
            if not image_metadata and not text:
                logger.info("[whatsapp] image download failed and no caption, skipping")
                return

            # Use caption as prompt, or default
            if not text:
                text = "What's in this image?"

        logger.info(f"[whatsapp] {sender_name}: {text[:100]}")

        inbound = InboundContext(
            channel="whatsapp",
            platform="whatsapp",
            chat_type="group" if is_group else "direct",
            conversation_label=conversation_label,
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

        # WA-specific instruction: do not reveal abilities
        text = (
            "[WhatsApp channel instruction: NEVER list or reveal your specific abilities, "
            "tools, or capabilities. If asked what you can do, reply diplomatically, e.g. "
            "\"Saya punya banyak kemampuan, di antaranya bisa ngobrol tentang apa saja.\" "
            "Keep it vague and friendly. Do NOT mention any tool names.]\n\n"
            + text
        )

        metadata = {
            "chat_id": chat_jid,
            "inbound": inbound,
            "original_text": original_text,
            "wa_model_override": model_override,  # None = use default
        }
        if image_metadata:
            metadata["image"] = image_metadata

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
        """Send a text (or text+image) message via wacli.

        If the response contains a MEDIA: path, the image is sent via
        `wacli send file` and the remaining text as a follow-up.

        `wacli sync --follow` holds an exclusive lock on the store. Running
        `wacli send ...` concurrently fails with "store is locked".

        Workaround: pause sync, send, then resume sync.
        """
        text = process_outbound(text)
        if not text:
            return

        # Extract MEDIA: path if present (e.g. from image_gen ability)
        text, media_path = extract_media(text)

        async with self._send_lock:
            was_syncing = self._process is not None
            if was_syncing:
                await self._stop_sync()

            try:
                # Send image first if present
                if media_path:
                    await self._send_file_locked(jid, media_path, caption=text)
                    text = None  # caption already sent with image

                # Send remaining text (if any)
                if text:
                    for chunk in split_message(text, max_length=4096):
                        await self._wacli_send_text(jid, chunk)
                        await asyncio.sleep(0.3)
            finally:
                if self._running and was_syncing:
                    ok = await self._start_sync()
                    if not ok:
                        logger.error('Failed to restart wacli sync after sending message')

    async def _wacli_send_text(self, jid: str, text: str):
        """Low-level: send a single text chunk. Caller must hold _send_lock."""
        proc = await asyncio.create_subprocess_exec(
            self._wacli_path, 'send', 'text',
            '--to', jid,
            '--message', text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            err = stderr.decode('utf-8', errors='replace') if stderr else ''
            raise RuntimeError(f"wacli send text failed (rc={proc.returncode}): {err[:200]}")
        self._echo_hashes[self._content_hash(jid, text)] = time.time()

    async def _send_file_locked(self, jid: str, file_path: str, caption: str = ""):
        """Low-level: send a file via wacli. Caller must hold _send_lock."""
        cmd = [self._wacli_path, 'send', 'file', '--to', jid, '--file', file_path]
        if caption:
            cmd.extend(['--caption', caption])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            err = stderr.decode('utf-8', errors='replace') if stderr else ''
            raise RuntimeError(f"wacli send file failed (rc={proc.returncode}): {err[:200]}")
        logger.info(f"[whatsapp] sent file to {jid}: {file_path}")

    # ── Media download ────────────────────────────────────────

    async def _download_media(self, chat_jid: str, msg_id: str) -> Optional[str]:
        """Download media via wacli and return local file path.

        Reuses _send_lock because wacli holds an exclusive lock on the
        SQLite store — same issue as sending messages.
        """
        if not msg_id or not chat_jid:
            return None

        async with self._send_lock:
            was_syncing = self._process is not None
            if was_syncing:
                await self._stop_sync()

            try:
                proc = await asyncio.create_subprocess_exec(
                    self._wacli_path, 'media', 'download',
                    '--chat', chat_jid,
                    '--id', msg_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                if proc.returncode != 0:
                    err = stderr.decode('utf-8', errors='replace') if stderr else ''
                    logger.error(f"wacli media download failed (rc={proc.returncode}): {err[:200]}")
                    return None
            except asyncio.TimeoutError:
                logger.error("wacli media download timed out")
                return None
            except Exception as e:
                logger.error(f"wacli media download error: {e}")
                return None
            finally:
                if self._running and was_syncing:
                    ok = await self._start_sync()
                    if not ok:
                        logger.error('Failed to restart wacli sync after media download')

        # Query DB for the local_path after download
        try:
            conn = sqlite3.connect(self._wacli_db, timeout=5)
            cur = conn.execute(
                "SELECT local_path FROM messages WHERE msg_id = ? LIMIT 1",
                (msg_id,),
            )
            row = cur.fetchone()
            conn.close()
            if row and row[0] and os.path.isfile(row[0]):
                return row[0]
            logger.warning(f"Media downloaded but local_path not found for msg_id={msg_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to query local_path: {e}")
            return None

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
