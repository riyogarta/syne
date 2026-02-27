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
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._running = False

    # ── Dependencies ────────────────────────────────────────────

    async def ensure_dependencies(self) -> tuple[bool, str]:
        """Check for wacli binary; install if missing.

        Install strategy (no sudo — never escalate privileges):
        1. Check PATH and ~/.local/bin
        2. go install (if Go available)
        3. macOS only: download pre-built binary from GitHub releases
        4. Fail with instructions for the owner
        """
        # Already installed?
        if shutil.which("wacli"):
            return True, ""

        local_bin = os.path.join(_WACLI_INSTALL_DIR, "wacli")
        if os.path.isfile(local_bin) and os.access(local_bin, os.X_OK):
            self._wacli_path = local_bin
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
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
            if proc.returncode == 0:
                wacli_path = os.path.join(_WACLI_INSTALL_DIR, "wacli")
                if os.path.isfile(wacli_path):
                    self._wacli_path = wacli_path
                    logger.info(f"wacli installed via go install → {wacli_path}")
                    return True, f"wacli installed to {wacli_path}"
            logger.warning(f"go install failed: {stderr.decode()[:300]}")
            return False, f"go install failed: {stderr.decode()[:200]}"
        except asyncio.TimeoutError:
            return False, "go install timed out (180s)"
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

        await self._send_text(to, message)
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

    # ── Bridge lifecycle ───────────────────────────────────────

    async def start_bridge(self, agent, wacli_path: str = "wacli"):
        """Start the wacli background bridge.

        Called from main.py when whatsapp.enabled is true.
        """
        global _bridge_instance
        self._agent = agent
        self._wacli_path = wacli_path

        if not shutil.which(self._wacli_path):
            logger.error(
                f"wacli binary not found at '{self._wacli_path}'. "
                "Install from https://github.com/steipete/wacli"
            )
            return False

        try:
            self._process = await asyncio.create_subprocess_exec(
                self._wacli_path, "sync", "--follow", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            logger.error(f"Failed to start wacli: {e}")
            return False

        self._running = True
        self._reader_task = asyncio.create_task(self._reader_loop())
        _bridge_instance = self
        logger.info("WhatsApp bridge started.")
        return True

    async def stop(self):
        """Stop the WhatsApp bridge."""
        global _bridge_instance
        self._running = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._process.kill()
            self._process = None
        _bridge_instance = None
        logger.info("WhatsApp bridge stopped.")

    # ── Message loop ───────────────────────────────────────────

    async def _reader_loop(self):
        """Read JSON lines from wacli stdout and process messages."""
        while self._running and self._process and self._process.stdout:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    if self._running:
                        logger.warning("wacli process ended unexpectedly, restarting...")
                        await self._restart_wacli()
                    break

                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON line from wacli: {line[:200]}")
                    continue

                await self._handle_inbound(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reader loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _handle_inbound(self, msg: dict):
        """Process a single inbound wacli JSON message."""
        if msg.get("FromMe"):
            return

        text = msg.get("Text", "").strip()
        if not text:
            return

        chat_jid = msg.get("ChatJID", "")

        # DM only — skip groups
        if chat_jid.endswith("@g.us"):
            return

        sender_jid = msg.get("SenderJID", chat_jid)
        chat_name = msg.get("ChatName", "")
        sender_name = msg.get("PushName", chat_name or sender_jid.split("@")[0])

        logger.info(f"[whatsapp] {sender_name}: {text[:100]}")

        inbound = InboundContext(
            channel="whatsapp",
            platform="whatsapp",
            chat_type="direct",
            conversation_label=sender_name,
            chat_id=chat_jid,
            sender_name=sender_name,
            sender_id=sender_jid,
        )

        metadata = {
            "chat_id": chat_jid,
            "inbound": inbound,
        }

        try:
            response = await self._agent.handle_message(
                platform="whatsapp",
                chat_id=chat_jid,
                user_name=sender_name,
                user_platform_id=sender_jid,
                message=text,
                display_name=sender_name,
                message_metadata=metadata,
            )
            if response:
                await self._send_text(chat_jid, response)
        except Exception as e:
            logger.error(f"Error handling WhatsApp message: {e}", exc_info=True)
            await self._send_text(chat_jid, "Sorry, an error occurred.")

    # ── Outbound ───────────────────────────────────────────────

    async def _send_text(self, jid: str, text: str):
        """Send a text message via wacli."""
        text = process_outbound(text)
        if not text:
            return

        for chunk in split_message(text, max_length=4096):
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._wacli_path, "send", "text",
                    "--to", jid,
                    "--message", chunk,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode != 0:
                    logger.error(f"wacli send failed: {stderr.decode()[:200]}")
            except asyncio.TimeoutError:
                logger.error(f"wacli send timed out for {jid}")
            except Exception as e:
                logger.error(f"Failed to send WhatsApp message: {e}")
            await asyncio.sleep(0.3)

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
            self._process = await asyncio.create_subprocess_exec(
                self._wacli_path, "sync", "--follow", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._reader_task = asyncio.create_task(self._reader_loop())
        except Exception as e:
            logger.error(f"Failed to restart wacli: {e}")
