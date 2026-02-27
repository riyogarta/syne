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
import re
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

# Unicode half-block characters used by qrterminal
_QR_CHARS = frozenset("\u2588\u2580\u2584 ")  # █ ▀ ▄ and space

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
        self._auth_monitor_task: Optional[asyncio.Task] = None
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

    # ── Authentication ─────────────────────────────────────────

    async def is_authenticated(self) -> bool:
        """Check if wacli has a valid WhatsApp session.

        Fast path: no ~/.wacli/ dir → definitely not authed.
        Then actually probe via `wacli chats list` — catches expired sessions.
        """
        store_dir = Path.home() / ".wacli"
        if not store_dir.is_dir():
            return False

        # Find wacli binary
        wacli = self._wacli_path
        if not shutil.which(wacli):
            local_bin = os.path.join(_WACLI_INSTALL_DIR, "wacli")
            if os.path.isfile(local_bin) and os.access(local_bin, os.X_OK):
                wacli = local_bin
            else:
                return False

        # Actually validate: run a quick command that requires auth
        try:
            proc = await asyncio.create_subprocess_exec(
                wacli, "chats", "list", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            return proc.returncode == 0
        except (asyncio.TimeoutError, Exception):
            return False

    async def start_auth_flow(self, agent) -> Optional[str]:
        """Run `wacli auth`, capture QR code, render as PNG.

        Returns the path to the QR PNG image, or None on failure.
        A background task monitors the auth process for completion.
        """
        self._agent = agent
        wacli = self._wacli_path

        if not shutil.which(wacli):
            local_bin = os.path.join(_WACLI_INSTALL_DIR, "wacli")
            if os.path.isfile(local_bin) and os.access(local_bin, os.X_OK):
                wacli = local_bin
                self._wacli_path = wacli
            else:
                logger.error("wacli binary not found for auth flow")
                return None

        try:
            proc = await asyncio.create_subprocess_exec(
                wacli, "auth",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            logger.error(f"Failed to start wacli auth: {e}")
            return None

        # Read stderr — wacli outputs QR to stderr (stdout reserved for JSON)
        qr_lines: list[str] = []
        collecting = False

        try:
            while True:
                raw = await asyncio.wait_for(
                    proc.stderr.readline(), timeout=30,
                )
                if not raw:
                    break  # EOF

                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                # Strip ANSI escape sequences
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line)

                if _is_qr_line(clean):
                    collecting = True
                    qr_lines.append(clean)
                elif collecting:
                    # End of QR block
                    break
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for QR from wacli auth")
            proc.kill()
            return None

        if not qr_lines:
            logger.warning("No QR lines captured from wacli auth")
            proc.kill()
            return None

        # Parse and render
        matrix = _parse_qr_output(qr_lines)
        if not matrix:
            logger.warning("Failed to parse QR matrix from wacli auth output")
            proc.kill()
            return None

        out_dir = self.get_output_dir()
        qr_path = os.path.join(out_dir, "whatsapp_qr.png")
        rendered = _render_qr_png(matrix, qr_path)
        if not rendered:
            proc.kill()
            return None

        # Start background monitor for auth completion
        self._auth_monitor_task = asyncio.create_task(
            self._monitor_auth(proc, agent)
        )

        logger.info(f"QR code rendered to {qr_path}")
        return qr_path

    async def _monitor_auth(self, proc: asyncio.subprocess.Process, agent):
        """Background task: wait for wacli auth to complete.

        On success, starts the bridge and notifies the owner via Telegram.
        On failure/timeout, notifies the owner to retry.
        """
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode == 0:
                logger.info("WhatsApp authentication successful!")
                # Start the bridge now
                from ..db.models import get_config
                wa_path = await get_config("whatsapp.wacli_path", "wacli")
                started = await self.start_bridge(agent, wacli_path=wa_path)
                msg = (
                    "WhatsApp connected! Bridge sudah aktif."
                    if started
                    else "WhatsApp authenticated, tapi bridge gagal start. Coba restart Syne."
                )
                await self._notify_owner(agent, msg)
            else:
                err = stderr.decode()[:200] if stderr else "unknown error"
                logger.warning(f"wacli auth failed (rc={proc.returncode}): {err}")
                await self._notify_owner(
                    agent,
                    "WhatsApp QR expired atau auth gagal. Kirim 'enable whatsapp' lagi untuk retry.",
                )
        except asyncio.TimeoutError:
            logger.warning("wacli auth timed out (120s)")
            proc.kill()
            await self._notify_owner(
                agent,
                "WhatsApp QR timeout (2 menit). Kirim 'enable whatsapp' lagi untuk retry.",
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring wacli auth: {e}", exc_info=True)

    @staticmethod
    async def _notify_owner(agent, message: str):
        """Send a notification to the owner via Telegram."""
        try:
            from ..db.connection import get_connection
            async with get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT platform_id FROM users "
                    "WHERE access_level = 'owner' AND platform = 'telegram' LIMIT 1"
                )
            if row and row["platform_id"]:
                tg = getattr(agent, "_telegram_channel", None)
                if tg and hasattr(tg, "_bot"):
                    await tg._bot.send_message(
                        chat_id=int(row["platform_id"]),
                        text=message,
                    )
                    return
            logger.info(f"Owner notification (no Telegram): {message}")
        except Exception as e:
            logger.error(f"Failed to notify owner: {e}")

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


# ── QR parsing / rendering helpers ─────────────────────────────


def _is_qr_line(line: str) -> bool:
    """Check if a line is part of a QR code (Unicode half-block output)."""
    stripped = line.strip()
    if not stripped:
        return False
    return all(ch in _QR_CHARS for ch in stripped)


def _parse_qr_output(lines: list[str]) -> list[list[int]]:
    """Parse Unicode half-block QR output into a binary pixel matrix.

    Each character encodes 2 vertical pixels using half-blocks:
        █ (U+2588) = top black, bottom black
        ▀ (U+2580) = top black, bottom white
        ▄ (U+2584) = top white, bottom black
        (space)    = top white, bottom white

    Returns:
        2D list of ints (1=black, 0=white), or empty list on failure.
    """
    if not lines:
        return []

    rows_top: list[list[int]] = []
    rows_bot: list[list[int]] = []

    for line in lines:
        top_row: list[int] = []
        bot_row: list[int] = []
        for ch in line:
            if ch == "\u2588":    # █ — both black
                top_row.append(1)
                bot_row.append(1)
            elif ch == "\u2580":  # ▀ — top black, bottom white
                top_row.append(1)
                bot_row.append(0)
            elif ch == "\u2584":  # ▄ — top white, bottom black
                top_row.append(0)
                bot_row.append(1)
            elif ch == " ":       # space — both white
                top_row.append(0)
                bot_row.append(0)
            # skip any other characters
        rows_top.append(top_row)
        rows_bot.append(bot_row)

    # Interleave: top row, bottom row, top row, bottom row...
    matrix: list[list[int]] = []
    for t, b in zip(rows_top, rows_bot):
        matrix.append(t)
        matrix.append(b)

    if not matrix or not matrix[0]:
        return []

    # Normalize row widths (pad shorter rows with white)
    max_w = max(len(r) for r in matrix)
    for r in matrix:
        while len(r) < max_w:
            r.append(0)

    return matrix


def _render_qr_png(matrix: list[list[int]], path: str, scale: int = 10) -> Optional[str]:
    """Render a binary pixel matrix as a PNG image.

    Args:
        matrix: 2D list of 0/1 values (1=black, 0=white)
        path: Output file path
        scale: Pixels per QR module (default 10)

    Returns:
        The output path on success, None on failure.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow not installed — cannot render QR PNG")
        return None

    h = len(matrix)
    w = len(matrix[0]) if h > 0 else 0
    if h == 0 or w == 0:
        return None

    # Add quiet zone (4 modules on each side)
    quiet = 4
    img_w = (w + quiet * 2) * scale
    img_h = (h + quiet * 2) * scale

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    pixels = img.load()

    for y, row in enumerate(matrix):
        for x, val in enumerate(row):
            if val:
                # Draw a scale x scale black square
                px = (x + quiet) * scale
                py = (y + quiet) * scale
                for dy in range(scale):
                    for dx in range(scale):
                        pixels[px + dx, py + dy] = (0, 0, 0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, "PNG")
    logger.info(f"QR PNG rendered: {w}x{h} modules, {img_w}x{img_h} px → {path}")
    return path
