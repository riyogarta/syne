"""Website Screenshot Ability — capture a screenshot of a URL using Playwright.

This ability is intentionally minimal and uses the same Ability interface as other
built-in Syne abilities.

Dependencies (system/runtime):
- playwright (python)
- Playwright Chromium browser installed (playwright install chromium)

Security:
- Uses Syne URL safety check (SSRF protection) before navigation.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Optional

from .base import Ability
from ..security import is_url_safe

logger = logging.getLogger("syne.ability.website_screenshot")


class WebsiteScreenshotAbility(Ability):
    name = "website_screenshot"
    description = "Take a screenshot of a website URL (viewport or full page)"
    version = "1.1"

    async def ensure_dependencies(self) -> tuple[bool, str]:
        """Install playwright + Chromium browser in the current venv."""
        # 1. Check if playwright is already importable
        try:
            import playwright.async_api  # noqa: F401
            has_pkg = True
        except ImportError:
            has_pkg = False

        installed = []

        # 2. pip install playwright if missing
        if not has_pkg:
            logger.info("Installing playwright via pip...")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", "playwright",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
            if proc.returncode != 0:
                err = stderr.decode().strip()
                return False, f"Failed to install playwright: {err}"
            installed.append("playwright")

        # 3. Install Chromium browser if missing
        # playwright install chromium — downloads ~130MB browser binary
        playwright_bin = os.path.join(os.path.dirname(sys.executable), "playwright")
        if not os.path.isfile(playwright_bin):
            # Fallback: run via python -m playwright
            cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
        else:
            cmd = [playwright_bin, "install", "chromium"]

        logger.info("Installing Chromium browser via playwright...")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
        if proc.returncode != 0:
            err = stderr.decode().strip()
            return False, f"Failed to install Chromium: {err}"
        installed.append("chromium")

        if installed:
            return True, f"Installed: {', '.join(installed)}"
        return True, ""

    async def execute(self, params: dict, context: dict) -> dict:
        url = (params.get("url") or "").strip()
        if not url:
            return {"success": False, "error": "url is required"}

        safe, reason = is_url_safe(url)
        if not safe:
            return {"success": False, "error": f"URL blocked: {reason}"}

        full_page = bool(params.get("full_page", False))
        wait_ms = int(params.get("wait_ms", 1000))
        viewport_width = int(params.get("viewport_width", 1366))
        viewport_height = int(params.get("viewport_height", 768))

        # Output path (centralized)
        session_id = str(context.get("session_id") or int(time.time()))
        ts = int(time.time())
        outdir = self.get_output_dir(session_id=session_id)
        out_path = os.path.join(outdir, f"webshot_{session_id}_{ts}.png")

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(args=["--no-sandbox"])
                page = await browser.new_page(viewport={"width": viewport_width, "height": viewport_height})

                # Reasonable defaults
                page.set_default_navigation_timeout(45000)
                page.set_default_timeout(45000)

                await page.goto(url, wait_until="networkidle")
                if wait_ms > 0:
                    await page.wait_for_timeout(wait_ms)

                await page.screenshot(path=out_path, full_page=full_page)
                await browser.close()

            return {
                "success": True,
                "result": {
                    "url": url,
                    "full_page": full_page,
                    "viewport": {"width": viewport_width, "height": viewport_height},
                },
                "media": out_path,
            }

        except Exception as e:
            msg = str(e)
            hint: Optional[str] = None
            if "Executable doesn't exist" in msg or "chromium" in msg.lower():
                hint = "Playwright browser not installed. Try: playwright install chromium"
            elif "No module named" in msg and "playwright" in msg:
                hint = "Playwright python package not installed. Try: pip install playwright"

            return {"success": False, "error": msg if not hint else f"{msg} | Hint: {hint}"}

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "website_screenshot",
                "description": "Take a PNG screenshot of a website URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Website URL (https://...)"},
                        "full_page": {"type": "boolean", "description": "Capture full page (scrolling)", "default": False},
                        "viewport_width": {"type": "integer", "description": "Viewport width in px", "default": 1366},
                        "viewport_height": {"type": "integer", "description": "Viewport height in px", "default": 768},
                        "wait_ms": {"type": "integer", "description": "Extra wait after load (ms)", "default": 1000},
                    },
                    "required": ["url"],
                },
            },
        }



    def get_guide(self, enabled: bool, config: dict) -> str:
        if not enabled:
            return (
                "- Status: **disabled**\n"
                "- Enable: `update_ability(action='enable', name='website_screenshot')`"
            )

        # Non-fatal dependency check
        try:
            import playwright.async_api  # noqa: F401
            has_playwright = True
        except Exception:
            has_playwright = False

        if has_playwright:
            return (
                "- Status: **ready** (Playwright installed)\n"
                "- Use: `website_screenshot(url='https://...', full_page=false)`\n"
                "- If Chromium missing: run `playwright install chromium`"
            )

        return (
            "- Status: **not ready** (missing Playwright)\n"
            "- Setup (server): `pip install playwright && playwright install chromium`"
        )


    def get_required_config(self) -> list[str]:
        return []
