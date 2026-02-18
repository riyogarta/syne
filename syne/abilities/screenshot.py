"""Screenshot Ability — capture webpage screenshots using Playwright."""

import time
import logging
from pathlib import Path

from playwright.async_api import async_playwright

from .base import Ability

logger = logging.getLogger("syne.abilities.screenshot")


class ScreenshotAbility(Ability):
    """Capture full-page or viewport screenshots of any URL using headless Chromium."""

    name = "screenshot"
    description = "Take a screenshot of a webpage given its URL. Returns a PNG image."
    version = "1.0"

    async def execute(self, params: dict, context: dict) -> dict:
        """Take a screenshot of the given URL.

        Args:
            params:
                url (str): The webpage URL to capture (required).
                full_page (bool): Capture the entire scrollable page (default True).
                width (int): Viewport width in pixels (default 1280).
                height (int): Viewport height in pixels (default 720).
                wait (float): Seconds to wait after page load before capturing (default 2).
                selector (str): Optional CSS selector — screenshot only that element.
            context: Execution context (user_id, session_id, etc.)

        Returns:
            dict with success, result, error, media keys.
        """
        url = params.get("url", "").strip()
        if not url:
            return {"success": False, "error": "URL is required"}

        # Normalise URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        full_page = params.get("full_page", True)
        width = int(params.get("width", 1280))
        height = int(params.get("height", 720))
        wait = float(params.get("wait", 2))
        selector = params.get("selector", "")

        # Clamp wait to sane limits
        wait = max(0, min(wait, 15))

        session_id = context.get("session_id", int(time.time()))
        filepath = f"/tmp/syne_screenshot_{session_id}_{int(time.time())}.png"

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                )
                page_ctx = await browser.new_context(
                    viewport={"width": width, "height": height},
                    locale="en-US",
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                )
                page = await page_ctx.new_page()

                # Navigate with generous timeout
                await page.goto(url, wait_until="networkidle", timeout=30_000)

                # Extra wait for JS-heavy pages
                if wait > 0:
                    await page.wait_for_timeout(int(wait * 1000))

                if selector:
                    element = await page.query_selector(selector)
                    if element:
                        await element.screenshot(path=filepath)
                    else:
                        return {
                            "success": False,
                            "error": f"Selector '{selector}' not found on page",
                        }
                else:
                    await page.screenshot(path=filepath, full_page=full_page)

                title = await page.title()
                await browser.close()

            return {
                "success": True,
                "result": f"Screenshot captured: {title or url}",
                "media": filepath,
            }

        except Exception as e:
            logger.error(f"Screenshot failed for {url}: {e}")
            return {"success": False, "error": f"Screenshot failed: {str(e)}"}

    def get_schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "screenshot",
                "description": (
                    "Take a screenshot of a webpage. Use this when the user wants to "
                    "see how a website looks, capture a page, or preview a URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL of the webpage to capture (e.g. https://example.com)",
                        },
                        "full_page": {
                            "type": "boolean",
                            "description": "Whether to capture the entire scrollable page (default true)",
                        },
                        "selector": {
                            "type": "string",
                            "description": "Optional CSS selector to screenshot only a specific element",
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    def get_required_config(self) -> list[str]:
        """No external API keys needed — uses local Playwright."""
        return []
