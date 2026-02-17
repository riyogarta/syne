"""Screenshot Ability — captures website screenshots via Playwright Chromium."""

import hashlib
import logging
import os
from pathlib import Path

from .base import Ability

logger = logging.getLogger("syne.abilities.screenshot")


class ScreenshotAbility(Ability):
    """Capture a screenshot of a website using Playwright's Chromium browser."""

    name = "screenshot"
    description = "Capture a screenshot of a website"
    version = "1.0"

    async def execute(self, params: dict, context: dict) -> dict:
        """Take a screenshot of the given URL.

        Args:
            params: Must contain 'url'. Optional: 'full_page' (bool, default False),
                    'width' (int, default 1280), 'height' (int, default 800).
            context: Execution context (not used for this ability).

        Returns:
            dict with success, result, media keys
        """
        url = params.get("url", "").strip()
        if not url:
            return {"success": False, "error": "URL is required"}

        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        full_page = params.get("full_page", False)
        width = int(params.get("width", 1280))
        height = int(params.get("height", 800))

        # Output directory
        output_dir = Path.home() / ".syne" / "screenshots"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = hashlib.md5(url.encode()).hexdigest() + ".png"
        output_path = str(output_dir / filename)

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                if not full_page:
                    await page.set_viewport_size({"width": width, "height": height})

                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                # Brief wait for JS rendering
                await page.wait_for_timeout(2000)

                await page.screenshot(path=output_path, full_page=full_page)
                await browser.close()

            logger.info(f"Screenshot saved: {output_path} ({url})")
            return {
                "success": True,
                "result": f"Screenshot of {url} saved.",
                "media": output_path,
            }

        except ImportError:
            return {
                "success": False,
                "error": "Playwright is not installed. Run: pip install playwright && playwright install chromium",
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
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the website to capture (e.g., https://example.com)",
                        },
                        "full_page": {
                            "type": "boolean",
                            "description": "Capture the full scrollable page (default: false)",
                        },
                        "width": {
                            "type": "integer",
                            "description": "Browser viewport width in pixels (default: 1280)",
                        },
                        "height": {
                            "type": "integer",
                            "description": "Browser viewport height in pixels (default: 800)",
                        },
                    },
                    "required": ["url"],
                },
            },
        }
