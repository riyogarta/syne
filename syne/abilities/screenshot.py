"""Screenshot Ability — capture full-page screenshots of websites."""

import uuid
import os
from pathlib import Path

from playwright.async_api import async_playwright
from syne.abilities.base import Ability


class ScreenshotAbility(Ability):
    name = "screenshot"
    description = "Capture a full-page screenshot of a website URL."
    version = "1.0"

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to capture (e.g. https://example.com)",
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(self, params: dict, context: dict) -> dict:
        url = params.get("url", "")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Resolve output dir relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"screenshot_{uuid.uuid4()}.png"
        output_path = output_dir / filename

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={"width": 1280, "height": 720})
                await page.goto(url, wait_until="networkidle", timeout=60000)
                await page.screenshot(path=str(output_path), full_page=True)
                await browser.close()

            return {
                "success": True,
                "result": f"Screenshot of {url} saved.",
                "media": str(output_path),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to screenshot {url}: {e}",
            }
