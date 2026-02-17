import asyncio
import logging
import os
import uuid
from playwright.async_api import async_playwright
from syne.abilities.base import Ability

# Configure logging
logger = logging.getLogger(__name__)

class ScreenshotAbility(Ability):
    name = "screenshot"
    description = "Capture a screenshot of a website"
    version = "1.0"

    def get_schema(self) -> dict:
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
                            "description": "URL of the website to capture. Must start with http:// or https://",
                        },
                        "full_page": {
                            "type": "boolean",
                            "description": "Whether to capture the full scrollable page. Default is True.",
                            "default": True,
                        }
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(self, params: dict, context: dict) -> dict:
        url = params.get("url")
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return {"success": False, "result": "Invalid or missing URL. Please provide a full URL starting with http:// or https://."}

        full_page = params.get("full_page", True)
        
        # Ensure the /tmp directory exists
        output_dir = "/tmp"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/{uuid.uuid4()}.png"

        try:
            async with async_playwright() as p:
                logger.info(f"Launching browser to screenshot {url}")
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Increase timeout to 60 seconds for slow pages
                page.set_default_timeout(60000)
                
                logger.info(f"Navigating to {url}")
                await page.goto(url, wait_until="networkidle")
                
                logger.info(f"Taking screenshot and saving to {filename}")
                await page.screenshot(path=filename, full_page=full_page)
                
                await browser.close()
                logger.info(f"Browser closed. Screenshot successful.")

                return {
                    "success": True,
                    "result": f"Screenshot of {url} has been captured.",
                    "media": filename,
                }
        except Exception as e:
            logger.error(f"Error taking screenshot of {url}: {e}", exc_info=True)
            # Clean up failed screenshot file if it exists
            if os.path.exists(filename):
                os.remove(filename)
            return {"success": False, "result": f"An error occurred while taking the screenshot: {str(e)}"}

