import asyncio
import os
from datetime import datetime
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from syne.abilities.base import Ability

class ScreenshotAbility(Ability):
    """
    An ability to capture a screenshot of a web page.
    """
    name = "screenshot"
    description = "Capture a screenshot of a given URL and return the image."
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
                            "description": "The full URL (including http:// or https://) to capture.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Optional filename for the saved image (e.g., 'mysite.png'). Defaults to a generated name.",
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(self, params: dict, context: dict) -> dict:
        url = params.get("url")
        if not url or not url.startswith(('http://', 'https://')):
            return {"success": False, "result": "Invalid URL. It must start with http:// or https://"}

        filename = params.get("filename")
        if not filename:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.replace('.', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{domain}_{timestamp}.png"
            except Exception:
                # Fallback for invalid URLs that somehow bypass the initial check
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'

        output_dir = "media/screenshots"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=60000)
                await page.screenshot(path=save_path, full_page=True)
                await browser.close()
            
            return {
                "success": True,
                "result": f"Screenshot of {url} saved to {save_path}",
                "media": save_path,
            }
        except Exception as e:
            error_message = f"An error occurred while taking screenshot: {str(e)}"
            if "net::ERR_NAME_NOT_RESOLVED" in str(e):
                error_message = f"Could not resolve the hostname for the URL: {url}. Please check the domain name."
            elif "Timeout" in str(e):
                error_message = f"The page at {url} took too long to load and timed out."
            
            return {"success": False, "result": error_message}

