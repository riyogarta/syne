"""Shared OAuth helpers for all providers.

Provides hybrid code retrieval: callback server + manual URL paste.
Works on both desktop (browser redirects to localhost) and headless
(user copies redirect URL from browser address bar and pastes here).
"""

import asyncio
import sys
import threading
from urllib.parse import urlparse, parse_qs
from typing import Optional


def _extract_code_from_url(url: str) -> Optional[str]:
    """Extract authorization code from a redirect URL.

    Works with full callback URLs like:
        http://localhost:8085/oauth2callback?code=4/xxxxx&state=xxxxx
        http://localhost:9742/oauth/callback?code=abc123&state=def456
    """
    try:
        parsed = urlparse(url.strip())
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        return code
    except Exception:
        return None


async def wait_for_auth_code(
    handler_class,
    callback_path: str,
    timeout_seconds: int = 300,
) -> str:
    """Wait for auth code from callback server OR manual paste.

    Two methods run concurrently:
    1. Callback server listening for browser redirect (works on desktop)
    2. User prompt to paste the redirect URL (works on headless/remote)

    Whichever arrives first wins.

    Args:
        handler_class: The HTTP handler class (must have .code class attribute)
        callback_path: The path portion of callback URL (e.g. "/oauth2callback")
        timeout_seconds: Max wait time

    Returns:
        Authorization code string

    Raises:
        TimeoutError if neither method produces a code in time
    """
    code_event = asyncio.Event()
    result_code: list[str] = []

    # Task 1: Poll callback server for code (set by browser redirect)
    async def wait_callback():
        for _ in range(timeout_seconds):
            await asyncio.sleep(1)
            if handler_class.code:
                if not result_code:  # Don't override paste result
                    result_code.append(handler_class.code)
                code_event.set()
                return

    # Task 2: Prompt user to paste redirect URL
    paste_result: list[str] = []

    def _read_stdin():
        """Prompt and read from stdin in a thread (blocking)."""
        while not code_event.is_set():
            try:
                line = input(">>> Paste URL: ")
                line = line.strip()
                if not line:
                    continue

                code = _extract_code_from_url(line)
                if code:
                    paste_result.append(code)
                    code_event.set()
                    return
                else:
                    # Maybe they pasted the authorize URL instead of callback
                    if "/authorize" in line or "oauth2/v2/auth" in line:
                        print(
                            "\n⚠  That's the authorize URL. You need the REDIRECT URL instead."
                        )
                        print("   Click 'Authorize' in the browser first, then copy the URL")
                        print("   from the address bar (even if the page shows an error).\n")
                    else:
                        print(
                            "\n⚠  Could not find authorization code in that URL."
                        )
                        print("   The URL should contain '?code=...' in it.\n")
            except (EOFError, OSError, KeyboardInterrupt):
                return

    async def wait_paste():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _read_stdin)

    # Run both concurrently
    try:
        await asyncio.wait_for(
            asyncio.gather(wait_callback(), wait_paste(), return_exceptions=True),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        pass

    if result_code:
        return result_code[0]
    if paste_result:
        return paste_result[0]

    raise TimeoutError("OAuth timed out — no authorization code received.")
