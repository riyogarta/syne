"""Shared OAuth helpers for all providers.

Provides hybrid code retrieval: callback server + manual URL paste.
"""

import asyncio
import sys
import threading
from urllib.parse import urlparse, parse_qs
from typing import Optional


def _extract_code_from_url(url: str, callback_path: str) -> Optional[str]:
    """Extract authorization code from a redirect URL.
    
    Works with full URLs like:
        http://localhost:8085/oauth2callback?code=4/xxxxx&state=xxxxx
    Or just the query string.
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
    
    Runs two tasks concurrently:
    1. Callback server listening for redirect
    2. stdin reader waiting for user to paste redirect URL
    
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
    result_code: list[str] = []  # Using list to allow mutation in closures

    # Task 1: Wait for callback server to receive code
    async def wait_callback():
        for _ in range(timeout_seconds):
            await asyncio.sleep(1)
            if handler_class.code:
                result_code.append(handler_class.code)
                code_event.set()
                return

    # Task 2: Wait for manual paste from stdin (in a thread)
    paste_result: list[str] = []
    
    def _read_stdin():
        """Read from stdin in a thread (blocking)."""
        while not code_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    continue
                line = line.strip()
                if not line:
                    continue
                # Try to extract code from pasted URL
                code = _extract_code_from_url(line, callback_path)
                if code:
                    paste_result.append(code)
                    code_event.set()
                    return
            except (EOFError, OSError):
                return

    async def wait_paste():
        thread = threading.Thread(target=_read_stdin, daemon=True)
        thread.start()
        # Wait until code_event is set (by either callback or paste)
        while not code_event.is_set():
            await asyncio.sleep(0.5)

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
    
    raise TimeoutError("OAuth callback timed out — no code received")
