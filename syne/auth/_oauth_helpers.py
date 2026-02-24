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
    """
    code_event = threading.Event()  # Use threading.Event, not asyncio.Event
    result_code: list[str] = []
    paste_result: list[str] = []

    # Thread 1 (via asyncio): Poll callback server for code
    async def wait_callback():
        for _ in range(timeout_seconds):
            await asyncio.sleep(1)
            if code_event.is_set():
                return
            if handler_class.code:
                if not result_code:
                    result_code.append(handler_class.code)
                code_event.set()
                return

    # Thread 2: Read stdin line-by-line with select for non-blocking check
    def _read_stdin():
        """Read callback URL from stdin. Exits promptly when code_event is set."""
        import select

        sys.stdout.write(">>> Paste URL: ")
        sys.stdout.flush()

        buf = ""
        while not code_event.is_set():
            try:
                # Use select to avoid blocking forever on read(1).
                # This lets us check code_event regularly.
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if not ready:
                    continue  # timeout — loop back and check code_event

                ch = sys.stdin.read(1)
                if not ch:  # EOF
                    break
                if ch in ("\n", "\r"):
                    line = buf.strip()
                    buf = ""
                    if code_event.is_set():
                        return  # Code already received via callback
                    if not line:
                        if not code_event.is_set():
                            sys.stdout.write(">>> Paste URL: ")
                            sys.stdout.flush()
                        continue

                    code = _extract_code_from_url(line)
                    if code:
                        paste_result.append(code)
                        code_event.set()
                        return
                    else:
                        if code_event.is_set():
                            return  # Don't print error if already authenticated
                        if "/authorize" in line or "oauth2/v2/auth" in line:
                            sys.stdout.write(
                                "\n⚠  That's the authorize URL. You need the REDIRECT URL.\n"
                                "   Click 'Authorize' first, then copy the URL from the\n"
                                "   address bar (even if the page shows an error).\n\n"
                            )
                        else:
                            sys.stdout.write(
                                "\n⚠  Could not find authorization code in that URL.\n"
                                "   The URL should contain '?code=...' in it.\n\n"
                            )
                        sys.stdout.write(">>> Paste URL: ")
                        sys.stdout.flush()
                else:
                    buf += ch
            except (EOFError, OSError, KeyboardInterrupt, ValueError):
                return

    # Start stdin reader thread
    stdin_thread = threading.Thread(target=_read_stdin, daemon=True)
    stdin_thread.start()

    # Wait for either method
    try:
        await asyncio.wait_for(wait_callback(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        pass

    # Also check if paste got it while we were waiting
    if not result_code and not paste_result:
        # Give stdin thread a moment
        code_event.wait(timeout=2)

    if result_code:
        return result_code[0]
    if paste_result:
        return paste_result[0]

    raise TimeoutError("OAuth timed out — no authorization code received.")
