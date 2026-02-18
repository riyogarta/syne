"""Codex (ChatGPT) OAuth for Syne.

Same flow as OpenClaw/Codex CLI — user logs in via browser with their
ChatGPT account. Requires ChatGPT Plus/Pro/Team subscription.

Flow:
  1. Open browser to auth.openai.com
  2. User logs in with their ChatGPT account
  3. Callback to localhost:1455 → exchange code for tokens
  4. Tokens saved to PostgreSQL database
"""

import asyncio
import hashlib
import base64
import json
import logging
import secrets
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

logger = logging.getLogger("syne.auth.codex")

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_REDIRECT_URI = "http://localhost:1455/auth/callback"
_SCOPE = "openid profile email offline_access"


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier and challenge."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for Codex OAuth callback."""

    code: Optional[str] = None
    state: Optional[str] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        error = params.get("error", [None])[0]
        if error:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<h1>Authentication Failed</h1><p>{error}</p>".encode())
            return

        _OAuthCallbackHandler.code = params.get("code", [None])[0]
        _OAuthCallbackHandler.state = params.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<h1>Authentication Successful!</h1>"
            b"<p>You can close this window and return to the terminal.</p>"
        )

    def log_message(self, format, *args):
        pass


async def login_codex() -> dict:
    """Run Codex OAuth PKCE flow.

    Returns dict with: access_token, refresh_token, expires_at
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None

    server = HTTPServer(("127.0.0.1", 1455), _OAuthCallbackHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        params = urlencode({
            "response_type": "code",
            "client_id": _CLIENT_ID,
            "redirect_uri": _REDIRECT_URI,
            "scope": _SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
        })
        auth_url = f"{_AUTHORIZE_URL}?{params}"

        print("\n🔐 Opening browser for ChatGPT sign-in...")
        print(f"   If the browser doesn't open, visit:\n   {auth_url}\n")
        webbrowser.open(auth_url)

        print("⏳ Waiting for sign-in...")
        for _ in range(300):
            await asyncio.sleep(1)
            if _OAuthCallbackHandler.code:
                break
        else:
            raise TimeoutError("OAuth callback timed out (5 minutes)")

        code = _OAuthCallbackHandler.code
        callback_state = _OAuthCallbackHandler.state

        if callback_state != state:
            raise ValueError("OAuth state mismatch")

        print("🔄 Exchanging authorization code...")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": _CLIENT_ID,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": _REDIRECT_URI,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 0)

        if not access_token or not refresh_token:
            raise RuntimeError("Token exchange failed — missing access or refresh token.")

        expires_at = time.time() + expires_in

        print("✅ ChatGPT authentication successful")

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
        }

    finally:
        server.shutdown()
