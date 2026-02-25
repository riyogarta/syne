"""Claude/Anthropic OAuth for Syne.

Standalone OAuth flow — NO dependency on Claude Code CLI.
User logs in via browser with their claude.ai account.
Requires claude.ai Pro/Max/Team subscription.

Two flows supported:
  A) Authorization Code + PKCE (interactive — needs localhost callback)
  B) Device Code Flow / RFC 8628 (headless — no localhost needed)

Flow A (Auth Code + PKCE):
  1. Start local HTTP server for callback
  2. Open browser to claude.ai/oauth/authorize
  3. User logs in with their claude.ai account
  4. Callback receives auth code
  5. Exchange code for tokens via platform.claude.com
  6. Tokens saved to PostgreSQL database
  7. Auto-refresh when expired

Flow B (Device Code — headless):
  1. POST to /v1/oauth/device/code → get user_code + verification_uri
  2. User opens verification_uri in ANY browser (phone, laptop, etc.)
  3. User enters the user_code
  4. Syne polls token endpoint until authorized
  5. Tokens saved to PostgreSQL database
  6. Auto-refresh when expired
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

logger = logging.getLogger("syne.auth.claude")

# Claude OAuth — same as Claude Code CLI
_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_REDIRECT_PORT = 9742
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}/oauth/callback"
_SCOPES = "user:inference user:profile"
_BETA_HEADER = "oauth-2025-04-20"

# API
_API_URL = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier and challenge."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for Claude OAuth callback."""

    code: Optional[str] = None
    state: Optional[str] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/oauth/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        error = params.get("error", [None])[0]
        if error:
            desc = params.get("error_description", [""])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<h1>Authentication Failed</h1><p>{error}: {desc}</p>".encode()
            )
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
        pass  # suppress HTTP logs


class ClaudeCredentials:
    """Manages Claude OAuth credentials with auto-refresh and DB persistence."""

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        expires_at: float,
        email: Optional[str] = None,
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
        self.email = email
        self._lock = asyncio.Lock()

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    async def get_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if not self.is_expired:
            return self.access_token
        async with self._lock:
            if not self.is_expired:
                return self.access_token
            await self._refresh()
            return self.access_token

    async def _refresh(self):
        """Refresh the access token."""
        logger.info("Refreshing Claude access token...")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": _CLIENT_ID,
                    "refresh_token": self.refresh_token,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "anthropic-beta": _BETA_HEADER,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        self.access_token = data["access_token"]
        self.expires_at = time.time() + data.get("expires_in", 3600) - 300
        if data.get("refresh_token"):
            self.refresh_token = data["refresh_token"]

        await self.save_to_db()
        logger.info("Claude access token refreshed and saved to DB.")

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClaudeCredentials":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            email=data.get("email"),
        )

    async def save_to_db(self):
        """Save credentials to database."""
        from ..db.credentials import set_claude_oauth_credentials
        await set_claude_oauth_credentials(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            expires_at=self.expires_at,
            email=self.email,
        )

    @classmethod
    async def load_from_db(cls) -> Optional["ClaudeCredentials"]:
        """Load credentials from database."""
        from ..db.credentials import get_claude_oauth_credentials
        data = await get_claude_oauth_credentials()
        if not data or not data.get("refresh_token"):
            return None
        try:
            return cls.from_dict(data)
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to load Claude credentials from DB: {e}")
            return None


async def _get_user_profile(access_token: str) -> Optional[str]:
    """Get user email/profile from Claude API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{_API_URL}/v1/me",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "anthropic-version": _API_VERSION,
                    "anthropic-beta": _BETA_HEADER,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("email") or data.get("name")
    except Exception:
        pass
    return None


async def login_claude() -> ClaudeCredentials:
    """Run Claude OAuth PKCE flow.

    1. Start local callback server on port 9742
    2. Open browser for claude.ai sign-in
    3. Wait for callback with auth code
    4. Exchange code for tokens
    5. Return credentials
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None

    server = HTTPServer(("127.0.0.1", _REDIRECT_PORT), _OAuthCallbackHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        params = urlencode({
            "response_type": "code",
            "client_id": _CLIENT_ID,
            "redirect_uri": _REDIRECT_URI,
            "scope": _SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        })
        auth_url = f"{_AUTHORIZE_URL}?{params}"

        print("\n🔐 Open this URL in your browser to sign in with Claude:")
        print(f"\n   {auth_url}\n")
        print("Steps:")
        print("   1. Open the URL above in your browser")
        print("   2. Click 'Authorize' to grant access")
        print('   3. Browser will show "This site can\'t be reached" — that\'s normal!')
        print("   4. Copy the ENTIRE URL from the browser address bar")
        print(f"      (looks like: http://localhost:{_REDIRECT_PORT}/oauth/callback?code=...)")
        print("   5. Paste it below\n")

        from ._oauth_helpers import wait_for_auth_code
        code = await wait_for_auth_code(
            _OAuthCallbackHandler, "/oauth/callback", timeout_seconds=300
        )

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
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "anthropic-beta": _BETA_HEADER,
                },
            )
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token or not refresh_token:
            raise RuntimeError("Token exchange failed — missing access or refresh token.")

        expires_at = time.time() + expires_in - 300  # 5 min buffer

        email = await _get_user_profile(access_token)

        print(f"✅ Claude authentication successful{f' ({email})' if email else ''}")

        return ClaudeCredentials(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            email=email,
        )

    finally:
        server.shutdown()


# ============================================================
# Device Code Flow (RFC 8628) — headless servers
# ============================================================

_DEVICE_CODE_URL = f"https://platform.claude.com/v1/oauth/device/code"
_DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


async def login_claude_device() -> ClaudeCredentials:
    """Run Claude Device Code Flow (RFC 8628).

    No localhost callback needed. Works on headless servers.
    User opens a URL on ANY device (phone, laptop) and enters a code.

    1. POST to device/code endpoint → get user_code + verification_uri
    2. Display code to user
    3. Poll token endpoint until authorized
    4. Return credentials
    """
    print("\n🔐 Claude Device Code Authentication (headless)")
    print("   No browser needed on this machine.\n")

    # Step 1: Request device code
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _DEVICE_CODE_URL,
            data={
                "client_id": _CLIENT_ID,
                "scope": _SCOPES,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "anthropic-beta": _BETA_HEADER,
            },
        )
        if resp.status_code != 200:
            error_body = resp.text
            raise RuntimeError(
                f"Device code request failed ({resp.status_code}): {error_body}"
            )
        device_data = resp.json()

    user_code = device_data["user_code"]
    verification_uri = device_data.get("verification_uri_complete") or device_data["verification_uri"]
    device_code = device_data["device_code"]
    interval = device_data.get("interval", 5)
    expires_in = device_data.get("expires_in", 600)

    # Step 2: Display instructions
    print(f"   1. Open this URL on any device:")
    print(f"      {verification_uri}")
    print()
    print(f"   2. Enter this code:  [  {user_code}  ]")
    print()
    print(f"   ⏳ Waiting for authorization (expires in {expires_in // 60} minutes)...")

    # Step 3: Poll token endpoint
    deadline = time.time() + expires_in
    poll_interval = interval

    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() < deadline:
            await asyncio.sleep(poll_interval)

            resp = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": _DEVICE_CODE_GRANT,
                    "client_id": _CLIENT_ID,
                    "device_code": device_code,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "anthropic-beta": _BETA_HEADER,
                },
            )

            if resp.status_code == 200:
                token_data = resp.json()
                break

            # Handle polling responses
            try:
                error_data = resp.json()
            except Exception:
                error_data = {"error": "unknown", "error_description": resp.text}

            error = error_data.get("error", "")

            if error == "authorization_pending":
                # User hasn't authorized yet — keep polling
                continue
            elif error == "slow_down":
                # Server wants us to slow down
                poll_interval = min(poll_interval + 5, 30)
                logger.debug(f"Slowing down polling to {poll_interval}s")
                continue
            elif error == "expired_token":
                raise RuntimeError(
                    "Device code expired. Please run authentication again."
                )
            elif error == "access_denied":
                raise RuntimeError(
                    "Authorization denied by user."
                )
            else:
                raise RuntimeError(
                    f"Device code auth error: {error} — "
                    f"{error_data.get('error_description', '')}"
                )
        else:
            raise TimeoutError(
                "Device code authorization timed out. Please try again."
            )

    # Step 4: Extract tokens
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_in_token = token_data.get("expires_in", 3600)

    if not access_token or not refresh_token:
        raise RuntimeError(
            "Token exchange failed — missing access or refresh token."
        )

    expires_at = time.time() + expires_in_token - 300  # 5 min buffer

    email = await _get_user_profile(access_token)

    print(f"\n✅ Claude authentication successful{f' ({email})' if email else ''}")

    return ClaudeCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        email=email,
    )


async def get_credentials(auto_refresh: bool = True) -> Optional[ClaudeCredentials]:
    """Get Claude credentials from database.

    Returns None if not configured — user must run `syne init`.
    """
    creds = await ClaudeCredentials.load_from_db()
    if creds:
        if auto_refresh and creds.is_expired:
            await creds.get_token()
        logger.info(f"Using DB credentials for Claude ({creds.email})")
        return creds

    logger.warning("No Claude OAuth credentials found. Run 'syne init' to authenticate.")
    return None
