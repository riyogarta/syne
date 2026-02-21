"""Google OAuth for Syne.

Uses the same OAuth flow as OpenClaw/Gemini CLI (Google Cloud Code Assist).
CCA client ID + 3 scopes (cloud-platform, email, profile).
User logs in via browser with their own Google account.

Flow:
  1. `syne init` opens browser for Google sign-in
  2. User logs in with their Google account
  3. Callback to localhost → exchange code for tokens
  4. Discover/provision GCP project via Code Assist API
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
from pathlib import Path
from threading import Thread
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

logger = logging.getLogger("syne.auth.google")

# Google Cloud Code Assist OAuth — same as OpenClaw/Gemini CLI
# Public OAuth app credentials (not secrets — same as any OAuth client ID)
_CCA_CID_B64 = (
    "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLm"
    "FwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t"
)
_CCA_CSC_B64 = "R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw="
_CLIENT_ID = base64.b64decode(_CCA_CID_B64).decode()
_CLIENT_SECRET = base64.b64decode(_CCA_CSC_B64).decode()

_REDIRECT_URI = "http://localhost:8085/oauth2callback"

# Same 3 scopes as OpenClaw — cloud-platform covers Generative Language API
_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_CODE_ASSIST_URL = "https://cloudcode-pa.googleapis.com"


class GoogleCredentials:
    """Manages Google OAuth credentials with auto-refresh and DB persistence."""

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        expires_at: float,
        project_id: str,
        email: Optional[str] = None,
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
        self.project_id = project_id
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
        logger.info("Refreshing Google access token...")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _TOKEN_URL,
                data={
                    "client_id": _CLIENT_ID,
                    "client_secret": _CLIENT_SECRET,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        self.access_token = data["access_token"]
        self.expires_at = time.time() + data["expires_in"] - 300
        if data.get("refresh_token"):
            self.refresh_token = data["refresh_token"]

        await self.save_to_db()
        logger.info("Google access token refreshed and saved to DB.")

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "project_id": self.project_id,
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoogleCredentials":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            project_id=data.get("project_id", ""),
            email=data.get("email"),
        )

    async def save_to_db(self):
        """Save credentials to database."""
        from ..db.credentials import set_google_oauth_credentials
        await set_google_oauth_credentials(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            expires_at=self.expires_at,
            project_id=self.project_id,
            email=self.email,
        )

    @classmethod
    async def load_from_db(cls) -> Optional["GoogleCredentials"]:
        """Load credentials from database."""
        from ..db.credentials import get_google_oauth_credentials
        data = await get_google_oauth_credentials()
        if not data or not data.get("refresh_token"):
            return None
        try:
            return cls.from_dict(data)
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to load credentials from DB: {e}")
            return None


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier and challenge."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    code: Optional[str] = None
    state: Optional[str] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/oauth2callback":
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


async def _discover_project(access_token: str) -> str:
    """Discover or provision Google Cloud project via Code Assist API.
    
    Same approach as OpenClaw/Gemini CLI — auto-provisions free tier project.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        # Try loading existing project
        resp = await client.post(
            f"{_CODE_ASSIST_URL}/v1internal:loadCodeAssist",
            headers=headers,
            json={
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                },
            },
        )

        if resp.status_code == 200:
            data = resp.json()
            if project_id := data.get("cloudaicompanionProject"):
                return project_id

        # Need onboarding — provision free tier
        resp = await client.post(
            f"{_CODE_ASSIST_URL}/v1internal:onboardUser",
            headers=headers,
            json={
                "tierId": "free-tier",
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # Poll if needed
        if not data.get("done") and data.get("name"):
            for _ in range(30):
                await asyncio.sleep(5)
                poll_resp = await client.get(
                    f"{_CODE_ASSIST_URL}/v1internal/{data['name']}",
                    headers=headers,
                )
                poll_resp.raise_for_status()
                data = poll_resp.json()
                if data.get("done"):
                    break

        project_id = (
            data.get("response", {})
            .get("cloudaicompanionProject", {})
            .get("id")
        )
        if project_id:
            return project_id

    raise RuntimeError("Failed to discover or provision Google Cloud project.")


async def _get_user_email(access_token: str) -> Optional[str]:
    """Get user email from access token."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code == 200:
                return resp.json().get("email")
    except Exception:
        pass
    return None


async def login_google() -> GoogleCredentials:
    """Run Google OAuth PKCE flow.

    1. Start local callback server on port 8085
    2. Open browser for Google sign-in
    3. Wait for callback with auth code
    4. Exchange code for tokens
    5. Discover/provision project
    6. Return credentials
    """
    verifier, challenge = _generate_pkce()

    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None

    server = HTTPServer(("127.0.0.1", 8085), _OAuthCallbackHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        params = urlencode({
            "client_id": _CLIENT_ID,
            "response_type": "code",
            "redirect_uri": _REDIRECT_URI,
            "scope": " ".join(_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": verifier,
            "access_type": "offline",
            "prompt": "consent",
        })
        auth_url = f"{_AUTH_URL}?{params}"

        print("\n🔐 Open this URL in your browser to sign in with Google:")
        print(f"\n   {auth_url}\n")
        print("⏳ Waiting for sign-in...")
        print('   After login, your browser will show "This site can\'t be reached".')
        print("   That's normal! Copy the ENTIRE URL from the address bar and paste it here.")
        print("   (It looks like: http://localhost:8085/oauth2callback?code=...)\n")

        from ._oauth_helpers import wait_for_auth_code
        code = await wait_for_auth_code(
            _OAuthCallbackHandler, "/oauth2callback", timeout_seconds=300
        )

        print("🔄 Exchanging authorization code...")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _TOKEN_URL,
                data={
                    "client_id": _CLIENT_ID,
                    "client_secret": _CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": _REDIRECT_URI,
                    "code_verifier": verifier,
                },
            )
            resp.raise_for_status()
            token_data = resp.json()

        if not token_data.get("refresh_token"):
            raise RuntimeError("No refresh token received. Try again.")

        access_token = token_data["access_token"]
        expires_at = time.time() + token_data["expires_in"] - 300

        print("🔍 Setting up project...")
        email = await _get_user_email(access_token)
        project_id = await _discover_project(access_token)

        print(f"✅ Authenticated as {email or 'unknown'}")
        print(f"   Project: {project_id}")

        return GoogleCredentials(
            access_token=access_token,
            refresh_token=token_data["refresh_token"],
            expires_at=expires_at,
            project_id=project_id,
            email=email,
        )

    finally:
        server.shutdown()


async def get_credentials(auto_refresh: bool = True) -> Optional[GoogleCredentials]:
    """Get Google credentials from database.

    Returns None if not configured — user must run `syne init`.
    """
    creds = await GoogleCredentials.load_from_db()
    if creds:
        if auto_refresh and creds.is_expired:
            await creds.get_token()
        logger.info(f"Using DB credentials for {creds.email}")
        return creds

    logger.warning("No Google OAuth credentials found. Run 'syne init' to authenticate.")
    return None
