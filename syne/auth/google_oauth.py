"""Google OAuth for Syne.

User authenticates with their OWN Google Cloud OAuth client ID.
No hardcoded client IDs — everything belongs to the user.

Flow:
  1. User creates GCP project + OAuth client ID (Desktop app)
  2. `syne init` reads client_secret.json, runs OAuth PKCE flow
  3. User logs in via browser with their own Google account
  4. Tokens + client ID/secret saved to PostgreSQL database
  5. Auto-refresh using stored client credentials
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

logger = logging.getLogger("syne.auth.google")

_REDIRECT_URI = "http://localhost:8085/oauth2callback"
_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/generative-language.retriever",
    "https://www.googleapis.com/auth/generative-language.tuning",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"


class GoogleCredentials:
    """Manages Google OAuth credentials with auto-refresh and DB persistence.
    
    All credentials belong to the user — client_id, client_secret, tokens.
    Nothing is hardcoded.
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        expires_at: float,  # epoch seconds
        client_id: str,
        client_secret: str,
        project_id: str = "",
        email: Optional[str] = None,
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
        self.client_id = client_id
        self.client_secret = client_secret
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
            # Double-check after acquiring lock
            if not self.is_expired:
                return self.access_token
            await self._refresh()
            return self.access_token

    async def _refresh(self):
        """Refresh the access token using the user's own client credentials."""
        logger.info("Refreshing Google access token...")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        self.access_token = data["access_token"]
        # Expire 5 minutes early as buffer
        self.expires_at = time.time() + data["expires_in"] - 300
        if data.get("refresh_token"):
            self.refresh_token = data["refresh_token"]
        
        # Save refreshed token to DB
        await self.save_to_db()
        logger.info("Google access token refreshed and saved to DB.")

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "project_id": self.project_id,
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoogleCredentials":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            client_id=data["client_id"],
            client_secret=data["client_secret"],
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
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

    @classmethod
    async def load_from_db(cls) -> Optional["GoogleCredentials"]:
        """Load credentials from database."""
        from ..db.credentials import get_google_oauth_credentials
        data = await get_google_oauth_credentials()
        if not data or not data.get("refresh_token"):
            return None
        # Must have client_id and client_secret (user's own)
        if not data.get("client_id") or not data.get("client_secret"):
            logger.warning("Credentials missing client_id/client_secret. Re-run 'syne init'.")
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


def load_client_secret(path: str) -> tuple[str, str]:
    """Load client_id and client_secret from a Google client_secret.json file.
    
    Args:
        path: Path to client_secret.json downloaded from GCP Console
        
    Returns:
        (client_id, client_secret) tuple
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in {p}")
    
    # Google client_secret.json format:
    # {"installed": {"client_id": "...", "client_secret": "...", ...}}
    # or {"web": {"client_id": "...", "client_secret": "...", ...}}
    for key in ("installed", "web"):
        if key in data:
            app = data[key]
            client_id = app.get("client_id")
            client_secret = app.get("client_secret")
            if client_id and client_secret:
                return client_id, client_secret
    
    raise ValueError(
        f"Could not find client_id/client_secret in {p}. "
        "Make sure this is a valid client_secret.json from Google Cloud Console."
    )


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
        pass  # Suppress default logging


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


async def login_google(client_id: str, client_secret: str) -> GoogleCredentials:
    """Run the full Google OAuth PKCE flow using the user's own client credentials.

    1. Start local server on port 8085
    2. Open browser to Google consent screen
    3. Wait for callback with auth code
    4. Exchange code for tokens
    5. Return credentials
    
    Args:
        client_id: User's OAuth client ID from GCP Console
        client_secret: User's OAuth client secret from GCP Console
    """
    verifier, challenge = _generate_pkce()

    # Reset handler state
    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None

    # Start callback server
    server = HTTPServer(("127.0.0.1", 8085), _OAuthCallbackHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # Build auth URL with user's own client_id
        params = urlencode({
            "client_id": client_id,
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

        print("\n🔐 Opening browser for Google sign-in...")
        print(f"   If the browser doesn't open, visit:\n   {auth_url}\n")
        webbrowser.open(auth_url)

        # Wait for callback
        print("⏳ Waiting for OAuth callback...")
        for _ in range(300):  # 5 min timeout
            await asyncio.sleep(1)
            if _OAuthCallbackHandler.code:
                break
        else:
            raise TimeoutError("OAuth callback timed out (5 minutes)")

        code = _OAuthCallbackHandler.code
        state = _OAuthCallbackHandler.state

        if state != verifier:
            raise ValueError("OAuth state mismatch — possible CSRF attack")

        # Exchange code for tokens using user's own client credentials
        print("🔄 Exchanging authorization code...")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
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

        # Get email
        email = await _get_user_email(access_token)

        # Extract project_id from client_id (format: PROJECT_NUMBER-xxx.apps.googleusercontent.com)
        project_id = client_id.split("-")[0] if "-" in client_id else ""

        print(f"✅ Authenticated as {email or 'unknown'}")

        return GoogleCredentials(
            access_token=access_token,
            refresh_token=token_data["refresh_token"],
            expires_at=expires_at,
            client_id=client_id,
            client_secret=client_secret,
            project_id=project_id,
            email=email,
        )

    finally:
        server.shutdown()


async def get_credentials(auto_refresh: bool = True) -> Optional[GoogleCredentials]:
    """Get Google credentials from database.

    Users must run `syne init` to authenticate with their own Google account
    and their own OAuth client ID. No hardcoded credentials, no imports from
    other tools.

    Args:
        auto_refresh: Whether to auto-refresh expired tokens

    Returns:
        GoogleCredentials with valid access token, or None if not configured
    """
    # Load from database (only source)
    creds = await GoogleCredentials.load_from_db()
    if creds:
        if auto_refresh and creds.is_expired:
            await creds.get_token()  # Triggers refresh and saves to DB
        logger.info(f"Using DB credentials for {creds.email}")
        return creds

    # No credentials found — user must run `syne init`
    logger.warning("No Google OAuth credentials found. Run 'syne init' to authenticate.")
    return None
