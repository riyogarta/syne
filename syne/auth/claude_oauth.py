"""Claude/Anthropic OAuth for Syne.

Standalone OAuth flow — NO dependency on Claude Code CLI.
User logs in via browser with their claude.ai account.
Requires claude.ai Pro/Max/Team subscription.

Paste-code flow (works everywhere — no localhost, no polling):
  1. Open browser to claude.ai/oauth/authorize
  2. User authorizes → Anthropic console shows the code
  3. User pastes code back into terminal
  4. Exchange code for tokens
  5. Tokens saved to PostgreSQL database
  6. Auto-refresh when expired
"""

import asyncio
import base64
import hashlib
import logging
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger("syne.auth.claude")

# Claude OAuth — same as Claude Code CLI
_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_SCOPES = "org:create_api_key user:inference user:profile"
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
    """Run Claude OAuth paste-code flow.

    Works everywhere — no localhost server, no polling.
    User opens URL in any browser, Anthropic console shows the code,
    user pastes it back.
    """
    verifier, challenge = _generate_pkce()

    params = urlencode({
        "code": "true",
        "client_id": _CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    })
    auth_url = f"{_AUTHORIZE_URL}?{params}"

    print("\n  Open this URL in your browser to sign in with Claude:\n")
    print(f"    {auth_url}\n")
    print("  After authorizing, the page will display a code.")
    print("  Copy and paste it below.\n")

    pasted = input("  Paste code: ").strip()
    if not pasted:
        raise RuntimeError("No code provided.")

    # Parse: may be "code#state" or just "code"
    if "#" in pasted:
        code, state = pasted.split("#", 1)
    else:
        code = pasted
        state = verifier

    print("\n  Exchanging authorization code...")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": _CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": verifier,
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

    print(f"  Claude authentication successful{f' ({email})' if email else ''}")

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
