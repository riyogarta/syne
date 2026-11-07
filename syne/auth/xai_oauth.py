"""xAI Grok OAuth (SuperGrok / X Premium+) for Syne.

Device-code flow (RFC 8628) against auth.x.ai — the same public client the
Hermes Agent uses. No callback URL, nothing to paste: the user opens a link,
approves, and Syne polls the token endpoint until the grant is issued.

Flow:
  1. POST /oauth2/device/code  → device_code, user_code, verification_uri
  2. User opens verification_uri_complete and approves
  3. Poll POST /oauth2/token (device_code grant) until access + refresh token
  4. Tokens saved to the DB (credential.xai_*)

Endpoints verified against https://auth.x.ai/.well-known/openid-configuration
(token_endpoint=https://auth.x.ai/oauth2/token,
 device_authorization_endpoint=https://auth.x.ai/oauth2/device/code).
"""

import asyncio
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger("syne.auth.xai")

XAI_ISSUER = "https://auth.x.ai"
XAI_DEVICE_CODE_URL = f"{XAI_ISSUER}/oauth2/device/code"
XAI_TOKEN_URL = f"{XAI_ISSUER}/oauth2/token"
XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_API_BASE_URL = "https://api.x.ai/v1"

_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
_FORM_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
}

# Config keys (DB is the source of truth)
CRED_ACCESS = "credential.xai_access_token"
CRED_REFRESH = "credential.xai_refresh_token"
CRED_EXPIRES = "credential.xai_expires_at"


class XaiOAuthError(RuntimeError):
    """xAI OAuth failure with the OAuth error code (if any)."""

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


def _error_text(resp: httpx.Response) -> tuple[str, str]:
    """Return (oauth_error_code, human description) from an error response."""
    try:
        data = resp.json()
    except Exception:
        return "", (resp.text or f"HTTP {resp.status_code}")[:300]
    if not isinstance(data, dict):
        return "", str(data)[:300]
    code = str(data.get("error") or "")
    desc = str(data.get("error_description") or code or f"HTTP {resp.status_code}")
    return code, desc[:300]


async def request_device_code() -> dict:
    """Start the device flow.

    Returns dict with: device_code, user_code, verification_uri,
    verification_uri_complete, expires_in, interval.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            XAI_DEVICE_CODE_URL,
            headers=_FORM_HEADERS,
            data={"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPE},
        )
    if resp.status_code != 200:
        code, desc = _error_text(resp)
        raise XaiOAuthError(f"xAI device-code request failed ({resp.status_code}): {desc}", code)

    data = resp.json()
    required = ("device_code", "user_code", "verification_uri", "expires_in", "interval")
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise XaiOAuthError(f"xAI device-code response missing: {', '.join(missing)}", "device_code_invalid")
    data.setdefault("verification_uri_complete", data["verification_uri"])
    return data


async def poll_device_token(
    device_code: str,
    interval: int,
    expires_in: int,
    cancel_event: Optional[asyncio.Event] = None,
) -> dict:
    """Poll the token endpoint until the user approves (RFC 8628 §3.4/3.5).

    Handles authorization_pending (keep waiting), slow_down (interval += 5),
    access_denied / expired_token (stop). Raises asyncio.CancelledError if
    cancel_event is set.

    Returns the raw token response (access_token, refresh_token, expires_in, ...).
    """
    interval = max(1, int(interval or 5))
    deadline = time.monotonic() + max(30, int(expires_in or 900))

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            # Wait first: RFC 8628 says the client must wait `interval`
            # seconds between requests, including before the first poll.
            try:
                if cancel_event is not None:
                    await asyncio.wait_for(cancel_event.wait(), timeout=interval)
                    raise asyncio.CancelledError("xAI device authorization cancelled")
                await asyncio.sleep(interval)
            except asyncio.TimeoutError:
                pass  # interval elapsed, not cancelled

            if time.monotonic() >= deadline:
                raise XaiOAuthError("Timed out waiting for xAI authorization.", "expired_token")

            try:
                resp = await client.post(
                    XAI_TOKEN_URL,
                    headers=_FORM_HEADERS,
                    data={
                        "grant_type": _DEVICE_GRANT,
                        "client_id": XAI_CLIENT_ID,
                        "device_code": device_code,
                    },
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                logger.warning(f"xAI token poll network error (will retry): {e}")
                continue

            if resp.status_code == 200:
                data = resp.json()
                if not data.get("access_token") or not data.get("refresh_token"):
                    raise XaiOAuthError("xAI token response missing access/refresh token.", "token_invalid")
                return data

            code, desc = _error_text(resp)
            if code == "authorization_pending":
                continue
            if code == "slow_down":
                interval += 5
                continue
            if code == "access_denied":
                raise XaiOAuthError("xAI authorization was denied.", code)
            if code == "expired_token":
                raise XaiOAuthError("xAI login code expired. Start again.", code)
            raise XaiOAuthError(f"xAI token polling failed ({resp.status_code}): {desc}", code)


def refresh_tokens_sync(refresh_token: str, timeout: float = 30) -> dict:
    """Refresh xAI tokens (sync — called from the provider's sync header path).

    Returns dict with access_token, refresh_token (rotated or the old one),
    expires_at. Raises XaiOAuthError on failure.
    """
    if not refresh_token:
        raise XaiOAuthError("No xAI refresh token. Re-authenticate via /models.", "missing_refresh_token")

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            XAI_TOKEN_URL,
            headers=_FORM_HEADERS,
            data={
                "grant_type": "refresh_token",
                "client_id": XAI_CLIENT_ID,
                "refresh_token": refresh_token,
            },
        )
    if resp.status_code != 200:
        code, desc = _error_text(resp)
        raise XaiOAuthError(f"xAI token refresh failed ({resp.status_code}): {desc}", code)

    data = resp.json()
    access = data.get("access_token")
    if not access:
        raise XaiOAuthError("xAI refresh response missing access_token.", "token_invalid")
    return {
        "access_token": access,
        # xAI may or may not rotate the refresh token — keep the old one if absent
        "refresh_token": data.get("refresh_token") or refresh_token,
        "expires_at": time.time() + float(data.get("expires_in") or 0),
    }


async def save_tokens(token_data: dict) -> float:
    """Persist a token response to the DB. Returns expires_at (unix)."""
    from ..db.models import set_config

    expires_at = float(token_data.get("expires_at") or 0) or (
        time.time() + float(token_data.get("expires_in") or 0)
    )
    await set_config(CRED_ACCESS, token_data["access_token"], "xAI OAuth access token")
    await set_config(CRED_REFRESH, token_data["refresh_token"], "xAI OAuth refresh token")
    await set_config(CRED_EXPIRES, expires_at, "xAI OAuth token expiry")
    return expires_at


async def load_tokens() -> tuple[str, str, float]:
    """Load (access_token, refresh_token, expires_at) from the DB."""
    from ..db.models import get_config

    access = await get_config(CRED_ACCESS, "") or ""
    refresh = await get_config(CRED_REFRESH, "") or ""
    expires_at = float(await get_config(CRED_EXPIRES, 0) or 0)
    if not access:
        raise RuntimeError(
            "No xAI OAuth tokens found. Add/re-authenticate the model via /models → xAI Grok (OAuth)."
        )
    return access, refresh, expires_at


async def login_xai() -> dict:
    """CLI device-code login (used by `syne reauth`).

    Returns dict with: access_token, refresh_token, expires_at
    """
    dev = await request_device_code()
    print("\n🔐 Sign in to xAI (SuperGrok / X Premium+):")
    print(f"\n   {dev['verification_uri_complete']}\n")
    print(f"   Code: {dev['user_code']}")
    print(f"\nWaiting for approval (expires in {int(dev['expires_in']) // 60} min)...")

    data = await poll_device_token(dev["device_code"], dev["interval"], dev["expires_in"])
    print("✅ xAI authentication successful")
    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_at": time.time() + float(data.get("expires_in") or 0),
    }
