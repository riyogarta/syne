"""xAI Grok provider (SuperGrok / X Premium+ OAuth, Responses API).

xAI's /v1/responses speaks the same streaming event format as the Codex
Responses backend (output_text.delta, output_item.added, function_call_
arguments.done, response.completed), so this reuses CodexProvider's request
and SSE parsing and only swaps the endpoint, headers and token refresh.
"""

import logging
import time
from typing import Optional

from .codex import CodexProvider, _REFRESH_BUFFER_SECONDS
from ..auth.xai_oauth import (
    XAI_API_BASE_URL, CRED_ACCESS, CRED_REFRESH, CRED_EXPIRES,
    XaiOAuthError, refresh_tokens_sync,
)

logger = logging.getLogger("syne.llm.xai")


class XaiProvider(CodexProvider):
    """xAI Grok via OAuth device-code login (uses the SuperGrok subscription).

    Tokens stored in DB (credential.xai_access_token, credential.xai_refresh_token,
    credential.xai_expires_at).
    """

    _responses_path = "/responses"

    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        chat_model: str = "grok-4.6",
        base_url: str = XAI_API_BASE_URL,
    ):
        super().__init__(
            access_token=access_token,
            refresh_token=refresh_token,
            chat_model=chat_model,
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return "xai"

    @property
    def context_window(self) -> int:
        # Registry entry's context_window is what Syne actually budgets with;
        # this is only the driver fallback.
        return 256_000

    def _refresh_token_sync(self) -> bool:
        """Refresh the xAI access token if expired or about to expire."""
        now = time.time()
        if self._token_expires_at > 0 and (now + _REFRESH_BUFFER_SECONDS) < self._token_expires_at:
            return False

        if not self.refresh_token:
            logger.warning("No refresh token available for xAI")
            return False

        try:
            data = refresh_tokens_sync(self.refresh_token)
        except XaiOAuthError as e:
            logger.error(str(e))
            self._auth_failure = (
                "xAI OAuth token refresh failed. Re-authenticate via /models → Re-authenticate."
            )
            return False
        except Exception as e:
            logger.error(f"xAI token refresh error: {e}")
            return False

        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self._token_expires_at = data["expires_at"]
        self._save_tokens_to_db(self.access_token, self.refresh_token, self._token_expires_at)
        logger.info("xAI OAuth token refreshed successfully")
        return True

    def _save_tokens_to_db(self, access_token: str, refresh_token: str, expires_at: float):
        """Save refreshed tokens to DB in background (same pattern as Codex)."""
        import asyncio
        import threading

        async def _save():
            try:
                from ..db.models import set_config
                await set_config(CRED_ACCESS, access_token)
                await set_config(CRED_REFRESH, refresh_token)
                await set_config(CRED_EXPIRES, expires_at)
                logger.debug("xAI tokens saved to DB")
            except Exception as e:
                logger.warning(f"Failed to save xAI tokens to DB: {e}")

        def _run():
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_save())
                loop.close()
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

    def _get_headers(self) -> dict:
        self._refresh_token_sync()
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
