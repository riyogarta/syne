"""Channel-agnostic error formatting for user-facing messages.

Policy (owner-approved):
- Sender with access_level "owner" or "family" (DM or group, any channel)
  sees the ORIGINAL error: ``⚠️ [<model>] <ExceptionType>: <str(e)>``.
- Everyone else gets one generic message (GENERIC_ERROR_MESSAGE).

Callers are responsible for logging the full error (logger.error with
exc_info) — this function only builds the text shown to the user.
"""

GENERIC_ERROR_MESSAGE = (
    "⚠️ There is a problem with the model currently in use. Please try again later."
)

# Keep the raw error under Telegram's 4096-char message limit.
_MAX_RAW_LEN = 3500
_TRUNC_SUFFIX = "… (truncated, full error in log)"

_PRIVILEGED_LEVELS = ("owner", "family")


def classify_error(e: Exception, model: str = "", access_level: str = "public") -> str:
    """Build the user-facing error message for an exception.

    Owner/family get the raw error (type + message, model-tagged);
    all other access levels get GENERIC_ERROR_MESSAGE.
    """
    if (access_level or "").lower() not in _PRIVILEGED_LEVELS:
        return GENERIC_ERROR_MESSAGE

    tag = f"[{model}] " if model else ""
    text = f"⚠️ {tag}{type(e).__name__}: {e}"
    if len(text) > _MAX_RAW_LEN:
        text = text[:_MAX_RAW_LEN] + _TRUNC_SUFFIX
    return text
