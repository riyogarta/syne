"""Channel-agnostic error formatting for user-facing messages.

Policy (owner-approved):
- Sender with access_level "owner" or "family" (DM or group, any channel)
  sees the ORIGINAL error: ``⚠️ [<model>] <ExceptionType>: <str(e)>``.
- Everyone else gets one generic message (GENERIC_ERROR_MESSAGE).

Callers are responsible for logging the full error (logger.error with
exc_info) — this function only builds the text shown to the user.
"""

import json
import re

GENERIC_ERROR_MESSAGE = (
    "⚠️ There is a problem with the model currently in use. Please try again later."
)

# Keep the raw error under Telegram's 4096-char message limit.
_MAX_RAW_LEN = 3500
_TRUNC_SUFFIX = "… (truncated, full error in log)"

_PRIVILEGED_LEVELS = ("owner", "family")

_MSG_RE = re.compile(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"')
_PARAM_RE = re.compile(r'"param"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _humanize(text: str) -> str:
    """Collapse a provider error that embeds a raw JSON body into one line.

    "OpenAI 400: {"error": {"message": "...", "param": "x"}}"
    -> "OpenAI 400: ... (param: x)"

    Works on truncated JSON too (regex fallback). Anything without a JSON
    error message is returned unchanged.
    """
    idx = text.find("{")
    if idx < 0:
        return text
    prefix = text[:idx].strip().rstrip(":").strip()
    blob = text[idx:]
    message = param = None
    try:
        data = json.loads(blob)
        err = data.get("error", data) if isinstance(data, dict) else None
        if isinstance(err, dict):
            message, param = err.get("message"), err.get("param")
        elif isinstance(err, str):
            message = err
    except Exception:
        m = _MSG_RE.search(blob)
        if m:
            try:
                message = json.loads(f'"{m.group(1)}"')
            except Exception:
                message = m.group(1)
            pm = _PARAM_RE.search(blob)
            param = pm.group(1) if pm else None
    if not message or not isinstance(message, str):
        return text
    message = " ".join(message.split())
    out = f"{prefix}: {message}" if prefix else message
    if param and isinstance(param, str):
        out += f" (param: {param})"
    return out


def classify_error(e: Exception, model: str = "", access_level: str = "public") -> str:
    """Build the user-facing error message for an exception.

    Owner/family get the raw error (type + message, model-tagged);
    all other access levels get GENERIC_ERROR_MESSAGE.
    """
    if (access_level or "").lower() not in _PRIVILEGED_LEVELS:
        return GENERIC_ERROR_MESSAGE

    tag = f"[{model}] " if model else ""
    text = f"⚠️ {tag}{type(e).__name__}: {_humanize(str(e))}"
    if len(text) > _MAX_RAW_LEN:
        text = text[:_MAX_RAW_LEN] + _TRUNC_SUFFIX
    return text
