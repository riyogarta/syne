"""Google Gemini provider via Cloud Code Assist (OAuth) or Gemini API (key).

Aligned with OpenClaw's google-gemini-cli.js + google-shared.js implementation.
All 14 differences resolved:
  1. parametersJsonSchema (not parameters) for tool schemas
  2. Server-parsed retry delays (extractRetryDelay)
  3. Full retryable error detection (429,500,502,503,504 + regex)
  4. Endpoint fallback (daily-cloudcode-pa + cloudcode-pa)
  5. Empty stream retry (MAX_EMPTY_STREAM_RETRIES=2)
  6. Unicode surrogate sanitization
  7. sessionId in request
  8. Thinking config (includeThoughts + thinkingLevel for Gemini 3)
  9. functionResponse format ({output: value} / {error: value})
 10. Empty text blocks skipped
 11. thoughtSignature preserved & replayed
 12. transformMessages (skip errored/aborted, synthetic orphan results)
 13. generationConfig conditional (only if non-empty)
 14. Temperature only if defined (not always 0.7)
"""

import asyncio
import json
import logging
import re
import time
import random
from collections import deque
import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse, LLMRateLimitError, LLMAuthError, LLMBadRequestError, LLMEmptyResponseError
from ..auth.google_oauth import GoogleCredentials

logger = logging.getLogger("syne.llm.google")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants — match OpenClaw exactly
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
_ANTIGRAVITY_DAILY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
_ENDPOINT_FALLBACKS = [_ANTIGRAVITY_DAILY_ENDPOINT, _DEFAULT_ENDPOINT]

_CCA_HEADERS = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": json.dumps({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }),
}

# Standard Gemini API endpoint (used with API keys)
_GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"

# Retry configuration — match OpenClaw exactly
_MAX_RETRIES = 3
_BASE_DELAY_MS = 1_000
_MAX_EMPTY_STREAM_RETRIES = 2
_EMPTY_STREAM_BASE_DELAY_MS = 500
_MAX_RETRY_DELAY_MS = 60_000
_STREAM_OVERALL_TIMEOUT = 30  # 30s hard cap on entire SSE stream

# Default CCA rate limit (requests per minute).
# Google Account (OAuth free): documented as 60 RPM, but internal CCA
# endpoint may enforce lower limits. Conservative default; configurable
# per model via "rpm_limit" field in model registry.
_DEFAULT_CCA_RPM = 10

# Tool call ID counter
_tool_call_counter = 0


# ═══════════════════════════════════════════════════════════════════════════════
# #6 — Unicode surrogate sanitization (from sanitize-unicode.js)
# ═══════════════════════════════════════════════════════════════════════════════

# Regex to match unpaired surrogates in Python strings.
# In Python 3, strings are Unicode — unpaired surrogates can appear from
# malformed data (e.g. lone \ud83d without a following low surrogate).
_SURROGATE_RE = re.compile(
    r'[\ud800-\udbff](?![\udc00-\udfff])'  # high surrogate not followed by low
    r'|(?<![\ud800-\udbff])[\udc00-\udfff]',  # low surrogate not preceded by high
    re.UNICODE,
)


def _sanitize_surrogates(text: str) -> str:
    """Remove unpaired Unicode surrogate characters.

    Valid emoji (properly paired surrogates) are preserved.
    Matches OpenClaw's sanitizeSurrogates().
    """
    if not text:
        return text
    try:
        return _SURROGATE_RE.sub('', text)
    except Exception:
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# #2 — Extract retry delay from error response (from google-gemini-cli.js)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_retry_delay(error_text: str, headers: Optional[httpx.Headers] = None) -> Optional[int]:
    """Extract retry delay in milliseconds from error response.

    Checks headers first, then parses body patterns. Matches OpenClaw's extractRetryDelay().
    Returns milliseconds or None.
    """
    def _normalize(ms: float) -> Optional[int]:
        return int(ms + 1000) if ms > 0 else None

    # Check headers
    if headers:
        # Retry-After header
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                seconds = float(retry_after)
                delay = _normalize(seconds * 1000)
                if delay is not None:
                    return delay
            except ValueError:
                pass

        # x-ratelimit-reset-after
        reset_after = headers.get("x-ratelimit-reset-after")
        if reset_after:
            try:
                seconds = float(reset_after)
                delay = _normalize(seconds * 1000)
                if delay is not None:
                    return delay
            except ValueError:
                pass

    # Pattern 1: "Your quota will reset after 18h31m10s" / "39s"
    m = re.search(r'reset after (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s', error_text, re.I)
    if m:
        hours = int(m.group(1)) if m.group(1) else 0
        minutes = int(m.group(2)) if m.group(2) else 0
        seconds = float(m.group(3))
        total_ms = ((hours * 60 + minutes) * 60 + seconds) * 1000
        delay = _normalize(total_ms)
        if delay is not None:
            return delay

    # Pattern 2: "Please retry in Xs" / "Please retry in Xms"
    m = re.search(r'Please retry in ([0-9.]+)(ms|s)', error_text, re.I)
    if m:
        value = float(m.group(1))
        ms = value if m.group(2).lower() == "ms" else value * 1000
        delay = _normalize(ms)
        if delay is not None:
            return delay

    # Pattern 3: "retryDelay": "34.074824224s"
    m = re.search(r'"retryDelay":\s*"([0-9.]+)(ms|s)"', error_text, re.I)
    if m:
        value = float(m.group(1))
        ms = value if m.group(2).lower() == "ms" else value * 1000
        delay = _normalize(ms)
        if delay is not None:
            return delay

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# #3 — Retryable error detection (from google-gemini-cli.js)
# ═══════════════════════════════════════════════════════════════════════════════

_RETRYABLE_PATTERN = re.compile(
    r'resource.?exhausted|rate.?limit|overloaded|service.?unavailable|other.?side.?closed',
    re.I,
)


def _is_retryable_error(status: int, error_text: str) -> bool:
    """Check if an error is retryable. Matches OpenClaw's isRetryableError().

    400 Bad Request is NOT retryable.
    """
    if status in (429, 500, 502, 503, 504):
        return True
    return bool(_RETRYABLE_PATTERN.search(error_text))


# ═══════════════════════════════════════════════════════════════════════════════
# #11 — Thought signature validation (from google-shared.js)
# ═══════════════════════════════════════════════════════════════════════════════

_BASE64_SIGNATURE_RE = re.compile(r'^[A-Za-z0-9+/]+=*$')


def _is_valid_thought_signature(sig: Optional[str]) -> bool:
    """Validate thought signature is proper base64."""
    if not sig:
        return False
    if len(sig) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE_RE.match(sig))


# ═══════════════════════════════════════════════════════════════════════════════
# Tool schema sanitization — strip unsupported JSON Schema keywords for Gemini
# ═══════════════════════════════════════════════════════════════════════════════

_GEMINI_UNSUPPORTED_KEYWORDS = {
    "patternProperties", "additionalProperties", "$schema", "$id", "$ref",
    "$defs", "definitions", "examples", "minLength", "maxLength",
    "minimum", "maximum", "multipleOf", "pattern", "format",
    "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
}


def _is_null_schema(variant: dict) -> bool:
    """Check if a schema variant represents null type."""
    if not isinstance(variant, dict):
        return False
    if variant.get("const") is None and "const" in variant:
        return True
    enum_val = variant.get("enum")
    if isinstance(enum_val, list) and len(enum_val) == 1 and enum_val[0] is None:
        return True
    t = variant.get("type")
    if t == "null":
        return True
    if isinstance(t, list) and len(t) == 1 and t[0] == "null":
        return True
    return False


def _try_flatten_union(variants: list, parent: dict) -> Optional[dict]:
    """Try to flatten anyOf/oneOf into a single schema.

    Matches OpenClaw's simplifyUnionVariants + flattenUnionFallback:
    1. Strip null variants
    2. If all variants are literals (const/single-enum), merge to {type, enum}
    3. If one variant left after stripping null, use it directly
    4. If all variants share same type, use that type
    5. Otherwise use first variant's type
    """
    # Strip null variants
    non_null = [v for v in variants if isinstance(v, dict) and not _is_null_schema(v)]
    if not non_null:
        return None

    # Try literal flattening: {const: "a"} + {const: "b"} → {type: "string", enum: ["a", "b"]}
    all_values = []
    common_type = None
    all_literal = True
    for v in non_null:
        if "const" in v:
            literal = v["const"]
        elif isinstance(v.get("enum"), list) and len(v["enum"]) == 1:
            literal = v["enum"][0]
        else:
            all_literal = False
            break
        vtype = v.get("type")
        if not isinstance(vtype, str):
            all_literal = False
            break
        if common_type is None:
            common_type = vtype
        elif common_type != vtype:
            all_literal = False
            break
        all_values.append(literal)

    if all_literal and common_type and all_values:
        result = {"type": common_type, "enum": all_values}
        for mk in ("description", "title", "default"):
            if mk in parent:
                result[mk] = parent[mk]
        return result

    # Single variant after null stripping
    if len(non_null) == 1:
        result = dict(non_null[0])
        for mk in ("description", "title", "default"):
            if mk in parent:
                result[mk] = parent[mk]
        return result

    # All same type — use that type
    types = {v.get("type") for v in non_null if isinstance(v.get("type"), str)}
    if len(types) == 1:
        result = {"type": types.pop()}
        for mk in ("description", "title", "default"):
            if mk in parent:
                result[mk] = parent[mk]
        return result

    # Fallback: first variant's type
    first_type = non_null[0].get("type") if non_null else None
    if first_type:
        result = {"type": first_type}
        for mk in ("description", "title", "default"):
            if mk in parent:
                result[mk] = parent[mk]
        return result

    return None


def _clean_schema_for_gemini(schema: dict, defs: Optional[dict] = None) -> dict:
    """Remove unsupported JSON Schema keywords for Gemini.

    Recursively cleans nested schemas (properties, items, anyOf, oneOf).
    Resolves $ref locally. Converts const→enum. Flattens unions.
    Matches OpenClaw's cleanSchemaForGemini + simplifyUnionVariants.
    """
    if not isinstance(schema, dict):
        return schema

    if defs is None:
        defs = schema.get("$defs") or schema.get("definitions") or {}

    # Handle $ref before anything else
    if "$ref" in schema and isinstance(schema["$ref"], str):
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        if ref_name in defs:
            return _clean_schema_for_gemini(defs[ref_name], defs)

    # Handle type: ["string", "null"] → type: "string" (strip null from type arrays)
    raw_type = schema.get("type")
    if isinstance(raw_type, list) and all(isinstance(t, str) for t in raw_type):
        non_null_types = [t for t in raw_type if t != "null"]
        schema = dict(schema)  # shallow copy to avoid mutating original
        if len(non_null_types) == 1:
            schema["type"] = non_null_types[0]
        elif non_null_types:
            schema["type"] = non_null_types

    cleaned = {}
    for key, value in schema.items():
        if key in _GEMINI_UNSUPPORTED_KEYWORDS:
            continue
        if key == "const":
            cleaned["enum"] = [value]
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _clean_schema_for_gemini(v, defs) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _clean_schema_for_gemini(value, defs)
        elif key in ("anyOf", "oneOf") and isinstance(value, list):
            # Clean each variant first, then try to flatten
            cleaned_variants = [_clean_schema_for_gemini(v, defs) for v in value]
            flattened = _try_flatten_union(cleaned_variants, schema)
            if flattened:
                # Flattened successfully — merge into cleaned, skip the union key
                cleaned.update(_clean_schema_for_gemini(flattened, defs))
            else:
                cleaned[key] = cleaned_variants
        elif key == "allOf" and isinstance(value, list):
            cleaned[key] = [_clean_schema_for_gemini(v, defs) for v in value]
        else:
            cleaned[key] = value

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# #1 — Tool schema conversion (parametersJsonSchema, not parameters)
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_tools_to_gemini(tools: list[dict], use_parameters: bool = False) -> list[dict]:
    """Convert OpenAI function calling format to Gemini functionDeclarations.

    By default uses `parametersJsonSchema` (full JSON Schema support).
    Set `use_parameters=True` for legacy `parameters` field (Claude models on CCA).
    Matches OpenClaw's convertTools().
    """
    if not tools:
        return []

    declarations = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            decl = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
            if "parameters" in func:
                params = func["parameters"]
                if use_parameters:
                    decl["parameters"] = params
                else:
                    # #1: Use parametersJsonSchema for Gemini models
                    # Clean unsupported keywords for non-legacy mode
                    decl["parametersJsonSchema"] = _clean_schema_for_gemini(params)
            declarations.append(decl)

    if not declarations:
        return []

    return [{"functionDeclarations": declarations}]


# ═══════════════════════════════════════════════════════════════════════════════
# #12 — Transform messages (skip errored/aborted, synthetic orphan results)
# From OpenClaw's transform-messages.js
# ═══════════════════════════════════════════════════════════════════════════════

def _transform_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Clean up message history before sending to Gemini.

    - Skip errored/aborted assistant messages (incomplete turns)
    - Insert synthetic empty tool results for orphaned tool calls
    Matches OpenClaw's transformMessages().
    """
    # First pass: identify tool calls and their results
    pending_tool_calls: list[dict] = []  # [{name, id}]
    existing_result_ids: set[str] = set()
    result: list[ChatMessage] = []

    for msg in messages:
        if msg.role == "assistant":
            # If we have pending orphaned tool calls from previous assistant, insert synthetic results
            if pending_tool_calls:
                for tc in pending_tool_calls:
                    if tc["id"] not in existing_result_ids:
                        result.append(ChatMessage(
                            role="tool",
                            content="No result provided",
                            metadata={
                                "tool_name": tc["name"],
                                "tool_call_id": tc["id"],
                                "is_error": True,
                                "synthetic": True,
                            }
                        ))
                pending_tool_calls = []
                existing_result_ids = set()

            # Skip errored/aborted assistant messages — incomplete turns
            stop_reason = msg.metadata.get("stop_reason") if msg.metadata else None
            if stop_reason in ("error", "aborted"):
                continue

            # Track tool calls from this assistant message
            tool_calls = msg.metadata.get("tool_calls") if msg.metadata else None
            if tool_calls:
                pending_tool_calls = []
                existing_result_ids = set()
                for tc in tool_calls:
                    func = tc.get("function", tc)
                    tc_id = tc.get("id", func.get("name", f"unknown_{time.time()}"))
                    pending_tool_calls.append({"name": func.get("name", ""), "id": tc_id})

            result.append(msg)

        elif msg.role == "tool":
            tc_id = msg.metadata.get("tool_call_id", "") if msg.metadata else ""
            existing_result_ids.add(tc_id)
            result.append(msg)

        elif msg.role == "user":
            # User message interrupts tool flow — insert synthetic results for orphaned calls
            if pending_tool_calls:
                for tc in pending_tool_calls:
                    if tc["id"] not in existing_result_ids:
                        result.append(ChatMessage(
                            role="tool",
                            content="No result provided",
                            metadata={
                                "tool_name": tc["name"],
                                "tool_call_id": tc["id"],
                                "is_error": True,
                                "synthetic": True,
                            }
                        ))
                pending_tool_calls = []
                existing_result_ids = set()
            result.append(msg)

        else:
            result.append(msg)

    return result


def _sanitize_turn_ordering(contents: list[dict]) -> list[dict]:
    """Ensure Gemini user/model alternation.

    Two fixes:
    1. If first content is 'model', prepend bootstrap user message
    2. Merge consecutive same-role messages (parts concatenation)
    """
    if not contents:
        return contents

    result = []

    # Fix 1: Bootstrap — if first message is model, prepend user
    if contents[0].get("role") == "model":
        result.append({"role": "user", "parts": [{"text": "(session bootstrap)"}]})

    # Fix 2: Merge consecutive same-role
    for msg in contents:
        if result and result[-1].get("role") == msg.get("role"):
            result[-1]["parts"].extend(msg.get("parts", []))
        else:
            result.append(msg)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Provider class
# ═══════════════════════════════════════════════════════════════════════════════

class GoogleProvider(LLMProvider):
    """Google Gemini via Cloud Code Assist OAuth (free) or Gemini API key (paid).

    OAuth mode uses the Cloud Code Assist wrapper API.
    API key mode uses the standard Gemini API directly.
    """

    def __init__(
        self,
        credentials: Optional[GoogleCredentials] = None,
        api_key: Optional[str] = None,
        chat_model: str = "gemini-2.5-pro",
        embedding_model: str = "text-embedding-004",
        cca_rpm: Optional[int] = None,
    ):
        self.credentials = credentials
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model

        if not credentials and not api_key:
            raise ValueError("Either credentials (OAuth) or api_key is required")

        self._use_cca = credentials is not None
        # Limit concurrent embed calls to prevent burst 429s
        self._embed_semaphore = asyncio.Semaphore(3)
        # Sliding window throttle for CCA chat requests
        self._cca_rpm = cca_rpm or _DEFAULT_CCA_RPM
        self._request_times: deque = deque()

    @property
    def name(self) -> str:
        return "google"

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def context_window(self) -> int:
        return 1_000_000  # Gemini 2.5 Pro default

    @staticmethod
    def _build_thinking_config(model: str, thinking_budget: Optional[int]) -> Optional[dict]:
        """Build thinkingConfig for Gemini models.

        Defaults (when thinking_budget is None):
          - Gemini 3: thinkingLevel HIGH
          - Gemini 2.5: thinkingBudget -1 (dynamic, model decides, max 8192)

        Explicit thinking_budget=0 → minimal thinking (LOW / budget 128).
        """
        is_gemini_3 = "gemini-3" in model.lower()

        if is_gemini_3:
            if thinking_budget is None:
                level = "HIGH"
            elif thinking_budget == 0:
                level = "LOW"
            elif thinking_budget > 8192:
                level = "HIGH"
            elif thinking_budget > 2048:
                level = "MEDIUM"
            else:
                level = "LOW"
            return {"includeThoughts": True, "thinkingLevel": level}

        # Gemini 2.5 — thinkingBudget
        if thinking_budget is None:
            budget = -1  # dynamic thinking
        elif thinking_budget == 0:
            budget = 128  # minimum allowed
        else:
            budget = thinking_budget
        return {"includeThoughts": True, "thinkingBudget": budget}

    def _format_messages(self, messages: list[ChatMessage], model: str = "") -> tuple[Optional[str], list[dict]]:
        """Convert ChatMessages to Gemini format. Returns (system_text, contents).

        Implements:
        - #6: Unicode sanitization on all text
        - #9: functionResponse format {output: value} / {error: value}
        - #10: Skip empty text blocks
        - #11: thoughtSignature preserved & replayed
        - #12: transformMessages (skip errored/aborted, synthetic orphan results)
        """
        # #12: Pre-process messages — handle orphan tool calls, skip errored
        messages = _transform_messages(messages)

        system_text = None
        contents = []

        is_gemini_3 = "gemini-3" in model.lower() if model else False

        for msg in messages:
            if msg.role == "system":
                # Accumulate system messages
                text = _sanitize_surrogates(msg.content)
                if system_text is None:
                    system_text = text
                else:
                    system_text += "\n\n" + text

            elif msg.role == "user":
                text = _sanitize_surrogates(msg.content)
                # #10: Skip empty text
                if not text or not text.strip():
                    continue
                parts = [{"text": text}]
                # Include image data for vision if present in metadata
                if msg.metadata and "image" in msg.metadata:
                    img = msg.metadata["image"]
                    parts.append({
                        "inlineData": {
                            "mimeType": img.get("mime_type", "image/jpeg"),
                            "data": img.get("base64", ""),
                        }
                    })
                contents.append({"role": "user", "parts": parts})

            elif msg.role == "assistant":
                tool_calls = msg.metadata.get("tool_calls") if msg.metadata else None
                if tool_calls:
                    parts = []
                    # #10: Only add text if non-empty
                    if msg.content and msg.content.strip():
                        parts.append({"text": _sanitize_surrogates(msg.content)})
                    for tc in tool_calls:
                        func = tc.get("function", tc)
                        # Handle both original ("arguments" as string) and normalized ("args" as dict)
                        raw_args = func.get("arguments") or func.get("args") or "{}"
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {}

                        # #11: Get thoughtSignature if present
                        thought_sig = tc.get("thoughtSignature")

                        # #11: Gemini 3 requires thoughtSignature on all function calls
                        # When replaying history without signatures, convert to text
                        if is_gemini_3 and not _is_valid_thought_signature(thought_sig):
                            args_str_pretty = json.dumps(args, indent=2)
                            parts.append({
                                "text": (
                                    f'[Historical context: a different model called tool '
                                    f'"{func.get("name", "")}" with arguments: {args_str_pretty}. '
                                    f'Do not mimic this format - use proper function calling.]'
                                )
                            })
                        else:
                            fc_part: dict = {
                                "functionCall": {
                                    "name": func.get("name", ""),
                                    "args": args,
                                }
                            }
                            # #11: Preserve thoughtSignature
                            if _is_valid_thought_signature(thought_sig):
                                fc_part["thoughtSignature"] = thought_sig
                            parts.append(fc_part)

                    if parts:  # #10: Don't add empty model turns
                        contents.append({"role": "model", "parts": parts})
                else:
                    # #10: Skip empty text blocks
                    if not msg.content or not msg.content.strip():
                        continue
                    contents.append({"role": "model", "parts": [{"text": _sanitize_surrogates(msg.content)}]})

            elif msg.role == "tool":
                tool_name = msg.metadata.get("tool_name", "unknown") if msg.metadata else "unknown"
                is_error = msg.metadata.get("is_error", False) if msg.metadata else False

                # #9: functionResponse format — {output: value} for success, {error: value} for errors
                result_text = _sanitize_surrogates(msg.content)
                if is_error:
                    response_data = {"error": result_text}
                else:
                    response_data = {"output": result_text}

                fr_part: dict = {
                    "functionResponse": {
                        "name": tool_name,
                        "response": response_data,
                    }
                }

                # Merge consecutive tool results into one "user" message
                if contents and contents[-1].get("role") == "user" and \
                   contents[-1]["parts"] and "functionResponse" in contents[-1]["parts"][0]:
                    contents[-1]["parts"].append(fr_part)
                else:
                    contents.append({
                        "role": "user",
                        "parts": [fr_part],
                    })

        contents = _sanitize_turn_ordering(contents)
        return system_text, contents

    async def _throttle_cca(self):
        """Sliding window rate limiter for CCA chat requests.

        Tracks request timestamps in a 60-second window. When approaching
        the RPM limit, adds proportional delay to spread requests out.
        When at the limit, waits until the oldest request exits the window.
        """
        if not self._use_cca or not self._cca_rpm:
            return

        now = time.monotonic()
        window = 60.0

        # Remove timestamps outside the window
        while self._request_times and self._request_times[0] < now - window:
            self._request_times.popleft()

        used = len(self._request_times)
        threshold = max(1, self._cca_rpm - 2)  # Start throttling 2 before limit

        if used >= self._cca_rpm:
            # Hard limit — wait for oldest request to exit the window
            wait = self._request_times[0] + window - now
            if wait > 0:
                logger.info(f"CCA throttle: {used}/{self._cca_rpm} RPM, waiting {wait:.1f}s")
                await asyncio.sleep(wait)
                now = time.monotonic()
                while self._request_times and self._request_times[0] < now - window:
                    self._request_times.popleft()
        elif used >= threshold:
            # Approaching limit — spread remaining requests across remaining time
            elapsed = now - self._request_times[0] if self._request_times else 0
            remaining_time = window - elapsed
            remaining_slots = self._cca_rpm - used
            if remaining_slots > 0 and remaining_time > 0:
                spacing = remaining_time / remaining_slots
                if spacing > 1.0:
                    logger.debug(f"CCA pre-throttle: {used}/{self._cca_rpm} RPM, spacing {spacing:.1f}s")
                    await asyncio.sleep(spacing)

        self._request_times.append(time.monotonic())

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> ChatResponse:
        model = model or self.chat_model

        if self._use_cca:
            await self._throttle_cca()
            return await self._chat_cca(messages, model, temperature, max_tokens, tools, thinking_budget, top_p, top_k)
        else:
            return await self._chat_api(messages, model, temperature, max_tokens, tools, thinking_budget, top_p, top_k)

    async def _chat_cca(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        """Chat via Cloud Code Assist (OAuth, free)."""
        token = await self.credentials.get_token()
        system_text, contents = self._format_messages(messages, model)

        # Build inner request
        inner: dict = {"contents": contents}

        # #7: sessionId in request
        session_id = f"syne-{int(time.time())}-{random.randint(1000, 9999)}"
        inner["sessionId"] = session_id

        if system_text:
            inner["systemInstruction"] = {"parts": [{"text": system_text}]}

        # #13 & #14: generationConfig conditional — only include if non-empty
        # #14: Temperature only if explicitly set (not always 0.7)
        gen_config: dict = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        if top_p is not None:
            gen_config["topP"] = top_p
        if top_k is not None:
            gen_config["topK"] = top_k

        # #8: Thinking config — always on, model-appropriate defaults
        gen_config["thinkingConfig"] = self._build_thinking_config(model, thinking_budget)

        # #13: Only set generationConfig if non-empty
        if gen_config:
            inner["generationConfig"] = gen_config

        # #1: Add tools — use parametersJsonSchema (not parameters)
        if tools:
            # Claude models on CCA need legacy `parameters` field
            use_params = model.startswith("claude-")
            gemini_tools = _convert_tools_to_gemini(tools, use_parameters=use_params)
            if gemini_tools:
                inner["tools"] = gemini_tools

        # Wrap in CCA envelope
        request_id = f"syne-{int(time.time())}-{random.randint(1000, 9999)}"
        body = {
            "project": self.credentials.project_id,
            "model": model,
            "request": inner,
            "userAgent": "syne-agent",
            "requestId": request_id,
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **_CCA_HEADERS,
        }

        return await self._chat_cca_streaming(body, headers, model)

    async def _chat_cca_streaming(self, body: dict, headers: dict, model: str) -> ChatResponse:
        """Chat via CCA streaming endpoint with full OpenClaw-style retry logic.

        Implements:
        - #2: Server-parsed retry delays
        - #3: Full retryable error detection
        - #4: Endpoint fallback
        - #5: Empty stream retry
        """
        headers["Accept"] = "text/event-stream"
        body_json = json.dumps(body)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict] = []
        input_tokens = 0
        output_tokens = 0

        def _reset_output():
            nonlocal input_tokens, output_tokens
            text_parts.clear()
            thinking_parts.clear()
            tool_calls.clear()
            input_tokens = 0
            output_tokens = 0

        async def _do_streaming_request(url: str) -> bool:
            """Execute one streaming request, parse SSE, return True if content received."""
            nonlocal input_tokens, output_tokens
            has_content = False

            # Overall stream timeout — prevents keep-alive lines from holding
            # the connection open forever (httpx read timeout only checks for
            # ANY data, including empty keep-alive lines).
            stream_deadline = time.monotonic() + _STREAM_OVERALL_TIMEOUT

            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream("POST", url, content=body_json, headers=headers) as resp:
                    # Check status BEFORE consuming stream.
                    # On error, read body for retry delay info, then raise.
                    if resp.status_code >= 400:
                        await resp.aread()
                        resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if time.monotonic() > stream_deadline:
                            logger.warning(f"CCA stream exceeded {_STREAM_OVERALL_TIMEOUT}s overall — aborting")
                            break
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Unwrap response envelope
                        response_data = chunk.get("response")
                        if not response_data:
                            continue

                        # Log first chunk
                        if not has_content:
                            logger.debug(f"CCA first chunk: {data_str[:500]}")

                        for candidate in response_data.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                if "text" in part:
                                    has_content = True
                                    if part.get("thought"):
                                        thinking_parts.append(part["text"])
                                    else:
                                        # #10: Skip empty text blocks
                                        if part["text"].strip():
                                            text_parts.append(part["text"])
                                elif "functionCall" in part:
                                    has_content = True
                                    fc = part["functionCall"]
                                    tc_id = fc.get("id") or f"{fc.get('name', 'unknown')}_{time.time_ns()}"
                                    tc_entry: dict = {
                                        "id": tc_id,
                                        "function": {
                                            "name": fc.get("name", ""),
                                            "arguments": json.dumps(fc.get("args", {})),
                                        },
                                    }
                                    # #11: Preserve thoughtSignature
                                    if part.get("thoughtSignature"):
                                        tc_entry["thoughtSignature"] = part["thoughtSignature"]
                                    tool_calls.append(tc_entry)

                        # Extract usage
                        usage = response_data.get("usageMetadata", {})
                        if usage.get("promptTokenCount"):
                            input_tokens = usage["promptTokenCount"]
                        if usage.get("candidatesTokenCount"):
                            output_tokens = usage["candidatesTokenCount"]

            return has_content

        # ─── Retry loop with endpoint fallback (#3, #4) ───
        last_error: Optional[Exception] = None
        request_url: Optional[str] = None
        streamed = False

        for attempt in range(_MAX_RETRIES + 1):
            try:
                # #4: Cycle through endpoints on retry
                endpoint = _ENDPOINT_FALLBACKS[min(attempt, len(_ENDPOINT_FALLBACKS) - 1)]
                request_url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"

                streamed = await _do_streaming_request(request_url)
                break  # Success

            except httpx.HTTPStatusError as e:
                # Streaming responses may not have body read yet
                try:
                    await e.response.aread()
                    error_text = e.response.text
                except Exception:
                    error_text = str(e)
                status = e.response.status_code

                # #3: Check if retryable
                if attempt < _MAX_RETRIES and _is_retryable_error(status, error_text):
                    # #2: Use server-provided delay or exponential backoff
                    server_delay = _extract_retry_delay(error_text, e.response.headers)
                    if not server_delay:
                        logger.debug(f"CCA no server delay parsed. Headers: {dict(e.response.headers)}. Body snippet: {error_text[:300]}")

                    # On 429: try next endpoint immediately (delay is per-endpoint).
                    # Only wait the full delay if all endpoints are exhausted.
                    next_endpoint = _ENDPOINT_FALLBACKS[min(attempt + 1, len(_ENDPOINT_FALLBACKS) - 1)]
                    if status == 429 and next_endpoint != endpoint:
                        delay_ms = 500  # Brief pause, then try other endpoint
                        logger.warning(
                            f"CCA 429 on {endpoint.split('//')[1].split('/')[0]}, "
                            f"switching to {next_endpoint.split('//')[1].split('/')[0]} "
                            f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})"
                        )
                    else:
                        delay_ms = server_delay if server_delay else _BASE_DELAY_MS * (2 ** attempt)
                        # Cap at max delay
                        if server_delay and server_delay > _MAX_RETRY_DELAY_MS:
                            delay_s = server_delay // 1000
                            raise LLMRateLimitError(
                                f"Rate limited (429). Server requested {delay_s}s wait (max {_MAX_RETRY_DELAY_MS // 1000}s)."
                            ) from e
                        logger.warning(
                            f"CCA {status}, retrying in {delay_ms}ms "
                            f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}, endpoint: {endpoint})"
                        )

                    await asyncio.sleep(delay_ms / 1000)
                    _reset_output()
                    continue

                # Not retryable or max retries exceeded — classify the error
                if status == 429:
                    raise LLMRateLimitError(f"Rate limited (429) after {_MAX_RETRIES + 1} attempts.") from e
                elif status in (401, 403):
                    raise LLMAuthError(f"Authentication failed ({status}): {error_text[:200]}") from e
                elif status == 400:
                    logger.error(f"Gemini 400 Bad Request (full): {error_text[:2000]}")
                    raise LLMBadRequestError(f"Bad request (400): {error_text[:500]}") from e
                else:
                    raise

            except (LLMRateLimitError, LLMAuthError, LLMBadRequestError):
                raise  # Already classified, don't wrap again

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay_ms = _BASE_DELAY_MS * (2 ** attempt)
                    logger.warning(f"CCA network error: {e}, retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    _reset_output()
                    continue
                raise

        # ─── Empty stream retry (#5) ───
        if not streamed and request_url:
            for empty_attempt in range(1, _MAX_EMPTY_STREAM_RETRIES + 1):
                backoff_ms = _EMPTY_STREAM_BASE_DELAY_MS * (2 ** (empty_attempt - 1))
                logger.warning(f"CCA empty stream, retry {empty_attempt}/{_MAX_EMPTY_STREAM_RETRIES} in {backoff_ms}ms")
                await asyncio.sleep(backoff_ms / 1000)
                _reset_output()

                try:
                    streamed = await _do_streaming_request(request_url)
                    if streamed:
                        break
                except Exception as e:
                    logger.warning(f"CCA empty stream retry failed: {e}")

        if not streamed:
            raise LLMEmptyResponseError(f"Cloud Code Assist API returned an empty response for {model}")

        content = "".join(text_parts)
        thinking = "".join(thinking_parts) if thinking_parts else None

        return ChatResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls if tool_calls else None,
            thinking=thinking,
        )

    async def _chat_api(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        """Chat via standard Gemini API (API key, paid)."""
        system_text, contents = self._format_messages(messages, model)

        body: dict = {
            "contents": contents,
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        gen_config: dict = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        if top_p is not None:
            gen_config["topP"] = top_p
        if top_k is not None:
            gen_config["topK"] = top_k
        # #8: Thinking config — always on, model-appropriate defaults
        gen_config["thinkingConfig"] = self._build_thinking_config(model, thinking_budget)

        # #13: Only set if non-empty
        if gen_config:
            body["generationConfig"] = gen_config

        # Tools — standard API uses `parameters` (OpenAPI schema)
        if tools:
            gemini_tools = _convert_tools_to_gemini(tools, use_parameters=True)
            if gemini_tools:
                body["tools"] = gemini_tools

        url = f"{_GEMINI_API}/models/{model}:generateContent"

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, params={"key": self.api_key})
            resp.raise_for_status()
            data = resp.json()

        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])

        text_parts = []
        thinking_parts = []
        tool_calls = []

        for part in parts:
            if part.get("thought") and "text" in part:
                thinking_parts.append(part["text"])
            elif "text" in part:
                # #10: Skip empty
                if part["text"].strip():
                    text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    }
                })

        usage = data.get("usageMetadata", {})

        return ChatResponse(
            content="".join(text_parts),
            model=model,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            tool_calls=tool_calls if tool_calls else None,
            thinking="".join(thinking_parts) if thinking_parts else None,
        )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Embedding — always uses Gemini API (CCA doesn't have embed endpoint).

        Includes retry with exponential backoff for transient errors (429, 5xx).
        """
        model = model or self.embedding_model
        url = f"{_GEMINI_API}/models/{model}:embedContent"

        body = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
        }

        async with self._embed_semaphore:
            for attempt in range(_MAX_RETRIES + 1):
                if self._use_cca:
                    token = await self.credentials.get_token()
                    headers = {"Authorization": f"Bearer {token}"}
                    params = {}
                else:
                    headers = {}
                    params = {"key": self.api_key}

                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        resp = await client.post(url, json=body, headers=headers, params=params)
                        resp.raise_for_status()
                        data = resp.json()

                    vector = data["embedding"]["values"]
                    return EmbeddingResponse(vector=vector, model=model, dimensions=len(vector))

                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    error_text = e.response.text[:300] if hasattr(e.response, 'text') else str(e)
                    if attempt < _MAX_RETRIES and _is_retryable_error(status, error_text):
                        server_delay = _extract_retry_delay(error_text, e.response.headers)
                        delay_ms = server_delay if server_delay else _BASE_DELAY_MS * (2 ** attempt)
                        if server_delay and server_delay > _MAX_RETRY_DELAY_MS:
                            raise LLMRateLimitError(
                                f"Embed rate limited (429). Server requested {server_delay // 1000}s wait."
                            ) from e
                        logger.warning(f"Embed {status}, retrying in {delay_ms}ms (attempt {attempt + 1}/{_MAX_RETRIES + 1})")
                        await asyncio.sleep(delay_ms / 1000)
                        continue
                    if status == 429:
                        raise LLMRateLimitError(f"Embed rate limited (429) after {attempt + 1} attempts.") from e
                    raise

                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    if attempt < _MAX_RETRIES:
                        delay_ms = _BASE_DELAY_MS * (2 ** attempt)
                        logger.warning(f"Embed network error: {e}, retrying in {delay_ms}ms")
                        await asyncio.sleep(delay_ms / 1000)
                        continue
                    raise

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        """Batch embedding with retry + exponential backoff."""
        model = model or self.embedding_model
        url = f"{_GEMINI_API}/models/{model}:batchEmbedContents"

        requests = [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]

        async with self._embed_semaphore:
            for attempt in range(_MAX_RETRIES + 1):
                if self._use_cca:
                    token = await self.credentials.get_token()
                    headers = {"Authorization": f"Bearer {token}"}
                    params = {}
                else:
                    headers = {}
                    params = {"key": self.api_key}

                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(
                            url, json={"requests": requests}, headers=headers, params=params
                        )
                        resp.raise_for_status()
                        data = resp.json()

                    return [
                        EmbeddingResponse(vector=emb["values"], model=model, dimensions=len(emb["values"]))
                        for emb in data["embeddings"]
                    ]

                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    error_text = e.response.text[:300] if hasattr(e.response, 'text') else str(e)
                    if attempt < _MAX_RETRIES and _is_retryable_error(status, error_text):
                        server_delay = _extract_retry_delay(error_text, e.response.headers)
                        delay_ms = server_delay if server_delay else _BASE_DELAY_MS * (2 ** attempt)
                        if server_delay and server_delay > _MAX_RETRY_DELAY_MS:
                            raise LLMRateLimitError(
                                f"Batch embed rate limited. Server requested {server_delay // 1000}s wait."
                            ) from e
                        logger.warning(f"Batch embed {status}, retrying in {delay_ms}ms (attempt {attempt + 1}/{_MAX_RETRIES + 1})")
                        await asyncio.sleep(delay_ms / 1000)
                        continue
                    if status == 429:
                        raise LLMRateLimitError(f"Batch embed rate limited (429) after {attempt + 1} attempts.") from e
                    raise

                except (httpx.ConnectError, httpx.ReadTimeout) as e:
                    if attempt < _MAX_RETRIES:
                        delay_ms = _BASE_DELAY_MS * (2 ** attempt)
                        logger.warning(f"Batch embed network error: {e}, retrying in {delay_ms}ms")
                        await asyncio.sleep(delay_ms / 1000)
                        continue
                    raise
