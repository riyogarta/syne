"""Shared helpers for Google Gemini-family providers (google.py + vertex.py).

Extracted from google.py — all functions here are used by both the CCA/API-key
GoogleProvider and the Vertex AI VertexProvider.
"""

import json
import logging
import re
import time
from typing import Optional

import httpx

from .provider import ChatMessage

logger = logging.getLogger("syne.llm.gemini_common")


# ═══════════════════════════════════════════════════════════════════════════════
# Unicode surrogate sanitization (from sanitize-unicode.js)
# ═══════════════════════════════════════════════════════════════════════════════

_SURROGATE_RE = re.compile(
    r'[\ud800-\udbff](?![\udc00-\udfff])'
    r'|(?<![\ud800-\udbff])[\udc00-\udfff]',
    re.UNICODE,
)


def _sanitize_surrogates(text: str) -> str:
    """Remove unpaired Unicode surrogate characters."""
    if not text:
        return text
    try:
        return _SURROGATE_RE.sub('', text)
    except Exception:
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# Extract retry delay from error response
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_retry_delay(error_text: str, headers: Optional[httpx.Headers] = None) -> Optional[int]:
    """Extract retry delay in milliseconds from error response.

    Checks headers first, then parses body patterns. Returns milliseconds or None.
    """
    def _normalize(ms: float) -> Optional[int]:
        return int(ms + 1000) if ms > 0 else None

    if headers:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                seconds = float(retry_after)
                delay = _normalize(seconds * 1000)
                if delay is not None:
                    return delay
            except ValueError:
                pass

        reset_after = headers.get("x-ratelimit-reset-after")
        if reset_after:
            try:
                seconds = float(reset_after)
                delay = _normalize(seconds * 1000)
                if delay is not None:
                    return delay
            except ValueError:
                pass

    # Pattern 1: "Your quota will reset after 18h31m10s"
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
# Retryable error detection
# ═══════════════════════════════════════════════════════════════════════════════

_RETRYABLE_PATTERN = re.compile(
    r'resource.?exhausted|rate.?limit|overloaded|service.?unavailable|other.?side.?closed',
    re.I,
)


def _is_retryable_error(status: int, error_text: str) -> bool:
    """Check if an error is retryable. 400 Bad Request is NOT retryable."""
    if status in (429, 499, 500, 502, 503, 504):
        return True
    return bool(_RETRYABLE_PATTERN.search(error_text))


# ═══════════════════════════════════════════════════════════════════════════════
# Error classification — Gemini CLI-style quota analysis
# ═══════════════════════════════════════════════════════════════════════════════

class _ErrorClass:
    TERMINAL = "terminal"
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"


class ErrorClassification:
    __slots__ = ("kind", "delay_ms", "reason")

    def __init__(self, kind: str, delay_ms: int = 0, reason: str = ""):
        self.kind = kind
        self.delay_ms = delay_ms
        self.reason = reason

    @property
    def is_terminal(self) -> bool:
        return self.kind == _ErrorClass.TERMINAL

    @property
    def is_retryable(self) -> bool:
        return self.kind == _ErrorClass.RETRYABLE


_CLOUDCODE_DOMAINS = {
    "cloudcode-pa.googleapis.com",
    "staging-cloudcode-pa.googleapis.com",
    "autopush-cloudcode-pa.googleapis.com",
}


def _parse_duration_seconds(duration: str) -> Optional[float]:
    """Parse duration string like '34.07s' or '500ms' to seconds."""
    if duration.endswith("ms"):
        try:
            return float(duration[:-2]) / 1000
        except ValueError:
            return None
    if duration.endswith("s"):
        try:
            return float(duration[:-1])
        except ValueError:
            return None
    return None


def _classify_google_error(
    status: int,
    error_text: str,
    headers: Optional[httpx.Headers] = None,
) -> ErrorClassification:
    """Classify a Google API error for retry decisions.

    Follows Gemini CLI's classifyGoogleError() order exactly:
    1. 503 → Retryable (no delay)
    2. 429/499 without structured details → parse "Please retry in X" or plain retryable
    3. QuotaFailure.violations[].quotaId PerDay/Daily → Terminal
    4. ErrorInfo INSUFFICIENT_G1_CREDITS_BALANCE → Terminal
    5. ErrorInfo cloudcode-pa domain: RATE_LIMIT_EXCEEDED → Retryable (10s),
       QUOTA_EXHAUSTED → Terminal
    6. RetryInfo.retryDelay → Retryable (server delay)
    7. QuotaFailure PerMinute → Retryable (60s)
    8. ErrorInfo.metadata.quota_limit PerMinute → Retryable (60s)
    9. Fallback 429/499 → Retryable (no delay)
    """
    # Parse structured error
    err_message = ""
    details: list[dict] = []
    parsed_status: Optional[int] = None
    try:
        err_json = json.loads(error_text)
        err_obj = err_json.get("error", err_json)
        if isinstance(err_obj, dict):
            details = err_obj.get("details", [])
            err_message = err_obj.get("message", "")
            parsed_status = err_obj.get("code")
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    effective_status = parsed_status or status

    # ── Step 1: 503 → Retryable (no delay) ──
    if effective_status == 503:
        return ErrorClassification(
            _ErrorClass.RETRYABLE, 0,
            err_message or f"Service unavailable (503)"
        )

    # Only classify 429/499 with structured detail parsing
    if effective_status not in (429, 499):
        # Non-429/499 status codes — simple classification
        if status == 400:
            return ErrorClassification(_ErrorClass.NON_RETRYABLE, 0, "Bad request (400)")
        if status in (401, 403):
            return ErrorClassification(_ErrorClass.NON_RETRYABLE, 0, f"Auth error ({status})")
        if status in (500, 502, 504):
            return ErrorClassification(_ErrorClass.RETRYABLE, 0, f"Server error ({status})")
        if _is_retryable_error(status, error_text):
            return ErrorClassification(_ErrorClass.RETRYABLE, 0, f"Retryable error ({status})")
        return ErrorClassification(_ErrorClass.NON_RETRYABLE, 0, f"Non-retryable error ({status})")

    # ── Step 2: 429/499 without structured details ──
    if not details:
        # Try to parse "Please retry in X[s|ms]" from message
        msg = err_message or error_text
        m = re.search(r'Please retry in ([0-9.]+(?:ms|s))', msg, re.I)
        if m:
            parsed_secs = _parse_duration_seconds(m.group(1))
            if parsed_secs is not None:
                return ErrorClassification(
                    _ErrorClass.RETRYABLE, int(parsed_secs * 1000),
                    f"Rate limited ({effective_status}) — retry in {m.group(1)}"
                )
        # Plain 429/499 without details or parseable delay
        return ErrorClassification(
            _ErrorClass.RETRYABLE, 0,
            f"Rate limited ({effective_status})"
        )

    # ── Extract detail types separately (like Gemini CLI) ──
    quota_failure: Optional[dict] = None
    error_info: Optional[dict] = None
    retry_info: Optional[dict] = None

    for detail in details:
        if not isinstance(detail, dict):
            continue
        at_type = detail.get("@type", "")
        if "QuotaFailure" in at_type and quota_failure is None:
            quota_failure = detail
        elif "ErrorInfo" in at_type and error_info is None:
            error_info = detail
        elif "RetryInfo" in at_type and retry_info is None:
            retry_info = detail

    # ── Step 3: QuotaFailure PerDay/Daily → Terminal ──
    if quota_failure:
        for violation in quota_failure.get("violations", []):
            quota_id = violation.get("quotaId", "")
            if "PerDay" in quota_id or "Daily" in quota_id:
                return ErrorClassification(
                    _ErrorClass.TERMINAL, 0,
                    "Daily quota exhausted — cannot retry"
                )

    # Parse RetryInfo delay (used by steps 4-6)
    delay_seconds: Optional[float] = None
    if retry_info and retry_info.get("retryDelay"):
        delay_seconds = _parse_duration_seconds(retry_info["retryDelay"])

    # ── Step 4: ErrorInfo INSUFFICIENT_G1_CREDITS_BALANCE → Terminal ──
    if error_info:
        reason = error_info.get("reason", "")

        if reason == "INSUFFICIENT_G1_CREDITS_BALANCE":
            return ErrorClassification(
                _ErrorClass.TERMINAL, 0,
                "Insufficient G1 credits balance — cannot retry"
            )

        # ── Step 5: cloudcode-pa domain checks ──
        domain = error_info.get("domain", "")
        if domain in _CLOUDCODE_DOMAINS:
            if reason == "RATE_LIMIT_EXCEEDED":
                delay_ms = int(delay_seconds * 1000) if delay_seconds else 10_000
                return ErrorClassification(
                    _ErrorClass.RETRYABLE, delay_ms,
                    "CloudCode rate limit exceeded"
                )
            if reason == "QUOTA_EXHAUSTED":
                return ErrorClassification(
                    _ErrorClass.TERMINAL, 0,
                    "CloudCode quota exhausted — cannot retry"
                )

    # ── Step 6: RetryInfo with server delay → Retryable ──
    if retry_info and delay_seconds:
        delay_ms = int(delay_seconds * 1000)
        return ErrorClassification(
            _ErrorClass.RETRYABLE, delay_ms,
            f"Server requested retry in {retry_info['retryDelay']}"
        )

    # ── Step 7: QuotaFailure PerMinute → Retryable (60s) ──
    if quota_failure:
        for violation in quota_failure.get("violations", []):
            quota_id = violation.get("quotaId", "")
            if "PerMinute" in quota_id:
                return ErrorClassification(
                    _ErrorClass.RETRYABLE, 60_000,
                    "Per-minute quota — retry in 60s"
                )

    # ── Step 8: ErrorInfo metadata quota_limit PerMinute → Retryable (60s) ──
    if error_info:
        metadata = error_info.get("metadata", {})
        if isinstance(metadata, dict):
            quota_limit = metadata.get("quota_limit", "")
            if "PerMinute" in quota_limit:
                return ErrorClassification(
                    _ErrorClass.RETRYABLE, 60_000,
                    "Per-minute quota (metadata) — retry in 60s"
                )

    # ── Step 9: Fallback 429/499 → Retryable (no delay) ──
    return ErrorClassification(
        _ErrorClass.RETRYABLE, 0,
        f"Rate limited ({effective_status})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Thought signature validation
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
    """Try to flatten anyOf/oneOf into a single schema."""
    non_null = [v for v in variants if isinstance(v, dict) and not _is_null_schema(v)]
    if not non_null:
        return None

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

    if len(non_null) == 1:
        result = dict(non_null[0])
        for mk in ("description", "title", "default"):
            if mk in parent:
                result[mk] = parent[mk]
        return result

    types = {v.get("type") for v in non_null if isinstance(v.get("type"), str)}
    if len(types) == 1:
        result = {"type": types.pop()}
        for mk in ("description", "title", "default"):
            if mk in parent:
                result[mk] = parent[mk]
        return result

    first_type = non_null[0].get("type") if non_null else None
    if first_type:
        result = {"type": first_type}
        for mk in ("description", "title", "default"):
            if mk in parent:
                result[mk] = parent[mk]
        return result

    return None


def _clean_schema_for_gemini(schema: dict, defs: Optional[dict] = None) -> dict:
    """Remove unsupported JSON Schema keywords for Gemini."""
    if not isinstance(schema, dict):
        return schema

    if defs is None:
        defs = schema.get("$defs") or schema.get("definitions") or {}

    if "$ref" in schema and isinstance(schema["$ref"], str):
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        if ref_name in defs:
            return _clean_schema_for_gemini(defs[ref_name], defs)

    raw_type = schema.get("type")
    if isinstance(raw_type, list) and all(isinstance(t, str) for t in raw_type):
        non_null_types = [t for t in raw_type if t != "null"]
        schema = dict(schema)
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
            cleaned_variants = [_clean_schema_for_gemini(v, defs) for v in value]
            flattened = _try_flatten_union(cleaned_variants, schema)
            if flattened:
                cleaned.update(_clean_schema_for_gemini(flattened, defs))
            else:
                cleaned[key] = cleaned_variants
        elif key == "allOf" and isinstance(value, list):
            cleaned[key] = [_clean_schema_for_gemini(v, defs) for v in value]
        else:
            cleaned[key] = value

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# Tool schema conversion (parametersJsonSchema / parameters)
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_tools_to_gemini(tools: list[dict], use_parameters: bool = False) -> list[dict]:
    """Convert OpenAI function calling format to Gemini functionDeclarations.

    By default uses `parametersJsonSchema` (full JSON Schema support).
    Set `use_parameters=True` for legacy `parameters` field.
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
                    decl["parametersJsonSchema"] = _clean_schema_for_gemini(params)
            declarations.append(decl)

    if not declarations:
        return []

    return [{"functionDeclarations": declarations}]


# ═══════════════════════════════════════════════════════════════════════════════
# Transform messages (skip errored/aborted, synthetic orphan results)
# ═══════════════════════════════════════════════════════════════════════════════

def _transform_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Clean up message history before sending to Gemini."""
    pending_tool_calls: list[dict] = []
    existing_result_ids: set[str] = set()
    result: list[ChatMessage] = []

    for msg in messages:
        if msg.role == "assistant":
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

            stop_reason = msg.metadata.get("stop_reason") if msg.metadata else None
            if stop_reason in ("error", "aborted"):
                continue

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
    """Ensure Gemini user/model alternation."""
    if not contents:
        return contents

    result = []

    if contents[0].get("role") == "model":
        result.append({"role": "user", "parts": [{"text": "(session bootstrap)"}]})

    for msg in contents:
        if result and result[-1].get("role") == msg.get("role"):
            result[-1]["parts"].extend(msg.get("parts", []))
        else:
            result.append(msg)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Build thinking config
# ═══════════════════════════════════════════════════════════════════════════════

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

    _CCA_MAX_THINKING = 24576
    if thinking_budget is None:
        budget = -1
    elif thinking_budget == 0:
        budget = 128
    else:
        budget = min(thinking_budget, _CCA_MAX_THINKING)
    return {"includeThoughts": True, "thinkingBudget": budget}


# ═══════════════════════════════════════════════════════════════════════════════
# Format messages for Gemini (shared between GoogleProvider and VertexProvider)
# ═══════════════════════════════════════════════════════════════════════════════

def format_messages_for_gemini(
    messages: list[ChatMessage],
    model: str = "",
) -> tuple[Optional[str], list[dict]]:
    """Convert ChatMessages to Gemini format. Returns (system_text, contents).

    Implements:
    - Unicode sanitization on all text
    - functionResponse format {output: value} / {error: value}
    - Skip empty text blocks
    - thoughtSignature preserved & replayed
    - transformMessages (skip errored/aborted, synthetic orphan results)
    """
    messages = _transform_messages(messages)

    system_text = None
    contents = []

    is_gemini_3 = "gemini-3" in model.lower() if model else False

    for msg in messages:
        if msg.role == "system":
            text = _sanitize_surrogates(msg.content)
            if system_text is None:
                system_text = text
            else:
                system_text += "\n\n" + text

        elif msg.role == "user":
            text = _sanitize_surrogates(msg.content)
            if not text or not text.strip():
                continue
            parts = [{"text": text}]
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
                if msg.content and msg.content.strip():
                    parts.append({"text": _sanitize_surrogates(msg.content)})
                for tc in tool_calls:
                    func = tc.get("function", tc)
                    raw_args = func.get("arguments") or func.get("args") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}

                    thought_sig = tc.get("thoughtSignature")

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
                        if _is_valid_thought_signature(thought_sig):
                            fc_part["thoughtSignature"] = thought_sig
                        parts.append(fc_part)

                if parts:
                    contents.append({"role": "model", "parts": parts})
            else:
                if not msg.content or not msg.content.strip():
                    continue
                contents.append({"role": "model", "parts": [{"text": _sanitize_surrogates(msg.content)}]})

        elif msg.role == "tool":
            tool_name = msg.metadata.get("tool_name", "unknown") if msg.metadata else "unknown"
            is_error = msg.metadata.get("is_error", False) if msg.metadata else False

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
