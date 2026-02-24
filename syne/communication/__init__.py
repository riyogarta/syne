"""Communication sub-core — channel-agnostic message handling.

This package is the single source of truth for all communication logic:
- Inbound: InboundContext, system metadata, user context prefix
- Outbound: path stripping, media extraction, message splitting
- Tags: reply/react tag parsing from LLM output
- Formatting: markdown → platform-specific conversion
- Telegram: core/default communication channel (always active)
"""

from .inbound import InboundContext, build_system_metadata, build_user_context_prefix, load_group_settings
from .outbound import strip_server_paths, strip_narration, extract_media, split_message, process_outbound
from .tags import parse_reply_tag, parse_react_tags

__all__ = [
    # Inbound
    "InboundContext",
    "build_system_metadata",
    "build_user_context_prefix",
    "load_group_settings",
    # Outbound
    "strip_server_paths",
    "strip_narration",
    "extract_media",
    "split_message",
    "process_outbound",
    # Tags
    "parse_reply_tag",
    "parse_react_tags",
]
