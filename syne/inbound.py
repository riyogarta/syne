"""DEPRECATED — Use syne.communication.inbound instead.

This file re-exports for backward compatibility during migration.
"""
from .communication.inbound import (  # noqa: F401
    InboundContext,
    build_system_metadata,
    build_user_context_prefix,
    load_group_settings,
)
