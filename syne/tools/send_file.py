"""Send File — Send a file to the current chat (owner-only).

Uses the MEDIA: protocol that channels already understand.
The channel layer (Telegram, CLI) handles the actual delivery.

For Telegram: images → send_photo, other files → send_document.
For CLI: prints the file path.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("syne.tools.send_file")

# Max file size (50MB — Telegram Bot API limit)
_MAX_FILE_SIZE = 50 * 1024 * 1024


async def send_file_handler(
    path: str,
    caption: str = "",
) -> str:
    """Send a file to the current chat.
    
    Args:
        path: Absolute or relative path to the file
        caption: Optional caption/description for the file
        
    Returns:
        MEDIA: protocol string (channel layer handles delivery)
    """
    if not path:
        return "Error: path is required."
    
    # Resolve path
    file_path = os.path.expanduser(path)
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    file_path = os.path.realpath(file_path)
    
    # Validate file exists
    if not os.path.exists(file_path):
        return f"Error: File not found: {path}"
    
    if not os.path.isfile(file_path):
        return f"Error: Not a file: {path}"
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return f"Error: File is empty: {path}"
    
    if file_size > _MAX_FILE_SIZE:
        size_mb = file_size / (1024 * 1024)
        return f"Error: File too large ({size_mb:.1f} MB). Telegram limit is 50 MB."
    
    filename = os.path.basename(file_path)
    size_str = _format_size(file_size)
    
    logger.info(f"send_file: {file_path} ({size_str})")
    
    # Return MEDIA: protocol — channel layer picks this up
    if caption:
        return f"{caption}\n\nMEDIA: {file_path}"
    else:
        return f"📎 {filename} ({size_str})\n\nMEDIA: {file_path}"


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ── Tool Registration Dict ──

SEND_FILE_TOOL = {
    "name": "send_file",
    "description": (
        "Send a file to the current chat. Works with any file type. "
        "Images are sent as photos, other files as documents. "
        "Max file size: 50 MB (Telegram limit)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to send (absolute or relative to CWD)",
            },
            "caption": {
                "type": "string",
                "description": "Optional caption for the file",
            },
        },
        "required": ["path"],
    },
    "handler": send_file_handler,
    "requires_access_level": "owner",
}
