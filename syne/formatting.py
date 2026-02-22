"""Markdown to Telegram HTML converter.

Telegram supports a limited HTML subset:
  <b>bold</b>, <i>italic</i>, <u>underline</u>, <s>strikethrough</s>,
  <code>inline code</code>, <pre>code block</pre>,
  <a href="url">link</a>, <tg-spoiler>spoiler</tg-spoiler>

This module converts common LLM markdown output to safe Telegram HTML.
"""

import re
import html as _html


def _escape(text: str) -> str:
    """Escape HTML special characters in plain text segments."""
    return _html.escape(text, quote=False)


def markdown_to_telegram_html(text: str) -> str:
    """Convert markdown-formatted text to Telegram-safe HTML.
    
    Handles:
    - **bold** / __bold__ → <b>bold</b>
    - *italic* / _italic_ → <i>italic</i>
    - `inline code` → <code>inline code</code>
    - ```code blocks``` → <pre>code blocks</pre>
    - [text](url) → <a href="url">text</a>
    - ~~strikethrough~~ → <s>strikethrough</s>
    - Markdown tables → wrapped in <pre> for monospace alignment
    
    Text outside of these patterns is HTML-escaped for safety.
    """
    if not text:
        return text
    
    result = []
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Code block: ```...```
        if line.strip().startswith('```'):
            code_lines = []
            # Skip opening ```(optional language)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            # Skip closing ```
            if i < len(lines):
                i += 1
            code_content = _escape('\n'.join(code_lines))
            result.append(f'<pre>{code_content}</pre>')
            continue
        
        # Markdown table detection: line starts with | and contains |
        if line.strip().startswith('|') and '|' in line.strip()[1:]:
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                # Skip separator lines (|---|---|)
                stripped = lines[i].strip()
                if re.match(r'^\|[\s\-:|]+\|$', stripped):
                    i += 1
                    continue
                table_lines.append(lines[i])
                i += 1
            # Render table as <pre> for monospace alignment
            if table_lines:
                table_text = _escape('\n'.join(table_lines))
                result.append(f'<pre>{table_text}</pre>')
            continue
        
        # Regular line — apply inline formatting
        result.append(_format_inline(line))
        i += 1
    
    return '\n'.join(result)


def _format_inline(text: str) -> str:
    """Apply inline markdown formatting to a single line."""
    # Process segments — protect code spans first
    segments = []
    pos = 0
    
    # Find all inline code spans first
    code_pattern = re.compile(r'`([^`]+)`')
    last_end = 0
    
    for match in code_pattern.finditer(text):
        # Process text before this code span
        if match.start() > last_end:
            segments.append(('text', text[last_end:match.start()]))
        segments.append(('code', match.group(1)))
        last_end = match.end()
    
    # Remaining text after last code span
    if last_end < len(text):
        segments.append(('text', text[last_end:]))
    
    # Now format each segment
    parts = []
    for seg_type, seg_text in segments:
        if seg_type == 'code':
            parts.append(f'<code>{_escape(seg_text)}</code>')
        else:
            parts.append(_format_text_segment(seg_text))
    
    return ''.join(parts)


def _format_text_segment(text: str) -> str:
    """Apply bold, italic, links, strikethrough to a text segment."""
    # Escape HTML first
    text = _escape(text)
    
    # Links: [text](url) — must be before bold/italic processing
    text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2">\1</a>',
        text
    )
    
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    
    # Italic: *text* or _text_ (but not inside words like file_name)
    # Only match _text_ when preceded by space/start and followed by space/end/punctuation
    text = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^_]+?)_(?!\w)', r'<i>\1</i>', text)
    
    # Strikethrough: ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    
    # Headers: # Header → <b>Header</b> (Telegram has no headers)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text)
    
    return text
