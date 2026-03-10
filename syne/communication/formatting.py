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


def _html_to_telegram(text: str) -> str:
    """Convert raw HTML from LLM output to Telegram-compatible format.

    - HTML tables → monospace <pre> table (using _render_table)
    - Supported tags preserved: <b>, <i>, <u>, <s>, <code>, <pre>, <a>
    - Unsupported tags stripped or converted:
      <strong> → <b>, <em> → <i>, <del> → <s>
      <h1>-<h6> → **bold** (let markdown pass handle it)
      <hr> → --- line
      <br> → newline
      <p> → double newline
      <ul>/<ol>/<li> → markdown list
      <table>/<tr>/<td>/<th> → pipe-delimited table (let table pass handle it)
      <div>/<span> → content only
    """
    # === HTML table → markdown pipe table ===
    def _table_to_md(m):
        table_html = m.group(0)
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        if not rows:
            return ''
        md_rows = []
        for row in rows:
            # Extract cells (th or td)
            cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', row, re.DOTALL | re.IGNORECASE)
            # Strip nested HTML from cell content
            clean_cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            md_rows.append('| ' + ' | '.join(clean_cells) + ' |')
        # Add separator after first row (header)
        if len(md_rows) > 1:
            md_rows.insert(1, '| ' + ' | '.join(['---'] * len(re.findall(r'<(?:td|th)[^>]*>', rows[0], re.IGNORECASE))) + ' |')
        return '\n'.join(md_rows)

    text = re.sub(r'<table[^>]*>.*?</table>', _table_to_md, text, flags=re.DOTALL | re.IGNORECASE)

    # === Convert ALL HTML tags to markdown equivalents ===
    # This way the main markdown pass handles everything uniformly.

    # Bold: <b>, <strong>
    text = re.sub(r'<(?:b|strong)>(.*?)</(?:b|strong)>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
    # Italic: <i>, <em>
    text = re.sub(r'<(?:i|em)>(.*?)</(?:i|em)>', r'*\1*', text, flags=re.DOTALL | re.IGNORECASE)
    # Strikethrough: <s>, <del>
    text = re.sub(r'<(?:s|del)>(.*?)</(?:s|del)>', r'~~\1~~', text, flags=re.DOTALL | re.IGNORECASE)
    # Underline: <u> — Telegram supports, but no markdown equiv; use bold as fallback
    text = re.sub(r'<u>(.*?)</u>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
    # Inline code: <code>
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL | re.IGNORECASE)
    # Code block: <pre>
    text = re.sub(r'<pre>(.*?)</pre>', r'```\n\1\n```', text, flags=re.DOTALL | re.IGNORECASE)
    # Links: <a href="url">text</a>
    text = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.DOTALL | re.IGNORECASE)

    # === Headings → bold markdown ===
    text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n**\1**\n', text, flags=re.DOTALL | re.IGNORECASE)

    # === Block elements ===
    text = re.sub(r'<hr\s*/?>', '\n---\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<p[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)

    # === Lists ===
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'\n• \1', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?(?:ul|ol)[^>]*>', '', text, flags=re.IGNORECASE)

    # === Strip any remaining HTML tags ===
    text = re.sub(r'<[^>]+>', '', text)

    # Clean up excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


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
    - Raw HTML from LLM → converted to Telegram-safe format

    Text outside of these patterns is HTML-escaped for safety.
    """
    if not text:
        return text

    # Pre-process: if LLM output contains raw HTML tags, convert first
    if re.search(r'<(?:b|i|u|s|pre|code|a |table|tr|td|th|h[1-6]|hr|div|span|p|br|strong|em|del|ul|ol|li)\b', text, re.IGNORECASE):
        text = _html_to_telegram(text)
    
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
        # Also detect lines that look like table rows (at least 2 pipes)
        stripped = line.strip()
        is_table_row = (
            (stripped.startswith('|') and '|' in stripped[1:]) or
            (stripped.count('|') >= 2 and not stripped.startswith('```'))
        )
        if is_table_row:
            table_lines = []
            while i < len(lines):
                s = lines[i].strip()
                # Stop if line doesn't look like a table row
                if not s or (not s.startswith('|') and s.count('|') < 2):
                    break
                # Skip separator lines (|---|---|)
                if re.match(r'^[\s|]*[\-:]+[\s\-:|]*$', s):
                    i += 1
                    continue
                table_lines.append(lines[i])
                i += 1
            # Render table as aligned <pre> monospace
            if table_lines:
                result.append(_render_table(table_lines))
            continue
        
        # Regular line — apply inline formatting
        result.append(_format_inline(line))
        i += 1
    
    output = '\n'.join(result)
    
    # Safety net: catch any remaining raw markdown tables that slipped through
    # Look for consecutive lines with | that aren't inside <pre> tags
    output = _wrap_stray_tables(output)
    
    return output


def _strip_md(text: str) -> str:
    """Strip markdown formatting from text (for monospace display)."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_]+?)_(?!\w)', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text


def _render_table(table_lines: list[str]) -> str:
    """Render markdown table rows as padded monospace <pre> block."""
    rows = []
    for line in table_lines:
        stripped = line.strip()
        # Remove leading/trailing pipes and split
        if stripped.startswith('|'):
            stripped = stripped[1:]
        if stripped.endswith('|'):
            stripped = stripped[:-1]
        cells = [_strip_md(c.strip()) for c in stripped.split('|')]
        rows.append(cells)

    if not rows:
        return ''

    # Calculate column widths
    col_count = max(len(r) for r in rows)
    widths = [0] * col_count
    for row in rows:
        for j, cell in enumerate(row):
            if j < col_count and len(cell) > widths[j]:
                widths[j] = len(cell)

    # Build padded output
    out = []
    for idx, row in enumerate(rows):
        parts = []
        for j in range(col_count):
            cell = row[j] if j < len(row) else ''
            parts.append(cell + ' ' * (widths[j] - len(cell)))
        out.append('| ' + ' | '.join(parts) + ' |')
        # Add separator after header row
        if idx == 0:
            sep_parts = ['-' * max(3, w) for w in widths]
            out.append('|-' + '-|-'.join(sep_parts) + '-|')

    return '<pre>' + _escape('\n'.join(out)) + '</pre>'


def _wrap_stray_tables(html: str) -> str:
    """Catch any markdown tables that weren't wrapped in <pre> during main pass.
    
    Scans for consecutive lines containing | pipes that are NOT already
    inside <pre>...</pre> blocks, and wraps them.
    """
    # Split by <pre> blocks to avoid double-wrapping
    parts = re.split(r'(<pre>.*?</pre>)', html, flags=re.DOTALL)
    
    new_parts = []
    for part in parts:
        if part.startswith('<pre>'):
            new_parts.append(part)
            continue
        
        # Check for stray table rows in non-pre content
        lines = part.split('\n')
        out_lines = []
        j = 0
        while j < len(lines):
            line = lines[j]
            stripped = line.strip()
            # Detect table-like lines (2+ pipes, not a code element)
            if stripped.count('|') >= 2 and not stripped.startswith('<code>'):
                raw_lines = []
                while j < len(lines):
                    s = lines[j].strip()
                    if not s and raw_lines:
                        break
                    if s.count('|') >= 2 or re.match(r'^[\s\-:|]+$', s):
                        # Skip separator lines
                        if not re.match(r'^[\s|]*[\-:]+[\s\-:|]*$', s):
                            raw_lines.append(_html.unescape(lines[j]))
                        j += 1
                        continue
                    break
                if raw_lines:
                    out_lines.append(_render_table(raw_lines))
                    continue
            out_lines.append(line)
            j += 1
        new_parts.append('\n'.join(out_lines))
    
    return ''.join(new_parts)


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
