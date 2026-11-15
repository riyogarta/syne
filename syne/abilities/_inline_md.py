"""Shared inline-markup parser for the document abilities (PDF, Word, PowerPoint).

Turns one line / paragraph of text into a flat list of styled ``Segment``s so
every renderer applies exactly the same rules.

Two syntaxes are supported:

- ``markdown`` (default): ``**bold**`` / ``__bold__``, ``*italic*`` /
  ``_italic_``, ``***bold italic***``, ``~~strike~~``, ```code```,
  ``[label](url)`` and backslash escapes (``\\*``). Markers may nest, e.g.
  ``**bold with *italic* inside**`` or ``**see [docs](https://x)**``.
  Emphasis follows the CommonMark delimiter-run rules, so ``snake_case`` and
  ``5 * 3`` stay literal.
- ``whatsapp``: ``*bold*``, ``_italic_``, ``~strike~``, ```code``` and
  ```````mono```````. Markers only open/close at word boundaries, like WhatsApp
  itself.

The syntax is passed per call, or set for a whole document build with
``syntax_scope()``. That uses a ContextVar, so concurrent builds never leak
into each other.
"""

from __future__ import annotations

import contextlib
import contextvars
import string
import unicodedata
from dataclasses import dataclass

MARKDOWN = "markdown"
WHATSAPP = "whatsapp"
SYNTAXES = (MARKDOWN, WHATSAPP)

_ALIASES = {
    "": MARKDOWN,
    "markdown": MARKDOWN,
    "md": MARKDOWN,
    "whatsapp": WHATSAPP,
    "wa": WHATSAPP,
}

_current_syntax: contextvars.ContextVar[str] = contextvars.ContextVar(
    "syne_inline_md_syntax", default=MARKDOWN
)

_ASCII_PUNCT = frozenset(string.punctuation)
_DELIM_CHARS = "*_~"


def normalize_syntax(value) -> str:
    """Map a user-supplied syntax name to MARKDOWN / WHATSAPP.

    ``None`` / empty means markdown. Unknown names raise ValueError so a typo
    is reported instead of silently rendering with the wrong rules.
    """
    key = str(value or "").strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown syntax '{value}'. Use one of: {', '.join(SYNTAXES)}"
        )
    return _ALIASES[key]


@contextlib.contextmanager
def syntax_scope(syntax):
    """Make ``syntax`` the default for parse_inline() inside the block."""
    token = _current_syntax.set(normalize_syntax(syntax))
    try:
        yield
    finally:
        _current_syntax.reset(token)


def current_syntax() -> str:
    return _current_syntax.get()


@dataclass(frozen=True)
class Segment:
    text: str
    bold: bool = False
    italic: bool = False
    strike: bool = False
    code: bool = False
    link: str | None = None

    @property
    def plain(self) -> bool:
        return not (self.bold or self.italic or self.strike or self.code or self.link)


@dataclass
class _Delim:
    char: str
    start: int      # first unconsumed position
    end: int        # one past the last unconsumed position
    orig: int       # original run length (CommonMark "rule of 3")
    can_open: bool
    can_close: bool
    cont: int       # container: 0 = top level, N = label of link N
    active: bool = True

    @property
    def remaining(self) -> int:
        return self.end - self.start


def _is_ws(ch: str | None) -> bool:
    return ch is None or ch.isspace()


def _is_punct(ch: str | None) -> bool:
    if ch is None:
        return False
    return ch in _ASCII_PUNCT or unicodedata.category(ch)[:1] in ("P", "S")


def _run_len(text: str, i: int, ch: str) -> int:
    j = i
    while j < len(text) and text[j] == ch:
        j += 1
    return j - i


def _find_backtick_run(text: str, start: int, run: int) -> int:
    """Index of the next backtick run of exactly ``run`` chars, or -1."""
    j = start
    while j < len(text):
        if text[j] == "`":
            m = _run_len(text, j, "`")
            if m == run:
                return j
            j += m
        else:
            j += 1
    return -1


class _Parser:
    def __init__(self, text: str, mode: str):
        self.mode = mode
        self.chars: list[str] = []
        self.literal: list[bool] = []   # escaped or code: never a delimiter
        self.code: list[bool] = []
        self._scan(text)
        n = len(self.chars)
        self.deleted = [False] * n
        self.link: list[str | None] = [None] * n
        self.container = [0] * n
        self.flags = {
            "bold": [False] * n,
            "italic": [False] * n,
            "strike": [False] * n,
        }

    # -- pass 1: escapes + code spans -------------------------------------

    def _emit(self, ch: str, literal: bool = False, code: bool = False) -> None:
        self.chars.append(ch)
        self.literal.append(literal)
        self.code.append(code)

    def _scan(self, text: str) -> None:
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            if (
                c == "\\"
                and self.mode == MARKDOWN
                and i + 1 < n
                and text[i + 1] in _ASCII_PUNCT
            ):
                self._emit(text[i + 1], literal=True)
                i += 2
                continue
            if c == "`":
                run = _run_len(text, i, "`")
                close = _find_backtick_run(text, i + run, run)
                if close >= 0:
                    body = text[i + run:close]
                    if len(body) >= 2 and body[0] == " " and body[-1] == " " and body.strip():
                        body = body[1:-1]
                    for ch in body:
                        self._emit(ch, literal=True, code=True)
                    i = close + run
                else:
                    for _ in range(run):
                        self._emit("`", literal=True)
                    i += run
                continue
            self._emit(c)
            i += 1

    # -- pass 2: links ---------------------------------------------------

    def _special(self, k: int, ch: str) -> bool:
        return k < len(self.chars) and self.chars[k] == ch and not self.literal[k]

    def mark_links(self) -> None:
        n = len(self.chars)
        link_id = 0
        k = 0
        while k < n:
            if not self._special(k, "["):
                k += 1
                continue
            j = k + 1
            while j < n and not self._special(j, "]") and not self._special(j, "["):
                j += 1
            if j >= n or not self._special(j, "]") or j == k + 1 or not self._special(j + 1, "("):
                k += 1
                continue
            u = j + 2
            while u < n and not self._special(u, ")") and not self.chars[u].isspace():
                u += 1
            if u >= n or not self._special(u, ")") or u == j + 2:
                k += 1
                continue
            url = "".join(self.chars[j + 2:u])
            link_id += 1
            for p in (k, j, j + 1):
                self.deleted[p] = True
            for p in range(j + 2, u + 1):
                self.deleted[p] = True
            for p in range(k + 1, j):
                self.link[p] = url
                self.container[p] = link_id
            k = u + 1

    # -- pass 3: emphasis (CommonMark delimiter runs) ---------------------

    def _collect_delims(self) -> list[_Delim]:
        chars, n = self.chars, len(self.chars)
        runs: list[_Delim] = []
        p = 0
        while p < n:
            ch = chars[p]
            if ch not in _DELIM_CHARS or self.literal[p] or self.deleted[p]:
                p += 1
                continue
            cont = self.container[p]
            q = p
            while (
                q < n
                and chars[q] == ch
                and not self.literal[q]
                and not self.deleted[q]
                and self.container[q] == cont
            ):
                q += 1
            prev = chars[p - 1] if p > 0 else None
            nxt = chars[q] if q < n else None
            left = not _is_ws(nxt) and (not _is_punct(nxt) or _is_ws(prev) or _is_punct(prev))
            right = not _is_ws(prev) and (not _is_punct(prev) or _is_ws(nxt) or _is_punct(nxt))
            if self.mode == MARKDOWN and ch in "*~":
                can_open, can_close = left, right
            else:
                # '_' in markdown, every marker in WhatsApp: no intraword use.
                can_open = left and (not right or _is_punct(prev))
                can_close = right and (not left or _is_punct(nxt))
            if self.mode == MARKDOWN and ch == "~" and q - p != 2:
                can_open = can_close = False
            runs.append(_Delim(ch, p, q, q - p, can_open, can_close, cont))
            p = q
        return runs

    def _find_opener(self, runs: list[_Delim], ci: int) -> int:
        closer = runs[ci]
        for oi in range(ci - 1, -1, -1):
            o = runs[oi]
            if not o.active or o.remaining <= 0 or not o.can_open:
                continue
            if o.char != closer.char or o.cont != closer.cont:
                continue
            if self.mode == MARKDOWN:
                if o.char == "~":
                    if o.remaining != closer.remaining:
                        continue
                elif (
                    (o.can_close or closer.can_open)
                    and (o.orig + closer.orig) % 3 == 0
                    and not (o.orig % 3 == 0 and closer.orig % 3 == 0)
                ):
                    continue
            return oi
        return -1

    def _match(self, opener: _Delim, closer: _Delim) -> tuple[int, str]:
        if self.mode == WHATSAPP:
            return 1, {"*": "bold", "_": "italic", "~": "strike"}[opener.char]
        if opener.char == "~":
            return 2, "strike"
        use = 2 if opener.remaining >= 2 and closer.remaining >= 2 else 1
        return use, ("bold" if use == 2 else "italic")

    def process_emphasis(self) -> None:
        runs = self._collect_delims()
        for ci, closer in enumerate(runs):
            if not closer.can_close:
                continue
            while closer.active and closer.remaining > 0:
                oi = self._find_opener(runs, ci)
                if oi < 0:
                    break
                opener = runs[oi]
                use, style = self._match(opener, closer)
                marks = self.flags[style]
                for p in range(opener.end, closer.start):
                    marks[p] = True
                for p in range(opener.end - use, opener.end):
                    self.deleted[p] = True
                for p in range(closer.start, closer.start + use):
                    self.deleted[p] = True
                opener.end -= use
                closer.start += use
                for k in range(oi + 1, ci):
                    if runs[k].cont == closer.cont:
                        runs[k].active = False
                if opener.remaining <= 0:
                    opener.active = False

    # -- output ------------------------------------------------------------

    def segments(self) -> list[Segment]:
        out: list[Segment] = []
        buf: list[str] = []
        cur = None
        bold, italic, strike = self.flags["bold"], self.flags["italic"], self.flags["strike"]
        for p, ch in enumerate(self.chars):
            if self.deleted[p]:
                continue
            key = (bold[p], italic[p], strike[p], self.code[p], self.link[p])
            if buf and key != cur:
                out.append(Segment("".join(buf), *cur))
                buf = []
            cur = key
            buf.append(ch)
        if buf:
            out.append(Segment("".join(buf), *cur))
        return out


def parse_inline(text, syntax: str | None = None) -> list[Segment]:
    """Parse inline markup into styled segments.

    ``syntax`` overrides the scope default (see ``syntax_scope``). Unmatched
    markers are kept as literal text; nothing is ever dropped except the
    markup itself.
    """
    text = "" if text is None else str(text)
    if not text:
        return []
    mode = normalize_syntax(syntax) if syntax is not None else _current_syntax.get()
    parser = _Parser(text, mode)
    parser.mark_links()
    parser.process_emphasis()
    return parser.segments()


def has_markup(text, syntax: str | None = None) -> bool:
    """True when parsing changes anything (styling, links or removed markers)."""
    text = "" if text is None else str(text)
    segs = parse_inline(text, syntax)
    if not segs:
        return False
    return not (len(segs) == 1 and segs[0].plain and segs[0].text == text)


def plain_text(text, syntax: str | None = None) -> str:
    """Text with all inline markup removed (link labels kept, URLs dropped)."""
    return "".join(seg.text for seg in parse_inline(text, syntax))
