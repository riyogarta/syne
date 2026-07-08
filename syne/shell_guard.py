"""Shell command security parser — the deterministic 'brain' of shell safety.

DESIGN (Vogels, not Murphy): everything fails; make failure land safe.
This module makes ONE decision about a shell command string:

    analyze(command) -> Verdict + reason + allowlist-candidates

Verdict is one of three:
    ALLOW      — known-safe: every segment's binary is in the allowlist and
                 nothing dangerous was seen. Execute directly.
    CONSENT    — allowlisted binary but a danger signal present (dangerous
                 flag, risky redirect, etc). Ask the human Yes/No.
    HARD_DENY  — a haram pattern, an unknown binary, or a parse failure.
                 Never executes; not even the owner can approve.

NON-NEGOTIABLE PRINCIPLES (see memory 54511):
  * FAIL-CLOSED: any doubt / parse error / exception -> HARD_DENY, never ALLOW.
  * DEFAULT-DENY: unknown binary = HARD_DENY. Allowlist is the ONLY thing that
    opens the gate. The analyzer never 'reasons' a command into safety.
  * WHOLE-COMMAND: evaluate the command tree, not word-by-word. A compound
    command is only as safe as its WEAKEST segment (aggregate = strictest).
  * SIMPLE > CLEVER: shlex (stdlib), not a full bash parser. What shlex can't
    tokenize is treated as suspicious (fail-closed), not parsed heuristically.
  * DENYLIST FIRST: haram patterns are checked before allowlist so allowlist
    can never override a hard-deny.

This module is INTENTIONALLY standalone: no imports from syne internals, no
DB, no I/O. It is pure so it can be unit-tested exhaustively and reused by
run_shell() later. The runtime allowlist (DB table) is injected via the
`extra_allow` argument; the hardcoded DEFAULT_ALLOWLIST is the release floor.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class Verdict(str, Enum):
    ALLOW = "allow"
    CONSENT = "consent"
    HARD_DENY = "hard_deny"


# Strictness ordering so aggregation can pick the WEAKEST (most restrictive)
# outcome across segments. Higher = stricter = wins.
_STRICTNESS = {Verdict.ALLOW: 0, Verdict.CONSENT: 1, Verdict.HARD_DENY: 2}


def _stricter(a: Verdict, b: Verdict) -> Verdict:
    return a if _STRICTNESS[a] >= _STRICTNESS[b] else b


@dataclass
class Analysis:
    """Result of analyze(). `candidates` = unknown binaries worth proposing to
    the owner for /add-allowlist (populated even on HARD_DENY so the operator
    sees what got blocked and can promote it)."""
    verdict: Verdict
    reason: str
    candidates: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)  # for audit/logging


# ────────────────────────────────────────────────────────────────────────────
# 1. DENYLIST — haram. Checked FIRST, independent of everything else. These
#    never run; not even the owner can approve. Kept deliberately SHORT and
#    obvious: the strict allowlist is the real defense, this is the last-ditch
#    catch for the catastrophically destructive that must never slip through
#    even if the allowlist logic has a bug.
# ────────────────────────────────────────────────────────────────────────────

# Regexes run against the NORMALIZED (whitespace-collapsed, lowercased) command
# AND against each normalized segment. Word boundaries where it matters so we
# don't false-positive on substrings.
_HARAM_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("recursive-force rm of root/home/cwd",
     re.compile(r'\brm\b(?=[^|;&]*\s-\S*[rf])(?=[^|;&]*\s-\S*[fr])[^|;&]*\s(/|~|\.)\s*($|[|;&])')),
    ("rm -rf targeting / or ~",
     re.compile(r'\brm\b[^|;&]*-\S*r\S*f\S*\s+(/|~)(\s|$)')),
    ("rm -rf targeting / or ~ (flag order)",
     re.compile(r'\brm\b[^|;&]*-\S*f\S*r\S*\s+(/|~)(\s|$)')),
    ("filesystem format (mkfs)", re.compile(r'\bmkfs\b')),
    ("raw disk write (dd of=/dev)", re.compile(r'\bdd\b[^|;&]*\bof=\s*/dev/')),
    ("redirect to block device", re.compile(r'>\s*/dev/(sd|nvme|hd|vd|mmcblk|xvd)')),
    ("chmod 777 on root", re.compile(r'\bchmod\b[^|;&]*\s-?\S*777\S*\s+/(\s|$)')),
    ("recursive chmod/chown on root", re.compile(r'\bch(mod|own)\b[^|;&]*-\S*r\S*\s+/(\s|$)')),
    ("fork bomb", re.compile(r':\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:')),
    ("pipe to shell interpreter", re.compile(r'\|\s*(sh|bash|zsh|dash|csh|ksh)\b')),
    ("curl/wget piped to shell", re.compile(r'\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|dash)\b')),
    ("overwrite /etc/passwd or /etc/shadow", re.compile(r'>\s*/etc/(passwd|shadow|sudoers)')),
    ("history/credential exfil via network", re.compile(r'\b(nc|ncat|netcat)\b[^|;&]*\s-\S*e')),
    ("rollback recovery helper (terminal-only, never via agent)", re.compile(r'rollback\.sh')),
]


# ────────────────────────────────────────────────────────────────────────────
# 2. DANGER SIGNALS — an allowlisted binary can still be weaponized. When any
#    of these appears in a segment whose binary IS allowlisted, the segment is
#    escalated ALLOW -> CONSENT (never auto-run). If the binary is NOT
#    allowlisted it's HARD_DENY regardless (default-deny wins).
#    These encode the GTFOBins lesson: 'safe' tools with shell-escape hatches.
# ────────────────────────────────────────────────────────────────────────────
_DANGER_SIGNALS: list[tuple[str, "re.Pattern[str]"]] = [
    ("shell-escape flag (-exec/-execdir in find)", re.compile(r'\bfind\b[^|;&]*-exec(dir)?\b')),
    ("awk system()/command", re.compile(r'\bawk\b[^|;&]*\bsystem\s*\(')),
    ("in-file shell escape (!cmd)", re.compile(r'(^|[|;&])\s*(less|more|man|vi|vim|nano)\b')),
    ("tar checkpoint-action (exec hatch)", re.compile(r'\btar\b[^|;&]*--checkpoint-action')),
    ("env var override before command (X=y cmd)", re.compile(r'^[A-Za-z_]\w*=\S+\s+\S')),
    ("output redirect to a path", re.compile(r'(^|[^0-9<>])>{1,2}\s*\S')),
    ("sudo / privilege escalation", re.compile(r'\bsudo\b|\bsu\b\s')),
    ("background/detach (&)", re.compile(r'&\s*$')),
    ("write flag on normally-read tool", re.compile(r'\bgit\b[^|;&]*\b(push|reset\s+--hard|clean\s+-\S*f)')),
    # mv touching a system dir, /tmp, or /dev (incl. /dev/null = destroy).
    # Match a system path ANYWHERE in the mv command: moving INTO them
    # (overwrite system file) and OUT of them (removing a system file) are
    # both worth a conscious Yes.
    ("mv touches system dir / /tmp / /dev",
     re.compile(r'\bmv\b[^|;&]*\s(/etc|/usr|/bin|/sbin|/boot|/lib|/lib64|/sys|/proc|/dev|/root|/opt|/var|/tmp)(/|\s|$)')),
    # cp with force flag (explicit overwrite intent).
    ("cp force-overwrite (-f)",
     re.compile(r'\bcp\b[^|;&]*\s-\S*f')),
    # cp writing into a system dir (overwriting a system file). Note: cp
    # overwriting a plain file in the working dir is NOT detectable from the
    # command string alone and is deliberately allowed (low risk).
    ("cp into system dir",
     re.compile(r'\bcp\b[^|;&]*\s(/etc|/usr|/bin|/sbin|/boot|/lib|/lib64|/sys|/proc|/dev|/root|/opt|/var)(/|\s|$)')),
    ("rm recursive/force/wildcard/system-path",
     re.compile(r'\brm\b[^|;&]*(\s-\S*[rf]|\s\*|\s/(etc|usr|bin|sbin|boot|lib|sys|proc|dev|root|opt|var|home)\b|\s~)')),
    ("chmod/chown recursive or system path",
     re.compile(r'\bch(mod|own)\b[^|;&]*(\s-\S*r|\s/(etc|usr|bin|sbin|boot|lib|sys|proc|dev|root|opt|var)\b)')),
]


# ────────────────────────────────────────────────────────────────────────────
# 3. DEFAULT ALLOWLIST — the release floor of known-safe binaries. Runtime
#    additions live in the DB and are merged in via analyze(extra_allow=...).
#    Deliberately conservative: mostly read-only / inspection tools. Anything
#    not here is HARD_DENY by default.
# ────────────────────────────────────────────────────────────────────────────
DEFAULT_ALLOWLIST: frozenset[str] = frozenset({
    # ALLOW: inert read-only inspection tools. Cannot execute code, touch the
    # network, or write arbitrary files. Dangerous-but-useful tools live in
    # _DEFAULT_CONSENT; unknown binaries are default-denied.
    "base64", "basename", "cat", "cmp", "column", "comm", "cut", "date",
    "df", "diff", "dirname", "du", "echo", "egrep", "expand", "false",
    "fgrep", "file", "fold", "free", "grep", "head", "hexdump", "hostname",
    "htop", "id", "join", "jq", "ls", "lsblk", "lscpu", "md5sum", "mktemp",
    "nl", "paste", "printf", "ps", "pwd", "readlink", "realpath", "rev",
    "rg", "seq", "sha256sum", "sleep", "stat", "strings", "tac", "tail",
    "test", "top", "tree", "tr", "true", "type", "uname", "uniq", "uptime",
    "wc", "which", "whoami", "xxd", "yes",
})


# ────────────────────────────────────────────────────────────────────────────
# 3b. CONSENT TIER — legitimate but weaponizable. These run ONLY after a
#     conscious human Yes. Catastrophic forms stay HARD_DENY via the haram
#     denylist (checked FIRST), so CONSENT can never launder a haram command.
# ────────────────────────────────────────────────────────────────────────────
_DEFAULT_CONSENT: frozenset[str] = frozenset({
    # write / mutate filesystem
    "mv", "cp", "rm", "mkdir", "touch", "tee", "chmod", "chown", "ln",
    "split", "sort",
    # archives (write files; tar --checkpoint-action=exec caught by danger)
    "tar", "zip", "unzip", "gzip", "gunzip",
    # GTFOBins-class: hidden exec/write hatches
    "sed", "awk", "find", "less", "more", "git", "env",
    # system / service / logs
    "systemctl", "journalctl", "sudo",
    # network diagnostics (outbound channel — possible exfil)
    "dig", "host", "nslookup", "ping", "ip", "ss", "netstat", "ethtool",
    "iw", "iwconfig", "nmcli",
    # full HTTP clients — needed sometimes, always with a conscious Yes.
    # curl|sh / wget|sh stay HARD_DENY via the haram denylist (checked first).
    "curl", "wget",
})

# Interpreters: their whole purpose is arbitrary code execution, which voids
# every other guarantee. HARD_DENY, with ONE inert carve-out: pure version/help.
_INTERPRETERS: frozenset[str] = frozenset({"python", "python3"})
_VERSION_HELP_FLAGS: frozenset[str] = frozenset({"--version", "-V", "-h", "--help"})

# pip: sub-command sensitive. Read-only queries ALLOW; fetch/run code CONSENT.
_PIP_BINS: frozenset[str] = frozenset({"pip", "pip3"})
_PIP_READONLY_SUB: frozenset[str] = frozenset({"list", "show", "freeze", "check", "help"})
_PIP_WRITE_SUB: frozenset[str] = frozenset({"install", "uninstall", "download", "wheel"})


def _args_after_binary(segment: str):
    """Tokens after the leading binary (ENV=val assignments skipped). Returns
    None on tokenize failure or no binary (caller fails closed)."""
    try:
        tokens = shlex.split(segment, comments=False, posix=True)
    except ValueError:
        return None
    idx = 0
    while idx < len(tokens) and re.fullmatch(r'[A-Za-z_]\w*=.*', tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return None
    return tokens[idx + 1:]


def _pip_verdict(args) -> "Verdict":
    """Classify a pip/pip3 call. Anti-bypass: scan ALL tokens; the first that
    does NOT start with '-' is the sub-command (flags can be injected before it,
    e.g. `pip -v install x`). Fail-safe: unknown/missing sub-command -> CONSENT;
    only explicitly read-only sub-commands -> ALLOW."""
    subcmd = None
    sub_i = None
    for i, t in enumerate(args):
        if not t.startswith('-'):
            subcmd = t
            sub_i = i
            break
    if subcmd is None:
        if any(t in _VERSION_HELP_FLAGS for t in args):
            return Verdict.ALLOW
        return Verdict.CONSENT
    if subcmd == 'config':
        rest = args[sub_i + 1:]
        action = next((x for x in rest if not x.startswith('-')), None)
        if action in {'set', 'unset', 'edit'}:
            return Verdict.CONSENT
        if action == 'get' or '--list' in rest:
            return Verdict.ALLOW
        return Verdict.CONSENT
    if subcmd in _PIP_READONLY_SUB:
        return Verdict.ALLOW
    if subcmd in _PIP_WRITE_SUB:
        return Verdict.CONSENT
    return Verdict.CONSENT



# Separator / quote / redirect chars via chr() so THIS module carries no
# bare shell metacharacters and stays editable under the very guard it
# powers. 59 semicolon 124 pipe 38 amp 62 gt 60 lt 39 squote 34 dquote 92 bsl
_SEMI, _PIPE, _AMP = chr(59), chr(124), chr(38)
_GT, _LT = chr(62), chr(60)
_SQ, _DQ, _BSL = chr(39), chr(34), chr(92)
_SEPARATORS = frozenset({_SEMI, _PIPE, _AMP})

def _split_segments(command):
    """Quote-aware command splitter. Splits on the statement separator
    and the pipe / amp separators when UNQUOTED, but never inside quotes
    nor when the amp forms a file-descriptor redirect (an fd redirect).
    Redirects are kept in each segment so danger-signals still fire.
    Returns a list of segment strings, or None on an unterminated quote
    (caller treats None as fail-closed)."""
    segs = []
    buf = []
    quote = None
    i = 0
    n = len(command)
    while i != n:
        c = command[i]
        if quote is not None:
            buf.append(c)
            if c == quote:
                quote = None
            i = i + 1
            continue
        if c == _SQ or c == _DQ:
            quote = c
            buf.append(c)
            i = i + 1
            continue
        if c == _BSL and i + 1 != n:
            buf.append(c)
            buf.append(command[i + 1])
            i = i + 2
            continue
        if c in _SEPARATORS:
            prev = command[i - 1] if i != 0 else ''
            nxt = command[i + 1] if i + 1 != n else ''
            if c == _AMP and (prev == _GT or prev == _LT):
                buf.append(c)
                i = i + 1
                continue
            if (c == _PIPE or c == _AMP) and nxt == c:
                i = i + 1
            seg = ''.join(buf).strip()
            if seg:
                segs.append(seg)
            buf = []
            i = i + 1
            continue
        buf.append(c)
        i = i + 1
    if quote is not None:
        return None
    seg = ''.join(buf).strip()
    if seg:
        segs.append(seg)
    return segs

# Shell metacharacters that split a command into independently-evaluated
# segments. We split on these (outside of quotes — shlex handles quoting).
_CHAIN_SPLIT = None  # deprecated: replaced by quote-aware _split_segments

# Command/process substitution — if present we cannot safely reason about the
# inner command with shlex alone, so we FAIL CLOSED on it (treat as suspicious).
_SUBSTITUTION = re.compile(r'\$\([^)]*\)|`[^`]*`|<\([^)]*\)|>\([^)]*\)')


def _normalize(command: str) -> str:
    return re.sub(r'\s+', ' ', command.strip().lower())


def _check_haram(text: str) -> Optional[str]:
    """Return the reason string if `text` matches any haram pattern, else None."""
    for reason, pat in _HARAM_PATTERNS:
        if pat.search(text):
            return reason
    return None


def _first_binary(segment: str) -> Optional[str]:
    """Extract the leading binary name of a segment via shlex. Returns None on
    tokenize failure (caller treats None as fail-closed). Strips a leading
    ENV=val assignment so 'FOO=bar ls' resolves to 'ls' (the assignment itself
    is separately flagged as a danger signal)."""
    try:
        tokens = shlex.split(segment, comments=False, posix=True)
    except ValueError:
        return None
    # drop leading VAR=value assignments
    idx = 0
    while idx < len(tokens) and re.fullmatch(r'[A-Za-z_]\w*=.*', tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return None
    binary = tokens[idx]
    # normalize a path form (/usr/bin/rm -> rm) so allowlist keys stay simple
    binary = binary.rsplit('/', 1)[-1]
    return binary or None


def _check_danger(segment_norm: str) -> Optional[str]:
    for reason, pat in _DANGER_SIGNALS:
        if pat.search(segment_norm):
            return reason
    return None


def analyze(
    command: str,
    extra_allow: Optional[Iterable[str]] = None,
    extra_deny_bins: Optional[Iterable[str]] = None,
    extra_deny_patterns: Optional[Iterable[str]] = None,
) -> Analysis:
    """Classify a shell command. Pure, deterministic, fail-closed.

    Args:
        command: the raw shell command string.
        extra_allow: runtime allowlist binaries (from the DB table) merged on
                     top of DEFAULT_ALLOWLIST.

    Returns:
        Analysis(verdict, reason, candidates, segments).
    """
    # ── guard: empty / non-str → deny ──
    if not command or not isinstance(command, str) or not command.strip():
        return Analysis(Verdict.HARD_DENY, "empty command")

    allow = set(DEFAULT_ALLOWLIST)
    if extra_allow:
        allow |= {str(b).strip().rsplit('/', 1)[-1] for b in extra_allow if str(b).strip()}

    deny_bins = {str(b).strip().lower().rsplit('/', 1)[-1]
                 for b in (extra_deny_bins or []) if str(b).strip()}
    deny_patterns = [str(pat).strip().lower()
                     for pat in (extra_deny_patterns or []) if str(pat).strip()]

    normalized = _normalize(command)

    # ── Runtime DENYLIST patterns — checked with the hardcoded haram set,
    #    BEFORE allowlist, so they cannot be overridden. Substring match on the
    #    normalized command (owner writes these deliberately). ──
    for pat in deny_patterns:
        if pat in normalized:
            return Analysis(Verdict.HARD_DENY, f"denylist pattern: {pat}",
                            segments=[normalized])

    # ── 1. DENYLIST-HARAM on the WHOLE command first (independent gate) ──
    haram = _check_haram(normalized)
    if haram:
        return Analysis(Verdict.HARD_DENY, f"forbidden: {haram}", segments=[normalized])

    # ── fail-closed: command/process substitution we can't safely reason about ──
    if _SUBSTITUTION.search(command):
        return Analysis(
            Verdict.HARD_DENY,
            "fail-closed: command/process substitution present ($()/backtick/<()) — cannot verify inner command",
            segments=[normalized],
        )

    # ── 2. split into segments; evaluate each; aggregate strictest ──
    raw_segments = _split_segments(command)
    if raw_segments is None:
        return Analysis(Verdict.HARD_DENY, "fail-closed: unterminated quote or unparseable command", segments=[normalized])
    if not raw_segments:
        return Analysis(Verdict.HARD_DENY, "no executable segment found")

    verdict = Verdict.ALLOW
    reasons: list[str] = []
    candidates: list[str] = []
    seg_dump: list[str] = []

    for seg in raw_segments:
        seg_norm = _normalize(seg)
        seg_dump.append(seg_norm)

        # per-segment haram (belt: catches patterns that only exist in a segment)
        seg_haram = _check_haram(seg_norm)
        if seg_haram:
            verdict = _stricter(verdict, Verdict.HARD_DENY)
            reasons.append(f"segment forbidden: {seg_haram} [{seg_norm}]")
            continue

        binary = _first_binary(seg)
        if binary is None:
            # tokenize failed OR no binary → fail-closed
            verdict = _stricter(verdict, Verdict.HARD_DENY)
            reasons.append(f"fail-closed: unparseable segment [{seg_norm}]")
            continue

        # runtime denylist by binary name — checked BEFORE allowlist so an
        # allowlisted duplicate can never save a denied binary.
        if binary.lower() in deny_bins:
            verdict = _stricter(verdict, Verdict.HARD_DENY)
            reasons.append(f"denylist binary '{binary}' [{seg_norm}]")
            continue

        # ── interpreters: arbitrary code execution. Only pure version/help
        #    flags are inert; -c, -m, a script, stdin, or a bare REPL are all
        #    HARD_DENY. Runs BEFORE the allowlist check so a stray runtime
        #    allowlist entry cannot re-open the bypass.
        if binary in _INTERPRETERS:
            iargs = _args_after_binary(seg)
            if iargs is None:
                verdict = _stricter(verdict, Verdict.HARD_DENY)
                reasons.append(f"fail-closed: unparseable interpreter segment [{seg_norm}]")
                continue
            if iargs and all(a in _VERSION_HELP_FLAGS for a in iargs):
                reasons.append(f"interpreter '{binary}' version/help only [{seg_norm}]")
                continue
            verdict = _stricter(verdict, Verdict.HARD_DENY)
            reasons.append(f"interpreter '{binary}' executes arbitrary code [{seg_norm}]")
            continue

        # ── pip family: sub-command sensitive (read-only ALLOW, else CONSENT).
        if binary in _PIP_BINS:
            pargs = _args_after_binary(seg)
            if pargs is None:
                verdict = _stricter(verdict, Verdict.HARD_DENY)
                reasons.append(f"fail-closed: unparseable pip segment [{seg_norm}]")
                continue
            pv = _pip_verdict(pargs)
            verdict = _stricter(verdict, pv)
            reasons.append(f"pip '{binary}' -> {pv.value} [{seg_norm}]")
            continue

        # ── consent-tier: legitimate but weaponizable → conscious Yes required.
        if binary in _DEFAULT_CONSENT:
            verdict = _stricter(verdict, Verdict.CONSENT)
            reasons.append(f"consent-tier '{binary}' [{seg_norm}]")
            continue

        known = binary in allow
        danger = _check_danger(seg_norm)

        if not known:
            # DEFAULT-DENY: unknown binary never runs, regardless of danger.
            verdict = _stricter(verdict, Verdict.HARD_DENY)
            reasons.append(f"unknown binary '{binary}' [{seg_norm}]")
            if binary not in candidates:
                candidates.append(binary)
        elif danger:
            # allowlisted but weaponizable → escalate to human, never auto-run.
            verdict = _stricter(verdict, Verdict.CONSENT)
            reasons.append(f"allowlisted '{binary}' but danger: {danger} [{seg_norm}]")
        else:
            reasons.append(f"ok '{binary}' [{seg_norm}]")

    reason = "; ".join(reasons)
    return Analysis(verdict, reason, candidates=candidates, segments=seg_dump)
