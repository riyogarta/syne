"""Written-reference recognition for the rule-built KG (no AI).

Pure functions, no database access:
  * surah-name normalization (spelling variants fold to one key),
  * a surah index built from the stored Qur'an reference chunks,
  * parsers for references written in chunk titles and text:
    QS mentions, tafsir / asbab-nuzul titles, "(bagian i/n)" parts.

The KG builder loads the rows and config and calls these. Anything that
cannot be resolved with certainty is returned as a reject (with a reason)
instead of being linked (Riyo, 14 Nov 2026).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ── Surah names ──────────────────────────────────────────────────────────

# Arabic article as a separate first word ("Al-Baqarah", "Asy Syams").
_ARTICLES = frozenset({
    "al", "an", "ar", "as", "ash", "asy", "at", "ath", "ats",
    "ad", "adh", "adz", "az", "el",
})

# Transliteration variants folded to one form, applied in this order
# (digraphs before y->i so "sy"/"sh" are seen first).
_FOLDS = (
    ("sy", "s"), ("sh", "s"), ("ts", "s"), ("th", "s"),
    ("dz", "z"), ("dh", "z"), ("gh", "g"), ("w", "u"), ("y", "i"), ("o", "a"),
)

# Names that folding cannot join (a different name, or a vowel change).
# Code fallback: used only when the Qur'an entry in config kg.ref_sources
# has no 'aliases' (the config list, when present, replaces this one).
DEFAULT_SURAH_ALIASES: dict[str, int] = {
    "Al-Kahfi": 18,
    "Thaha": 20,
    "Al-Mujadalah": 58,
    "Al-Insyirah": 94,
    "Alam Nasyrah": 94,
    "Bani Israil": 17,
    "Ad-Dahr": 76,
    "Al-Lahab": 111,
    "Ath-Thur": 52,
    "Adhdhuha": 93,
    "Adh-Dhuha": 93,
    "Al-Mu'min": 40,
    "Alu 'Imran": 3,
    "Qn-Nisa": 4,
    "Lukman": 31,
    "As Sajadah": 32,
    "Al Hujarat": 49,
}


def _tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [t for t in re.split(r"[^a-z]+", s) if t]


def _fold(s: str) -> str:
    for a, b in _FOLDS:
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)
    if len(s) > 3 and s.endswith("h") and s[-2] in "aiu":
        s = s[:-1]
    return s


def surah_keys(name: str) -> list[str]:
    """Lookup keys for a surah name: whole name, then without the article.

    "Al Baqarah", "Al-Baqarah", "Albaqarah" -> "albaqara" (+ "baqara").
    """
    toks = _tokens(name)
    if not toks:
        return []
    keys = [_fold("".join(toks))]
    if len(toks) > 1 and toks[0] in _ARTICLES:
        stripped = _fold("".join(toks[1:]))
        if stripped and stripped not in keys:
            keys.append(stripped)
    return keys


# "Al-Quran - Al-Kahf (الكهف) - ...\nSurat ke-18, Makkiyah, Ayat 91-110"
_QURAN_HEAD = re.compile(
    r"^Al-Quran - (?P<name>.+?) \([^\n]*\n\s*Surat ke-(?P<surah>\d+),[^\n]*?"
    r"Ayat (?P<a1>\d+)(?:\s*-\s*(?P<a2>\d+))?"
)


def parse_quran_chunk(content: str) -> Optional[tuple[int, str, int, int]]:
    """(surah, name, first_ayah, last_ayah) of a Qur'an chunk, else None."""
    m = _QURAN_HEAD.match(content or "")
    if not m:
        return None
    a1 = int(m["a1"])
    a2 = int(m["a2"]) if m["a2"] else a1
    return int(m["surah"]), m["name"].strip(), a1, a2


@dataclass
class SurahIndex:
    names: dict[int, str] = field(default_factory=dict)
    ayah_counts: dict[int, int] = field(default_factory=dict)
    keys: dict[str, int] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)

    def resolve(self, name: str) -> Optional[int]:
        for k in surah_keys(name):
            if k in self.keys:
                return self.keys[k]
        return None


def build_surah_index(quran_chunks: Iterable[str],
                      aliases: Optional[dict] = None) -> SurahIndex:
    """Index from stored Qur'an chunks + aliases {name: surah_number}.

    A key claimed by two different surahs is dropped (ambiguous) unless
    an alias pins it explicitly.
    """
    idx = SurahIndex()
    for content in quran_chunks:
        p = parse_quran_chunk(content)
        if not p:
            continue
        surah, name, _a1, a2 = p
        idx.names.setdefault(surah, name)
        idx.ayah_counts[surah] = max(idx.ayah_counts.get(surah, 0), a2)

    claims: dict[str, set] = {}
    for surah, name in idx.names.items():
        for k in surah_keys(name):
            claims.setdefault(k, set()).add(surah)
    for k, owners in claims.items():
        if len(owners) == 1:
            idx.keys[k] = next(iter(owners))
        else:
            idx.ambiguous.add(k)

    for alias, surah in (aliases or {}).items():
        try:
            surah = int(surah)
        except (TypeError, ValueError):
            continue
        if surah not in idx.names:
            continue
        for k in surah_keys(alias):
            idx.keys[k] = surah
            idx.ambiguous.discard(k)
    return idx


# ── References ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ref:
    surah: int
    start: int
    end: int

    def ayahs(self) -> range:
        return range(self.start, self.end + 1)


def ayah_entity_name(surah: int, ayah: int) -> str:
    return f"QS {surah}:{ayah}"


def make_ref(index: SurahIndex, name: Optional[str], surah_num,
             a1, a2=None) -> tuple[Optional[Ref], str]:
    """Validated Ref, or (None, reason). Never guesses."""
    by_name = index.resolve(name) if name and name.strip() else None
    snum = int(surah_num) if surah_num else None
    if name and name.strip() and by_name is None and snum is None:
        return None, "unknown_surah"
    if snum and by_name and snum != by_name:
        return None, "surah_conflict"
    surah = snum or by_name
    if not surah:
        return None, "no_surah"
    count = index.ayah_counts.get(surah)
    if not count:
        return None, "unknown_surah"
    start = int(a1)
    end = int(a2) if a2 else start
    if start < 1 or start > count:
        return None, "ayah_out_of_range"
    if end < start or end > count:
        return None, "bad_range"
    return Ref(surah, start, end), "ok"


_L = r"A-Za-z\u00C0-\u024F\u1E00-\u1EFF"

# "QS. Al-Baqarah: 190-193", "QS Al Lail; 5", "QS. Ibrahim (14): 48",
# "QS al-Kahfi ayat 71 - 73", "QS. An-Nisa' 4:43", "QS. 2:255".
DEFAULT_QS_PATTERN = (
    r"\bQ\.?\s?S\.?\s*"
    rf"(?P<name>[{_L}][{_L}'\u2019\u2018\u02bf\u02be`\- ]{{0,38}}?)?"
    r"\s*(?:[(\[]\s*(?P<snum>\d{1,3})\s*[)\]])?"
    r"\s*(?:(?P<snum2>\d{1,3})\s*:\s*)?"
    r"\s*(?:[:;,.]|(?i:ayat|ayah))?"
    r"\s*(?P<a1>\d{1,3})(?:\s*[-\u2013\u2014]\s*(?P<a2>\d{1,3}))?(?!\d)"
)

# Title patterns per reference category. Named groups: name, surah, a1, a2.
DEFAULT_TITLE_PATTERNS: dict[str, str] = {
    "ibnukatsir": (
        r"^Tafsir Ibnu Katsir \u2014 (?P<name>.+?) Ayat (?P<a1>\d+)"
        r"(?:\s*-\s*(?P<a2>\d+))?(?:\s*\((?:bagian|part) \d+/\d+\))?\s*$"
    ),
    "asbabnuzul": (
        r"^Asbab Nuzul \u2014 QS\.? (?P<name>.+?) (?P<surah>\d+):(?P<a1>\d+)"
        r"(?:\s*-\s*(?P<a2>\d+))?\s*$"
    ),
}

DEFAULT_PART_PATTERN = r"\s*\((?:bagian|part)\s+(?P<i>\d+)\s*/\s*(?P<n>\d+)\)"


def parse_mentions(text: str, index: SurahIndex,
                   pattern: str = DEFAULT_QS_PATTERN
                   ) -> tuple[list[Ref], list[tuple[str, str]]]:
    """QS mentions in text -> (refs, rejects[(raw, reason)])."""
    refs: list[Ref] = []
    rejects: list[tuple[str, str]] = []
    for m in re.finditer(pattern, text or ""):
        name = (m.group("name") or "").strip(" -'")
        ref, why = make_ref(index, name or None,
                            m.group("snum") or m.group("snum2"),
                            m.group("a1"), m.group("a2"))
        if ref is None:
            rejects.append((m.group(0).strip(), why))
        elif ref not in refs:
            refs.append(ref)
    return refs, rejects


def parse_title(title: str, pattern: str,
                index: SurahIndex) -> tuple[Optional[Ref], str]:
    """Ref written in a chunk title (tafsir / asbab), or (None, reason)."""
    m = re.match(pattern, (title or "").strip())
    if not m:
        return None, "no_match"
    g = m.groupdict()
    return make_ref(index, (g.get("name") or "").strip() or None,
                    g.get("surah"), g.get("a1"), g.get("a2"))


def parse_part(title: str, pattern: str = DEFAULT_PART_PATTERN
               ) -> Optional[tuple[str, int, int]]:
    """(base_title, i, n) for "... (bagian i/n)", else None."""
    t = (title or "").strip()
    m = re.search(pattern, t)
    if not m:
        return None
    i, n = int(m["i"]), int(m["n"])
    if i < 1 or n < 1 or i > n:
        return None
    base = (t[:m.start()] + t[m.end():]).strip()
    return base, i, n
