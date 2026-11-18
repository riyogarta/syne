"""Written-reference recognition for the rule-built KG (no AI).

Pure functions, no database access:
  * name normalization (spelling variants fold to one key; rules per target),
  * an index of a link target built from its stored chunks,
  * parsers for references written in chunk titles and text:
    mentions ("QS ..."), tafsir / asbab-nuzul titles, "(bagian i/n)" parts.

A link target is a reference whose chunks hold numbered items: the Qur'an
(surah:ayah) or e.g. an insurance policy (policy:article). Its rules live in
the 'target' object of its kg.ref_sources entry (v1.23.02); the Qur'an
rules below are the code defaults, so the Qur'an behaves exactly as before.
Anything that cannot be resolved with certainty is returned as a reject
(with a reason) instead of being linked (Riyo, 14 Nov 2026).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ── Qur'an defaults ──────────────────────────────────────────────────────

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
# has no aliases (the config list, when present, replaces this one).
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
    "At Taqwir": 81,
    "At Thaghobun": 64,
}

_L = r"A-Za-z\u00C0-\u024F\u1E00-\u1EFF"

# "QS. Al-Baqarah: 190-193", "QS Al Lail; 5", "QS. Ibrahim (14): 48",
# "QS al-Kahfi ayat 71 - 73", "QS. An-Nisa' 4:43", "QS. 2:255",
# "QS surat an-Naml ayat 39", "QS an-Nisaa' ayat; 6", "QS Al Ma'idah ayat: 3",
# "QS. As-Saff, 61:4" (name, surah:ayah).
DEFAULT_QS_PATTERN = (
    r"\bQ\.?\s?S\.?\s*"
    r"(?:(?i:surat|surah)\s+)?"
    rf"(?P<name>[{_L}][{_L}'\u2019\u2018\u02bf\u02be`\- ]{{0,38}}?)?"
    r"\s*(?:[(\[]\s*(?P<snum>\d{1,3})\s*[)\]])?"
    r"\s*(?:,?\s*(?P<snum2>\d{1,3})\s*:\s*)?"
    r"\s*(?:(?i:ayat|ayah)\s*[:;,.]?|[:;,.]\s*(?i:ayat|ayah)?)?"
    r"\s*(?P<a1>\d{1,3})(?:\s*[-\u2013\u2014]\s*(?P<a2>\d{1,3}))?(?!\d)"
)

# "Al-Quran - Al-Kahf (الكهف) - ...\nSurat ke-18, Makkiyah, Ayat 91-110"
_QURAN_HEAD_PATTERN = (
    r"^Al-Quran - (?P<name>.+?) \([^\n]*\n\s*Surat ke-(?P<unit>\d+),[^\n]*?"
    r"Ayat (?P<a1>\d+)(?:\s*-\s*(?P<a2>\d+))?"
)
_QURAN_HEAD = re.compile(_QURAN_HEAD_PATTERN)

# A whole entity name may start with "QS"/"Q.S." or not; it is stripped and
# "QS " put back so DEFAULT_QS_PATTERN can match the whole name.
_QS_PREFIX_PATTERN = r"^\s*(?i:q\.?\s?s\.?)\s*"
_QS_PREFIX = re.compile(_QS_PREFIX_PATTERN)

_QURAN_NODE_TYPE = "ayah"
_QURAN_NODE_FORMAT = "QS {unit}:{item}"
_QURAN_NODE_REGEX = r"QS (\d{1,3}):(\d{1,3})"


# ── Name normalization ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Normalizer:
    """Folds spelling variants of a unit name to lookup keys.

    Empty (the default for a non-Qur'an target): lowercase, accents and
    non-letters removed. The Qur'an adds Arabic articles, transliteration
    folds, doubled letters squeezed and a final 'h' after a vowel dropped.
    """
    articles: frozenset = frozenset()
    folds: tuple = ()
    squeeze: bool = False
    drop_final_h: bool = False

    def fold(self, s: str) -> str:
        for a, b in self.folds:
            s = s.replace(a, b)
        if self.squeeze:
            s = re.sub(r"(.)\1+", r"\1", s)
        if self.drop_final_h and len(s) > 3 and s.endswith("h") and s[-2] in "aiu":
            s = s[:-1]
        return s

    def keys(self, name: str) -> list[str]:
        """Whole name, then without a leading article."""
        toks = _tokens(name)
        if not toks:
            return []
        keys = [self.fold("".join(toks))]
        if len(toks) > 1 and toks[0] in self.articles:
            stripped = self.fold("".join(toks[1:]))
            if stripped and stripped not in keys:
                keys.append(stripped)
        return keys


QURAN_NORMALIZER = Normalizer(_ARTICLES, _FOLDS, True, True)


def normalizer_from(cfg) -> Normalizer:
    """Normalizer from a target's 'normalize' object (missing = empty)."""
    if not isinstance(cfg, dict):
        return Normalizer()
    arts = frozenset(str(a).lower() for a in (cfg.get("articles") or [])
                     if isinstance(a, str) and a)
    folds = tuple((p[0].lower(), p[1].lower()) for p in (cfg.get("folds") or [])
                  if isinstance(p, (list, tuple)) and len(p) == 2
                  and isinstance(p[0], str) and isinstance(p[1], str) and p[0])
    return Normalizer(arts, folds, bool(cfg.get("squeeze")), bool(cfg.get("drop_final_h")))


def _tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [t for t in re.split(r"[^a-z]+", s) if t]


def _fold(s: str) -> str:
    return QURAN_NORMALIZER.fold(s)


def surah_keys(name: str) -> list[str]:
    """Lookup keys for a surah name: whole name, then without the article.

    "Al Baqarah", "Al-Baqarah", "Albaqarah" -> "albaqara" (+ "baqara").
    """
    return QURAN_NORMALIZER.keys(name)


# ── Link targets (config) ────────────────────────────────────────────────

_TARGET_KEYS = ("node_type", "node_format", "node_regex", "head", "mention")


def quran_target(mention: Optional[str] = None, aliases: Optional[dict] = None) -> dict:
    """The Qur'an's target rules (code defaults of kg.ref_sources)."""
    return {
        "node_type": _QURAN_NODE_TYPE,
        "node_format": _QURAN_NODE_FORMAT,
        "node_regex": _QURAN_NODE_REGEX,
        "head": _QURAN_HEAD_PATTERN,
        "mention": mention or DEFAULT_QS_PATTERN,
        "whole_prefix": "QS ",
        "whole_strip": _QS_PREFIX_PATTERN,
        "normalize": {"articles": sorted(_ARTICLES), "folds": [list(f) for f in _FOLDS],
                      "squeeze": True, "drop_final_h": True},
        "aliases": dict(DEFAULT_SURAH_ALIASES if aliases is None else aliases),
    }


def target_config(entry, legacy_mention: Optional[str] = None) -> Optional[dict]:
    """Full target rules of one kg.ref_sources entry, or None (not a target).

    kind 'quran' = the Qur'an defaults, overridden by the entry's 'target'
    object (an entry from before v1.23.02 has kind + aliases only). Any other
    entry is a target only with a complete 'target' object. legacy_mention
    (kg.qs_pattern, until migration 55 removes it) replaces the Qur'an mention
    pattern unless the target sets its own. Invalid rules give None.
    """
    if not isinstance(entry, dict):
        return None
    t = entry.get("target") if isinstance(entry.get("target"), dict) else None
    if entry.get("kind") == "quran":
        aliases = entry["aliases"] if isinstance(entry.get("aliases"), dict) else None
        base = quran_target(aliases=aliases)
        if legacy_mention and not (t and t.get("mention")):
            base["mention"] = legacy_mention
    elif t is not None:
        base = {"whole_prefix": "", "whole_strip": "", "normalize": {}, "aliases": {}}
    else:
        return None
    if t:
        base.update(t)
    if not all(isinstance(base.get(k), str) and base.get(k) for k in _TARGET_KEYS):
        return None
    try:
        for k in ("node_regex", "head", "mention"):
            re.compile(base[k])
        if base.get("whole_strip"):
            re.compile(base["whole_strip"])
        base["node_format"].format(unit=1, item=1)
    except (re.error, KeyError, IndexError, ValueError, TypeError):
        return None
    if not isinstance(base.get("aliases"), dict):
        base["aliases"] = {}
    if not isinstance(base.get("whole_prefix"), str):
        base["whole_prefix"] = ""
    if not isinstance(base.get("whole_strip"), str):
        base["whole_strip"] = ""
    return base


def target_configs(sources, legacy_mention: Optional[str] = None) -> dict[str, dict]:
    """{category: target rules} for every link target in kg.ref_sources."""
    out: dict[str, dict] = {}
    for cat, entry in (sources or {}).items():
        t = target_config(entry, legacy_mention)
        if t:
            out[cat] = t
    return out


def target_node_types(sources) -> list[str]:
    """Node types of the link targets ('ayah' for the Qur'an), in order."""
    out: list[str] = []
    for t in target_configs(sources).values():
        if t["node_type"] not in out:
            out.append(t["node_type"])
    return out


# ── Index ────────────────────────────────────────────────────────────────

def parse_head(content: str, pattern: str) -> Optional[tuple[int, str, int, int]]:
    """(unit, name, first_item, last_item) from a target chunk's head, else None.

    Groups: unit (or surah), a1, optional a2, optional name.
    """
    m = re.match(pattern, content or "")
    if not m:
        return None
    g = m.groupdict()
    unit = g.get("unit") or g.get("surah")
    if not unit or not g.get("a1"):
        return None
    a1 = int(g["a1"])
    a2 = int(g["a2"]) if g.get("a2") else a1
    return int(unit), (g.get("name") or "").strip(), a1, a2


def parse_quran_chunk(content: str) -> Optional[tuple[int, str, int, int]]:
    """(surah, name, first_ayah, last_ayah) of a Qur'an chunk, else None."""
    return parse_head(content, _QURAN_HEAD_PATTERN)


@dataclass
class SurahIndex:
    """Index of one link target: unit names, item counts and lookup keys,
    plus the target's rules. Defaults are the Qur'an's."""
    names: dict[int, str] = field(default_factory=dict)
    ayah_counts: dict[int, int] = field(default_factory=dict)
    keys: dict[str, int] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)
    norm: Normalizer = QURAN_NORMALIZER
    node_type: str = _QURAN_NODE_TYPE
    node_format: str = _QURAN_NODE_FORMAT
    node_regex: str = _QURAN_NODE_REGEX
    head: str = _QURAN_HEAD_PATTERN
    mention: str = DEFAULT_QS_PATTERN
    whole_prefix: str = "QS "
    whole_strip: str = _QS_PREFIX_PATTERN

    def resolve(self, name: str) -> Optional[int]:
        for k in self.norm.keys(name):
            if k in self.keys:
                return self.keys[k]
        return None

    def node_name(self, unit: int, item: int) -> str:
        return self.node_format.format(unit=unit, item=item)

    def node_desc(self, unit: int, item: int) -> str:
        return f"{self.names.get(unit, '')} {item}".strip()


RefIndex = SurahIndex


def _empty_index(cfg: Optional[dict]) -> SurahIndex:
    if cfg is None:
        return SurahIndex()
    return SurahIndex(
        norm=normalizer_from(cfg.get("normalize")),
        node_type=cfg["node_type"], node_format=cfg["node_format"],
        node_regex=cfg["node_regex"], head=cfg["head"], mention=cfg["mention"],
        whole_prefix=cfg.get("whole_prefix") or "", whole_strip=cfg.get("whole_strip") or "",
    )


def build_target_index(chunks: Iterable[str], cfg: dict,
                       extra_aliases: Optional[dict] = None) -> SurahIndex:
    """Index of a target from its stored chunks (heads) + its aliases."""
    idx = _empty_index(cfg)
    for content in chunks:
        p = parse_head(content, idx.head)
        if not p:
            continue
        unit, name, _a1, a2 = p
        if name:
            idx.names.setdefault(unit, name)
        else:
            idx.names.setdefault(unit, "")
        idx.ayah_counts[unit] = max(idx.ayah_counts.get(unit, 0), a2)
    return _index_keys(idx, {**(cfg.get("aliases") or {}), **(extra_aliases or {})})


def target_index_from(names: dict, counts: dict, cfg: dict) -> SurahIndex:
    """Index of a target from known unit names and item counts (read back
    from its node entities)."""
    idx = _empty_index(cfg)
    idx.names, idx.ayah_counts = dict(names), dict(counts)
    return _index_keys(idx, cfg.get("aliases") or {})


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

    return _index_keys(idx, aliases)


def _index_keys(idx: SurahIndex, aliases: Optional[dict]) -> SurahIndex:
    """Lookup keys from idx.names, then aliases (see build_surah_index)."""
    claims: dict[str, set] = {}
    for unit, name in idx.names.items():
        for k in idx.norm.keys(name):
            claims.setdefault(k, set()).add(unit)
    for k, owners in claims.items():
        if len(owners) == 1:
            idx.keys[k] = next(iter(owners))
        else:
            idx.ambiguous.add(k)

    for alias, unit in (aliases or {}).items():
        try:
            unit = int(unit)
        except (TypeError, ValueError):
            continue
        if unit not in idx.names:
            continue
        for k in idx.norm.keys(alias):
            idx.keys[k] = unit
            idx.ambiguous.discard(k)
    return idx


def surah_index_from(names: dict, ayah_counts: dict,
                     aliases: Optional[dict] = None) -> SurahIndex:
    """Index from known surah names {number: name} and ayah counts
    {number: last ayah}, e.g. read back from the 'ayah' entities."""
    idx = SurahIndex(names=dict(names), ayah_counts=dict(ayah_counts))
    return _index_keys(idx, aliases)


# ── References ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ref:
    surah: int   # the unit (surah for the Qur'an)
    start: int
    end: int

    def ayahs(self) -> range:
        return range(self.start, self.end + 1)


def ayah_entity_name(surah: int, ayah: int) -> str:
    return _QURAN_NODE_FORMAT.format(unit=surah, item=ayah)


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


def _unit_num(g: dict):
    return g.get("snum") or g.get("snum2") or g.get("unit") or g.get("surah")


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
                   pattern: Optional[str] = None
                   ) -> tuple[list[Ref], list[tuple[str, str]]]:
    """Mentions of the index's target in text -> (refs, rejects[(raw, reason)]).

    pattern defaults to the target's mention pattern ("QS ..." for the Qur'an).
    """
    refs: list[Ref] = []
    rejects: list[tuple[str, str]] = []
    for m in re.finditer(pattern or index.mention, text or ""):
        g = m.groupdict()
        name = (g.get("name") or "").strip(" -'")
        ref, why = make_ref(index, name or None, _unit_num(g), g.get("a1"), g.get("a2"))
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
                    g.get("surah") or g.get("unit"), g.get("a1"), g.get("a2"))


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


def parse_ayah_ref(name: str, index: SurahIndex,
                   pattern: Optional[str] = None) -> Optional[Ref]:
    """Ref when the WHOLE name is one item or one range of the target, else None.

    Qur'an: "QS 54:46", "QS. Al-Qamar: 46", "Al-Baqarah ayat 255",
    "Surah Al-Kahfi 10", "QS. Al-Baqarah (2): 255", "QS 18:71-73". An
    unknown unit, an item out of range or any other text gives None.
    """
    s = (name or "").strip()
    if not s or not re.search(r"\d", s):
        return None
    body = re.sub(index.whole_strip, "", s) if index.whole_strip else s
    m = re.fullmatch(pattern or index.mention, index.whole_prefix + body)
    if not m:
        return None
    g = m.groupdict()
    nm = (g.get("name") or "").strip(" -'")
    ref, _why = make_ref(index, nm or None, _unit_num(g), g.get("a1"), g.get("a2"))
    return ref


parse_ref = parse_ayah_ref


def parse_single_ayah(name: str, index: SurahIndex,
                      pattern: Optional[str] = None) -> Optional[tuple[int, int]]:
    """(unit, item) when the WHOLE name is exactly one item, else None."""
    ref = parse_ayah_ref(name, index, pattern)
    if ref is None or ref.end != ref.start:
        return None
    return ref.surah, ref.start
