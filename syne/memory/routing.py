"""Keyword-based category routing for memory recall.

Memory recall is semantic top-K. When a deployment loads a large domain corpus
(scripture, case law, medical references), that corpus dominates the memory table
and therefore dominates *every* recall — regardless of what the conversation is
actually about. A chat about token quota pulls tafsir passages because the word
"morning" happens to have scripture as its nearest neighbour.

Routing fixes that by mapping trigger keywords to target categories:

    memory.category_routes = {
        "islam, fiqih, ayat, hadits": ["fiqih", "alquran", "bukhari", ...],
        "rumah, listrik, pln":        ["home"],
    }

Rules:
  - a route fires when any of its keywords appears in the message (word-boundary)
  - fired  -> recall is restricted to the union of the fired routes' categories
  - none   -> recall excludes every routed category (i.e. searches the rest)
  - empty config -> no routing at all, original behaviour

Categories that appear in no route are the "default" set: always reachable.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger("syne.memory.routing")


def _normalize(text: str) -> str:
    """Lowercase + strip accents so 'Qur'ān' and 'quran' compare equal."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def _split_keywords(raw: str) -> list[str]:
    """Route keys are comma-separated keyword lists: 'islam, fiqih, ayat'."""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _matches(keyword: str, haystack: str) -> bool:
    """Word-boundary match.

    Substring matching is wrong here: the Indonesian trigger 'ayat' would fire on
    'pembayaran', and 'pln' on 'plnya'. Boundaries are computed on the normalized
    text so multi-word keywords ('surat al kahfi') still work.
    """
    kw = _normalize(keyword)
    if not kw:
        return False
    return re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", haystack) is not None


def parse_routes(raw) -> dict[str, list[str]]:
    """Coerce the config value into {keyword_csv: [category, ...]}.

    Tolerates a JSON string (some config backends round-trip as text) and a
    single category given as a bare string instead of a list.
    """
    if not raw:
        return {}
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except Exception:
            logger.warning("memory.category_routes is not valid JSON — routing disabled")
            return {}
    if not isinstance(raw, dict):
        logger.warning("memory.category_routes must be an object — routing disabled")
        return {}

    routes: dict[str, list[str]] = {}
    for key, cats in raw.items():
        if isinstance(cats, str):
            cats = [cats]
        if not isinstance(cats, (list, tuple)):
            continue
        clean = [str(c).strip().lower() for c in cats if str(c).strip()]
        if clean:
            routes[str(key)] = clean
    return routes


def resolve(
    message: str,
    raw_routes,
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """Decide which categories recall may search.

    Returns (include_categories, exclude_categories):
      - (targets, None) when at least one route fired
      - (None, all_routed) when none fired — search everything unrouted
      - (None, None) when routing is not configured
    """
    routes = parse_routes(raw_routes)
    if not routes:
        return None, None

    all_routed: list[str] = []
    for cats in routes.values():
        for c in cats:
            if c not in all_routed:
                all_routed.append(c)

    haystack = _normalize(message or "")

    hits: list[str] = []
    fired: list[str] = []
    for key, cats in routes.items():
        for kw in _split_keywords(key):
            if _matches(kw, haystack):
                fired.append(kw)
                for c in cats:
                    if c not in hits:
                        hits.append(c)
                break

    if hits:
        logger.debug(f"Memory routing: triggers {fired} -> categories {hits}")
        return hits, None

    logger.debug(f"Memory routing: no trigger -> excluding {len(all_routed)} routed categories")
    return None, all_routed
