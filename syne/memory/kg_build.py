"""Rule-built KG: links between reference chunks from written references.

No AI. Reads chunks from `memory`, recognizes references with kg_refs,
and writes them into the existing KG tables (Riyo, 14 Nov 2026):

  * one 'ayah' entity per Qur'an verse, named "QS surah:ayah";
  * one 'chunk' entity per reference chunk (description "memory:<id>");
  * relations chunk -> ayah ('contains' / title predicate / 'cites'),
    and part i -> part i+1 ('next_part');
  * every relation gets a kg_relation_sources row, origin 'rule'.

Per-reference rules live in config (kg.ref_sources, kg.surah_aliases,
kg.qs_pattern, kg.part_pattern, kg.max_mention_range); the defaults
below apply when a key is not set. Idempotent: re-running upserts.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from . import kg_refs as R

logger = logging.getLogger("syne.memory.kg_build")

QURAN = "quran"

DEFAULT_REF_SOURCES: dict[str, dict] = {
    "alquran": {"kind": QURAN},
    "ibnukatsir": {"title": R.DEFAULT_TITLE_PATTERNS["ibnukatsir"],
                   "title_predicate": "explains", "mentions": True},
    "asbabnuzul": {"title": R.DEFAULT_TITLE_PATTERNS["asbabnuzul"],
                   "title_predicate": "asbab_of", "mentions": True},
    "bukhari": {"mentions": True},
    "muslim": {"mentions": True},
    "riyadhus": {"mentions": True},
    "aqidah": {"mentions": True},
    "fiqih": {"mentions": True},
    "fiqih_ext": {"mentions": True},
    "sirah": {"mentions": True},
    "kaidah": {"mentions": True},
}
DEFAULT_MAX_MENTION_RANGE = 50

AYAH = "ayah"
CHUNK = "chunk"


def chunk_entity_name(memory_id: int, title: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip())[:200]
    return f"{t} [#{memory_id}]"


def first_line(content: str) -> str:
    return (content or "").strip().split("\n", 1)[0].strip()


@dataclass
class BuildPlan:
    entities: dict = field(default_factory=dict)   # (lname, type) -> (name, type, desc)
    relations: dict = field(default_factory=dict)  # (subj_key, pred, obj_key) -> memory_id
    rejects: Counter = field(default_factory=Counter)
    reject_samples: dict = field(default_factory=lambda: defaultdict(list))
    per_predicate: Counter = field(default_factory=Counter)
    chunks: int = 0

    def entity(self, name: str, etype: str, desc: str = "") -> tuple:
        key = (name.lower(), etype)
        self.entities.setdefault(key, (name, etype, desc))
        return key

    def relate(self, subj: tuple, pred: str, obj: tuple, memory_id: int) -> None:
        k = (subj, pred, obj)
        if k not in self.relations:
            self.relations[k] = memory_id
            self.per_predicate[pred] += 1

    def reject(self, category: str, raw: str, why: str) -> None:
        self.rejects[(category, why)] += 1
        s = self.reject_samples[(category, why)]
        if len(s) < 5:
            s.append(raw[:100])


def plan_links(rows: list[dict], sources: dict, aliases: Optional[dict] = None,
               qs_pattern: str = R.DEFAULT_QS_PATTERN,
               part_pattern: str = R.DEFAULT_PART_PATTERN,
               max_range: int = DEFAULT_MAX_MENTION_RANGE) -> BuildPlan:
    """Pure: rows [{id, category, content}] -> BuildPlan."""
    plan = BuildPlan()
    quran_cats = {c for c, s in sources.items() if s.get("kind") == QURAN}
    index = R.build_surah_index(
        (r["content"] for r in rows if r["category"] in quran_cats),
        {**R.DEFAULT_SURAH_ALIASES, **(aliases or {})},
    )

    def ayah_key(surah: int, ayah: int) -> tuple:
        return plan.entity(R.ayah_entity_name(surah, ayah), AYAH,
                           f"{index.names.get(surah, '')} {ayah}".strip())

    parts: dict = defaultdict(dict)  # (category, base) -> {i: chunk_key}
    for row in rows:
        cat = row["category"]
        src = sources.get(cat)
        if src is None:
            continue
        mid = int(row["id"])
        content = row["content"] or ""
        title = first_line(content)
        ck = plan.entity(chunk_entity_name(mid, title), CHUNK, f"memory:{mid}")
        plan.chunks += 1

        if src.get("kind") == QURAN:
            p = R.parse_quran_chunk(content)
            if not p:
                plan.reject(cat, title, "no_match")
            else:
                surah, _n, a1, a2 = p
                for a in range(a1, a2 + 1):
                    plan.relate(ck, "contains", ayah_key(surah, a), mid)
            continue

        title_ref = None
        if src.get("title"):
            title_ref, why = R.parse_title(title, src["title"], index)
            if title_ref is None:
                plan.reject(cat, title, why)
            else:
                pred = src.get("title_predicate") or "about"
                for a in title_ref.ayahs():
                    plan.relate(ck, pred, ayah_key(title_ref.surah, a), mid)

        if src.get("mentions"):
            refs, rej = R.parse_mentions(content, index, qs_pattern)
            for raw, why in rej:
                plan.reject(cat, raw, why)
            for ref in refs:
                if ref == title_ref:
                    continue
                if ref.end - ref.start + 1 > max_range:
                    plan.reject(cat, f"{ref.surah}:{ref.start}-{ref.end}", "range_too_long")
                    continue
                for a in ref.ayahs():
                    plan.relate(ck, "cites", ayah_key(ref.surah, a), mid)

        pp = R.parse_part(title, part_pattern)
        if pp:
            base, i, _n = pp
            parts[(cat, base)].setdefault(i, (ck, mid))

    for group in parts.values():
        for i in sorted(group):
            nxt = group.get(i + 1)
            if nxt:
                ck, mid = group[i]
                plan.relate(ck, "next_part", nxt[0], mid)
    return plan


async def load_settings_from_config() -> dict:
    from ..db.models import get_config
    return {
        "sources": await get_config("kg.ref_sources", None) or DEFAULT_REF_SOURCES,
        "aliases": await get_config("kg.surah_aliases", None) or {},
        "qs_pattern": await get_config("kg.qs_pattern", None) or R.DEFAULT_QS_PATTERN,
        "part_pattern": await get_config("kg.part_pattern", None) or R.DEFAULT_PART_PATTERN,
        "max_range": int(await get_config("kg.max_mention_range", None)
                         or DEFAULT_MAX_MENTION_RANGE),
    }


async def build(conn, categories: Optional[list[str]] = None,
                dry_run: bool = False, settings: Optional[dict] = None) -> BuildPlan:
    """Plan from the database, then write it (unless dry_run).

    The Qur'an chunks are always read so mentions can be validated, even
    when `categories` limits which references get linked.
    """
    st = settings or await load_settings_from_config()
    sources = dict(st["sources"])
    if categories:
        keep = set(categories) | {c for c, s in sources.items() if s.get("kind") == QURAN}
        sources = {c: s for c, s in sources.items() if c in keep}
    rows = await conn.fetch(
        "SELECT id, category, content FROM memory WHERE category = ANY($1::text[]) ORDER BY id",
        list(sources),
    )
    plan = plan_links([dict(r) for r in rows], sources, st["aliases"],
                      st["qs_pattern"], st["part_pattern"], st["max_range"])
    if not dry_run:
        await write_plan(conn, plan)
    return plan


_BATCH = 5000


async def write_plan(conn, plan: BuildPlan) -> None:
    """Upsert entities, relations and their 'rule' sources in batches."""
    ids: dict = {}
    ents = list(plan.entities.items())
    for i in range(0, len(ents), _BATCH):
        part = ents[i:i + _BATCH]
        rows = await conn.fetch(
            """INSERT INTO kg_entities (name, entity_type, description)
               SELECT * FROM unnest($1::text[], $2::text[], $3::text[])
               ON CONFLICT (LOWER(name), entity_type)
               DO UPDATE SET description = EXCLUDED.description, updated_at = NOW()
               RETURNING id, LOWER(name) AS lname, entity_type""",
            [v[0] for _k, v in part], [v[1] for _k, v in part], [v[2] for _k, v in part],
        )
        for r in rows:
            ids[(r["lname"], r["entity_type"])] = r["id"]

    rels = [(ids[s], p, ids[o], m) for (s, p, o), m in plan.relations.items()
            if s in ids and o in ids]
    for i in range(0, len(rels), _BATCH):
        part = rels[i:i + _BATCH]
        async with conn.transaction():
            rows = await conn.fetch(
                """INSERT INTO kg_relations (subject_id, predicate, object_id, source_memory_id)
                   SELECT * FROM unnest($1::int[], $2::text[], $3::int[], $4::int[])
                   ON CONFLICT (subject_id, predicate, object_id)
                   DO UPDATE SET source_memory_id = EXCLUDED.source_memory_id, updated_at = NOW()
                   RETURNING id, source_memory_id""",
                [x[0] for x in part], [x[1] for x in part],
                [x[2] for x in part], [x[3] for x in part],
            )
            await conn.execute(
                """INSERT INTO kg_relation_sources (relation_id, source_type, source_id, origin)
                   SELECT rid, 'memory', mid, 'rule' FROM unnest($1::int[], $2::int[]) AS t(rid, mid)
                   ON CONFLICT DO NOTHING""",
                [r["id"] for r in rows], [r["source_memory_id"] for r in rows],
            )


async def prune_orphan_chunks(conn) -> int:
    """Delete chunk entities whose memory row is gone (relations cascade)."""
    res = await conn.execute(
        """DELETE FROM kg_entities e
           WHERE e.entity_type = 'chunk' AND e.description LIKE 'memory:%'
             AND NOT EXISTS (SELECT 1 FROM memory m
                             WHERE m.id = substring(e.description from 7)::int)"""
    )
    try:
        return int(res.split()[-1])
    except (ValueError, IndexError, AttributeError):
        return 0
