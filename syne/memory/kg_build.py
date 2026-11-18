"""Rule-built KG: links between reference chunks from written references.

No AI. Reads chunks from `memory`, recognizes references with kg_refs,
and writes them into the existing KG tables (Riyo, 14 Nov 2026):

  * one node entity per item of a link target: for the Qur'an one 'ayah'
    entity per verse, named "QS surah:ayah" (v1.23.02: any reference with
    a 'target' object in kg.ref_sources is a link target, e.g. a policy
    with numbered articles);
  * one 'chunk' entity per reference chunk (description "memory:<id>");
  * relations chunk -> node ('contains' / title predicate / 'cites'),
    and part i -> part i+1 ('next_part');
  * every relation gets a kg_relation_sources row, origin 'rule'.

Rules live in config (kg.ref_sources with each target's rules and aliases,
kg.part_pattern, kg.max_mention_range), seeded by migration 43 from
default_config(); the code defaults below apply only when a key is
missing. Idempotent: re-running upserts.
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
    "alquran": {"kind": QURAN, "target": R.quran_target()},
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
DEFAULT_CHUNK_LINK_LIMIT = 10


def default_config() -> dict:
    """{config key: (default value, description)} for the rule-built KG."""
    return {
        "kg.ref_sources": (
            DEFAULT_REF_SOURCES,
            "Rule-built KG: references to link, by memory category. "
            "{category: {target: {node_type, node_format ('QS {unit}:{item}'), "
            "node_regex, head (regex on the target's chunks: unit, a1, a2, name), "
            "mention (regex for mentions: name, snum/snum2, a1, a2), normalize "
            "{articles, folds, squeeze, drop_final_h}, aliases {name: unit}} = a link "
            "target | title: regex, title_predicate, title_target | mentions: true}}. "
            "kind 'quran' = Qur'an defaults for the target. "
            "Re-run `syne memory kg-build` after a change.",
        ),
        "kg.part_pattern": (
            R.DEFAULT_PART_PATTERN,
            "Rule-built KG: regex for '(bagian i/n)' in a chunk title (groups i, n); "
            "consecutive parts are linked next_part.",
        ),
        "kg.max_mention_range": (
            DEFAULT_MAX_MENTION_RANGE,
            "Rule-built KG: a mentioned range of items (e.g. ayahs) longer than this "
            "is not linked.",
        ),
        "kg.chunk_link_limit": (
            DEFAULT_CHUNK_LINK_LIMIT,
            "Max KG lines added per turn from the links of recalled reference "
            "chunks (chunk -> target item -> other chunks). 0 = off.",
        ),
    }

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
               qs_pattern: Optional[str] = None,
               part_pattern: str = R.DEFAULT_PART_PATTERN,
               max_range: int = DEFAULT_MAX_MENTION_RANGE) -> BuildPlan:
    """Pure: rows [{id, category, content}] -> BuildPlan.

    Every link target in `sources` gets an index from its own chunks; text
    of the other references is matched against every target. `aliases`
    adds to each target's aliases; `qs_pattern` is the legacy
    kg.qs_pattern (Qur'an mention pattern, until migration 55 moves it).
    """
    plan = BuildPlan()
    targets = R.target_configs(sources, qs_pattern)
    indexes: dict = {
        cat: R.build_target_index(
            (r["content"] for r in rows if r["category"] == cat), cfg, aliases)
        for cat, cfg in targets.items()
    }

    def node_key(index, unit: int, item: int) -> tuple:
        return plan.entity(index.node_name(unit, item), index.node_type,
                           index.node_desc(unit, item))

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

        if cat in indexes:
            index = indexes[cat]
            p = R.parse_head(content, index.head)
            if not p:
                plan.reject(cat, title, "no_match")
            else:
                unit, _n, a1, a2 = p
                for a in range(a1, a2 + 1):
                    plan.relate(ck, "contains", node_key(index, unit, a), mid)
            continue
        if src.get("kind") == QURAN:
            plan.reject(cat, title, "bad_target")
            continue

        title_ref = None
        title_idx = None
        if src.get("title") and indexes:
            tcat = src.get("title_target")
            title_idx = indexes.get(tcat) if tcat else next(iter(indexes.values()))
            if title_idx is None:
                plan.reject(cat, title, "unknown_target")
            else:
                title_ref, why = R.parse_title(title, src["title"], title_idx)
                if title_ref is None:
                    plan.reject(cat, title, why)
                else:
                    pred = src.get("title_predicate") or "about"
                    for a in title_ref.ayahs():
                        plan.relate(ck, pred, node_key(title_idx, title_ref.surah, a), mid)

        if src.get("mentions"):
            for index in indexes.values():
                refs, rej = R.parse_mentions(content, index)
                for raw, why in rej:
                    plan.reject(cat, raw, why)
                for ref in refs:
                    if index is title_idx and ref == title_ref:
                        continue
                    if ref.end - ref.start + 1 > max_range:
                        plan.reject(cat, f"{ref.surah}:{ref.start}-{ref.end}", "range_too_long")
                        continue
                    for a in ref.ayahs():
                        plan.relate(ck, "cites", node_key(index, ref.surah, a), mid)

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
    used = {k for rel in plan.relations for k in (rel[0], rel[2])}
    plan.entities = {k: v for k, v in plan.entities.items() if k in used}
    return plan


async def load_settings_from_config() -> dict:
    from ..db.models import get_config
    return {
        "sources": await get_config("kg.ref_sources", None) or DEFAULT_REF_SOURCES,
        "aliases": {},  # per reference, inside kg.ref_sources
        # Legacy key, removed by migration 55; None = each target's own pattern.
        "qs_pattern": await get_config("kg.qs_pattern", None),
        "part_pattern": await get_config("kg.part_pattern", None) or R.DEFAULT_PART_PATTERN,
        "max_range": int(await get_config("kg.max_mention_range", None)
                         or DEFAULT_MAX_MENTION_RANGE),
    }


async def build(conn, categories: Optional[list[str]] = None,
                dry_run: bool = False, settings: Optional[dict] = None) -> BuildPlan:
    """Plan from the database, then write it (unless dry_run).

    The link targets' chunks are always read so mentions can be validated,
    even when `categories` limits which references get linked.
    """
    st = settings or await load_settings_from_config()
    sources = dict(st["sources"])
    if categories:
        keep = set(categories) | set(R.target_configs(sources))
        sources = {c: s for c, s in sources.items() if c in keep}
    rows = await conn.fetch(
        "SELECT id, category, content FROM memory WHERE category = ANY($1::text[]) ORDER BY id",
        list(sources),
    )
    plan = plan_links([dict(r) for r in rows], sources, st["aliases"],
                      st.get("qs_pattern"), st["part_pattern"], st["max_range"])
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
                             WHERE m.id = substring(e.description from 8)::int)"""
    )
    try:
        return int(res.split()[-1])
    except (ValueError, IndexError, AttributeError):
        return 0
