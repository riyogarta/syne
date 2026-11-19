"""KG ingestion path (Riyo, 18 Nov 2026) - KG-RAG stage 2.

Gambar 1, "Jalur KG": pick the chunk -> LLM extracts entities and relations
-> entity resolution (name normalisation + name embedding, duplicates
joined) -> Graph DB (kg_entities / kg_relations, every relation keeps its
source chunk ID in kg_relation_sources).

Only NEW chunks are processed (Riyo, 18 Nov 2026, memory #57758): memory
rows and chat messages created at or after config kg.extract_era_start.
Old chunks are never extracted again. The era date is set once by
migration 53; relations from the old AI extraction are told apart from the
new ones by the created_at of their kg_relation_sources rows.

A chunk is one memory row or one chat message (user = layer 1, assistant
= layer 2, memory = layer 0). A message shorter than kg.extract_min_words
words is skipped only when no entity is detected in it, without an LLM:
a capitalised word inside the sentence, a word with digits, or a phrase
that already is an entity in the graph ("Anakku lulus Paket C" is kept,
"ya, yang kedua saja" is skipped). A message is extracted together with
the message before it, which only serves to resolve words like "ya" or
"dia"; facts are taken from the message itself.

A memory of the era changed by memory_update (content or category) loses
its LLM relations and its log row in the same transaction
(reset_extraction) and is extracted again; a worker round that read the
old content writes nothing (status stale). Rule links are kept.

Runs in the background (run_worker), polling the tables, so memory_store
and replies are never slowed down and nothing is lost on a restart. Every
processed chunk gets one kg_extract_log row (done / skipped / error);
errors are retried up to kg.extract_max_attempts times.

Chat rows follow the memory_store access rule (Riyo, 18 Nov 2026): a
message is extracted only when metadata.access (the sender's level when
it was written, set by save_message) may write memory; any other row,
or one without the field, is logged as skipped. The speaker is named from
metadata.speaker (users table / member registry), never from the display
names inside the message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

from .. import maintenance as _maint
from ..anchor_text import split_context_prefix
from ..db.connection import get_connection
from ..db.models import get_config, set_config
from . import kg_refs as _refs

logger = logging.getLogger("syne.memory.kg_extract")

# Entities of the rule-built reference graph: never resolution targets,
# except the nodes of a link target ('ayah' for the Qur'an): a name that is
# exactly one item ("QS 54:46", "Al-Baqarah ayat 255") joins the existing
# node (resolve_ref). Node types of other targets are added at runtime
# (reference_types()).
REFERENCE_TYPES = ["chunk", "ayah", "quran_ayah_ref", "quran_chunk",
                   "quran_surah", "kaidah_entry"]

DEFAULT_ENTITY_TYPES = ["person", "place", "org", "project", "item",
                        "concept", "event", "date", "role"]

# Fixed relation list (decision 1, #57757). Names match the predicates
# already in kg_relations; the meaning line goes into the prompt.
DEFAULT_PREDICATES = {
    "part_of": "A is part of B",
    "member_of": "A is a member of group/organisation B",
    "has_role": "A has role/position B",
    "located_in": "A is located in / lives in B",
    "works_at": "A works at B",
    "owns": "A owns B",
    "uses": "A uses B",
    "studies": "A studies / attends / takes B",
    "created": "A made / built / wrote B",
    "child_of": "A is a child of B",
    "parent_of": "A is a parent of B",
    "sibling_of": "A is a sibling of B",
    "married_to": "A is married to B",
    "friend_of": "A is a friend / colleague of B",
    "prefers": "A likes / prefers B",
    "decided": "A decided / agreed on B",
    "explains": "A explains / is about B",
    "mentions": "A mentions B",
    "cites": "A quotes / cites B",
    "narrated_from": "A (narrator) narrates from B",
    "happened_on": "A happened on date B",
}

# key -> (default, description). Seeded by migration 53 and schema.sql
# (kg.extract_era_start is seeded there from now(), not from this table).
DEFAULTS: dict = {
    "kg.extract_enabled": (
        True,
        "KG ingestion (KG-RAG stage 2): extract entities and relations with an LLM from new memories and chat messages (created at or after kg.extract_era_start), in the background. false = stop.",
    ),
    "kg.extract_model": (
        "",
        "Model that extracts entities and relations: a key from /models, empty = the default model (provider.active_model).",
    ),
    "kg.extract_min_words": (
        10,
        "Chat messages shorter than this many words are skipped by KG extraction, but only when no entity is detected in them (capitalised word, word with digits, or a phrase already in the graph). Clamped 0-100.",
    ),
    "kg.extract_max_chars": (6000, "Characters of one chunk sent to KG extraction; the rest is cut. Clamped 500-50000."),
    "kg.extract_batch": (5, "Chunks extracted per round of the KG worker. Clamped 1-50."),
    "kg.extract_interval": (30, "Seconds the KG worker waits between rounds when nothing is pending. Clamped 5-3600."),
    "kg.extract_timeout": (120, "Seconds one KG extraction call may take. Clamped 10-600."),
    "kg.extract_max_attempts": (3, "Attempts per chunk before a failed KG extraction is given up (kg_extract_log status error). Clamped 1-10."),
    "kg.extract_daily_limit": (300, "Most chunks sent to the KG extraction model per day (system.timezone), retries included. At the limit the worker waits for the next day. 0 = no limit. Clamped 0-100000."),
    "kg.extract_skip_categories": ([], "Memory categories KG extraction skips, e.g. [\"alquran\"]."),
    "kg.extract_skip_scheduled": (True, "Skip messages of reminder / scheduled-task sessions (scheduled:...) in KG extraction."),
    "kg.extract_max_entities": (15, "Most entities kept from one chunk. Clamped 1-50."),
    "kg.extract_max_relations": (20, "Most relations kept from one chunk. Clamped 1-50."),
    "kg.extract_entity_types": (DEFAULT_ENTITY_TYPES, "Entity types KG extraction may use; anything else becomes concept."),
    "kg.extract_predicates": (DEFAULT_PREDICATES, "Fixed relation list for KG extraction: {predicate: meaning}. Relations with another predicate are dropped."),
    "kg.resolve_min_similarity": (
        0.92,
        "Entity resolution: a new name joins an existing entity of the same type by name embedding only at this cosine similarity or higher (after normalisation found nothing). Clamped 0.5-1.",
    ),
    "kg.resolve_min_trgm": (
        0.4,
        "Entity resolution: the embedding match must also share this much spelling (pg_trgm similarity of the normalised names), so different people with similar-sounding names are not joined. Clamped 0-1.",
    ),
}

_CLAMPS = {
    "kg.extract_min_words": (0, 100),
    "kg.extract_max_chars": (500, 50000),
    "kg.extract_batch": (1, 50),
    "kg.extract_interval": (5, 3600),
    "kg.extract_timeout": (10, 600),
    "kg.extract_max_attempts": (1, 10),
    "kg.extract_daily_limit": (0, 100000),
    "kg.extract_max_entities": (1, 50),
    "kg.extract_max_relations": (1, 50),
    "kg.resolve_min_similarity": (0.5, 1.0),
    "kg.resolve_min_trgm": (0.0, 1.0),
}
_FLOATS = {"kg.resolve_min_similarity", "kg.resolve_min_trgm"}


def _clamp(key: str, v):
    default = DEFAULTS[key][0]
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = float(default)
    if math.isnan(x):
        x = float(default)
    lo, hi = _CLAMPS[key]
    x = max(lo, min(hi, x))
    return x if key in _FLOATS else int(x)


async def load_settings() -> dict:
    out = {}
    for key, (default, _) in DEFAULTS.items():
        try:
            v = await get_config(key, default)
        except Exception:
            v = default
        if key in _CLAMPS:
            v = _clamp(key, v)
        elif isinstance(default, bool):
            v = v if isinstance(v, bool) else str(v).strip().lower() not in ("false", "0", "no", "off", "")
        elif isinstance(default, list):
            v = [str(x).strip().lower() for x in (v if isinstance(v, list) else default) if str(x).strip()]
        elif isinstance(default, dict):
            v = {str(k).strip().lower(): str(d) for k, d in (v if isinstance(v, dict) else default).items() if str(k).strip()}
            v = v or dict(default)
        else:
            v = "" if v is None else str(v)
        out[key] = v
    try:
        out["kg.extract_era_start"] = parse_era(await get_config("kg.extract_era_start", None))
    except Exception:
        out["kg.extract_era_start"] = None
    return out


def parse_era(raw) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).strip())
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── Name normalisation (from workspace/workspace/kg_entity_merge.py) ──────
# Decision 13 (#57757): Arabic-name normalisation for now; the Indonesian
# (KPU) approach is postponed.

_TITLE_SUFFIX = re.compile(
    r"\s+(saw\.?|r\.a\.?|ra\.?|rahimahullah|radhiallahu.+|alaihissalam|as\.?)$")
_ARTICLE = re.compile(r"^(al[- ]|an[- ]|asy[- ]|ath[- ]|at[- ]|ar[- ]|adh[- ]|ad[- ]|ash[- ])")


def normalize_name(name: str) -> str:
    n = (name or "").strip().lower()
    n = _TITLE_SUFFIX.sub("", n)
    n = re.sub(r"\s*\u2642\s*", " ", n)
    n = _ARTICLE.sub("", n)
    n = re.sub(r"^(al[- ])?imam ", "", n)
    n = re.sub(r"\b(ibnu?|bin)\b", "ibn", n)
    n = re.sub(r"['\u2018\u2019`\u02BB\u02BF]", "'", n)
    for c in ("yy", "ff", "ll", "mm", "nn", "ss", "tt", "dd"):
        n = n.replace(c, c[0])
    n = n.replace("dz", "z").replace("zh", "z").replace("sy", "sh")
    n = re.sub(r"[\"“”()\[\]{}.,;:!?]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


# ── Chunk selection (decision 9, #57757) ──────────────────────────────────

_WORD = re.compile(r"[^\W_][\w'’\-]*", re.U)


def words(text: str) -> list:
    return _WORD.findall(text or "")


def cheap_entities(text: str) -> list:
    """Entity candidates found without an LLM: a capitalised word that does
    not start a sentence, an all-caps word, or a word with digits."""
    out = []
    start = True
    for m in re.finditer(r"[^\W_][\w'’\-]*|[.!?\n]", text or "", re.U):
        tok = m.group(0)
        if tok in ".!?\n":
            start = True
            continue
        if any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
            out.append(tok)
        elif len(tok) >= 2 and tok.isupper():
            out.append(tok)
        elif not start and tok[0].isupper():
            out.append(tok)
        start = False
    return out


def ngrams(text: str, n_max: int = 3) -> list:
    ws = [w.lower() for w in words(text)]
    out = []
    for n in range(1, n_max + 1):
        for i in range(len(ws) - n + 1):
            g = normalize_name(" ".join(ws[i:i + n]))
            if len(g) >= 3:
                out.append(g)
    return list(dict.fromkeys(out))


async def select_chunk(conn, text: str, min_words: int) -> tuple:
    """(extract?, reason)."""
    if not (text or "").strip():
        return False, "empty"
    if len(words(text)) >= min_words:
        return True, "long"
    if cheap_entities(text):
        return True, "entity"
    grams = ngrams(text)
    if grams:
        hit = await conn.fetchval(
            "SELECT 1 FROM kg_entities WHERE norm_name = ANY($1::text[]) "
            "AND entity_type <> ALL($2::text[]) LIMIT 1",
            grams, REFERENCE_TYPES,
        )
        if hit:
            return True, "known_entity"
    return False, "short_no_entity"


# ── Extraction ────────────────────────────────────────────────────────────

PROMPT = """Extract a knowledge graph from TARGET.
Return JSON only, no prose:
{{"entities":[{{"name":"...","type":"..."}}],"relations":[{{"subject":"...","predicate":"...","object":"..."}}]}}

Rules:
- Only facts stated in TARGET. CONTEXT is the message before it and only helps to understand words like "ya", "itu", "dia"; never extract a fact that appears only in CONTEXT.
- "aku", "saya", "I", "my", "-ku" in TARGET refer to the SPEAKER.
- Entity types: {types}.
- Predicates, use exactly one of these; when none fits, leave the relation out:
{predicates}
- Names as written in the text, in their original language, without titles of respect added. At most {max_e} entities and {max_r} relations.
- Every entity name must point to ONE specific thing. A person known only by a relation word (anakku, istrinya, "my son") is named "<relation> <name of that person>", e.g. "anak Riyogarta Pratikto", never just "Anak".
- A label in square brackets at the start of TARGET (e.g. "[KEPUTUSAN RIYO - 18 Nov 2026]") is a title, not an entity; its date may be used with happened_on.
- Examples, illustrations, quotations and hypothetical scenarios ("misalnya", "contoh", "seandainya", "for example", "suppose") are not facts: extract nothing from them.
- Subject and object of every relation must be in entities. Leave out greetings, opinions about the conversation itself, and anything uncertain.
- Never make an entity of a commit hash, an ID or record number (memory #123, message 456), a file path, a URL, or a chat command (/models).
- Nothing worth storing: {{"entities":[],"relations":[]}}

SPEAKER: {speaker}
CONTEXT:
{context}

TARGET:
{target}
"""


def build_prompt(target: str, *, speaker: str, context: str, s: dict) -> str:
    preds = "\n".join(f"  {k}: {v}" for k, v in s["kg.extract_predicates"].items())
    return PROMPT.format(
        types=", ".join(s["kg.extract_entity_types"]),
        predicates=preds,
        max_e=s["kg.extract_max_entities"],
        max_r=s["kg.extract_max_relations"],
        speaker=speaker or "unknown",
        context=(context or "(none)").strip(),
        target=target.strip(),
    )


def _json_object(raw: str):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in extraction reply")
    return json.loads(raw[i:j + 1])


# Names that are technical references, not things (Riyo, 18 Nov 2026):
# commit hashes, IDs / record numbers, file paths, URLs, chat commands.
_NOT_ENTITY = [
    re.compile(r"^(?=[0-9a-f]*\d)[0-9a-f]{7,40}$", re.I),
    re.compile(r"^#\d+$"),
    re.compile(r"^(memori|memory|pesan|message|msg|commit|id|row|baris)\s*#?\s*\d+$", re.I),
    re.compile(r"^(https?://|www\.)", re.I),
    re.compile(r"^/[a-z_][a-z0-9_]*$", re.I),
    re.compile(r"^(~|\.{1,2})?/?([\w.-]+/)+[\w.-]*$"),
    re.compile(r"^[\w-]+\.(py|sql|md|json|ya?ml|toml|txt|log|sh|js|ts|html|css)$", re.I),
]


def technical_name(name: str) -> bool:
    n = (name or "").strip()
    return any(p.search(n) for p in _NOT_ENTITY)


def parse_extraction(raw: str, s: dict) -> dict:
    """Validated {entities: [{name, type}], relations: [{subject, predicate, object}]}."""
    data = _json_object(raw)
    types = set(s["kg.extract_entity_types"])
    preds = set(s["kg.extract_predicates"])
    ents: dict = {}
    for e in data.get("entities") or []:
        if not isinstance(e, dict):
            continue
        name = re.sub(r"\s+", " ", str(e.get("name") or "")).strip(" \"'")
        if not name or len(name) > 120 or not normalize_name(name) or technical_name(name):
            continue
        t = str(e.get("type") or "").strip().lower()
        key = name.lower()
        if key not in ents:
            ents[key] = {"name": name, "type": t if t in types else "concept"}
        if len(ents) >= s["kg.extract_max_entities"]:
            break
    rels = []
    seen = set()
    for r in data.get("relations") or []:
        if not isinstance(r, dict):
            continue
        sub = str(r.get("subject") or "").strip().lower()
        obj = str(r.get("object") or "").strip().lower()
        p = str(r.get("predicate") or "").strip().lower()
        if p not in preds or sub not in ents or obj not in ents or sub == obj:
            continue
        k = (sub, p, obj)
        if k in seen:
            continue
        seen.add(k)
        rels.append({"subject": ents[sub]["name"], "predicate": p, "object": ents[obj]["name"]})
        if len(rels) >= s["kg.extract_max_relations"]:
            break
    used = {x for r in rels for x in (r["subject"].lower(), r["object"].lower())}
    return {"entities": [e for k, e in ents.items() if k in used], "relations": rels}


# ── Entity resolution + Graph DB write ────────────────────────────────────

Embedder = Callable[[str], Awaitable[Optional[list]]]


_AYAH_INDEX: dict = {"at": 0.0, "index": None, "pattern": None, "max_range": 50,
                     "types": []}
_AYAH_INDEX_TTL = 3600.0


async def _target_rules() -> tuple:
    """(target rules {category: cfg}, max range) from kg.ref_sources,
    the legacy kg.qs_pattern (until migration 55) and kg.max_mention_range."""
    from .kg_build import DEFAULT_REF_SOURCES
    sources, legacy, max_range = DEFAULT_REF_SOURCES, None, 50
    try:
        src = await get_config("kg.ref_sources", None)
        if isinstance(src, dict) and src:
            sources = src
        legacy = await get_config("kg.qs_pattern", None)
        mr = await get_config("kg.max_mention_range", None)
        if mr:
            max_range = max(1, int(mr))
    except Exception:
        pass
    targets = _refs.target_configs(sources, legacy if isinstance(legacy, str) else None)
    if not targets:
        targets = _refs.target_configs(DEFAULT_REF_SOURCES)
    return targets, max_range


async def target_indexes(conn) -> list:
    """One index per link target (v1.23.02), built from the target's node
    entities (unit names from their descriptions, "Al-Qamar 46") and its
    rules in kg.ref_sources. Cached for an hour."""
    now = time.monotonic()
    if _AYAH_INDEX["index"] is not None and now - _AYAH_INDEX["at"] < _AYAH_INDEX_TTL:
        return _AYAH_INDEX["index"]
    targets, max_range = await _target_rules()
    out: list = []
    for cfg in targets.values():
        rows = await conn.fetch(
            "SELECT name, description FROM kg_entities WHERE entity_type = $1",
            cfg["node_type"])
        node_re = re.compile(cfg["node_regex"])
        names: dict = {}
        counts: dict = {}
        for r in rows:
            m = node_re.fullmatch(r["name"] or "")
            if not m:
                continue
            unit, item = int(m.group(1)), int(m.group(2))
            counts[unit] = max(counts.get(unit, 0), item)
            desc = (r["description"] or "").strip()
            head, _, tail = desc.rpartition(" ")
            nm = head if tail.isdigit() else desc
            if nm:
                names.setdefault(unit, nm)
        out.append(_refs.target_index_from(names, counts, cfg))
    _AYAH_INDEX.update(at=now, index=out, pattern=None, max_range=max_range,
                       types=[i.node_type for i in out])
    return out


def reference_types() -> list:
    """REFERENCE_TYPES plus the node types of the link targets loaded so far."""
    return REFERENCE_TYPES + [t for t in _AYAH_INDEX.get("types") or []
                              if t not in REFERENCE_TYPES]


async def ayah_index(conn) -> tuple:
    """(index, mention pattern) of the Qur'an target, or of the first target
    when there is no 'ayah' target. Kept for callers from before v1.23.02."""
    idxs = await target_indexes(conn)
    if not idxs:
        return _refs.SurahIndex(), _refs.DEFAULT_QS_PATTERN
    idx = next((i for i in idxs if i.node_type == "ayah"), idxs[0])
    return idx, idx.mention


async def resolve_ref(conn, name: str) -> Optional[int]:
    """ID of the target node when the whole name is exactly one item of a
    link target ("QS 54:46", "Al-Baqarah ayat 255")."""
    for index in await target_indexes(conn):
        ref = _refs.parse_single_ayah(name, index)
        if not ref:
            continue
        return await conn.fetchval(
            "SELECT id FROM kg_entities WHERE name = $1 AND entity_type = $2 LIMIT 1",
            index.node_name(*ref), index.node_type,
        )
    return None


async def resolve_ref_span(conn, name: str) -> list:
    """Target node IDs when the whole name is one item or one range of a link
    target. A range becomes one ID per item (decision 14 Nov 2026: QS 18:71-73
    = QS 18:71, 18:72, 18:73); a range longer than kg.max_mention_range gives []."""
    for index in await target_indexes(conn):
        ref = _refs.parse_ayah_ref(name, index)
        if ref is None:
            continue
        if ref.end - ref.start + 1 > _AYAH_INDEX["max_range"]:
            return []
        names = [index.node_name(ref.surah, a) for a in ref.ayahs()]
        rows = await conn.fetch(
            "SELECT id, name FROM kg_entities WHERE name = ANY($1::text[]) AND entity_type = $2",
            names, index.node_type,
        )
        by = {r["name"]: int(r["id"]) for r in rows}
        return [by[n] for n in names if n in by]
    return []


# Names from before v1.23.02.
resolve_ayah = resolve_ref
resolve_ayah_span = resolve_ref_span


async def resolve_entity(conn, name: str, etype: str, embed: Optional[Embedder],
                         s: dict, scope: Optional[str] = None) -> tuple:
    """(entity id, how): how = ayah | norm | embedding | new.

    A name that is exactly one item of a link target (a Qur'an verse) joins
    that target's node of the reference graph (not for persons: "Yusuf 12"
    may be a name). how stays 'ayah' for any target.

    v1.23.15: a name joins only an entity of the same type and the same
    scope (person_scope): "Al Azhar" the org and "Azhar" the person stay
    apart, and a person named in one chat is not merged with a namesake
    from another chat.
    """
    if etype != "person" and re.search(r"\d", name or ""):
        try:
            aid = await resolve_ayah(conn, name)
        except Exception as e:
            logger.debug(f"KG ayah link skipped ({name!r}): {e}")
            aid = None
        if aid:
            return aid, "ayah"
    norm = normalize_name(name)
    row = await conn.fetchrow(
        "SELECT id FROM kg_entities WHERE norm_name = $1 AND entity_type <> ALL($2::text[]) "
        "AND entity_type = $3 AND scope IS NOT DISTINCT FROM $4 ORDER BY id LIMIT 1",
        norm, reference_types(), etype, scope,
    )
    if row:
        return row["id"], "norm"
    vec = None
    if embed is not None:
        try:
            vec = await embed(name)
        except Exception as e:
            logger.debug(f"KG entity name not embedded ({name!r}): {e}")
            vec = None
    if vec:
        row = await conn.fetchrow(
            "SELECT id, 1 - (embedding <=> $1::vector) AS sim, "
            "coalesce(similarity(norm_name, $2), 0) AS trgm "
            "FROM kg_entities WHERE entity_type = $3 AND embedding IS NOT NULL "
            "AND vector_dims(embedding) = $4 AND scope IS NOT DISTINCT FROM $5 "
            "ORDER BY embedding <=> $1::vector LIMIT 1",
            str(vec), norm, etype, len(vec), scope,
        )
        if (row and row["sim"] is not None
                and row["sim"] >= s["kg.resolve_min_similarity"]
                and row["trgm"] >= s["kg.resolve_min_trgm"]):
            await conn.execute(
                "UPDATE kg_entities SET aliases = array_append(aliases, $2), updated_at = now() "
                "WHERE id = $1 AND NOT (lower($2) = ANY (SELECT lower(a) FROM unnest(aliases) a)) "
                "AND lower(name) <> lower($2)",
                row["id"], name,
            )
            return row["id"], "embedding"
    eid = await conn.fetchval(
        "INSERT INTO kg_entities (name, entity_type, norm_name, embedding, scope) "
        "VALUES ($1, $2, $3, $4::vector, $5) "
        "ON CONFLICT (LOWER(name), entity_type, coalesce(scope, '')) DO UPDATE SET updated_at = now(), "
        "norm_name = coalesce(kg_entities.norm_name, EXCLUDED.norm_name), "
        "embedding = coalesce(kg_entities.embedding, EXCLUDED.embedding) "
        "RETURNING id",
        name, etype, norm, str(vec) if vec else None, scope,
    )
    return eid, "new"


def person_scope(name: str, etype: str, chat_scope: Optional[str], known) -> Optional[str]:
    """Scope of an entity extracted from a chunk (v1.23.15).

    A person named in a group message belongs to that group ('chat:<platform>:
    <chat id>'), so two people called Budi in two groups stay two entities.
    DM messages carry no chat_scope (v1.23.16), so their people are shared.
    Shared (None): every other type, memory chunks, and people who are
    registered owner/family users or listed in family.directory (their name,
    or its first or last word, e.g. "Pak Riyo", "Agha Gamitra").
    """
    if etype != "person" or not chat_scope:
        return None
    norm = normalize_name(name)
    toks = norm.split()
    if norm in known or (toks and (toks[0] in known or toks[-1] in known)):
        return None
    return chat_scope


async def known_people(conn) -> frozenset:
    """Normalised names of owner/family users, their aliases, the
    family.directory names and the assistant's own name."""
    names: set = set()
    for r in await conn.fetch(
            "SELECT name, display_name, aliases FROM users "
            "WHERE access_level IN ('owner', 'family')"):
        names.update([r["name"], r["display_name"]])
        al = r["aliases"]
        if isinstance(al, str):
            try:
                al = json.loads(al)
            except ValueError:
                al = {}
        if isinstance(al, dict):
            names.add(al.get("default"))
            if isinstance(al.get("groups"), dict):
                names.update(al["groups"].values())
    # v1.23.16: one malformed entry no longer drops the whole directory,
    # and "names" given as a string is one name, not a list of letters.
    try:
        d = await get_config("family.directory", [])
        if isinstance(d, str):
            d = json.loads(d)
    except Exception as e:
        logger.warning(f"KG extract: family.directory unreadable: {e}")
        d = []
    for p in d if isinstance(d, list) else []:
        if not isinstance(p, dict):
            continue
        n = p.get("names") or []
        names.update([n] if isinstance(n, str) else (n if isinstance(n, list) else []))
    try:
        names.add(await conn.fetchval(
            "SELECT value FROM identity WHERE key = 'name' AND active = true"))
    except Exception:
        pass
    out: set = set()
    for x in names:
        if not isinstance(x, str):
            continue
        norm = normalize_name(x)
        if not norm:
            continue
        out.add(norm)
        # Each word of a multi-word name too: "Riyogarta" alone is the owner.
        out.update(t for t in norm.split() if len(t) >= 3)
    return frozenset(out)


def _rel_pairs(subs: list, objs: list) -> list:
    """Entity pairs for one extracted relation. One side a single entity:
    every item of the other side. Range to range: item by item when both
    have the same length, else nothing - a cross product would turn one
    stated fact into up to range^2 relations."""
    if len(subs) <= 1 or len(objs) <= 1:
        return [(a, b) for a in subs for b in objs]
    if len(subs) == len(objs):
        return list(zip(subs, objs))
    logger.debug(f"KG range relation skipped: {len(subs)} x {len(objs)} items")
    return []


async def write_graph(conn, extracted: dict, source_type: str, source_id: int,
                      embed: Optional[Embedder], s: dict,
                      chat_scope: Optional[str] = None) -> tuple:
    """Resolve entities, upsert relations with their source. (entities, relations).

    A link-target reference (Qur'an verse) resolves to its node entities; a
    range gives one relation per item.
    """
    ids: dict = {}
    for e in extracted["entities"]:
        span: list = []
        if e["type"] != "person" and re.search(r"\d", e["name"] or ""):
            try:
                span = await resolve_ayah_span(conn, e["name"])
            except Exception as ex:
                logger.debug(f"KG ayah link skipped ({e['name']!r}): {ex}")
                span = []
        if span:
            ids[e["name"].lower()] = span
            continue
        scope = person_scope(e["name"], e["type"], chat_scope, s.get("_known_people") or frozenset())
        eid, _how = await resolve_entity(conn, e["name"], e["type"], embed, s, scope)
        ids[e["name"].lower()] = [eid]
    n_rel = 0
    for r in extracted["relations"]:
        for a, b in _rel_pairs(ids.get(r["subject"].lower()) or [],
                               ids.get(r["object"].lower()) or []):
            if not a or not b or a == b:
                continue
            rel_id = await conn.fetchval(
                "INSERT INTO kg_relations (subject_id, predicate, object_id, source_memory_id) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT (subject_id, predicate, object_id) "
                "DO UPDATE SET updated_at = now() RETURNING id",
                a, r["predicate"], b, source_id if source_type == "memory" else None,
            )
            await conn.execute(
                "INSERT INTO kg_relation_sources (relation_id, source_type, source_id, origin) "
                "VALUES ($1, $2, $3, 'llm') ON CONFLICT DO NOTHING",
                rel_id, source_type, source_id,
            )
            n_rel += 1
    return len({i for v in ids.values() for i in v}), n_rel


# ── Worker ────────────────────────────────────────────────────────────────

def writer_levels() -> frozenset:
    """Access levels that may write memory: the memory_store permission in
    security.py. Chat rows written at another level never reach the graph,
    so a chat cannot write the KG where it could not store a memory."""
    from ..security import TOOL_PERMISSIONS, get_permission_digit, has_permission
    perm = TOOL_PERMISSIONS.get("memory_store", 0)
    return frozenset(lvl for lvl in ("owner", "family", "public")
                     if has_permission(get_permission_digit(perm, lvl), "w"))


def speaker_of(content: str, role: str, name: str = "") -> tuple:
    """(speaker label, text without the channel header).

    name must come from a trusted source (metadata.speaker). The sender and
    conversation blocks inside the message carry display names anyone can
    set, so they are stripped and never read.
    """
    if role == "assistant":
        return "Molt (the assistant)", (content or "").strip()
    _prefix, body = split_context_prefix(content or "")
    name = (name or "").strip()
    return (f"{name} (the user)" if name else "the user"), body or (content or "").strip()


_LOG_UPSERT = """
    INSERT INTO kg_extract_log (source_type, source_id, layer, session_id, status,
                                reason, model, n_entities, n_relations, latency_ms)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    ON CONFLICT (source_type, source_id) DO UPDATE SET
        status = EXCLUDED.status, reason = EXCLUDED.reason, model = EXCLUDED.model,
        n_entities = EXCLUDED.n_entities, n_relations = EXCLUDED.n_relations,
        latency_ms = EXCLUDED.latency_ms, attempts = kg_extract_log.attempts + 1,
        updated_at = now()
"""

# A row is pending unless it has a log row that is final (done / skipped /
# error with no attempts left) or an error still waiting: 10 min after the
# first attempt, doubling per attempt up to 640 min (v1.23.13).
_PENDING = """
    NOT EXISTS (SELECT 1 FROM kg_extract_log l
                WHERE l.source_type = '{t}' AND l.source_id = x.id
                  AND NOT (l.status = 'error' AND l.attempts < $3
                           AND l.updated_at < now() - interval '10 minutes'
                               * power(2, least(l.attempts - 1, 6))))
"""


async def era_floor(conn, table: str, era: datetime) -> int:
    """Largest id created before the era (rows above it are candidates)."""
    return int(await conn.fetchval(
        f"SELECT coalesce(max(id), 0) FROM {table} WHERE created_at < $1", era) or 0)


async def pending(conn, floors: dict, era: datetime, s: dict) -> list:
    from ..tools.history import group_session_sql
    n = s["kg.extract_batch"]
    out = []
    rows = await conn.fetch(
        "SELECT x.id, x.content, x.category, x.updated_at FROM memory x "
        "WHERE x.id > $1 AND x.created_at >= $2 AND " + _PENDING.format(t="memory") +
        " AND NOT (lower(coalesce(x.category, '')) = ANY($5::text[])) "
        "ORDER BY x.id LIMIT $4",
        floors["memory"], era, s["kg.extract_max_attempts"], n,
        s["kg.extract_skip_categories"],
    )
    out += [{"type": "memory", **dict(r)} for r in rows]
    if len(out) >= n:
        return out
    skip_sched = "AND coalesce(se.platform_chat_id, '') NOT LIKE 'scheduled:%' " if s["kg.extract_skip_scheduled"] else ""
    rows = await conn.fetch(
        "SELECT x.id, x.content, x.role, x.session_id, u.name AS owner_name, "
        # v1.23.16: people are scoped to the chat only in groups. A DM is
        # written by one owner/family user, so a person named there is the
        # same person when that user asks from another chat or platform.
        "CASE WHEN " + group_session_sql("se") + " THEN "
        "'chat:' || se.platform || ':' || coalesce(se.platform_chat_id, '') END AS chat_scope, "
        "x.metadata->>'access' AS access, x.metadata->>'speaker' AS speaker FROM messages x "
        "LEFT JOIN sessions se ON se.id = x.session_id "
        "LEFT JOIN users u ON u.id = se.user_id "
        "WHERE x.id > $1 AND x.created_at >= $2 AND x.role IN ('user', 'assistant') "
        "AND btrim(coalesce(x.content, '')) <> '' " + skip_sched + "AND " +
        _PENDING.format(t="message") + " ORDER BY x.id LIMIT $4",
        floors["messages"], era, s["kg.extract_max_attempts"], n - len(out),
    )
    out += [{"type": "message", **dict(r)} for r in rows]
    return out


async def previous_message(conn, session_id: int, message_id: int) -> str:
    """The message before, as context for words like 'ya' or 'dia'. Left out
    when it was not written with write access: text from a sender who may
    not write memory must not steer what is stored."""
    row = await conn.fetchrow(
        "SELECT role, content, metadata->>'access' AS access, "
        "metadata->>'speaker' AS speaker FROM messages WHERE session_id = $1 AND id < $2 "
        "AND role IN ('user', 'assistant') AND btrim(coalesce(content, '')) <> '' "
        "ORDER BY id DESC LIMIT 1", session_id, message_id)
    if not row or row["access"] not in writer_levels():
        return ""
    who, body = speaker_of(row["content"], row["role"], row["speaker"] or "")
    return f"[{who}] {body[-1500:]}"


async def process_one(item: dict, judge, embed: Optional[Embedder], s: dict) -> dict:
    """Extract one chunk and write its graph. Returns the log fields."""
    t0 = time.monotonic()
    label = getattr(judge, "label", "")
    st, sid = item["type"], item["id"]
    layer = 0 if st == "memory" else (1 if item.get("role") == "user" else 2)
    log = {"layer": layer, "session_id": item.get("session_id"), "model": label,
           "n_entities": 0, "n_relations": 0}
    if st == "memory":
        speaker, text, context = f"stored memory, category {item.get('category')}", (item["content"] or "").strip(), ""
    else:
        if item.get("access") not in writer_levels():
            return {**log, "status": "skipped",
                    "reason": f"access {item.get('access') or 'unknown'}", "model": None,
                    "latency_ms": int((time.monotonic() - t0) * 1000)}
        speaker, text = speaker_of(item["content"], item.get("role"), item.get("speaker") or "")
        context = ""
    async with get_connection() as conn:
        if st == "message":
            ok, reason = await select_chunk(conn, text, s["kg.extract_min_words"])
            if not ok:
                return {**log, "status": "skipped", "reason": reason, "model": None,
                        "latency_ms": int((time.monotonic() - t0) * 1000)}
            context = await previous_message(conn, item["session_id"], sid)
        else:
            reason = "memory"
    prompt = build_prompt(text[: s["kg.extract_max_chars"]], speaker=speaker, context=context, s=s)
    try:
        raw = await judge(prompt, float(s["kg.extract_timeout"]))
    except asyncio.CancelledError:
        raise
    except Exception as e:
        if provider_down(e):
            raise ProviderDown(f"{type(e).__name__}: {e}") from e
        raise
    extracted = parse_extraction(raw, s)
    vecs = await prepare_embeddings(extracted, embed, item.get("chat_scope"),
                                    s.get("_known_people") or frozenset())

    async def _cached(name: str):
        return vecs.get(name)

    async with get_connection() as conn:
        async with conn.transaction():
            # The graph and the log row are written together, and only for the
            # content that was read: a memory changed (memory_update) or
            # deleted meanwhile is left to the next round.
            if st == "memory" and "updated_at" in item:
                cur = await conn.fetchrow(
                    "SELECT updated_at FROM memory WHERE id = $1 FOR UPDATE", sid)
                if cur is None or cur["updated_at"] != item["updated_at"]:
                    return {**log, "status": "stale", "_logged": True,
                            "reason": "memory changed or deleted during extraction",
                            "latency_ms": int((time.monotonic() - t0) * 1000)}
            n_e, n_r = await write_graph(conn, extracted, st, sid,
                                         _cached if embed is not None else None, s,
                                         chat_scope=item.get("chat_scope"))
            res = {**log, "status": "done", "reason": reason, "n_entities": n_e,
                   "n_relations": n_r, "latency_ms": int((time.monotonic() - t0) * 1000)}
            await _write_log(conn, item, res)
    return {**res, "_logged": True}


async def _write_log(conn, item: dict, res: dict) -> None:
    await conn.execute(
        _LOG_UPSERT, item["type"], item["id"], res.get("layer"), res.get("session_id"),
        res["status"], (res.get("reason") or "")[:500], res.get("model"),
        res.get("n_entities", 0), res.get("n_relations", 0), res.get("latency_ms"),
    )


async def write_log(item: dict, res: dict) -> None:
    async with get_connection() as conn:
        await _write_log(conn, item, res)


# ── memory_update (Riyo, 19 Nov 2026) - v1.23.08 ─────────────────────────

async def current_era() -> Optional[datetime]:
    return parse_era(await get_config("kg.extract_era_start", None))


def needs_reextract(old, content: str, category: str, era: Optional[datetime]) -> bool:
    """True when an update changes what the KG worker read from a memory of
    the era (content or category). Older memories are never re-extracted."""
    if old is None or era is None or old["created_at"] is None:
        return False
    created = old["created_at"]
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if created < era:
        return False
    return ((old["content"] or "") != (content or "")
            or (old["category"] or "") != (category or ""))


async def reset_extraction(conn, memory_id: int) -> int:
    """Remove what the KG worker extracted from one memory, so it is
    extracted again. Run inside the caller's transaction.

    Drops its kg_extract_log row and its LLM sources (rule links stay), the
    relations left with no source, and the entities those touched that are
    left with no relation (reference types kept). A relation kept by another
    source points source_memory_id at a living memory source, or NULL.
    Returns the number of sources removed.
    """
    await conn.execute(
        "DELETE FROM kg_extract_log WHERE source_type = 'memory' AND source_id = $1", memory_id)
    rows = await conn.fetch(
        "DELETE FROM kg_relation_sources WHERE source_type = 'memory' AND source_id = $1 "
        "AND origin = 'llm' RETURNING relation_id", memory_id)
    rels = sorted({r["relation_id"] for r in rows})
    if not rels:
        return 0
    ends = await conn.fetch(
        "DELETE FROM kg_relations r WHERE r.id = ANY($1::int[]) "
        "AND NOT EXISTS (SELECT 1 FROM kg_relation_sources s WHERE s.relation_id = r.id) "
        "RETURNING r.subject_id, r.object_id", rels)
    await conn.execute(
        "UPDATE kg_relations r SET source_memory_id = ("
        " SELECT max(s.source_id) FROM kg_relation_sources s JOIN memory m ON m.id = s.source_id"
        " WHERE s.relation_id = r.id AND s.source_type = 'memory') "
        "WHERE r.id = ANY($1::int[]) AND r.source_memory_id = $2 "
        "AND NOT EXISTS (SELECT 1 FROM kg_relation_sources s WHERE s.relation_id = r.id "
        "AND s.source_type = 'memory' AND s.source_id = $2)", rels, memory_id)
    ents = sorted({e for r in ends for e in (r["subject_id"], r["object_id"]) if e})
    if ents:
        await conn.execute(
            "DELETE FROM kg_entities e WHERE e.id = ANY($1::int[]) "
            "AND COALESCE(e.entity_type, '') <> ALL($2::text[]) "
            "AND NOT EXISTS (SELECT 1 FROM kg_relations r "
            "WHERE r.subject_id = e.id OR r.object_id = e.id)", ents, reference_types())
    return len(rows)


async def backfill_norm_names(batch: int = 2000) -> int:
    """Fill kg_entities.norm_name of old entities (reference types excluded)."""
    done = 0
    while True:
        async with get_connection() as conn:
            rows = await conn.fetch(
                "SELECT id, name FROM kg_entities WHERE norm_name IS NULL "
                "AND entity_type <> ALL($1::text[]) ORDER BY id LIMIT $2",
                REFERENCE_TYPES, batch)
            if not rows:
                return done
            await conn.execute(
                "UPDATE kg_entities k SET norm_name = v.n "
                "FROM unnest($1::int[], $2::text[]) AS v(id, n) WHERE k.id = v.id",
                [r["id"] for r in rows], [normalize_name(r["name"]) for r in rows])
        done += len(rows)
        await asyncio.sleep(0.2)


# ── Monthly prune (Riyo, 18 Nov 2026) - v1.23.05 ──────────────────────────
# The delete triggers (migration 56) clean the graph of a deleted memory or
# message at once. What they can miss - a prune that failed (WARNING, the
# delete went through) or a message deleted while it was being extracted -
# is cleaned here: once every PRUNE_EVERY, the last run kept in config
# kg.prune_last_run so a restart does not run it again. A failed run is
# retried an hour later and not recorded.

PRUNE_EVERY = timedelta(days=30)
PRUNE_RETRY_SECONDS = 3600
PRUNE_KEY = "kg.prune_last_run"
_PRUNE_DESC = ("Last monthly KG prune by the KG worker (ISO, UTC): graph sources whose memory or "
               "chat message is gone are removed every 30 days. Set by the worker; delete the key "
               "to run it again on the next round.")
_prune_retry_at = 0.0


async def maybe_prune(now: Optional[datetime] = None) -> Optional[dict]:
    """Prune orphan graph sources when the last run is PRUNE_EVERY old.

    Returns {source type: ids pruned} when it ran, None when not due or
    failed. Never raises (except cancellation). Checks the config at most
    once an hour per process.
    """
    global _prune_retry_at
    if time.monotonic() < _prune_retry_at:
        return None
    now = now or datetime.now(timezone.utc)
    try:
        last = parse_era(await get_config(PRUNE_KEY, None))
        if last is not None and now - last < PRUNE_EVERY:
            _prune_retry_at = time.monotonic() + PRUNE_RETRY_SECONDS
            return None
        from ..db.kg_prune_ddl import prune_orphan_sources
        async with get_connection() as conn:
            res = await prune_orphan_sources(conn)
        await set_config(PRUNE_KEY, now.isoformat(timespec="seconds"), _PRUNE_DESC)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _prune_retry_at = time.monotonic() + PRUNE_RETRY_SECONDS
        logger.warning(f"KG prune failed, retried in an hour: {e}")
        return None
    _prune_retry_at = time.monotonic() + PRUNE_RETRY_SECONDS
    logger.info(f"KG prune: {res.get('memory', 0)} memories and {res.get('message', 0)} "
                f"messages that are gone removed from the graph")
    return res


class ProviderDown(Exception):
    """The extraction model is unreachable (timeout, 429, 5xx, connection).
    The chunk is kept: no log row, no attempt used (v1.23.13)."""


# Rounds the same chunk may hit ProviderDown before it counts as an attempt
# anyway (a chunk that always breaks the provider must not stall the worker).
PROVIDER_DOWN_STRIKES = 6
PROVIDER_DOWN_MAX_WAIT = 1800
# v1.23.16: a "model unavailable" failure costs the chunk an attempt only
# when the model answered another chunk after this one first failed (the
# chunk itself is the problem, e.g. too slow for a local model), or after
# PROVIDER_DOWN_STRIKES failures spread over at least this long (a daily
# quota resets within it). A plain outage never costs attempts.
PROVIDER_DOWN_GIVE_UP = 24 * 3600


def provider_down(err: BaseException) -> bool:
    from ..embed_health import is_soft_failure
    if is_soft_failure(err) or isinstance(err, (ConnectionError, OSError)):
        return True
    try:
        import httpx
        if isinstance(err, httpx.TransportError):
            return True
    except ImportError:
        pass
    status = getattr(err, "status_code", None) or getattr(
        getattr(err, "response", None), "status_code", None)
    if isinstance(status, int):
        return status >= 500
    text = str(err).lower()
    if (re.search(r"\b(500|502|503|504|529)\b", text)
            or "overloaded" in text or "unavailable" in text):
        return True
    # Syne's drivers (anthropic / openai / codex) end their own retries with
    # a RuntimeError that has no status code, e.g. "Anthropic stream failed
    # after 3 attempts: ConnectError" (v1.23.16).
    return bool(_DRIVER_DOWN.search(text))


_DRIVER_DOWN = re.compile(
    r"stream failed after|transient error after|api failed after"
    r"|connecterror|connecttimeout|readtimeout|writetimeout|pooltimeout|readerror"
    r"|remoteprotocolerror|timed out|connection (?:refused|reset|closed|error)")


async def _nap(seconds: float) -> None:
    """Sleep up to ``seconds``; wake early when /restore starts or
    kg.extract_enabled turns false (v1.23.16: a 30-minute outage wait used
    to ignore both)."""
    end = time.monotonic() + seconds
    while True:
        left = end - time.monotonic()
        if left <= 0:
            return
        await asyncio.sleep(min(30.0, left))
        if _maint.active():
            return
        try:
            v = await get_config("kg.extract_enabled", True)
        except Exception:
            continue
        if v is False or str(v).strip().lower() in ("false", "0", "no", "off", ""):
            return


async def prepare_embeddings(extracted: dict, embed: Optional[Embedder],
                             chat_scope: Optional[str] = None,
                             known: frozenset = frozenset()) -> dict:
    """{entity name: vector} for names write_graph may need to embed.

    Done before the write transaction opens (v1.23.13): with embedding
    retries the transaction stayed open for minutes holding entity and
    relation locks. Names that resolve without a vector (a reference item,
    a known norm_name of the same type AND scope) are not embedded. v1.23.16:
    the scope was missing, so a person scoped to a group never got a vector
    when a shared namesake existed.
    """
    if embed is None or not extracted.get("entities"):
        return {}
    need: list = []
    async with get_connection() as conn:
        for e in extracted["entities"]:
            name = e["name"]
            if name in need:
                continue
            if e["type"] != "person" and re.search(r"\d", name or ""):
                try:
                    if await resolve_ayah_span(conn, name) or await resolve_ayah(conn, name):
                        continue
                except Exception:
                    pass
            scope = person_scope(name, e["type"], chat_scope, known)
            hit = await conn.fetchval(
                "SELECT 1 FROM kg_entities WHERE norm_name = $1 "
                "AND entity_type <> ALL($2::text[]) AND entity_type = $3 "
                "AND scope IS NOT DISTINCT FROM $4 LIMIT 1",
                normalize_name(name), reference_types(), e["type"], scope)
            if not hit:
                need.append(name)
    vecs: dict = {}
    for name in need:
        try:
            vec = await embed(name)
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            logger.debug(f"KG entity name not embedded ({name!r}): {ex}")
            continue
        if vec:
            vecs[name] = vec
    return vecs


async def advance_floor(conn, floor: int, era: datetime, s: dict) -> int:
    """New message floor: every candidate row at or below it is final.

    The queue scan starts above the floor; before v1.23.13 it stayed at the
    era start and every round rescanned all rows since. Rows of the last
    five minutes are never passed (a row may commit after a higher id).
    Only messages: a memory row is extracted again after memory_update.
    """
    skip_sched = ("AND coalesce(se.platform_chat_id, '') NOT LIKE 'scheduled:%' "
                  if s["kg.extract_skip_scheduled"] else "")
    first = await conn.fetchval(
        "SELECT x.id FROM messages x LEFT JOIN sessions se ON se.id = x.session_id "
        "WHERE x.id > $1 AND x.created_at >= $2 AND x.role IN ('user', 'assistant') "
        "AND btrim(coalesce(x.content, '')) <> '' " + skip_sched +
        "AND NOT EXISTS (SELECT 1 FROM kg_extract_log l WHERE l.source_type = 'message' "
        "AND l.source_id = x.id AND (l.status <> 'error' OR l.attempts >= $3)) "
        "ORDER BY x.id LIMIT 1",
        floor, era, s["kg.extract_max_attempts"])
    settled = int(await conn.fetchval(
        "SELECT coalesce(max(id), 0) FROM messages "
        "WHERE id > $1 AND created_at < now() - interval '5 minutes'", floor) or 0)
    new = settled if first is None else min(int(first) - 1, settled)
    return max(floor, new)


def judge_cache_key(model_key: str, default_provider) -> tuple:
    """Rebuild the judge when this changes. An empty kg.extract_model
    follows the default provider, which /models replaces (v1.23.13)."""
    if model_key:
        return (model_key,)
    return ("", default_provider, getattr(default_provider, "chat_model", None))


def _error_result(item: dict, judge, reason: str) -> dict:
    return {"status": "error", "reason": reason,
            "layer": 0 if item["type"] == "memory" else (1 if item.get("role") == "user" else 2),
            "session_id": item.get("session_id"), "model": getattr(judge, "label", "")}


async def extracted_today(conn) -> int:
    """Chunks sent to the extraction model today (system.timezone)."""
    tz = str(await get_config("system.timezone", "UTC") or "UTC")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(tz)
    except Exception:
        tz = "UTC"
    return int(await conn.fetchval(
        "SELECT count(*) FROM kg_extract_log WHERE status IN ('done', 'error') "
        "AND coalesce(model, '') <> '' AND updated_at >= "
        "(date_trunc('day', now() AT TIME ZONE $1) AT TIME ZONE $1)", tz) or 0)


# One KG worker per database (v1.23.15): a session advisory lock held on a
# dedicated pool connection. A second process (syne cli, repair --fix, a
# second service) stays idle instead of extracting the same chunks.
WORKER_LOCK_KEY = 7311603215


async def take_worker_lock():
    """The connection holding the worker lock, or None when it is taken."""
    from ..db.connection import get_pool
    pool = get_pool()
    conn = await pool.acquire()
    try:
        if await conn.fetchval("SELECT pg_try_advisory_lock($1::bigint)", WORKER_LOCK_KEY):
            return conn
    except asyncio.CancelledError:
        # v1.23.16: the connection went back to the pool on errors only; a
        # cancellation here leaked it and Pool.close() then hung at shutdown.
        # Releasing resets it (asyncpg unlocks advisory locks on reset).
        try:
            await pool.release(conn)
        except BaseException:
            pass
        raise
    except Exception as e:
        logger.debug(f"KG extract: worker lock not taken: {e}")
    await pool.release(conn)
    return None


async def drop_worker_lock(conn) -> None:
    if conn is None:
        return
    from ..db.connection import get_pool
    try:
        if not conn.is_closed():
            await conn.execute("SELECT pg_advisory_unlock($1::bigint)", WORKER_LOCK_KEY)
    except Exception:
        pass
    try:
        await get_pool().release(conn)
    except Exception:
        pass


async def run_worker(default_provider_getter: Callable[[], object], embed_provider=None) -> None:
    """Background loop. Never raises (except cancellation)."""
    from ..judge_model import build_judge

    async def embed(text: str):
        if embed_provider is None:
            return None
        resp = await embed_provider.embed(text)
        return getattr(resp, "vector", None) or None

    try:
        n = await backfill_norm_names()
        if n:
            logger.info(f"KG extract: normalised names of {n} existing entities")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"KG extract: name normalisation backfill failed: {e}")

    floors: dict = {}
    floors_key = None
    judge, judge_key = None, None
    # (type, id) -> (monotonic time of first "model unavailable", strikes)
    down_wait, suspects, last_ok = 0, {}, 0.0
    lock_conn, lock_idle_logged, limit_logged = None, False, False
    try:
        while True:
            try:
                if _maint.active():
                    await asyncio.sleep(5)
                    continue
                if lock_conn is None or lock_conn.is_closed():
                    await drop_worker_lock(lock_conn)
                    lock_conn = await take_worker_lock()
                    if lock_conn is None:
                        if not lock_idle_logged:
                            logger.info("KG extract: another Syne process runs the KG worker; "
                                        "this one stays idle")
                            lock_idle_logged = True
                        await asyncio.sleep(60)
                        continue
                    lock_idle_logged = False
                await maybe_prune()
                s = await load_settings()
                era = s["kg.extract_era_start"]
                if not s["kg.extract_enabled"] or era is None:
                    await asyncio.sleep(s["kg.extract_interval"])
                    continue
                # Settings that decide what is final restart the floors.
                fkey = (era, s["kg.extract_max_attempts"], s["kg.extract_skip_scheduled"])
                if floors_key != fkey:
                    async with get_connection() as conn:
                        floors = {"memory": await era_floor(conn, "memory", era),
                                  "messages": await era_floor(conn, "messages", era)}
                    floors_key = fkey
                async with get_connection() as conn:
                    floors["messages"] = await advance_floor(conn, floors["messages"], era, s)
                    items = await pending(conn, floors, era, s)
                    if items:
                        s["_known_people"] = await known_people(conn)
                if not items:
                    await asyncio.sleep(s["kg.extract_interval"])
                    continue
                limit = s["kg.extract_daily_limit"]
                if limit:
                    async with get_connection() as conn:
                        used = await extracted_today(conn)
                    if used >= limit:
                        if not limit_logged:
                            logger.warning(f"KG extract: daily limit reached ({used}/{limit} chunks, "
                                           "kg.extract_daily_limit); waiting for the next day")
                            limit_logged = True
                        await asyncio.sleep(300)
                        continue
                    limit_logged = False
                    items = items[: limit - used]
                key = judge_cache_key(s["kg.extract_model"], default_provider_getter())
                if judge is None or judge_key != key:
                    judge = await build_judge(
                        s["kg.extract_model"], default_provider=default_provider_getter(),
                        get_config=get_config, who="KG extract")
                    judge_key = key
                    if judge is None:
                        logger.warning("KG extract: no usable model (kg.extract_model); waiting")
                        await asyncio.sleep(max(60, s["kg.extract_interval"]))
                        continue
                # Chunks that failed as "model unavailable" go last: if another
                # chunk gets an answer first, the model is up.
                items.sort(key=lambda it: (it["type"], it["id"]) in suspects)
                for item in items:
                    if _maint.active():
                        break
                    # Tracked like a chat turn (v1.23.13): maintenance.begin()
                    # cancels this chunk - its transaction rolls back - and the
                    # worker waits at the gate. No await between check and track.
                    task = asyncio.create_task(process_one(item, judge, embed, s))
                    _maint.track(task)
                    try:
                        await asyncio.wait({task})
                    except asyncio.CancelledError:
                        task.cancel()
                        raise
                    finally:
                        _maint.untrack(task)
                    if task.cancelled():
                        break
                    try:
                        res = task.result()
                    except ProviderDown as e:
                        this = (item["type"], item["id"])
                        now = time.monotonic()
                        first, strikes = suspects.get(this, (now, 0))
                        strikes += 1
                        if last_ok > first or (strikes >= PROVIDER_DOWN_STRIKES
                                               and now - first >= PROVIDER_DOWN_GIVE_UP):
                            # The model is up for other chunks (or this has
                            # failed for a day): this chunk costs an attempt
                            # and waits its normal error backoff.
                            suspects.pop(this, None)
                            res = _error_result(item, judge, f"model unavailable: {e}")
                            logger.warning(f"KG extract {item['type']} {item['id']} failed: {res['reason'][:200]}")
                        else:
                            suspects[this] = (first, strikes)
                            down_wait = min(max(60, down_wait * 2), PROVIDER_DOWN_MAX_WAIT)
                            logger.warning(f"KG extract: model unavailable ({str(e)[:200]}); "
                                           f"chunks kept, trying again in {down_wait}s")
                            await _nap(down_wait)
                            break
                    except Exception as e:
                        res = _error_result(item, judge, f"{type(e).__name__}: {e}")
                        logger.warning(f"KG extract {item['type']} {item['id']} failed: {res['reason'][:200]}")
                    else:
                        down_wait = 0
                        suspects.pop((item["type"], item["id"]), None)
                        if res.get("status") in ("done", "stale"):
                            last_ok = time.monotonic()
                    if not res.get("_logged"):
                        await write_log(item, res)
                    if res["status"] == "stale":
                        logger.info(f"KG extract memory {item['id']}: changed during extraction, "
                                    "extracted again next round")
                    if res["status"] == "done":
                        logger.info(
                            f"KG extract {item['type']} {item['id']}: {res['n_entities']} entities, "
                            f"{res['n_relations']} relations ({res['latency_ms']} ms)")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"KG extract round failed: {e}")
                await asyncio.sleep(60)
    finally:
        await drop_worker_lock(lock_conn)
