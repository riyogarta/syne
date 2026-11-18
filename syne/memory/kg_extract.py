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

Runs in the background (run_worker), polling the tables, so memory_store
and replies are never slowed down and nothing is lost on a restart. Every
processed chunk gets one kg_extract_log row (done / skipped / error);
errors are retried up to kg.extract_max_attempts times.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from .. import maintenance as _maint
from ..anchor_text import split_context_prefix
from ..db.connection import get_connection
from ..db.models import get_config
from . import kg_refs as _refs

logger = logging.getLogger("syne.memory.kg_extract")

# Entities of the rule-built reference graph: never resolution targets,
# except 'ayah': a name that is exactly one Qur'an verse ("QS 54:46",
# "Al-Baqarah ayat 255") joins the existing ayah entity (resolve_ayah).
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


_AYAH_INDEX: dict = {"at": 0.0, "index": None, "pattern": None, "max_range": 50}
_AYAH_INDEX_TTL = 3600.0


async def ayah_index(conn) -> tuple:
    """(SurahIndex, QS pattern) built from the 'ayah' entities (names from
    their descriptions, "Al-Qamar 46"), the Qur'an aliases of kg.ref_sources
    and kg.qs_pattern. Cached for an hour."""
    now = time.monotonic()
    if _AYAH_INDEX["index"] is not None and now - _AYAH_INDEX["at"] < _AYAH_INDEX_TTL:
        return _AYAH_INDEX["index"], _AYAH_INDEX["pattern"]
    rows = await conn.fetch(
        "SELECT name, description FROM kg_entities WHERE entity_type = 'ayah'")
    names: dict = {}
    counts: dict = {}
    for r in rows:
        m = re.fullmatch(r"QS (\d{1,3}):(\d{1,3})", r["name"] or "")
        if not m:
            continue
        su, ay = int(m.group(1)), int(m.group(2))
        counts[su] = max(counts.get(su, 0), ay)
        desc = (r["description"] or "").strip()
        head, _, tail = desc.rpartition(" ")
        nm = head if tail.isdigit() else desc
        if nm:
            names.setdefault(su, nm)
    aliases = dict(_refs.DEFAULT_SURAH_ALIASES)
    pattern = _refs.DEFAULT_QS_PATTERN
    max_range = 50
    try:
        src = await get_config("kg.ref_sources", None)
        if isinstance(src, dict):
            for v in src.values():
                if (isinstance(v, dict) and v.get("kind") == "quran"
                        and isinstance(v.get("aliases"), dict)):
                    aliases = dict(v["aliases"])
        pattern = await get_config("kg.qs_pattern", None) or pattern
        mr = await get_config("kg.max_mention_range", None)
        if mr:
            max_range = max(1, int(mr))
    except Exception:
        pass
    idx = _refs.surah_index_from(names, counts, aliases)
    _AYAH_INDEX.update(at=now, index=idx, pattern=pattern, max_range=max_range)
    return idx, pattern


async def resolve_ayah(conn, name: str) -> Optional[int]:
    """ID of the 'ayah' entity when the whole name is one Qur'an verse."""
    index, pattern = await ayah_index(conn)
    ref = _refs.parse_single_ayah(name, index, pattern)
    if not ref:
        return None
    return await conn.fetchval(
        "SELECT id FROM kg_entities WHERE entity_type = 'ayah' AND name = $1 LIMIT 1",
        _refs.ayah_entity_name(*ref),
    )


async def resolve_ayah_span(conn, name: str) -> list:
    """'ayah' entity IDs when the whole name is a Qur'an reference. A range
    becomes one ID per verse (decision 14 Nov 2026: QS 18:71-73 = QS 18:71,
    18:72, 18:73); a range longer than kg.max_mention_range gives []."""
    index, pattern = await ayah_index(conn)
    ref = _refs.parse_ayah_ref(name, index, pattern)
    if ref is None or ref.end - ref.start + 1 > _AYAH_INDEX["max_range"]:
        return []
    names = [_refs.ayah_entity_name(ref.surah, a) for a in ref.ayahs()]
    rows = await conn.fetch(
        "SELECT id, name FROM kg_entities WHERE entity_type = 'ayah' AND name = ANY($1::text[])",
        names,
    )
    by = {r["name"]: int(r["id"]) for r in rows}
    return [by[n] for n in names if n in by]


async def resolve_entity(conn, name: str, etype: str, embed: Optional[Embedder],
                         s: dict) -> tuple:
    """(entity id, how): how = ayah | norm | embedding | new.

    A name that is exactly one Qur'an verse joins the 'ayah' entity of the
    reference graph (not for persons: "Yusuf 12" may be a name).
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
        "ORDER BY (entity_type = $3) DESC, id LIMIT 1",
        norm, REFERENCE_TYPES, etype,
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
            "ORDER BY embedding <=> $1::vector LIMIT 1",
            str(vec), norm, etype,
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
        "INSERT INTO kg_entities (name, entity_type, norm_name, embedding) "
        "VALUES ($1, $2, $3, $4::vector) "
        "ON CONFLICT (LOWER(name), entity_type) DO UPDATE SET updated_at = now(), "
        "norm_name = coalesce(kg_entities.norm_name, EXCLUDED.norm_name), "
        "embedding = coalesce(kg_entities.embedding, EXCLUDED.embedding) "
        "RETURNING id",
        name, etype, norm, str(vec) if vec else None,
    )
    return eid, "new"


async def write_graph(conn, extracted: dict, source_type: str, source_id: int,
                      embed: Optional[Embedder], s: dict) -> tuple:
    """Resolve entities, upsert relations with their source. (entities, relations).

    A Qur'an reference resolves to its 'ayah' entities; a range gives one
    relation per verse.
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
        eid, _how = await resolve_entity(conn, e["name"], e["type"], embed, s)
        ids[e["name"].lower()] = [eid]
    n_rel = 0
    for r in extracted["relations"]:
        for a in ids.get(r["subject"].lower()) or []:
            for b in ids.get(r["object"].lower()) or []:
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

def speaker_of(content: str, role: str, owner_name: str = "") -> tuple:
    """(speaker label, text without the channel header)."""
    if role == "assistant":
        return "Molt (the assistant)", (content or "").strip()
    prefix, body = split_context_prefix(content or "")
    name = ""
    m = re.search(r"Sender \(untrusted metadata\):\s*```json\s*(\{.*?\})\s*```", prefix, re.S)
    if m:
        try:
            name = str(json.loads(m.group(1)).get("name") or "")
        except Exception:
            name = ""
    if not name:
        m = re.search(r'"conversation_label"\s*:\s*"([^"]+)"', prefix)
        name = m.group(1) if m else owner_name
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
# error with no attempts left) or an error retried less than 10 min ago.
_PENDING = """
    NOT EXISTS (SELECT 1 FROM kg_extract_log l
                WHERE l.source_type = '{t}' AND l.source_id = x.id
                  AND NOT (l.status = 'error' AND l.attempts < $3
                           AND l.updated_at < now() - interval '10 minutes'))
"""


async def era_floor(conn, table: str, era: datetime) -> int:
    """Largest id created before the era (rows above it are candidates)."""
    return int(await conn.fetchval(
        f"SELECT coalesce(max(id), 0) FROM {table} WHERE created_at < $1", era) or 0)


async def pending(conn, floors: dict, era: datetime, s: dict) -> list:
    n = s["kg.extract_batch"]
    out = []
    rows = await conn.fetch(
        "SELECT x.id, x.content, x.category FROM memory x "
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
        "SELECT x.id, x.content, x.role, x.session_id, u.name AS owner_name FROM messages x "
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
    row = await conn.fetchrow(
        "SELECT role, content FROM messages WHERE session_id = $1 AND id < $2 "
        "AND role IN ('user', 'assistant') AND btrim(coalesce(content, '')) <> '' "
        "ORDER BY id DESC LIMIT 1", session_id, message_id)
    if not row:
        return ""
    who, body = speaker_of(row["content"], row["role"])
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
        speaker, text = speaker_of(item["content"], item.get("role"), item.get("owner_name") or "")
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
    raw = await judge(prompt, float(s["kg.extract_timeout"]))
    extracted = parse_extraction(raw, s)
    async with get_connection() as conn:
        async with conn.transaction():
            n_e, n_r = await write_graph(conn, extracted, st, sid, embed, s)
    return {**log, "status": "done", "reason": reason, "n_entities": n_e,
            "n_relations": n_r, "latency_ms": int((time.monotonic() - t0) * 1000)}


async def write_log(item: dict, res: dict) -> None:
    async with get_connection() as conn:
        await conn.execute(
            _LOG_UPSERT, item["type"], item["id"], res.get("layer"), res.get("session_id"),
            res["status"], (res.get("reason") or "")[:500], res.get("model"),
            res.get("n_entities", 0), res.get("n_relations", 0), res.get("latency_ms"),
        )


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
    floors_era = None
    judge, judge_key = None, None
    while True:
        try:
            if _maint.active():
                await asyncio.sleep(5)
                continue
            s = await load_settings()
            era = s["kg.extract_era_start"]
            if not s["kg.extract_enabled"] or era is None:
                await asyncio.sleep(s["kg.extract_interval"])
                continue
            if floors_era != era:
                async with get_connection() as conn:
                    floors = {"memory": await era_floor(conn, "memory", era),
                              "messages": await era_floor(conn, "messages", era)}
                floors_era = era
            async with get_connection() as conn:
                items = await pending(conn, floors, era, s)
            if not items:
                await asyncio.sleep(s["kg.extract_interval"])
                continue
            if judge is None or judge_key != s["kg.extract_model"]:
                judge = await build_judge(
                    s["kg.extract_model"], default_provider=default_provider_getter(),
                    get_config=get_config, who="KG extract")
                judge_key = s["kg.extract_model"]
                if judge is None:
                    logger.warning("KG extract: no usable model (kg.extract_model); waiting")
                    await asyncio.sleep(max(60, s["kg.extract_interval"]))
                    continue
            for item in items:
                try:
                    res = await process_one(item, judge, embed, s)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    res = {"status": "error", "reason": f"{type(e).__name__}: {e}",
                           "layer": 0 if item["type"] == "memory" else (1 if item.get("role") == "user" else 2),
                           "session_id": item.get("session_id"), "model": getattr(judge, "label", "")}
                    logger.warning(f"KG extract {item['type']} {item['id']} failed: {res['reason'][:200]}")
                await write_log(item, res)
                if res["status"] == "done":
                    logger.info(
                        f"KG extract {item['type']} {item['id']}: {res['n_entities']} entities, "
                        f"{res['n_relations']} relations ({res['latency_ms']} ms)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"KG extract round failed: {e}")
            await asyncio.sleep(60)
