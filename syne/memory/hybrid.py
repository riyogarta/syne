"""Hybrid search: cosine + BM25, fused by reciprocal rank (Riyo, 18 Nov 2026).

KG-RAG stage 1, step 3. Used by MemoryEngine.recall (memory + references)
and history_search (chat history, level 1 / 2).

Two ranked lists are built over the SAME filters:
  * cosine - nearest embeddings (as before), rows below min_similarity dropped;
  * BM25   - rows whose keyword index (tsv, migration 51) holds query words,
    scored with the statistics in bm25_corpus / bm25_terms:
        idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
        score(d) = sum idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))
and fused with reciprocal rank fusion: rrf(d) = sum 1 / (rrf_k + rank).
The cosine similarity of every fused row is still reported (it is what the
model sees as confidence); only the ORDER and the row set change.

Query words: the query goes through the same syne_tsv() as the rows. Words
in no document are dropped, words in more than max_df_ratio of the corpus
are dropped (they match almost everything and say little: "dan", "yang"),
the rest is kept up to max_terms, rarest first.

Any BM25 error (e.g. migration 51 not applied yet) is logged and the search
falls back to cosine alone; it never fails a recall.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("syne.memory.hybrid")

# key -> (default, description). Seeded by migration 52 and schema.sql.
DEFAULTS: dict = {
    "bm25.recall_enabled": (
        True,
        "Memory recall (memory + references) ranks by cosine + BM25 keyword score, fused by reciprocal rank. false = cosine only, as before 1.22.99.",
    ),
    "bm25.history_enabled": (
        True,
        "history_search (sort_by=similarity) ranks by cosine + BM25 keyword score, fused by reciprocal rank. false = cosine only. sort_by=recency is unchanged.",
    ),
    "bm25.k1": (1.2, "BM25 k1: how fast repeated words stop adding score. Clamped 0-3."),
    "bm25.b": (0.75, "BM25 b: how much long documents are penalised (0 = not at all, 1 = fully). Clamped 0-1."),
    "bm25.max_terms": (16, "Most query words used for BM25, rarest first. Clamped 1-64."),
    "bm25.max_df_ratio": (
        0.05,
        "Query words found in more than this share of the corpus are not used for BM25 (e.g. dan, yang, hukum, shalat in a fiqih-heavy memory). Higher = more words scored but slower: measured on 30k memories, 0.3 took 0.3-1.0 s per query, 0.05 under 0.2 s. Clamped 0.01-1.",
    ),
    "bm25.candidates": (
        20,
        "Rows taken from each list (cosine and BM25) before fusion. Clamped 1-200.",
    ),
    "bm25.rrf_k": (60, "Reciprocal rank fusion constant: score = sum 1/(rrf_k + rank). Clamped 1-1000."),
    "memory.recall_fallback": (
        True,
        "Automatic recall: when keyword routing (memory.category_routes) restricted the search to some categories and found nothing, or the best cosine similarity is below memory.recall_fallback_min_similarity, search again without the category filter.",
    ),
    "memory.recall_fallback_min_similarity": (
        0.5,
        "Best cosine similarity under which a category-filtered automatic recall is retried without the filter (memory.recall_fallback). Clamped 0-1.",
    ),
}

_CLAMPS = {
    "bm25.k1": (0.0, 3.0),
    "bm25.b": (0.0, 1.0),
    "bm25.max_terms": (1, 64),
    "bm25.max_df_ratio": (0.01, 1.0),
    "bm25.candidates": (1, 200),
    "bm25.rrf_k": (1, 1000),
    "memory.recall_fallback_min_similarity": (0.0, 1.0),
}
_INTS = {"bm25.max_terms", "bm25.candidates", "bm25.rrf_k"}


def _bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "off", "")
    if v is None:
        return default
    return bool(v)


def _num(key: str, v):
    default = DEFAULTS[key][0]
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = float(default)
    if math.isnan(x):
        x = float(default)
    lo, hi = _CLAMPS[key]
    x = max(lo, min(hi, x))
    return int(x) if key in _INTS else x


async def load_settings() -> dict:
    """Current hybrid-search config, clamped. Missing keys use DEFAULTS."""
    from ..db.models import get_config
    out = {}
    for key, (default, _) in DEFAULTS.items():
        try:
            v = await get_config(key, default)
        except Exception:
            v = default
        out[key] = _bool(v, default) if isinstance(default, bool) else _num(key, v)
    return out


@dataclass
class BM25Query:
    terms: list
    idfs: list
    avg_len: float

    @property
    def tsquery(self) -> str:
        return tsquery_text(self.terms)

    def params(self, s: dict) -> list:
        """Values for bm25_cte placeholders p_first .. p_first+5."""
        return [self.terms, self.idfs, self.tsquery,
                float(s["bm25.k1"]), float(s["bm25.b"]), float(self.avg_len)]


def tsquery_text(terms: list) -> str:
    """OR-query of literal lexemes, quoted for tsquery input."""
    out = []
    for t in terms:
        t = str(t).replace("\\", "\\\\").replace("'", "''")
        out.append(f"'{t}'")
    return " | ".join(out)


def idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


async def query_terms(conn, query: str, corpus: str, s: dict) -> Optional[BM25Query]:
    """Query words worth scoring in ``corpus``, or None when there are none."""
    stats = await conn.fetchrow(
        "SELECT n_docs, total_len FROM bm25_corpus WHERE corpus = $1", corpus)
    if not stats or not stats["n_docs"] or stats["n_docs"] <= 0:
        return None
    n = int(stats["n_docs"])
    avg_len = max(1.0, float(stats["total_len"]) / n)
    rows = await conn.fetch(
        "SELECT q.term, coalesce(t.df, 0) AS df "
        "FROM (SELECT DISTINCT unnest(tsvector_to_array(syne_tsv($1))) AS term) q "
        "LEFT JOIN bm25_terms t ON t.corpus = $2 AND t.term = q.term",
        query, corpus,
    )
    max_df = s["bm25.max_df_ratio"] * n
    picked = [(r["term"], idf(n, int(r["df"])))
              for r in rows if 0 < int(r["df"]) <= max_df]
    if not picked:
        return None
    picked.sort(key=lambda x: (-x[1], x[0]))
    picked = picked[: s["bm25.max_terms"]]
    return BM25Query([t for t, _ in picked], [w for _, w in picked], avg_len)


def bm25_cte(from_sql: str, where: str, p_limit: int, p_first: int) -> str:
    """CTEs q / cand / sc; ``sc`` holds (id, bm25) of the best rows.

    ``from_sql`` must alias the searched table as m (joins allowed, e.g.
    sessions s); ``where`` is the caller's filter over those aliases.
    Placeholders: $p_limit = rows kept; $p_first.. = BM25Query.params().
    """
    t, w, q, k1, b, avg = (f"${p_first + i}" for i in range(6))
    return f"""
        q AS (
            SELECT * FROM unnest({t}::text[], {w}::float8[]) AS q(term, idf)
        ),
        cand AS (
            SELECT m.id, m.tsv, m.tsv_len
            FROM {from_sql}
            WHERE {where}
              AND m.tsv @@ {q}::tsquery
              AND m.tsv_len > 0
        ),
        sc AS (
            SELECT c.id,
                   sum(q.idf * (x.tf * ({k1}::float8 + 1))
                       / (x.tf + {k1}::float8 * (1 - {b}::float8
                          + {b}::float8 * c.tsv_len / {avg}::float8))) AS bm25
            FROM cand c
            CROSS JOIN LATERAL (
                SELECT u.lexeme, coalesce(array_length(u.positions, 1), 1)::float8 AS tf
                FROM unnest(c.tsv) AS u
            ) x
            JOIN q ON q.term = x.lexeme
            GROUP BY c.id
            ORDER BY bm25 DESC, c.id DESC
            LIMIT ${p_limit}
        )"""


def fuse(cosine: list, bm25: list, rrf_k: int, limit: Optional[int] = None) -> list:
    """Reciprocal rank fusion of two ranked lists of row dicts (key 'id').

    A row in both lists keeps the cosine dict, gains 'bm25'. Every result
    carries 'rrf', 'rank_cosine' and 'rank_bm25' (None when absent).
    Ties: higher cosine similarity first.
    """
    merged: dict = {}
    for rank, r in enumerate(cosine, 1):
        d = dict(r)
        d.setdefault("bm25", None)
        d["rank_cosine"], d["rank_bm25"] = rank, None
        d["rrf"] = 1.0 / (rrf_k + rank)
        merged[d["id"]] = d
    for rank, r in enumerate(bm25, 1):
        d = merged.get(r["id"])
        if d is None:
            d = dict(r)
            d["rank_cosine"], d["rrf"] = None, 0.0
            merged[d["id"]] = d
        else:
            d["bm25"] = r.get("bm25")
        d["rank_bm25"] = rank
        d["rrf"] += 1.0 / (rrf_k + rank)
    out = sorted(
        merged.values(),
        key=lambda d: (-d["rrf"], -(d.get("similarity") or 0.0), -int(d["id"])),
    )
    return out if limit is None else out[:limit]
