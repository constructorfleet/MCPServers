import hashlib
import math
import re
from collections import Counter
from typing import Optional

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    KeywordIndexParams,
    KeywordIndexType,
    PayloadSchemaType,
    ScoredPoint,
    Snowball,
    SnowballLanguage,
    SnowballParams,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)
from qdrant_client.models import SparseVector

import logging

from plex.knowledge.types import DataPoint, PlexMediaPayload, TModel

_LOGGER = logging.getLogger(__name__)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "it",
    "as",
    "from",
}


def _sparse_from_text(text: str) -> SparseVector:
    """Build a Qdrant SparseVector from text using a stable hashed bag-of-words.

    - Tokenize on alphanumerics, lowercase
    - Drop short/common stopwords
    - Weight = 1 + ln(tf)
    - Index = blake2b(token) -> int -> modulo 2^31-1
    - Merge collisions (sum values) and ensure unique indices
    """
    if not text:
        return SparseVector(indices=[], values=[])
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    counts = Counter(t for t in tokens if t not in _STOPWORDS and len(t) > 1)
    if not counts:
        return SparseVector(indices=[], values=[])
    index_to_val: dict[int, float] = {}
    for tok, tf in counts.items():
        h = (
            int.from_bytes(hashlib.blake2b(tok.encode("utf-8"),
                           digest_size=8).digest(), "little")
            % 2147483647
        )
        val = 1.0 + math.log(tf)
        index_to_val[h] = index_to_val.get(h, 0.0) + val
    indices = list(index_to_val.keys())
    values = [index_to_val[i] for i in indices]
    return SparseVector(indices=indices, values=values)


def _word_count(s: str | None) -> int:
    """Count the number of words in a string.

    Args:
        s: String to count words in, can be None

    Returns:
        int: Number of words, 0 if string is None or empty
    """
    if not s:
        return 0
    return len([w for w in s.split() if w.strip()])


# --- Qdrant collection and index helpers ---
async def ensure_collection(
    client: AsyncQdrantClient,
    name: str,
    dim: int,
) -> None:
    """Create the collection if it doesn't exist, with dense + sparse vectors."""
    _LOGGER.warning(
        f"Ensuring collection '{name}' exists with {dim} dimensions.")
    collections = await client.get_collections()
    if any(c.name == name for c in collections.collections):
        return
        # collection = await client.get_collection(name)
        # if not collection.config.params.vectors:
        #     recreate = True
        # elif not isinstance(collection.config.params.vectors, dict):
        #     recreate = True
        # elif not collection.config.params.vectors.get("dense"):
        #     recreate = True
        # elif collection.config.params.vectors["dense"].size != dim:
        #     recreate = True
        # elif not collection.config.params.sparse_vectors:
        #     recreate = True
        # elif not isinstance(collection.config.params.sparse_vectors, dict):
        #     recreate = True
        # elif collection.config.params.sparse_vectors.get("sparse"):
        #     recreate = True
        # if not recreate:
        #     return
        # _LOGGER.warning(f"Deleting collection '{name}'")
        # await client.delete_collection(name)
    _LOGGER.warning(f"Creating collection '{name}'")
    await client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(
            size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
        on_disk_payload=True,
    )


async def ensure_payload_indexes(client: AsyncQdrantClient, name: str) -> None:
    """Create payload indexes used by filters and lexical search."""
    # TEXT indexes
    for fld in [
        "title",
        "summary",
        "show_title",
    ]:
        await client.create_payload_index(
            collection_name=name,
            field_name=fld,
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                lowercase=True,
                stemmer=SnowballParams(
                    type=Snowball.SNOWBALL,
                    language=SnowballLanguage.ENGLISH,
                ),
                on_disk=True,
                phrase_matching=True,
            ),
        )

    # KEYWORD indexes
    for fld in [
        "genres",
        "actors",
        "directors",
        "writers",
        "studio",
        "collection",
        "content_rating",
        "status",
        "show_status",
        "reviews.text",
    ]:
        await client.create_payload_index(
            collection_name=name,
            field_name=fld,
            field_schema=KeywordIndexParams(
                type=KeywordIndexType.KEYWORD,
                on_disk=True,
            ),
        )

    for fld in ["rating", "ratings.score"]:
        await client.create_payload_index(
            collection_name=name, field_name=fld, field_schema=PayloadSchemaType.FLOAT
        )

    for fld in ["air_date"]:
        await client.create_payload_index(
            collection_name=name, field_name=fld, field_schema=PayloadSchemaType.DATETIME
        )
    for fld in ["duration_seconds", "season", "episode"]:
        await client.create_payload_index(
            collection_name=name, field_name=fld, field_schema=PayloadSchemaType.INTEGER
        )

    for fld in ["watched"]:
        await client.create_payload_index(
            collection_name=name, field_name=fld, field_schema=PayloadSchemaType.BOOL
        )


def heuristic_rerank(
    q: PlexMediaPayload, points: list[DataPoint[TModel]]
) -> list[DataPoint[TModel]]:
    """Rerank search results using domain-specific heuristics.

    This method applies additional scoring based on genre, cast, and director
    overlap to improve relevance of search results.

    Args:
        q: Original query payload
        points: Search results to rerank

    Returns:
        list[DataPoint[TModel]]: Reranked results
    """

    def jaccard(a: list[str] | None, b: list[str] | None) -> float:
        sa, sb = set(a or []), set(b or [])
        if not sa and not sb:
            return 0.0
        i = len(sa & sb)
        u = len(sa | sb)
        return (i / u) if u else 0.0

    rescored: list[tuple[float, DataPoint[TModel]]] = []
    for dp in points:
        item = dp.payload_data()
        base = float(getattr(dp, "score", 0.0) or 0.0)
        g = jaccard(getattr(q, "genres", None), getattr(item, "genres", None))
        c = jaccard(getattr(q, "actors", None), getattr(item, "actors", None))
        d = jaccard(getattr(q, "directors", None),
                    getattr(item, "directors", None))
        blend = 0.75 * base + 0.15 * g + 0.06 * c + 0.04 * d
        rescored.append((blend, dp))
    rescored.sort(key=lambda t: t[0], reverse=True)
    return [dp for _, dp in rescored]


def explain_match(q: PlexMediaPayload, item: PlexMediaPayload) -> Optional[str]:
    """Generate an explanation of why a search result matched the query.

    Args:
        q: Query payload
        item: Matched item payload

    Returns:
        Optional[str]: Human-readable explanation of the match, or None
    """
    reasons: list[str] = []
    if q.title and item.title and q.title.lower() in item.title.lower():
        reasons.append("title match")
    if q.show_title and item.show_title and q.show_title.lower() in item.show_title.lower():
        reasons.append("show match")
    if q.genres:
        common = sorted(set(q.genres).intersection(set(item.genres or [])))
        if common:
            reasons.append(f"genres overlap: {', '.join(common)}")
    if q.actors:
        common = sorted(set(q.actors).intersection(set(item.actors or [])))
        if common:
            reasons.append(f"cast overlap: {', '.join(common)}")
    if q.directors:
        common = sorted(set(q.directors).intersection(
            set(item.directors or [])))
        if common:
            reasons.append(f"director overlap: {', '.join(common)}")
    return ", ".join(reasons) or None


def series_key(payload: dict) -> Optional[str]:
    """Extract a series/collection identifier from payload data.

    Args:
        payload: Media item payload data

    Returns:
        Optional[str]: Series identifier or None if not found
    """
    return str(payload.get("collection") or payload.get("show_title") or "") or None


def title_shingles(title: Optional[str]) -> set[str]:
    """Generate 3-gram shingles from a title for similarity comparison.

    Args:
        title: Title text to process

    Returns:
        set[str]: Set of 3-gram shingles (or individual tokens if < 3 tokens)
    """
    if not title:
        return set()
    t = "".join(ch.lower() if ch.isalnum() else " " for ch in title)
    tokens = [tok for tok in t.split() if tok]
    # 3-gram shingles over tokens
    if len(tokens) < 3:
        return set(tokens)
    return {" ".join(tokens[i: i + 3]) for i in range(len(tokens) - 2)}


def sim_items(a: DataPoint[TModel], b: DataPoint[TModel]) -> float:
    """Calculate similarity between two media items for diversity filtering.

    Args:
        a: First media item
        b: Second media item

    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    pa, pb = a.payload or {}, b.payload or {}
    # Title similarity via shingles
    ta = title_shingles(pa.get("title"))
    tb = title_shingles(pb.get("title"))
    title_j = (len(ta & tb) / len(ta | tb)) if (ta or tb) else 0.0
    # Genre similarity
    ga = set((pa.get("genres") or []))
    gb = set((pb.get("genres") or []))
    genre_j = (len(ga & gb) / len(ga | gb)) if (ga or gb) else 0.0
    # Weighted blend for similarity (not score)
    return 0.7 * title_j + 0.3 * genre_j


def apply_diversity(
    points: list["DataPoint[TModel]"], limit: int, diversity_lambda: float, max_per_series: int
) -> list["DataPoint[TModel]"]:
    """Apply diversity filtering using Maximal Marginal Relevance (MMR).

    This method reduces redundancy in search results by selecting items that
    balance relevance with diversity, preventing too many similar results.

    Args:
        points: Input search results
        limit: Maximum number of results to return

    Returns:
        list[DataPoint[TModel]]: Diversified results up to the limit
    """
    if limit <= 0 or not points:
        return points
    lam = max(0.0, min(1.0, diversity_lambda))
    selected: list[DataPoint[TModel]] = []
    series_counts: dict[str, int] = {}
    # Greedy MMR selection
    # Assume input is roughly sorted by model-score descending
    pool = points[:]
    while pool and len(selected) < limit:
        best_idx = 0
        best_val = float("-inf")
        for i, cand in enumerate(pool):
            payload = cand.payload or {}
            skey = series_key(payload)
            if skey and series_counts.get(skey, 0) >= max_per_series:
                continue
            rel = float(getattr(cand, "score", 0.0) or 0.0)
            div = 0.0
            if selected:
                div = max(sim_items(cand, s) for s in selected)
            val = lam * rel - (1.0 - lam) * div
            if val > best_val:
                best_val = val
                best_idx = i
        choice = pool.pop(best_idx)
        payload = choice.payload or {}
        skey = series_key(payload)
        if skey:
            series_counts[skey] = series_counts.get(skey, 0) + 1
        selected.append(choice)
    return selected


def fuse_two_pass(
    dense: list[ScoredPoint],
    sparse: list[ScoredPoint],
    dense_w: float,
    sparse_w: float,
    rrf_k: int = 60,
) -> list[ScoredPoint]:
    """Fuse dense and sparse search results using weighted Reciprocal Rank Fusion.

    Args:
        dense: Results from dense vector search
        sparse: Results from sparse vector search
        dense_w: Weight for dense results
        sparse_w: Weight for sparse results
        rrf_k: RRF parameter (typically 60)

    Returns:
        list[ScoredPoint]: Fused results sorted by combined score
    """
    # Build rank maps
    d_rank = {p.id: i for i, p in enumerate(dense, start=1)}
    s_rank = {p.id: i for i, p in enumerate(sparse, start=1)}
    # Union ids
    ids = list({*d_rank.keys(), *s_rank.keys()})
    # Index lookup
    by_id: dict = {}
    for p in dense:
        by_id.setdefault(p.id, p)
    for p in sparse:
        by_id.setdefault(p.id, p)
    # Weighted RRF score
    fused: list[tuple[float, ScoredPoint]] = []
    for pid in ids:
        dr = d_rank.get(pid)
        sr = s_rank.get(pid)
        d_score = (1.0 / (rrf_k + dr)) if dr is not None else 0.0
        s_score = (1.0 / (rrf_k + sr)) if sr is not None else 0.0
        score = dense_w * d_score + sparse_w * s_score
        fused.append((score, by_id[pid]))
    fused.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in fused]
