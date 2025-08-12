import hashlib
from datetime import date, timedelta
import math
import re
from collections import Counter
from typing import Iterable, Optional, Sequence, Tuple, Type, cast

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Condition,
    Filter,
    DatetimeRange,
    Range,
    SparseVector,
    Distance,
    KeywordIndexParams,
    FieldCondition,
    KeywordIndexType,
    PayloadSchemaType,
    ScoredPoint,
    Document,
    MatchValue,
    MinShould,
    Snowball,
    SnowballLanguage,
    SnowballParams,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

import logging

from plex.knowledge.types import (
    DataPoint,
    ExplainContext,
    MediaResult,
    MinMax,
    PlexMediaPayload,
    PlexMediaQuery,
    TModel,
)

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


def sparse_from_text(text: str) -> SparseVector:
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
            int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")
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
    _LOGGER.warning(f"Ensuring collection '{name}' exists with {dim} dimensions.")
    collections = await client.get_collections()
    if any(c.name == name for c in collections.collections):
        return
    _LOGGER.warning(f"Creating collection '{name}'")
    await client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
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
        d = jaccard(getattr(q, "directors", None), getattr(item, "directors", None))
        blend = 0.75 * base + 0.15 * g + 0.06 * c + 0.04 * d
        rescored.append((blend, dp))
    rescored.sort(key=lambda t: t[0], reverse=True)
    return [dp for _, dp in rescored]


def explain_match(q: PlexMediaQuery, item: PlexMediaPayload) -> Optional[str]:
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
        common = sorted(set(q.directors).intersection(set(item.directors or [])))
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
    return {" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


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


def _to_set(xs: Optional[Iterable[str]]) -> set[str]:
    return set(map(str.lower, xs or []))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = _to_set(a), _to_set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)


def _explain_condition(
    payload: PlexMediaPayload, c: Condition | list[Condition]
) -> Tuple[bool, str]:
    """
    Return (passed?, human_reason)
    Supports MatchValue/MatchAny style (by presence of 'value' or 'any') and numeric/datetime ranges.
    """
    if isinstance(c, list):
        response = []
        for cond in c:
            result = _explain_condition(payload, cond)
            response.append(result[1])
            if not result[0]:
                return result
        return True, f"~ Conditions Passed: {', '.join(response)}"
    if not isinstance(c, FieldCondition):
        raise TypeError(f"Expected FieldCondition, got {type(c).__name__}")
    key = c.key
    val = getattr(payload, key, payload.__dict__.get(key, None))

    # match …
    if c.match is not None:
        match = c.match.model_dump(exclude_none=True, exclude_unset=True)
        if "value" in match:
            wanted = str(match["value"]).lower()
            if isinstance(val, list):
                ok = wanted in _to_set(val)
            else:
                ok = (str(val).lower() == wanted) if val is not None else False
            return ok, f"{'✓' if ok else '✗'} {key} == {match['value']!r}"
        if "any" in match:
            wanted_any = _to_set(match["any"])
            cand = _to_set(val if isinstance(val, list) else [val] if val is not None else [])
            ok = bool(wanted_any & cand)
            return ok, f"{'✓' if ok else '✗'} {key} intersects {sorted(wanted_any)}"
        if "phrase" in match:
            wanted_phrase = str(match["phrase"]).lower()
            if isinstance(val, list):
                ok = any(wanted_phrase in _to_set(v) for v in val)
            else:
                ok = (str(val).lower() == wanted_phrase) if val is not None else False
            return ok, f"{'✓' if ok else '✗'} {key} contains {match['phrase']!r}"

    # range …
    if c.range is not None:
        gte = c.range.gte
        lte = c.range.lte
        ok = True
        pieces = []
        if gte is not None:
            ok &= val is not None and val >= gte
            pieces.append(f">={gte!r}")
        if lte is not None:
            ok &= val is not None and val <= lte
            pieces.append(f"<={lte!r}")
        return (
            ok,
            f"{'✓' if ok else '✗'} {key} within {' & '.join(pieces) or 'range'} (got {val!r})",
        )

    # default fall-through
    return True, f"~ {key} (unrecognized condition type; assumed pass)"


def _explain_filter(name: str, f: Optional[Filter], payload: PlexMediaPayload) -> list[str]:
    notes: list[str] = []
    if not f:
        return notes

    if f.must:
        results = [
            _explain_condition(payload, c)
            for c in (
                None if f.must is None else (f.must if isinstance(f.must, list) else [f.must])
            )
            or []
        ]
        ok = all(x for x, _ in results)
        notes.append(f"{'PASS' if ok else 'FAIL'} must: " + "; ".join(msg for _, msg in results))
    if f.should:
        results = [
            _explain_condition(payload, c)
            for c in (
                None
                if f.should is None
                else (f.should if isinstance(f.should, list) else [f.should])
            )
            or []
        ]
        # should is advisory; count hits
        hits = sum(1 for ok, _ in results if ok)
        notes.append(
            f"{hits}/{len(results)} should matched: " + "; ".join(msg for _, msg in results)
        )
    if f.must_not:
        results = [
            _explain_condition(payload, c)
            for c in (
                None
                if f.must_not is None
                else (f.must_not if isinstance(f.must_not, list) else [f.must_not])
            )
            or []
        ]
        # ok=True means it *hit* a must_not
        violations = [msg for ok, msg in results if ok]
        if violations:
            notes.append("VIOLATED must_not: " + "; ".join(violations))
        else:
            notes.append("PASS must_not: none violated")
    return notes


def _overlap_against_seeds(item: PlexMediaPayload, seeds: Sequence[PlexMediaPayload]) -> list[str]:
    if not seeds:
        return []
    msgs = []
    # Compute max overlaps across seeds (simple and useful)

    def max_j(field: str) -> Tuple[float, Optional[str]]:
        best = 0.0
        best_title = None
        for s in seeds:
            a = getattr(item, field, []) or []
            b = getattr(s, field, []) or []
            j = _jaccard(a, b)
            if j > best:
                best, best_title = j, s.title
        return best, best_title

    for field, label in [
        ("genres", "genres"),
        ("actors", "actors"),
        ("directors", "directors"),
        ("writers", "writers"),
    ]:
        j, with_title = max_j(field)
        if j > 0:
            msgs.append(f"{label} overlap J={j:.2f} vs seed “{with_title}”")
    return msgs


# --- Main explainer ----------------------------------------------------------


def explain_match_from_context(
    result: ScoredPoint,
    p: PlexMediaPayload,
    ctx: ExplainContext,
) -> str:
    lines: list[str] = []

    # Header
    lines.append(f"{p.title} ({p.year})  — score={result.score:.4f} [{ctx.score_interpretation}]")
    lines.append(
        f"type={p.type}  duration={p.duration_seconds}s  content_rating={p.content_rating or 'N/A'}"
    )

    # Prefetch filters applied (candidate set)
    if ctx.prefetch and ctx.prefetch.filter:
        notes = _explain_filter("prefetch", ctx.prefetch.filter, p)
        if notes:
            lines.append("• Prefetch filter: " + " | ".join(notes))
    elif ctx.prefetch:
        if ctx.prefetch.query:
            lines.append("• Prefetch query present (vector/text)")

    # Outer filter (if any)
    if ctx.outer_filter:
        notes = _explain_filter("outer", ctx.outer_filter, p)
        if notes:
            lines.append("• Outer filter: " + " | ".join(notes))

    # Query kind
    if ctx.query_kind == "recommend":
        if ctx.positive_point_ids:
            lines.append(f"• Ranked by similarity to positive IDs: {ctx.positive_point_ids}")
        else:
            lines.append("• Ranked by recommend() style query (no IDs listed)")
    elif ctx.query_kind == "text":
        lines.append(
            f"• Ranked by text embedding: “{cast(Document, ctx.query).text if isinstance(ctx.query, Document) else ''}”"
        )
    elif ctx.query_kind == "vector":
        lines.append("• Ranked by raw vector similarity")
    else:
        lines.append(f"• Ranked by: {ctx.query_kind}")

    # Seed overlaps (if provided)
    if ctx.seed_payloads:
        overlaps = _overlap_against_seeds(p, ctx.seed_payloads)
        if overlaps:
            lines.append("• Overlap with seeds: " + " | ".join(overlaps))

    # Content snippets that help LLM justify to users
    # keep short to avoid turning this into a novel
    if p.genres:
        lines.append("• Genres: " + ", ".join(sorted(set(p.genres), key=str.lower)))
    if p.actors:
        lines.append(
            "• Actors: "
            + ", ".join(sorted(set(p.actors), key=str.lower)[:8])
            + ("…" if len(p.actors) > 8 else "")
        )
    if p.directors:
        lines.append("• Directors: " + ", ".join(sorted(set(p.directors), key=str.lower)))
    if p.writers:
        lines.append("• Writers: " + ", ".join(sorted(set(p.writers), key=str.lower)))

    return "\n".join(lines)


def point_to_media_result(
    payload_class: Type[PlexMediaPayload],
    p: ScoredPoint,
    context: ExplainContext,
) -> MediaResult:
    """Convert a Qdrant search result to a standardized MediaResult.

    Args:
        p: Scored point from Qdrant search
        why: Optional explanation of why this result matched

    Returns:
        MediaResult: Standardized result format for API responses
    """
    payload = PlexMediaPayload(**p.payload)  # type: ignore
    item = payload_class.model_validate(payload)
    series = None
    if payload.type == "episode":
        series = payload.show_title
    elif payload.type == "movie":
        collection = payload.collection
        if collection and len(collection) > 0:
            series = collection[0].tag

    return MediaResult(
        key=item.key,
        result_type=item.type,
        title=item.title,
        year=item.year,
        status=None,
        series=series,
        genres=item.genres,
        actors=item.actors,
        directors=item.directors,
        writers=item.writers,
        similar_media=[],
        synopsis=item.summary,
        content_rating=item.content_rating,
        rating=item.rating,
        why=explain_match_from_context(p, payload, context),
    )


def build_filters(
    genres: Optional[list[str]] = None,
    directors: Optional[list[str]] = None,
    writers: Optional[list[str]] = None,
    actors: Optional[list[str]] = None,
    aired_date: Optional[MinMax[date | int]] = None,
    series: Optional[str] = None,
    season: Optional[list[int]] = None,
    episode: Optional[list[int]] = None,
    rating: Optional[MinMax[float]] = None,
    watched: Optional[bool] = None,
) -> Optional[Filter]:
    musts: list[Condition] = []
    # must_nots: list[Condition] = []
    shoulds: list[Condition] = []
    min_should: MinShould | None = None
    if genres:
        musts.extend(
            [FieldCondition(key="genres", match=MatchValue(value=genre)) for genre in genres]
        )
    if directors:
        musts.extend(
            [
                FieldCondition(key="directors", match=MatchValue(value=director))
                for director in directors
            ]
        )
    if writers:
        musts.extend(
            [FieldCondition(key="writers", match=MatchValue(value=writer)) for writer in writers]
        )
    if actors:
        musts.extend(
            [FieldCondition(key="actors", match=MatchValue(value=actor)) for actor in actors]
        )
    if aired_date:
        after: date | None = None
        before: date | None = None
        if aired_date.minimum:
            after = (
                aired_date.minimum
                if isinstance(aired_date.minimum, date)
                else date.today() - timedelta(days=aired_date.minimum)
            )
        if aired_date.maximum:
            before = (
                aired_date.maximum
                if isinstance(aired_date.maximum, date)
                else date.today() - timedelta(days=aired_date.maximum)
            )
        if after or before:
            musts.append(
                FieldCondition(key="aired_date", range=DatetimeRange(gte=after, lte=before))
            )
    if series:
        musts.append(FieldCondition(key="season", match=MatchValue(value=series)))
    if season:
        shoulds.extend([FieldCondition(key="season", match=MatchValue(value=e)) for e in season])
    if episode:
        shoulds.extend([FieldCondition(key="episode", match=MatchValue(value=e)) for e in episode])
    if rating:
        if rating.minimum:
            musts.append(FieldCondition(key="rating", range=Range(gte=rating.minimum)))
        if rating.maximum:
            musts.append(FieldCondition(key="rating", range=Range(lte=rating.maximum)))
    if watched:
        musts.append(FieldCondition(key="watched", match=MatchValue(value=watched)))
    if len(musts) == 0 and len(shoulds) == 0 and min_should is None:
        return None
    return Filter(
        must=musts,
        should=shoulds,
        min_should=min_should,
    )
