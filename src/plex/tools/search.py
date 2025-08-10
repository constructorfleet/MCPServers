from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from enum import StrEnum
from typing import Annotated, Any, Generic, Iterable, List, Optional, Sequence, Tuple, cast
from typing import Type as ClassType
from typing import TypeVar

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from qdrant_client.models import (
    Condition,
    DatetimeRange,
    Document,
    FieldCondition,
    Filter,
    MatchPhrase,
    MatchText,
    MatchValue,
    MinShould,
    Prefetch,
    QueryInterface,
    Range,
    RecommendInput,
    RecommendQuery,
    ScoredPoint,
    VectorInput,
)
from scipy import sparse

from plex.knowledge import KnowledgeBase
from plex.knowledge.collection import Collection
from plex.knowledge.types import DataPoint, MediaType, PlexMediaPayload, PlexMediaQuery
from plex.knowledge.utils import sparse_from_text

# from plex.knowledge.utils import _word_count, heuristic_rerank

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class Scope(StrEnum):
    MOVIE = "movies"
    # SERIES = "series"
    # SEASON = "season"
    EPISODE = "episodes"


async def ScopeCollection(scope: Scope) -> Collection[PlexMediaPayload]:
    collection: Collection[PlexMediaPayload] | None
    if scope == Scope.MOVIE:
        collection = await KnowledgeBase.instance().movies()
    elif scope == Scope.EPISODE:
        collection = await KnowledgeBase.instance().episodes()
    if collection is None:
        raise ValueError(f"Unknown scope: {scope}")
    return collection


class ResultMode(StrEnum):
    AUTO = "auto"
    SERIES_ONLY = "series_only"
    EPISODES_ONLY = "episodes_only"
    MIXED = "mixed"


class AnyRole(StrEnum):
    ANY = "any"
    ACTOR = "actor"
    DIRECTOR = "director"
    WRITER = "writer"


class Seed(BaseModel):
    title: Annotated[
        Optional[str],
        Field(
            description=(
                "Exact title of a known movie, show, or episode to use as the anchor for similarity search. "
                "The system retrieves media most similar to this title before applying other filters."
            ),
            examples=["Alien: Romulus", "Inception"],
        ),
    ] = None

    key: Annotated[
        Optional[int],
        Field(
            description=(
                "Unique Plex library key of a known media item to use as the anchor for similarity search."
            ),
            examples=[12345, 67890],
        ),
    ] = None

    genres: Annotated[
        Optional[str],
        Field(
            description=(
                "Comma-separated genres to anchor the similarity search. "
                "Retrieves media with genres most similar to these values before applying other filters."
            ),
            examples=["horror", "action, adventure"],
        ),
    ] = None

    summary: Annotated[
        Optional[str],
        Field(
            description=(
                "Brief plot or synopsis to anchor the similarity search. "
                "Retrieves media with similar summaries before applying other filters."
            ),
            examples=["A thrilling sci-fi adventure.",
                      "A mind-bending thriller."],
        ),
    ] = None

    series: Annotated[
        Optional[str],
        Field(
            description=(
                "Series title to anchor the similarity search, focusing results on similar shows or episodes."
            ),
            examples=["Alien", "Inception"],
        ),
    ] = None

    season: Annotated[
        Optional[int],
        Field(
            description=(
                "Season number to anchor the similarity search within a series context."),
            examples=[1, 2],
        ),
    ] = None

    episode: Annotated[
        Optional[int],
        Field(
            description=(
                "Episode number to anchor the similarity search within a specific season."
            ),
            examples=[1, 2],
        ),
    ] = None

    directors: Annotated[
        Optional[str],
        Field(
            description=(
                "Comma-separated directors to anchor the similarity search by creative style."
            ),
            examples=["Ridley Scott", "Christopher Nolan, Lisa Joy"],
        ),
    ] = None

    writers: Annotated[
        Optional[str],
        Field(
            description=(
                "Comma-separated writers to anchor the similarity search by writing style or story tone."
            ),
            examples=["Dan O'Bannon", "Jonathan Nolan, Lisa Joy"],
        ),
    ] = None

    actors: Annotated[
        Optional[str],
        Field(
            description=(
                "Comma-separated actors to anchor the similarity search by performance style or cast overlap."
            ),
            examples=["Sigourney Weaver", "Leonardo DiCaprio, Tom Hardy"],
        ),
    ] = None


class Intent(StrEnum):
    AUTO = "auto"
    SIMILAR_TO = "similar_to"
    BY_ACTOR = "by_actor"
    BY_DIRECTOR = "by_director"
    BY_WRITER = "by_writer"
    BY_GENRE = "by_genre"
    KEYWORD = "keyword"
    VAGUE_PLOT = "vague_plot"
    CONTINUE_WATCHING_LIKE = "continue_watching_like"


class Role(StrEnum):
    ACTOR = "actor"
    DIRECTOR = "director"
    WRITER = "writer"


class SeriesStatus(StrEnum):
    ANY = "any"
    ONGOING = "ongoing"
    ENDED = "ended"


class Filters(BaseModel):
    similar: Annotated[
        Optional[str], Field(
            description="Title or key of similar media to anchor the query.")
    ] = None
    genres_any: Annotated[
        Optional[List[Annotated[str, Field(description="Genre to match")]]],
        Field(description="At least one genre must match."),
    ] = None
    genres_all: Annotated[
        Optional[List[Annotated[str, Field(description="Genre to match")]]],
        Field(description="All genres must match."),
    ] = None
    actors_any: Annotated[
        Optional[List[Annotated[str, Field(description="Actor to match")]]],
        Field(description="At least one actor must match."),
    ] = None
    directors_any: Annotated[
        Optional[List[Annotated[str, Field(description="Director to match")]]],
        Field(description="At least one director must match."),
    ] = None
    writers_any: Annotated[
        Optional[List[Annotated[str, Field(description="Writer to match")]]],
        Field(description="At least one writer must match."),
    ] = None
    actors_all: Annotated[
        Optional[List[Annotated[str, Field(description="Actor to match")]]],
        Field(description="All actors must match."),
    ] = None
    directors_all: Annotated[
        Optional[List[Annotated[str, Field(description="Director to match")]]],
        Field(description="All directors must match."),
    ] = None
    writers_all: Annotated[
        Optional[List[Annotated[str, Field(description="Writer to match")]]],
        Field(description="All writers must match."),
    ] = None
    air_date_range_min: Annotated[
        Optional[date],
        Field(description="Minimum air date"),
    ] = None
    air_date_range_max: Annotated[
        Optional[date],
        Field(description="Maximum air date"),
    ] = None
    runtime_range_min: Annotated[
        Optional[int], Field(description="Minimum runtime in minutes.")
    ] = None
    runtime_range_max: Annotated[
        Optional[int], Field(description="Maximum runtime in minutes.")
    ] = None
    content_rating_any: Annotated[
        Optional[List[Annotated[str, Field(
            description="Content rating to include.")]]],
        Field(description="Content ratings to include."),
    ] = None
    exclude_titles: Annotated[
        Optional[List[Annotated[str, Field(
            description="Titles to exclude.")]]],
        Field(description="Titles to exclude."),
    ] = None
    season_range_min: Annotated[Optional[int], Field(
        description="Minimum season number.")] = None
    season_range_max: Annotated[Optional[int], Field(
        description="Maximum season number.")] = None
    season: Annotated[Optional[int], Field(
        description="Exact season number.")] = None
    episode_range_min: Annotated[
        Optional[int],
        Field(description="Minimum episode number."),
    ] = None
    episode_range_max: Annotated[
        Optional[int],
        Field(description="Maximum episode number."),
    ] = None
    episode: Annotated[Optional[int], Field(
        description="Exact episode number.")] = None
    year_range_min: Annotated[Optional[int], Field(
        description="Minimum release year.")] = None
    year_range_max: Annotated[Optional[int], Field(
        description="Maximum release year.")] = None
    year: Annotated[Optional[int], Field(
        description="Exact release year.")] = None


class EpisodeFocus(BaseModel):
    series_title: Annotated[Optional[str], Field(
        description="Title of the series.")] = None
    season: Annotated[Optional[int], Field(
        description="Season number.")] = None
    episode: Annotated[Optional[int], Field(
        description="Episode number.")] = None
    episode_title: Annotated[Optional[str], Field(
        description="Title of the episode.")] = None
    arc_keywords: Annotated[
        Optional[List[Annotated[str, Field(description="Story arc")]]],
        Field(
            description="Filter episodes by story arc",
            examples=["bottle episode", "anthology", "heist", "time loop"],
        ),
    ] = None


class Pacing(StrEnum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


class Vibes(BaseModel):
    tone: Annotated[
        Optional[List[Annotated[str, Field(description="Tone to match")]]],
        Field(description="Filter media by tone."),
    ] = None
    themes: Annotated[
        Optional[List[Annotated[str, Field(description="Theme to match")]]],
        Field(description="Filter media by themes."),
    ] = None
    pacing: Annotated[Optional[Pacing], Field(
        description="Filter media by pacing.")] = None
    scariness: Annotated[
        Optional[int], Field(
            ge=1, le=10, description="Filter media by scariness level (1-10).")
    ] = None


class Hybrid(BaseModel):
    dense_weight: Annotated[
        Optional[float], Field(
            description="Weight for dense representations", ge=0.0, le=1.0)
    ] = 0.7
    sparse_weight: Annotated[
        Optional[float], Field(
            description="Weight for sparse representations", ge=0.0, le=1.0)
    ] = 0.3


class Rerank(BaseModel):
    model: Annotated[Optional[str], Field(
        description="Model to use for reranking.")] = None
    explain: Annotated[Optional[bool], Field(description="Whether to explain the reranking.")] = (
        True
    )


class Diversity(BaseModel):
    mmr_lambda: Annotated[
        Optional[float],
        Field(description="Weight for maximum marginal relevance (MMR)",
              ge=0.0, le=1.0),
    ] = 0.3
    max_per_person: Annotated[Optional[int], Field(
        description="Maximum results per person.")] = 2
    max_per_series: Annotated[Optional[int], Field(
        description="Maximum results per series.")] = 2


class Ranking(BaseModel):
    popularity_boost: Annotated[Optional[float], Field(
        description="Boost for popular items")] = 0.2
    recency_boost: Annotated[Optional[float], Field(
        description="Boost for recent items")] = 0.1
    critic_score_boost: Annotated[
        Optional[float], Field(description="Boost for high critic scores")
    ] = 0.0
    episode_weight: Annotated[
        Optional[float], Field(
            description="Bias toward episode-level matches when mixed")
    ] = 1.0


class IncludeEnum(StrEnum):
    SYNOPSIS = "synopsis"
    SUMMARY = "summary"
    CAST = "cast"
    CREW = "crew"
    GENRES = "genres"
    RUNTIME = "runtime"
    YEAR = "year"
    RATING = "rating"
    SIMILAR_TITLES = "similar_titles"
    SERIES_TITLE = "series_title"
    SEASON_NUMBER = "season_number"
    EPISODE_NUMBER = "episode_number"
    EPISODE_TITLE = "episode_title"
    AIR_DATE = "air_date"
    STATUS = "status"


class Safety(BaseModel):
    avoid_spoilers: Annotated[
        Optional[bool], Field(
            default=True, description="Whether to avoid spoilers")
    ] = True
    content_warnings_any: Annotated[
        Optional[List[str]], Field(
            description="List of content warnings to consider")
    ] = None
    exclude_content_warnings: Annotated[
        Optional[List[str]], Field(
            description="List of content warnings to exclude")
    ] = None


class SimilarMedia(BaseModel):
    key: Annotated[str, Field(description="Unique Plex key for similar media")]
    title: Annotated[str, Field(description="Title of the similar media")]
    type: Annotated[MediaType, Field(description="Type of the similar media")]


class MediaResult(BaseModel):
    result_type: Annotated[MediaType, Field(
        description="Type of media result")]
    key: Annotated[int, Field(
        description="Unique Plex key for this media result")]
    series: Annotated[Optional[str], Field(
        description="Series or collection title, if applicable")]
    title: Annotated[str, Field(
        description="Primary title of the media result")]
    season: Annotated[Optional[int], Field(description="Season number, if type is 'episode'")] = (
        None
    )
    episode: Annotated[Optional[int], Field(description="Episode number, if type is 'episode'")] = (
        None
    )
    year: Annotated[Optional[int], Field(description="Release year")] = None
    status: Annotated[Optional[str], Field(
        description="Status of the media result")] = None
    genres: Annotated[
        Optional[List[str]], Field(
            description="List of genres associated with this media result")
    ] = None
    synopsis: Annotated[Optional[str], Field(
        description="Synopsis of the media result")] = None
    summary: Annotated[Optional[str], Field(
        description="Summary of the media result")] = None
    rating: Annotated[Optional[Any], Field(
        description="Rating of the media result")] = None
    directors: Annotated[
        Optional[List[str]], Field(
            description="List of directors associated with the media result")
    ] = None
    writers: Annotated[
        Optional[List[str]], Field(
            description="List of writers associated with the media result")
    ] = None
    actors: Annotated[
        Optional[List[str]], Field(
            description="List of actors associated with the media result")
    ] = None
    content_rating: Annotated[
        Optional[str], Field(description="Content rating of the media result")
    ] = None
    runtime_seconds: Annotated[
        Optional[int], Field(
            description="Runtime of the media result in seconds")
    ] = None
    tagline: Annotated[Optional[str], Field(
        description="Tagline of the media result")] = None
    similar_media: Annotated[
        Optional[List[SimilarMedia]],
        Field(description="List of media similar to this media result"),
    ] = None
    why: Annotated[
        Optional[str], Field(
            description="Reasoning for the media result's inclusion")
    ] = None


class Retrieval(BaseModel):
    dense_weight: Annotated[float, Field(
        description="Weight for dense retrieval")]
    sparse_weight: Annotated[float, Field(
        description="Weight for sparse retrieval")]


class Diagnostics(BaseModel):
    retrieval: Annotated[Retrieval, Field(
        description="Details about the retrieval process")]
    reranker: Annotated[Optional[str], Field(description="Details about the reranking process")] = (
        None
    )
    filters_applied: Annotated[bool, Field(
        description="Whether any filters were applied")] = False
    fallback_used: Annotated[bool, Field(description="Whether a fallback mechanism was used")] = (
        False
    )


class MediaSearchResponse(BaseModel):
    results: Annotated[List[MediaResult], Field(
        description="List of media search results")]
    total: Annotated[int, Field(
        description="Total number of results found", ge=0)]
    used_intent: Annotated[str, Field(
        description="Intent used for the search")]
    used_scope: Annotated[Scope | str, Field(
        description="Scope used for the search")]
    diagnostics: Annotated[
        Diagnostics, Field(
            description="Diagnostics information about the search")
    ]


class Pagination(BaseModel):
    limit: Annotated[
        Optional[int], Field(
            default=10, description="Maximum number of results to return")
    ] = 10
    offset: Annotated[
        Optional[int],
        Field(
            default=0, description="Number of results to skip before starting to collect"),
    ] = 0


class ExplainContext(BaseModel):
    # what you sent to Qdrant
    prefetch: Optional[Prefetch] = None
    outer_filter: Optional[Filter] = None
    query_kind: str = "vector"  # "recommend" | "text" | "vector"
    query: Optional[QueryInterface] = None
    positive_point_ids: list[VectorInput] = []
    # optional: seed payloads to compute overlap against
    seed_payloads: list[PlexMediaPayload] = []
    # distance or similarity? (Qdrant returns "score" that depends on distance/sim metric)
    score_interpretation: str = "similarity"  # or "distance"


def _to_set(xs: Optional[Iterable[str]]) -> set[str]:
    return set(map(str.lower, xs or []))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = _to_set(a), _to_set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)


def _explain_condition(
    payload: PlexMediaPayload, c: Condition | List[Condition]
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
            cand = _to_set(val if isinstance(val, list) else [
                           val] if val is not None else [])
            ok = bool(wanted_any & cand)
            return ok, f"{'✓' if ok else '✗'} {key} intersects {sorted(wanted_any)}"
        if "phrase" in match:
            wanted_phrase = str(match["phrase"]).lower()
            if isinstance(val, list):
                ok = any(wanted_phrase in _to_set(v) for v in val)
            else:
                ok = (str(val).lower() ==
                      wanted_phrase) if val is not None else False
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
                None if f.must is None else (
                    f.must if isinstance(f.must, list) else [f.must])
            )
            or []
        ]
        ok = all(x for x, _ in results)
        notes.append(f"{'PASS' if ok else 'FAIL'} must: " +
                     "; ".join(msg for _, msg in results))
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
            f"{hits}/{len(results)} should matched: " +
            "; ".join(msg for _, msg in results)
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


def explain_match(
    result: ScoredPoint,
    p: PlexMediaPayload,
    ctx: ExplainContext,
) -> str:
    lines: list[str] = []

    # Header
    lines.append(
        f"{p.title} ({p.year})  — score={result.score:.4f} [{ctx.score_interpretation}]")
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
            lines.append(
                f"• Ranked by similarity to positive IDs: {ctx.positive_point_ids}")
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
        lines.append("• Genres: " +
                     ", ".join(sorted(set(p.genres), key=str.lower)))
    if p.actors:
        lines.append(
            "• Actors: "
            + ", ".join(sorted(set(p.actors), key=str.lower)[:8])
            + ("…" if len(p.actors) > 8 else "")
        )
    if p.directors:
        lines.append("• Directors: " +
                     ", ".join(sorted(set(p.directors), key=str.lower)))
    if p.writers:
        lines.append("• Writers: " +
                     ", ".join(sorted(set(p.writers), key=str.lower)))

    return "\n".join(lines)


def point_to_media_result(
    payload_class: ClassType[PlexMediaPayload],
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
    series = payload.show_title
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
        why=explain_match(p, payload, context),
    )


# async def query_by_id_as_tool(
#     point_ids: list[Union[int, str]],
#     limit: int | None = None,
#     used_intent: str = "by_id",
#     used_scope: str = "auto",
# ) -> MediaSearchResponse:
#     """Query for similar items based on a specific point ID.

#     Args:
#         point_ids: ID of the point to find similar items for
#         limit: Maximum number of results to return
#         used_intent: Intent used for this query (for diagnostics)
#         used_scope: Scope used for this query (for diagnostics)

#     Returns:
#         ToolResponse: Formatted response with similar items
#     """
#     result = await KnowledgeBase.instance().qdrant_client.retrieve(
#         collection_name="media",
#         ids=point_ids,
#         limit=limit or 10,
#         with_payload=True,
#         with_vectors=True,
#     )
#     points = [
#         DataPoint(payload_class=PlexMediaPayload,
#                   version=0, score=1.0, **p.model_dump())
#         for p in result
#     ]
#     results = [point_to_media_result(
#         PlexMediaPayload, dp, why=None) for dp in points]
#     return MediaSearchResponse(
#         results=results,
#         total=len(results),
#         used_intent=used_intent,
#         used_scope=used_scope,
#         diagnostics=Diagnostics(
#             retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
#             reranker=None,
#             filters_applied=False,
#             fallback_used=False,
#         ),
#     )


# async def recommend_as_tool(
#     positive: list[VectorInput],
#     negative: Optional[list[VectorInput]] = None,
#     limit: int | None = None,
#     used_intent: str = "recommend",
#     used_scope: str = "auto",
# ) -> MediaSearchResponse:
#     """Generate recommendations based on positive and negative examples.

#     Args:
#         positive: List of positive examples (IDs or vectors)
#         negative: Optional list of negative examples (IDs or vectors)
#         limit: Maximum number of recommendations to return
#         used_intent: Intent used for this query (for diagnostics)
#         used_scope: Scope used for this query (for diagnostics)

#     Returns:
#         ToolResponse: Formatted response with recommendations
#     """
#     # The client passes arbitrary dict as query; Qdrant accepts {"recommend": {"positive": [...], "negative": [...]}}
#     q: dict[str, Any] = {"recommend": {"positive": positive}}
#     if negative:
#         q["recommend"]["negative"] = negative
#     result = await KnowledgeBase.instance().qdrant_client.query_points(
#         collection_name="media",
#         query=RecommendQuery(
#             recommend=RecommendInput(positive=positive, negative=negative)
#         ),  # type: ignore[arg-type]
#         limit=limit or 10,
#         with_payload=True,
#     )
#     points = [
#         DataPoint.model_validate(
#             {"payload_class": PlexMediaPayload, **p.model_dump()})
#         for p in result.points
#     ]
#     results = [point_to_media_result(
#         PlexMediaPayload, dp, why=None) for dp in points]
#     return MediaSearchResponse(
#         results=results,
#         total=len(results),
#         used_intent=used_intent,
#         used_scope=used_scope,
#         diagnostics=Diagnostics(
#             retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
#             reranker=None,
#             filters_applied=False,
#             fallback_used=False,
#         ),
#     )


# async def search_as_tool_boosted(
#     scope: Scope,
#     data: PlexMediaQuery,
#     boosts: dict[str, float],
#     limit: int | None = None,
#     used_intent: str = "auto",
#     fusion_prelimit: int = 200,
#     enable_rerank: bool = False,
#     reranker_name: str = "heuristic-v1",
# ) -> MediaSearchResponse:
#     """Search with boosted scoring for specific fields.

#     This method performs search with additional boost scoring applied to
#     specified fields that match the query payload.

#     Args:
#         data: Media payload to search for
#         boosts: Dictionary mapping field names to boost weights
#         limit: Maximum number of results to return
#         used_intent: Intent used for this query (for diagnostics)
#         used_scope: Scope used for this query (for diagnostics)

#     Returns:
#         ToolResponse: Formatted response with boosted search results
#     """
#     # Build dense prefetch from the structured payload’s document
#     doc_text = PlexMediaPayload.document(data)
#     dense_doc = Document(
#         text=doc_text, model=KnowledgeBase.instance().model, options={
#             "cuda": True}
#     )  # type: ignore
#     # Build formula: sum of $score + weighted matches on payload keys
#     # Example boosts: {"genres": 0.5, "actors": 0.25}
#     sum_terms: SumExpression = SumExpression(sum=["$score"])
#     for key, w in boosts.items():
#         # If the query payload has a value for this key, boost documents matching ANY of those values
#         values = getattr(data, key, None)
#         if not values:
#             continue
#         if not isinstance(values, list):
#             values = [values]
#         sum_terms.sum.append(
#             MultExpression(
#                 mult=[float(w), FieldCondition(
#                     key=key, match=MatchAny(any=list(values)))]
#             )
#         )
#     result = await KnowledgeBase.instance().qdrant_client.query_points(
#         collection_name=scope.value,
#         prefetch=Prefetch(
#             query=dense_doc, limit=max(fusion_prelimit, (limit or 50) * 3), using="dense"
#         ),
#         query=FormulaQuery(formula=sum_terms),
#         limit=limit or 10,
#         with_payload=True,
#     )
#     points = [
#         DataPoint.model_validate(
#             {"payload_class": PlexMediaPayload, **p.model_dump()})
#         for p in result.points
#     ]
#     # Optional rerank on top
#     if enable_rerank:
#         points = heuristic_rerank(data, points)
#     results = [point_to_media_result(
#         PlexMediaPayload, dp, why=None) for dp in points]
#     return MediaSearchResponse(
#         results=results,
#         total=len(results),
#         used_intent=used_intent,
#         used_scope=scope.value,
#         diagnostics=Diagnostics(
#             retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
#             reranker=reranker_name if enable_rerank else None,
#             filters_applied=True,
#             fallback_used=False,
#         ),
#     )


# async def search_as_tool(
#     scope: Scope,
#     data: PlexMediaQuery,
#     limit: int | None = None,
#     used_intent: str = "auto",
#     enable_two_pass_fusion: bool = False,
#     fusion_dense_weight: float = 0.7,
#     fusion_sparse_weight: float = 0.3,
#     reranker_name: str = "heuristic-v1",
# ) -> MediaSearchResponse:
#     """Search the collection and return results in tool response format.

#     Args:
#         data: Media payload to search for
#         limit: Maximum number of results to return
#         used_intent: Intent used for this query (for diagnostics)
#         used_scope: Scope used for this query (for diagnostics)

#     Returns:
#         ToolResponse: Formatted response with search results and diagnostics
#     """
#     collection = await ScopeCollection(scope)
#     points = await collection.search(data, limit=limit)
#     results: list[MediaResult] = []
#     for dp in points:
#         item = dp.payload_data()
#         why = explain_match(data, item)
#         results.append(point_to_media_result(PlexMediaPayload, dp, why=why))
#     hint = " ".join([data.title or "", data.summary or "",
#                     data.show_title or ""]).strip()
#     wc = _word_count(hint)
#     if enable_two_pass_fusion:
#         dense_w = fusion_dense_weight
#         sparse_w = fusion_sparse_weight
#     else:
#         dense_w = 0.8 if wc > 12 else 0.7
#         sparse_w = 0.2 if wc > 12 else 0.3
#     return MediaSearchResponse(
#         results=results,
#         total=len(results),
#         used_intent=used_intent,
#         used_scope=scope.value,
#         diagnostics=Diagnostics(
#             retrieval=Retrieval(dense_weight=dense_w, sparse_weight=sparse_w),
#             reranker=reranker_name,
#             filters_applied=True,
#             fallback_used=False,
#         ),
#     )


async def filter_points(
    collection: str,
    filters: PlexMediaQuery,
) -> list[DataPoint[PlexMediaPayload]]:
    """Filter data points based on the provided filters.

    Args:
        scope: The scope to filter within.
        filters: The filters to apply.

    Returns:
        A list of filtered data points.
    """
    musts: list[Condition] = []
    if filters.genres:
        musts.extend(
            [
                FieldCondition(key="genres", match=MatchValue(value=genre))
                for genre in filters.genres
            ]
        )
    if filters.directors:
        musts.extend(
            [
                FieldCondition(key="directors",
                               match=MatchValue(value=director))
                for director in filters.directors
            ]
        )
    if filters.writers:
        musts.extend(
            [
                FieldCondition(key="writers", match=MatchValue(value=writer))
                for writer in filters.writers
            ]
        )
    if filters.title:
        musts.append(FieldCondition(
            key="title", match=MatchText(text=filters.title)))
    if filters.summary:
        musts.append(FieldCondition(
            key="summary", match=MatchPhrase(phrase=filters.summary)))
    if filters.season:
        musts.append(FieldCondition(
            key="season", match=MatchValue(value=filters.season)))
    if filters.episode:
        musts.append(FieldCondition(
            key="episode", match=MatchValue(value=filters.episode)))
    if filters.show_title:
        musts.append(FieldCondition(key="show_title",
                     match=MatchPhrase(phrase=filters.show_title)))
    _LOGGER.info(
        f'Filtering points with conditions: {json.dumps({
            "collection_name": collection,
            "query_filter": Filter(must=musts).model_dump(),
            "using": "dense",
            "limit": 100
        }, indent=2)}'
    )
    result = await KnowledgeBase.instance().qdrant_client.query_points(
        collection_name=collection, query_filter=Filter(must=musts), using="dense", limit=100
    )
    _LOGGER.info(
        f"Found {len(result.points)} points matching the query and filters.")
    _LOGGER.info(json.dumps(result.model_dump(), indent=2))
    return [DataPoint(payload_class=PlexMediaPayload, **p.model_dump()) for p in result.points]


# async def query_as_tool(
#     scope: Scope,
#     query: str,
#     limit: int | None = None,
#     used_intent: str = "auto",
#     used_scope: str = "auto",
#     enable_two_pass_fusion: bool = False,
#     fusion_dense_weight: float = 0.7,
#     fusion_sparse_weight: float = 0.3,
#     reranker_name: str = "heuristic-v1",
# ) -> MediaSearchResponse:
#     """Perform free-text search and return results in tool response format.

#     Args:
#         query: Free-text search query
#         limit: Maximum number of results to return
#         used_intent: Intent used for this query (for diagnostics)
#         used_scope: Scope used for this query (for diagnostics)

#     Returns:
#         MediaSearchResponse: Formatted response with search results and diagnostics
#     """
#     collection = await ScopeCollection(scope)
#     points = await collection.query(query, limit=limit)
#     results = [point_to_media_result(
#         PlexMediaPayload, dp, why=None) for dp in points]
#     wc = _word_count(query)
#     if enable_two_pass_fusion:
#         dense_w = fusion_dense_weight
#         sparse_w = fusion_sparse_weight
#     else:
#         dense_w = 0.8 if wc > 12 else 0.6
#         sparse_w = 0.2 if wc > 12 else 0.4

#     return MediaSearchResponse(
#         results=results,
#         total=len(results),
#         used_intent=used_intent,
#         used_scope=used_scope,
#         diagnostics=Diagnostics(
#             retrieval=Retrieval(dense_weight=dense_w, sparse_weight=sparse_w),
#             reranker=reranker_name,
#             filters_applied=False,
#             fallback_used=False,
#         ),
#     )


empty_seed = Seed()
empty_filters = Filters()
empty_pagination = Pagination()


class MinMax(BaseModel, Generic[T]):
    minimum: Optional[T] = None
    maximum: Optional[T] = None


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
            [FieldCondition(key="genres", match=MatchValue(value=genre))
             for genre in genres]
        )
    if directors:
        musts.extend(
            [
                FieldCondition(key="directors",
                               match=MatchValue(value=director))
                for director in directors
            ]
        )
    if writers:
        musts.extend(
            [FieldCondition(key="writers", match=MatchValue(value=writer))
             for writer in writers]
        )
    if actors:
        musts.extend(
            [FieldCondition(key="actors", match=MatchValue(value=actor))
             for actor in actors]
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
                FieldCondition(key="aired_date",
                               range=DatetimeRange(gte=after, lte=before))
            )
    if series:
        musts.append(FieldCondition(
            key="season", match=MatchValue(value=series)))
    if season:
        shoulds.extend(
            [FieldCondition(key="season", match=MatchValue(value=e)) for e in season])
    if episode:
        shoulds.extend(
            [FieldCondition(key="episode", match=MatchValue(value=e)) for e in episode])
    if rating:
        if rating.minimum:
            musts.append(FieldCondition(
                key="rating", range=Range(gte=rating.minimum)))
        if rating.maximum:
            musts.append(FieldCondition(
                key="rating", range=Range(lte=rating.maximum)))
    if watched:
        musts.append(FieldCondition(
            key="watched", match=MatchValue(value=watched)))
    if len(musts) == 0 and len(shoulds) == 0 and min_should is None:
        return None
    return Filter(
        must=musts,
        should=shoulds,
        min_should=min_should,
    )


def find_media_tool(mcp: FastMCP) -> None:
    @mcp.tool(
        name="find_media",
        description="Retrieve/recommend films and TV (series, seasons, episodes) by similarity, cast/crew, genres, keywords, or vague plot clues.",
        # output_schema=MediaSearchResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Search or Recommend Media"),
        tags={"plex", "media", "search", "recommend"},
    )
    async def tool(
        media_type: Annotated[
            str,
            Field(
                title="Media Type",
                description="The media library section to query 'movies' or 'episodes'.",
                examples=["movies", "episodes"],
            ),
        ],
        similar_to: Annotated[
            Optional[str],
            Field(
                default=None,
                title="Similarity Title Anchor Filter",
                description="The title of another media to use as a similarity anchor for the query: similar genres, plots, synopsis. etc/",
            ),
        ] = None,
        with_genre: Annotated[
            Optional[list[str]],
            Field(
                default=None,
                title="Genres",
                description="Query for media that is categorized by these genres.",
            ),
        ] = None,
        directed_by: Annotated[
            Optional[list[str]],
            Field(
                default=None,
                title="Directors",
                description="Query for media that was directed by these individuals.",
            ),
        ] = None,
        written_by: Annotated[
            Optional[list[str]],
            Field(
                default=None,
                title="Writers",
                description="Query for media that was written by these individuals.",
            ),
        ] = None,
        starring: Annotated[
            Optional[list[str]],
            Field(
                default=None,
                title="Actors",
                description="Query for media that the following individuals act in.",
            ),
        ] = None,
        with_title: Annotated[
            Optional[str],
            Field(
                default=None,
                title="Title",
                description="Query for media that has titles similar to this.",
            ),
        ] = None,
        summary: Annotated[
            Optional[str],
            Field(
                default=None,
                title="Plot",
                description="Query for media with similar plots.",
            ),
        ] = None,
        aired_before: Annotated[
            Optional[date],
            Field(
                default=None,
                title="Aired Before",
                description="Query for media aired before: <date> or <int> days ago",
            ),
        ] = None,
        aired_after: Annotated[
            Optional[date],
            Field(
                default=None,
                title="Aired After",
                description="Query for media that aired after: <date> or <int> days ago",
            ),
        ] = None,
        series: Annotated[
            Optional[str],
            Field(
                default=None,
                title="Series",
                description="Query for media that is part of this series.",
            ),
        ] = None,
        season: Annotated[
            Optional[list[int]],
            Field(
                default=None,
                title="Seasons",
                description="Query for media with these season number.",
            ),
        ] = None,
        episode: Annotated[
            Optional[list[int]],
            Field(
                default=None,
                title="Episodes",
                description="Query for media with these episode numbers.",
            ),
        ] = None,
        rating_min: Annotated[
            Optional[float],
            Field(
                default=None,
                title="Minimum Rating",
                description="Query for media with a minimum rating.",
            ),
        ] = None,
        rating_max: Annotated[
            Optional[float],
            Field(
                default=None,
                title="Maximum Rating",
                description="Query for media with a maximum rating.",
            ),
        ] = None,
        watched: Annotated[
            Optional[bool],
            Field(
                default=None,
                title="Watched Status",
                description="Query for media that has or has not been watched.",
            ),
        ] = None,
        limit: Annotated[
            Optional[int],
            Field(
                default=None,
                title="Limit",
                description="Query for a maximum number of results.",
            ),
        ] = None,
        offset: Annotated[
            Optional[int],
            Field(
                default=None,
                title="Offset",
                description="Query for the number of results to skip.",
            ),
        ] = None,
        full_text_query: Annotated[
            Optional[str],
            Field(
                default=None,
                title="Full Text Query",
                description="Generic full-text, natural language request for vague prompts.",
                examples=[
                    "What's that episode where a journalist, an artist, a musician are invited to a billionaire's house and there's a meteor at the end?"
                ],
            ),
        ] = None,
    ) -> MediaSearchResponse:
        """Find media items (movies or episodes) based on various criteria."""
        collection = str(media_type)
        if collection not in ("movies", "episodes"):
            if collection in ["movie", "episode"]:
                collection = str(media_type) + "s"
            else:
                raise ValueError("media_type must be 'movies' or 'episodes'")
        query_filter: Filter | None = None
        context = ExplainContext()
        if query_filter := build_filters(
            genres=with_genre,
            directors=directed_by,
            writers=written_by,
            actors=starring,
            aired_date=MinMax(
                minimum=aired_after,
                maximum=aired_before,
            ),
            series=series,
            season=season,
            episode=episode,
            rating=MinMax(minimum=rating_min, maximum=rating_max),
            watched=watched,
        ):
            context.outer_filter = query_filter
            context.prefetch = Prefetch(filter=query_filter)

        if similar_to:
            if full_text_query:
                context.prefetch = Prefetch(
                    prefetch=context.prefetch,
                    query=KnowledgeBase.instance().document(full_text_query),
                )
            context.query_kind = "similar"
            context.positive_point_ids = [
                p.id
                for p in await filter_points(collection, filters=PlexMediaQuery(title=similar_to))
            ]
            context.query = RecommendQuery(
                recommend=RecommendInput(positive=context.positive_point_ids)
            )
        elif full_text_query:
            context.query = KnowledgeBase.instance().document(full_text_query)

        if summary:
            context.prefetch = Prefetch(
                prefetch=context.prefetch,
                query=sparse_from_text(summary),
                using="sparse",
            )

        _LOGGER.info(
            f'Filtering points with conditions: {json.dumps({
                "collection_name": collection,
                "prefetch": context.prefetch.model_dump() if context.prefetch else None,
                "query": context.query.model_dump() if context.query and isinstance(context.query, BaseModel) else None,
                "query_filter": query_filter.model_dump() if query_filter else None,
                "using": "dense",
                "limit": (limit if limit is not None else 1000),
                "offset": (offset if offset is not None else None),
                "with_payload": True,
            }, indent=2)}'
        )

        results = await KnowledgeBase.instance().qdrant_client.query_points(
            collection_name=collection,
            prefetch=context.prefetch,
            query=context.query,
            query_filter=context.outer_filter,
            using="dense",
            limit=(limit if limit is not None else 1000),
            offset=(offset if offset is not None else None),
            with_payload=True,
        )

        _LOGGER.info(
            f"Found {len(results.points)} points matching the query and filters.")
        _LOGGER.info(json.dumps(results.model_dump(), indent=2))
        must = cast(
            list[Condition], context.outer_filter.must if context.outer_filter is not None else []
        )
        should = cast(
            list[Condition],
            context.outer_filter.should if context.outer_filter is not None else [],
        )
        return MediaSearchResponse(
            results=[
                point_to_media_result(PlexMediaPayload, point, context)
                for point in results.points[: (limit if limit else None) or 10]
            ],
            total=len(results.points),
            used_intent="auto",
            used_scope=media_type,
            diagnostics=Diagnostics(
                retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
                reranker=None,
                filters_applied=len(must) > 0 or len(should) > 0,
                fallback_used=context.query is not None,
            ),
        )
