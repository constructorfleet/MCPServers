from __future__ import annotations

import asyncio
import itertools
import logging
from datetime import date
from enum import Enum, StrEnum
from typing import Annotated, Any, List, Literal, Optional
from typing import Type as ClassType
from typing import Union

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from qdrant_client.models import (
    Condition,
    DatetimeRange,
    Document,
    FieldCondition,
    Filter,
    FormulaQuery,
    MatchAny,
    MatchPhrase,
    MatchValue,
    MultExpression,
    Prefetch,
    Range,
    RecommendInput,
    RecommendQuery,
    ScoredPoint,
    SumExpression,
    VectorInput,
)

from plex.knowledge import KnowledgeBase
from plex.knowledge.collection import Collection
from plex.knowledge.types import DataPoint, MediaType, PlexMediaPayload, PlexMediaQuery
from plex.knowledge.utils import _word_count, explain_match, heuristic_rerank

_LOGGER = logging.getLogger(__name__)


class Scope(Enum):
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


class ResultMode(Enum):
    AUTO = "auto"
    SERIES_ONLY = "series_only"
    EPISODES_ONLY = "episodes_only"
    MIXED = "mixed"


class SeedType(Enum):
    TITLE = "title"
    KEY = "key"
    GENRE = "genre"
    SUMMARY = "summary"
    SERIES = "series"
    SEASON = "season"
    EPISODE = "episode"
    DIRECTOR = "director"
    WRITER = "writer"
    ACTOR = "actor"


class AnyRole(Enum):
    ANY = "any"
    ACTOR = "actor"
    DIRECTOR = "director"
    WRITER = "writer"


class Seed(BaseModel):
    type: Annotated[
        SeedType, Field(description="Seed entity type.",
                        examples=["title", "key", "person"])
    ]
    value: Annotated[str, Field(description="Value of the seed entity.")]


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


class Role(Enum):
    ACTOR = "actor"
    DIRECTOR = "director"
    WRITER = "writer"


class SeriesStatus(Enum):
    ANY = "any"
    ONGOING = "ongoing"
    ENDED = "ended"


class Filters(BaseModel):
    genres_any: Annotated[
        Optional[List[str]], Field(
            description="At least one genre must match.")
    ] = None
    genres_all: Annotated[Optional[List[str]], Field(
        description="All genres must match.")] = None
    actors_any: Annotated[
        Optional[List[str]], Field(
            description="At least one actor must match.")
    ] = None
    directors_any: Annotated[
        Optional[List[str]], Field(
            description="At least one director must match.")
    ] = None
    writers_any: Annotated[
        Optional[List[str]], Field(
            description="At least one writer must match.")
    ] = None
    actors_all: Annotated[Optional[List[str]], Field(
        description="All actors must match.")] = None
    directors_all: Annotated[
        Optional[List[str]], Field(description="All directors must match.")
    ] = None
    writers_all: Annotated[Optional[List[str]], Field(
        description="All writers must match.")] = None
    air_date_range: Annotated[
        Optional[List[date]],
        Field(description="Filter media by air date by range",
              max_length=2, min_length=2),
    ] = None
    runtime_range_min: Annotated[
        Optional[int], Field(description="Minimum runtime in minutes.")
    ] = None
    runtime_range_max: Annotated[
        Optional[int], Field(description="Maximum runtime in minutes.")
    ] = None
    content_rating_any: Annotated[
        Optional[List[str]], Field(description="Content ratings to include.")
    ] = None
    exclude_titles: Annotated[Optional[List[str]],
                              Field(description="Titles to exclude.")] = None
    season_range: Annotated[
        Optional[List[int]],
        Field(description="Filter media by season number by range",
              max_length=2, min_length=2),
    ] = None
    episode_range: Annotated[
        Optional[List[int]],
        Field(description="Filter media by episode number by range",
              max_length=2, min_length=2),
    ] = None


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
        Optional[List[str]],
        Field(
            description="Filter episodes by story arc",
            examples=["bottle episode", "anthology", "heist", "time loop"],
        ),
    ] = None


class Pacing(Enum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


class Vibes(BaseModel):
    tone: Annotated[Optional[List[str]], Field(
        description="Filter media by tone.")] = None
    themes: Annotated[Optional[List[str]], Field(
        description="Filter media by themes.")] = None
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


class IncludeEnum(Enum):
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
    total: Annotated[int, Field(description="Total number of results found")]
    used_intent: Annotated[str, Field(
        description="Intent used for the search")]
    used_scope: Annotated[str, Field(description="Scope used for the search")]
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


def point_to_media_result(
    payload_class: ClassType[PlexMediaPayload], p: ScoredPoint, why: Optional[str] = None
) -> MediaResult:
    """Convert a Qdrant search result to a standardized MediaResult.

    Args:
        p: Scored point from Qdrant search
        why: Optional explanation of why this result matched

    Returns:
        MediaResult: Standardized result format for API responses
    """
    payload = p.payload or {}
    item = payload_class.model_validate(payload)
    # Determine result type (we only store movies/episodes today)
    rtype: Literal["series", "episode", "movie"]
    if getattr(item, "type", None) == "episode":
        rtype = "episode"
    elif getattr(item, "type", None) == "series":
        rtype = "series"
    else:
        rtype = "movie"
    # IDs
    series = None
    # Prefer a stable collection/series/franchise key if present; fallback to show_title for episodes
    if payload.get("collection"):
        series = str(payload.get("collection"))
    elif rtype == "episode":
        series = str(getattr(item, "show_title", None) or "") or None
    return MediaResult(
        key=item.key,
        result_type=item.type,
        title=item.title,
        year=item.year,
        status=(payload.get("show_status") if rtype ==
                "episode" else payload.get("status")),
        series=series,
        genres=item.genres,
        actors=item.actors,
        directors=item.directors,
        writers=item.writers,
        similar_media=[],
        synopsis=item.summary,
        content_rating=item.content_rating,
        rating=item.rating,
        why=why,
    )


async def query_by_id_as_tool(
    point_ids: list[Union[int, str]],
    limit: int | None = None,
    used_intent: str = "by_id",
    used_scope: str = "auto",
) -> MediaSearchResponse:
    """Query for similar items based on a specific point ID.

    Args:
        point_ids: ID of the point to find similar items for
        limit: Maximum number of results to return
        used_intent: Intent used for this query (for diagnostics)
        used_scope: Scope used for this query (for diagnostics)

    Returns:
        ToolResponse: Formatted response with similar items
    """
    result = await KnowledgeBase.instance().qdrant_client.retrieve(
        collection_name="media",
        ids=point_ids,
        limit=limit or 10,
        with_payload=True,
        with_vectors=True,
    )
    points = [
        DataPoint(payload_class=PlexMediaPayload,
                  version=0, score=1.0, **p.model_dump())
        for p in result
    ]
    results = [point_to_media_result(
        PlexMediaPayload, dp, why=None) for dp in points]
    return MediaSearchResponse(
        results=results,
        total=len(results),
        used_intent=used_intent,
        used_scope=used_scope,
        diagnostics=Diagnostics(
            retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
            reranker=None,
            filters_applied=False,
            fallback_used=False,
        ),
    )


async def recommend_as_tool(
    positive: list[VectorInput],
    negative: Optional[list[VectorInput]] = None,
    limit: int | None = None,
    used_intent: str = "recommend",
    used_scope: str = "auto",
) -> MediaSearchResponse:
    """Generate recommendations based on positive and negative examples.

    Args:
        positive: List of positive examples (IDs or vectors)
        negative: Optional list of negative examples (IDs or vectors)
        limit: Maximum number of recommendations to return
        used_intent: Intent used for this query (for diagnostics)
        used_scope: Scope used for this query (for diagnostics)

    Returns:
        ToolResponse: Formatted response with recommendations
    """
    # The client passes arbitrary dict as query; Qdrant accepts {"recommend": {"positive": [...], "negative": [...]}}
    q: dict[str, Any] = {"recommend": {"positive": positive}}
    if negative:
        q["recommend"]["negative"] = negative
    result = await KnowledgeBase.instance().qdrant_client.query_points(
        collection_name="media",
        query=RecommendQuery(
            recommend=RecommendInput(positive=positive, negative=negative)
        ),  # type: ignore[arg-type]
        limit=limit or 10,
        with_payload=True,
    )
    points = [
        DataPoint.model_validate(
            {"payload_class": PlexMediaPayload, **p.model_dump()})
        for p in result.points
    ]
    results = [point_to_media_result(
        PlexMediaPayload, dp, why=None) for dp in points]
    return MediaSearchResponse(
        results=results,
        total=len(results),
        used_intent=used_intent,
        used_scope=used_scope,
        diagnostics=Diagnostics(
            retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
            reranker=None,
            filters_applied=False,
            fallback_used=False,
        ),
    )


async def search_as_tool_boosted(
    scope: Scope,
    data: PlexMediaQuery,
    boosts: dict[str, float],
    limit: int | None = None,
    used_intent: str = "auto",
    fusion_prelimit: int = 200,
    enable_rerank: bool = False,
    reranker_name: str = "heuristic-v1",
) -> MediaSearchResponse:
    """Search with boosted scoring for specific fields.

    This method performs search with additional boost scoring applied to
    specified fields that match the query payload.

    Args:
        data: Media payload to search for
        boosts: Dictionary mapping field names to boost weights
        limit: Maximum number of results to return
        used_intent: Intent used for this query (for diagnostics)
        used_scope: Scope used for this query (for diagnostics)

    Returns:
        ToolResponse: Formatted response with boosted search results
    """
    # Build dense prefetch from the structured payload’s document
    doc_text = PlexMediaPayload.document(data)
    dense_doc = Document(
        text=doc_text, model=KnowledgeBase.instance().model, options={
            "cuda": True}
    )  # type: ignore
    # Build formula: sum of $score + weighted matches on payload keys
    # Example boosts: {"genres": 0.5, "actors": 0.25}
    sum_terms: SumExpression = SumExpression(sum=["$score"])
    for key, w in boosts.items():
        # If the query payload has a value for this key, boost documents matching ANY of those values
        values = getattr(data, key, None)
        if not values:
            continue
        if not isinstance(values, list):
            values = [values]
        sum_terms.sum.append(
            MultExpression(
                mult=[float(w), FieldCondition(
                    key=key, match=MatchAny(any=list(values)))]
            )
        )
    result = await KnowledgeBase.instance().qdrant_client.query_points(
        collection_name=scope.value,
        prefetch=Prefetch(
            query=dense_doc, limit=max(fusion_prelimit, (limit or 50) * 3), using="dense"
        ),
        query=FormulaQuery(formula=sum_terms),
        limit=limit or 10,
        with_payload=True,
    )
    points = [
        DataPoint.model_validate(
            {"payload_class": PlexMediaPayload, **p.model_dump()})
        for p in result.points
    ]
    # Optional rerank on top
    if enable_rerank:
        points = heuristic_rerank(data, points)
    results = [point_to_media_result(
        PlexMediaPayload, dp, why=None) for dp in points]
    return MediaSearchResponse(
        results=results,
        total=len(results),
        used_intent=used_intent,
        used_scope=scope.value,
        diagnostics=Diagnostics(
            retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
            reranker=reranker_name if enable_rerank else None,
            filters_applied=True,
            fallback_used=False,
        ),
    )


async def search_as_tool(
    scope: Scope,
    data: PlexMediaQuery,
    limit: int | None = None,
    used_intent: str = "auto",
    enable_two_pass_fusion: bool = False,
    fusion_dense_weight: float = 0.7,
    fusion_sparse_weight: float = 0.3,
    reranker_name: str = "heuristic-v1",
) -> MediaSearchResponse:
    """Search the collection and return results in tool response format.

    Args:
        data: Media payload to search for
        limit: Maximum number of results to return
        used_intent: Intent used for this query (for diagnostics)
        used_scope: Scope used for this query (for diagnostics)

    Returns:
        ToolResponse: Formatted response with search results and diagnostics
    """
    collection = await ScopeCollection(scope)
    points = await collection.search(data, limit=limit)
    results: list[MediaResult] = []
    for dp in points:
        item = dp.payload_data()
        why = explain_match(data, item)
        results.append(point_to_media_result(PlexMediaPayload, dp, why=why))
    hint = " ".join([data.title or "", data.summary or "",
                    data.show_title or ""]).strip()
    wc = _word_count(hint)
    if enable_two_pass_fusion:
        dense_w = fusion_dense_weight
        sparse_w = fusion_sparse_weight
    else:
        dense_w = 0.8 if wc > 12 else 0.7
        sparse_w = 0.2 if wc > 12 else 0.3
    return MediaSearchResponse(
        results=results,
        total=len(results),
        used_intent=used_intent,
        used_scope=scope.value,
        diagnostics=Diagnostics(
            retrieval=Retrieval(dense_weight=dense_w, sparse_weight=sparse_w),
            reranker=reranker_name,
            filters_applied=True,
            fallback_used=False,
        ),
    )


async def filter_points(
    scope: Scope,
    filters: PlexMediaQuery,
) -> list[DataPoint]:
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
            key="title", match=MatchPhrase(phrase=filters.title)))
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
    result = await KnowledgeBase.instance().qdrant_client.query_points(
        collection_name=scope.value, query_filter=Filter(must=musts), limit=10000
    )
    return [
        DataPoint(payload_class=PlexMediaPayload,
                  version=0, score=1.0, **p.model_dump())
        for p in result.points
    ]


async def query_as_tool(
    scope: Scope,
    query: str,
    limit: int | None = None,
    used_intent: str = "auto",
    used_scope: str = "auto",
    enable_two_pass_fusion: bool = False,
    fusion_dense_weight: float = 0.7,
    fusion_sparse_weight: float = 0.3,
    reranker_name: str = "heuristic-v1",
) -> MediaSearchResponse:
    """Perform free-text search and return results in tool response format.

    Args:
        query: Free-text search query
        limit: Maximum number of results to return
        used_intent: Intent used for this query (for diagnostics)
        used_scope: Scope used for this query (for diagnostics)

    Returns:
        MediaSearchResponse: Formatted response with search results and diagnostics
    """
    collection = await ScopeCollection(scope)
    points = await collection.query(query, limit=limit)
    results = [point_to_media_result(
        PlexMediaPayload, dp, why=None) for dp in points]
    wc = _word_count(query)
    if enable_two_pass_fusion:
        dense_w = fusion_dense_weight
        sparse_w = fusion_sparse_weight
    else:
        dense_w = 0.8 if wc > 12 else 0.6
        sparse_w = 0.2 if wc > 12 else 0.4

    return MediaSearchResponse(
        results=results,
        total=len(results),
        used_intent=used_intent,
        used_scope=used_scope,
        diagnostics=Diagnostics(
            retrieval=Retrieval(dense_weight=dense_w, sparse_weight=sparse_w),
            reranker=reranker_name,
            filters_applied=False,
            fallback_used=False,
        ),
    )


find_media_tool_response_schema = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "result_type": {"type": "string", "enum": ["episode", "movie"]},
                    "key": {"type": "integer"},
                    "series": {"type": "string"},
                    "title": {"type": "string"},
                    "season": {"type": "integer"},
                    "episode": {"type": "integer"},
                    "year": {"type": "integer"},
                    "status": {"type": "string"},
                    "genres": {"type": "array", "items": {"type": "string"}},
                    "synopsis": {"type": "string"},
                    "summary": {"type": "string"},
                    "rating": {"type": "number"},
                    "directors": {"type": "array", "items": {"type": "string"}},
                    "writers": {"type": "array", "items": {"type": "string"}},
                    "actors": {"type": "array", "items": {"type": "string"}},
                    "content_rating": {"type": "string"},
                    "runtime_seconds": {"type": "integer"},
                    "tagline": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["result_type", "key", "title"],
            },
        },
        "total": {"type": "integer"},
        "used_intent": {"type": "string"},
        "used_scope": {"type": "string"},
        "diagnostics": {
            "type": "object",
            "properties": {
                "retrieval": {
                    "type": "object",
                    "properties": {
                        "dense_weight": {"type": "number"},
                        "sparse_weight": {"type": "number"},
                    },
                    "required": ["dense_weight", "sparse_weight"],
                },
                "reranker": {"type": "string"},
                "filters_applied": {"type": "boolean"},
                "fallback_used": {"type": "boolean"},
            },
            "required": ["retrieval", "filters_applied", "fallback_used"],
        },
    },
    "required": ["results", "total", "used_intent", "used_scope", "diagnostics"],
}


def find_media_tool(mcp: FastMCP) -> None:
    @mcp.tool(
        name="find_media",
        description="Retrieve/recommend films and TV (series, seasons, episodes) by similarity, cast/crew, genres, keywords, or vague plot clues.",
        output_schema=find_media_tool_response_schema,
        annotations=ToolAnnotations(title="Search or Recommend Media"),
        tags={"plex", "media", "search", "recommend"},
    )
    async def tool(
        scope: Annotated[
            Scope,
            Field(
                description="Granularity to target.",
                examples=["movies", "episodes"],
            ),
        ],
        uncategorized_query: Annotated[
            Optional[str],
            Field(
                description="Natural language request for vague prompts.",
                examples=[
                    "What's that episode where a journalist, an artist, a musician are invited to a billionaire's house and there's a meteor at the end?"
                ],
            ),
        ] = None,
        query_seeds: Annotated[
            Optional[List[Seed]], Field(
                description="Anchors to guide the retrieval of media.")
        ] = None,
        filters: Annotated[
            Optional[Filters], Field(
                description="Filters to apply to the search.")
        ] = None,
        episode_focus: Annotated[
            Optional[EpisodeFocus], Field(
                description="Episode-specific targeting for TV.")
        ] = None,
        vibes: Annotated[Optional[Vibes], Field(
            description="Vibes to guide the search.")] = None,
        hybrid: Annotated[Optional[Hybrid], Field(
            description="Hybrid search options.")] = None,
        rerank: Annotated[Optional[Rerank], Field(
            description="Reranking options.")] = None,
        diversity: Annotated[
            Optional[Diversity], Field(
                description="Options to diversify the results.")
        ] = None,
        ranking: Annotated[
            Optional[Ranking], Field(
                description="Options to rerank the results.")
        ] = None,
        pagination: Annotated[
            Optional[Pagination], Field(description="Pagination options.")
        ] = None,
        # include: Annotated[
        #     Optional[List[IncludeEnum]],
        #     Field(description="Specify the content to include in the search results."),
        # ] = None,
        # safety: Annotated[Optional[Safety], Field(
        #     description="Safety options for the search.")] = None,
    ) -> MediaSearchResponse:
        if (
            uncategorized_query is None
            and (not query_seeds or len(query_seeds) == 0)
            and filters is None
        ):
            raise ValueError(
                "At least one of query, seeds, or filters must be provided")
        prefetch: list[Prefetch] = []
        if query_seeds:
            positive_seeds = await asyncio.gather(
                *[
                    filter_points(
                        scope,
                        PlexMediaQuery(
                            title=s.value if s.type == SeedType.TITLE else None,
                            summary=s.value if s.type == SeedType.SUMMARY else None,
                            show_title=s.value if s.type == SeedType.SERIES else None,
                            season=(
                                int(s.value)
                                if s.type == SeedType.SEASON and s.value.isdigit()
                                else None
                            ),
                            episode=(
                                int(s.value)
                                if s.type == SeedType.EPISODE and s.value.isdigit()
                                else None
                            ),
                            genres=[
                                s.value] if s.type == SeedType.GENRE else None,
                            directors=[
                                s.value] if s.type == SeedType.DIRECTOR else None,
                            writers=[
                                s.value] if s.type == SeedType.WRITER else None,
                            actors=[
                                s.value] if s.type == SeedType.ACTOR else None,
                        ),
                    )
                    for s in query_seeds
                ]
            )
            prefetch.append(
                Prefetch(
                    query=RecommendQuery(
                        recommend=RecommendInput(
                            positive=[
                                x.id for x in itertools.chain.from_iterable(positive_seeds)]
                        )
                    )
                )
            )
        musts: list[Condition] = []
        must_nots: list[Condition] = []
        shoulds: list[Condition] = []
        if filters is not None:
            if filters.genres_all:
                musts.extend(
                    [
                        FieldCondition(
                            key="genres", match=MatchValue(value=genre))
                        for genre in filters.genres_all
                    ]
                )
            if filters.genres_any:
                shoulds.extend(
                    [
                        FieldCondition(
                            key="air_date", match=MatchValue(value=genre))
                        for genre in filters.genres_any
                    ]
                )
            if filters.air_date_range:
                musts.append(
                    FieldCondition(
                        key="air_date",
                        range=DatetimeRange(
                            gte=filters.air_date_range[0], lte=filters.air_date_range[1]
                        ),
                    )
                )
            if filters.season_range:
                musts.append(
                    FieldCondition(
                        key="season",
                        range=Range(
                            gte=filters.season_range[0], lte=filters.season_range[1]),
                    )
                )
            if filters.episode_range:
                musts.append(
                    FieldCondition(
                        key="episode",
                        range=Range(
                            gte=filters.episode_range[0], lte=filters.episode_range[1]),
                    )
                )
            if filters.exclude_titles:
                must_nots.extend(
                    [
                        FieldCondition(
                            key="title", match=MatchValue(value=title))
                        for title in filters.exclude_titles
                    ]
                )
            if filters.content_rating_any:
                shoulds.extend(
                    [
                        FieldCondition(key="content_rating",
                                       match=MatchValue(value=rating))
                        for rating in filters.content_rating_any
                    ]
                )
            if filters.runtime_range_min:
                musts.append(
                    FieldCondition(
                        key="runtime",
                        range=Range(gte=filters.runtime_range_min),
                    )
                )
            if filters.runtime_range_max:
                musts.append(
                    FieldCondition(
                        key="runtime",
                        range=Range(lte=filters.runtime_range_max),
                    )
                )
            if filters.directors_all:
                musts.extend(
                    [
                        FieldCondition(key="directors",
                                       match=MatchValue(value=director))
                        for director in filters.directors_all
                    ]
                )
            if filters.directors_any:
                shoulds.extend(
                    [
                        FieldCondition(key="directors",
                                       match=MatchValue(value=director))
                        for director in filters.directors_any
                    ]
                )
            if filters.actors_all:
                musts.extend(
                    [
                        FieldCondition(
                            key="actors", match=MatchValue(value=actor))
                        for actor in filters.actors_all
                    ]
                )
            if filters.actors_any:
                shoulds.extend(
                    [
                        FieldCondition(
                            key="actors", match=MatchValue(value=actor))
                        for actor in filters.actors_any
                    ]
                )
            if filters.writers_all:
                musts.extend(
                    [
                        FieldCondition(
                            key="writers", match=MatchValue(value=writer))
                        for writer in filters.writers_all
                    ]
                )
            if filters.writers_any:
                shoulds.extend(
                    [
                        FieldCondition(
                            key="writers", match=MatchValue(value=writer))
                        for writer in filters.writers_any
                    ]
                )

        results = await KnowledgeBase.instance().qdrant_client.query_points(
            collection_name=scope.value,
            prefetch=prefetch,
            query=uncategorized_query,
            query_filter=Filter(must=musts, should=shoulds),
            limit=(
                pagination.limit if pagination and pagination.limit is not None else 1000),
            offset=pagination.offset if pagination else None,
            with_payload=True,
        )

        return MediaSearchResponse(
            results=[
                point_to_media_result(PlexMediaPayload, point)
                for point in results.points[
                    : pagination.limit if pagination and pagination.limit else 10
                ]
            ],
            total=len(results.points),
            used_intent="auto",
            used_scope=scope.value,
            diagnostics=Diagnostics(
                retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
                reranker=None,
                filters_applied=len(musts) > 0 or len(shoulds) > 0,
                fallback_used=uncategorized_query is not None,
            ),
        )

        # return await query_as_tool(
        #     scope=scope,
        #     query=uncategorized_query,,
        #     limit=pagination.limit if pagination else None,
        #     used_intent=intent.value if intent else "auto",
        #     used_scope=scope.value if scope else "auto",
        #     enable_two_pass_fusion=hybrid is not None,
        #     fusion_dense_weight=(
        #         hybrid.dense_weight if hybrid is not None else None) or 0.7,
        #     fusion_sparse_weight=(
        #         hybrid.sparse_weight if hybrid is not None else None) or 0.3,
        #     reranker_name=(
        #         rerank.model if rerank is not None else None) or "heuristic-v1",
        # )

        # return await search_as_tool(
        #     scope=scope,
        #     seeds=seeds,
        #     filters=filters,
        #     used_intent=intent.value if intent else "auto",
        #     used_scope=scope.value if scope else "auto",
        #     pagination=pagination,
        # )
