from __future__ import annotations

import json
import logging
from datetime import date
from enum import StrEnum
import re
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
            examples=["A thrilling sci-fi adventure.", "A mind-bending thriller."],
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
            description=("Season number to anchor the similarity search within a series context."),
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
        Optional[str], Field(description="Title or key of similar media to anchor the query.")
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
        Optional[List[Annotated[str, Field(description="Content rating to include.")]]],
        Field(description="Content ratings to include."),
    ] = None
    exclude_titles: Annotated[
        Optional[List[Annotated[str, Field(description="Titles to exclude.")]]],
        Field(description="Titles to exclude."),
    ] = None
    season_range_min: Annotated[Optional[int], Field(description="Minimum season number.")] = None
    season_range_max: Annotated[Optional[int], Field(description="Maximum season number.")] = None
    season: Annotated[Optional[int], Field(description="Exact season number.")] = None
    episode_range_min: Annotated[
        Optional[int],
        Field(description="Minimum episode number."),
    ] = None
    episode_range_max: Annotated[
        Optional[int],
        Field(description="Maximum episode number."),
    ] = None
    episode: Annotated[Optional[int], Field(description="Exact episode number.")] = None
    year_range_min: Annotated[Optional[int], Field(description="Minimum release year.")] = None
    year_range_max: Annotated[Optional[int], Field(description="Maximum release year.")] = None
    year: Annotated[Optional[int], Field(description="Exact release year.")] = None


class EpisodeFocus(BaseModel):
    series_title: Annotated[Optional[str], Field(description="Title of the series.")] = None
    season: Annotated[Optional[int], Field(description="Season number.")] = None
    episode: Annotated[Optional[int], Field(description="Episode number.")] = None
    episode_title: Annotated[Optional[str], Field(description="Title of the episode.")] = None
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
    pacing: Annotated[Optional[Pacing], Field(description="Filter media by pacing.")] = None
    scariness: Annotated[
        Optional[int], Field(ge=1, le=10, description="Filter media by scariness level (1-10).")
    ] = None


class Hybrid(BaseModel):
    dense_weight: Annotated[
        Optional[float], Field(description="Weight for dense representations", ge=0.0, le=1.0)
    ] = 0.7
    sparse_weight: Annotated[
        Optional[float], Field(description="Weight for sparse representations", ge=0.0, le=1.0)
    ] = 0.3


class Rerank(BaseModel):
    model: Annotated[Optional[str], Field(description="Model to use for reranking.")] = None
    explain: Annotated[Optional[bool], Field(description="Whether to explain the reranking.")] = (
        True
    )


class Diversity(BaseModel):
    mmr_lambda: Annotated[
        Optional[float],
        Field(description="Weight for maximum marginal relevance (MMR)", ge=0.0, le=1.0),
    ] = 0.3
    max_per_person: Annotated[Optional[int], Field(description="Maximum results per person.")] = 2
    max_per_series: Annotated[Optional[int], Field(description="Maximum results per series.")] = 2


class Ranking(BaseModel):
    popularity_boost: Annotated[Optional[float], Field(description="Boost for popular items")] = 0.2
    recency_boost: Annotated[Optional[float], Field(description="Boost for recent items")] = 0.1
    critic_score_boost: Annotated[
        Optional[float], Field(description="Boost for high critic scores")
    ] = 0.0
    episode_weight: Annotated[
        Optional[float], Field(description="Bias toward episode-level matches when mixed")
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
        Optional[bool], Field(default=True, description="Whether to avoid spoilers")
    ] = True
    content_warnings_any: Annotated[
        Optional[List[str]], Field(description="List of content warnings to consider")
    ] = None
    exclude_content_warnings: Annotated[
        Optional[List[str]], Field(description="List of content warnings to exclude")
    ] = None


class SimilarMedia(BaseModel):
    key: Annotated[str, Field(description="Unique Plex key for similar media")]
    title: Annotated[str, Field(description="Title of the similar media")]
    type: Annotated[MediaType, Field(description="Type of the similar media")]


class MediaResult(BaseModel):
    result_type: Annotated[MediaType, Field(description="Type of media result")]
    key: Annotated[int, Field(description="Unique Plex key for this media result")]
    series: Annotated[Optional[str], Field(description="Series or collection title, if applicable")]
    title: Annotated[str, Field(description="Primary title of the media result")]
    season: Annotated[Optional[int], Field(description="Season number, if type is 'episode'")] = (
        None
    )
    episode: Annotated[Optional[int], Field(description="Episode number, if type is 'episode'")] = (
        None
    )
    year: Annotated[Optional[int], Field(description="Release year")] = None
    status: Annotated[Optional[str], Field(description="Status of the media result")] = None
    genres: Annotated[
        Optional[List[str]], Field(description="List of genres associated with this media result")
    ] = None
    synopsis: Annotated[Optional[str], Field(description="Synopsis of the media result")] = None
    summary: Annotated[Optional[str], Field(description="Summary of the media result")] = None
    rating: Annotated[Optional[Any], Field(description="Rating of the media result")] = None
    directors: Annotated[
        Optional[List[str]], Field(description="List of directors associated with the media result")
    ] = None
    writers: Annotated[
        Optional[List[str]], Field(description="List of writers associated with the media result")
    ] = None
    actors: Annotated[
        Optional[List[str]], Field(description="List of actors associated with the media result")
    ] = None
    content_rating: Annotated[
        Optional[str], Field(description="Content rating of the media result")
    ] = None
    runtime_seconds: Annotated[
        Optional[int], Field(description="Runtime of the media result in seconds")
    ] = None
    tagline: Annotated[Optional[str], Field(description="Tagline of the media result")] = None
    similar_media: Annotated[
        Optional[List[SimilarMedia]],
        Field(description="List of media similar to this media result"),
    ] = None
    why: Annotated[
        Optional[str], Field(description="Reasoning for the media result's inclusion")
    ] = None


class Retrieval(BaseModel):
    dense_weight: Annotated[float, Field(description="Weight for dense retrieval")]
    sparse_weight: Annotated[float, Field(description="Weight for sparse retrieval")]


class Diagnostics(BaseModel):
    retrieval: Annotated[Retrieval, Field(description="Details about the retrieval process")]
    reranker: Annotated[Optional[str], Field(description="Details about the reranking process")] = (
        None
    )
    filters_applied: Annotated[bool, Field(description="Whether any filters were applied")] = False
    fallback_used: Annotated[bool, Field(description="Whether a fallback mechanism was used")] = (
        False
    )


class MediaSearchResponse(BaseModel):
    results: Annotated[List[MediaResult], Field(description="List of media search results")]
    total: Annotated[int, Field(description="Total number of results found", ge=0)]
    used_intent: Annotated[str, Field(description="Intent used for the search")]
    used_scope: Annotated[Scope | str, Field(description="Scope used for the search")]
    diagnostics: Annotated[
        Diagnostics, Field(description="Diagnostics information about the search")
    ]


class Pagination(BaseModel):
    limit: Annotated[
        Optional[int], Field(default=10, description="Maximum number of results to return")
    ] = 10
    offset: Annotated[
        Optional[int],
        Field(default=0, description="Number of results to skip before starting to collect"),
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
        status=(payload.get("show_status") if rtype == "episode" else payload.get("status")),
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
        DataPoint(payload_class=PlexMediaPayload, version=0, score=1.0, **p.model_dump())
        for p in result
    ]
    results = [point_to_media_result(PlexMediaPayload, dp, why=None) for dp in points]
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
        DataPoint.model_validate({"payload_class": PlexMediaPayload, **p.model_dump()})
        for p in result.points
    ]
    results = [point_to_media_result(PlexMediaPayload, dp, why=None) for dp in points]
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
        text=doc_text, model=KnowledgeBase.instance().model, options={"cuda": True}
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
                mult=[float(w), FieldCondition(key=key, match=MatchAny(any=list(values)))]
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
        DataPoint.model_validate({"payload_class": PlexMediaPayload, **p.model_dump()})
        for p in result.points
    ]
    # Optional rerank on top
    if enable_rerank:
        points = heuristic_rerank(data, points)
    results = [point_to_media_result(PlexMediaPayload, dp, why=None) for dp in points]
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
    hint = " ".join([data.title or "", data.summary or "", data.show_title or ""]).strip()
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
                FieldCondition(key="directors", match=MatchValue(value=director))
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
        musts.append(FieldCondition(key="title", match=MatchPhrase(phrase=filters.title)))
    if filters.summary:
        musts.append(FieldCondition(key="summary", match=MatchPhrase(phrase=filters.summary)))
    if filters.season:
        musts.append(FieldCondition(key="season", match=MatchValue(value=filters.season)))
    if filters.episode:
        musts.append(FieldCondition(key="episode", match=MatchValue(value=filters.episode)))
    if filters.show_title:
        musts.append(FieldCondition(key="show_title", match=MatchPhrase(phrase=filters.show_title)))
    _LOGGER.info(
        f'Filtering points with conditions: {json.dumps({
            "collection_name": collection,
            "query_filter": Filter(must=musts).model_dump(),
            "using": "dense",
            "limit": 10000
        }, indent=2)}'
    )
    result = await KnowledgeBase.instance().qdrant_client.query_points(
        collection_name=collection, query_filter=Filter(must=musts), using="dense", limit=10000
    )
    _LOGGER.info(f"Found {len(result.points)} points matching the query and filters.")
    _LOGGER.info(json.dumps(result.model_dump(), indent=2))
    return [DataPoint(payload_class=PlexMediaPayload, **p.model_dump()) for p in result.points]


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
    results = [point_to_media_result(PlexMediaPayload, dp, why=None) for dp in points]
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


empty_seed = Seed()
empty_filters = Filters()
empty_pagination = Pagination()


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
        uncategorized_query: Annotated[
            Optional[str],
            Field(
                default=None,
                title="Uncategorized Query",
                description="Natural language request for vague prompts.",
                examples=[
                    "What's that episode where a journalist, an artist, a musician are invited to a billionaire's house and there's a meteor at the end?"
                ],
            ),
        ] = None,
        similar_to_filter: Annotated[
            Seed,
            Field(
                default=empty_seed,
                title="Similarity Anchor Filter",
                description=(
                    "Attributes of known media used as an anchor for similarity search. "
                    "The search process first finds media most similar to these attributes, "
                    "then applies all other filters and query terms to that set.\n\n"
                    "Use when the request is about 'things like this' or 'similar to this'.\n\n"
                    "Examples:\n"
                    '1. What movies are like Together? -> title="Together"\n'
                    '2. Find me a thriller movie with Scarlett Johansson -> actors="Scarlett Johansson", genres="thriller"\n'
                    '3. Episodes like the one where the doctor fights the alien queen -> summary="doctor fights the alien queen"'
                ),
                json_schema_extra=Seed.model_json_schema(),
            ),
        ] = empty_seed,
        filters: Annotated[
            Filters,
            Field(
                default=empty_filters,
                title="Filters",
                description="Filters to apply to the search.",
                json_schema_extra=Filters.model_json_schema(),
            ),
        ] = empty_filters,
        pagination: Annotated[
            Pagination,
            Field(
                default=empty_pagination,
                title="Pagination",
                description="Pagination options.",
                json_schema_extra=Pagination.model_json_schema(),
            ),
        ] = empty_pagination,
        # episode_focus: Annotated[
        #     Optional[EpisodeFocus], Field(description="Episode-specific targeting for TV.")
        # ] = None,
        # vibes: Annotated[Optional[Vibes], Field(description="Vibes to guide the search.")] = None,
        # hybrid: Annotated[Optional[Hybrid], Field(description="Hybrid search options.")] = None,
        # rerank: Annotated[Optional[Rerank], Field(description="Reranking options.")] = None,
        # diversity: Annotated[
        #     Optional[Diversity], Field(description="Options to diversify the results.")
        # ] = None,
        # ranking: Annotated[
        #     Optional[Ranking], Field(description="Options to rerank the results.")
        # ] = None,
        # include: Annotated[
        #     Optional[List[IncludeEnum]],
        #     Field(description="Specify the content to include in the search results."),
        # ] = None,
        # safety: Annotated[Optional[Safety], Field(
        #     description="Safety options for the search.")] = None,
    ) -> MediaSearchResponse:
        if uncategorized_query is None and similar_to_filter is None and filters is None:
            raise ValueError("At least one of query, seeds, or filters must be provided")
        collection = str(media_type)
        if collection not in ("movies", "episodes"):
            if collection in ["movie", "episode"]:
                collection = str(media_type) + "s"
            else:
                raise ValueError("media_type must be 'movies' or 'episodes'")
        prefetch: list[Prefetch] = []
        if any(
            [
                similar_to_filter.title,
                similar_to_filter.summary,
                similar_to_filter.series,
                similar_to_filter.season,
                similar_to_filter.episode,
                similar_to_filter.genres,
                similar_to_filter.directors,
                similar_to_filter.writers,
                similar_to_filter.actors,
                filters.similar,
            ]
        ):
            positive_seeds = await filter_points(
                collection,
                PlexMediaQuery(
                    title=similar_to_filter.title or filters.similar,
                    summary=similar_to_filter.summary,
                    show_title=similar_to_filter.series,
                    season=similar_to_filter.season,
                    episode=(
                        similar_to_filter.episode if similar_to_filter.episode is not None else None
                    ),
                    genres=(
                        re.split(r"[, ]", similar_to_filter.genres)
                        if similar_to_filter.genres
                        else None
                    ),
                    directors=(
                        similar_to_filter.directors.split(",")
                        if similar_to_filter.directors
                        else None
                    ),
                    writers=(
                        similar_to_filter.writers.split(",") if similar_to_filter.writers else None
                    ),
                    actors=(
                        similar_to_filter.actors.split(",") if similar_to_filter.actors else None
                    ),
                ),
            )

            prefetch.append(
                Prefetch(
                    query=RecommendQuery(
                        recommend=RecommendInput(positive=[seed.id for seed in positive_seeds])
                    )
                )
            )
        musts: list[Condition] = []
        must_nots: list[Condition] = []
        shoulds: list[Condition] = []
        if filters.genres_all:
            musts.extend(
                [
                    FieldCondition(key="genres", match=MatchValue(value=genre))
                    for genre in filters.genres_all
                ]
            )
        if filters.genres_any:
            shoulds.extend(
                [
                    FieldCondition(key="air_date", match=MatchValue(value=genre))
                    for genre in filters.genres_any
                ]
            )
        if filters.air_date_range_min:
            musts.append(
                FieldCondition(
                    key="air_date",
                    range=DatetimeRange(gte=filters.air_date_range_min),
                )
            )
        if filters.air_date_range_max:
            musts.append(
                FieldCondition(
                    key="air_date",
                    range=DatetimeRange(lte=filters.air_date_range_max),
                )
            )
        if filters.season_range_min:
            musts.append(
                FieldCondition(
                    key="season",
                    range=Range(gte=filters.season_range_min),
                )
            )
        if filters.season_range_max:
            musts.append(
                FieldCondition(
                    key="season",
                    range=Range(lte=filters.season_range_max),
                )
            )
        if filters.season:
            musts.append(FieldCondition(key="season", match=MatchValue(value=filters.season)))
        if filters.episode_range_min:
            musts.append(
                FieldCondition(
                    key="episode",
                    range=Range(gte=filters.episode_range_min),
                )
            )
        if filters.episode_range_max:
            musts.append(
                FieldCondition(
                    key="episode",
                    range=Range(lte=filters.episode_range_max),
                )
            )
        if filters.episode:
            musts.append(FieldCondition(key="episode", match=MatchValue(value=filters.episode)))
        if filters.exclude_titles:
            must_nots.extend(
                [
                    FieldCondition(key="title", match=MatchValue(value=title))
                    for title in filters.exclude_titles
                ]
            )
        if filters.content_rating_any:
            shoulds.extend(
                [
                    FieldCondition(key="content_rating", match=MatchValue(value=rating))
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
                    FieldCondition(key="directors", match=MatchValue(value=director))
                    for director in filters.directors_all
                ]
            )
        if filters.directors_any:
            shoulds.extend(
                [
                    FieldCondition(key="directors", match=MatchValue(value=director))
                    for director in filters.directors_any
                ]
            )
        if filters.actors_all:
            musts.extend(
                [
                    FieldCondition(key="actors", match=MatchValue(value=actor))
                    for actor in filters.actors_all
                ]
            )
        if filters.actors_any:
            shoulds.extend(
                [
                    FieldCondition(key="actors", match=MatchValue(value=actor))
                    for actor in filters.actors_any
                ]
            )
        if filters.writers_all:
            musts.extend(
                [
                    FieldCondition(key="writers", match=MatchValue(value=writer))
                    for writer in filters.writers_all
                ]
            )
        if filters.writers_any:
            shoulds.extend(
                [
                    FieldCondition(key="writers", match=MatchValue(value=writer))
                    for writer in filters.writers_any
                ]
            )
        if filters.year_range_min:
            musts.append(
                FieldCondition(
                    key="year",
                    range=Range(gte=filters.year_range_min),
                )
            )
        if filters.year_range_max:
            musts.append(
                FieldCondition(
                    key="year",
                    range=Range(lte=filters.year_range_max),
                )
            )

        _LOGGER.info(
            f'Filtering points with conditions: {json.dumps({
                "collection_name": collection,
                "prefetch": [p.model_dump() for p in prefetch or []],
                "query": (
                    Document(
                        text=uncategorized_query,
                        model=KnowledgeBase.instance().model,
                        options={"cuda": True},
                    ).model_dump()
                    if uncategorized_query
                    else None
                ),
                "query_filter": Filter(must=musts, should=shoulds).model_dump(),
                "using": "dense",
                "limit": (pagination.limit if pagination and pagination.limit is not None else 1000),
                "offset": pagination.offset if pagination else None,
                "with_payload": True,
            }, indent=2)}'
        )

        results = await KnowledgeBase.instance().qdrant_client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=(
                Document(
                    text=uncategorized_query,
                    model=KnowledgeBase.instance().model,
                    options={"cuda": True},
                )
                if uncategorized_query
                else None
            ),
            query_filter=Filter(must=musts, should=shoulds),
            using="dense",
            limit=(pagination.limit if pagination and pagination.limit is not None else 1000),
            offset=pagination.offset if pagination else None,
            with_payload=True,
        )

        _LOGGER.info(f"Found {len(results.points)} points matching the query and filters.")
        _LOGGER.info(json.dumps(results.model_dump(), indent=2))

        return MediaSearchResponse(
            results=[
                point_to_media_result(PlexMediaPayload, point)
                for point in results.points[
                    : (pagination.limit if pagination.limit else None) or 10
                ]
            ],
            total=len(results.points),
            used_intent="auto",
            used_scope=media_type,
            diagnostics=Diagnostics(
                retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
                reranker=None,
                filters_applied=len(musts) > 0 or len(shoulds) > 0,
                fallback_used=uncategorized_query is not None,
            ),
        )
