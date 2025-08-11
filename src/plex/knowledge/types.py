from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Generic, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field
from qdrant_client.models import (
    Filter,
    Prefetch,
    QueryInterface,
    ScoredPoint,
    VectorInput,
)

MediaType = Literal["movie"] | Literal["episode"]
T = TypeVar("T")


class MinMax(BaseModel, Generic[T]):
    minimum: Optional[T] = None
    maximum: Optional[T] = None


class Review(BaseModel):
    key: int
    reviewer: str
    text: str


class Rating(BaseModel):
    source: str
    type: str
    score: float


class PlexMediaPayload(BaseModel):
    """Data model for Plex media items (movies and episodes).

    This model represents the core data structure for media items stored in
    the knowledge base, containing all relevant metadata for movies and TV episodes.
    """

    key: int
    title: str
    summary: str
    year: int
    rating: float
    watched: bool
    type: MediaType
    genres: list[str]
    actors: list[str]
    studio: str
    directors: list[str]
    writers: list[str]
    duration_seconds: int
    content_rating: Optional[str]
    show_title: Optional[str]
    season: Optional[int]
    episode: Optional[int]
    air_date: Optional[date]
    producers: Optional[list[str]] = None
    reviews: Optional[list[Review]] = None
    ratings: Optional[list[Rating]] = None

    @classmethod
    def document(cls, item: "PlexMediaPayload") -> str:
        """Convert a media item to a searchable text document.

        Args:
            item: Media item to convert to document text

        Returns:
            str: Formatted text representation for search indexing
        """
        parts = []
        if item.title:
            parts.append("Title: " + item.title)
        if item.summary:
            parts.append("Summary: " + item.summary)
        if item.reviews:
            parts.append(
                "Reviews: " + " ".join(review.text for review in item.reviews))
        if item.genres:
            parts.append("Genres: " + ", ".join(item.genres))
        if item.actors:
            parts.append("Actors: " + ", ".join(item.actors))
        if item.directors:
            parts.append("Directed by: " + ", ".join(item.directors))
        if item.writers:
            parts.append("Written by: " + ", ".join(item.writers))
        if item.type == "episode" and item.show_title:
            parts.append(f"Show: {item.show_title}")
            season_episode = []
            if item.season is not None:
                season_episode.append(f"Season {item.season}")
            if item.episode is not None:
                season_episode.append(f"Episode {item.episode}")
            if season_episode:
                parts.append(" ".join(season_episode))
        if item.rating is not None:
            parts.append(f"Rating: {item.rating}/10")
        return "\n".join(parts).lower()


class PlexMediaQuery(PlexMediaPayload):
    """Query model for searching Plex media with optional fields.

    This model extends PlexMediaPayload but makes all fields optional,
    allowing for flexible search queries where any combination of fields
    can be specified.
    """

    key: Optional[int] = None  # type: ignore
    title: Optional[str] = None  # type: ignore
    summary: Optional[str] = None  # type: ignore
    year: Optional[int] = None  # type: ignore
    rating: Optional[float] = None  # type: ignore
    watched: Optional[bool] = None  # type: ignore
    type: Optional[MediaType] = None  # type: ignore
    genres: Optional[list[str]] = None  # type: ignore
    actors: Optional[list[str]] = None  # type: ignore
    studio: Optional[str] = None  # type: ignore
    directors: Optional[list[str]] = None  # type: ignore
    writers: Optional[list[str]] = None  # type: ignore
    duration_seconds: Optional[int] = None  # type: ignore
    content_rating: Optional[str] = None  # type: ignore
    show_title: Optional[str] = None  # type: ignore
    season: Optional[int] = None  # type: ignore
    episode: Optional[int] = None  # type: ignore
    air_date: Optional[date] = None  # type: ignore
    similar_to: Optional[int] = None  # type: ignore
    roducers: Optional[list[str]] = None
    reviews: Optional[list[Review]] = None
    ratings: Optional[list[Rating]] = None


TModel = TypeVar("TModel", bound=PlexMediaPayload)


class DataPoint(ScoredPoint, Generic[TModel]):
    """Enhanced ScoredPoint with typed payload access for search results.

    This class extends Qdrant's ScoredPoint to provide type-safe access to
    the payload data using the specified model class.
    """

    payload_class: Type[TModel]

    def payload_data(self) -> TModel:
        """Get the typed payload data from this point.

        Returns:
            TModel: Validated payload data as the specified model type

        Raises:
            ValueError: If no payload is available
        """
        if not self.payload:
            raise ValueError("No payload available")
        return self.payload_class.model_validate(self.payload)


class Scope(StrEnum):
    MOVIE = "movies"
    # SERIES = "series"
    # SEASON = "season"
    EPISODE = "episodes"


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
