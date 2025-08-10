from datetime import date
from typing import Generic, Literal, Optional, Type, TypeVar

from pydantic import BaseModel
from qdrant_client.models import ScoredPoint

MediaType = Literal["movie"] | Literal["episode"]


class Review(BaseModel):
    key: str
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
