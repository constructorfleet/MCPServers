from dataclasses import dataclass
from enum import StrEnum
import os
from typing import Any, Dict, Optional
import logging

from plexapi.exceptions import Unauthorized
from plexapi.server import PlexServer

logger = logging.getLogger(__name__)

class MediaType(StrEnum):
    ACTOR = 'actor'
    ALBUM = 'album'
    ARTIST = 'artist'
    AUTO_TAG = 'autotag'
    COLLECTION = 'collection'
    DIRECTOR = 'director'
    EPISODE = 'episode'
    GAME = 'game'
    GENRE = 'genre'
    MOVIE = 'movie'
    PHOTO = 'photo'
    PHOTO_ALBUM = 'photoalbum'
    PLACE = 'place'
    PLAYLIST = 'playlist'
    SHARED = 'shared'
    SHOW = 'show'
    TAG = 'tag'
    TRACK = 'track'


@dataclass
class ShowSearchParams:
    show_title: Optional[str] = None
    show_year: Optional[int] = None
    director: Optional[str] = None
    studio: Optional[str] = None
    genre: Optional[str] = None
    actor: Optional[str] = None
    rating: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    watched: Optional[bool] = None  # True=only watched, False=only unwatched
    season: Optional[int] = None
    episode: Optional[int] = None
    episode_title: Optional[str] = None
    episode_year: Optional[int] = None

    def to_filters(self) -> Dict[str, Any]:
        FIELD_MAP = {
            "show_title": "show.title",
            "show_year": "show.year",
            "director": "show.director",
            "studio": "episode.studio",
            "genre": "episode.genre",
            "actor": "show.actor",
            "rating": "show.contentRating",
            "country": "show.country",
            "language": "episode.audioLanguage",
            "watched": "episode.unwatched",
            "season": "season.index",
            "episode": "episode.index",
            "episode_title": "episode.title",
            "episode_year": "episode.year",
        }

        params: Dict[str, Any] = {"libtype": "episode"}
        filters: Dict[str, Any] = {}
        for field_name, plex_arg in FIELD_MAP.items():
            value = getattr(self, field_name)
            if value is None:
                continue

            if field_name == "watched":
                # invert for Plex 'unwatched' flag
                filters["unwatched"] = not value
                continue

            filters[plex_arg] = value
        params["filters"] = filters
        return params


@dataclass
class MovieSearchParams:
    title: Optional[str] = None
    year: Optional[int] = None
    director: Optional[str] = None
    studio: Optional[str] = None
    genre: Optional[str] = None
    actor: Optional[str] = None
    rating: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    watched: Optional[bool] = None  # True=only watched, False=only unwatched
    summary: Optional[str] = None

    def to_filters(self) -> Dict[str, Any]:
        FIELD_MAP = {
            "title": "title",
            "year": "year",
            "director": "directors",
            "studio": "studio",
            "genre": "genres",
            "actor": "actors",
            "rating": "contentRating",
            "country": "countries",
            "language": "audioLanguage",
            "watched": "unwatched",
            "summary": "summary",
        }

        filters: Dict[str, Any] = {"libtype": "movie"}

        for field_name, plex_arg in FIELD_MAP.items():
            value = getattr(self, field_name)
            if value is None:
                continue

            if field_name == "watched":
                # invert for Plex 'unwatched' flag
                filters["unwatched"] = not value
                continue

            filters[plex_arg] = value

        return filters


class PlexClient:
    """
    Encapsulate the Plex connection logic.
    This class handles initialization and caching of the PlexServer instance.
    """

    def __init__(self, server_url: str | None = None, token: str | None = None):
        self.server_url = server_url or os.environ.get("PLEX_SERVER_URL", "").rstrip(
            "/"
        )
        self.token = token or os.environ.get("PLEX_TOKEN")

        if not self.server_url or not self.token:
            raise ValueError(
                "Missing required configuration: Ensure PLEX_SERVER_URL and PLEX_TOKEN are set."
            )

        self._server = None

    def get_server(self) -> PlexServer:
        """
        Return a cached PlexServer instance or initialize one if not already available.

        Returns:
            A connected PlexServer instance.

        Raises:
            Exception: If connection initialization fails.
        """
        if self._server is None:
            try:
                logger.info("Initializing PlexServer with URL: %s", self.server_url)
                self._server = PlexServer(self.server_url, self.token)
                logger.info("Successfully initialized PlexServer.")

                # Validate the connection
                self._server.library.sections()  # Attempt to fetch library sections
                logger.info("Plex server connection validated.")
            except Unauthorized as exc:
                logger.error("Unauthorized: Invalid Plex token provided.")
                raise Exception("Unauthorized: Invalid Plex token provided.") from exc
            except Exception as exc:
                logger.exception("Error initializing Plex server: %s", exc)
                raise Exception(f"Error initializing Plex server: {exc}") from exc
        return self._server