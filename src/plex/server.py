"""
Module: plex_mcp

This module provides tools for interacting with a Plex server via FastMCP.
It includes functions to search for movies, retrieve movie details, manage playlists,
and obtain recent movies and movie genres. Logging and asynchronous execution are used
to handle non-blocking I/O and to provide informative error messages.
"""
import argparse
import asyncio
import datetime
import itertools
import json
from enum import StrEnum
import logging

import os
from dataclasses import dataclass

# --- Import Statements ---
from typing import Annotated, Any, Dict, List, Optional, Callable
from base import run_server, mcp
from starlette.requests import Request
from starlette.responses import Response
from mcp.types import ToolAnnotations
from plex.utils import from_json, initialize_pickle, to_json
from plexapi.base import PlexSession as PlexAPISession
from plexapi.exceptions import NotFound, Unauthorized
from plexapi.library import MovieSection, ShowSection
from plexapi.client import PlexClient as PlexAPIClient
from plexapi.server import PlexServer
from plexapi.video import Movie, Episode
from plexapi.utils import toJson
from pydantic import Field
from rapidfuzz import process
from rapidfuzz.fuzz import token_set_ratio
import aiofiles
try:
    import http.client as http_client
except ImportError:
    # Python 2
    import httplib as http_client
http_client.HTTPConnection.debuglevel = 1

# You must initialize logging, otherwise you'll not see debug output.
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for more verbosity during development
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

# --- Logging Setup ---
logger = logging.getLogger(__name__)
media_cache: dict[str, dict] = {}

def recursive_get(s, key):
    if "." in key:
        first, rest = key.split(".", 1)
        return recursive_get(s.get(first, {}), rest)
    else:
        return s.get(key, None)

def object_similarity_score(obj, filter_dict):
    score = 0
    for key, value in filter_dict.items():
        obj_value = recursive_get(obj, key)
        if obj_value is None:
            continue
        score += token_set_ratio(str(obj_value), str(value))
    return score

def sort_plex_objects_by_similarity(objects, filter_dict):
    return sorted(
        objects,
        key=lambda obj: object_similarity_score(obj, filter_dict),
        reverse=True
    )

def cache_result(key: str, value: dict) -> None:
    media_cache[key] = value


def get_cached_result(key: str) -> Optional[dict]:
    return media_cache.get(key, None)

# --- Utility Formatting Functions ---
def default_filter(client: PlexAPIClient) -> bool:
    return True

def match_client_name(requested_client: str, candidates: list[PlexAPIClient], filter: Callable[[PlexAPIClient], bool] = default_filter) -> Optional[PlexAPIClient]:
    """
    Try to match the requested client to a candidate
    """
    if not requested_client or not candidates or not filter:
        return None
    candidate_map = {c.title: c for c in candidates if filter(c)}
    choice, score, _ = process.extractOne(requested_client, list(candidate_map.keys()), score_cutoff=60)
    return candidate_map.get(choice, None) if score >= 60 else None

def format_movie(movie) -> str:
    """
    Format a movie object into a human-readable string.

    Parameters:
        movie: A Plex movie object.

    Returns:
        A formatted string containing movie details.
    """
    title = getattr(movie, "title", "Unknown Title")
    year = getattr(movie, "year", "Unknown Year")
    summary = getattr(movie, "summary", "No summary available")
    duration = (
        getattr(movie, "duration", 0) // 60000 if hasattr(movie, "duration") else 0
    )
    rating = getattr(movie, "rating", "Unrated")
    studio = getattr(movie, "studio", "Unknown Studio")
    directors = [director.tag for director in getattr(movie, "directors", [])[:3]]
    actors = [role.tag for role in getattr(movie, "roles", [])[:5]]

    return (
        f"Title: {title} ({year})\n"
        f"Rating: {rating}\n"
        f"Duration: {duration} minutes\n"
        f"Studio: {studio}\n"
        f"Directors: {', '.join(directors) if directors else 'Unknown'}\n"
        f"Starring: {', '.join(actors) if actors else 'Unknown'}\n"
        f"Summary: {summary}\n"
    )


def format_episode(episode) -> str:
    """
    Format an episode object into a human-readable string.

    Parameters:
        episode: A Plex episode object.

    Returns:
        A formatted string containing episode details.
    """
    show_title = getattr(episode, "grandparentTitle", "Unknown Show")
    season_number = getattr(episode, "parentIndex", "Unknown Season")
    episode_number = getattr(episode, "index", "Unknown Episode")
    title = getattr(episode, "title", "Unknown Title")
    summary = getattr(episode, "summary", "No summary available")
    duration = (
        getattr(episode, "duration", 0) // 60000 if hasattr(episode, "duration") else 0
    )
    rating = getattr(episode, "rating", "Unrated")
    studio = getattr(episode, "studio", "Unknown Studio")
    directors = [director.tag for director in getattr(episode, "directors", [])[:3]]
    actors = [role.tag for role in getattr(episode, "roles", [])[:5]]
    year = getattr(episode, "year", "Unknown Year")

    return (
        f"Show: {show_title}\n"
        f"Season: {season_number}, Episode: {episode_number}\n"
        f"Year: {year}\n"
        f"Title: {title} ({year})\n"
        f"Rating: {rating}\n"
        f"Duration: {duration} minutes\n"
        f"Studio: {studio}\n"
        f"Directors: {', '.join(directors) if directors else 'Unknown'}\n"
        f"Starring: {', '.join(actors) if actors else 'Unknown'}\n"
        f"Summary: {summary}\n"
    )

def format_session(session: PlexAPISession) -> str:
    """
    Format a Plex session object into a human-readable string.
    Parameters:
        session: A Plex session object
    Returns:
        A formatted string containing session details.
    """
    source = session.source()
    logger.error(json.dumps(toJson(source), indent=2))
    return (
        f"User: {session.user.username}\n"
        f"Media: {format_movie(source) if 'grandFatherTitle' not in source else format_episode(source)}\n"
    )


def format_client(client: PlexAPIClient) -> str:
    """
    Format a Plex client object into a human-readable string.
    Parameters:
        client: A Plex client object.
    Returns:
        A formatted string containing client details.
    """
    return (
        f"Client: {client.title}\n"
        f"Platform: {client.platform}\n"
        f"Product: {client.product}\n"
        f"Version: {client.version}\n"
        f"Device: {client.device}\n"
        f"State: {client.state}\n"
        f"address: {client.address}\n"
    )


def format_playlist(playlist) -> str:
    """
    Format a playlist into a human-readable string.

    Parameters:
        playlist: A Plex playlist object.

    Returns:
        A formatted string containing playlist details.
    """
    duration_mins = (
        sum(item.duration for item in playlist.items()) // 60000
        if playlist.items()
        else 0
    )
    updated = (
        playlist.updatedAt.strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(playlist, "updatedAt")
        else "Unknown"
    )
    return (
        f"Playlist: {playlist.title}\n"
        f"Items: {len(playlist.items())}\n"
        f"Duration: {duration_mins} minutes\n"
        f"Last Updated: {updated}\n"
    )


def movie_section(server: PlexServer) -> Optional[MovieSection]:
    """Get the first movie section from the Plex server."""
    return next(
        (
            section
            for section in server.library.sections()
            if isinstance(section, MovieSection)
        ),
        None,
    )


def show_section(server: PlexServer) -> Optional[ShowSection]:
    """Get the first show section from the Plex server."""
    return next(
        (
            section
            for section in server.library.sections()
            if isinstance(section, ShowSection)
        ),
        None,
    )


# --- Plex Client Class ---


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


# --- Data Classes ---


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

    def to_filters(self) -> Dict[str, Any]:
        FIELD_MAP = {
            "title": "title",
            "year": "year",
            "director": "director",
            "studio": "studio",
            "genre": "genre",
            "actor": "actor",
            "rating": "contentRating",
            "country": "country",
            "language": "audioLanguage",
            "watched": "unwatched",
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


# --- Global Singleton and Access Functions ---

_plex_client_instance: PlexClient | None = None


def get_plex_client() -> PlexClient:
    """
    Return the singleton PlexClient instance, initializing it if necessary.

    Returns:
        A PlexClient instance.
    """
    global _plex_client_instance
    if _plex_client_instance is None:
        _plex_client_instance = PlexClient()
    return _plex_client_instance


async def get_plex_server() -> PlexServer:
    """
    Asynchronously get a PlexServer instance via the singleton PlexClient.

    Returns:
        A PlexServer instance.

    Raises:
        Exception: When the Plex server connection fails.
    """
    try:
        plex_client = get_plex_client()  # Singleton accessor
        plex = await asyncio.to_thread(plex_client.get_server)
        return plex
    except Exception as e:
        logger.exception("Failed to get Plex server instance")
        raise e

@mcp.custom_route("/health", ["GET"], "health", False)
async def health(request: Request) -> Response:
    try:
        await get_plex_server()
        return Response("OK", status_code=200, media_type="text/plain")
    except Exception:
        return Response("ERROR", status_code=500, media_type="text/plain")


# --- Tool Methods ---
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

@mcp.tool(
    name="search_media",
    description="'Smart' search for movies, shows, episodes, and more across the entire Plex library.",
    annotations=ToolAnnotations(
        title="Media Smart Search",
    )
)
async def search_media(
    query: Annotated[
        str,
        Field(
            description="Media search query",
            examples=["Terminator", "Schwartzengger", "Horror"]
        )
    ],
    media_type: Annotated[
        Optional[str],
        Field(
            description="The optional type of media to query",
            examples=[m.value for m in MediaType],
            default=None
        )
    ],
    # limit: Annotated[
    #     Optional[int],
    #     Field(
    #         description="Optionally limit the number of items returned.",
    #         default=None,
    #         examples=[None,1,10,5],
    #     )
    # ] = None,
) -> str:
    """
    Perform a smart media search.

    parameters:
      query: Media search query
      limit: The number of items to return
    returns:
      A formatted string or error message
    """
    logger.info("Searching Plex with query: %s", query)

    try:
        plex = await get_plex_server()
        media = await asyncio.to_thread(plex.search, query, mediatype=media_type)
    except Exception as e:
        logger.exception("smart_search failed connecting to Plex")
        return f"ERROR: Could not search Plex. {e}"

    if not media:
        return f"No results found matching query {query}."

    logger.info("Found %d results matching the query: %s", len(media), query)

    results: List[str] = []
    # limit = max(1, limit) if limit else len(media)  # Default to len(media) if limit is 0 or negative
    limit = len(media)
    for i, m in enumerate(media, start=1):
        cache_result(m.ratingKey, m.title)
        if len(results) > limit:
            break
        if not isinstance(m, Movie) and not isinstance(m, Episode):
            continue
        results.append(
            f"Result #{i}: {m.title} ({m.year})\nKey: {m.ratingKey}\n"
        )  # type: ignore

    if len(media) > limit:
        results.append(f"\n... and {len(media)-limit} more results.")
    logger.info("Returning %s.", "\n---\n".join(results))
    return "\n---\n".join(results)


@mcp.tool(
    name="search_movies",
    description="Search for movies in your Plex library using various filters.",
    annotations=ToolAnnotations(
        title="Search For Movies",
    ),
)
async def search_movies(
    title: Annotated[
        Optional[str],
        Field(
            description="Title or substring to match",
            examples=["Inception", "Terminator", "Star Wars"],
            default=None,
        ),
    ] = None,
    year: Annotated[
        Optional[int],
        Field(
            description="Release year to filter by",
            default=None,
            examples=[1994, 2020, 1985],
        ),
    ] = None,
    director: Annotated[
        Optional[str],
        Field(
            description="Director name to filter by",
            default=None,
            examples=["Christopher Nolan", "James Cameron", "George Lucas"],
        ),
    ] = None,
    studio: Annotated[
        Optional[str],
        Field(
            description="Studio name to filter by",
            default=None,
            examples=["Warner Bros.", "20th Century Fox", "Universal Pictures"],
        ),
    ] = None,
    genre: Annotated[
        Optional[str],
        Field(
            description="Genre tag to filter by",
            default=None,
            examples=["Action", "Sci-Fi", "Drama"],
        ),
    ] = None,
    actor: Annotated[
        Optional[str],
        Field(
            description="Actor name to filter by",
            default=None,
            examples=["Leonardo DiCaprio", "Arnold Schwarzenegger", "Harrison Ford"],
        ),
    ] = None,
    rating: Annotated[
        Optional[str],
        Field(
            description="Rating to filter by (e.g., 'PG-13', 'R')",
            default=None,
            examples=["PG-13", "R", "G"],
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        Field(
            description="Country of origin to filter by",
            default=None,
            examples=["USA", "UK", "France"],
        ),
    ] = None,
    language: Annotated[
        Optional[str],
        Field(
            description="Audio or subtitle language to filter by",
            default=None,
            examples=["English", "Spanish", "French"],
        ),
    ] = None,
    watched: Annotated[
        Optional[bool],
        Field(
            description="Filter by watched status; True for watched, False for unwatched",
            default=None,
            examples=[True, False],
        ),
    ] = None,
    # limit: Annotated[
    #     Optional[int],
    #     Field(
    #         description="Optionally limit the number of items returned.",
    #         default=None,
    #         examples=[None,1, 5, 10],
    #     )
    # ] = None,
) -> str:
    """
    Search for movies in your Plex library using optional filters.

    Parameters:
        title: Optional title or substring to match.
        year: Optional release year to filter by.
        director: Optional director name to filter by.
        studio: Optional studio name to filter by.
        genre: Optional genre tag to filter by.
        actor: Optional actor name to filter by.
        rating: Optional rating (e.g., "PG-13") to filter by.
        country: Optional country of origin to filter by.
        language: Optional audio or subtitle language to filter by.
        watched: Optional boolean; True returns only watched movies, False only unwatched.
        min_duration: Optional minimum duration in minutes.
        max_duration: Optional maximum duration in minutes.

    Returns:
        A formatted string of up to 5 matching movies (with a count of any additional results),
        or an error message if the search fails or no movies are found.
    """

    params = MovieSearchParams(
        title,
        year,
        director,
        studio,
        genre,
        actor,
        rating,
        country,
        language,
        watched,
    )
    filters = params.to_filters()
    logger.info("Searching Plex with filters: %r", filters)

    try:
        plex = await get_plex_server()
        library_section = movie_section(plex)
        if not library_section:
            return "ERROR: No movie section found in your Plex library."
        movies = await asyncio.to_thread(library_section.search, **filters)
    except Exception as e:
        logger.exception("search_movies failed connecting to Plex")
        return f"ERROR: Could not search Plex. {e}"

    if not movies:
        return f"No movies found matching filters {filters!r}."

    logger.info("Found %d movies matching filters: %r", len(movies), filters)

    results: List[str] = []
    # Validate the limit parameter
    # limit = max(1, limit) if limit else len(movies)  # Default to 5 if limit is 0 or negative
    limit = len(movies)
    for i, m in enumerate(movies[:limit], start=1):
        cache_result(m.ratingKey, m.title)
        results.append(f"Result #{i}: {m.title} ({m.year})\nKey: {m.ratingKey}\n")  # type: ignore
        # results.append(f"Result #{i}:\nKey: {m.ratingKey}\n{format_movie(m)}")  # type: ignore

    # if limit and len(movies) > limit:
    #     results.append(f"\n... and {len(movies)-limit} more results.")
    logger.info("Returning %s.", "\n---\n".join(results))
    return "\n---\n".join(results)


@mcp.tool(
    name="get_movie_details",
    description="Get detailed information about a specific movie by its key.",
    annotations=ToolAnnotations(
        title="Get Movie Details",
    ),
)
async def get_movie_details(
    movie_key: Annotated[
        str,
        Field(
            description="The key identifying the movie to retrieve details for.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Get detailed information about a specific movie.

    Parameters:
        movie_key: The key identifying the movie.

    Returns:
        A formatted string with movie details or an error message.
    """
    if result := get_cached_result(movie_key):
        return result
    logger.info("Fetching movie details for key '%s'", movie_key)
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(movie_key)
        library_section = movie_section(plex)
        if not library_section:
            return "ERROR: No movie section found in your Plex library."
        movie = await asyncio.to_thread(library_section.fetchItem, key)  # type: ignore

        if not movie:
            return f"No movie found with key {movie_key}."
        logger.info("Returning %s", format_movie(movie))
        return format_movie(movie)
    except NotFound:
        return f"ERROR: Movie with key {movie_key} not found."
    except Exception as e:
        logger.exception("Failed to fetch movie details for key '%s'", movie_key)
        return f"ERROR: Failed to fetch movie details. {str(e)}"


@mcp.tool(
    name="get_new_movies",
    description="Get a list of recently added movies in your Plex library.",
    annotations=ToolAnnotations(
        title="Get New Movies",
    ),
)
async def get_new_movies() -> str:
    """
    Get list of recently added movies in your Plex library.

    Returns:
        A formatted string with the new movie details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        library_section = movie_section(plex)
        if not library_section:
            return "ERROR: No movie section found in your Plex library."
        movies = await asyncio.to_thread(library_section.recentlyAdded, 10)  # type: ignore

        if not movies:
            return "No new movies found in your Plex library."
        results: List[str] = []
        for i, m in enumerate(movies[:10], start=1):
            cache_result(m.ratingKey, m.title)
            results.append(f"Result #{i}:\nKey: {m.ratingKey}\n{format_movie(m)}")  # type: ignore
        logger.info(("Returning %s new movies.", "\n---\n".join(results)))
        return "\n---\n".join(results)
    except Exception as e:
        logger.exception("Failed to fetch new movie list.")
        return f"ERROR: Failed to fetch new movie list. {str(e)}"


@mcp.tool(
    name="search_shows",
    description="Search for shows or episodes in your Plex library using various filters.",
    annotations=ToolAnnotations(
        title="Search For Shows or Episodes",
    ),
)
async def search_shows(
    show_title: Annotated[
        Optional[str],
        Field(
            description="Title or substring of the show to match",
            examples=["Family Guy", "South Park", "3 body problem"],
            default=None,
        ),
    ] = None,
    show_year: Annotated[
        Optional[int],
        Field(
            description="Show release year to filter by",
            default=None,
            examples=[1994, 2020, 1985],
        ),
    ] = None,
    director: Annotated[
        Optional[str],
        Field(
            description="Director name to filter by",
            default=None,
            examples=["Christopher Nolan", "James Cameron", "George Lucas"],
        ),
    ] = None,
    studio: Annotated[
        Optional[str],
        Field(
            description="Studio name to filter by",
            default=None,
            examples=["Warner Bros.", "20th Century Fox", "Universal Pictures"],
        ),
    ] = None,
    genre: Annotated[
        Optional[str],
        Field(
            description="Genre tag to filter by",
            default=None,
            examples=["Action", "Sci-Fi", "Drama"],
        ),
    ] = None,
    actor: Annotated[
        Optional[str],
        Field(
            description="Actor name to filter by",
            default=None,
            examples=["Leonardo DiCaprio", "Arnold Schwarzenegger", "Harrison Ford"],
        ),
    ] = None,
    rating: Annotated[
        Optional[str],
        Field(
            description="Rating to filter by (e.g., 'PG-13', 'R')",
            default=None,
            examples=["PG-13", "R", "G"],
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        Field(
            description="Country of origin to filter by",
            default=None,
            examples=["USA", "UK", "France"],
        ),
    ] = None,
    language: Annotated[
        Optional[str],
        Field(
            description="Audio or subtitle language to filter by",
            default=None,
            examples=["English", "Spanish", "French"],
        ),
    ] = None,
    watched: Annotated[
        Optional[bool],
        Field(
            description="Filter by watched status; True for watched, False for unwatched",
            default=None,
            examples=[True, False],
        ),
    ] = None,
    season: Annotated[
        Optional[int],
        Field(
            description="Season number to filter by",
            default=None,
            examples=[1, 2, 3],
        ),
    ] = None,
    episode: Annotated[
        Optional[int],
        Field(
            description="Episode number to filter by",
            default=None,
            examples=[1, 2, 3],
        ),
    ] = None,
    episode_title: Annotated[
        Optional[str],
        Field(
            description="Episode title or substring to match",
            default=None,
            examples=["Pilot", "The End"],
        ),
    ] = None,
    episode_year: Annotated[
        Optional[int],
        Field(
            description="Episode release year to filter by",
            default=None,
            examples=[2020, 2021, 2022],
        ),
    ] = None,
    # limit: Annotated[
    #     Optional[int],
    #     Field(
    #         description="Optionally limit the number of items returned.",
    #         default=None,
    #         examples=[None, 1, 5, 10],
    #     )
    # ] = None,
) -> str:
    """
    Search for movies in your Plex library using optional filters.

    Parameters:
        show_title: Optional show title or substring to match.
        show_year: Optional show release year to filter by.
        director: Optional director name to filter by.
        studio: Optional studio name to filter by.
        genre: Optional genre tag to filter by.
        actor: Optional actor name to filter by.
        rating: Optional rating (e.g., "PG-13") to filter by.
        country: Optional country of origin to filter by.
        language: Optional audio or subtitle language to filter by.
        watched: Optional boolean; True returns only watched movies, False only unwatched.
        season: Optional season number to filter by.
        episode: Optional episode number to filter by.
        episode_title: Optional episode title or substring to match.
        episode_year: Optional episode release year to filter by.


    Returns:
        A formatted string of up to 5 matching shows (with a count of any additional results),
        or an error message if the search fails or no movies are found.
    """

    params = ShowSearchParams(
        show_title,
        show_year,
        director,
        studio,
        genre,
        actor,
        rating,
        country,
        language,
        watched,
        season,
        episode,
        episode_title,
        episode_year,
    )
    filters = params.to_filters()
    logger.info("Searching Plex with filters: %r", filters)

    try:
        plex = await get_plex_server()
        library_section = show_section(plex)
        if not library_section:
            return "ERROR: No show section found in your Plex library."
        episodes = await asyncio.to_thread(library_section.search, **filters)
    except Exception as e:
        logger.exception("search_shows failed connecting to Plex")
        return f"ERROR: Could not search Plex. {e}"

    if not episodes:
        logger.info("No shows found matching filters: %r", filters)
        return f"No shows found matching filters {filters!r}."

    logger.info("Found %d shows matching filters: %r", len(episodes), filters)
    # Validate the limit parameter
    # limit = max(1, limit) if limit else len(episodes)  # Default to 5 if limit is 0 or negative
    limit = len(episodes)
    results: List[str] = []
    for i, m in enumerate(episodes[:limit], start=1):
        cache_result(m.ratingKey, m)
        results.append(f"Result #{i}:\nKey: {m.ratingKey}\n{format_episode(m)}")  # type: ignore

    if len(episodes) > limit:
        results.append(f"\n... and {len(episodes)-limit} more results.")

    return "\n---\n".join(results)


@mcp.tool(
    name="get_episode_details",
    description="Get detailed information about a specific episode identified by its key.",
    annotations=ToolAnnotations(
        title="Get Movie Details",
    ),
)
async def get_episode_details(
    episode_key: Annotated[
        str,
        Field(
            description="The key identifying the episode to retrieve details for.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Get detailed information about a specific episode.

    Parameters:
        episode_key (str): The key identifying the episode.

    Returns:
        A formatted string with episode details or an error message.
    """
    if result := get_cached_result(episode_key):
        return result
    logger.info("Fetching episode details for key '%s'", episode_key)
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(episode_key)
        library_section = show_section(plex)
        if not library_section:
            return "ERROR: No show section found in your Plex library."
        episode = await asyncio.to_thread(library_section.fetchItem, f"/library/metadata/{key}")  # type: ignore

        if not episode:
            return f"No episode found with key {episode_key}."
        logger.info("Found episode: %s", episode.title)
        return format_episode(episode)  # type: ignore
    except NotFound:
        return f"ERROR: Episode with key {episode_key} not found."
    except Exception as e:
        logger.exception("Failed to fetch episode details for key '%s'", episode_key)
        return f"ERROR: Failed to fetch episode details. {str(e)}"


@mcp.tool(
    name="get_new_shows",
    description="Get a list of recently added episodes in your Plex library.",
    annotations=ToolAnnotations(
        title="Get New Shows",
    ),
)
async def get_new_shows() -> str:
    """
    Get list of recently added episodes in your Plex library.

    Returns:
        A formatted string with the new episodes details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        library_section = show_section(plex)
        if not library_section:
            return "ERROR: No show section found in your Plex library."
        episodes = await asyncio.to_thread(library_section.recentlyAdded, 10)  # type: ignore
        logger.info("Found %d new episodes.", len(episodes))
        if not episodes:
            return "No new episodes found in your Plex library."
        results: List[str] = []
        for i, m in enumerate(episodes[:10], start=1):
            cache_result(m.ratingKey, m.title)
            results.append(f"Result #{i}:\nKey: {m.ratingKey}\n{format_episode(m)}")  # type: ignore
        logger.info("Returning %s new episodes.", "\n---\n".join(results))
        return "\n---\n".join(results)
    except Exception as e:
        logger.exception("Failed to fetch new episode list.")
        return f"ERROR: Failed to fetch new episode list. {str(e)}"


@mcp.tool(
    name="get_active_clients",
    description="Get a list of active clients on your Plex server.",
    annotations=ToolAnnotations(
        title="Get Active Clients",
    ),
)
async def get_active_clients(
    controllable: Annotated[
        Optional[bool],
        Field(
            description="If True, only return clients that can be controlled.",
            default=True,
            examples=[True, False],
        ),
    ] = True,
) -> str:
    """
    Get list active clients on your Plex server.

    Parameters:
        controllable: If True, only return clients that can be controlled.

    Returns:
        A formatted string with the clients details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        clients, sessions = await asyncio.gather(*[asyncio.to_thread(plex.clients), asyncio.to_thread(plex.sessions)])

        if not clients and not sessions:
            return "No active clients connected to your Plex server."
        logger.info("Found %d active clients and %d sessions.", len(clients), len(sessions))
        results: List[str] = []
        for i, m in enumerate(clients):
            logger.info(f"Client {m.title} {m.protocolCapabilities}")
            if controllable and "playback" not in m.protocolCapabilities: # type: ignore
                continue
            session = [s for _, s in enumerate(sessions)]
            results.append(
                f"Result #{i}:\nMachine Identifier: {m.machineIdentifier}\n{format_client(m)}\n{format_session(session[0]) if len(session) > 0 else ''}"
            )  # type: ignore
        logger.info("Returning %s.", "\n---\n".join(results))
        return "\n---\n".join(results)
    except Exception as e:
        logger.exception("Failed to fetch client list.")
        return f"ERROR: Failed to fetch client list. {str(e)}"


@mcp.tool(
    name="play_media_on_client",
    description="Play specified media on a given Plex client.",
    annotations=ToolAnnotations(
        title="Play Media on Client",
    ),
)
async def play_media_on_client(
    machine_identifier_or_client_name: Annotated[
        str,
        Field(
            description="Either the machine identifier or the name of the of the Plex client. Find this by calling the get_client_machine_identifier tool.",
            examples=["abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx"],
        ),
    ],
    media_key: Annotated[
        int,
        Field(
            description="The key of the media item to play.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Play specified media on a given Plex client.
    Parameters:
        machine_identifier: The machine identifier of the Plex client.
        media_key: The key of the media item to play.
    Returns:
        A success message or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        clients: list[PlexAPIClient] = await asyncio.to_thread(plex.clients)

        if not clients:
            return "No active clients connected to your Plex server."
        logger.info("Found %d active clients.", len(clients))
        if len([c for c in clients if c.machineIdentifier == machine_identifier_or_client_name or c.title == machine_identifier_or_client_name]) > 0:
            client = [c for c in clients if c.machineIdentifier == machine_identifier_or_client_name or c.title == machine_identifier_or_client_name][0]
        else:
            client = match_client_name(machine_identifier_or_client_name, clients, filter=lambda c: "playback" in c.protocolCapabilities)
        if not client:
            return f"No client found with machine identifier {machine_identifier}."
        if "playback" not in client.protocolCapabilities:
            return f"Client {client.title} does not support playback control."
        logger.info("Found client: %s with media key: %s", client.title, media_key)
        media = plex.fetchItem(f"/library/metadata/{media_key}")
        if not media:
            return f"No media found with key {media_key}."
        logger.info("Playing media: %s on client: %s", media.title, client.title)
        await asyncio.to_thread(client.playMedia, media)
        return f"Playing {media.title} on {client.title}." # type: ignore
    except Exception as e:
        logger.exception("Failed to play media on client.")
        return f"ERROR: Failed to play media on client. {str(e)}"


class MediaCommand(StrEnum):
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    RESUME = "resume"
    FAST_FORWARD = "fast forward"
    REWIND = "rewind"
    NEXT = "next"
    PREVIOUS = "previous"
    SKIP = "skip"
    START_OVER = "start over"
    SEEK = "seek"


@mcp.tool(
        name="control_client_playback",
        description="Control playback on a specified Plex client (play, pause, stop).",
        annotations=ToolAnnotations(
            title="Control Client Playback",
        ),)
async def control_client_playback(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client find this by calling the get_client_machine_identifier tool.",
            examples=["1234567890abcdef", "abcdef1234567890"],
        ),
    ],
    command: Annotated[
        str,
        Field(
            description="The playback command to send (play, pause, stop).",
            examples=["play", "resume", "pause", "stop", "fast forward", "rewind", "next", "previous", "seek", "skip", "start over"],
        ),
    ],
    seek_position: Annotated[
        Optional[int],
        Field(
            description="The position in seconds to seek to (required if command is 'seek').",
            examples=[60, 120, 300],
            default=None,
        ),
    ] = None,
) -> str:
    """
    Control playback on a specified Plex client (play, pause, stop).
    Parameters
        machine_identifier: The machine identifier of the Plex client.
        command: The playback command to send (play, pause, stop).
        seek_position: The position in seconds to seek to (required if command is 'seek').
        duration: The duration in seconds to fast forward or rewind (required if command is 'fastForward' or 'rewind').
    Returns:
        A success message or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"
    try:
        client: PlexAPIClient = plex.client(machine_identifier)
    except NotFound:
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        return f"ERROR: Could not retrieve client. {str(e)}"

    if "playback" not in client.protocolCapabilities:
        logger.info("Client '%s' does not support playback control.", client.title)
        return f"ERROR: Client '{client.title}' does not support playback control."

    try:
        command_enum = MediaCommand(command)
        logger.info("Sending command '%s' to client '%s'.", command_enum, client.title)
        if command_enum == MediaCommand.SEEK and seek_position is not None:
            client.seekTo(seek_position * 1000)
        if command_enum == MediaCommand.START_OVER:
            client.seekTo(0)
        elif command_enum == MediaCommand.FAST_FORWARD:
            client.stepForward()
        elif command_enum == MediaCommand.REWIND:
            client.stepBack()
        elif command_enum == MediaCommand.NEXT or command_enum.SKIP:
            client.skipNext()
        elif command_enum == MediaCommand.PREVIOUS:
            client.skipPrevious()
        elif command_enum == MediaCommand.STOP:
            client.stop()
        elif command_enum == MediaCommand.PAUSE:
            client.pause()
        elif command_enum == MediaCommand.PLAY or command_enum == MediaCommand.RESUME:
            client.play()
        else:
            return f"ERROR: Invalid command '{command}'."
        return f"Command '{command}' executed on client '{client.title}'."
    except Exception as e:
        logger.exception("Failed to execute command '%s' on client '%s'.", command, machine_identifier)
        return f"ERROR: Failed to execute command '{command}' on client '{client.title}'. {str(e)}"


@mcp.tool(
    name="turn_off_client_subtitles",
    description="Turn off Plex client subtitles.",
    annotations=ToolAnnotations(
        title="Turn Off Client Subtitles",
    ),
)
async def turn_off_client_subtitles(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client find this by calling the get_client_machine_identifier tool.",
            examples=["1234567890abcdef", "abcdef1234567890"],
        ),
    ],
) -> str:
    """
    Turns off subtitles for a specified Plex client.
    """
    return await set_client_subtitles(machine_identifier, False)


@mcp.tool(
    name="turn_on_client_subtitles",
    description="Turn on Plex client subtitles.",
    annotations=ToolAnnotations(
        title="Turn On Client Subtitles",
    ),
)
async def turn_on_client_subtitles(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client find this by calling the get_client_machine_identifier tool.",
            examples=["1234567890abcdef", "abcdef1234567890"],
        ),
    ],
) -> str:
    """
    Turns on subtitles for a specified Plex client.
    """
    return await set_client_subtitles(machine_identifier, True)

@mcp.tool(
    name="get_client_machine_identifier",
    description="Get the machine identifier of a Plex client.",
    annotations=ToolAnnotations(
        title="Get Client Machine Identifier",
    ),
)
async def get_client_machine_identifier(
    client_name: Annotated[
        str,
        Field(
            description="The name of the Plex client.",
            examples=["Living Room TV", "Bedroom TV"],
        ),
    ],
) -> str:
    """
    Retrieves the machine identifier of a specified Plex client.
    Parameters:
        client_name: The name of the Plex client.
    Returns:
        The machine identifier of the client or an error message if the client is not found.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        clients = await asyncio.to_thread(plex.clients)

        if not clients:
            return "No active clients connected to your Plex server."
        logger.info("Searching for client with name '%s'.", client_name)
        client = match_client_name(client_name, clients, filter=lambda c: "playback" in c.protocolCapabilities)
        if client:
            return f"Machine Identifier for client '{client_name}': {client.machineIdentifier}" # type: ignore
        return f"ERROR: No client found with name '{client_name}'."
    except Exception as e:
        logger.exception("Failed to fetch client list.")
        return f"ERROR: Failed to fetch client list. {str(e)}"


async def set_client_subtitles(
    machine_identifier: Annotated[
        str,
        Field(
            description="The machine identifier of the Plex client.",
            examples=["1234567890abcdef", "abcdef1234567890"],
        ),
    ],
    subtitles_on: Annotated[
        bool,
        Field(
            description="Whether to turn subtitles on or off.",
            examples=[True, False],
        ),
    ],
) -> str:
    """
    Sets subtitles on or off for a specified Plex client.
    Parameters:
        machine_identifier: The machine identifier of the Plex client.
        subtitles_on: Whether to turn subtitles on or off.
    Returns
        a success message or an error message if the operation fails.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"
    try:
        client: PlexAPIClient = plex.client(machine_identifier)
        if "playback" not in client.protocolCapabilities:
            return f"ERROR: Client '{client.title}' does not support playback control."
    except NotFound:
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        return f"ERROR: Could not retrieve client. {str(e)}"
    logger.info("Found client '%s' for subtitle control.", client.title)
    try:
        sessions = await asyncio.to_thread(plex.sessions)
        logger.info("Found %d active sessions on Plex server.", len(sessions))
        session: PlexAPISession = next((s for s in sessions if s.player.machineIdentifier == machine_identifier), None) # type: ignore
        if not session:
            return f"ERROR: No active session found for client '{client.title}'."
        if not session.key:
            return f"ERROR: No session key found for client '{client.title}'."
        if not subtitles_on:
            client.setSubtitleStream(-1)
            return f"Subtitles disabled on client '{client.title}'."
        items = plex.fetchItems(session.key)
        logger.info("Found %d media items in session on client '%s'.", len(items), client.title)
        if not items:
            return f"ERROR: No media items found for session on client '{client.title}'."
        for _, item in enumerate(items):
            if not item.media or not item.media[0].parts:
                return f"ERROR: No media found for item  session on client '{client.title}'."
            for _, part in enumerate(item.media[0].parts):
                if not part.subtitleStreams():
                    continue
                for _, subtitle in enumerate(part.subtitleStreams()):
                    if subtitle.language.lower() == "english" and (not subtitle.forced and "force" not in subtitle.extendedDisplayTitle.lower()):
                        logger.info(
                            "Found English subtitle stream on client '%s' '%s'.",
                            client.title,
                            subtitle.extendedDisplayTitle,
                        )
                        client.connect()
                        client.setSubtitleStream(subtitle, "subtitle")
                        return f"Subtitles enabled on client '{client.title}'."
        return f"ERROR: No English subtitles found for current media on client '{client.title}'."
    except NotFound:
        logger.info("Client with machine identifier '%s' not found.", machine_identifier)
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        logger.exception("Failed to set subtitles on client '%s'.", machine_identifier)
        return f"ERROR: Could not set subtitles on client. {str(e)}"


@mcp.tool(
    name="list_playlists",
    description="List all playlists in the Plex server.",
    annotations=ToolAnnotations(
        title="List Playlists",
    ),
)
async def list_playlists() -> str:
    """
    List all playlists in the Plex server.

    Returns:
        A formatted string of playlists or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        playlists = await asyncio.to_thread(plex.playlists)
        if not playlists:
            return "No playlists found in your Plex server."
        formatted_playlists = []
        for i, playlist in enumerate(playlists, 1):
            formatted_playlists.append(
                f"Playlist #{i}:\nKey: {playlist.ratingKey}\n{format_playlist(playlist)}"  # type: ignore
            )
        return "\n---\n".join(formatted_playlists)
    except Exception as e:
        logger.exception("Failed to fetch playlists")
        return f"ERROR: Failed to fetch playlists. {str(e)}"


@mcp.tool(
    name="get_playlist_items",
    description="Get items in a specific playlist by its key.",
    annotations=ToolAnnotations(
        title="Get Playlist Items",
    ),
)
async def get_playlist_items(
    playlist_key: Annotated[
        str,
        Field(
            description="The key of the playlist to retrieve items from.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Get the items in a specific playlist.

    Parameters:
        playlist_key: The key of the playlist to retrieve items from.

    Returns:
        A formatted string of playlist items or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(playlist_key)
        all_playlists = await asyncio.to_thread(plex.playlists)
        playlist = next((p for p in all_playlists if p.ratingKey == key), None)  # type: ignore
        if not playlist:
            return f"No playlist found with key {playlist_key}."

        items = playlist.items()
        if not items:
            return "No items found in this playlist."

        formatted_items = []
        for i, item in enumerate(items, 1):
            title = item.title
            year = getattr(item, "year", "")
            type_str = item.type.capitalize()
            formatted_items.append(f"{i}. {title} ({year}) - {type_str}")
        return "\n".join(formatted_items)
    except NotFound:
        return f"ERROR: Playlist with key {playlist_key} not found."
    except Exception as e:
        logger.exception("Failed to fetch items for playlist key '%s'", playlist_key)
        return f"ERROR: Failed to fetch playlist items. {str(e)}"


@mcp.tool(
    name="create_playlist",
    description="Create a new playlist with specified movies.",
    annotations=ToolAnnotations(
        title="Create Playlist",
    ),
)
async def create_playlist(
    name: Annotated[
        str,
        Field(
            description="The name for the new playlist.",
            examples=["My Favorite Movies", "Action Classics", "Sci-Fi Hits"],
            min_length=1,
            max_length=100,
        ),
    ],
    movie_keys: Annotated[
        list[str],
        Field(
            description="List of movie keys to include in the playlist.",
            examples=[["12345", "67890", "112233"]],
        ),
    ],
) -> str:
    """
    Create a new playlist with specified movies.

    Parameters:
        name: The desired name for the new playlist.
        movie_keys: A list of movie keys to include.

    Returns:
        A success message with playlist details or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        movie_key_list = [int(key.strip()) for key in movie_keys if key.strip()]
        if not movie_key_list:
            return "ERROR: No valid movie keys provided."

        logger.info("Creating playlist '%s' with movie keys: %s", name, movie_keys)
        all_movies = await asyncio.to_thread(
            lambda: plex.library.search(libtype="movie")
        )
        logger.info("Found %d total movies in library", len(all_movies))
        movie_map = {movie.ratingKey: movie for movie in all_movies}  # type: ignore
        movies = []
        not_found_keys = []

        for key in movie_key_list:
            if key in movie_map:
                movies.append(movie_map[key])
                logger.info("Found movie: %s (Key: %d)", movie_map[key].title, key)  # type: ignore
            else:
                not_found_keys.append(key)
                logger.warning("Could not find movie with key: %d", key)

        if not_found_keys:
            return f"ERROR: Some movie keys were not found: {', '.join(str(k) for k in not_found_keys)}"
        if not movies:
            return "ERROR: No valid movies found with the provided keys."

        try:
            playlist_future = asyncio.create_task(
                asyncio.to_thread(lambda: plex.createPlaylist(name, items=movies))
            )
            playlist = await asyncio.wait_for(playlist_future, timeout=15.0)
            logger.info("Playlist created successfully: %s", playlist.title)
            return f"Successfully created playlist '{name}' with {len(movies)} movie(s).\nPlaylist Key: {playlist.ratingKey}"
        except asyncio.TimeoutError:
            logger.warning(
                "Playlist creation is taking longer than expected for '%s'", name
            )
            return (
                "PENDING: Playlist creation is taking longer than expected. "
                "The operation might still complete in the background. "
                "Please check your Plex server to confirm."
            )
    except ValueError as e:
        logger.error("Invalid input format for movie keys: %s", e)
        return f"ERROR: Invalid input format. Please check movie keys are valid numbers. {str(e)}"
    except Exception as e:
        logger.exception("Error creating playlist")
        return f"ERROR: Failed to create playlist. {str(e)}"


@mcp.tool(
    name="delete_playlist",
    description="Delete a playlist from the Plex server by its key.",
    annotations=ToolAnnotations(
        title="Delete Playlist",
    ),
)
async def delete_playlist(
    playlist_key: Annotated[
        str,
        Field(
            description="The key of the playlist to delete.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Delete a playlist from the Plex server.

    Parameters:
        playlist_key: The key of the playlist to delete.

    Returns:
        A success message if deletion is successful, or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(playlist_key)
        all_playlists = await asyncio.to_thread(plex.playlists)
        playlist = next((p for p in all_playlists if p.ratingKey == key), None)  # type: ignore
        if not playlist:
            return f"No playlist found with key {playlist_key}."
        await asyncio.to_thread(playlist.delete)
        logger.info(
            "Playlist '%s' with key %s successfully deleted.",
            playlist.title,
            playlist_key,
        )
        return (
            f"Successfully deleted playlist '{playlist.title}' with key {playlist_key}."
        )
    except NotFound:
        return f"ERROR: Playlist with key {playlist_key} not found."
    except Exception as e:
        logger.exception("Failed to delete playlist with key '%s'", playlist_key)
        return f"ERROR: Failed to delete playlist. {str(e)}"


@mcp.tool(
    name="add_to_playlist",
    description="Add a movie to an existing playlist by their keys.",
    annotations=ToolAnnotations(
        title="Add Movie to Playlist",
    ),
)
async def add_to_playlist(
    playlist_key: Annotated[
        str,
        Field(
            description="The key of the playlist to add the movie to.",
            examples=["12345", "67890"],
        ),
    ],
    movie_key: Annotated[
        str,
        Field(
            description="The key of the movie to add to the playlist.",
            examples=["54321", "98765"],
        ),
    ],
) -> str:
    """
    Add a movie to an existing playlist.

    Parameters:
        playlist_key: The key of the playlist.
        movie_key: The key of the movie to add.

    Returns:
        A success message if the movie is added, or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        p_key = int(playlist_key)
        m_key = int(movie_key)

        # Find the playlist
        all_playlists = await asyncio.to_thread(plex.playlists)
        playlist = next((p for p in all_playlists if p.ratingKey == p_key), None)  # type: ignore
        if not playlist:
            return f"No playlist found with key {playlist_key}."

        # Perform a global search for the movie
        movies = await asyncio.to_thread(
            lambda: plex.library.search(libtype="movie", ratingKey=m_key)
        )
        if not movies:
            return f"No movie found with key {movie_key}."

        movie = movies[
            0
        ]  # Since the search is scoped to the ratingKey, there should be at most one result

        # Add the movie to the playlist
        await asyncio.to_thread(lambda p=playlist, m=movie: p.addItems([m]))
        logger.info("Added movie '%s' to playlist '%s'", movie.title, playlist.title)  # type: ignore
        return f"Successfully added '{movie.title}' to playlist '{playlist.title}'."  # type: ignore
    except ValueError:
        return "ERROR: Invalid playlist or movie key. Please provide valid numbers."
    except Exception as e:
        logger.exception("Failed to add movie to playlist")
        return f"ERROR: Failed to add movie to playlist. {str(e)}"


@mcp.tool(
    name="get_movie_genres",
    description="Get genres for a specific movie by its key.",
    annotations=ToolAnnotations(
        title="Get Movie Genres",
    ),
)
async def get_movie_genres(
    movie_key: Annotated[
        str,
        Field(
            description="The key of the movie to retrieve genres for.",
            examples=["12345", "67890"],
        ),
    ],
) -> str:
    """
    Get genres for a specific movie.

    Parameters:
        movie_key: The key of the movie.

    Returns:
        A formatted string of movie genres or an error message.
    """
    try:
        plex = await get_plex_server()
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"

    try:
        key = int(movie_key)

        # Perform a global search for the movie
        all_movies = await asyncio.to_thread(
            lambda: plex.library.search(libtype="movie")
        )
        movie = next((m for m in all_movies if m.ratingKey == key), None)  # type: ignore
        if not movie:
            return f"No movie found with key {movie_key}."
        logger.info("Found movie: %s (Key: %d)", movie.title, key)  # type: ignore
        # Extract genres
        genres = (
            [genre.tag for genre in movie.genres] if hasattr(movie, "genres") else []
        )
        logger.info("Genres for movie '%s': %s", movie.title, genres)  # type: ignore
        if not genres:
            return f"No genres found for movie '{movie.title}'."
        return f"Genres for '{movie.title}':\n{', '.join(genres)}"
    except ValueError:
        return f"ERROR: Invalid movie key '{movie_key}'. Please provide a valid number."
    except Exception as e:
        logger.exception("Failed to fetch genres for movie with key '%s'", movie_key)
        return f"ERROR: Failed to fetch movie genres. {str(e)}"

@mcp.tool(
    name="get_movie_recommendations",
    description="Get movie recommendations.",
    annotations=ToolAnnotations(
        title="Get Movie Recommendations",
    ),
)
async def get_movie_recommendations(
    title: Annotated[
        Optional[str],
        Field(
            description="Title or substring to match",
            examples=["Inception", "Terminator", "Star Wars"],
            default=None,
        ),
    ] = None,
    year: Annotated[
        Optional[int],
        Field(
            description="Release year to filter by",
            default=None,
            examples=[1994, 2020, 1985],
        ),
    ] = None,
    director: Annotated[
        Optional[str],
        Field(
            description="Director name to filter by",
            default=None,
            examples=["Christopher Nolan", "James Cameron", "George Lucas"],
        ),
    ] = None,
    studio: Annotated[
        Optional[str],
        Field(
            description="Studio name to filter by",
            default=None,
            examples=["Warner Bros.", "20th Century Fox", "Universal Pictures"],
        ),
    ] = None,
    genre: Annotated[
        Optional[str],
        Field(
            description="Genre tag to filter by",
            default=None,
            examples=["Action", "Sci-Fi", "Drama"],
        ),
    ] = None,
    actor: Annotated[
        Optional[str],
        Field(
            description="Actor name to filter by",
            default=None,
            examples=["Leonardo DiCaprio", "Arnold Schwarzenegger", "Harrison Ford"],
        ),
    ] = None,
    rating: Annotated[
        Optional[str],
        Field(
            description="Rating to filter by (e.g., 'PG-13', 'R')",
            default=None,
            examples=["PG-13", "R", "G"],
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        Field(
            description="Country of origin to filter by",
            default=None,
            examples=["USA", "UK", "France"],
        ),
    ] = None,
    language: Annotated[
        Optional[str],
        Field(
            description="Audio or subtitle language to filter by",
            default=None,
            examples=["English", "Spanish", "French"],
        ),
    ] = None,
    watched: Annotated[
        Optional[bool],
        Field(
            description="Filter by watched status; True for watched, False for unwatched",
            default=None,
            examples=[True, False],
        ),
    ] = None,
    similar_movie_key: Annotated[
        Optional[str],
        Field(
            description="The key of the movie to retrieve recommendations for.",
            default=None,
            examples=["12345", "67890"],
        ),
    ] = None,
) -> str:
    """
    Get recommendations for movies based on various filters.

    Parameters:
        similar_movie_key (str): The key of the movie to retrieve similar movies for.
        title (str): Title or substring to match.
        year (int): Release year to filter by.
        director (str): Director name to filter by.
        studio (str): Studio name to filter by.
        genre (str): Genre tag to filter by.
        actor (str): Actor name to filter by.
        rating (str): Rating to filter by (e.g., 'PG-13', 'R').
        country (str): Country of origin to filter by.
        language (str): Audio or subtitle language to filter by.
        watched (bool): Filter by watched status; True for watched, False for unwatched.

    Returns:
        A list of movie keys and titles that match the given filters sorted by relevance, or an error message.
    """
    movies = [m for m in media_cache.values() if m.get("type") == "movie"]
    filter = media_cache.get(similar_movie_key or "", {})
    for key, value in {
        "year": year,
        "director": director,
        "studio": studio,
        "genre": genre,
        "actor": actor,
        "rating": rating,
        "country": country,
        "language": language,
        "watched": watched,
        "title": title,
    }.items():
        if value is not None:
            filter[key] = value
    return "\n".join([
        f"{m['ratingKey']}: {m['title']}"
        for m
        in sort_plex_objects_by_similarity(movies, filter)][:50]
    ) or "No recommendations found."

@mcp.tool(
    name="get_show_recommendations",
    description="Get tv show recommendations.",
    annotations=ToolAnnotations(
        title="Get Show Recommendations",
    ),
)
async def get_show_recommendations(
    show_title: Annotated[
        Optional[str],
        Field(
            description="Title or substring of the show to match",
            examples=["Family Guy", "South Park", "3 body problem"],
            default=None,
        ),
    ] = None,
    show_year: Annotated[
        Optional[int],
        Field(
            description="Show release year to filter by",
            default=None,
            examples=[1994, 2020, 1985],
        ),
    ] = None,
    director: Annotated[
        Optional[str],
        Field(
            description="Director name to filter by",
            default=None,
            examples=["Christopher Nolan", "James Cameron", "George Lucas"],
        ),
    ] = None,
    studio: Annotated[
        Optional[str],
        Field(
            description="Studio name to filter by",
            default=None,
            examples=["Warner Bros.", "20th Century Fox", "Universal Pictures"],
        ),
    ] = None,
    genre: Annotated[
        Optional[str],
        Field(
            description="Genre tag to filter by",
            default=None,
            examples=["Action", "Sci-Fi", "Drama"],
        ),
    ] = None,
    actor: Annotated[
        Optional[str],
        Field(
            description="Actor name to filter by",
            default=None,
            examples=["Leonardo DiCaprio", "Arnold Schwarzenegger", "Harrison Ford"],
        ),
    ] = None,
    rating: Annotated[
        Optional[str],
        Field(
            description="Rating to filter by (e.g., 'PG-13', 'R')",
            default=None,
            examples=["PG-13", "R", "G"],
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        Field(
            description="Country of origin to filter by",
            default=None,
            examples=["USA", "UK", "France"],
        ),
    ] = None,
    language: Annotated[
        Optional[str],
        Field(
            description="Audio or subtitle language to filter by",
            default=None,
            examples=["English", "Spanish", "French"],
        ),
    ] = None,
    watched: Annotated[
        Optional[bool],
        Field(
            description="Filter by watched status; True for watched, False for unwatched",
            default=None,
            examples=[True, False],
        ),
    ] = None,
    season: Annotated[
        Optional[int],
        Field(
            description="Season number to filter by",
            default=None,
            examples=[1, 2, 3],
        ),
    ] = None,
    episode: Annotated[
        Optional[int],
        Field(
            description="Episode number to filter by",
            default=None,
            examples=[1, 2, 3],
        ),
    ] = None,
    episode_title: Annotated[
        Optional[str],
        Field(
            description="Episode title or substring to match",
            default=None,
            examples=["Pilot", "The End"],
        ),
    ] = None,
    episode_year: Annotated[
        Optional[int],
        Field(
            description="Episode release year to filter by",
            default=None,
            examples=[2020, 2021, 2022],
        ),
    ] = None,
    similar_show_key: Annotated[
        Optional[str],
        Field(
            description="The key of the show or episode to retrieve recommendations for.",
            default=None,
            examples=["12345", "67890"],
        ),
    ] = None,
) -> str:
    """
    Get recommendations for show based on various filters.

    Parameters:
        similar_show_key (Optional[str]): The key of the show or episode to retrieve recommendations for.
        episode_title (Optional[str]): Episode title or substring to match.
        episode_year (Optional[int]): Episode release year to filter by.
        show_title (Optional[str]): Title or substring of the show to match.
        show_year (Optional[int]): Show release year to filter by.
        director (Optional[str]): Director name to filter by.
        studio (Optional[str]): Studio name to filter by.
        genre (Optional[str]): Genre tag to filter by.
        actor (Optional[str]): Actor name to filter by.
        rating (Optional[str]): Rating to filter by (e.g., 'PG-13', 'R').
        country (Optional[str]): Country of origin to filter by.
        language (Optional[str]): Language to filter by.
        watched (Optional[bool]): Filter by watched status.
        title (Optional[str]): Title or substring to match.
        season (Optional[int]): Season number to filter by.
        episode (Optional[int]): Episode number to filter by.

    Returns:
        A list of movie keys and titles that match the given filters sorted by relevance, or an error message.
    """
    episodes = [m for m in media_cache.values() if m.get("type") == "episode"]
    filter = media_cache.get(similar_show_key or "", {})

    for key, value in {
        "show.title": show_title,
        "show.year": show_year,
        "season.index": season,
        "index": episode,
        "title": episode_title,
        "director": director,
        "studio": studio,
        "genre": genre,
        "actor": actor,
        "rating": rating,
        "country": country,
        "language": language,
        "watched": watched,
        "year": episode_year,
    }.items():
        if value is  None:
            filter[key] = value
    return "\n".join([
        f"{m['ratingKey']}: {m['title']}"
        for m
        in sort_plex_objects_by_similarity(episodes, filter)][:50]
    ) or "No recommendations found."

def add_plex_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-u", "--plex-url",
        type=str,
        help="Base URL for the Plex server (e.g., http://localhost:32400)",
    )
    parser.add_argument(
        "-k", "--plex-token",
        type=str,
        help="Authentication token for accessing the Plex server",
    )

    return parser

def on_run_server(args):
    if not os.environ.get("PLEX_SERVER_URL"):
        if not args.plex_url:
            raise ValueError("Plex server URL must be provided via --plex-url or PLEX_SERVER_URL environment variable.")
        os.environ["PLEX_SERVER_URL"] = args.plex_url
    if not os.environ.get("PLEX_TOKEN"):
        if not args.plex_token:
            raise ValueError("Plex token must be provided via --plex-token or PLEX_TOKEN environment variable.")
        os.environ["PLEX_TOKEN"] = args.plex_token
    initialize_pickle(os.environ["PLEX_SERVER_URL"])
    PlexClient(
        os.environ["PLEX_SERVER_URL"], os.environ["PLEX_TOKEN"]
    )  # Initialize singleton
    asyncio.run(get_plex_server())
    if os.path.exists(os.path.join(os.environ.get("DATA_DIR", "."), "plex_cache.json")):
        logger.info("Loading Plex cache from disk...")
        with open(os.path.join(os.environ.get("DATA_DIR", "."), "plex_cache.json"), "r") as f:
            global media_cache
            media_cache = from_json(f.read())
    logger.info("Loaded Plex cache with %d items.", len(media_cache))
    asyncio.run(run_periodically(60.0 * 60.0))

async def refresh_cache(new_only: bool = False):
    logger.info("Refreshing Plex cache...")
    async def get_all_items(sections, libtype) -> dict[str, Any]:
        return {
            m.ratingKey: m
            for m
            in list(itertools.chain(*await asyncio.gather(*[
                asyncio.to_thread(section.all, libtype=libtype)
                for section
                in sections
        ])))}

    def build_cache(item, seasons, shows) -> Optional[dict[str, Any]]:
        if not item or not hasattr(item, "ratingKey"):
            return None
        item_dict = item.__dict__
        item_dict["reviews"] = (
            [
                r
                for r
                in item.reviews()
            ] if hasattr(item, "reviews") else []
        )
        item_dict["actors"] = (
            [
                a
                for a
                in item.actors
            ] if hasattr(item, "actors") else []
        )
        item_dict["directors"] = (
            [
                a
                for a
                in item.directors
            ] if hasattr(item, "directors") else []
        )
        item_dict["writers"] = (
            [
                a
                for a
                in item.writers
            ] if hasattr(item, "writers") else []
        )
        item_dict["genres"] = (
            [a for a in item.genres]
            if hasattr(item, "genres")
            else []
        )
        if hasattr(item, "parentRatingKey") and item.parentRatingKey:
            season = seasons.get(item.parentRatingKey, None)
            if season:
                item_dict["season"] = season
        if hasattr(item, "grandparentRatingKey") and item.grandparentRatingKey:
            show = shows.get(item.grandparentRatingKey, None)
            if show:
                item_dict["show"] = show

        return {
            k: v
            for k, v in item_dict.items()
            if not k.startswith("_")
            and k not in ["server", "section", "media", "parts", "sessionKey"]
            and v is not None
        }

    try:
        plex = await get_plex_server()

        if plex is None:
            raise ValueError("PlexClient instance is not initialized.")
        sections = await asyncio.to_thread(plex.library.sections)
        shows, seasons = await asyncio.gather(
            get_all_items(sections, "show"),
            get_all_items(sections, "season")
        )
        movies, episodes = await asyncio.gather(
            *[
                get_all_items(sections, libtype)
                for libtype
                in ["movie", "episode"]
            ]
        )
        cache = [
            build_cache(item, seasons, shows)
            for item in itertools.chain(movies.values(), episodes.values())
            if not new_only or (
                item is not None and item.ratingKey not in media_cache
            )
        ]
        for item in itertools.chain(shows.values(), seasons.values()):
            cache.append(build_cache(item, {}, {}))
        logger.info(f"Plex cache built with {len(cache)} items.")
        global media_cache
        if new_only:
            media_cache.update({item["ratingKey"]: item for item in cache if item is not None})
        else:
            media_cache = {item["ratingKey"]: item for item in cache if item is not None}  # type: ignore

        logger.info("Plex cache refreshed successfully.")
        async with aiofiles.open(
            os.path.join(os.environ.get("DATA_DIR", "."), "plex_cache.json"),
            mode='w'
        ) as f:
            await f.write(to_json(media_cache))
        logger.info("Plex cache saved successfully.")

    except Exception as e:
        logger.error(f"Failed to refresh Plex cache: {e}", exc_info=e)

async def run_periodically(interval_sec: float):
    while True:
        try:
            await refresh_cache()
        except Exception as e:
            print(f"Something went wrong, as expected: {e}")
        await asyncio.sleep(interval_sec)

def main():
    run_server("plex", add_args_fn=add_plex_args, run_callback=on_run_server)

# --- Main Execution ---
if __name__ == "__main__":
    main()
