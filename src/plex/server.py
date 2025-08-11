"""
Module: plex_mcp

This module provides tools for interacting with a Plex server via FastMCP.
It includes functions to search for movies, retrieve movie details, manage playlists,
and obtain recent movies and movie genres. Logging and asynchronous execution are used
to handle non-blocking I/O and to provide informative error messages.
"""

import argparse
import asyncio
import logging
import os
from enum import StrEnum

# --- Import Statements ---
from typing import Annotated, Callable, List, Literal, Optional

from mcp.types import ToolAnnotations
from plexapi.base import PlexSession as PlexAPISession
from plexapi.client import PlexClient as PlexAPIClient
from plexapi.exceptions import NotFound
from pydantic import Field
from rapidfuzz import process
from starlette.requests import Request
from starlette.responses import Response

from base import mcp, run_server
from plex.format import (
    format_client,
    format_episode,
    format_movie,
    format_session,
)
from plex.knowledge import KnowledgeBase, PlexMediaPayload, PlexMediaQuery
from plex.plex_api import PlexAPI, PlexTextSearch

# You must initialize logging, otherwise you'll not see debug output.
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for more verbosity during development
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.INFO)
fastmcp_logger = logging.getLogger("fastmcp")
fastmcp_logger.setLevel(logging.DEBUG)
fastmcp_logger.propagate = True
mcp_logger = logging.getLogger("mcp")
mcp_logger.setLevel(logging.DEBUG)
mcp_logger.propagate = True
# requests_log.propagate = True

# --- Logging Setup ---
logger = logging.getLogger(__name__)


# --- Utility Formatting Functions ---
def default_filter(client: PlexAPIClient) -> bool:
    return True


def match_client_name(
    requested_client: str,
    candidates: list[PlexAPIClient],
    filter: Callable[[PlexAPIClient], bool] = default_filter,
) -> Optional[PlexAPIClient]:
    """
    Try to match the requested client to a candidate
    """
    if not requested_client or not candidates or not filter:
        return None
    candidate_map = {c.title: c for c in candidates if filter(c)}
    choice, score, _ = process.extractOne(
        requested_client, list(candidate_map.keys()), score_cutoff=60
    )
    return candidate_map.get(choice, None) if score >= 60 else None


# --- Global Singleton and Access Functions ---
plex_api: PlexAPI | None = None
plex_search: PlexTextSearch | None = None


def get_plex_search() -> PlexTextSearch:
    if not plex_api:
        raise ValueError("PlexAPI is not initialized")
    if not plex_search:
        raise ValueError("PlexTextSearch is not initialized")
    return plex_search


@mcp.custom_route("/health", ["GET"], "health", False)
async def health(request: Request) -> Response:
    try:
        if not plex_api:
            return Response("ERROR", status_code=500, media_type="text/plain")
        await plex_api.get_sessions()
        return Response("OK", status_code=200, media_type="text/plain")
    except Exception:
        return Response("ERROR", status_code=500, media_type="text/plain")


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
    if not plex_api:
        return "ERROR: Plex API is not initialized."
    logger.info("Fetching movie details for key '%s'", movie_key)
    try:
        key = int(movie_key)
        movie = await plex_api.get_item(key)

        if not movie:
            return f"No movie found with key {movie_key}."
        payload = PlexMediaPayload(
            key=int(movie["ratingKey"]),
            title=movie["title"],
            summary=movie["summary"],
            year=int(movie["year"]) if movie["year"] else 0,
            rating=float(movie["rating"]) * 10.0 if movie["rating"] else 0.0,
            watched=movie["isWatched"],
            type="movie",
            genres=[g.tag for g in movie["Genre"]] if movie["Genre"] else [],
            actors=[a.tag for a in movie["Actor"]] if movie["Actor"] else [],
            studio=movie["studio"] or "",
            directors=[d.tag for d in movie["Director"]
                       ] if movie["Director"] else [],
            writers=[w.tag for w in movie["Writer"]
                     ] if movie["Writer"] else [],
            duration_seconds=(movie["duration"] //
                              1000) if movie["duration"] else 0,
            content_rating=movie["contentRating"] if "contentRating" in movie else None,
            show_title=None,
            season=None,
            episode=None,
            air_date=None,
        )
        logger.info("Returning %s", format_movie(payload))
        return format_movie(payload)
    except NotFound:
        return f"ERROR: Movie with key {movie_key} not found."
    except Exception as e:
        logger.exception(
            "Failed to fetch movie details for key '%s'", movie_key)
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
    if not plex_api:
        return "ERROR: Plex API is not initialized."
    try:
        movies = await plex_api.get_new_movies()
        if not movies:
            return "No new movies found in your Plex library."
        movies = [
            PlexMediaPayload(
                key=int(m["ratingKey"]),
                title=m["title"],
                summary=m["summary"],
                year=int(m["year"]) if m["year"] else 0,
                rating=float(m["rating"]) * 10.0 if m["rating"] else 0.0,
                watched=m["isWatched"],
                type="movie",
                genres=[g.tag for g in m["Genre"]] if m["Genre"] else [],
                actors=[a.tag for a in m["Actor"]] if m["Actor"] else [],
                studio=m["studio"] or "",
                directors=[d.tag for d in m["Director"]
                           ] if m["Director"] else [],
                writers=[w.tag for w in m["Writer"]] if m["Writer"] else [],
                duration_seconds=(
                    m["duration"] // 1000) if m["duration"] else 0,
                content_rating=m["contentRating"] if "contentRating" in m else None,
                show_title=None,
                season=None,
                episode=None,
                air_date=None,
            )
            for m in movies
        ]
        results: List[str] = []
        for i, m in enumerate(movies[:10], start=1):
            # type: ignore
            results.append(f"Result #{i}:\nKey: {m.key}\n{format_movie(m)}")
        logger.info(("Returning %s new movies.", "\n---\n".join(results)))
        return "\n---\n".join(results)
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"


@mcp.tool(
    name="get_episode_details",
    description="Get detailed information about a specific episode identified by its key.",
    annotations=ToolAnnotations(
        title="Get Movie Details",
    ),
)
async def get_episode_details(
    episode_key: Annotated[
        int,
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
    if not plex_api:
        return "ERROR: Plex API is not initialized."

    try:
        episode = await plex_api.get_item(episode_key)
        if not episode:
            return f"No episode found with key {episode_key}."
        payload = PlexMediaPayload(
            key=int(episode["ratingKey"]),
            title=episode["title"],
            summary=episode["summary"],
            year=int(episode["year"]) if episode["year"] else 0,
            rating=float(episode["rating"]) *
            10.0 if episode["rating"] else 0.0,
            watched=episode["isWatched"],
            type="episode",
            genres=[g.tag for g in episode["Genre"]
                    ] if episode["Genre"] else [],
            actors=[a.tag for a in episode["Actor"]
                    ] if episode["Actor"] else [],
            studio=episode["studio"] or "",
            directors=[d.tag for d in episode["Director"]
                       ] if episode["Director"] else [],
            writers=[w.tag for w in episode["Writer"]
                     ] if episode["Writer"] else [],
            duration_seconds=(episode["duration"] //
                              1000) if episode["duration"] else 0,
            content_rating=episode["contentRating"] if "contentRating" in episode else None,
            show_title=episode["grandparentTitle"] if "grandparentTitle" in episode else None,
            season=(
                episode["parentTitle"]
                if "parentTitle" in episode and episode["parentTitle"]
                else None
            ),
            episode=int(
                episode["index"]) if "index" in episode and episode["index"] else None,
            air_date=None,
        )
        return format_episode(payload)
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"


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
    if not plex_api:
        return "ERROR: Plex API is not initialized."
    try:
        episodes = await plex_api.get_new_episodes()
        if not episodes:
            return "No new episodes found in your Plex library."
        episodes = [
            PlexMediaPayload(
                key=int(e["ratingKey"]),
                title=e["title"],
                summary=e["summary"],
                year=int(e["year"]) if e["year"] else 0,
                rating=float(e["rating"]) * 10.0 if e["rating"] else 0.0,
                watched=e["isWatched"],
                type="episode",
                genres=[g.tag for g in e["Genre"]] if e["Genre"] else [],
                actors=[a.tag for a in e["Actor"]] if e["Actor"] else [],
                studio=e["studio"] or "",
                directors=[d.tag for d in e["Director"]
                           ] if e["Director"] else [],
                writers=[w.tag for w in e["Writer"]] if e["Writer"] else [],
                duration_seconds=(
                    e["duration"] // 1000) if e["duration"] else 0,
                content_rating=e["contentRating"] if "contentRating" in e else None,
                show_title=e["grandparentTitle"] if "grandparentTitle" in e else None,
                season=e["parentTitle"] if "parentTitle" in e and e["parentTitle"] else None,
                episode=int(
                    e["index"]) if "index" in e and e["index"] else None,
                air_date=None,
            )
            for e in episodes
        ]
        results: List[str] = []
        for i, m in enumerate(episodes[:10], start=1):
            # type: ignore
            results.append(f"Result #{i}:\nKey: {m.key}\n{format_episode(m)}")
        logger.info("Returning %s new episodes.", "\n---\n".join(results))
        return "\n---\n".join(results)
    except Exception as e:
        return f"ERROR: Could not connect to Plex server. {str(e)}"


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
    if not plex_api:
        return "ERROR: Plex server not configured."

    try:
        clients, sessions = await asyncio.gather(*[plex_api.get_clients, plex_api.get_sessions])

        if not clients and not sessions:
            return "No active clients connected to your Plex server."
        logger.info("Found %d active clients and %d sessions.",
                    len(clients), len(sessions))
        results: List[str] = []
        for i, m in enumerate(clients):
            logger.info(f"Client {m.title} {m.protocolCapabilities}")
            if controllable and "playback" not in m.protocolCapabilities:  # type: ignore
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
        Optional[int],
        Field(
            default=None,
            description="The key of the media item to play (if media_title is not provided).",
            examples=["12345", "67890"],
        ),
    ],
    media_title: Annotated[
        Optional[str],
        Field(
            default=None,
            description="The title of the media item to play (if media_key is not provided).",
            examples=["Inception", "The Matrix"],
        ),
    ] = None,
    media_type: Annotated[
        Optional[Literal["movie"] | Literal["episode"]],
        Field(
            description="The type of media to play when searching by title. (movie or episode)",
            examples=["movie", "episode"],
        ),
    ] = None,
) -> str:
    """
    Play specified media on a given Plex client.
    Parameters:
        machine_identifier: The machine identifier of the Plex client.
        media_key: The key of the media item to play.
    Returns:
        A success message or an error message.
    """
    if not media_key and not media_title:
        return "ERROR: Either media_key or media_title must be provided."
    if not plex_api:
        return "ERROR: Plex server not configured."

    try:
        clients: list[PlexAPIClient] = await plex_api.get_clients()

        if not clients:
            return "No active clients connected to your Plex server."
        logger.info("Found %d active clients.", len(clients))
        if (
            len(
                [
                    c
                    for c in clients
                    if c.machineIdentifier == machine_identifier_or_client_name
                    or c.title == machine_identifier_or_client_name
                ]
            )
            > 0
        ):
            client = [
                c
                for c in clients
                if c.machineIdentifier == machine_identifier_or_client_name
                or c.title == machine_identifier_or_client_name
            ][0]
        else:
            client = match_client_name(
                machine_identifier_or_client_name,
                clients,
                filter=lambda c: "playback" in c.protocolCapabilities,
            )
        if not client:
            return (
                f"No client found with machine identifier/name {machine_identifier_or_client_name}."
            )
        if "playback" not in client.protocolCapabilities:
            return f"Client {client.title} does not support playback control."
        if not media_key and media_title:
            media = await get_plex_search().find_media(
                PlexMediaQuery(type=media_type, title=media_title), limit=1
            )
            if not media:
                return f"No media found with title {media_title}."
            media_key = media[0].key if media else None
        if not media_key:
            return f"No media found with title {media_title}."
        logger.info("Found client: %s with media key: %s",
                    client.title, media_key)
        media = await plex_api.get_media(media_key)
        if not media:
            return f"No media found with key {media_key}."
        logger.info("Playing media: %s on client: %s",
                    media.title, client.title)
        await asyncio.to_thread(client.playMedia, media)
        return f"Playing {media.title} on {client.title}."  # type: ignore
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
    description="Control playback on a specified Plex client ('play', 'resume', 'pause', 'stop', 'fast forward', 'rewind', 'next', 'previous', 'seek', 'skip', 'start over).",
    annotations=ToolAnnotations(
        title="Control Client Playback",
    ),
)
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
            examples=[
                "play",
                "resume",
                "pause",
                "stop",
                "fast forward",
                "rewind",
                "next",
                "previous",
                "seek",
                "skip",
                "start over",
            ],
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
    if not plex_api:
        return "ERROR: Plex server not configured."

    try:
        client = await plex_api.get_client(machine_identifier)
        if not client:
            return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except NotFound:
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        return f"ERROR: Could not retrieve client. {str(e)}"

    if "playback" not in client.protocolCapabilities:
        logger.info(
            "Client '%s' does not support playback control.", client.title)
        return f"ERROR: Client '{client.title}' does not support playback control."

    try:
        command_enum = MediaCommand(command)
        logger.info("Sending command '%s' to client '%s'.",
                    command_enum, client.title)
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
        logger.exception(
            "Failed to execute command '%s' on client '%s'.", command, machine_identifier
        )
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
    if not plex_api:
        return "ERROR: Plex API is not initialized."

    try:
        clients = await plex_api.get_clients()

        if not clients:
            return "No active clients connected to your Plex server."
        logger.info("Searching for client with name '%s'.", client_name)
        client = match_client_name(
            client_name, clients, filter=lambda c: "playback" in c.protocolCapabilities
        )
        if client:
            # type: ignore
            return f"Machine Identifier for client '{client_name}': {client.machineIdentifier}"
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
    if not plex_api:
        return "ERROR: Plex server not configured."
    try:
        client: PlexAPIClient | None = await plex_api.get_client(machine_identifier)
        if not client:
            return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
        if "playback" not in client.protocolCapabilities:
            return f"ERROR: Client '{client.title}' does not support playback control."
    except NotFound:
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        return f"ERROR: Could not retrieve client. {str(e)}"
    logger.info("Found client '%s' for subtitle control.", client.title)
    try:
        sessions = await plex_api.get_sessions()
        logger.info("Found %d active sessions on Plex server.", len(sessions))
        session: PlexAPISession = next(
            # type: ignore
            (s for s in sessions if s.player.machineIdentifier == machine_identifier),
            None,
        )
        if not session:
            return f"ERROR: No active session found for client '{client.title}'."
        source = session.source()
        if not source:
            return f"ERROR: No session key found for client '{client.title}'."
        if not subtitles_on:
            client.setSubtitleStream(-1)
            return f"Subtitles disabled on client '{client.title}'."
        if not isinstance(source, list):
            source = [source]
        logger.info("Found %d media items in session on client '%s'.",
                    len(source), client.title)
        if not source:
            return f"ERROR: No media items found for session on client '{client.title}'."
        for _, item in enumerate(source):
            if not item.media or not item.media[0].parts:
                return f"ERROR: No media found for item  session on client '{client.title}'."
            for _, part in enumerate(item.media[0].parts):
                if not part.subtitleStreams():
                    continue
                for _, subtitle in enumerate(part.subtitleStreams()):
                    if subtitle.language.lower() == "english" and (
                        not subtitle.forced and "force" not in subtitle.extendedDisplayTitle.lower()
                    ):
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
        logger.info("Client with machine identifier '%s' not found.",
                    machine_identifier)
        return f"ERROR: Client with machine identifier '{machine_identifier}' not found."
    except Exception as e:
        logger.exception(
            "Failed to set subtitles on client '%s'.", machine_identifier)
        return f"ERROR: Could not set subtitles on client. {str(e)}"


def add_plex_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-u",
        "--plex-url",
        type=str,
        help="Base URL for the Plex server (e.g., http://localhost:32400)",
    )
    parser.add_argument(
        "-k",
        "--plex-token",
        type=str,
        help="Authentication token for accessing the Plex server",
    )
    parser.add_argument(
        "-qh",
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Host for the Qdrant vector database (default: localhost)",
    )
    parser.add_argument(
        "-qp",
        "--qdrant-port",
        type=int,
        default=6333,
        help="Port for the Qdrant vector database (default: 6333)",
    )
    parser.add_argument(
        "-mn",
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name for the Qdrant vector database (default: sentence-transformers/all-MiniLM-L6-v2)",
    )

    return parser


async def on_run_server(args):
    if not os.environ.get("PLEX_SERVER_URL"):
        if not args.plex_url:
            raise ValueError(
                "Plex server URL must be provided via --plex-url or PLEX_SERVER_URL environment variable."
            )
        os.environ["PLEX_SERVER_URL"] = args.plex_url
    if not os.environ.get("PLEX_TOKEN"):
        if not args.plex_token:
            raise ValueError(
                "Plex token must be provided via --plex-token or PLEX_TOKEN environment variable."
            )
        os.environ["PLEX_TOKEN"] = args.plex_token
    if not os.environ.get("QDRANT_HOST"):
        if not args.qdrant_host:
            raise ValueError(
                "Qdrant host must be provided via --qdrant-host or QDRANT_HOST environment variable."
            )
        os.environ["QDRANT_HOST"] = args.qdrant_host
    if not os.environ.get("QDRANT_PORT"):
        if not args.qdrant_port:
            raise ValueError(
                "Qdrant port must be provided via --qdrant-port or QDRANT_PORT environment variable."
            )
        os.environ["QDRANT_PORT"] = str(args.qdrant_port)
    if not os.environ.get("MODEL_NAME"):
        if not args.model_name:
            raise ValueError(
                "Model name must be provided via --model-name or MODEL_NAME environment variable."
            )
        os.environ["MODEL_NAME"] = args.model_name

    global plex_api, plex_search
    plex_api = PlexAPI(os.environ["PLEX_SERVER_URL"], os.environ["PLEX_TOKEN"])
    plex_search = PlexTextSearch(
        plex_api,
        KnowledgeBase(
            os.environ["MODEL_NAME"], os.environ["QDRANT_HOST"], int(
                os.environ["QDRANT_PORT"])
        ),
    )
    logger.info("Connected to Plex server at %s",
                os.environ["PLEX_SERVER_URL"])


def main():
    run_server("plex", add_args_fn=add_plex_args, run_callback=on_run_server)


# --- Main Execution ---
if __name__ == "__main__":
    main()
