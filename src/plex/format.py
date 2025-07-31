from plexapi.client import PlexClient as PlexAPIClient
import logging

from plexapi.base import PlexSession as PlexAPISession
from plexapi.utils import toJson
from plexapi.video import Video as PlexAPIVideo

import json

logger = logging.getLogger(__name__)


def format_movie(movie) -> str:
    """
    Format a movie object into a human-readable string.

    Parameters:
        movie: A Plex movie object.

    Returns:
        A formatted string containing movie details.
    """
    if not isinstance(movie, dict):
        movie = json.loads(toJson(movie))
    title = movie['title']
    year = movie['year']
    summary = movie['summary']
    duration = (
        movie['duration'] // 60000
    )
    rating = movie['rating']
    studio = movie['studio']
    directors = [
        director["tag"]
        for director
        in movie['director']
    ]
    actors = [
        role["tag"]
        for role
        in movie['actor']
    ]

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
    show_title = episode.get("grandparentTitle", "Unknown Show")
    season_number = episode.get("parentIndex", "Unknown Season")
    episode_number = episode.get("index", "Unknown Episode")
    title = episode.get("title", "Unknown Title")
    summary = episode.get("summary", "No summary available")
    duration = (
        episode.get("duration", 0) // 60000
    )
    rating = episode.get("rating", "Unrated")
    studio = episode.get("studio", "Unknown Studio")
    directors = [director.tag for director in episode.get("director", [])[:3]]
    actors = [role.tag for role in episode.get("role", [])[:5]]
    year = episode.get("year", "Unknown Year")

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
        "Media: {\n"
        f"{format_episode(source) if getattr(source, 'type', None) == 'episode' else format_movie(source)}\n"
        "}\n"
    )


def format_media(video: PlexAPIVideo) -> str:
    """
    Format a Plex video object into a human-readable string.
    Parameters:
        video: A Plex video object
    Returns:
        A formatted string containing video details.
    """
    logger.error(json.dumps(toJson(video), indent=2))
    is_movie = (video['type'] if isinstance(video, dict) else getattr(video, "type", None)) == "movie"
    logger.error(f"is_movie: {is_movie}")
    return (
        f"{format_episode(video) if not is_movie else format_movie(video)}\n"
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
        sum(item["duration"] for item in playlist["items"]) // 60000
        if playlist["items"] and "duration" in playlist["items"][0]
        else 0
    )
    return (
        f"Playlist: {playlist.title}\n"
        f"Items: {len(playlist.items())}\n"
        f"Duration: {duration_mins} minutes\n"
    )
