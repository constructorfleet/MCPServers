from plexapi.client import PlexClient as PlexAPIClient
import logging

from plexapi.base import PlexSession as PlexAPISession
from plexapi.utils import toJson

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