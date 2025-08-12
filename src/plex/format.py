from plexapi.client import PlexClient as PlexAPIClient
import logging

from plexapi.base import PlexSession as PlexAPISession
from plexapi.utils import toJson
from plexapi.video import Video as PlexAPIVideo

import json

from plex.knowledge import PlexMediaPayload

logger = logging.getLogger(__name__)


def format_movie(movie: PlexMediaPayload) -> str:
    """
    Format a movie object into a human-readable string.

    Parameters:
        movie: A Plex movie object.

    Returns:
        A formatted string containing movie details.
    """
    title = movie.title
    year = movie.year
    summary = movie.summary
    directors = movie.directors or "Unknown"
    actors = movie.actors or []
    rating = movie.rating

    return (
        f"Title: {title} ({year})\n"
        f"Rating: {rating}\n"
        f"Directors: {directors}\n"
        f"Starring: {actors}\n"
        f"Summary: {summary}\n"
    )


def format_episode(episode: PlexMediaPayload) -> str:
    """
    Format an episode object into a human-readable string.

    Parameters:
        episode: A Plex episode object.

    Returns:
        A formatted string containing episode details.
    """
    show_title = episode.show_title
    season = episode.season
    episode_number = episode.episode
    title = episode.title
    summary = episode.summary
    rating = episode.rating or "Unrated"
    directors = episode.directors or "Unknown Director"
    actors = episode.actors or "Unknown Actors"
    year = episode.year or "Unknown Year"

    return (
        f"Show: {show_title}\n"
        f"Season {season}, Episode: {episode_number}\n"
        f"Year: {year}\n"
        f"Title: {title} ({year})\n"
        f"Rating: {rating}\n"
        f"Directors: {directors}\n"
        f"Starring: {actors}\n"
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
    is_movie = (
        video["type"] if isinstance(
            video, dict) else getattr(video, "type", None)
    ) == "movie"
    logger.error(f"is_movie: {is_movie}")
    if video.type == "movie":
        payload = PlexMediaPayload(
            key=int(video.ratingKey),
            title=video.title,
            summary=video.summary,
            year=int(video.year) if video.year else 0,
            rating=float(video.rating) * 10.0 if video.rating else 0.0,
            watched=video.isWatched,
            type="movie",
            genres=[g.tag for g in video.genres] if video.genres else [],
            actors=[a.tag for a in video.actors] if video.actors else [],
            studio=video.studio or "",
            directors=[
                d.tag for d in video.directors] if video.directors else [],
            writers=[w.tag for w in video.writers] if video.writers else [],
            duration_seconds=(video.duration // 1000) if video.duration else 0,
            content_rating=video.contentRating if hasattr(
                video, "contentRating") else None,
            show_title=None,
            season=None,
            episode=None,
            air_date=None,
        )
    else:
        payload = PlexMediaPayload(
            key=int(video.ratingKey),
            title=video.title,
            summary=video.summary,
            year=int(video.year) if video.year else 0,
            rating=float(video.rating) * 10.0 if video.rating else 0.0,
            watched=video.isWatched,
            type="episode",
            genres=[g.tag for g in video.genres] if video.genres else [],
            actors=[a.tag for a in video.actors] if video.actors else [],
            studio=video.studio or "",
            directors=[
                d.tag for d in video.directors] if video.directors else [],
            writers=[w.tag for w in video.writers] if video.writers else [],
            duration_seconds=(video.duration // 1000) if video.duration else 0,
            content_rating=video.contentRating if hasattr(
                video, "contentRating") else None,
            show_title=video.grandparentTitle if hasattr(
                video, "grandparentTitle") else None,
            season=(
                video.parentIndex
                if hasattr(video, "parentIndex") and video.parentIndex is not None
                else None
            ),
            episode=(
                int(video.index) if hasattr(
                    video, "index") and video.index is not None else None
            ),
            air_date=None,
        )
    return f"{format_episode(payload) if not is_movie else format_movie(payload)}\n"


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
