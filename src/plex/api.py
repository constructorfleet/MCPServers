import asyncio
import json
from plexapi.server import PlexServer
from plexapi.client import PlexClient
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import httpx
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    Type,
    TypedDict,
    Unpack,
    get_type_hints,
)

from plex.search import BaseTextSearch
_LOGGER = logging.getLogger(__name__)

class Command:

    def __init__(
        self,
        *,
        path: str,
        method: Literal["POST", "GET", "PUT", "DELETE"] = "GET",
        schema: Mapping[str, Type] | None = None,
        transform: Optional[Callable[[dict], dict]] = None,
    ):
        self.path = path
        self.method = method
        self.schema = schema
        self.transform = transform
        self.controller_path = None  # Filled in by Controller metaclass later

    def bind(self, controller_path: str, http_client: Any) -> Callable:
        async def call(**kwargs):
            try:
                annotations = get_type_hints(self.schema)
            except:
                annotations = {}
            for k, expected in annotations.items():
                if k not in kwargs:
                    raise TypeError(f"Missing required argument: {k}")
                if not isinstance(
                    kwargs[k], expected.__args__[0] if hasattr(expected, "__args__") else expected
                ):
                    raise TypeError(f"{k} must be {expected}, got {type(kwargs[k])}")
            if self.transform:
                kwargs = self.transform(kwargs)
            return await http_client(controller_path, self.path, params=kwargs)

        return call
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise RuntimeError("Commands must be called via a Controller instance.")


class ControllerMeta(type):
    def __new__(cls, name, bases, namespace, path=None):
        obj = super().__new__(cls, name, bases, dict(namespace))
        obj._controller_path = path or name.lower()  # type: ignore
        obj._commands = {  # type: ignore
            attr_name: attr for attr_name, attr in namespace.items() if isinstance(attr, Command)
        }
        return obj


class Controller(metaclass=ControllerMeta):
    def __init__(self, send_request: Callable[[str, str, str], Any]) -> None:
        self.send_request = send_request
        for name, cmd in self._commands.items():  # type: ignore
            bound = cmd.bind(self._controller_path, self.send_request)  # type: ignore
            setattr(self, name, bound)
    
    def __getattr__(self, name: str):
        cmd = self._commands.get(name)
        if cmd:
            return cmd.bind(self._controller_path, self._http_client)
        raise AttributeError(f"{self.__class__.__name__} has no command '{name}'")


class Playback(Controller, path="playback"):

    def __init__(
        self, send_request: Callable[[str, str, str], Any]
    ) -> None:
        super().__init__(send_request)
        self.play = Command(path="play", schema={"id": str}).bind(self._controller_path, self.send_request)
        self.pause = Command(path="pause").bind(self._controller_path, self.send_request)
        self.stop = Command(path="stop").bind(self._controller_path, self.send_request)
        self.seek_to = Command(path="seekTo", schema={"position": int}).bind(self._controller_path, self.send_request)
        self.skip_next = Command(path="skipNext").bind(self._controller_path, self.send_request)
        self.skip_previous = Command(path="skipPrevious").bind(self._controller_path, self.send_request)
        self.fast_forward = Command(path="stepForward", schema={"step": int}).bind(self._controller_path, self.send_request)
        self.rewind = Command(path="stepBack", schema={"step": int}).bind(self._controller_path, self.send_request)
        self.play_media = Command(
            path="playMedia",
            schema={
                "providerIdentifier": Type[str],
                "machineIdentifier": Type[str],
                "protocol": Type[str],
                "address": Type[str],
                "port": Type[int],
                "offset": Type[int],
                "key": Type[str],
                "type": Type[str],
                "containerKey": Type[str],
            },
        ).bind(self._controller_path, self.send_request)
        self.set_subtitle_stream = Command(path="setStreams", schema={"subtitleStreamID": int}).bind(self._controller_path, self.send_request)
        self.set_audio_stream = Command(path="setStreams", schema={"audioStreamID": int}).bind(self._controller_path, self.send_request)


class Navigation(Controller, path="navigation"):
    def __init__(self, http_client: Any):
        super().__init__(http_client)
        self.move_up = Command(path="moveUp").bind(self._controller_path, self.send_request)
        self.move_down = Command(path="moveDown").bind(self._controller_path, self.send_request)
        self.move_left = Command(path="moveLeft").bind(self._controller_path, self.send_request)
        self.move_right = Command(path="moveRight").bind(self._controller_path, self.send_request)
        self.select = Command(path="select").bind(self._controller_path, self.send_request)
        self.back = Command(path="back").bind(self._controller_path, self.send_request)
        self.home = Command(path="home").bind(self._controller_path, self.send_request)
        self.context_menu = Command(path="contextMenu").bind(self._controller_path, self.send_request)
        self.show_osd = Command(path="showOSD").bind(self._controller_path, self.send_request)
        self.hide_osd = Command(path="hideOSD").bind(self._controller_path, self.send_request)
        self.toggle_osd = Command(path="toggleOSD").bind(self._controller_path, self.send_request)
        self.page_up = Command(path="pageUp").bind(self._controller_path, self.send_request)
        self.page_down = Command(path="pageDown").bind(self._controller_path, self.send_request)
        self.next_letter = Command(path="nextLetter").bind(self._controller_path, self.send_request)
        self.previous_letter = Command(path="previousLetter").bind(self._controller_path, self.send_request)
        self.toggle_fullscreen = Command(path="toggleFullscreen").bind(self._controller_path, self.send_request)


DEFAULT_QUERY_PARAMETERS: Mapping[str, httpx._types.PrimitiveData] = {
    "includeGuids": 0,
    "includeRelated": 1,
    "includeChapters": 1,
    "includeReviews": 1,
    "includeMarkers": 1,
    "includePopularLeaves": 1,
    "includePreferences": 1,
    "includeAdvanced": 0,
    "includeMeta": 0,
    "includeAllLeaves": 1,
    "includeChildren": 1,
    "includeArt": 1,
    "includeThumbs": 1,
    "includeSummary": 1,
    "includeRatings": 1,
    "includeTags": 1,
    "includeGenres": 1,
    "includeDirectors": 1,
    "includeWriters": 1,
    "includeActors": 1,
    "includeCountries": 1,
    "includeStudios": 1,
    "includeLanguages": 1,
}


class SearchParameters(TypedDict):
    type: str | None
    title: str | None
    year: int | None
    tagline: str | None
    writer: str | list[str] | None
    director: str | list[str] | None
    studio: str | list[str] | None
    genre: str | list[str] | None
    actor: str | list[str] | None
    rating: str | None
    country: str | None
    summary: str | None
    contentRating: str | None
    watched: bool | None


class PlexAPI:
    """A class to interact with the Plex Media Server API."""

    def __init__(self, base_url: str, token: str) -> None:
        """Initialize the PlexAPI class."""
        self.base_url = base_url
        self.token = token
        self.client = httpx.AsyncClient(
            headers={"X-Plex-Token": self.token, "Accept": "application/json"},
            params={
                **DEFAULT_QUERY_PARAMETERS,
                "X-Plex-Token": self.token,
            },
            base_url=self.base_url,
            timeout=httpx.Timeout(1000.0, read=3600.0),
        )
        self.server = PlexServer(base_url, token)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: httpx._types.QueryParamTypes | None = None,
        **kwargs,
    ) -> dict:
        """Make an HTTP request to the Plex API."""
        response = await self.client.request(method, endpoint, params=params, **kwargs)
        response.raise_for_status()
        return response.json()

    async def get_servers(self) -> dict:
        """Get a list of available Plex servers."""
        return await self._make_request("GET", "/servers")

    async def get_library_sections(self) -> dict:
        """Get a list of library sections."""
        return await self._make_request("GET", "/library/sections")

    async def get_library_section(self, section_id: int) -> dict:
        """Get details about a specific library section."""
        return await self._make_request("GET", f"/library/sections/{section_id}")

    async def get_library_section_contents(self, section_id: int) -> dict:
        """Get contents of a specific library section."""
        return await self._make_request("GET", f"/library/sections/{section_id}/all")

    async def get_item(self, rating_key: int) -> dict:
        """Get details about a specific item by its rating key."""
        return await self._make_request("GET", f"/library/metadata/{rating_key}")

    async def _get_all_items(
        self,
        section_id: int | None = None,
        **kwargs: Unpack[SearchParameters],
    ) -> dict:
        """Get all items of a specific type (e.g., 'movie', 'episode')."""
        url = f"/library/{f'{section_id}/' if section_id else ''}all"
        params: httpx._types.QueryParamTypes = {
            k: str(v) for k, v in kwargs.items() if v is not None
        }
        return await self._make_request("GET", url, params=params)

    async def get_all_movies(
        self, section_id: int | None = None, **kwargs: Unpack[SearchParameters]
    ) -> dict:
        """Get all movies."""
        if not kwargs:
            kwargs = SearchParameters()
        kwargs["type"] = "movie"
        return await self._get_all_items(section_id, **kwargs)

    async def get_all_shows(
        self, section_id: int | None = None, **kwargs: Unpack[SearchParameters]
    ) -> dict:
        """Get all shows."""
        if not kwargs:
            kwargs = SearchParameters()
        kwargs["type"] = "show"
        return await self._get_all_items(section_id, **kwargs)

    async def get_all_seasons(
        self, section_id: int | None = None, **kwargs: Unpack[SearchParameters]
    ) -> dict:
        """Get all seasons."""
        if not kwargs:
            kwargs = SearchParameters()
        kwargs["type"] = "season"
        return await self._get_all_items(section_id, **kwargs)

    async def get_all_episodes(
        self, section_id: int | None = None, **kwargs: Unpack[SearchParameters]
    ) -> dict:
        """Get all episodes."""
        if not kwargs:
            kwargs = SearchParameters()
        kwargs["type"] = "episode"
        return await self._get_all_items(section_id, **kwargs)

    async def get_sessions(self) -> list:
        """Get active clients."""
        return await asyncio.to_thread(self.server.sessions)

    async def get_clients(self) -> list[PlexClient]:
        """Get all known clients."""
        return await asyncio.to_thread(self.server.clients)

    async def get_playlists(self) -> list:
        """Get all playlists."""
        response = await self._make_request("GET", "/playlists")
        playlists = response.get("MediaContainer", {}).get("Metadata", [])
        for playlist in playlists:
            items = await self._make_request("GET", playlist["key"])
            playlist["items"] = items.get("MediaContainer", {}).get("Metadata", [])
        return playlists

    async def get_client(self, machine_identifier: str) -> PlexClient | None:
        """Get a client by ID."""
        return next((c for c in await self.get_clients() if c.machineIdentifier == machine_identifier), None)


class PlexTextSearch(BaseTextSearch):
    def __init__(self, plex: PlexAPI, fields_with_weights: dict, mode: str='tfidf', model_name: str='all-MiniLM-L6-v2'):
        """
        Initialize the text search with specified fields and weights.
        
        :param fields_with_weights: dict of {field_name: weight}
        :param mode: 'tfidf' or 'embedding'
        :param model_name: model for embedding mode
        """
        """
        Initialize the text search with specified fields and weights.
        
        :param fields_with_weights: dict of {field_name: weight}
        :param mode: 'tfidf' or 'embedding'
        :param model_name: model for embedding mode
        """
        super().__init__(fields_with_weights, mode, model_name)
        self.plex = plex
        self._media_items: dict[int, list[dict]] = {}

    async def _load_items(self) -> None:
        """
        Load items from Plex server for the specified section.
        
        :return: list of media items
        """
        sections_data = await self.plex.get_library_sections()
        sections = sections_data.get("MediaContainer", {}).get("Directory", [])

        for section in sections:
            items: list[dict] = []
            sec_id = int(section["key"])
            all_items_data = await self.plex.get_library_section_contents(sec_id)
            items.extend(all_items_data.get("MediaContainer", {}).get("Metadata", []))

            self._media_items[sec_id] = items
        with open("items.json", "w") as f:
            f.write(json.dumps(self._media_items, indent=2))

    def flatten_media_items(self, items: list[dict]) -> pd.DataFrame:
        def extract_tags(tag_list):
            return [x["tag"] for x in tag_list] if isinstance(tag_list, list) else []

        normalized = []
        for item in items:
            normalized.append(
                {
                    "ratingKey": item.get("ratingKey", ""),
                    "key": item.get("key", ""),
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "genre": extract_tags(item.get("Genre", [])),
                    "director": extract_tags(item.get("Director", [])),
                    "actor": extract_tags(item.get("Role", [])),
                    "writer": extract_tags(item.get("Writer", [])),
                    "year": item.get("year", ""),
                    "studio": item.get("studio", ""),
                    "country": item.get("country", ""),
                    "contentRating": item.get("contentRating", ""),
                    "originalTitle": item.get("originalTitle", ""),
                    "tagline": item.get("tagline", ""),
                    "originallyAvailableAt": item.get("originallyAvailableAt", ""),
                    "addedAt": item.get("addedAt", ""),
                    "updatedAt": item.get("updatedAt", ""),
                    "rating": item.get("rating", 0.0) * 10,
                    "type": item.get("type", ""),
                    "watched": item.get("lastViewedAt", 0) > 0,
                    "duration": item.get("duration", 0),
                }
            )
        return pd.DataFrame(normalized)
    
    async def _schedule_load_items(self):
        await self._load_items()
        self._load_items_task = asyncio.create_task(self._schedule_load_items())

    async def find_media(
            self, 
            section_id: int | None = None, 
            query: str | None = None, 
            type: str | None = None,
            title: str | None = None,
            year: int | None = None,
            tagline: str | None = None,
            writer: str | list[str] | None = None,
            director: str | list[str] | None = None,
            studio: str | list[str] | None = None,
            genre: str | list[str] | None = None,
            actor: str | list[str] | None = None,
            rating: str | None = None,
            country: str | None = None,
            summary: str | None = None,
            contentRating: str | None = None,
            watched: bool | None = None,
    ) -> list:
        """
        Perform a text search on the Plex library.
        
        :param query: search query
        :param section_id: optional section ID to limit search
        :param kwargs: additional search parameters
        :return: list of search results
        """
        if not self._media_items:
            await self._schedule_load_items()

        if section_id is not None:
            items = self._media_items.get(section_id, [])
        elif type is not None:
            items = [item for sublist in self._media_items.values() for item in sublist if item.get("type") == type]
        else:
            items = [item for sublist in self._media_items.values() for item in sublist]
        self.fit(self.flatten_media_items(items))
        query_dict = {
            "type": type,
            "title": title,
            "year": year,
            "tagline": tagline,
            "writer": writer,
            "director": director,
            "studio": studio,
            "genre": genre,
            "actor": actor,
            "rating": rating,
            "country": country,
            "summary": summary,
            "contentRating": contentRating,
            "watched": watched,
        }   
        if query:
            query_dict["summary"] = query

        return self.search(query_dict, len(items)).to_dict(orient="records") if items else []  # type: ignore
