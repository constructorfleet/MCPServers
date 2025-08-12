import asyncio
import itertools
import logging
from datetime import date
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypedDict,
    Unpack,
    cast,
    get_type_hints,
)

import httpx
from plexapi.client import PlexClient
from plexapi.media import Media
from plexapi.server import PlexServer
from qdrant_client.models import ExtendedPointId

from plex.knowledge import Collection, KnowledgeBase, PlexMediaPayload, PlexMediaQuery
from plex.knowledge.types import MediaCollection, Rating, Review, Services
from plex.utils import batch_map

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
            # trunk-ignore(ruff/E722)
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

    def __init__(self, send_request: Callable[[str, str, str], Any]) -> None:
        super().__init__(send_request)
        self.play = Command(path="play", schema={"id": str}).bind(
            self._controller_path, self.send_request
        )
        self.pause = Command(path="pause").bind(self._controller_path, self.send_request)
        self.stop = Command(path="stop").bind(self._controller_path, self.send_request)
        self.seek_to = Command(path="seekTo", schema={"position": int}).bind(
            self._controller_path, self.send_request
        )
        self.skip_next = Command(path="skipNext").bind(self._controller_path, self.send_request)
        self.skip_previous = Command(path="skipPrevious").bind(
            self._controller_path, self.send_request
        )
        self.fast_forward = Command(path="stepForward", schema={"step": int}).bind(
            self._controller_path, self.send_request
        )
        self.rewind = Command(path="stepBack", schema={"step": int}).bind(
            self._controller_path, self.send_request
        )
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
        self.set_subtitle_stream = Command(
            path="setStreams", schema={"subtitleStreamID": int}
        ).bind(self._controller_path, self.send_request)
        self.set_audio_stream = Command(path="setStreams", schema={"audioStreamID": int}).bind(
            self._controller_path, self.send_request
        )


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
        self.context_menu = Command(path="contextMenu").bind(
            self._controller_path, self.send_request
        )
        self.show_osd = Command(path="showOSD").bind(self._controller_path, self.send_request)
        self.hide_osd = Command(path="hideOSD").bind(self._controller_path, self.send_request)
        self.toggle_osd = Command(path="toggleOSD").bind(self._controller_path, self.send_request)
        self.page_up = Command(path="pageUp").bind(self._controller_path, self.send_request)
        self.page_down = Command(path="pageDown").bind(self._controller_path, self.send_request)
        self.next_letter = Command(path="nextLetter").bind(self._controller_path, self.send_request)
        self.previous_letter = Command(path="previousLetter").bind(
            self._controller_path, self.send_request
        )
        self.toggle_fullscreen = Command(path="toggleFullscreen").bind(
            self._controller_path, self.send_request
        )


DEFAULT_QUERY_PARAMETERS: Mapping[str, httpx._types.PrimitiveData] = {
    "includeGuids": 1,
    "includeRelated": 0,
    "includeChapters": 1,
    "includeReviews": 1,
    "includeMarkers": 1,
    "includePopularLeaves": 1,
    "includePreferences": 1,
    "includeAdvanced": 0,
    "includeMeta": 0,
    "includeAllLeaves": 1,
    "includeChildren": 1,
    "includeCollections": 1,
    "includeArt": 0,
    "includeThumbs": 0,
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
    "includeTitle": 1,
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
        
    async def close(self):
        await self.client.aclose()

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

    async def get_library_sections(self) -> list:
        """Get a list of library sections."""
        data = await self._make_request("GET", "/library/sections")
        return (
            cast(list, data.get("MediaContainer", {}).get("Directory", []))
            if data.get("MediaContainer", {}).get("Directory")
            else []
        )

    async def get_library_section(self, section_id: int) -> dict:
        """Get details about a specific library section."""
        data = await self._make_request("GET", f"/library/sections/{section_id}")
        return (
            data.get("MediaContainer", {}).get("Metadata", [])[0]
            if data.get("MediaContainer", {}).get("Metadata")
            else {}
        )

    async def get_library_section_contents(self, section_id: int) -> list:
        """Get contents of a specific library section."""
        all_data = await asyncio.gather(
            *[
                self._make_request(
                    "GET", f"/library/sections/{section_id}/search", params={"type": libtype}
                )
                for libtype in [1, 4]
            ]
        )
        return list(
            itertools.chain.from_iterable(
                [
                    (
                        data.get("MediaContainer", {}).get("Metadata", [])
                        if data.get("MediaContainer", {}).get("Metadata")
                        else []
                    )
                    for data in all_data
                ]
            )
        )

    async def get_item(self, rating_key: int) -> dict:
        """Get details about a specific item by its rating key."""
        data = await self._make_request(
            "GET",
            f"/library/metadata/{rating_key}",
            params={"includeReviews": 1, "includeCollections": 1, "includeTags": 1},
        )
        return (
            data.get("MediaContainer", {}).get("Metadata", [{}])[0]
            if data.get("MediaContainer", {}).get("Metadata")
            else {}
        )

    async def _get_all_items(
        self,
        section_id: int | None = None,
        **kwargs: Unpack[SearchParameters],
    ) -> list:
        """Get all items of a specific type (e.g., 'movie', 'episode')."""
        url = f"/library/{f'{section_id}/' if section_id else ''}all"
        params: httpx._types.QueryParamTypes = {
            k: str(v) for k, v in kwargs.items() if v is not None
        }
        data = await self._make_request("GET", url, params=params)
        return (
            data.get("MediaContainer", {}).get("Metadata", [])
            if data.get("MediaContainer", {}).get("Metadata")
            else []
        )

    async def get_all_movies(
        self, section_id: int | None = None, **kwargs: Unpack[SearchParameters]
    ) -> list:
        """Get all movies."""
        if not kwargs:
            kwargs = SearchParameters()
        kwargs["type"] = "movie"
        return await self._get_all_items(section_id, **kwargs)

    async def get_new_movies(
        self,
    ) -> list:
        """Get recently added movies."""
        data = await self._make_request("GET", "/library/sections/1/recentlyAdded")
        return (
            data.get("MediaContainer", {}).get("Metadata", [])
            if data.get("MediaContainer", {}).get("Metadata")
            else []
        )

    async def get_new_episodes(
        self,
    ) -> list:
        """Get recently added episodes."""
        data = await self._make_request("GET", "/library/sections/2/recentlyAdded")
        return (
            data.get("MediaContainer", {}).get("Metadata", [])
            if data.get("MediaContainer", {}).get("Metadata")
            else []
        )

    async def get_all_shows(
        self, section_id: int | None = None, **kwargs: Unpack[SearchParameters]
    ) -> list:
        """Get all shows."""
        if not kwargs:
            kwargs = SearchParameters()
        kwargs["type"] = "show"
        return await self._get_all_items(section_id, **kwargs)

    async def get_all_seasons(
        self, section_id: int | None = None, **kwargs: Unpack[SearchParameters]
    ) -> list:
        """Get all seasons."""
        if not kwargs:
            kwargs = SearchParameters()
        kwargs["type"] = "season"
        return await self._get_all_items(section_id, **kwargs)

    async def get_all_episodes(
        self, section_id: int | None = None, **kwargs: Unpack[SearchParameters]
    ) -> list:
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
        return next(
            (c for c in await self.get_clients() if c.machineIdentifier == machine_identifier), None
        )

    async def get_media(self, key: int | str) -> Media | None:
        """Get media item by key."""
        return await asyncio.to_thread(self.server.fetchItem, key)


class PlexTextSearch:
    def __init__(self, plex: PlexAPI, knowledge_base: KnowledgeBase):
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
        self.plex = plex
        self.knowledge_base = knowledge_base
        self._loaded = False
        self._media: Collection[PlexMediaPayload]
        
    async def close(self):
        await self.knowledge_base.close()

    async def _load_items(self) -> None:
        """
        Load items from Plex server for the specified section.

        :return: list of media items
        """
        _LOGGER.info("Loading items")
        if media := await self.knowledge_base.ensure_media():
            self._media = media
        else:
            return None
        sections = await self.plex.get_library_sections()
        items: list[PlexMediaPayload] = []
        all_payloads = await asyncio.gather(
            *[self.plex.get_library_section_contents(int(section["key"])) for section in sections]
        )
        _LOGGER.info("Loading details...")
        results = await batch_map([
            item["ratingKey"] for item in list(itertools.chain.from_iterable(all_payloads))
        ], self._load_details, batch_size=50, concurrency=10, return_exceptions=False)
        items.extend([
            item
            for item in results if item is not None and isinstance(item, PlexMediaPayload)
        ])

        await self._do_upload(items)
        self._loaded = True

    async def _do_upload(self, items: list[PlexMediaPayload]) -> None:
        _LOGGER.info("Prepping media items for upload")
        media_collection = self._media
        movie_collection = await self.knowledge_base.ensure_movies()
        episode_collection = await self.knowledge_base.ensure_episodes()
        media: Sequence[PlexMediaPayload] = [point for point in items if point.type == "media"]
        movies: Sequence[PlexMediaPayload] = [point for point in items if point.type == "movie"]
        episodes: Sequence[PlexMediaPayload] = [
            point for point in items if  point.type == "episode"
        ]
        if movie_collection:
            _LOGGER.info("Upserting %d items into movie collection", len(movies))
            await movie_collection.upsert_data(
                movies,
                lambda x: x.key,
                False,
            )
        if episode_collection:
            _LOGGER.info("Upserting %d items into episode collection", len(episodes))
            await episode_collection.upsert_data(
                episodes,
                lambda x: x.key,
                False,
            )
        if media_collection:
            _LOGGER.info("Upserting %d items into media collection", len(items))
            await media_collection.upsert_data(
                media,  # type: ignore
                lambda x: x.key,
                False,
            )

    async def _load_details(self, id: ExtendedPointId) -> PlexMediaPayload:
        item = await self.plex.get_item(id if isinstance(id, int) else int(id))
        if not item:
            return PlexMediaQuery(key=int(id))
        return PlexMediaPayload(
            key=int(item.get("ratingKey", "")),
            title=item.get("title", ""),
            summary=item.get("summary", ""),
            genres=([g["tag"] for g in item.get("Genre", [])] if item.get("Genre") else []),
            directors=(
                [d["tag"] for d in item.get("Director", [])] if item.get("Director") else []
            ),
            actors=[a["tag"] for a in item.get("Role", [])] if item.get("Role") else [],
            writers=([w["tag"] for w in item.get("Writer", [])] if item.get("Writer") else []),
            producers=(
                [p["tag"] for p in item.get("Producer", [])] if item.get("Producer") else []
            ),
            year=item["year"] if item.get("year", None) is not None else 0,
            studio=item.get("studio", ""),
            rating=float(item.get("rating", 0.0)) * 10 if item.get("rating") else 0.0,
            content_rating=item.get("contentRating"),
            type=item.get("type", ""),
            watched=item.get("viewCount", 0) > 0,
            duration_seconds=int(item.get("duration", 0)),
            show_title=item.get("grandparentTitle"),
            season=item.get("parentIndex") if item.get("parentTitle") else None,
            episode=item.get("index") if item.get("index") else None,
            air_date=(
                date.fromisoformat(str(item.get("originallyAvailableAt")))
                if isinstance(item.get("originallyAvailableAt"), str)
                else item.get("originallyAvailableAt")
            ),
            reviews=[
                Review(
                    reviewer=item.get("tag"),
                    text=item.get("text"),
                    key=item.get("id"),
                )
                for item in item.get("Review", [])
            ],
            ratings=[
                Rating(
                    source=item.get("image").split(":")[0],
                    type=item.get("type"),
                    score=item.get("value"),
                )
                for item in item.get("Rating", [])
            ],
            services=Services(
                tmdb={guid['id'].split(':')[0]: guid['id'] for guid in item.get("Guid", [])}.get('tmdb'),
                imdb={guid['id'].split(':')[0]: guid['id'] for guid in item.get("Guid", [])}.get('imdb'),
                tvdb={guid['id'].split(':')[0]: guid['id'] for guid in item.get("Guid", [])}.get('tvdb'),
            ),
            collection=[
                MediaCollection(**c)
                for c in item.get("Collection", [])
            ]
                if item.get("Collection", None) else None
        )

    async def schedule_load_items(self, sleep: int = 60):
        await self._load_items()

        try:
            while True:
                await asyncio.sleep(sleep * 60)
                await self.schedule_load_items()
        except asyncio.CancelledError:
            _LOGGER.error("schedule_load_items cancelled")
            raise

    async def find_media(
        self,
        query: PlexMediaQuery,
        limit: int | None = None,
    ) -> list[PlexMediaPayload]:
        if query.type == "movie":
            media = await self.knowledge_base.movies()
        elif query.type == "episode":
            media = await self.knowledge_base.episodes()
        else:
            media = await self.knowledge_base.media()
        if not media:
            return []
        results = await media.search(query, limit=limit)
        return [r.payload_data() for r in results]
