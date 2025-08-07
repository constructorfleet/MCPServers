from optparse import make_option
from typing import Generic, Literal, Optional, Type, TypeAlias, TypeVar, TypedDict, get_type_hints
from click import Option
from pydantic import BaseModel, create_model
import qdrant_client
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, CollectionInfo, Document, ScoredPoint
from qdrant_client.conversions.common_types import Points

from typing import Callable, Awaitable
import asyncio

import asyncio
from typing import Callable, Awaitable

TModel = TypeVar("TModel", bound=BaseModel)
T = TypeVar("T")

def make_optional_model(model_name: str, model: Type[T]) -> Type[BaseModel]:
    fields = get_type_hints(model)
    optional_fields = {k: (Optional[v], None) for k, v in fields.items()}
    return create_model(model_name, kwargs=optional_fields)

class PlexMediaPayload(BaseModel):
    key: int
    title: str
    summary: str
    year: int
    rating: float
    watched: bool
    type: Literal['movie'] | Literal['episode']
    genres: list[str]
    actors: list[str]
    studio: str
    directors: list[str]
    writers: list[str]
    duration_seconds: int
    content_rating: Optional[str]
    show_title: Optional[str]
    season: Optional[str]
    episode: Optional[int]
    
    @classmethod
    def document(cls, item: "PlexMediaPayload") -> str:
        parts = []
        if item.title:
            parts = [item.title]
        if item.summary:
            parts.append(item.summary)
        if item.genres:
            parts.append("Genres: " + ", ".join(item.genres))
        if item.actors:
            parts.append("Actors: " + ", ".join(item.actors))
        if item.directors:
            parts.append("Directored by: " + ", ".join(item.directors))
        if item.writers:
            parts.append("Written by: " + ", ".join(item.writers))
        if item.type == 'episode' and item.show_title:
            parts.append(f"Show: {item.show_title}")
            season_episode = []
            if item.season is not None:
                season_episode.append(item.season)
            if item.episode is not None:
                season_episode.append(f"Episode {item.episode}")
            if season_episode:
                parts.append(" ".join(season_episode))
        if item.year:
            parts.append(f"Year: {item.year}")
        if item.rating:
            parts.append(f"Rating: {item.rating}")
        if item.watched is not None:
            parts.append(f"Watched: {'Yes' if item.watched else 'No'}")
        return "\n".join(parts)

class PlexMediaQuery(PlexMediaPayload):
    key: Optional[int] = None # type: ignore
    title: Optional[str] = None # type: ignore
    summary: Optional[str] = None # type: ignore
    year: Optional[int] = None # type: ignore
    rating: Optional[float] = None # type: ignore
    watched: Optional[bool] = None # type: ignore
    type: Optional[Literal['movie'] | Literal['episode']] = None # type: ignore
    genres: Optional[list[str]] = None # type: ignore
    actors: Optional[list[str]] = None # type: ignore
    studio: Optional[str] = None # type: ignore
    directors: Optional[list[str]] = None # type: ignore
    writers: Optional[list[str]] = None # type: ignore
    duration_seconds: Optional[int] = None # type: ignore
    content_rating: Optional[str] = None # type: ignore
    show_title: Optional[str] = None # type: ignore
    season: Optional[str] = None # type: ignore
    episode: Optional[int] = None # type: ignore

class DataPoint(Generic[TModel], ScoredPoint):
    payload_class: Type[TModel]
    def payload_data(self) -> TModel:
        if not self.payload:
            raise ValueError("No payload available")
        return self.payload_class.model_validate(self.payload)

class Collection(Generic[TModel], CollectionInfo):
    qdrant_client: AsyncQdrantClient
    payload_class: Type[TModel]
    make_document: Callable[[TModel], str]
    name: str
    model: str
    
    async def upsert_points(self, points: Points, wait: bool = True):
        await self.qdrant_client.upsert(
            collection_name=self.name,
            points=points,
            wait=wait
        )

    async def upsert_data(self, data: list[TModel], id_getter: Callable[[TModel], Optional[int | str]] = lambda x: getattr(x, "id", None), wait: bool = True):
        ids: list[int | str] = []
        vectors: list[Document] = []
        payloads: list[dict] = []
        for item in data:
            ids.append(id_getter(item) or "")
            vectors.append(Document(text=self.make_document(item), model=self.model))
            payloads.append(item.model_dump(exclude_unset=True))
        await asyncio.to_thread(self.qdrant_client.upload_collection,
            collection_name=self.name,
            ids=ids,
            vectors=vectors,
            payload=payloads,
            wait=wait
        )

    async def search(self, data: TModel, limit: int | None = None, filter: Optional[dict] = None) -> list[DataPoint[TModel]]:
        return await self.query(self.make_document(data), limit=limit, filter=filter)

    async def query(self, query: str, limit: int | None = None, filter: Optional[dict] = None) -> list[DataPoint[TModel]]:
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            query=Document(text=query, model=self.model),
            limit=limit or 10000,
            filter=filter
        )
        return sorted([DataPoint.model_validate({
            "payload_class": self.payload_class,
            **p.model_dump()
        }) for p in result.points], key=lambda x: x.score, reverse=True)

class LazyCollections:
    def __init__(self, qdrant_client: AsyncQdrantClient, model: str, fetch_collection: Callable[[str], Awaitable[Optional[Collection]]]):
        self._qdrant_client = qdrant_client
        self._model = model
        self._fetch_collection = fetch_collection
        self._cache: dict[str, Collection] = {}

    async def _load_names(self) -> list[str]:
        if self._names is None:
            print("Fetching list of collection names from /collections")
            self._names = await self._fetch_collection_names()
        return self._names

    async def _fetch_collection_names(self) -> list[str]:
        response = await self._qdrant_client.get_collections()
        return [collection.name for collection in response.collections]
    
    async def media(self) -> Optional[Collection[PlexMediaPayload]]:
        if self._names is None:
            await self._load_names()
        return await self._fetch_collection("media")

    def __getattr__(self, name: str):
        async def getter():
            if self._names is None:
                await self._load_names()
                
            if not self._names:
                raise AttributeError("No collections available")

            if name not in self._names:
                raise AttributeError(f"No such collection: {name}")

            if name not in self._cache:
                print(f"Fetching /collections/{name}")
                collection = await self._fetch_collection(name)
                if collection is None:
                    raise AttributeError(f"Collection {name} not found")
                self._cache[name] = collection

            return self._cache[name]

        return AwaitableProxy(getter)

class AwaitableProxy:
    def __init__(self, coro_fn: Callable[[], Awaitable]):
        self._coro_fn = coro_fn

    def __await__(self):
        return self._coro_fn().__await__()

class KnowledgeBase:
    def __init__(self, model: str, qdrant_host: str, qdrant_port: int):
        self.model = model or "text-embedding-ada-002"
        self.qdrant_client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        self.model = model
        self._collection_cache: dict[str, Collection] = {}

    async def _has_collection(self, name: str) -> bool:
        collections = await self.qdrant_client.get_collections()
        return any(c.name == name for c in collections.collections)

    async def _fetch_collection(self, name: str, payload_class: Type[TModel], make_document: Callable[[TModel], str]) -> Optional[Collection]:
        if name in self._collection_cache:
            return self._collection_cache[name]
        if not await self._has_collection(name):
            return None
        info = await self.qdrant_client.get_collection(name)
        if info is None:
            return None
        collection = Collection(
            qdrant_client=self.qdrant_client,
            payload_class=payload_class,
            make_document=make_document,
            name=name,
            model=self.model,
            **info.model_dump()
        )
        self._collection_cache[name] = collection
        return collection
    
    async def media(self) -> Optional[Collection[PlexMediaPayload]]:
        if "media" in self._collection_cache:
            return self._collection_cache["media"]
        if not await self._has_collection("media"):
            return None
        return await self._fetch_collection("media", PlexMediaPayload, make_document=PlexMediaPayload.document)
    
