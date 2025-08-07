from re import M
from typing import Generic, Literal, Optional, Sequence, Type, TypeVar, cast, get_type_hints
from click import Option
from pydantic import BaseModel, create_model, ConfigDict
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import  CollectionInfo, Document, ScoredPoint
from qdrant_client.conversions.common_types import Points
from qdrant_client.http.models import Filter, MinShould, FieldCondition, MatchPhrase, MatchValue

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
            parts.append("Title: " + item.title)
        if item.summary:
            parts.append("Summary: " + item.summary)
        if item.genres:
            parts.append("Genres: " + ", ".join(item.genres))
        if item.actors:
            parts.append("Actors: " + ", ".join(item.actors))
        if item.directors:
            parts.append("Directed by: " + ", ".join(item.directors))
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
        if item.rating:
            if item.rating > 90:
                parts.append(f"Rating: Excellent")
            elif item.rating > 75:
                parts.append(f"Rating: Great")
            elif item.rating > 50:
                parts.append(f"Rating: Good")
            elif item.rating > 25:
                parts.append(f"Rating: Okay")
            else:
                parts.append(f"Rating: Bad")
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
    similar_to: Optional[int] = None # type: ignore

class DataPoint(ScoredPoint, Generic[TModel]):
    payload_class: Type[TModel]
    def payload_data(self) -> TModel:
        if not self.payload:
            raise ValueError("No payload available")
        return self.payload_class.model_validate(self.payload)

class Collection(CollectionInfo, Generic[TModel]):
    qdrant_client: AsyncQdrantClient
    payload_class: Type[TModel]
    make_document: Callable[[TModel], str]
    name: str
    model: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
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
            vectors.append(Document(text=self.make_document(item), model=self.model, options={"cuda": True}))
            payloads.append(item.model_dump(exclude_unset=True))
        await asyncio.to_thread(self.qdrant_client.upload_collection,
            collection_name=self.name,
            ids=ids,
            vectors=vectors,
            payload=payloads,
            wait=wait
        )

    async def search(self, data: PlexMediaPayload, limit: int | None = None) -> list[DataPoint[TModel]]:
        shoulds: list = []
        musts: list = []
        if data.title:
            shoulds.append(FieldCondition(key="title", match=MatchPhrase(phrase=data.title)))
        if data.show_title:
            shoulds.append(FieldCondition(key="show_title", match=MatchPhrase(phrase=data.show_title)))
        if data.genres:
            for genre in data.genres:
                shoulds.append(FieldCondition(key="genres", match=MatchPhrase(phrase=genre)))
        if data.watched is not None:
            musts.append(FieldCondition(key="watched", match=MatchValue(value=data.watched)))
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            query=Document(text=PlexMediaPayload.document(cast(PlexMediaPayload, data)), model=self.model, options={"cuda": True}),
            query_filter=Filter(
                should=[],
                must=[],
                must_not=[],
                min_should=MinShould(
                    conditions=shoulds,
                    min_count=1,
                )
            ),
            limit=limit or 10000,
        )
        return sorted([DataPoint.model_validate({
            "payload_class": self.payload_class,
            **p.model_dump()
        }) for p in result.points], key=lambda x: x.score, reverse=True)

    async def query(self, query: str, limit: int | None = None) -> list[DataPoint[TModel]]:
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            query=Document(text=query, model=self.model, options={"cuda": True}),
            limit=limit or 10000,
        )
        return sorted([DataPoint.model_validate({
            "payload_class": self.payload_class,
            **p.model_dump()
        }) for p in result.points], key=lambda x: x.score, reverse=True)

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
    
    async def movies(self) -> Optional[Collection[PlexMediaPayload]]:
        if "movies" in self._collection_cache:
            return self._collection_cache["movies"]
        if not await self._has_collection("movies"):
            return None
        return await self._fetch_collection("movies", PlexMediaPayload, make_document=PlexMediaPayload.document)
    
    async def episodes(self) -> Optional[Collection[PlexMediaPayload]]:
        if "episodes" in self._collection_cache:
            return self._collection_cache["episodes"]
        if not await self._has_collection("episodes"):
            return None
        return await self._fetch_collection("episodes", PlexMediaPayload, make_document=PlexMediaPayload.document)
    
