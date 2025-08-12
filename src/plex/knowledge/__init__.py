import json
import logging
import os
from datetime import date
from typing import Annotated, Callable, Literal, Optional, Type, cast

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Condition,
    Document,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    Prefetch,
    RecommendInput,
    RecommendQuery,
    VectorInput,
)

from plex.knowledge.collection import Collection
from plex.knowledge.types import (
    DataPoint,
    Diagnostics,
    ExplainContext,
    MediaSearchResponse,
    MinMax,
    PlexMediaPayload,
    PlexMediaQuery,
    Retrieval,
    TModel,
)
from plex.knowledge.utils import (
    build_filters,
    ensure_collection,
    ensure_payload_indexes,
    point_to_media_result,
    sparse_from_text,
)

logging.getLogger("qdrant_client").setLevel(logging.DEBUG)
_LOGGER = logging.getLogger(__name__)

__all__ = ["KnowledgeBase", "PlexMediaPayload", "PlexMediaQuery"]


class KnowledgeBase:
    """High-level interface for managing Plex media collections in Qdrant.

    This class provides methods for setting up and accessing different types
    of media collections (movies, episodes, etc.) with caching support.
    """

    _instance: "KnowledgeBase"

    @staticmethod
    def instance() -> "KnowledgeBase":
        """Get the singleton instance of the knowledge base."""
        if not hasattr(KnowledgeBase, "_instance"):
            raise RuntimeError("KnowledgeBase is not initialized")
        return KnowledgeBase._instance

    def __init__(self, model: str, qdrant_host: str, qdrant_port: int):
        """Initialize the knowledge base.

        Args:
            model: Name of the embedding model to use
            qdrant_host: Hostname of the Qdrant server
            qdrant_port: Port of the Qdrant server
        """
        KnowledgeBase._instance = self
        self.model = model or "text-embedding-ada-002"
        self.qdrant_client = AsyncQdrantClient(
            host=qdrant_host, port=qdrant_port, grpc_port=6334, prefer_grpc=True
        )
        self.model = model
        self._collection_cache: dict[str, Collection] = {}
        self.enable_rerank: bool = os.environ.get(
            "ENABLE_RERANK", "false") == "true"
        self.reranker_name: Optional[str] = os.environ.get(
            "RERANKER_NAME", None)
        self.enable_two_pass_fusion: bool = (
            os.environ.get("ENABLE_TWO_PASS_FUSION", "false") == "true"
        )
        self.fusion_dense_weight: float = float(
            os.environ.get("FUSION_DENSE_WEIGHT", "0.7"))
        self.fusion_sparse_weight: float = float(
            os.environ.get("FUSION_SPARSE_WEIGHT", "0.3"))
        self.fusion_prelimit: int = int(
            os.environ.get("FUSION_PRELIMIT", "200"))
        self.enable_server_fusion: bool = os.environ.get(
            "ENABLE_SERVER_FUSION", "false") == "true"
        # only rrf supported currently
        self.server_fusion_method: Literal["rrf"] = "rrf"
        self.enable_diversity: bool = os.environ.get(
            "ENABLE_DIVERSITY", "false") == "true"
        self.diversity_lambda: float = float(
            os.environ.get("DIVERSITY_LAMBDA", "0.3"))
        # cap results per series/franchise
        self.max_per_series: int = int(os.environ.get("MAX_PER_SERIES", "1"))
        _LOGGER.error(self.qdrant_client.get_embedding_size(self.model))

    async def close(self):
        await self.qdrant_client.close()

    def document(self, doc: str | PlexMediaQuery | PlexMediaPayload) -> Document:
        """Create a document representation of the knowledge base."""
        return Document(
            text=doc if isinstance(
                doc, str) else PlexMediaPayload.document(doc),
            model=self.model,
            options={"cuda": True},
        )

    async def ensure_movies(self, dim: int | None = None) -> Collection[PlexMediaPayload]:
        """Ensure the movies collection exists with proper configuration.

        Args:
            dim: Dimension of the dense vector embeddings
        """
        await ensure_collection(
            self.qdrant_client, "movies", dim or self.qdrant_client.get_embedding_size(
                self.model)
        )
        await ensure_payload_indexes(self.qdrant_client, "movies")
        collection = await self._fetch_collection(
            "movies", PlexMediaPayload, make_document=PlexMediaPayload.document
        )
        if not collection:
            raise RuntimeError("Failed to create or fetch movies collection")
        return collection

    async def ensure_episodes(self, dim: int | None = None) -> Collection[PlexMediaPayload]:
        """Ensure the episodes collection exists with proper configuration.

        Args:
            dim: Dimension of the dense vector embeddings
        """
        await ensure_collection(
            self.qdrant_client, "episodes", dim or self.qdrant_client.get_embedding_size(
                self.model)
        )
        await ensure_payload_indexes(self.qdrant_client, "episodes")
        collection = await self._fetch_collection(
            "episodes", PlexMediaPayload, make_document=PlexMediaPayload.document
        )
        if not collection:
            raise RuntimeError("Failed to create or fetch episodes collection")
        return collection

    async def ensure_media(self, dim: int | None = None) -> Collection[PlexMediaPayload]:
        """Ensure the media collection exists with proper configuration.

        Args:
            dim: Dimension of the dense vector embeddings
        """
        await ensure_collection(
            self.qdrant_client, "media", dim or self.qdrant_client.get_embedding_size(
                self.model)
        )
        await ensure_payload_indexes(self.qdrant_client, "media")
        collection = await self._fetch_collection(
            "media", PlexMediaPayload, make_document=PlexMediaPayload.document
        )
        if not collection:
            raise RuntimeError("Failed to create or fetch media collection")
        return collection

    async def _has_collection(self, name: str) -> bool:
        """Check if a collection exists in Qdrant.

        Args:
            name: Name of the collection to check

        Returns:
            bool: True if collection exists, False otherwise
        """
        collections = await self.qdrant_client.get_collections()
        return any(c.name == name for c in collections.collections)

    async def _fetch_collection(
        self, name: str, payload_class: Type[TModel], make_document: Callable[[TModel], str]
    ) -> Optional[Collection]:
        """Fetch a collection from Qdrant and wrap it in a Collection object.

        Args:
            name: Name of the collection
            payload_class: Class for validating payload data
            make_document: Function to convert payload to document text

        Returns:
            Optional[Collection]: Collection wrapper or None if not found
        """
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
            **info.model_dump(),
        )
        self._collection_cache[name] = collection
        return collection

    async def media(self) -> Optional[Collection[PlexMediaPayload]]:
        """Get the media collection containing all media types.

        Returns:
            Optional[Collection[PlexMediaPayload]]: Media collection or None if not found
        """
        _LOGGER.warning("Fetching media collection")
        if "media" in self._collection_cache:
            return self._collection_cache["media"]
        if not await self._has_collection("media"):
            return None
        return await self._fetch_collection(
            "media", PlexMediaPayload, make_document=PlexMediaPayload.document
        )

    async def movies(self) -> Optional[Collection[PlexMediaPayload]]:
        """Get the movies collection.

        Returns:
            Optional[Collection[PlexMediaPayload]]: Movies collection or None if not found
        """
        _LOGGER.warning("Fetching movies collection")
        if "movies" in self._collection_cache:
            return self._collection_cache["movies"]
        if not await self._has_collection("movies"):
            return None
        return await self._fetch_collection(
            "movies", PlexMediaPayload, make_document=PlexMediaPayload.document
        )

    async def episodes(self) -> Optional[Collection[PlexMediaPayload]]:
        """Get the episodes collection.

        Returns:
            Optional[Collection[PlexMediaPayload]]: Episodes collection or None if not found
        """
        _LOGGER.warning("Fetching episodes collection")
        if "episodes" in self._collection_cache:
            return self._collection_cache["episodes"]
        if not await self._has_collection("episodes"):
            return None
        return await self._fetch_collection(
            "episodes", PlexMediaPayload, make_document=PlexMediaPayload.document
        )

    def find_media_tool(self, mcp: FastMCP) -> None:
        async def filter_points(
            collection: str,
            filters: PlexMediaQuery,
        ) -> list[DataPoint[PlexMediaPayload]]:
            """Filter data points based on the provided filters.

            Args:
                scope: The scope to filter within.
                filters: The filters to apply.

            Returns:
                A list of filtered data points.
            """
            musts: list[Condition] = []
            if filters.genres:
                musts.extend(
                    [
                        FieldCondition(
                            key="genres", match=MatchValue(value=genre.title()))
                        for genre in filters.genres
                    ]
                )
            if filters.directors:
                musts.extend(
                    [
                        FieldCondition(key="directors", match=MatchValue(
                            value=director.title()))
                        for director in filters.directors
                    ]
                )
            if filters.writers:
                musts.extend(
                    [
                        FieldCondition(key="writers", match=MatchValue(
                            value=writer.title()))
                        for writer in filters.writers
                    ]
                )
            if filters.title:
                musts.append(FieldCondition(
                    key="title", match=MatchText(text=filters.title)))
            if filters.summary:
                musts.append(FieldCondition(
                    key="summary", match=MatchText(text=filters.summary)))
            if filters.season:
                musts.append(FieldCondition(
                    key="season", match=MatchValue(value=filters.season)))
            if filters.episode:
                musts.append(FieldCondition(
                    key="episode", match=MatchValue(value=filters.episode)))
            if filters.show_title:
                musts.append(
                    FieldCondition(key="show_title", match=MatchText(
                        text=filters.show_title))
                )
            _LOGGER.info(
                f'Filtering points with conditions: {json.dumps({
                    "collection_name": collection,
                    "query_filter": Filter(must=musts).model_dump(),
                    "using": "dense",
                    "limit": 100
                }, indent=2)}'
            )
            result = await KnowledgeBase.instance().qdrant_client.query_points(
                collection_name=collection,
                query_filter=Filter(must=musts),
                using="dense",
                limit=100,
            )
            _LOGGER.info(
                f"Found {len(result.points)} points matching the query and filters.")
            _LOGGER.info(json.dumps(result.model_dump(), indent=2))
            return [
                DataPoint(payload_class=PlexMediaPayload, **p.model_dump()) for p in result.points
            ]

        async def find_media(
            media_type: Annotated[
                str,
                Field(
                    title="Media Type",
                    description="The media library section to query 'movies' or 'episodes'.",
                    examples=["movies", "episodes"],
                ),
            ],
            similar_to: Annotated[
                Optional[str],
                Field(
                    default=None,
                    title="Similarity Title Anchor Filter",
                    description="The title of another media to use as a similarity anchor for the query: similar genres, plots, synopsis. etc/",
                ),
            ] = None,
            with_genre: Annotated[
                Optional[list[str] | str],
                Field(
                    default=None,
                    title="Genres",
                    description="Query for media that is categorized by these genres.",
                ),
            ] = None,
            directed_by: Annotated[
                Optional[list[str] | str],
                Field(
                    default=None,
                    title="Directors",
                    description="Query for media that was directed by these individuals.",
                ),
            ] = None,
            written_by: Annotated[
                Optional[list[str] | str],
                Field(
                    default=None,
                    title="Writers",
                    description="Query for media that was written by these individuals.",
                ),
            ] = None,
            starring: Annotated[
                Optional[list[str] | str],
                Field(
                    default=None,
                    title="Actors",
                    description="Query for media that the following individuals act in.",
                ),
            ] = None,
            summary: Annotated[
                Optional[str],
                Field(
                    default=None,
                    title="Plot",
                    description="Query for media with similar plots.",
                ),
            ] = None,
            aired_before: Annotated[
                Optional[date],
                Field(
                    default=None,
                    title="Aired Before",
                    description="Query for media aired before: <date> or <int> days ago",
                ),
            ] = None,
            aired_after: Annotated[
                Optional[date],
                Field(
                    default=None,
                    title="Aired After",
                    description="Query for media that aired after: <date> or <int> days ago",
                ),
            ] = None,
            series: Annotated[
                Optional[str],
                Field(
                    default=None,
                    title="Series",
                    description="Query for media that is part of this series.",
                ),
            ] = None,
            season: Annotated[
                Optional[list[int] | int],
                Field(
                    default=None,
                    title="Seasons",
                    description="Query for media with these season number.",
                ),
            ] = None,
            episode: Annotated[
                Optional[list[int] | int],
                Field(
                    default=None,
                    title="Episodes",
                    description="Query for media with these episode numbers.",
                ),
            ] = None,
            rating_min: Annotated[
                Optional[float],
                Field(
                    default=None,
                    title="Minimum Rating",
                    description="Query for media with a minimum rating.",
                ),
            ] = None,
            rating_max: Annotated[
                Optional[float],
                Field(
                    default=None,
                    title="Maximum Rating",
                    description="Query for media with a maximum rating.",
                ),
            ] = None,
            watched: Annotated[
                Optional[bool],
                Field(
                    default=None,
                    title="Watched Status",
                    description="Query for media that has or has not been watched.",
                ),
            ] = None,
            limit: Annotated[
                Optional[int],
                Field(
                    default=None,
                    title="Limit",
                    description="Query for a maximum number of results.",
                ),
            ] = None,
            offset: Annotated[
                Optional[int],
                Field(
                    default=None,
                    title="Offset",
                    description="Query for the number of results to skip.",
                ),
            ] = None,
            theme_or_story: Annotated[
                Optional[str],
                Field(
                    default=None,
                    title="Theme or story",
                    description="Query based on theme, story or other vague characteristics.",
                    examples=[
                        "What's that episode where a journalist, an artist, a musician are invited to a billionaire's house and there's a meteor at the end?"
                    ],
                ),
            ] = None,
        ) -> MediaSearchResponse:
            """Find media items (movies or episodes) based on various criteria."""
            collection = str(media_type)
            if collection not in ("movies", "episodes"):
                if collection in ["movie", "episode"]:
                    collection = str(media_type) + "s"
                else:
                    raise ValueError(
                        "media_type must be 'movies' or 'episodes'")
            query_filter: Filter | None = None
            context = ExplainContext()
            query = ""
            if query_filter := build_filters(
                genres=with_genre,
                directors=directed_by,
                writers=written_by,
                actors=starring,
                aired_date=MinMax(
                    minimum=aired_after,
                    maximum=aired_before,
                ),
                series=series,
                season=season,
                episode=episode,
                rating=MinMax(minimum=rating_min, maximum=rating_max),
                watched=watched,
            ):
                context.outer_filter = query_filter

            if similar_to:
                if theme_or_story:
                    context.prefetch = Prefetch(
                        prefetch=context.prefetch,
                        query=KnowledgeBase.instance().document(theme_or_story),
                    )
                context.query_kind = "similar"
                context.positive_point_ids = [
                    p.id
                    for p in await filter_points(
                        collection, filters=PlexMediaQuery(title=similar_to)
                    )
                ]
                if context.positive_point_ids:
                    context.query = RecommendQuery(
                        recommend=RecommendInput(
                            positive=cast(list[VectorInput],
                                          context.positive_point_ids)
                        )
                    )
                    if context.outer_filter:
                        context.outer_filter = Filter(
                            must=context.outer_filter.must if context.outer_filter.must else [],
                            must_not=[
                                *cast(
                                    list["Condition"],
                                    (
                                        context.outer_filter.must_not
                                        if context.outer_filter.must_not
                                        else []
                                    ),
                                ),
                                FieldCondition(
                                    key="title",
                                    match=MatchValue(value=similar_to),
                                ),
                            ],
                            should=(
                                context.outer_filter.should if context.outer_filter.should else []
                            ),
                        )
                    else:
                        context.outer_filter = Filter(
                            must_not=[
                                FieldCondition(
                                    key="title",
                                    match=MatchValue(value=similar_to),
                                )
                            ]
                        )
                else:
                    query = similar_to
            elif theme_or_story:
                context.query = KnowledgeBase.instance().document(theme_or_story)

            if summary:
                context.prefetch = Prefetch(
                    prefetch=context.prefetch,
                    query=sparse_from_text(summary),
                    using="sparse",
                )
                query = query + " " + summary

            context.query = KnowledgeBase.instance().document(query)

            _LOGGER.info(
                f'Filtering points with conditions: {json.dumps({
                    "collection_name": collection,
                    "prefetch": context.prefetch.model_dump() if context.prefetch else None,
                    "query": context.query.model_dump() if context.query and isinstance(context.query, BaseModel) else None,
                    "using": "dense",
                    "limit": (limit if limit is not None else 10),
                    "offset": (offset if offset is not None else None),
                    "with_payload": True,
                }, indent=2)}'
            )

            results = await KnowledgeBase.instance().qdrant_client.query_points(
                collection_name=collection,
                prefetch=context.prefetch,
                query=context.query,
                using="dense",
                limit=(limit if limit is not None else 10),
                offset=(offset if offset is not None else None),
                with_payload=True,
            )

            _LOGGER.info(
                f"Found {len(results.points)} points matching the query and filters.")
            _LOGGER.info(json.dumps(results.model_dump(), indent=2))
            must = cast(
                list[Condition],
                context.outer_filter.must if context.outer_filter is not None else [],
            )
            should = cast(
                list[Condition],
                context.outer_filter.should if context.outer_filter is not None else [],
            )
            return MediaSearchResponse(
                results=[
                    point_to_media_result(PlexMediaPayload, point, context)
                    for point in results.points[: (limit if limit else None) or 10]
                ],
                total=len(results.points),
                used_intent="auto",
                used_scope=media_type,
                diagnostics=Diagnostics(
                    retrieval=Retrieval(dense_weight=1.0, sparse_weight=0.0),
                    reranker=None,
                    filters_applied=len(must or []) > 0 or len(
                        should or []) > 0,
                    fallback_used=context.query is not None,
                ),
            )

        @mcp.custom_route(
            path="/find_media",
            methods=["POST"],
            name="Find Media",
        )
        async def find_media_handler(
            request: Request,
        ):
            data = await request.json()
            response = await find_media(**data)
            return JSONResponse(response.model_dump_json(), status_code=200)

        @mcp.tool(
            name="find_media",
            description="Retrieve/recommend films and TV (series, seasons, episodes) by similarity, cast/crew, genres, keywords, or vague plot clues.",
            output_schema=MediaSearchResponse.model_json_schema(),
            annotations=ToolAnnotations(title="Search or Recommend Media"),
            tags={"plex", "media", "search", "recommend"},
        )
        async def tool(
            media_type: Annotated[
                str,
                Field(
                    title="Media Type",
                    description="The media library section to query 'movies' or 'episodes'.",
                    examples=["movies", "episodes"],
                ),
            ],
            similar_to: Annotated[
                Optional[str],
                Field(
                    default=None,
                    title="Similarity Title Anchor Filter",
                    description="The title of another media to use as a similarity anchor for the query: similar genres, plots, synopsis. etc/",
                ),
            ] = None,
            with_genre: Annotated[
                Optional[list[str] | str],
                Field(
                    default=None,
                    title="Genres",
                    description="Query for media that is categorized by these genres.",
                ),
            ] = None,
            directed_by: Annotated[
                Optional[list[str] | str],
                Field(
                    default=None,
                    title="Directors",
                    description="Query for media that was directed by these individuals.",
                ),
            ] = None,
            written_by: Annotated[
                Optional[list[str] | str],
                Field(
                    default=None,
                    title="Writers",
                    description="Query for media that was written by these individuals.",
                ),
            ] = None,
            starring: Annotated[
                Optional[list[str] | str],
                Field(
                    default=None,
                    title="Actors",
                    description="Query for media that the following individuals act in.",
                ),
            ] = None,
            summary: Annotated[
                Optional[str],
                Field(
                    default=None,
                    title="Plot",
                    description="Query for media with similar plots.",
                ),
            ] = None,
            aired_before: Annotated[
                Optional[date],
                Field(
                    default=None,
                    title="Aired Before",
                    description="Query for media aired before: <date> or <int> days ago",
                ),
            ] = None,
            aired_after: Annotated[
                Optional[date],
                Field(
                    default=None,
                    title="Aired After",
                    description="Query for media that aired after: <date> or <int> days ago",
                ),
            ] = None,
            series: Annotated[
                Optional[str],
                Field(
                    default=None,
                    title="Series",
                    description="Query for media that is part of this series.",
                ),
            ] = None,
            season: Annotated[
                Optional[list[int] | int],
                Field(
                    default=None,
                    title="Seasons",
                    description="Query for media with these season number.",
                ),
            ] = None,
            episode: Annotated[
                Optional[list[int] | int],
                Field(
                    default=None,
                    title="Episodes",
                    description="Query for media with these episode numbers.",
                ),
            ] = None,
            rating_min: Annotated[
                Optional[float],
                Field(
                    default=None,
                    title="Minimum Rating",
                    description="Query for media with a minimum rating.",
                ),
            ] = None,
            rating_max: Annotated[
                Optional[float],
                Field(
                    default=None,
                    title="Maximum Rating",
                    description="Query for media with a maximum rating.",
                ),
            ] = None,
            watched: Annotated[
                Optional[bool],
                Field(
                    default=None,
                    title="Watched Status",
                    description="Query for media that has or has not been watched.",
                ),
            ] = None,
            limit: Annotated[
                Optional[int],
                Field(
                    default=None,
                    title="Limit",
                    description="Query for a maximum number of results.",
                ),
            ] = None,
            offset: Annotated[
                Optional[int],
                Field(
                    default=None,
                    title="Offset",
                    description="Query for the number of results to skip.",
                ),
            ] = None,
            theme_or_story: Annotated[
                Optional[str],
                Field(
                    default=None,
                    title="Theme or story",
                    description="Query based on theme, story or other vague characteristics.",
                    examples=[
                        "What's that episode where a journalist, an artist, a musician are invited to a billionaire's house and there's a meteor at the end?"
                    ],
                ),
            ] = None,
        ) -> MediaSearchResponse:
            """Find media items (movies or episodes) based on various criteria."""
            return await find_media(
                media_type=media_type,
                similar_to=similar_to,
                with_genre=with_genre,
                summary=summary,
                directed_by=directed_by,
                starring=starring,
                written_by=written_by,
                aired_before=aired_before,
                aired_after=aired_after,
                series=series,
                season=season,
                episode=episode,
                rating_min=rating_min,
                rating_max=rating_max,
                watched=watched,
                limit=limit,
                offset=offset,
                theme_or_story=theme_or_story,
            )
