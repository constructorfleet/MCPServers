import logging
import os
from datetime import date
from typing import Annotated, Callable, Literal, Optional, Type

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import Field
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from plex.knowledge.collection import Collection
from plex.knowledge.types import (
    MediaSearchResponse,
    PlexMediaPayload,
    PlexMediaQuery,
    TModel,
)
from plex.knowledge.utils import (
    ensure_collection,
    ensure_payload_indexes,
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

    async def get_collection(self, media_type: str | None) -> Collection[PlexMediaPayload] | None:
        if media_type is not None:
            if media_type in ["movie", "episode"]:
                media_type = f"{media_type}s"
            if media_type not in ["movies", "episodes"]:
                raise ValueError("Media type must be 'movies' or 'episodes'")
        else:
            media_type = "media"
        if media_type == "movies":
            return await self.ensure_movies()
        if media_type == "episodes":
            return await self.ensure_episodes()
        return await self.ensure_media()

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
                Optional[list[int]],
                Field(
                    default=None,
                    title="Seasons",
                    description="Query for media with this season number.",
                ),
            ] = None,
            episode: Annotated[
                Optional[list[int]],
                Field(
                    default=None,
                    title="Episodes",
                    description="Query for media with this episode numbers.",
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
            collection = await self.get_collection(media_type)
            if not collection:
                raise ValueError(f"Unknown media type {media_type}")
            return await collection.find_media(
                similar_to=similar_to,
                with_genre=with_genre,
                directed_by=directed_by,
                written_by=written_by,
                starring=starring,
                aired_after=aired_after,
                aired_before=aired_before,
                series=series,
                season=season if isinstance(season, list) else [
                    season] if season else None,
                episode=episode if isinstance(episode, list) else [
                    episode] if episode else None,
                rating_min=rating_min,
                rating_max=rating_max,
                watched=watched,
                limit=limit,
                offset=offset,
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
            name="media_query",
            description="Query for films and TV (series, seasons, episodes) by similarity, cast/crew, genres, keywords, or vague plot clues to recommend or retrieve information about.",
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
                season=season if isinstance(season, list) else [
                    season] if season else None,
                episode=episode if isinstance(episode, list) else [
                    episode] if episode else None,
                rating_min=rating_min,
                rating_max=rating_max,
                watched=watched,
                limit=limit,
                offset=offset,
                theme_or_story=theme_or_story,
            )
