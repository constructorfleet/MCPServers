import os
from typing import Callable, Literal, Type, Optional

from qdrant_client.async_qdrant_client import AsyncQdrantClient

from plex.knowledge.collection import Collection
from plex.knowledge.types import PlexMediaPayload, PlexMediaQuery, TModel
from plex.knowledge.utils import ensure_collection, ensure_payload_indexes
import logging

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
