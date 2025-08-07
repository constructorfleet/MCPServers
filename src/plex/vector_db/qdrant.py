"""Code for interacting with a vector database."""

from typing import List, Dict, Any
import httpx

from plex.utils.mixins import BaseAPIClient
from plex.vector_db.models import CollectionInfo, CreateCollection, GetCollection, ListCollections, UpdateCollection

class QdrantClient(BaseAPIClient):
    def __init__(
            self,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            qdrant_embedding_size: int = 384,
            qdrant_api_key: str | None = None,
        ):
        headers = {}
        if qdrant_api_key:
            headers["api-key"] = qdrant_api_key
        self.client = httpx.AsyncClient(base_url=f"http://{qdrant_host}:{qdrant_port}", headers=headers)
        self.embedding_size = qdrant_embedding_size

    async def _list_collections(self) -> List[str]:
        """List collections in the vector database."""
        response = await ListCollections.model_construct().with_client(self).send()
        return [col.name for col in response.collections]
    
    async def get_collection(self, collection_name: str) -> CollectionInfo:
        """Get collection info from the vector database."""
        response = await GetCollection.model_construct(collection_name=collection_name).with_client(self).send()
        return response

    async def create_collection(self, collection_name: str, schema: Dict[str, Any]) -> CollectionInfo:
        """Create a collection in the vector database."""
        await CreateCollection.model_construct(collection_name=collection_name, schema=schema).with_client(self).send()
        return await self.get_collection(collection_name)
    
    async def update_collection(self, collection_name: str, schema: Dict[str, Any]) -> CollectionInfo:
        """Update a collection in the vector database."""
        await UpdateCollection.model_construct(collection_name=collection_name, schema=schema).with_client(self).send()
        return await self.get_collection(collection_name)
