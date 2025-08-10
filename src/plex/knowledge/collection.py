import asyncio
import json
from typing import Callable, Generic, Type, Optional

from pydantic import ConfigDict
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchPhrase,
    MatchValue,
    MinShould,
)
from qdrant_client.models import (
    CollectionInfo,
    Condition,
    Document,
    Prefetch,
    Vector,
)

from plex.knowledge.types import TModel
from plex.knowledge.utils import (
    _sparse_from_text,
    apply_diversity,
    fuse_two_pass,
    heuristic_rerank,
)
from plex.knowledge.types import DataPoint, PlexMediaPayload

import logging

_LOGGER = logging.getLogger(__name__)


class Collection(CollectionInfo, Generic[TModel]):
    """Enhanced Qdrant collection with advanced search capabilities.

    This class provides a high-level interface for searching and managing
    Qdrant collections with support for hybrid search, reranking, diversity,
    and various fusion methods.
    """

    qdrant_client: AsyncQdrantClient
    payload_class: Type[TModel]
    make_document: Callable[[TModel], str]
    name: str
    model: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def upsert_points(self, points: types.Points, wait: bool = True):
        """Insert or update points in the Qdrant collection.

        Args:
            points: Points to upsert
            wait: Whether to wait for the operation to complete
        """
        await self.qdrant_client.upsert(collection_name=self.name, points=points, wait=wait)

    async def upsert_data(
        self,
        data: list[TModel],
        id_getter: Callable[[TModel], Optional[int | str]] = lambda x: getattr(x, "id", None),
        wait: bool = True,
    ):
        """Insert or update typed data objects in the collection.

        This method handles the conversion of typed data objects to vectors and
        uploads them to Qdrant with both dense and sparse vector representations.

        Args:
            data: List of data objects to upsert
            id_getter: Function to extract ID from data objects
            wait: Whether to wait for the operation to complete
        """
        ids: list[int | str] = []
        vectors: list[dict[str, Vector]] = []
        payloads: list[dict] = []
        for item in data:
            ids.append(id_getter(item) or "")
            doc_text = self.make_document(item)
            vectors.append(
                {
                    "dense": Document(
                        text=doc_text, model=self.model, options={"cuda": True, "device_ids": [1]}
                    ),
                    "sparse": _sparse_from_text(doc_text),
                }
            )
            payloads.append(item.model_dump(exclude_unset=True))
        await asyncio.to_thread(
            self.qdrant_client.upload_collection,
            collection_name=self.name,
            ids=ids,
            vectors=vectors,
            payload=payloads,
            parallel=10,
            wait=wait,
        )

    async def search(
        self,
        data: TModel,
        limit: int | None = None,
        enable_server_fusion: bool = False,
        fusion_prelimit: int = 200,
        enable_rerank: bool = False,
        enable_diversity: bool = False,
        enable_two_pass_fusion: bool = False,
        fusion_dense_weight: float = 0.7,
        fusion_sparse_weight: float = 0.3,
        reranker_name: str = "heuristic-v1",
    ) -> list[DataPoint[TModel]]:
        """Search the collection using structured media data.

        This method performs hybrid search combining dense and sparse vectors with
        structured filters based on the media payload. It supports various fusion
        methods and optional reranking and diversity filtering.

        Args:
            data: Media payload to search for
            limit: Maximum number of results to return

        Returns:
            list[DataPoint[TModel]]: Search results sorted by relevance
        """
        shoulds: list[Condition] = []
        musts: list[Condition] = []
        if data.title:
            shoulds.append(FieldCondition(key="title", match=MatchPhrase(phrase=data.title)))
        if data.show_title:
            shoulds.append(
                FieldCondition(key="show_title", match=MatchPhrase(phrase=data.show_title))
            )
        if data.genres:
            for genre in data.genres:
                shoulds.append(FieldCondition(key="genres", match=MatchPhrase(phrase=genre)))
        if data.watched is not None:
            musts.append(FieldCondition(key="watched", match=MatchValue(value=data.watched)))
        if data.actors:
            for actor in data.actors:
                musts.append(FieldCondition(key="actors", match=MatchPhrase(phrase=actor)))
        if data.directors:
            for director in data.directors:
                musts.append(FieldCondition(key="directors", match=MatchPhrase(phrase=director)))
        query_filter = Filter(
            must=musts if len(musts) > 0 else None,
            min_should=(
                MinShould(
                    conditions=shoulds,
                    min_count=1,
                )
                if len(shoulds) > 0
                else None
            ),
            should=None,
            must_not=None,
        )
        # Server-side fusion path (prefetch + fusion), if enabled
        if enable_server_fusion:
            doc_text = PlexMediaPayload.document(data)
            dense_doc = Document(
                text=doc_text, model=self.model, options={"cuda": True}
            )  # type: ignore
            sparse_vec = _sparse_from_text(doc_text)
            prelimit = max(fusion_prelimit, (limit or 50) * 3)
            prefetch = [
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=prelimit,
                    filter=query_filter if query_filter else None,
                ),
                Prefetch(
                    # Let the client serialize the dense vector; if not supported, you can embed client-side and pass raw vector here
                    query=dense_doc,
                    using="dense",
                    limit=prelimit,
                    filter=query_filter if query_filter else None,
                ),
            ]
            result = await self.qdrant_client.query_points(
                collection_name=self.name,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit or 10000,
            )
            points = [
                DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
                for p in result.points
            ]
            if enable_rerank:
                points = heuristic_rerank(data, points)
                self.reranker_name = self.reranker_name or "heuristic-v1"
            if enable_diversity:
                points = apply_diversity(points, limit or 10000, 0.3, 1)
            return points
        if enable_two_pass_fusion:
            # Two-pass: run dense-only and sparse-only, fuse client-side
            doc_text = self.make_document(data)
            dense_doc = Document(
                text=doc_text, model=self.model, options={"cuda": True}
            )  # type: ignore
            sparse_vec = _sparse_from_text(doc_text)
            prelimit = max(fusion_prelimit, (limit or 50) * 3)
            d_res, s_res = await asyncio.gather(
                *[
                    self.qdrant_client.query_points(
                        collection_name=self.name,
                        query=dense_doc,
                        using="dense",
                        query_filter=query_filter,
                        limit=prelimit,
                        with_payload=True,
                    ),
                    self.qdrant_client.query_points(
                        collection_name=self.name,
                        query=sparse_vec,
                        using="sparse",
                        query_filter=query_filter,
                        limit=prelimit,
                        with_payload=True,
                    ),
                ]
            )
            fused_points = fuse_two_pass(
                list(d_res.points),
                list(s_res.points),
                fusion_dense_weight,
                fusion_sparse_weight,
            )
            # Truncate to requested limit and adapt to DataPoint
            points = [
                DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
                for p in fused_points[: (limit or 10000)]
            ]
            if enable_rerank:
                points = heuristic_rerank(data, points)
                self.reranker_name = self.reranker_name or "heuristic-v1"
            if enable_diversity:
                points = apply_diversity(points, limit or 10000, 0.3, 1)
            return points
        # else fallback to built-in hybrid heuristics
        # Heuristic: short or fielded queries benefit more from sparse; long summaries lean dense
        # query_hint = " ".join(
        #     [data.title or "", data.summary or "", data.show_title or ""]).strip()
        # wc = _word_count(query_hint)
        use_sparse = False
        # (
        #     True
        #     if (wc <= 12 or any([data.title, data.genres, data.actors, data.directors]))
        #     else True
        # )
        doc_text = self.make_document(data)
        query = Document(text=doc_text, model=self.model, options={"cuda": True})  # type: ignore
        sparse = _sparse_from_text(doc_text)
        vecs = []
        for x in self.qdrant_client._embed_documents([query.text], self.model):
            vecs.append(x[1])
        _LOGGER.warn(
            f'Searching collection {self.name} with: {
                json.dumps({
                    "collection_name": self.name,
                    "query": sparse.model_dump() if use_sparse else vecs[0],
                    "using": "sparse" if use_sparse else "dense",
                    "prefetch": (
                        Prefetch(query=sparse, using="sparse").model_dump()
                        if not use_sparse and sparse is not None
                        else None
                    ),
                    "query_filter": query_filter.model_dump() if query_filter else None,
                    "limit": limit or 10000,
                    "with_payload": True,
                }, indent=2)
            }'
        )
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            query=vecs[0],  # sparse if use_sparse else query,
            using="dense",  # "sparse" if use_sparse else "dense",
            # prefetch=(
            #     Prefetch(query=sparse, using="sparse")
            #     if not use_sparse and sparse is not None
            #     else None
            # ),
            # query_filter=query_filter,
            limit=limit or 10000,
            with_payload=True,
        )
        _LOGGER.warn(f"Qdrant query result: {json.dumps(result.model_dump(), indent=2)}")
        points = sorted(
            [
                DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
                for p in result.points
            ],
            key=lambda x: x.score,
            reverse=True,
        )
        if enable_rerank:
            points = heuristic_rerank(data, points)
            self.reranker_name = reranker_name or "heuristic-v1"
        if enable_diversity:
            points = apply_diversity(points, limit or 10000, 0.3, 1)
        return points

    async def query(
        self,
        query: str,
        limit: int | None = None,
        enable_two_pass_fusion: bool = False,
        fusion_prelimit: int = 200,
        fusion_dense_weight: float = 0.7,
        fusion_sparse_weight: float = 0.3,
        enable_rerank: bool = False,
        enable_diversity: bool = False,
    ) -> list[DataPoint[TModel]]:
        """Perform free-text search on the collection.

        Args:
            query: Free-text search query
            limit: Maximum number of results to return

        Returns:
            list[DataPoint[TModel]]: Search results sorted by relevance
        """
        doc = Document(text=query, model=self.model, options={"cuda": True})
        if enable_two_pass_fusion:
            sparse = _sparse_from_text(query)
            prelimit = max(fusion_prelimit, (limit or 50) * 3)
            d_res, s_res = await asyncio.gather(
                *[
                    self.qdrant_client.query_points(
                        collection_name=self.name,
                        query=doc,
                        limit=prelimit,
                        using="dense",
                        with_payload=True,
                    ),
                    self.qdrant_client.query_points(
                        collection_name=self.name,
                        sparse_vector=sparse,
                        limit=prelimit,
                        using="sparse",
                        with_payload=True,
                    ),
                ]
            )
            fused_points = fuse_two_pass(
                list(d_res.points),
                list(s_res.points),
                fusion_dense_weight,
                fusion_sparse_weight,
            )
            points = [
                DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
                for p in fused_points[: (limit or 10000)]
            ]
            if enable_rerank:
                q = PlexMediaPayload(
                    key=0,
                    title=query,
                    summary=query,
                    year=0,
                    rating=0.0,
                    watched=False,
                    type="movie",
                    genres=[],
                    actors=[],
                    studio="",
                    directors=[],
                    writers=[],
                    duration_seconds=0,
                    content_rating=None,
                    show_title=None,
                    season=None,
                    episode=None,
                )
                points = heuristic_rerank(q, points)
                self.reranker_name = self.reranker_name or "heuristic-v1"
            if enable_diversity:
                points = apply_diversity(points, limit or 10000, 0.3, 1)
            return points
        # else: built-in hybrid path
        sparse = _sparse_from_text(query)
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            query=doc,
            using="dense",
            prefetch=Prefetch(query=sparse, using="sparse"),
            limit=limit or 10000,
            with_payload=True,
        )
        points = sorted(
            [
                DataPoint.model_validate({"payload_class": PlexMediaPayload, **p.model_dump()})
                for p in result.points
            ],
            key=lambda x: x.score,
            reverse=True,
        )
        if enable_rerank:
            q = PlexMediaPayload(
                key=0,
                title=query,
                summary=query,
                year=0,
                rating=0.0,
                watched=False,
                type="movie",
                genres=[],
                actors=[],
                studio="",
                directors=[],
                writers=[],
                duration_seconds=0,
                content_rating=None,
                show_title=None,
                season=None,
                episode=None,
            )
            points = heuristic_rerank(q, points)
            self.reranker_name = self.reranker_name or "heuristic-v1"
        if enable_diversity:
            points = apply_diversity(points, limit or 10000, 0.3, 1)
        return points
