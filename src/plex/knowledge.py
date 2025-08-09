import asyncio
import hashlib
import json
import logging
import math
import re
from collections import Counter
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, create_model
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.common_types import Points
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchPhrase,
    MatchValue,
    MinShould,
    MultExpression,
    PayloadSchemaType,
    SparseVectorParams,
    SumExpression,
    VectorParams,
)
from qdrant_client.models import (
    CollectionInfo,
    Document,
    FormulaQuery,
    Prefetch,
    ScoredPoint,
    SparseVector,
    Vector,
)

_LOGGER = logging.getLogger(__name__)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "it",
    "as",
    "from",
}


def _sparse_from_text(text: str) -> SparseVector:
    """Build a Qdrant SparseVector from text using a stable hashed bag-of-words.

    - Tokenize on alphanumerics, lowercase
    - Drop short/common stopwords
    - Weight = 1 + ln(tf)
    - Index = blake2b(token) -> int -> modulo 2^31-1
    - Merge collisions (sum values) and ensure unique indices
    """
    if not text:
        return SparseVector(indices=[], values=[])
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    counts = Counter(t for t in tokens if t not in _STOPWORDS and len(t) > 1)
    if not counts:
        return SparseVector(indices=[], values=[])
    index_to_val: dict[int, float] = {}
    for tok, tf in counts.items():
        h = (
            int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")
            % 2147483647
        )
        val = 1.0 + math.log(tf)
        index_to_val[h] = index_to_val.get(h, 0.0) + val
    indices = list(index_to_val.keys())
    values = [index_to_val[i] for i in indices]
    return SparseVector(indices=indices, values=values)


def _word_count(s: str | None) -> int:
    """Count the number of words in a string.

    Args:
        s: String to count words in, can be None

    Returns:
        int: Number of words, 0 if string is None or empty
    """
    if not s:
        return 0
    return len([w for w in s.split() if w.strip()])


# --- Qdrant collection and index helpers ---
async def ensure_collection(
    client: AsyncQdrantClient,
    name: str,
    dim: int,
) -> None:
    """Create the collection if it doesn't exist, with dense + sparse vectors."""
    _LOGGER.warning(f"Ensuring collection '{name}' exists with {dim} dimensions.")
    collections = await client.get_collections()
    if any(c.name == name for c in collections.collections):
        return
        # collection = await client.get_collection(name)
        # if not collection.config.params.vectors:
        #     recreate = True
        # elif not isinstance(collection.config.params.vectors, dict):
        #     recreate = True
        # elif not collection.config.params.vectors.get("dense"):
        #     recreate = True
        # elif collection.config.params.vectors["dense"].size != dim:
        #     recreate = True
        # elif not collection.config.params.sparse_vectors:
        #     recreate = True
        # elif not isinstance(collection.config.params.sparse_vectors, dict):
        #     recreate = True
        # elif collection.config.params.sparse_vectors.get("sparse"):
        #     recreate = True
        # if not recreate:
        #     return
        # _LOGGER.warning(f"Deleting collection '{name}'")
        # await client.delete_collection(name)
    _LOGGER.warning(f"Creating collection '{name}'")
    await client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
        on_disk_payload=True,
    )


async def ensure_payload_indexes(
    client: AsyncQdrantClient, name: str, is_episode: bool = False
) -> None:
    """Create payload indexes used by filters and lexical search."""
    # TEXT indexes
    await client.create_payload_index(
        collection_name=name, field_name="title", field_schema=PayloadSchemaType.TEXT
    )
    await client.create_payload_index(
        collection_name=name, field_name="summary", field_schema=PayloadSchemaType.TEXT
    )
    if is_episode:
        await client.create_payload_index(
            collection_name=name, field_name="show_title", field_schema=PayloadSchemaType.TEXT
        )

    # KEYWORD indexes
    for fld in [
        "genres",
        "actors",
        "directors",
        "writers",
        "studio",
        "collection",
        "content_rating",
        "status",
    ]:
        await client.create_payload_index(
            collection_name=name, field_name=fld, field_schema=PayloadSchemaType.KEYWORD
        )
    if is_episode:
        await client.create_payload_index(
            collection_name=name, field_name="show_status", field_schema=PayloadSchemaType.KEYWORD
        )

    # NUMERIC / DATETIME / BOOL
    await client.create_payload_index(
        collection_name=name, field_name="rating", field_schema=PayloadSchemaType.FLOAT
    )
    await client.create_payload_index(
        collection_name=name, field_name="duration_seconds", field_schema=PayloadSchemaType.INTEGER
    )
    await client.create_payload_index(
        collection_name=name, field_name="date", field_schema=PayloadSchemaType.DATETIME
    )
    await client.create_payload_index(
        collection_name=name, field_name="watched", field_schema=PayloadSchemaType.BOOL
    )
    if is_episode:
        await client.create_payload_index(
            collection_name=name, field_name="season", field_schema=PayloadSchemaType.INTEGER
        )
        await client.create_payload_index(
            collection_name=name, field_name="episode", field_schema=PayloadSchemaType.INTEGER
        )


# --- Tool API schema models ---
class ToolResult(BaseModel):
    """Result model for media search tools, representing a single media item.

    This model represents the standardized output format for media search results,
    containing information about movies, TV series, or episodes.
    """

    result_type: Literal["series", "episode", "movie"]
    title: Optional[str] = None
    year: Optional[int] = None
    status: Optional[str] = None
    series_id: Optional[str] = None
    movie_id: Optional[str] = None
    genres: Optional[list[str]] = None
    season_count: Optional[int] = None
    episode_count: Optional[int] = None
    network: Optional[str] = None
    showrunner: Optional[list[str]] = None
    cast: Optional[list[str]] = None
    crew: Optional[dict[str, list[str]]] = None
    similar_titles: Optional[list[str]] = None
    synopsis: Optional[str] = None
    scores: Optional[dict[str, float]] = None
    why: Optional[str] = None


class ToolResponse(BaseModel):
    """Response wrapper for tool API calls containing search results and metadata.

    This model wraps the search results with additional metadata about how the
    search was performed and diagnostic information.
    """

    results: list[ToolResult]
    total: int
    used_intent: str
    used_scope: str
    diagnostics: dict


T = TypeVar("T")


def make_optional_model(model_name: str, model: Type[T]) -> Type[BaseModel]:
    """Create a new Pydantic model with all fields from the source model made optional.

    Args:
        model_name: Name for the new model class
        model: Source model type to create optional version from

    Returns:
        Type[BaseModel]: New model class with all optional fields
    """
    fields = get_type_hints(model)
    optional_fields = {k: (Optional[v], None) for k, v in fields.items()}
    return create_model(model_name, kwargs=optional_fields)


class PlexMediaPayload(BaseModel):
    """Data model for Plex media items (movies and episodes).

    This model represents the core data structure for media items stored in
    the knowledge base, containing all relevant metadata for movies and TV episodes.
    """

    key: int
    title: str
    summary: str
    year: int
    rating: float
    watched: bool
    type: Literal["movie"] | Literal["episode"]
    genres: list[str]
    actors: list[str]
    studio: str
    directors: list[str]
    writers: list[str]
    duration_seconds: int
    content_rating: Optional[str]
    show_title: Optional[str]
    season: Optional[int]
    episode: Optional[int]

    @classmethod
    def document(cls, item: "PlexMediaPayload") -> str:
        """Convert a media item to a searchable text document.

        Args:
            item: Media item to convert to document text

        Returns:
            str: Formatted text representation for search indexing
        """
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
        if item.type == "episode" and item.show_title:
            parts.append(f"Show: {item.show_title}")
            season_episode = []
            if item.season is not None:
                season_episode.append(f"Season {item.season}")
            if item.episode is not None:
                season_episode.append(f"Episode {item.episode}")
            if season_episode:
                parts.append(" ".join(season_episode))
        if item.rating is not None:
            parts.append(f"Rating: {item.rating}/10")
        return "\n".join(parts)


TModel = TypeVar("TModel", bound=PlexMediaPayload)


class PlexMediaQuery(PlexMediaPayload):
    """Query model for searching Plex media with optional fields.

    This model extends PlexMediaPayload but makes all fields optional,
    allowing for flexible search queries where any combination of fields
    can be specified.
    """

    key: Optional[int] = None  # type: ignore
    title: Optional[str] = None  # type: ignore
    summary: Optional[str] = None  # type: ignore
    year: Optional[int] = None  # type: ignore
    rating: Optional[float] = None  # type: ignore
    watched: Optional[bool] = None  # type: ignore
    type: Optional[  # type: ignore
        Literal["movie"] | Literal["episode"]  # type: ignore
    ] = None  # type: ignore
    genres: Optional[list[str]] = None  # type: ignore
    actors: Optional[list[str]] = None  # type: ignore
    studio: Optional[str] = None  # type: ignore
    directors: Optional[list[str]] = None  # type: ignore
    writers: Optional[list[str]] = None  # type: ignore
    duration_seconds: Optional[int] = None  # type: ignore
    content_rating: Optional[str] = None  # type: ignore
    show_title: Optional[str] = None  # type: ignore
    season: Optional[int] = None  # type: ignore
    episode: Optional[int] = None  # type: ignore
    similar_to: Optional[int] = None  # type: ignore


class DataPoint(ScoredPoint, Generic[TModel]):
    """Enhanced ScoredPoint with typed payload access for search results.

    This class extends Qdrant's ScoredPoint to provide type-safe access to
    the payload data using the specified model class.
    """

    payload_class: Type[TModel]

    def payload_data(self) -> TModel:
        """Get the typed payload data from this point.

        Returns:
            TModel: Validated payload data as the specified model type

        Raises:
            ValueError: If no payload is available
        """
        if not self.payload:
            raise ValueError("No payload available")
        return self.payload_class.model_validate(self.payload)


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
    enable_rerank: bool = False
    reranker_name: Optional[str] = None
    enable_two_pass_fusion: bool = False
    fusion_dense_weight: float = 0.7
    fusion_sparse_weight: float = 0.3
    fusion_prelimit: int = 200
    # --- Server-fusion flags and knobs ---
    enable_server_fusion: bool = False
    server_fusion_method: Literal["rrf"] = "rrf"
    # --- Diversity knobs ---
    enable_diversity: bool = False
    diversity_lambda: float = 0.3  # 0=only diversity, 1=only score
    max_per_series: int = 1  # cap results per series/franchise

    def _fuse_two_pass(
        self,
        dense: list[ScoredPoint],
        sparse: list[ScoredPoint],
        dense_w: float,
        sparse_w: float,
        rrf_k: int = 60,
    ) -> list[ScoredPoint]:
        """Fuse dense and sparse search results using weighted Reciprocal Rank Fusion.

        Args:
            dense: Results from dense vector search
            sparse: Results from sparse vector search
            dense_w: Weight for dense results
            sparse_w: Weight for sparse results
            rrf_k: RRF parameter (typically 60)

        Returns:
            list[ScoredPoint]: Fused results sorted by combined score
        """
        # Build rank maps
        d_rank = {p.id: i for i, p in enumerate(dense, start=1)}
        s_rank = {p.id: i for i, p in enumerate(sparse, start=1)}
        # Union ids
        ids = list({*d_rank.keys(), *s_rank.keys()})
        # Index lookup
        by_id: dict = {}
        for p in dense:
            by_id.setdefault(p.id, p)
        for p in sparse:
            by_id.setdefault(p.id, p)
        # Weighted RRF score
        fused: list[tuple[float, ScoredPoint]] = []
        for pid in ids:
            dr = d_rank.get(pid)
            sr = s_rank.get(pid)
            d_score = (1.0 / (rrf_k + dr)) if dr is not None else 0.0
            s_score = (1.0 / (rrf_k + sr)) if sr is not None else 0.0
            score = dense_w * d_score + sparse_w * s_score
            fused.append((score, by_id[pid]))
        fused.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in fused]

    def _series_key(self, payload: dict) -> Optional[str]:
        """Extract a series/collection identifier from payload data.

        Args:
            payload: Media item payload data

        Returns:
            Optional[str]: Series identifier or None if not found
        """
        return str(payload.get("collection") or payload.get("show_title") or "") or None

    def _title_shingles(self, title: Optional[str]) -> set[str]:
        """Generate 3-gram shingles from a title for similarity comparison.

        Args:
            title: Title text to process

        Returns:
            set[str]: Set of 3-gram shingles (or individual tokens if < 3 tokens)
        """
        if not title:
            return set()
        t = "".join(ch.lower() if ch.isalnum() else " " for ch in title)
        tokens = [tok for tok in t.split() if tok]
        # 3-gram shingles over tokens
        if len(tokens) < 3:
            return set(tokens)
        return {" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)}

    def _sim_items(self, a: DataPoint[TModel], b: DataPoint[TModel]) -> float:
        """Calculate similarity between two media items for diversity filtering.

        Args:
            a: First media item
            b: Second media item

        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        pa, pb = a.payload or {}, b.payload or {}
        # Title similarity via shingles
        ta = self._title_shingles(pa.get("title"))
        tb = self._title_shingles(pb.get("title"))
        title_j = (len(ta & tb) / len(ta | tb)) if (ta or tb) else 0.0
        # Genre similarity
        ga = set((pa.get("genres") or []))
        gb = set((pb.get("genres") or []))
        genre_j = (len(ga & gb) / len(ga | gb)) if (ga or gb) else 0.0
        # Weighted blend for similarity (not score)
        return 0.7 * title_j + 0.3 * genre_j

    def _apply_diversity(
        self, points: list["DataPoint[TModel]"], limit: int
    ) -> list["DataPoint[TModel]"]:
        """Apply diversity filtering using Maximal Marginal Relevance (MMR).

        This method reduces redundancy in search results by selecting items that
        balance relevance with diversity, preventing too many similar results.

        Args:
            points: Input search results
            limit: Maximum number of results to return

        Returns:
            list[DataPoint[TModel]]: Diversified results up to the limit
        """
        if limit <= 0 or not points:
            return points
        lam = max(0.0, min(1.0, self.diversity_lambda))
        selected: list[DataPoint[TModel]] = []
        series_counts: dict[str, int] = {}
        # Greedy MMR selection
        # Assume input is roughly sorted by model-score descending
        pool = points[:]
        while pool and len(selected) < limit:
            best_idx = 0
            best_val = float("-inf")
            for i, cand in enumerate(pool):
                payload = cand.payload or {}
                skey = self._series_key(payload)
                if skey and series_counts.get(skey, 0) >= self.max_per_series:
                    continue
                rel = float(getattr(cand, "score", 0.0) or 0.0)
                div = 0.0
                if selected:
                    div = max(self._sim_items(cand, s) for s in selected)
                val = lam * rel - (1.0 - lam) * div
                if val > best_val:
                    best_val = val
                    best_idx = i
            choice = pool.pop(best_idx)
            payload = choice.payload or {}
            skey = self._series_key(payload)
            if skey:
                series_counts[skey] = series_counts.get(skey, 0) + 1
            selected.append(choice)
        return selected

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _point_to_tool_result(self, p: ScoredPoint, why: Optional[str] = None) -> ToolResult:
        """Convert a Qdrant search result to a standardized ToolResult.

        Args:
            p: Scored point from Qdrant search
            why: Optional explanation of why this result matched

        Returns:
            ToolResult: Standardized result format for API responses
        """
        payload = p.payload or {}
        item = self.payload_class.model_validate(payload)
        # Determine result type (we only store movies/episodes today)
        rtype: Literal["series", "episode", "movie"]
        if getattr(item, "type", None) == "episode":
            rtype = "episode"
        elif getattr(item, "type", None) == "series":
            rtype = "series"
        else:
            rtype = "movie"
        # IDs
        movie_id = str(getattr(item, "key", None)) if rtype == "movie" else None
        series_id = None
        # Prefer a stable collection/series/franchise key if present; fallback to show_title for episodes
        if payload.get("collection"):
            series_id = str(payload.get("collection"))
        elif rtype == "episode":
            series_id = str(getattr(item, "show_title", None) or "") or None
        # Crew mapping
        crew: dict[str, list[str]] = {}
        if getattr(item, "directors", None):
            crew["director"] = list(getattr(item, "directors"))
        if getattr(item, "writers", None):
            crew["writer"] = list(getattr(item, "writers"))
        # Scores mapping (we only have rating)
        scores = None
        if getattr(item, "rating", None) is not None:
            try:
                scores = {"rating": float(item.rating)}
            except Exception:
                _LOGGER.debug("Failed to convert rating to float")
        return ToolResult(
            result_type=rtype,
            title=getattr(item, "title", None),
            year=getattr(item, "year", None),
            status=(payload.get("show_status") if rtype == "episode" else payload.get("status")),
            series_id=series_id,
            movie_id=movie_id,
            genres=getattr(item, "genres", None),
            season_count=None,
            episode_count=None,
            network=payload.get("network"),
            showrunner=payload.get("showrunner"),
            cast=getattr(item, "actors", None),
            crew=crew or None,
            similar_titles=[],
            synopsis=getattr(item, "summary", None),
            scores=scores,
            why=why,
        )

    def _explain_match(self, q: PlexMediaPayload, item: PlexMediaPayload) -> Optional[str]:
        """Generate an explanation of why a search result matched the query.

        Args:
            q: Query payload
            item: Matched item payload

        Returns:
            Optional[str]: Human-readable explanation of the match, or None
        """
        reasons: list[str] = []
        if q.title and item.title and q.title.lower() in item.title.lower():
            reasons.append("title match")
        if q.show_title and item.show_title and q.show_title.lower() in item.show_title.lower():
            reasons.append("show match")
        if q.genres:
            common = sorted(set(q.genres).intersection(set(item.genres or [])))
            if common:
                reasons.append(f"genres overlap: {', '.join(common)}")
        if q.actors:
            common = sorted(set(q.actors).intersection(set(item.actors or [])))
            if common:
                reasons.append(f"cast overlap: {', '.join(common)}")
        if q.directors:
            common = sorted(set(q.directors).intersection(set(item.directors or [])))
            if common:
                reasons.append(f"director overlap: {', '.join(common)}")
        return ", ".join(reasons) or None

    def _heuristic_rerank(
        self, q: PlexMediaPayload, points: list[DataPoint[TModel]]
    ) -> list[DataPoint[TModel]]:
        """Rerank search results using domain-specific heuristics.

        This method applies additional scoring based on genre, cast, and director
        overlap to improve relevance of search results.

        Args:
            q: Original query payload
            points: Search results to rerank

        Returns:
            list[DataPoint[TModel]]: Reranked results
        """

        def jaccard(a: list[str] | None, b: list[str] | None) -> float:
            sa, sb = set(a or []), set(b or [])
            if not sa and not sb:
                return 0.0
            i = len(sa & sb)
            u = len(sa | sb)
            return (i / u) if u else 0.0

        rescored: list[tuple[float, DataPoint[TModel]]] = []
        for dp in points:
            item = dp.payload_data()
            base = float(getattr(dp, "score", 0.0) or 0.0)
            g = jaccard(getattr(q, "genres", None), getattr(item, "genres", None))
            c = jaccard(getattr(q, "actors", None), getattr(item, "actors", None))
            d = jaccard(getattr(q, "directors", None), getattr(item, "directors", None))
            blend = 0.75 * base + 0.15 * g + 0.06 * c + 0.04 * d
            rescored.append((blend, dp))
        rescored.sort(key=lambda t: t[0], reverse=True)
        return [dp for _, dp in rescored]

    async def upsert_points(self, points: Points, wait: bool = True):
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
                    "dense": Document(text=doc_text, model=self.model, options={"cuda": True}),
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
            wait=wait,
        )

    async def search(self, data: TModel, limit: int | None = None) -> list[DataPoint[TModel]]:
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
        shoulds: list = []
        musts: list = []
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
        )
        # Server-side fusion path (prefetch + fusion), if enabled
        if self.enable_server_fusion:
            doc_text = self.make_document(data)
            dense_doc = Document(
                text=doc_text, model=self.model, options={"cuda": True}
            )  # type: ignore
            sparse_vec = _sparse_from_text(doc_text)
            prelimit = max(self.fusion_prelimit, (limit or 50) * 3)
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
            if self.enable_rerank:
                points = self._heuristic_rerank(data, points)
                self.reranker_name = self.reranker_name or "heuristic-v1"
            if self.enable_diversity:
                points = self._apply_diversity(points, limit or 10000)
            return points
        if self.enable_two_pass_fusion:
            # Two-pass: run dense-only and sparse-only, fuse client-side
            doc_text = self.make_document(data)
            dense_doc = Document(
                text=doc_text, model=self.model, options={"cuda": True}
            )  # type: ignore
            sparse_vec = _sparse_from_text(doc_text)
            prelimit = max(self.fusion_prelimit, (limit or 50) * 3)
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
            fused_points = self._fuse_two_pass(
                list(d_res.points),
                list(s_res.points),
                self.fusion_dense_weight,
                self.fusion_sparse_weight,
            )
            # Truncate to requested limit and adapt to DataPoint
            points = [
                DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
                for p in fused_points[: (limit or 10000)]
            ]
            if self.enable_rerank:
                points = self._heuristic_rerank(data, points)
                self.reranker_name = self.reranker_name or "heuristic-v1"
            if self.enable_diversity:
                points = self._apply_diversity(points, limit or 10000)
            return points
        # else fallback to built-in hybrid heuristics
        # Heuristic: short or fielded queries benefit more from sparse; long summaries lean dense
        query_hint = " ".join([data.title or "", data.summary or "", data.show_title or ""]).strip()
        wc = _word_count(query_hint)
        use_sparse = (
            True
            if (wc <= 12 or any([data.title, data.genres, data.actors, data.directors]))
            else True
        )
        doc_text = self.make_document(data)
        query = Document(text=doc_text, model=self.model, options={"cuda": True})  # type: ignore
        sparse = _sparse_from_text(doc_text)
        _LOGGER.warn(
            f"Searching collection {self.name} with filter: {json.dumps(query_filter.model_dump(exclude_none=True), indent=2)} and query: {json.dumps(query.model_dump(exclude_none=True), indent=2)}"
        )
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            query=sparse if use_sparse else query,
            using="sparse" if use_sparse else "dense",
            prefetch=(
                Prefetch(query=sparse, using="sparse")
                if not use_sparse and sparse is not None
                else None
            ),
            query_filter=query_filter,
            limit=limit or 10000,
            with_payload=True,
        )
        points = sorted(
            [
                DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
                for p in result.points
            ],
            key=lambda x: x.score,
            reverse=True,
        )
        if self.enable_rerank:
            points = self._heuristic_rerank(data, points)
            self.reranker_name = self.reranker_name or "heuristic-v1"
        if self.enable_diversity:
            points = self._apply_diversity(points, limit or 10000)
        return points

    async def query_by_id_as_tool(
        self,
        point_id: Union[int, str],
        limit: int | None = None,
        used_intent: str = "by_id",
        used_scope: str = "auto",
    ) -> ToolResponse:
        """Query for similar items based on a specific point ID.

        Args:
            point_id: ID of the point to find similar items for
            limit: Maximum number of results to return
            used_intent: Intent used for this query (for diagnostics)
            used_scope: Scope used for this query (for diagnostics)

        Returns:
            ToolResponse: Formatted response with similar items
        """
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            query=point_id,
            limit=limit or 10,
        )
        points = [
            DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
            for p in result.points
        ]
        results = [self._point_to_tool_result(dp, why=None) for dp in points]
        diagnostics = {
            "retrieval": {"dense_weight": 1.0, "sparse_weight": 0.0},
            "reranker": None,
            "filters_applied": False,
            "fallback_used": False,
        }
        return ToolResponse(
            results=results,
            total=len(results),
            used_intent=used_intent,
            used_scope=used_scope,
            diagnostics=diagnostics,
        )

    async def recommend_as_tool(
        self,
        positive: list[Union[int, str, list[float]]],
        negative: Optional[list[Union[int, str, list[float]]]] = None,
        limit: int | None = None,
        used_intent: str = "recommend",
        used_scope: str = "auto",
    ) -> ToolResponse:
        """Generate recommendations based on positive and negative examples.

        Args:
            positive: List of positive examples (IDs or vectors)
            negative: Optional list of negative examples (IDs or vectors)
            limit: Maximum number of recommendations to return
            used_intent: Intent used for this query (for diagnostics)
            used_scope: Scope used for this query (for diagnostics)

        Returns:
            ToolResponse: Formatted response with recommendations
        """
        # The client passes arbitrary dict as query; Qdrant accepts {"recommend": {"positive": [...], "negative": [...]}}
        q: dict[str, Any] = {"recommend": {"positive": positive}}
        if negative:
            q["recommend"]["negative"] = negative
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            query=q,  # type: ignore[arg-type]
            limit=limit or 10,
            with_payload=True,
        )
        points = [
            DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
            for p in result.points
        ]
        results = [self._point_to_tool_result(dp, why=None) for dp in points]
        diagnostics = {
            "retrieval": {"dense_weight": 1.0, "sparse_weight": 0.0},
            "reranker": None,
            "filters_applied": False,
            "fallback_used": False,
        }
        return ToolResponse(
            results=results,
            total=len(results),
            used_intent=used_intent,
            used_scope=used_scope,
            diagnostics=diagnostics,
        )

    async def search_as_tool_boosted(
        self,
        data: TModel,
        boosts: dict[str, float],
        limit: int | None = None,
        used_intent: str = "auto",
        used_scope: str = "auto",
    ) -> ToolResponse:
        """Search with boosted scoring for specific fields.

        This method performs search with additional boost scoring applied to
        specified fields that match the query payload.

        Args:
            data: Media payload to search for
            boosts: Dictionary mapping field names to boost weights
            limit: Maximum number of results to return
            used_intent: Intent used for this query (for diagnostics)
            used_scope: Scope used for this query (for diagnostics)

        Returns:
            ToolResponse: Formatted response with boosted search results
        """
        # Build dense prefetch from the structured payload’s document
        doc_text = self.make_document(data)
        dense_doc = Document(
            text=doc_text, model=self.model, options={"cuda": True}
        )  # type: ignore
        # Build formula: sum of $score + weighted matches on payload keys
        # Example boosts: {"genres": 0.5, "actors": 0.25}
        sum_terms: SumExpression = SumExpression(sum=["$score"])
        for key, w in boosts.items():
            # If the query payload has a value for this key, boost documents matching ANY of those values
            values = getattr(data, key, None)
            if not values:
                continue
            if not isinstance(values, list):
                values = [values]
            sum_terms.sum.append(
                MultExpression(
                    mult=[float(w), FieldCondition(key=key, match=MatchAny(any=list(values)))]
                )
            )
        result = await self.qdrant_client.query_points(
            collection_name=self.name,
            prefetch=Prefetch(
                query=dense_doc, limit=max(self.fusion_prelimit, (limit or 50) * 3), using="dense"
            ),
            query=FormulaQuery(formula=sum_terms),
            limit=limit or 10,
            with_payload=True,
        )
        points = [
            DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
            for p in result.points
        ]
        # Optional rerank on top
        if self.enable_rerank:
            points = self._heuristic_rerank(data, points)
            self.reranker_name = self.reranker_name or "heuristic-v1"
        results = [self._point_to_tool_result(dp, why=None) for dp in points]
        diagnostics = {
            "retrieval": {"dense_weight": 1.0, "sparse_weight": 0.0},
            "reranker": self.reranker_name,
            "filters_applied": True,
            "fallback_used": False,
        }
        return ToolResponse(
            results=results,
            total=len(results),
            used_intent=used_intent,
            used_scope=used_scope,
            diagnostics=diagnostics,
        )

    async def search_as_tool(
        self,
        data: TModel,
        limit: int | None = None,
        used_intent: str = "auto",
        used_scope: str = "auto",
    ) -> ToolResponse:
        """Search the collection and return results in tool response format.

        Args:
            data: Media payload to search for
            limit: Maximum number of results to return
            used_intent: Intent used for this query (for diagnostics)
            used_scope: Scope used for this query (for diagnostics)

        Returns:
            ToolResponse: Formatted response with search results and diagnostics
        """
        points = await self.search(data, limit=limit)
        results: list[ToolResult] = []
        for dp in points:
            item = dp.payload_data()
            why = self._explain_match(data, item)
            results.append(self._point_to_tool_result(dp, why=why))
        hint = " ".join([data.title or "", data.summary or "", data.show_title or ""]).strip()
        wc = _word_count(hint)
        if self.enable_two_pass_fusion:
            dense_w = self.fusion_dense_weight
            sparse_w = self.fusion_sparse_weight
        else:
            dense_w = 0.8 if wc > 12 else 0.7
            sparse_w = 0.2 if wc > 12 else 0.3
        diagnostics = {
            "retrieval": {"dense_weight": dense_w, "sparse_weight": sparse_w},
            "reranker": self.reranker_name,
            "filters_applied": True,
            "fallback_used": False,
        }
        return ToolResponse(
            results=results,
            total=len(results),
            used_intent=used_intent,
            used_scope=used_scope,
            diagnostics=diagnostics,
        )

    async def query(self, query: str, limit: int | None = None) -> list[DataPoint[TModel]]:
        """Perform free-text search on the collection.

        Args:
            query: Free-text search query
            limit: Maximum number of results to return

        Returns:
            list[DataPoint[TModel]]: Search results sorted by relevance
        """
        doc = Document(text=query, model=self.model, options={"cuda": True})
        if self.enable_two_pass_fusion:
            sparse = _sparse_from_text(query)
            prelimit = max(self.fusion_prelimit, (limit or 50) * 3)
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
            fused_points = self._fuse_two_pass(
                list(d_res.points),
                list(s_res.points),
                self.fusion_dense_weight,
                self.fusion_sparse_weight,
            )
            points = [
                DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
                for p in fused_points[: (limit or 10000)]
            ]
            if self.enable_rerank:
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
                points = self._heuristic_rerank(q, points)
                self.reranker_name = self.reranker_name or "heuristic-v1"
            if self.enable_diversity:
                points = self._apply_diversity(points, limit or 10000)
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
                DataPoint.model_validate({"payload_class": self.payload_class, **p.model_dump()})
                for p in result.points
            ],
            key=lambda x: x.score,
            reverse=True,
        )
        if self.enable_rerank:
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
            points = self._heuristic_rerank(q, points)
            self.reranker_name = self.reranker_name or "heuristic-v1"
        if self.enable_diversity:
            points = self._apply_diversity(points, limit or 10000)
        return points

    async def query_as_tool(
        self,
        query: str,
        limit: int | None = None,
        used_intent: str = "auto",
        used_scope: str = "auto",
    ) -> ToolResponse:
        """Perform free-text search and return results in tool response format.

        Args:
            query: Free-text search query
            limit: Maximum number of results to return
            used_intent: Intent used for this query (for diagnostics)
            used_scope: Scope used for this query (for diagnostics)

        Returns:
            ToolResponse: Formatted response with search results and diagnostics
        """
        points = await self.query(query, limit=limit)
        results = [self._point_to_tool_result(dp, why=None) for dp in points]
        wc = _word_count(query)
        if self.enable_two_pass_fusion:
            dense_w = self.fusion_dense_weight
            sparse_w = self.fusion_sparse_weight
        else:
            dense_w = 0.8 if wc > 12 else 0.6
            sparse_w = 0.2 if wc > 12 else 0.4
        diagnostics = {
            "retrieval": {"dense_weight": dense_w, "sparse_weight": sparse_w},
            "reranker": self.reranker_name,
            "filters_applied": False,
            "fallback_used": False,
        }
        return ToolResponse(
            results=results,
            total=len(results),
            used_intent=used_intent,
            used_scope=used_scope,
            diagnostics=diagnostics,
        )


class KnowledgeBase:
    """High-level interface for managing Plex media collections in Qdrant.

    This class provides methods for setting up and accessing different types
    of media collections (movies, episodes, etc.) with caching support.
    """

    def __init__(self, model: str, qdrant_host: str, qdrant_port: int):
        """Initialize the knowledge base.

        Args:
            model: Name of the embedding model to use
            qdrant_host: Hostname of the Qdrant server
            qdrant_port: Port of the Qdrant server
        """
        self.model = model or "text-embedding-ada-002"
        self.qdrant_client = AsyncQdrantClient(
            host=qdrant_host, port=qdrant_port, grpc_port=6334, prefer_grpc=True
        )
        self.model = model
        self._collection_cache: dict[str, Collection] = {}

    async def ensure_movies(self, dim: int | None = None) -> Collection[PlexMediaPayload]:
        """Ensure the movies collection exists with proper configuration.

        Args:
            dim: Dimension of the dense vector embeddings
        """
        await ensure_collection(
            self.qdrant_client, "movies", dim or self.qdrant_client.get_embedding_size(self.model)
        )
        await ensure_payload_indexes(self.qdrant_client, "movies", is_episode=False)
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
            self.qdrant_client, "episodes", dim or self.qdrant_client.get_embedding_size(self.model)
        )
        await ensure_payload_indexes(self.qdrant_client, "episodes", is_episode=True)
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
            self.qdrant_client, "media", dim or self.qdrant_client.get_embedding_size(self.model)
        )
        await ensure_payload_indexes(self.qdrant_client, "media", is_episode=False)
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
