"""Milvus vector store — client wrapper with 3 collections.

Collections:
  - review_embeddings: All cleaned review vectors
  - pain_point_embeddings: Historical pain point vectors
  - historical_issues: Past issues + resolutions for RAG

Gracefully degrades if Milvus is unreachable.
"""

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

_connected: bool = False
_milvus_client = None


class MilvusStore:
    """Wraps pymilvus with collection management and graceful fallback."""

    def __init__(self):
        settings = get_settings()
        self._host = settings.milvus_host
        self._port = settings.milvus_port
        self._dimension = settings.milvus_dimension
        self._connected = False
        self._client = None
        self._connect()

    def _connect(self) -> None:
        """Attempt to connect to Milvus. Gracefully fails if unreachable."""
        global _milvus_client, _connected

        if _milvus_client is not None:
            self._client = _milvus_client
            self._connected = _connected
            return

        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

            connections.connect(
                alias="default",
                host=self._host,
                port=self._port,
                timeout=5,
            )
            self._client = connections
            self._connected = True
            _milvus_client = self._client
            _connected = True
            logger.info("Connected to Milvus at %s:%d", self._host, self._port)

            # Ensure collections exist
            self._ensure_collections()

        except Exception as e:
            self._connected = False
            _connected = False
            logger.warning(
                "Milvus unavailable at %s:%d (%s). RAG features disabled. "
                "Start Milvus with: docker-compose up milvus",
                self._host, self._port, e,
            )

    def _ensure_collections(self) -> None:
        """Create collections if they don't exist."""
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

        settings = get_settings()

        collections_config = [
            (settings.milvus_collection_reviews, "Cleaned review embeddings"),
            (settings.milvus_collection_pain_points, "Pain point embeddings"),
            (settings.milvus_collection_issues, "Historical issues for RAG"),
        ]

        for name, desc in collections_config:
            try:
                Collection(name)
                logger.debug("Collection '%s' exists", name)
            except Exception:
                # Create collection
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dimension),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                ]
                schema = CollectionSchema(fields=fields, description=desc)
                coll = Collection(name=name, schema=schema)

                # Create index
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                }
                coll.create_index(field_name="embedding", index_params=index_params)
                coll.load()
                logger.info("Created collection '%s' (dim=%d)", name, self._dimension)

    @property
    def is_available(self) -> bool:
        return self._connected

    def insert_reviews(self, ids: list[str], texts: list[str], embeddings: list[list[float]],
                       metadatas: list[dict]) -> bool:
        """Insert review embeddings into the review_embeddings collection."""
        if not self._connected:
            return False
        try:
            from pymilvus import Collection
            settings = get_settings()
            coll = Collection(settings.milvus_collection_reviews)
            data = [
                ids,
                texts,
                embeddings,
                metadatas,
            ]
            coll.insert(data)
            coll.flush()
            logger.debug("Inserted %d reviews into Milvus", len(ids))
            return True
        except Exception as e:
            logger.warning("Milvus insert failed: %s", e)
            return False

    def search_reviews(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        """Search for similar reviews by embedding."""
        if not self._connected:
            return []
        try:
            from pymilvus import Collection
            settings = get_settings()
            coll = Collection(settings.milvus_collection_reviews)
            coll.load()

            results = coll.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                output_fields=["text", "metadata"],
            )
            return [
                {"id": hit.id, "distance": hit.distance, "text": hit.entity.get("text", ""),
                 "metadata": hit.entity.get("metadata", {})}
                for hit in results[0]
            ]
        except Exception as e:
            logger.warning("Milvus search failed: %s", e)
            return []

    def insert_pain_points(self, ids: list[str], texts: list[str],
                           embeddings: list[list[float]], metadatas: list[dict]) -> bool:
        """Insert pain point vectors."""
        if not self._connected:
            return False
        try:
            from pymilvus import Collection
            settings = get_settings()
            coll = Collection(settings.milvus_collection_pain_points)
            coll.insert([ids, texts, embeddings, metadatas])
            coll.flush()
            return True
        except Exception as e:
            logger.warning("Milvus pain point insert failed: %s", e)
            return False

    def search_pain_points(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar historical pain points."""
        if not self._connected:
            return []
        try:
            from pymilvus import Collection
            settings = get_settings()
            coll = Collection(settings.milvus_collection_pain_points)
            coll.load()
            results = coll.search(
                data=[query_embedding], anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k, output_fields=["text", "metadata"],
            )
            return [
                {"id": hit.id, "distance": hit.distance, "text": hit.entity.get("text", ""),
                 "metadata": hit.entity.get("metadata", {})}
                for hit in results[0]
            ]
        except Exception as e:
            logger.warning("Milvus pain point search failed: %s", e)
            return []

    def insert_issues(self, ids: list[str], texts: list[str],
                      embeddings: list[list[float]], metadatas: list[dict]) -> bool:
        """Insert historical issue with resolution."""
        if not self._connected:
            return False
        try:
            from pymilvus import Collection
            settings = get_settings()
            coll = Collection(settings.milvus_collection_issues)
            coll.insert([ids, texts, embeddings, metadatas])
            coll.flush()
            return True
        except Exception as e:
            logger.warning("Milvus issue insert failed: %s", e)
            return False

    def search_issues(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar historical issues with resolutions."""
        if not self._connected:
            return []
        try:
            from pymilvus import Collection
            settings = get_settings()
            coll = Collection(settings.milvus_collection_issues)
            coll.load()
            results = coll.search(
                data=[query_embedding], anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k, output_fields=["text", "metadata"],
            )
            return [
                {"id": hit.id, "distance": hit.distance, "text": hit.entity.get("text", ""),
                 "metadata": hit.entity.get("metadata", {})}
                for hit in results[0]
            ]
        except Exception as e:
            logger.warning("Milvus issue search failed: %s", e)
            return []
