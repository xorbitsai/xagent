from __future__ import annotations

import importlib
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import VectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "MilvusConnectionManager",
    "MilvusVectorStore",
    "get_client",
    "get_client_from_env",
]


if TYPE_CHECKING:
    # NOTE:
    # Readability alias: returned runtime object is `pymilvus.MilvusClient`.
    # We keep this as `Any` in typing to avoid strict mypy failures on untyped
    # third-party imports (`disallow_any_unimported=true` in this project).
    MilvusClient = Any
else:
    MilvusClient = Any


def _import_milvus_client_class() -> Any:
    try:
        pymilvus_module = importlib.import_module("pymilvus")
    except ImportError as e:
        raise ImportError(
            "pymilvus is not installed. Please install it with: pip install pymilvus"
        ) from e
    return getattr(pymilvus_module, "MilvusClient")


class MilvusConnectionManager:
    """Milvus connection manager."""

    _shared_clients: ClassVar[Dict[Tuple[Any, ...], Any]] = {}
    _shared_clients_lock: ClassVar[threading.Lock] = threading.Lock()

    def get_client(
        self,
        uri: str,
        token: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> "MilvusClient":
        if not uri or not uri.strip():
            raise ValueError("Milvus uri must be non-empty")

        milvus_client_class = _import_milvus_client_class()
        return milvus_client_class(
            uri=uri.strip(),
            token=(token or "").strip(),
            db_name=(db_name or "").strip(),
        )

    def get_client_from_env(
        self,
        uri_env_var: str = "MILVUS_URI",
        token_env_var: str = "MILVUS_TOKEN",
        db_name_env_var: str = "MILVUS_DB_NAME",
    ) -> "MilvusClient":
        uri, token, db_name = self._settings_from_env(
            uri_env_var, token_env_var, db_name_env_var
        )
        return self.get_client(uri=uri, token=token, db_name=db_name)

    def get_shared_client_from_env(
        self,
        uri_env_var: str = "MILVUS_URI",
        token_env_var: str = "MILVUS_TOKEN",
        db_name_env_var: str = "MILVUS_DB_NAME",
    ) -> "MilvusClient":
        """Return the process-wide client for the environment's (uri, token, db).

        Never closed: pymilvus shares one connection per alias, and ``close()``
        drops it for every other client. The lock covers only a miss, because
        pymilvus does not lock the first connect of an alias.
        """
        uri, token, db_name = self._settings_from_env(
            uri_env_var, token_env_var, db_name_env_var
        )
        key = (type(self), uri.strip(), (token or "").strip(), (db_name or "").strip())
        client = self._shared_clients.get(key)
        if client is None:
            with self._shared_clients_lock:
                client = self._shared_clients.get(key)
                if client is None:
                    client = self._shared_clients[key] = self.get_client(
                        uri=uri, token=token, db_name=db_name
                    )
        return client

    @staticmethod
    def _settings_from_env(
        uri_env_var: str, token_env_var: str, db_name_env_var: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        uri = os.getenv(uri_env_var)
        if uri is None:
            raise KeyError(f"Environment variable {uri_env_var} is not set")
        if not uri.strip():
            raise ValueError(f"Environment variable {uri_env_var} is empty")
        return uri, os.getenv(token_env_var), os.getenv(db_name_env_var)


class MilvusVectorStore(VectorStore):
    """Milvus vector store implementation."""

    support_store_texts: ClassVar[bool] = True

    def __init__(
        self,
        uri: str,
        collection_name: str = "vectors",
        token: Optional[str] = None,
        db_name: Optional[str] = None,
        metric_type: str = "COSINE",
        connection_manager: Optional[MilvusConnectionManager] = None,
    ):
        self._uri = uri
        self._collection_name = collection_name
        self._token = token
        self._db_name = db_name
        self._metric_type = metric_type
        self._conn_manager = connection_manager or MilvusConnectionManager()
        self._client = self._conn_manager.get_client(
            uri=uri,
            token=token,
            db_name=db_name,
        )
        self._vector_dim: Optional[int] = None

    def _ensure_collection(self, vector_dim: int) -> None:
        if vector_dim <= 0:
            raise ValueError("vector dimension must be greater than zero")

        if self._vector_dim is None:
            self._vector_dim = vector_dim

        if not self._client.has_collection(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=vector_dim,
                primary_field_name="id",
                id_type="string",
                vector_field_name="vector",
                metric_type=self._metric_type,
                auto_id=False,
                enable_dynamic_field=True,
            )

    @staticmethod
    def _matches_filters(
        metadata: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        if not filters:
            return True

        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    def add_vectors(
        self,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if not vectors:
            return []

        if ids is None:
            ids = [str(uuid4()) for _ in vectors]
        elif len(ids) != len(vectors):
            raise ValueError("ids length must match vectors length")

        if metadatas is None:
            metadatas = [{} for _ in vectors]
        elif len(metadatas) != len(vectors):
            raise ValueError("metadatas length must match vectors length")

        vector_dim = len(vectors[0])
        self._ensure_collection(vector_dim=vector_dim)

        payload = []
        for i, vector in enumerate(vectors):
            if len(vector) != vector_dim:
                raise ValueError("all vectors must have the same dimension")
            payload.append(
                {
                    "id": ids[i],
                    "vector": vector,
                    "metadata": metadatas[i],
                }
            )

        self._client.insert(collection_name=self._collection_name, data=payload)
        return ids

    def delete_vectors(self, ids: List[str]) -> bool:
        if not ids:
            return True

        try:
            if not self._client.has_collection(self._collection_name):
                return True
            self._client.delete(collection_name=self._collection_name, ids=ids)
            return True
        except Exception as e:
            logger.error("Failed to delete vectors in Milvus: %s", e)
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        if not query_vector:
            return []

        self._ensure_collection(vector_dim=len(query_vector))

        # Fetch more candidates when filters are provided, then filter in Python.
        limit = max(top_k, top_k * 5 if filters else top_k)
        raw = self._client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["metadata"],
        )

        hits = raw[0] if raw else []
        results: List[Dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity", {})
            metadata = entity.get("metadata", hit.get("metadata", {}))
            if not isinstance(metadata, dict):
                metadata = {}

            if not self._matches_filters(metadata, filters):
                continue

            item_id = hit.get("id", entity.get("id"))
            score = hit.get("distance", hit.get("score", 0.0))
            results.append(
                {
                    "id": str(item_id) if item_id is not None else "",
                    "score": float(score),
                    "metadata": metadata,
                }
            )
            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        if not self._client.has_collection(self._collection_name):
            return
        self._client.truncate_collection(self._collection_name)


def get_client(
    uri: str,
    token: Optional[str] = None,
    db_name: Optional[str] = None,
) -> "MilvusClient":
    manager = MilvusConnectionManager()
    return manager.get_client(uri=uri, token=token, db_name=db_name)


def get_client_from_env(
    uri_env_var: str = "MILVUS_URI",
    token_env_var: str = "MILVUS_TOKEN",
    db_name_env_var: str = "MILVUS_DB_NAME",
) -> "MilvusClient":
    manager = MilvusConnectionManager()
    return manager.get_client_from_env(
        uri_env_var=uri_env_var,
        token_env_var=token_env_var,
        db_name_env_var=db_name_env_var,
    )
