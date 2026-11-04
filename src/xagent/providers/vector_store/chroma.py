from __future__ import annotations

from typing import Any, ClassVar, Optional
from uuid import uuid4

try:
    from chromadb import Client
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except ImportError as e:
    raise ImportError(
        "ChromaDB is not installed. Please install it with: pip install chromadb"
    ) from e

from .base import VectorStore


class ChromaVectorStore(VectorStore):
    support_store_texts: ClassVar[Optional[bool]] = True

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        if client is not None:
            self.client = client
        else:
            settings = Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
            self.client = Client(settings=settings)

        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_vectors(
        self,
        vectors: list[list[float]],
        ids: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid4()) for _ in vectors]

        kwargs = {
            "embeddings": vectors,
            "ids": ids,
        }

        if metadatas:
            kwargs["metadatas"] = metadatas

        self.collection.add(**kwargs)
        return ids

    def delete_vectors(self, ids: list[str]) -> bool:
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["metadatas", "distances"],
        }

        if filters:
            query_kwargs["where"] = filters

        results = self.collection.query(**query_kwargs)

        return [
            {
                "id": results["ids"][0][i],
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def clear(self) -> None:
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name, embedding_function=DefaultEmbeddingFunction()
        )
