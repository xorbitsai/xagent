"""Helpers shared by the embedding migration tests: seeded stores and a fake hub."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from xagent.core.model.embedding.base import BaseEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.model.storage.error import ModelNotFoundError
from xagent.migrations.embedding import scope

SOURCE = "BAAI/old-model"
TARGET = "new-model"
# Its writer table and its once-tagged table are two names.
VENDOR_TARGET = "Vendor/new-model"
VECTOR = [0.25, 0.75]


def _config(model_id: str, **overrides: Any) -> EmbeddingModelConfig:
    fields: Dict[str, Any] = {
        "id": model_id,
        "model_name": model_id,
        "model_provider": "dashscope",
        "api_key": "k",
        "dimension": 2,
        "abilities": ["embedding"],
    }
    fields.update(overrides)
    return EmbeddingModelConfig(**fields)


class _Hub:
    def __init__(self, models: Dict[str, EmbeddingModelConfig]) -> None:
        self.models = models

    def load(self, model_id: str, **_kwargs: Any) -> EmbeddingModelConfig:
        for config in self.models.values():
            if model_id in (config.id, config.model_name):
                return config
        raise ModelNotFoundError(model_id)

    def list(self) -> Dict[str, EmbeddingModelConfig]:
        return self.models

    def close(self) -> None:
        pass


class _Adapter(BaseEmbedding):
    def __init__(self) -> None:
        self.calls: List[List[str]] = []
        self.builds = 0
        self.before_encode: Optional[Callable[[List[str]], None]] = None
        self.vectors: Callable[[List[str]], Any] = lambda texts: [
            list(VECTOR) for _ in texts
        ]

    def encode(self, text: Any, dimension: Any = None, instruct: Any = None) -> Any:
        texts = [text] if isinstance(text, str) else list(text)
        self.calls.append(texts)
        if self.before_encode is not None:
            self.before_encode(texts)
        return self.vectors(texts)

    def get_dimension(self) -> Optional[int]:
        return 2

    @property
    def abilities(self) -> List[str]:
        return ["embedding"]


def _save_collection(
    name: str = "kb",
    model_id: Optional[str] = SOURCE,
    dimension: Optional[int] = 2,
    indexed: Optional[str] = None,
) -> None:
    import asyncio

    from xagent.core.tools.core.RAG_tools.core.schemas import (
        CollectionInfo,
        IngestionConfig,
    )
    from xagent.core.tools.core.RAG_tools.management import collection_manager as cm

    asyncio.run(
        cm.collection_manager.save_collection(
            CollectionInfo(
                name=name,
                embedding_model_id=model_id,
                embedding_dimension=dimension,
                embeddings=4,
                ingestion_config=IngestionConfig(embedding_model_id=indexed)
                if indexed
                else None,
            )
        )
    )


def _create_table(name: str) -> None:
    """An embeddings table with no rows, made by the write path's own call."""
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_embeddings_table,
    )

    tag = name.removeprefix("embeddings_")
    ensure_embeddings_table(scope.connection(), tag, vector_dim=2)


def _store() -> Any:
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    return get_vector_index_store()


def _seed_core(
    chunks: Dict[str, Sequence[str]],
    *,
    collection: str = "kb",
    stale: Sequence[str] = (),
) -> None:
    """Document, parse and chunk rows; a ``stale`` chunk was made from older content."""
    from xagent.core.tools.core.RAG_tools.utils.metadata_utils import (
        serialize_metadata,
    )

    now = datetime.now(timezone.utc)
    store = _store()
    store.upsert_documents(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "file_id": None,
                "source_path": f"/{doc_id}.txt",
                "file_type": "txt",
                "content_hash": "C2",
                "uploaded_at": now,
                "title": None,
                "language": None,
                "user_id": None,
            }
            for doc_id in chunks
        ]
    )
    store.upsert_parses(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": "p",
                "parser": "local:default@v1.0.0",
                "created_at": now,
                "params_json": json.dumps({"content_hash": "C2"}),
                "parsed_content": "[]",
                "user_id": None,
            }
            for doc_id in chunks
        ]
    )
    store.upsert_chunks(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": "p",
                "chunk_id": chunk_id,
                "index": index,
                "text": chunk_id,
                "page_number": None,
                "section": None,
                "anchor": None,
                "json_path": None,
                "chunk_hash": "x",
                "config_hash": "cfg-old" if chunk_id in stale else "cfg-new",
                "created_at": now,
                "metadata": serialize_metadata(
                    {"content_hash": "C1" if chunk_id in stale else "C2"}
                ),
                "user_id": None,
            }
            for doc_id, chunk_ids in chunks.items()
            for index, chunk_id in enumerate(chunk_ids)
        ]
    )


def _src(
    chunk_id: str,
    doc_id: str = "d1",
    *,
    user_id: Optional[int] = 7,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "user_id": user_id,
        "text": f"text {chunk_id}" if text is None else text,
        "metadata": metadata,
    }


def _write(
    rows: List[Dict[str, Any]],
    *,
    model: str = SOURCE,
    collection: str = "kb",
    vector: Optional[List[float]] = None,
) -> None:
    """Rows written through the ingestion write path, one call per owner."""
    from xagent.core.tools.core.RAG_tools.core.schemas import ChunkEmbeddingData
    from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
        write_vectors_to_db,
    )

    owners: Dict[Optional[int], List[Dict[str, Any]]] = {}
    for row in rows:
        owners.setdefault(row["user_id"], []).append(row)
    for owner, items in owners.items():
        write_vectors_to_db(
            collection,
            [
                ChunkEmbeddingData(
                    doc_id=row["doc_id"],
                    chunk_id=row["chunk_id"],
                    parse_hash="p",
                    model=model,
                    vector=vector or [0.5, 0.5],
                    text=row["text"],
                    chunk_hash=f"h-{row['chunk_id']}",
                    metadata=row["metadata"],
                )
                for row in items
            ],
            create_index=False,
            user_id=owner,
            verify_bound=False,
        )


def _rows(table: str) -> List[Dict[str, Any]]:
    conn = scope.connection()
    if table not in scope.table_names(conn):
        return []
    return conn.open_table(table).search().where("true").to_arrow().to_pylist()


def _target() -> List[Dict[str, Any]]:
    return _rows(scope.source_table(TARGET))


def _versions() -> Dict[str, int]:
    conn = scope.connection()
    return {name: conn.open_table(name).version for name in scope.table_names(conn)}


def _seed_target(hub: _Hub, target: str, *, writer: bool) -> str:
    """``target`` in the hub and its tables but, without ``writer``, the writer's."""
    hub.models[target] = _config(target)
    conn = scope.connection()
    table, *legacy = scope.own_tables(target)
    for name in legacy + [table] * writer:
        if name not in scope.table_names(conn):
            _create_table(name)
    if not writer and table in scope.table_names(conn):
        conn.drop_table(table)
    return table


def _retype_owner(owner_type: Any) -> str:
    """The target's writer table made again, empty, with ``user_id`` of that type."""
    import pyarrow as pa

    conn = scope.connection()
    writer = scope.source_table(TARGET)
    schema = conn.open_table(writer).schema
    owner = schema.get_field_index("user_id")
    conn.drop_table(writer)
    conn.create_table(writer, schema=schema.set(owner, pa.field("user_id", owner_type)))
    return writer
