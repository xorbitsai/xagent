"""Vector storage operations for RAG tools.

This module exposes the legacy, synchronous vector-storage helper functions:

1. Reading chunks from the database for embedding computation
2. Writing embedding vectors to the database with idempotency
3. Query-vector validation

The functions keep their historical import paths and signatures, but the
storage mechanics now live in the coordinator-owned collection handle. Each
helper delegates to the ``KBVectorStorageCompatibilityFacade`` (which opens the
handle through the active coordinator), so model-tag routing, dimension checks,
merge error handling, and result models are all owned by the handle/store
layers. This module performs no text-to-vector conversion (embedding generation
stays in the pipeline/provider layer).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..core.schemas import (
    ChunkEmbeddingData,
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..kb.vector_storage_compatibility import (
        KBVectorStorageCompatibilityFacade,
    )


def _get_vector_storage_compatibility_facade() -> "KBVectorStorageCompatibilityFacade":
    """Return the coordinator-owned vector storage compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().vector_storage_compatibility


def validate_query_vector(
    query_vector: List[float],
    model_tag: Optional[str] = None,
    conn: Any = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> None:
    """Validate query vector format and content through the coordinator facade."""
    _get_vector_storage_compatibility_facade().validate_query_vector(
        query_vector,
        model_tag=model_tag,
        conn=conn,
        user_id=user_id,
        is_admin=is_admin,
    )


def read_chunks_for_embedding(
    collection: str,
    doc_id: str,
    parse_hash: str,
    model: str,
    filters: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> EmbeddingReadResponse:
    """Read chunks from database for embedding computation through the facade."""
    return _get_vector_storage_compatibility_facade().read_chunks_for_embedding(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        model=model,
        filters=filters,
        user_id=user_id,
        is_admin=is_admin,
    )


def write_vectors_to_db(
    collection: str,
    embeddings: List[ChunkEmbeddingData],
    create_index: bool = True,
    user_id: Optional[int] = None,
    *,
    verify_bound: bool = True,
) -> EmbeddingWriteResponse:
    """Write embedding vectors to database with idempotency through the facade.

    Serialized per collection: web ingestion runs up to page_ingest_concurrency
    pages on separate threads, and concurrent commits to the same LanceDB table
    (including the create_index build) can raise CommitConflict. The dominant
    cost, embedding computation, already finished before this call, so the lock
    only serializes the cheap write and barely dents throughput.

    With ``verify_bound``, returns only if, after the commit, the migration row is not
    frozen and the collection has no metadata row, no binding, or the batch model bound.
    """
    from ..kb.coordinator import KBCoordinator
    from ..management.collection_manager import _get_collection_thread_lock

    # The handle writes under this name; the guard must read the same rows.
    collection = KBCoordinator._normalize_collection(collection)
    with _get_collection_thread_lock(collection):
        response = _get_vector_storage_compatibility_facade().write_vectors_to_db(
            collection=collection,
            embeddings=embeddings,
            create_index=create_index,
            user_id=user_id,
        )
    if verify_bound and embeddings:
        _refuse_unless_still_bound(collection, embeddings)
    return response


def _refuse_unless_still_bound(
    collection: str, embeddings: List[ChunkEmbeddingData]
) -> None:
    from ..core.exceptions import CollectionMigratingError
    from ..management.collection_manager import (
        _normalize_model_id,
        get_collection_if_present_sync,
    )
    from ..management.migration_state import FROZEN, read_migration_state

    written = {_normalize_model_id(item.model) for item in embeddings}
    error: Optional[Exception] = None
    read = "migration row"
    try:
        # Row before binding: a switch flips the binding before it unfreezes.
        if read_migration_state(collection).state == FROZEN:
            reason = "is frozen for an embedding migration"
        else:
            read = "binding"
            info = get_collection_if_present_sync(collection)
            if info is None:
                return
            bound = _normalize_model_id(info.embedding_model_id)
            # No binding cannot be a finished switch: a switch always leaves one.
            if bound is None or {bound} == written:
                return
            reason = f"is now bound to '{info.embedding_model_id}'"
    except Exception as exc:  # noqa: BLE001 - unreadable proves nothing either way
        error = exc
        reason = f"could not be checked ({read} read: {type(exc).__name__})"
        logger.warning(
            "Write guard on collection %s: %s read failed",
            collection,
            read,
            exc_info=True,
        )
    raise CollectionMigratingError(
        f"Collection '{collection}' {reason}; this write is refused, retry it.",
        details={
            "collection": collection,
            "written_models": sorted({item.model for item in embeddings}),
        },
    ) from error


def commit_vectors_to_db(
    collection: str,
    doc_id: str,
    parse_hash: str,
    model: str,
    commit_gate: Optional[Callable[[], None]] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> None:
    """Make the written vectors searchable through the facade (no-op on LanceDB)."""
    _get_vector_storage_compatibility_facade().commit_vectors(
        collection=collection,
        doc_id=doc_id,
        parse_hash=parse_hash,
        model=model,
        commit_gate=commit_gate,
        user_id=user_id,
        is_admin=is_admin,
    )
