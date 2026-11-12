"""Collection management with embedding binding and lazy initialization.

This module provides collection management functionality for RAG (Retrieval-Augmented Generation)
systems, including lazy initialization, embedding configuration, and statistics tracking.
"""

import asyncio
import contextlib
import logging
import os
import threading
from contextvars import copy_context
from datetime import datetime, timezone
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    TypeVar,
)

import pyarrow as pa  # type: ignore

from ..core.parser_registry import get_supported_parsers, validate_parser_compatibility
from ..core.schemas import CollectionInfo
from ..LanceDB.schema_manager import _safe_close_table
from ..storage.contracts import MetadataStore
from ..storage.factory import get_metadata_store, get_vector_index_store
from ..utils.lancedb_query_utils import list_table_names
from ..utils.model_resolution_pin import pinned_resolution
from ..utils.model_resolver import resolve_embedding_adapter
from ..utils.tag_mapping import register_tag_mapping

if TYPE_CHECKING:
    from ..kb import (
        CollectionConfigSnapshot,
        CollectionRollbackMaintenanceResult,
        KBCoordinator,
        KBMaintenanceCompatibilityFacade,
    )

T = TypeVar("T")

logger = logging.getLogger(__name__)

# In-memory locks for collection operations to prevent concurrent initialization conflicts.
# Lock key is (loop_id, collection_name): avoids cross-thread deadlock and avoids sharing
# asyncio.Lock instances across different event loops.
_collection_locks: dict[tuple[int, str], asyncio.Lock] = {}
_collection_locks_lock = threading.Lock()

# Cross-thread locks for collection metadata read-modify-write. The asyncio locks
# above are keyed by event-loop id, so they do NOT serialize concurrent web-page
# ingestion where each page runs in its own executor thread + event loop. These
# thread locks close that gap for the metadata RMW critical sections (stats update,
# embedding init, and access-timestamp bump). Ordering: always acquire the asyncio
# lock first (same-loop reentry) then this thread lock, so a task holding it across
# an await can never block its own loop. The same per-collection lock also serializes
# the LanceDB vector writes in write_vectors_to_db (concurrent commits to one table,
# including the create_index build, can raise CommitConflict); it's an RLock so the
# metadata sites can nest the write under their own guard on the same thread.
_collection_thread_locks: dict[str, threading.RLock] = {}
_collection_thread_locks_guard = threading.Lock()


def _get_collection_thread_lock(collection_name: str) -> threading.RLock:
    """Return a process-global cross-thread RLock serializing one collection's
    metadata RMW and LanceDB vector writes across threads."""
    lock = _collection_thread_locks.get(collection_name)
    if lock is None:
        with _collection_thread_locks_guard:
            lock = _collection_thread_locks.setdefault(
                collection_name, threading.RLock()
            )
    return lock


@contextlib.asynccontextmanager
async def _collection_thread_guard(collection_name: str) -> AsyncIterator[None]:
    """Hold a collection's cross-thread RLock across an ``async with`` body.

    Acquired via a non-blocking poll so we never freeze the event loop this
    coroutine runs on while a different thread holds the lock: yielding with
    ``asyncio.sleep`` lets other tasks on the same loop keep running. The
    enclosing asyncio lock guarantees we never re-enter on the same event loop
    while the lock is held across an await; the RLock is reentrant regardless.
    """
    lock = _get_collection_thread_lock(collection_name)
    while not lock.acquire(blocking=False):
        await asyncio.sleep(0.005)
    try:
        yield
    finally:
        lock.release()


def reset_locks_for_testing() -> None:
    """Clear in-memory collection locks for test isolation."""
    with _collection_locks_lock:
        _collection_locks.clear()
    with _collection_thread_locks_guard:
        _collection_thread_locks.clear()


def _normalize_collection_config_user_id(user_id: Optional[int]) -> int:
    """Normalize legacy collection_config ownership scope."""
    return 0 if user_id is None else int(user_id)


# Columns the stats paths own. Binding columns are deliberately absent: they are
# written only through the guarded binding paths or first-time initialization.
_STATS_FIELDS = (
    "documents",
    "processed_documents",
    "parses",
    "chunks",
    "embeddings",
    "last_accessed_at",
)
_COUNT_FIELDS = _STATS_FIELDS[:-1]
_BINDING_FIELDS = ("embedding_model_id", "embedding_dimension")


def _normalize_model_id(model_id: Optional[str]) -> Optional[str]:
    if not isinstance(model_id, str):
        return None
    normalized = model_id.strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def _model_dimension(model_config: Any) -> Optional[int]:
    dimension = getattr(model_config, "dimension", None)
    return dimension if isinstance(dimension, int) else None


def _embedding_model_ids_equivalent(
    existing_model_id: str, requested_model_id: str
) -> tuple[bool, Optional[str], Optional[int]]:
    """Return whether two IDs resolve to the same embedding model.

    The model hub supports loading by both model_id and model_name, so this lets
    legacy name/alias bindings migrate to the canonical hub ID without treating
    the operation as a model switch.
    """
    existing_config, _ = resolve_embedding_adapter(existing_model_id)
    requested_config, _ = resolve_embedding_adapter(requested_model_id)

    existing_canonical_id = _normalize_model_id(getattr(existing_config, "id", None))
    requested_canonical_id = _normalize_model_id(getattr(requested_config, "id", None))
    dimension = _model_dimension(requested_config)
    if dimension is None:
        dimension = _model_dimension(existing_config)

    return (
        existing_canonical_id is not None
        and existing_canonical_id == requested_canonical_id,
        requested_canonical_id,
        dimension,
    )


def _run_in_separate_loop(coro: Awaitable[T]) -> T:
    """Safely run an async coroutine synchronously, even if an event loop is already running.

    This function implements thread isolation to avoid "asyncio.run() cannot be called
    from a running event loop" errors. It automatically detects the execution context
    and chooses the appropriate execution strategy.

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine execution

    Raises:
        Any exception raised by the coroutine
    """
    result: Optional[T] = None
    exception: Optional[BaseException] = None
    context = copy_context()

    def target() -> None:
        """Thread target function that runs the coroutine in its own event loop."""
        nonlocal result, exception
        # Create a fresh event loop for this thread
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = context.run(lambda: loop.run_until_complete(coro))
        except BaseException as e:
            exception = e
        finally:
            loop.close()

    try:
        # Check if we are already in a running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If yes, we MUST offload to a thread to avoid "nested loop" error
            logger.debug(
                "Detected running event loop, using thread isolation for async execution"
            )
            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
        else:
            # No running event loop, we can use asyncio.run() directly (lighter weight)
            logger.debug("No running event loop detected, using direct asyncio.run()")
            return asyncio.run(coro)  # type: ignore
    except Exception as e:
        # Fallback for unexpected errors in loop detection
        logger.error("Error in async execution wrapper: %s", e)
        raise e

    # Handle results from thread execution
    if exception:
        raise exception
    # Allow None as valid return for async void functions
    return result  # type: ignore[return-value]


def _sync_wrapper(async_func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """Decorator to create synchronous versions of async functions.

    Args:
        async_func: The async function to wrap

    Returns:
        Synchronous wrapper function
    """

    @wraps(async_func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
        # Import here to avoid circular imports and allow mocking in tests
        coro: Awaitable[T] = async_func(*args, **kwargs)
        return _run_in_separate_loop(coro)

    return sync_wrapper


def _get_collection_lock(collection_name: str) -> asyncio.Lock:
    """Get or create a lock for collection operations with thread-safe double-checked locking.

    This function implements double-checked locking to safely handle concurrent access
    to the global collection locks dictionary without unnecessary lock contention.

    Lock key includes event loop ID to prevent cross-thread deadlock when locks are
    shared across event loops (e.g., thread pool execution).

    Returns:
        asyncio.Lock for the specific (loop_id, collection_name) combination
    """
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        # No running event loop (e.g., in sync context), use 0 as default
        loop_id = 0

    lock_key = (loop_id, collection_name)

    # First check (lock-free) - for performance on hot path
    if lock_key in _collection_locks:
        return _collection_locks[lock_key]

    # Second check (with lock) - for safety on cold path
    with _collection_locks_lock:
        if lock_key not in _collection_locks:
            logger.debug(
                "[COLLECTION_LOCK] Creating new lock for '%s' (loop %s)",
                collection_name,
                loop_id,
            )
            _collection_locks[lock_key] = asyncio.Lock()
        return _collection_locks[lock_key]


class CollectionManager:
    """Manager for collection metadata operations with LanceDB storage.

    This class handles collection lifecycle management including initialization,
    statistics tracking, and persistence to LanceDB.
    """

    def __init__(self) -> None:
        pass

    def _get_metadata_store(self) -> MetadataStore:
        """Return the current metadata store singleton.

        Tests reset the storage factory between isolated LanceDB directories, so
        collection manager must resolve the store lazily instead of caching the
        first instance forever.
        """
        return get_metadata_store()

    async def _get_connection(self) -> Any:
        """Get raw metadata storage connection for legacy helper methods."""
        return self._get_metadata_store().get_raw_connection()

    async def get_collection(self, collection_name: str) -> CollectionInfo:
        """Get collection metadata from storage.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo instance

        Raises:
            ValueError: If collection not found
        """
        try:
            return await self._get_metadata_store().get_collection(collection_name)

        except Exception as e:
            # Table might not exist yet, or other backend errors
            logger.debug("Error reading collection %s: %s", collection_name, e)
            raise ValueError(f"Collection '{collection_name}' not found")

    async def save_collection(self, collection: CollectionInfo) -> None:
        """Save collection metadata to storage with retry mechanism.

        Takes the cross-thread guard as well as the asyncio lock: this does a
        full-row overwrite, so without the lock a stale-read overwrite could
        clobber a concurrent stats or binding write on the same collection.
        Callers that only mean to create a row must use
        ``save_collection_if_absent`` instead.

        Args:
            collection: CollectionInfo to save
        """
        lock = _get_collection_lock(collection.name)

        async with lock, _collection_thread_guard(collection.name):
            await self._save_collection_with_retry(collection)

    async def save_collection_if_absent(self, collection: CollectionInfo) -> bool:
        """Create the row only if no row with that name exists; never overwrites."""
        lock = _get_collection_lock(collection.name)

        async with lock, _collection_thread_guard(collection.name):
            return await self._save_collection_if_absent_with_retry(collection)

    async def _save_collection_if_absent_with_retry(
        self, collection: CollectionInfo, max_retries: int = 3
    ) -> bool:
        for attempt in range(max_retries):
            try:
                return await self._get_metadata_store().save_collection_if_absent(
                    collection
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to create collection %s after %s attempts: %s",
                        collection.name,
                        max_retries,
                        e,
                    )
                    raise
                await asyncio.sleep(0.1 * (2**attempt))
        raise AssertionError("unreachable")

    async def update_collection_fields(
        self,
        collection: CollectionInfo,
        field_names: tuple[str, ...],
        *,
        require_uninitialized: bool = False,
        expected_model_id: Optional[str] = None,
    ) -> bool:
        """Write only ``field_names`` of an existing row, under the collection locks."""
        lock = _get_collection_lock(collection.name)

        async with lock, _collection_thread_guard(collection.name):
            return await self._update_fields_with_retry(
                collection,
                field_names,
                require_uninitialized=require_uninitialized,
                expected_model_id=expected_model_id,
            )

    async def _update_fields_with_retry(
        self,
        collection: CollectionInfo,
        field_names: tuple[str, ...],
        *,
        require_uninitialized: bool = False,
        expected_model_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> bool:
        """Partial write with the same commit-conflict retry as ``save_collection``.

        A retried compare-and-set re-evaluates its guard, so a concurrent switch
        still wins and the call returns ``False``. A missing row is deterministic
        and is re-raised immediately: search/ingestion resolve the model before
        the row exists, and retrying would stall that path.
        """
        for attempt in range(max_retries):
            try:
                return await self._get_metadata_store().update_collection_fields(
                    collection,
                    field_names,
                    require_uninitialized=require_uninitialized,
                    expected_model_id=expected_model_id,
                )
            except Exception as e:
                if isinstance(e, ValueError) and "not found" in str(e):
                    raise
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to update fields of collection %s after %s attempts: %s",
                        collection.name,
                        max_retries,
                        e,
                    )
                    raise
                logger.warning(
                    "Field update attempt %s failed for %s, retrying: %s",
                    attempt + 1,
                    collection.name,
                    e,
                )
                await asyncio.sleep(0.1 * (2**attempt))
        raise AssertionError("unreachable")

    async def _write_initial_binding(
        self,
        collection_name: str,
        updated: CollectionInfo,
        *,
        requested_model_id: str,
        require_uninitialized: bool = False,
        expected_model_id: Optional[str] = None,
    ) -> CollectionInfo:
        """Apply a first/canonical binding only if the stored one did not move first."""
        applied = await self._update_fields_with_retry(
            updated,
            _BINDING_FIELDS,
            require_uninitialized=require_uninitialized,
            expected_model_id=expected_model_id,
        )
        if applied:
            return updated
        fresh = await self.get_collection(collection_name)
        if _normalize_model_id(fresh.embedding_model_id) == _normalize_model_id(
            updated.embedding_model_id
        ):
            return fresh
        raise ValueError(
            f"Collection '{collection_name}' already initialized with "
            f"model '{fresh.embedding_model_id}'. Cannot change to '{requested_model_id}'."
        )

    async def _update_fields_locked(
        self, collection: CollectionInfo, field_names: tuple[str, ...]
    ) -> None:
        """Partial write for callers already holding the collection locks."""
        await self._update_fields_with_retry(collection, field_names)

    async def delete_collection_metadata(
        self,
        collection_name: str,
        user_id: Optional[int],
        is_admin: bool = False,
        delete_orphaned_metadata: bool = False,
    ) -> dict[str, int]:
        """Delete persisted metadata/config rows for a collection."""
        lock = _get_collection_lock(collection_name)

        async with lock:
            return await self._get_metadata_store().delete_collection_metadata(
                collection_name,
                user_id=user_id,
                is_admin=is_admin,
                delete_orphaned_metadata=delete_orphaned_metadata,
            )

    async def _save_collection_with_retry(
        self, collection: CollectionInfo, max_retries: int = 3
    ) -> None:
        """Save collection with retry mechanism for concurrent updates.

        Args:
            collection: CollectionInfo to save
            max_retries: Maximum number of retry attempts

        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                await self._get_metadata_store().save_collection(collection)
                return

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        "Failed to save collection %s after %s attempts: %s",
                        collection.name,
                        max_retries,
                        e,
                    )
                    raise

                # Exponential backoff
                wait_time = 0.1 * (2**attempt)
                logger.warning(
                    "Save attempt %s failed for %s, retrying in %ss: %s",
                    attempt + 1,
                    collection.name,
                    wait_time,
                    e,
                )
                await asyncio.sleep(wait_time)

    async def _ensure_metadata_table(self) -> None:
        """Ensure collection_metadata table exists in LanceDB.

        Creates the table if it doesn't exist, otherwise does nothing.
        """

        conn = await self._get_connection()

        schema = pa.schema(
            [
                ("name", pa.string()),
                ("schema_version", pa.string()),
                ("embedding_model_id", pa.string()),  # Nullable
                ("embedding_dimension", pa.int32()),  # Nullable
                ("documents", pa.int32()),
                ("processed_documents", pa.int32()),
                ("parses", pa.int32()),
                ("chunks", pa.int32()),
                ("embeddings", pa.int32()),
                ("document_names", pa.string()),  # JSON string
                (
                    "owners",
                    pa.string(),
                ),  # Schema-only; not maintained (derived at list time from user_id)
                ("collection_locked", pa.bool_()),
                ("allow_mixed_parse_methods", pa.bool_()),
                ("skip_config_validation", pa.bool_()),
                ("ingestion_config", pa.string()),  # JSON string
                ("created_at", pa.timestamp("us")),
                ("updated_at", pa.timestamp("us")),
                ("last_accessed_at", pa.timestamp("us")),
                ("extra_metadata", pa.string()),  # JSON string
            ]
        )

        # Check if table already exists (prefer list_tables(); avoids deprecation warnings)
        table_exists = False
        try:
            existing_tables = list_table_names(conn)
            table_exists = "collection_metadata" in existing_tables
        except Exception as e:
            logger.debug("Table names check failed: %s", e)

        if not table_exists:
            try:
                conn.create_table("collection_metadata", schema=schema)
            except Exception as e:
                logger.debug("Table creation failed (may already exist): %s", e)
                # Table might already exist, continue
        else:
            # Table exists: ensure it has the "owners" column (schema compat; column is not maintained)
            table = None
            try:
                table = conn.open_table("collection_metadata")
                if hasattr(table, "schema") and table.schema is not None:
                    names = getattr(table.schema, "names", None) or []
                    if "owners" not in names:
                        add_fn = getattr(table, "add_columns", None)
                        if add_fn is not None:
                            add_fn({"owners": "cast('[]' as string)"})
                            logger.info(
                                "collection_metadata: added missing 'owners' column (schema-only)"
                            )
            except Exception as e:
                logger.debug(
                    "Could not migrate collection_metadata schema (add owners): %s", e
                )
            finally:
                _safe_close_table(table)

    async def initialize_collection_embedding(
        self, collection_name: str, embedding_model_id: str
    ) -> "CollectionInfo":
        """Initialize collection with embedding configuration.

        This implements the "First Document Wins" lazy initialization strategy.

        Args:
            collection_name: Name of the collection
            embedding_model_id: Embedding model ID to use

        Returns:
            Initialized CollectionInfo

        Raises:
            ValueError: If collection already initialized with different model
        """
        logger.debug(
            "[COLLECTION_INIT] Initializing '%s' with model '%s'",
            collection_name,
            embedding_model_id,
        )
        lock = _get_collection_lock(collection_name)
        logger.debug("[COLLECTION_INIT] Acquiring lock for '%s'...", collection_name)

        async with lock, _collection_thread_guard(collection_name):
            logger.debug("[COLLECTION_INIT] Lock acquired for '%s'", collection_name)
            # Get current state
            try:
                collection = await self.get_collection(collection_name)
                row_exists = True
            except ValueError:
                # Collection doesn't exist, create it
                collection = CollectionInfo(name=collection_name)
                row_exists = False

            # Check if already initialized
            if collection.is_initialized:
                existing_model_id = _normalize_model_id(collection.embedding_model_id)
                requested_model_id = _normalize_model_id(embedding_model_id)
                if existing_model_id != requested_model_id:
                    equivalent = False
                    canonical_model_id: Optional[str] = None
                    canonical_dimension: Optional[int] = None
                    if existing_model_id and requested_model_id:
                        try:
                            (
                                equivalent,
                                canonical_model_id,
                                canonical_dimension,
                            ) = _embedding_model_ids_equivalent(
                                existing_model_id, requested_model_id
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Failed to compare embedding model IDs for collection '%s': %s",
                                collection_name,
                                exc,
                            )

                    if equivalent and canonical_model_id:
                        updated_collection = collection.model_copy(
                            update={
                                "embedding_model_id": canonical_model_id,
                                "embedding_dimension": canonical_dimension
                                if canonical_dimension is not None
                                else collection.embedding_dimension,
                                "updated_at": datetime.now(timezone.utc).replace(
                                    tzinfo=None
                                ),
                            }
                        )
                        migrated = await self._write_initial_binding(
                            collection_name,
                            updated_collection,
                            requested_model_id=embedding_model_id,
                            expected_model_id=collection.embedding_model_id,
                        )
                        logger.info(
                            "Migrated collection '%s' embedding model from '%s' to canonical ID '%s'",
                            collection_name,
                            existing_model_id,
                            canonical_model_id,
                        )
                        return migrated

                    raise ValueError(
                        f"Collection '{collection_name}' already initialized with "
                        f"model '{collection.embedding_model_id}'. Cannot change to '{embedding_model_id}'."
                    )
                # Already initialized with same model, return as-is
                return collection

            # Initialize embedding config
            embedding_config, _ = resolve_embedding_adapter(embedding_model_id)
            canonical_model_id = (
                _normalize_model_id(getattr(embedding_config, "id", None))
                or embedding_model_id
            )

            # Update collection
            updated_collection = collection.model_copy(
                update={
                    "embedding_model_id": canonical_model_id,
                    "embedding_dimension": embedding_config.dimension,
                    "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
                }
            )

            if row_exists:
                updated_collection = await self._write_initial_binding(
                    collection_name,
                    updated_collection,
                    requested_model_id=embedding_model_id,
                    require_uninitialized=True,
                )
            else:
                inserted = await self._save_collection_if_absent_with_retry(
                    updated_collection
                )
                if not inserted:
                    updated_collection = await self._write_initial_binding(
                        collection_name,
                        updated_collection,
                        requested_model_id=embedding_model_id,
                        require_uninitialized=True,
                    )

            logger.info(
                "Initialized collection '%s' with embedding model '%s'",
                collection_name,
                canonical_model_id,
            )
            logger.debug("[COLLECTION_INIT] Releasing lock for '%s'", collection_name)
            return updated_collection

    async def update_collection_stats(
        self,
        collection_name: str,
        documents_delta: int = 0,
        processed_documents_delta: int = 0,
        parses_delta: int = 0,
        chunks_delta: int = 0,
        embeddings_delta: int = 0,
        document_name: Optional[str] = None,
    ) -> "CollectionInfo":
        """Update collection statistics after document processing.

        Args:
            collection_name: Name of the collection
            documents_delta: Change in total document count
            processed_documents_delta: Change in successfully processed document count
            parses_delta: Change in parse count
            chunks_delta: Change in chunk count
            embeddings_delta: Change in embedding count
            document_name: Document name to add to document_names list

        Returns:
            Updated CollectionInfo
        """
        lock = _get_collection_lock(collection_name)

        async with lock, _collection_thread_guard(collection_name):
            try:
                collection = await self.get_collection(collection_name)
            except ValueError:
                # Auto-create collection if it doesn't exist
                collection = CollectionInfo(name=collection_name)
                # Save the newly created collection to storage
                await self._save_collection_if_absent_with_retry(collection)
            # Update statistics
            updated_data = {
                "documents": collection.documents + documents_delta,
                "processed_documents": collection.processed_documents
                + processed_documents_delta,
                "parses": collection.parses + parses_delta,
                "chunks": collection.chunks + chunks_delta,
                "embeddings": collection.embeddings + embeddings_delta,
                "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "last_accessed_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }

            # Update document names
            if document_name:
                new_document_names = collection.document_names.copy()
                if document_name not in new_document_names:
                    new_document_names.append(document_name)
                updated_data["document_names"] = new_document_names

            updated_collection = collection.model_copy(update=updated_data)
            fields = _STATS_FIELDS + (
                ("document_names",) if "document_names" in updated_data else ()
            )
            await self._update_fields_locked(updated_collection, fields)

            return updated_collection

    async def validate_document_processing(
        self,
        collection_name: str,
        file_path: str,
        parsing_method: str,
        chunking_method: str,
    ) -> None:
        """Validate document processing config against collection settings.

        Args:
            collection_name: Name of the collection
            file_path: Path to the document file
            parsing_method: Parsing method to use
            chunking_method: Chunking method to use

        Raises:
            ValueError: If validation fails
        """
        # First-layer: basic type compatibility based purely on extension and parser method.
        # This runs regardless of collection existence, enforcing a consistent baseline.
        try:
            collection = await self.get_collection(collection_name)
        except ValueError:
            collection = None

        # Respect collection-level skip flag only for collection-specific rules,
        # not for the basic type compatibility check.
        allow_mixed = (
            bool(collection.allow_mixed_parse_methods) if collection else False
        )

        # Always use strict parser/file-type compatibility here (allow_mixed=False).
        # The allow_mixed flag only controls whether we skip this compatibility check entirely.
        if parsing_method != "default" and not allow_mixed:
            file_ext = os.path.splitext(file_path)[1]

            if not validate_parser_compatibility(file_ext, parsing_method, False):
                supported = get_supported_parsers(file_ext)
                raise ValueError(
                    f"Parser method '{parsing_method}' not compatible with file type '{file_ext}'. "
                    f"Supported methods: {supported}"
                )

        # Second-layer: collection-level rules (locks, future policies).
        if collection is None or collection.skip_config_validation:
            return

        if collection.collection_locked:
            # Placeholder for any stricter rules on locked collections.
            return

    async def mark_collection_accessed(self, collection_name: str) -> None:
        """Update last_accessed_at.

        Only that column is written, so a stale read here cannot clobber a
        concurrent stat increment or a completed model switch.
        """
        lock = _get_collection_lock(collection_name)
        try:
            async with lock, _collection_thread_guard(collection_name):
                await self._update_fields_locked(
                    CollectionInfo(
                        name=collection_name,
                        last_accessed_at=datetime.now(timezone.utc).replace(
                            tzinfo=None
                        ),
                    ),
                    ("last_accessed_at",),
                )
        except Exception as e:
            logger.debug(
                "Failed to update last_accessed_at for %s: %s", collection_name, e
            )


# Global singleton instance
collection_manager = CollectionManager()


def _get_maintenance_compatibility_facade() -> "KBMaintenanceCompatibilityFacade":
    """Return the coordinator-owned collection maintenance facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().maintenance_compatibility


# Synchronous wrapper functions using decorator
def get_collection_sync(collection_name: str) -> "CollectionInfo":
    """Synchronous version of get_collection for non-async contexts.

    Args:
        collection_name: Name of the collection to retrieve

    Returns:
        CollectionInfo instance

    Raises:
        ValueError: If collection not found
    """
    return _get_maintenance_compatibility_facade().get_collection_sync(collection_name)


def _get_collection_sync_impl(collection_name: str) -> "CollectionInfo":
    return _sync_wrapper(collection_manager.get_collection)(collection_name)


def initialize_collection_embedding_sync(
    collection_name: str, embedding_model_id: str
) -> "CollectionInfo":
    """Synchronous version of initialize_collection_embedding.

    Args:
        collection_name: Name of the collection
        embedding_model_id: Embedding model ID to use

    Returns:
        Initialized CollectionInfo

    Raises:
        ValueError: If collection already initialized with different model
    """
    return _get_maintenance_compatibility_facade().initialize_collection_embedding_sync(
        collection_name, embedding_model_id
    )


def _initialize_collection_embedding_sync_impl(
    collection_name: str, embedding_model_id: str
) -> "CollectionInfo":
    return _sync_wrapper(collection_manager.initialize_collection_embedding)(
        collection_name, embedding_model_id
    )


def validate_document_processing_sync(
    collection_name: str, file_path: str, parsing_method: str, chunking_method: str
) -> None:
    """Synchronous version of validate_document_processing.

    Args:
        collection_name: Name of the collection
        file_path: Path to the document file
        parsing_method: Parsing method to use
        chunking_method: Chunking method to use

    Raises:
        ValueError: If validation fails
    """
    _get_maintenance_compatibility_facade().validate_document_processing_sync(
        collection_name, file_path, parsing_method, chunking_method
    )


def _validate_document_processing_sync_impl(
    collection_name: str, file_path: str, parsing_method: str, chunking_method: str
) -> None:
    _sync_wrapper(collection_manager.validate_document_processing)(
        collection_name, file_path, parsing_method, chunking_method
    )


def update_collection_stats_sync(
    collection_name: str,
    documents_delta: int = 0,
    processed_documents_delta: int = 0,
    parses_delta: int = 0,
    chunks_delta: int = 0,
    embeddings_delta: int = 0,
    document_name: Optional[str] = None,
) -> "CollectionInfo":
    """Synchronous version of update_collection_stats.

    Args:
        collection_name: Name of the collection
        documents_delta: Change in total document count
        processed_documents_delta: Change in successfully processed document count
        parses_delta: Change in parse count
        chunks_delta: Change in chunk count
        embeddings_delta: Change in embedding count
        document_name: Document name to add to document_names list

    Returns:
        Updated CollectionInfo
    """
    return _get_maintenance_compatibility_facade().update_collection_stats_sync(
        collection_name,
        documents_delta,
        processed_documents_delta,
        parses_delta,
        chunks_delta,
        embeddings_delta,
        document_name,
    )


def _update_collection_stats_sync_impl(
    collection_name: str,
    documents_delta: int = 0,
    processed_documents_delta: int = 0,
    parses_delta: int = 0,
    chunks_delta: int = 0,
    embeddings_delta: int = 0,
    document_name: Optional[str] = None,
) -> "CollectionInfo":
    return _sync_wrapper(collection_manager.update_collection_stats)(
        collection_name,
        documents_delta,
        processed_documents_delta,
        parses_delta,
        chunks_delta,
        embeddings_delta,
        document_name,
    )


def mark_collection_accessed_sync(collection_name: str) -> None:
    """Synchronous version of mark_collection_accessed.

    Args:
        collection_name: Name of the collection to mark as accessed
    """
    _get_maintenance_compatibility_facade().mark_collection_accessed_sync(
        collection_name
    )


async def write_collection_binding(
    collection_name: str, embedding_model_id: str, embedding_dimension: int
) -> None:
    """Write a collection's bound model pair in one partial update."""
    await collection_manager.update_collection_fields(
        CollectionInfo(
            name=collection_name,
            embedding_model_id=embedding_model_id,
            embedding_dimension=embedding_dimension,
        ),
        _BINDING_FIELDS,
    )


def write_collection_binding_sync(
    collection_name: str, embedding_model_id: str, embedding_dimension: int
) -> None:
    """Synchronous version of write_collection_binding."""
    _sync_wrapper(write_collection_binding)(
        collection_name, embedding_model_id, embedding_dimension
    )


def _mark_collection_accessed_sync_impl(collection_name: str) -> None:
    _sync_wrapper(collection_manager.mark_collection_accessed)(collection_name)


def delete_collection_metadata_sync(
    collection_name: str,
    user_id: Optional[int],
    is_admin: bool = False,
    delete_orphaned_metadata: bool = False,
) -> dict[str, int]:
    """Synchronous version of delete_collection_metadata."""
    return _get_maintenance_compatibility_facade().delete_collection_metadata_sync(
        collection_name,
        user_id,
        is_admin,
        delete_orphaned_metadata,
    )


def _delete_collection_metadata_sync_impl(
    collection_name: str,
    user_id: Optional[int],
    is_admin: bool = False,
    delete_orphaned_metadata: bool = False,
) -> dict[str, int]:
    return _sync_wrapper(collection_manager.delete_collection_metadata)(
        collection_name,
        user_id,
        is_admin,
        delete_orphaned_metadata,
    )


def _collection_rollback_guard_warning(
    *, rollback_complete: bool, side_effects_may_remain: bool
) -> tuple[Optional[str], tuple[str, ...]]:
    if side_effects_may_remain:
        return (
            "side_effects_may_remain",
            (
                "Skipped collection rollback maintenance because side effects may remain.",
            ),
        )
    if not rollback_complete:
        return (
            "rollback_not_complete",
            (
                "Skipped collection rollback maintenance because rollback is incomplete.",
            ),
        )
    return None, ()


def _collection_rollback_result(
    collection_name: str,
    *,
    status: str,
    skipped: bool = False,
    reason: Optional[str] = None,
    cleanup_counts: Optional[dict[str, int]] = None,
    warnings: tuple[str, ...] = (),
    side_effects_may_remain: bool = False,
) -> "CollectionRollbackMaintenanceResult":
    from ..kb.maintenance_compatibility import CollectionRollbackMaintenanceResult

    return CollectionRollbackMaintenanceResult(
        collection_name=collection_name,
        status=status,
        skipped=skipped,
        reason=reason,
        cleanup_counts=cleanup_counts or {},
        warnings=warnings,
        side_effects_may_remain=side_effects_may_remain,
    )


async def capture_collection_config_snapshot(
    collection_name: str, user_id: Optional[int]
) -> "CollectionConfigSnapshot":
    """Capture the exact collection_config row before rollback-sensitive mutation."""
    return await _get_maintenance_compatibility_facade().capture_collection_config_snapshot(
        collection_name, user_id
    )


async def _capture_collection_config_snapshot_impl(
    collection_name: str, user_id: Optional[int]
) -> "CollectionConfigSnapshot":
    from ..kb.maintenance_compatibility import CollectionConfigSnapshot

    config_user_id = _normalize_collection_config_user_id(user_id)
    config_json = await get_metadata_store().get_collection_config(
        collection_name,
        config_user_id,
        is_admin=False,
    )
    return CollectionConfigSnapshot(
        collection_name=collection_name,
        user_id=user_id,
        config_user_id=config_user_id,
        config_json=config_json,
        existed=config_json is not None,
    )


def capture_collection_config_snapshot_sync(
    collection_name: str, user_id: Optional[int]
) -> "CollectionConfigSnapshot":
    """Synchronous version of capture_collection_config_snapshot."""
    return (
        _get_maintenance_compatibility_facade().capture_collection_config_snapshot_sync(
            collection_name, user_id
        )
    )


def _capture_collection_config_snapshot_sync_impl(
    collection_name: str, user_id: Optional[int]
) -> "CollectionConfigSnapshot":
    return _sync_wrapper(_capture_collection_config_snapshot_impl)(
        collection_name, user_id
    )


async def restore_collection_config_snapshot(
    snapshot: "CollectionConfigSnapshot",
    *,
    rollback_complete: bool,
    side_effects_may_remain: bool = False,
) -> "CollectionRollbackMaintenanceResult":
    """Restore or remove collection_config from a rollback snapshot."""
    return await _get_maintenance_compatibility_facade().restore_collection_config_snapshot(
        snapshot,
        rollback_complete=rollback_complete,
        side_effects_may_remain=side_effects_may_remain,
    )


async def _restore_collection_config_snapshot_impl(
    snapshot: "CollectionConfigSnapshot",
    *,
    rollback_complete: bool,
    side_effects_may_remain: bool = False,
) -> "CollectionRollbackMaintenanceResult":
    reason, warnings = _collection_rollback_guard_warning(
        rollback_complete=rollback_complete,
        side_effects_may_remain=side_effects_may_remain,
    )
    if reason is not None:
        return _collection_rollback_result(
            snapshot.collection_name,
            status="skipped",
            skipped=True,
            reason=reason,
            warnings=warnings,
            side_effects_may_remain=side_effects_may_remain,
        )

    lock = _get_collection_lock(snapshot.collection_name)
    async with lock:
        if snapshot.existed:
            if snapshot.config_json is None:
                return _collection_rollback_result(
                    snapshot.collection_name,
                    status="skipped",
                    skipped=True,
                    reason="snapshot_missing_config",
                    warnings=(
                        "Skipped collection config restore because the snapshot has no config JSON.",
                    ),
                    side_effects_may_remain=side_effects_may_remain,
                )
            await get_metadata_store().save_collection_config(
                snapshot.collection_name,
                snapshot.config_json,
                snapshot.config_user_id,
            )
            return _collection_rollback_result(
                snapshot.collection_name,
                status="restored",
                side_effects_may_remain=side_effects_may_remain,
            )

        cleanup_counts = await get_metadata_store().delete_collection_metadata(
            snapshot.collection_name,
            user_id=snapshot.config_user_id,
            is_admin=False,
            delete_orphaned_metadata=False,
        )
        return _collection_rollback_result(
            snapshot.collection_name,
            status="removed",
            cleanup_counts=cleanup_counts,
            side_effects_may_remain=side_effects_may_remain,
        )


def restore_collection_config_snapshot_sync(
    snapshot: "CollectionConfigSnapshot",
    *,
    rollback_complete: bool,
    side_effects_may_remain: bool = False,
) -> "CollectionRollbackMaintenanceResult":
    """Synchronous version of restore_collection_config_snapshot."""
    return (
        _get_maintenance_compatibility_facade().restore_collection_config_snapshot_sync(
            snapshot,
            rollback_complete=rollback_complete,
            side_effects_may_remain=side_effects_may_remain,
        )
    )


def _restore_collection_config_snapshot_sync_impl(
    snapshot: "CollectionConfigSnapshot",
    *,
    rollback_complete: bool,
    side_effects_may_remain: bool = False,
) -> "CollectionRollbackMaintenanceResult":
    return _sync_wrapper(_restore_collection_config_snapshot_impl)(
        snapshot,
        rollback_complete=rollback_complete,
        side_effects_may_remain=side_effects_may_remain,
    )


async def cleanup_collection_metadata_after_rollback(
    collection_name: str,
    user_id: Optional[int],
    is_admin: bool = False,
    *,
    rollback_complete: bool,
    side_effects_may_remain: bool = False,
    delete_orphaned_metadata: bool = True,
) -> "CollectionRollbackMaintenanceResult":
    """Delete metadata/config rows only when rollback completed safely."""
    return await _get_maintenance_compatibility_facade().cleanup_collection_metadata_after_rollback(
        collection_name,
        user_id,
        is_admin,
        rollback_complete=rollback_complete,
        side_effects_may_remain=side_effects_may_remain,
        delete_orphaned_metadata=delete_orphaned_metadata,
    )


async def _cleanup_collection_metadata_after_rollback_impl(
    collection_name: str,
    user_id: Optional[int],
    is_admin: bool = False,
    *,
    rollback_complete: bool,
    side_effects_may_remain: bool = False,
    delete_orphaned_metadata: bool = True,
) -> "CollectionRollbackMaintenanceResult":
    reason, warnings = _collection_rollback_guard_warning(
        rollback_complete=rollback_complete,
        side_effects_may_remain=side_effects_may_remain,
    )
    if reason is not None:
        return _collection_rollback_result(
            collection_name,
            status="skipped",
            skipped=True,
            reason=reason,
            warnings=warnings,
            side_effects_may_remain=side_effects_may_remain,
        )

    cleanup_counts = await collection_manager.delete_collection_metadata(
        collection_name,
        user_id=user_id,
        is_admin=is_admin,
        delete_orphaned_metadata=delete_orphaned_metadata,
    )
    return _collection_rollback_result(
        collection_name,
        status="cleaned",
        cleanup_counts=cleanup_counts,
        side_effects_may_remain=side_effects_may_remain,
    )


def cleanup_collection_metadata_after_rollback_sync(
    collection_name: str,
    user_id: Optional[int],
    is_admin: bool = False,
    *,
    rollback_complete: bool,
    side_effects_may_remain: bool = False,
    delete_orphaned_metadata: bool = True,
) -> "CollectionRollbackMaintenanceResult":
    """Synchronous version of cleanup_collection_metadata_after_rollback."""
    return _get_maintenance_compatibility_facade().cleanup_collection_metadata_after_rollback_sync(
        collection_name,
        user_id,
        is_admin,
        rollback_complete=rollback_complete,
        side_effects_may_remain=side_effects_may_remain,
        delete_orphaned_metadata=delete_orphaned_metadata,
    )


def _cleanup_collection_metadata_after_rollback_sync_impl(
    collection_name: str,
    user_id: Optional[int],
    is_admin: bool = False,
    *,
    rollback_complete: bool,
    side_effects_may_remain: bool = False,
    delete_orphaned_metadata: bool = True,
) -> "CollectionRollbackMaintenanceResult":
    return _sync_wrapper(_cleanup_collection_metadata_after_rollback_impl)(
        collection_name,
        user_id,
        is_admin,
        rollback_complete=rollback_complete,
        side_effects_may_remain=side_effects_may_remain,
        delete_orphaned_metadata=delete_orphaned_metadata,
    )


def _coerce_stat_count(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key)
    return int(value) if value is not None else 0


async def rebuild_collection_stats(
    collection_name: str,
) -> Optional["CollectionInfo"]:
    """Rebuild global collection metadata stats after commit/rollback."""
    return await _get_maintenance_compatibility_facade().rebuild_collection_stats(
        collection_name
    )


async def _rebuild_collection_stats_impl(
    collection_name: str,
    coordinator: "KBCoordinator | None" = None,
) -> Optional["CollectionInfo"]:
    from .collections import _resolve_coordinator

    try:
        existing_info: Optional[
            CollectionInfo
        ] = await get_metadata_store().get_collection(collection_name)
    except Exception:
        existing_info = None

    # collection_metadata is a global cache; never persist user-scoped counts here.
    # aggregate_collection_stats is a synchronous LanceDB scan: keep it off the
    # event loop so it cannot stall every other coroutine in the process.
    stats_by_collection = await asyncio.to_thread(
        _resolve_coordinator(coordinator).aggregate_collection_stats_sync,
        user_id=None,
        is_admin=True,
    )
    rebuilt_stats = stats_by_collection.get(collection_name)
    if existing_info is None and rebuilt_stats is None:
        return None

    if rebuilt_stats is None:
        rebuilt_stats = {
            "documents": 0,
            "parses": 0,
            "chunks": 0,
            "embeddings": 0,
        }

    document_count = _coerce_stat_count(rebuilt_stats, "documents")
    parse_count = _coerce_stat_count(rebuilt_stats, "parses")
    chunk_count = _coerce_stat_count(rebuilt_stats, "chunks")
    embedding_count = _coerce_stat_count(rebuilt_stats, "embeddings")
    base_info = existing_info or CollectionInfo(name=collection_name)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    update_data: dict[str, Any] = {
        "documents": document_count,
        "processed_documents": parse_count,
        "parses": parse_count,
        "chunks": chunk_count,
        "embeddings": embedding_count,
        "updated_at": now,
        "last_accessed_at": now,
    }
    if document_count == 0:
        update_data.update(
            {
                "document_names": [],
                "document_metadata": [],
                "owners": [],
            }
        )

    rebuilt = base_info.model_copy(update=update_data)
    fields = _STATS_FIELDS + (("document_names",) if document_count == 0 else ())
    if existing_info is None:
        inserted = await collection_manager.save_collection_if_absent(rebuilt)
        if inserted:
            return rebuilt
    await collection_manager.update_collection_fields(rebuilt, fields)
    return rebuilt


def rebuild_collection_stats_sync(
    collection_name: str,
) -> Optional["CollectionInfo"]:
    """Synchronous version of rebuild_collection_stats."""
    return _get_maintenance_compatibility_facade().rebuild_collection_stats_sync(
        collection_name
    )


def _rebuild_collection_stats_sync_impl(
    collection_name: str,
    coordinator: "KBCoordinator | None" = None,
) -> Optional["CollectionInfo"]:
    return _sync_wrapper(_rebuild_collection_stats_impl)(collection_name, coordinator)


def resolve_effective_embedding_model_sync(
    collection_name: str, config_model_id: Optional[str] = None
) -> str:
    """Resolve the effective embedding model ID for a collection.

    Logic:
    1. If collection is initialized, use its bound model ID.
    2. Else if collection ingestion_config stores an embedding model, use it.
    3. Else if existing embedding tables can be inferred for this collection, use that.
    4. Else if config provides an embedding model, use it.
    5. If none are available, raise ValueError.

    Args:
        collection_name: Name of the collection
        config_model_id: Model ID from configuration (optional)

    Returns:
        The resolved model ID string.

    Raises:
        ValueError: If model cannot be resolved or collection not found.
    """
    # Once per request under a ModelResolutionPin: a binding written between
    # the usage gate and the pipeline must not change the model called.
    return pinned_resolution(
        ("effective_embedding_model", collection_name, config_model_id),
        lambda: (
            _get_maintenance_compatibility_facade().resolve_effective_embedding_model_sync(
                collection_name, config_model_id
            )
        ),
    )


def _resolve_effective_embedding_model_sync_impl(
    collection_name: str, config_model_id: Optional[str] = None
) -> str:
    """Implementation for resolve_effective_embedding_model_sync."""

    # Treat empty/whitespace-only IDs and the tool-layer "none" placeholder as missing.
    config_model_id = _normalize_model_id(config_model_id)
    try:
        mark_collection_accessed_sync(collection_name)
        collection_info = get_collection_sync(collection_name)

        bound_model_id = _normalize_model_id(collection_info.embedding_model_id)
        indexed_model_id = _normalize_model_id(
            collection_info.ingestion_config.embedding_model_id
            if collection_info.ingestion_config is not None
            else None
        )

        if collection_info.is_initialized and bound_model_id:
            if config_model_id and config_model_id != bound_model_id:
                logger.warning(
                    "Config embedding_model_id '%s' overridden by "
                    "collection '%s' bound model '%s'",
                    config_model_id,
                    collection_name,
                    bound_model_id,
                )
            return bound_model_id

        if indexed_model_id:
            if config_model_id and config_model_id != indexed_model_id:
                logger.warning(
                    "Config embedding_model_id '%s' overridden by "
                    "collection '%s' ingestion config model '%s'",
                    config_model_id,
                    collection_name,
                    indexed_model_id,
                )
            logger.info(
                "Collection '%s' using ingestion config embedding_model_id '%s'",
                collection_name,
                indexed_model_id,
            )
            return indexed_model_id

        inferred_model_id: Optional[str] = None
        inferred_dimension: Optional[int] = None
        if collection_info.embeddings > 0:
            try:
                from ..utils.migration_utils import (
                    _infer_embedding_config_from_collection,
                )

                inferred_model_id, inferred_dimension = (
                    _infer_embedding_config_from_collection(collection_name)
                )
                inferred_model_id = _normalize_model_id(inferred_model_id)
            except Exception as exc:
                logger.warning(
                    "Embedding inference failed for collection '%s': %s",
                    collection_name,
                    exc,
                )

        if inferred_model_id:
            logger.info(
                "Collection '%s' inferred embedding_model_id '%s' from existing embedding tables",
                collection_name,
                inferred_model_id,
            )
            try:
                candidate = collection_info.model_copy(
                    update={
                        "embedding_model_id": inferred_model_id,
                        "embedding_dimension": inferred_dimension
                        if inferred_dimension is not None
                        else collection_info.embedding_dimension,
                    }
                )
                applied = _sync_wrapper(collection_manager.update_collection_fields)(
                    candidate, _BINDING_FIELDS, require_uninitialized=True
                )
                if not applied:
                    # Inferring is slow enough for a switch to land meanwhile; it wins.
                    bound = _normalize_model_id(
                        get_collection_sync(collection_name).embedding_model_id
                    )
                    if bound:
                        logger.info(
                            "Collection '%s' was bound to '%s' while inferring",
                            collection_name,
                            bound,
                        )
                        return bound
            except Exception as save_error:
                logger.warning(
                    "Failed to persist inferred embedding metadata for collection '%s': %s",
                    collection_name,
                    save_error,
                )
            return inferred_model_id

        if config_model_id:
            logger.info(
                "Collection '%s' not initialized, using config embedding_model_id '%s'",
                collection_name,
                config_model_id,
            )
            return config_model_id

        raise ValueError(
            f"Collection '{collection_name}' is not initialized with an embedding model. "
            "Please ingest documents first or specify embedding_model_id in config."
        )

    except ValueError as e:
        if "not found" in str(e):
            if config_model_id:
                return config_model_id
            raise ValueError(
                f"Collection '{collection_name}' not found and no model ID provided."
            )
        raise


async def rebuild_collection_metadata() -> None:
    """Rebuild collection_metadata table from existing data.

    This function reads all collections from documents/parses/chunks/embeddings tables
    and creates corresponding entries in the collection_metadata table.

    Use this to migrate existing data when collection_metadata table is missing or outdated.
    """
    await _get_maintenance_compatibility_facade().rebuild_collection_metadata()


async def _read_stored_collection(
    collection_name: str,
) -> Optional["CollectionInfo"]:
    """Read the current row; ``None`` means it does not exist."""
    try:
        return await get_metadata_store().get_collection(collection_name)
    except ValueError as exc:
        if "not found" in str(exc):
            return None
        raise


_ENGINE_COUNTS = ("documents", "parses", "chunks", "embeddings")


async def _rebuild_collection_metadata_impl() -> None:
    """Implementation for rebuild_collection_metadata."""
    from ..kb.collection_handle import ledger_holds_vectors
    from . import collections
    from .collections import _resolve_coordinator

    # Get all existing collections (use is_admin=True to bypass user filtering)
    result = await collections.list_collections(is_admin=True, force_realtime=True)

    if result.status != "success":
        logger.error("Failed to list collections: %s", result.message)
        return

    if not result.collections:
        return

    # Unreported collections have unknown counts: a stored row with vectors is
    # left as is, one without keeps its counts and the normal binding rule.
    engine_stats = await asyncio.to_thread(
        _resolve_coordinator(None).aggregate_collection_stats_sync,
        user_id=None,
        is_admin=True,
    )

    # Get connection and find embeddings tables
    vector_store = get_vector_index_store()
    embeddings_tables: list[str] = []
    if ledger_holds_vectors():
        table_names = vector_store.list_table_names()
        embeddings_tables = [t for t in table_names if t.startswith("embeddings_")]

    # Build lookup from legacy/new table tags to Hub model IDs.
    hub_tag_to_id: dict[str, tuple[str, Optional[int]]] = {}
    try:
        from xagent.core.model.model import EmbeddingModelConfig

        from ..LanceDB.model_tag_utils import model_tag_candidates
        from ..utils.model_resolver import _list_model_hub_configs

        configs = [
            cfg
            for cfg in _list_model_hub_configs().values()
            if isinstance(cfg, EmbeddingModelConfig)
        ]
        # Ids first: a model_name alias must not claim a tag another config writes.
        for field in ("id", "model_name"):
            for cfg in configs:
                for tag in model_tag_candidates(getattr(cfg, field)):
                    register_tag_mapping(
                        hub_tag_to_id,
                        tag,
                        (cfg.id, cfg.dimension),
                        get_identity=lambda item: item[0],
                        logger=logger,
                    )
    except Exception as e:
        logger.warning(
            "Model hub initialization failed during collection metadata rebuild: "
            "error_type=%s, error_message=%s, fallback_behavior=%s, impact=%s",
            type(e).__name__,
            str(e),
            "legacy_model_resolution",
            "May use suboptimal model selection or missing embeddings",
            exc_info=True,
        )
        hub_tag_to_id = {}

    # Save each collection to metadata table
    for collection in result.collections:
        try:
            # Infer embedding_model_id from embeddings tables
            embedding_model_id = None
            embedding_dimension = None

            engine = engine_stats.get(collection.name)
            if engine is not None:
                # Counts and the reported decision come from this one read.
                collection = collection.model_copy(
                    update={
                        **{k: engine.get(k, 0) for k in _ENGINE_COUNTS},
                        "processed_documents": engine.get("parses", 0),
                    }
                )
            else:
                try:
                    stored = await get_metadata_store().get_collection(collection.name)
                except Exception as e:  # noqa: BLE001 - unknown state, keep the row
                    # A corrupt row raises pydantic's ValidationError, also a ValueError.
                    if not (isinstance(e, ValueError) and "not found" in str(e)):
                        logger.warning(
                            "Skipping rebuild of '%s': cannot read its row: %s",
                            collection.name,
                            e,
                        )
                        continue
                    stored = None
                if stored is not None:
                    if stored.embeddings > 0:
                        continue
                    collection = stored

            # Redundant at run time; a direct gate the bypass guard can see.
            if collection.embeddings > 0 and ledger_holds_vectors():
                # Find which embeddings table has data for this collection
                for table_name in embeddings_tables:
                    # Use abstraction layer to count rows
                    count = vector_store.count_rows_or_zero(
                        table_name,
                        filters={"collection": collection.name},
                        is_admin=True,
                    )
                    if count > 0:
                        suffix = table_name.replace("embeddings_", "", 1)
                        # Prefer Hub ID mapping (single source of truth).
                        if suffix in hub_tag_to_id:
                            embedding_model_id, inferred_dim = hub_tag_to_id[suffix]
                            if inferred_dim is not None:
                                embedding_dimension = inferred_dim
                        else:
                            # Legacy fallback: best-effort reverse normalization.
                            embedding_model_id = suffix.replace("_", "-")

                        # Use abstraction layer to get vector dimension from schema
                        table_dim = vector_store.get_vector_dimension(table_name)
                        if table_dim is not None:
                            embedding_dimension = table_dim

                        break

            # Vectors the engine counts but no embeddings_* table holds: keep the
            # stored model. Zero vectors still clear it so the KB can be rebound.
            keep_binding = collection.embeddings > 0 and embedding_model_id is None
            binding: dict[str, Any] = (
                {}
                if keep_binding
                else {
                    "embedding_model_id": embedding_model_id,
                    "embedding_dimension": embedding_dimension,
                }
            )

            current = await _read_stored_collection(collection.name)
            updated_collection = collection.model_copy(update=binding)
            inserted = False
            if current is None:
                inserted = _sync_wrapper(collection_manager.save_collection_if_absent)(
                    updated_collection
                )
            if current is not None or not inserted:
                _sync_wrapper(collection_manager.update_collection_fields)(
                    updated_collection, _COUNT_FIELDS
                )
                if binding:
                    # A switch may have landed since the list snapshot; only an
                    # unbound row may receive an inferred binding.
                    _sync_wrapper(collection_manager.update_collection_fields)(
                        updated_collection,
                        tuple(binding),
                        require_uninitialized=True,
                    )
        except Exception as e:
            logger.error("Failed to rebuild collection '%s': %s", collection.name, e)
