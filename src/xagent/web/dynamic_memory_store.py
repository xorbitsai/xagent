"""Dynamic memory store manager for web application."""

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union, cast

from ..core.memory.base import (
    MEMORY_BACKEND_UNAVAILABLE_REASON,
    MemoryBackendUnavailableError,
)
from ..core.memory.in_memory import InMemoryMemoryStore
from ..core.memory.lancedb import LanceDBMemoryStore
from ..core.model import EmbeddingModelConfig
from ..core.model.embedding import create_embedding_adapter
from ..core.storage.manager import get_storage_root
from .models.database import get_db
from .models.model import Model as DBModel
from .models.user import UserDefaultModel
from .services.db_runtime import is_database_pool_timeout
from .user_isolated_memory import UserIsolatedMemoryStore, current_user_id

logger = logging.getLogger(__name__)

# Type alias for our memory store types that includes user isolation
MemoryStoreType = Union[
    InMemoryMemoryStore, LanceDBMemoryStore, UserIsolatedMemoryStore
]


def _embedding_model_fingerprint(model: Optional[DBModel]) -> Optional[tuple]:
    """Identity of an embedding model config, including reconfigurations.

    ``updated_at`` changes when the model row is edited (API key rotation,
    endpoint change), so comparing the fingerprint instead of only the id
    lets the store pick up new credentials without a backend restart.
    """
    if model is None:
        return None
    return (
        model.id,
        str(model.updated_at),
        model.model_provider,
        model.model_name,
        model.base_url,
        model.dimension,
    )


class _ModelLookupStatus(Enum):
    FOUND = auto()
    NONE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class _ModelLookupResult:
    status: _ModelLookupStatus
    model: Optional[DBModel] = None
    error: Optional[BaseException] = None


class DynamicMemoryStoreManager:
    """Dynamic memory store manager that supports lazy initialization and reconfiguration."""

    def __init__(self, similarity_threshold: Optional[float] = None):
        """
        Initialize the dynamic memory store manager.

        Args:
            similarity_threshold: Optional similarity threshold for vector search.
        """
        self._similarity_threshold = similarity_threshold
        self._memory_store: Optional[MemoryStoreType] = None
        self._lock = threading.RLock()
        self._lookup_lock = threading.Lock()
        self._last_embedding_model_id: Optional[int] = None
        # (id, updated_at) of the embedding model the store was built with.
        # Comparing the full fingerprint (not just the id) makes API key or
        # endpoint rotation on the same model take effect without a restart.
        self._last_embedding_model_fingerprint: Optional[tuple] = None
        self._is_lancedb: bool = False

        # Initialize with in-memory store (will be replaced with LanceDB when embedding model is configured)
        self._initialize_in_memory_store()

    def _initialize_in_memory_store(self) -> None:
        """Initialize with basic in-memory store."""
        with self._lock:
            in_memory_store = InMemoryMemoryStore()
            self._memory_store = UserIsolatedMemoryStore(in_memory_store)
            self._is_lancedb = False
            self._last_embedding_model_id = None
            self._last_embedding_model_fingerprint = None
            logger.info("Initialized with in-memory store")

    def _get_embedding_model_from_db(self) -> Optional[DBModel]:
        """Get the current embedding model from database."""
        db = next(get_db())
        try:
            user_id = current_user_id.get()
            from .services.model_service import _is_model_visible_to_user

            if user_id:
                user_default = (
                    db.query(UserDefaultModel)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "embedding",
                    )
                    .first()
                )
                if user_default:
                    embedding_model = (
                        db.query(DBModel)
                        .filter(
                            DBModel.id == user_default.model_id,
                            DBModel.category == "embedding",
                            DBModel.is_active,
                        )
                        .first()
                    )
                    if embedding_model and _is_model_visible_to_user(
                        db, embedding_model.id, user_id
                    ):
                        return embedding_model
                    logger.warning(
                        "User default embedding model %s is unavailable",
                        user_default.model_id,
                    )

            all_active_embeddings = (
                db.query(DBModel)
                .filter(DBModel.category == "embedding", DBModel.is_active)
                .all()
            )
            for embedding_model in all_active_embeddings:
                if _is_model_visible_to_user(db, embedding_model.id, user_id):
                    return embedding_model
            return None
        finally:
            db.close()

    def _lookup_embedding_model(self) -> _ModelLookupResult:
        try:
            model = self._get_embedding_model_from_db()
        except Exception as exc:
            if is_database_pool_timeout(exc):
                raise
            logger.error("Error checking for embedding model: %s", exc)
            return _ModelLookupResult(_ModelLookupStatus.FAILED, error=exc)
        if model is None:
            return _ModelLookupResult(_ModelLookupStatus.NONE)
        return _ModelLookupResult(_ModelLookupStatus.FOUND, model=model)

    def _create_lancedb_store(
        self,
        embedding_model: Optional[DBModel],
        *,
        allow_schema_migration: bool = False,
    ) -> UserIsolatedMemoryStore:
        """Create LanceDB store with the given embedding model."""
        # Check legacy location (project root) first for backward compatibility
        legacy_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "memory_store",
        )
        if os.path.exists(legacy_dir) and os.listdir(legacy_dir):
            logger.info(f"Using legacy memory store location: {legacy_dir}")
            db_dir = legacy_dir
        else:
            new_dir = get_storage_root() / "memory_store"
            os.makedirs(new_dir, exist_ok=True)
            db_dir = str(new_dir)

        embedding_adapter = None
        vector_space_identity = None
        if embedding_model is not None:
            dimension = embedding_model.dimension
            if (
                dimension is None
                and str(embedding_model.model_provider).lower().strip() == "dashscope"
            ):
                dimension = 1024
            embedding_adapter = create_embedding_adapter(
                EmbeddingModelConfig(
                    id=str(embedding_model.model_id),
                    model_name=embedding_model.model_name,
                    model_provider=embedding_model.model_provider,
                    api_key=str(embedding_model.api_key),
                    base_url=embedding_model.base_url,
                    dimension=dimension,
                )
            )
            vector_space_identity = {
                "provider": str(embedding_model.model_provider).lower().strip(),
                "model": embedding_model.model_name,
                "endpoint": embedding_model.base_url,
                "dimension": dimension,
            }
        lancedb_store = LanceDBMemoryStore(
            db_dir=db_dir,
            embedding_model=embedding_adapter,
            similarity_threshold=self._similarity_threshold or 1.5,
            vector_space_identity=vector_space_identity,
            allow_schema_migration=allow_schema_migration,
        )
        logger.info("Created LanceDB memory store")
        return UserIsolatedMemoryStore(lancedb_store)

    def _check_and_update_store(
        self,
        *,
        require_persistence: bool = False,
        require_vector_search: bool = False,
    ) -> MemoryStoreType:
        """Check if embedding model configuration has changed and update store accordingly."""
        with self._lookup_lock:
            return self._check_and_update_store_once(
                require_persistence=require_persistence,
                require_vector_search=require_vector_search,
            )

    def _check_and_update_store_once(
        self,
        *,
        require_persistence: bool,
        require_vector_search: bool,
    ) -> MemoryStoreType:
        lookup = self._lookup_embedding_model()

        with self._lock:
            strict = require_persistence or require_vector_search
            if lookup.status is _ModelLookupStatus.FAILED:
                if require_vector_search:
                    raise MemoryBackendUnavailableError(
                        MEMORY_BACKEND_UNAVAILABLE_REASON
                    ) from lookup.error
                if require_persistence and not self._is_lancedb:
                    try:
                        self._install_lancedb_store(None, None)
                    except Exception as exc:
                        raise MemoryBackendUnavailableError(
                            MEMORY_BACKEND_UNAVAILABLE_REASON
                        ) from exc
                if require_persistence:
                    self._require_persistence()
                assert self._memory_store is not None
                return self._memory_store

            embedding_model = lookup.model
            current_fingerprint = _embedding_model_fingerprint(embedding_model)
            should_install = bool(
                embedding_model
                and (
                    not self._is_lancedb
                    or current_fingerprint != self._last_embedding_model_fingerprint
                )
            )
            # An authoritative NONE clears a stale adapter while preserving a
            # durable handle and all shared-table data.
            should_clear_adapter = bool(
                embedding_model is None
                and self._is_lancedb
                and self._last_embedding_model_fingerprint is not None
            )
            should_create_text_store = bool(
                embedding_model is None and require_persistence and not self._is_lancedb
            )

            if should_install or should_clear_adapter or should_create_text_store:
                try:
                    self._install_lancedb_store(embedding_model, current_fingerprint)
                except Exception as exc:
                    logger.exception("Error creating LanceDB memory store")
                    if strict:
                        raise MemoryBackendUnavailableError(
                            MEMORY_BACKEND_UNAVAILABLE_REASON
                        ) from exc
                    assert self._memory_store is not None
                    return self._memory_store

            if require_persistence:
                self._require_persistence()
            if require_vector_search:
                if embedding_model is None:
                    raise MemoryBackendUnavailableError(
                        MEMORY_BACKEND_UNAVAILABLE_REASON
                    )
                self._require_vector_search()
            assert self._memory_store is not None
            return self._memory_store

    def _install_lancedb_store(
        self, embedding_model: Optional[DBModel], fingerprint: Optional[tuple]
    ) -> None:
        memory_store = self._create_lancedb_store(
            embedding_model, allow_schema_migration=False
        )
        self._memory_store = memory_store
        self._is_lancedb = True
        self._last_embedding_model_id = (
            cast(int, embedding_model.id) if embedding_model else None
        )
        self._last_embedding_model_fingerprint = fingerprint

    def _require_persistence(self) -> None:
        if self._memory_store is None:
            raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)
        try:
            self._memory_store.ensure_persistence()
        except MemoryBackendUnavailableError:
            raise
        except Exception as exc:
            raise MemoryBackendUnavailableError(
                MEMORY_BACKEND_UNAVAILABLE_REASON
            ) from exc

    def _require_vector_search(self) -> None:
        if self._memory_store is None:
            raise MemoryBackendUnavailableError(MEMORY_BACKEND_UNAVAILABLE_REASON)
        try:
            self._memory_store.ensure_required_vector_search()
        except MemoryBackendUnavailableError:
            raise
        except Exception as exc:
            raise MemoryBackendUnavailableError(
                MEMORY_BACKEND_UNAVAILABLE_REASON
            ) from exc

    def get_memory_store(
        self,
        *,
        require_persistence: bool = False,
        require_vector_search: bool = False,
    ) -> MemoryStoreType:
        """
        Get the current memory store, initializing or updating as necessary.

        Returns:
            Current memory store instance
        """
        store = self._check_and_update_store(
            require_persistence=require_persistence,
            require_vector_search=require_vector_search,
        )
        if require_vector_search and isinstance(store, UserIsolatedMemoryStore):
            return UserIsolatedMemoryStore(
                store._base_store, require_vector_search=True
            )
        return store

    def force_reinitialize(self) -> None:
        """Force reinitialization of the memory store."""
        self._initialize_in_memory_store()
        self._check_and_update_store()
        logger.info("Force reinitialized memory store")

    def check_embedding_model_change(self) -> bool:
        """Check if embedding model configuration has changed and update if necessary.

        Returns:
            True if the store was updated, False otherwise.
        """
        with self._lock:
            old_is_lancedb = self._is_lancedb
            old_fingerprint = self._last_embedding_model_fingerprint

        self._check_and_update_store()

        with self._lock:
            return (
                old_is_lancedb != self._is_lancedb
                or old_fingerprint != self._last_embedding_model_fingerprint
            )

    def get_store_info(self) -> dict:
        """
        Get information about the current memory store.

        Returns:
            Dictionary with store information
        """
        with self._lock:
            base_store = (
                self._memory_store._base_store
                if isinstance(self._memory_store, UserIsolatedMemoryStore)
                else self._memory_store
            )

            supports_vector_search = False
            if self._is_lancedb and base_store is not None:
                try:
                    base_store.ensure_required_vector_search()
                except MemoryBackendUnavailableError:
                    pass
                else:
                    supports_vector_search = True

            return {
                "store_type": type(base_store).__name__,
                "is_lancedb": self._is_lancedb,
                "embedding_model_id": self._last_embedding_model_id,
                "similarity_threshold": self._similarity_threshold,
                "supports_vector_search": supports_vector_search,
            }


# Global instance
_dynamic_manager: Optional[DynamicMemoryStoreManager] = None
_manager_lock = threading.Lock()


def get_memory_store_manager(
    similarity_threshold: Optional[float] = None,
) -> DynamicMemoryStoreManager:
    """Get or create the global memory store manager."""
    global _dynamic_manager

    if _dynamic_manager is None:
        with _manager_lock:
            if _dynamic_manager is None:
                _dynamic_manager = DynamicMemoryStoreManager(similarity_threshold)

    return _dynamic_manager


def get_memory_store(
    *,
    require_persistence: bool = False,
    require_vector_search: bool = False,
) -> MemoryStoreType:
    """Get the current memory store (for backward compatibility)."""
    manager = get_memory_store_manager()
    return manager.get_memory_store(
        require_persistence=require_persistence,
        require_vector_search=require_vector_search,
    )


def force_reinitialize_memory_store() -> None:
    """Force reinitialization of the memory store."""
    manager = get_memory_store_manager()
    manager.force_reinitialize()
