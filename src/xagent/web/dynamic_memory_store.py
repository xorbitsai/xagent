"""Dynamic memory store manager for web application."""

import logging
import os
import threading
from typing import Optional, Union

from ..core.memory.in_memory import InMemoryMemoryStore
from ..core.memory.lancedb import LanceDBMemoryStore
from ..core.model.embedding.adapter import create_embedding_adapter
from ..core.model.model import EmbeddingModelConfig
from ..core.storage.manager import get_storage_root
from .models.database import get_db
from .models.model import Model as DBModel
from .models.user import UserDefaultModel
from .services.db_runtime import is_database_pool_timeout
from .user_isolated_memory import UserIsolatedMemoryStore, current_user_id

logger = logging.getLogger(__name__)

# Matches DashScopeEmbedding's own default. Memory vectors written before the
# store routed through the metered adapter used this model, so it stays the
# fallback whenever the DB row does not name one.
_DEFAULT_MEMORY_EMBEDDING_MODEL = "text-embedding-v4"

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
    return (model.id, str(model.updated_at))


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
        try:
            db = next(get_db())
            try:
                # Get current user ID from context
                user_id = current_user_id.get()

                from .services.model_service import _is_model_visible_to_user

                if user_id:
                    # First, try to get user's default embedding model
                    user_default = (
                        db.query(UserDefaultModel)
                        .filter(
                            UserDefaultModel.user_id == user_id,
                            UserDefaultModel.config_type == "embedding",
                        )
                        .first()
                    )

                    if user_default:
                        # Get the actual model
                        embedding_model = (
                            db.query(DBModel)
                            .filter(
                                DBModel.id == user_default.model_id,
                                DBModel.category == "embedding",
                                DBModel.is_active,
                            )
                            .first()
                        )
                        if embedding_model:
                            if not _is_model_visible_to_user(
                                db, embedding_model.id, user_id
                            ):
                                logger.warning(
                                    f"User default embedding model {user_default.model_id} is no longer visible"
                                )
                                # fall through to system fallback
                            else:
                                logger.info(
                                    f"Found user's default embedding model: {embedding_model.model_id}"
                                )
                                return embedding_model
                        else:
                            logger.warning(
                                f"User default embedding model {user_default.model_id} not found or inactive"
                            )

                # Fallback: look for first active embedding model visible to user
                all_active_embeddings = (
                    db.query(DBModel)
                    .filter(
                        DBModel.category == "embedding",
                        DBModel.is_active,
                    )
                    .all()
                )

                for embedding_model in all_active_embeddings:
                    if _is_model_visible_to_user(db, embedding_model.id, user_id):
                        logger.info(
                            f"Using visible active embedding model: {embedding_model.model_id}"
                        )
                        return embedding_model

                logger.info("No visible active embedding model found")
                return None
            finally:
                db.close()
        except Exception as e:
            if is_database_pool_timeout(e):
                raise
            logger.error(f"Error checking for embedding model: {e}")
            return None

    def _create_lancedb_store(
        self, embedding_model: DBModel
    ) -> UserIsolatedMemoryStore:
        """Create LanceDB store with the given embedding model."""
        try:
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
                # Use new default location
                new_dir = get_storage_root() / "memory_store"
                os.makedirs(new_dir, exist_ok=True)
                db_dir = str(new_dir)

            if embedding_model.model_provider == "dashscope":
                # Keep the previous class default when the DB row carries no
                # usable name. Memories already on disk were embedded with this
                # model, and switching it would silently put old and new
                # vectors in incompatible spaces — a recall-quality regression
                # with no error at store-creation time (DashScopeEmbedding does
                # not validate the name; a bad one only fails later at encode).
                # The embedding model is PINNED, not read from the DB row.
                # Memory vectors already stored were written with
                # _DEFAULT_MEMORY_EMBEDDING_MODEL, and switching the model puts
                # new vectors in a different space than the stored ones — a
                # silent recall-quality regression, since
                # schema_migration.py's mismatch check compares only vector
                # presence and width, not model identity, so nothing triggers
                # a rebuild. The configured name is still threaded through as
                # the billing label so usage is attributed to the row the
                # deployment actually configured.
                configured_name = str(
                    getattr(embedding_model, "model_name", "") or ""
                ).strip()
                if (
                    configured_name
                    and configured_name != _DEFAULT_MEMORY_EMBEDDING_MODEL
                ):
                    logger.info(
                        "Memory store embeds with pinned %s for compatibility "
                        "with existing vectors; billing usage as %s per the "
                        "configured embedding row",
                        _DEFAULT_MEMORY_EMBEDDING_MODEL,
                        configured_name,
                    )
                # Built through the adapter rather than instantiating the
                # provider directly: the adapter is where embedding usage is
                # metered, so a direct DashScopeEmbedding() would make every
                # memory-store embedding invisible to billing.
                lancedb_store = LanceDBMemoryStore(
                    db_dir=db_dir,
                    embedding_model=create_embedding_adapter(
                        EmbeddingModelConfig(
                            id=str(getattr(embedding_model, "model_id", "") or ""),
                            model_name=_DEFAULT_MEMORY_EMBEDDING_MODEL,
                            billing_model_name=configured_name or None,
                            model_provider="dashscope",
                            api_key=str(embedding_model.api_key),
                            dimension=int(embedding_model.dimension or 1024),
                        )
                    ),
                    similarity_threshold=self._similarity_threshold or 1.5,
                )
                logger.info("Created LanceDB store with DashScope embedding model")
                return UserIsolatedMemoryStore(lancedb_store)
            else:
                # Fallback to in-memory if embedding type not supported
                logger.warning(
                    f"Unsupported embedding model type: {embedding_model.model_provider}"
                )
                self._initialize_in_memory_store()
                return self._memory_store  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Error creating LanceDB store: {e}")
            # Fallback to in-memory store
            self._initialize_in_memory_store()
            return self._memory_store  # type: ignore[return-value]

    def _check_and_update_store(self) -> None:
        """Check if embedding model configuration has changed and update store accordingly."""
        with self._lock:
            embedding_model = self._get_embedding_model_from_db()
            current_model_id = embedding_model.id if embedding_model else None
            current_fingerprint = _embedding_model_fingerprint(embedding_model)

            # Check if we need to update the store
            should_update = False

            if embedding_model and not self._is_lancedb:
                # We have an embedding model but using in-memory store
                should_update = True
                logger.info("Embedding model detected, upgrading to LanceDB store")
            elif (
                embedding_model
                and self._is_lancedb
                and current_fingerprint != self._last_embedding_model_fingerprint
            ):
                # Embedding model changed, or the same model was reconfigured
                # (e.g. API key rotation) — rebuild so the new config is used.
                should_update = True
                logger.info(
                    "Embedding model configuration changed, updating LanceDB store"
                )
            elif not embedding_model and self._is_lancedb:
                # No embedding model available but using LanceDB (shouldn't happen normally)
                should_update = True
                logger.info(
                    "No embedding model available, falling back to in-memory store"
                )

            if should_update:
                if embedding_model:
                    self._memory_store = self._create_lancedb_store(embedding_model)
                    self._is_lancedb = True
                    self._last_embedding_model_id = current_model_id  # type: ignore[assignment]
                    self._last_embedding_model_fingerprint = current_fingerprint
                    logger.info("Switched to LanceDB memory store")
                else:
                    self._initialize_in_memory_store()
                    logger.info("Switched to in-memory memory store")

    def get_memory_store(self) -> MemoryStoreType:
        """
        Get the current memory store, initializing or updating as necessary.

        Returns:
            Current memory store instance
        """
        self._check_and_update_store()
        return self._memory_store  # type: ignore[return-value]

    def force_reinitialize(self) -> None:
        """Force reinitialization of the memory store."""
        with self._lock:
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

            # Return true if anything changed
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

            return {
                "store_type": type(base_store).__name__,
                "is_lancedb": self._is_lancedb,
                "embedding_model_id": self._last_embedding_model_id,
                "similarity_threshold": self._similarity_threshold,
                "supports_vector_search": self._is_lancedb,
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


def get_memory_store() -> MemoryStoreType:
    """Get the current memory store (for backward compatibility)."""
    manager = get_memory_store_manager()
    return manager.get_memory_store()


def force_reinitialize_memory_store() -> None:
    """Force reinitialization of the memory store."""
    manager = get_memory_store_manager()
    manager.force_reinitialize()
