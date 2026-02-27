"""Dynamic memory store manager for web application."""

import logging
import threading
from typing import Optional, Union

from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore

from .models.database import get_db
from .models.model import Model as DBModel
from .models.user import UserDefaultModel
from .user_isolated_memory import UserIsolatedMemoryStore, current_user_id

logger = logging.getLogger(__name__)

# Type alias for our memory store types that includes user isolation
MemoryStoreType = Union[
    InMemoryMemoryStore, LanceDBMemoryStore, UserIsolatedMemoryStore
]


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
            logger.info("Initialized with in-memory store")

    def _get_embedding_model_from_db(self) -> Optional[DBModel]:
        """Get the current embedding model from database."""
        try:
            db = next(get_db())
            try:
                # Get current user ID from context
                user_id = current_user_id.get()

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
                            logger.info(
                                f"Found user's default embedding model: {embedding_model.model_id}"
                            )
                            return embedding_model
                        else:
                            logger.warning(
                                f"User default embedding model {user_default.model_id} not found or inactive"
                            )

                # Fallback: look for any active embedding model
                embedding_model = (
                    db.query(DBModel)
                    .filter(
                        DBModel.category == "embedding",
                        DBModel.is_active,
                    )
                    .first()
                )

                if embedding_model:
                    logger.info(
                        f"Using system active embedding model (user has no default): {embedding_model.model_id}"
                    )
                else:
                    logger.info("No active embedding model found")

                return embedding_model
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error checking for embedding model: {e}")
            return None

    def _create_lancedb_store(
        self, embedding_model: DBModel
    ) -> UserIsolatedMemoryStore:
        """Create LanceDB store with the given embedding model."""
        try:
            import os

            from xagent.core.model.embedding import DashScopeEmbedding

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
                new_dir = os.path.expanduser("~/.xagent/memory_store")
                os.makedirs(new_dir, exist_ok=True)
                db_dir = new_dir

            if embedding_model.model_provider == "dashscope":
                lancedb_store = LanceDBMemoryStore(
                    db_dir=db_dir,
                    embedding_model=DashScopeEmbedding(
                        api_key=str(embedding_model.api_key),
                        dimension=int(embedding_model.dimension or 1024),
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
        embedding_model = self._get_embedding_model_from_db()
        current_model_id = embedding_model.id if embedding_model else None

        with self._lock:
            # Check if we need to update the store
            should_update = False

            if embedding_model and not self._is_lancedb:
                # We have an embedding model but using in-memory store
                should_update = True
                logger.info("Embedding model detected, upgrading to LanceDB store")
            elif (
                embedding_model
                and self._is_lancedb
                and current_model_id != self._last_embedding_model_id
            ):
                # Embedding model has changed
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
            old_model_id = self._last_embedding_model_id

            self._check_and_update_store()

            # Return true if anything changed
            return (
                old_is_lancedb != self._is_lancedb
                or old_model_id != self._last_embedding_model_id
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
