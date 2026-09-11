"""Dynamic memory store manager for web application."""

import json
import logging
import os
import threading
from contextlib import contextmanager
from hashlib import sha256
from typing import Any, Iterator, Optional, Union, cast

from filelock import FileLock, Timeout

from ..core.memory.in_memory import InMemoryMemoryStore
from ..core.memory.lancedb import LanceDBMemoryStore
from ..core.memory.lancedb_maintenance import (
    DEFAULT_LOCK_TIMEOUT,
    MaintenanceStatus,
    maintain_lancedb_memory_table,
)
from ..core.memory.vector_compatibility import (
    VECTOR_IDENTITY_METADATA_KEY,
    VectorCompatibility,
    canonical_embedding_identity,
    create_or_recreate_vector_capable_table,
    open_lancedb_table_if_exists,
)
from ..core.model import EmbeddingModelConfig
from ..core.model.embedding.adapter import (
    UnsupportedEmbeddingProviderError,
    create_embedding_adapter,
)
from ..core.model.embedding.base import BaseEmbedding
from ..core.storage.manager import get_storage_root
from ..core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from .models.database import get_db
from .models.model import Model as DBModel
from .models.user import UserDefaultModel
from .services.db_runtime import is_database_pool_timeout
from .user_isolated_memory import UserIsolatedMemoryStore

logger = logging.getLogger(__name__)


class MemoryAdmissionLockTimeout(RuntimeError):
    """Raised when startup cannot acquire the cross-process admission lock."""


class InvalidEmbeddingModelConfiguration(ValueError):
    """Raised when persisted embedding fields cannot form a model config."""


class MemoryStoreStartupAdmissionError(RuntimeError):
    """Raised when persistent memory cannot be safely admitted at startup."""


class MemoryStoreRestartRequired(RuntimeError):
    """Raised when the published store no longer matches authoritative config."""


MEMORY_STORE_RESTART_REQUIRED_DETAIL = (
    "Persistent memory configuration changed after startup. Quiesce all API, "
    "worker, and scheduler memory writers, restart every worker, and keep memory "
    "ingress closed until startup admission completes. Identity-changing repairs "
    "must be performed offline."
)
MEMORY_STORE_STARTUP_REPAIR_DETAIL = (
    "Persistent memory startup admission found legacy data that cannot be safely "
    "projected. Quiesce all API, worker, and scheduler memory writers and repair "
    "the table offline before restarting every worker."
)


# Type alias for our memory store types that includes user isolation
MemoryStoreType = Union[
    InMemoryMemoryStore, LanceDBMemoryStore, UserIsolatedMemoryStore
]


def _embedding_model_config(model: DBModel) -> EmbeddingModelConfig:
    """Preserve the complete shared embedding configuration and credential."""
    api_key = model.api_key
    try:
        dimension = int(model.dimension) if model.dimension is not None else None
        max_retries = int(model.max_retries) if model.max_retries is not None else 10
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingModelConfiguration from exc
    try:
        return EmbeddingModelConfig(
            id=str(model.model_id),
            model_provider=str(model.model_provider),
            model_name=str(model.model_name),
            api_key=str(api_key) if api_key is not None else None,
            base_url=str(model.base_url) if model.base_url else None,
            dimension=dimension,
            instruct=None,
            max_retries=max_retries,
        )
    except ValueError as exc:
        raise InvalidEmbeddingModelConfiguration from exc


def _embedding_model_fingerprint(model: Optional[DBModel]) -> str:
    """Hash the complete authoritative adapter input without retaining secrets."""
    if model is None:
        payload: dict[str, Any] = {"default": None}
    else:
        api_key = getattr(model, "api_key", None)
        api_key_digest = sha256(
            ("" if api_key is None else str(api_key)).encode("utf-8")
        ).hexdigest()
        payload = {
            "database_id": getattr(model, "id", None),
            "default_id": getattr(model, "_memory_default_id", None),
            "default_user_id": getattr(model, "_memory_default_user_id", None),
            "model_id": getattr(model, "model_id", None),
            "category": getattr(model, "category", None),
            "model_provider": getattr(model, "model_provider", None),
            "model_name": getattr(model, "model_name", None),
            "base_url": getattr(model, "base_url", None),
            "dimension": getattr(model, "dimension", None),
            "instruct": None,
            "max_retries": getattr(model, "max_retries", None),
            "is_active": getattr(model, "is_active", None),
            "created_at": str(getattr(model, "created_at", None)),
            "updated_at": str(getattr(model, "updated_at", None)),
            "api_key_sha256": api_key_digest,
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


class DynamicMemoryStoreManager:
    """Publish one startup-admitted store and reject configuration drift."""

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
        self._admitted_embedding_fingerprint: Optional[str] = None
        self._startup_admission_error: Optional[MemoryStoreStartupAdmissionError] = None

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

    def _get_embedding_model_from_db(
        self, *, fail_fast: bool = False
    ) -> Optional[DBModel]:
        """Resolve one deterministic shared/admin default for the shared table."""
        try:
            db = next(get_db())
            try:
                from .services.model_service import _get_visible_user_ids

                default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.config_type == "embedding",
                        UserDefaultModel.user_id.in_(_get_visible_user_ids(db, None)),
                        DBModel.category == "embedding",
                        DBModel.is_active.is_(True),
                    )
                    .order_by(UserDefaultModel.user_id, UserDefaultModel.id)
                    .first()
                )
                if default is None:
                    return None
                model = cast(DBModel, default.model)
                model._memory_default_id = default.id
                model._memory_default_user_id = default.user_id
                return model
            finally:
                db.close()
        except Exception as e:
            if fail_fast or is_database_pool_timeout(e):
                raise
            logger.error(f"Error checking for embedding model: {e}")
            return None

    def _memory_db_dir(self, *, create: bool) -> Optional[str]:
        """Resolve storage without creating artifacts during discovery."""
        legacy_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "memory_store",
        )
        if os.path.exists(legacy_dir) and os.listdir(legacy_dir):
            logger.info(f"Using legacy memory store location: {legacy_dir}")
            return legacy_dir

        new_dir = get_storage_root() / "memory_store"
        if create:
            os.makedirs(new_dir, exist_ok=True)
        elif not new_dir.is_dir():
            return None
        return str(new_dir)

    def _create_lancedb_store(
        self,
        embedding_model: Optional[BaseEmbedding],
        *,
        db_dir: Optional[str] = None,
    ) -> UserIsolatedMemoryStore:
        """Create a dormant filesystem-backed store without touching its table."""
        resolved_db_dir = db_dir or self._memory_db_dir(create=True)
        if resolved_db_dir is None:  # pragma: no cover - create=True always resolves
            raise RuntimeError("memory storage directory could not be resolved")

        lancedb_store = LanceDBMemoryStore(
            db_dir=resolved_db_dir,
            embedding_model=embedding_model,
            similarity_threshold=self._similarity_threshold or 1.5,
            initialize_schema=False,
            include_null_vector_fallback=True,
        )
        logger.info("Created dormant LanceDB memory store")
        return UserIsolatedMemoryStore(lancedb_store)

    @staticmethod
    @contextmanager
    def _lifecycle_lock(connection: Any, table_name: str) -> Iterator[None]:
        """Serialize startup admission across processes for one local table.

        This outer lock is always acquired before the maintenance lock. Keeping
        the two lock files distinct and the acquisition order one-way avoids a
        lock cycle while covering the re-open/classify/overwrite window.
        """
        uri = str(getattr(connection, "uri", "") or "")
        if not uri or "://" in uri or not os.path.isdir(uri):
            raise ValueError(
                "LanceDB memory admission requires a writable local database URI"
            )
        digest = sha256(table_name.encode()).hexdigest()[:16]
        lock = FileLock(
            os.path.join(uri, f".memory-admission-{digest}.lock"),
            timeout=DEFAULT_LOCK_TIMEOUT,
        )
        try:
            lock.acquire()
        except Timeout as exc:
            raise MemoryAdmissionLockTimeout(
                f"Timed out after {DEFAULT_LOCK_TIMEOUT}s acquiring memory "
                f"admission lock for table {table_name!r}"
            ) from exc
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def _require_complete_maintenance(connection: Any, table_name: str) -> None:
        outcome = maintain_lancedb_memory_table(connection, table_name)
        if outcome.status is not MaintenanceStatus.COMPLETE:
            raise RuntimeError(
                f"memory maintenance failed: {outcome.status.value}: {outcome.detail}"
            )

    def run_startup_compatibility_lifecycle(self) -> None:
        """Admit and maintain shared memory before runtime writers start."""
        with self._lock:
            model = self._get_embedding_model_from_db(fail_fast=True)
            admitted_fingerprint = _embedding_model_fingerprint(model)
            if model is None:
                db_dir = self._memory_db_dir(create=False)
                if db_dir is None:
                    self._admitted_embedding_fingerprint = admitted_fingerprint
                    self._startup_admission_error = None
                    return
                new_store = self._create_lancedb_store(None, db_dir=db_dir)
                text_base_store = cast(LanceDBMemoryStore, new_store._base_store)
                connection = text_base_store._vector_store.get_raw_connection()
                table_name = text_base_store._collection_name
                with self._lifecycle_lock(connection, table_name):
                    table = open_lancedb_table_if_exists(connection, table_name)
                    if table is None:
                        self._admitted_embedding_fingerprint = admitted_fingerprint
                        self._startup_admission_error = None
                        return
                    _safe_close_table(table)
                    outcome = maintain_lancedb_memory_table(connection, table_name)
                    if outcome.status is MaintenanceStatus.INVALID_LEGACY_DATA:
                        logger.error(MEMORY_STORE_STARTUP_REPAIR_DETAIL)
                        self._startup_admission_error = (
                            MemoryStoreStartupAdmissionError(
                                MEMORY_STORE_STARTUP_REPAIR_DETAIL
                            )
                        )
                        raise self._startup_admission_error
                    if outcome.status is not MaintenanceStatus.COMPLETE:
                        raise RuntimeError(
                            "memory maintenance failed: "
                            f"{outcome.status.value}: {outcome.detail}"
                        )
                self._memory_store = new_store
                self._is_lancedb = True
                self._last_embedding_model_id = None
                self._admitted_embedding_fingerprint = admitted_fingerprint
                self._startup_admission_error = None
                return
            # Model validation and adapter construction are intentionally kept
            # ahead of every LanceDB/filesystem operation. Only definite model
            # configuration errors degrade; backend failures still abort startup.
            try:
                embedding_config = _embedding_model_config(model)
            except InvalidEmbeddingModelConfiguration:
                logger.warning(
                    "Configured default embedding model has invalid fields; "
                    "preserving the current memory store"
                )
                self._admitted_embedding_fingerprint = admitted_fingerprint
                return
            try:
                canonical_embedding_identity(embedding_config)
            except ValueError:
                logger.warning(
                    "Configured default embedding model has an invalid vector "
                    "identity; preserving the current memory store"
                )
                self._admitted_embedding_fingerprint = admitted_fingerprint
                return
            try:
                embedding_model = create_embedding_adapter(embedding_config)
            except UnsupportedEmbeddingProviderError:
                logger.warning(
                    "Configured default embedding model uses an unsupported "
                    "provider; preserving the current memory store"
                )
                self._admitted_embedding_fingerprint = admitted_fingerprint
                return

            new_store = self._create_lancedb_store(embedding_model)
            base_store = new_store._base_store
            if not isinstance(base_store, LanceDBMemoryStore):
                raise TypeError(
                    "startup memory lifecycle requires a LanceDB base store"
                )
            connection = base_store._vector_store.get_raw_connection()
            table_name = base_store._collection_name

            with self._lifecycle_lock(connection, table_name):
                table = None
                table_existed = False
                had_vector = False
                missing_identity = False
                try:
                    table = open_lancedb_table_if_exists(connection, table_name)
                    table_existed = table is not None
                    if table is not None:
                        had_vector = "vector" in table.schema.names
                        missing_identity = (
                            had_vector
                            and VECTOR_IDENTITY_METADATA_KEY
                            not in (table.schema.metadata or {})
                        )
                        outcome = maintain_lancedb_memory_table(connection, table_name)
                        if outcome.status is MaintenanceStatus.INVALID_LEGACY_DATA:
                            logger.error(MEMORY_STORE_STARTUP_REPAIR_DETAIL)
                            self._startup_admission_error = (
                                MemoryStoreStartupAdmissionError(
                                    MEMORY_STORE_STARTUP_REPAIR_DETAIL
                                )
                            )
                            raise self._startup_admission_error
                        elif outcome.status is not MaintenanceStatus.COMPLETE:
                            raise RuntimeError(
                                "memory maintenance failed: "
                                f"{outcome.status.value}: {outcome.detail}"
                            )
                finally:
                    _safe_close_table(table)

                if base_store._embedding_model is not None:
                    compatibility = create_or_recreate_vector_capable_table(
                        connection, table_name, embedding_config
                    )
                    if compatibility is VectorCompatibility.MISMATCHING:
                        caveat = (
                            " The table has no exact legacy vector identity; "
                            "rolling back to the historical dimension or endpoint "
                            "will not make it safe. Quiesce writers and repair or "
                            "re-embed the table offline."
                            if missing_identity
                            else ""
                        )
                        logger.warning(
                            "Persistent memory vector identity does not match the "
                            "configured embedding model; continuing with text-only "
                            "search.%s",
                            caveat,
                        )
                        base_store._embedding_model = None
                    elif not table_existed or not had_vector:
                        # Creation and vectorless overwrite both replace schema
                        # metadata, so establish the completion marker now.
                        self._require_complete_maintenance(connection, table_name)

            self._memory_store = new_store
            self._is_lancedb = True
            self._last_embedding_model_id = cast(int, model.id)
            self._admitted_embedding_fingerprint = admitted_fingerprint
            self._startup_admission_error = None

    def get_memory_store(self) -> MemoryStoreType:
        """
        Get the startup-admitted store after verifying authoritative identity.

        Returns:
            Current memory store instance
        """
        with self._lock:
            if self._startup_admission_error is not None:
                raise self._startup_admission_error
            admitted = self._admitted_embedding_fingerprint
            if admitted is not None:
                try:
                    current_model = self._get_embedding_model_from_db(fail_fast=True)
                    current = _embedding_model_fingerprint(current_model)
                except Exception as exc:
                    raise MemoryStoreRestartRequired(
                        MEMORY_STORE_RESTART_REQUIRED_DETAIL
                    ) from exc
                if current != admitted:
                    raise MemoryStoreRestartRequired(
                        MEMORY_STORE_RESTART_REQUIRED_DETAIL
                    )
            return self._memory_store  # type: ignore[return-value]

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
                "supports_vector_search": bool(
                    self._is_lancedb
                    and isinstance(base_store, LanceDBMemoryStore)
                    and base_store._embedding_model is not None
                ),
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
    """Get the admitted store, failing closed after configuration drift."""
    manager = get_memory_store_manager()
    return manager.get_memory_store()
