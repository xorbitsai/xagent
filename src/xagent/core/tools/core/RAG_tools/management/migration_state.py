"""Cross-process embedding-migration state and collection write freeze."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from filelock import FileLock

from ..core.exceptions import CollectionMigratingError, ConfigurationError
from ..storage.factory import get_metadata_store
from .collection_manager import (
    _collection_thread_guard,
    _get_collection_lock,
    _get_collection_sync_impl,
    _run_in_separate_loop,
    collection_manager,
)

MIGRATION_METADATA_KEY = "embedding_migration"
DEFAULT_LOCK_TIMEOUT_SECONDS = 120


class EmbeddingMigrationPhase:
    """Values the later migration job may store; only ``frozen`` is acted on."""

    IDLE = "idle"
    FROZEN = "frozen"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    VALIDATING = "validating"
    SWITCHED = "switched"
    FAILED = "failed"


@dataclass(frozen=True)
class CollectionMigrationState:
    """Embedding-migration state persisted under ``extra_metadata``."""

    state: str = EmbeddingMigrationPhase.IDLE
    source_model_id: Optional[str] = None
    target_model_id: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_metadata(
        cls, extra_metadata: Optional[Dict[str, Any]]
    ) -> "CollectionMigrationState":
        raw = (extra_metadata or {}).get(MIGRATION_METADATA_KEY)
        if not isinstance(raw, dict):
            return cls()
        return cls(
            state=str(raw.get("state") or EmbeddingMigrationPhase.IDLE),
            source_model_id=raw.get("source_model_id"),
            target_model_id=raw.get("target_model_id"),
            updated_at=raw.get("updated_at"),
        )

    def to_metadata(self) -> Dict[str, Any]:
        return asdict(self)


def _lock_path(collection_name: str) -> Path:
    conn = get_metadata_store().get_raw_connection()
    uri = getattr(conn, "uri", None)
    if not uri:
        raise ConfigurationError(
            "Metadata store exposes no LanceDB directory for locks"
        )
    digest = hashlib.sha256(collection_name.encode("utf-8")).hexdigest()
    directory = Path(uri)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f".collection_migration.{digest}.lock"


@contextmanager
def collection_migration_lock(
    collection_name: str,
    timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Hold the cross-process lock guarding one collection's migration state."""
    lock = FileLock(str(_lock_path(collection_name)), timeout=timeout)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def read_migration_state(collection_name: str) -> CollectionMigrationState:
    """Return the stored migration state; a missing collection is idle."""
    with collection_migration_lock(collection_name):
        try:
            collection = _get_collection_sync_impl(collection_name)
        except ValueError:
            return CollectionMigrationState()
    return CollectionMigrationState.from_metadata(collection.extra_metadata)


async def _write_migration_state_async(
    collection_name: str,
    state: CollectionMigrationState,
) -> None:
    with collection_migration_lock(collection_name):
        lock = _get_collection_lock(collection_name)
        async with lock, _collection_thread_guard(collection_name):
            collection = await collection_manager.get_collection(collection_name)
            extra_metadata = dict(collection.extra_metadata or {})
            extra_metadata[MIGRATION_METADATA_KEY] = state.to_metadata()
            await collection_manager._save_collection_with_retry(
                collection.model_copy(update={"extra_metadata": extra_metadata})
            )


def write_migration_state(
    collection_name: str,
    state: CollectionMigrationState,
) -> CollectionMigrationState:
    """Persist ``state`` without touching other keys or collection fields."""
    stamped = replace(
        state,
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    )
    _run_in_separate_loop(_write_migration_state_async(collection_name, stamped))
    return stamped


def ensure_collection_writable(collection_name: str) -> None:
    """Raise before any write when the collection is frozen for migration."""
    state = read_migration_state(collection_name)
    if state.state == EmbeddingMigrationPhase.FROZEN:
        raise CollectionMigratingError(
            f"Collection '{collection_name}' is being migrated to a new embedding "
            "model; writes are disabled until it finishes.",
            details={"collection": collection_name, "migration_state": state.state},
        )
