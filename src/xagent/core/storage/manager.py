"""Storage root directory manager for xagent.

This module provides a centralized way to manage storage root directories
and related storage paths across the entire application.
"""

import threading
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Alias to avoid naming conflict with this module's get_storage_root()
from ...config import (
    get_database_url,
    get_db_pool_kwargs,
    get_default_sqlite_db_path,
)
from ...config import get_storage_root as get_config_storage_root
from ...db.sqlite import apply_sqlite_concurrency_pragmas

__all__ = [
    "StorageRootManager",
    "get_storage_root",
    "get_upload_dir",
    "initialize_storage_manager",
    "get_default_sqlite_db_path",  # re-expose for backward compatibility
    "get_default_db_url",
    "create_db_session",
    "get_db_session",
    "Base",
]


class StorageRootManager:
    """Thread-safe storage root directory manager."""

    def __init__(
        self, storage_root: str | None = None, upload_dir: str | None = None
    ) -> None:
        """Initialize the storage root manager.

        Args:
            storage_root: Path to the storage root directory. If None, uses default path.
            upload_dir: Path to the upload directory. If None, uses /tmp.
        """
        self._lock = threading.RLock()

        # Initialize storage root
        if storage_root is not None:
            self._storage_root = Path(storage_root)
        else:
            self._storage_root = get_config_storage_root()

        # Initialize upload directory
        if upload_dir is not None:
            self._upload_dir = Path(upload_dir)
        else:
            self._upload_dir = Path("/tmp")

        # Ensure the directories exist
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    def get_storage_root(self) -> Path:
        """Get the storage root directory.

        Returns:
            Path to the storage root directory
        """
        with self._lock:
            return self._storage_root

    def get_upload_dir(self) -> Path:
        """Get the upload directory.

        Returns:
            Path to the upload directory
        """
        with self._lock:
            return self._upload_dir


# Global storage root manager instance will be initialized in server startup code
_storage_manager: Optional[StorageRootManager] = None


def get_storage_root() -> Path:
    """Get the storage root directory using the global manager.

    Initializes the global storage manager with the default value if not already initialized.

    Returns:
        Path to the storage root directory
    """
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageRootManager()
    return _storage_manager.get_storage_root()


def get_upload_dir() -> Path:
    """Get the upload directory using the global manager.

    Returns:
        Path to the upload directory
    """
    if _storage_manager is None:
        raise RuntimeError(
            "Storage manager is not initialized. Call initialize_storage_manager() first."
        )
    return _storage_manager.get_upload_dir()


def initialize_storage_manager(
    storage_root: str | None = None, upload_dir: str | None = None
) -> None:
    """Initialize the global storage manager.

    Args:
        storage_root: Path to the storage root directory. If None, uses default path.
        upload_dir: Path to the upload directory. If None, uses /tmp.
    """
    global _storage_manager
    _storage_manager = StorageRootManager(storage_root, upload_dir)


# Create base model class
Base = declarative_base()

# Process-wide engine for ad-hoc sessions. ``create_db_session`` used to
# build a brand-new engine per call and never dispose it; each call left up
# to a full connection pool alive until GC, which piles real database
# connections up under load (issue #889). Cache one engine per database URL
# instead — the URL check lets tests that swap DATABASE_URL still work.
_adhoc_engine = None
_adhoc_engine_url: Optional[str] = None
_adhoc_engine_lock = threading.Lock()


def _get_adhoc_engine():  # type: ignore[no-untyped-def]
    global _adhoc_engine, _adhoc_engine_url

    database_url = get_database_url()
    # Check-and-swap runs entirely under the lock: a lock-free fast path
    # could read a torn (engine, url) pair mid-swap and hand out an engine
    # that is about to be disposed. This function is not on a hot path —
    # correctness over the micro-optimization.
    with _adhoc_engine_lock:
        if _adhoc_engine is None or _adhoc_engine_url != database_url:
            old_engine = _adhoc_engine
            if "sqlite" in database_url:
                engine = create_engine(
                    database_url,
                    connect_args={"check_same_thread": False},
                )
                # WAL + busy_timeout so concurrent writes wait for the lock
                # instead of failing with "database is locked".
                apply_sqlite_concurrency_pragmas(engine)
            else:
                # Same pool tunables as the shared web engine so operators
                # size ONE set of knobs — but remember this is a SECOND
                # pool in the same process: worst case per process is
                # 2 x (pool_size + max_overflow) against the database's
                # max_connections (see example.env).
                engine = create_engine(database_url, **get_db_pool_kwargs())
            Base.metadata.create_all(engine)
            _adhoc_engine = engine
            _adhoc_engine_url = database_url
            if old_engine is not None:
                old_engine.dispose()
        return _adhoc_engine


def create_db_session() -> Session:
    engine = _get_adhoc_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def get_db_session() -> Generator[Session, None, None]:
    db = create_db_session()
    try:
        yield db
    finally:
        db.close()


def get_default_db_url() -> str:
    """Get the default database URL.

    Deprecated: Use get_database_url() from xagent.config instead.
    This function is kept for backward compatibility.

    Returns:
        Database connection string
    """
    return get_database_url()
