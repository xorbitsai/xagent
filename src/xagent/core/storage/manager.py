"""Storage root directory manager for xagent.

This module provides a centralized way to manage storage root directories
and related storage paths across the entire application.
"""

import os
import threading
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker


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
            self._storage_root = Path.home() / ".xagent"

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


def get_default_sqlite_db_path() -> str:
    import os

    storage_root = get_storage_root()
    return os.path.join(storage_root, "xagent.db")


def get_default_db_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url is not None:
        return database_url

    db_path = get_default_sqlite_db_path()
    return f"sqlite:///{db_path}"


# Create base model class
Base = declarative_base()


def create_db_session() -> Session:
    database_url = get_default_db_url()
    connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}
    # Create database engine
    engine = create_engine(database_url, connect_args=connect_args)
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    Base.metadata.create_all(engine)
    return db


def get_db_session() -> Generator[Session, None, None]:
    db = create_db_session()
    try:
        yield db
    finally:
        db.close()
