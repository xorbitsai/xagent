"""
LanceDB vector store implementation with integrated connection management.

This module provides both connection management and vector store implementation
for LanceDB, combining the functionality that was previously split across
lancedb_client.py and a separate vector store implementation.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import lancedb
from lancedb.db import DBConnection

from .base import VectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "LanceDBConnectionManager",
    "LanceDBVectorStore",
    "get_connection",
    "get_connection_from_env",
]

# Connection cache: key -> (connection, last_accessed_timestamp)
_connection_cache: Dict[str, Tuple[DBConnection, float]] = {}
_cache_lock = RLock()

# Connection TTL (seconds), default 5 minutes
CONNECTION_TTL = int(os.getenv("LANCEDB_CONNECTION_TTL", "300"))


class LanceDBConnectionManager:
    """
    LanceDB connection manager with caching and automatic cleanup.

    This class handles connection lifecycle management, caching, and
    automatic cleanup of expired connections.
    """

    @staticmethod
    def _normalize_dirpath(db_dir: str) -> str:
        """Normalize database directory path."""
        return str(Path(db_dir).expanduser().resolve())

    @staticmethod
    def _ensure_dir(db_dir: str) -> None:
        """Ensure database directory exists."""
        Path(db_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_connection_expired(last_accessed: float) -> bool:
        """Check if a connection has expired based on TTL."""
        return time.time() - last_accessed > CONNECTION_TTL

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_default_lancedb_dir() -> str:
        """Get the default LanceDB directory path.

        Returns:
            Default LanceDB directory path

        Note:
            Priority: legacy_location (if has data) > ~/.xagent/data/lancedb
            Result is cached after first call.
        """
        # Check legacy location (project root) first for backward compatibility
        legacy_dir = (
            Path(__file__).parent.parent.parent.parent.parent / "data" / "lancedb"
        )
        if legacy_dir.is_dir() and list(legacy_dir.iterdir()):
            logger.info(f"Using legacy LanceDB location: {legacy_dir}")
            return str(legacy_dir)

        # Use new default location
        new_dir = Path.home() / ".xagent" / "data" / "lancedb"
        new_dir.mkdir(parents=True, exist_ok=True)
        return str(new_dir)

    @staticmethod
    def _cleanup_expired_connections() -> None:
        """Remove expired connections from cache."""
        current_time = time.time()
        expired_keys = []

        for key, (conn, last_accessed) in _connection_cache.items():
            if current_time - last_accessed > CONNECTION_TTL:
                expired_keys.append(key)

        for key in expired_keys:
            conn, _ = _connection_cache.pop(key)
            try:
                # Close connection if it has a close method
                if hasattr(conn, "close"):
                    conn.close()
            except Exception as e:
                # Ignore errors during connection cleanup but log them
                logger.warning("Error closing expired connection for %s: %s", key, e)
                pass

    def get_connection(self, db_dir: str) -> DBConnection:
        """
        Get LanceDB connection with caching.

        Args:
            db_dir: Database directory path

        Returns:
            LanceDB connection

        Raises:
            ValueError: If db_dir is empty
        """
        if not db_dir:
            raise ValueError("LanceDB directory path must be non-empty")

        normalized = self._normalize_dirpath(db_dir)
        self._ensure_dir(normalized)

        current_time = time.time()

        with _cache_lock:
            # Cleanup expired connections
            self._cleanup_expired_connections()

            if normalized in _connection_cache:
                conn, last_accessed = _connection_cache[normalized]
                # Check if connection is expired
                if not self._is_connection_expired(last_accessed):
                    # Update last access time
                    _connection_cache[normalized] = (conn, current_time)
                    return conn
                else:
                    # Remove expired connection
                    _connection_cache.pop(normalized)

            # Create new connection
            conn = lancedb.connect(normalized)
            _connection_cache[normalized] = (conn, current_time)
            return conn

    def get_connection_from_env(self, env_var: str = "LANCEDB_DIR") -> DBConnection:
        """
        Get LanceDB connection from environment variable with fallback to default path.

        If the environment variable is not set, uses get_default_lancedb_dir() which:
        1. Checks legacy location (project root data/lancedb) if it contains data
        2. Otherwise uses ~/.xagent/data/lancedb

        Args:
            env_var: Environment variable name containing database directory

        Returns:
            LanceDB connection

        Raises:
            ValueError: If environment variable is empty
            KeyError: If environment variable (other than LANCEDB_DIR) is not set
        """
        db_dir = os.getenv(env_var)

        if db_dir is None:
            if env_var == "LANCEDB_DIR":
                # Use default path only for the standard LANCEDB_DIR environment variable
                db_dir = self.get_default_lancedb_dir()
                logger.info(f"Using default LanceDB directory: {db_dir}")
            else:
                # For other environment variables, raise KeyError as before
                raise KeyError(f"Environment variable {env_var} is not set")
        elif db_dir.strip() == "":
            raise ValueError(f"Environment variable {env_var} is empty")

        return self.get_connection(db_dir)


class LanceDBVectorStore(VectorStore):
    """
    LanceDB vector store implementation.

    This class implements the standard VectorStore interface using LanceDB
    as the backend storage engine.
    """

    support_store_texts: ClassVar[bool] = True

    def __init__(
        self,
        db_dir: str,
        collection_name: str = "vectors",
        connection_manager: Optional[LanceDBConnectionManager] = None,
    ):
        """
        Initialize LanceDB vector store.

        Args:
            db_dir: Database directory path
            collection_name: Collection/table name for vectors
            connection_manager: Optional connection manager instance
        """
        self._db_dir = db_dir
        self._collection_name = collection_name
        self._conn_manager = connection_manager or LanceDBConnectionManager()
        self._conn = self._conn_manager.get_connection(db_dir)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure the vector table exists."""
        try:
            self._conn.open_table(self._collection_name)
        except Exception as e:
            logger.debug(
                "Table %s does not exist or open failed (%s), creating new table.",
                self._collection_name,
                e,
            )
            # Table doesn't exist, create it with sample data
            # LanceDB needs actual data to properly infer vector column type
            sample_data = [
                {
                    "id": "sample",
                    "vector": [0.0, 0.0, 0.0],  # Sample vector
                    "text": "sample",
                    "metadata": "{}",
                }
            ]
            table = self._conn.create_table(self._collection_name, data=sample_data)
            # Remove sample data
            table.delete("id = 'sample'")

    def add_vectors(
        self,
        vectors: List[List[float]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add vectors to the store.

        Args:
            vectors: List of vectors to add
            ids: Optional list of IDs for each vector
            metadatas: Optional list of metadata dicts

        Returns:
            List of vector IDs stored
        """
        import json
        from uuid import uuid4

        if ids is None:
            ids = [str(uuid4()) for _ in vectors]

        if metadatas is None:
            metadatas = [{} for _ in vectors]

        # Prepare data for insertion
        data = []
        for i, vector in enumerate(vectors):
            record = {
                "id": ids[i],
                "vector": vector,
                "text": metadatas[i].get("text", ""),
                "metadata": json.dumps(metadatas[i], ensure_ascii=False),
            }
            data.append(record)

        # Insert data
        table = self._conn.open_table(self._collection_name)
        table.add(data)

        return ids

    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors from the store by their IDs.

        Args:
            ids: List of vector IDs to delete

        Returns:
            True if deletion was successful
        """
        try:
            table = self._conn.open_table(self._collection_name)

            # Build delete condition
            id_conditions = " OR ".join([f"id = '{id_}'" for id_ in ids])
            table.delete(id_conditions)

            return True
        except Exception as e:
            logger.error("Failed to delete vectors: %s", e)
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for vectors similar to the query vector.

        Args:
            query_vector: Vector to search for
            top_k: Number of top similar vectors to return
            filters: Optional metadata filters

        Returns:
            List of search results with id, score, and metadata
        """
        import json

        table = self._conn.open_table(self._collection_name)

        # Perform vector search, explicitly specify vector column
        results = (
            table.search(query_vector, vector_column_name="vector")
            .limit(top_k)
            .to_pandas()
        )

        # Format results
        formatted_results = []
        for _, row in results.iterrows():
            try:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            result = {
                "id": row["id"],
                "score": float(row["_distance"]) if "_distance" in row else 0.0,
                "metadata": metadata,
            }
            formatted_results.append(result)

        return formatted_results

    def clear(self) -> None:
        """Clear all vectors and metadata from the store."""
        try:
            # Try to delete all records
            table = self._conn.open_table(self._collection_name)
            table.delete("true")  # Delete all records
        except Exception as e:
            logger.error("Failed to clear vector store: %s", e)
            # Table doesn't exist, just ensure it's created
            self._ensure_table()

    def get_raw_connection(self) -> DBConnection:
        """
        Get raw LanceDB connection for advanced operations.

        This method provides access to the underlying LanceDB connection
        for operations that go beyond the standard VectorStore interface.

        Returns:
            Raw LanceDB connection
        """
        return self._conn


# Convenience functions
def get_connection(db_dir: str) -> DBConnection:
    """
    Get LanceDB connection with caching.

    Args:
        db_dir: Database directory path

    Returns:
        LanceDB connection
    """
    manager = LanceDBConnectionManager()
    return manager.get_connection(db_dir)


def get_connection_from_env(env_var: str = "LANCEDB_DIR") -> DBConnection:
    """
    Get LanceDB connection from environment variable with fallback to default path.

    If LANCEDB_DIR is not set, uses get_default_lancedb_dir() which checks legacy
    location first, then falls back to ~/.xagent/data/lancedb.

    Args:
        env_var: Environment variable name

    Returns:
        LanceDB connection
    """
    manager = LanceDBConnectionManager()
    return manager.get_connection_from_env(env_var)
