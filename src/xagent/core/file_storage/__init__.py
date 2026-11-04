"""Durable file storage abstraction for user-visible files."""

from .factory import (
    get_file_storage_backend,
    get_unscoped_file_storage,
    get_user_file_storage,
)
from .scoped import ScopedFileStorage, StorageKeyScopeError
from .storage import FsspecFileStorage, normalize_storage_key
from .types import StoredObject

__all__ = [
    "FsspecFileStorage",
    "ScopedFileStorage",
    "StorageKeyScopeError",
    "StoredObject",
    "get_file_storage_backend",
    "get_unscoped_file_storage",
    "get_user_file_storage",
    "normalize_storage_key",
]
