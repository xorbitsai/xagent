from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from .storage import FsspecFileStorage, normalize_storage_key
from .types import StoredObject


class StorageKeyScopeError(Exception):
    """Raised when a storage key falls outside a handle's bound prefix.

    Deliberately not a ``ValueError`` so scope violations stay distinguishable
    from the structural normalization errors raised by
    ``normalize_storage_key``.
    """


class ScopedFileStorage:
    """Storage view bound to a required key prefix.

    Every operation normalizes the key (same strict/tolerant policy as
    ``FsspecFileStorage``), then verifies it falls under the bound prefix
    before delegating. Containment is separator-aware: binding ``users/1``
    does not admit ``users/10/...``.
    """

    def __init__(self, *, storage: FsspecFileStorage, prefix: str) -> None:
        self._storage = storage
        self._prefix = normalize_storage_key(prefix)

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def backend(self) -> str:
        return self._storage.backend

    @property
    def base_uri(self) -> str:
        return self._storage.base_uri

    def _scoped(self, key: str, *, strict: bool = True) -> str:
        normalized = normalize_storage_key(key, strict=strict)
        if normalized != self._prefix and not normalized.startswith(self._prefix + "/"):
            raise StorageKeyScopeError(
                f"Storage key {key!r} is outside the bound prefix {self._prefix!r}"
            )
        return normalized

    def put_file(
        self, source: Path, key: str, content_type: str | None = None
    ) -> StoredObject:
        return self._storage.put_file(source, self._scoped(key), content_type)

    def put_bytes(
        self, data: bytes, key: str, content_type: str | None = None
    ) -> StoredObject:
        return self._storage.put_bytes(data, self._scoped(key), content_type)

    def open_read(self, key: str) -> BinaryIO:
        return self._storage.open_read(self._scoped(key, strict=False))

    def exists(self, key: str) -> bool:
        return self._storage.exists(self._scoped(key, strict=False))

    def stat(self, key: str) -> StoredObject:
        return self._storage.stat(self._scoped(key, strict=False))

    def signed_url(
        self,
        key: str,
        *,
        expires: int,
        content_type: str | None = None,
        content_disposition: str | None = None,
    ) -> str | None:
        return self._storage.signed_url(
            self._scoped(key, strict=False),
            expires=expires,
            content_type=content_type,
            content_disposition=content_disposition,
        )

    def content_hash(self, key: str) -> str:
        return self._storage.content_hash(self._scoped(key, strict=False))

    def list(self, prefix: str) -> list[StoredObject]:
        return self._storage.list(self._scoped(prefix))

    def delete(self, key: str) -> None:
        self._storage.delete(self._scoped(key, strict=False))

    def materialize(self, key: str, filename: str | None = None) -> Path:
        return self._storage.materialize(self._scoped(key, strict=False), filename)

    def materialized_path(self, key: str, filename: str | None = None) -> Path:
        return self._storage.materialized_path(
            self._scoped(key, strict=False), filename
        )

    def copy_to_path(self, key: str, target_path: Path) -> Path:
        return self._storage.copy_to_path(self._scoped(key, strict=False), target_path)
