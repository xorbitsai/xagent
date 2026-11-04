from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoredObject:
    """Metadata for an object stored in durable file storage."""

    backend: str
    key: str
    uri: str
    size: int
    checksum: str | None = None
    etag: str | None = None
