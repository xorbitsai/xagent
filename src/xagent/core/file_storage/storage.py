from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import quote

from .types import StoredObject

_SHA256_METADATA_KEY = "xagent-sha256"


class FsspecFileStorage:
    """Small fsspec-backed storage wrapper using keys relative to a root URI."""

    def __init__(
        self,
        *,
        fs: Any,
        root: str,
        backend: str,
        base_uri: str,
        materialize_dir: Path,
    ) -> None:
        self._fs = fs
        self._root = root.rstrip("/")
        self._backend = backend
        self._base_uri = base_uri.rstrip("/")
        self._materialize_dir = materialize_dir

    def put_file(
        self, source: Path, key: str, content_type: str | None = None
    ) -> StoredObject:
        normalized_key = self._normalize_key(key)
        destination = self._full_path(normalized_key)
        self._makedirs_for_key(normalized_key)
        digest = hashlib.sha256()
        with (
            source.open("rb") as src,
            self._fs.open(
                destination, "wb", **self._write_open_kwargs(content_type)
            ) as dst,
        ):
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                digest.update(chunk)
                dst.write(chunk)
        checksum = digest.hexdigest()
        self._store_content_hash(normalized_key, checksum, content_type=content_type)
        return self._stored_object(normalized_key, checksum=checksum)

    def put_bytes(
        self, data: bytes, key: str, content_type: str | None = None
    ) -> StoredObject:
        normalized_key = self._normalize_key(key)
        destination = self._full_path(normalized_key)
        self._makedirs_for_key(normalized_key)
        with self._fs.open(
            destination, "wb", **self._write_open_kwargs(content_type)
        ) as dst:
            dst.write(data)
        checksum = hashlib.sha256(data).hexdigest()
        self._store_content_hash(normalized_key, checksum, content_type=content_type)
        return self._stored_object(normalized_key, checksum=checksum)

    def open_read(self, key: str) -> BinaryIO:
        return cast(
            BinaryIO,
            self._fs.open(self._full_path(self._normalize_key(key)), "rb"),
        )

    def exists(self, key: str) -> bool:
        return bool(self._fs.exists(self._full_path(self._normalize_key(key))))

    def stat(self, key: str) -> StoredObject:
        return self._stored_object(self._normalize_key(key))

    def signed_url(
        self,
        key: str,
        *,
        expires: int,
        content_type: str | None = None,
        content_disposition: str | None = None,
    ) -> str | None:
        """Return a temporary direct-access URL when the backend supports it."""
        if self._backend != "s3" or not hasattr(self._fs, "url"):
            return None

        url_kwargs: dict[str, str] = {}
        if content_type:
            url_kwargs["ResponseContentType"] = content_type
        if content_disposition:
            url_kwargs["ResponseContentDisposition"] = content_disposition

        full_path = self._full_path(self._normalize_key(key))
        try:
            return str(self._fs.url(full_path, expires=expires, **url_kwargs))
        except TypeError:
            return str(self._fs.url(full_path, expires=expires))

    def content_hash(self, key: str) -> str:
        normalized_key = self._normalize_key(key)
        if self._backend == "file":
            return self._sha256(Path(self._full_path(normalized_key)))

        metadata_hash = self._metadata_content_hash(normalized_key)
        if metadata_hash:
            return metadata_hash

        info_hash = self._info_content_hash(
            self._fs.info(self._full_path(normalized_key))
        )
        if info_hash:
            return info_hash

        raise RuntimeError(
            f"Durable storage backend {self._backend!r} did not provide a content hash "
            f"for {normalized_key!r}"
        )

    def list(self, prefix: str) -> list[StoredObject]:
        normalized_prefix = self._normalize_key(prefix).rstrip("/")
        full_prefix = self._full_path(normalized_prefix)
        if not self._fs.exists(full_prefix):
            return []
        entries = self._fs.find(full_prefix, detail=True)
        return [
            self._stored_object_from_info(self._relative_key(path), info)
            for path, info in sorted(entries.items())
            if not self._is_directory_entry(path, info)
        ]

    def delete(self, key: str) -> None:
        full_path = self._full_path(self._normalize_key(key))
        if self._fs.exists(full_path):
            self._fs.rm(full_path)

    def materialize(self, key: str, filename: str | None = None) -> Path:
        normalized_key = self._normalize_key(key)
        target_name = Path(filename or normalized_key).name or "file"
        key_digest = hashlib.sha256(normalized_key.encode("utf-8")).hexdigest()[:16]
        target_path = (
            self._materialize_dir
            / key_digest
            / self.content_hash(normalized_key)
            / target_name
        )
        if target_path.exists() and target_path.is_file():
            return target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return self._copy_to_path_atomic(normalized_key, target_path)

    def copy_to_path(self, key: str, target_path: Path) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return self._copy_to_path_atomic(self._normalize_key(key), target_path)

    def _copy_to_path_atomic(self, key: str, target_path: Path) -> Path:
        temp_file = tempfile.NamedTemporaryFile(
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        try:
            with self.open_read(key) as src, temp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            temp_path.replace(target_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return target_path

    def _full_path(self, key: str) -> str:
        if not self._root:
            return key
        return f"{self._root}/{key}"

    def _relative_key(self, full_path: str) -> str:
        normalized = str(full_path).lstrip("/")
        root = self._root.lstrip("/")
        if root and normalized.startswith(root.rstrip("/") + "/"):
            return normalized[len(root.rstrip("/") + "/") :]
        return normalized

    def _makedirs_for_key(self, key: str) -> None:
        parent = str(Path(self._full_path(key)).parent)
        if parent and parent != ".":
            self._fs.makedirs(parent, exist_ok=True)

    def _stored_object(self, key: str, checksum: str | None = None) -> StoredObject:
        info = self._fs.info(self._full_path(key))
        return self._stored_object_from_info(key, info, checksum=checksum)

    def _stored_object_from_info(
        self,
        key: str,
        info: dict[str, Any],
        checksum: str | None = None,
    ) -> StoredObject:
        etag = info.get("ETag") or info.get("etag")
        return StoredObject(
            backend=self._backend,
            key=key,
            uri=self._object_uri(key),
            size=int(info.get("size", 0)),
            checksum=checksum,
            etag=str(etag) if etag is not None else None,
        )

    @staticmethod
    def _is_directory_entry(path: str, info: dict[str, Any]) -> bool:
        entry_type = str(info.get("type", "")).lower()
        return entry_type == "directory" or str(path).rstrip("/").endswith("/")

    def _object_uri(self, key: str) -> str:
        quoted_key = quote(key, safe="/")
        return f"{self._base_uri}/{quoted_key}"

    @staticmethod
    def _write_open_kwargs(content_type: str | None) -> dict[str, str]:
        if not content_type:
            return {}
        return {"content_type": content_type}

    @staticmethod
    def _normalize_key(key: str) -> str:
        normalized = key.strip().lstrip("/")
        if not normalized or ".." in Path(normalized).parts:
            raise ValueError(f"Invalid storage key: {key!r}")
        return normalized

    def _store_content_hash(
        self, key: str, checksum: str, *, content_type: str | None
    ) -> None:
        if self._backend != "s3" or not hasattr(self._fs, "copy"):
            return
        copy_kwargs = {
            "Metadata": {_SHA256_METADATA_KEY: checksum},
            "MetadataDirective": "REPLACE",
        }
        if content_type:
            copy_kwargs["ContentType"] = content_type
        self._fs.copy(
            self._full_path(key),
            self._full_path(key),
            **copy_kwargs,
        )

    def _metadata_content_hash(self, key: str) -> str | None:
        if self._backend != "s3" or not hasattr(self._fs, "call_s3"):
            return None
        bucket, object_key, _version_id = self._fs.split_path(self._full_path(key))
        head = self._fs.call_s3("head_object", Bucket=bucket, Key=object_key)
        metadata = head.get("Metadata") or {}
        value = metadata.get(_SHA256_METADATA_KEY)
        if value:
            return str(value)
        return self._info_content_hash(head)

    @staticmethod
    def _info_content_hash(info: dict[str, Any]) -> str | None:
        for key in (
            "checksum",
            "ChecksumSHA256",
            "checksum_sha256",
            _SHA256_METADATA_KEY,
        ):
            value = info.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
