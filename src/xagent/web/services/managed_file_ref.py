from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import mimetypes
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, NoReturn, Protocol

from ...core.file_storage import (
    FsspecFileStorage,
    ScopedFileStorage,
    StoredObject,
    get_user_file_storage,
)
from ...core.file_storage.keys import (
    build_task_output_storage_key as build_task_output_storage_key,
)
from ...core.file_storage.keys import (
    build_upload_storage_key as build_upload_storage_key,
)
from ...core.file_storage.keys import safe_storage_filename as safe_storage_filename
from ..models.uploaded_file import UploadedFile

logger = logging.getLogger(__name__)
FILE_INTEGRITY_REUPLOAD_MESSAGE = (
    "File integrity verification failed. Please re-upload this file."
)


class DurableObjectMissingError(FileNotFoundError):
    """Raised when a registered file has no local copy or durable object."""


class DurableStorageOperationError(RuntimeError):
    """Raised when durable object storage is unavailable for an operation."""


class DurableObjectIntegrityError(DurableStorageOperationError):
    """Raised when restored durable bytes do not match the DB checksum."""


class UploadedFileLocalPathRecord(Protocol):
    """Fields needed to restore or locate an uploaded file without an ORM session."""

    @property
    def user_id(self) -> Any: ...

    @property
    def file_id(self) -> Any: ...

    @property
    def filename(self) -> Any: ...

    @property
    def storage_path(self) -> Any: ...

    @property
    def storage_key(self) -> Any: ...

    @property
    def storage_status(self) -> Any: ...

    @property
    def checksum(self) -> Any: ...


def guess_media_type(filename: str) -> str:
    media_type, _ = mimetypes.guess_type(filename)
    return media_type or "application/octet-stream"


def iter_file_handle(handle: Any) -> Any:
    try:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        handle.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_to_sha256_hex(checksum: str) -> str | None:
    normalized = checksum.strip()
    if normalized.lower().startswith("sha256:"):
        normalized = normalized.split(":", 1)[1]

    try:
        bytes.fromhex(normalized)
    except ValueError:
        pass
    else:
        if len(normalized) == 64:
            return normalized.lower()

    try:
        decoded = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError):
        return None
    return decoded.hex() if len(decoded) == 32 else None


def _checksum_matches(expected_checksum: str, actual_checksum: str) -> bool:
    expected_sha256 = _checksum_to_sha256_hex(expected_checksum)
    actual_sha256 = _checksum_to_sha256_hex(actual_checksum)
    return (
        expected_sha256 is not None
        and actual_sha256 is not None
        and expected_sha256 == actual_sha256
    )


@dataclass
class ManagedFileRef:
    """Registered file handle with local-first durable fallback semantics."""

    record: UploadedFileLocalPathRecord
    storage: FsspecFileStorage | ScopedFileStorage = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.storage is None:
            user_id = self.record.user_id
            if user_id is None:
                raise ValueError(
                    "Record user_id is required to bind user-scoped storage; "
                    "pass an explicit storage handle for records without an owner"
                )
            self.storage = get_user_file_storage(int(user_id))

    @property
    def local_path(self) -> Path:
        return Path(str(self.record.storage_path))

    @property
    def filename(self) -> str:
        return str(self.record.filename)

    @property
    def storage_key(self) -> str:
        return str(self.record.storage_key or "")

    @property
    def has_durable_object(self) -> bool:
        return bool(self.storage_key and self.record.storage_status == "available")

    def ensure_local(self) -> Path:
        path = self.local_path
        if path.exists() and path.is_file():
            return path

        if not self.has_durable_object:
            raise DurableObjectMissingError(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        try:
            self.storage.copy_to_path(self.storage_key, temp_path)
            self._verify_content_checksum(temp_path)
            temp_path.replace(path)
            return path
        except DurableObjectIntegrityError:
            temp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise DurableStorageOperationError(
                f"Failed to restore durable object: {self.storage_key}"
            ) from exc

    def materialize(self, *, allow_existing_local: bool = True) -> Path:
        path = self.local_path
        if allow_existing_local and path.exists() and path.is_file():
            return path

        if not self.has_durable_object:
            raise DurableObjectMissingError(path)

        last_integrity_error: DurableObjectIntegrityError | None = None
        for _attempt in range(2):
            materialized_path: Path | None = None
            try:
                materialized_path = self.storage.materialize(
                    self.storage_key, self.filename
                )
                self._verify_content_checksum(materialized_path)
                return materialized_path
            except DurableObjectIntegrityError as exc:
                last_integrity_error = exc
                if materialized_path is not None:
                    materialized_path.unlink(missing_ok=True)
            except Exception as exc:
                raise DurableStorageOperationError(
                    f"Failed to materialize durable object: {self.storage_key}"
                ) from exc

        if last_integrity_error is not None:
            raise last_integrity_error
        raise DurableStorageOperationError(
            f"Failed to materialize durable object: {self.storage_key}"
        )

    def open_read(self) -> BinaryIO:
        return self.ensure_local().open("rb")

    def signed_access_url(
        self,
        *,
        expires: int,
        content_type: str | None = None,
        content_disposition: str | None = None,
    ) -> str | None:
        if not self.has_durable_object:
            return None
        if not self._verify_durable_checksum_for_direct_access():
            return None
        try:
            return self.storage.signed_url(
                self.storage_key,
                expires=expires,
                content_type=content_type,
                content_disposition=content_disposition,
            )
        except Exception as exc:
            raise DurableStorageOperationError(
                f"Failed to sign durable object URL: {self.storage_key}"
            ) from exc

    def sync_to_durable(
        self,
        *,
        storage_key: str | None = None,
        mime_type: str | None = None,
    ) -> StoredObject:
        path = self.local_path
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        resolved_key = (
            storage_key
            or self.storage_key
            or build_upload_storage_key(
                int(self.record.user_id),
                str(self.record.file_id),
                self.filename or path.name,
            )
        )
        try:
            stored_object = self.storage.put_file(
                path,
                resolved_key,
                mime_type or getattr(self.record, "mime_type", None),
            )
        except Exception as exc:
            raise DurableStorageOperationError(
                f"Failed to write durable object: {resolved_key}"
            ) from exc
        self.apply_stored_object(stored_object)
        setattr(self.record, "file_size", path.stat().st_size)
        return stored_object

    def adopt_existing_object(
        self, expected_key: str
    ) -> Literal["adopted", "uploaded", "missing"]:
        local_path = self.local_path
        local_exists = local_path.exists() and local_path.is_file()
        try:
            stored_object = self.storage.stat(expected_key)
        except FileNotFoundError:
            return "missing"
        except Exception as exc:
            raise DurableStorageOperationError(
                f"Failed to inspect durable object metadata: {expected_key}"
            ) from exc

        checksum = stored_object.checksum
        if local_exists:
            local_size = local_path.stat().st_size
            remote_size_raw = getattr(stored_object, "size", None)
            remote_size = None if remote_size_raw is None else int(remote_size_raw)
            if remote_size is not None and remote_size != local_size:
                self.sync_to_durable(
                    storage_key=expected_key,
                    mime_type=getattr(self.record, "mime_type", None),
                )
                return "uploaded"
            if not checksum:
                try:
                    checksum = self.storage.content_hash(expected_key)
                except Exception:
                    self.sync_to_durable(
                        storage_key=expected_key,
                        mime_type=getattr(self.record, "mime_type", None),
                    )
                    return "uploaded"

            if checksum != _sha256_file(local_path):
                self.sync_to_durable(
                    storage_key=expected_key,
                    mime_type=getattr(self.record, "mime_type", None),
                )
                return "uploaded"

        if not checksum:
            try:
                checksum = self.storage.content_hash(expected_key)
            except Exception as exc:
                raise DurableStorageOperationError(
                    f"Failed to inspect durable object metadata: {expected_key}"
                ) from exc

        self.apply_stored_object(
            StoredObject(
                backend=stored_object.backend,
                key=stored_object.key,
                uri=stored_object.uri,
                size=stored_object.size,
                checksum=checksum,
                etag=stored_object.etag,
            )
        )
        return "adopted"

    def apply_stored_object(self, stored_object: StoredObject) -> None:
        if not stored_object.checksum:
            raise ValueError(
                f"Cannot mark durable object available without checksum: {stored_object.key}"
            )
        setattr(self.record, "storage_backend", stored_object.backend)
        setattr(self.record, "storage_key", stored_object.key)
        setattr(self.record, "storage_uri", stored_object.uri)
        setattr(self.record, "checksum", stored_object.checksum)
        setattr(self.record, "etag", stored_object.etag)
        setattr(self.record, "storage_status", "available")

    def delete_durable(self) -> None:
        if self.has_durable_object:
            self.storage.delete(self.storage_key)

    def _verify_content_checksum(self, path: Path) -> None:
        expected_checksum = getattr(self.record, "checksum", None)
        if not expected_checksum:
            return

        actual_checksum = _sha256_file(path)
        if _checksum_matches(str(expected_checksum), actual_checksum):
            return

        self._raise_integrity_error(str(expected_checksum), actual_checksum)

    def _verify_durable_checksum_for_direct_access(self) -> bool:
        expected_checksum = getattr(self.record, "checksum", None)
        if not expected_checksum:
            logger.warning(
                "Skipping direct durable access without DB checksum: "
                "file_id=%s storage_key=%s",
                getattr(self.record, "file_id", None),
                self.storage_key,
            )
            return False

        try:
            actual_checksum = self.storage.content_hash(self.storage_key)
        except Exception as exc:
            logger.warning(
                "Falling back to backend-mediated durable access because content "
                "hash is unavailable: file_id=%s storage_key=%s error=%s",
                getattr(self.record, "file_id", None),
                self.storage_key,
                exc,
            )
            return False

        if _checksum_matches(str(expected_checksum), str(actual_checksum)):
            return True

        self._raise_integrity_error(str(expected_checksum), str(actual_checksum))

    def _raise_integrity_error(
        self, expected_checksum: str, actual_checksum: str
    ) -> NoReturn:
        logger.error(
            "Durable object integrity check failed: file_id=%s storage_key=%s "
            "expected_checksum=%s actual_checksum=%s",
            getattr(self.record, "file_id", None),
            self.storage_key,
            expected_checksum,
            actual_checksum,
        )
        raise DurableObjectIntegrityError(FILE_INTEGRITY_REUPLOAD_MESSAGE)


def managed_file_from_record(file_record: UploadedFile) -> ManagedFileRef:
    return ManagedFileRef(file_record)


def ensure_uploaded_file_local_path(file_record: UploadedFileLocalPathRecord) -> Path:
    try:
        return ManagedFileRef(file_record).ensure_local()
    except DurableObjectMissingError:
        return Path(str(file_record.storage_path))


def create_uploaded_file_from_local_path(
    *,
    local_path: Path,
    user_id: int,
    filename: str | None = None,
    file_id: str | None = None,
    task_id: int | None = None,
    mime_type: str | None = None,
    storage_key: str | None = None,
    workspace_relative_path: str | None = None,
    workspace_category: str | None = None,
) -> UploadedFile:
    from .uploaded_file_store import create_unbound_uploaded_file_from_local_path

    return create_unbound_uploaded_file_from_local_path(
        local_path=local_path,
        user_id=user_id,
        filename=filename,
        file_id=file_id,
        task_id=task_id,
        mime_type=mime_type,
        storage_key=storage_key,
        workspace_relative_path=workspace_relative_path,
        workspace_category=workspace_category,
    )
