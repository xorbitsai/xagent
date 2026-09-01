from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, cast
from uuid import UUID, uuid4

from sqlalchemy import update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ...config import get_storage_root, get_uploads_dir
from ...core.execution_scope import (
    EXECUTION_SCOPE_NOT_PROVIDED,
    ExecutionScope,
    ExecutionScopeInput,
)
from ...core.file_storage import get_user_file_storage
from ...core.file_storage.keys import (
    build_upload_generation_storage_key,
    build_upload_storage_key,
)
from ...core.workspace import scoped_user_root
from ..models.database import get_session_local, release_db_connection_if_clean
from ..models.task import Task
from ..models.uploaded_file import UploadedFile, uploaded_file_bind_values
from .managed_file_ref import (
    DurableStorageOperationError,
    ManagedFileRef,
    guess_media_type,
)

logger = logging.getLogger(__name__)

_SessionFactory = Callable[[], Session]


def _session_factory_from_reference(db: Session) -> _SessionFactory:
    """Create short store-owned Sessions on the caller's database engine."""

    bind = db.get_bind()
    if isinstance(bind, Connection):
        bind = bind.engine
    return cast(
        _SessionFactory,
        sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=bind,
        ),
    )


@dataclass(frozen=True)
class LocalUploadRegistration:
    """Primitive input for one durable upload registration."""

    local_path: Path
    user_id: int
    file_id: str
    task_id: int | None
    filename: str
    mime_type: str | None
    upload_source: str | None = None
    execution_scope: ExecutionScope | None = None

    @property
    def compensation_claim(self) -> "RegisteredUploadCompensationClaim":
        """Return the exact request-created identity eligible for rollback."""

        return RegisteredUploadCompensationClaim(
            user_id=self.user_id,
            file_id=self.file_id,
            expected_task_id=self.task_id,
            expected_storage_key=build_upload_storage_key(
                self.user_id,
                self.file_id,
                self.filename,
                scope_segments=(
                    self.execution_scope.durable_storage_segments
                    if self.execution_scope is not None
                    else ()
                ),
            ),
        )


@dataclass(frozen=True)
class RegisteredUploadCompensationClaim:
    """CAS fence for rolling back one request-created upload registration."""

    user_id: int
    file_id: str
    expected_task_id: int | None
    expected_storage_key: str


@dataclass(frozen=True)
class ClaimedRegisteredUploadCompensation:
    """Exact metadata version hidden while its durable object is reconciled."""

    row_id: int
    user_id: int
    file_id: str
    expected_task_id: int | None
    expected_storage_key: str
    claimed_at: datetime


@dataclass(frozen=True)
class UploadedFileRegistrationSnapshot:
    """Detached response fields returned after metadata commit."""

    file_id: str
    filename: str
    file_size: int
    mime_type: str | None


@dataclass(frozen=True)
class StagedUploadedFile:
    """Detached metadata for bytes durably written before a DB transaction.

    The object represented here is immutable and already checksum-verified by
    :class:`ManagedFileRef`.  It is deliberately not an ORM instance: storage
    staging can cross an async boundary, while the later metadata transaction
    owns its own Session and materializes or updates the row there.
    """

    file_id: str
    user_id: int
    task_id: int | None
    filename: str
    storage_path: str
    storage_backend: str | None
    storage_key: str
    storage_uri: str | None
    checksum: str
    etag: str | None
    workspace_relative_path: str | None
    workspace_category: str | None
    mime_type: str | None
    file_size: int
    upload_source: str | None = None

    @classmethod
    def from_record(cls, file_record: UploadedFile) -> "StagedUploadedFile":
        storage_key = str(file_record.storage_key or "").strip()
        checksum = str(file_record.checksum or "").strip()
        if file_record.storage_status != "available" or not storage_key or not checksum:
            raise ValueError(
                "Staged file must reference checksum-verified durable bytes"
            )
        return cls(
            file_id=str(file_record.file_id),
            user_id=int(file_record.user_id),
            task_id=(
                int(file_record.task_id) if file_record.task_id is not None else None
            ),
            filename=str(file_record.filename),
            storage_path=str(file_record.storage_path),
            storage_backend=(
                str(file_record.storage_backend)
                if file_record.storage_backend is not None
                else None
            ),
            storage_key=storage_key,
            storage_uri=(
                str(file_record.storage_uri)
                if file_record.storage_uri is not None
                else None
            ),
            checksum=checksum,
            etag=str(file_record.etag) if file_record.etag is not None else None,
            workspace_relative_path=(
                str(file_record.workspace_relative_path)
                if file_record.workspace_relative_path is not None
                else None
            ),
            workspace_category=(
                str(file_record.workspace_category)
                if file_record.workspace_category is not None
                else None
            ),
            mime_type=(
                str(file_record.mime_type)
                if file_record.mime_type is not None
                else None
            ),
            file_size=int(file_record.file_size or 0),
            upload_source=(
                str(file_record.upload_source)
                if file_record.upload_source is not None
                else None
            ),
        )

    def to_record(self) -> UploadedFile:
        """Materialize a transient ORM row inside the metadata owner."""

        return UploadedFile(
            file_id=self.file_id,
            user_id=self.user_id,
            task_id=self.task_id,
            filename=self.filename,
            storage_path=self.storage_path,
            storage_backend=self.storage_backend,
            storage_key=self.storage_key,
            storage_uri=self.storage_uri,
            checksum=self.checksum,
            etag=self.etag,
            workspace_relative_path=self.workspace_relative_path,
            workspace_category=self.workspace_category,
            storage_status="available",
            mime_type=self.mime_type,
            file_size=self.file_size,
            upload_source=self.upload_source,
        )


@dataclass(frozen=True)
class UploadedFileVersionSnapshot:
    """Detached identity of one committed uploaded-file metadata version."""

    row_id: int
    file_id: str
    user_id: int
    task_id: int | None
    filename: str
    storage_path: str
    storage_backend: str | None
    storage_key: str | None
    storage_uri: str | None
    checksum: str | None
    etag: str | None
    workspace_relative_path: str | None
    workspace_category: str | None
    storage_status: str
    mime_type: str | None
    file_size: int


class UploadedFileVersionConflict(RuntimeError):
    """The expected uploaded-file metadata version is no longer current."""


@dataclass(frozen=True)
class SupersededObjectCleanupClaim:
    """Detached ownership claim for one superseded immutable object."""

    user_id: int
    storage_backend: str | None
    storage_key: str


@dataclass(frozen=True)
class AppliedUploadedFileVersion:
    """Receipt for one metadata insert or compare-and-swap replacement."""

    snapshot: UploadedFileVersionSnapshot
    superseded_cleanup_claim: SupersededObjectCleanupClaim | None = None


def snapshot_uploaded_file_version(
    file_record: UploadedFile,
) -> UploadedFileVersionSnapshot:
    """Detach every metadata field that fences a later replacement."""

    return UploadedFileVersionSnapshot(
        row_id=int(file_record.id),
        file_id=str(file_record.file_id),
        user_id=int(file_record.user_id),
        task_id=(int(file_record.task_id) if file_record.task_id is not None else None),
        filename=str(file_record.filename),
        storage_path=str(file_record.storage_path),
        storage_backend=(
            str(file_record.storage_backend)
            if file_record.storage_backend is not None
            else None
        ),
        storage_key=(
            str(file_record.storage_key)
            if file_record.storage_key is not None
            else None
        ),
        storage_uri=(
            str(file_record.storage_uri)
            if file_record.storage_uri is not None
            else None
        ),
        checksum=(
            str(file_record.checksum) if file_record.checksum is not None else None
        ),
        etag=str(file_record.etag) if file_record.etag is not None else None,
        workspace_relative_path=(
            str(file_record.workspace_relative_path)
            if file_record.workspace_relative_path is not None
            else None
        ),
        workspace_category=(
            str(file_record.workspace_category)
            if file_record.workspace_category is not None
            else None
        ),
        storage_status=str(file_record.storage_status),
        mime_type=(
            str(file_record.mime_type) if file_record.mime_type is not None else None
        ),
        file_size=int(file_record.file_size or 0),
    )


def _snapshot_staged_uploaded_file_version(
    staged: StagedUploadedFile,
    *,
    row_id: int,
) -> UploadedFileVersionSnapshot:
    return UploadedFileVersionSnapshot(
        row_id=row_id,
        file_id=staged.file_id,
        user_id=staged.user_id,
        task_id=staged.task_id,
        filename=staged.filename,
        storage_path=staged.storage_path,
        storage_backend=staged.storage_backend,
        storage_key=staged.storage_key,
        storage_uri=staged.storage_uri,
        checksum=staged.checksum,
        etag=staged.etag,
        workspace_relative_path=staged.workspace_relative_path,
        workspace_category=staged.workspace_category,
        storage_status="available",
        mime_type=staged.mime_type,
        file_size=staged.file_size,
    )


def _nullable_version_match(column: Any, expected: Any) -> Any:
    """Build a portable exact match for a nullable snapshot field."""

    return column.is_(None) if expected is None else column == expected


def _uploaded_file_version_match_predicates(
    expected: UploadedFileVersionSnapshot,
) -> tuple[Any, ...]:
    """Return the complete SQL fence for one uploaded-file metadata version."""

    return (
        UploadedFile.id == expected.row_id,
        UploadedFile.file_id == expected.file_id,
        UploadedFile.user_id == expected.user_id,
        _nullable_version_match(UploadedFile.task_id, expected.task_id),
        UploadedFile.filename == expected.filename,
        UploadedFile.storage_path == expected.storage_path,
        _nullable_version_match(
            UploadedFile.storage_backend,
            expected.storage_backend,
        ),
        _nullable_version_match(
            UploadedFile.storage_key,
            expected.storage_key,
        ),
        _nullable_version_match(
            UploadedFile.storage_uri,
            expected.storage_uri,
        ),
        _nullable_version_match(UploadedFile.checksum, expected.checksum),
        _nullable_version_match(UploadedFile.etag, expected.etag),
        _nullable_version_match(
            UploadedFile.workspace_relative_path,
            expected.workspace_relative_path,
        ),
        _nullable_version_match(
            UploadedFile.workspace_category,
            expected.workspace_category,
        ),
        UploadedFile.storage_status == expected.storage_status,
        _nullable_version_match(UploadedFile.mime_type, expected.mime_type),
        UploadedFile.file_size == expected.file_size,
    )


def _is_owned_output_generation_key(*, user_id: int, storage_key: str) -> bool:
    """Return whether a key is an immutable task-output generation.

    Generation keys are created with a fresh UUID and are never reused. That
    non-reuse invariant is what makes a zero-reference check sufficient before
    deleting a superseded object after its metadata transaction has committed.
    Legacy deterministic output keys deliberately do not satisfy this shape.
    """

    components = storage_key.strip("/").split("/")
    if len(components) < 9 or components[:2] != ["users", str(user_id)]:
        return False
    for index in range(2, len(components) - 6):
        if (
            components[index] != "tasks"
            or not components[index + 1].isdigit()
            or components[index + 2] != "outputs"
            or not components[index + 3]
            or components[index + 4] != "_versions"
        ):
            continue
        generation = components[index + 5]
        normalized_generation = generation.replace("-", "")
        if len(normalized_generation) != 32:
            continue
        try:
            UUID(generation)
        except ValueError:
            continue
        return bool(components[index + 6 :])
    return False


def _is_owned_upload_key(
    *,
    user_id: int,
    storage_key: str,
    expected_file_id: str | None = None,
) -> bool:
    """Return whether an upload key belongs to an immutable object namespace.

    A UUID logical file id is itself a single-use namespace for an initial
    upload. Legacy logical ids become safe only under a UUID ``_versions``
    component. Scoped keys are admitted because the ``uploads`` component may
    appear after validated workspace segments.
    """

    components = storage_key.strip("/").split("/")
    if len(components) < 5 or components[:2] != ["users", str(user_id)]:
        return False
    for index in range(2, len(components) - 2):
        if components[index] != "uploads":
            continue
        logical_file_id = components[index + 1]
        if expected_file_id is not None and logical_file_id != expected_file_id:
            continue
        if not logical_file_id:
            continue
        remainder = components[index + 2 :]
        if not remainder:
            continue
        if remainder[0] == "_versions":
            if len(remainder) < 3:
                continue
            try:
                UUID(remainder[1])
            except ValueError:
                continue
            return bool(remainder[2:])
        try:
            UUID(logical_file_id)
        except ValueError:
            continue
        return True
    return False


def _is_owned_immutable_object_key(*, user_id: int, storage_key: str) -> bool:
    """Return whether post-commit cleanup may safely consider this key."""

    return _is_owned_output_generation_key(
        user_id=user_id,
        storage_key=storage_key,
    ) or _is_owned_upload_key(
        user_id=user_id,
        storage_key=storage_key,
    )


def _is_owned_unique_staging_key(
    *,
    user_id: int,
    file_id: str,
    storage_key: str,
) -> bool:
    """Return whether a staging key is provably single-use.

    Replacing or deleting an ambiguous key after a failed write acknowledgement
    could destroy an older committed object.  Staging therefore accepts only
    task-output generation keys or keys whose file-id namespace is itself a
    fresh UUID (normal uploads, legacy previews, and channel attachments).
    """

    if _is_owned_output_generation_key(
        user_id=user_id,
        storage_key=storage_key,
    ):
        return True
    if _is_owned_upload_key(
        user_id=user_id,
        storage_key=storage_key,
        expected_file_id=file_id,
    ):
        return True

    try:
        UUID(file_id)
    except ValueError:
        return False

    components = storage_key.strip("/").split("/")
    if len(components) < 5 or components[:2] != ["users", str(user_id)]:
        return False
    for index in range(2, len(components) - 2):
        if (
            components[index] == "tasks"
            and index + 4 < len(components)
            and components[index + 1].isdigit()
            and components[index + 2] == "outputs"
            and components[index + 3] == file_id
            and components[index + 4 :]
        ):
            return True
    return False


def _load_referenced_uploaded_file_storage_keys(
    owned_keys: Sequence[tuple[int, str]],
    *,
    session_factory: _SessionFactory | None = None,
) -> set[tuple[int, str]] | None:
    """Return referenced object identities, or ``None`` when unknown.

    Durable objects are selected through the user's storage namespace and key.
    ``storage_backend`` is metadata about the configured implementation, not
    part of the object identity.  A reference query failure must fail safe:
    callers retain the object and defer cleanup instead of guessing that it is
    unreferenced.
    """

    normalized_keys = tuple(
        dict.fromkeys(
            (int(user_id), str(storage_key))
            for user_id, storage_key in owned_keys
            if str(storage_key)
        )
    )
    if not normalized_keys:
        return set()

    user_ids = tuple(dict.fromkeys(user_id for user_id, _ in normalized_keys))
    storage_keys = tuple(dict.fromkeys(key for _, key in normalized_keys))
    try:
        SessionLocal = session_factory or get_session_local()
        with SessionLocal() as db:
            candidates = (
                db.query(
                    UploadedFile.user_id,
                    UploadedFile.storage_key,
                )
                .filter(
                    UploadedFile.user_id.in_(user_ids),
                    UploadedFile.storage_key.in_(storage_keys),
                )
                .all()
            )
    except Exception:
        logger.exception(
            "Failed to verify uploaded-file references before object cleanup"
        )
        return None

    requested = set(normalized_keys)
    return {
        (int(user_id), str(storage_key))
        for user_id, storage_key in candidates
        if storage_key is not None and (int(user_id), str(storage_key)) in requested
    }


StoragePresence = Literal["exists", "absent", "unknown"]
UploadedFileCompensationSettlement = Literal["deleted"]


def probe_uploaded_file_storage_presence(
    *,
    user_id: int,
    storage_key: str,
) -> StoragePresence:
    """Probe one owner-scoped key without opening a database Session."""

    try:
        return (
            "exists" if get_user_file_storage(user_id).exists(storage_key) else "absent"
        )
    except Exception:
        logger.exception(
            "Failed to determine durable file presence for key %s",
            storage_key,
        )
        return "unknown"


def delete_uploaded_file_compensation_object(
    *,
    user_id: int,
    storage_key: str,
) -> StoragePresence:
    """Continue compensation without a Session and report the durable outcome.

    An acknowledged delete is treated as absent. If the acknowledgement is
    lost, a follow-up presence probe distinguishes a completed delete from an
    object that must remain hidden and be retried. ``exists`` and ``unknown``
    deliberately never authorize restoring metadata to ``available`` because
    another compensation owner may still have a delete in flight.
    """

    try:
        get_user_file_storage(user_id).delete(storage_key)
    except Exception:
        logger.exception(
            "Failed to delete compensating durable file key %s",
            storage_key,
        )
        return probe_uploaded_file_storage_presence(
            user_id=user_id,
            storage_key=storage_key,
        )
    return "absent"


def settle_uploaded_file_compensation_no_commit(
    db: Session,
    *,
    row_id: int,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_key: str,
    expected_updated_at: datetime | None,
    presence: StoragePresence,
    storage_path: str | None = None,
) -> UploadedFileCompensationSettlement | None:
    """CAS-settle a probed compensation claim without storage I/O."""

    if presence != "absent":
        return None
    task_filter = (
        UploadedFile.task_id.is_(None)
        if task_id is None
        else UploadedFile.task_id == task_id
    )
    updated_at_filter = (
        UploadedFile.updated_at.is_(None)
        if expected_updated_at is None
        else UploadedFile.updated_at == expected_updated_at
    )
    query = db.query(UploadedFile).filter(
        UploadedFile.id == row_id,
        UploadedFile.user_id == user_id,
        UploadedFile.file_id == file_id,
        UploadedFile.storage_key == storage_key,
        UploadedFile.storage_status == "compensating",
        task_filter,
        updated_at_filter,
    )
    if storage_path is not None:
        query = query.filter(UploadedFile.storage_path == storage_path)
    changed = query.delete(synchronize_session=False)
    return "deleted" if changed == 1 else None


def _load_uploaded_file_compensation_token_no_commit(
    db: Session,
    *,
    row_id: int,
) -> datetime | None:
    token = (
        db.query(UploadedFile.updated_at)
        .filter(
            UploadedFile.id == row_id,
            UploadedFile.storage_status == "compensating",
        )
        .scalar()
    )
    return cast(datetime | None, token)


def take_over_uploaded_file_compensation_no_commit(
    db: Session,
    *,
    row_id: int,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_key: str,
    expected_updated_at: datetime | None,
    storage_path: str | None = None,
) -> datetime | None:
    """CAS-take ownership of one stale compensation claim.

    The returned timestamp is the exact persisted generation token that must
    fence the later metadata settlement. Advancing by at least one second keeps
    the generation distinct on databases whose datetime precision is limited
    to whole seconds.
    """

    if expected_updated_at is None:
        return None
    task_filter = (
        UploadedFile.task_id.is_(None)
        if task_id is None
        else UploadedFile.task_id == task_id
    )
    expected_utc = (
        expected_updated_at.replace(tzinfo=timezone.utc)
        if expected_updated_at.tzinfo is None
        else expected_updated_at.astimezone(timezone.utc)
    )
    takeover_at = max(
        datetime.now(timezone.utc),
        expected_utc + timedelta(seconds=1),
    )
    query = db.query(UploadedFile).filter(
        UploadedFile.id == row_id,
        UploadedFile.user_id == user_id,
        UploadedFile.file_id == file_id,
        UploadedFile.storage_key == storage_key,
        UploadedFile.storage_status == "compensating",
        task_filter,
        UploadedFile.updated_at == expected_updated_at,
    )
    if storage_path is not None:
        query = query.filter(UploadedFile.storage_path == storage_path)
    changed = query.update(
        {UploadedFile.updated_at: takeover_at},
        synchronize_session=False,
    )
    if changed != 1:
        return None
    return _load_uploaded_file_compensation_token_no_commit(
        db,
        row_id=row_id,
    )


def delete_uploaded_file_local_copy_if_owned(
    *,
    storage_path: str,
    user_id: int,
) -> bool:
    """Delete a local copy only when it resolves under its user's upload root.

    Returns ``False`` for external/shared paths, which are deliberately left
    untouched. Files already absent count as successfully cleaned.
    """

    local_path = Path(storage_path)
    try:
        resolved_path = local_path.resolve()
        user_root = scoped_user_root(get_uploads_dir(), user_id).resolve()
        resolved_path.relative_to(user_root)
    except (OSError, RuntimeError, ValueError):
        return False
    if resolved_path.exists() and resolved_path.is_file():
        resolved_path.unlink()
    return True


def _compensate_possible_immutable_staging_key(
    *,
    user_id: int,
    file_id: str,
    storage_key: str,
) -> bool:
    """Delete a possibly-written immutable key only when DB state is known."""

    if not _is_owned_unique_staging_key(
        user_id=user_id,
        file_id=file_id,
        storage_key=storage_key,
    ):
        logger.error(
            "Refusing ambiguous staging compensation for key %s",
            storage_key,
        )
        return False
    referenced = _load_referenced_uploaded_file_storage_keys(((user_id, storage_key),))
    if referenced is None or (user_id, storage_key) in referenced:
        return False
    return not delete_durable_storage_keys(((user_id, storage_key),))


def cleanup_superseded_uploaded_file_objects(
    claims: Sequence[SupersededObjectCleanupClaim],
    *,
    session_factory: _SessionFactory | None = None,
) -> tuple[SupersededObjectCleanupClaim, ...]:
    """Delete unreferenced immutable generations after metadata commit.

    Only keys with a single-use UUID namespace or generation are eligible.
    Deterministic legacy keys can be reused and are therefore never deleted
    here. The metadata reference check owns a short Session; all
    object-storage deletes run after it is closed.

    Returns:
        Claims whose object could not be deleted safely. Referenced or legacy
        keys are intentionally skipped and are not failures.
    """

    normalized_claims = tuple(
        dict.fromkeys(
            claim
            for claim in claims
            if claim.storage_key
            and _is_owned_immutable_object_key(
                user_id=claim.user_id,
                storage_key=claim.storage_key,
            )
        )
    )
    if not normalized_claims:
        return ()

    referenced_objects = _load_referenced_uploaded_file_storage_keys(
        tuple((claim.user_id, claim.storage_key) for claim in normalized_claims),
        session_factory=session_factory,
    )
    if referenced_objects is None:
        return normalized_claims

    failed: list[SupersededObjectCleanupClaim] = []
    for claim in normalized_claims:
        if (claim.user_id, claim.storage_key) in referenced_objects:
            continue
        try:
            storage = get_user_file_storage(claim.user_id)
            if (
                claim.storage_backend is not None
                and storage.backend != claim.storage_backend
            ):
                logger.warning(
                    "Skipping superseded object cleanup because backend changed: %s",
                    claim.storage_key,
                )
                failed.append(claim)
                continue
            storage.delete(claim.storage_key)
        except Exception:
            logger.exception(
                "Failed to delete superseded durable file key %s",
                claim.storage_key,
            )
            failed.append(claim)
    return tuple(failed)


def delete_pptx_pdf_cache(file_id: str) -> None:
    """Remove the server-side LibreOffice PDF preview cache for a .pptx upload.

    Must be called from every path that deletes an ``UploadedFile`` row so
    the derived cache entry does not outlive the source artifact.  Silently
    no-ops when the file does not exist (first delete, non-PPTX uploads, etc.).
    """
    if not file_id:
        return
    if Path(file_id).name != file_id or file_id in {".", ".."}:
        logger.warning("Skipping invalid registered preview cache ID: %r", file_id)
        return
    cache_path = get_storage_root() / "pptx_pdf_cache" / f"{file_id}.preview.pdf"
    try:
        cache_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to remove PDF preview cache for %s", file_id)


def delete_svg_png_cache(file_id: str) -> None:
    """Remove all rasterized SVG preview PNGs cached for an upload.

    Unlike the PPTX preview cache (one fixed file per ``file_id``), an SVG
    ``file_id`` can have multiple derived previews — one per relative-path
    asset resolved under it (see ``_svg_png_cache_path`` in
    ``web/api/files.py``), each keyed as ``<file_id>.<path-hash>.preview.png``.
    Must be called from every path that deletes an ``UploadedFile`` row so
    none of those derived previews outlive the source artifact. Silently
    no-ops when there is nothing cached.

    ``file_id`` is treated as a literal filename prefix, never as a glob
    expression. Registered identifiers normally come from the database, but
    keeping pattern syntax inert also makes this shared cleanup boundary safe
    if a malformed historical row is encountered.
    """

    cache_dir = get_storage_root() / "svg_png_cache"
    if not file_id:
        return
    prefix = f"{file_id}."
    try:
        matches = [
            path
            for path in cache_dir.iterdir()
            if path.name.startswith(prefix) and path.name.endswith(".preview.png")
        ]
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("Failed to list SVG preview cache for %s", file_id)
        return
    for cache_path in matches:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove SVG preview cache %s", cache_path)


def delete_registered_preview_caches(file_id: str) -> None:
    """Invalidate every derived preview owned by one registered upload."""

    delete_pptx_pdf_cache(file_id)
    delete_svg_png_cache(file_id)


def delete_legacy_preview_caches(source_path: Path) -> None:
    """Invalidate path-keyed previews for a legacy unregistered source.

    ``source_path`` must be the canonical path captured while the source still
    exists. Keeping this entry point separate prevents a route path from ever
    being interpreted as a registered cache identifier.
    """

    path_key = hashlib.sha256(str(source_path).encode()).hexdigest()[:24]
    cache_paths = (
        get_storage_root() / "pptx_pdf_cache" / f"{path_key}.preview.pdf",
        get_storage_root() / "svg_png_cache" / f"{path_key}.preview.png",
    )
    for cache_path in cache_paths:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove legacy preview cache %s", cache_path)


def create_unbound_uploaded_file_from_local_path(
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
    upload_source: str | None = None,
    execution_scope: ExecutionScopeInput = EXECUTION_SCOPE_NOT_PROVIDED,
) -> UploadedFile:
    file_record = build_uploaded_file_record(
        local_path=local_path,
        user_id=user_id,
        filename=filename,
        file_id=file_id,
        task_id=task_id,
        mime_type=mime_type,
        workspace_relative_path=workspace_relative_path,
        workspace_category=workspace_category,
        upload_source=upload_source,
    )
    ManagedFileRef(file_record, execution_scope=execution_scope).sync_to_durable(
        storage_key=storage_key,
        mime_type=str(file_record.mime_type),
    )
    return file_record


def stage_uploaded_file_from_local_path(
    *,
    local_path: Path,
    user_id: int,
    filename: str | None = None,
    file_id: str,
    task_id: int | None = None,
    mime_type: str | None = None,
    storage_key: str,
    workspace_relative_path: str | None = None,
    workspace_category: str | None = None,
    upload_source: str | None = None,
    execution_scope: ExecutionScopeInput = EXECUTION_SCOPE_NOT_PROVIDED,
) -> StagedUploadedFile:
    """Write one immutable object without retaining a Session.

    A storage backend can persist bytes and then fail while attaching metadata
    or acknowledging the write.  Because ``storage_key`` is required to be
    single-use, that ambiguous result can be compensated after a short DB
    reference check without risking an older object.
    """

    if not _is_owned_unique_staging_key(
        user_id=user_id,
        file_id=file_id,
        storage_key=storage_key,
    ):
        raise ValueError("Staged durable storage key must be unique and immutable")
    try:
        return StagedUploadedFile.from_record(
            create_unbound_uploaded_file_from_local_path(
                local_path=local_path,
                user_id=user_id,
                filename=filename,
                file_id=file_id,
                task_id=task_id,
                mime_type=mime_type,
                storage_key=storage_key,
                workspace_relative_path=workspace_relative_path,
                workspace_category=workspace_category,
                upload_source=upload_source,
                execution_scope=execution_scope,
            )
        )
    except Exception:
        _compensate_possible_immutable_staging_key(
            user_id=user_id,
            file_id=file_id,
            storage_key=storage_key,
        )
        raise


def compensate_staged_uploaded_files(
    staged_files: Sequence[StagedUploadedFile],
    *,
    session_factory: _SessionFactory | None = None,
) -> tuple[str, ...]:
    """Delete staged objects only after confirming metadata does not reference them.

    A database commit can succeed even when the client receives an exception.
    Consequently, an in-process ``metadata_committed`` flag cannot authorize a
    destructive compensation.  The short reference query is the source of
    truth; an unavailable database causes the object to be retained because its
    reference state is unknown. The row-based compensation recovery loop cannot
    discover such an object, so this function does not claim eventual cleanup.
    """

    normalized_staged = tuple(
        dict.fromkeys(
            staged for staged in staged_files if str(staged.storage_key).strip()
        )
    )
    referenced_keys = _load_referenced_uploaded_file_storage_keys(
        tuple((staged.user_id, staged.storage_key) for staged in normalized_staged),
        session_factory=session_factory,
    )
    if referenced_keys is None:
        return tuple(staged.file_id for staged in normalized_staged)

    unreferenced_staged = tuple(
        staged
        for staged in normalized_staged
        if (staged.user_id, staged.storage_key) not in referenced_keys
    )
    failed_keys = delete_durable_storage_keys(unreferenced_staged)
    failed_key_set = set(failed_keys)
    return tuple(
        staged.file_id
        for staged in unreferenced_staged
        if (staged.user_id, staged.storage_key) in failed_key_set
    )


def delete_durable_storage_keys(
    owned_keys: Iterable[tuple[int, str] | StagedUploadedFile],
) -> tuple[tuple[int, str], ...]:
    """Delete detached, owner-scoped object keys without opening a Session."""

    # Accept any finite iterable while keeping the public type narrow enough
    # for callers and static analysis.  Converting once also deduplicates keys
    # so repeated output aliases cannot delete the same object twice.
    raw_keys = [
        (
            (item.user_id, item.storage_key)
            if isinstance(item, StagedUploadedFile)
            else item
        )
        for item in owned_keys
    ]
    normalized_keys = tuple(
        dict.fromkeys((int(user_id), str(key)) for user_id, key in raw_keys if str(key))
    )
    failed: list[tuple[int, str]] = []
    for user_id, storage_key in normalized_keys:
        try:
            get_user_file_storage(user_id).delete(storage_key)
        except Exception:
            failed.append((user_id, storage_key))
            logger.exception(
                "Failed to delete durable file key %s",
                storage_key,
            )
    return tuple(failed)


def register_local_uploads_sync(
    registrations: Sequence[LocalUploadRegistration],
) -> tuple[UploadedFileRegistrationSnapshot, ...]:
    """Stage durable bytes first, then commit metadata in one short Session.

    Object storage and checksum latency happen before a Session exists. The
    only connection-owning phase revalidates any task binding, inserts the
    already-durable records, and commits the whole batch atomically.
    """

    if not registrations:
        return ()
    staged_files: list[StagedUploadedFile] = []
    metadata_committed = False
    try:
        for registration in registrations:
            staged = stage_uploaded_file_from_local_path(
                local_path=registration.local_path,
                user_id=registration.user_id,
                file_id=registration.file_id,
                task_id=registration.task_id,
                filename=registration.filename,
                mime_type=registration.mime_type,
                storage_key=registration.compensation_claim.expected_storage_key,
                upload_source=registration.upload_source,
                execution_scope=registration.execution_scope,
            )
            staged_files.append(staged)

        SessionLocal = get_session_local()
        with SessionLocal() as db:
            for registration in registrations:
                if registration.task_id is None:
                    continue
                task_exists = (
                    db.query(Task.id)
                    .filter(
                        Task.id == registration.task_id,
                        Task.user_id == registration.user_id,
                    )
                    .first()
                    is not None
                )
                if not task_exists:
                    raise ValueError(
                        f"Task {registration.task_id} is not owned by "
                        f"user {registration.user_id}"
                    )
            for staged in staged_files:
                UploadedFileStore(db).add_already_durable(staged.to_record())
            snapshots = tuple(
                UploadedFileRegistrationSnapshot(
                    file_id=staged.file_id,
                    filename=staged.filename,
                    file_size=staged.file_size,
                    mime_type=(
                        str(staged.mime_type) if staged.mime_type is not None else None
                    ),
                )
                for staged in staged_files
            )
            db.commit()
            metadata_committed = True
            return snapshots
    finally:
        if not metadata_committed:
            failed_file_ids = compensate_staged_uploaded_files(staged_files)
            if failed_file_ids:
                logger.warning(
                    "Retained %s staged upload object(s) because reference or "
                    "deletion state was unknown",
                    len(failed_file_ids),
                )


def compensate_registered_uploads_sync(
    claims: Sequence[RegisteredUploadCompensationClaim],
) -> None:
    """Rollback only registrations still owned by the cancelled request.

    The metadata row is first claimed with an exact compare-and-swap. Consumers
    can bind only ``available`` rows, so either a consumer changes ``task_id``
    first and compensation becomes a no-op, or compensation changes the status
    first and the consumer observes the file as unavailable. Object storage I/O
    happens only after that short transaction has released its connection.
    """

    normalized_claims = tuple(
        dict.fromkeys(
            claim
            for claim in claims
            if str(claim.file_id).strip() and str(claim.expected_storage_key).strip()
        )
    )
    if not normalized_claims:
        return

    SessionLocal = get_session_local()
    claimed: list[ClaimedRegisteredUploadCompensation] = []
    with SessionLocal() as db:
        for registration_claim in normalized_claims:
            claimed_at = datetime.now(timezone.utc)
            task_filter = (
                UploadedFile.task_id.is_(None)
                if registration_claim.expected_task_id is None
                else UploadedFile.task_id == registration_claim.expected_task_id
            )
            row = (
                db.query(UploadedFile.id)
                .filter(
                    UploadedFile.user_id == registration_claim.user_id,
                    UploadedFile.file_id == registration_claim.file_id,
                    UploadedFile.storage_key == registration_claim.expected_storage_key,
                    UploadedFile.storage_status == "available",
                    task_filter,
                )
                .first()
            )
            if row is None:
                continue
            claimed_count = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.id == int(row[0]),
                    UploadedFile.user_id == registration_claim.user_id,
                    UploadedFile.file_id == registration_claim.file_id,
                    UploadedFile.storage_key == registration_claim.expected_storage_key,
                    UploadedFile.storage_status == "available",
                    task_filter,
                )
                .update(
                    {
                        UploadedFile.storage_status: "compensating",
                        UploadedFile.updated_at: claimed_at,
                    },
                    synchronize_session=False,
                )
            )
            if claimed_count == 1:
                persisted_claimed_at = _load_uploaded_file_compensation_token_no_commit(
                    db,
                    row_id=int(row[0]),
                )
                if persisted_claimed_at is None:
                    raise RuntimeError(
                        "Compensation claim did not persist its generation token"
                    )
                claimed.append(
                    ClaimedRegisteredUploadCompensation(
                        row_id=int(row[0]),
                        user_id=registration_claim.user_id,
                        file_id=registration_claim.file_id,
                        expected_task_id=registration_claim.expected_task_id,
                        expected_storage_key=(registration_claim.expected_storage_key),
                        claimed_at=persisted_claimed_at,
                    )
                )
        db.commit()

    cleaned: list[ClaimedRegisteredUploadCompensation] = []
    unresolved: list[ClaimedRegisteredUploadCompensation] = []
    for claimed_version in claimed:
        presence = delete_uploaded_file_compensation_object(
            user_id=claimed_version.user_id,
            storage_key=claimed_version.expected_storage_key,
        )
        if presence == "absent":
            cleaned.append(claimed_version)
        else:
            unresolved.append(claimed_version)

    if cleaned:
        cleaned_file_ids: list[str] = []
        with SessionLocal() as db:
            for claimed_version in cleaned:
                settlement = settle_uploaded_file_compensation_no_commit(
                    db,
                    row_id=claimed_version.row_id,
                    user_id=claimed_version.user_id,
                    file_id=claimed_version.file_id,
                    task_id=claimed_version.expected_task_id,
                    storage_key=claimed_version.expected_storage_key,
                    expected_updated_at=claimed_version.claimed_at,
                    presence="absent",
                )
                if settlement == "deleted":
                    cleaned_file_ids.append(claimed_version.file_id)
            db.commit()
        for file_id in cleaned_file_ids:
            delete_registered_preview_caches(file_id)

    if unresolved:
        failed_ids = ", ".join(claim.file_id for claim in unresolved)
        # No single key: this is a batch compensation failure. The file ids in
        # the message are #1642's scope, alongside widening the purity scan to
        # this module.
        raise DurableStorageOperationError(
            f"Failed to compensate durable uploads: {failed_ids}",
            storage_key=None,
        )


def build_uploaded_file_record(
    *,
    local_path: Path,
    user_id: int,
    filename: str | None = None,
    file_id: str | None = None,
    task_id: int | None = None,
    mime_type: str | None = None,
    workspace_relative_path: str | None = None,
    workspace_category: str | None = None,
    upload_source: str | None = None,
) -> UploadedFile:
    resolved_filename = filename or local_path.name
    resolved_mime_type = mime_type or guess_media_type(resolved_filename)
    return UploadedFile(
        file_id=file_id or str(uuid4()),
        user_id=user_id,
        task_id=task_id,
        filename=Path(resolved_filename).name,
        storage_path=str(local_path),
        mime_type=resolved_mime_type,
        file_size=local_path.stat().st_size,
        storage_status="pending",
        workspace_relative_path=workspace_relative_path,
        workspace_category=workspace_category,
        upload_source=upload_source,
    )


class UploadedFileStore:
    """Coordinates UploadedFile rows with durable object storage."""

    def __init__(self, db: Session):
        self.db = db

    def create_from_local_path(
        self,
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
        upload_source: str | None = None,
    ) -> UploadedFile:
        file_record = build_uploaded_file_record(
            local_path=local_path,
            user_id=user_id,
            filename=filename,
            file_id=file_id,
            task_id=task_id,
            mime_type=mime_type,
            workspace_relative_path=workspace_relative_path,
            workspace_category=workspace_category,
            upload_source=upload_source,
        )
        self.db.add(file_record)
        self.db.flush()
        try:
            self._sync_new_pending(
                file_record,
                storage_key=storage_key,
                mime_type=str(file_record.mime_type),
            )
        except Exception:
            self.db.delete(file_record)
            self.db.flush()
            raise
        return file_record

    def add_already_durable(self, file_record: UploadedFile) -> UploadedFile:
        """Persist metadata for an object uploaded before opening this Session.

        Durable writes can be slow and must not run while a database transaction
        owns a pooled connection.  Callers stage the object through
        :func:`create_unbound_uploaded_file_from_local_path`, then use this
        method for the short metadata transaction.  No storage I/O is performed
        here.
        """

        if getattr(file_record, "storage_status", None) != "available":
            raise ValueError("Pre-uploaded file must have storage_status='available'")
        if not getattr(file_record, "storage_key", None):
            raise ValueError("Pre-uploaded file must have a durable storage key")
        if not getattr(file_record, "checksum", None):
            raise ValueError("Pre-uploaded file must have a durable checksum")
        self.db.add(file_record)
        self.db.flush()
        return file_record

    def upsert_already_durable(
        self,
        staged: StagedUploadedFile,
        *,
        expected: UploadedFileVersionSnapshot | None,
        allow_task_rebind: bool = False,
    ) -> AppliedUploadedFileVersion:
        """Insert or atomically replace metadata for pre-staged durable bytes.

        ``expected=None`` is an explicit insert contract. A snapshot is an
        explicit replacement contract: one SQL ``UPDATE`` compares every
        durable identity field before applying the staged version. The caller
        owns commit/rollback; this method performs no object-storage I/O.

        Task ownership is immutable by default. A caller that has separately
        verified the target task belongs to the same user may explicitly opt
        into rebinding; the original task id remains part of the SQL CAS.
        """

        if expected is None:
            file_record = staged.to_record()
            try:
                self.add_already_durable(file_record)
            except IntegrityError as exc:
                raise UploadedFileVersionConflict(
                    f"Uploaded file {staged.file_id} already exists"
                ) from exc
            return AppliedUploadedFileVersion(
                snapshot=snapshot_uploaded_file_version(file_record)
            )

        if expected.file_id != staged.file_id:
            raise ValueError("Cannot replace an uploaded file with another file_id")
        if expected.user_id != staged.user_id:
            raise ValueError("Cannot replace an uploaded file owned by another user")
        if expected.task_id != staged.task_id and not allow_task_rebind:
            raise ValueError("Cannot replace an uploaded file bound to another task")

        replacement_values: dict[Any, Any] = {
            UploadedFile.file_id: staged.file_id,
            UploadedFile.user_id: staged.user_id,
            UploadedFile.task_id: staged.task_id,
            UploadedFile.filename: staged.filename,
            UploadedFile.storage_path: staged.storage_path,
            UploadedFile.storage_backend: staged.storage_backend,
            UploadedFile.storage_key: staged.storage_key,
            UploadedFile.storage_uri: staged.storage_uri,
            UploadedFile.checksum: staged.checksum,
            UploadedFile.etag: staged.etag,
            UploadedFile.workspace_relative_path: staged.workspace_relative_path,
            UploadedFile.workspace_category: staged.workspace_category,
            UploadedFile.storage_status: "available",
            UploadedFile.mime_type: staged.mime_type,
            UploadedFile.file_size: staged.file_size,
        }
        if staged.task_id is not None:
            replacement_values.update(uploaded_file_bind_values(staged.task_id))

        statement = (
            update(UploadedFile)
            .where(
                *_uploaded_file_version_match_predicates(expected),
                UploadedFile.storage_status != "compensating",
            )
            .values(replacement_values)
        )
        result = self.db.execute(statement.execution_options(synchronize_session=False))
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            raise UploadedFileVersionConflict(
                f"Uploaded file {staged.file_id} changed after durable staging"
            )

        cleanup_claim = (
            SupersededObjectCleanupClaim(
                user_id=expected.user_id,
                storage_backend=expected.storage_backend,
                storage_key=expected.storage_key,
            )
            if expected.storage_key and expected.storage_key != staged.storage_key
            else None
        )
        return AppliedUploadedFileVersion(
            snapshot=_snapshot_staged_uploaded_file_version(
                staged,
                row_id=expected.row_id,
            ),
            superseded_cleanup_claim=cleanup_claim,
        )

    def restore_metadata_version_no_commit(
        self,
        *,
        expected: UploadedFileVersionSnapshot,
        replacement: UploadedFileVersionSnapshot,
    ) -> AppliedUploadedFileVersion:
        """CAS-restore metadata when the previous version had no source bytes.

        This is intentionally narrower than durable replacement: it is for
        rollback of a previously missing object, where no local bytes exist to
        stage. The caller owns the short transaction and must perform any
        returned superseded-object cleanup only after commit and Session close.
        """

        if (
            expected.row_id != replacement.row_id
            or expected.file_id != replacement.file_id
            or expected.user_id != replacement.user_id
        ):
            raise ValueError("Metadata rollback must preserve uploaded-file identity")
        if replacement.storage_status not in {"available", "legacy"}:
            raise ValueError("Metadata rollback target must be a stable file state")
        replacement_values: dict[Any, Any] = {
            UploadedFile.file_id: replacement.file_id,
            UploadedFile.user_id: replacement.user_id,
            UploadedFile.task_id: replacement.task_id,
            UploadedFile.filename: replacement.filename,
            UploadedFile.storage_path: replacement.storage_path,
            UploadedFile.storage_backend: replacement.storage_backend,
            UploadedFile.storage_key: replacement.storage_key,
            UploadedFile.storage_uri: replacement.storage_uri,
            UploadedFile.checksum: replacement.checksum,
            UploadedFile.etag: replacement.etag,
            UploadedFile.workspace_relative_path: replacement.workspace_relative_path,
            UploadedFile.workspace_category: replacement.workspace_category,
            UploadedFile.storage_status: replacement.storage_status,
            UploadedFile.mime_type: replacement.mime_type,
            UploadedFile.file_size: replacement.file_size,
        }
        if replacement.task_id is not None:
            replacement_values.update(uploaded_file_bind_values(replacement.task_id))

        statement = (
            update(UploadedFile)
            .where(
                *_uploaded_file_version_match_predicates(expected),
                UploadedFile.storage_status != "compensating",
            )
            .values(replacement_values)
        )
        result = self.db.execute(statement.execution_options(synchronize_session=False))
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            raise UploadedFileVersionConflict(
                f"Uploaded file {expected.file_id} changed before metadata rollback"
            )
        cleanup_claim = (
            SupersededObjectCleanupClaim(
                user_id=expected.user_id,
                storage_backend=expected.storage_backend,
                storage_key=expected.storage_key,
            )
            if expected.storage_key and expected.storage_key != replacement.storage_key
            else None
        )
        return AppliedUploadedFileVersion(
            snapshot=replacement,
            superseded_cleanup_claim=cleanup_claim,
        )

    def _sync_new_pending(
        self,
        file_record: UploadedFile,
        *,
        storage_key: str | None = None,
        mime_type: str | None = None,
    ) -> UploadedFile:
        storage_status = str(getattr(file_record, "storage_status", ""))
        if storage_status != "pending":
            raise UploadedFileVersionConflict(
                f"Uploaded file {file_record.file_id} is {storage_status or 'unknown'}"
            )
        ManagedFileRef(file_record).sync_to_durable(
            storage_key=storage_key,
            mime_type=mime_type,
        )
        self.db.flush()
        return file_record

    def upsert_by_storage_path(
        self,
        *,
        user_id: int,
        filename: str,
        storage_path: Path,
        mime_type: str | None,
        file_size: int,
        file_id: str | None = None,
        storage_key: str | None = None,
        task_id: int | None = None,
        workspace_relative_path: str | None = None,
        workspace_category: str | None = None,
        expected_version: UploadedFileVersionSnapshot | None = None,
        replacement_metadata: UploadedFileVersionSnapshot | None = None,
    ) -> UploadedFile:
        """Insert or refresh a path through immutable staging and exact CAS.

        The caller Session is used only to preserve the historical return
        contract. Every metadata transaction is owned by this method and all
        checksum/object-storage work runs while no Session holds a connection.
        Existing rows are replaceable only from stable ``available`` or
        ``legacy`` states; compensation and pending-registration owners fail
        closed.
        """

        del file_size  # Durable metadata always uses the source file's real size.
        if not release_db_connection_if_clean(self.db):
            raise RuntimeError(
                "Cannot upsert an uploaded file while the caller database "
                "session has pending writes"
            )
        SessionLocal = _session_factory_from_reference(self.db)
        storage_path_str = str(storage_path)
        expected: UploadedFileVersionSnapshot | None = None
        resolved_file_id = file_id or str(uuid4())
        resolved_task_id = task_id

        with SessionLocal() as planning_db:
            existing = (
                planning_db.query(UploadedFile)
                .filter(UploadedFile.storage_path == storage_path_str)
                .first()
            )
            if existing is not None:
                if int(existing.user_id) != int(user_id):
                    raise PermissionError("Uploaded file path belongs to another user")
                existing_status = str(existing.storage_status)
                if existing_status not in {"available", "legacy"}:
                    raise UploadedFileVersionConflict(
                        f"Uploaded file {existing.file_id} is {existing_status}"
                    )
                if file_id is not None and str(existing.file_id) != str(file_id):
                    raise UploadedFileVersionConflict(
                        "Uploaded file storage_path belongs to another file_id"
                    )
                current_version = snapshot_uploaded_file_version(existing)
                if expected_version is not None and current_version != expected_version:
                    raise UploadedFileVersionConflict(
                        f"Uploaded file {existing.file_id} no longer matches "
                        "the expected version"
                    )
                expected = expected_version or current_version
                resolved_file_id = expected.file_id
                if replacement_metadata is not None:
                    if (
                        replacement_metadata.file_id != expected.file_id
                        or replacement_metadata.user_id != expected.user_id
                        or replacement_metadata.storage_path != storage_path_str
                    ):
                        raise ValueError(
                            "Replacement metadata must describe the same "
                            "uploaded-file identity and storage_path"
                        )
                    resolved_task_id = replacement_metadata.task_id
                elif task_id is None:
                    resolved_task_id = expected.task_id
            elif expected_version is not None or replacement_metadata is not None:
                raise UploadedFileVersionConflict(
                    "Uploaded file expected version no longer exists"
                )

            if resolved_task_id is not None:
                task_exists = (
                    planning_db.query(Task.id)
                    .filter(
                        Task.id == resolved_task_id,
                        Task.user_id == user_id,
                    )
                    .first()
                    is not None
                )
                if not task_exists:
                    raise ValueError(
                        f"Task {resolved_task_id} is not owned by user {user_id}"
                    )

        actual_file_size = storage_path.stat().st_size
        source_checksum = self._sha256(storage_path)
        if replacement_metadata is not None:
            resolved_filename = replacement_metadata.filename
            resolved_mime_type = replacement_metadata.mime_type
            resolved_workspace_relative_path = (
                replacement_metadata.workspace_relative_path
            )
            resolved_workspace_category = replacement_metadata.workspace_category
        else:
            resolved_filename = Path(filename).name
            resolved_mime_type = (
                mime_type
                if mime_type is not None
                else (
                    expected.mime_type
                    if expected is not None and expected.mime_type is not None
                    else guess_media_type(resolved_filename)
                )
            )
            resolved_workspace_relative_path = (
                workspace_relative_path
                if workspace_relative_path is not None
                else (
                    expected.workspace_relative_path if expected is not None else None
                )
            )
            resolved_workspace_category = (
                workspace_category
                if workspace_category is not None
                else (expected.workspace_category if expected is not None else None)
            )

        durable_key_changed = (
            expected is not None
            and storage_key is not None
            and storage_key != expected.storage_key
        )
        current_durable_bytes = (
            expected is not None
            and expected.storage_status == "available"
            and bool(expected.storage_key)
            and bool(expected.checksum)
            and expected.file_size == actual_file_size
            and expected.checksum == source_checksum
            and (mime_type is None or expected.mime_type == mime_type)
            and not durable_key_changed
        )

        wrote_staged_object = False
        if current_durable_bytes:
            assert expected is not None
            assert expected.storage_key is not None
            assert expected.checksum is not None
            staged = StagedUploadedFile(
                file_id=resolved_file_id,
                user_id=user_id,
                task_id=resolved_task_id,
                filename=resolved_filename,
                storage_path=storage_path_str,
                storage_backend=expected.storage_backend,
                storage_key=expected.storage_key,
                storage_uri=expected.storage_uri,
                checksum=expected.checksum,
                etag=expected.etag,
                workspace_relative_path=resolved_workspace_relative_path,
                workspace_category=resolved_workspace_category,
                mime_type=resolved_mime_type,
                file_size=actual_file_size,
            )
        else:
            resolved_storage_key = storage_key
            if resolved_storage_key is None or (
                expected is not None and resolved_storage_key == expected.storage_key
            ):
                scope_segments = self._upload_scope_segments(
                    user_id=user_id,
                    file_id=resolved_file_id,
                    storage_key=(
                        expected.storage_key if expected is not None else None
                    ),
                )
                if expected is None:
                    try:
                        UUID(resolved_file_id)
                    except ValueError:
                        resolved_storage_key = build_upload_generation_storage_key(
                            user_id,
                            resolved_file_id,
                            resolved_filename,
                            generation=uuid4().hex,
                            scope_segments=scope_segments,
                        )
                    else:
                        resolved_storage_key = build_upload_storage_key(
                            user_id,
                            resolved_file_id,
                            resolved_filename,
                            scope_segments=scope_segments,
                        )
                else:
                    resolved_storage_key = build_upload_generation_storage_key(
                        user_id,
                        resolved_file_id,
                        resolved_filename,
                        generation=uuid4().hex,
                        scope_segments=scope_segments,
                    )

            staged = stage_uploaded_file_from_local_path(
                local_path=storage_path,
                user_id=user_id,
                file_id=resolved_file_id,
                task_id=resolved_task_id,
                filename=resolved_filename,
                mime_type=resolved_mime_type,
                storage_key=resolved_storage_key,
                workspace_relative_path=resolved_workspace_relative_path,
                workspace_category=resolved_workspace_category,
                # The already-validated key is the authority here. Binding the
                # storage handle at the owner root admits both owner-root and
                # deeper execution-scope keys without ambient-scope drift.
                execution_scope=None,
            )
            wrote_staged_object = True

        applied: AppliedUploadedFileVersion
        try:
            with SessionLocal() as metadata_db:
                if resolved_task_id is not None:
                    task_exists = (
                        metadata_db.query(Task.id)
                        .filter(
                            Task.id == resolved_task_id,
                            Task.user_id == user_id,
                        )
                        .first()
                        is not None
                    )
                    if not task_exists:
                        raise ValueError(
                            f"Task {resolved_task_id} is not owned by user {user_id}"
                        )
                applied = UploadedFileStore(metadata_db).upsert_already_durable(
                    staged,
                    expected=expected,
                    allow_task_rebind=(
                        expected is not None and expected.task_id != resolved_task_id
                    ),
                )
                metadata_db.commit()
        except Exception:
            if wrote_staged_object:
                failed_file_ids = compensate_staged_uploaded_files(
                    (staged,),
                    session_factory=SessionLocal,
                )
                if failed_file_ids:
                    logger.warning(
                        "Retained staged upload object after metadata conflict "
                        "because cleanup state was unknown: %s",
                        ", ".join(failed_file_ids),
                    )
            raise

        if expected is not None:
            # The committed generation is now authoritative. Any preview
            # derived from the previous bytes must be discarded before a
            # subsequent request can reuse it.
            delete_registered_preview_caches(applied.snapshot.file_id)
            delete_legacy_preview_caches(Path(expected.storage_path).resolve())

        if applied.superseded_cleanup_claim is not None:
            failed_claims = cleanup_superseded_uploaded_file_objects(
                (applied.superseded_cleanup_claim,),
                session_factory=SessionLocal,
            )
            if failed_claims:
                logger.warning(
                    "Failed to clean %s superseded uploaded-file object(s)",
                    len(failed_claims),
                )

        self.db.expire_all()
        return (
            self.db.query(UploadedFile)
            .filter(UploadedFile.id == applied.snapshot.row_id)
            .one()
        )

    def delete(
        self,
        file_record: UploadedFile,
        *,
        delete_local: bool = True,
        local_root: Optional[Path] = None,
    ) -> None:
        ManagedFileRef(file_record).delete_durable()
        if delete_local:
            self._delete_local(file_record, local_root=local_root)
        # Remove any server-side PDF preview cache so derived content doesn't
        # outlive the source upload.  Called here (not only in the HTTP route)
        # so reconcile / orphan-cleanup paths that go through this service also
        # clean up the cache.
        file_id = str(getattr(file_record, "file_id", "") or "")
        delete_registered_preview_caches(file_id)
        self.db.delete(file_record)
        self.db.flush()

    @staticmethod
    def ensure_local(file_record: UploadedFile) -> Path:
        return ManagedFileRef(file_record).ensure_local()

    @staticmethod
    def _upload_scope_segments(
        *,
        user_id: int,
        file_id: str,
        storage_key: str | None,
    ) -> tuple[str, ...]:
        """Recover validated scope segments from an existing upload key."""

        if not storage_key:
            return ()
        components = storage_key.strip("/").split("/")
        if components[:2] != ["users", str(user_id)]:
            return ()
        for index in range(2, len(components) - 2):
            if components[index] == "uploads" and components[index + 1] == file_id:
                return tuple(components[2:index])
        return ()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _delete_local(
        file_record: UploadedFile, *, local_root: Optional[Path] = None
    ) -> None:
        local_path = Path(str(file_record.storage_path))
        if local_root is not None:
            resolved_path = local_path.resolve()
            if not resolved_path.is_relative_to(local_root.resolve()):
                return
            local_path = resolved_path
        if local_path.exists() and local_path.is_file():
            local_path.unlink()
