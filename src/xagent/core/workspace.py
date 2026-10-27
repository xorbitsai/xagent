"""
Agent-aware workspace management for xagent

This module provides workspace management that supports multiple concurrent agents,
ensuring that each agent has its own isolated workspace context.
"""

import contextvars
import errno
import logging
import mimetypes
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from uuid import uuid4

from ..config import (
    get_file_materialize_dir,
    get_uploads_dir,
    in_sandbox_tool_runner,
)
from .execution_scope import validate_scope_component
from .file_ref import (
    SANDBOX_FILE_ID_PREFIX,
    is_sandbox_local_file_id,
    parse_file_id_ref,
)
from .file_storage.keys import build_user_key_prefix

if TYPE_CHECKING:
    from ..web.services.uploaded_file_store import (
        StagedUploadedFile,
        SupersededObjectCleanupClaim,
        UploadedFileVersionSnapshot,
    )

logger = logging.getLogger(__name__)

DEFAULT_USER_FILE_LIST_LIMIT = 50

# Context variable for auto-registration mode
_auto_register = contextvars.ContextVar("_auto_register", default=False)

# Runtime scratch files must remain resolvable across equivalent TaskWorkspace
# instances in one process without becoming user-visible UploadedFile records.
# The canonical workspace root is part of every key, preventing a file ID from
# crossing task/workspace boundaries.
_internal_file_registry: Dict[Tuple[str, str], Path] = {}
_internal_path_registry: Dict[Tuple[str, str], str] = {}
_internal_file_registry_lock = RLock()
_INTERNAL_TEMP_DIR_NAME = ".xagent-internal"
_SANDBOX_PERMISSION_MIGRATION_MARKER = ".xagent-permissions-v2"
_SANDBOX_RUNTIME_GID = 1010
# A failed legacy migration is retried after a process restart or through an
# explicit migration call, but ordinary TaskWorkspace reconstruction must not
# repeat a synchronous tree walk on every tool call.
_sandbox_permission_migrations_attempted: set[str] = set()
_sandbox_permission_migrations_lock = Lock()
_SANDBOX_EXTERNAL_UPLOAD_STAGING_PARTS = (_INTERNAL_TEMP_DIR_NAME, "mcp-upload")


class ExternalUploadStagingError(ValueError):
    """A validated source could not be copied into upload staging."""


class ExternalUploadSizeLimitError(ExternalUploadStagingError):
    """An upload source exceeded the caller's staging byte limit."""


# The name of the output subtree the engine owns for spilled tool results.
# It is defined here, on the workspace side, and the spill module imports it
# from here: tool modules import core.workspace, never the other way round.
SPILL_DIR_NAME = "tool-results"
# Reserved-name collisions already announced by this process, keyed by the
# reserved path, so that a workspace built many times over -- once per tool
# family, per call on some paths -- announces the same collision once.
_announced_reserved_names: set[str] = set()
_announced_reserved_names_lock = Lock()


def _is_sandbox_prepared(entry_stat: os.stat_result) -> bool:
    """Return whether a directory carries the sandbox group contract."""

    return entry_stat.st_gid == _SANDBOX_RUNTIME_GID and bool(
        entry_stat.st_mode & stat.S_ISGID
    )


def scoped_user_root(
    base_dir: Union[str, Path, None],
    user_id: int,
    scope_segments: Sequence[str] = (),
) -> Path:
    """Single owner of the ``user root + scope segments`` path composition.

    Every user-root path — workspace base dirs, upload dirs, sandbox mounts,
    read paths — must be composed through this entry point instead of
    hand-building ``.../user_{id}`` literals, so an active
    :class:`ExecutionScope`'s ``workspace_segments`` cannot be applied to one
    path and missed on another (the "partially applied scope" bug class).

    Args:
        base_dir: Root the user directory lives under; None means the
            configured uploads dir.
        user_id: Platform user id (owns the root regardless of scope).
        scope_segments: ExecutionScope.workspace_segments, inserted after the
            user root. Empty (the default) composes today's unscoped path
            byte-for-byte.

    Returns:
        ``{base_dir}/user_{user_id}[/{segment}...]``

    Raises:
        InvalidScopeComponentError: a segment fails validation. Rejecting is
            deliberate — falling back to the unscoped path would silently
            merge namespaces.
    """
    root = Path(base_dir).expanduser() if base_dir is not None else get_uploads_dir()
    root = root / f"user_{int(user_id)}"
    for segment in scope_segments:
        validate_scope_component(segment, field_name="workspace_segments entry")
        root = root / segment
    return root


@dataclass
class AgentContext:
    """Agent execution context"""

    id: str
    workspace: Optional["TaskWorkspace"] = None


@dataclass(frozen=True)
class WorkspaceFileRegistration:
    """Validated local metadata needed to register a workspace file."""

    path: Path
    relative_path: str
    category: str
    is_workspace_file: bool


@dataclass(frozen=True)
class WorkspaceUploadedFileSnapshot:
    """Detached metadata consumed after the read Session releases its connection."""

    version: "UploadedFileVersionSnapshot"
    file_id: str
    user_id: int
    task_id: Optional[int]
    mime_type: Optional[str]
    workspace_relative_path: Optional[str]
    workspace_category: Optional[str]


@dataclass(frozen=True)
class WorkspaceFileRegistrationPlan:
    """Read-phase result for one workspace registration."""

    registration: WorkspaceFileRegistration
    file_id: str
    task_id: Optional[int]
    user_id: Optional[int]
    existing: Optional[WorkspaceUploadedFileSnapshot]
    occupied_storage_paths: tuple[tuple[str, str], ...]

    @property
    def should_persist(self) -> bool:
        return self.task_id is not None and self.user_id is not None


@dataclass(frozen=True)
class PreparedWorkspaceFileRegistration:
    """Storage-phase result passed into the short metadata transaction."""

    plan: WorkspaceFileRegistrationPlan
    staged: "StagedUploadedFile"


class TaskWorkspace:
    """
    Task workspace manager that provides isolated working directories for tasks.

    Each task gets its own workspace with:
    - input/: For input files
    - output/: For output files
    - temp/: For temporary files

    The workspace also supports access to external user directories (e.g., knowledge base files)
    through an allowed external directories whitelist.
    """

    def __init__(
        self,
        id: str,
        base_dir: Optional[str] = None,
        allowed_external_dirs: Optional[List[str]] = None,
        db_task_id: Optional[int] = None,
        scope_segments: Sequence[str] = (),
        durable_storage_segments: Sequence[str] | None = None,
        migrate_sandbox_permissions: bool = True,
    ):
        self.id = id
        self.db_task_id = db_task_id
        # ExecutionScope.workspace_segments this workspace was built under.
        # base_dir already ends with these segments for scoped workspaces;
        # they are carried separately so storage keys and canonical task
        # roots insert them after the user root instead of re-deriving them
        # from the path.
        self.scope_segments: tuple[str, ...] = tuple(scope_segments)
        for segment in self.scope_segments:
            validate_scope_component(segment, field_name="workspace_segments entry")
        # Logical workspace paths and durable object keys intentionally diverge
        # when an ExecutionScope does not isolate external directories. Keep the
        # write-side durable segments explicit so marked reads validate the key
        # layout that UploadedFileStore actually produced.
        self.durable_storage_segments: tuple[str, ...] = tuple(
            self.scope_segments
            if durable_storage_segments is None
            else durable_storage_segments
        )
        for segment in self.durable_storage_segments:
            validate_scope_component(segment, field_name="workspace_segments entry")
        if base_dir is None:
            base_dir = str(get_uploads_dir())
        self.base_dir = (
            Path(base_dir).expanduser().resolve()
        )  # Resolve base_dir to absolute path for consistent workspace reconstruction
        self.db_session = None  # Optional database session for file registration
        self._registration_lock = RLock()
        self._host_external_upload_staging_root: Optional[Path] = None
        self._host_external_upload_staging_cleanup: Any = None
        self._recently_registered_files: Dict[str, str] = {}  # path -> file_id mapping
        self._file_id_to_path: Dict[str, Path] = {}  # file_id -> path reverse mapping
        self.owner_user_id: Optional[int] = None
        self._migrate_sandbox_permissions = migrate_sandbox_permissions
        # Server-derived rollout policy for File Operation. ``None`` means the
        # exact upstream legacy path and therefore requires no policy query.
        # Public runtime construction sets version 1 only after loading a
        # persisted, server-stamped task marker.
        self.file_operation_access_version: Any = None
        self.current_task_id: Optional[int] = (
            db_task_id
            if db_task_id is not None
            else self._parse_task_id_from_workspace_id(id)
        )

        # Create workspace directory
        self.workspace_dir = self.base_dir / id
        self.input_dir = self.workspace_dir / "input"
        self.output_dir = self.workspace_dir / "output"
        self.temp_dir = self.workspace_dir / "temp"

        # Create the workspace tree before validating external roots.  The
        # configured base directory is also the canonical per-user upload root
        # in web runtimes, so an eagerly-created workspace must not discard it
        # merely because no upload has created that root yet.
        self._ensure_directories()

        # Allowed external directories (e.g., user upload directories with knowledge base files)
        self.allowed_external_dirs: List[Path] = []
        if allowed_external_dirs:
            for dir_path in allowed_external_dirs:
                path = Path(dir_path).resolve()
                if path.exists():
                    self.allowed_external_dirs.append(path)
                else:
                    logger.warning(
                        f"Allowed external directory does not exist: {dir_path}"
                    )

        self._warn_if_reserved_output_name_is_taken()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize durable workspace state without process-local synchronization."""

        state = dict(self.__dict__)
        state.pop("_registration_lock", None)
        state.pop("_host_external_upload_staging_root", None)
        state.pop("_host_external_upload_staging_cleanup", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a fresh registration owner in the receiving process."""

        self.__dict__.update(state)
        self.__dict__.setdefault("_migrate_sandbox_permissions", True)
        self._registration_lock = RLock()
        self._host_external_upload_staging_root = None
        self._host_external_upload_staging_cleanup = None

    @property
    def internal_temp_dir(self) -> Path:
        """Return the reserved root for process-local runtime scratch data."""

        return self.temp_dir / _INTERNAL_TEMP_DIR_NAME

    @property
    def engine_owned_output_dir(self) -> Path:
        """Return the output subtree the engine owns and the model may not write.

        :meth:`is_engine_owned_path` says what that reservation covers.
        """

        return self.output_dir / SPILL_DIR_NAME

    def is_engine_owned_path(self, file_path: Path) -> bool:
        """Return whether a path is the engine-owned output subtree or inside it.

        The argument and the parent of the reserved segment are resolved; the
        reserved segment itself is compared by name, so a symlink standing at
        the reserved name is an ordinary entry whose target is not reserved.
        A file at such a hijacked location is kept free of the engine's bytes
        by the spill writer's refusal to write through the link, not by this
        check. Equality counts as containment. A symlink loop in the argument
        propagates from ``resolve()``: a write guard fails rather than
        answering False on an unanswered question.
        """

        reserved_root = self.engine_owned_output_dir
        # The reserved segment is not resolved because a direct writer can
        # put a symlink there, and following it would let that writer choose
        # which directory is protected.
        parent_root = reserved_root.parent.resolve()
        resolved_path = file_path.resolve()
        # Every compared segment is case-folded on every file system: one
        # rule and one test expectation hold everywhere, at the price of also
        # reserving other spellings on a case-sensitive file system.
        parent_parts = tuple(part.casefold() for part in parent_root.parts)
        path_parts = tuple(part.casefold() for part in resolved_path.parts)
        if path_parts[: len(parent_parts)] != parent_parts:
            return False
        relative_parts = path_parts[len(parent_parts) :]
        if not relative_parts:
            return False
        return relative_parts[0] == reserved_root.name.casefold()

    def refuse_engine_owned_write(self, resolved_path: Path, requested: str) -> Path:
        """Refuse one resolved write target that lands in the engine-owned subtree.

        The one place :meth:`is_engine_owned_path` becomes a write refusal;
        the file tools' resolvers and :meth:`resolve_write_path` reach it,
        reads never do. Raises ValueError, the class the resolvers raise for
        a target outside the workspace, naming the path and the reason.
        """

        if self.is_engine_owned_path(resolved_path):
            raise ValueError(
                f"Path '{requested}' is inside the engine-owned "
                f"'{SPILL_DIR_NAME}' directory. The engine stores oversized "
                "tool results there and that directory is read-only for file "
                "tools: you can read those files, but you cannot write, edit, "
                "delete or create anything inside it. Write your own files "
                "somewhere else under output/."
            )
        return resolved_path

    def resolve_write_path(self, file_path: str, default_dir: str = "output") -> Path:
        """Resolve a target the caller is about to write, then apply the policy.

        Same resolution as :meth:`resolve_path`, followed by
        :meth:`refuse_engine_owned_write`. Tools that take a destination from
        the model and write to it directly -- rather than through the
        workspace file tools -- resolve it here, so the engine-owned subtree
        is refused on that path as well.
        """

        return self.refuse_engine_owned_write(
            self.resolve_path(file_path, default_dir), file_path
        )

    def stage_file_for_external_upload(
        self,
        file_id: str,
        *,
        max_bytes: int | None = None,
        host_private: bool = False,
    ) -> Path:
        """Materialize a durable file into this task's private staging area.

        Connectors intentionally accept only paths under the current task
        workspace.  A durable ``FileRef`` may resolve to the shared
        materialization cache instead, so copy it into the workspace before
        handing it to an external upload connector.  The staging directory is
        internal scratch and is removed with the task workspace; it is never
        registered as a user-visible file.
        """

        source = self.resolve_file_id_detached(file_id)
        if source is None:
            raise FileNotFoundError(f"File not found: {file_id}")
        return self._stage_authorized_source(
            Path(source),
            file_id,
            max_bytes=max_bytes,
            host_private=host_private,
        )

    def issue_external_media_url_detached(self, file_id: str, purpose: str) -> str:
        """Issue a purpose-bound URL for one exact task-owned registered image.

        This host-only bridge is deliberately separate from local-path staging:
        connectors such as Instagram ask their provider to fetch a URL rather
        than accepting uploaded bytes. The URL is never persisted in workspace
        state or returned to the model.
        """

        if self.owner_user_id is None or self.db_task_id is None:
            raise FileNotFoundError("Registered image was not found")

        from ..web.services.external_media import (
            ExternalMediaNotFound,
            ExternalMediaPurpose,
            ExternalMediaUnavailable,
            issue_external_media_url,
        )
        from .storage.manager import create_db_session

        try:
            external_purpose = ExternalMediaPurpose(purpose)
        except ValueError:
            raise ValueError("External media purpose is not supported") from None

        db = create_db_session()
        try:
            try:
                issued = issue_external_media_url(
                    db,
                    file_id=file_id,
                    user_id=self.owner_user_id,
                    task_id=self.db_task_id,
                    purpose=external_purpose,
                )
            except ExternalMediaNotFound:
                raise FileNotFoundError("Registered image was not found") from None
            except ExternalMediaUnavailable as exc:
                raise ValueError(str(exc)) from None
            return issued.url
        finally:
            db.close()

    @staticmethod
    def _open_regular_file_nofollow(source: Path, selector: str) -> int:
        """Open an absolute file without following any path component."""
        absolute = source.absolute()
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        current_fd = os.open(os.path.sep, directory_flags)
        try:
            for component in absolute.parts[1:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            file_fd = os.open(
                absolute.name,
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current_fd,
            )
        except OSError as exc:
            raise FileNotFoundError(f"File not found: {selector}") from exc
        finally:
            os.close(current_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            os.close(file_fd)
            raise FileNotFoundError(f"File not found: {selector}")
        return file_fd

    @classmethod
    def _validate_regular_file_nofollow(cls, source: Path, selector: str) -> Path:
        """Validate a lexical file capability without following any symlink."""
        absolute = source.absolute()
        file_fd = cls._open_regular_file_nofollow(absolute, selector)
        os.close(file_fd)
        return absolute

    def _stage_authorized_source(
        self,
        source: Path,
        selector: str,
        *,
        max_bytes: int | None = None,
        host_private: bool = False,
    ) -> Path:
        """Copy an already-authorized source fd into private task staging."""
        source_fd = self._open_regular_file_nofollow(source, selector)
        staging_dir: Path | None = None
        try:
            staging_root = self.external_upload_staging_root(host_private=host_private)
            staging_dir = staging_root / uuid4().hex
            target = staging_dir / source.name
            source_size = os.fstat(source_fd).st_size
            if max_bytes is not None and source_size > max_bytes:
                raise ExternalUploadSizeLimitError(
                    f"File exceeds maximum staging size of {max_bytes} bytes: "
                    f"{selector}"
                )
            staging_dir.mkdir(parents=True, exist_ok=False)
            with os.fdopen(source_fd, "rb", closefd=True) as source_file:
                source_fd = -1
                with target.open("xb") as target_file:
                    copied = 0
                    while chunk := source_file.read(1024 * 1024):
                        copied += len(chunk)
                        if max_bytes is not None and copied > max_bytes:
                            raise ExternalUploadSizeLimitError(
                                "File grew beyond maximum staging size of "
                                f"{max_bytes} bytes: {selector}"
                            )
                        target_file.write(chunk)
            os.chmod(target, 0o600)
        except BaseException:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        finally:
            if source_fd >= 0:
                os.close(source_fd)
        return target

    def stage_path_for_external_upload(
        self,
        file_path: str,
        *,
        max_bytes: int | None = None,
        host_private: bool = False,
    ) -> Path:
        """Copy one workspace-authorized selector into private task staging."""
        source = Path(self.resolve_path_with_search(file_path, detached_file_ids=True))
        if source.is_symlink():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            source = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FileNotFoundError(f"File not found: {file_path}") from exc
        if not source.is_absolute():
            raise FileNotFoundError(f"File not found: {file_path}")
        return self._stage_authorized_source(
            source,
            file_path,
            max_bytes=max_bytes,
            host_private=host_private,
        )

    def stage_resolved_path_for_external_upload(
        self,
        source: Path,
        selector: str,
        *,
        max_bytes: int | None = None,
        host_private: bool = False,
    ) -> Path:
        """Stage the capability returned by ``resolve_path_with_search`` once."""
        raw_source = Path(source)
        if raw_source.is_symlink():
            raise FileNotFoundError(f"File not found: {selector}")
        try:
            resolved = raw_source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FileNotFoundError(f"File not found: {selector}") from exc
        if not resolved.is_absolute():
            raise FileNotFoundError(f"File not found: {selector}")
        return self._stage_authorized_source(
            resolved,
            selector,
            max_bytes=max_bytes,
            host_private=host_private,
        )

    def discard_staged_external_upload(self, file_path: str | Path) -> None:
        """Remove a file previously returned by staging.

        Cleanup is deliberately confined to the per-task internal staging
        root, so a connector cannot cause arbitrary workspace deletion.
        """

        try:
            candidate = Path(file_path).resolve()
        except RuntimeError:
            # A symlink loop must not escape cleanup's confined-path check.
            candidate = Path(file_path)
        private_root = self._host_external_upload_staging_root
        staging_root = None
        if private_root is not None and candidate != private_root:
            if candidate.is_relative_to(private_root):
                staging_root = private_root
        if staging_root is None:
            try:
                sandbox_root = self.external_upload_staging_root()
            except ExternalUploadStagingError as exc:
                raise ValueError(
                    "staged upload path is outside the task staging area"
                ) from exc
            if candidate != sandbox_root and candidate.is_relative_to(sandbox_root):
                staging_root = sandbox_root
        if staging_root is None:
            raise ValueError("staged upload path is outside the task staging area")
        try:
            candidate.unlink(missing_ok=True)
        finally:
            parent = candidate.parent
            while parent != staging_root and parent.is_relative_to(staging_root):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    def external_upload_staging_root(self, *, host_private: bool = False) -> Path:
        """Return sandbox-visible connector staging or private media staging."""
        if host_private and not in_sandbox_tool_runner():
            with self._registration_lock:
                existing = self._host_external_upload_staging_root
                if existing is not None and existing.is_dir():
                    return existing
                cleanup = self._host_external_upload_staging_cleanup
                if cleanup is not None:
                    cleanup.detach()
                staging_root = Path(
                    tempfile.mkdtemp(prefix="xagent-external-upload-")
                ).resolve()
                os.chmod(staging_root, 0o700)
                self._host_external_upload_staging_root = staging_root
                self._host_external_upload_staging_cleanup = weakref.finalize(
                    self, shutil.rmtree, staging_root, ignore_errors=True
                )
                return staging_root

        parent_root = self.temp_dir.resolve(strict=True)
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(parent_root, flags)
        current_fd = parent_fd
        try:
            for component in _SANDBOX_EXTERNAL_UPLOAD_STAGING_PARTS:
                created = False
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    pass
                next_fd = os.open(component, flags, dir_fd=current_fd)
                try:
                    if created:
                        os.fchmod(next_fd, 0o700)
                except BaseException:
                    os.close(next_fd)
                    raise
                if current_fd != parent_fd:
                    os.close(current_fd)
                current_fd = next_fd
        except OSError as exc:
            raise ExternalUploadStagingError(
                "task upload staging area is unavailable"
            ) from exc
        finally:
            if current_fd != parent_fd:
                os.close(current_fd)
            os.close(parent_fd)
        return parent_root.joinpath(*_SANDBOX_EXTERNAL_UPLOAD_STAGING_PARTS)

    def _external_upload_staging_root(self) -> Path:
        """Backward-compatible accessor for sandbox-visible connector staging."""
        return self.external_upload_staging_root()

    def register_internal_file(
        self,
        file_path: str,
    ) -> str:
        """Register workspace scratch data without creating a file record.

        Internal registrations are process-local by design and are not
        available in sandbox subprocesses reconstructed from a serialized
        workspace. A checkpoint may retain the safe FileRef after its scratch
        bytes expire; model input then degrades to the reference's text fallback
        instead of resurrecting a user-visible artifact.
        """

        resolved_path = Path(file_path).resolve(strict=True)
        workspace_root = self.workspace_dir.resolve()
        temp_root = self.temp_dir.resolve()
        if not resolved_path.is_file() or not resolved_path.is_relative_to(temp_root):
            raise ValueError("internal files must be regular workspace temp files")

        workspace_key = str(workspace_root)
        path_key = (workspace_key, str(resolved_path))
        with _internal_file_registry_lock:
            existing_file_id = _internal_path_registry.get(path_key)
            if existing_file_id is not None:
                registered_path = _internal_file_registry.get(
                    (workspace_key, existing_file_id)
                )
                if registered_path == resolved_path:
                    final_file_id = existing_file_id
                else:
                    _internal_path_registry.pop(path_key, None)
                    final_file_id = f"internal-{uuid4()}"
            else:
                final_file_id = f"internal-{uuid4()}"

            file_key = (workspace_key, final_file_id)
            occupied_path = _internal_file_registry.get(file_key)
            if occupied_path is not None and occupied_path != resolved_path:
                raise ValueError("internal file_id is already registered")
            _internal_file_registry[file_key] = resolved_path
            _internal_path_registry[path_key] = final_file_id

        return final_file_id

    def _resolve_internal_file_id(self, file_id: str) -> Optional[Path]:
        workspace_key = str(self.workspace_dir.resolve())
        file_key = (workspace_key, file_id)
        with _internal_file_registry_lock:
            registered_path = _internal_file_registry.get(file_key)
            if registered_path is None:
                return None
            if (
                not registered_path.exists()
                or not registered_path.is_file()
                or not registered_path.resolve().is_relative_to(self.temp_dir.resolve())
            ):
                _internal_file_registry.pop(file_key, None)
                _internal_path_registry.pop(
                    (workspace_key, str(registered_path)),
                    None,
                )
                return None

        return registered_path

    def unregister_internal_file(self, file_path: str) -> Optional[str]:
        """Forget one process-local scratch file registration by path."""

        resolved_path = Path(file_path).resolve()
        workspace_key = str(self.workspace_dir.resolve())
        if not resolved_path.is_relative_to(self.temp_dir.resolve()):
            raise ValueError("internal files must be regular workspace temp files")
        path_key = (workspace_key, str(resolved_path))
        with _internal_file_registry_lock:
            file_id = _internal_path_registry.pop(path_key, None)
            if file_id is not None:
                _internal_file_registry.pop((workspace_key, file_id), None)
        return file_id

    def _get_internal_file_id_from_path(self, file_path: Path) -> Optional[str]:
        workspace_key = str(self.workspace_dir.resolve())
        path_key = (workspace_key, str(file_path))
        with _internal_file_registry_lock:
            file_id = _internal_path_registry.get(path_key)
            if file_id is None:
                return None
            registered_path = _internal_file_registry.get((workspace_key, file_id))
            if (
                registered_path != file_path
                or not file_path.exists()
                or not file_path.is_file()
                or not file_path.resolve().is_relative_to(self.temp_dir.resolve())
            ):
                _internal_path_registry.pop(path_key, None)
                _internal_file_registry.pop((workspace_key, file_id), None)
                return None
            return file_id

    def _is_internal_workspace_path(self, file_path: Path) -> bool:
        """Return whether a path is reserved or registered runtime scratch data.

        Three reserved kinds, skipped by every listing that consults this
        predicate (get_all_files, get_output_files, _scan_all_files): the
        process-local scratch root under temp (and, when that name is an
        alias, the directory directly under temp/ it points at), the engine-owned
        output subtree (see :meth:`is_engine_owned_path`), and any path
        registered as an internal file id. The file tool's named-directory
        listing calls is_engine_owned_path directly instead, so it hides only
        the engine tool-results directory, not the other two kinds.
        """

        resolved_path = file_path.resolve()
        if (
            file_path.parent.resolve() == self.workspace_dir.resolve()
            and file_path.name == _SANDBOX_PERMISSION_MIGRATION_MARKER
        ):
            return True
        # The reserved root is the name under temp/, not whatever that name
        # points at: resolve the parent, never the reserved segment itself.
        # Following it would put the failure, and the answer, in the hands of
        # whoever can create a symlink there -- and a loop there would make
        # this shared check raise for every caller, about every path. The
        # name itself is compared as written, not case folded: it is a
        # dot-name the engine creates itself, and no file tool resolves a
        # write through it.
        reserved_root = self.temp_dir.resolve() / _INTERNAL_TEMP_DIR_NAME
        if resolved_path.is_relative_to(reserved_root):
            return True
        # Unlike the engine-owned output subtree, a directory this name
        # currently points at is ALSO treated as reserved, but only when
        # that directory is a direct child of temp/ -- the one layout the
        # engine's own scratch writer accepts for this root (computer/store):
        # the engine never puts a symlink here itself, so the only way the
        # name points elsewhere is an alias placed by something else, and
        # scratch data that already lives at that alias stays internal
        # rather than surfacing as a new user file. An alias to temp/ itself
        # would make every temp/ file "inside" it and empty the temp listing;
        # a deeper directory is one the engine never writes into; and nothing
        # the engine writes ever lives outside temp/, so an alias reaching
        # further out names no scratch data to protect -- honouring it there
        # would let a symlink placed under temp/ blank out listings anywhere
        # else in the workspace. A loop has no target to follow, so it is
        # left to the name-based answer above instead of raising here.
        try:
            aliased_target = reserved_root.resolve()
        except RuntimeError:
            aliased_target = None
        if (
            aliased_target is not None
            and aliased_target != reserved_root
            and aliased_target.parent == reserved_root.parent
            and resolved_path.is_relative_to(aliased_target)
        ):
            return True
        if self.is_engine_owned_path(resolved_path):
            return True
        return self._get_internal_file_id_from_path(resolved_path) is not None

    def _forget_internal_files(self) -> None:
        workspace_key = str(self.workspace_dir.resolve())
        with _announced_reserved_names_lock:
            _announced_reserved_names.discard(str(self.engine_owned_output_dir))
        with _internal_file_registry_lock:
            file_keys = [
                key for key in _internal_file_registry if key[0] == workspace_key
            ]
            for key in file_keys:
                registered_path = _internal_file_registry.pop(key)
                _internal_path_registry.pop(
                    (workspace_key, str(registered_path)),
                    None,
                )

    def register_file(
        self, file_path: str, file_id: Optional[str] = None, db_session: Any = None
    ) -> str:
        registered = self.register_files(
            ((file_path, file_id),),
            db_session=db_session,
        )
        return registered[0]

    def register_delivery_file(self, file_path: str) -> str:
        """Register a host-delivered attachment only with durable task ownership."""
        if in_sandbox_tool_runner():
            raise ValueError("User delivery requires host-side registration")
        with self._registration_lock:
            return self._register_files_locked(
                ((file_path, None),), db_session=None, require_persistence=True
            )[0]

    def register_files(
        self,
        files: Sequence[tuple[str, Optional[str]]],
        *,
        db_session: Any = None,
    ) -> tuple[str, ...]:
        """Serialize one workspace's registration read/stage/commit sequence."""

        if not files:
            return ()
        if in_sandbox_tool_runner():
            return tuple(
                self._remember_sandbox_registration(path, file_id)
                for path, file_id in files
            )
        with self._registration_lock:
            return self._register_files_locked(files, db_session=db_session)

    def _register_files_locked(
        self,
        files: Sequence[tuple[str, Optional[str]]],
        *,
        db_session: Any,
        require_persistence: bool = False,
    ) -> tuple[str, ...]:
        """Register files through detached read, storage, and metadata phases.

        ``db_session`` is a compatibility reference for the caller's database
        bind, not the registration transaction owner. It must be clean so its
        connection can be released before any filesystem or object-storage I/O.
        Workspace-owned short Sessions perform the detached read and metadata
        compare-and-swap, and the metadata Session commits before this method
        returns. A later caller rollback therefore cannot orphan durable bytes.
        """

        resolved_db_session = db_session if db_session is not None else self.db_session
        self._release_registration_session_if_clean(resolved_db_session)

        # Path resolution, validation, and stat-like filesystem work happen
        # only after any clean caller transaction has returned its connection.
        registrations, result_indexes = self._normalize_file_registrations(files)
        plans = self._load_file_registration_plans(
            registrations,
            db_session=resolved_db_session,
        )
        if require_persistence and any(not plan.should_persist for plan in plans):
            raise ValueError("Attachment registration requires a persisted task owner")

        # Grant access only after the complete batch has passed path, ownership,
        # and persistence validation. Permission repair is best effort because a
        # caller may concurrently remove an otherwise valid path.
        for registration, _file_id in registrations:
            self._grant_sandbox_access_best_effort(registration)

        prepared: list[PreparedWorkspaceFileRegistration] = []
        try:
            for plan in plans:
                if plan.should_persist:
                    prepared.append(self._prepare_file_registration(plan))
        except Exception:
            self._compensate_prepared_file_registrations(prepared)
            raise

        cleanup_claims: tuple["SupersededObjectCleanupClaim", ...] = ()
        try:
            if prepared:
                cleanup_claims = self._apply_prepared_file_registrations(
                    prepared,
                    reference_db=resolved_db_session,
                )
        except Exception:
            self._compensate_prepared_file_registrations(prepared)
            raise

        # Cleanup happens only after the Workspace-owned metadata commit.
        if cleanup_claims:
            from ..web.services.uploaded_file_store import (
                cleanup_superseded_uploaded_file_objects,
            )

            failed_claims = cleanup_superseded_uploaded_file_objects(cleanup_claims)
            if failed_claims:
                logger.warning(
                    "Failed to clean %s superseded workspace file generation(s)",
                    len(failed_claims),
                )

        for plan in plans:
            self._remember_file_registration(
                plan.file_id,
                plan.registration.path,
            )
        unique_file_ids = tuple(plan.file_id for plan in plans)
        return tuple(unique_file_ids[index] for index in result_indexes)

    def _normalize_file_registrations(
        self,
        files: Sequence[tuple[str, Optional[str]]],
    ) -> tuple[
        tuple[tuple[WorkspaceFileRegistration, Optional[str]], ...],
        tuple[int, ...],
    ]:
        """Collapse duplicate canonical paths before staging durable generations."""

        unique: list[tuple[WorkspaceFileRegistration, Optional[str]]] = []
        index_by_path: dict[str, int] = {}
        path_by_requested_file_id: dict[str, str] = {}
        result_indexes: list[int] = []

        for file_path, file_id in files:
            registration = self.describe_file_registration(file_path)
            path_key = str(registration.path)
            requested_file_id = str(file_id).strip() if file_id is not None else None
            if not requested_file_id:
                requested_file_id = None

            existing_index = index_by_path.get(path_key)
            if existing_index is not None:
                existing_registration, existing_file_id = unique[existing_index]
                if (
                    existing_file_id is not None
                    and requested_file_id is not None
                    and existing_file_id != requested_file_id
                ):
                    raise ValueError(
                        "One workspace path cannot be registered with multiple "
                        "file ids in the same batch"
                    )
                if existing_file_id is None and requested_file_id is not None:
                    unique[existing_index] = (
                        existing_registration,
                        requested_file_id,
                    )
                if requested_file_id is not None:
                    existing_path = path_by_requested_file_id.get(requested_file_id)
                    if existing_path is not None and existing_path != path_key:
                        raise ValueError(
                            "One file id cannot identify multiple workspace paths in the same batch"
                        )
                    path_by_requested_file_id[requested_file_id] = path_key
                result_indexes.append(existing_index)
                continue

            if requested_file_id is not None:
                existing_path = path_by_requested_file_id.get(requested_file_id)
                if existing_path is not None and existing_path != path_key:
                    raise ValueError(
                        "One file id cannot identify multiple workspace paths in the same batch"
                    )
                path_by_requested_file_id[requested_file_id] = path_key

            unique_index = len(unique)
            index_by_path[path_key] = unique_index
            unique.append((registration, requested_file_id))
            result_indexes.append(unique_index)

        return tuple(unique), tuple(result_indexes)

    def _grant_registered_path_sandbox_access(
        self, registration: WorkspaceFileRegistration
    ) -> None:
        """Keep host-created workspace deliverables editable by uid 1100."""
        if not registration.is_workspace_file or in_sandbox_tool_runner():
            return
        workspace_root = self.workspace_dir.resolve()
        root_stat = workspace_root.stat()
        if not _is_sandbox_prepared(root_stat):
            return
        try:
            relative = registration.path.absolute().relative_to(workspace_root)
        except ValueError:
            return

        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        current_fd = os.open(workspace_root, directory_flags)
        try:
            for index, component in enumerate(relative.parts):
                is_last = index == len(relative.parts) - 1
                try:
                    component_stat = os.stat(
                        component, dir_fd=current_fd, follow_symlinks=False
                    )
                except OSError:
                    return
                is_directory = stat.S_ISDIR(component_stat.st_mode)
                is_regular = stat.S_ISREG(component_stat.st_mode)
                if not is_directory and not is_regular:
                    return
                if is_last and is_regular and component_stat.st_nlink > 1:
                    return
                flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
                if not is_last:
                    flags |= os.O_DIRECTORY
                try:
                    next_fd = os.open(component, flags, dir_fd=current_fd)
                except OSError:
                    return
                os.close(current_fd)
                current_fd = next_fd
                entry_stat = os.fstat(current_fd)
                if (
                    entry_stat.st_dev != component_stat.st_dev
                    or entry_stat.st_ino != component_stat.st_ino
                ):
                    return
                # Preserve the creating uid. In particular, sandbox-created
                # files must stay owned by uid 1100 so chmod/utime/copy2 keep
                # working on later sandbox calls.
                os.fchown(current_fd, -1, _SANDBOX_RUNTIME_GID)
                os.fchmod(
                    current_fd,
                    self._sandbox_compatible_mode(
                        entry_stat.st_mode, directory=is_directory
                    ),
                )
        finally:
            if current_fd != -1:
                os.close(current_fd)

    def _grant_sandbox_access_best_effort(
        self, registration: WorkspaceFileRegistration
    ) -> None:
        """Best-effort permission repair for an already validated path."""

        try:
            self._grant_registered_path_sandbox_access(registration)
        except OSError:
            logger.warning(
                "Failed to grant sandbox access to workspace path %s",
                registration.path,
                exc_info=True,
            )

    def grant_sandbox_access(self, path: Union[str, Path]) -> None:
        """Best-effort sandbox access for one host-created file or directory.

        Some host-side tools create files without registering them as user
        deliverables.  They still need the same descriptor-relative
        directory and file permission repair as registered outputs when the
        workspace is prepared for uid 1100/gid 1010.
        """

        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_dir / candidate
        candidate = Path(os.path.abspath(candidate))
        workspace_root = self.workspace_dir.resolve()
        try:
            relative = candidate.relative_to(workspace_root)
        except ValueError:
            return
        registration = WorkspaceFileRegistration(
            path=candidate,
            relative_path=relative.as_posix(),
            category=relative.parts[0] if relative.parts else "workspace",
            is_workspace_file=True,
        )
        self._grant_sandbox_access_best_effort(registration)

    @staticmethod
    def _release_registration_session_if_clean(db: Any) -> None:
        if db is None:
            return
        from ..web.models.database import release_db_connection_if_clean

        if not release_db_connection_if_clean(db):
            raise RuntimeError(
                "Cannot register workspace files while the caller database "
                "session has pending writes"
            )

    def _load_file_registration_plans(
        self,
        registrations: Sequence[tuple[WorkspaceFileRegistration, Optional[str]]],
        *,
        db_session: Any,
    ) -> tuple[WorkspaceFileRegistrationPlan, ...]:
        db = self._create_registration_session(db_session)

        try:
            return tuple(
                self._load_file_registration_plan(
                    db,
                    registration=registration,
                    requested_file_id=requested_file_id,
                )
                for registration, requested_file_id in registrations
            )
        finally:
            db.close()

    @staticmethod
    def _create_registration_session(reference_db: Any) -> Any:
        """Open one Workspace-owned Session on the caller's database engine."""

        if reference_db is None:
            from ..web.models.database import get_session_local
            from .storage.manager import create_db_session

            try:
                RegistrationSession = get_session_local()
            except RuntimeError:
                # Core-only callers may initialize the ad-hoc storage database
                # without bootstrapping the web application's session factory.
                return create_db_session()
            return RegistrationSession()

        from sqlalchemy.engine import Connection
        from sqlalchemy.orm import sessionmaker

        bind = reference_db.get_bind()
        if isinstance(bind, Connection):
            bind = bind.engine
        RegistrationSession = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=bind,
        )
        return RegistrationSession()

    def _load_file_registration_plan(
        self,
        db: Any,
        *,
        registration: WorkspaceFileRegistration,
        requested_file_id: Optional[str],
    ) -> WorkspaceFileRegistrationPlan:
        from ..web.models.task import Task
        from ..web.models.uploaded_file import UploadedFile
        from ..web.services.uploaded_file_store import snapshot_uploaded_file_version

        task_id = self.db_task_id
        if task_id is None:
            task_id = self._parse_task_id_from_workspace_id(self.id)

        task_row = None
        if task_id is not None:
            task_row = (
                db.query(Task.id, Task.user_id).filter(Task.id == int(task_id)).first()
            )
        if task_row is None:
            if task_id is not None:
                logger.warning("Task %s not found, cannot create file record", task_id)
            final_file_id = requested_file_id or str(uuid4())
            return WorkspaceFileRegistrationPlan(
                registration=registration,
                file_id=final_file_id,
                task_id=None,
                user_id=None,
                existing=None,
                occupied_storage_paths=(),
            )

        task_id = int(task_row.id)
        task_user_id = int(task_row.user_id)
        cached_file_id = self._recently_registered_files.get(str(registration.path))
        candidate_file_id = cached_file_id or requested_file_id

        existing_record = None
        if candidate_file_id:
            existing_record = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.file_id == candidate_file_id,
                    UploadedFile.storage_status != "compensating",
                )
                .first()
            )
        if existing_record is None:
            existing_record = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.storage_path == str(registration.path),
                    UploadedFile.storage_status != "compensating",
                )
                .first()
            )

        existing = None
        if existing_record is not None:
            record_user_id = int(existing_record.user_id)
            if record_user_id != task_user_id:
                raise PermissionError(
                    "Cannot register a workspace file over metadata owned by another user"
                )
            if existing_record.detached_reason is not None:
                raise ValueError(
                    "Retained files must be explicitly reattached before registration"
                )
            existing = WorkspaceUploadedFileSnapshot(
                version=snapshot_uploaded_file_version(existing_record),
                file_id=str(existing_record.file_id),
                user_id=record_user_id,
                task_id=(
                    int(existing_record.task_id)
                    if existing_record.task_id is not None
                    else None
                ),
                mime_type=(
                    str(existing_record.mime_type)
                    if existing_record.mime_type is not None
                    else None
                ),
                workspace_relative_path=(
                    str(existing_record.workspace_relative_path)
                    if existing_record.workspace_relative_path is not None
                    else None
                ),
                workspace_category=(
                    str(existing_record.workspace_category)
                    if existing_record.workspace_category is not None
                    else None
                ),
            )

        final_file_id = (
            existing.file_id
            if existing is not None
            else requested_file_id or str(uuid4())
        )
        occupied_storage_paths = tuple(
            (str(storage_path), str(file_id))
            for storage_path, file_id in (
                db.query(UploadedFile.storage_path, UploadedFile.file_id)
                .filter(
                    UploadedFile.user_id == task_user_id,
                    UploadedFile.storage_path.isnot(None),
                )
                .all()
            )
            if storage_path is not None
        )
        return WorkspaceFileRegistrationPlan(
            registration=registration,
            file_id=final_file_id,
            task_id=task_id,
            user_id=task_user_id,
            existing=existing,
            occupied_storage_paths=occupied_storage_paths,
        )

    def _prepare_file_registration(
        self,
        plan: WorkspaceFileRegistrationPlan,
    ) -> PreparedWorkspaceFileRegistration:
        from ..web.services.uploaded_file_store import (
            stage_uploaded_file_from_local_path,
        )
        from .execution_scope import ExecutionScope
        from .file_storage.keys import build_task_output_storage_key

        assert plan.task_id is not None
        assert plan.user_id is not None

        local_path = plan.registration.path
        relative_path: Optional[str] = plan.registration.relative_path
        category: Optional[str] = plan.registration.category
        if not plan.registration.is_workspace_file and plan.existing is not None:
            relative_path = plan.existing.workspace_relative_path
            category = plan.existing.workspace_category

        if category == "output" and self._is_delegated_db_task_workspace(plan.task_id):
            local_path, relative_path = self._materialize_delegated_output(
                plan,
                source_path=local_path,
                relative_path=relative_path or local_path.name,
            )
            category = "output"

        mime_type = plan.existing.mime_type if plan.existing is not None else None
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(local_path.name)
        mime_type = mime_type or "application/octet-stream"

        logical_relative_path = relative_path or local_path.name
        generation_relative_path = f"_versions/{uuid4().hex}/{logical_relative_path}"
        staged = stage_uploaded_file_from_local_path(
            local_path=local_path,
            user_id=plan.user_id,
            file_id=plan.file_id,
            task_id=plan.task_id,
            filename=local_path.name,
            mime_type=mime_type,
            storage_key=build_task_output_storage_key(
                plan.user_id,
                plan.task_id,
                plan.file_id,
                generation_relative_path,
                scope_segments=self.scope_segments,
            ),
            workspace_relative_path=relative_path,
            workspace_category=category,
            execution_scope=ExecutionScope(
                workspace_segments=self.scope_segments,
                isolate_external_dirs=bool(self.scope_segments),
            ),
        )
        return PreparedWorkspaceFileRegistration(plan=plan, staged=staged)

    def _materialize_delegated_output(
        self,
        plan: WorkspaceFileRegistrationPlan,
        *,
        source_path: Path,
        relative_path: str,
    ) -> tuple[Path, str]:
        assert plan.task_id is not None
        assert plan.user_id is not None

        relative_parts = Path(relative_path).parts
        output_parts = [
            part for part in relative_parts[1:] if part not in ("", ".", "..")
        ]
        if not output_parts:
            output_parts = [source_path.name]

        task_root = self._user_workspace_base_dir(plan.user_id) / (
            f"web_task_{plan.task_id}"
        )
        output_root = (task_root / "output").resolve()
        candidate = (task_root / Path("output", *output_parts)).resolve()
        try:
            candidate.relative_to(output_root)
        except ValueError:
            candidate = (output_root / source_path.name).resolve()

        occupied_by_path = dict(plan.occupied_storage_paths)
        stem = candidate.stem
        suffix = candidate.suffix
        index = 1
        source_resolved = source_path.resolve()
        while True:
            candidate_resolved = candidate.resolve()
            occupying_file_id = occupied_by_path.get(str(candidate))
            same_record = occupying_file_id == plan.file_id
            record_conflicts = (
                occupying_file_id is not None and occupying_file_id != plan.file_id
            )
            path_conflicts = (
                candidate.exists()
                and candidate_resolved != source_resolved
                and not same_record
            )
            if not record_conflicts and not path_conflicts:
                break
            candidate = candidate.parent / f"{stem}_{index}{suffix}"
            index += 1

        canonical_relative_path = candidate.relative_to(task_root).as_posix()
        if candidate.resolve() != source_resolved:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, candidate)
        return candidate, canonical_relative_path

    def _apply_prepared_file_registrations(
        self,
        prepared: Sequence[PreparedWorkspaceFileRegistration],
        *,
        reference_db: Any,
    ) -> tuple["SupersededObjectCleanupClaim", ...]:
        from ..web.models.task import Task
        from ..web.services.uploaded_file_store import UploadedFileStore

        db = self._create_registration_session(reference_db)

        cleanup_claims: list[SupersededObjectCleanupClaim] = []
        try:
            for item in prepared:
                plan = item.plan
                task_exists = (
                    db.query(Task.id)
                    .filter(
                        Task.id == plan.task_id,
                        Task.user_id == plan.user_id,
                    )
                    .first()
                    is not None
                )
                if not task_exists:
                    raise ValueError(
                        f"Task {plan.task_id} is not owned by user {plan.user_id}"
                    )
                applied = UploadedFileStore(db).upsert_already_durable(
                    item.staged,
                    expected=(
                        plan.existing.version if plan.existing is not None else None
                    ),
                    allow_task_rebind=(
                        plan.existing is not None
                        and plan.existing.task_id != plan.task_id
                    ),
                )
                if applied.superseded_cleanup_claim is not None:
                    cleanup_claims.append(applied.superseded_cleanup_claim)
            db.commit()
            return tuple(cleanup_claims)
        except Exception:
            try:
                db.rollback()
            except Exception:
                logger.warning(
                    "Failed to roll back workspace file metadata Session",
                    exc_info=True,
                )
            raise
        finally:
            db.close()

    @staticmethod
    def _compensate_prepared_file_registrations(
        prepared: Sequence[PreparedWorkspaceFileRegistration],
    ) -> None:
        if not prepared:
            return
        from ..web.services.uploaded_file_store import (
            compensate_staged_uploaded_files,
        )

        failed_file_ids = compensate_staged_uploaded_files(
            tuple(item.staged for item in prepared)
        )
        if failed_file_ids:
            logger.warning(
                "Failed to compensate %s staged workspace file generation(s)",
                len(failed_file_ids),
            )

    def _create_file_record(
        self,
        file_id: str,
        file_path: Path,
        db_session: Any = None,
    ) -> None:
        """Compatibility facade for legacy callers and test seams.

        Registration is owned by :meth:`register_files`; this private method
        remains callable while downstream integrations migrate away from
        patching the former single-phase persistence hook.
        """

        self.register_files(
            ((str(file_path), file_id),),
            db_session=db_session,
        )

    def bind_already_durable_file(
        self,
        registration: WorkspaceFileRegistration,
        *,
        file_id: str,
    ) -> str:
        """Bind a separately persisted durable file into this workspace cache.

        This is deliberately distinct from :meth:`register_file`: it never
        queries the database or uploads bytes.  The caller must first commit an
        ``UploadedFile`` row whose durable metadata matches ``file_id``.
        """

        normalized_file_id = str(file_id).strip()
        if not normalized_file_id:
            raise ValueError("file_id is required for an already durable file")
        self._grant_sandbox_access_best_effort(registration)
        with self._registration_lock:
            self._remember_file_registration(normalized_file_id, registration.path)
        return normalized_file_id

    def describe_file_registration(self, file_path: str) -> WorkspaceFileRegistration:
        """Return validated, persistence-ready metadata without side effects."""

        resolved_path = self._resolve_file_for_registration(file_path)
        try:
            relative_path = resolved_path.relative_to(
                self.workspace_dir.resolve()
            ).as_posix()
            is_workspace_file = True
        except ValueError:
            relative_path = resolved_path.name
            is_workspace_file = False
        category = relative_path.split("/", 1)[0] if relative_path else "workspace"
        return WorkspaceFileRegistration(
            path=resolved_path,
            relative_path=relative_path,
            category=category,
            is_workspace_file=is_workspace_file,
        )

    def _resolve_file_for_registration(self, file_path: str) -> Path:
        """Resolve and validate the shared path precondition for registration."""

        resolved_path = self.resolve_path(file_path, default_dir="output")
        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(f"File not found for registration: {file_path}")

        workspace_abs = self.workspace_dir.resolve()
        is_valid = False
        try:
            resolved_path.relative_to(workspace_abs)
            is_valid = True
        except ValueError:
            for allowed_dir in self.allowed_external_dirs:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    is_valid = True
                    break
                except ValueError:
                    pass

        if not is_valid:
            raise ValueError(
                f"Path {file_path} is outside workspace and allowed directories"
            )
        return resolved_path

    def _remember_sandbox_registration(
        self, file_path: str, file_id: Optional[str]
    ) -> str:
        """Mint a process-local id inside the sandbox runner.

        The sandbox reaches no real database or object storage, so the host
        process re-registers these files once the tool call returns. Minted ids
        carry a prefix so anything the host fails to re-register stays
        recognizable instead of passing as a database-backed id.
        """
        resolved_path = Path(file_path).resolve()
        with self._registration_lock:
            cached = self._recently_registered_files.get(str(resolved_path))
        resolved_id = file_id or cached or f"{SANDBOX_FILE_ID_PREFIX}{uuid4()}"
        self._remember_file_registration(resolved_id, resolved_path)
        logger.debug(
            "Sandbox-local file id %s for %s; host process owns registration",
            resolved_id,
            resolved_path,
        )
        return resolved_id

    def _remember_file_registration(self, file_id: str, file_path: Path) -> None:
        with self._registration_lock:
            path_str = str(file_path)
            self._recently_registered_files[path_str] = file_id
            self._file_id_to_path[file_id] = file_path

    def _is_delegated_db_task_workspace(self, task_id: int) -> bool:
        if self.db_task_id is None:
            return False
        parsed_task_id = self._parse_task_id_from_workspace_id(self.id)
        return parsed_task_id != int(task_id)

    def _user_workspace_base_dir(self, user_id: int) -> Path:
        # base_dir may already be the (scoped) user base
        # (``.../user_{id}[/{segments}]``) or a raw uploads root; in the
        # latter case the user root + scope segments are appended.
        expected_tail = scoped_user_root(Path("/"), user_id, self.scope_segments).parts[
            1:
        ]
        if self.base_dir.parts[-len(expected_tail) :] == expected_tail:
            return self.base_dir
        return scoped_user_root(self.base_dir, user_id, self.scope_segments)

    def _get_file_id_from_db(
        self, file_path: Path, db_session: Any = None
    ) -> Optional[str]:
        """Get file_id from database by file path."""
        from .storage.manager import create_db_session

        try:
            from ..web.models.uploaded_file import UploadedFile

            if db_session:
                db = db_session
                should_close = False
            else:
                db = create_db_session()
                should_close = True

            try:
                record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.storage_path == str(file_path))
                    .first()
                )
                if record:
                    return str(record.file_id)
                return None
            finally:
                if should_close:
                    db.close()
        except Exception as e:
            logger.warning(f"Failed to query file_id from database: {e}")
            return None

    def get_registered_file_id(self, file_path: str) -> Optional[str]:
        try:
            resolved_path = self.resolve_path(file_path, default_dir="output")
            return self._get_file_id_from_db(resolved_path)
        except Exception:
            return None

    @staticmethod
    def _parse_task_id_from_workspace_id(workspace_id: str) -> Optional[int]:
        try:
            return int(str(workspace_id).split("_")[-1])
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _normalize_file_selector(file_path: str) -> str:
        """Normalize a file reference without resolving its authority."""

        normalized = str(file_path).strip()
        referenced_file_id = parse_file_id_ref(normalized)
        if referenced_file_id is not None:
            return referenced_file_id
        if normalized.startswith("file:") and not normalized.startswith("file://"):
            # Preserve legacy workspace path refs such as
            # ``file:output/report.csv``. They are paths, not file ids, and
            # still pass through the normal workspace containment checks.
            return normalized[5:].strip()
        return normalized

    def requires_exact_file_operation_scope(self) -> bool:
        """Return whether File Operation must use exact owner/task authority.

        Runtime construction supplies the server-derived marker. Its absence
        delegates directly to upstream behavior without a policy database
        lookup. A marked workspace revalidates the persisted task, owner, and
        explicit database task identity before any legacy file cache or
        allowlist can be consulted.
        """

        marker = self.file_operation_access_version
        if marker is None:
            return False

        from ..web.models.task import Task
        from .storage.manager import create_db_session
        from .task_runtime import (
            SUPPORTED_FILE_OPERATION_ACCESS_VERSIONS,
            FileOperationAccessPolicyError,
            requires_exact_file_operation_scope,
        )

        # Validate the propagated workspace marker independently. The task's
        # persisted marker is loaded and revalidated at the database authority
        # boundary below so construction-time state cannot substitute for it.
        if (
            isinstance(marker, bool)
            or not isinstance(marker, int)
            or marker not in SUPPORTED_FILE_OPERATION_ACCESS_VERSIONS
        ):
            raise FileOperationAccessPolicyError(
                "File Operation workspace policy version is unsupported"
            )
        if self.db_task_id is None or self.owner_user_id is None:
            raise FileOperationAccessPolicyError(
                "Marked File Operation workspace has no authoritative identity"
            )

        # File Operation callables run in worker threads. Always use an
        # operation-local session for policy checks instead of carrying a
        # potentially thread-bound caller session into that worker.
        db = create_db_session()
        try:
            task = db.query(Task).filter(Task.id == self.db_task_id).first()
            if task is None:
                raise FileOperationAccessPolicyError(
                    "Marked File Operation task no longer exists"
                )
            if not requires_exact_file_operation_scope(task):
                raise FileOperationAccessPolicyError(
                    "Marked File Operation task lost its persisted policy"
                )
            if (
                int(task.id) != self.db_task_id
                or int(task.user_id) != self.owner_user_id
            ):
                raise FileOperationAccessPolicyError(
                    "Marked File Operation workspace authority disagrees with task"
                )
            return True
        finally:
            db.close()

    def _file_record_allowed_for_workspace(
        self, record: Any, path: Optional[Path] = None
    ) -> bool:
        if (
            getattr(record, "detached_reason", None) is not None
            or getattr(record, "storage_status", None) == "compensating"
        ):
            return False
        if path is not None:
            workspace_abs = self.workspace_dir.resolve()
            resolved_path = path.resolve()
            if resolved_path == workspace_abs or resolved_path.is_relative_to(
                workspace_abs
            ):
                return True

        owner_user_id = self.owner_user_id
        if owner_user_id is None:
            return True

        record_user_id = getattr(record, "user_id", None)
        if record_user_id != owner_user_id:
            return False

        # For a scoped workspace, an owner match is not sufficient: the record
        # must also live under this workspace's scope subtree. Otherwise a task
        # could resolve, by file_id, a record belonging to the same owner but a
        # different scope (e.g. another end user's upload, which has
        # task_id=None and would pass the checks below). Unscoped workspaces
        # (no scope_segments) keep the owner-only behavior byte-for-byte.
        if self.scope_segments and not self._record_in_scope_subtree(record, path):
            return False

        record_task_id = getattr(record, "task_id", None)
        if record_task_id is None:
            return True
        return (
            self.current_task_id is not None and record_task_id == self.current_task_id
        )

    def _record_in_scope_subtree(
        self, record: Any, path: Optional[Path] = None
    ) -> bool:
        """Whether a file record lives under this workspace's scope subtree.

        Only meaningful when ``scope_segments`` is non-empty. Validates the
        artifact that ``resolve_file_id`` will actually hand back:

        - When an explicit local ``path`` is supplied, that path *is* the
          return value, so it is authoritative and must lie under the scoped
          user root. An in-scope durable ``storage_key`` must never admit an
          out-of-scope local path.
        - Otherwise resolution materializes from the durable handle, so the
          ``storage_key`` prefix (``users/{owner}/{segments}/``) is checked,
          falling back to the record's own ``storage_path`` only when no key
          is present.

        Fails closed (returns False) when nothing can be confirmed, so an
        owner match alone can never admit a sibling scope's file.
        """
        owner_user_id = self.owner_user_id
        if owner_user_id is None:
            return True

        # Branch 1: the explicit local path is what gets returned. Confine it
        # regardless of any storage_key, so an in-scope key can't smuggle an
        # out-of-scope local path past the gate.
        if path is not None:
            return self._path_under_scoped_user_root(path, owner_user_id)

        # Branch 2: no local path, so resolution materializes from the durable
        # handle -- validate the key prefix when present.
        storage_key = getattr(record, "storage_key", None)
        if storage_key:
            key = str(storage_key)
            key_prefix = "/".join(["users", str(owner_user_id), *self.scope_segments])
            return key == key_prefix or key.startswith(key_prefix + "/")

        # No key: fall back to the record's own local storage path.
        storage_path = getattr(record, "storage_path", None)
        if storage_path:
            return self._path_under_scoped_user_root(Path(storage_path), owner_user_id)

        return False

    def _path_under_scoped_user_root(self, candidate: Path, owner_user_id: Any) -> bool:
        """Whether ``candidate`` resolves under the scoped user workspace root."""
        scoped_root = self._user_workspace_base_dir(int(owner_user_id)).resolve()
        resolved = candidate.resolve()
        return resolved == scoped_root or resolved.is_relative_to(scoped_root)

    def resolve_file_id(self, file_id: str) -> Optional[Path]:
        return self._resolve_file_id(file_id, use_bound_db_session=True)

    def resolve_file_id_detached(self, file_id: str) -> Optional[Path]:
        """Resolve a file ID without reusing the caller's database session.

        This variant is safe to invoke from a worker thread. It preserves the
        same workspace ownership and scope checks as :meth:`resolve_file_id`,
        while opening and closing its own session when a database lookup is
        needed.
        """
        return self._resolve_file_id(file_id, use_bound_db_session=False)

    def _resolve_file_id(
        self,
        file_id: str,
        *,
        use_bound_db_session: bool,
    ) -> Optional[Path]:
        raw_file_id = str(file_id).strip()
        file_id = parse_file_id_ref(raw_file_id) or raw_file_id
        if not file_id:
            return None

        # Check in-memory cache first
        with self._registration_lock:
            cached_path = self._file_id_to_path.get(file_id)
        if cached_path is not None:
            try:
                resolved_cached_path = cached_path.resolve(strict=True)
                try:
                    cached_path = self._validate_regular_file_nofollow(
                        cached_path, file_id
                    )
                except FileNotFoundError:
                    # A configured external root may itself be an
                    # operator-controlled symlink. Its canonical target was
                    # frozen when the workspace was constructed; only that
                    # trusted-prefix case may retry with the canonical path.
                    if not any(
                        resolved_cached_path == root
                        or resolved_cached_path.is_relative_to(root)
                        for root in self.allowed_external_dirs
                    ):
                        raise
                    cached_path = self._validate_regular_file_nofollow(
                        resolved_cached_path, file_id
                    )
            except (FileNotFoundError, OSError, RuntimeError):
                resolved_cached_path = None
            if resolved_cached_path is not None and (
                self._file_operation_path_in_authorized_storage(resolved_cached_path)
            ):
                logger.debug(
                    f"resolve_file_id: Found in cache: {file_id} -> {cached_path}"
                )
                return cached_path
            else:
                logger.warning(
                    f"resolve_file_id: Cached path doesn't exist: {cached_path}"
                )
                # Remove stale cache entry
                with self._registration_lock:
                    if self._file_id_to_path.get(file_id) == cached_path:
                        self._file_id_to_path.pop(file_id, None)

        internal_path = self._resolve_internal_file_id(file_id)
        if internal_path is not None:
            logger.debug(
                "resolve_file_id: Found workspace-internal file: %s -> %s",
                file_id,
                internal_path,
            )
            return internal_path
        if file_id.startswith("internal-"):
            logger.warning(
                "resolve_file_id: Process-local internal file is unavailable in "
                "this process or has expired: %s",
                file_id,
            )
            return None
        if is_sandbox_local_file_id(file_id):
            logger.warning(
                "resolve_file_id: Sandbox-minted id never reached the database, so "
                "host re-registration failed for it: %s",
                file_id,
            )
            return None

        # Query from database
        from .storage.manager import create_db_session

        try:
            from ..web.models.uploaded_file import UploadedFile

            if use_bound_db_session and self.db_session is not None:
                db = self.db_session
                should_close = False
            else:
                db = create_db_session()
                should_close = True
            try:
                record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == file_id)
                    .first()
                )
                if record and record.storage_path:
                    raw_path = Path(record.storage_path)
                    if not self._file_record_allowed_for_workspace(record, raw_path):
                        logger.warning(
                            "Rejected file_id outside workspace scope: %s",
                            file_id,
                        )
                        return None
                    try:
                        return self._validate_regular_file_nofollow(raw_path, file_id)
                    except FileNotFoundError:
                        pass
                if (
                    record
                    and getattr(record, "storage_key", None)
                    and getattr(record, "storage_status", None) == "available"
                ):
                    if not self._file_record_allowed_for_workspace(record):
                        logger.warning(
                            "Rejected durable file_id outside workspace scope: %s",
                            file_id,
                        )
                        return None
                    from ..web.services.managed_file_ref import ManagedFileRef

                    raw_materialized = ManagedFileRef(record).materialize()
                    materialized = raw_materialized.resolve(strict=True)
                    if self._file_operation_path_in_authorized_storage(materialized):
                        return self._validate_regular_file_nofollow(
                            raw_materialized, file_id
                        )
                    logger.warning(
                        "Rejected materialized file_id outside authorized storage: %s",
                        file_id,
                    )
                    return None
                return None
            finally:
                if should_close:
                    db.close()
        except Exception as e:
            # ``exc_info`` because this fault is swallowed -- the caller only
            # sees ``None`` -- so this is its only record. A durable-storage
            # fault arrives here from ``materialize()`` carrying just the
            # storage key; its cause lives in ``__cause__`` (#1467).
            logger.warning(
                f"Failed to resolve file_id from database: {e}", exc_info=True
            )
            return None

    def _file_operation_path_in_authorized_storage(self, path: Path) -> bool:
        """Return whether a record-backed path stays in configured storage."""

        resolved = path.resolve()
        # The materialization cache is host-shared, so containment within it
        # is never sufficient authority. Callers reach this check only after an
        # exact UploadedFile owner/task/status/storage-key authorization.
        roots = [
            self.base_dir.resolve(),
            self.workspace_dir.resolve(),
            get_file_materialize_dir().expanduser().resolve(),
            *self.allowed_external_dirs,
        ]
        return any(resolved == root or resolved.is_relative_to(root) for root in roots)

    def _exact_file_operation_record_path(self, record: Any) -> Optional[Path]:
        """Resolve an already exact-authorized file record to a local file."""

        storage_status = getattr(record, "storage_status", None)
        if storage_status not in {None, "available", "legacy"}:
            return None

        storage_key = getattr(record, "storage_key", None)
        if storage_key:
            if self.owner_user_id is None:
                return None
            owner_prefix = build_user_key_prefix(
                self.owner_user_id,
                self.durable_storage_segments,
            )
            key = str(storage_key)
            if key != owner_prefix and not key.startswith(owner_prefix + "/"):
                return None

        storage_path = getattr(record, "storage_path", None)
        if storage_path:
            try:
                local_path = self._validate_regular_file_nofollow(
                    Path(str(storage_path)), str(getattr(record, "file_id", "file"))
                )
                if self._file_operation_path_in_authorized_storage(local_path):
                    return local_path
            except FileNotFoundError:
                pass
        if storage_key and storage_status == "available":
            from ..web.services.managed_file_ref import ManagedFileRef

            raw_materialized = ManagedFileRef(record).materialize()
            materialized = raw_materialized.resolve(strict=True)
            if self._file_operation_path_in_authorized_storage(materialized):
                return self._validate_regular_file_nofollow(
                    raw_materialized, str(getattr(record, "file_id", "file"))
                )
        return None

    def resolve_file_operation_path(self, file_path: str) -> Path:
        """Resolve one File Operation selector under its persisted task policy.

        Unmarked tasks delegate byte-for-byte to the shared resolver. Marked
        public tasks may use workspace-local files or external files backed by
        an exact owner/task record; the ambient external-directory allowlist is
        never sufficient authority on its own in that mode.
        """

        try:
            exact_scope = self.requires_exact_file_operation_scope()
        except Exception as exc:
            # Preserve File Operation's public not-found shape while failing
            # closed on malformed policy state or database infrastructure.
            logger.warning(
                "File Operation selector policy validation failed for workspace %s and task %s",
                self.id,
                self.db_task_id,
                exc_info=True,
            )
            raise FileNotFoundError(f"File not found: {file_path}") from exc
        if not exact_scope:
            return self.resolve_path_with_search(file_path)

        from ..web.models.database import release_db_connection_if_clean
        from ..web.models.uploaded_file import UploadedFile
        from .storage.manager import create_db_session

        if self.owner_user_id is None or self.db_task_id is None:
            raise FileNotFoundError(f"File not found: {file_path}")

        normalized = self._normalize_file_selector(file_path)

        # Exact-scope resolution is worker-dispatched by File Operation, so
        # its record lookup must not reuse a caller-owned session.
        db = create_db_session()
        try:
            candidate_ref = Path(normalized)
            if normalized and len(candidate_ref.parts) == 1 and "/" not in normalized:
                record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == normalized)
                    .first()
                )
                if record is not None and (
                    int(getattr(record, "user_id", 0) or 0) == self.owner_user_id
                    and int(getattr(record, "task_id", 0) or 0) == self.db_task_id
                ):
                    db.expunge(record)
                    if not release_db_connection_if_clean(db):
                        raise FileNotFoundError(f"File not found: {file_path}")
                    resolved_record = self._exact_file_operation_record_path(record)
                    if resolved_record is None:
                        raise FileNotFoundError(f"File not found: {file_path}")
                    return resolved_record

            if not release_db_connection_if_clean(db):
                raise FileNotFoundError(f"File not found: {file_path}")

            if candidate_ref.is_absolute():
                resolved = candidate_ref.resolve()
            else:
                try:
                    # Exact mode already performed its authorized file-id lookup.
                    # Restrict the fallback to workspace-local discovery so a
                    # foreign or taskless durable record cannot be materialized
                    # through the legacy owner-wide resolver.
                    resolved = self.resolve_path_with_search(
                        normalized,
                        resolve_file_ids=False,
                    ).resolve()
                except FileNotFoundError as exc:
                    raise FileNotFoundError(f"File not found: {file_path}") from exc

            workspace_root = self.workspace_dir.resolve()
            if resolved == workspace_root or resolved.is_relative_to(workspace_root):
                return resolved

            path_spellings = {str(resolved), str(candidate_ref)}
            records = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.user_id == self.owner_user_id,
                    UploadedFile.task_id == self.db_task_id,
                    UploadedFile.storage_path.in_(path_spellings),
                )
                .all()
            )
            db.expunge_all()
            if not release_db_connection_if_clean(db):
                raise FileNotFoundError(f"File not found: {file_path}")
            for record in records:
                record_path = Path(str(record.storage_path)).resolve()
                if record_path != resolved:
                    continue
                resolved_record = self._exact_file_operation_record_path(record)
                if resolved_record is not None:
                    return resolved_record
            raise FileNotFoundError(f"File not found: {file_path}")
        finally:
            db.close()

    def _prepare_sandbox_workspace_parent(self) -> bool:
        """Repair descendants of the first prepared sandbox-group ancestor.

        The mounted prefix can be a strict ancestor of an actor/task layout.
        Those intermediate directories may predate the non-root sandbox
        contract, so detecting only the immediate parent would silently skip
        their repair. Walk the already-resolved parent by descriptor and, once
        a gid/setgid anchor is found, prepare every remaining component.
        """
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        traversal_flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY
        traversal_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent = self.workspace_dir.parent.resolve()
        uploads_root = get_uploads_dir().expanduser().resolve()
        try:
            relative_parts = parent.relative_to(uploads_root).parts
        except ValueError:
            return False
        try:
            current_fd = os.open(uploads_root, traversal_flags)
            prepared_ancestor_found = _is_sandbox_prepared(os.fstat(current_fd))
        except OSError:
            return False
        try:
            for component in relative_parts:
                next_fd = -1
                try:
                    next_fd = os.open(component, traversal_flags, dir_fd=current_fd)
                    component_stat = os.fstat(next_fd)
                    if prepared_ancestor_found:
                        mutation_fd = os.open(
                            component, directory_flags, dir_fd=current_fd
                        )
                        try:
                            os.fchown(mutation_fd, -1, _SANDBOX_RUNTIME_GID)
                            os.fchmod(
                                mutation_fd,
                                stat.S_IMODE(component_stat.st_mode) | 0o2050,
                            )
                        finally:
                            os.close(mutation_fd)
                    elif _is_sandbox_prepared(component_stat):
                        prepared_ancestor_found = True
                except OSError:
                    if next_fd >= 0:
                        os.close(next_fd)
                    return False
                os.close(current_fd)
                current_fd = next_fd
            return prepared_ancestor_found
        finally:
            os.close(current_fd)

    def _ensure_directories(self) -> None:
        """Ensure task directories inherit writable sandbox-group access."""
        self.workspace_dir.parent.mkdir(parents=True, exist_ok=True)
        parent = self.workspace_dir.parent.resolve()
        uploads_root = get_uploads_dir().expanduser().resolve()
        parent_is_managed = parent == uploads_root or parent.is_relative_to(
            uploads_root
        )
        parent_is_prepared = parent_is_managed and _is_sandbox_prepared(parent.stat())
        repaired_parent_chain = False
        if not parent_is_prepared:
            repaired_parent_chain = self._prepare_sandbox_workspace_parent()
            parent_is_prepared = repaired_parent_chain
        if not parent_is_prepared:
            self._ensure_legacy_directories()
            return

        try:
            self._ensure_prepared_sandbox_directories(repaired_parent_chain)
        except OSError:
            logger.warning(
                "Failed to repair prepared sandbox workspace %s; using legacy layout",
                self.workspace_dir,
                exc_info=True,
            )
            self._ensure_legacy_directories()

    def _ensure_legacy_directories(self) -> None:
        self.workspace_dir.mkdir(exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    @staticmethod
    def _open_prepared_directory(
        parent_fd: int,
        name: str,
        flags: int,
        *,
        create_mode: int,
        final_mode: int,
    ) -> int:
        try:
            os.mkdir(name, mode=create_mode, dir_fd=parent_fd)
        except FileExistsError:
            pass
        directory_fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            os.fchown(directory_fd, -1, _SANDBOX_RUNTIME_GID)
            os.fchmod(directory_fd, final_mode)
        except BaseException:
            os.close(directory_fd)
            raise
        return directory_fd

    def _ensure_prepared_sandbox_directories(self, repaired_parent_chain: bool) -> None:
        """Create and repair one workspace beneath a trusted prepared parent."""

        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        parent = self.workspace_dir.parent.resolve()
        parent_open_flags = flags | getattr(os, "O_PATH", 0)
        parent_fd = os.open(parent, parent_open_flags)
        try:
            workspace_fd = self._open_prepared_directory(
                parent_fd,
                self.workspace_dir.name,
                flags,
                create_mode=0o750,
                final_mode=0o2750,
            )
        finally:
            os.close(parent_fd)

        try:
            for directory in (self.input_dir, self.output_dir, self.temp_dir):
                fd = self._open_prepared_directory(
                    workspace_fd,
                    directory.name,
                    flags,
                    create_mode=0o770,
                    final_mode=0o2770,
                )
                os.close(fd)
            temp_fd = os.open(self.temp_dir.name, flags, dir_fd=workspace_fd)
            try:
                self._ensure_internal_temp_directory(temp_fd, flags)
            finally:
                os.close(temp_fd)
            if self._migrate_sandbox_permissions:
                migration_key = str(self.workspace_dir)
                with _sandbox_permission_migrations_lock:
                    should_migrate = (
                        repaired_parent_chain
                        or migration_key not in _sandbox_permission_migrations_attempted
                    )
                    if should_migrate:
                        _sandbox_permission_migrations_attempted.add(migration_key)
                if should_migrate:
                    try:
                        self._migrate_existing_sandbox_permissions(
                            workspace_fd, flags, force=repaired_parent_chain
                        )
                    except BaseException:
                        with _sandbox_permission_migrations_lock:
                            _sandbox_permission_migrations_attempted.discard(
                                migration_key
                            )
                        raise
        finally:
            os.close(workspace_fd)

    def _ensure_internal_temp_directory(
        self, temp_fd: int, directory_flags: int
    ) -> None:
        """Create the reserved temp directory without trusting sandbox state.

        Legacy ``temp/`` layouts may have allowed the sandbox to replace the
        reserved name. Remove non-directory entries descriptor-relatively and
        retry, but do not let a process that continuously races the name make
        workspace construction or cleanup impossible.
        """
        recoverable_open_errors = {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}
        internal_flags = directory_flags | getattr(os, "O_NONBLOCK", 0)
        for _attempt in range(3):
            try:
                os.mkdir(_INTERNAL_TEMP_DIR_NAME, mode=0o750, dir_fd=temp_fd)
            except FileExistsError:
                pass
            try:
                internal_fd = os.open(
                    _INTERNAL_TEMP_DIR_NAME,
                    internal_flags,
                    dir_fd=temp_fd,
                )
            except OSError as exc:
                if exc.errno not in recoverable_open_errors:
                    raise
                if exc.errno == errno.ENOENT:
                    # The name vanished between mkdir/open. There is nothing
                    # safe to remove; retry the descriptor-relative lookup.
                    continue
                try:
                    os.unlink(_INTERNAL_TEMP_DIR_NAME, dir_fd=temp_fd)
                except FileNotFoundError:
                    pass
                except IsADirectoryError:
                    # The entry changed after the failed open; retry it as a
                    # directory rather than removing recursively.
                    pass
                continue
            try:
                os.fchown(internal_fd, -1, _SANDBOX_RUNTIME_GID)
                os.fchmod(internal_fd, 0o2750)
            finally:
                os.close(internal_fd)
            return
        logger.warning(
            "Sandbox kept replacing reserved temp entry %s; continuing without "
            "repair so workspace cleanup remains available",
            self.internal_temp_dir,
        )

    @staticmethod
    def _sandbox_compatible_mode(mode: int, *, directory: bool) -> int:
        owner_permissions = (stat.S_IMODE(mode) >> 6) & 0o7
        # Do not copy setuid/setgid bits from legacy files into a sandbox bind
        # mount. Directories need only setgid so new entries inherit the sandbox gid.
        special = stat.S_ISGID if directory else 0
        return special | (owner_permissions << 6) | (owner_permissions << 3)

    def _migrate_existing_sandbox_permissions(
        self, workspace_fd: int, directory_flags: int, *, force: bool = False
    ) -> None:
        """One-time, descriptor-safe migration for pre-nonroot task contents."""
        marker_flags = (
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            marker_fd = os.open(
                _SANDBOX_PERMISSION_MIGRATION_MARKER,
                marker_flags,
                dir_fd=workspace_fd,
            )
        except OSError:
            marker_fd = -1
        if marker_fd >= 0:
            try:
                marker_stat = os.fstat(marker_fd)
                if (
                    not force
                    and stat.S_ISREG(marker_stat.st_mode)
                    and marker_stat.st_uid == os.geteuid()
                ):
                    return
            finally:
                os.close(marker_fd)

        migration_succeeded = True
        for directory_name in ("input", "output", "temp"):
            directory_fd = os.open(directory_name, directory_flags, dir_fd=workspace_fd)
            try:
                migration_succeeded = (
                    self._migrate_sandbox_tree(
                        directory_fd,
                        skip_names=(
                            frozenset({_INTERNAL_TEMP_DIR_NAME})
                            if directory_name == "temp"
                            else None
                        ),
                    )
                    and migration_succeeded
                )
            finally:
                os.close(directory_fd)

        if not migration_succeeded:
            logger.warning(
                "Sandbox permission migration remains incomplete for workspace %s",
                self.workspace_dir,
            )
            return

        try:
            os.unlink(_SANDBOX_PERMISSION_MIGRATION_MARKER, dir_fd=workspace_fd)
        except FileNotFoundError:
            pass
        except OSError:
            # Never recursively remove an unexpected entry at this reserved
            # name. A directory or protected inode can safely leave migration
            # incomplete without breaking workspace reconstruction.
            logger.warning(
                "Cannot replace sandbox permission migration marker for %s",
                self.workspace_dir,
                exc_info=True,
            )
            return
        try:
            marker_fd = os.open(
                _SANDBOX_PERMISSION_MIGRATION_MARKER,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=workspace_fd,
            )
        except FileExistsError:
            # Another process completed the same idempotent migration first.
            return
        os.close(marker_fd)

    def _migrate_sandbox_tree(
        self, directory_fd: int, *, skip_names: Optional[frozenset[str]] = None
    ) -> bool:
        """Repair a tree iteratively without retaining one fd per directory."""
        succeeded = True
        pending: list[tuple[str, ...]] = [()]
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        while pending:
            relative_parts = pending.pop()
            try:
                current_fd = os.dup(directory_fd)
            except OSError:
                logger.warning(
                    "Failed to duplicate the legacy sandbox workspace root",
                    exc_info=True,
                )
                return False
            try:
                try:
                    for component in relative_parts:
                        next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                        os.close(current_fd)
                        current_fd = next_fd
                except FileNotFoundError:
                    # A concurrently removed directory no longer needs repair.
                    continue
                except OSError:
                    succeeded = False
                    continue
                try:
                    names = os.listdir(current_fd)
                except OSError:
                    logger.warning(
                        "Failed to list a legacy sandbox workspace directory",
                        exc_info=True,
                    )
                    succeeded = False
                    continue
                for name in names:
                    if not relative_parts and skip_names and name in skip_names:
                        continue
                    try:
                        entry_stat = os.stat(
                            name,
                            dir_fd=current_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        # A vanished entry no longer needs repair.
                        continue
                    except OSError:
                        succeeded = False
                        continue
                    is_directory = stat.S_ISDIR(entry_stat.st_mode)
                    is_regular = stat.S_ISREG(entry_stat.st_mode)
                    if not is_directory and not is_regular:
                        continue
                    if is_regular and entry_stat.st_nlink > 1:
                        continue
                    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
                    if is_directory:
                        flags |= os.O_DIRECTORY
                    try:
                        entry_fd = os.open(name, flags, dir_fd=current_fd)
                    except FileNotFoundError:
                        # A vanished entry no longer needs repair.
                        continue
                    except OSError:
                        succeeded = False
                        continue
                    try:
                        opened_stat = os.fstat(entry_fd)
                        if (
                            opened_stat.st_dev != entry_stat.st_dev
                            or opened_stat.st_ino != entry_stat.st_ino
                        ):
                            succeeded = False
                            continue
                        try:
                            os.fchown(entry_fd, -1, _SANDBOX_RUNTIME_GID)
                            os.fchmod(
                                entry_fd,
                                self._sandbox_compatible_mode(
                                    opened_stat.st_mode, directory=is_directory
                                ),
                            )
                        except OSError:
                            logger.warning(
                                "Failed to migrate sandbox workspace entry %s",
                                name,
                                exc_info=True,
                            )
                            succeeded = False
                            continue
                        if is_directory:
                            pending.append((*relative_parts, name))
                    finally:
                        os.close(entry_fd)
            finally:
                os.close(current_fd)
        return succeeded

    def _warn_if_reserved_output_name_is_taken(self) -> None:
        """Log once per process and workspace that the reserved output name is taken."""
        reserved = self.engine_owned_output_dir
        # Any entry at the name counts, a dangling symlink included (lexists).
        # The reservation goes by name, not by origin, so the engine's own
        # directory is announced too; once per process keeps that off the
        # happy path.
        if os.path.lexists(reserved):
            key = str(reserved)
            with _announced_reserved_names_lock:
                if key in _announced_reserved_names:
                    return
                _announced_reserved_names.add(key)
            logger.warning(
                "Workspace %s already has an entry at %s. That name is reserved "
                "for the engine's spilled tool results: nothing under it is "
                "listed as a deliverable and the workspace file tools refuse "
                "to write there.",
                self.id,
                reserved,
            )

    def get_allowed_dirs(self) -> List[str]:
        """Get list of allowed directories for this workspace"""
        dirs = [
            str(self.workspace_dir),
            str(self.input_dir),
            str(self.output_dir),
            str(self.temp_dir),
        ]
        # Add external allowed directories (e.g., user upload directories)
        dirs.extend([str(d) for d in self.allowed_external_dirs])
        return dirs

    def _resolve_allowed_absolute_path(self, path: Path) -> Path:
        """Resolve an absolute path after checking workspace allowlists."""
        return self.resolve_authorized_path(
            path,
            base_dir=self.workspace_dir,
            include_external_dirs=True,
        )

    def resolve_authorized_path(
        self,
        file_path: str | Path,
        *,
        base_dir: str | Path,
        include_external_dirs: bool = True,
    ) -> Path:
        """Resolve one path against an explicit base and workspace allowlists.

        Unlike :meth:`resolve_path`, this method never consults the Python
        process CWD. Callers must provide the filesystem base whose semantics
        they own. Shell-specific expansion, including ``~``, is caller-owned.
        """
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            raise ValueError("base_dir must be absolute")

        try:
            path = Path(file_path)
            candidate = path if path.is_absolute() else base_path / path
            abs_path = candidate.resolve()
            workspace_abs = self.workspace_dir.resolve()

            if abs_path.is_relative_to(workspace_abs):
                return abs_path

            if include_external_dirs:
                for allowed_dir in self.allowed_external_dirs:
                    allowed_abs = allowed_dir.resolve()
                    if abs_path.is_relative_to(allowed_abs):
                        logger.debug(
                            f"Accessing external file via allowed directory: {abs_path}"
                        )
                        return abs_path
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Failed to resolve path {file_path}") from exc

        allowed_dirs_str = ", ".join(
            [str(self.workspace_dir)]
            + (
                [str(d) for d in self.allowed_external_dirs]
                if include_external_dirs
                else []
            )
        )
        raise ValueError(
            f"Path {file_path} is outside allowed directories: {allowed_dirs_str}"
        )

    def _resolve_existing_cwd_relative_path(self, path: Path) -> Optional[Path]:
        """Resolve a CWD-relative path if it exists and is explicitly allowed."""
        if path.is_absolute():
            return None

        cwd_candidate = (Path.cwd() / path).resolve()
        if not cwd_candidate.exists():
            return None

        try:
            return self._resolve_allowed_absolute_path(cwd_candidate)
        except ValueError:
            return None

    def _escapes_before_symlinks(self, candidate: Path) -> bool:
        """True if ``candidate`` leaves every allowed root by path arithmetic alone.

        ``os.path.normpath`` collapses ``..`` lexically without following
        symlinks, so a ``..`` traversal is detected independently of what is on
        disk — distinguishing it from a lexically-contained candidate whose
        leaf only escapes once a symlink is resolved.
        """
        lexical = Path(os.path.normpath(candidate))
        roots = [self.workspace_dir, *self.allowed_external_dirs]
        for root in roots:
            root_lexical = Path(os.path.normpath(root))
            if lexical == root_lexical or lexical.is_relative_to(root_lexical):
                return False
        return True

    def _find_existing_candidate(
        self,
        search_dirs: List[Tuple[str, Path]],
        relative_path: Path,
    ) -> Optional[Path]:
        """Return the first existing ``dir_path / relative_path``, else None.

        Containment is validated (via ``_resolve_allowed_absolute_path``)
        BEFORE touching the filesystem, so the check runs before ``.exists()``
        and a traversal cannot become a ``ValueError``/``FileNotFoundError``
        cross-workspace existence oracle.

        Two kinds of escape raise ``ValueError`` from the containment check and
        are handled differently:

        * A ``..`` traversal escapes lexically — by path arithmetic alone,
          before any symlink is followed — and does so identically for every
          (sibling) search dir, so no in-workspace match is possible. Re-raise
          to fail closed uniformly; this is also the existence-oracle guard,
          which must raise regardless of whether the out-of-tree target exists.
        * A candidate that is lexically contained but escapes only when a
          symlinked leaf is resolved is per-entry: skip it and keep searching
          so a legitimate same-named file in a later dir stays reachable.
        """
        for _dir_name, dir_path in search_dirs:
            candidate = dir_path / relative_path
            try:
                resolved_candidate = self._resolve_allowed_absolute_path(candidate)
            except ValueError:
                if self._escapes_before_symlinks(candidate):
                    raise
                continue
            if resolved_candidate.exists():
                return resolved_candidate
        return None

    def resolve_path(self, file_path: str, default_dir: str = "output") -> Path:
        """
        Resolve a file path within the workspace or allowed external directories.

        Args:
            file_path: Relative or absolute file path
            default_dir: Default subdirectory if path is relative

        Returns:
            Resolved absolute path within workspace or allowed external directories

        Raises:
            ValueError: If path is outside both workspace and allowed external directories
        """
        path = Path(file_path)

        if path.is_absolute():
            # For absolute paths, verify it's within workspace or allowed external directories
            return self._resolve_allowed_absolute_path(path)
        else:
            cwd_relative = self._resolve_existing_cwd_relative_path(path)
            if cwd_relative is not None:
                return cwd_relative

            # For relative paths, resolve relative to the default directory,
            # then re-check containment. ``.resolve()`` collapses ``..``
            # segments, so a relative path such as ``../../other/file`` can
            # escape the workspace; returning it without re-verifying would let
            # a relative path reach outside the allowed subtree.
            if default_dir == "input":
                candidate = (self.input_dir / path).resolve()
            elif default_dir == "output":
                candidate = (self.output_dir / path).resolve()
            elif default_dir == "temp":
                candidate = (self.temp_dir / path).resolve()
            else:
                candidate = (self.workspace_dir / path).resolve()
            return self._resolve_allowed_absolute_path(candidate)

    @staticmethod
    def _normalize_filename_for_search(filename: str) -> str:
        """Normalize a filename for fuzzy matching.

        Applies the same normalization as the upload handler:
        spaces -> underscores, remove special chars like brackets.
        """
        name_part = Path(filename).stem
        extension = Path(filename).suffix

        name_part = unicodedata.normalize("NFC", name_part)
        name_part = re.sub(r"\s+", "_", name_part)
        name_part = re.sub(r"[^\w\u4e00-\u9fff\-_.]", "", name_part)
        name_part = re.sub(r"_+", "_", name_part)
        name_part = name_part.strip("_")

        if not name_part:
            return filename
        return name_part + extension

    def resolve_path_with_search(
        self,
        file_path: str,
        *,
        resolve_file_ids: bool = True,
        detached_file_ids: bool = False,
    ) -> Path:
        """
        Resolve a file path within the workspace with intelligent directory search.
        Searches for the file in input -> output -> temp -> workspace root order.
        For absolute paths, checks workspace and allowed external directories.

        Args:
            file_path: Relative or absolute file path
            resolve_file_ids: Whether a single path component may be a file id.
            detached_file_ids: Resolve file ids with a worker-owned DB session.

        Returns:
            Resolved absolute path within workspace or allowed external directories

        Raises:
            ValueError: If path is outside both workspace and allowed external directories
            FileNotFoundError: If relative path doesn't exist in any searched directory
        """
        normalized_input = self._normalize_file_selector(file_path)

        path = Path(normalized_input)

        file_id_candidate = normalized_input
        if (
            resolve_file_ids
            and file_id_candidate
            and len(path.parts) == 1
            and "/" not in file_id_candidate
        ):
            resolver = (
                self.resolve_file_id_detached
                if detached_file_ids
                else self.resolve_file_id
            )
            resolved_by_id = resolver(file_id_candidate)
            if resolved_by_id is not None:
                return resolved_by_id

        if path.is_absolute():
            # For absolute paths, verify it's within workspace or allowed external directories
            return self._resolve_allowed_absolute_path(path)
        else:
            cwd_relative = self._resolve_existing_cwd_relative_path(path)
            if cwd_relative is not None:
                return cwd_relative

            # For relative paths, search in priority order
            # Strip directory prefixes if present to avoid duplicates
            clean_path = path
            if len(path.parts) > 0:
                first_part = path.parts[0].lower()
                if first_part in ["input", "output", "temp"]:
                    # Strip the prefix to avoid duplicate directories
                    clean_path = Path(*path.parts[1:])

            # Search directories in priority order
            search_dirs = [
                ("input", self.input_dir),
                ("output", self.output_dir),
                ("temp", self.temp_dir),
            ]

            # 1. Try exact match first
            match = self._find_existing_candidate(search_dirs, clean_path)
            if match is not None:
                return match

            # 2. Try normalized filename (handles spaces, brackets, etc.)
            normalized_name = self._normalize_filename_for_search(clean_path.name)
            if normalized_name != clean_path.name:
                normalized_clean = clean_path.parent / normalized_name
                match = self._find_existing_candidate(search_dirs, normalized_clean)
                if match is not None:
                    logger.info(
                        f"File '{file_path}' matched via normalized name: '{normalized_name}'"
                    )
                    return match

            # 3. Try fuzzy match — also collect file list for error message
            request_stem = clean_path.stem.replace(" ", "").replace("_", "")
            request_suffix = clean_path.suffix.lower()
            all_files: List[str] = []
            for dir_name, dir_path in search_dirs:
                if not dir_path.exists():
                    continue
                for existing_file in dir_path.iterdir():
                    if not existing_file.is_file():
                        continue
                    all_files.append(f"{dir_name}/{existing_file.name}")
                    if (
                        request_suffix
                        and existing_file.suffix.lower() != request_suffix
                    ):
                        continue
                    existing_stem = existing_file.stem.replace(" ", "").replace("_", "")
                    if (
                        request_stem
                        and existing_stem
                        and (
                            request_stem in existing_stem
                            or existing_stem in request_stem
                        )
                    ):
                        # Containment gate, mirroring the exact-match and
                        # normalized-name branches. ``iterdir`` can surface a
                        # symlink that resolves outside the workspace; skip such
                        # a rogue candidate and keep searching rather than
                        # aborting, so a legitimate later match is still found.
                        resolved = existing_file.resolve()
                        try:
                            self._resolve_allowed_absolute_path(resolved)
                        except ValueError:
                            continue
                        logger.info(
                            f"File '{file_path}' fuzzy matched to: '{existing_file.name}'"
                        )
                        return resolved

            # 4. Not found — include available files in error message
            hint = ""
            if all_files:
                hint = f". Available files: {', '.join(all_files[:10])}"
            raise FileNotFoundError(
                f"File '{file_path}' not found in workspace directories "
                f"(tried: input, output, temp){hint}"
            )

    def get_output_files(self, include_subdirs: bool = True) -> List[Dict[str, Any]]:
        """
        Get all output files in the workspace.

        Reserved paths are left out on both branches, matching get_all_files:
        this list is what the user and a calling agent receive as the task's
        deliverables, and engine-owned files are not deliverables.

        Args:
            include_subdirs: Whether to include files in subdirectories

        Returns:
            List of file information dictionaries
        """
        output_files = []

        if include_subdirs:
            # Recursively scan output directory
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file() and not self._is_internal_workspace_path(
                    file_path
                ):
                    output_files.append(self._get_file_info(file_path, "output"))
        else:
            # Only scan top-level of output directory
            for file_path in self.output_dir.iterdir():
                if file_path.is_file() and not self._is_internal_workspace_path(
                    file_path
                ):
                    output_files.append(self._get_file_info(file_path, "output"))

        return output_files

    def get_all_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all files in workspace categorized by directory"""
        result: Dict[str, List[Dict[str, Any]]] = {
            "input": [],
            "output": [],
            "temp": [],
            "workspace": [],
        }

        # Scan input directory
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file() and not self._is_internal_workspace_path(file_path):
                result["input"].append(self._get_file_info(file_path, "input"))

        # Scan output directory
        for file_path in self.output_dir.rglob("*"):
            if file_path.is_file() and not self._is_internal_workspace_path(file_path):
                result["output"].append(self._get_file_info(file_path, "output"))

        # Scan temp directory
        for file_path in self.temp_dir.rglob("*"):
            if file_path.is_file() and not self._is_internal_workspace_path(file_path):
                result["temp"].append(self._get_file_info(file_path, "temp"))

        # Scan workspace root (excluding subdirs)
        for file_path in self.workspace_dir.iterdir():
            if (
                file_path.is_file()
                and not self._is_internal_workspace_path(file_path)
                and file_path.name not in ["input", "output", "temp"]
            ):
                result["workspace"].append(self._get_file_info(file_path, "workspace"))

        return result

    def _get_file_info(self, file_path: Path, location: str) -> Dict[str, Any]:
        """Get file information for a given path.

        Note: file_id will be None if the file is not registered in the database.
        Callers should handle this case appropriately.
        """
        stat = file_path.stat()
        # Get file_id from cache or DB (None if not registered)
        file_id = self.get_file_id_from_path(str(file_path))

        return {
            "file_id": file_id,
            "file_path": str(file_path),
            "relative_path": str(file_path.relative_to(self.workspace_dir)),
            "location": location,
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "is_readable": os.access(file_path, os.R_OK),
            "is_writable": os.access(file_path, os.W_OK),
        }

    def clean_temp_files(self) -> None:
        """Clean up temporary files"""
        for file_path in self.temp_dir.rglob("*"):
            if file_path.is_file():
                try:
                    registered_path = file_path.resolve()
                    file_path.unlink(missing_ok=True)
                    self.unregister_internal_file(str(registered_path))
                except (OSError, ValueError):
                    logger.debug(
                        "Could not remove temporary workspace file %s",
                        file_path,
                        exc_info=True,
                    )

    def cleanup(self) -> None:
        """Clean up the entire workspace.

        Tolerates a concurrent remover. Task deletion can race a cancelled
        turn's own runtime cleanup -- both unwind through ``remove_agent``, on
        separate worker threads, onto the same tree -- and the loser would
        otherwise surface a ``FileNotFoundError`` from the middle of the walk.
        The tree being gone is the outcome this method promises, so the caller
        learns nothing useful from which thread removed it, and reporting it as
        a failure would tell deletion the directory leaked when it did not.
        """
        self._forget_internal_files()
        with _sandbox_permission_migrations_lock:
            _sandbox_permission_migrations_attempted.discard(str(self.workspace_dir))
        if self.workspace_dir.exists():
            logger.info(f"Removing workspace directory: {self.workspace_dir}")
            try:
                shutil.rmtree(self.workspace_dir)
            except FileNotFoundError:
                logger.info(
                    f"Workspace directory already removed concurrently: "
                    f"{self.workspace_dir}"
                )
                return
            logger.info(f"Workspace directory removed: {self.workspace_dir}")

    def copy_to_workspace(self, source_path: str, target_subdir: str = "input") -> Path:
        """
        Copy a file to the workspace.

        Args:
            source_path: Source file path
            target_subdir: Target subdirectory (input, output, temp)

        Returns:
            Path to the copied file
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if target_subdir == "input":
            target_dir = self.input_dir
        elif target_subdir == "output":
            target_dir = self.output_dir
        elif target_subdir == "temp":
            target_dir = self.temp_dir
        else:
            target_dir = self.workspace_dir

        target_path = target_dir / source.name
        shutil.copy2(source, target_path)
        return target_path

    @contextmanager
    def auto_register_files(self) -> "Iterator[TaskWorkspace]":
        """
        Context manager to automatically register files created during execution.

        Usage:
            with workspace.auto_register_files():
                # All files created here will be automatically registered
                write_file("test.txt", "content")
                process_and_save_image("output.png")

        This is safer than relying on manual register_file() calls.
        """
        self._release_registration_session_if_clean(self.db_session)
        files_before = self._scan_all_files()

        try:
            yield self
        finally:
            self._release_registration_session_if_clean(self.db_session)
            files_after = self._scan_all_files()
            changed_files = files_after - files_before
            for file_path in files_after & files_before:
                with self._registration_lock:
                    cached_file_id = self._recently_registered_files.get(str(file_path))
                if cached_file_id or self._get_file_id_from_db(
                    file_path, self.db_session
                ):
                    changed_files.add(file_path)
            self._release_registration_session_if_clean(self.db_session)

            if changed_files:
                ordered_files = tuple(sorted(changed_files, key=str))
                # Each file owns its metadata transaction and compensation so
                # one failed artifact cannot roll back unrelated outputs. This
                # deliberately trades batch Session/commit efficiency for
                # per-file failure isolation.
                for file_path in ordered_files:
                    try:
                        file_id = self.register_file(
                            str(file_path),
                            db_session=self.db_session,
                        )
                        logger.debug(
                            "Auto-registered file: %s -> %s", file_path, file_id
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to auto-register workspace file %s: %s. "
                            "The file exists on disk but is not in the database "
                            "and requires backfill.",
                            file_path,
                            e,
                        )

    def _scan_all_files(self) -> set[Path]:
        """Scan all files in workspace and return as set."""
        files: set[Path] = set()
        if not self.workspace_dir.exists():
            return files

        for file_path in self.workspace_dir.rglob("*"):
            if file_path.is_file():
                # Skip hidden files and cache directories
                if any(part.startswith(".") for part in file_path.parts):
                    continue
                if (
                    "__pycache__" in file_path.parts
                    or "node_modules" in file_path.parts
                ):
                    continue
                if self._is_internal_workspace_path(file_path):
                    continue
                files.add(file_path)
        return files

    def get_file_id_from_path(self, file_path: str) -> Optional[str]:
        """Get file_id from file path using database or in-memory cache."""
        try:
            resolved_path = Path(file_path).resolve()
            resolved_str = str(resolved_path)

            # Check in-memory cache first (for files just registered)
            logger.debug(f"get_file_id_from_path: Looking for {resolved_str}")
            with self._registration_lock:
                cache_snapshot = dict(self._recently_registered_files)
            logger.debug(
                f"get_file_id_from_path: Cache has {len(cache_snapshot)} entries: {list(cache_snapshot.keys())}"
            )

            if resolved_str in cache_snapshot:
                logger.debug(
                    f"get_file_id_from_path: Found in cache: {cache_snapshot[resolved_str]}"
                )
                return cache_snapshot[resolved_str]

            internal_file_id = self._get_internal_file_id_from_path(resolved_path)
            if internal_file_id is not None:
                return internal_file_id

            # Also try the original path (not resolved)
            if file_path in cache_snapshot:
                logger.debug(
                    f"get_file_id_from_path: Found in cache with original path: {cache_snapshot[file_path]}"
                )
                return cache_snapshot[file_path]

            logger.debug("get_file_id_from_path: Not found in cache, checking DB")
            # Fall back to database query
            return self._get_file_id_from_db(resolved_path)
        except Exception as e:
            logger.warning(f"get_file_id_from_path: Exception: {e}")
            return None

    def list_all_user_files(
        self,
        include_workspace_files: bool = True,
        limit: int = DEFAULT_USER_FILE_LIST_LIMIT,
        offset: int = 0,
        *,
        _exact_task_scope: bool = False,
        openable_only: bool = False,
    ) -> Dict[str, Any]:
        """List the user's uploaded files, narrowed by scope and by the flags.

        Args:
            include_workspace_files: Whether to include current workspace files
            limit: Maximum number of files to return (default: 50)
            offset: Number of files to skip for pagination (default: 0)
            openable_only: Narrow to this task's own uploads before counting and
                slicing. Narrower than the read authority, which also grants
                task-less records; off by default so callers that carry their own
                authority keep the full listing.

        Returns:
            Dictionary with list of all user files with metadata including file_id,
            filename, storage_path, size, mime_type, etc.
        """
        import os
        from pathlib import Path

        from ..web.models.task import Task
        from ..web.models.uploaded_file import UploadedFile
        from .storage.manager import create_db_session

        # Marked File Operation uses the explicit database task identity.
        # Unmarked callers retain the historical workspace-id parsing behavior.
        task_id = self.db_task_id if _exact_task_scope else None
        user_id = None
        if task_id is None and not _exact_task_scope:
            try:
                task_id = int(self.id.split("_")[-1])
            except (ValueError, IndexError):
                task_id = None

        # Only open a database session when this workspace can actually map to a task.
        db = None
        should_close = False
        if task_id is not None:
            # Marked File Operation listing runs in a worker thread and must
            # use an operation-local session. Preserve bound-session behavior
            # only for historical, unmarked callers.
            if _exact_task_scope or self.db_session is None:
                db = create_db_session()
                should_close = True
            else:
                db = self.db_session

        try:
            # Try to get user_id from task if we have a valid task_id and db session
            if task_id and db is not None:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task is None and _exact_task_scope:
                    raise RuntimeError("Marked File Operation listing task is missing")
                if task:
                    user_id = task.user_id

            # Build file list - start with uploaded files if we have user_id
            result_files = []
            total_count = 0

            if user_id and db is not None:
                # Query uploaded files for this user. Marked File Operation
                # narrows before count/offset/limit so sibling rows cannot
                # distort pagination.
                query = db.query(UploadedFile).filter(UploadedFile.user_id == user_id)
                if _exact_task_scope:
                    if self.owner_user_id is None or int(user_id) != self.owner_user_id:
                        raise RuntimeError(
                            "Marked File Operation listing authority disagrees"
                        )
                    from sqlalchemy import and_, or_

                    owner_prefix = build_user_key_prefix(
                        self.owner_user_id,
                        self.durable_storage_segments,
                    )
                    storage_roots = [
                        self.base_dir.resolve(),
                        self.workspace_dir.resolve(),
                        *self.allowed_external_dirs,
                    ]
                    local_storage_filters = [
                        predicate
                        for root in storage_roots
                        for predicate in (
                            UploadedFile.storage_path == str(root),
                            UploadedFile.storage_path.startswith(
                                str(root) + os.sep, autoescape=True
                            ),
                        )
                    ]
                    query = query.filter(
                        UploadedFile.task_id == task_id,
                        or_(
                            UploadedFile.storage_status.is_(None),
                            UploadedFile.storage_status.in_(["available", "legacy"]),
                        ),
                        or_(
                            UploadedFile.storage_key.is_(None),
                            UploadedFile.storage_key == "",
                            UploadedFile.storage_key == owner_prefix,
                            UploadedFile.storage_key.startswith(
                                owner_prefix + "/", autoescape=True
                            ),
                        ),
                        or_(
                            and_(
                                UploadedFile.storage_key.is_not(None),
                                UploadedFile.storage_key != "",
                            ),
                            *local_storage_filters,
                        ),
                    )
                if openable_only and self.owner_user_id is not None:
                    # Before count and the slice, or later pages become unreachable.
                    # Task-bound only: deleting a task detaches its files, not drops them.
                    query = query.filter(UploadedFile.task_id == self.current_task_id)
                total_count = query.count()
                files = (
                    query.order_by(UploadedFile.id.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )

                # Build file list from database
                for file_record in files:
                    file_path = Path(file_record.storage_path)
                    has_local_file = file_path.exists()
                    has_durable_file = bool(
                        file_record.storage_key
                        and file_record.storage_status == "available"
                    )
                    if has_local_file or has_durable_file:
                        result_files.append(
                            {
                                "file_id": file_record.file_id,
                                "filename": file_record.filename,
                                "storage_path": file_record.storage_path,
                                "relative_path": str(file_path),
                                "size": file_record.file_size,
                                "mime_type": file_record.mime_type,
                                "task_id": file_record.task_id,
                                "uploaded_at": file_record.created_at.isoformat()
                                if file_record.created_at
                                else None,
                                "in_current_workspace": file_path.is_relative_to(
                                    self.workspace_dir
                                )
                                if has_local_file
                                else False,
                            }
                        )

            # Optionally include current workspace files (not yet uploaded)
            if include_workspace_files:
                try:
                    workspace_files_dict = self.get_all_files()
                    # Flatten the dict values to get all files
                    for category in ["input", "output", "temp", "workspace"]:
                        for file_info in workspace_files_dict.get(category, []):
                            file_path = file_info.get("file_path", "")
                            relative_path = file_info.get("relative_path", "")
                            if not file_path:
                                continue
                            is_already_listed = any(
                                f.get("storage_path") == file_path for f in result_files
                            )
                            if not is_already_listed:
                                stat = (
                                    os.stat(file_path)
                                    if os.path.exists(file_path)
                                    else None
                                )
                                if stat:
                                    result_files.append(
                                        {
                                            "file_id": None,
                                            "filename": Path(file_path).name,
                                            "storage_path": file_path,
                                            "relative_path": relative_path,
                                            "size": stat.st_size,
                                            "mime_type": "unknown",
                                            "task_id": task_id,
                                            "uploaded_at": None,
                                            "in_current_workspace": True,
                                            "is_unregistered": True,
                                        }
                                    )
                except Exception as e:
                    logger.warning(f"Failed to get workspace files: {e}")

            return {
                "success": True,
                "files": result_files,
                "total_count": total_count,
                "workspace_id": self.id,
                "user_id": user_id,
                "limit": limit,
                "offset": offset,
            }

        finally:
            if should_close and db is not None:
                db.close()

    def __enter__(self) -> "TaskWorkspace":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        # Don't automatically cleanup on exit, let the caller decide
        pass


# Simple workspace management functions
def create_workspace(
    id: str,
    base_dir: Optional[str] = None,
    allowed_external_dirs: Optional[List[str]] = None,
    db_task_id: Optional[int] = None,
    scope_segments: Sequence[str] = (),
) -> TaskWorkspace:
    """
    Create a new workspace for the given id.

    Args:
        id: Workspace identifier
        base_dir: Base directory for workspaces (uses default if None)
        allowed_external_dirs: List of allowed external directories
        scope_segments: ExecutionScope.workspace_segments the workspace
            runs under (base_dir is expected to already include them)

    Returns:
        TaskWorkspace instance
    """
    if base_dir is None:
        base_dir = str(get_uploads_dir())
    return TaskWorkspace(
        id,
        base_dir,
        allowed_external_dirs,
        db_task_id=db_task_id,
        scope_segments=scope_segments,
    )


def get_workspace_output_files(
    id: str, base_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get output files for a specific workspace.

    Args:
        id: Workspace identifier
        base_dir: Base directory for workspaces (uses default if None)

    Returns:
        List of output file information
    """
    if base_dir is None:
        base_dir = str(get_uploads_dir())
    workspace = TaskWorkspace(id, base_dir)
    return workspace.get_output_files()


class WorkspaceManager:
    """
    Manager for creating and accessing workspaces.

    Provides a centralized way to manage workspaces with proper cleanup
    and lifecycle management.
    """

    def __init__(self) -> None:
        self._workspaces: Dict[str, TaskWorkspace] = {}

    def get_or_create_workspace(
        self,
        base_dir: str,
        task_id: str,
        allowed_external_dirs: Optional[List[str]] = None,
        db_task_id: Optional[int] = None,
        scope_segments: Sequence[str] = (),
        durable_storage_segments: Sequence[str] | None = None,
    ) -> TaskWorkspace:
        """
        Get existing workspace or create new one.

        Args:
            base_dir: Base directory for workspaces
            task_id: Task/workspace identifier
            allowed_external_dirs: List of allowed external directories
            scope_segments: ExecutionScope.workspace_segments the workspace
                runs under (base_dir already includes them, so the cache key
                distinguishes scopes)

        Returns:
            TaskWorkspace instance
        """
        cache_key = f"{base_dir}:{task_id}"

        if cache_key not in self._workspaces:
            workspace = TaskWorkspace(
                task_id,
                base_dir,
                allowed_external_dirs,
                db_task_id=db_task_id,
                scope_segments=scope_segments,
                durable_storage_segments=durable_storage_segments,
            )
            self._workspaces[cache_key] = workspace
        elif db_task_id is not None and self._workspaces[cache_key].db_task_id is None:
            self._workspaces[cache_key].db_task_id = db_task_id
            self._workspaces[cache_key].current_task_id = db_task_id

        return self._workspaces[cache_key]

    def cleanup_workspace(self, base_dir: str, task_id: str) -> None:
        """
        Clean up a specific workspace.

        Args:
            base_dir: Base directory for workspaces
            task_id: Task/workspace identifier
        """
        cache_key = f"{base_dir}:{task_id}"

        if cache_key in self._workspaces:
            workspace = self._workspaces[cache_key]
            workspace.cleanup()
            del self._workspaces[cache_key]

    def cleanup_all_workspaces(self) -> None:
        """Clean up all managed workspaces."""
        for workspace in self._workspaces.values():
            workspace.cleanup()
        self._workspaces.clear()


# Global workspace instance, used in yaml server
_global_workspace: Optional[TaskWorkspace] = None


def init_global_workspace(
    id: str = "default", base_dir: str = "default_workspace"
) -> TaskWorkspace:
    """Initialize the global workspace."""
    global _global_workspace
    if _global_workspace is None:
        _global_workspace = TaskWorkspace(id, base_dir)
    return _global_workspace


def get_global_workspace() -> TaskWorkspace:
    """Get the global workspace instance."""
    global _global_workspace
    if _global_workspace is None:
        raise RuntimeError(
            "Global workspace not initialized. Call init_global_workspace() first."
        )
    return _global_workspace


class MockWorkspace:
    """
    Mock workspace that doesn't create actual directories on disk.

    This is used for scenarios like tool listing where we need a workspace
    object for tool creation but don't want to create directories on disk.

    All paths are virtual and won't be created. File operations will fail if
    attempted, which is fine for read-only operations like tool metadata retrieval.
    """

    def __init__(
        self,
        id: str = "_mock_",
        base_dir: str = "/mock/workspace",
    ):
        """
        Initialize mock workspace.

        Args:
            id: Workspace identifier
            base_dir: Virtual base directory (won't be created)
        """
        self.id = id
        self.base_dir = Path(base_dir)

        # Virtual paths (not created on disk)
        self.workspace_dir = self.base_dir / id
        self.input_dir = self.workspace_dir / "input"
        self.output_dir = self.workspace_dir / "output"
        self.temp_dir = self.workspace_dir / "temp"

        # No external allowed directories for mock
        self.allowed_external_dirs: List[Path] = []

        logger.debug(
            f"Created mock workspace: {self.workspace_dir} (not created on disk)"
        )

    def get_allowed_dirs(self) -> List[str]:
        """Get list of allowed directories for this workspace (virtual paths)."""
        return [
            str(self.workspace_dir),
            str(self.input_dir),
            str(self.output_dir),
            str(self.temp_dir),
        ]

    def resolve_path(self, file_path: str, default_dir: str = "output") -> Path:
        """
        Resolve a file path within the workspace.

        For mock workspace, this returns a virtual path without creating it.

        Args:
            file_path: Relative or absolute file path
            default_dir: Default subdirectory if path is relative

        Returns:
            Resolved absolute path (virtual, not created)
        """
        path = Path(file_path)

        # If absolute path, just return it (for mock workspace)
        if path.is_absolute():
            return path

        # Relative path - resolve to default directory
        if default_dir == "input":
            return self.input_dir / file_path
        elif default_dir == "output":
            return self.output_dir / file_path
        elif default_dir == "temp":
            return self.temp_dir / file_path
        else:
            return self.workspace_dir / file_path

    def resolve_write_path(self, file_path: str, default_dir: str = "output") -> Path:
        """Resolve a write target the same way as :meth:`resolve_path`.

        The mock never writes to disk, so there is no engine-owned subtree to
        refuse; the method exists so that tools built against this workspace
        for listing keep the write-side entry point they call.
        """

        return self.resolve_path(file_path, default_dir)

    def is_engine_owned_path(self, file_path: Path) -> bool:
        """Answer False: a workspace that never writes to disk owns nothing."""

        return False

    def refuse_engine_owned_write(self, resolved_path: Path, requested: str) -> Path:
        """Return the target unchanged; see :meth:`is_engine_owned_path`."""

        return resolved_path

    def register_file(self, file_path: str, file_id: Optional[str] = None) -> str:
        """
        Mock register_file - returns a UUID without creating database record.

        Args:
            file_path: Virtual file path
            file_id: Optional file ID

        Returns:
            A UUID string
        """
        from uuid import uuid4

        return str(file_id).strip() if file_id else str(uuid4())

    def __repr__(self) -> str:
        return f"MockWorkspace(id='{self.id}', path='{self.workspace_dir}')"
