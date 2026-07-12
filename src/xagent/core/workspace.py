"""
Agent-aware workspace management for xagent

This module provides workspace management that supports multiple concurrent agents,
ensuring that each agent has its own isolated workspace context.
"""

import contextvars
import logging
import mimetypes
import os
import re
import shutil
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

from ..config import get_uploads_dir
from .execution_scope import validate_scope_component

logger = logging.getLogger(__name__)

DEFAULT_USER_FILE_LIST_LIMIT = 50

# Context variable for auto-registration mode
_auto_register = contextvars.ContextVar("_auto_register", default=False)


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
        if base_dir is None:
            base_dir = str(get_uploads_dir())
        self.base_dir = (
            Path(base_dir).expanduser().resolve()
        )  # Resolve base_dir to absolute path for consistent workspace reconstruction
        self.db_session = None  # Optional database session for file registration
        self._recently_registered_files: Dict[str, str] = {}  # path -> file_id mapping
        self._file_id_to_path: Dict[str, Path] = {}  # file_id -> path reverse mapping
        self.owner_user_id: Optional[int] = None
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

        # Create directory structure
        self._ensure_directories()

    def register_file(
        self, file_path: str, file_id: Optional[str] = None, db_session: Any = None
    ) -> str:
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

        # Check if file already exists in database
        resolved_db_session = db_session or self.db_session
        cached_file_id = self._recently_registered_files.get(str(resolved_path))
        if cached_file_id:
            self._sync_existing_file_record(
                cached_file_id, resolved_path, resolved_db_session
            )
            self._remember_file_registration(cached_file_id, resolved_path)
            return cached_file_id

        existing_file_id = self._get_file_id_from_db(resolved_path, resolved_db_session)
        if existing_file_id:
            self._sync_existing_file_record(
                existing_file_id, resolved_path, resolved_db_session
            )
            self._remember_file_registration(existing_file_id, resolved_path)
            return existing_file_id

        # Generate new file_id if not provided
        final_file_id = str(file_id).strip() if file_id else ""
        if not final_file_id:
            final_file_id = str(uuid4())

        # Create database record
        self._create_file_record(final_file_id, resolved_path, db_session)
        self._remember_file_registration(final_file_id, resolved_path)

        return final_file_id

    def _remember_file_registration(self, file_id: str, file_path: Path) -> None:
        path_str = str(file_path)
        resolved_str = str(file_path.resolve())
        self._recently_registered_files[path_str] = file_id
        self._recently_registered_files[resolved_str] = file_id
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

    @staticmethod
    def _storage_path_record(
        db: Any, storage_path: Path, file_id: Optional[str] = None
    ) -> Any:
        from ..web.models.uploaded_file import UploadedFile

        query = db.query(UploadedFile).filter(
            UploadedFile.storage_path == str(storage_path)
        )
        record = query.first()
        if record is None or file_id is None or str(record.file_id) == str(file_id):
            return record
        return record

    @classmethod
    def _unique_registration_path(
        cls, target_path: Path, source_path: Path, db: Any, file_id: str
    ) -> Path:
        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent
        candidate = target_path
        index = 1

        source_resolved = source_path.resolve()
        while True:
            try:
                candidate_resolved = candidate.resolve()
            except OSError:
                candidate_resolved = candidate.absolute()
            record = cls._storage_path_record(db, candidate, file_id)
            same_record = record is not None and str(record.file_id) == str(file_id)
            record_conflicts = record is not None and str(record.file_id) != str(
                file_id
            )
            path_conflicts = (
                candidate.exists()
                and candidate_resolved != source_resolved
                and not same_record
            )
            if not record_conflicts and not path_conflicts:
                return candidate
            candidate = parent / f"{stem}_{index}{suffix}"
            index += 1

    def _canonicalize_delegated_output_registration(
        self,
        *,
        db: Any,
        file_id: str,
        file_path: Path,
        task_id: int,
        user_id: int,
        relative_path: str,
        category: str,
    ) -> tuple[Path, str, str]:
        if category != "output" or not self._is_delegated_db_task_workspace(task_id):
            return file_path, relative_path, category

        relative_parts = Path(relative_path).parts
        output_parts = [
            part for part in relative_parts[1:] if part not in ("", ".", "..")
        ]
        if not output_parts:
            output_parts = [file_path.name]

        canonical_relative_path = Path("output", *output_parts).as_posix()
        task_root = self._user_workspace_base_dir(user_id) / f"web_task_{task_id}"
        output_root = (task_root / "output").resolve()
        canonical_path = (task_root / canonical_relative_path).resolve()
        try:
            canonical_path.relative_to(output_root)
        except ValueError:
            canonical_relative_path = Path("output", file_path.name).as_posix()
            canonical_path = (task_root / canonical_relative_path).resolve()

        canonical_path = self._unique_registration_path(
            canonical_path, file_path, db, file_id
        )
        canonical_relative_path = canonical_path.relative_to(task_root).as_posix()
        if canonical_path.resolve() != file_path.resolve():
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, canonical_path)

        return canonical_path, canonical_relative_path, "output"

    def _create_file_record(
        self, file_id: str, file_path: Path, db_session: Any = None
    ) -> None:
        """Create UploadedFile record in database"""
        from .storage.manager import create_db_session

        # Use provided session or create temporary one
        if db_session:
            db = db_session
            should_close = False
        else:
            db = self.db_session if self.db_session else create_db_session()
            should_close = self.db_session is None

        try:
            from ..web.models.task import Task
            from ..web.models.uploaded_file import UploadedFile

            # Check if record already exists
            existing = (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
            )
            if existing:
                return

            task_id = self.db_task_id
            if task_id is None:
                # Extract task_id from workspace id (e.g., 'web_task_265' -> 265).
                # Delegated agent workspaces use non-DB ids such as
                # 'agent_2_abcd1234', so callers should pass db_task_id explicitly.
                try:
                    task_id = int(self.id.split("_")[-1])
                except (ValueError, IndexError):
                    logger.debug(
                        f"Skipping database registration for workspace '{self.id}' "
                        f"without db_task_id, file_id={file_id}"
                    )
                    return

            # Get user_id from task
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.warning(f"Task {task_id} not found, cannot create file record")
                return

            # Guess MIME type
            mime_type, _ = mimetypes.guess_type(file_path.name)
            if not mime_type:
                mime_type = "application/octet-stream"

            try:
                relative_path = str(file_path.relative_to(self.workspace_dir))
            except ValueError:
                relative_path = file_path.name
            category = relative_path.split("/", 1)[0] if relative_path else "workspace"

            file_path, relative_path, category = (
                self._canonicalize_delegated_output_registration(
                    db=db,
                    file_id=file_id,
                    file_path=file_path,
                    task_id=int(task_id),
                    user_id=int(task.user_id),
                    relative_path=relative_path,
                    category=category,
                )
            )

            # Import lazily: module-level file_storage imports would pull
            # fsspec into sandboxed executions that ship minimal deps.
            from ..web.services.uploaded_file_store import UploadedFileStore
            from .file_storage.keys import build_task_output_storage_key

            UploadedFileStore(db).create_from_local_path(
                local_path=file_path,
                user_id=int(task.user_id),
                file_id=file_id,
                task_id=task_id,
                filename=file_path.name,
                storage_key=build_task_output_storage_key(
                    int(task.user_id),
                    task_id,
                    file_id,
                    relative_path,
                    scope_segments=self.scope_segments,
                ),
                workspace_relative_path=relative_path,
                workspace_category=category,
                mime_type=mime_type,
            )
            if should_close:
                db.commit()
            else:
                db.flush()
            logger.info(f"Created file record: file_id={file_id}, task_id={task_id}")
        except Exception as e:
            logger.error(f"Failed to create file record: {e}")
            if should_close:
                db.rollback()
            raise  # Re-raise so caller knows registration failed
        finally:
            if should_close and db is not None:
                db.close()

    def _sync_existing_file_record(
        self, file_id: str, file_path: Path, db_session: Any = None
    ) -> None:
        """Sync an existing UploadedFile row with current local bytes."""
        from .file_storage.keys import build_task_output_storage_key
        from .storage.manager import create_db_session

        if db_session:
            db = db_session
            should_close = False
        else:
            db = self.db_session if self.db_session else create_db_session()
            should_close = self.db_session is None

        try:
            from ..web.models.uploaded_file import UploadedFile
            from ..web.services.uploaded_file_store import UploadedFileStore

            record = (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
            )
            if record is None:
                return

            mime_type, _ = mimetypes.guess_type(file_path.name)
            if not mime_type:
                mime_type = "application/octet-stream"

            task_id = getattr(record, "task_id", None)
            user_id = int(getattr(record, "user_id"))

            try:
                relative_path = str(file_path.relative_to(self.workspace_dir))
            except ValueError:
                UploadedFileStore(db).sync_existing(
                    record,
                    storage_key=getattr(record, "storage_key", None),
                    mime_type=getattr(record, "mime_type", None) or mime_type,
                )
                if should_close:
                    db.commit()
                else:
                    db.flush()
                return

            if self.db_task_id is not None:
                from ..web.models.task import Task

                task = db.query(Task).filter(Task.id == self.db_task_id).first()
                if not task:
                    logger.warning(
                        f"Task {self.db_task_id} not found, cannot rebind file record"
                    )
                    return
                task_user_id = int(task.user_id)
                if user_id != task_user_id:
                    logger.warning(
                        "Skipping file record rebind across users: "
                        f"file_id={file_id}, record_user_id={user_id}, "
                        f"task_user_id={task_user_id}"
                    )
                    return
                task_id = self.db_task_id
                user_id = task_user_id

            category = relative_path.split("/", 1)[0] if relative_path else "workspace"
            file_path, relative_path, category = (
                self._canonicalize_delegated_output_registration(
                    db=db,
                    file_id=file_id,
                    file_path=file_path,
                    task_id=int(task_id) if task_id is not None else 0,
                    user_id=user_id,
                    relative_path=relative_path,
                    category=category,
                )
            )
            storage_key = build_task_output_storage_key(
                user_id,
                int(task_id) if task_id is not None else 0,
                file_id,
                relative_path,
                scope_segments=self.scope_segments,
            )
            if task_id is None:
                storage_key = getattr(record, "storage_key", None) or storage_key

            record.user_id = user_id
            record.task_id = int(task_id) if task_id is not None else None
            record.filename = file_path.name
            record.storage_path = str(file_path)
            record.file_size = file_path.stat().st_size
            record.mime_type = mime_type
            record.workspace_relative_path = relative_path
            record.workspace_category = category
            UploadedFileStore(db).sync_existing(
                record,
                storage_key=storage_key,
                mime_type=mime_type,
            )
            if should_close:
                db.commit()
            else:
                db.flush()
        except Exception as e:
            logger.error(f"Failed to sync existing file record: {e}")
            if should_close:
                db.rollback()
            raise
        finally:
            if should_close and db is not None:
                db.close()

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

    def _file_record_allowed_for_workspace(
        self, record: Any, path: Optional[Path] = None
    ) -> bool:
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
        file_id = str(file_id).strip()
        if not file_id:
            return None

        # Check in-memory cache first
        if file_id in self._file_id_to_path:
            cached_path = self._file_id_to_path[file_id]
            if cached_path.exists():
                logger.debug(
                    f"resolve_file_id: Found in cache: {file_id} -> {cached_path}"
                )
                return cached_path
            else:
                logger.warning(
                    f"resolve_file_id: Cached path doesn't exist: {cached_path}"
                )
                # Remove stale cache entry
                del self._file_id_to_path[file_id]

        # Query from database
        from .storage.manager import create_db_session

        try:
            from ..web.models.uploaded_file import UploadedFile

            if self.db_session is not None:
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
                    resolved_path = Path(record.storage_path)
                    if resolved_path.exists() and resolved_path.is_file():
                        if not self._file_record_allowed_for_workspace(
                            record, resolved_path
                        ):
                            logger.warning(
                                "Rejected file_id outside workspace scope: %s",
                                file_id,
                            )
                            return None
                        return resolved_path
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

                    return ManagedFileRef(record).materialize()
                return None
            finally:
                if should_close:
                    db.close()
        except Exception as e:
            logger.warning(f"Failed to resolve file_id from database: {e}")
            return None

    def _ensure_directories(self) -> None:
        """Ensure all workspace directories exist"""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

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
        abs_path = path.resolve()
        workspace_abs = self.workspace_dir.resolve()
        if abs_path == workspace_abs or abs_path.is_relative_to(workspace_abs):
            return abs_path

        for allowed_dir in self.allowed_external_dirs:
            allowed_abs = allowed_dir.resolve()
            if abs_path == allowed_abs or abs_path.is_relative_to(allowed_abs):
                logger.debug(
                    f"Accessing external file via allowed directory: {abs_path}"
                )
                return abs_path

        allowed_dirs_str = ", ".join(
            [str(self.workspace_dir)] + [str(d) for d in self.allowed_external_dirs]
        )
        raise ValueError(
            f"Path {path} is outside allowed directories: {allowed_dirs_str}"
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

    def resolve_path_with_search(self, file_path: str) -> Path:
        """
        Resolve a file path within the workspace with intelligent directory search.
        Searches for the file in input -> output -> temp -> workspace root order.
        For absolute paths, checks workspace and allowed external directories.

        Args:
            file_path: Relative or absolute file path

        Returns:
            Resolved absolute path within workspace or allowed external directories

        Raises:
            ValueError: If path is outside both workspace and allowed external directories
            FileNotFoundError: If relative path doesn't exist in any searched directory
        """
        normalized_input = file_path.strip()
        if normalized_input.startswith("file:") and not normalized_input.startswith(
            "file://"
        ):
            normalized_input = normalized_input[5:].strip()

        path = Path(normalized_input)

        file_id_candidate = normalized_input
        if file_id_candidate and len(path.parts) == 1 and "/" not in file_id_candidate:
            resolved_by_id = self.resolve_file_id(file_id_candidate)
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
                        f"File '{file_path}' matched via normalized name: "
                        f"'{normalized_name}'"
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
                            f"File '{file_path}' fuzzy matched to: "
                            f"'{existing_file.name}'"
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

        Args:
            include_subdirs: Whether to include files in subdirectories

        Returns:
            List of file information dictionaries
        """
        output_files = []

        if include_subdirs:
            # Recursively scan output directory
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    output_files.append(self._get_file_info(file_path, "output"))
        else:
            # Only scan top-level of output directory
            for file_path in self.output_dir.iterdir():
                if file_path.is_file():
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
            if file_path.is_file():
                result["input"].append(self._get_file_info(file_path, "input"))

        # Scan output directory
        for file_path in self.output_dir.rglob("*"):
            if file_path.is_file():
                result["output"].append(self._get_file_info(file_path, "output"))

        # Scan temp directory
        for file_path in self.temp_dir.rglob("*"):
            if file_path.is_file():
                result["temp"].append(self._get_file_info(file_path, "temp"))

        # Scan workspace root (excluding subdirs)
        for file_path in self.workspace_dir.iterdir():
            if file_path.is_file() and file_path.name not in [
                "input",
                "output",
                "temp",
            ]:
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
                    file_path.unlink()
                except OSError:
                    pass

    def cleanup(self) -> None:
        """Clean up the entire workspace"""
        if self.workspace_dir.exists():
            logger.info(f"Removing workspace directory: {self.workspace_dir}")
            shutil.rmtree(self.workspace_dir)
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
        # Scan files before operation
        files_before = self._scan_all_files()

        try:
            yield self
        finally:
            # Scan files after operation and register new/modified files.
            files_after = self._scan_all_files()
            changed_files = files_after - files_before
            changed_files.update(
                file_path
                for file_path in files_after & files_before
                if self._get_file_id_from_db(file_path, self.db_session) is not None
            )

            for file_path in changed_files:
                try:
                    file_id = self.register_file(str(file_path))
                    # Store path -> file_id mapping
                    path_str = str(file_path)
                    resolved_str = str(file_path.resolve())
                    self._recently_registered_files[path_str] = file_id
                    self._recently_registered_files[resolved_str] = file_id
                    # Store file_id -> path reverse mapping
                    self._file_id_to_path[file_id] = file_path
                    logger.debug(f"Auto-registered file: {file_path} -> {file_id}")
                except Exception as e:
                    # Don't generate fake file_id - file will need to be backfilled later
                    logger.error(
                        f"Failed to auto-register file {file_path}: {e}. "
                        f"File exists on disk but is not in database - will require backfill."
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
                files.add(file_path)
        return files

    def get_file_id_from_path(self, file_path: str) -> Optional[str]:
        """Get file_id from file path using database or in-memory cache."""
        try:
            resolved_path = Path(file_path).resolve()
            resolved_str = str(resolved_path)

            # Check in-memory cache first (for files just registered)
            logger.debug(f"get_file_id_from_path: Looking for {resolved_str}")
            logger.debug(
                f"get_file_id_from_path: Cache has {len(self._recently_registered_files)} entries: {list(self._recently_registered_files.keys())}"
            )

            if resolved_str in self._recently_registered_files:
                logger.debug(
                    f"get_file_id_from_path: Found in cache: {self._recently_registered_files[resolved_str]}"
                )
                return self._recently_registered_files[resolved_str]

            # Also try the original path (not resolved)
            if file_path in self._recently_registered_files:
                logger.debug(
                    f"get_file_id_from_path: Found in cache with original path: {self._recently_registered_files[file_path]}"
                )
                return self._recently_registered_files[file_path]

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
    ) -> Dict[str, Any]:
        """List all user files across all workspaces and uploaded files.

        Args:
            include_workspace_files: Whether to include current workspace files
            limit: Maximum number of files to return (default: 50)
            offset: Number of files to skip for pagination (default: 0)

        Returns:
            Dictionary with list of all user files with metadata including file_id,
            filename, storage_path, size, mime_type, etc.
        """
        import os
        from pathlib import Path

        from ..web.models.task import Task
        from ..web.models.uploaded_file import UploadedFile
        from .storage.manager import create_db_session

        # Extract user_id from workspace id (e.g., 'web_task_265' -> 265)
        task_id = None
        user_id = None
        try:
            task_id = int(self.id.split("_")[-1])
        except (ValueError, IndexError):
            task_id = None

        # Only open a database session when this workspace can actually map to a task.
        db = None
        should_close = False
        if task_id is not None:
            db = self.db_session if self.db_session else create_db_session()
            should_close = self.db_session is None

        try:
            # Try to get user_id from task if we have a valid task_id and db session
            if task_id and db is not None:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task:
                    user_id = task.user_id

            # Build file list - start with uploaded files if we have user_id
            result_files = []
            total_count = 0

            if user_id and db is not None:
                # Query uploaded files for this user
                query = db.query(UploadedFile).filter(UploadedFile.user_id == user_id)
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
