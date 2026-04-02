"""
Agent-aware workspace management for xagent

This module provides workspace management that supports multiple concurrent agents,
ensuring that each agent has its own isolated workspace context.
"""

import contextvars
import logging
import os
import re
import shutil
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Context variable for auto-registration mode
_auto_register = contextvars.ContextVar("_auto_register", default=False)


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
        base_dir: str = "uploads",
        allowed_external_dirs: Optional[List[str]] = None,
    ):
        self.id = id
        self.base_dir = Path(base_dir)
        self.db_session = None  # Optional database session for file registration
        self._recently_registered_files: Dict[str, str] = {}  # path -> file_id mapping
        self._file_id_to_path: Dict[str, Path] = {}  # file_id -> path reverse mapping

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
        existing_file_id = self._get_file_id_from_db(resolved_path, db_session)
        if existing_file_id:
            return existing_file_id

        # Generate new file_id if not provided
        final_file_id = str(file_id).strip() if file_id else ""
        if not final_file_id:
            final_file_id = str(uuid4())

        # Create database record
        self._create_file_record(final_file_id, resolved_path, db_session)

        return final_file_id

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

            # Extract task_id from workspace id (e.g., 'web_task_265' -> 265)
            # Handle test environment workspaces (e.g., 'test_task')
            try:
                task_id = int(self.id.split("_")[-1])
            except (ValueError, IndexError):
                # Not a valid task ID, likely a test workspace
                # Skip database registration in test environments
                logger.debug(
                    f"Skipping database registration for test workspace '{self.id}', file_id={file_id}"
                )
                return

            # Get user_id from task
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.warning(f"Task {task_id} not found, cannot create file record")
                return

            # Guess MIME type
            import mimetypes

            mime_type, _ = mimetypes.guess_type(file_path.name)
            if not mime_type:
                mime_type = "application/octet-stream"

            # Create file record
            file_record = UploadedFile(
                file_id=file_id,
                user_id=task.user_id,
                task_id=task_id,
                filename=file_path.name,
                storage_path=str(file_path),
                mime_type=mime_type,
                file_size=file_path.stat().st_size,
            )
            db.add(file_record)
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
            if should_close:
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

            db = create_db_session()
            try:
                record = (
                    db.query(UploadedFile)
                    .filter(UploadedFile.file_id == file_id)
                    .first()
                )
                if record and record.storage_path:
                    resolved_path = Path(record.storage_path)
                    if resolved_path.exists() and resolved_path.is_file():
                        return resolved_path
                return None
            finally:
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
            abs_path = path.resolve()

            # Check if within workspace
            workspace_abs = self.workspace_dir.resolve()
            if abs_path == workspace_abs or abs_path.is_relative_to(workspace_abs):
                return abs_path

            # Check if within any allowed external directory
            for allowed_dir in self.allowed_external_dirs:
                if abs_path.is_relative_to(allowed_dir):
                    logger.debug(
                        f"Accessing external file via allowed directory: {abs_path}"
                    )
                    return abs_path

            # Not in any allowed directory
            allowed_dirs_str = ", ".join(
                [str(self.workspace_dir)] + [str(d) for d in self.allowed_external_dirs]
            )
            raise ValueError(
                f"Path {file_path} is outside allowed directories: {allowed_dirs_str}"
            )
        else:
            # For relative paths, resolve relative to default directory
            if default_dir == "input":
                return (self.input_dir / path).resolve()
            elif default_dir == "output":
                return (self.output_dir / path).resolve()
            elif default_dir == "temp":
                return (self.temp_dir / path).resolve()
            else:
                return (self.workspace_dir / path).resolve()

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
            abs_path = path.resolve()

            # Check if within workspace
            workspace_abs = self.workspace_dir.resolve()
            if abs_path == workspace_abs or abs_path.is_relative_to(workspace_abs):
                return abs_path

            # Check if within any allowed external directory
            for allowed_dir in self.allowed_external_dirs:
                if abs_path.is_relative_to(allowed_dir):
                    logger.debug(
                        f"Accessing external file via allowed directory: {abs_path}"
                    )
                    return abs_path

            # Not in any allowed directory
            allowed_dirs_str = ", ".join(
                [str(self.workspace_dir)] + [str(d) for d in self.allowed_external_dirs]
            )
            raise ValueError(
                f"Path {file_path} is outside allowed directories: {allowed_dirs_str}"
            )
        else:
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
            for _dir_name, dir_path in search_dirs:
                candidate = dir_path / clean_path
                if candidate.exists():
                    return candidate.resolve()

            # 2. Try normalized filename (handles spaces, brackets, etc.)
            normalized_name = self._normalize_filename_for_search(clean_path.name)
            if normalized_name != clean_path.name:
                normalized_clean = clean_path.parent / normalized_name
                for _dir_name, dir_path in search_dirs:
                    candidate = dir_path / normalized_clean
                    if candidate.exists():
                        logger.info(
                            f"File '{file_path}' matched via normalized name: "
                            f"'{normalized_name}'"
                        )
                        return candidate.resolve()

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
                        logger.info(
                            f"File '{file_path}' fuzzy matched to: "
                            f"'{existing_file.name}'"
                        )
                        return existing_file.resolve()

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
            # Scan files after operation and register new ones
            files_after = self._scan_all_files()
            new_files = files_after - files_before

            for file_path in new_files:
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
        limit: int = 1000,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List all user files across all workspaces and uploaded files.

        Args:
            include_workspace_files: Whether to include current workspace files
            limit: Maximum number of files to return (default: 1000)
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

        # Use existing session or create temporary one
        db = self.db_session if self.db_session else create_db_session()
        should_close = self.db_session is None

        try:
            # Extract user_id from workspace id (e.g., 'web_task_265' -> 265)
            task_id = None
            user_id = None
            try:
                task_id = int(self.id.split("_")[-1])
            except (ValueError, IndexError):
                task_id = None

            # Try to get user_id from task if we have a valid task_id
            if task_id:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task:
                    user_id = task.user_id

            # Build file list - start with uploaded files if we have user_id
            result_files = []
            total_count = 0

            if user_id:
                # Query uploaded files for this user
                query = db.query(UploadedFile).filter(UploadedFile.user_id == user_id)
                total_count = query.count()
                files = query.offset(offset).limit(limit).all()

                # Build file list from database
                for file_record in files:
                    file_path = Path(file_record.storage_path)
                    if file_path.exists():
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
                                if file_path.exists()
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
            if should_close:
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
    base_dir: str = "uploads",
    allowed_external_dirs: Optional[List[str]] = None,
) -> TaskWorkspace:
    """
    Create a new workspace for the given id.

    Args:
        id: Workspace identifier
        base_dir: Base directory for workspaces
        allowed_external_dirs: List of allowed external directories

    Returns:
        TaskWorkspace instance
    """
    return TaskWorkspace(id, base_dir, allowed_external_dirs)


def get_workspace_output_files(
    id: str, base_dir: str = "uploads"
) -> List[Dict[str, Any]]:
    """
    Get output files for a specific workspace.

    Args:
        id: Workspace identifier
        base_dir: Base directory for workspaces

    Returns:
        List of output file information
    """
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
    ) -> TaskWorkspace:
        """
        Get existing workspace or create new one.

        Args:
            base_dir: Base directory for workspaces
            task_id: Task/workspace identifier
            allowed_external_dirs: List of allowed external directories

        Returns:
            TaskWorkspace instance
        """
        cache_key = f"{base_dir}:{task_id}"

        if cache_key not in self._workspaces:
            workspace = TaskWorkspace(task_id, base_dir, allowed_external_dirs)
            self._workspaces[cache_key] = workspace

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
