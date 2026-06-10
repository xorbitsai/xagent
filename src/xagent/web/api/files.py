import asyncio
import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, cast
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    Response,
)
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import (
    get_file_delivery_accel_redirect_enabled,
    get_file_delivery_accel_redirect_prefix,
    get_file_delivery_redirect_enabled,
    get_file_delivery_signed_url_ttl_seconds,
    get_storage_root,
    get_uploads_dir,
)
from ...core.file_storage.factory import get_file_storage
from ...core.tools.adapters.vibe.file_tool import read_file
from ...core.tools.core.file_analysis import collect_pptx_slide_blocks
from ..auth_dependencies import get_current_user
from ..config import (
    BINARY_EXTENSIONS,
    MAX_FILE_SIZE,
    MAX_FILE_SIZE_LABEL,
    get_upload_path,
    is_allowed_file,
)
from ..models.database import get_db
from ..models.task import Task
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..services.kb_file_service import aggregate_uploaded_file_statuses
from ..services.managed_file_ref import (
    FILE_INTEGRITY_REUPLOAD_MESSAGE,
    DurableObjectIntegrityError,
    DurableObjectMissingError,
    DurableStorageOperationError,
    ManagedFileRef,
    guess_media_type,
)
from ..services.uploaded_file_store import UploadedFileStore, delete_pptx_pdf_cache
from .legacy_file import (
    infer_user_id_from_legacy_path,
    is_valid_uuid,
    resolve_legacy_file_path,
    resolve_legacy_file_path_cross_user,
)

# Optional import for python-pptx (used for upload-time text preview).
pptx: ModuleType | None = None
# Kept around as a defensive placeholder so future call sites can raise a
# helpful error if python-pptx is missing instead of failing with
# AttributeError. Currently unused — _pptx_text_preview just no-ops when
# pptx is None.
pptx_not_installed_exception = RuntimeError(
    "python-pptx is not installed. "
    "Install with: pip install 'xagent[document-processing]'"
)
try:
    import pptx
except ImportError:
    pptx = None

logger = logging.getLogger(__name__)

file_router = APIRouter(prefix="/api/files", tags=["files"])


def _durable_storage_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Durable storage is temporarily unavailable",
    )


def _file_integrity_failed() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=FILE_INTEGRITY_REUPLOAD_MESSAGE,
    )


def _content_disposition_header(disposition: str, filename: str) -> str:
    safe_name = Path(filename).name or "file"
    fallback_name = "".join(
        character if 0x20 <= ord(character) <= 0x7E else "_" for character in safe_name
    )
    escaped_name = fallback_name.replace("\\", "\\\\").replace('"', '\\"')
    encoded_name = quote(safe_name, safe="")
    return (
        f"{disposition}; filename=\"{escaped_name}\"; filename*=UTF-8''{encoded_name}"
    )


def _inline_download_disposition(media_type: str) -> str:
    return (
        "inline"
        if media_type.startswith(("image/", "video/", "audio/", "text/"))
        else "attachment"
    )


def _preview_can_redirect(path: Path, media_type: str) -> bool:
    if path.suffix.lower() in {".pptx", ".ppt"}:
        return False
    return media_type not in {"text/html", "application/xhtml+xml", "image/svg+xml"}


def _durable_redirect_response(
    file_ref: ManagedFileRef,
    *,
    filename: str,
    media_type: str,
    disposition: str,
) -> RedirectResponse | None:
    if not get_file_delivery_redirect_enabled():
        return None

    try:
        signed_url = file_ref.signed_access_url(
            expires=get_file_delivery_signed_url_ttl_seconds(),
            content_type=media_type,
            content_disposition=_content_disposition_header(disposition, filename),
        )
    except DurableObjectIntegrityError as exc:
        raise _file_integrity_failed() from exc
    except DurableStorageOperationError as exc:
        raise _durable_storage_unavailable() from exc

    if not signed_url:
        return None

    return RedirectResponse(url=signed_url, status_code=307)


def _accel_redirect_response(
    path: Path,
    *,
    owner_user_id: int,
    filename: str,
    media_type: str,
    disposition: str,
) -> Response | None:
    if not get_file_delivery_accel_redirect_enabled():
        return None
    if not path.exists() or not path.is_file():
        return None

    _ensure_under_uploads(path, owner_user_id)
    uploads_root = get_uploads_dir().resolve()
    try:
        relative_path = path.resolve().relative_to(uploads_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc

    internal_prefix = get_file_delivery_accel_redirect_prefix()
    internal_uri = (
        internal_prefix.rstrip("/") + "/" + quote(relative_path.as_posix(), safe="/")
    )
    return Response(
        status_code=200,
        media_type=media_type,
        headers={
            "X-Accel-Redirect": internal_uri,
            "Content-Disposition": _content_disposition_header(disposition, filename),
        },
    )


async def _write_upload_with_size_limit(uploaded: UploadFile, target_path: Path) -> int:
    """Persist an uploaded file while enforcing the configured size limit."""
    total_size = 0
    read_buffer_size = 1024 * 1024  # 1MB chunks keep memory bounded for large files.

    try:
        with open(target_path, "wb") as buffer:
            while True:
                chunk = await uploaded.read(read_buffer_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File size exceeds maximum limit of {MAX_FILE_SIZE_LABEL}"
                        ),
                    )
                buffer.write(chunk)
    except Exception:
        try:
            if target_path.exists():
                target_path.unlink()
        except OSError:
            pass
        raise

    return total_size


async def store_uploaded_files(
    *,
    upload_items: list[UploadFile],
    task_type: str,
    task_id: str | None,
    folder: str | None,
    user: User,
    db: Session,
    single_file_mode: bool,
) -> Dict[str, Any]:
    parsed_task_id = _parse_task_id(task_id)
    uploaded_files = []
    written_paths: list[Path] = []
    written_storage_keys: list[str] = []

    try:
        for uploaded in upload_items:
            if not uploaded.filename or not uploaded.filename.strip():
                raise HTTPException(status_code=422, detail="No filename provided")
            if not is_allowed_file(uploaded.filename, task_type):
                raise HTTPException(
                    status_code=500,
                    detail=f"File type {Path(uploaded.filename).suffix.lower()} not supported for task type {task_type}",
                )

            try:
                target_path = _build_unique_file_path(
                    get_upload_path(
                        uploaded.filename, task_id, folder, _user_id_value(user)
                    )
                )
            except ValueError as e:
                logger.warning(f"Invalid folder name rejected: {folder!r} - {e}")
                raise HTTPException(
                    status_code=422, detail=f"Invalid folder name: {str(e)}"
                ) from e

            file_size = await _write_upload_with_size_limit(uploaded, target_path)
            written_paths.append(target_path)
            file_id = str(uuid4())
            file_record = UploadedFileStore(db).create_from_local_path(
                local_path=target_path,
                user_id=_user_id_value(user),
                file_id=file_id,
                task_id=parsed_task_id,
                filename=Path(uploaded.filename).name,
                mime_type=uploaded.content_type,
            )
            if file_record.storage_key:
                written_storage_keys.append(str(file_record.storage_key))
            setattr(file_record, "file_size", file_size)
            db.flush()

            content_preview = ""
            file_extension = Path(uploaded.filename).suffix.lower()
            if file_extension == ".pptx":
                try:
                    preview_content = await asyncio.to_thread(
                        _pptx_text_preview, target_path
                    )
                    content_preview = (
                        preview_content[:500] + "..."
                        if len(preview_content) > 500
                        else preview_content
                    )
                except Exception:
                    content_preview = ""
            elif file_extension == ".ppt":
                content_preview = ""
            elif file_extension not in BINARY_EXTENSIONS:
                try:
                    preview_content = read_file(str(target_path))
                    content_preview = (
                        preview_content[:500] + "..."
                        if isinstance(preview_content, str)
                        and len(preview_content) > 500
                        else preview_content
                    )
                except Exception:
                    content_preview = ""

            uploaded_files.append(
                {
                    "file_id": file_record.file_id,
                    "filename": file_record.filename,
                    "file_size": file_record.file_size,
                    "mime_type": file_record.mime_type,
                    "content_preview": content_preview,
                }
            )

        db.commit()
    except DurableStorageOperationError as exc:
        db.rollback()
        for storage_key in written_storage_keys:
            try:
                get_file_storage().delete(storage_key)
            except Exception:
                logger.warning("Failed to clean up durable upload: %s", storage_key)
        for path in written_paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        logger.warning("Durable storage unavailable during upload: %s", exc)
        raise _durable_storage_unavailable() from exc
    except Exception:
        db.rollback()
        for storage_key in written_storage_keys:
            try:
                get_file_storage().delete(storage_key)
            except Exception:
                logger.warning("Failed to clean up durable upload: %s", storage_key)
        for path in written_paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        raise

    if single_file_mode:
        first_file = uploaded_files[0]
        return {
            "success": True,
            "file_id": first_file["file_id"],
            "filename": first_file["filename"],
            "file_size": first_file["file_size"],
            "mime_type": first_file["mime_type"],
            "task_type": task_type,
            "content_preview": first_file["content_preview"],
            "message": f"Successfully uploaded {first_file['filename']}",
        }

    return {
        "success": True,
        "files": uploaded_files,
        "total_files": len(uploaded_files),
        "task_type": task_type,
        "message": f"Successfully uploaded {len(uploaded_files)} files",
    }


def _user_id_value(user: User) -> int:
    return int(getattr(user, "id"))


def _file_user_id_value(file_record: UploadedFile) -> int:
    return int(getattr(file_record, "user_id"))


def _is_admin_user(user: User) -> bool:
    return bool(getattr(user, "is_admin", False))


def _parse_task_id(task_id: Optional[str]) -> Optional[int]:
    if task_id is None or task_id == "":
        return None
    try:
        return int(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid task_id") from exc


def _build_unique_file_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _ensure_under_uploads(path: Path, user_id: int) -> None:
    resolved_path = path.resolve()
    uploads_dir = get_uploads_dir()
    uploads_root = uploads_dir.resolve()
    user_root = (uploads_dir / f"user_{user_id}").resolve()
    try:
        resolved_path.relative_to(uploads_root)
        resolved_path.relative_to(user_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc


def _resolve_public_preview_target(
    base_path: Path, relative_path: Optional[str], user_id: int
) -> Path:
    _ensure_under_uploads(base_path, user_id)
    if not relative_path:
        return base_path

    base_dir = base_path.parent.resolve()
    candidate = (base_dir / relative_path).resolve()

    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Access denied") from exc

    _ensure_under_uploads(candidate, user_id)
    return candidate


def _validate_public_task_file_access(
    db: Session,
    file_record: UploadedFile,
    token: str | None,
) -> None:
    if not file_record.task_id:
        return

    task = db.query(Task).filter(Task.id == file_record.task_id).first()
    if not task or not isinstance(task.agent_config, dict):
        return

    if not token:
        raise HTTPException(status_code=403, detail="Public file access token required")
    auth_mode = task.agent_config.get("auth_mode")

    if auth_mode == "share":
        from .public_chat_access import get_share_chat_user, get_task_for_share_context

        access_context = get_share_chat_user(token, db)
        get_task_for_share_context(db, int(task.id), access_context)
        return

    guest_id = task.agent_config.get("guest_id")
    if not isinstance(guest_id, str) or not guest_id:
        return

    from .public_chat_access import get_public_chat_user, get_task_for_public_context

    access_context = get_public_chat_user(token, db)
    get_task_for_public_context(db, int(task.id), access_context)


def _find_registered_preview_asset(
    db: Session,
    *,
    base_record: UploadedFile,
    target_path: Path,
    relative_path: str,
) -> Optional[UploadedFile]:
    target_path_candidates = {str(target_path)}
    try:
        target_path_candidates.add(str(target_path.resolve(strict=False)))
    except OSError:
        pass
    try:
        base_path = Path(str(base_record.storage_path))
        target_path_candidates.add(str(base_path.parent / relative_path))
    except (TypeError, ValueError):
        pass
    asset_record = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == base_record.user_id,
            UploadedFile.storage_path.in_(target_path_candidates),
        )
        .first()
    )
    if asset_record is not None:
        return asset_record

    base_workspace_path = str(base_record.workspace_relative_path or "")
    if not base_workspace_path:
        return None
    base_dir = Path(base_workspace_path).parent
    asset_workspace_path = str((base_dir / relative_path).as_posix())
    return (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == base_record.user_id,
            UploadedFile.task_id == base_record.task_id,
            UploadedFile.workspace_relative_path == asset_workspace_path,
        )
        .first()
    )


def _to_unix_timestamp(path: Path, fallback: Any) -> int:
    if path.exists():
        return int(path.stat().st_mtime)
    if fallback is not None and hasattr(fallback, "timestamp"):
        return int(fallback.timestamp())
    return 0


def _extract_relative_path(storage_path: Path, user_id: int) -> str:
    user_root = get_uploads_dir() / f"user_{user_id}"
    try:
        return str(storage_path.relative_to(user_root))
    except ValueError:
        return storage_path.name


def _collect_backfill_user_ids(user: User) -> list[int]:
    if not _is_admin_user(user):
        return [_user_id_value(user)]

    user_ids: list[int] = []
    if not get_uploads_dir().exists():
        return user_ids

    for child in get_uploads_dir().iterdir():
        if not child.is_dir() or not child.name.startswith("user_"):
            continue
        try:
            user_ids.append(int(child.name.replace("user_", "", 1)))
        except ValueError:
            continue
    return user_ids


def _infer_backfill_task_id(
    db: Session, file_path: Path, user_id: int
) -> Optional[int]:
    from ..models.task import Task

    user_root = get_uploads_dir() / f"user_{user_id}"
    try:
        rel_parts = file_path.relative_to(user_root).parts
    except ValueError:
        return None

    if not rel_parts:
        return None
    first_part = rel_parts[0]
    task_id_part: Optional[str] = None
    if first_part.startswith("web_task_"):
        task_id_part = first_part.replace("web_task_", "", 1)
    elif first_part.startswith("task_"):
        task_id_part = first_part.replace("task_", "", 1)

    if task_id_part is None:
        return None

    try:
        task_id = int(task_id_part)
    except ValueError:
        return None

    task = db.query(Task.id).filter(Task.id == task_id, Task.user_id == user_id).first()
    return task_id if task is not None else None


def _backfill_uploaded_file_records(db: Session, user: User) -> None:
    if not get_uploads_dir().exists():
        return

    target_user_ids = _collect_backfill_user_ids(user)
    if not target_user_ids:
        return

    existing_records: dict[str, UploadedFile] = {
        cast(str, row.storage_path): row
        for row in db.query(UploadedFile)
        .filter(UploadedFile.user_id.in_(target_user_ids))
        .all()
    }

    created = 0
    for target_user_id in target_user_ids:
        user_root = get_uploads_dir() / f"user_{target_user_id}"
        if not user_root.exists() or not user_root.is_dir():
            continue

        for candidate in user_root.rglob("*"):
            if not candidate.is_file():
                continue

            storage_path: str = str(candidate)
            existing_record = existing_records.get(storage_path)
            if existing_record is not None:
                if not existing_record.storage_key:
                    setattr(existing_record, "user_id", target_user_id)
                    setattr(existing_record, "storage_path", str(candidate))
                    UploadedFileStore(db).sync_existing(
                        existing_record, mime_type=guess_media_type(candidate.name)
                    )
                    created += 1
                continue

            file_id = str(uuid4())
            file_record = UploadedFileStore(db).create_from_local_path(
                local_path=candidate,
                user_id=target_user_id,
                file_id=file_id,
                task_id=_infer_backfill_task_id(db, candidate, target_user_id),
                filename=candidate.name,
                mime_type=guess_media_type(candidate.name),
            )
            existing_records[storage_path] = file_record
            created += 1

    if created > 0:
        try:
            db.commit()
            logger.info(f"Backfilled {created} uploaded_files records")
        except IntegrityError:
            db.rollback()
            logger.warning(
                "Backfill commit hit unique constraint race; rolled back safely"
            )


def _get_file_record(db: Session, file_id: str) -> UploadedFile:
    file_record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
    if file_record is None:
        raise HTTPException(status_code=404, detail="File not found")
    return file_record


def _resolve_file_path(
    db: Session, file_id_or_path: str, user_id: int
) -> Tuple[Optional[UploadedFile], Path, int]:
    """
    Resolve file_id or legacy path to file record and actual path.

    This function handles both:
    - New system: UUID file_id that maps to a database record
    - Legacy system: Relative paths like "web_task_235/output/file.jpeg"
    - Workspace system: Files created by agents in workspace directories

    Args:
        db: Database session
        file_id_or_path: Either a UUID file_id or a legacy file path
        user_id: Current user's ID for permission checks

    Returns:
        Tuple of (file_record or None, file_path, owner_user_id)

    Raises:
        HTTPException: If file is not found
    """
    # If it's a valid UUID, try to find by file_id (new system)
    if is_valid_uuid(file_id_or_path):
        file_record = (
            db.query(UploadedFile)
            .filter(UploadedFile.file_id == file_id_or_path)
            .first()
        )
        if file_record:
            return (
                file_record,
                Path(str(file_record.storage_path)),
                _file_user_id_value(file_record),
            )

    # For legacy paths, resolve from filesystem
    # First try to find in current user's directory
    file_path = resolve_legacy_file_path(file_id_or_path, user_id)
    owner_user_id = user_id

    # If not found and user is admin, try to infer owner from path and search all users
    if file_path is None:
        # Try to infer the correct user_id from the path
        inferred_user_id = infer_user_id_from_legacy_path(db, file_id_or_path)
        if inferred_user_id is not None:
            file_path = resolve_legacy_file_path(file_id_or_path, inferred_user_id)
            if file_path is not None:
                owner_user_id = inferred_user_id

        # If still not found and user is admin, try searching in all user directories
        if file_path is None and _is_admin_user_by_id(db, user_id):
            result = resolve_legacy_file_path_cross_user(file_id_or_path)
            if result is not None:
                file_path, owner_user_id = result

    if file_path is None:
        raise HTTPException(status_code=404, detail="File not found")

    # Try to find a matching database record (might exist for backfilled files)
    file_record = (
        db.query(UploadedFile)
        .filter(UploadedFile.storage_path == str(file_path))
        .first()
    )

    return (file_record, file_path, owner_user_id)


def _is_admin_user_by_id(db: Session, user_id: int) -> bool:
    """Check if a user is admin by user ID."""
    from ..models.user import User

    user = db.query(User).filter(User.id == user_id).first()
    return user is not None and getattr(user, "is_admin", False)


def _check_file_access(file_record: UploadedFile, user: User) -> None:
    if _is_admin_user(user):
        return
    if _file_user_id_value(file_record) != _user_id_value(user):
        raise HTTPException(status_code=403, detail="Access denied")


def _pptx_text_preview(path: Path) -> str:
    """Extract a plain-text preview from a .pptx file for upload responses.

    Concatenates each slide's shape text and speaker notes. Returns an empty
    string if python-pptx is not installed, the file is not .pptx, or
    extraction fails.
    """
    if not pptx or path.suffix.lower() != ".pptx":
        return ""
    try:
        prs = pptx.Presentation(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to open pptx for preview: %s", exc)
        return ""

    blocks, _stats = collect_pptx_slide_blocks(prs)
    return "\n\n".join(blocks)


def _pptx_pdf_cache_path(pptx_path: Path, file_id: Optional[str] = None) -> Path:
    """Return the on-disk cache path for a converted PDF preview.

    Caches are stored in ``<storage_root>/pptx_pdf_cache/`` — a dedicated
    directory outside the uploads tree.  Keeping them out of uploads:

    * prevents reconcile / backfill from treating ``.pptx.preview.pdf``
      sidecars as normal user-uploaded files;
    * avoids leaking derived content after a source upload is deleted
      (delete_file() explicitly removes the cache entry by file_id);
    * lets the uploads tree stay clean — only files written by users or
      the upload endpoint live there.

    For registered UUID files the cache key is the ``file_id``, which lets
    ``delete_file()`` remove it with a single unlink.  For legacy / non-UUID
    paths (no lifecycle management) we fall back to a SHA-256 prefix of the
    resolved source path — entries there age out naturally.
    """
    import hashlib

    cache_dir = get_storage_root() / "pptx_pdf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if file_id:
        return cache_dir / f"{file_id}.preview.pdf"
    path_key = hashlib.sha256(str(pptx_path.resolve()).encode()).hexdigest()[:24]
    return cache_dir / f"{path_key}.preview.pdf"


async def _convert_pptx_to_pdf(
    pptx_path: Path, file_id: Optional[str] = None
) -> Optional[Path]:
    """Convert a .pptx to PDF via LibreOffice with on-disk caching.

    Returns the path to the cached PDF on success, or ``None`` when soffice
    isn't available / conversion failed. The caller decides whether to
    surface a 503 (so the frontend can fall back) or just keep going.

    The fact that this can return ``None`` is the whole point of the
    preview-pdf endpoint: on developer machines without LibreOffice we
    don't break the UX — we let the frontend fall back to its canvas
    renderer. Production images (Docker) pre-install libreoffice so this
    path always succeeds there.

    ``file_id`` is forwarded to ``_pptx_pdf_cache_path`` so the cache entry
    lands in the managed cache directory (keyed by file_id) rather than as
    a sidecar next to the source file.
    """
    if pptx_path.suffix.lower() != ".pptx":
        return None

    cache_path = _pptx_pdf_cache_path(pptx_path, file_id)
    try:
        if (
            cache_path.exists()
            and cache_path.stat().st_mtime >= pptx_path.stat().st_mtime
        ):
            return cache_path
    except OSError:
        pass

    import tempfile

    try:
        # Pin the temp dir to the same directory as `cache_path` so the final
        # rename is a single intra-filesystem syscall. Without `dir=...`,
        # `tempfile.TemporaryDirectory()` uses the system default (typically
        # /tmp), which on Docker/production layouts is a different mount than
        # the uploads volume — the later `replace()` would then fail with
        # EXDEV (Invalid cross-device link). Per PR #542 review.
        with tempfile.TemporaryDirectory(dir=cache_path.parent) as temp_dir:
            proc = await asyncio.create_subprocess_exec(
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                temp_dir,
                str(pptx_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("pptx→pdf conversion timed out for %s", pptx_path)
                return None
            if proc.returncode != 0:
                logger.warning(
                    "soffice exited %s for %s: %s",
                    proc.returncode,
                    pptx_path,
                    stderr.decode("utf-8", "replace")[:200],
                )
                return None
            pdf_files = list(Path(temp_dir).glob("*.pdf"))
            if not pdf_files:
                return None
            # The PDF is already fully written and closed inside the temp dir,
            # and the temp dir lives in the same filesystem as `cache_path`
            # (see `dir=` above), so `replace()` is a single atomic rename.
            # No `.tmp` intermediate needed — that previous two-step dance
            # could leave orphans if a concurrent request raced. Per PR #542
            # review.
            try:
                pdf_files[0].replace(cache_path)
            except OSError as exc:
                logger.warning(
                    "Failed to cache pptx preview PDF at %s: %s", cache_path, exc
                )
                return None
            return cache_path
    except FileNotFoundError:
        # soffice binary is missing — this is the expected "fallback" path
        # on developer machines. Log once at INFO so it doesn't spam.
        logger.info(
            "soffice not on PATH; .pptx PDF previews will fall back to "
            "client-side rendering"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("pptx→pdf conversion failed for %s: %s", pptx_path, exc)
        return None


@file_router.post("/upload")
async def upload_file(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    task_type: str = Form(...),
    message: str = Form(""),
    task_id: str = Form(None),
    folder: str = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    del message
    upload_items: list[UploadFile] = []
    if file is not None:
        upload_items.append(file)
    if files:
        upload_items.extend(files)

    if not upload_items:
        raise HTTPException(status_code=422, detail="No files provided")

    single_file_mode = file is not None and (not files)
    return await store_uploaded_files(
        upload_items=upload_items,
        task_type=task_type,
        task_id=task_id,
        folder=folder,
        user=user,
        db=db,
        single_file_mode=single_file_mode,
    )


@file_router.get("/list")
async def list_files(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    search: str = Query("", description="Search filename or path"),
    task_id: int | None = Query(None, description="Filter by task ID"),
    uploads_only: bool = Query(False, description="Only include user uploads"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    query = db.query(UploadedFile)
    if not _is_admin_user(user):
        query = query.filter(UploadedFile.user_id == _user_id_value(user))

    normalized_search = search.strip()
    if normalized_search:
        like_pattern = f"%{normalized_search}%"
        query = query.filter(
            or_(
                UploadedFile.filename.ilike(like_pattern),
                UploadedFile.workspace_relative_path.ilike(like_pattern),
                UploadedFile.storage_path.ilike(like_pattern),
            )
        )

    if task_id is not None:
        query = query.filter(UploadedFile.task_id == task_id)
    elif uploads_only:
        query = query.filter(UploadedFile.task_id.is_(None))

    total_count = query.count()
    total_pages = max((total_count + size - 1) // size, 1)
    current_page = min(page, total_pages)
    offset = (current_page - 1) * size
    records = (
        query.order_by(UploadedFile.created_at.desc(), UploadedFile.id.desc())
        .offset(offset)
        .limit(size)
        .all()
    )
    file_status_map = aggregate_uploaded_file_statuses(
        file_ids=[str(record.file_id) for record in records if record.file_id],
        user_id=_user_id_value(user),
        is_admin=_is_admin_user(user),
    )
    files = []
    for record in records:
        path = Path(str(record.storage_path))
        record_user_id = _file_user_id_value(record)
        relative_path = _extract_relative_path(path, record_user_id)
        files.append(
            {
                "file_id": record.file_id,
                "filename": record.filename,
                "file_size": record.file_size,
                "modified_time": _to_unix_timestamp(path, record.created_at),
                "file_type": path.suffix.lower().lstrip("."),
                "relative_path": relative_path,
                "task_id": record.task_id,
                "user_id": record_user_id,
                "ingestion_status": file_status_map.get(str(record.file_id), "UNKNOWN"),
            }
        )

    return {
        "files": files,
        "total_count": total_count,
        "page": current_page,
        "size": size,
        "pages": total_pages,
    }


@file_router.get("/tasks")
async def list_file_tasks(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    query = (
        db.query(
            Task.id.label("task_id"),
            Task.title.label("title"),
            func.count(UploadedFile.id).label("file_count"),
            func.max(UploadedFile.created_at).label("latest_file_at"),
        )
        .join(UploadedFile, UploadedFile.task_id == Task.id)
        .filter(UploadedFile.task_id.isnot(None))
    )

    if not _is_admin_user(user):
        query = query.filter(
            UploadedFile.user_id == _user_id_value(user),
            Task.user_id == _user_id_value(user),
        )

    rows = (
        query.group_by(Task.id, Task.title)
        .order_by(
            func.max(UploadedFile.created_at).desc(),
            Task.id.desc(),
        )
        .all()
    )

    return {
        "tasks": [
            {
                "task_id": int(row.task_id),
                "title": str(row.title or f"Task {row.task_id}"),
                "file_count": int(row.file_count or 0),
            }
            for row in rows
        ]
    }


@file_router.get("/task/{task_id}")
async def list_task_files(
    task_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get all files for a specific task.

    More efficient than /api/files/list as it filters at database level.
    Only returns files that are already registered in the database.
    """
    # Query files for this task
    query = db.query(UploadedFile).filter(UploadedFile.task_id == task_id)

    # Permission check: only show user's own files unless admin
    if not _is_admin_user(user):
        query = query.filter(UploadedFile.user_id == _user_id_value(user))

    records = query.order_by(UploadedFile.created_at.desc()).all()

    files = []
    for record in records:
        file_ref = ManagedFileRef(record)
        path = file_ref.local_path
        if not path.exists() and not file_ref.has_durable_object:
            # Skip files that no longer exist on disk
            continue

        record_user_id = _file_user_id_value(record)
        relative_path = _extract_relative_path(path, record_user_id)

        # Categorize by directory (input/output/temp)
        path_parts = relative_path.split("/")
        file_category = "other"
        if len(path_parts) >= 2:
            subdir = path_parts[1]  # e.g., "input", "output", "temp"
            if subdir in ["input", "output", "temp"]:
                file_category = subdir

        files.append(
            {
                "file_id": record.file_id,
                "filename": record.filename,
                "file_size": record.file_size,
                "modified_time": _to_unix_timestamp(path, record.created_at),
                "file_type": path.suffix.lower().lstrip("."),
                "relative_path": relative_path,
                "category": file_category,
                "task_id": record.task_id,
                "user_id": record_user_id,
            }
        )

    return {"files": files, "total_count": len(files), "task_id": task_id}


@file_router.get("/download/{file_id:path}", response_model=None)
async def download_file(
    file_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    file_record, full_path, owner_user_id = _resolve_file_path(
        db, file_id, _user_id_value(user)
    )

    # Check access permissions
    if file_record:
        _check_file_access(file_record, user)
        file_ref = ManagedFileRef(file_record)
        file_name = str(file_record.filename)
        media_type = guess_media_type(file_name)
        _ensure_under_uploads(full_path, owner_user_id)
        content_disposition = _inline_download_disposition(media_type)
        redirect_response = _durable_redirect_response(
            file_ref,
            filename=file_name,
            media_type=media_type,
            disposition=content_disposition,
        )
        if redirect_response is not None:
            return redirect_response
        if full_path.exists() and full_path.is_file():
            accel_response = _accel_redirect_response(
                full_path,
                owner_user_id=owner_user_id,
                filename=file_name,
                media_type=media_type,
                disposition=content_disposition,
            )
            if accel_response is not None:
                return accel_response
            return FileResponse(
                path=str(full_path),
                filename=file_name,
                media_type=media_type,
                headers={
                    "Content-Disposition": _content_disposition_header(
                        content_disposition, file_name
                    )
                },
            )
        if file_ref.has_durable_object:
            try:
                restored_path = file_ref.ensure_local()
                accel_response = _accel_redirect_response(
                    restored_path,
                    owner_user_id=owner_user_id,
                    filename=file_name,
                    media_type=media_type,
                    disposition=content_disposition,
                )
                if accel_response is not None:
                    return accel_response
                return FileResponse(
                    path=str(restored_path),
                    filename=file_name,
                    media_type=media_type,
                    headers={
                        "Content-Disposition": _content_disposition_header(
                            content_disposition, file_name
                        )
                    },
                )
            except DurableObjectIntegrityError as exc:
                raise _file_integrity_failed() from exc
            except DurableStorageOperationError as exc:
                raise _durable_storage_unavailable() from exc
    else:
        # For legacy files without records, check ownership
        if owner_user_id != _user_id_value(user) and not _is_admin_user(user):
            raise HTTPException(status_code=403, detail="Access denied")
        file_name = full_path.name
        media_type = guess_media_type(file_name)

    _ensure_under_uploads(full_path, owner_user_id)

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # For images and other viewable content, set Content-Disposition to inline
    # to allow browser to display the file instead of downloading it
    content_disposition = _inline_download_disposition(media_type)

    accel_response = _accel_redirect_response(
        full_path,
        owner_user_id=owner_user_id,
        filename=file_name,
        media_type=media_type,
        disposition=content_disposition,
    )
    if accel_response is not None:
        return accel_response

    return FileResponse(
        path=str(full_path),
        filename=file_name,
        media_type=media_type,
        headers={
            "Content-Disposition": _content_disposition_header(
                content_disposition, file_name
            )
        },
    )


@file_router.get("/preview/{file_id:path}", response_model=None)
async def preview_file(
    file_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    file_record, full_path, owner_user_id = _resolve_file_path(
        db, file_id, _user_id_value(user)
    )

    # Check access permissions
    if file_record:
        _check_file_access(file_record, user)
        file_ref = ManagedFileRef(file_record)
        file_name = str(file_record.filename)
        media_type = guess_media_type(file_name)
        _ensure_under_uploads(full_path, owner_user_id)
        if _preview_can_redirect(full_path, media_type):
            redirect_response = _durable_redirect_response(
                file_ref,
                filename=file_name,
                media_type=media_type,
                disposition="inline",
            )
            if redirect_response is not None:
                return redirect_response
            accel_response = _accel_redirect_response(
                full_path,
                owner_user_id=owner_user_id,
                filename=file_name,
                media_type=media_type,
                disposition="inline",
            )
            if accel_response is not None:
                return accel_response
        if file_ref.has_durable_object:
            try:
                materialized_path = file_ref.materialize()
            except DurableObjectIntegrityError as exc:
                raise _file_integrity_failed() from exc
            except DurableStorageOperationError as exc:
                raise _durable_storage_unavailable() from exc
            except DurableObjectMissingError:
                materialized_path = file_ref.local_path
            return FileResponse(
                path=str(materialized_path),
                filename=file_name,
                media_type=media_type,
                headers={"Content-Disposition": "inline"},
            )
    else:
        # For legacy files without records, check ownership
        if owner_user_id != _user_id_value(user) and not _is_admin_user(user):
            raise HTTPException(status_code=403, detail="Access denied")
        file_name = full_path.name
        media_type = guess_media_type(file_name)

    _ensure_under_uploads(full_path, owner_user_id)

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if _preview_can_redirect(full_path, media_type):
        accel_response = _accel_redirect_response(
            full_path,
            owner_user_id=owner_user_id,
            filename=file_name,
            media_type=media_type,
            disposition="inline",
        )
        if accel_response is not None:
            return accel_response

    return FileResponse(
        path=str(full_path),
        filename=file_name,
        media_type=media_type,
        headers={"Content-Disposition": "inline"},
    )


@file_router.get("/preview-pdf/{file_id:path}", response_model=None)
async def preview_pptx_as_pdf(
    file_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Return a PDF rendering of a .pptx for high-fidelity in-browser preview.

    The frontend hits this endpoint first; if it succeeds it shows the PDF
    in an ``<iframe>`` (vector, text-selectable, perfect font metrics).
    On a 503 response — meaning the server doesn't have LibreOffice on
    PATH — the frontend falls back to the existing ``pptxviewjs`` canvas
    renderer so the UX still works on developer machines without soffice
    installed.

    Mirrors the durable-storage restore path that ``/preview`` uses: for
    files whose local copy was reaped but whose ``storage_key`` is still in
    durable storage, we materialize before conversion (returning 409 on
    checksum mismatch, 503 on durable backend failure — same semantics as
    ``/preview``). Without this, durable-only ``.pptx`` files would 404 here
    and silently fall back to the canvas renderer, which is exactly the
    UX regression flagged in PR #542 review.
    """
    file_record, full_path, owner_user_id = _resolve_file_path(
        db, file_id, _user_id_value(user)
    )

    # Resolve the local .pptx path. If the file is backed by durable
    # storage and not currently on disk, restore it first.
    pptx_path = full_path
    if file_record:
        _check_file_access(file_record, user)
        file_name = str(file_record.filename)
        _ensure_under_uploads(full_path, owner_user_id)
        file_ref = ManagedFileRef(file_record)
        if file_ref.has_durable_object:
            try:
                pptx_path = file_ref.materialize()
            except DurableObjectIntegrityError as exc:
                raise _file_integrity_failed() from exc
            except DurableStorageOperationError as exc:
                raise _durable_storage_unavailable() from exc
            except DurableObjectMissingError:
                # Durable record points at nothing; fall back to whatever
                # is on disk (or 404 below if it's gone too).
                pptx_path = file_ref.local_path
    else:
        if owner_user_id != _user_id_value(user) and not _is_admin_user(user):
            raise HTTPException(status_code=403, detail="Access denied")
        file_name = full_path.name
        _ensure_under_uploads(full_path, owner_user_id)

    if pptx_path is None or not pptx_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if pptx_path.suffix.lower() != ".pptx":
        raise HTTPException(
            status_code=400, detail="preview-pdf only supports .pptx files"
        )

    # Pass the UUID file_id so the cache entry lands in the managed
    # cache directory (keyed by file_id) rather than as a sidecar next to
    # the source file.  Legacy / non-UUID ids pass None and fall back to
    # the path-hash key.
    managed_file_id = file_id if is_valid_uuid(file_id) else None
    pdf_path = await _convert_pptx_to_pdf(pptx_path, file_id=managed_file_id)
    if pdf_path is None:
        # Distinct 503 lets the frontend fall back gracefully instead of
        # showing an error banner.
        raise HTTPException(
            status_code=503,
            detail="LibreOffice unavailable; client should fall back to canvas renderer",
        )

    return FileResponse(
        path=str(pdf_path),
        filename=Path(file_name).stem + ".pdf",
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@file_router.get("/public/download/{file_id:path}", response_model=None)
async def public_download_file(
    file_id: str,
    token: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> Any:
    """Public source-download endpoint for the chat file-card 'Open' link.

    Mirrors ``public_preview_file`` for resolution (UUID file id or
    legacy cross-user path) but always returns
    ``Content-Disposition: attachment; filename="..."`` so the browser
    saves under the source filename instead of trying to render the
    bytes inline. Used by ``InlineFilePreview`` 'Open' affordances
    that render as plain ``<a href>``: those clicks — plus middle-
    click / right-click "open in new tab" / "copy link" — don't carry
    the frontend's bearer token, so the auth'd
    ``/api/files/download/{file_id}`` route would 401 every plain
    browser navigation.

    The ``file_id`` is the only required capability, matching the
    existing ``public/preview`` contract. ``relative_path`` is
    intentionally NOT supported: 'Open' always targets the registered
    source artifact, never a sub-path inside it.
    """
    file_record: Optional[UploadedFile] = None
    target_path: Optional[Path] = None
    file_name: str

    if is_valid_uuid(file_id):
        file_record = (
            db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
        )

    if file_record:
        _validate_public_task_file_access(db, file_record, token)
        file_ref = ManagedFileRef(file_record)
        owner_user_id = _file_user_id_value(file_record)
        _ensure_under_uploads(file_ref.local_path, owner_user_id)
        file_name = str(file_record.filename)
        if file_ref.has_durable_object:
            try:
                target_path = file_ref.ensure_local()
            except DurableObjectIntegrityError as exc:
                raise _file_integrity_failed() from exc
            except DurableStorageOperationError as exc:
                raise _durable_storage_unavailable() from exc
            except DurableObjectMissingError:
                target_path = file_ref.local_path
        else:
            target_path = file_ref.local_path
    else:
        # Legacy non-UUID file id: resolve across user directories
        # (same fallback as public_preview_file).
        result = resolve_legacy_file_path_cross_user(file_id)
        if result is None:
            raise HTTPException(status_code=404, detail="File not found")
        target_path, owner_user_id = result
        _ensure_under_uploads(target_path, owner_user_id)
        file_name = target_path.name

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Pass filename= and let Starlette compose the Content-Disposition header.
    # Starlette 0.36+ generates ``filename*=utf-8''<percent-encoded>`` for
    # non-ASCII names (e.g. 报告.pptx) via urllib.parse.quote, so we must NOT
    # set the header manually: a raw ``filename="报告.pptx"`` string would be
    # latin-1 encoded by the ASGI layer and raise UnicodeEncodeError.
    # The default content_disposition_type is already "attachment", so we
    # only need to pass filename= here.
    return FileResponse(
        path=str(target_path),
        filename=file_name,
        media_type=guess_media_type(file_name),
    )


@file_router.get("/public/preview/{file_id:path}", response_model=None)
async def public_preview_file(
    file_id: str,
    relative_path: Optional[str] = Query(default=None),
    token: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> Any:
    # For public preview, we need to handle both file_id and legacy paths
    # Try UUID first
    file_record = None
    base_path = None
    owner_user_id = None

    if is_valid_uuid(file_id):
        file_record = (
            db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
        )

    if file_record:
        _validate_public_task_file_access(db, file_record, token)
        file_ref = ManagedFileRef(file_record)
        base_path = file_ref.local_path
        owner_user_id = _file_user_id_value(file_record)
        _ensure_under_uploads(base_path, owner_user_id)
        if file_ref.has_durable_object and not relative_path:
            try:
                target_path = file_ref.materialize()
            except DurableObjectIntegrityError as exc:
                raise _file_integrity_failed() from exc
            except DurableStorageOperationError as exc:
                raise _durable_storage_unavailable() from exc
            except DurableObjectMissingError:
                target_path = file_ref.local_path
            return FileResponse(
                path=str(target_path),
                filename=str(file_record.filename),
                media_type=guess_media_type(str(file_record.filename)),
                headers={"Content-Disposition": "inline"},
            )
    else:
        # Try to resolve as legacy path across all user directories
        result = resolve_legacy_file_path_cross_user(file_id)
        if result is None:
            raise HTTPException(status_code=404, detail="File not found")

        base_path, owner_user_id = result

    target_path = _resolve_public_preview_target(
        base_path,
        relative_path,
        owner_user_id,
    )

    if (
        file_record is not None
        and relative_path
        and (not target_path.exists() or not target_path.is_file())
    ):
        asset_record = _find_registered_preview_asset(
            db,
            base_record=file_record,
            target_path=target_path,
            relative_path=relative_path,
        )
        if asset_record is not None:
            _validate_public_task_file_access(db, asset_record, token)
            asset_ref = ManagedFileRef(asset_record)
            try:
                target_path = asset_ref.ensure_local()
            except DurableObjectIntegrityError as exc:
                raise _file_integrity_failed() from exc
            except DurableStorageOperationError as exc:
                raise _durable_storage_unavailable() from exc
            except DurableObjectMissingError:
                target_path = asset_ref.local_path
            _ensure_under_uploads(target_path, owner_user_id)

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(target_path),
        filename=target_path.name,
        media_type=guess_media_type(target_path.name),
        headers={"Content-Disposition": "inline"},
    )


@file_router.post("/backfill")
async def backfill_files(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Manually trigger file backfill to sync filesystem with database.

    This is a maintenance operation that scans the filesystem and creates
    database records for any unregistered files. Only available to admins.
    """
    if not _is_admin_user(user):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        _backfill_uploaded_file_records(db, user)
        return {"success": True, "message": "File backfill completed successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Backfill failed: {str(e)}") from e


@file_router.delete("/{file_id:path}")
async def delete_file(
    file_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    file_record, file_path, owner_user_id = _resolve_file_path(
        db, file_id, _user_id_value(user)
    )

    # Check access permissions
    if file_record:
        _check_file_access(file_record, user)
        file_name = str(file_record.filename)
    else:
        # For legacy files without records, check ownership
        if owner_user_id != _user_id_value(user) and not _is_admin_user(user):
            raise HTTPException(status_code=403, detail="Access denied")
        file_name = file_path.name

    _ensure_under_uploads(file_path, owner_user_id)

    if file_record:
        storage_key = str(file_record.storage_key or "")
        storage_status = str(file_record.storage_status or "")
        if storage_key and storage_status == "available":
            try:
                get_file_storage().delete(storage_key)
            except Exception as exc:
                logger.warning(
                    "Failed to clean up durable file before deleting row: %s",
                    storage_key,
                )
                raise _durable_storage_unavailable() from exc

        db.delete(file_record)
        db.commit()
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
        except OSError:
            logger.warning("Failed to clean up deleted local file: %s", file_path)
    elif file_path.exists() and file_path.is_file():
        file_path.unlink()

    # Remove any server-side PDF preview cache so derived content doesn't
    # outlive the source upload.  Uses the same helper as UploadedFileStore.delete()
    # so this HTTP route and every other deletion path (reconcile, orphan cleanup)
    # behave identically.  Non-UUID ids are silently ignored by the helper.
    delete_pptx_pdf_cache(file_id)

    return {
        "success": True,
        "message": f"File {file_name} deleted successfully",
        "file_id": file_id,
    }
