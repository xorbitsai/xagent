"""Collection-level filesystem and UploadedFile coordination helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

from filelock import Timeout
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ..config import get_upload_path
from ..kb_physical_sync import collection_physical_lock, move_collection_dir_to_trash
from ..models.uploaded_file import UploadedFile
from ..utils.file import to_relative_path
from .kb_file_service import delete_uploaded_file_if_orphaned

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionPhysicalDeleteResult:
    """Outcome of attempting to move a collection directory to trash."""

    status: str
    error: Optional[str] = None
    collection_dir: Optional[Path] = None


@dataclass(frozen=True)
class CollectionPhysicalRenameResult:
    """Outcome of attempting to rename a collection directory and file paths."""

    status: str
    error: Optional[str] = None
    old_collection_dir: Optional[Path] = None
    new_collection_dir: Optional[Path] = None


def delete_collection_physical_dir(
    *,
    user_id: int,
    collection_name: str,
) -> CollectionPhysicalDeleteResult:
    """Move a collection directory to trash if it exists."""
    collection_dir: Optional[Path] = None
    try:
        collection_dir = get_upload_path(
            "", user_id=user_id, collection=collection_name
        )
        if not collection_dir.exists() or not collection_dir.is_dir():
            logger.debug(
                "Collection directory does not exist (or is not a directory): %s. "
                "This is normal for collections without physical files.",
                collection_dir,
            )
            return CollectionPhysicalDeleteResult(
                status="not_found",
                collection_dir=collection_dir,
            )

        with collection_physical_lock(collection_dir):
            move_collection_dir_to_trash(
                collection_dir,
                get_uploads_dir(),
                user_id,
                collection_name,
            )
        logger.info("Collection directory moved to trash: %s", collection_dir)
        return CollectionPhysicalDeleteResult(
            status="success",
            collection_dir=collection_dir,
        )
    except Timeout:
        return CollectionPhysicalDeleteResult(
            status="failed",
            error="Another operation is in progress; please try again later.",
            collection_dir=collection_dir,
        )
    except (PermissionError, OSError) as exc:
        logger.error(
            "Failed to move collection directory to trash for %s: %s",
            collection_name,
            exc,
        )
        return CollectionPhysicalDeleteResult(
            status="failed",
            error=str(exc),
            collection_dir=collection_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Error determining collection directory path for %s: %s",
            collection_name,
            exc,
        )
        return CollectionPhysicalDeleteResult(
            status="error",
            error=f"Path resolution error: {exc}",
            collection_dir=collection_dir,
        )


def delete_collection_uploaded_files(
    db: Session,
    *,
    user_id: int,
    collection_file_ids: Set[str],
    remaining_file_ids: Set[str],
    collection_dir: Optional[Path],
) -> int:
    """Delete orphan UploadedFile rows for a collection, with legacy path fallback."""
    deleted_uploaded_files = 0
    deleted_file_ids: Set[str] = set()

    for current_file_id in collection_file_ids:
        if delete_uploaded_file_if_orphaned(
            db,
            file_id=current_file_id,
            user_id=user_id,
            remaining_file_ids=remaining_file_ids,
        ):
            deleted_uploaded_files += 1
            deleted_file_ids.add(current_file_id)

    if collection_dir is not None:
        abs_prefix = str(collection_dir.resolve()) + os.sep
        abs_dir_str = str(collection_dir.resolve())
        # Also compute relative prefix for new-style relative storage_path
        rel_prefix = to_relative_path(collection_dir.resolve(), user_id)
        if not rel_prefix.endswith(os.sep):
            rel_prefix_str = rel_prefix + "/"
        else:
            rel_prefix_str = rel_prefix
        query = db.query(UploadedFile).filter(
            UploadedFile.user_id == user_id,
            or_(
                UploadedFile.storage_path.startswith(abs_prefix),
                UploadedFile.storage_path == abs_dir_str,
                UploadedFile.storage_path.startswith(rel_prefix_str),
                UploadedFile.storage_path == rel_prefix,
            ),
        )
        # Exclude file_ids already deleted in the first pass to avoid double-count
        if deleted_file_ids:
            query = query.filter(UploadedFile.file_id.notin_(deleted_file_ids))
        deleted = query.delete(synchronize_session=False)
        deleted_uploaded_files += int(deleted or 0)

    if deleted_uploaded_files:
        db.commit()

    return deleted_uploaded_files


def rename_collection_storage(
    db: Session,
    *,
    user_id: int,
    old_collection_name: str,
    new_collection_name: str,
    collection_file_ids: Set[str],
) -> CollectionPhysicalRenameResult:
    """Rename collection directory and update UploadedFile storage paths."""
    old_collection_dir: Optional[Path] = None
    new_collection_dir: Optional[Path] = None

    try:
        old_collection_dir = get_upload_path(
            "",
            user_id=user_id,
            collection=old_collection_name,
            create_if_not_exists=False,
        )
        new_collection_dir = get_upload_path(
            "",
            user_id=user_id,
            collection=new_collection_name,
            create_if_not_exists=False,
        )

        if not old_collection_dir.exists() or not old_collection_dir.is_dir():
            logger.debug(
                "Collection directory does not exist (or is not a directory): %s. "
                "This is normal for collections without physical files.",
                old_collection_dir,
            )
            return CollectionPhysicalRenameResult(
                status="not_found",
                old_collection_dir=old_collection_dir,
                new_collection_dir=new_collection_dir,
            )

        if new_collection_dir.exists():
            return CollectionPhysicalRenameResult(
                status="failed",
                error=(
                    "Cannot rename collection: target directory already exists. "
                    f"A collection named '{new_collection_name}' already has physical files."
                ),
                old_collection_dir=old_collection_dir,
                new_collection_dir=new_collection_dir,
            )

        with collection_physical_lock(old_collection_dir):
            old_abs = str(old_collection_dir)
            new_abs = str(new_collection_dir)
            old_rel = to_relative_path(old_collection_dir, user_id)
            new_rel = to_relative_path(new_collection_dir, user_id)
            uploads_resolved = get_uploads_dir().resolve()

            # Build filter to match both absolute (old data) and relative (new data) paths
            old_abs_prefix = old_abs + os.sep
            old_rel_prefix = old_rel + "/" if not old_rel.endswith("/") else old_rel

            records_query = db.query(UploadedFile).filter(
                UploadedFile.user_id == user_id
            )
            if collection_file_ids:
                records_query = records_query.filter(
                    or_(
                        UploadedFile.file_id.in_(sorted(collection_file_ids)),
                        UploadedFile.storage_path.startswith(old_abs_prefix),
                        UploadedFile.storage_path.startswith(old_rel_prefix),
                    )
                )
            else:
                records_query = records_query.filter(
                    or_(
                        UploadedFile.storage_path.startswith(old_abs_prefix),
                        UploadedFile.storage_path.startswith(old_rel_prefix),
                    )
                )
            records = records_query.all()
            previous_paths: dict[int, str] = {
                int(getattr(rec, "id")): str(getattr(rec, "storage_path"))
                for rec in records
            }
            for rec in records:
                storage_str = str(rec.storage_path)
                # Determine if this is absolute or relative path
                if storage_str.startswith(old_abs_prefix):
                    suffix = storage_str[len(old_abs) :]
                    new_path = new_abs + suffix
                elif storage_str.startswith(old_rel_prefix):
                    suffix = storage_str[len(old_rel) :]
                    new_path = new_rel + suffix
                else:
                    continue
                if ".." in suffix:
                    logger.warning(
                        "Skipping storage_path update (invalid suffix): %s",
                        suffix,
                    )
                    continue
                # Security check: ensure new path stays within uploads
                try:
                    if Path(new_path).is_absolute():
                        Path(new_path).resolve().relative_to(uploads_resolved)
                    else:
                        (
                            uploads_resolved / f"user_{user_id}" / new_path
                        ).resolve().relative_to(uploads_resolved)
                except ValueError:
                    logger.warning(
                        "Skipping storage_path update (path outside uploads directory): %s",
                        new_path,
                    )
                    continue
                rec.storage_path = new_path  # type: ignore[assignment]
            db.commit()
            if records:
                logger.info(
                    "Updated %d uploaded_files record(s) for renamed collection %s -> %s",
                    len(records),
                    old_collection_name,
                    new_collection_name,
                )

            import shutil

            try:
                shutil.move(str(old_collection_dir), str(new_collection_dir))
            except Exception as move_exc:  # noqa: BLE001
                logger.error(
                    "Physical collection move failed after DB update for %s -> %s: %s; rolling back DB paths",
                    old_collection_name,
                    new_collection_name,
                    move_exc,
                )
                for rec in records:
                    prior = previous_paths.get(int(getattr(rec, "id")), None)
                    if prior is not None:
                        rec.storage_path = prior  # type: ignore[assignment]
                try:
                    db.commit()
                except Exception as rollback_exc:  # noqa: BLE001
                    logger.exception(
                        "Rollback DB paths failed for collection rename %s -> %s: %s",
                        old_collection_name,
                        new_collection_name,
                        rollback_exc,
                    )
                raise

        logger.info(
            "Physically renamed collection directory: %s -> %s",
            old_collection_dir,
            new_collection_dir,
        )
        return CollectionPhysicalRenameResult(
            status="success",
            old_collection_dir=old_collection_dir,
            new_collection_dir=new_collection_dir,
        )
    except Timeout:
        return CollectionPhysicalRenameResult(
            status="failed",
            error="Another operation is in progress; please try again later.",
            old_collection_dir=old_collection_dir,
            new_collection_dir=new_collection_dir,
        )
    except (PermissionError, OSError) as exc:
        logger.error(
            "Failed to physically rename collection directory for %s: %s",
            old_collection_name,
            exc,
        )
        return CollectionPhysicalRenameResult(
            status="failed",
            error=str(exc),
            old_collection_dir=old_collection_dir,
            new_collection_dir=new_collection_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Error determining collection directory path for rename %s -> %s: %s",
            old_collection_name,
            new_collection_name,
            exc,
        )
        return CollectionPhysicalRenameResult(
            status="error",
            error=f"Path resolution error: {exc}",
            old_collection_dir=old_collection_dir,
            new_collection_dir=new_collection_dir,
        )
