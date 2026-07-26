"""Shared file-attachment pipeline for a single task turn.

One owner for the resolve -> bind -> LLM-context steps so both transports
stay identical and cannot drift:

  - ``api/websocket.py`` (WebSocket UI path)
  - ``api/v1/tasks.py``  (SDK ``/v1`` path)

The resolve and bind steps are deliberately split:

  - :func:`resolve_turn_file_infos` is read-only. The SDK path calls it to
    validate file ids *before* it commits a task or claims a turn, so a bad
    id fails with 400 without leaving an orphan PENDING task behind, and no
    file gets bound to a turn that then 409s.
  - :func:`bind_turn_files_no_commit` stages the mutation inside the
    caller-owned claim transaction. :func:`bind_turn_files` is the
    compatibility wrapper for owners that need an immediate commit.

The WebSocket path keeps its resolve-then-bind-in-one behavior via
``handle_file_upload_for_task``, which now delegates to both.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ...core.agent.attachments import project_file_info_to_chip
from ...core.execution_scope import (
    ExecutionScope,
    execution_scope_from_agent_config,
    resolve_execution_scope,
)
from ...core.file_ref import FILE_REF_MODEL_INSTRUCTIONS
from ..models.database import release_db_connection_if_clean
from ..models.task import Task
from ..models.uploaded_file import UploadedFile
from .managed_file_ref import ensure_uploaded_file_local_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TurnFileRecordSnapshot:
    """Detached fields needed to materialize one authorized upload."""

    user_id: int
    file_id: str
    filename: str
    file_size: int
    mime_type: str | None
    storage_path: str
    storage_key: str | None
    storage_status: str | None
    checksum: str | None


def _task_execution_scope_in_session(
    db: Session,
    task_id: int | None,
) -> tuple[int | None, ExecutionScope | None]:
    """Read the persisted half of canonical scope resolution in this Session."""

    if task_id is None:
        return None, None
    row = db.query(Task.agent_config).filter(Task.id == task_id).first()
    return task_id, execution_scope_from_agent_config(row[0] if row else None)


def normalize_filename(filename: str) -> str:
    """
    Normalize filename by removing special characters and spaces.

    Args:
        filename: Original filename

    Returns:
        Normalized filename safe for file operations
    """
    # Keep file extension
    name_part = Path(filename).stem
    extension = Path(filename).suffix

    # Unicode normalize (NFD to NFC, remove diacritics)
    name_part = unicodedata.normalize("NFC", name_part)

    # Replace spaces with underscores
    name_part = re.sub(r"\s+", "_", name_part)

    # Remove special characters, keep only letters, numbers, underscores, Chinese characters
    name_part = re.sub(r"[^\w一-鿿\-_.]", "", name_part)

    # Remove consecutive underscores
    name_part = re.sub(r"_+", "_", name_part)

    # Remove leading and trailing underscores
    name_part = name_part.strip("_")

    # Use default name if filename is empty
    if not name_part:
        name_part = "file"

    # Reassemble filename
    normalized_name = name_part + extension

    # Ensure filename doesn't start with a dot (hidden file)
    if normalized_name.startswith("."):
        normalized_name = "_" + normalized_name

    return normalized_name


def resolve_turn_file_infos(
    *,
    file_ids: List[str],
    owner_user_id: int,
    db: Session,
    task_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolve file ids to bindable file-info dicts WITHOUT mutating them.

    A file id resolves when a row exists that is owned by ``owner_user_id``,
    is either unbound (``task_id IS NULL``) or already bound to ``task_id``,
    and whose bytes are present on disk.

    Args:
        file_ids: Requested file ids, in caller order.
        owner_user_id: The only user whose files are reachable.
        db: Session for the read.
        task_id: When set, files already bound to this task also resolve
            (so re-attaching within the same task is idempotent). When
            ``None`` (task not created yet), only unbound files resolve.

    Returns:
        ``(file_info_list, missing_ids)``. ``file_info_list`` preserves input
        order and carries the same shape the WebSocket path produces
        (file_id, name, original_name, size, type, path). ``missing_ids``
        lists ids that did not resolve (bad id, wrong owner, bound to another
        task, or missing bytes) so callers can decide strict-vs-lenient.
    """
    if task_id is not None:
        bind_filter: Any = or_(
            UploadedFile.task_id == task_id, UploadedFile.task_id.is_(None)
        )
    else:
        bind_filter = UploadedFile.task_id.is_(None)

    normalized_ids: list[str] = []
    missing: List[str] = []
    seen: set[str] = set()
    for raw_file_id in file_ids:
        file_id = str(raw_file_id or "").strip()
        if not file_id:
            missing.append(str(raw_file_id))
            continue
        # Dedup at the source so a repeated file_id doesn't produce duplicate
        # UPLOADED FILES lines / attachment chips downstream.
        if file_id in seen:
            continue
        seen.add(file_id)
        normalized_ids.append(file_id)

    if not normalized_ids:
        return [], missing

    records = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.file_id.in_(normalized_ids),
            UploadedFile.user_id == owner_user_id,
            UploadedFile.storage_status != "compensating",
            bind_filter,
        )
        .all()
    )
    records_by_id = {str(record.file_id): record for record in records}
    snapshots: list[_TurnFileRecordSnapshot] = []
    for file_id in normalized_ids:
        record = records_by_id.get(file_id)
        if record is None:
            missing.append(file_id)
            continue

        snapshots.append(
            _TurnFileRecordSnapshot(
                user_id=int(record.user_id),
                file_id=str(record.file_id),
                filename=str(record.filename),
                file_size=int(record.file_size or 0),
                mime_type=(
                    str(record.mime_type) if record.mime_type is not None else None
                ),
                storage_path=str(record.storage_path),
                storage_key=(
                    str(record.storage_key) if record.storage_key is not None else None
                ),
                storage_status=(
                    str(record.storage_status)
                    if record.storage_status is not None
                    else None
                ),
                checksum=(
                    str(record.checksum) if record.checksum is not None else None
                ),
            )
        )

    scope_task_id, persisted_scope = _task_execution_scope_in_session(db, task_id)
    if snapshots and not release_db_connection_if_clean(db):
        raise RuntimeError(
            "Turn file materialization requires a read-only database phase"
        )
    execution_scope = (
        resolve_execution_scope(
            scope_task_id,
            persisted_snapshot=persisted_scope,
        )
        if scope_task_id is not None
        else None
    )

    file_infos: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        source_path = ensure_uploaded_file_local_path(
            snapshot,
            execution_scope=execution_scope,
        )
        if not source_path.exists():
            logger.warning(
                "Physical file not found for %s: %s",
                snapshot.file_id,
                source_path,
            )
            missing.append(snapshot.file_id)
            continue

        original_name = Path(snapshot.filename).name
        file_infos.append(
            {
                "file_id": snapshot.file_id,
                "name": normalize_filename(original_name),
                "original_name": original_name,
                "size": snapshot.file_size,
                "type": snapshot.mime_type,
                "path": str(source_path),
                "workspace_path": None,
            }
        )

    return file_infos, missing


def bind_turn_files(
    *,
    file_ids: List[str],
    task_id: int,
    owner_user_id: int,
    db: Session,
) -> None:
    """Stamp ``task_id`` onto the given still-unbound files and commit.

    Compatibility wrapper around :func:`bind_turn_files_no_commit`.
    """
    missing = bind_turn_files_no_commit(
        file_ids=file_ids,
        task_id=task_id,
        owner_user_id=owner_user_id,
        db=db,
    )
    if missing:
        raise ValueError("Files are no longer bindable: " + ", ".join(missing))
    db.commit()


def bind_turn_files_no_commit(
    *,
    file_ids: List[str],
    task_id: int,
    owner_user_id: int,
    db: Session,
) -> List[str]:
    """Stage file binding without ending the caller-owned transaction.

    Every requested id must still be owned by ``owner_user_id`` and either
    unbound or already bound to this task. Returning the missing ids lets the
    domain owner roll back its claim when another request won the file between
    read-only preparation and the atomic turn transaction.
    """

    ids = [str(f).strip() for f in file_ids if str(f).strip()]
    if not ids:
        return []
    ids = list(dict.fromkeys(ids))
    # Claim every currently-unbound row first. On PostgreSQL the conditional
    # UPDATE waits for a concurrent writer and then re-evaluates its predicate;
    # on SQLite the serialized writer lock provides the same winner/loser
    # outcome. Verify ownership after the UPDATE so a transaction that lost a
    # race cannot report a successful bind from an earlier SELECT snapshot.
    db.query(UploadedFile).filter(
        UploadedFile.file_id.in_(ids),
        UploadedFile.user_id == owner_user_id,
        UploadedFile.task_id.is_(None),
        UploadedFile.storage_status != "compensating",
    ).update({UploadedFile.task_id: task_id}, synchronize_session=False)
    bound_rows = (
        db.query(UploadedFile.file_id)
        .filter(
            UploadedFile.file_id.in_(ids),
            UploadedFile.user_id == owner_user_id,
            UploadedFile.task_id == task_id,
            UploadedFile.storage_status != "compensating",
        )
        .all()
    )
    bound = {str(row.file_id) for row in bound_rows}
    return [file_id for file_id in ids if file_id not in bound]


def build_uploaded_files_context(
    file_info_list: List[Dict[str, Any]], *, is_agent_builder: bool = False
) -> str:
    """Build stable LLM context for files already uploaded for this turn."""
    if not file_info_list:
        return ""

    file_summaries = []
    file_ids = []
    for file_info in file_info_list:
        file_id = str(file_info.get("file_id") or "").strip()
        if not file_id:
            continue
        name = str(
            file_info.get("original_name") or file_info.get("name") or "uploaded file"
        )
        file_ids.append(file_id)
        file_summaries.append(f"- {name}: file_id={file_id}")

    if not file_ids:
        return ""

    lines = [
        "## UPLOADED FILES",
        "The user has uploaded file(s) for this turn. Use these exact file_id values:",
        *file_summaries,
        "",
        FILE_REF_MODEL_INSTRUCTIONS,
    ]
    if is_agent_builder:
        joined_file_ids = ", ".join(f'"{file_id}"' for file_id in file_ids)
        lines.extend(
            [
                "",
                "For knowledge-base creation, call `create_knowledge_base_from_file` with:",
                f"  file_ids = [{joined_file_ids}]",
                "Do NOT ask the user to upload again unless these file_ids fail.",
            ]
        )
    return "\n".join(lines)


def append_uploaded_files_context(message: str, uploaded_files_context: str) -> str:
    if not uploaded_files_context:
        return message
    if uploaded_files_context in message:
        return message
    return f"{message.rstrip()}\n\n{uploaded_files_context}"


def normalize_attachments_for_persistence(
    file_info_list: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Project ``file_info_list`` to the minimal chip shape persisted on rows.

    Thin wrapper around the shared
    ``core.agent.attachments.project_file_info_to_chip`` so the trace
    callback and the persistence path can't drift on what fields the
    browser sees (paths must never leak — the attachments column and the
    user_message trace events both reach the UI).
    """
    return project_file_info_to_chip(file_info_list)
