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
  - :func:`bind_turn_files` performs the mutation (stamping ``task_id``) and
    is called only once the turn is committed to running.

The WebSocket path keeps its resolve-then-bind-in-one behavior via
``handle_file_upload_for_task``, which now delegates to both.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ...core.agent.attachments import project_file_info_to_chip
from ...core.file_ref import FILE_REF_MODEL_INSTRUCTIONS
from ..models.uploaded_file import UploadedFile
from .managed_file_ref import ensure_uploaded_file_local_path

logger = logging.getLogger(__name__)


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

    file_infos: List[Dict[str, Any]] = []
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

        record = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.file_id == file_id,
                UploadedFile.user_id == owner_user_id,
                bind_filter,
            )
            .first()
        )
        if record is None:
            missing.append(file_id)
            continue

        source_path = ensure_uploaded_file_local_path(record)
        if not source_path.exists():
            logger.warning("Physical file not found for %s: %s", file_id, source_path)
            missing.append(file_id)
            continue

        original_name = Path(record.filename).name
        file_infos.append(
            {
                "file_id": record.file_id,
                "name": normalize_filename(original_name),
                "original_name": original_name,
                "size": record.file_size,
                "type": record.mime_type,
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

    Only rows owned by ``owner_user_id`` and currently ``task_id IS NULL``
    are moved; rows already bound (to this or another task) are left as-is.
    Committing here makes the binding visible to the background execution
    session that reads the files.
    """
    ids = [str(f).strip() for f in file_ids if str(f).strip()]
    if not ids:
        return
    db.query(UploadedFile).filter(
        UploadedFile.file_id.in_(ids),
        UploadedFile.user_id == owner_user_id,
        UploadedFile.task_id.is_(None),
    ).update({UploadedFile.task_id: task_id}, synchronize_session=False)
    db.commit()


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
