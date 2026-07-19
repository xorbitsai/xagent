"""Validate and repair model-authored file links before they reach the UI."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ...core.file_ref import build_file_id_ref, parse_file_id_ref
from ..models.uploaded_file import UploadedFile

logger = logging.getLogger(__name__)

_MARKDOWN_FILE_REFERENCE_RE = re.compile(
    r"(?P<image>!)?\[(?P<label>[^\]]*)\]\((?P<target>file:[^)\s]+)\)"
)


def reconcile_assistant_file_references(
    db: Session,
    *,
    task_id: int,
    user_id: int,
    content: Any,
) -> Any:
    """Canonicalize valid links, repair unique filename matches, and unlink fakes.

    Models occasionally copy a filename correctly while inventing the UUID in
    ``file:<id>``. A broken UUID must never become a clickable preview. When the
    markdown label uniquely identifies a file in the current task/user scope,
    replace it with that record's canonical id; otherwise keep only plain text.
    """
    if not isinstance(content, str) or "file:" not in content:
        return content

    records = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.user_id == int(user_id),
            or_(UploadedFile.task_id == int(task_id), UploadedFile.task_id.is_(None)),
        )
        .all()
    )
    records_by_id = {str(record.file_id): record for record in records}
    records_by_filename: dict[str, list[UploadedFile]] = defaultdict(list)
    task_records_by_filename: dict[str, list[UploadedFile]] = defaultdict(list)
    for record in records:
        filename_key = str(record.filename).casefold()
        records_by_filename[filename_key].append(record)
        if record.task_id is not None and int(record.task_id) == int(task_id):
            task_records_by_filename[filename_key].append(record)

    def replacement(match: re.Match[str]) -> str:
        prefix = match.group("image") or ""
        label = match.group("label")
        target = match.group("target")
        referenced_id = parse_file_id_ref(target)
        record = records_by_id.get(referenced_id or "")

        if record is None:
            filename = Path(label.strip()).name
            filename_key = filename.casefold()
            candidates = task_records_by_filename.get(filename_key, [])
            if not candidates:
                candidates = records_by_filename.get(filename_key, [])
            if len(candidates) == 1:
                record = candidates[0]
                logger.info(
                    "Repaired invalid assistant FileRef %s using unique filename %s "
                    "for task %s",
                    referenced_id or target,
                    filename,
                    task_id,
                )

        if record is None:
            logger.warning(
                "Removed invalid assistant FileRef %s for task %s",
                referenced_id or target,
                task_id,
            )
            return label

        return f"{prefix}[{label}]({build_file_id_ref(str(record.file_id))})"

    return _MARKDOWN_FILE_REFERENCE_RE.sub(replacement, content)
