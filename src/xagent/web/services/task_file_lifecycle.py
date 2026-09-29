"""Task attachment provenance inside the caller's database transaction."""

from datetime import UTC, datetime
from typing import Literal

from sqlalchemy.orm import Session

from ..models.task import Task
from ..models.uploaded_file import UploadedFile

DetachmentReason = Literal["task_deleted", "task_create_failed"]


def lock_attachment_task(
    db: Session,
    task_id: int,
    *,
    owner_user_id: int | None = None,
    deleting: bool = False,
) -> Task | None:
    """Serialize deletion/binding with the task locked before upload rows.

    Binders use NO KEY UPDATE so child FK inserts can proceed. Deletion uses
    FOR UPDATE to fence new attachments before the detachment scan, just as
    retention's authoritative task lock fences new execution commands.
    """
    query = db.query(Task).filter(Task.id == task_id)
    if owner_user_id is not None:
        query = query.filter(Task.user_id == owner_user_id)
    return query.with_for_update(key_share=not deleting).first()


def detach_task_files(db: Session, *, task_id: int, reason: DetachmentReason) -> None:
    """Mark only this task's attachments; never turn drafts into expiring files."""
    if reason not in {"task_deleted", "task_create_failed"}:
        raise ValueError(f"Unsupported attachment detachment reason: {reason}")
    db.query(UploadedFile).filter(UploadedFile.task_id == task_id).update(
        {
            UploadedFile.task_id: None,
            UploadedFile.detached_reason: reason,
            UploadedFile.detached_at: datetime.now(UTC),
        },
        synchronize_session="fetch",
    )
