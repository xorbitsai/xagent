"""Small transaction helpers shared by task/file lifecycle operations."""

from sqlalchemy.orm import Session

from ..models.task import Task


def lock_task_no_commit(
    db: Session,
    *,
    task_id: int,
    owner_user_id: int | None = None,
) -> Task | None:
    """Lock and return a task without committing the caller-owned transaction."""

    query = db.query(Task).filter(Task.id == task_id)
    if owner_user_id is not None:
        query = query.filter(Task.user_id == owner_user_id)
    return query.with_for_update().one_or_none()
