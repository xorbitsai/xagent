import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from ...core.task_runtime import TaskRuntimeContext
from ..auth_dependencies import get_current_user
from ..models.database import (
    get_db,
    get_session_local,
    release_db_connection_if_clean,
)
from ..models.task import Task
from ..models.user import User
from ..schemas.user import UserListResponse, UserResponse
from ..services.model_store import ModelStore
from ..services.task_runtime import TaskRuntimeExtensionError, delete_task_extensions
from ..services.user_admin_scope import hidden_user_ids

router = APIRouter(prefix="/api/admin/users", tags=["admin-users"])
logger = logging.getLogger(__name__)


def _delete_legacy_text2sql_rows(db: Session, user_id: int) -> None:
    """Delete legacy Text2SQL rows without importing removed ORM models."""
    inspector = inspect(db.get_bind())
    if not inspector.has_table("text2sql_databases"):
        return
    db.execute(
        text("DELETE FROM text2sql_databases WHERE user_id = :user_id"),
        {"user_id": user_id},
    )


@router.get("", response_model=UserListResponse)
async def get_users(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    search: str = Query("", description="Search username"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserListResponse:
    """Get paginated list of users (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Build query
    query = db.query(User)

    # Hide app-managed service users (e.g. SaaS team storage principals) that
    # own data but are not real accounts.
    hidden = hidden_user_ids(db)
    if hidden:
        query = query.filter(User.id.notin_(hidden))

    # Apply search filter
    if search:
        query = query.filter(User.username.like(f"%{search}%"))

    # Get total count
    total = query.count()

    # Apply pagination
    offset = (page - 1) * size
    users = query.offset(offset).limit(size).all()

    return UserListResponse(
        users=[UserResponse.model_validate(user) for user in users],
        total=total,
        page=page,
        size=size,
        pages=(total + size - 1) // size,
    )


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Delete a user (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    # Cannot delete yourself
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    # App-managed service users (e.g. SaaS team storage principals) are hidden
    # from the directory and must not be deletable — doing so would orphan the
    # data they back.
    if user_id in set(hidden_user_ids(db)):
        raise HTTPException(status_code=404, detail="User not found")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session_factory = get_session_local()
    task_runtime_contexts = [
        TaskRuntimeContext(
            task_id=int(task_id),
            user_id=int(task_user_id),
            source=str(source) if source is not None else None,
            session_factory=session_factory,
        )
        for task_id, task_user_id, source in db.query(
            Task.id,
            Task.user_id,
            Task.source,
        )
        .filter(Task.user_id == user_id)
        .all()
    ]
    release_db_connection_if_clean(db)

    for context in task_runtime_contexts:
        try:
            await delete_task_extensions(context)
        except TaskRuntimeExtensionError:
            logger.warning(
                "User %s was deleted but runtime extension cleanup failed for task %s",
                user_id,
                context.task_id,
                exc_info=True,
            )

    # Re-read after the await: releasing the clean read transaction above may
    # expire ORM state, and extension hooks use their own short sessions.
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete related data in correct order to respect foreign key constraints.
    from ..models.mcp import UserMCPServer

    # Existing deployments may still have the removed Text2SQL table. Clean it
    # up by table name so user deletion keeps working under strict FK checks.
    _delete_legacy_text2sql_rows(db, user_id)

    # Delete user's tasks
    db.query(Task).filter(Task.user_id == user_id).delete()

    # Delete user's MCP server associations (not the servers themselves)
    db.query(UserMCPServer).filter(UserMCPServer.user_id == user_id).delete()

    # Delete the user (UserModel and UserDefaultModel have cascade delete)
    db.delete(user)
    db.commit()
    ModelStore(db).invalidate_after_user_delete()

    return {"message": "User deleted successfully"}
