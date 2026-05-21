from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.task import Task
from ..models.user import User
from ..schemas.user import UserListResponse, UserResponse
from ..services.model_store import ModelStore

router = APIRouter(prefix="/api/admin/users", tags=["admin-users"])


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

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete related data in correct order to respect foreign key constraints
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
