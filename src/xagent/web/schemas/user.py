from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator


class UserResponse(BaseModel):
    """Administrative account data with separate identity and display fields."""

    id: int
    username: str
    email: str | None = None
    is_admin: bool
    created_at: str
    updated_at: str

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def format_datetime(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int
    page: int
    size: int
    pages: int
