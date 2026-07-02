"""Scoped credential API for env-backed tools."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.user import User
from ..services.tool_credentials import (
    SQL_TOOL_NAME,
    TOOL_CREDENTIAL_SPECS,
    ApiScopeType,
    clear_scoped_tool_credential,
    get_tool_credential_view,
    list_tool_credential_views,
    set_scoped_tool_credentials,
)

CredentialScope = ApiScopeType

tool_credentials_router = APIRouter(
    prefix="/api/tool-credentials", tags=["tool-credentials"]
)


class CredentialFieldUpdate(BaseModel):
    value: str


class ToolCredentialUpdateRequest(BaseModel):
    credentials: dict[str, CredentialFieldUpdate]


def _require_user_id(current_user: User) -> int:
    user_id: object = getattr(current_user, "id", None)
    if isinstance(user_id, int):
        return user_id
    raise HTTPException(status_code=500, detail="Authenticated user is missing an id")


def _scope_context(
    *,
    scope: CredentialScope,
    current_user: User,
    db: Session,
) -> dict[str, Any]:
    user_id = _require_user_id(current_user)

    if scope == "user":
        return {
            "scope_type": "user",
            "scope_id": user_id,
            "user_id": user_id,
            "user": current_user,
            "include_instance": True,
        }
    if not bool(getattr(current_user, "is_admin", False)):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return {
        "scope_type": "instance",
        "scope_id": None,
        "user_id": user_id,
        "user": current_user,
        "include_instance": True,
    }


def _ensure_configurable_tool(tool_name: str) -> None:
    if tool_name != SQL_TOOL_NAME and tool_name not in TOOL_CREDENTIAL_SPECS:
        raise HTTPException(
            status_code=404, detail=f"Tool '{tool_name}' is not configurable"
        )


@tool_credentials_router.get("")
async def list_tool_credentials(
    scope: CredentialScope = Query("user"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    ctx = _scope_context(scope=scope, current_user=current_user, db=db)
    items = list_tool_credential_views(db, **ctx)
    return {"tools": items, "count": len(items)}


@tool_credentials_router.get("/{tool_name}")
def get_tool_credentials(
    tool_name: str,
    scope: CredentialScope = Query("user"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _ensure_configurable_tool(tool_name)
    ctx = _scope_context(scope=scope, current_user=current_user, db=db)
    return get_tool_credential_view(db, tool_name, **ctx)


@tool_credentials_router.put("/{tool_name}")
def update_tool_credentials(
    tool_name: str,
    payload: ToolCredentialUpdateRequest,
    scope: CredentialScope = Query("user"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _ensure_configurable_tool(tool_name)
    ctx = _scope_context(scope=scope, current_user=current_user, db=db)
    updates = {
        field_name: field_update.value
        for field_name, field_update in payload.credentials.items()
    }
    try:
        set_scoped_tool_credentials(
            db,
            scope_type=ctx["scope_type"],
            scope_id=ctx["scope_id"],
            tool_name=tool_name,
            values=updates,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return get_tool_credential_view(db, tool_name, **ctx)


@tool_credentials_router.delete("/{tool_name}/{field_name}")
def delete_tool_credential(
    tool_name: str,
    field_name: str,
    scope: CredentialScope = Query("user"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _ensure_configurable_tool(tool_name)
    if (
        tool_name != SQL_TOOL_NAME
        and field_name not in TOOL_CREDENTIAL_SPECS[tool_name]
    ):
        raise HTTPException(
            status_code=404, detail=f"Field '{field_name}' is not configurable"
        )
    ctx = _scope_context(scope=scope, current_user=current_user, db=db)
    clear_scoped_tool_credential(
        db,
        scope_type=ctx["scope_type"],
        scope_id=ctx["scope_id"],
        tool_name=tool_name,
        field_name=field_name,
    )
    return get_tool_credential_view(db, tool_name, **ctx)
