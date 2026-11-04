"""Scoped management endpoints for personal SDK keys."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.user import User
from ..schemas.user_api_key import (
    PersonalAPIKeyListResponse,
    PersonalAPIKeyRevokeResponse,
)
from ..services.api_keys import PersonalApiKeyManagementService
from ..services.personal_key_scope import get_personal_key_access_scope

router = APIRouter(prefix="/api/personal-api-keys", tags=["personal-api-keys"])


@router.get("", response_model=PersonalAPIKeyListResponse)
async def list_personal_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PersonalAPIKeyListResponse:
    scope = get_personal_key_access_scope(db, current_user)
    return PersonalAPIKeyListResponse(
        items=PersonalApiKeyManagementService(db).list_keys(scope),
        can_manage_others=scope.can_manage_others,
    )


@router.delete("/{key_id}", response_model=PersonalAPIKeyRevokeResponse)
async def revoke_personal_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PersonalAPIKeyRevokeResponse:
    scope = get_personal_key_access_scope(db, current_user)
    result = PersonalApiKeyManagementService(db).revoke_key(scope, key_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Personal API key not found")
    return result
