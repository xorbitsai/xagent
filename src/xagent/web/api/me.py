"""Current-user management endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.user import User
from ..schemas.user_api_key import (
    PersonalAPIKeyCreateResponse,
    PersonalAPIKeyMetadata,
    PersonalAPIKeyRevokeResponse,
)
from ..services.api_keys import UserApiKeyService

router = APIRouter(prefix="/api/me", tags=["me"])


@router.post("/personal-keys", response_model=PersonalAPIKeyCreateResponse)
async def create_personal_key(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PersonalAPIKeyCreateResponse:
    return UserApiKeyService(db).create_key(int(current_user.id))


@router.get("/personal-keys", response_model=list[PersonalAPIKeyMetadata])
async def list_personal_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[PersonalAPIKeyMetadata]:
    return UserApiKeyService(db).list_keys(int(current_user.id))


@router.delete("/personal-keys/{key_id}", response_model=PersonalAPIKeyRevokeResponse)
async def revoke_personal_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PersonalAPIKeyRevokeResponse:
    return UserApiKeyService(db).revoke_key(int(current_user.id), key_id)
