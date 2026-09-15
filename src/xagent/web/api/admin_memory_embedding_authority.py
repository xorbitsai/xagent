from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user, is_admin_user
from ..models.database import get_db
from ..models.user import User
from ..services.global_memory_embedding_authority import (
    AuthorityConfiguration,
    GlobalMemoryEmbeddingAuthorityService,
)

router = APIRouter(
    prefix="/api/admin/memory/embedding-authority",
    tags=["admin-memory"],
)


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not is_admin_user(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


class AuthorityState(BaseModel):
    configured: bool
    provider: str | None = None
    model_name: str | None = None
    endpoint: str | None = None
    dimension: int | None = None
    instruct: str | None = None
    max_retries: int | None = None
    credential_status: str | None = None
    configured_at: datetime | None = None
    updated_at: datetime | None = None


def _public_state(service: GlobalMemoryEmbeddingAuthorityService) -> AuthorityState:
    row = service.get_row()
    if row is None:
        return AuthorityState(configured=False)
    return AuthorityState(
        configured=True,
        provider=str(row.model_provider),
        model_name=str(row.model_name),
        endpoint=str(row.base_url),
        dimension=int(row.dimension),
        instruct=str(row.instruct) if row.instruct is not None else None,
        max_retries=int(row.max_retries),
        credential_status=service.credential_status(),
        configured_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("", response_model=AuthorityState)
def get_authority(
    _admin: User = Depends(require_admin), db: Session = Depends(get_db)
) -> AuthorityState:
    return _public_state(GlobalMemoryEmbeddingAuthorityService(db))


@router.put("", response_model=AuthorityState)
def set_authority(
    request: AuthorityConfiguration,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> AuthorityState:
    actor_subject = str(admin.actor_subject or "")
    if not actor_subject:
        raise HTTPException(409, detail="Admin actor identity is unavailable")
    service = GlobalMemoryEmbeddingAuthorityService(db)
    try:
        service.set(request, actor_subject=actor_subject)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(400, detail=str(exc)) from exc
    except RuntimeError:
        db.rollback()
        raise HTTPException(
            503, detail="Global memory embedding authority could not be stored"
        ) from None
    return _public_state(service)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def delete_authority(
    _admin: User = Depends(require_admin), db: Session = Depends(get_db)
) -> Response:
    GlobalMemoryEmbeddingAuthorityService(db).delete()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
