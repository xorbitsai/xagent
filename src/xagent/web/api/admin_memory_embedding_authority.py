from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user, is_admin_user
from ..models.database import get_db
from ..models.user import User
from ..services.global_memory_embedding_authority import (
    AuthorityConfiguration,
    CredentialSource,
    GlobalMemoryEmbeddingAuthorityRecord,
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
    """Public view of the authority. Actor subjects are deliberately absent."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    configured: bool = True
    provider: str | None = None
    model_name: str | None = None
    endpoint: str | None = None
    dimension: int | None = None
    instruct: str | None = None
    max_retries: int | None = None
    credential_source: CredentialSource | None = None
    global_sharing_consent: bool | None = None
    consented_at: datetime | None = None
    credential_status: str | None = None
    configured_at: datetime | None = Field(default=None, validation_alias="created_at")
    updated_at: datetime | None = None


def _state(record: GlobalMemoryEmbeddingAuthorityRecord | None) -> AuthorityState:
    """The one renderer for both routes. A record carries the status its own
    path established -- the read path from the stored row, the write path from
    the ciphertext it just encrypted -- so neither hardcodes nor re-reads."""
    if record is None:
        return AuthorityState(configured=False)
    return AuthorityState.model_validate(record)


@router.get("", response_model=AuthorityState)
def get_authority(
    _admin: User = Depends(require_admin), db: Session = Depends(get_db)
) -> AuthorityState:
    return _state(GlobalMemoryEmbeddingAuthorityService(db).read_record())


# Declared, not inferred: the route reads the body itself.
_REQUEST_BODY = {
    "required": True,
    "content": {
        "application/json": {"schema": AuthorityConfiguration.model_json_schema()}
    },
}


@router.put(
    "", response_model=AuthorityState, openapi_extra={"requestBody": _REQUEST_BODY}
)
async def set_authority(
    request: Request,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> AuthorityState:
    """Parse and validate the body here instead of through the framework.

    The shared ``/api`` ``RequestValidationError`` handler echoes each error's
    raw ``input`` into the 422 body and logs it, and here that input is
    credential bearing: an ``api_key`` sent as a list or object comes back
    verbatim. Parsing here keeps every malformed body -- undecodable JSON
    included -- on the sanitized 400 path, which names no request value.
    """
    actor_subject = str(admin.actor_subject or "")
    if not actor_subject:
        raise HTTPException(409, detail="Admin actor identity is unavailable")
    try:
        config = AuthorityConfiguration.model_validate_json(await request.body())
    except ValueError:
        raise HTTPException(
            400, detail="Global memory embedding authority request is invalid"
        ) from None
    service = GlobalMemoryEmbeddingAuthorityService(db)
    try:
        record = service.set(config, actor_subject=actor_subject)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(400, detail=str(exc)) from exc
    except RuntimeError:
        db.rollback()
        raise HTTPException(
            503, detail="Global memory embedding authority could not be stored"
        ) from None
    return _state(record)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def delete_authority(
    _admin: User = Depends(require_admin), db: Session = Depends(get_db)
) -> Response:
    GlobalMemoryEmbeddingAuthorityService(db).delete()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
