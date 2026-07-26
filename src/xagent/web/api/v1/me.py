"""GET /v1/me -- personal management key identity probe."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)

router = APIRouter()


class MeResponse(BaseModel):
    """Response model for ``GET /v1/me``."""

    principal_type: str = Field(default="user", description="Authenticated principal.")
    user_id: int = Field(..., description="User bound to the presented personal key.")
    username: str = Field(..., description="User's unique login name.")
    email: str | None = Field(
        default=None, description="User's email address, or null if unset."
    )
    key_prefix: str = Field(
        ...,
        description=(
            "Public-safe 6-char lookup handle of the presented key. "
            "Lets the SDK log which key is in use without exposing the secret."
        ),
    )


@router.get("/me", response_model=MeResponse)
async def get_me(
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> MeResponse:
    """Probe the user identity bound to the caller's personal key."""
    user, key = authed
    return MeResponse(
        principal_type="user",
        user_id=int(user.id),
        username=str(user.username),
        email=str(user.email) if user.email is not None else None,
        key_prefix=key.key_prefix,
    )
