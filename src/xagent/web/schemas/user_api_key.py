"""Schemas for user personal SDK management keys."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..utils.db_timezone import normalize_datetime_from_db


class _PersonalAPIKeyTimestampResponse(BaseModel):
    @field_validator(
        "created_at",
        "expires_at",
        "revoked_at",
        check_fields=False,
    )
    @classmethod
    def _normalize_api_timestamp(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        return normalize_datetime_from_db(value)


class PersonalAPIKeyCreateResponse(_PersonalAPIKeyTimestampResponse):
    id: int
    full_key: str = Field(
        ...,
        description=(
            "Plaintext personal key in the format "
            "xag_personal_<6 chars>_<32 chars>. Returned exactly once."
        ),
    )
    key_prefix: str
    created_at: datetime
    expires_at: Optional[datetime] = None


class PersonalAPIKeyMetadata(_PersonalAPIKeyTimestampResponse):
    id: int
    key_prefix: str
    masked_key: str
    revoked_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime


class PersonalAPIKeyRevokeResponse(_PersonalAPIKeyTimestampResponse):
    revoked: bool
    revoked_at: Optional[datetime] = None


class PersonalAPIKeyOwner(BaseModel):
    id: int
    username: str
    email: Optional[str] = None


PersonalAPIKeyStatus = Literal["active", "expired", "revoked"]


class PersonalAPIKeyListItem(PersonalAPIKeyMetadata):
    status: PersonalAPIKeyStatus
    owner: PersonalAPIKeyOwner


class PersonalAPIKeyListResponse(BaseModel):
    items: list[PersonalAPIKeyListItem]
    can_manage_others: bool
