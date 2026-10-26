"""Purpose-bound, short-lived delivery URLs for external media processors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
from uuid import UUID

from sqlalchemy.orm import Session

from ...config import (
    get_external_media_url_ttl_seconds,
    get_file_delivery_redirect_enabled,
    get_public_api_base_url,
)
from ...core.file_ref import parse_file_id_ref
from .. import jwt_codec as jwt
from ..auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from ..jwt_codec import JWTError
from ..models.uploaded_file import UploadedFile
from .managed_file_ref import (
    DurableObjectIntegrityError,
    DurableStorageOperationError,
    ManagedFileRef,
)

EXTERNAL_MEDIA_TICKET_TYPE = "external_media_ticket"
EXTERNAL_MEDIA_ROUTE_PREFIX = "/api/files/external-media"
_IMAGE_MEDIA_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}


class ExternalMediaPurpose(str, Enum):
    """Providers allowed to receive a purpose-scoped media URL."""

    INSTAGRAM_PUBLISH = "instagram-publish"


class ExternalMediaError(RuntimeError):
    """Base error with a caller-safe message."""


class ExternalMediaNotFound(ExternalMediaError):
    """The requested registered image is absent or outside the exact scope."""


class ExternalMediaUnavailable(ExternalMediaError):
    """The image exists but cannot currently be delivered externally."""


class ExternalMediaTicketRejected(ExternalMediaError):
    """A public delivery ticket is invalid, expired, or scope-mismatched."""


@dataclass(frozen=True)
class ExternalMediaURL:
    url: str
    expires_in: int
    direct_object_url: bool


@dataclass(frozen=True)
class RedeemedExternalMedia:
    record: UploadedFile
    purpose: ExternalMediaPurpose
    media_type: str


def _normalized_file_id(value: str) -> str:
    raw = str(value).strip()
    normalized = parse_file_id_ref(raw) or raw
    try:
        return str(UUID(normalized))
    except (ValueError, AttributeError):
        raise ExternalMediaNotFound("Registered image was not found") from None


def _image_media_type(record: Any) -> str:
    filename = str(getattr(record, "filename", "") or "").strip().lower()
    media_type = _IMAGE_MEDIA_TYPES.get(Path(filename).suffix)
    if media_type is None:
        raise ExternalMediaNotFound("Registered image was not found")
    return media_type


def _load_record(
    db: Session,
    *,
    file_id: str,
    user_id: int,
    task_id: int,
) -> UploadedFile:
    record = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.file_id == file_id,
            UploadedFile.user_id == user_id,
            UploadedFile.task_id == task_id,
        )
        .first()
    )
    if (
        record is None
        or int(record.user_id) != user_id
        or int(record.task_id or 0) != task_id
        or record.storage_status not in {"available", "legacy"}
    ):
        raise ExternalMediaNotFound("Registered image was not found")
    _image_media_type(record)
    return record


def _encode_ticket(
    *,
    file_id: str,
    user_id: int,
    task_id: int,
    purpose: ExternalMediaPurpose,
    ttl: int,
) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
    return jwt.encode(
        {
            "type": EXTERNAL_MEDIA_TICKET_TYPE,
            "purpose": purpose.value,
            "file_id": file_id,
            "user_id": user_id,
            "task_id": task_id,
            "exp": expires_at,
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def issue_external_media_url(
    db: Session,
    *,
    file_id: str,
    user_id: int,
    task_id: int,
    purpose: ExternalMediaPurpose,
) -> ExternalMediaURL:
    """Issue one exact-file URL without exposing arbitrary host paths."""

    normalized_file_id = _normalized_file_id(file_id)
    record = _load_record(
        db,
        file_id=normalized_file_id,
        user_id=int(user_id),
        task_id=int(task_id),
    )
    media_type = _image_media_type(record)
    ttl = get_external_media_url_ttl_seconds()
    signed_url = None
    if get_file_delivery_redirect_enabled():
        try:
            signed_url = ManagedFileRef(record).signed_access_url(
                expires=ttl,
                content_type=media_type,
            )
        except DurableObjectIntegrityError as exc:
            raise ExternalMediaUnavailable(
                "Registered image failed integrity verification"
            ) from exc
        except DurableStorageOperationError as exc:
            raise ExternalMediaUnavailable(
                "Registered image could not be prepared for external delivery"
            ) from exc
    if signed_url and urlsplit(signed_url).scheme == "https":
        return ExternalMediaURL(
            url=signed_url,
            expires_in=ttl,
            direct_object_url=True,
        )

    public_base_url = get_public_api_base_url()
    if not public_base_url or urlsplit(public_base_url).scheme != "https":
        raise ExternalMediaUnavailable(
            "A public HTTPS XAGENT_PUBLIC_API_BASE_URL is required for external "
            "media delivery"
        )
    ticket = _encode_ticket(
        file_id=normalized_file_id,
        user_id=int(user_id),
        task_id=int(task_id),
        purpose=purpose,
        ttl=ttl,
    )
    path = f"{EXTERNAL_MEDIA_ROUTE_PREFIX}/{quote(normalized_file_id, safe='')}"
    return ExternalMediaURL(
        url=f"{public_base_url.rstrip('/')}{path}?{urlencode({'ticket': ticket})}",
        expires_in=ttl,
        direct_object_url=False,
    )


def redeem_external_media_ticket(
    db: Session,
    *,
    file_id: str,
    ticket: str,
) -> RedeemedExternalMedia:
    """Validate a public ticket and return its exact registered image."""

    try:
        normalized_file_id = _normalized_file_id(file_id)
        claims = jwt.decode(
            ticket,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_sub": False},
        )
    except (ExternalMediaNotFound, JWTError):
        raise ExternalMediaTicketRejected(
            "Invalid or expired external media ticket"
        ) from None

    user_id = claims.get("user_id")
    task_id = claims.get("task_id")
    try:
        purpose = ExternalMediaPurpose(claims.get("purpose"))
    except (TypeError, ValueError):
        purpose = None
    if (
        claims.get("type") != EXTERNAL_MEDIA_TICKET_TYPE
        or claims.get("file_id") != normalized_file_id
        or purpose is None
        or type(user_id) is not int
        or type(task_id) is not int
    ):
        raise ExternalMediaTicketRejected("Invalid or expired external media ticket")

    try:
        record = _load_record(
            db,
            file_id=normalized_file_id,
            user_id=user_id,
            task_id=task_id,
        )
    except ExternalMediaNotFound:
        raise ExternalMediaTicketRejected(
            "Invalid or expired external media ticket"
        ) from None
    return RedeemedExternalMedia(
        record=record,
        purpose=purpose,
        media_type=_image_media_type(record),
    )
