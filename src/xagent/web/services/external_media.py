"""Purpose-bound, short-lived delivery URLs for external media processors."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
from uuid import UUID

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session

from ...config import (
    get_external_media_url_ttl_seconds,
    get_public_api_base_url,
    get_uploads_dir,
)
from ...core.file_ref import parse_file_id_ref
from ...core.workspace import scoped_user_root
from .. import jwt_codec as jwt
from ..auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from ..jwt_codec import JWTError
from ..models.uploaded_file import UploadedFile
from .managed_file_ref import (
    DurableObjectIntegrityError,
    DurableObjectMissingError,
    DurableStorageOperationError,
    ManagedFileRef,
)
from .uploaded_file_cleanup_publication import FilePublicationUnavailable

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


INSTAGRAM_MAX_JPEG_BYTES = 8 * 1024 * 1024
INSTAGRAM_MIN_ASPECT_RATIO = 4 / 5
INSTAGRAM_MAX_ASPECT_RATIO = 1.91
INSTAGRAM_MIN_WIDTH = 320
INSTAGRAM_MAX_WIDTH = 1440
INSTAGRAM_MAX_SOURCE_PIXELS = 25_000_000
INSTAGRAM_MAX_JPEG_SOURCE_PIXELS = 100_000_000
_INSTAGRAM_IMAGE_FORMATS = ["PNG", "JPEG"]
_INSTAGRAM_PROCESSING_ERRORS = (
    OSError,
    UnidentifiedImageError,
    Image.DecompressionBombError,
    ValueError,
    MemoryError,
    SyntaxError,
    struct.error,
)


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
        raise ExternalMediaUnavailable("Instagram accepts PNG or JPEG images")
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
    ttl = get_external_media_url_ttl_seconds()
    public_base_url = get_public_api_base_url()
    if not public_base_url or urlsplit(public_base_url).scheme != "https":
        raise ExternalMediaUnavailable(
            "A public HTTPS XAGENT_PUBLIC_API_BASE_URL is required for external "
            "media delivery"
        )
    _validate_instagram_record(record)
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


def _validate_instagram_source(image: Image.Image) -> None:
    if image.width <= 0 or image.height <= 0:
        raise ExternalMediaUnavailable("Registered image has invalid dimensions")
    pixel_limit = (
        INSTAGRAM_MAX_JPEG_SOURCE_PIXELS
        if image.format == "JPEG"
        else INSTAGRAM_MAX_SOURCE_PIXELS
    )
    if image.width * image.height > pixel_limit:
        raise ExternalMediaUnavailable(
            "Registered image exceeds the Instagram source pixel limit"
        )


def _local_record_path_is_trusted(record: UploadedFile, path: Path) -> bool:
    uploads_root = get_uploads_dir().resolve()
    user_root = scoped_user_root(get_uploads_dir(), int(record.user_id)).resolve()
    resolved = path.resolve()
    return resolved.is_relative_to(uploads_root) and resolved.is_relative_to(user_root)


def _validate_instagram_header(path: Path) -> None:
    try:
        with Image.open(path, formats=_INSTAGRAM_IMAGE_FORMATS) as image:
            _validate_instagram_source(image)
    except ExternalMediaUnavailable:
        raise
    except _INSTAGRAM_PROCESSING_ERRORS as exc:
        raise ExternalMediaUnavailable(
            "Registered image could not be decoded for Instagram"
        ) from exc


def _validate_instagram_record(record: UploadedFile) -> None:
    file_ref = ManagedFileRef(record)
    try:
        path = file_ref.materialize(
            allow_existing_local=_local_record_path_is_trusted(
                record,
                file_ref.local_path,
            )
        )
    except DurableObjectIntegrityError as exc:
        raise ExternalMediaUnavailable(
            "Registered image failed integrity verification"
        ) from exc
    except (DurableObjectMissingError, FilePublicationUnavailable) as exc:
        raise ExternalMediaUnavailable(
            "Registered image bytes are unavailable"
        ) from exc
    except DurableStorageOperationError as exc:
        raise ExternalMediaUnavailable(
            "Registered image could not be prepared for external delivery"
        ) from exc
    _validate_instagram_header(path)


def instagram_image_requires_normalization(path: Path) -> bool:
    """Return whether a local image must be normalized before Instagram fetches it."""

    try:
        file_size = path.stat().st_size
        with Image.open(path, formats=_INSTAGRAM_IMAGE_FORMATS) as image:
            _validate_instagram_source(image)
            if image.format != "JPEG":
                return True
            orientation = image.getexif().get(ExifTags.Base.Orientation, 1)
            ratio = image.width / image.height
            requires_normalization = (
                image.mode != "RGB"
                or orientation != 1
                or file_size > INSTAGRAM_MAX_JPEG_BYTES
                or not INSTAGRAM_MIN_WIDTH <= image.width <= INSTAGRAM_MAX_WIDTH
                or not INSTAGRAM_MIN_ASPECT_RATIO <= ratio <= INSTAGRAM_MAX_ASPECT_RATIO
            )
            if requires_normalization:
                return True
            image.load()
            return False
    except ExternalMediaUnavailable:
        raise
    except _INSTAGRAM_PROCESSING_ERRORS as exc:
        raise ExternalMediaUnavailable(
            "Registered image could not be decoded for Instagram"
        ) from exc


def _instagram_canvas_size(image: Image.Image) -> tuple[int, int]:
    ratio = image.width / image.height
    if ratio < INSTAGRAM_MIN_ASPECT_RATIO:
        return ceil(image.height * INSTAGRAM_MIN_ASPECT_RATIO), image.height
    if ratio > INSTAGRAM_MAX_ASPECT_RATIO:
        return image.width, ceil(image.width / INSTAGRAM_MAX_ASPECT_RATIO)
    return image.size


def _resize_for_instagram_canvas(image: Image.Image) -> Image.Image:
    max_height = floor(INSTAGRAM_MAX_WIDTH / INSTAGRAM_MIN_ASPECT_RATIO)
    scale = min(
        1.0,
        INSTAGRAM_MAX_WIDTH / image.width,
        max_height / image.height,
    )
    if scale < 1.0:
        image = image.resize(
            (
                max(1, floor(image.width * scale)),
                max(1, floor(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    canvas_width, _ = _instagram_canvas_size(image)
    if canvas_width < INSTAGRAM_MIN_WIDTH:
        scale = INSTAGRAM_MIN_WIDTH / canvas_width
        if image.width / image.height < INSTAGRAM_MIN_ASPECT_RATIO:
            minimum_height = ceil(INSTAGRAM_MIN_WIDTH / INSTAGRAM_MIN_ASPECT_RATIO)
            scale = max(scale, minimum_height / image.height)
        image = image.resize(
            (
                max(1, ceil(image.width * scale)),
                max(1, ceil(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    return image


def instagram_jpeg_bytes(path: Path) -> bytes:
    """Normalize a local image to Instagram's JPEG and aspect-ratio contract."""

    try:
        with Image.open(path, formats=_INSTAGRAM_IMAGE_FORMATS) as opened:
            _validate_instagram_source(opened)
            if opened.format == "JPEG":
                opened.draft("RGB", (INSTAGRAM_MAX_WIDTH, 1800))
            image = ImageOps.exif_transpose(opened)
            image.load()

        if image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            flattened = Image.new("RGB", rgba.size, "white")
            flattened.paste(rgba, mask=rgba.getchannel("A"))
            image = flattened
        else:
            image = image.convert("RGB")

        image = _resize_for_instagram_canvas(image)
        canvas_size = _instagram_canvas_size(image)
        if canvas_size != image.size:
            canvas = Image.new("RGB", canvas_size, "white")
            offset = (
                (canvas.width - image.width) // 2,
                (canvas.height - image.height) // 2,
            )
            canvas.paste(image, offset)
            image = canvas

        for quality in (92, 85, 75, 65):
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            data = output.getvalue()
            if len(data) <= INSTAGRAM_MAX_JPEG_BYTES:
                return data
    except ExternalMediaUnavailable:
        raise
    except _INSTAGRAM_PROCESSING_ERRORS as exc:
        raise ExternalMediaUnavailable(
            "Registered image could not be normalized for Instagram"
        ) from exc
    raise ExternalMediaUnavailable(
        "Registered image remains too large for Instagram after JPEG conversion"
    )
