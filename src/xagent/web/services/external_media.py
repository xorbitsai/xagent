"""Purpose-bound, short-lived delivery URLs for external media processors."""

from __future__ import annotations

import os
import stat
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote, urlencode, urlsplit
from uuid import UUID

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session

from ...config import (
    get_external_media_url_ttl_seconds,
    get_s2s_api_base_url,
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
    FACEBOOK_PUBLISH = "facebook-publish"


class ExternalMediaError(RuntimeError):
    """Base error with a caller-safe message."""


class ExternalMediaNotFound(ExternalMediaError):
    """The requested registered image is absent or outside the exact scope."""


class ExternalMediaUnavailable(ExternalMediaError):
    """The image exists but cannot currently be delivered externally."""


class ExternalMediaNotConfigured(ExternalMediaUnavailable):
    """This deployment has no public HTTPS base URL to deliver media from."""


class ExternalMediaUnsupportedImage(ExternalMediaUnavailable):
    """The image's type, size or encoding cannot be delivered for this purpose."""


class ExternalMediaOutOfScope(ExternalMediaUnavailable):
    """External media cannot reach the image's bytes within its own scope.

    It trusts a local copy only inside the user's upload root with no symbolic
    link below it, and otherwise needs the durable object. A workspace may
    still read a local copy elsewhere under its own rules.
    """


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


# Matches the Facebook connector's own limit for an uploaded image.
FACEBOOK_MAX_IMAGE_BYTES = 10 * 1024 * 1024
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
        raise ExternalMediaUnsupportedImage("Only PNG or JPEG images can be delivered")
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
    base_url = get_s2s_api_base_url()
    if not base_url:
        raise ExternalMediaNotConfigured(
            "A public HTTPS XAGENT_S2S_API_BASE_URL or XAGENT_PUBLIC_API_BASE_URL "
            "is required for external media delivery"
        )
    if urlsplit(base_url).scheme != "https":
        raise ExternalMediaNotConfigured(
            "XAGENT_S2S_API_BASE_URL (or XAGENT_PUBLIC_API_BASE_URL when S2S is "
            "unset) must be HTTPS for external media delivery"
        )
    if purpose is ExternalMediaPurpose.FACEBOOK_PUBLISH:
        _validate_facebook_record(record)
    else:
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
        url=f"{base_url.rstrip('/')}{path}?{urlencode({'ticket': ticket})}",
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


def _below_user_root(record: UploadedFile, path: Path) -> tuple[Path, Path] | None:
    """The owner's upload root a path lies in, and the part of the path below it.

    The root may be a link (a mounted volume can be), so it is tried both as
    configured and resolved; it must still resolve inside the uploads directory.
    """
    uploads_root = get_uploads_dir().resolve()
    user_root = scoped_user_root(get_uploads_dir(), int(record.user_id))
    candidate = Path(os.path.abspath(path))
    for base in (Path(os.path.abspath(user_root)), user_root.resolve()):
        try:
            below = candidate.relative_to(base)
        except ValueError:
            continue
        if not base.resolve().is_relative_to(uploads_root):
            return None
        return base, below
    return None


def local_record_path_is_trusted(record: UploadedFile, path: Path) -> bool:
    """Whether a local copy may be delivered externally instead of the durable one.

    The path must lie inside the record owner's upload root, and nothing below
    that root may be a symbolic link: a registered file replaced by a link
    would otherwise deliver another task's image. The root itself may be a
    link, as a mounted volume can be. Delivery opens the file again with
    ``open_delivery_file``, which holds the same rule at the moment of opening.
    """
    anchored = _below_user_root(record, path)
    if anchored is None:
        return False
    current, below = anchored
    for part in below.parts:
        current = current / part
        if current.is_symlink():
            return False
    return True


def open_delivery_file(record: UploadedFile, path: Path) -> BinaryIO:
    """Open the file an external-media response will send, without following links.

    Checking a path and opening it are separate steps, and a file replaced by a
    link in between would send the link's target. Below the owner's upload root
    each component is opened relative to its parent's descriptor with
    ``O_NOFOLLOW``, so the bytes come from the file the check allowed, or the
    open fails. The root itself may be a link. Any other path is a host-private
    copy from durable storage; only the file itself must not be a link.

    Raises ``OSError`` when the file cannot be opened this way or is not a
    regular file.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    anchored = _below_user_root(record, path)
    names: tuple[str, ...]
    if anchored is None:
        directory, names = Path(os.path.abspath(path)).parent, (path.name,)
    else:
        directory, below = anchored
        names = below.parts
    if not names:
        raise IsADirectoryError(str(path))
    current_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in names[:-1]:
            next_fd = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=current_fd
            )
            os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(
            names[-1],
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | nofollow,
            dir_fd=current_fd,
        )
    finally:
        os.close(current_fd)
    if not stat.S_ISREG(os.fstat(file_fd).st_mode):
        os.close(file_fd)
        raise FileNotFoundError(str(path))
    return os.fdopen(file_fd, "rb")


def _source_size(source: Path | BinaryIO) -> int:
    if isinstance(source, Path):
        return source.stat().st_size
    return os.fstat(source.fileno()).st_size


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


def _materialize_record(record: UploadedFile) -> Path:
    file_ref = ManagedFileRef(record)
    try:
        path = file_ref.materialize(
            allow_existing_local=local_record_path_is_trusted(
                record,
                file_ref.local_path,
            )
        )
    except DurableObjectIntegrityError as exc:
        raise ExternalMediaUnavailable(
            "Registered image failed integrity verification"
        ) from exc
    except (DurableObjectMissingError, FilePublicationUnavailable) as exc:
        raise ExternalMediaOutOfScope("Registered image bytes are unavailable") from exc
    except DurableStorageOperationError as exc:
        raise ExternalMediaUnavailable(
            "Registered image could not be prepared for external delivery"
        ) from exc
    return path


def _validate_instagram_record(record: UploadedFile) -> None:
    _validate_instagram_header(_materialize_record(record))


def _validate_facebook_record(record: UploadedFile) -> None:
    """Facebook fetches the image as stored: a whole PNG or JPEG within 10 MB.

    Reading and decoding fail differently on purpose. An unreadable file is a
    storage problem (base ``ExternalMediaUnavailable``); a file that reads but
    is the wrong type, too large or does not decode is an unsupported image,
    which a connector able to upload the bytes itself may still take.
    """

    _image_media_type(record)
    path = _materialize_record(record)
    try:
        if path.stat().st_size > FACEBOOK_MAX_IMAGE_BYTES:
            raise ExternalMediaUnsupportedImage(
                "Registered image exceeds the 10 MB limit"
            )
        content = path.read_bytes()
    except OSError as exc:
        raise ExternalMediaUnavailable(
            "Registered image could not be read for external delivery"
        ) from exc
    if len(content) > FACEBOOK_MAX_IMAGE_BYTES:
        raise ExternalMediaUnsupportedImage("Registered image exceeds the 10 MB limit")
    try:
        # verify() checks the whole stream (a truncated body fails) without
        # decoding pixels. The bytes are in memory, so any error is a decode error.
        with Image.open(BytesIO(content), formats=_INSTAGRAM_IMAGE_FORMATS) as image:
            image.verify()
    except Exception as exc:
        raise ExternalMediaUnsupportedImage(
            "Registered image could not be decoded as PNG or JPEG"
        ) from exc


def instagram_image_requires_normalization(source: Path | BinaryIO) -> bool:
    """Return whether a local image must be normalized before Instagram fetches it."""

    try:
        file_size = _source_size(source)
        with Image.open(source, formats=_INSTAGRAM_IMAGE_FORMATS) as image:
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


def instagram_jpeg_bytes(source: Path | BinaryIO) -> bytes:
    """Normalize a local image to Instagram's JPEG and aspect-ratio contract."""

    try:
        with Image.open(source, formats=_INSTAGRAM_IMAGE_FORMATS) as opened:
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
