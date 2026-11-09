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
from typing import IO, Any, BinaryIO
from urllib.parse import quote, urlencode, urlsplit
from uuid import UUID

from PIL import ExifTags, Image, ImageFile, ImageOps, JpegImagePlugin, UnidentifiedImageError
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
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_JPEG_END_OF_IMAGE = b"\xff\xd9"


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
# A source may be larger than the JPEG Meta accepts, since it is converted
# first (a large generated PNG, say). This limit bounds what Pillow reads: it
# reads a PNG chunk or JPEG segment it does not decode in full, and keeps some,
# so its memory follows the file's size, not its pixels.
INSTAGRAM_MAX_SOURCE_BYTES = 50 * 1024 * 1024
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


def delivered_image_media_type(content: bytes, registered_type: str) -> str:
    """The type of the image bytes being delivered, read from their signature.

    Issuance accepts PNG or JPEG bytes under any of .png, .jpg or .jpeg, and
    image_tool saves every edited image as .png, so the extension can name the
    wrong one of the two; delivery sends nosniff, so the declared type is
    final. Bytes with neither signature keep the registered type.
    """
    if content.startswith(_PNG_SIGNATURE):
        return "image/png"
    if content.startswith(_JPEG_SIGNATURE):
        return "image/jpeg"
    return registered_type


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


# SOFn markers carry the frame header; C4 (DHT), C8 (JPG) and CC (DAC) do not.
_JPEG_FRAME_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# TEM and RST0-RST7 stand alone, with no length.
_JPEG_MARKERS_WITHOUT_LENGTH = frozenset({0x01, *range(0xD0, 0xD8)})


def _jpeg_first_scan_omits_components(stream: IO[bytes] | None) -> bool:
    """Whether a JPEG's first scan codes fewer components than its frame.

    libjpeg then keeps coefficients for the whole image, as for a progressive
    JPEG (``has_multiple_scans`` in its ``jdinput.c``). The markers are read up
    to the first scan as libjpeg reads them: stray bytes, FF fill bytes and
    stuffed FF00 pairs before a marker are skipped, and each segment by its
    length. Markers that cannot be read that far count as omitting components,
    so they never earn the higher limit. The stream's position is kept.
    """
    if stream is None:
        return True
    position = stream.tell()
    try:
        stream.seek(0)
        if stream.read(2) != b"\xff\xd8":
            return True
        frame_components = None
        while True:
            byte = stream.read(1)
            if byte != b"\xff":
                if not byte:
                    return True
                continue
            while byte == b"\xff":
                byte = stream.read(1)
            if not byte or byte in (b"\xd8", b"\xd9"):
                return True
            marker = byte[0]
            if marker == 0 or marker in _JPEG_MARKERS_WITHOUT_LENGTH:
                continue
            header = stream.read(2)
            length = int.from_bytes(header, "big") - 2
            if len(header) < 2 or length < 0:
                return True
            if marker == 0xDA:
                scan = stream.read(1)
                return (
                    frame_components is None or not scan or scan[0] < frame_components
                )
            if marker in _JPEG_FRAME_MARKERS:
                frame = stream.read(length)
                if len(frame) < 6:
                    return True
                frame_components = frame[5]
            else:
                stream.seek(length, os.SEEK_CUR)
    finally:
        stream.seek(position)


def _validate_instagram_source(image: ImageFile.ImageFile) -> None:
    if image.width <= 0 or image.height <= 0:
        raise ExternalMediaUnavailable("Registered image has invalid dimensions")
    # The higher JPEG limit relies on draft() letting libjpeg decode a JPEG at
    # a fraction of its size. That takes a draft target well below the image's
    # size, which instagram_jpeg_bytes() asks for by drafting to the size it
    # sends, and a JPEG coded in one scan. With more than one (progressive, or
    # a first scan that codes only some components) libjpeg keeps coefficients
    # for the whole image whatever draft() asks for: about 370 MB at 100
    # million pixels, and about 650 MB without chroma subsampling. Those JPEGs
    # get the PNG limit.
    if image.format != "JPEG":
        pixel_limit, kind = INSTAGRAM_MAX_SOURCE_PIXELS, str(image.format)
    elif image.info.get("progressive"):
        pixel_limit, kind = INSTAGRAM_MAX_SOURCE_PIXELS, "progressive JPEG"
    elif _jpeg_first_scan_omits_components(image.fp):
        pixel_limit, kind = INSTAGRAM_MAX_SOURCE_PIXELS, "multi-scan JPEG"
    else:
        pixel_limit, kind = INSTAGRAM_MAX_JPEG_SOURCE_PIXELS, "single-scan JPEG"
    if image.width * image.height > pixel_limit:
        raise ExternalMediaUnavailable(
            "Registered image exceeds the Instagram source pixel limit of "
            f"{pixel_limit:,} for {kind}; downscale it and try again"
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
    """Open a registered image for issuing or delivery, without following links.

    Checking a path and opening it are separate steps, and a file replaced by a
    link in between would read the link's target. Below the owner's upload root
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
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise FileNotFoundError(str(path))
        return os.fdopen(file_fd, "rb")
    except BaseException:
        os.close(file_fd)
        raise


def read_instagram_source(source: BinaryIO) -> BytesIO:
    """Read an opened Instagram source into memory, within its byte limit.

    Pillow is given this copy, never the file, so it cannot read more than the
    limit. A file already over the limit is refused unread; reading at most one
    byte past it catches a file the sandbox grows or rewrites after that check.

    Raises ``ExternalMediaUnavailable`` when the file is over the limit (the
    route answers it with 422; it is not an ``OSError``, so issuance's
    ``except OSError`` lets it through), and ``OSError`` when the file cannot
    be read.
    """
    limit = INSTAGRAM_MAX_SOURCE_BYTES
    if os.fstat(source.fileno()).st_size <= limit:
        content = source.read(limit + 1)
        if len(content) <= limit:
            return BytesIO(content)
    raise ExternalMediaUnavailable(
        f"Registered image exceeds the {limit // (1024 * 1024)} MB Instagram "
        "source limit; compress or downscale it and try again"
    )


def _validate_instagram_header(source: BinaryIO) -> None:
    try:
        with Image.open(source, formats=_INSTAGRAM_IMAGE_FORMATS) as image:
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
    path = _materialize_record(record)
    try:
        # The same safe open as delivery: a FIFO or link swapped in after
        # materialization fails here instead of blocking or being followed.
        with open_delivery_file(record, path) as source:
            content = read_instagram_source(source)
    except OSError as exc:
        raise ExternalMediaUnavailable(
            "Registered image could not be read for external delivery"
        ) from exc
    _validate_instagram_header(content)


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
        # Pin the opened file and bound the read even if a sandbox writer grows it.
        with open_delivery_file(record, path) as source:
            content = source.read(FACEBOOK_MAX_IMAGE_BYTES + 1)
    except OSError as exc:
        raise ExternalMediaUnavailable(
            "Registered image could not be read for external delivery"
        ) from exc
    if len(content) > FACEBOOK_MAX_IMAGE_BYTES:
        raise ExternalMediaUnsupportedImage("Registered image exceeds the 10 MB limit")
    try:
        # Neither check decodes pixels. PNG's verify() reads every chunk and
        # checks its CRC, so a truncated or damaged PNG fails. JPEG's verify()
        # only parses the header, so the main image must also be seen to end;
        # a JPEG damaged in the middle still passes. The bytes are in memory,
        # so any error is a decode error.
        with Image.open(BytesIO(content), formats=_INSTAGRAM_IMAGE_FORMATS) as image:
            is_jpeg = image.format in {"JPEG", "MPO"}
            image.verify()
        if is_jpeg and not _jpeg_main_image_ends(content):
            raise SyntaxError("JPEG ends before its main image does")
    except Exception as exc:
        raise ExternalMediaUnsupportedImage(
            "Registered image could not be decoded as PNG or JPEG"
        ) from exc


def _jpeg_main_image_ends(content: bytes) -> bool:
    """Whether the end-of-image marker follows the main image's scan data.

    Pillow's JPEG open stops right after the first start-of-scan header, so
    the position it leaves is where the main image's scan data begins. An EXIF
    thumbnail ends with its own marker before that point and does not count;
    data after the marker, such as extra MPF images, is allowed. The plugin is
    opened directly because ``Image.open`` turns a JPEG carrying MPF images
    into an ``MpoImageFile``, which rewinds to the start of the file.
    """
    buffer = BytesIO(content)
    with JpegImagePlugin.JpegImageFile(buffer):
        scan_start = buffer.tell()
    return content.find(_JPEG_END_OF_IMAGE, scan_start) != -1


def instagram_image_requires_normalization(source: BytesIO) -> bool:
    """Return whether a local image must be normalized before Instagram fetches it.

    ``source`` is the bounded copy ``read_instagram_source`` returns, so Pillow
    never reads the file itself.
    """

    try:
        with source.getbuffer() as content:
            file_size = content.nbytes
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


def _instagram_fitted_size(width: int, height: int) -> tuple[int, int]:
    """An upright size scaled down to Instagram's maximum width and height.

    It is never scaled up.
    """
    max_height = floor(INSTAGRAM_MAX_WIDTH / INSTAGRAM_MIN_ASPECT_RATIO)
    scale = min(1.0, INSTAGRAM_MAX_WIDTH / width, max_height / height)
    return max(1, floor(width * scale)), max(1, floor(height * scale))


def _resize_for_instagram_canvas(
    image: Image.Image, upright_size: tuple[int, int]
) -> Image.Image:
    # Fit from the source's full upright size, not the decoded one: a drafted
    # JPEG is decoded smaller, and the size sent must not depend on that.
    fitted = _instagram_fitted_size(*upright_size)
    if image.size != fitted:
        image = image.resize(fitted, Image.Resampling.LANCZOS)

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


def instagram_jpeg_bytes(source: BytesIO) -> bytes:
    """Normalize a local image to Instagram's JPEG and aspect-ratio contract.

    ``source`` is the bounded copy ``read_instagram_source`` returns.
    """

    try:
        with Image.open(source, formats=_INSTAGRAM_IMAGE_FORMATS) as opened:
            _validate_instagram_source(opened)
            # Orientations 5-8 swap width and height, as exif_transpose() does.
            width, height = opened.size
            turned = opened.getexif().get(ExifTags.Base.Orientation, 1) in (5, 6, 7, 8)
            upright_size = (height, width) if turned else (width, height)
            if opened.format == "JPEG":
                # libjpeg scales a single-scan JPEG down by up to 8 toward the
                # draft target, so ask for the size it is sent at. A fixed
                # target box left a long, narrow image at full size. libjpeg
                # does not scale a lossless JPEG, so drafting one by 2 or more
                # fails its decode; that refusal is accepted (#3596).
                fitted_width, fitted_height = _instagram_fitted_size(*upright_size)
                opened.draft(
                    "RGB",
                    (fitted_height, fitted_width)
                    if turned
                    else (fitted_width, fitted_height),
                )
            image = ImageOps.exif_transpose(opened)
            image.load()

        if image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            flattened = Image.new("RGB", rgba.size, "white")
            flattened.paste(rgba, mask=rgba.getchannel("A"))
            image = flattened
        else:
            image = image.convert("RGB")

        image = _resize_for_instagram_canvas(image, upright_size)
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
