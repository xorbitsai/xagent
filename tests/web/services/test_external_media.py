import os
import struct
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from PIL import Image, JpegImagePlugin
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests.web.services.external_media_source_shared import (
    LOSSLESS_JPEG,
    MULTI_SCAN_PIXEL_LIMIT_CASES,
    OVERSIZED,
    OVERSIZED_SOURCES,
    SOURCE_LIMIT,
    count_source_reads,
    jpeg_400px,
    write_png_with_private_chunk,
    write_small_image,
)
from xagent.web import models
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import external_media


@pytest.fixture(autouse=True)
def _isolate_s2s_base(monkeypatch):
    monkeypatch.delenv("XAGENT_S2S_API_BASE_URL", raising=False)


class _DB:
    """Small real SQLite session holder used by service-level tests."""

    def __init__(self, record: UploadedFile):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        models.Base.metadata.create_all(bind=self.engine)
        self.session = sessionmaker(bind=self.engine)()
        self.record = self.session.merge(record)
        self.session.commit()

    def query(self, model):
        return self.session.query(model)


def _record(
    file_id: str,
    *,
    filename: str | None = None,
    mime_type: str = "image/png",
    storage_path: str = "/unused/launch.png",
    storage_status: str = "available",
    storage_key: str | None = "users/7/launch.png",
) -> UploadedFile:
    filename = filename or ("launch.jpg" if mime_type == "image/jpeg" else "launch.png")
    return UploadedFile(
        file_id=file_id,
        user_id=7,
        task_id=11,
        filename=filename,
        mime_type=mime_type,
        storage_status=storage_status,
        storage_path=storage_path,
        storage_key=storage_key,
        checksum="a" * 64,
        file_size=1,
    )


def test_issue_instagram_uses_public_ticket_url(monkeypatch):
    file_id = str(uuid4())
    record = _record(file_id, mime_type="image/jpeg")
    monkeypatch.delenv("XAGENT_EXTERNAL_MEDIA_URL_TTL_SECONDS", raising=False)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    validate = Mock()
    monkeypatch.setattr(external_media, "_validate_instagram_record", validate)

    issued = external_media.issue_external_media_url(
        _DB(record),
        file_id=f"file:{file_id}",
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    )

    assert issued.direct_object_url is False
    assert issued.expires_in == 600
    assert issued.url.startswith("https://api.example.test/")
    validate.assert_called_once()


def test_issue_returns_public_ticket_url(monkeypatch):
    file_id = str(uuid4())
    record = _record(file_id)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test/")
    monkeypatch.setattr(external_media, "_validate_instagram_record", Mock())

    issued = external_media.issue_external_media_url(
        _DB(record),
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    )

    parsed = urlsplit(issued.url)
    ticket = parse_qs(parsed.query)["ticket"][0]
    assert issued.direct_object_url is False
    assert parsed.path == f"/api/files/external-media/{file_id}"
    redeemed = external_media.redeem_external_media_ticket(
        _DB(record), file_id=file_id, ticket=ticket
    )
    assert redeemed.record.file_id == record.file_id
    assert redeemed.purpose is external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH
    assert redeemed.media_type == "image/png"


def test_issue_requires_public_base_for_local_storage(monkeypatch):
    file_id = str(uuid4())
    monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)

    with pytest.raises(
        external_media.ExternalMediaUnavailable,
        match="XAGENT_PUBLIC_API_BASE_URL",
    ):
        external_media.issue_external_media_url(
            _DB(_record(file_id)),
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )


def _count_encode_calls(monkeypatch):
    calls = []
    real = external_media._encode_ticket

    def counting(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(external_media, "_encode_ticket", counting)
    return calls


@pytest.mark.parametrize(
    "public", [None, "https://api.example.test", "http://api.example.test", "ftp://x"]
)
@pytest.mark.parametrize(
    "s2s",
    [
        "https://s2s.example.test",
        "https://s2s.example.test/",
        "https://s2s.example.test/base",
    ],
)
def test_issue_prefers_s2s_base(monkeypatch, s2s, public):
    file_id = str(uuid4())
    record = _record(file_id)
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", s2s)
    if public is None:
        monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", public)
    monkeypatch.setattr(external_media, "_validate_instagram_record", Mock())

    issued = external_media.issue_external_media_url(
        _DB(record),
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    )

    prefix = f"{s2s.rstrip('/')}/api/files/external-media/{file_id}?ticket="
    assert issued.url.startswith(prefix)
    ticket = parse_qs(urlsplit(issued.url).query)["ticket"][0]
    redeemed = external_media.redeem_external_media_ticket(
        _DB(record), file_id=file_id, ticket=ticket
    )
    assert redeemed.record.file_id == file_id


@pytest.mark.parametrize(
    ("s2s", "public", "message"),
    [
        (None, None, "is required for external media delivery"),
        (None, "http://api.example.test", "must be HTTPS"),
        ("http://s2s.example.test", "https://api.example.test", "must be HTTPS"),
    ],
)
def test_issue_rejects_unusable_base(monkeypatch, s2s, public, message):
    file_id = str(uuid4())
    for name, value in (
        ("XAGENT_S2S_API_BASE_URL", s2s),
        ("XAGENT_PUBLIC_API_BASE_URL", public),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(external_media, "_validate_instagram_record", Mock())
    calls = _count_encode_calls(monkeypatch)

    with pytest.raises(external_media.ExternalMediaUnavailable, match=message):
        external_media.issue_external_media_url(
            _DB(_record(file_id)),
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )

    assert calls == []


def test_issue_rejects_invalid_s2s_value(monkeypatch):
    file_id = str(uuid4())
    monkeypatch.setenv("XAGENT_S2S_API_BASE_URL", "ftp://x")
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    monkeypatch.setattr(external_media, "_validate_instagram_record", Mock())
    calls = _count_encode_calls(monkeypatch)

    with pytest.raises(ValueError, match="XAGENT_S2S_API_BASE_URL"):
        external_media.issue_external_media_url(
            _DB(_record(file_id)),
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )

    assert calls == []


def test_issue_rejects_cross_task_record():
    file_id = str(uuid4())

    with pytest.raises(external_media.ExternalMediaNotFound):
        external_media.issue_external_media_url(
            _DB(_record(file_id)),
            file_id=file_id,
            user_id=7,
            task_id=12,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )


def test_issue_rejects_unsupported_image_format():
    file_id = str(uuid4())

    with pytest.raises(
        external_media.ExternalMediaUnavailable,
        match="PNG or JPEG",
    ):
        external_media.issue_external_media_url(
            _DB(_record(file_id, filename="payload.svg", mime_type="image/svg+xml")),
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )


def test_ticket_is_bound_to_exact_file(monkeypatch):
    file_id = str(uuid4())
    other_file_id = str(uuid4())
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    monkeypatch.setattr(external_media, "_validate_instagram_record", Mock())
    issued = external_media.issue_external_media_url(
        _DB(_record(file_id)),
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    )
    ticket = parse_qs(urlsplit(issued.url).query)["ticket"][0]

    with pytest.raises(external_media.ExternalMediaTicketRejected):
        external_media.redeem_external_media_ticket(
            _DB(_record(other_file_id)),
            file_id=other_file_id,
            ticket=ticket,
        )


def test_ticket_rejects_tampering():
    file_id = str(uuid4())

    with pytest.raises(external_media.ExternalMediaTicketRejected):
        external_media.redeem_external_media_ticket(
            _DB(_record(file_id)),
            file_id=file_id,
            ticket="not-a-jwt",
        )


def test_ticket_rejects_expiration():
    file_id = str(uuid4())
    ticket = external_media._encode_ticket(
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        ttl=-1,
    )

    with pytest.raises(external_media.ExternalMediaTicketRejected):
        external_media.redeem_external_media_ticket(
            _DB(_record(file_id)),
            file_id=file_id,
            ticket=ticket,
        )


def test_external_media_endpoint_serves_inline_without_caching(monkeypatch, tmp_path):
    from xagent.web.api import files

    file_id = str(uuid4())
    record = _record(file_id)
    image_path = tmp_path / "launch.png"
    Image.new("RGB", (640, 640), "blue").save(image_path)

    monkeypatch.setattr(
        files,
        "redeem_external_media_ticket",
        lambda db, **kwargs: external_media.RedeemedExternalMedia(
            record=record,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
            media_type="image/png",
        ),
    )

    class _FileRef:
        def __init__(self, value):
            assert value is record

        @property
        def local_path(self):
            return image_path

        def materialize(self, *, allow_existing_local):
            assert allow_existing_local is False
            return image_path

    monkeypatch.setattr(files, "ManagedFileRef", _FileRef)

    response = files.external_media_file(file_id, ticket="ticket", db=_DB(record))

    assert response.body.startswith(b"\xff\xd8")
    assert response.media_type == "image/jpeg"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["content-disposition"].startswith("inline;")
    assert "launch.jpg" in response.headers["content-disposition"]
    assert response.headers["referrer-policy"] == "no-referrer"


def test_instagram_jpeg_conversion_flattens_alpha_and_pads_aspect_ratio(tmp_path):
    source = tmp_path / "tall.png"
    Image.new("RGBA", (100, 300), (255, 0, 0, 128)).save(source)

    converted = external_media.instagram_jpeg_bytes(BytesIO(source.read_bytes()))
    output = tmp_path / "converted.jpg"
    output.write_bytes(converted)

    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.width >= 320
        assert image.width / image.height >= external_media.INSTAGRAM_MIN_ASPECT_RATIO
        assert image.getpixel((0, 0)) == pytest.approx((255, 255, 255), abs=3)


def test_instagram_conversion_upscales_tiny_tall_image_to_minimum_width(tmp_path):
    source = tmp_path / "tiny-tall.png"
    Image.new("RGB", (3, 4), "blue").save(source)

    converted = external_media.instagram_jpeg_bytes(BytesIO(source.read_bytes()))
    output = tmp_path / "tiny-tall.jpg"
    output.write_bytes(converted)

    with Image.open(output) as image:
        assert image.width >= external_media.INSTAGRAM_MIN_WIDTH
        assert image.width / image.height >= external_media.INSTAGRAM_MIN_ASPECT_RATIO


def test_instagram_conversion_resizes_before_padding_extreme_tall_image(tmp_path):
    source = tmp_path / "extreme-tall.png"
    Image.new("RGB", (10, 50_000), "blue").save(source)

    converted = external_media.instagram_jpeg_bytes(BytesIO(source.read_bytes()))
    output = tmp_path / "extreme-tall.jpg"
    output.write_bytes(converted)

    with Image.open(output) as image:
        assert image.width <= external_media.INSTAGRAM_MAX_WIDTH
        assert image.width / image.height >= external_media.INSTAGRAM_MIN_ASPECT_RATIO


def test_instagram_conversion_pads_wide_image_and_downscales_width(tmp_path):
    source = tmp_path / "wide.png"
    Image.new("RGB", (2000, 500), "blue").save(source)

    converted = external_media.instagram_jpeg_bytes(BytesIO(source.read_bytes()))
    output = tmp_path / "wide.jpg"
    output.write_bytes(converted)

    with Image.open(output) as image:
        assert image.width == external_media.INSTAGRAM_MAX_WIDTH
        assert image.width / image.height <= external_media.INSTAGRAM_MAX_ASPECT_RATIO


def test_instagram_conversion_flattens_palette_transparency(tmp_path):
    source = tmp_path / "palette.png"
    image = Image.new("P", (640, 640))
    image.putpalette([255, 0, 0, 0, 0, 0] + [0, 0, 0] * 254)
    image.info["transparency"] = 0
    image.save(source)

    converted = external_media.instagram_jpeg_bytes(BytesIO(source.read_bytes()))
    output = tmp_path / "palette.jpg"
    output.write_bytes(converted)

    with Image.open(output) as normalized:
        assert normalized.mode == "RGB"
        assert normalized.getpixel((0, 0)) == pytest.approx((255, 255, 255), abs=3)


def test_instagram_conversion_applies_exif_orientation(tmp_path):
    source = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (400, 800), "blue").save(source, exif=exif)

    assert (
        external_media.instagram_image_requires_normalization(
            BytesIO(source.read_bytes())
        )
        is True
    )
    converted = external_media.instagram_jpeg_bytes(BytesIO(source.read_bytes()))
    output = tmp_path / "oriented-normalized.jpg"
    output.write_bytes(converted)

    with Image.open(output) as image:
        assert image.width > image.height
        assert image.getexif().get(274, 1) == 1


def _sized_jpeg(size, orientation=1):
    exif = Image.Exif()
    exif[274] = orientation
    buffer = BytesIO()
    Image.new("RGB", size, "blue").save(buffer, "JPEG", exif=exif.tobytes())
    return BytesIO(buffer.getvalue())


@pytest.mark.parametrize(
    ("stored", "orientation"),
    [((400, 12000), 1), ((12000, 400), 6)],
    ids=["narrow-tall", "narrow-tall-after-exif-orientation-6"],
)
def test_instagram_conversion_decodes_a_narrow_jpeg_at_reduced_size(
    monkeypatch, stored, orientation
):
    # Why: a fixed draft target left a long, narrow JPEG decoding at full size,
    # about 1.2 GB at 100 million pixels (#3596). Drafting to the size it is
    # sent at lets libjpeg scale it down, but never below that size: upright,
    # this 400x12000 image is resized to 60x1800 before padding.
    decoded = []
    exif_transpose = external_media.ImageOps.exif_transpose

    def record_decoded_size(image, **kwargs):
        decoded.append(image.size)  # drafted, not yet loaded
        return exif_transpose(image, **kwargs)

    monkeypatch.setattr(external_media.ImageOps, "exif_transpose", record_decoded_size)

    external_media.instagram_jpeg_bytes(_sized_jpeg(stored, orientation))

    [(width, height)] = decoded
    assert stored[0] // width > 1 and stored[1] // height > 1
    upright = (width, height) if orientation == 1 else (height, width)
    assert upright[0] >= 60 and upright[1] >= 1800


@pytest.mark.parametrize(
    ("stored", "orientation", "sent"),
    [
        ((4000, 4000), 1, (1440, 1440)),
        ((4000, 3000), 1, (1440, 1080)),
        ((3013, 1800), 1, (1440, 860)),
        ((400, 12000), 1, (1440, 1800)),
        ((6000, 1000), 1, (1440, 754)),
        ((3000, 1200), 2, (1440, 754)),
        ((3000, 1200), 3, (1440, 754)),
        ((3000, 1200), 4, (1440, 754)),
        ((3000, 1200), 5, (1440, 1800)),
        ((3000, 1200), 6, (1440, 1800)),
        ((3000, 1200), 7, (1440, 1800)),
        ((3000, 1200), 8, (1440, 1800)),
    ],
    ids=[
        "square",
        "photo",
        "photo-fitted-from-full-size",
        "narrow-tall",
        "wide-panorama",
        "exif-orientation-2",
        "exif-orientation-3",
        "exif-orientation-4",
        "exif-orientation-5",
        "exif-orientation-6",
        "exif-orientation-7",
        "exif-orientation-8",
    ],
)
def test_instagram_conversion_sends_the_same_size_whatever_it_decodes_at(
    stored, orientation, sent
):
    # Why: how small a JPEG is decoded must not change what is sent. These are
    # the sizes the fixed draft target produced before (#3596). The 3013x1800
    # photo was decoded at full size then; fitting it from a drafted copy
    # instead would send 1440x859. Orientations 2-4 keep the stored 3000x1200
    # image's width and height; 5-8 (mirrored or not) turn it upright as
    # 1200x3000.
    converted = external_media.instagram_jpeg_bytes(_sized_jpeg(stored, orientation))

    with Image.open(BytesIO(converted)) as image:
        assert image.size == sent


def test_instagram_conforming_jpeg_does_not_require_normalization(tmp_path):
    source = tmp_path / "ready.jpg"
    Image.new("RGB", (1080, 1080), "blue").save(source)

    assert (
        external_media.instagram_image_requires_normalization(
            BytesIO(source.read_bytes())
        )
        is False
    )


def test_instagram_source_pixel_limit_is_checked_before_load(monkeypatch, tmp_path):
    source = tmp_path / "oversized.png"
    source.write_bytes(b"placeholder")

    class OversizedImage:
        format = "PNG"
        width = 5001
        height = 5001

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def load(self):
            raise AssertionError("pixel data must not be loaded")

    monkeypatch.setattr(
        external_media.Image, "open", lambda *args, **kwargs: OversizedImage()
    )

    with pytest.raises(external_media.ExternalMediaUnavailable, match="pixel limit"):
        external_media.instagram_image_requires_normalization(
            BytesIO(source.read_bytes())
        )


@pytest.mark.parametrize(
    "error", [SyntaxError("bad JPEG"), external_media.struct.error()]
)
def test_instagram_decoder_errors_are_mapped(monkeypatch, tmp_path, error):
    source = tmp_path / "image.jpg"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(external_media.Image, "open", Mock(side_effect=error))

    with pytest.raises(external_media.ExternalMediaUnavailable, match="decoded"):
        external_media.instagram_image_requires_normalization(
            BytesIO(source.read_bytes())
        )


def test_issue_preflights_image_header(monkeypatch, tmp_path):
    uploads_dir = tmp_path / "uploads"
    source = uploads_dir / "user_7" / "web_task_11" / "launch.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (640, 640), "blue").save(source)
    file_id = str(uuid4())
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")

    issued = external_media.issue_external_media_url(
        _DB(
            _record(
                file_id,
                storage_path=str(source),
                storage_status="legacy",
                storage_key=None,
            )
        ),
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    )

    assert issued.url.startswith("https://api.example.test/")


def test_issue_rejects_corrupt_image_before_ticket(monkeypatch, tmp_path):
    uploads_dir = tmp_path / "uploads"
    source = uploads_dir / "user_7" / "web_task_11" / "broken.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not-an-image")
    file_id = str(uuid4())
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")

    with pytest.raises(external_media.ExternalMediaUnavailable, match="decoded"):
        external_media.issue_external_media_url(
            _DB(
                _record(
                    file_id,
                    storage_path=str(source),
                    storage_status="legacy",
                    storage_key=None,
                )
            ),
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )


def test_large_jpeg_header_uses_higher_source_pixel_limit():
    # The scan layout is read from the stream Pillow opened: one interleaved scan.
    image = Mock(
        format="JPEG",
        width=8000,
        height=6000,
        info={},
        fp=BytesIO(jpeg_400px("interleaved")),
    )

    external_media._validate_instagram_source(image)

    image.format = "PNG"
    with pytest.raises(external_media.ExternalMediaUnavailable, match="pixel limit"):
        external_media._validate_instagram_source(image)


def test_instagram_processing_memory_error_is_mapped(monkeypatch, tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (640, 640), "blue").save(source)
    monkeypatch.setattr(
        external_media.ImageOps,
        "exif_transpose",
        Mock(side_effect=MemoryError),
    )

    with pytest.raises(external_media.ExternalMediaUnavailable, match="normalized"):
        external_media.instagram_jpeg_bytes(BytesIO(source.read_bytes()))


def _instagram_issue(monkeypatch, tmp_path, *, name="launch.png", write):
    uploads_dir = tmp_path / "uploads"
    source = uploads_dir / "user_7" / "web_task_11" / name
    source.parent.mkdir(parents=True)
    write(source)
    file_id = str(uuid4())
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    db = _DB(
        _record(
            file_id,
            filename=name,
            storage_path=str(source),
            storage_status="legacy",
            storage_key=None,
        )
    )
    issue = lambda: external_media.issue_external_media_url(  # noqa: E731
        db,
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    )
    return source, issue


@OVERSIZED_SOURCES
@pytest.mark.parametrize(
    ("rewritten", "bytes_read"),
    [("before-open", 0), ("while-open", SOURCE_LIMIT + 1)],
)
def test_issue_instagram_bounds_what_it_reads_of_an_oversized_source(
    monkeypatch, tmp_path, filename, write_oversized, rewritten, bytes_read
):
    # Why: Pillow reads a PNG chunk or JPEG segment it does not decode in full
    # and keeps it, so a few pixels in a large file cost memory by the file's
    # size, and the sandbox can rewrite its outputs at any time (#3596). A file
    # already over the limit is refused unread; one rewritten after that check
    # is read only to one byte past the limit.
    monkeypatch.setattr(external_media, "INSTAGRAM_MAX_SOURCE_BYTES", SOURCE_LIMIT)
    if rewritten == "before-open":
        _, issue = _instagram_issue(
            monkeypatch,
            tmp_path,
            name=filename,
            write=lambda path: write_oversized(path, OVERSIZED),
        )
        opened = count_source_reads(monkeypatch, external_media)
    else:
        source, issue = _instagram_issue(
            monkeypatch, tmp_path, name=filename, write=write_small_image
        )
        opened = count_source_reads(
            monkeypatch,
            external_media,
            before_first_read=lambda: write_oversized(source, OVERSIZED),
        )
    tickets = _count_encode_calls(monkeypatch)

    with pytest.raises(
        external_media.ExternalMediaUnavailable,
        match="exceeds the 1 MB Instagram source limit; compress or downscale",
    ):
        issue()

    assert [reader.bytes_read for reader in opened] == [bytes_read]
    assert tickets == []


def test_issue_instagram_refuses_a_source_over_50_mb_unread(monkeypatch, tmp_path):
    # Why: the other byte-limit tests lower the limit to stay fast; this one
    # pins the production default. A sparse file one byte over 50 MB is
    # refused from its size alone, without reading any of it.
    source, issue = _instagram_issue(
        monkeypatch,
        tmp_path,
        write=lambda path: write_png_with_private_chunk(path, 50 * 1024 * 1024 + 1),
    )
    opened = count_source_reads(monkeypatch, external_media)

    with pytest.raises(
        external_media.ExternalMediaUnavailable,
        match="exceeds the 50 MB Instagram source limit",
    ):
        issue()

    assert source.stat().st_size == 50 * 1024 * 1024 + 1
    assert [reader.bytes_read for reader in opened] == [0]


@pytest.mark.parametrize(("slack", "issued"), [(0, True), (-1, False)])
def test_issue_instagram_source_byte_limit_includes_the_limit(
    monkeypatch, tmp_path, slack, issued
):
    # Why: an ordinary image is held to the same byte limit, and one within it
    # still issues; the limit itself is allowed.
    source, issue = _instagram_issue(monkeypatch, tmp_path, write=write_small_image)
    monkeypatch.setattr(
        external_media, "INSTAGRAM_MAX_SOURCE_BYTES", source.stat().st_size + slack
    )

    if issued:
        assert issue().url.startswith("https://api.example.test/")
    else:
        with pytest.raises(
            external_media.ExternalMediaUnavailable, match="Instagram source limit"
        ):
            issue()


@MULTI_SCAN_PIXEL_LIMIT_CASES
def test_issue_instagram_holds_a_multi_scan_jpeg_to_the_png_pixel_limit(
    monkeypatch, tmp_path, layout, slack, refused_as
):
    # Why: the higher JPEG pixel limit relies on draft() letting libjpeg decode
    # a JPEG at a fraction of its size. With more than one scan (progressive,
    # or a first scan that codes only some components) libjpeg keeps
    # coefficients for the whole image, so such a JPEG gets the PNG limit
    # (#3596). One interleaved scan of the same size keeps the higher limit.
    _, issue = _instagram_issue(
        monkeypatch,
        tmp_path,
        name="launch.jpg",
        write=lambda path: path.write_bytes(jpeg_400px(layout)),
    )
    monkeypatch.setattr(
        external_media, "INSTAGRAM_MAX_SOURCE_PIXELS", 400 * 400 + slack
    )

    if refused_as is None:
        assert issue().url.startswith("https://api.example.test/")
    else:
        with pytest.raises(
            external_media.ExternalMediaUnavailable,
            match=f"pixel limit of 159,999 for {refused_as}; downscale it",
        ):
            issue()


@pytest.mark.parametrize(("slack", "refused"), [(-1, True), (0, False)])
def test_issue_instagram_holds_a_lossless_jpeg_to_the_png_pixel_limit(
    monkeypatch, tmp_path, slack, refused
):
    # Why: libjpeg does not scale a lossless JPEG, so draft() cannot keep its
    # decode small; drafting one by 2 or more breaks its decode, and with
    # Pillow 12.3.0 can crash the process. Conversion therefore decodes it at
    # full size, and it gets the PNG limit like any image decoded in full.
    # Issuance refuses it, before a URL is handed out, not Meta's fetch later.
    _, issue = _instagram_issue(
        monkeypatch,
        tmp_path,
        name="launch.jpg",
        write=lambda path: path.write_bytes(LOSSLESS_JPEG.read_bytes()),
    )
    monkeypatch.setattr(
        external_media, "INSTAGRAM_MAX_SOURCE_PIXELS", 160 * 160 + slack
    )

    if refused:
        with pytest.raises(
            external_media.ExternalMediaUnavailable,
            match="pixel limit of 25,599 for lossless JPEG; downscale it",
        ):
            issue()
    else:
        assert issue().url.startswith("https://api.example.test/")


def test_instagram_conversion_decodes_a_lossless_jpeg_without_drafting(monkeypatch):
    # Why: drafting a lossless JPEG by 2 or more breaks its decode (see the
    # pixel limit test above). It is decoded at full size and then resized.
    # A maximum width of 64 makes the 160x160 fixture one draft() would scale
    # by 2.
    monkeypatch.setattr(external_media, "INSTAGRAM_MAX_WIDTH", 64)
    drafted = []
    monkeypatch.setattr(
        JpegImagePlugin.JpegImageFile,
        "draft",
        lambda image, mode, size: drafted.append(size),
    )

    converted = external_media.instagram_jpeg_bytes(BytesIO(LOSSLESS_JPEG.read_bytes()))

    assert drafted == []
    with Image.open(BytesIO(converted)) as image:
        assert image.format == "JPEG"
        assert image.size == (320, 320)
        assert image.getpixel((160, 160)) == pytest.approx((0, 0, 254), abs=2)


def test_instagram_converts_a_lossless_jpeg_that_otherwise_conforms(monkeypatch):
    # Why: few decoders read a lossless JPEG, and Meta is not known to. One
    # that meets every other condition would be sent as stored and fail only
    # when Meta fetches it, so it is converted to a baseline JPEG first. A
    # minimum width of 100 makes the 160x160 fixture otherwise conforming.
    monkeypatch.setattr(external_media, "INSTAGRAM_MIN_WIDTH", 100)
    source = LOSSLESS_JPEG.read_bytes()

    assert external_media.instagram_image_requires_normalization(BytesIO(source))
    converted = external_media.instagram_jpeg_bytes(BytesIO(source))
    assert external_media._jpeg_markers(BytesIO(converted))[0] == 0xC0


def test_issue_instagram_reads_jpeg_markers_only_over_the_png_pixel_limit(
    monkeypatch, tmp_path
):
    # Why: within the PNG limit both limits pass, so the scan layout cannot
    # change the outcome; the byte-by-byte marker walk is skipped there.
    _, issue = _instagram_issue(
        monkeypatch,
        tmp_path,
        name="launch.jpg",
        write=lambda path: path.write_bytes(jpeg_400px("interleaved")),
    )
    walked = []
    jpeg_markers = external_media._jpeg_markers
    monkeypatch.setattr(
        external_media,
        "_jpeg_markers",
        lambda stream: walked.append(True) or jpeg_markers(stream),
    )

    issue()
    assert walked == []

    monkeypatch.setattr(external_media, "INSTAGRAM_MAX_SOURCE_PIXELS", 400 * 400 - 1)
    issue()
    assert walked == [True]


def _with_fill_bytes(jpeg: bytes) -> bytes:
    # Extra FF bytes may precede any marker.
    return jpeg.replace(b"\xff\xc0", b"\xff\xff\xff\xc0").replace(
        b"\xff\xda", b"\xff\xff\xda"
    )


def _with_fake_scan_in_a_comment(jpeg: bytes) -> bytes:
    # A COM segment whose payload looks like a one-component scan header.
    return (
        jpeg[:2] + b"\xff\xfe\x00\x0a" + b"\xff\xda\x00\x08\x01\x01\x00\x00" + jpeg[2:]
    )


def _with_second_soi(jpeg: bytes) -> bytes:
    # A second SOI before the first scan: Pillow opens it, libjpeg refuses it.
    return jpeg[:2] + b"\xff\xd8" + jpeg[2:]


def _with_zero_length_segment(jpeg: bytes) -> bytes:
    # An APP5 segment before the frame whose length field is 0, below the 2 it
    # must count; Pillow and libjpeg both read past it.
    return jpeg[:2] + b"\xff\xe5\x00\x00" + jpeg[2:]


@pytest.mark.parametrize(
    ("transform", "layout", "refused"),
    [
        (_with_fill_bytes, "interleaved", False),
        (_with_fill_bytes, "non-interleaved", True),
        (_with_fake_scan_in_a_comment, "interleaved", False),
        (_with_second_soi, "interleaved", True),
        (_with_zero_length_segment, "interleaved", True),
    ],
    ids=[
        "fill-bytes-interleaved",
        "fill-bytes-non-interleaved",
        "fake-scan-in-a-comment",
        "unreadable-second-soi",
        "unreadable-zero-length-segment",
    ],
)
def test_issue_instagram_reads_the_jpeg_scan_layout_from_its_markers(
    monkeypatch, tmp_path, transform, layout, refused
):
    # Why: the scan layout comes from the markers up to the first scan. Fill
    # bytes before a marker are skipped, and each segment by its length, so a
    # scan header inside a comment does not count. Markers that cannot be read
    # that far (a second SOI, which Pillow opens but libjpeg refuses, or a
    # segment length below 2) must not earn the higher limit.
    _, issue = _instagram_issue(
        monkeypatch,
        tmp_path,
        name="launch.jpg",
        write=lambda path: path.write_bytes(transform(jpeg_400px(layout))),
    )
    monkeypatch.setattr(external_media, "INSTAGRAM_MAX_SOURCE_PIXELS", 400 * 400 - 1)

    if refused:
        with pytest.raises(
            external_media.ExternalMediaUnavailable, match="for multi-scan JPEG"
        ):
            issue()
    else:
        assert issue().url.startswith("https://api.example.test/")


def _facebook_issue(monkeypatch, tmp_path, *, name="post.png", write=None):
    uploads_dir = tmp_path / "uploads"
    source = uploads_dir / "user_7" / "web_task_11" / name
    source.parent.mkdir(parents=True)
    if write is None:
        Image.new("RGB", (3000, 400), "blue").save(source)
    else:
        write(source)
    file_id = str(uuid4())
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    record = _record(
        file_id,
        filename=name,
        storage_path=str(source),
        storage_status="legacy",
        storage_key=None,
    )
    db = _DB(record)
    issued = lambda: external_media.issue_external_media_url(  # noqa: E731
        db,
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
    )
    return db, file_id, issued


def test_issue_facebook_ticket_names_its_purpose(monkeypatch, tmp_path):
    # Why: the delivery route decides whether to normalize by the ticket's
    # purpose, so a Facebook ticket must carry the Facebook purpose.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    db, file_id, issue = _facebook_issue(monkeypatch, tmp_path)

    ticket = parse_qs(urlsplit(issue().url).query)["ticket"][0]
    redeemed = external_media.redeem_external_media_ticket(
        db, file_id=file_id, ticket=ticket
    )

    assert redeemed.purpose is external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH
    assert redeemed.media_type == "image/png"


@pytest.mark.parametrize(
    ("name", "write", "message"),
    [
        ("post.png", lambda path: path.write_bytes(b"not-an-image"), "PNG or JPEG"),
        (
            "post.gif",
            lambda path: Image.new("RGB", (8, 8)).save(path, "GIF"),
            "PNG or JPEG",
        ),
        ("post.png", lambda path: path.write_bytes(b"x" * 64), "10 MB"),
    ],
)
def test_issue_facebook_rejects_what_the_page_cannot_take(
    monkeypatch, tmp_path, name, write, message
):
    # Why: an image the URL path cannot deliver must fail at issue, before a
    # ticket exists, as its own error so native grants can fall back to the
    # connector's upload; the size cap matches the connector's.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    if message == "10 MB":
        monkeypatch.setattr(external_media, "FACEBOOK_MAX_IMAGE_BYTES", 32)
    _, _, issue = _facebook_issue(monkeypatch, tmp_path, name=name, write=write)

    with pytest.raises(external_media.ExternalMediaUnsupportedImage, match=message):
        issue()


@pytest.mark.parametrize("base", [None, "http://api.example.test"])
def test_issue_without_a_public_https_base_is_not_configured(monkeypatch, base):
    # Why: the adapter falls back to a native upload only for this case, so it
    # must be its own error, distinct from an image that cannot be delivered.
    monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
    if base:
        monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", base)
    file_id = str(uuid4())

    with pytest.raises(external_media.ExternalMediaNotConfigured):
        external_media.issue_external_media_url(
            _DB(_record(file_id)),
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
        )


def test_issue_facebook_rejects_a_truncated_image(monkeypatch, tmp_path):
    # Why: a header that parses is not enough; Meta would fail to fetch a
    # truncated body, and it must fall back while the native upload still can.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")

    def truncated_png(path):
        Image.new("RGB", (640, 640), "blue").save(path)
        path.write_bytes(path.read_bytes()[:-40])

    _, _, issue = _facebook_issue(monkeypatch, tmp_path, write=truncated_png)

    with pytest.raises(external_media.ExternalMediaUnsupportedImage, match="decoded"):
        issue()


def _exif_with_thumbnail() -> bytes:
    """EXIF whose IFD1 holds a small JPEG thumbnail, with its own end marker."""
    thumbnail = BytesIO()
    Image.new("RGB", (16, 16), "red").save(thumbnail, "JPEG")
    thumbnail_bytes = thumbnail.getvalue()
    ifd1_offset = 8 + 2 + 4  # TIFF header, then IFD0 with no entries
    thumbnail_offset = ifd1_offset + 2 + 2 * 12 + 4
    tiff = (
        b"II*\x00"
        + struct.pack("<I", 8)
        + struct.pack("<HI", 0, ifd1_offset)
        + struct.pack("<H", 2)
        + struct.pack("<HHII", 0x0201, 4, 1, thumbnail_offset)
        + struct.pack("<HHII", 0x0202, 4, 1, len(thumbnail_bytes))
        + struct.pack("<I", 0)
        + thumbnail_bytes
    )
    return b"Exif\x00\x00" + tiff


def _jpeg(kind: str) -> bytes:
    """A noisy 640x640 JPEG, so the main image has plenty of scan data."""
    image = Image.effect_noise((640, 640), 64).convert("RGB")
    output = BytesIO()
    if kind == "plain":
        image.save(output, "JPEG")
    elif kind == "progressive":
        image.save(output, "JPEG", progressive=True)
    elif kind == "exif thumbnail":
        image.save(output, "JPEG", exif=_exif_with_thumbnail())
    elif kind == "trailing data":
        # Like a phone's motion photo: a video appended after the end marker.
        image.save(output, "JPEG")
        output.write(b"\x00\x00\x00\x18ftypmp42" + bytes(4096))
    else:
        assert kind == "MPO with an EXIF thumbnail"
        image.save(
            output,
            "MPO",
            save_all=True,
            append_images=[Image.new("RGB", (32, 32), "green")],
            exif=_exif_with_thumbnail(),
        )
    return output.getvalue()


@pytest.mark.parametrize(
    "kind", ["plain", "exif thumbnail", "MPO with an EXIF thumbnail"]
)
def test_issue_facebook_rejects_a_truncated_jpeg(monkeypatch, tmp_path, kind):
    # Why: JPEG's verify() parses only the header, so a JPEG cut short by an
    # interrupted write would get a URL that Meta then fails to fetch, and a
    # native grant would lose its upload fallback. An EXIF thumbnail ends with
    # its own end marker before the main image's scan data; that one must not
    # count, and Image.open rewinds an MPO to its start after parsing it.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    complete = _jpeg(kind)
    truncated = complete[: len(complete) // 2]
    if kind != "plain":
        assert b"\xff\xd9" in truncated  # the thumbnail's end marker
    if kind.startswith("MPO"):
        assert Image.open(BytesIO(truncated)).format == "MPO"

    _, _, issue = _facebook_issue(
        monkeypatch,
        tmp_path,
        name="post.jpg",
        write=lambda path: path.write_bytes(truncated),
    )

    with pytest.raises(external_media.ExternalMediaUnsupportedImage, match="decoded"):
        issue()


@pytest.mark.parametrize(
    "kind",
    [
        "plain",
        "progressive",
        "exif thumbnail",
        "MPO with an EXIF thumbnail",
        "trailing data",
    ],
)
def test_issue_facebook_accepts_a_complete_jpeg(monkeypatch, tmp_path, kind):
    # Why: the end-marker check must not refuse whole JPEGs: progressive ones
    # with several scans, ones carrying an EXIF thumbnail, MPO files whose
    # extra images follow the main image's end marker, and files with other
    # data after that marker, so the file need not end with one.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    content = _jpeg(kind)
    if kind == "trailing data":
        assert not content.endswith(b"\xff\xd9")
    _, _, issue = _facebook_issue(
        monkeypatch,
        tmp_path,
        name="post.jpg",
        write=lambda path: path.write_bytes(content),
    )

    assert issue().url.startswith("https://api.example.test/")


@pytest.mark.parametrize("registered_type", ["image/png", "image/jpeg"])
def test_delivered_image_type_keeps_the_registered_type_for_other_bytes(
    registered_type,
):
    # Why: delivery names the type of the bytes it sends, but a file rewritten
    # after issue may carry neither signature; the registered type then stays
    # rather than a guess.
    gif = BytesIO()
    Image.new("RGB", (8, 8), "blue").save(gif, "GIF")

    assert (
        external_media.delivered_image_media_type(gif.getvalue(), registered_type)
        == registered_type
    )


def test_issue_facebook_read_failure_is_not_an_unsupported_image(monkeypatch, tmp_path):
    # Why: an IO failure is a storage problem, not the image's fault; it must
    # stay base ExternalMediaUnavailable so the adapter does not fall back.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    _, _, issue = _facebook_issue(monkeypatch, tmp_path)

    reader = MagicMock()
    reader.__enter__.return_value.read.side_effect = OSError("Input/output error")
    monkeypatch.setattr(external_media, "open_delivery_file", Mock(return_value=reader))

    with pytest.raises(external_media.ExternalMediaUnavailable, match="read") as raised:
        issue()
    assert not isinstance(raised.value, external_media.ExternalMediaUnsupportedImage)


@pytest.mark.timeout(5)
@pytest.mark.parametrize("swapped", ["file", "directory", "fifo"])
def test_issue_facebook_refuses_a_path_swapped_after_materialization(
    monkeypatch, tmp_path, swapped
):
    # The sandbox can replace an output after materialization. Signing must
    # neither follow the new link nor block on a FIFO, before any ticket exists.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    db, _, issue = _facebook_issue(monkeypatch, tmp_path)
    source = Path(db.record.storage_path)
    other = tmp_path / "other" / source.name
    other.parent.mkdir()
    Image.new("RGB", (64, 64), "red").save(other)
    materialize = external_media._materialize_record
    tickets = _count_encode_calls(monkeypatch)

    def materialize_then_swap(record):
        path = materialize(record)
        assert path == source
        if swapped == "directory":
            source.parent.rename(tmp_path / "original-output")
            source.parent.symlink_to(other.parent)
        else:
            source.unlink()
            if swapped == "fifo":
                os.mkfifo(source)
            else:
                source.symlink_to(other)
        return path

    monkeypatch.setattr(external_media, "_materialize_record", materialize_then_swap)

    with pytest.raises(external_media.ExternalMediaUnavailable, match="read") as raised:
        issue()

    assert type(raised.value) is external_media.ExternalMediaUnavailable
    assert tickets == []


def test_open_delivery_file_closes_the_descriptor_when_wrapping_fails(
    monkeypatch, tmp_path
):
    # Each issuance or GET opens a descriptor; a failure after the open must
    # not leak it.
    uploads_dir = tmp_path / "uploads"
    source = uploads_dir / "user_7" / "web_task_11" / "launch.png"
    _png(source)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    record = _record(str(uuid4()), storage_path=str(source))
    opened = []
    real_open = os.open

    def tracking_open(name, flags, *args, **kwargs):
        fd = real_open(name, flags, *args, **kwargs)
        if name == source.name:
            opened.append(fd)
        return fd

    monkeypatch.setattr(external_media.os, "open", tracking_open)
    monkeypatch.setattr(
        external_media.os, "fdopen", Mock(side_effect=OSError("fdopen failed"))
    )

    with pytest.raises(OSError, match="fdopen failed"):
        external_media.open_delivery_file(record, source)

    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.timeout(5)
@pytest.mark.parametrize("swapped", ["file", "directory", "fifo"])
def test_issue_instagram_refuses_a_path_swapped_after_materialization(
    monkeypatch, tmp_path, swapped
):
    # Instagram issuance reads the header the same way, and must not block on a
    # FIFO or follow a link swapped in after materialization either.
    uploads_dir = tmp_path / "uploads"
    source = uploads_dir / "user_7" / "web_task_11" / "launch.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (640, 640), "blue").save(source)
    other = tmp_path / "other" / source.name
    other.parent.mkdir()
    Image.new("RGB", (640, 640), "red").save(other)
    file_id = str(uuid4())
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    materialize = external_media._materialize_record
    tickets = _count_encode_calls(monkeypatch)

    def materialize_then_swap(record):
        path = materialize(record)
        assert path == source
        if swapped == "directory":
            source.parent.rename(tmp_path / "original-output")
            source.parent.symlink_to(other.parent)
        else:
            source.unlink()
            if swapped == "fifo":
                os.mkfifo(source)
            else:
                source.symlink_to(other)
        return path

    monkeypatch.setattr(external_media, "_materialize_record", materialize_then_swap)

    with pytest.raises(external_media.ExternalMediaUnavailable, match="read"):
        external_media.issue_external_media_url(
            _DB(
                _record(
                    file_id,
                    storage_path=str(source),
                    storage_status="legacy",
                    storage_key=None,
                )
            ),
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )
    assert tickets == []


def test_issue_facebook_bounds_read_when_file_grows(monkeypatch, tmp_path):
    # Grow a real, initially valid image immediately before reading. Count the
    # bytes actually returned: checking len() after an unbounded read is too late.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    db, _, issue = _facebook_issue(monkeypatch, tmp_path)
    source = Path(db.record.storage_path)
    source_inode = source.stat().st_ino
    limit = source.stat().st_size
    monkeypatch.setattr(external_media, "FACEBOOK_MAX_IMAGE_BYTES", limit)
    tickets = _count_encode_calls(monkeypatch)
    reads = []
    opened = []
    path_open = Path.open
    fdopen = os.fdopen

    def track_read(stream):
        opened.append(stream)
        reader = MagicMock(wraps=stream)
        reader.__enter__.return_value = reader
        reader.__exit__.side_effect = stream.__exit__

        def read(*args, **kwargs):
            with path_open(source, "ab") as writer:
                writer.write(b"x" * (limit * 2))
            content = stream.read(*args, **kwargs)
            reads.append(len(content))
            return content

        reader.read.side_effect = read
        return reader

    def open_path(path, *args, **kwargs):
        stream = path_open(path, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else "r")
        if path == source and mode == "rb":
            return track_read(stream)
        return stream

    def open_fd(fd, *args, **kwargs):
        stream = fdopen(fd, *args, **kwargs)
        if os.fstat(stream.fileno()).st_ino == source_inode:
            return track_read(stream)
        return stream

    # Observe both the old pathname reader and the descriptor reader, so this
    # regression fails on the old implementation without allocating a large file.
    monkeypatch.setattr(Path, "open", open_path)
    monkeypatch.setattr(os, "fdopen", open_fd)

    with pytest.raises(external_media.ExternalMediaUnsupportedImage, match="10 MB"):
        issue()

    assert reads == [limit + 1]
    assert opened and all(stream.closed for stream in opened)
    assert tickets == []


def _png(path, color="blue"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


def test_trust_refuses_a_symlink_below_the_user_root(monkeypatch, tmp_path):
    # Why: a registered file replaced by a link to another task's image must
    # not be delivered from the local copy (#3588 scope review, P1).
    uploads = tmp_path / "uploads"
    other = uploads / "user_7" / "web_task_99" / "private.png"
    _png(other, "red")
    registered = uploads / "user_7" / "web_task_11" / "post.png"
    registered.parent.mkdir(parents=True)
    registered.symlink_to(other)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads)
    record = _record(str(uuid4()), storage_path=str(registered))

    assert external_media.local_record_path_is_trusted(record, registered) is False
    _png(registered.with_name("plain.png"))
    plain = registered.with_name("plain.png")
    assert external_media.local_record_path_is_trusted(record, plain) is True


def test_trust_accepts_an_upload_root_that_is_itself_a_link(monkeypatch, tmp_path):
    # Why: a mounted volume may make the upload root a link; only links below
    # the user root are refused, so ordinary deployments keep local delivery.
    real = tmp_path / "volume"
    _png(real / "user_7" / "web_task_11" / "post.png")
    uploads = tmp_path / "uploads"
    uploads.symlink_to(real)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads)
    path = uploads / "user_7" / "web_task_11" / "post.png"

    assert (
        external_media.local_record_path_is_trusted(_record(str(uuid4())), path) is True
    )


def test_issue_facebook_does_not_deliver_a_symlinked_local_copy(monkeypatch, tmp_path):
    # Why: with the local copy refused and no durable object, issuing must
    # fail as out of scope (staging then applies its own rules), never sign
    # a URL for the link's target.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    uploads = tmp_path / "uploads"
    other = uploads / "user_7" / "web_task_99" / "private.png"
    _png(other, "red")

    def link(path):
        path.symlink_to(other)

    _, _, issue = _facebook_issue(monkeypatch, tmp_path, write=link)

    with pytest.raises(external_media.ExternalMediaOutOfScope):
        issue()


def test_issue_local_only_copy_outside_uploads_is_out_of_scope(monkeypatch, tmp_path):
    # Why: a workspace may keep a local-only file outside the upload root that
    # its own staging can read; issuing cannot, and must say so as its own
    # error so native grants fall back instead of failing (scope review, P2).
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: tmp_path / "uploads")
    elsewhere = tmp_path / "custom-workspace" / "post.png"
    _png(elsewhere)
    file_id = str(uuid4())
    db = _DB(
        _record(
            file_id,
            filename="post.png",
            storage_path=str(elsewhere),
            storage_status="legacy",
            storage_key=None,
        )
    )

    with pytest.raises(external_media.ExternalMediaOutOfScope):
        external_media.issue_external_media_url(
            db,
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
        )
