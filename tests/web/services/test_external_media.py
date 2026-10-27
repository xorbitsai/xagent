from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web import models
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import external_media


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

    converted = external_media.instagram_jpeg_bytes(source)
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

    converted = external_media.instagram_jpeg_bytes(source)
    output = tmp_path / "tiny-tall.jpg"
    output.write_bytes(converted)

    with Image.open(output) as image:
        assert image.width >= external_media.INSTAGRAM_MIN_WIDTH
        assert image.width / image.height >= external_media.INSTAGRAM_MIN_ASPECT_RATIO


def test_instagram_conversion_resizes_before_padding_extreme_tall_image(tmp_path):
    source = tmp_path / "extreme-tall.png"
    Image.new("RGB", (10, 50_000), "blue").save(source)

    converted = external_media.instagram_jpeg_bytes(source)
    output = tmp_path / "extreme-tall.jpg"
    output.write_bytes(converted)

    with Image.open(output) as image:
        assert image.width <= external_media.INSTAGRAM_MAX_WIDTH
        assert image.width / image.height >= external_media.INSTAGRAM_MIN_ASPECT_RATIO


def test_instagram_conversion_pads_wide_image_and_downscales_width(tmp_path):
    source = tmp_path / "wide.png"
    Image.new("RGB", (2000, 500), "blue").save(source)

    converted = external_media.instagram_jpeg_bytes(source)
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

    converted = external_media.instagram_jpeg_bytes(source)
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

    assert external_media.instagram_image_requires_normalization(source) is True
    converted = external_media.instagram_jpeg_bytes(source)
    output = tmp_path / "oriented-normalized.jpg"
    output.write_bytes(converted)

    with Image.open(output) as image:
        assert image.width > image.height
        assert image.getexif().get(274, 1) == 1


def test_instagram_conforming_jpeg_does_not_require_normalization(tmp_path):
    source = tmp_path / "ready.jpg"
    Image.new("RGB", (1080, 1080), "blue").save(source)

    assert external_media.instagram_image_requires_normalization(source) is False


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
        external_media.instagram_image_requires_normalization(source)


@pytest.mark.parametrize(
    "error", [SyntaxError("bad JPEG"), external_media.struct.error()]
)
def test_instagram_decoder_errors_are_mapped(monkeypatch, tmp_path, error):
    source = tmp_path / "image.jpg"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(external_media.Image, "open", Mock(side_effect=error))

    with pytest.raises(external_media.ExternalMediaUnavailable, match="decoded"):
        external_media.instagram_image_requires_normalization(source)


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
    image = Mock(format="JPEG", width=8000, height=6000)

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
        external_media.instagram_jpeg_bytes(source)
