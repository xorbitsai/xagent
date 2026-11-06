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


def test_issue_facebook_read_failure_is_not_an_unsupported_image(monkeypatch, tmp_path):
    # Why: an IO failure is a storage problem, not the image's fault; it must
    # stay base ExternalMediaUnavailable so the adapter does not fall back.
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    _, _, issue = _facebook_issue(monkeypatch, tmp_path)

    def unreadable(self):
        raise OSError("Input/output error")

    monkeypatch.setattr(external_media.Path, "read_bytes", unreadable)

    with pytest.raises(external_media.ExternalMediaUnavailable, match="read") as raised:
        issue()
    assert not isinstance(raised.value, external_media.ExternalMediaUnsupportedImage)


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
