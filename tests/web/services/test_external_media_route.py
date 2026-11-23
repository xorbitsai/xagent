import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from tests.web.services.external_media_source_shared import (
    MULTI_SCAN_PIXEL_LIMIT_CASES,
    OVERSIZED,
    OVERSIZED_SOURCES,
    SOURCE_LIMIT,
    count_source_reads,
    jpeg_400px,
    write_small_image,
)
from xagent.web import models
from xagent.web.api import files
from xagent.web.models.database import get_db
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import external_media
from xagent.web.services.managed_file_ref import (
    DurableObjectIntegrityError,
    DurableObjectMissingError,
    DurableStorageOperationError,
)
from xagent.web.services.uploaded_file_cleanup_publication import (
    FilePublicationUnavailable,
)


@pytest.fixture(autouse=True)
def _isolate_s2s_base(monkeypatch):
    monkeypatch.delenv("XAGENT_S2S_API_BASE_URL", raising=False)


@pytest.fixture(scope="module")
def db() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    models.Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        models.Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(scope="module")
def client(db: Session) -> TestClient:
    app = FastAPI()
    app.include_router(files.file_router)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app, raise_server_exceptions=False)


def _record(
    db: Session,
    file_id: str,
    *,
    filename: str = "launch.png",
    mime_type: str = "image/png",
    storage_path: str | None = None,
    storage_status: str = "available",
    storage_key: str | None = "__default__",
) -> UploadedFile:
    if storage_path is None:
        storage_path = f"/unused/{file_id}.png"
    if storage_key == "__default__":
        storage_key = f"users/7/{file_id}.png"
    record = UploadedFile(
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
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _ticket(file_id: str, **overrides: object) -> str:
    claims = {
        "type": external_media.EXTERNAL_MEDIA_TICKET_TYPE,
        "purpose": external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH.value,
        "file_id": file_id,
        "user_id": 7,
        "task_id": 11,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    claims.update(overrides)
    return external_media.jwt.encode(
        claims,
        external_media.JWT_SECRET_KEY,
        algorithm=external_media.JWT_ALGORITHM,
    )


def _issue_ticket_url(
    monkeypatch: pytest.MonkeyPatch,
    db: Session,
    file_id: str,
) -> str:
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://testserver")
    monkeypatch.delenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", raising=False)
    return external_media.issue_external_media_url(
        db,
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    ).url


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "access"},
        {"purpose": "unknown"},
        {"user_id": "7"},
        {"task_id": "11"},
    ],
)
def test_ticket_rejects_signed_invalid_claims(
    db: Session,
    overrides: dict[str, object],
):
    file_id = str(uuid4())
    _record(db, file_id)

    with pytest.raises(external_media.ExternalMediaTicketRejected):
        external_media.redeem_external_media_ticket(
            db,
            file_id=file_id,
            ticket=_ticket(file_id, **overrides),
        )


def test_ticket_rejects_invalid_path_file_id(db: Session):
    with pytest.raises(external_media.ExternalMediaTicketRejected):
        external_media.redeem_external_media_ticket(
            db,
            file_id="not-a-uuid",
            ticket="not-a-jwt",
        )


def test_issue_rejects_http_public_base(
    monkeypatch: pytest.MonkeyPatch,
    db: Session,
):
    file_id = str(uuid4())
    _record(db, file_id)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "http://api.example.test")
    monkeypatch.delenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", raising=False)

    with pytest.raises(external_media.ExternalMediaUnavailable):
        external_media.issue_external_media_url(
            db,
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )


def test_redirect_disabled_does_not_expose_signed_object_url(
    monkeypatch: pytest.MonkeyPatch,
    db: Session,
):
    file_id = str(uuid4())
    _record(db, file_id)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test")
    monkeypatch.delenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", raising=False)
    monkeypatch.setattr(external_media, "_validate_instagram_record", Mock())
    signed_access_url = Mock(
        return_value="https://internal-minio.example.test/private-object"
    )
    monkeypatch.setattr(
        external_media.ManagedFileRef,
        "signed_access_url",
        signed_access_url,
    )

    issued = external_media.issue_external_media_url(
        db,
        file_id=file_id,
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    )

    assert issued.direct_object_url is False
    assert issued.url.startswith("https://api.example.test/api/files/external-media/")
    signed_access_url.assert_not_called()


def test_route_issues_and_redeems_get_and_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / "launch.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 640), "blue").save(image_path)
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    issued_url = _issue_ticket_url(monkeypatch, db, file_id)
    target = urlsplit(issued_url)
    path = f"{target.path}?{target.query}"

    response = client.get(path)
    head = client.head(path)

    assert response.status_code == 200
    assert response.content.startswith(b"\xff\xd8")
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["content-disposition"].startswith("inline;")
    assert "launch.jpg" in response.headers["content-disposition"]
    assert response.headers["referrer-policy"] == "no-referrer"
    assert head.status_code == 200
    assert head.content == b""


def test_route_passes_through_conforming_jpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / "ready.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (1080, 1080), "blue").save(image_path, format="JPEG")
    expected = image_path.read_bytes()
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename="ready.png",
        mime_type="image/png",
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(_issue_ticket_url(monkeypatch, db, file_id))

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-type"] == "image/jpeg"
    assert "ready.png" in response.headers["content-disposition"]


def test_route_maps_decode_failure_to_422(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / "broken.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 640), "blue").save(image_path)
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename="broken.png",
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(_issue_ticket_url(monkeypatch, db, file_id))
    image_path.write_bytes(b"not-an-image")

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Registered image could not be decoded for Instagram"
    )


def _instagram_source_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    *,
    filename: str,
    write: Callable[[Path], None],
) -> tuple[Path, str]:
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / filename
    image_path.parent.mkdir(parents=True)
    write(image_path)
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename=filename,
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    return image_path, f"/api/files/external-media/{file_id}?ticket={_ticket(file_id)}"


@OVERSIZED_SOURCES
@pytest.mark.parametrize(
    ("rewritten", "bytes_read"),
    [("before-open", 0), ("while-open", SOURCE_LIMIT + 1)],
)
def test_route_bounds_what_it_reads_of_an_oversized_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    filename: str,
    write_oversized: Callable[[Path, int], None],
    rewritten: str,
    bytes_read: int,
):
    # Why: every GET of an issued URL parses the file again, and the sandbox
    # can rewrite it after issuing (#3596). A file over the limit is refused
    # unread; one rewritten after that check is read only to one byte past it.
    monkeypatch.setattr(external_media, "INSTAGRAM_MAX_SOURCE_BYTES", SOURCE_LIMIT)
    if rewritten == "before-open":
        _, url = _instagram_source_url(
            monkeypatch,
            tmp_path,
            db,
            filename=filename,
            write=lambda path: write_oversized(path, OVERSIZED),
        )
        opened = count_source_reads(monkeypatch, files)
    else:
        image_path, url = _instagram_source_url(
            monkeypatch, tmp_path, db, filename=filename, write=write_small_image
        )
        opened = count_source_reads(
            monkeypatch,
            files,
            before_first_read=lambda: write_oversized(image_path, OVERSIZED),
        )

    response = client.get(url)

    assert [reader.bytes_read for reader in opened] == [bytes_read]
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Registered image exceeds the 1 MB Instagram source limit; "
        "compress or downscale it and try again"
    )


@pytest.mark.parametrize(("slack", "status"), [(0, 200), (-1, 422)])
def test_route_holds_a_conforming_jpeg_to_the_source_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    slack: int,
    status: int,
):
    # Why: a JPEG that needs no conversion is sent from the same bounded copy
    # that was checked, byte for byte; over the limit it is refused.
    image_path, url = _instagram_source_url(
        monkeypatch,
        tmp_path,
        db,
        filename="ready.jpg",
        write=lambda path: Image.new("RGB", (1080, 1080), "blue").save(path, "JPEG"),
    )
    expected = image_path.read_bytes()
    monkeypatch.setattr(
        external_media, "INSTAGRAM_MAX_SOURCE_BYTES", len(expected) + slack
    )

    response = client.get(url)

    assert response.status_code == status
    if status == 200:
        assert response.content == expected


@pytest.mark.parametrize(
    ("filename", "image_format"),
    [("ready.jpg", "JPEG"), ("converted.png", "PNG")],
    ids=["sent-as-read", "converted"],
)
def test_route_reads_a_delivered_source_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    filename: str,
    image_format: str,
):
    # Why: Pillow must be given the bounded copy, never the file (#3596). The
    # refusal paths above cannot tell: a route that checked the copy but then
    # handed the file to Pillow would still refuse those. On success the file
    # is read once, in full, and never again.
    image_path, url = _instagram_source_url(
        monkeypatch,
        tmp_path,
        db,
        filename=filename,
        write=lambda path: Image.new("RGB", (1080, 1080), "blue").save(
            path, image_format
        ),
    )
    opened = count_source_reads(monkeypatch, files)

    response = client.get(url)

    assert response.status_code == 200
    assert [reader.bytes_read for reader in opened] == [image_path.stat().st_size]


def test_route_converts_a_conforming_jpeg_over_the_jpeg_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    # Why: the route sends a JPEG that needs no conversion without checking its
    # size again; that holds only because one over Meta's JPEG byte limit is
    # always converted.
    image_path, url = _instagram_source_url(
        monkeypatch,
        tmp_path,
        db,
        filename="ready.jpg",
        write=lambda path: (
            Image.radial_gradient("L")
            .resize((1080, 1080))
            .convert("RGB")
            .save(path, "JPEG", quality=100)
        ),
    )
    original = image_path.read_bytes()
    monkeypatch.setattr(external_media, "INSTAGRAM_MAX_JPEG_BYTES", len(original) - 1)

    response = client.get(url)

    assert response.status_code == 200
    assert response.content != original
    assert len(response.content) < len(original)
    assert response.content.startswith(b"\xff\xd8")


@MULTI_SCAN_PIXEL_LIMIT_CASES
def test_route_holds_a_multi_scan_jpeg_to_the_png_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    layout: str,
    slack: int,
    refused_as: str | None,
):
    # Why: every GET decodes the image, and draft() does not shrink what
    # libjpeg holds for a JPEG with more than one scan (progressive, or a first
    # scan that codes only some components), so concurrent fetches of a large
    # one add up (#3596). It gets the PNG limit; one interleaved scan of the
    # same size does not.
    source = jpeg_400px(layout)
    _, url = _instagram_source_url(
        monkeypatch,
        tmp_path,
        db,
        filename="launch.jpg",
        write=lambda path: path.write_bytes(source),
    )
    monkeypatch.setattr(
        external_media, "INSTAGRAM_MAX_SOURCE_PIXELS", 400 * 400 + slack
    )

    response = client.get(url)

    if refused_as is None:
        assert response.status_code == 200
        assert response.content == source
    else:
        assert response.status_code == 422
        assert response.json()["detail"] == (
            "Registered image exceeds the Instagram source pixel limit of 159,999 "
            f"for {refused_as}; downscale it and try again"
        )


def test_route_requires_and_validates_ticket(client: TestClient):
    file_id = str(uuid4())

    assert client.get(f"/api/files/external-media/{file_id}").status_code == 422
    assert (
        client.get("/api/files/external-media/not-a-uuid?ticket=x").status_code == 401
    )


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            DurableObjectIntegrityError(
                "checksum mismatch",
                storage_key="users/7/launch.png",
            ),
            409,
        ),
        (DurableObjectMissingError(Path("missing")), 404),
        (FilePublicationUnavailable("unavailable"), 404),
        (
            DurableStorageOperationError(
                "storage unavailable",
                storage_key="users/7/launch.png",
            ),
            503,
        ),
    ],
)
def test_route_maps_storage_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    expected_status: int,
):
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "launch.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 640), "blue").save(image_path)
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    file_id = str(uuid4())
    _record(db, file_id, storage_path=str(image_path))
    issued_url = _issue_ticket_url(monkeypatch, db, file_id)
    monkeypatch.setattr(
        files.ManagedFileRef,
        "materialize",
        Mock(side_effect=error),
    )
    target = urlsplit(issued_url)
    caplog.set_level(logging.WARNING, logger=ROUTE_LOGGER)

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == expected_status
    # Why: the purpose label is for connector images only.
    assert _delivery_logs(caplog) == []


def test_route_does_not_trust_local_path_outside_uploads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    uploads_dir = tmp_path / "uploads"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        storage_path=str(outside),
        storage_status="legacy",
        storage_key=None,
    )
    response = client.get(
        f"/api/files/external-media/{file_id}?ticket={_ticket(file_id)}"
    )

    assert response.status_code == 404


def test_route_serves_the_stored_png_for_facebook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    # Why: only Instagram needs JPEG normalization; Facebook must receive the
    # stored bytes with their own type, directly and without a redirect.
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / "wide.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGBA", (3000, 400), (0, 0, 255, 128)).save(image_path)
    expected = image_path.read_bytes()
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://testserver")
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename="wide.png",
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(
        external_media.issue_external_media_url(
            db,
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
        ).url
    )

    response = client.get(f"{target.path}?{target.query}", follow_redirects=False)

    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    ("filename", "image_format", "expected_type"),
    [("edited.png", "JPEG", "image/jpeg"), ("photo.jpg", "PNG", "image/png")],
)
def test_route_serves_facebook_images_with_the_type_of_their_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    filename: str,
    image_format: str,
    expected_type: str,
):
    # Why: issuance accepts PNG or JPEG bytes whatever the extension, and
    # image_tool saves every edit as .png. Under nosniff the declared type is
    # final, so it must name the bytes sent, not the file name.
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / filename
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "blue").save(image_path, image_format)
    expected = image_path.read_bytes()
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://testserver")
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename=filename,
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(
        external_media.issue_external_media_url(
            db,
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
        ).url
    )

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-type"] == expected_type


def test_route_refuses_a_facebook_image_over_the_delivery_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    # Why: the sandbox can rewrite a registered image after its URL is issued;
    # delivery reads at most the connector's limit and refuses anything larger
    # rather than sending a cut-off image.
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / "post.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "blue").save(image_path)
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://testserver")
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename="post.png",
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(
        external_media.issue_external_media_url(
            db,
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
        ).url
    )
    # files imports the limit by value, so the route's copy is the one to patch.
    monkeypatch.setattr(files, "FACEBOOK_MAX_IMAGE_BYTES", 32)

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Registered image exceeds the delivery size limit"
    )


def test_route_does_not_serve_a_file_swapped_for_a_link_after_issue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    # Why: the ticket is checked again at delivery; replacing the registered
    # file with a link after issuing must not deliver the link's target.
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / "post.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "blue").save(image_path)
    other = uploads_dir / "user_7" / "web_task_99" / "private.png"
    other.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "red").save(other)
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://testserver")
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename="post.png",
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(
        external_media.issue_external_media_url(
            db,
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
        ).url
    )
    image_path.unlink()
    image_path.symlink_to(other)

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == 404
    assert response.content != other.read_bytes()


@pytest.mark.parametrize(
    "purpose",
    [
        external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
        external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    ],
)
@pytest.mark.parametrize("swapped", ["file", "directory"])
def test_route_does_not_serve_a_path_swapped_after_its_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    purpose: external_media.ExternalMediaPurpose,
    swapped: str,
):
    # Why: the trust check and the response must cover the same bytes. A path
    # replaced by a link after the check passes, before the response opens it,
    # must not send the link's target.
    uploads_dir = tmp_path / "uploads"
    task_dir = uploads_dir / "user_7" / "web_task_11"
    image_path = task_dir / "post.png"
    task_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), "blue").save(image_path)
    other_dir = uploads_dir / "user_7" / "web_task_99"
    other_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), "red").save(other_dir / "post.png")
    other_bytes = (other_dir / "post.png").read_bytes()
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://testserver")
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename="post.png",
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(
        external_media.issue_external_media_url(
            db, file_id=file_id, user_id=7, task_id=11, purpose=purpose
        ).url
    )
    original_materialize = files.ManagedFileRef.materialize

    def materialize_then_swap(self, *args, **kwargs):
        path = original_materialize(self, *args, **kwargs)
        assert path == image_path
        if swapped == "file":
            image_path.unlink()
            image_path.symlink_to(other_dir / "post.png")
        else:
            task_dir.rename(tmp_path / "moved_task")
            task_dir.symlink_to(other_dir)
        return path

    monkeypatch.setattr(files.ManagedFileRef, "materialize", materialize_then_swap)

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == 404
    assert response.content != other_bytes


def test_route_serves_through_a_linked_upload_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    # Why: an operator may mount the upload root as a link; only links below
    # the user root are refused when the response opens the file.
    volume = tmp_path / "volume"
    (volume / "user_7" / "web_task_11").mkdir(parents=True)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.symlink_to(volume)
    image_path = uploads_dir / "user_7" / "web_task_11" / "post.png"
    Image.new("RGB", (64, 64), "blue").save(image_path)
    expected = image_path.read_bytes()
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://testserver")
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename="post.png",
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(
        external_media.issue_external_media_url(
            db,
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.FACEBOOK_PUBLISH,
        ).url
    )

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == 200
    assert response.content == expected


# ---------------------------------------------------------------------------
# connector-image purpose
# ---------------------------------------------------------------------------

ROUTE_LOGGER = files.logger.name
INVALID_TICKET = {"detail": "Invalid or expired external media ticket"}


def _image_content(image_format: str) -> bytes:
    output = BytesIO()
    save = {"lossless": True} if image_format == "WEBP" else {}
    Image.effect_noise((96, 64), 64).convert("RGB").save(output, image_format, **save)
    return output.getvalue()


def _serve_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    *,
    name: str = "logo.png",
    content: bytes | None = None,
    purpose: str = "connector-image",
) -> tuple[str, str]:
    """Register an image under ``name``, issue a ticket and return (id, path)."""
    uploads_dir = tmp_path / "uploads"
    image_path = uploads_dir / "user_7" / "web_task_11" / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(_image_content("PNG") if content is None else content)
    monkeypatch.setattr(files, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setattr(external_media, "get_uploads_dir", lambda: uploads_dir)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://testserver")
    file_id = str(uuid4())
    _record(
        db,
        file_id,
        filename=name,
        storage_path=str(image_path),
        storage_status="legacy",
        storage_key=None,
    )
    target = urlsplit(
        external_media.issue_external_media_url(
            db,
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose(purpose),
        ).url
    )
    return file_id, f"{target.path}?{target.query}"


@pytest.mark.parametrize(
    ("name", "image_format", "media_type", "extension"),
    [
        ("logo.png", "PNG", "image/png", "png"),
        ("logo.jpg", "JPEG", "image/jpeg", "jpg"),
        ("logo.webp", "WEBP", "image/webp", "webp"),
        # The bytes decide what is delivered, not the name.
        ("logo.webp", "PNG", "image/png", "png"),
        ("logo.png", "WEBP", "image/webp", "webp"),
    ],
)
def test_route_serves_connector_images_as_stored_under_the_file_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    name: str,
    image_format: str,
    media_type: str,
    extension: str,
):
    content = _image_content(image_format)
    file_id, path = _serve_image(monkeypatch, tmp_path, db, name=name, content=content)

    response = client.get(path, follow_redirects=False)

    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"] == media_type
    assert response.headers["content-disposition"] == (
        f'inline; filename="{file_id}.{extension}"; '
        f"filename*=UTF-8''{file_id}.{extension}"
    )


def test_route_keeps_the_facebook_file_name_and_type_rule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    # Why: a WebP swapped in after a Facebook issue keeps the registered type,
    # and the Facebook download keeps the name the file was uploaded under.
    name = "visitor-upload.png"
    _, path = _serve_image(
        monkeypatch, tmp_path, db, name=name, purpose="facebook-publish"
    )
    (tmp_path / "uploads" / "user_7" / "web_task_11" / name).write_bytes(
        _image_content("WEBP")
    )

    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert name in response.headers["content-disposition"]


@pytest.mark.parametrize(
    "purpose", [member.value for member in external_media.ExternalMediaPurpose]
)
def test_every_purpose_has_an_explicit_branch_at_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    purpose: str,
):
    # Why: a purpose added to the enum without a delivery branch is refused,
    # so this fails until that purpose is handled.
    _, path = _serve_image(monkeypatch, tmp_path, db, purpose=purpose)

    assert client.get(path).status_code == 200


def test_route_answers_a_purpose_without_a_branch_like_an_invalid_ticket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
):
    # Why: a purpose the route has no branch for must be refused the same way
    # as an invalid ticket, never served by a neighbouring branch.
    file_id, path = _serve_image(monkeypatch, tmp_path, db)
    redeemed = external_media.RedeemedExternalMedia(
        record=db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one(),
        purpose="future-purpose",  # type: ignore[arg-type]
        media_type="image/png",
    )
    monkeypatch.setattr(
        files, "redeem_external_media_ticket", Mock(return_value=redeemed)
    )

    response = client.get(path)

    assert response.status_code == 401
    assert response.json() == INVALID_TICKET


def _delivery_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "External media delivery" in record.getMessage()
    ]


@pytest.mark.parametrize(
    ("case", "status", "reason"),
    [
        ("over-the-delivery-cap", 422, "unprocessable"),
        ("cannot-be-opened", 404, "file_unavailable"),
        ("bytes-gone", 404, "file_unavailable"),
        ("integrity-failed", 409, "integrity_failed"),
        ("storage-unavailable", 503, "storage_unavailable"),
        ("record-gone", 401, "ticket_rejected"),
        ("wrong-scope", 401, "ticket_rejected"),
        ("expired", 401, "expired"),
    ],
)
def test_route_logs_a_refused_connector_image_by_category(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    case: str,
    status: int,
    reason: str,
):
    if case in {"record-gone", "wrong-scope", "expired"}:
        # The ticket decodes and names the purpose, but no record matches it,
        # or it is past its expiry.
        file_id = str(uuid4())
        claims: dict[str, object] = {}
        if case == "wrong-scope":
            claims = {"user_id": "7"}
        elif case == "expired":
            claims = {"exp": datetime.now(timezone.utc) - timedelta(minutes=1)}
        ticket = _ticket(file_id, purpose="connector-image", **claims)
        path = f"/api/files/external-media/{file_id}?ticket={ticket}"
    else:
        file_id, path = _serve_image(monkeypatch, tmp_path, db)
    if case == "over-the-delivery-cap":
        monkeypatch.setattr(files, "FACEBOOK_MAX_IMAGE_BYTES", 32)
    elif case == "cannot-be-opened":
        monkeypatch.setattr(files, "open_delivery_file", Mock(side_effect=OSError))
    elif case == "bytes-gone":
        monkeypatch.setattr(
            files.ManagedFileRef,
            "materialize",
            Mock(side_effect=DurableObjectMissingError(Path("missing"))),
        )
    elif case == "integrity-failed":
        monkeypatch.setattr(
            files.ManagedFileRef,
            "materialize",
            Mock(
                side_effect=DurableObjectIntegrityError(
                    "checksum mismatch", storage_key="users/7/logo.png"
                )
            ),
        )
    elif case == "storage-unavailable":
        monkeypatch.setattr(
            files.ManagedFileRef,
            "materialize",
            Mock(
                side_effect=DurableStorageOperationError(
                    "storage unavailable", storage_key="users/7/logo.png"
                )
            ),
        )
    caplog.set_level(logging.WARNING, logger=ROUTE_LOGGER)

    response = client.get(path)

    assert response.status_code == status
    refusal = (
        f"External media delivery refused purpose=connector-image "
        f"file_id={file_id} reason={reason}"
    )
    assert _delivery_logs(caplog) == [refusal]
    route_warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == ROUTE_LOGGER and r.levelno == logging.WARNING
    ]
    if case == "storage-unavailable":
        # The storage fault keeps its own record, after the refusal.
        assert route_warnings[0] == refusal
        assert route_warnings[1].startswith(
            "Durable storage unavailable during external media delivery "
        )
        assert len(route_warnings) == 2
    else:
        assert route_warnings == [refusal]
    if case == "expired":
        assert response.json() == INVALID_TICKET
        assert ticket_text_absent(caplog, path)


def ticket_text_absent(caplog: pytest.LogCaptureFixture, path: str) -> bool:
    ticket = path.split("ticket=", 1)[1]
    return all(ticket not in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "case",
    ["instagram", "facebook", "forged-signature", "other-file"],
)
def test_route_does_not_log_an_expired_ticket_that_is_not_a_connector_image(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    case: str,
) -> None:
    # Why: only an expired ticket whose verified signature, type, purpose and
    # file match the request may be attributed to a connector image.
    file_id = str(uuid4())
    expired = {"exp": datetime.now(timezone.utc) - timedelta(minutes=1)}
    if case == "instagram":
        ticket = _ticket(file_id, **expired)
    elif case == "facebook":
        ticket = _ticket(file_id, purpose="facebook-publish", **expired)
    elif case == "forged-signature":
        ticket = external_media.jwt.encode(
            {
                "type": external_media.EXTERNAL_MEDIA_TICKET_TYPE,
                "purpose": "connector-image",
                "file_id": file_id,
                "user_id": 7,
                "task_id": 11,
                **expired,
            },
            "not-the-signing-key" * 3,
            algorithm=external_media.JWT_ALGORITHM,
        )
    else:
        ticket = _ticket(str(uuid4()), purpose="connector-image", **expired)
    caplog.set_level(logging.DEBUG, logger=ROUTE_LOGGER)

    response = client.get(f"/api/files/external-media/{file_id}?ticket={ticket}")

    assert response.status_code == 401
    assert response.json() == INVALID_TICKET
    assert _delivery_logs(caplog) == []


@pytest.mark.parametrize("purpose", ["instagram-publish", "facebook-publish"])
def test_route_adds_no_log_for_the_existing_purposes_or_an_unread_ticket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db: Session,
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    purpose: str,
):
    _, path = _serve_image(monkeypatch, tmp_path, db, purpose=purpose)
    monkeypatch.setattr(files, "open_delivery_file", Mock(side_effect=OSError))
    gone = str(uuid4())
    caplog.set_level(logging.DEBUG, logger=ROUTE_LOGGER)

    missing = client.get(path)
    rejected = client.get(
        f"/api/files/external-media/{gone}?ticket={_ticket(gone, purpose=purpose)}"
    )
    unread = client.get(f"/api/files/external-media/{gone}?ticket=not-a-jwt")

    assert (missing.status_code, rejected.status_code) == (404, 401)
    assert unread.json() == INVALID_TICKET
    assert _delivery_logs(caplog) == []
