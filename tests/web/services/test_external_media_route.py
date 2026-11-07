from collections.abc import Callable
from datetime import datetime, timedelta, timezone
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
    OVERSIZED,
    OVERSIZED_SOURCES,
    SOURCE_LIMIT,
    count_source_reads,
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

    response = client.get(f"{target.path}?{target.query}")

    assert response.status_code == expected_status


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
