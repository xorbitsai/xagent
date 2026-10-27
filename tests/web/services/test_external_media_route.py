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
