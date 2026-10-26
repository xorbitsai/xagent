from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
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
    filename: str = "launch.png",
    mime_type: str = "image/png",
) -> UploadedFile:
    return UploadedFile(
        file_id=file_id,
        user_id=7,
        task_id=11,
        filename=filename,
        mime_type=mime_type,
        storage_status="available",
        storage_path="/unused/launch.png",
        storage_key="users/7/launch.png",
        checksum="a" * 64,
        file_size=1,
    )


def test_issue_uses_direct_durable_url_when_available(monkeypatch):
    file_id = str(uuid4())
    record = _record(file_id)
    monkeypatch.delenv("XAGENT_EXTERNAL_MEDIA_URL_TTL_SECONDS", raising=False)
    monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
    monkeypatch.setattr(
        external_media.ManagedFileRef,
        "signed_access_url",
        lambda self, **kwargs: (
            "https://objects.example.test/launch.png?signature=secret"
        ),
    )

    issued = external_media.issue_external_media_url(
        _DB(record),
        file_id=f"file:{file_id}",
        user_id=7,
        task_id=11,
        purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
    )

    assert issued.direct_object_url is True
    assert issued.expires_in == 600
    assert issued.url.startswith("https://objects.example.test/")


def test_issue_reports_direct_url_integrity_failure(monkeypatch):
    file_id = str(uuid4())
    record = _record(file_id, mime_type="image/jpeg")
    monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
    monkeypatch.setattr(
        external_media.ManagedFileRef,
        "signed_access_url",
        Mock(
            side_effect=external_media.DurableObjectIntegrityError(
                "checksum mismatch",
                storage_key="users/7/launch.jpg",
            )
        ),
    )

    with pytest.raises(
        external_media.ExternalMediaUnavailable,
        match="integrity verification",
    ):
        external_media.issue_external_media_url(
            _DB(record),
            file_id=file_id,
            user_id=7,
            task_id=11,
            purpose=external_media.ExternalMediaPurpose.INSTAGRAM_PUBLISH,
        )


def test_issue_falls_back_to_public_ticket_url(monkeypatch):
    file_id = str(uuid4())
    record = _record(file_id)
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.test/")
    monkeypatch.setattr(
        external_media.ManagedFileRef,
        "signed_access_url",
        lambda self, **kwargs: None,
    )

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
    assert redeemed.media_type == "image/png"


def test_issue_requires_public_base_for_local_storage(monkeypatch):
    file_id = str(uuid4())
    monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
    monkeypatch.setattr(
        external_media.ManagedFileRef,
        "signed_access_url",
        lambda self, **kwargs: None,
    )

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


def test_issue_rejects_non_image_record():
    file_id = str(uuid4())

    with pytest.raises(external_media.ExternalMediaNotFound):
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
    monkeypatch.setattr(
        external_media.ManagedFileRef,
        "signed_access_url",
        lambda self, **kwargs: None,
    )
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
    image_path.write_bytes(b"image-bytes")

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

    assert Path(response.path) == image_path
    assert response.media_type == "image/png"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["referrer-policy"] == "no-referrer"
