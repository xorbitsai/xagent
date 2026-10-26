from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest

from xagent.web.services import external_media


class _Query:
    def __init__(self, record):
        self.record = record

    def filter(self, *conditions):
        return self

    def first(self):
        return self.record


class _DB:
    def __init__(self, record):
        self.record = record

    def query(self, model):
        return _Query(self.record)


def _record(file_id: str, *, mime_type: str = "image/png"):
    return SimpleNamespace(
        file_id=file_id,
        user_id=7,
        task_id=11,
        filename="launch.png",
        mime_type=mime_type,
        storage_status="available",
        storage_path="/unused/launch.png",
        storage_key="users/7/launch.png",
        checksum="a" * 64,
    )


def test_issue_uses_direct_durable_url_when_available(monkeypatch):
    file_id = str(uuid4())
    record = _record(file_id)
    monkeypatch.setattr(
        external_media.ManagedFileRef,
        "signed_access_url",
        lambda self,
        **kwargs: "https://objects.example.test/launch.png?signature=secret",
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
    assert (
        external_media.redeem_external_media_ticket(
            _DB(record), file_id=file_id, ticket=ticket
        )
        is record
    )


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
            _DB(_record(file_id, mime_type="text/plain")),
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
        lambda db, **kwargs: record,
    )

    class _FileRef:
        def __init__(self, value):
            assert value is record

        def ensure_local(self):
            return image_path

    monkeypatch.setattr(files, "ManagedFileRef", _FileRef)

    response = files.external_media_file(file_id, ticket="ticket", db=_DB(record))

    assert Path(response.path) == image_path
    assert response.media_type == "image/png"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["referrer-policy"] == "no-referrer"
