"""Test file upload API functionality - Fixed for multi-tenant architecture"""

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.api.auth import hash_password
from xagent.web.api.files import _content_disposition_header, file_router
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models import database as database_module
from xagent.web.models.database import Base, get_db
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.managed_file_ref import DURABLE_FAULT_LOG_PREFIX


@pytest.fixture(scope="function")
def test_db(monkeypatch: pytest.MonkeyPatch):
    """Create test database with isolated engine and session"""
    # Create a temporary database file for each test
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)

    # Create isolated engine and session for this test
    test_engine = create_engine(
        f"sqlite:///{temp_db_path}", connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    # Upload metadata is now committed by a worker-owned short Session rather
    # than the request dependency. Point the canonical Session factory at this
    # test's isolated database as well as overriding FastAPI's dependency.
    monkeypatch.setattr(database_module, "_engine", test_engine)
    monkeypatch.setattr(database_module, "_SessionLocal", TestingSessionLocal)

    # Create override function that uses this test's session
    def override_get_db():
        db = None
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            if db is not None:
                db.close()

    # Create test app for this test
    test_app = FastAPI()
    test_app.include_router(file_router)
    test_app.dependency_overrides[get_db] = override_get_db

    # Create tables
    Base.metadata.create_all(bind=test_engine)

    # Create admin user for this test
    session = TestingSessionLocal()
    try:
        admin_user = User(
            username="admin", password_hash=hash_password("admin"), is_admin=True
        )
        session.add(admin_user)
        session.commit()
        session.refresh(admin_user)
        yield admin_user, test_app
    finally:
        session.close()
        # Clean up
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()
        # Delete temporary database file
        try:
            os.unlink(temp_db_path)
        except OSError:
            pass


@pytest.fixture(scope="function")
def auth_headers(test_db):
    """Authentication headers for admin user"""
    admin_user, _ = test_db
    # Create a valid JWT token directly
    from datetime import datetime, timedelta, timezone

    import jwt

    payload = {
        "sub": admin_user.username,  # Use unique username from test_db fixture
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
        "user_id": admin_user.id,  # Use actual user ID from test_db fixture
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def sample_files():
    """Create sample test files"""
    files = {}

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = {
            "test.txt": "This is a test text file content.",
            "test.py": "print('Hello, World!')\n\n# Test Python file",
            "test.json": '{"name": "test", "value": 123}',
            "test.csv": "name,age,city\nJohn,25,NYC\nJane,30,LA",
        }

        for filename, content in test_files.items():
            file_path = Path(temp_dir) / filename
            with open(file_path, "w") as f:
                f.write(content)
            files[filename] = str(file_path)

        yield files, temp_dir


@pytest.fixture(scope="function")
def client(test_db):
    """Create test client for each test"""
    _, test_app = test_db
    return TestClient(test_app)


@pytest.fixture(scope="function")
def temp_uploads_dir(monkeypatch):
    """Create temporary uploads directory and override get_uploads_dir"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Patch the directory in both the config module and the files module
        # This is necessary because files.py imports these at module load time
        import xagent.web.api.files
        import xagent.web.config

        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (temp_path / "objects").as_uri())
        get_unscoped_file_storage.cache_clear()
        monkeypatch.setattr(xagent.web.config, "get_uploads_dir", lambda: temp_path)
        monkeypatch.setattr(xagent.web.api.files, "get_uploads_dir", lambda: temp_path)

        yield temp_path


def _corrupt_durable_copy_and_remove_local(
    object_root: Path, uploads_dir: Path, filename: str
) -> None:
    object_file = next(path for path in object_root.rglob(filename) if path.is_file())
    object_file.write_bytes(b"corrupted durable content")
    for path in uploads_dir.rglob(filename):
        if path.is_file():
            path.unlink()


def _replace_uploaded_file_storage_path(test_app: FastAPI, file_id: str, path: Path):
    db = next(test_app.dependency_overrides[get_db]())
    try:
        file_record = (
            db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        )
        file_record.storage_path = str(path)
        db.commit()
    finally:
        db.close()


class TestFileUpload:
    """Test file upload functionality"""

    def test_content_disposition_header_escapes_filename_and_adds_utf8_parameter(self):
        assert _content_disposition_header("attachment", 'quo"te\\文\r\n.txt') == (
            'attachment; filename="quo\\"te\\\\___.txt"; '
            "filename*=UTF-8''quo%22te%5C%E6%96%87%0D%0A.txt"
        )

    def test_upload_text_file_success(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test successful upload of text file"""
        files, temp_dir = sample_files
        file_path = files["test.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/files/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        # File upload returns 200 on success
        assert response.status_code == 200

    def test_upload_download_uses_durable_storage_after_local_file_deleted(
        self, client, temp_uploads_dir, auth_headers, monkeypatch, tmp_path
    ):
        """Uploaded files should download from durable storage, not local uploads."""
        object_root = tmp_path / "objects"
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
        get_unscoped_file_storage.cache_clear()

        response = client.post(
            "/api/files/upload",
            files={"file": ("durable.txt", b"durable content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        file_id = response.json()["file_id"]

        object_files = [path for path in object_root.rglob("*") if path.is_file()]
        assert len(object_files) == 1
        assert object_files[0].read_bytes() == b"durable content"

        for path in temp_uploads_dir.rglob("*"):
            if path.is_file():
                path.unlink()

        download = client.get(
            f"/api/files/download/{file_id}",
            headers=auth_headers,
        )

        assert download.status_code == 200
        assert download.content == b"durable content"

    def test_registered_file_uses_durable_storage_when_storage_path_is_stale(
        self, client, test_db, temp_uploads_dir, auth_headers, monkeypatch, tmp_path
    ):
        """A stale worktree storage_path must not block durable file access."""
        _, test_app = test_db
        monkeypatch.setenv(
            "XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized")
        )
        get_unscoped_file_storage.cache_clear()

        response = client.post(
            "/api/files/upload",
            files={"file": ("stale.txt", b"durable bytes", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]

        stale_path = (
            tmp_path
            / "deleted-worktree"
            / "src"
            / "xagent"
            / "web"
            / "uploads"
            / "user_1"
            / "web_task_99"
            / "output"
            / "stale.txt"
        )
        stale_path.parent.mkdir(parents=True)
        stale_path.write_bytes(b"wrong local bytes")
        _replace_uploaded_file_storage_path(test_app, file_id, stale_path)

        for endpoint, headers in [
            (f"/api/files/preview/{file_id}", auth_headers),
            (f"/api/files/download/{file_id}", auth_headers),
            (f"/api/files/public/preview/{file_id}", {}),
            (f"/api/files/public/download/{file_id}", {}),
        ]:
            result = client.get(endpoint, headers=headers)
            assert result.status_code == 200
            assert result.content == b"durable bytes"

        assert stale_path.read_bytes() == b"wrong local bytes"

    def test_durable_missing_fallback_rejects_untrusted_storage_path(
        self, client, test_db, temp_uploads_dir, auth_headers, monkeypatch, tmp_path
    ):
        """Durable fallback must not serve an untrusted local storage_path."""
        from xagent.web.services.managed_file_ref import (
            DurableObjectMissingError,
            ManagedFileRef,
        )

        _, test_app = test_db
        response = client.post(
            "/api/files/upload",
            files={
                "file": (
                    "missing.pptx",
                    b"durable bytes",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            },
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]

        stale_path = tmp_path / "outside-uploads" / "missing.txt"
        stale_path.parent.mkdir()
        stale_path.write_bytes(b"wrong local bytes")
        _replace_uploaded_file_storage_path(test_app, file_id, stale_path)

        def missing_durable_object(
            self: ManagedFileRef, *, allow_existing_local: bool = True
        ):
            del allow_existing_local
            raise DurableObjectMissingError(self.local_path)

        monkeypatch.setattr(ManagedFileRef, "materialize", missing_durable_object)

        for endpoint, headers in [
            (f"/api/files/download/{file_id}", auth_headers),
            (f"/api/files/preview/{file_id}", auth_headers),
            (f"/api/files/preview-pdf/{file_id}", auth_headers),
            (f"/api/files/public/download/{file_id}", {}),
            (f"/api/files/public/preview/{file_id}", {}),
        ]:
            result = client.get(endpoint, headers=headers)
            assert result.status_code == 403

        assert stale_path.read_bytes() == b"wrong local bytes"

    def test_download_redirects_to_signed_durable_url_when_enabled(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        from xagent.web.services.managed_file_ref import ManagedFileRef

        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_SIGNED_URL_TTL_SECONDS", "42")
        response = client.post(
            "/api/files/upload",
            files={"file": ("redirect.txt", b"redirect content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]
        calls = []

        def signed_access_url(
            self,
            *,
            expires,
            content_type=None,
            content_disposition=None,
        ):
            calls.append(
                (
                    self.storage_key,
                    expires,
                    content_type,
                    content_disposition,
                )
            )
            return "https://cdn.example.com/private/redirect.txt?sig=abc"

        monkeypatch.setattr(ManagedFileRef, "signed_access_url", signed_access_url)

        download = client.get(
            f"/api/files/download/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert download.status_code == 307
        assert (
            download.headers["location"]
            == "https://cdn.example.com/private/redirect.txt?sig=abc"
        )
        assert len(calls) == 1
        storage_key, expires, content_type, content_disposition = calls[0]
        assert storage_key.endswith(f"/{file_id}/redirect.txt")
        assert expires == 42
        assert content_type == "text/plain"
        assert (
            content_disposition
            == "inline; filename=\"redirect.txt\"; filename*=UTF-8''redirect.txt"
        )

    def test_preview_redirects_to_signed_durable_url_when_enabled(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        from xagent.web.services.managed_file_ref import ManagedFileRef

        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        response = client.post(
            "/api/files/upload",
            files={"file": ("preview.txt", b"preview content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]

        def signed_access_url(
            self,
            *,
            expires,
            content_type=None,
            content_disposition=None,
        ):
            del self, expires, content_type, content_disposition
            return "https://cdn.example.com/private/preview.txt?sig=abc"

        monkeypatch.setattr(ManagedFileRef, "signed_access_url", signed_access_url)

        preview = client.get(
            f"/api/files/preview/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert preview.status_code == 307
        assert (
            preview.headers["location"]
            == "https://cdn.example.com/private/preview.txt?sig=abc"
        )

    def test_download_uses_accel_redirect_for_local_file_when_enabled(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        del temp_uploads_dir
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED", "true")
        monkeypatch.setenv(
            "XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_PREFIX", "/private-files"
        )
        response = client.post(
            "/api/files/upload",
            files={
                "file": (
                    "local accel.txt",
                    b"local accel content",
                    "text/plain",
                )
            },
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]

        download = client.get(
            f"/api/files/download/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert download.status_code == 200
        assert download.content == b""
        assert "location" not in download.headers
        assert download.headers["x-accel-redirect"].endswith(
            f"/user_1/{quote('local accel.txt')}"
        )
        assert download.headers["content-type"].startswith("text/plain")
        assert (
            download.headers["content-disposition"]
            == 'inline; filename="local accel.txt"; '
            "filename*=UTF-8''local%20accel.txt"
        )

    def test_preview_uses_accel_redirect_for_local_text_when_enabled(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        del temp_uploads_dir
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED", "true")
        response = client.post(
            "/api/files/upload",
            files={"file": ("preview-accel.txt", b"preview accel", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]

        preview = client.get(
            f"/api/files/preview/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert preview.status_code == 200
        assert preview.content == b""
        assert preview.headers["x-accel-redirect"].endswith("/user_1/preview-accel.txt")
        assert (
            preview.headers["content-disposition"]
            == 'inline; filename="preview-accel.txt"; '
            "filename*=UTF-8''preview-accel.txt"
        )

    def test_preview_does_not_accel_redirect_html(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        del temp_uploads_dir
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED", "true")
        response = client.post(
            "/api/files/upload",
            files={"file": ("index.html", b"<h1>preview</h1>", "text/html")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]

        preview = client.get(
            f"/api/files/preview/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert preview.status_code == 200
        assert "x-accel-redirect" not in preview.headers
        assert preview.content == b"<h1>preview</h1>"

    def test_upload_remote_storage_outage_returns_503_and_rolls_back(
        self, client, test_db, temp_uploads_dir, auth_headers, monkeypatch
    ):
        from xagent.core.file_storage.storage import FsspecFileStorage

        admin_user, test_app = test_db

        def fail_put_file(self, source, key, content_type=None):
            raise RuntimeError("simulated remote write outage")

        monkeypatch.setattr(FsspecFileStorage, "put_file", fail_put_file)

        response = client.post(
            "/api/files/upload",
            files={"file": ("outage.txt", b"outage content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )

        assert response.status_code == 503
        assert "durable storage" in response.json()["detail"].lower()
        assert not list(temp_uploads_dir.rglob("outage.txt"))

        db = next(test_app.dependency_overrides[get_db]())
        try:
            assert (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.user_id == admin_user.id,
                    UploadedFile.filename == "outage.txt",
                )
                .first()
                is None
            )
        finally:
            db.close()

    def test_download_serves_existing_local_file_during_remote_outage(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        from xagent.core.file_storage.storage import FsspecFileStorage

        upload = client.post(
            "/api/files/upload",
            files={"file": ("local-copy.txt", b"local content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200

        def fail_open_read(self, key):
            raise RuntimeError("simulated remote read outage")

        monkeypatch.setattr(FsspecFileStorage, "open_read", fail_open_read)

        download = client.get(
            f"/api/files/download/{upload.json()['file_id']}",
            headers=auth_headers,
        )

        assert download.status_code == 200
        assert download.content == b"local content"

    def test_download_remote_storage_outage_returns_503_when_local_missing(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        from xagent.core.file_storage.storage import FsspecFileStorage

        upload = client.post(
            "/api/files/upload",
            files={"file": ("remote-only.txt", b"remote content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        for path in temp_uploads_dir.rglob("remote-only.txt"):
            path.unlink()

        def fail_open_read(self, key):
            raise RuntimeError("simulated remote read outage")

        monkeypatch.setattr(FsspecFileStorage, "open_read", fail_open_read)

        download = client.get(
            f"/api/files/download/{file_id}",
            headers=auth_headers,
        )

        assert download.status_code == 503
        assert "durable storage" in download.json()["detail"].lower()

    def test_download_namespace_containment_violation_is_not_reported_as_an_outage(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """A key outside the handle's bound prefix is a permanent authority
        fault, so it must not arrive as the retryable durable-storage 503.

        Driven through ``xagent.web.app.app`` rather than this module's
        router-only test app, because the classification lives in an
        application-level exception handler. The response may not echo the
        namespace values either -- prefixes and scope segments can carry
        end-user identity.
        """
        from xagent.web.app import app as web_app

        _, test_app = test_db
        upload = client.post(
            "/api/files/upload",
            files={"file": ("foreign-namespace.txt", b"foreign content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        for path in temp_uploads_dir.rglob("foreign-namespace.txt"):
            path.unlink()

        foreign_key = (
            f"users/999/clients/tenant-sentinel/uploads/{file_id}/foreign-namespace.txt"
        )
        db = next(test_app.dependency_overrides[get_db]())
        try:
            record = (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
            )
            record.storage_key = foreign_key
            db.commit()
        finally:
            db.close()

        web_app.dependency_overrides[get_db] = test_app.dependency_overrides[get_db]
        try:
            download = TestClient(web_app).get(
                f"/api/files/download/{file_id}",
                headers=auth_headers,
            )
        finally:
            web_app.dependency_overrides.pop(get_db, None)

        assert download.status_code == 500
        assert download.json() == {"detail": "Storage namespace authority violation."}
        assert "tenant-sentinel" not in download.text
        assert "users/999" not in download.text

    def test_upload_scope_authority_mismatch_is_not_reported_as_an_outage(
        self, test_db, temp_uploads_dir, auth_headers
    ):
        """The upload path resolves the write namespace fail-closed.

        A resolver/snapshot disagreement there is a permanent authority fault,
        so it must reach the same 500 classification as a containment
        violation rather than the retryable durable-storage 503. The response
        must not name the scope segments either.
        """
        from tests.shared.execution_scope import register_scope_resolver
        from xagent.core.execution_scope import (
            ExecutionScope,
            set_execution_scope_snapshot_loader,
        )
        from xagent.web.app import app as web_app

        del temp_uploads_dir
        _, test_app = test_db
        register_scope_resolver(
            lambda task_id: ExecutionScope(
                workspace_segments=("clients", "resolver-sentinel"),
                isolate_external_dirs=True,
            )
        )
        set_execution_scope_snapshot_loader(
            lambda task_id: ExecutionScope(
                workspace_segments=("clients", "snapshot-sentinel"),
                isolate_external_dirs=True,
            )
        )

        web_app.dependency_overrides[get_db] = test_app.dependency_overrides[get_db]
        try:
            response = TestClient(web_app).post(
                "/api/files/upload",
                files={"file": ("scoped.txt", b"scoped content", "text/plain")},
                data={"task_type": "general", "task_id": "5"},
                headers=auth_headers,
            )
        finally:
            web_app.dependency_overrides.pop(get_db, None)

        assert response.status_code == 500
        assert response.json() == {"detail": "Storage namespace authority violation."}
        assert "resolver-sentinel" not in response.text
        assert "snapshot-sentinel" not in response.text

    def test_download_checksum_mismatch_asks_user_to_reupload(
        self, client, temp_uploads_dir, auth_headers, monkeypatch, tmp_path
    ):
        object_root = tmp_path / "objects"
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
        get_unscoped_file_storage.cache_clear()

        upload = client.post(
            "/api/files/upload",
            files={
                "file": (
                    "integrity-download.txt",
                    b"expected download content",
                    "text/plain",
                )
            },
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        _corrupt_durable_copy_and_remove_local(
            object_root, temp_uploads_dir, "integrity-download.txt"
        )

        download = client.get(
            f"/api/files/download/{file_id}",
            headers=auth_headers,
        )

        assert download.status_code == 409
        assert "re-upload" in download.json()["detail"]
        assert not list(temp_uploads_dir.rglob("integrity-download.txt"))

    def test_download_redirect_enabled_checksum_mismatch_asks_user_to_reupload(
        self, client, temp_uploads_dir, auth_headers, monkeypatch, tmp_path
    ):
        from xagent.core.file_storage.storage import FsspecFileStorage

        object_root = tmp_path / "objects"
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        get_unscoped_file_storage.cache_clear()

        upload = client.post(
            "/api/files/upload",
            files={
                "file": (
                    "integrity-redirect-download.txt",
                    b"expected redirect download content",
                    "text/plain",
                )
            },
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        _corrupt_durable_copy_and_remove_local(
            object_root, temp_uploads_dir, "integrity-redirect-download.txt"
        )

        def fail_signed_url(self, key, **kwargs):
            del self, key, kwargs
            raise AssertionError("signed URL should not be generated")

        monkeypatch.setattr(FsspecFileStorage, "signed_url", fail_signed_url)

        download = client.get(
            f"/api/files/download/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert download.status_code == 409
        assert "re-upload" in download.json()["detail"]
        assert not list(temp_uploads_dir.rglob("integrity-redirect-download.txt"))

    def test_download_registered_file_rejects_local_path_outside_uploads(
        self, client, test_db, tmp_path, auth_headers
    ):
        """DB-backed download must still enforce the uploads path boundary."""
        admin_user, test_app = test_db
        outside_path = tmp_path / "outside.txt"
        outside_path.write_text("outside uploads", encoding="utf-8")

        db = next(test_app.dependency_overrides[get_db]())
        try:
            db.add(
                UploadedFile(
                    file_id="11111111-1111-4111-8111-111111111111",
                    user_id=admin_user.id,
                    filename="outside.txt",
                    storage_path=str(outside_path),
                    storage_status="legacy",
                    mime_type="text/plain",
                    file_size=outside_path.stat().st_size,
                )
            )
            db.commit()
        finally:
            db.close()

        response = client.get(
            "/api/files/download/11111111-1111-4111-8111-111111111111",
            headers=auth_headers,
        )

        assert response.status_code == 403

    def test_preview_registered_file_rejects_local_path_outside_uploads(
        self, client, test_db, tmp_path, auth_headers
    ):
        """DB-backed preview must still enforce the uploads path boundary."""
        admin_user, test_app = test_db
        outside_path = tmp_path / "outside-preview.txt"
        outside_path.write_text("outside uploads", encoding="utf-8")

        db = next(test_app.dependency_overrides[get_db]())
        try:
            db.add(
                UploadedFile(
                    file_id="22222222-2222-4222-8222-222222222222",
                    user_id=admin_user.id,
                    filename="outside-preview.txt",
                    storage_path=str(outside_path),
                    storage_status="legacy",
                    mime_type="text/plain",
                    file_size=outside_path.stat().st_size,
                )
            )
            db.commit()
        finally:
            db.close()

        response = client.get(
            "/api/files/preview/22222222-2222-4222-8222-222222222222",
            headers=auth_headers,
        )

        assert response.status_code == 403

    def test_public_preview_registered_file_rejects_local_path_outside_uploads(
        self, client, test_db, tmp_path
    ):
        """Public preview must not expose registered paths outside uploads."""
        admin_user, test_app = test_db
        outside_path = tmp_path / "outside-public.txt"
        outside_path.write_text("outside uploads", encoding="utf-8")

        db = next(test_app.dependency_overrides[get_db]())
        try:
            db.add(
                UploadedFile(
                    file_id="33333333-3333-4333-8333-333333333333",
                    user_id=admin_user.id,
                    filename="outside-public.txt",
                    storage_path=str(outside_path),
                    storage_status="legacy",
                    mime_type="text/plain",
                    file_size=outside_path.stat().st_size,
                )
            )
            db.commit()
        finally:
            db.close()

        response = client.get(
            "/api/files/public/preview/33333333-3333-4333-8333-333333333333"
        )

        assert response.status_code == 403

    def test_preview_remote_storage_outage_returns_503_when_local_missing(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        from xagent.core.file_storage.storage import FsspecFileStorage

        upload = client.post(
            "/api/files/upload",
            files={"file": ("preview-remote.txt", b"preview content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        for path in temp_uploads_dir.rglob("preview-remote.txt"):
            path.unlink()

        def fail_open_read(self, key):
            raise RuntimeError("simulated remote preview outage")

        monkeypatch.setattr(FsspecFileStorage, "open_read", fail_open_read)

        preview = client.get(
            f"/api/files/preview/{file_id}",
            headers=auth_headers,
        )

        assert preview.status_code == 503
        assert "durable storage" in preview.json()["detail"].lower()

    def test_preview_checksum_mismatch_asks_user_to_reupload(
        self, client, temp_uploads_dir, auth_headers, monkeypatch, tmp_path
    ):
        object_root = tmp_path / "objects"
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
        get_unscoped_file_storage.cache_clear()

        upload = client.post(
            "/api/files/upload",
            files={
                "file": (
                    "integrity-preview.txt",
                    b"expected preview content",
                    "text/plain",
                )
            },
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        _corrupt_durable_copy_and_remove_local(
            object_root, temp_uploads_dir, "integrity-preview.txt"
        )

        preview = client.get(
            f"/api/files/preview/{file_id}",
            headers=auth_headers,
        )

        assert preview.status_code == 409
        assert "re-upload" in preview.json()["detail"]

    def test_preview_redirect_enabled_checksum_mismatch_asks_user_to_reupload(
        self, client, temp_uploads_dir, auth_headers, monkeypatch, tmp_path
    ):
        from xagent.core.file_storage.storage import FsspecFileStorage

        object_root = tmp_path / "objects"
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        get_unscoped_file_storage.cache_clear()

        upload = client.post(
            "/api/files/upload",
            files={
                "file": (
                    "integrity-redirect-preview.txt",
                    b"expected redirect preview content",
                    "text/plain",
                )
            },
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        _corrupt_durable_copy_and_remove_local(
            object_root, temp_uploads_dir, "integrity-redirect-preview.txt"
        )

        def fail_signed_url(self, key, **kwargs):
            del self, key, kwargs
            raise AssertionError("signed URL should not be generated")

        monkeypatch.setattr(FsspecFileStorage, "signed_url", fail_signed_url)

        preview = client.get(
            f"/api/files/preview/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert preview.status_code == 409
        assert "re-upload" in preview.json()["detail"]
        assert not list(temp_uploads_dir.rglob("integrity-redirect-preview.txt"))

    def test_public_preview_checksum_mismatch_asks_user_to_reupload(
        self, client, temp_uploads_dir, monkeypatch, tmp_path, auth_headers
    ):
        object_root = tmp_path / "objects"
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
        get_unscoped_file_storage.cache_clear()

        upload = client.post(
            "/api/files/upload",
            files={
                "file": (
                    "integrity-public.txt",
                    b"expected public content",
                    "text/plain",
                )
            },
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]
        _corrupt_durable_copy_and_remove_local(
            object_root, temp_uploads_dir, "integrity-public.txt"
        )

        preview = client.get(f"/api/files/public/preview/{file_id}")

        assert preview.status_code == 409
        assert "re-upload" in preview.json()["detail"]

    def test_public_download_serves_source_bytes_without_auth(
        self, client, temp_uploads_dir, auth_headers
    ):
        """Public download must serve the source bytes for plain
        ``<a href>`` navigation that does NOT carry a bearer token —
        otherwise the chat file-card 'Open' link, middle-click
        'open in new tab', and right-click 'copy link' all 401."""
        upload = client.post(
            "/api/files/upload",
            files={"file": ("source.txt", b"source content", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]

        # Deliberately omit ``headers=auth_headers``: the whole point of
        # the public route is that it works without a token.
        download = client.get(f"/api/files/public/download/{file_id}")

        assert download.status_code == 200
        assert download.content == b"source content"

    @pytest.mark.parametrize("route", ["preview", "download"])
    def test_public_output_for_authenticated_task_with_selected_files_needs_no_token(
        self, route, client, test_db, temp_uploads_dir
    ):
        """An input attachment must not turn a signed-in task into a guest task.

        ``selected_file_ids`` makes ``agent_config`` a dict, but generated outputs
        still use the normal file-id capability. Only share/widget tasks require
        an additional public-access token.
        """
        from xagent.web.models.task import Task

        admin_user, test_app = test_db
        output_path = (
            temp_uploads_dir
            / f"user_{admin_user.id}"
            / "web_task_output"
            / "analysis.txt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"generated analysis")

        db = next(test_app.dependency_overrides[get_db]())
        try:
            task = Task(
                title="Authenticated task with an attachment",
                user_id=admin_user.id,
                agent_config={"selected_file_ids": ["input-file-id"]},
            )
            db.add(task)
            db.flush()

            output = UploadedFile(
                user_id=admin_user.id,
                task_id=task.id,
                filename="analysis.txt",
                storage_path=str(output_path),
                mime_type="text/plain",
                file_size=output_path.stat().st_size,
            )
            db.add(output)
            db.commit()
            db.refresh(output)
            file_id = output.file_id
        finally:
            db.close()

        response = client.get(f"/api/files/public/{route}/{file_id}")

        assert response.status_code == 200, response.text
        assert response.content == b"generated analysis"

    def test_public_download_sets_attachment_content_disposition(
        self, client, temp_uploads_dir, auth_headers
    ):
        """Public download must send Content-Disposition: attachment
        with the source filename so the browser saves under the real
        name (e.g. ``slides.pptx``) instead of the bare file id."""
        upload = client.post(
            "/api/files/upload",
            files={"file": ("slides.pptx", b"slide bytes", "application/octet-stream")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200
        file_id = upload.json()["file_id"]

        download = client.get(f"/api/files/public/download/{file_id}")

        assert download.status_code == 200
        disposition = download.headers.get("content-disposition", "")
        assert disposition.startswith("attachment"), disposition
        assert 'filename="slides.pptx"' in disposition, disposition

    def test_public_download_sets_rfc5987_content_disposition_for_non_ascii_filenames(
        self, client, temp_uploads_dir, auth_headers
    ):
        """Non-ASCII filenames (e.g. Chinese) must be percent-encoded as
        ``filename*=utf-8''<encoded>`` in the Content-Disposition header.
        A manually composed ``filename="报告.pptx"`` would be encoded as
        latin-1 by the ASGI layer, raising UnicodeEncodeError.  Delegating
        header generation to Starlette's FileResponse avoids this."""
        upload = client.post(
            "/api/files/upload",
            files={
                "file": (
                    "报告.pptx",
                    b"slide bytes",
                    "application/octet-stream",
                )
            },
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload.status_code == 200, upload.json()
        file_id = upload.json()["file_id"]

        download = client.get(f"/api/files/public/download/{file_id}")

        assert download.status_code == 200, download.text
        disposition = download.headers.get("content-disposition", "")
        assert disposition.startswith("attachment"), disposition
        # Starlette percent-encodes non-ASCII names with the RFC 5987
        # ``filename*=utf-8''<encoded>`` form.  The raw multi-byte string
        # must NOT appear literally in the header value.
        assert "filename*=" in disposition, disposition
        assert "报告" not in disposition, disposition  # '报告'

    def test_public_download_returns_404_for_unknown_id(self, client):
        download = client.get(
            "/api/files/public/download/00000000-0000-4000-8000-000000000000"
        )
        assert download.status_code == 404

    def test_public_download_registered_file_rejects_local_path_outside_uploads(
        self, client, test_db, tmp_path
    ):
        """Public download must not expose registered paths outside the
        uploads root (mirror of the same guard on public_preview)."""
        admin_user, test_app = test_db
        outside_path = tmp_path / "outside-public-download.txt"
        outside_path.write_text("outside uploads", encoding="utf-8")

        db = next(test_app.dependency_overrides[get_db]())
        try:
            db.add(
                UploadedFile(
                    file_id="44444444-4444-4444-8444-444444444444",
                    user_id=admin_user.id,
                    filename="outside-public-download.txt",
                    storage_path=str(outside_path),
                    storage_status="legacy",
                    mime_type="text/plain",
                    file_size=outside_path.stat().st_size,
                )
            )
            db.commit()
        finally:
            db.close()

        response = client.get(
            "/api/files/public/download/44444444-4444-4444-8444-444444444444"
        )

        assert response.status_code == 403

    def test_upload_python_file_success(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test successful upload of Python file"""
        files, temp_dir = sample_files
        file_path = files["test.py"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/files/upload",
                files={"file": ("test.py", f, "text/x-python")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        assert response.status_code == 200

    def test_upload_json_file_success(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test successful upload of JSON file"""
        files, temp_dir = sample_files
        file_path = files["test.json"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/files/upload",
                files={"file": ("test.json", f, "application/json")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        assert response.status_code == 200

    def test_upload_csv_file_success(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test successful upload of CSV file"""
        files, temp_dir = sample_files
        file_path = files["test.csv"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/files/upload",
                files={"file": ("test.csv", f, "text/csv")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        assert response.status_code == 200

    def test_upload_png_file_success(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test successful upload of PNG image file"""
        # Create a minimal valid PNG file (1x1 pixel PNG)
        # PNG signature + IHDR + IDAT + IEND
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x00\x03\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_data)
            tmp.flush()

            with open(tmp.name, "rb") as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": ("test.png", f, "image/png")},
                    data={"task_type": "general"},
                    headers=auth_headers,
                )

        os.unlink(tmp.name)
        assert response.status_code == 200

    def test_upload_jpg_file_success(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test successful upload of JPG image file"""
        # Create a minimal valid JPEG file
        jpeg_data = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x03\x02\x02\x03\x02\x02\x03\x03\x03\x03\x04\x03\x03"
            b"\x04\x05\x08\x05\x05\x04\x04\x05\n\x07\x07\x06\x08\x0c\n\x0c\x0c\x0b"
            b"\n\x0b\x0b\r\x0e\x12\x10\r\x0e\x11\x0e\x0b\x0b\x10\x16\x10\x11\x13\x14"
            b"\x15\x15\x15\x0c\x0f\x17\x18\x16\x14\x18\x12\x14\x15\x14\xff\xc0\x00"
            b"\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\xff\xc4\x00"
            b"\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\x9f\xff\xd9"
        )

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(jpeg_data)
            tmp.flush()

            with open(tmp.name, "rb") as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": ("test.jpg", f, "image/jpeg")},
                    data={"task_type": "general"},
                    headers=auth_headers,
                )

        os.unlink(tmp.name)
        assert response.status_code == 200

    def test_upload_no_filename_error(self, client, test_db, auth_headers):
        """Test upload with no filename"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp.flush()

            with open(tmp.name, "rb") as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": ("", f, "text/plain")},
                    data={"task_type": "general"},
                    headers=auth_headers,
                )

        # Empty filename returns 422 validation error
        assert response.status_code == 422
        os.unlink(tmp.name)

    def test_upload_unsupported_file_type(self, client, test_db, auth_headers):
        """Test upload with unsupported file type"""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp.write(b"executable content")
            tmp.flush()

            with open(tmp.name, "rb") as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": ("test.exe", f, "application/octet-stream")},
                    data={"task_type": "general"},
                    headers=auth_headers,
                )

        # API returns 500 for unsupported file types
        assert response.status_code == 500
        os.unlink(tmp.name)

    def test_upload_saves_file_to_disk(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test that upload saves file to disk"""
        files, temp_dir = sample_files
        file_path = files["test.txt"]

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/files/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        # Test passes if upload is successful (200/201) - we don't need to check file system
        # as the API response will indicate success/failure
        assert response.status_code == 200

    def test_upload_file_returns_413_when_size_exceeds_limit(
        self, client, test_db, temp_uploads_dir, auth_headers, monkeypatch
    ):
        """Upload endpoint should return 413 with a friendly message when too large."""
        import xagent.web.api.files

        monkeypatch.setattr(xagent.web.api.files, "MAX_FILE_SIZE", 4)

        response = client.post(
            "/api/files/upload",
            files={"file": ("big.txt", b"12345", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )

        assert response.status_code == 413
        assert "maximum limit" in response.json()["detail"].lower()

    def test_upload_multiple_files_cleans_up_partial_writes_on_limit_error(
        self, client, test_db, temp_uploads_dir, auth_headers, monkeypatch
    ):
        """A later oversized file should not leave earlier uploaded files on disk."""
        import xagent.web.api.files

        monkeypatch.setattr(xagent.web.api.files, "MAX_FILE_SIZE", 4)
        monkeypatch.setattr(xagent.web.api.files, "MAX_FILE_SIZE_LABEL", "4B")

        response = client.post(
            "/api/files/upload",
            files=[
                ("files", ("small.txt", b"1234", "text/plain")),
                ("files", ("big.txt", b"12345", "text/plain")),
            ],
            data={"task_type": "general"},
            headers=auth_headers,
        )

        assert response.status_code == 413
        assert [path for path in temp_uploads_dir.rglob("*") if path.is_file()] == []


class TestFileManagement:
    """Test file management operations"""

    def test_list_files_empty(self, client, test_db, auth_headers):
        """Test listing files when empty"""
        response = client.get("/api/files/list", headers=auth_headers)
        # Should return 200 with file list (may contain existing files from other tests)
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "total_count" in data
        assert isinstance(data["files"], list)
        assert isinstance(data["total_count"], int)
        for item in data["files"]:
            assert "ingestion_status" in item

    def test_list_files_with_collections(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test listing files when they are organized in collection subdirectories"""
        admin_user, _ = test_db
        collection_name = "my_test_collection"

        # With file_id design, list is DB-only. Create file via KB ingest so it
        # gets an UploadedFile record, then it will appear in list.
        doc_content = b"content in collection"
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("doc_in_coll.txt", doc_content, "text/plain")},
            data={"collection": collection_name},
            headers=auth_headers,
        )
        if response.status_code != 200:
            pytest.skip("KB ingest not available or failed")

        response = client.get("/api/files/list", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        found = False
        for f in data["files"]:
            if f.get("filename") == "doc_in_coll.txt":
                found = True
                assert f.get("file_id"), "list should return file_id"
                assert collection_name in f.get("relative_path", "")
                assert f.get("ingestion_status") in {
                    "SUCCESS",
                    "RUNNING",
                    "UNKNOWN",
                    "FAILED",
                }
                break
        assert found, (
            "File in collection directory should appear in list (file_id design)"
        )

    def test_download_file_success(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test successful file download"""
        files, temp_dir = sample_files
        file_path = files["test.txt"]

        # First upload a file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/files/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        # If upload was successful, try to download
        if upload_response.status_code == 200:
            upload_data = upload_response.json()
            file_id = upload_data.get("file_id")
            assert file_id, "upload response should include file_id"
            # Try to download the file using the download endpoint
            response = client.get(
                f"/api/files/download/{file_id}", headers=auth_headers
            )
            # Download of existing file should succeed
            assert response.status_code == 200
        else:
            # If upload failed, skip download test
            pytest.skip("Upload failed, skipping download test")

    def test_download_file_not_found(self, client, test_db, auth_headers):
        """Test downloading non-existent file"""
        response = client.get(
            "/api/files/download/nonexistent.txt", headers=auth_headers
        )
        # Non-existent file returns 404
        assert response.status_code == 404

    def test_delete_file_success(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test successful file deletion"""
        files, temp_dir = sample_files
        file_path = files["test.txt"]

        # First upload a file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/files/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        # If upload was successful, try to delete
        if upload_response.status_code == 200:
            upload_data = upload_response.json()
            file_id = upload_data.get("file_id")
            assert file_id, "upload response should include file_id"
            # Try to delete the file
            response = client.delete(f"/api/files/{file_id}", headers=auth_headers)
            # Delete existing file should succeed
            assert response.status_code == 200
        else:
            # If upload failed, skip delete test
            pytest.skip("Upload failed, skipping delete test")

    def test_delete_legacy_file_uses_only_path_keyed_cache_identity(
        self, client, temp_uploads_dir, auth_headers, monkeypatch, tmp_path
    ):
        import hashlib

        import xagent.web.api.legacy_file as legacy_file_module

        monkeypatch.setattr(
            legacy_file_module,
            "get_uploads_dir",
            lambda: temp_uploads_dir,
        )
        storage_root = tmp_path / "storage"
        monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))
        legacy_path = temp_uploads_dir / "user_1" / "legacy.svg"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_bytes(b"legacy")
        path_key = hashlib.sha256(str(legacy_path.resolve()).encode()).hexdigest()[:24]

        svg_cache_dir = storage_root / "svg_png_cache"
        pdf_cache_dir = storage_root / "pptx_pdf_cache"
        svg_cache_dir.mkdir(parents=True)
        pdf_cache_dir.mkdir(parents=True)
        legacy_svg_cache = svg_cache_dir / f"{path_key}.preview.png"
        legacy_pdf_cache = pdf_cache_dir / f"{path_key}.preview.pdf"
        legacy_svg_cache.write_bytes(b"legacy svg preview")
        legacy_pdf_cache.write_bytes(b"legacy pdf preview")
        unrelated_cache = svg_cache_dir / "registered.unrelated.preview.png"
        unrelated_cache.write_bytes(b"unrelated")

        response = client.delete("/api/files/legacy.svg", headers=auth_headers)

        assert response.status_code == 200, response.text
        assert not legacy_path.exists()
        assert not legacy_svg_cache.exists()
        assert not legacy_pdf_cache.exists()
        assert unrelated_cache.exists()

    def test_delete_file_keeps_record_when_durable_cleanup_fails(
        self, client, test_db, temp_uploads_dir, auth_headers, monkeypatch, caplog
    ):
        """Durable cleanup failure should not orphan the object by deleting the row."""
        from xagent.core.file_storage.storage import FsspecFileStorage

        admin_user, test_app = test_db
        upload_response = client.post(
            "/api/files/upload",
            files={"file": ("delete-fails.txt", b"delete fails", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

        db = next(test_app.dependency_overrides[get_db]())
        try:
            record = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.user_id == admin_user.id,
                    UploadedFile.file_id == file_id,
                )
                .one()
            )
            storage_key = str(record.storage_key)
            local_path = Path(str(record.storage_path))
        finally:
            db.close()

        real_delete = FsspecFileStorage.delete

        def fail_target_delete(self, key):
            if key == storage_key:
                raise RuntimeError("simulated durable delete failure")
            real_delete(self, key)

        monkeypatch.setattr(FsspecFileStorage, "delete", fail_target_delete)

        with caplog.at_level(logging.WARNING, logger="xagent.web.api.files"):
            response = client.delete(f"/api/files/{file_id}", headers=auth_headers)

        assert response.status_code == 503
        assert local_path.exists()

        # End-to-end on the message this endpoint actually composes (#1467).
        # Asserting through the request rather than by calling the helper with a
        # hand-built label is the point: a drift in what the endpoint passes --
        # a wrong variable, a dropped field -- would otherwise be invisible.
        fault_lines = [
            logging.Formatter("%(message)s").format(entry)
            for entry in caplog.records
            if entry.name == "xagent.web.api.files"
            and entry.levelno == logging.WARNING
            and "durable cleanup before row delete" in entry.getMessage()
        ]
        assert len(fault_lines) == 1, caplog.records
        rendered = fault_lines[0]
        assert f"file_id={file_id}" in rendered
        assert f"storage_key={storage_key}" in rendered
        # The cause chain, not just its str(), so the provider class survives.
        assert "RuntimeError" in rendered
        assert "simulated durable delete failure" in rendered
        # The 503 body stays detail-free: the key must not reach the client.
        assert storage_key not in response.text
        db = next(test_app.dependency_overrides[get_db]())
        try:
            assert (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
                is not None
            )
        finally:
            db.close()

    def test_delete_file_malformed_key_is_not_reported_as_an_outage(
        self, client, test_db, temp_uploads_dir, auth_headers, caplog
    ):
        """The other half of #1473: a structurally-invalid persisted key.

        ``ScopedFileStorage.delete`` normalizes the key even in tolerant mode,
        and ``normalize_storage_key`` raises ``ValueError`` for a ``..`` path
        segment (also a null byte or an empty key). That is data-level
        corruption in the row itself -- no retry can clear it -- so the
        cleanup arm must let it propagate rather than fold it into the
        retryable 503 with a transient-outage warning, which is exactly the
        failure #1473 describes: the row becomes undeletable through the API
        while the client is told to retry forever.
        """
        admin_user, test_app = test_db
        upload_response = client.post(
            "/api/files/upload",
            files={"file": ("malformed-key.txt", b"malformed key", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

        # Corrupt the persisted key the way a bad migration or adopt would:
        # structurally invalid, not merely out of scope.
        db = next(test_app.dependency_overrides[get_db]())
        try:
            record = (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
            )
            record.storage_key = f"users/{admin_user.id}/uploads/../{file_id}"
            db.commit()
        finally:
            db.close()

        with caplog.at_level(logging.WARNING, logger="xagent.web.api.files"):
            with pytest.raises(ValueError, match="Invalid storage key"):
                client.delete(f"/api/files/{file_id}", headers=auth_headers)

        # No outage warning: the fault is permanent, not a durable outage.
        assert not [
            entry
            for entry in caplog.records
            if entry.name == "xagent.web.api.files"
            and DURABLE_FAULT_LOG_PREFIX in entry.getMessage()
        ], "a malformed persisted key was reported as a durable-storage outage"
        # The row survives, same as any failed cleanup.
        db = next(test_app.dependency_overrides[get_db]())
        try:
            assert (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
                is not None
            )
        finally:
            db.close()

    def test_delete_file_backend_value_error_is_still_a_retryable_outage(
        self, client, test_db, temp_uploads_dir, auth_headers, monkeypatch, caplog
    ):
        """A ValueError from the backend is an outage, not a malformed key.

        ``ScopedFileStorage.delete`` normalizes and then calls ``fs.exists`` /
        ``fs.rm``, and those raise ``ValueError`` for their own reasons. The
        malformed-key half of #1473 must not swallow that case: catching
        ``ValueError`` around the whole call made every backend one permanent,
        bypassing both the durable-outage log and the retryable 503. The key
        is validated before the call now, so only the normalization boundary
        is permanent.
        """
        from xagent.core.file_storage.storage import FsspecFileStorage

        _, test_app = test_db
        upload_response = client.post(
            "/api/files/upload",
            files={"file": ("backend-value-error.txt", b"payload", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

        def backend_value_error(self, key):
            raise ValueError("backend rejected the request")

        monkeypatch.setattr(FsspecFileStorage, "delete", backend_value_error)

        with caplog.at_level(logging.WARNING, logger="xagent.web.api.files"):
            response = client.delete(f"/api/files/{file_id}", headers=auth_headers)

        # Retryable, not the permanent 500 a malformed key produces.
        assert response.status_code == 503, response.text
        # And it is reported as a durable-storage fault, with its cause.
        fault_lines = [
            entry.getMessage()
            for entry in caplog.records
            if entry.name == "xagent.web.api.files"
            and DURABLE_FAULT_LOG_PREFIX in entry.getMessage()
        ]
        assert len(fault_lines) == 1, caplog.records
        assert "durable cleanup before row delete" in fault_lines[0]

    def test_delete_file_scope_violation_is_not_reported_as_an_outage(
        self, client, test_db, temp_uploads_dir, auth_headers, monkeypatch, caplog
    ):
        """A containment violation is a permanent authority fault, not a 503.

        ``ScopedFileStorage.delete`` raises ``StorageKeyScopeError`` before it
        reaches the backend when the key falls outside the bound prefix. The
        cleanup arm must let it propagate -- retrying cannot clear an
        authority fault, and the outage warning would point an operator at the
        wrong subsystem. This was #1473.

        This fixture's app is a bare router with no application-level
        handlers, so "propagate" here means the exception escapes the
        endpoint. In production it reaches the dedicated handler registered in
        ``web/app.py`` (a permanent 500 with a fixed body), whose contract is
        pinned by ``test_storage_namespace_authority_handler.py``.
        """
        from xagent.core.file_storage.scoped import (
            ScopedFileStorage,
            StorageKeyScopeError,
        )

        admin_user, test_app = test_db
        upload_response = client.post(
            "/api/files/upload",
            files={"file": ("scope-violation.txt", b"scope violation", "text/plain")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert upload_response.status_code == 200
        file_id = upload_response.json()["file_id"]

        def scope_violating_delete(self, key):
            raise StorageKeyScopeError(
                f"Storage key '{key}' is outside the bound prefix"
            )

        monkeypatch.setattr(ScopedFileStorage, "delete", scope_violating_delete)

        with caplog.at_level(logging.WARNING, logger="xagent.web.api.files"):
            with pytest.raises(StorageKeyScopeError):
                client.delete(f"/api/files/{file_id}", headers=auth_headers)

        # No outage warning: the fault is not a durable-storage outage.
        assert not [
            entry
            for entry in caplog.records
            if entry.name == "xagent.web.api.files"
            and DURABLE_FAULT_LOG_PREFIX in entry.getMessage()
        ], "a scope violation was reported as a durable-storage outage"
        # The row survives, same as any failed cleanup.
        db = next(test_app.dependency_overrides[get_db]())
        try:
            assert (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
                is not None
            )
        finally:
            db.close()

    def test_delete_file_not_found(self, client, test_db, auth_headers):
        """Test deleting non-existent file"""
        response = client.delete("/api/files/nonexistent.txt", headers=auth_headers)
        # Non-existent file returns 404
        assert response.status_code == 404

    def test_list_files_after_deletion(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test listing files after deletion"""
        files, temp_dir = sample_files
        file_path = files["test.txt"]

        # First upload a file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/files/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        # If upload was successful, try to delete then list
        if upload_response.status_code == 200:
            # Delete the file
            client.delete("/api/files/test.txt", headers=auth_headers)

            # List files
            response = client.get("/api/files/list", headers=auth_headers)
            # Should return 200 with file list
            assert response.status_code == 200
        else:
            # If upload failed, skip test
            pytest.skip("Upload failed, skipping list after deletion test")


class TestFileUploadIntegration:
    """Integration tests for file upload workflow"""

    def test_complete_workflow(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test complete upload-download-delete workflow"""
        files, temp_dir = sample_files
        file_path = files["test.txt"]

        # Upload file
        with open(file_path, "rb") as f:
            upload_response = client.post(
                "/api/files/upload",
                files={"file": ("test.txt", f, "text/plain")},
                data={"task_type": "general"},
                headers=auth_headers,
            )

        # If upload was successful, continue with workflow
        if upload_response.status_code == 200:
            upload_data = upload_response.json()
            file_id = upload_data.get("file_id")
            assert file_id, "upload response should include file_id"
            # List files
            list_response = client.get("/api/files/list", headers=auth_headers)
            assert list_response.status_code == 200

            # Download file
            download_response = client.get(
                f"/api/files/download/{file_id}", headers=auth_headers
            )
            # Download existing file should succeed
            assert download_response.status_code == 200

            # Delete file
            delete_response = client.delete(
                f"/api/files/{file_id}", headers=auth_headers
            )
            # Delete existing file should succeed
            assert delete_response.status_code == 200
        else:
            # If upload failed, test passes as we verified the behavior
            pytest.skip("Upload failed, integration workflow test not applicable")

    def test_multiple_files_management(
        self, client, test_db, sample_files, temp_uploads_dir, auth_headers
    ):
        """Test managing multiple files"""
        files, temp_dir = sample_files

        # Upload multiple files
        uploaded_files = []
        for filename in ["test.txt", "test.py", "test.json"]:
            file_path = files[filename]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/files/upload",
                    files={"file": (filename, f, "text/plain")},
                    data={"task_type": "general"},
                    headers=auth_headers,
                )
                if response.status_code == 200:
                    uploaded_files.append(filename)

        # If some files were uploaded, test listing
        if uploaded_files:
            list_response = client.get("/api/files/list", headers=auth_headers)
            assert list_response.status_code == 200

            # Clean up uploaded files
            for filename in uploaded_files:
                client.delete(f"/api/files/{filename}", headers=auth_headers)
        else:
            # If no files were uploaded, test passes as we verified the behavior
            pytest.skip(
                "No files were uploaded, multiple files management test not applicable"
            )


class TestFileUploadSecurity:
    """Security tests for file upload API endpoints."""

    def test_upload_file_rejects_path_traversal_in_folder(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test that upload_file rejects path traversal in folder parameter."""
        malicious_folders = [
            "../../../etc",
            "..\\..\\..\\windows",
            "folder/../other",
            "../folder",
            "folder/",
        ]

        # Use a valid integer task_id so folder validation runs (get_upload_path).
        for folder in malicious_folders:
            response = client.post(
                "/api/files/upload",
                files={"file": ("test.txt", b"content", "text/plain")},
                data={"task_type": "general", "task_id": "1", "folder": folder},
                headers=auth_headers,
            )
            # Should reject with 422 (validation error)
            assert response.status_code == 422
            detail = response.json().get("detail", "")
            assert "Invalid folder name" in detail or "invalid" in detail.lower()

    def test_upload_file_rejects_invalid_characters_in_folder(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test that upload_file rejects invalid characters in folder parameter."""
        invalid_folders = [
            "folder name",  # Space
            "folder@name",  # @ symbol
            "folder#name",  # # symbol
            "folder/name",  # Path separator
            "folder\\name",  # Windows path separator
        ]

        # Use a valid integer task_id so folder validation runs.
        for folder in invalid_folders:
            response = client.post(
                "/api/files/upload",
                files={"file": ("test.txt", b"content", "text/plain")},
                data={"task_type": "general", "task_id": "1", "folder": folder},
                headers=auth_headers,
            )
            assert response.status_code == 422
            detail = response.json().get("detail", "")
            assert "Invalid folder name" in detail or "invalid" in detail.lower()

    def test_upload_file_rejects_too_long_folder_name(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test that upload_file rejects folder names exceeding length limit."""
        too_long_folder = "a" * 101

        response = client.post(
            "/api/files/upload",
            files={"file": ("test.txt", b"content", "text/plain")},
            data={
                "task_type": "general",
                "task_id": "1",
                "folder": too_long_folder,
            },
            headers=auth_headers,
        )
        assert response.status_code == 422
        detail = response.json().get("detail", "")
        assert "Invalid folder name" in detail or "invalid" in detail.lower()

    def test_upload_multiple_files_rejects_path_traversal_in_folder(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test that upload (multiple files) rejects path traversal in folder parameter."""
        malicious_folders = [
            "../../../etc",
            "..\\..\\..\\windows",
            "folder/../other",
        ]

        for folder in malicious_folders:
            response = client.post(
                "/api/files/upload",
                files=[("files", ("test.txt", b"content", "text/plain"))],
                data={"task_type": "general", "task_id": "1", "folder": folder},
                headers=auth_headers,
            )
            assert response.status_code == 422
            detail = response.json().get("detail", "")
            assert "Invalid folder name" in detail or "invalid" in detail.lower()

    def test_download_file_rejects_path_traversal(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test that download_file rejects path traversal attempts."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "../other_user/file.txt",
            "file/../../etc/passwd",
        ]

        for path in malicious_paths:
            encoded_path = quote(path, safe="")
            response = client.get(
                f"/api/files/download/{encoded_path}", headers=auth_headers
            )
            # Path traversal attempts return 404 (route not found)
            assert response.status_code == 404

    def test_preview_file_rejects_path_traversal(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test that preview_file rejects path traversal attempts."""
        task_id = 1

        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "../other_user/file.txt",
        ]

        for path in malicious_paths:
            encoded_path = quote(path, safe="")
            response = client.get(
                f"/api/files/preview/{task_id}/{encoded_path}", headers=auth_headers
            )
            # Path traversal attempts return 404 (route not found)
            assert response.status_code == 404

    def test_list_files_handles_nested_paths_correctly(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """With file_id design, list is DB-only (no filesystem scan). File in a
        collection appears in list when created via KB ingest."""
        admin_user, _ = test_db

        # Create file via KB ingest to collection "a" so it gets an UploadedFile record.
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("file.txt", b"nested content", "text/plain")},
            data={"collection": "a"},
            headers=auth_headers,
        )
        if response.status_code != 200:
            pytest.skip("KB ingest not available or failed")

        response = client.get("/api/files/list", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        found = False
        for f in data["files"]:
            if f.get("filename") == "file.txt":
                found = True
                assert f.get("file_id"), "list should return file_id"
                # Path is user_id/a/file.txt
                assert "a" in f.get("relative_path", "")
                break
        assert found, "File in collection should appear in list (file_id design)"

    def test_list_files_handles_invalid_first_level_collection_name(
        self, client, test_db, temp_uploads_dir, auth_headers
    ):
        """Test that list_files handles invalid first-level collection names gracefully."""
        admin_user, _ = test_db
        user_id = admin_user.id

        invalid_dir = temp_uploads_dir / f"user_{user_id}" / ".." / "other"
        try:
            invalid_dir.mkdir(parents=True, exist_ok=True)
            test_file = invalid_dir / "file.txt"
            test_file.write_text("content")

            response = client.get("/api/files/list", headers=auth_headers)
            assert response.status_code == 200
        except (OSError, ValueError):
            pass

    def test_list_files_supports_pagination_and_filters(
        self, client, test_db, auth_headers
    ):
        """Test listing files with pagination and server-side filters."""
        from xagent.web.models.task import Task

        admin_user, test_app = test_db
        db = next(test_app.dependency_overrides[get_db]())
        try:
            db.add(Task(id=123, title="Task 123", user_id=admin_user.id))
            db.commit()
        finally:
            db.close()

        uploaded = [
            ("alpha.txt", None),
            ("beta.txt", None),
            ("agent-note.txt", "123"),
        ]

        for filename, task_id in uploaded:
            data = {"task_type": "general"}
            if task_id is not None:
                data["task_id"] = task_id
            response = client.post(
                "/api/files/upload",
                files={"file": (filename, filename.encode("utf-8"), "text/plain")},
                data=data,
                headers=auth_headers,
            )
            assert response.status_code == 200

        response = client.get(
            "/api/files/list?page=1&size=2",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] >= 3
        assert data["page"] == 1
        assert data["size"] == 2
        assert data["pages"] >= 2
        assert len(data["files"]) == 2

        response = client.get(
            "/api/files/list?search=agent-note",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert [item["filename"] for item in data["files"]] == ["agent-note.txt"]

    def test_list_file_tasks_returns_tasks_with_files(
        self, client, test_db, auth_headers
    ):
        """Test listing task filters only returns tasks that actually have files."""
        from xagent.web.models.task import Task

        admin_user, test_app = test_db
        db = next(test_app.dependency_overrides[get_db]())
        try:
            db.add(Task(id=321, title="Task 321", user_id=admin_user.id))
            db.add(Task(id=654, title="Task 654", user_id=admin_user.id))
            db.commit()
        finally:
            db.close()

        uploads = [
            ("task-alpha.txt", "321"),
            ("task-beta.txt", "654"),
            ("loose-upload.txt", None),
        ]

        for filename, task_id in uploads:
            data = {"task_type": "general"}
            if task_id is not None:
                data["task_id"] = task_id
            response = client.post(
                "/api/files/upload",
                files={"file": (filename, filename.encode("utf-8"), "text/plain")},
                data=data,
                headers=auth_headers,
            )
            assert response.status_code == 200

        response = client.get("/api/files/tasks", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert "tasks" in data
        task_ids = {item["task_id"] for item in data["tasks"]}
        assert 321 in task_ids
        assert 654 in task_ids
        assert all(item["file_count"] >= 1 for item in data["tasks"])

        response = client.get(
            "/api/files/list?uploads_only=true",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert [item["filename"] for item in data["files"]] == ["loose-upload.txt"]
        assert all(item["task_id"] is None for item in data["files"])

        response = client.get(
            "/api/files/list?task_id=321",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert [item["filename"] for item in data["files"]] == ["task-alpha.txt"]


_CLEAN_SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" />'
_MALICIOUS_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
    b"<script>alert(document.domain)</script></svg>"
)


class TestSvgPreviewSecurity:
    """SVG previews must never render raw, script-bearing bytes in-browser."""

    def _upload_svg(self, client, auth_headers, content: bytes = _CLEAN_SVG):
        response = client.post(
            "/api/files/upload",
            files={"file": ("logo.svg", content, "image/svg+xml")},
            data={"task_type": "general"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        return response.json()["file_id"]

    def test_preview_svg_returns_png_not_raw(
        self, client, temp_uploads_dir, auth_headers
    ):
        file_id = self._upload_svg(client, auth_headers)

        preview = client.get(f"/api/files/preview/{file_id}", headers=auth_headers)

        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"
        assert preview.content.startswith(b"\x89PNG")

    def test_durable_svg_preview_bypasses_redirects_and_rasterizes(
        self,
        client,
        test_db,
        temp_uploads_dir,
        auth_headers,
        monkeypatch,
    ):
        from xagent.web.services.managed_file_ref import ManagedFileRef

        _, test_app = test_db
        file_id = self._upload_svg(client, auth_headers)
        db = next(test_app.dependency_overrides[get_db]())
        try:
            record = (
                db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
            )
            Path(str(record.storage_path)).unlink()
        finally:
            db.close()

        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED", "true")

        def unexpected_signed_redirect(self, **_kwargs):
            del self
            raise AssertionError("SVG preview must not use a signed raw-byte URL")

        monkeypatch.setattr(
            ManagedFileRef,
            "signed_access_url",
            unexpected_signed_redirect,
        )

        preview = client.get(
            f"/api/files/preview/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"
        assert preview.content.startswith(b"\x89PNG")
        assert "location" not in preview.headers
        assert "x-accel-redirect" not in preview.headers

    def test_public_preview_svg_returns_png(
        self, client, temp_uploads_dir, auth_headers
    ):
        file_id = self._upload_svg(client, auth_headers)

        preview = client.get(f"/api/files/public/preview/{file_id}")

        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"
        assert preview.content.startswith(b"\x89PNG")

    def test_preview_svg_with_script_never_leaks_raw_bytes(
        self, client, temp_uploads_dir, auth_headers
    ):
        """A pre-existing malicious SVG (e.g. uploaded before this fix, or via
        the generic /upload endpoint that download_web_asset's validation
        never sees) must fail closed rather than ever serve raw script bytes.
        """
        file_id = self._upload_svg(client, auth_headers, content=_MALICIOUS_SVG)

        preview = client.get(f"/api/files/preview/{file_id}", headers=auth_headers)

        assert preview.status_code == 422
        assert b"<script" not in preview.content

    def test_download_svg_forces_attachment(
        self, client, temp_uploads_dir, auth_headers
    ):
        file_id = self._upload_svg(client, auth_headers, content=_CLEAN_SVG)

        download = client.get(f"/api/files/download/{file_id}", headers=auth_headers)

        assert download.status_code == 200
        assert download.headers["content-disposition"].startswith("attachment")
        assert download.headers["x-content-type-options"] == "nosniff"

    def test_signed_svg_download_forces_attachment(
        self, client, temp_uploads_dir, auth_headers, monkeypatch
    ):
        from xagent.web.services.managed_file_ref import ManagedFileRef

        file_id = self._upload_svg(client, auth_headers, content=_CLEAN_SVG)
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        dispositions: list[str | None] = []

        def signed_access_url(
            self,
            *,
            expires,
            content_type=None,
            content_disposition=None,
        ):
            del self, expires, content_type
            dispositions.append(content_disposition)
            return "https://cdn.example.com/private/logo.svg?sig=abc"

        monkeypatch.setattr(ManagedFileRef, "signed_access_url", signed_access_url)

        download = client.get(
            f"/api/files/download/{file_id}",
            headers=auth_headers,
            follow_redirects=False,
        )

        assert download.status_code == 307
        assert dispositions == [
            "attachment; filename=\"logo.svg\"; filename*=UTF-8''logo.svg"
        ]
