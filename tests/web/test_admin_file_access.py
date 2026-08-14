import os
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.web.api.files as files_module
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.api.auth import hash_password
from xagent.web.api.files import file_router
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import Base, get_db
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


@pytest.fixture(scope="function")
def test_db():
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(temp_db_fd)

    test_engine = create_engine(
        f"sqlite:///{temp_db_path}", connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    def override_get_db():
        db = None
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            if db is not None:
                db.close()

    test_app = FastAPI()
    test_app.include_router(file_router)
    test_app.dependency_overrides[get_db] = override_get_db

    Base.metadata.create_all(bind=test_engine)

    session = TestingSessionLocal()
    try:
        admin_user = User(
            username="admin", password_hash=hash_password("admin"), is_admin=True
        )
        regular_user = User(
            username="regular", password_hash=hash_password("regular"), is_admin=False
        )
        session.add(admin_user)
        session.add(regular_user)
        session.commit()
        session.refresh(admin_user)
        session.refresh(regular_user)
        yield admin_user, regular_user, test_app, session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()
        try:
            os.unlink(temp_db_path)
        except OSError:
            pass


@pytest.fixture(scope="function")
def temp_uploads_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        import xagent.web.api.files

        monkeypatch.setattr(xagent.web.api.files, "get_uploads_dir", lambda: temp_path)
        yield temp_path


def create_auth_headers(user):
    from datetime import datetime, timedelta, timezone

    import jwt

    payload = {
        "sub": user.username,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
        "user_id": user.id,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


def create_uploaded_file(
    session,
    uploads_dir: Path,
    user_id: int,
    task_id: int,
    filename: str,
    content: str,
) -> UploadedFile:
    user_dir = uploads_dir / f"user_{user_id}" / f"web_task_{task_id}" / "output"
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / filename
    file_path.write_text(content)

    uploaded_file = UploadedFile(
        user_id=user_id,
        task_id=task_id,
        filename=filename,
        storage_path=str(file_path),
        mime_type="text/html",
        file_size=len(content.encode("utf-8")),
    )
    session.add(uploaded_file)
    session.commit()
    session.refresh(uploaded_file)
    return uploaded_file


class TestAdminFileAccess:
    def test_admin_access_other_user_file(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=78,
            user_id=regular_user_id,
            title="Test Task",
            description="Test task for file access",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "report.html",
            "test content",
        )

        client = TestClient(test_app)
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            f"/api/files/download/{uploaded_file.file_id}", headers=admin_headers
        )

        assert response.status_code == 200
        assert response.content == b"test content"

    def test_regular_user_access_own_file(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=79,
            user_id=regular_user_id,
            title="Test Task",
            description="Test task for file access",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "my_report.html",
            "my content",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/download/{uploaded_file.file_id}", headers=user_headers
        )

        assert response.status_code == 200
        assert response.content == b"my content"

    def test_regular_user_access_other_user_file_denied(
        self, test_db, temp_uploads_dir
    ):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        another_user = User(
            username="another", password_hash=hash_password("another"), is_admin=False
        )
        session.add(another_user)
        session.commit()

        another_user_id = int(cast(Any, another_user.id))
        task = Task(
            id=80,
            user_id=another_user_id,
            title="Another User Task",
            description="Task belonging to another user",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            another_user_id,
            int(cast(Any, task.id)),
            "secret_report.html",
            "secret content",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/download/{uploaded_file.file_id}", headers=user_headers
        )

        assert response.status_code == 403

    def test_regular_user_cannot_get_signed_redirect_for_other_user_file(
        self, test_db, temp_uploads_dir, monkeypatch
    ):
        from xagent.web.services.managed_file_ref import ManagedFileRef

        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        another_user = User(
            username="signed-other",
            password_hash=hash_password("signed-other"),
            is_admin=False,
        )
        session.add(another_user)
        session.commit()

        another_user_id = int(cast(Any, another_user.id))
        task = Task(
            id=801,
            user_id=another_user_id,
            title="Signed redirect denied",
            description="other user's file",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            another_user_id,
            int(cast(Any, task.id)),
            "signed-secret.txt",
            "secret content",
        )
        uploaded_file.storage_status = "available"
        uploaded_file.storage_key = (
            f"users/{another_user_id}/uploads/{uploaded_file.file_id}/signed-secret.txt"
        )
        uploaded_file.storage_backend = "s3"
        session.commit()

        def fail_signed_access_url(self, **kwargs):
            del self, kwargs
            raise AssertionError("access check must happen before signing")

        monkeypatch.setattr(ManagedFileRef, "signed_access_url", fail_signed_access_url)

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)

        for path in [
            f"/api/files/download/{uploaded_file.file_id}",
            f"/api/files/preview/{uploaded_file.file_id}",
        ]:
            response = client.get(
                path,
                headers=user_headers,
                follow_redirects=False,
            )
            assert response.status_code == 403

    def test_missing_file_id_returns_404(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        del regular_user, session, temp_uploads_dir
        client = TestClient(test_app)
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            "/api/files/download/00000000-0000-0000-0000-000000000000",
            headers=admin_headers,
        )
        assert response.status_code == 404

    def test_delete_file_by_file_id(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=81,
            user_id=regular_user_id,
            title="Delete Task",
            description="delete test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "delete_me.txt",
            "to delete",
        )
        file_path = Path(str(uploaded_file.storage_path))
        assert file_path.exists()

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.delete(
            f"/api/files/{uploaded_file.file_id}", headers=user_headers
        )

        assert response.status_code == 200
        assert not file_path.exists()
        assert (
            session.query(UploadedFile)
            .filter(UploadedFile.file_id == uploaded_file.file_id)
            .first()
            is None
        )

    def test_public_preview_allows_relative_asset_in_same_directory(
        self, test_db, temp_uploads_dir
    ):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=82,
            user_id=regular_user_id,
            title="Preview Task",
            description="public preview test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "index.html",
            "<img src='assets/pic.txt'>",
        )

        asset_dir = Path(str(uploaded_file.storage_path)).parent / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        asset_file = asset_dir / "pic.txt"
        asset_file.write_text("asset content")

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/public/preview/{uploaded_file.file_id}",
            params={"relative_path": "assets/pic.txt"},
        )

        assert response.status_code == 200
        assert response.content == b"asset content"

    def test_public_preview_restores_durable_relative_asset(
        self, test_db, temp_uploads_dir, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
        get_unscoped_file_storage.cache_clear()
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=85,
            user_id=regular_user_id,
            title="Preview Task",
            description="public durable preview test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "index.html",
            "<script src='assets/app.js'></script>",
        )

        asset_path = Path(str(uploaded_file.storage_path)).parent / "assets" / "app.js"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text("console.log('asset');", encoding="utf-8")
        storage_key = (
            f"users/{regular_user_id}/tasks/{task.id}/outputs/asset-file/"
            "output/assets/app.js"
        )
        stored_object = get_unscoped_file_storage().put_file(
            asset_path,
            storage_key,
            "application/javascript",
        )
        asset_path.unlink()
        asset_record = UploadedFile(
            user_id=regular_user_id,
            task_id=int(cast(Any, task.id)),
            filename="app.js",
            storage_path=str(asset_path),
            storage_backend=stored_object.backend,
            storage_key=stored_object.key,
            storage_uri=stored_object.uri,
            checksum=stored_object.checksum,
            etag=stored_object.etag,
            storage_status="available",
            workspace_relative_path="output/assets/app.js",
            workspace_category="output",
            mime_type="application/javascript",
            file_size=len("console.log('asset');".encode("utf-8")),
        )
        session.add(asset_record)
        session.commit()

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/public/preview/{uploaded_file.file_id}",
            params={"relative_path": "assets/app.js"},
        )

        assert response.status_code == 200
        assert response.content == b"console.log('asset');"
        assert asset_path.exists()

    def test_public_preview_blocks_parent_traversal(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))

        task_a = Task(
            id=83,
            user_id=regular_user_id,
            title="Task A",
            description="base file",
        )
        task_b = Task(
            id=84,
            user_id=regular_user_id,
            title="Task B",
            description="secret file",
        )
        session.add(task_a)
        session.add(task_b)
        session.commit()

        base_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task_a.id)),
            "index.html",
            "base",
        )
        create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task_b.id)),
            "secret.txt",
            "top secret",
        )

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/public/preview/{base_file.file_id}",
            params={"relative_path": "../../web_task_84/output/secret.txt"},
        )

        assert response.status_code == 403

    def test_public_preview_sets_no_store_cache_control(
        self, test_db, temp_uploads_dir
    ):
        # The public route is loaded directly as <img>/<audio>/<video> src on
        # share surfaces, which fetches automatically on render. Without this
        # header the tokened URL and its bytes could land in browser disk
        # cache and outlive the guest session.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=85,
            user_id=regular_user_id,
            title="Cache header test",
            description="public preview cache header test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.txt",
            "media bytes",
        )

        client = TestClient(test_app)
        response = client.get(f"/api/files/public/preview/{uploaded_file.file_id}")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "private, no-store"

    def test_public_preview_sets_no_store_cache_control_for_rasterized_svg(
        self, test_db, temp_uploads_dir, monkeypatch, tmp_path
    ):
        # The SVG branch of _inline_preview_response returns a separate
        # FileResponse for the rasterized PNG -- the header must reach that
        # branch too, not just the generic pass-through one exercised above.
        # rasterize_svg_bytes is stubbed to avoid a hard dependency on the
        # native cairo library in this environment; the extra_headers
        # plumbing under test doesn't depend on the rasterized output.
        monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path))
        monkeypatch.setattr(
            files_module, "rasterize_svg_bytes", lambda svg_bytes: b"fake-png-bytes"
        )

        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=87,
            user_id=regular_user_id,
            title="SVG cache header test",
            description="public preview cache header test for svg",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "icon.svg",
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>",
        )

        client = TestClient(test_app)
        response = client.get(f"/api/files/public/preview/{uploaded_file.file_id}")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "private, no-store"

    def test_authenticated_preview_keeps_default_caching(
        self, test_db, temp_uploads_dir
    ):
        # The authenticated in-app preview route must not get the public
        # route's no-store override: chat images there rely on ordinary
        # browser caching across repeated renders of the same message.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=86,
            user_id=regular_user_id,
            title="Authenticated cache header test",
            description="authenticated preview cache header test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.txt",
            "media bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}", headers=user_headers
        )

        assert response.status_code == 200
        # Starlette's FileResponse sets no Cache-Control by default; assert
        # that exactly, not just "not the public value" -- a regression that
        # set some other Cache-Control here would slip past a weaker check.
        assert "cache-control" not in response.headers

    def test_stream_ticket_authorizes_preview_without_bearer_header(
        self, test_db, temp_uploads_dir
    ):
        # A media element (<video>/<audio>) cannot send an Authorization
        # header, so it loads the preview URL directly using a ticket minted
        # by a Bearer-authenticated request instead.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=90,
            user_id=regular_user_id,
            title="Stream ticket test",
            description="stream ticket test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}",
            headers=user_headers,
        )
        assert ticket_response.status_code == 200
        path = ticket_response.json()["path"]
        assert path.startswith(f"/api/files/preview/{uploaded_file.file_id}?ticket=")

        preview_response = client.get(path)

        assert preview_response.status_code == 200
        assert preview_response.content == b"video bytes"
        assert preview_response.headers["cache-control"] == "private, no-store"

    def test_stream_ticket_cannot_be_replayed_against_a_different_file(
        self, test_db, temp_uploads_dir
    ):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=91,
            user_id=regular_user_id,
            title="Stream ticket scope test",
            description="stream ticket scope test",
        )
        session.add(task)
        session.commit()

        clip = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )
        other_clip = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "other.mp4",
            "other video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{clip.file_id}", headers=user_headers
        )
        ticket = ticket_response.json()["path"].split("ticket=")[1]

        response = client.get(
            f"/api/files/preview/{other_clip.file_id}", params={"ticket": ticket}
        )

        assert response.status_code == 401

    def test_stream_ticket_cannot_bypass_file_ownership_check(
        self, test_db, temp_uploads_dir
    ):
        # Minting a ticket never checks ownership -- it defers to the same
        # _check_file_access the Bearer path enforces at redemption time. A
        # ticket minted for a file the caller doesn't own must still 403
        # rather than leak the file.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        another_user = User(
            username="another_ticket_user",
            password_hash=hash_password("another"),
            is_admin=False,
        )
        session.add(another_user)
        session.commit()

        another_user_id = int(cast(Any, another_user.id))
        task = Task(
            id=92,
            user_id=another_user_id,
            title="Owner-only file",
            description="owner-only file",
        )
        session.add(task)
        session.commit()

        owners_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            another_user_id,
            int(cast(Any, task.id)),
            "private.mp4",
            "private video bytes",
        )

        client = TestClient(test_app)
        regular_user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{owners_file.file_id}",
            headers=regular_user_headers,
        )
        assert ticket_response.status_code == 200
        ticket = ticket_response.json()["path"].split("ticket=")[1]

        response = client.get(
            f"/api/files/preview/{owners_file.file_id}", params={"ticket": ticket}
        )

        assert response.status_code == 403

    def test_stream_ticket_cannot_bypass_ownership_check_for_legacy_file_id(
        self, test_db, temp_uploads_dir, monkeypatch
    ):
        # test_stream_ticket_cannot_bypass_file_ownership_check above only
        # exercises the DB-record branch of preview_file's access check
        # (_check_file_access via file_record); this covers the other
        # branch -- a legacy file_id with no UploadedFile record at all,
        # where ownership is instead enforced by comparing owner_user_id
        # (derived from the filesystem path, via a Task-id lookup for a
        # cross-user request -- see infer_user_id_from_legacy_path) against
        # the requester.
        import xagent.web.api.legacy_file as legacy_file_module

        monkeypatch.setattr(
            legacy_file_module, "get_uploads_dir", lambda: temp_uploads_dir
        )

        admin_user, regular_user, test_app, session = test_db
        del admin_user
        owner_id = int(cast(Any, regular_user.id))
        other_user = User(
            username="legacy-ticket-other",
            password_hash=hash_password("legacy-ticket-other"),
            is_admin=False,
        )
        session.add(other_user)
        session.commit()

        task = Task(
            id=236,
            user_id=owner_id,
            title="Owner-only legacy task",
            description="owner-only legacy task",
        )
        session.add(task)
        session.commit()

        legacy_relative_path = "web_task_236/output/private.mp4"
        legacy_file_path = temp_uploads_dir / f"user_{owner_id}" / legacy_relative_path
        legacy_file_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file_path.write_bytes(b"private legacy video bytes")

        client = TestClient(test_app)
        other_user_headers = create_auth_headers(other_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{quote(legacy_relative_path, safe='')}",
            headers=other_user_headers,
        )
        assert ticket_response.status_code == 200
        ticket = ticket_response.json()["path"].split("ticket=")[1]

        response = client.get(
            f"/api/files/preview/{quote(legacy_relative_path, safe='')}",
            params={"ticket": ticket},
        )

        assert response.status_code == 403

    def test_stream_ticket_mint_and_redeem_for_slash_bearing_legacy_file_id(
        self, test_db, temp_uploads_dir, monkeypatch
    ):
        # A legacy file_id can be an arbitrary relative path containing
        # slashes, with no UploadedFile DB record at all -- confirm the
        # mint response's quote(..., safe='')-encoded path round-trips
        # correctly through {file_id:path} on redemption.
        import xagent.web.api.legacy_file as legacy_file_module

        # resolve_legacy_file_path resolves get_uploads_dir() through its
        # own module-local import binding, separate from the one
        # temp_uploads_dir already patches on xagent.web.api.files -- both
        # must point at the same temp directory for this filesystem-scan
        # path (unlike the DB-record path other tests exercise, which
        # stores an absolute path directly and never calls this).
        monkeypatch.setattr(
            legacy_file_module, "get_uploads_dir", lambda: temp_uploads_dir
        )

        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))

        legacy_relative_path = "web_task_235/output/clip.mp4"
        legacy_file_path = (
            temp_uploads_dir / f"user_{regular_user_id}" / legacy_relative_path
        )
        legacy_file_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file_path.write_bytes(b"legacy video bytes")

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{quote(legacy_relative_path, safe='')}",
            headers=user_headers,
        )
        assert ticket_response.status_code == 200
        path = ticket_response.json()["path"]
        assert path.startswith(
            f"/api/files/preview/{quote(legacy_relative_path, safe='')}?ticket="
        )

        preview_response = client.get(path)

        assert preview_response.status_code == 200
        assert preview_response.content == b"legacy video bytes"

    def test_minted_ticket_expiry_honors_the_configured_ttl(
        self, test_db, temp_uploads_dir, monkeypatch
    ):
        # tests/core/test_config.py already covers
        # get_file_stream_ticket_ttl_seconds() parsing/validating the env
        # var in isolation; this instead confirms the mint endpoint actually
        # wires that configured value into the minted JWT's own exp claim,
        # not just some other hardcoded default.
        import time

        import jwt as pyjwt

        monkeypatch.setenv("XAGENT_FILE_STREAM_TICKET_TTL_SECONDS", "120")
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=108,
            user_id=regular_user_id,
            title="TTL config test",
            description="ttl config test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        before_mint = time.time()
        ticket_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}",
            headers=user_headers,
        )
        assert ticket_response.status_code == 200
        ticket = ticket_response.json()["path"].split("ticket=")[1]

        claims = pyjwt.decode(ticket, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Bounded rather than exact: real wall-clock elapses between
        # before_mint and the token's own iat/exp computation, and
        # python-jose floors exp to an integer second (timegm), so a naive
        # ..120 upper bound flakes whenever that floor rounds up relative to
        # before_mint's own fractional second -- tightened to 119..121 (0
        # failures simulated across 400k runs) rather than widened, to keep
        # the lower bound meaningfully close to the configured 120s.
        assert 119 <= claims["exp"] - before_mint <= 121

    def test_preview_rejects_an_access_token_used_as_a_ticket(
        self, test_db, temp_uploads_dir
    ):
        # type: "access" must not be redeemable via ?ticket= -- the two
        # credential kinds must stay disjoint even though both are signed
        # with the same secret.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=97,
            user_id=regular_user_id,
            title="Access token as ticket test",
            description="access token as ticket test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        access_token = create_auth_headers(regular_user)["Authorization"].removeprefix(
            "Bearer "
        )

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": access_token},
        )

        assert response.status_code == 401

    def test_stream_ticket_rejected_as_a_bearer_credential(
        self, test_db, temp_uploads_dir
    ):
        # The symmetric direction: type: "file_stream_ticket" must not be
        # accepted as a plain Bearer credential either.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=98,
            user_id=regular_user_id,
            title="Ticket as bearer test",
            description="ticket as bearer test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}",
            headers=user_headers,
        )
        ticket = ticket_response.json()["path"].split("ticket=")[1]

        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            headers={"Authorization": f"Bearer {ticket}"},
        )

        assert response.status_code == 401

    def test_mint_endpoint_requires_a_bearer_credential(
        self, test_db, temp_uploads_dir
    ):
        # issue_preview_stream_ticket uses the standard get_current_user
        # dependency, same as every other authenticated file route -- no
        # special-cased auth logic to verify here beyond that it's actually
        # wired up. A garbage/expired Bearer is rejected 401 by
        # get_current_user's own code (_required_http_rejection), which is
        # deterministic regardless of FastAPI version. A completely missing
        # Authorization header is instead rejected directly by FastAPI's
        # HTTPBearer(auto_error=True) before get_current_user's body ever
        # runs, and that status code is NOT asserted here: it's 403 on the
        # project's locked fastapi==0.115.14 (uv.lock) but empirically 401
        # against fastapi==0.135.1, whatever happens to be installed in a
        # given environment -- this is the version sensitivity documented
        # at _user_from_bearer_or_stream_ticket's own "Not authenticated"
        # comment, and this endpoint has no equivalent explicit-403 guard.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=107,
            user_id=regular_user_id,
            title="Mint auth test",
            description="mint auth test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)

        no_header_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}"
        )
        assert no_header_response.status_code in (401, 403)

        garbage_bearer_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert garbage_bearer_response.status_code == 401

    def test_preview_rejects_ticket_signed_with_wrong_secret(
        self, test_db, temp_uploads_dir
    ):
        import jwt as pyjwt

        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=99,
            user_id=regular_user_id,
            title="Wrong secret ticket test",
            description="wrong secret ticket test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        forged_ticket = pyjwt.encode(
            {
                "type": "file_stream_ticket",
                "sub": regular_user.username,
                "user_id": regular_user.id,
                "file_id": uploaded_file.file_id,
            },
            "not-the-real-secret",
            algorithm="HS256",
        )

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": forged_ticket},
        )

        assert response.status_code == 401

    def test_preview_rejects_ticket_for_a_deleted_user(self, test_db, temp_uploads_dir):
        import jwt as pyjwt

        from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

        admin_user, regular_user, test_app, session = test_db
        del admin_user, regular_user
        deleted_user = User(
            username="soon_deleted",
            password_hash=hash_password("soon_deleted"),
            is_admin=False,
        )
        session.add(deleted_user)
        session.commit()

        deleted_user_id = int(cast(Any, deleted_user.id))
        deleted_username = str(deleted_user.username)

        # _user_from_stream_ticket rejects on the user lookup alone, before
        # ever resolving a file -- no task/upload needs to exist for this
        # deleted user, which also avoids a real cascade-delete headache.
        ticket = pyjwt.encode(
            {
                "type": "file_stream_ticket",
                "sub": deleted_username,
                "user_id": deleted_user_id,
                "file_id": "irrelevant-file-id",
            },
            JWT_SECRET_KEY,
            algorithm=JWT_ALGORITHM,
        )

        session.delete(deleted_user)
        session.commit()

        client = TestClient(test_app)
        response = client.get(
            "/api/files/preview/irrelevant-file-id",
            params={"ticket": ticket},
        )

        assert response.status_code == 401

    def test_empty_ticket_query_param_falls_through_to_bearer_path(
        self, test_db, temp_uploads_dir
    ):
        # ?ticket= with an empty string must not short-circuit into the
        # ticket branch (an empty ticket would then fail JWT decoding);
        # it should behave exactly as if no ticket param were sent.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=101,
            user_id=regular_user_id,
            title="Empty ticket test",
            description="empty ticket test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": ""},
            headers=user_headers,
        )

        assert response.status_code == 200
        assert response.content == b"video bytes"
        # No ticket was actually redeemed, so this must not get the
        # ticket path's no-store override.
        assert "cache-control" not in response.headers

    def test_ticket_takes_precedence_over_a_mismatched_bearer_header(
        self, test_db, temp_uploads_dir
    ):
        # _user_from_bearer_or_stream_ticket checks `if ticket:` before
        # ever looking at the Bearer header -- confirm a valid ticket for
        # one user is honored even when a *different* user's Bearer
        # header is also present, rather than the two being reconciled or
        # the Bearer header silently winning. The mismatched Bearer must
        # belong to a non-admin user with no access to the file: an admin
        # Bearer would also pass _check_file_access on its own, making a
        # 200 non-discriminating between "the ticket won" and "the admin
        # Bearer silently won instead".
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        other_user = User(
            username="precedence-other",
            password_hash=hash_password("precedence-other"),
            is_admin=False,
        )
        session.add(other_user)
        session.commit()

        task = Task(
            id=102,
            user_id=regular_user_id,
            title="Precedence test",
            description="precedence test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        regular_user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}",
            headers=regular_user_headers,
        )
        ticket = ticket_response.json()["path"].split("ticket=")[1]

        other_user_headers = create_auth_headers(other_user)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": ticket},
            headers=other_user_headers,
        )

        assert response.status_code == 200
        assert response.content == b"video bytes"

    def test_preview_rejects_garbage_ticket_even_with_a_valid_bearer_header(
        self, test_db, temp_uploads_dir
    ):
        # Completes the precedence test above from the other direction: a
        # present-but-broken ticket short-circuits before the Bearer header
        # is ever examined, so a valid Bearer for the file's own owner does
        # NOT rescue a garbage ticket via fallthrough.
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=106,
            user_id=regular_user_id,
            title="Garbage ticket precedence test",
            description="garbage ticket precedence test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": "not-a-real-ticket"},
            headers=user_headers,
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired ticket"

    def test_ticket_authenticated_preview_can_use_accel_redirect(
        self, test_db, temp_uploads_dir, monkeypatch
    ):
        # The redirect fast paths are no longer gated on "not ticket" --
        # confirm a ticketed request actually reaches the accel-redirect
        # branch instead of being forced through the app process, and
        # still carries the no-store guard on that response.
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED", "true")
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=103,
            user_id=regular_user_id,
            title="Ticket accel redirect test",
            description="ticket accel redirect test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}",
            headers=user_headers,
        )
        path = ticket_response.json()["path"]

        response = client.get(path, follow_redirects=False)

        assert response.status_code == 200
        assert "x-accel-redirect" in response.headers
        assert response.headers["cache-control"] == "private, no-store"

    def test_bearer_authenticated_preview_keeps_default_caching_on_accel_redirect(
        self, test_db, temp_uploads_dir, monkeypatch
    ):
        # test_authenticated_preview_keeps_default_caching only exercises
        # the direct content-serving exit; without this, a regression that
        # threaded no-store unconditionally into _accel_redirect_response
        # (instead of only via cache_headers, gated on ticket presence)
        # would pass the whole suite undetected, since no other Bearer
        # request reaches this exit.
        monkeypatch.setenv("XAGENT_FILE_DELIVERY_ACCEL_REDIRECT_ENABLED", "true")
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=109,
            user_id=regular_user_id,
            title="Bearer accel redirect caching test",
            description="bearer accel redirect caching test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            headers=user_headers,
            follow_redirects=False,
        )

        assert response.status_code == 200
        assert "x-accel-redirect" in response.headers
        assert "cache-control" not in response.headers

    def test_ticket_authenticated_preview_serves_partial_content_for_range_requests(
        self, test_db, temp_uploads_dir
    ):
        # The entire point of the ticket mechanism is progressive playback
        # via HTTP Range requests -- this exercises that end-to-end on a
        # ticket-authenticated request against the direct content-serving
        # exit (Starlette's FileResponse implements Range support; the
        # accel/durable redirect exits delegate real Range serving to
        # nginx/the object store instead and are covered separately).
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=104,
            user_id=regular_user_id,
            title="Ticket range request test",
            description="ticket range request test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "0123456789video bytes",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}",
            headers=user_headers,
        )
        path = ticket_response.json()["path"]

        response = client.get(path, headers={"Range": "bytes=0-4"})

        assert response.status_code == 206
        assert response.content == b"01234"
        assert response.headers["content-range"] == "bytes 0-4/21"
        assert response.headers["cache-control"] == "private, no-store"

    def test_ticket_authenticated_preview_can_use_durable_redirect(
        self, test_db, temp_uploads_dir, monkeypatch
    ):
        # Mirrors test_ticket_authenticated_preview_can_use_accel_redirect
        # for the other redirect fast path: the durable-object signed-URL
        # 307, which was also previously gated on "not ticket" before this
        # PR's fix and is where a real deployment would delegate Range
        # serving to the object store for large ticketed media.
        from xagent.web.services.managed_file_ref import ManagedFileRef

        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=105,
            user_id=regular_user_id,
            title="Ticket durable redirect test",
            description="ticket durable redirect test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )
        uploaded_file.storage_status = "available"
        uploaded_file.storage_key = (
            f"users/{regular_user_id}/uploads/{uploaded_file.file_id}/clip.mp4"
        )
        uploaded_file.storage_backend = "s3"
        session.commit()

        monkeypatch.setattr(
            ManagedFileRef,
            "signed_access_url",
            lambda self, **kwargs: "https://durable.example/clip.mp4?sig=abc",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        ticket_response = client.get(
            f"/api/files/stream-tickets/{uploaded_file.file_id}",
            headers=user_headers,
        )
        path = ticket_response.json()["path"]

        response = client.get(path, follow_redirects=False)

        assert response.status_code == 307
        assert (
            response.headers["location"] == "https://durable.example/clip.mp4?sig=abc"
        )
        assert response.headers["cache-control"] == "private, no-store"

    def test_bearer_authenticated_preview_keeps_default_caching_on_durable_redirect(
        self, test_db, temp_uploads_dir, monkeypatch
    ):
        # Durable-redirect counterpart to
        # test_bearer_authenticated_preview_keeps_default_caching_on_accel_redirect:
        # without this, a regression that threaded no-store unconditionally
        # into _durable_redirect_response would also pass the whole suite
        # undetected, since no other Bearer request reaches this exit either.
        from xagent.web.services.managed_file_ref import ManagedFileRef

        monkeypatch.setenv("XAGENT_FILE_DELIVERY_REDIRECT_ENABLED", "true")
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=110,
            user_id=regular_user_id,
            title="Bearer durable redirect caching test",
            description="bearer durable redirect caching test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )
        uploaded_file.storage_status = "available"
        uploaded_file.storage_key = (
            f"users/{regular_user_id}/uploads/{uploaded_file.file_id}/clip.mp4"
        )
        uploaded_file.storage_backend = "s3"
        session.commit()

        monkeypatch.setattr(
            ManagedFileRef,
            "signed_access_url",
            lambda self, **kwargs: "https://durable.example/clip.mp4?sig=abc",
        )

        client = TestClient(test_app)
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            headers=user_headers,
            follow_redirects=False,
        )

        assert response.status_code == 307
        assert (
            response.headers["location"] == "https://durable.example/clip.mp4?sig=abc"
        )
        assert "cache-control" not in response.headers

    def test_preview_rejects_garbage_ticket(self, test_db, temp_uploads_dir):
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=93,
            user_id=regular_user_id,
            title="Garbage ticket test",
            description="garbage ticket test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": "not-a-real-jwt"},
        )

        assert response.status_code == 401

    def test_preview_rejects_expired_ticket(self, test_db, temp_uploads_dir):
        from datetime import datetime, timedelta, timezone

        import jwt as pyjwt

        from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=94,
            user_id=regular_user_id,
            title="Expired ticket test",
            description="expired ticket test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        expired_ticket = pyjwt.encode(
            {
                "type": "file_stream_ticket",
                "sub": regular_user.username,
                "user_id": regular_user.id,
                "file_id": uploaded_file.file_id,
                "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            JWT_SECRET_KEY,
            algorithm=JWT_ALGORITHM,
        )

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": expired_ticket},
        )

        assert response.status_code == 401

    def test_preview_rejects_ticket_with_malformed_claims(
        self, test_db, temp_uploads_dir
    ):
        # A ticket signed with the real secret but carrying a string
        # user_id (instead of int) exercises the _AccessTokenRejected path
        # inside _validate_access_token_claim_bindability -- distinct from
        # the JWTError paths already covered by the garbage/expired ticket
        # tests above. This server is the sole minter today, so malformed
        # claims aren't attacker-reachable, but the shared hardening this
        # ticket path reuses from access tokens should still be verified.
        from datetime import datetime, timedelta, timezone

        import jwt as pyjwt

        from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=96,
            user_id=regular_user_id,
            title="Malformed claim ticket test",
            description="malformed claim ticket test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        malformed_ticket = pyjwt.encode(
            {
                "type": "file_stream_ticket",
                "sub": regular_user.username,
                "user_id": str(regular_user_id),
                "file_id": uploaded_file.file_id,
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            JWT_SECRET_KEY,
            algorithm=JWT_ALGORITHM,
        )

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": malformed_ticket},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired ticket"

    def test_preview_rejects_ticket_with_non_convertible_temporal_claim(
        self, test_db, temp_uploads_dir
    ):
        # python-jose's own jwt.decode can raise a bare OverflowError (not
        # JWTError) for a garbage-typed exp claim -- it does int(exp)
        # internally without catching that -- which would 500 instead of
        # the 401 every other malformed-ticket case here returns, without
        # _user_from_stream_ticket's own guard mirroring the one
        # get_current_user already has for access tokens
        # (has_matching_temporal_claim_conversion_failure).
        import jwt as pyjwt

        from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY

        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=111,
            user_id=regular_user_id,
            title="Non-convertible temporal claim ticket test",
            description="non-convertible temporal claim ticket test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        malformed_ticket = pyjwt.encode(
            {
                "type": "file_stream_ticket",
                "sub": regular_user.username,
                "user_id": regular_user_id,
                "file_id": uploaded_file.file_id,
                "exp": float("inf"),
            },
            JWT_SECRET_KEY,
            algorithm=JWT_ALGORITHM,
        )

        client = TestClient(test_app)
        response = client.get(
            f"/api/files/preview/{uploaded_file.file_id}",
            params={"ticket": malformed_ticket},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired ticket"

    def test_preview_requires_bearer_or_ticket(self, test_db, temp_uploads_dir):
        # 403, matching this pre-existing endpoint's error contract from
        # before the ticket param existed: FastAPI's default
        # HTTPBearer(auto_error=True) raises 403 "Not authenticated" for a
        # completely absent Authorization header, not 401 (401 is reserved
        # for a credential that IS present but rejected).
        admin_user, regular_user, test_app, session = test_db
        del admin_user
        regular_user_id = int(cast(Any, regular_user.id))
        task = Task(
            id=95,
            user_id=regular_user_id,
            title="No credential test",
            description="no credential test",
        )
        session.add(task)
        session.commit()

        uploaded_file = create_uploaded_file(
            session,
            temp_uploads_dir,
            regular_user_id,
            int(cast(Any, task.id)),
            "clip.mp4",
            "video bytes",
        )

        client = TestClient(test_app)
        response = client.get(f"/api/files/preview/{uploaded_file.file_id}")

        assert response.status_code == 403
