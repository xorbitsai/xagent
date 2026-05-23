import os
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.file_storage.factory import get_file_storage
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
        get_file_storage.cache_clear()
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
        stored_object = get_file_storage().put_file(
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
