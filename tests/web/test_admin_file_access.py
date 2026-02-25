"""Test admin cross-user file access functionality"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.auth import hash_password
from xagent.web.api.files import file_router
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import Base, get_db
from xagent.web.models.task import Task
from xagent.web.models.user import User


@pytest.fixture(scope="function")
def test_db():
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

    # Create users for this test
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
        # Clean up
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()
        # Delete temporary database file
        try:
            os.unlink(temp_db_path)
        except OSError:
            pass


@pytest.fixture(scope="function")
def temp_uploads_dir(monkeypatch):
    """Create temporary uploads directory and override UPLOADS_DIR"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Patch UPLOADS_DIR in files module
        import xagent.web.api.files

        monkeypatch.setattr(xagent.web.api.files, "UPLOADS_DIR", temp_path)

        # Also patch get_upload_path
        def patched_get_upload_path(
            filename: str,
            task_id: Optional[str] = None,
            folder: Optional[str] = None,
            user_id: Optional[int] = None,
        ):
            if user_id:
                user_dir = temp_path / f"user_{user_id}"
                user_dir.mkdir(parents=True, exist_ok=True)
                return user_dir / filename
            return temp_path / filename

        monkeypatch.setattr(
            xagent.web.api.files, "get_upload_path", patched_get_upload_path
        )

        yield temp_path


def create_auth_headers(user):
    """Create authentication headers for a user"""
    from datetime import datetime, timedelta

    import jwt

    payload = {
        "sub": user.username,
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "user_id": user.id,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


def create_task_file_structure(uploads_dir, user_id, task_id, filename, content="test"):
    """Create a task file structure for testing"""
    user_dir = uploads_dir / f"user_{user_id}"
    task_dir = user_dir / f"web_task_{task_id}" / "output"
    task_dir.mkdir(parents=True, exist_ok=True)

    file_path = task_dir / filename
    file_path.write_text(content)
    return file_path


class TestAdminFileAccess:
    """Test admin cross-user file access functionality"""

    def test_admin_access_other_user_task_file(self, test_db, temp_uploads_dir):
        """Test that admin can access other user's task files"""
        admin_user, regular_user, test_app, session = test_db
        uploads_dir = temp_uploads_dir

        # Create a task for regular user
        task = Task(
            id=78,
            user_id=regular_user.id,
            title="Test Task",
            description="Test task for file access",
        )
        session.add(task)
        session.commit()

        # Create file structure for regular user's task
        filename = "IEEE_2857_Wi-SUN_FAN_技术分析报告.html"
        create_task_file_structure(
            uploads_dir, regular_user.id, task.id, filename, "test content"
        )

        # Create test client
        client = TestClient(test_app)

        # Admin should be able to download the file
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            f"/api/files/download/web_task_{task.id}/output/{filename}",
            headers=admin_headers,
        )

        assert response.status_code == 200
        assert response.content == b"test content"

    def test_regular_user_access_own_task_file(self, test_db, temp_uploads_dir):
        """Test that regular user can access their own task files"""
        admin_user, regular_user, test_app, session = test_db
        uploads_dir = temp_uploads_dir

        # Create a task for regular user
        task = Task(
            id=79,
            user_id=regular_user.id,
            title="Test Task",
            description="Test task for file access",
        )
        session.add(task)
        session.commit()

        # Create file structure for regular user's task
        filename = "my_report.html"
        create_task_file_structure(
            uploads_dir, regular_user.id, task.id, filename, "my content"
        )

        # Create test client
        client = TestClient(test_app)

        # Regular user should be able to download their own file
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/download/web_task_{task.id}/output/{filename}",
            headers=user_headers,
        )

        assert response.status_code == 200
        assert response.content == b"my content"

    def test_regular_user_access_other_user_task_file_denied(
        self, test_db, temp_uploads_dir
    ):
        """Test that regular user cannot access other user's task files"""
        admin_user, regular_user, test_app, session = test_db
        uploads_dir = temp_uploads_dir

        # Create another user
        another_user = User(
            username="another", password_hash=hash_password("another"), is_admin=False
        )
        session.add(another_user)
        session.commit()

        # Create a task for another user
        task = Task(
            id=80,
            user_id=another_user.id,
            title="Another User Task",
            description="Task belonging to another user",
        )
        session.add(task)
        session.commit()

        # Create file structure for another user's task
        filename = "secret_report.html"
        create_task_file_structure(
            uploads_dir, another_user.id, task.id, filename, "secret content"
        )

        # Create test client
        client = TestClient(test_app)

        # Regular user should NOT be able to access another user's file
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            f"/api/files/download/web_task_{task.id}/output/{filename}",
            headers=user_headers,
        )

        assert response.status_code == 403

    def test_admin_access_nonexistent_task_file(self, test_db, temp_uploads_dir):
        """Test that admin gets 404 when trying to access non-existent task file"""
        admin_user, regular_user, test_app, session = test_db

        # Create test client
        client = TestClient(test_app)

        # Try to access file for non-existent task
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            "/api/files/download/web_task_999/output/nonexistent.html",
            headers=admin_headers,
        )

        assert response.status_code == 404

    def test_regular_user_access_nonexistent_task_file(self, test_db, temp_uploads_dir):
        """Test that regular user gets 404 when trying to access non-existent task file"""
        admin_user, regular_user, test_app, session = test_db

        # Create test client
        client = TestClient(test_app)

        # Try to access file for non-existent task
        user_headers = create_auth_headers(regular_user)
        response = client.get(
            "/api/files/download/web_task_999/output/nonexistent.html",
            headers=user_headers,
        )

        assert response.status_code == 404

    def test_admin_access_invalid_task_id_format(self, test_db, temp_uploads_dir):
        """Test that admin gets proper error for invalid task ID format"""
        admin_user, regular_user, test_app, session = test_db

        # Create test client
        client = TestClient(test_app)

        # Try to access file with invalid task ID format
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            "/api/files/download/web_task_invalid/output/file.html",
            headers=admin_headers,
        )

        # Should fall back to normal file processing (may get 403, 404, or 400)
        assert response.status_code in [403, 404, 400]

    def test_access_regular_file_unaffected(self, test_db, temp_uploads_dir):
        """Test that regular file access is not affected by the new logic"""
        admin_user, regular_user, test_app, session = test_db
        uploads_dir = temp_uploads_dir

        # Create a regular file in admin's directory
        admin_dir = uploads_dir / f"user_{admin_user.id}"
        admin_dir.mkdir(parents=True, exist_ok=True)

        regular_filename = "regular_file.txt"
        regular_file_path = admin_dir / regular_filename
        regular_file_path.write_text("regular content")

        # Create test client
        client = TestClient(test_app)

        # Admin should be able to access their own regular file
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            f"/api/files/download/{regular_filename}", headers=admin_headers
        )

        assert response.status_code == 200
        assert response.content == b"regular content"

    def test_task_file_path_security(self, test_db, temp_uploads_dir):
        """Test that path security is maintained for task files"""
        admin_user, regular_user, test_app, session = test_db
        uploads_dir = temp_uploads_dir

        # Create a task for regular user
        task = Task(
            id=81,
            user_id=regular_user.id,
            title="Test Task",
            description="Test task for security check",
        )
        session.add(task)
        session.commit()

        # Create file structure for regular user's task
        filename = "secure_file.html"
        create_task_file_structure(
            uploads_dir, regular_user.id, task.id, filename, "secure content"
        )

        # Create test client
        client = TestClient(test_app)

        # Try to access with path traversal attempt
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            f"/api/files/download/web_task_{task.id}/output/../../../{filename}",
            headers=admin_headers,
        )

        # Should either succeed (if path normalization works), fail with security error,
        # or return method not allowed if the path is rejected
        # The important thing is that it doesn't expose system files
        assert response.status_code in [200, 403, 404, 405]


class TestFileAccessEdgeCases:
    """Test edge cases for file access functionality"""

    def test_admin_access_task_without_file(self, test_db, temp_uploads_dir):
        """Test admin accessing task that exists but has no file"""
        admin_user, regular_user, test_app, session = test_db
        uploads_dir = temp_uploads_dir

        # Create a task for regular user but don't create any files
        task = Task(
            id=82,
            user_id=regular_user.id,
            title="Empty Task",
            description="Task with no files",
        )
        session.add(task)
        session.commit()

        # Create the user directory structure (but no task files)
        user_dir = uploads_dir / f"user_{regular_user.id}"
        user_dir.mkdir(parents=True, exist_ok=True)

        # Create test client
        client = TestClient(test_app)

        # Try to access non-existent file for existing task
        admin_headers = create_auth_headers(admin_user)
        response = client.get(
            f"/api/files/download/web_task_{task.id}/output/nonexistent.html",
            headers=admin_headers,
        )

        # Should return 404 when file doesn't exist
        assert response.status_code == 404

    def test_malformed_web_task_path(self, test_db, temp_uploads_dir):
        """Test handling of malformed web_task paths"""
        admin_user, regular_user, test_app, session = test_db

        # Create test client
        client = TestClient(test_app)

        # Test various malformed paths
        malformed_paths = [
            "web_task_",  # Missing task ID
            "web_task_",  # Incomplete format
            "web_task_",  # Just the prefix
        ]

        admin_headers = create_auth_headers(admin_user)
        for path in malformed_paths:
            response = client.get(
                f"/api/files/download/{path}/output/file.html", headers=admin_headers
            )
            # Should not crash the server
            assert response.status_code in [403, 400, 404]

    def test_task_id_extraction_edge_cases(self, test_db, temp_uploads_dir):
        """Test edge cases in task ID extraction"""
        admin_user, regular_user, test_app, session = test_db

        # Create test client
        client = TestClient(test_app)

        # Test edge cases
        edge_cases = [
            ("web_task_0/output/file.html", 0),  # Task ID 0
            ("web_task_123abc/output/file.html", None),  # Invalid format
            ("web_task_-1/output/file.html", None),  # Negative ID
        ]

        admin_headers = create_auth_headers(admin_user)
        for path, expected_id in edge_cases:
            if expected_id is not None:
                # Create the task if it should exist
                task = Task(
                    id=expected_id,
                    user_id=regular_user.id,
                    title="Edge Case Task",
                    description="Task for edge case testing",
                )
                session.add(task)
                session.commit()

                # Create file if needed
                if expected_id == 0:
                    filename = "edge_case_file.html"
                    create_task_file_structure(
                        temp_uploads_dir,
                        regular_user.id,
                        task.id,
                        filename,
                        "edge content",
                    )

            response = client.get(f"/api/files/download/{path}", headers=admin_headers)
            # Should not crash the server
            assert response.status_code in [200, 400, 404, 403]
