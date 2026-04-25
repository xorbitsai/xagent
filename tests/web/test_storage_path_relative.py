"""Test relative path storage in uploaded_files table"""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api.auth import hash_password
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.utils.file import to_absolute_path, to_relative_path


class TestRelativePathStorage:
    """Test that uploaded_files.storage_path stores relative paths correctly"""

    @pytest.fixture
    def test_db(self):
        """Create in-memory test database"""
        engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)

        session = SessionLocal()
        try:
            # Create test user
            user = User(
                username="testuser",
                password_hash=hash_password("password"),
                is_admin=False,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            yield user, session, SessionLocal
        finally:
            session.close()
            engine.dispose()

    def test_to_relative_path_converts_absolute_to_relative(self, test_db, monkeypatch):
        """Test to_relative_path converts absolute paths to relative.

        Relative paths are stored relative to user_{user_id} directory,
        NOT including the user_{user_id} prefix.
        """
        user, session, _ = test_db

        with tempfile.TemporaryDirectory() as tmpdir:
            uploads_dir = Path(tmpdir)
            monkeypatch.setattr(
                "xagent.web.utils.file.get_uploads_dir", lambda: uploads_dir
            )

            # Create test file path
            user_dir = uploads_dir / f"user_{user.id}"
            test_file = user_dir / "web_task_123" / "output" / "test.txt"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("test content")

            # Test conversion - should be relative to user_{user_id} directory
            relative = to_relative_path(test_file, user.id)
            # Note: does NOT include user_{user_id} prefix
            assert relative == "web_task_123/output/test.txt"
            assert "/" in relative  # POSIX separator

    def test_to_absolute_path_converts_relative_to_absolute(self, test_db, monkeypatch):
        """Test to_absolute_path converts relative paths to absolute"""
        user, session, _ = test_db

        with tempfile.TemporaryDirectory() as tmpdir:
            uploads_dir = Path(tmpdir)
            monkeypatch.setattr(
                "xagent.web.utils.file.get_uploads_dir", lambda: uploads_dir
            )

            # Relative path without user_{user_id} prefix
            relative_path = "web_task_123/output/test.txt"
            absolute = to_absolute_path(relative_path, user.id)

            expected = uploads_dir / f"user_{user.id}" / relative_path
            assert absolute == expected
            assert absolute.is_absolute()

    def test_to_absolute_path_handles_absolute_input(self, monkeypatch):
        """Test to_absolute_path returns absolute paths as-is"""
        with tempfile.TemporaryDirectory() as tmpdir:
            uploads_dir = Path(tmpdir)
            monkeypatch.setattr(
                "xagent.web.utils.file.get_uploads_dir", lambda: uploads_dir
            )

            abs_path = Path("/some/absolute/path/file.txt")
            result = to_absolute_path(str(abs_path))
            assert result == abs_path

    def test_uploaded_file_absolute_path_property_relative_storage(
        self, test_db, monkeypatch
    ):
        """Test UploadedFile.absolute_path resolves relative storage_path"""
        user, session, _ = test_db

        with tempfile.TemporaryDirectory() as tmpdir:
            uploads_dir = Path(tmpdir)
            monkeypatch.setattr(
                "xagent.web.models.uploaded_file.get_uploads_dir",
                lambda: uploads_dir,
            )

            # Create file record with relative path (new format)
            file_record = UploadedFile(
                file_id="test-file-id",
                user_id=user.id,
                filename="test.txt",
                storage_path="web_task_123/output/test.txt",
                file_size=100,
            )
            session.add(file_record)
            session.commit()
            session.refresh(file_record)

            # absolute_path should resolve correctly
            expected = uploads_dir / f"user_{user.id}" / "web_task_123/output/test.txt"
            assert file_record.absolute_path == expected

    def test_uploaded_file_absolute_path_property_absolute_storage(
        self, test_db, monkeypatch
    ):
        """Test UploadedFile.absolute_path handles absolute storage_path (old format)"""
        user, session, _ = test_db

        with tempfile.TemporaryDirectory() as tmpdir:
            uploads_dir = Path(tmpdir)
            monkeypatch.setattr(
                "xagent.web.models.uploaded_file.get_uploads_dir",
                lambda: uploads_dir,
            )

            # Create file record with absolute path (old format, backward compatibility)
            abs_path = uploads_dir / f"user_{user.id}/web_task_123/output/test.txt"
            file_record = UploadedFile(
                file_id="test-file-id",
                user_id=user.id,
                filename="test.txt",
                storage_path=str(abs_path),
                file_size=100,
            )
            session.add(file_record)
            session.commit()
            session.refresh(file_record)

            # absolute_path should return as-is for absolute paths
            assert file_record.absolute_path == abs_path

    def test_roundtrip_conversion(self, test_db, monkeypatch):
        """Test that absolute -> relative -> absolute roundtrip works"""
        user, session, _ = test_db

        with tempfile.TemporaryDirectory() as tmpdir:
            uploads_dir = Path(tmpdir)
            monkeypatch.setattr(
                "xagent.web.utils.file.get_uploads_dir", lambda: uploads_dir
            )

            # Create test file
            user_dir = uploads_dir / f"user_{user.id}"
            original = user_dir / "web_task_123" / "output" / "test.txt"
            original.parent.mkdir(parents=True)

            # Roundtrip: absolute -> relative -> absolute
            relative = to_relative_path(original, user.id)
            restored = to_absolute_path(relative, user.id)

            assert restored == original
