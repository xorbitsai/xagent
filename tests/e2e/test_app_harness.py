from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.e2e.app_harness import build_access_token, seed_registered_local_file
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


def test_build_access_token_contains_user_claims():
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    token = build_access_token(
        username="e2e-user",
        user_id=42,
        expires_at=expires_at,
    )

    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert payload["sub"] == "e2e-user"
    assert payload["user_id"] == 42
    assert payload["type"] == "access"


def test_seed_registered_local_file_creates_file_and_db_record(tmp_path):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = User(id=7, username="seed-user", password_hash="hash")
        db.add(user)
        db.commit()

        seeded = seed_registered_local_file(
            db,
            uploads_dir=tmp_path / "uploads",
            user_id=7,
            filename="report.txt",
            content="report body\n",
            file_id="file-1",
            relative_dir="startup",
            mime_type="text/plain",
        )

        assert seeded.path.read_text(encoding="utf-8") == "report body\n"
        assert seeded.path == tmp_path / "uploads" / "user_7" / "startup" / "report.txt"

        record = db.query(UploadedFile).filter(UploadedFile.file_id == "file-1").one()
        assert record.user_id == 7
        assert record.filename == "report.txt"
        assert record.storage_path == str(seeded.path)
        assert record.mime_type == "text/plain"
        assert record.file_size == len("report body\n")
        assert record.storage_status == "legacy"
    finally:
        db.close()
        engine.dispose()


@pytest.mark.parametrize("timeout", [0, -1])
def test_shared_wait_task_reports_timeout_before_first_read(tmp_path, timeout):
    from tests.e2e.shared_execution_harness import SharedExecutionApp

    app = SharedExecutionApp(root=tmp_path, environment={}, token="unused", user_id=1)
    with pytest.raises(
        pytest.fail.Exception, match="Task did not reach completed: None"
    ):
        app.wait_task(1, timeout=timeout)
