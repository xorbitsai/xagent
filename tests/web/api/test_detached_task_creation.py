"""Task creation must commit an actual file binding or roll the task back."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.web.services.test_detached_file_lifecycle import sessions as sessions_fixture
from xagent.web.api import chat
from xagent.web.models import database
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import file_turn
from xagent.web.services.task_deletion import purge_task_rows

sessions = sessions_fixture


@pytest.mark.parametrize("claim_wins", [False, True])
def test_create_with_detached_file_clears_markers_or_rolls_back(
    sessions, monkeypatch, claim_wins
):
    app = FastAPI()
    app.include_router(chat.chat_router)
    monkeypatch.setattr(database, "_SessionLocal", sessions)
    with sessions() as db:
        purge_task_rows(db, task_id=1, detached_reason="task_deleted")
        db.commit()
        owner = db.get(User, 1)
        app.dependency_overrides[chat.get_current_user] = lambda: owner
        app.dependency_overrides[chat.get_db] = lambda: db
        monkeypatch.setattr(
            chat, "ensure_uploaded_file_local_path", lambda row: Path(row.storage_path)
        )
        original_bind = file_turn.bind_turn_files_no_commit

        def bind_after_claim(**kwargs):
            # Inject the losing bind outcome at the transaction seam; real
            # competing transactions are exercised by the PostgreSQL suite.
            db.query(UploadedFile).filter_by(file_id="attached").update(
                {"storage_status": "compensating"}
            )
            return original_bind(**kwargs)

        if claim_wins:
            monkeypatch.setattr(
                file_turn, "bind_turn_files_no_commit", bind_after_claim
            )
        with TestClient(app) as client:
            response = client.post(
                "/api/chat/task/create",
                json={"title": "explicit reuse", "files": ["attached"]},
            )
        assert response.status_code == (409 if claim_wins else 200), response.text
        db.expire_all()
        row = db.query(UploadedFile).filter_by(file_id="attached").one()
        if claim_wins:
            assert db.query(Task).filter_by(title="explicit reuse").count() == 0
            assert row.detached_reason == "task_deleted"
        else:
            assert row.task_id == response.json()["task_id"]
            assert row.detached_reason is None
            assert row.detached_at is None
