import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.api import websocket as websocket_api
from xagent.web.api.files import public_download_file, public_preview_file
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User


@pytest.mark.parametrize("endpoint", [public_download_file, public_preview_file])
def test_public_endpoints_reject_existing_legacy_paths(monkeypatch, tmp_path, endpoint):
    target = tmp_path / "user_1/private/report.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-private")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint("private/report.pdf", db=object()))

    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("requesting_user_id,allowed", [(1, True), (2, False)])
def test_registered_legacy_file_requires_its_owner(
    monkeypatch, tmp_path, requesting_user_id, allowed
):
    target = tmp_path / "report.pdf"
    monkeypatch.setattr(
        websocket_api,
        "_resolve_legacy_preview_storage_path",
        lambda path: (target, "user_1/report.pdf"),
    )
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
        user_id=1, file_id="registered-file"
    )
    factory = MagicMock()
    factory.return_value.__enter__.return_value = session
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: factory)
    if allowed:
        assert (
            websocket_api._register_legacy_preview_isolated(
                "user_1/report.pdf", requesting_user_id
            )
            == "registered-file"
        )
    else:
        with pytest.raises(websocket_api._LegacyPreviewRegistrationError):
            websocket_api._register_legacy_preview_isolated(
                "user_1/report.pdf", requesting_user_id
            )


@pytest.fixture
def legacy_db(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    with sessions() as db:
        db.add_all(
            [
                User(id=1, username="owner", password_hash="x"),
                User(id=2, username="other", password_hash="x"),
            ]
        )
        db.commit()
    monkeypatch.setattr(websocket_api, "get_session_local", lambda: sessions)
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    target = tmp_path / "uploads/user_1/web_task_7/report.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-private")
    try:
        yield sessions, target
    finally:
        engine.dispose()


def test_unregistered_cross_user_path_never_stages_or_registers(legacy_db, monkeypatch):
    sessions, target = legacy_db
    stage = Mock()
    monkeypatch.setattr(websocket_api, "stage_uploaded_file_from_local_path", stage)
    with pytest.raises(
        websocket_api._LegacyPreviewRegistrationError,
        match="^Legacy preview target not found$",
    ):
        websocket_api._register_legacy_preview_isolated(
            "user_1/web_task_7/report.pdf", 2
        )
    stage.assert_not_called()
    with sessions() as db:
        assert db.query(UploadedFile).count() == 0
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("race", ["current", "winner"])
def test_cross_user_race_discards_staged_object(legacy_db, monkeypatch, tmp_path, race):
    sessions, target = legacy_db
    # Use a non-task path so task validation is orthogonal to this owner race.
    monkeypatch.setattr(
        websocket_api,
        "_resolve_legacy_preview_storage_path",
        lambda _: (target, "user_1/report.pdf"),
    )
    staged_path = tmp_path / "staged-object"
    staged = SimpleNamespace(file_id="staged", local_path=staged_path)

    def insert_winner():
        with sessions() as db:
            db.add(
                UploadedFile(
                    file_id="winner",
                    user_id=2,
                    filename="report.pdf",
                    storage_path=str(target),
                )
            )
            db.commit()

    def stage(**kwargs):
        staged_path.write_bytes(b"staged")
        if race == "current":
            insert_winner()
        return staged

    def conflict(*args, **kwargs):
        insert_winner()
        raise websocket_api.UploadedFileVersionConflict("concurrent winner")

    compensation = Mock(side_effect=lambda values: (staged_path.unlink(), ())[1])
    monkeypatch.setattr(websocket_api, "stage_uploaded_file_from_local_path", stage)
    monkeypatch.setattr(websocket_api, "compensate_staged_uploaded_files", compensation)
    if race == "winner":
        monkeypatch.setattr(
            websocket_api.UploadedFileStore, "upsert_already_durable", conflict
        )
    with pytest.raises(
        websocket_api._LegacyPreviewRegistrationError, match="target not found"
    ):
        websocket_api._register_legacy_preview_isolated("user_1/report.pdf", 1)
    compensation.assert_called_once_with((staged,))
    assert not staged_path.exists()
    with sessions() as db:
        assert db.query(UploadedFile).one().user_id == 2


def test_anonymous_legacy_preview_is_rejected_before_registration(monkeypatch):
    register = Mock()
    monkeypatch.setattr(websocket_api, "_register_legacy_preview_isolated", register)
    app = FastAPI()
    app.include_router(websocket_api.ws_router)
    app.dependency_overrides[websocket_api.get_db] = lambda: object()
    with TestClient(app) as client:
        assert client.get("/preview/user_1/report.pdf").status_code in (401, 403)
    register.assert_not_called()
