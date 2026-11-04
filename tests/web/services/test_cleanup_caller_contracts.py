"""Async upload consumers preserve cleanup-race and event-loop contracts."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.web.services.test_complete_upload_cleanup import (
    lifecycle as cleanup_lifecycle_fixture,
)
from tests.web.services.test_detached_file_lifecycle import (
    sessions as detached_sessions_fixture,
)
from xagent.core.tools.adapters.vibe.file_ingestion_tool import (
    CreateKnowledgeBaseFromFileTool,
)
from xagent.web.api import chat
from xagent.web.api.websocket import handle_file_upload_for_task
from xagent.web.models import database
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import agent_service_manager
from xagent.web.services.task_deletion import purge_task_rows

lifecycle = cleanup_lifecycle_fixture
detached_sessions = detached_sessions_fixture


async def _heartbeat_until(done: asyncio.Event) -> int:
    ticks = 0
    while not done.is_set():
        await asyncio.sleep(0.01)
        ticks += 1
    return ticks


def test_chat_claim_between_select_and_restore_keeps_missing_file_contract(
    detached_sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = FastAPI()
    app.include_router(chat.chat_router)
    monkeypatch.setattr(database, "_SessionLocal", detached_sessions)
    engine = detached_sessions.kw["bind"]
    validating = Event()
    resume = Event()
    upload_selects = 0

    def pause_second_upload_select(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        nonlocal upload_selects
        if "FROM uploaded_files" in statement:
            upload_selects += 1
            if upload_selects == 2:
                validating.set()
                assert resume.wait(10)

    with detached_sessions() as request_db:
        purge_task_rows(request_db, task_id=1, detached_reason="task_deleted")
        request_db.commit()
        owner = request_db.get(User, 1)
        attached = request_db.query(UploadedFile).filter_by(file_id="attached").one()
        source = attached.storage_path
        Path(source).write_text("attached", encoding="utf-8")
        app.dependency_overrides[chat.get_current_user] = lambda: owner
        app.dependency_overrides[chat.get_db] = lambda: request_db
        sa.event.listen(engine, "before_cursor_execute", pause_second_upload_select)
        try:
            with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
                response_future = pool.submit(
                    client.post,
                    "/api/chat/task/create",
                    json={"title": "claimed during restore", "files": ["attached"]},
                )
                assert validating.wait(10)
                with detached_sessions.begin() as claim_db:
                    claim_db.query(UploadedFile).filter_by(file_id="attached").delete()
                resume.set()
                response = response_future.result(timeout=10)
        finally:
            resume.set()
            sa.event.remove(engine, "before_cursor_execute", pause_second_upload_select)

        assert response.status_code == 200, response.text
        task = request_db.get(Task, response.json()["task_id"])
        assert "File does not exist" in task.description
        assert not task.agent_config or "selected_file_ids" not in task.agent_config
        assert Path(source).exists()


@pytest.mark.asyncio
async def test_websocket_claim_between_select_and_restore_is_missing_without_loop_stall(
    lifecycle,
) -> None:
    sessions, source, _, _, _, file_id = lifecycle
    engine = sessions.kw["bind"]
    caller_thread = get_ident()
    validating = Event()
    resume = Event()

    def pause_publication_validation(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        if (
            get_ident() != caller_thread
            and "FROM uploaded_files" in statement
            and not validating.is_set()
        ):
            validating.set()
            assert resume.wait(10)

    sa.event.listen(engine, "before_cursor_execute", pause_publication_validation)
    done = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat_until(done))
    try:
        with sessions() as db:
            request = asyncio.create_task(
                handle_file_upload_for_task(
                    99,
                    [{"file_id": file_id}],
                    db,
                    SimpleNamespace(id=1, is_admin=False),
                    task_owner_id=1,
                )
            )
            assert await asyncio.to_thread(validating.wait, 10)
            with sessions.begin() as claim_db:
                claim_db.query(UploadedFile).filter_by(file_id=file_id).delete()
            resume.set()
            result = await request
    finally:
        resume.set()
        sa.event.remove(engine, "before_cursor_execute", pause_publication_validation)
        done.set()
    ticks = await heartbeat

    assert result == {"uploaded_files": [], "file_info_list": []}
    assert ticks >= 1
    assert source.exists()


@pytest.mark.asyncio
async def test_knowledge_base_lock_wait_keeps_event_loop_responsive(
    lifecycle, monkeypatch: pytest.MonkeyPatch
) -> None:
    from filelock import FileLock

    from xagent.core.tools.core.RAG_tools.storage.file_reference import (
        file_cleanup_lock,
    )
    from xagent.web.services import managed_file_ref

    _, source, materialized, _, _, file_id = lifecycle
    source.unlink()
    materialized.unlink()
    entered = Event()
    release = Event()
    original_acquire = FileLock.acquire

    def bounded_acquire(lock, *args, **kwargs):
        if str(lock.lock_file).endswith(".cleanup.lock"):
            kwargs["timeout"] = 0.15
        return original_acquire(lock, *args, **kwargs)

    def hold_lock() -> None:
        with file_cleanup_lock(file_id):
            entered.set()
            assert release.wait(10)

    class FakeKnowledgeBaseService:
        def __init__(self, **_kwargs) -> None:
            pass

        async def prepare_collection(self, name: str) -> str:
            return name

        async def collection_exists(self, _name: str) -> bool:
            return True

        async def cleanup_failed_collection(self, _name: str) -> None:
            return None

    monkeypatch.setattr(FileLock, "acquire", bounded_acquire)
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.agent_kb_service.AgentKnowledgeBaseService",
        FakeKnowledgeBaseService,
    )
    # Prove the production async helper is used, rather than a patched resolver.
    assert managed_file_ref.async_ensure_uploaded_file_local_path is not None

    done = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat_until(done))
    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(hold_lock)
        assert entered.wait(5)
        try:
            result = await CreateKnowledgeBaseFromFileTool(user_id=1).run_json_async(
                {"file_ids": [file_id], "collection_name": "caller_contract"}
            )
        finally:
            release.set()
            owner.result(timeout=5)
            done.set()
    ticks = await heartbeat

    assert result["success"] is False
    assert "Failed to restore source.txt from durable storage" in result["message"]
    assert ticks >= 3


def test_selected_file_registration_skips_row_deleted_after_select(lifecycle) -> None:
    sessions, source, _, _, _, file_id = lifecycle
    engine = sessions.kw["bind"]
    selected = Event()
    resume = Event()
    worker_thread: list[int] = []
    upload_selects = 0

    def pause_second_upload_select(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        nonlocal upload_selects
        if (
            worker_thread
            and get_ident() == worker_thread[0]
            and "FROM uploaded_files" in statement
        ):
            upload_selects += 1
            if upload_selects == 2:
                selected.set()
                assert resume.wait(10)

    class Workspace:
        registrations: list[tuple[str, str | None]] = []

        def register_files(self, registrations) -> None:
            self.registrations.extend(registrations)

    def register() -> None:
        worker_thread.append(get_ident())
        agent_service_manager._register_selected_task_files_isolated(
            Workspace(),
            task_id=77,
            task_owner_id=1,
            selected_file_ids=[file_id],
        )

    sa.event.listen(engine, "before_cursor_execute", pause_second_upload_select)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            worker = pool.submit(register)
            assert selected.wait(10)
            with sessions.begin() as claim_db:
                claim_db.query(UploadedFile).filter_by(file_id=file_id).delete()
            resume.set()
            worker.result(timeout=10)
    finally:
        resume.set()
        sa.event.remove(engine, "before_cursor_execute", pause_second_upload_select)

    assert Workspace.registrations == []
    assert source.exists()
