"""Conversation Logs queries must not block the web event loop."""

import asyncio
from threading import Event
from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from tests.shared.postgres_disposable import disposable_session_factory
from xagent.web.api import conversation_logs
from xagent.web.api.auth import create_access_token
from xagent.web.auth_sessions import user_session_claims
from xagent.web.models.database import get_engine
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User


@pytest.mark.parametrize(
    ("path", "query_helper"),
    [
        ("/api/conversation-logs", "_conversation_source_query"),
        ("/api/conversation-logs/1", "_base_task_query"),
    ],
)
def test_slow_conversation_log_query_does_not_block_other_requests(
    monkeypatch, path, query_helper
):
    query_started = Event()
    release_query = Event()

    def slow_query(*_args, **_kwargs):
        query_started.set()
        assert release_query.wait(timeout=5), "query outlasted the probe"
        raise HTTPException(status_code=404, detail="released")

    monkeypatch.setattr(conversation_logs, query_helper, slow_query)
    app = FastAPI()
    app.include_router(conversation_logs.router)
    app.dependency_overrides[conversation_logs.get_current_user] = lambda: User(
        id=1, is_admin=False
    )
    app.dependency_overrides[conversation_logs.get_db] = lambda: MagicMock(spec=Session)

    @app.get("/probe")
    async def probe():
        return {"ok": True}

    async def check():
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as browser:
            logs = asyncio.create_task(browser.get(path))
            try:
                started = await asyncio.to_thread(query_started.wait, 3)
                probe = await browser.get("/probe")
                probe_finished_first = not logs.done()
            finally:
                release_query.set()
                (response,) = await asyncio.gather(logs, return_exceptions=True)
            assert started
            assert probe.json() == {"ok": True}
            assert probe_finished_first, "the probe must finish during the query"
            assert not isinstance(response, BaseException), response
            assert response.status_code == 404
            assert response.json() == {"detail": "released"}

    asyncio.run(check())


@pytest.mark.postgresql
def test_postgresql_locked_conversation_logs_release_sessions(monkeypatch):
    with (
        monkeypatch.context() as patch,
        disposable_session_factory("conversation_logs_loop", patch) as sessions,
    ):
        engine = get_engine()
        with sessions.begin() as db:
            user = User(username="conversation-logs-loop", password_hash="unused")
            db.add(user)
            db.flush()
            task = Task(
                user_id=user.id,
                title="Widget conversation",
                description="Widget conversation",
                status=TaskStatus.COMPLETED,
                source="widget",
                is_visible=False,
            )
            db.add(task)
            db.flush()
            user_id, task_id = user.id, task.id
            token = create_access_token(user_session_claims(user))

        logs_query_started = Event()

        @event.listens_for(engine, "before_cursor_execute")
        def bound_logs_lock_wait(conn, cursor, statement, parameters, context, many):
            if "FROM tasks" in statement:
                cursor.execute("SET LOCAL lock_timeout = '5s'")
                logs_query_started.set()

        app = FastAPI()
        app.include_router(conversation_logs.router)

        @app.get("/probe")
        def probe(user: User = Depends(conversation_logs.get_current_user)):
            return {"user_id": user.id}

        async def check(path):
            logs_query_started.clear()
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as browser:
                assert (await browser.get(path)).status_code == 401
                headers = {"Authorization": f"Bearer {token}"}
                with engine.connect() as blocker:
                    blocker.exec_driver_sql("LOCK TABLE tasks IN ACCESS EXCLUSIVE MODE")
                    logs = asyncio.create_task(browser.get(path, headers=headers))
                    try:
                        started = await asyncio.to_thread(logs_query_started.wait, 3)
                        probe = await browser.get("/probe", headers=headers)
                        probe_finished_first = not logs.done()
                    finally:
                        blocker.rollback()
                        (response,) = await asyncio.gather(logs, return_exceptions=True)
                assert started
                assert probe.json() == {"user_id": user_id}
                assert probe_finished_first, "the probe must finish during the query"
                assert not isinstance(response, BaseException), response
                assert response.status_code == 200
                assert engine.pool.checkedout() == 0
                return response.json()

        listed = asyncio.run(check("/api/conversation-logs"))
        assert [log["task_id"] for log in listed["logs"]] == [task_id]
        detail = asyncio.run(check(f"/api/conversation-logs/{task_id}"))
        assert detail["log"]["task_id"] == task_id
