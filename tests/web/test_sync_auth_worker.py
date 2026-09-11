"""Authentication must never return a live synchronous ORM object to the loop."""

import asyncio
import os
import threading
import uuid
from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.api import auth
from xagent.web.models import auth_database
from xagent.web.models.user import User


@pytest.fixture(params=["sqlite", "postgresql"])
def worker_db(tmp_path, request):
    schema = None
    if request.param == "postgresql":
        url = os.getenv("XAGENT_TEST_AUTH_DATABASE_URL")
        if not url:
            pytest.skip("Set XAGENT_TEST_AUTH_DATABASE_URL for PostgreSQL")
        schema = "sync_auth_test_" + uuid.uuid4().hex
        engine = create_engine(
            url, execution_options={"schema_translate_map": {None: schema}}
        )
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'auth.db'}")
    User.__table__.create(engine)
    with Session(engine) as session:
        session.add(
            User(username="worker", password_hash=auth.hash_password("password123"))
        )
        session.commit()
    events = []

    class OwnedSession(Session):
        def __init__(self, *args, **kwargs):
            events.append(("create", threading.get_ident()))
            super().__init__(*args, **kwargs)

        def commit(self):
            events.append(("commit", threading.get_ident()))
            super().commit()

        def close(self):
            events.append(("close", threading.get_ident()))
            super().close()

    @event.listens_for(engine, "before_cursor_execute")
    def statement(conn, cursor, sql, params, context, executemany):
        events.append((sql.split()[0], threading.get_ident()))

    factory = sessionmaker(bind=engine, class_=OwnedSession)
    try:
        yield factory, events, engine
    finally:
        if schema:
            with engine.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()


@pytest.mark.asyncio
async def test_dependency_is_lazy_and_worker_owned(worker_db, monkeypatch):
    factory, events, _ = worker_db

    def provider():
        with factory() as session:
            yield session

    monkeypatch.setattr(auth_database, "get_db", provider)
    dependency = auth_database.get_auth_db()
    lazy_factory = await anext(dependency)
    assert events == []
    response = await auth.login(
        auth.LoginRequest(username="worker", password="password123"), lazy_factory
    )
    assert response["success"]
    await dependency.aclose()
    assert len({owner for _, owner in events}) == 1
    assert events[0][1] != threading.get_ident()
    assert events[-1][0] == "close"


@pytest.mark.asyncio
async def test_login_and_refresh_have_single_thread_owners(worker_db):
    factory, events, _ = worker_db
    response = await auth.login(
        auth.LoginRequest(username="worker", password="password123"), factory
    )
    assert response["user"]["username"] == "worker"
    assert [name for name, _ in events] == [
        "create",
        "SELECT",
        "commit",
        "UPDATE",
        "close",
    ]
    assert len({owner for _, owner in events}) == 1
    assert events[0][1] != threading.get_ident()
    events.clear()
    # Force a different token so UPDATE is observable even within one JWT second.
    from unittest.mock import patch

    with patch.object(auth, "create_refresh_token", return_value="rotated-token"):
        renewed = await auth.refresh_token(
            auth.RefreshTokenRequest(refresh_token=response["refresh_token"]), factory
        )
    assert renewed.refresh_token == "rotated-token"
    assert [name for name, _ in events] == [
        "create",
        "SELECT",
        "commit",
        "UPDATE",
        "close",
    ]
    assert len({owner for _, owner in events}) == 1
    assert events[0][1] != threading.get_ident()


@pytest.mark.asyncio
async def test_commit_failure_rolls_back_and_closes(worker_db, monkeypatch):
    factory, events, engine = worker_db

    def fail(session):
        session.flush()
        raise SQLAlchemyError("private-db-detail")

    monkeypatch.setattr(factory.class_, "commit", fail)
    with pytest.raises(HTTPException) as caught:
        await auth.login(
            auth.LoginRequest(username="worker", password="password123"), factory
        )
    assert caught.value.status_code == 500
    assert "private-db-detail" not in caught.value.detail
    assert events[-1][0] == "close"
    with Session(engine) as session:
        assert session.scalar(select(User.refresh_token)) is None


@pytest.mark.asyncio
async def test_cancelled_login_drains_commit_and_close(worker_db, monkeypatch):
    factory, events, engine = worker_db
    entered = threading.Event()
    release = threading.Event()
    original = factory.class_.commit

    def blocked_commit(session):
        entered.set()
        assert release.wait(5)
        original(session)

    monkeypatch.setattr(factory.class_, "commit", blocked_commit)
    task = asyncio.create_task(
        auth.login(
            auth.LoginRequest(username="worker", password="password123"), factory
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not any(name == "close" for name, _ in events)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert events[-1][0] == "close"
    assert len({owner for _, owner in events}) == 1
    # Cancellation cannot promise rollback if commit has already begun.
    with Session(engine) as session:
        assert session.scalar(select(User.refresh_token)) is not None


@pytest.mark.asyncio
async def test_bad_password_closes_without_commit(worker_db):
    factory, events, _ = worker_db
    with pytest.raises(HTTPException) as caught:
        await auth.login(
            auth.LoginRequest(username="worker", password="wrong"), factory
        )
    assert caught.value.status_code == 401
    assert [name for name, _ in events] == ["create", "SELECT", "close"]


@pytest.mark.asyncio
async def test_cancelled_worker_before_session_entry_still_closes(worker_db):
    factory, events, _ = worker_db
    entered = threading.Event()
    release = threading.Event()

    @contextmanager
    def delayed_factory():
        entered.set()
        assert release.wait(5)
        with factory() as session:
            yield session

    task = asyncio.create_task(
        auth.login(
            auth.LoginRequest(username="worker", password="password123"),
            delayed_factory,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert events[-1][0] == "close"
