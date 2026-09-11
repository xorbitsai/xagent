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
        "UPDATE",
        "commit",
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
async def test_concurrent_refresh_consumes_token_once(worker_db, monkeypatch):
    factory, _, engine = worker_db
    logged_in = await auth.login(
        auth.LoginRequest(username="worker", password="password123"), factory
    )
    token = logged_in["refresh_token"]
    barrier = threading.Barrier(2)
    prepare = auth._prepare_refresh_response

    def overlapping_prepare(user, request):
        response = prepare(user, request)
        # Both independent sessions have validated the same old token before
        # either is allowed to attempt its conditional write.
        barrier.wait(timeout=10)
        return response

    monkeypatch.setattr(auth, "_prepare_refresh_response", overlapping_prepare)
    results = await asyncio.gather(
        *(
            auth.refresh_token(auth.RefreshTokenRequest(refresh_token=token), factory)
            for _ in range(2)
        ),
        return_exceptions=True,
    )
    winners = [r for r in results if isinstance(r, auth.RefreshTokenResponse)]
    losers = [r for r in results if isinstance(r, HTTPException)]
    assert len(winners) == len(losers) == 1, results
    assert losers[0].status_code == 401
    assert winners[0].refresh_token != token
    with Session(engine) as session:
        assert session.scalar(select(User.refresh_token)) == winners[0].refresh_token
    monkeypatch.setattr(auth, "_prepare_refresh_response", prepare)
    with pytest.raises(HTTPException) as caught:
        await auth.refresh_token(auth.RefreshTokenRequest(refresh_token=token), factory)
    assert caught.value.status_code == 401


def test_refresh_tokens_are_unique_at_identical_time(monkeypatch):
    from unittest.mock import Mock

    fixed = auth.datetime.now(auth.timezone.utc)
    monkeypatch.setattr(auth, "datetime", Mock(now=Mock(return_value=fixed)))
    first = auth.create_refresh_token({"sub": "worker", "user_id": 1})
    second = auth.create_refresh_token({"sub": "worker", "user_id": 1})
    assert first != second
    assert (
        auth.verify_refresh_token(first)["jti"]
        != auth.verify_refresh_token(second)["jti"]
    )


@pytest.mark.asyncio
async def test_refresh_commit_failure_preserves_old_token(worker_db, monkeypatch):
    factory, events, engine = worker_db
    logged_in = await auth.login(
        auth.LoginRequest(username="worker", password="password123"), factory
    )
    token = logged_in["refresh_token"]

    def fail(session):
        raise SQLAlchemyError("private-db-detail")

    monkeypatch.setattr(factory.class_, "commit", fail)
    with pytest.raises(HTTPException) as caught:
        await auth.refresh_token(auth.RefreshTokenRequest(refresh_token=token), factory)
    assert caught.value.status_code == 503
    assert "private-db-detail" not in caught.value.detail
    assert events[-1][0] == "close"
    with Session(engine) as session:
        assert session.scalar(select(User.refresh_token)) == token


@pytest.mark.asyncio
async def test_login_phase_histograms_use_configured_buckets(worker_db, monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from xagent.core import runtime_performance as metrics

    reader = InMemoryMetricReader()
    provider = MeterProvider(
        metric_readers=[reader],
        views=metrics._histogram_views(),
        shutdown_on_exit=False,
    )
    monkeypatch.setattr(
        metrics,
        "runtime_performance",
        metrics.RuntimePerformanceTelemetry(provider.get_meter("test")),
    )
    try:
        await auth.login(
            auth.LoginRequest(username="worker", password="password123"), worker_db[0]
        )
        recorded = {
            metric.name: metric
            for resource in reader.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        for phase in ("sync_lookup", "sync_commit"):
            point = recorded[f"xagent.auth.login.{phase}.duration"].data.data_points[0]
            assert point.count == 1
            assert tuple(point.explicit_bounds) == tuple(metrics._DURATION_BUCKETS_MS)
    finally:
        provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["login", "refresh"])
async def test_cancelled_auth_drains_commit_and_close(
    worker_db, monkeypatch, operation
):
    factory, events, engine = worker_db
    token = None
    if operation == "refresh":
        logged_in = await auth.login(
            auth.LoginRequest(username="worker", password="password123"), factory
        )
        token = logged_in["refresh_token"]
        events.clear()
    entered = threading.Event()
    release = threading.Event()
    original = factory.class_.commit

    def blocked_commit(session):
        entered.set()
        assert release.wait(5)
        original(session)

    monkeypatch.setattr(factory.class_, "commit", blocked_commit)
    request = (
        auth.refresh_token(auth.RefreshTokenRequest(refresh_token=token), factory)
        if operation == "refresh"
        else auth.login(
            auth.LoginRequest(username="worker", password="password123"), factory
        )
    )
    task = asyncio.create_task(request)
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
        persisted = session.scalar(select(User.refresh_token))
        assert persisted is not None
        if operation == "refresh":
            assert persisted != token


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
