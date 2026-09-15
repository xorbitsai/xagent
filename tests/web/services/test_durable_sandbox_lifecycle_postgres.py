from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.sandbox import DurableSandboxLifecycle
from xagent.web.services.chrome_lifecycle import ChromeLifecycleCoordinator
from xagent.web.services.durable_sandbox_lifecycle import (
    DurableSandboxLifecycleService,
    RegisterLifecycle,
)

pytestmark = pytest.mark.postgresql

NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)


@pytest.fixture
def postgres_engine_factory():
    with disposable_database_factory("chrome_lifecycle") as factory:
        yield factory


@contextmanager
def _sessions(engine: sa.Engine):
    metadata = sa.MetaData()
    sa.Table(
        "tasks",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("runner_id", sa.String(255)),
        sa.Column("run_id", sa.String(64)),
        sa.Column("lease_attempt_id", sa.String(64)),
        sa.Column("status", sa.String(32)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
    )
    metadata.create_all(engine)
    DurableSandboxLifecycle.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


class _Backend:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_durable_sandbox_strict(self, lifecycle_id: str) -> None:
        self.deleted.append(lifecycle_id)


@pytest.mark.asyncio
async def test_postgres_two_sessions_allow_one_sweeper_delete(
    postgres_engine_factory,
) -> None:
    engine = postgres_engine_factory("two_sweepers")
    with _sessions(engine) as sessions:
        service_a = DurableSandboxLifecycleService(sessions)
        service_b = DurableSandboxLifecycleService(sessions)
        fence = service_a.register(
            RegisterLifecycle(
                scope_digest="a" * 64,
                task_id=7,
                run_id="run-1",
                lease_attempt_id="attempt-1",
                turn_digest=None,
                eligible_at=NOW - timedelta(minutes=2),
                owner_lease_expires_at=NOW - timedelta(minutes=1),
            )
        )
        backend = _Backend()
        first = ChromeLifecycleCoordinator(service_a, backend, now=lambda: NOW)
        second = ChromeLifecycleCoordinator(service_b, backend, now=lambda: NOW)

        outcomes = await asyncio.gather(first.sweep_once(), second.sweep_once())

        assert sum(outcomes) == 1
        assert backend.deleted == [fence.backend_lifecycle_digest]
        assert service_a.get_by_scope(fence.scope_digest) is None
