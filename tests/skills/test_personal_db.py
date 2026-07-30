from __future__ import annotations

import asyncio
import os
import threading
import uuid

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema, DropSchema

from xagent.skills.library import SkillScopeContext
from xagent.skills.personal_db import XagentPersonalDbSkillProvider
from xagent.web.models.database import Base
from xagent.web.models.skill import UserSkill, UserSkillFile
from xagent.web.models.user import User

_SKILL_MD = b"---\ndescription: writer\n---\n# Writer\n"


def test_optional_session_factory_preserves_strict_database_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import database

    monkeypatch.setattr(database, "_SessionLocal", None)

    assert database.get_optional_session_local() is None
    with pytest.raises(RuntimeError) as caught:
        database.get_session_local()
    assert (
        str(caught.value) == "Session Local is not initialized. Call init_db() first."
    )

    session_factory = object()
    monkeypatch.setattr(database, "_SessionLocal", session_factory)

    assert database.get_optional_session_local() is session_factory
    assert database.get_session_local() is session_factory


@pytest.mark.asyncio
async def test_personal_provider_skips_unavailable_database_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from xagent.web.models import database

    monkeypatch.setattr(database, "get_optional_session_local", lambda: None)

    records = await XagentPersonalDbSkillProvider().list_records(
        SkillScopeContext(user_id=7)
    )

    assert records == []
    assert "personal skill" in caplog.text.lower()


def _seed_personal_skill(session_factory) -> None:
    with session_factory() as db:
        db.execute(
            User.__table__.insert().values(
                id=7,
                username="skill-owner",
                password_hash="hash",
                is_admin=False,
            )
        )
        db.add(
            UserSkill(
                user_id=7,
                name="writer",
                origin="custom",
                skill_metadata={"nested": {"enabled": True}},
                files=[
                    UserSkillFile(
                        path="SKILL.md",
                        content=_SKILL_MD,
                        size_bytes=len(_SKILL_MD),
                        sha256="0" * 64,
                    ),
                    UserSkillFile(
                        path="examples/example.md",
                        content=b"example",
                        size_bytes=7,
                        sha256="1" * 64,
                    ),
                ],
            )
        )
        db.commit()


@pytest.mark.asyncio
async def test_personal_provider_owns_and_closes_its_database_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'skills.db'}",
        pool_size=1,
        max_overflow=0,
    )
    Base.metadata.create_all(
        engine,
        tables=[User.__table__, UserSkill.__table__, UserSkillFile.__table__],
    )
    session_factory = sessionmaker(bind=engine)
    checked_out = 0

    @event.listens_for(engine, "checkout")
    def _record_checkout(*_args) -> None:
        nonlocal checked_out
        checked_out += 1

    @event.listens_for(engine, "checkin")  # codespell:ignore checkin
    def _record_connection_return(*_args) -> None:
        nonlocal checked_out
        checked_out -= 1

    _seed_personal_skill(session_factory)

    from xagent.web.models import database

    monkeypatch.setattr(database, "get_optional_session_local", lambda: session_factory)

    records = await XagentPersonalDbSkillProvider().list_records(
        SkillScopeContext(user_id=7)
    )

    assert checked_out == 0
    assert len(records) == 1
    assert records[0].name == "writer"
    assert records[0].files == {
        "SKILL.md": _SKILL_MD,
        "examples/example.md": b"example",
    }
    assert records[0].metadata == {"nested": {"enabled": True}}

    # Detached record data remains usable after the provider-owned session
    # returned its only pool connection.
    with session_factory() as db:
        assert db.query(UserSkill).count() == 1
    assert checked_out == 0


@pytest.mark.asyncio
@pytest.mark.postgresql
async def test_personal_provider_returns_postgresql_pool_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not configured")

    schema_name = f"xagent_skill_provider_{uuid.uuid4().hex}"
    admin_engine = create_engine(database_url)
    engine = None
    schema_created = False
    try:
        with admin_engine.begin() as connection:
            connection.execute(CreateSchema(schema_name))
        schema_created = True

        engine = create_engine(
            database_url,
            connect_args={"options": f"-csearch_path={schema_name}"},
            pool_size=1,
            max_overflow=0,
        )
        try:
            Base.metadata.create_all(
                engine,
                tables=[
                    User.__table__,
                    UserSkill.__table__,
                    UserSkillFile.__table__,
                ],
            )
            session_factory = sessionmaker(bind=engine)
            _seed_personal_skill(session_factory)

            from xagent.web.models import database

            monkeypatch.setattr(
                database,
                "get_optional_session_local",
                lambda: session_factory,
            )

            records = await XagentPersonalDbSkillProvider().list_records(
                SkillScopeContext(user_id=7)
            )

            assert [record.name for record in records] == ["writer"]
            assert engine.pool.checkedout() == 0
        finally:
            if engine is not None:
                engine.dispose()
    finally:
        try:
            if schema_created:
                with admin_engine.begin() as connection:
                    connection.execute(DropSchema(schema_name, cascade=True))
        finally:
            admin_engine.dispose()


@pytest.mark.asyncio
async def test_personal_provider_skips_database_without_runtime_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import database

    def _unexpected_session_factory():
        raise AssertionError("provider must not open a session without user_id")

    monkeypatch.setattr(
        database,
        "get_optional_session_local",
        _unexpected_session_factory,
    )

    records = await XagentPersonalDbSkillProvider().list_records(SkillScopeContext())

    assert records == []


@pytest.mark.asyncio
async def test_personal_provider_drains_owned_db_work_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.models import database

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _blocking_load(_session_factory, user_id: int):
        assert user_id == 7
        started.set()
        release.wait(timeout=5)
        finished.set()
        return []

    monkeypatch.setattr(
        "xagent.skills.personal_db._load_personal_skill_records_sync",
        _blocking_load,
    )
    monkeypatch.setattr(database, "get_optional_session_local", lambda: object())

    task = asyncio.create_task(
        XagentPersonalDbSkillProvider().list_records(SkillScopeContext(user_id=7))
    )
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    await asyncio.sleep(0)
    assert not finished.is_set()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
