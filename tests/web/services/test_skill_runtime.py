from collections.abc import Mapping
from types import SimpleNamespace

import pytest
from sqlalchemy import Column, Integer, create_engine, select
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.skills.library import SkillScopeContext
from xagent.web.services.skill_runtime import (
    SkillRuntimeSessionBoundaryError,
    build_runtime_skill_scope,
    get_skill_runtime_scope,
)

Base = declarative_base()


class _Item(Base):
    __tablename__ = "skill_runtime_items"

    id = Column(Integer, primary_key=True)


def _one_slot_session() -> tuple[Session, QueuePool]:
    engine = create_engine(
        "sqlite://",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    return db, engine.pool


def test_build_runtime_skill_scope_releases_clean_caller_connection() -> None:
    db, pool = _one_slot_session()
    try:
        db.scalars(select(_Item)).all()
        assert pool.checkedout() == 1

        context = build_runtime_skill_scope(
            user_id=7,
            metadata={"team_id": 11},
            caller_db=db,
        )

        assert context == SkillScopeContext(
            user_id=7,
            metadata={"team_id": 11},
        )
        assert pool.checkedout() == 0
    finally:
        db.close()


def test_build_runtime_skill_scope_fails_closed_on_pending_write() -> None:
    db, pool = _one_slot_session()
    try:
        pending = _Item()
        db.add(pending)

        with pytest.raises(
            SkillRuntimeSessionBoundaryError,
            match="pending writes",
        ):
            build_runtime_skill_scope(
                user_id=7,
                caller_db=db,
            )

        assert pending in db.new
        assert pool.checkedout() == 0
    finally:
        db.rollback()
        db.close()


def test_skill_runtime_dependency_detaches_identity_before_route_execution() -> None:
    db, pool = _one_slot_session()
    try:
        db.scalars(select(_Item)).all()
        user = SimpleNamespace(id=7, _saas_team_id=11)
        assert pool.checkedout() == 1

        context = get_skill_runtime_scope(current_user=user, db=db)

        assert context == SkillScopeContext(user_id=7)
        assert pool.checkedout() == 0
    finally:
        db.close()


def test_http_and_runtime_scope_builders_are_symmetric_and_ignore_saas_state() -> None:
    from xagent.web.services.skill_runtime import build_detached_skill_scope

    db, pool = _one_slot_session()
    try:
        db.scalars(select(_Item)).all()
        user = SimpleNamespace(id=7, _saas_team_id=11)

        http_context = get_skill_runtime_scope(current_user=user, db=db)
        runtime_context = build_detached_skill_scope(user_id=7)

        assert http_context == runtime_context == SkillScopeContext(user_id=7)
        assert pool.checkedout() == 0
        assert not hasattr(http_context, "user")
        assert not hasattr(http_context, "db")
        assert not hasattr(http_context, "request")
    finally:
        db.close()


def test_detached_scope_builder_preserves_falsey_nonempty_metadata() -> None:
    from xagent.web.services.skill_runtime import build_detached_skill_scope

    class _FalseyNonemptyMetadata(Mapping[str, int]):
        def __init__(self) -> None:
            self.values = {"team_id": 11}

        def __getitem__(self, key: str) -> int:
            return self.values[key]

        def __iter__(self):
            return iter(self.values)

        def __len__(self) -> int:
            return 0

    metadata = _FalseyNonemptyMetadata()
    context = build_detached_skill_scope(user_id=7, metadata=metadata)
    metadata.values["team_id"] = 12

    assert context == SkillScopeContext(user_id=7, metadata={"team_id": 11})


@pytest.mark.asyncio
async def test_skill_runtime_boundary_error_handler_is_registered_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from xagent.web.app import app
    from xagent.web.services.skill_runtime import logger as skill_runtime_logger
    from xagent.web.services.skill_runtime import (
        skill_runtime_session_boundary_error_handler,
    )

    async def _raise_boundary_error() -> None:
        raise SkillRuntimeSessionBoundaryError("boundary-secret-sentinel")

    assert app.exception_handlers[SkillRuntimeSessionBoundaryError] is (
        skill_runtime_session_boundary_error_handler
    )
    app.add_api_route("/_test/skill-runtime-boundary", _raise_boundary_error)
    logged: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        skill_runtime_logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/_test/skill-runtime-boundary")

    assert response.status_code == 500
    assert response.json() == {"detail": "Skill runtime is temporarily unavailable."}
    assert "boundary-secret-sentinel" not in response.text
    assert logged[0][0] == (
        "Skill runtime session boundary failed for %s",
        "/_test/skill-runtime-boundary",
    )
