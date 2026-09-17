from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import Column, Integer, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from xagent.skills.library import SkillScopeContext, SkillWriteContext
from xagent.web.api.skill_hub import _get_scoped_manager, _write_context
from xagent.web.api.skills import _request_skill_manager
from xagent.web.services.skill_runtime import SkillRuntimeSessionBoundaryError

Base = declarative_base()


class _PendingCallerState(Base):
    __tablename__ = "skill_write_pending_caller_state"

    id = Column(Integer, primary_key=True)


def test_skill_hub_write_context_reuses_only_detached_scope_identity() -> None:
    scope = SkillScopeContext(user_id=7, metadata={"team_id": 11})

    context = _write_context(scope)

    assert context == SkillWriteContext(user_id=7, metadata={"team_id": 11})
    assert not hasattr(context, "user")
    assert not hasattr(context, "db")
    assert not hasattr(context, "request")


@pytest.mark.asyncio
async def test_skills_api_manager_hands_off_caller_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    db = object()
    scope = SkillScopeContext(user_id=7)

    class _Manager:
        async def ensure_initialized(self) -> None:
            captured["initialized"] = True

    def _create_skill_manager(*, context):
        captured["context"] = context
        return _Manager()

    def _unexpected_session_factory():
        raise AssertionError("skills API must not own a read session")

    def _handoff(caller_db):
        captured["caller_db"] = caller_db

    monkeypatch.setattr(
        "xagent.skills.utils.create_skill_manager",
        _create_skill_manager,
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        _unexpected_session_factory,
    )
    monkeypatch.setattr(
        "xagent.web.api.skills.handoff_skill_runtime_session",
        _handoff,
    )

    manager = await _request_skill_manager(
        scope,
        db,
    )

    assert isinstance(manager, _Manager)
    assert captured == {
        "caller_db": db,
        "context": scope,
        "initialized": True,
    }


@pytest.mark.asyncio
async def test_skill_hub_manager_fails_before_provider_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = SkillScopeContext(user_id=7)
    db = object()

    def _reject_handoff(caller_db):
        assert caller_db is db
        raise SkillRuntimeSessionBoundaryError("pending writes")

    monkeypatch.setattr(
        "xagent.web.api.skill_hub.handoff_skill_runtime_session",
        _reject_handoff,
    )

    with pytest.raises(SkillRuntimeSessionBoundaryError, match="pending writes"):
        await _get_scoped_manager(SimpleNamespace(), scope, db)


@pytest.mark.asyncio
async def test_write_provider_invoker_uses_detached_context_and_leaves_caller_pending_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from xagent.web.services.skill_runtime import invoke_skill_write_provider

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    caller_db: Session = sessionmaker(bind=engine)()
    pending = _PendingCallerState()
    caller_db.add(pending)
    captured: dict[str, Any] = {}

    class _Writer:
        async def create_skill(self, context, **kwargs) -> None:
            captured["context"] = context
            captured["kwargs"] = kwargs

    try:
        await invoke_skill_write_provider(
            _Writer(),
            "create_skill",
            SkillWriteContext(user_id=7, metadata={"source_id": 11}),
            scope="team",
            name="writer",
            files={"SKILL.md": b"# writer"},
        )

        assert {
            field.name for field in captured["context"].__dataclass_fields__.values()
        } == {
            "user_id",
            "metadata",
        }
        assert pending in caller_db.new
        assert caplog.text == ""
    finally:
        caller_db.rollback()
        caller_db.close()


@pytest.mark.asyncio
async def test_write_provider_invoker_maps_only_typed_public_failures_and_redacts_unknowns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from xagent.skills.library import (
        SkillWriteProviderError,
        SkillWriteProviderErrorReason,
    )
    from xagent.web.services.skill_runtime import invoke_skill_write_provider

    class _TypedFailureWriter:
        async def delete_skill(self, context, **kwargs) -> None:
            raise SkillWriteProviderError(
                SkillWriteProviderErrorReason.FORBIDDEN,
                "You cannot modify this team skill.",
            )

    class _UnexpectedFailureWriter:
        async def delete_skill(self, context, **kwargs) -> None:
            raise RuntimeError("provider-secret-sentinel")

    class _InvalidTypedFailureWriter:
        async def delete_skill(self, context, **kwargs) -> None:
            raise SkillWriteProviderError(  # type: ignore[arg-type]
                "not-allowlisted",
                "invalid-reason-secret-sentinel",
            )

    class _UnhashableReasonWriter:
        def __init__(self, reason: object) -> None:
            self.reason = reason

        async def delete_skill(self, context, **kwargs) -> None:
            raise SkillWriteProviderError(  # type: ignore[arg-type]
                self.reason,
                "unhashable-reason-secret-sentinel",
            )

    class _AttachedPublicDetailWriter:
        async def delete_skill(self, context, **kwargs) -> None:
            class _AttachedStr(str):
                pass

            detail = _AttachedStr("attached-detail-secret-sentinel")
            detail.request = object()
            raise SkillWriteProviderError(
                SkillWriteProviderErrorReason.FORBIDDEN,
                detail,  # type: ignore[arg-type]
            )

    class _HostileProviderError(SkillWriteProviderError):
        def __getattribute__(self, name: str):
            if name == "reason":
                raise RuntimeError("hostile-reason-secret-sentinel")
            return super().__getattribute__(name)

    class _HostileProviderErrorWriter:
        async def delete_skill(self, context, **kwargs) -> None:
            raise _HostileProviderError(
                SkillWriteProviderErrorReason.FORBIDDEN,
                "hostile-detail-secret-sentinel",
            )

    context = SkillWriteContext(user_id=7)
    with pytest.raises(HTTPException) as typed:
        await invoke_skill_write_provider(
            _TypedFailureWriter(), "delete_skill", context, scope="team", name="writer"
        )
    assert typed.value.status_code == 403
    assert typed.value.detail == "You cannot modify this team skill."

    with pytest.raises(HTTPException) as unknown:
        await invoke_skill_write_provider(
            _UnexpectedFailureWriter(),
            "delete_skill",
            context,
            scope="team",
            name="writer",
        )
    assert unknown.value.status_code == 500
    assert unknown.value.detail == "Skill provider operation failed."
    assert "provider-secret-sentinel" not in str(unknown.value.detail)
    assert "provider-secret-sentinel" in caplog.text

    with pytest.raises(HTTPException) as invalid_typed:
        await invoke_skill_write_provider(
            _InvalidTypedFailureWriter(),
            "delete_skill",
            context,
            scope="team",
            name="writer",
        )
    assert invalid_typed.value.status_code == 500
    assert invalid_typed.value.detail == "Skill provider operation failed."
    assert "invalid-reason-secret-sentinel" not in str(invalid_typed.value.detail)

    with pytest.raises(HTTPException) as attached_detail:
        await invoke_skill_write_provider(
            _AttachedPublicDetailWriter(),
            "delete_skill",
            context,
            scope="team",
            name="writer",
        )
    assert attached_detail.value.status_code == 500
    assert attached_detail.value.detail == "Skill provider operation failed."
    assert "attached-detail-secret-sentinel" not in str(attached_detail.value.detail)

    app = FastAPI()

    @app.get("/write")
    async def _hostile_write_route() -> None:
        await invoke_skill_write_provider(
            _HostileProviderErrorWriter(),
            "delete_skill",
            context,
            scope="team",
            name="writer",
        )

    response = TestClient(app, raise_server_exceptions=False).get("/write")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Skill provider operation failed."}
    assert "hostile-reason-secret-sentinel" not in response.text
    assert "hostile-detail-secret-sentinel" not in response.text

    for unhashable_reason in ([], {}):
        with pytest.raises(HTTPException) as unhashable:
            await invoke_skill_write_provider(
                _UnhashableReasonWriter(unhashable_reason),
                "delete_skill",
                context,
                scope="team",
                name="writer",
            )
        assert unhashable.value.status_code == 500
        assert unhashable.value.detail == "Skill provider operation failed."
