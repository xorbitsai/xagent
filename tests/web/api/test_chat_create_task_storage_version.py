"""``create_task`` picks the storage entry: Builder preview vs web chat."""

from __future__ import annotations

import pytest

from xagent.web.api.chat import create_task
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.schemas.chat import TaskCreateRequest

from .conftest import _admin_headers, _direct_db_session, _register_second_user

pytestmark = pytest.mark.usefixtures("_test_db")

_ENV = "XAGENT_EXECUTION_EVENT_STORAGE_ENTRIES"


async def _create(request: TaskCreateRequest) -> int:
    _admin_headers()
    _register_second_user("storage-owner", "storagepass1")
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == "storage-owner").one()
        created = await create_task(request, db=db, user=owner)
        db.expire_all()
        return int(
            db.query(Task)
            .filter(Task.id == int(created.task_id))
            .one()
            .conversation_storage_version
        )
    finally:
        db.close()


def _chat_request() -> TaskCreateRequest:
    return TaskCreateRequest(title="chat", description="chat", agent_id=None)


def _ws_preview_request() -> TaskCreateRequest:
    # The shape websocket.py builds: is_preview rides in agent_config.
    return TaskCreateRequest(
        title="preview",
        description="preview",
        agent_id=None,
        agent_config={"instructions": "p", "is_preview": True},
        is_visible=False,
    )


def _rest_preview_request() -> TaskCreateRequest:
    return TaskCreateRequest(
        title="preview", description="preview", agent_id=None, is_preview=True
    )


@pytest.mark.asyncio
async def test_default_off_is_version_1_for_chat_and_preview(monkeypatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert await _create(_chat_request()) == 1
    assert await _create(_ws_preview_request()) == 1


@pytest.mark.asyncio
async def test_web_chat_entry_enables_chat_but_not_preview(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, "web_chat")
    assert await _create(_chat_request()) == 2
    assert await _create(_ws_preview_request()) == 1
    assert await _create(_rest_preview_request()) == 1


@pytest.mark.asyncio
async def test_agent_preview_entry_enables_both_preview_paths_only(monkeypatch) -> None:
    monkeypatch.setenv(_ENV, "agent_preview")
    assert await _create(_chat_request()) == 1
    assert await _create(_ws_preview_request()) == 2
    assert await _create(_rest_preview_request()) == 2
