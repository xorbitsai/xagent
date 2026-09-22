"""Shared channel database fixtures and delivery state helpers."""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services.channel_runtime import _prepare_channel_task_sync
from xagent.web.services.task_orchestrator import TaskTurnPayload


@pytest.fixture(
    params=["sqlite", pytest.param("postgresql", marks=pytest.mark.postgresql)]
)
def database_url(request, tmp_path):
    if request.param == "postgresql":
        with disposable_database_factory("shared_channel") as make:
            engine = make("worker")
            with engine.connect() as connection:
                name = connection.execute(
                    text("SELECT current_database()")
                ).scalar_one()
            yield (
                make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
                .set(database=name)
                .render_as_string(hide_password=False)
            )
    else:
        yield f"sqlite:///{tmp_path / 'channel.db'}"


@pytest.fixture
def selected(database_url, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    init_db(db_url=database_url)
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        channel = UserChannel(
            user_id=user.id,
            channel_type="feishu",
            channel_name="test",
            config={"allowed_users": ["sender"]},
            is_active=True,
        )
        db.add(channel)
        db.commit()
        channel_id = channel.id
    selection = _prepare_channel_task_sync(
        channel_id=channel_id,
        external_user_id="sender",
        active_task_id=None,
        text="hello",
        channel_name="test",
        expected_owner_user_id=None,
        defer_execution=True,
    )
    turn = shared.SharedChannelTurn(selection, workspace=SimpleNamespace())
    turn.origin = "origin-token"
    yield turn
    Base.metadata.drop_all(bind=get_engine())
    get_engine().dispose()


@pytest.fixture
def accepted(selected):
    selected.delivery_destination = {
        "chat_id": "conversation",
        "loading_message_id": "loading",
    }
    command_id = shared._accept_channel_turn(
        selected, TaskTurnPayload("hello"), "old-ingress"
    )
    return command_id


def complete(command_id):
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, command_id)
        command.status = "completed"
        command.result = {
            "channel_result": {
                "success": True,
                "status": "completed",
                "output": "saved answer",
            }
        }
        task = db.get(Task, command.task_id)
        task.run_id = command.target_run_id
        task.status = TaskStatus.COMPLETED
        db.commit()


def expire_claim(command_id):
    with get_session_local()() as db:
        db.get(TaskChannelDelivery, command_id).available_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
        db.commit()
