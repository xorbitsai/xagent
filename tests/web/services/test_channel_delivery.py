"""Final platform sends are fenced separately from Agent execution."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from tests.web.services.test_shared_channel_execution import (
    database_url as database_url,
)
from tests.web.services.test_shared_channel_execution import selected as selected
from xagent.web.models.database import get_session_local
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_channel_delivery import TaskChannelDelivery
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import channel_delivery as delivery
from xagent.web.services import shared_channel_execution as shared
from xagent.web.services.task_orchestrator import TaskTurnPayload


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


@pytest.mark.asyncio
async def test_recovery_sends_saved_result_once_to_saved_destination(
    accepted, selected
):
    complete(accepted)
    sender = AsyncMock()
    await delivery.recover_channel_results(selected.selection.channel_id, sender)
    await delivery.recover_channel_results(selected.selection.channel_id, sender)
    sender.assert_awaited_once()
    record, result = sender.await_args.args
    assert record.destination == selected.delivery_destination
    assert result["output"] == "saved answer"
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "delivered"
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.asyncio
async def test_concurrent_recovery_cannot_duplicate_active_send(accepted):
    complete(accepted)
    entered, release = asyncio.Event(), asyncio.Event()

    async def send(*args):
        entered.set()
        await release.wait()

    sender = AsyncMock(side_effect=send)
    first = asyncio.create_task(delivery.deliver_channel_result(accepted, sender))
    try:
        await entered.wait()
        await delivery.deliver_channel_result(accepted, sender)
        sender.assert_awaited_once()
    finally:
        release.set()
        await first


@pytest.mark.asyncio
async def test_platform_failure_retries_only_delivery(accepted):
    complete(accepted)
    sender = AsyncMock(side_effect=[ConnectionError("platform unavailable"), None])
    await delivery.deliver_channel_result(accepted, sender)
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_awaited_once()
    expire_claim(accepted)
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_count == 2
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "delivered"
        assert db.query(TaskExecutionCommand).count() == 1


@pytest.mark.asyncio
async def test_send_ack_commit_failure_can_repeat_send_but_never_agent(
    accepted, monkeypatch
):
    complete(accepted)
    settle = delivery._settle

    def fail_success_once(record, *, status="pending"):
        if status == "delivered":
            monkeypatch.setattr(delivery, "_settle", settle)
            raise ConnectionError("completion commit unavailable")
        settle(record, status=status)

    monkeypatch.setattr(delivery, "_settle", fail_success_once)
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender)
    expire_claim(accepted)
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_count == 2
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).count() == 1
        assert db.get(TaskChannelDelivery, accepted).status == "delivered"


@pytest.mark.asyncio
async def test_abandoned_claim_recovers_after_expiry(accepted):
    complete(accepted)
    claimed = delivery._claim(accepted)
    assert claimed is not None
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_not_awaited()
    expire_claim(accepted)
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_notice_does_not_complete_delivery(accepted):
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender, pending_notice=True)
    assert sender.await_args.args[1]["status"] == "accepted"
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "pending"
        assert db.get(TaskChannelDelivery, accepted).failure_count == 0
    complete(accepted)
    expire_claim(accepted)
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_args.args[1]["output"] == "saved answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["sender_not_authorized", "owner_changed"])
async def test_revoked_channel_cannot_receive_saved_answer(
    accepted, selected, monkeypatch, caplog, reason
):
    complete(accepted)
    with get_session_local()() as db:
        channel = db.get(UserChannel, selected.selection.channel_id)
        if reason == "sender_not_authorized":
            channel.config = {"allowed_users": ["someone-else"]}
        else:
            from xagent.web.models.user import User

            owner = User(username="replacement-owner", password_hash="unused")
            db.add(owner)
            db.flush()
            channel.user_id = owner.id
        db.commit()
    counter = Mock()
    monkeypatch.setattr(delivery, "increment_counter", counter)
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_not_awaited()
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, accepted)
        assert row.status == "discarded"
        assert row.claim_token is None
    assert f"status=discarded reason={reason}" in caplog.text
    counter.assert_called_once_with(
        "xagent.channel.delivery",
        attributes={"outcome": "discarded", "error.type": reason},
    )


@pytest.mark.asyncio
async def test_terminal_command_with_replaced_run_reports_interruption(accepted):
    with get_session_local()() as db:
        command = db.get(TaskExecutionCommand, accepted)
        command.status = "completed"
        db.get(Task, command.task_id).run_id = "replacement"
        db.commit()
    sender = AsyncMock()
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_args.args[1]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_channel_poll_retries_transient_read_failure(selected, monkeypatch):
    from sqlalchemy.exc import TimeoutError as DatabaseTimeout

    bridge = Mock(host_id="ingress")
    bridge.register_origin.return_value = "origin"
    monkeypatch.setattr(shared, "get_task_event_bridge", lambda: bridge)
    read = Mock(
        side_effect=[
            DatabaseTimeout(),
            {"success": True, "status": "completed", "output": "answer"},
        ]
    )
    monkeypatch.setattr(shared, "_read_channel_result", read)
    result = await selected.execute(TaskTurnPayload("hello"), None)
    assert result["output"] == "answer"
    assert read.call_count == 2


@pytest.mark.asyncio
async def test_channel_pending_wait_is_bounded_and_keeps_command(selected, monkeypatch):
    bridge = Mock(host_id="ingress")
    bridge.register_origin.return_value = "origin"
    monkeypatch.setattr(shared, "get_task_event_bridge", lambda: bridge)
    monkeypatch.setattr(shared, "get_task_reply_wait_timeout_seconds", lambda: 0.03)
    # Acceptance includes real database I/O before the short reply-wait deadline.
    result = await asyncio.wait_for(
        selected.execute(TaskTurnPayload("hello"), None), 10
    )
    assert result["status"] == "accepted"
    with get_session_local()() as db:
        assert db.query(TaskExecutionCommand).one().status == "pending"


@pytest.mark.asyncio
async def test_preaccept_stop_preserves_resumable_task_and_files(selected, tmp_path):
    from xagent.web.models.uploaded_file import UploadedFile

    path = tmp_path / "input.txt"
    path.write_text("input")
    with get_session_local()() as db:
        db.add(
            UploadedFile(
                task_id=selected.selection.task_id,
                user_id=selected.selection.user_id,
                filename=path.name,
                storage_path=str(path),
            )
        )
        db.commit()
    selected.origin = None
    await selected.stop()
    await selected.stop()
    await selected.close()
    with get_session_local()() as db:
        task = db.get(Task, selected.selection.task_id)
        assert task.status == TaskStatus.PAUSED
        assert task.control_state == "paused"
        assert task.run_id is not None
        assert task.state_version == selected.selection.state_version + 1
        assert db.query(UploadedFile).one().task_id == task.id
    assert path.read_text() == "input"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [True, False])
async def test_preaccept_cleanup_cannot_modify_replacement(selected, stop):
    with get_session_local()() as db:
        task = db.get(Task, selected.selection.task_id)
        task.run_id = "new-run"
        task.state_version += 1
        db.commit()
    selected.origin = None
    if stop:
        await selected.stop()
    await selected.close()
    with get_session_local()() as db:
        task = db.get(Task, selected.selection.task_id)
        assert task.run_id == "new-run"
        assert task.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_new_after_pending_stop_discards_the_old_delivery(accepted, selected):
    from tests.web.test_telegram_task_commands import _bot

    selected.accepted = True
    selected.command_db_id = accepted
    bot = _bot(selected.selection.channel_id)
    bot.user_active_executions[123] = (selected.selection.task_id, selected)
    assert bot._stop_current_conversation(123)
    await selected.stop_task
    assert not selected.discard_output
    assert bot._start_new_conversation(123) == (True, True)
    await selected.stop_task
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "discarded"
        assert db.query(TaskExecutionCommand).filter_by(kind="pause").count() == 1


@pytest.mark.asyncio
async def test_lost_delivery_claim_cancels_inflight_sender(accepted, monkeypatch):
    complete(accepted)
    monkeypatch.setattr(delivery, "_DELIVERY_LEASE_SECONDS", 0.03)
    monkeypatch.setattr(delivery, "_renew", lambda record: False)
    cancelled = asyncio.Event()

    async def send(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    assert not await asyncio.wait_for(
        delivery.deliver_channel_result(accepted, send), 1
    )
    assert cancelled.is_set()
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).status == "pending"


@pytest.mark.asyncio
async def test_failed_destinations_do_not_starve_newer_replies(
    accepted, selected, monkeypatch
):
    complete(accepted)
    ids = [accepted]
    with get_session_local()() as db:
        first = db.get(TaskExecutionCommand, accepted)
        for i in range(10):
            command = TaskExecutionCommand(
                task_id=first.task_id,
                actor_user_id=first.actor_user_id,
                command_id=f"retry-fairness-{i}",
                kind=first.kind,
                payload=first.payload,
                target_run_id=first.target_run_id,
                status="completed",
                result=first.result,
            )
            db.add(command)
            db.flush()
            ids.append(command.id)
            db.add(
                TaskChannelDelivery(
                    command_id=command.id,
                    channel_id=selected.selection.channel_id,
                    destination={"chat_id": str(i)},
                )
            )
        db.commit()

    class Clock(datetime):
        tick = datetime.now(timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.tick

    monkeypatch.setattr(delivery, "datetime", Clock)
    sent = []

    async def sender(record, result):
        sent.append(record.command_id)
        if record.command_id != ids[-1]:
            raise ConnectionError("destination permanently refuses the reply")

    await delivery.recover_channel_results(selected.selection.channel_id, sender)
    assert ids[-1] not in sent
    Clock.tick += timedelta(seconds=31)
    await delivery.recover_channel_results(selected.selection.channel_id, sender)
    # Even though the ten failures are eligible again, the fresh reply wins.
    assert sent[10] == ids[-1]
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, ids[-1]).status == "delivered"
        assert db.query(TaskExecutionCommand).count() == 11


@pytest.mark.asyncio
async def test_permanent_send_failure_exhausts_budget(accepted, monkeypatch, caplog):
    complete(accepted)
    sender = AsyncMock(side_effect=ConnectionError("permanent refusal"))
    counter = Mock()
    monkeypatch.setattr(delivery, "increment_counter", counter)
    for attempt in range(10):
        expire_claim(accepted)
        await delivery.deliver_channel_result(accepted, sender)
        with get_session_local()() as db:
            row = db.get(TaskChannelDelivery, accepted)
            assert row.failure_count == attempt + 1
            assert row.status == ("failed" if attempt == 9 else "pending")
    await delivery.deliver_channel_result(accepted, sender)
    assert sender.await_count == 10
    assert "status=failed reason=delivery_error" in caplog.text
    counter.assert_any_call(
        "xagent.channel.delivery",
        attributes={"outcome": "failed", "error.type": "delivery_error"},
    )
    with get_session_local()() as db:
        assert db.get(TaskChannelDelivery, accepted).available_at is None
        assert (
            db.get(TaskExecutionCommand, accepted).result["channel_result"]["output"]
            == "saved answer"
        )
        assert db.query(TaskExecutionCommand).count() == 1


def test_replaced_delivery_claim_cannot_consume_retry_budget(accepted):
    complete(accepted)
    original, _ = delivery._claim(accepted)
    expire_claim(accepted)
    replacement, _ = delivery._claim(accepted)
    delivery._settle(original, failed=True)
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, accepted)
        assert row.claim_token == replacement.claim_token
        assert row.failure_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [True, False])
async def test_unavailable_channel_retries_with_budget_and_can_recover(
    accepted, selected, repair, monkeypatch, caplog
):
    complete(accepted)
    with get_session_local()() as db:
        db.get(UserChannel, selected.selection.channel_id).is_active = False
        db.commit()
    sender = AsyncMock()
    counter = Mock()
    monkeypatch.setattr(delivery, "increment_counter", counter)
    await delivery.deliver_channel_result(accepted, sender)
    await delivery.deliver_channel_result(accepted, sender)
    sender.assert_not_awaited()
    with get_session_local()() as db:
        row = db.get(TaskChannelDelivery, accepted)
        assert row.status == "pending"
        assert row.failure_count == 1
        assert row.available_at is not None
        assert row.claim_token is None
    assert "reason=configuration_unavailable" in caplog.text
    if repair:
        with get_session_local()() as db:
            db.get(UserChannel, selected.selection.channel_id).is_active = True
            db.commit()
        expire_claim(accepted)
        await delivery.recover_channel_results(selected.selection.channel_id, sender)
        sender.assert_awaited_once()
        with get_session_local()() as db:
            assert db.get(TaskChannelDelivery, accepted).status == "delivered"
    else:
        for _ in range(9):
            expire_claim(accepted)
            await delivery.recover_channel_results(
                selected.selection.channel_id, sender
            )
        with get_session_local()() as db:
            assert db.get(TaskChannelDelivery, accepted).status == "failed"
        sender.assert_not_awaited()
        counter.assert_any_call(
            "xagent.channel.delivery",
            attributes={"outcome": "failed", "error.type": "configuration_unavailable"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["file_unavailable", "owner_changed"])
async def test_preaccept_failure_preserves_resumable_task_and_files(
    selected, tmp_path, monkeypatch, failure
):
    from sqlalchemy import text

    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.models.user import User
    from xagent.web.services.task_orchestrator import TaskTurnError

    path = tmp_path / "input.txt"
    path.write_text("input")
    with get_session_local()() as db:
        if db.get_bind().dialect.name == "sqlite":
            db.execute(text("PRAGMA foreign_keys=ON"))
        db.add(
            UploadedFile(
                task_id=selected.selection.task_id,
                user_id=selected.selection.user_id,
                filename=path.name,
                storage_path=str(path),
            )
        )
        if failure == "owner_changed":
            replacement = User(username="replacement", password_hash="unused")
            db.add(replacement)
            db.flush()
            db.get(UserChannel, selected.selection.channel_id).user_id = replacement.id
        db.commit()
    bridge = Mock(host_id="ingress")
    bridge.register_origin.return_value = "origin"
    monkeypatch.setattr(shared, "get_task_event_bridge", lambda: bridge)
    payload = TaskTurnPayload(
        "hello", file_ids=("missing",) if failure == "file_unavailable" else ()
    )
    with pytest.raises(TaskTurnError, match=failure):
        await selected.execute(payload, None)
    await selected.close()
    await selected.close()
    with get_session_local()() as db:
        task = db.get(Task, selected.selection.task_id)
        assert task.status == TaskStatus.PAUSED
        assert task.control_state == "paused"
        assert task.state_version == selected.selection.state_version + 1
        assert db.query(UploadedFile).one().task_id == task.id
        assert db.query(TaskExecutionCommand).count() == 0
    assert path.read_text() == "input"
