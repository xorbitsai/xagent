"""Live input commands: ledger admission and the in-memory command executor."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from xagent.core.agent.live_input import (
    LiveInputClaim,
    LiveInputOffer,
    LiveInputResult,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services import live_input_command as lic
from xagent.web.services import task_command_execution
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_PENDING,
)
from xagent.web.services.task_command_execution import _execute_durable_task_command
from xagent.web.services.task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    stage_task_command,
)

TURN = "turn-1"
RUN = "run-1"


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Any]:
    init_db(db_url=f"sqlite:///{tmp_path / 'live-input.db'}")
    session = next(get_db())
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture
def task(db: Any) -> Any:
    user = User(username="owner", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    row = Task(
        user_id=user.id,
        title="t",
        description="t",
        status=TaskStatus.RUNNING,
        execution_mode="auto",
        run_id=RUN,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


class FakeHost:
    def __init__(self, owner: int | None, error: BaseException | None = None) -> None:
        self.owner = owner
        self.error = error
        self.authorized: list[lic.LiveInputPayload] = []
        self.applied: list[lic.LiveInputPayload] = []

    def authorize(self, db: Any, payload: lic.LiveInputPayload) -> int | None:
        self.authorized.append(payload)
        if self.error is not None:
            raise self.error
        return self.owner

    def record_applied(self, payload: lic.LiveInputPayload) -> None:
        self.applied.append(payload)


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []

    async def publish(frame: dict[str, Any], task_id: int) -> None:
        frames.append(frame)

    monkeypatch.setattr(lic, "publish_task_event", publish)
    return frames


async def settle() -> None:
    await asyncio.gather(*list(lic._background))


def payload_for(task: Any, turn_id: str = TURN, message: str = " raw text ") -> Any:
    return lic.LiveInputPayload(task_id=task.id, turn_id=turn_id, message=message)


def rows(db: Any, task: Any) -> list[TaskChatMessage]:
    db.expire_all()
    return list(db.query(TaskChatMessage).filter_by(task_id=task.id).all())


def command(task: Any, **overrides: Any) -> ClaimedTaskCommand:
    payload = {
        "scope": lic.LIVE_INPUT_COMMAND_SCOPE,
        "message": "hello",
        "turn_id": TURN,
    }
    payload.update(overrides.pop("payload", {}))
    return ClaimedTaskCommand(
        id=1,
        task_id=task.id,
        actor_user_id=None,
        command_id=lic.live_input_command_id(RUN, TURN, 0),
        kind=TaskCommandKind.MESSAGE,
        payload=payload,
        target_run_id=overrides.pop("target_run_id", RUN),
        attempt_count=1,
    )


async def test_claim_uses_raw_content_and_no_attachments(
    monkeypatch: pytest.MonkeyPatch, task: Any, published: list
) -> None:
    seen: dict[str, Any] = {}
    real = lic.claim_user_message_delivery_no_commit

    def spy(db: Any, task_id: int, user_id: int, content: str, **kwargs: Any) -> Any:
        seen.update(user_id=user_id, content=content, **kwargs)
        return real(db, task_id, user_id, content, **kwargs)

    monkeypatch.setattr(lic, "claim_user_message_delivery_no_commit", spy)
    admission = lic.DeliveryLedgerAdmission(payload_for(task), FakeHost(task.user_id))
    assert await admission.claim() is LiveInputClaim.CLAIMED
    assert seen["attachments"] is None
    assert seen["content"] == " raw text "
    assert seen["user_id"] == task.user_id


@pytest.mark.parametrize(
    "owner,error,reason",
    [
        (None, None, "unavailable"),
        (None, None, "unavailable"),
        (1, OperationalError("select", {}, Exception("down")), "run_ended"),
    ],
    ids=["quota-exhausted", "authorization-revoked", "database-error"],
)
async def test_live_input_reauthorized_at_claim(
    db: Any,
    task: Any,
    published: list,
    owner: int | None,
    error: BaseException | None,
    reason: str,
) -> None:
    admission = lic.DeliveryLedgerAdmission(payload_for(task), FakeHost(owner, error))
    assert await admission.claim() is LiveInputClaim.UNAVAILABLE
    await admission.finish(LiveInputResult.RETURNED)
    await settle()
    assert rows(db, task) == []
    assert [f["reason"] for f in published] == [reason]
    assert published[0]["type"] == "live_input_returned"


async def test_authorize_called_per_claim(task: Any, published: list) -> None:
    host = FakeHost(task.user_id)
    for index in range(3):
        admission = lic.DeliveryLedgerAdmission(
            payload_for(task, turn_id=f"turn-{index}"), host
        )
        assert await admission.claim() is LiveInputClaim.CLAIMED
    assert len(host.authorized) == 3


async def test_applied_marks_dispatched_and_meters(
    db: Any, task: Any, published: list
) -> None:
    host = FakeHost(task.user_id)
    admission = lic.DeliveryLedgerAdmission(payload_for(task), host)
    await admission.claim()
    assert [r.delivery_status for r in rows(db, task)] == [DELIVERY_PENDING]
    await admission.finish(LiveInputResult.APPLIED)
    assert [r.delivery_status for r in rows(db, task)] == [DELIVERY_DISPATCHED]
    assert len(host.applied) == 1 and published == []


async def test_returned_after_claim_withdraws_row_and_reports_not_applied(
    db: Any, task: Any, published: list
) -> None:
    admission = lic.DeliveryLedgerAdmission(payload_for(task), FakeHost(task.user_id))
    await admission.claim()
    await admission.finish(LiveInputResult.RETURNED)
    await settle()
    assert rows(db, task) == []
    assert [f["reason"] for f in published] == ["not_applied"]


async def test_failed_claim_commit_is_withdrawn_and_reports_not_applied(
    monkeypatch: pytest.MonkeyPatch, db: Any, task: Any, published: list
) -> None:
    real = lic.claim_user_message_delivery_no_commit

    def claim_then_fail(db_: Any, *args: Any, **kwargs: Any) -> Any:
        claim = real(db_, *args, **kwargs)
        db_.commit = lambda: (_ for _ in ()).throw(RuntimeError("commit lost"))
        return claim

    monkeypatch.setattr(lic, "claim_user_message_delivery_no_commit", claim_then_fail)
    admission = lic.DeliveryLedgerAdmission(payload_for(task), FakeHost(task.user_id))
    assert await admission.claim() is LiveInputClaim.UNAVAILABLE
    await admission.finish(LiveInputResult.RETURNED)
    await settle()
    assert rows(db, task) == []
    assert [f["reason"] for f in published] == ["not_applied"]


async def test_live_input_and_new_turn_share_one_delivery_row(
    db: Any, task: Any, published: list
) -> None:
    host = FakeHost(task.user_id)
    first = lic.DeliveryLedgerAdmission(payload_for(task), host)
    second = lic.DeliveryLedgerAdmission(payload_for(task), host)
    assert await first.claim() is LiveInputClaim.CLAIMED
    assert await second.claim() is LiveInputClaim.TAKEN
    assert len(rows(db, task)) == 1


async def test_racing_insert_is_reported_as_taken(
    monkeypatch: pytest.MonkeyPatch, db: Any, task: Any, published: list
) -> None:
    winner = lic.DeliveryLedgerAdmission(payload_for(task), FakeHost(task.user_id))
    assert await winner.claim() is LiveInputClaim.CLAIMED

    def lose_race(*args: Any, **kwargs: Any) -> Any:
        raise IntegrityError("insert", {}, Exception("unique"))

    monkeypatch.setattr(lic, "claim_user_message_delivery_no_commit", lose_race)
    loser = lic.DeliveryLedgerAdmission(payload_for(task), FakeHost(task.user_id))
    assert await loser.claim() is LiveInputClaim.TAKEN
    assert len(rows(db, task)) == 1


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> Any:
    offers: list[dict[str, Any]] = []

    def offer(eid: str, **kwargs: Any) -> LiveInputOffer:
        offers.append({"execution_id": eid, **kwargs})
        return LiveInputOffer.QUEUED

    service = SimpleNamespace(offer_live_user_message=offer)
    fake = SimpleNamespace(
        running_agent_for_task=lambda task_id: service,
        _agent_run_ids={},
        offers=offers,
        service=service,
    )
    monkeypatch.setattr(lic, "get_agent_manager", lambda: fake)
    return fake


@pytest.fixture
def running_tasks(monkeypatch: pytest.MonkeyPatch) -> dict[int, Any]:
    tasks: dict[int, Any] = {}
    monkeypatch.setattr(
        lic.task_execution_service,
        "background_task_manager",
        SimpleNamespace(running_tasks=tasks),
    )
    return tasks


@pytest.fixture
def host() -> Iterator[FakeHost]:
    registered = FakeHost(1)
    lic.register_live_input_host(registered)
    yield registered
    lic.unregister_live_input_host()


async def test_live_input_executor_touches_no_database(
    monkeypatch: pytest.MonkeyPatch,
    task: Any,
    manager: Any,
    running_tasks: dict,
    host: FakeHost,
    published: list,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the executor must not touch the database")

    monkeypatch.setattr(lic, "run_db_io_cancellation_safe", forbidden)
    monkeypatch.setattr("xagent.web.models.database.get_session_local", forbidden)
    manager._agent_run_ids[task.id] = RUN
    result = await _execute_durable_task_command(command(task))
    assert result == {"result": "queued"}
    assert host.authorized == [] and host.applied == []
    assert manager.offers[0]["execution_id"] == str(task.id)
    assert manager.offers[0]["turn_id"] == TURN


async def test_cancel_claimable_while_publish_blocks(
    monkeypatch: pytest.MonkeyPatch,
    task: Any,
    manager: Any,
    running_tasks: dict,
    host: FakeHost,
) -> None:
    async def slow_publish(frame: dict[str, Any], task_id: int) -> None:
        await asyncio.sleep(5)

    monkeypatch.setattr(lic, "publish_task_event", slow_publish)
    started = time.monotonic()
    result = await lic.execute_live_input_command(command(task, target_run_id="other"))
    assert result == {"result": "run_ended"}
    assert time.monotonic() - started < 0.5
    for pending in list(lic._background):
        pending.cancel()


async def test_executor_reports_not_started_before_control_registers(
    task: Any,
    manager: Any,
    running_tasks: dict,
    host: FakeHost,
    published: list,
) -> None:
    manager.running_agent_for_task = lambda task_id: None
    running_tasks[task.id] = object()
    result = await lic.execute_live_input_command(command(task))
    await settle()
    assert result == {"result": "not_started"}
    assert [f["reason"] for f in published] == ["not_started"]


async def test_executor_run_mismatch_broadcasts_run_ended(
    task: Any,
    manager: Any,
    running_tasks: dict,
    host: FakeHost,
    published: list,
) -> None:
    manager._agent_run_ids[task.id] = "another-run"
    result = await lic.execute_live_input_command(command(task))
    await settle()
    assert result == {"result": "run_ended"}
    assert [f["reason"] for f in published] == ["run_ended"]
    assert manager.offers == []


def test_host_registration_is_single_and_idempotent() -> None:
    first, other = FakeHost(1), FakeHost(2)
    lic.register_live_input_host(first)
    try:
        lic.register_live_input_host(first)
        with pytest.raises(ValueError):
            lic.register_live_input_host(other)
        assert lic.registered_live_input_host() is first
    finally:
        lic.unregister_live_input_host()
    assert lic.registered_live_input_host() is None


def test_live_input_command_id_is_stable_and_valid() -> None:
    first = lic.live_input_command_id("run", "turn", 0)
    assert first == lic.live_input_command_id("run", "turn", 0)
    assert first != lic.live_input_command_id("run", "turn", 1)
    assert first.startswith("live:") and len(first) == 53


@pytest.mark.parametrize("reserved_live", [False, True])
def test_live_prefix_reserved(db: Any, task: Any, reserved_live: bool) -> None:
    def stage() -> Any:
        return stage_task_command(
            db,
            task_id=task.id,
            actor_user_id=None,
            command_id=lic.live_input_command_id(RUN, TURN, 0),
            kind=TaskCommandKind.MESSAGE,
            payload={"scope": lic.LIVE_INPUT_COMMAND_SCOPE},
            reserved_live=reserved_live,
        )

    if reserved_live:
        assert stage().created
    else:
        with pytest.raises(ValueError, match="reserved"):
            stage()


@pytest.mark.parametrize(
    "error",
    [RuntimeError("executor bug"), TaskCommandDeferred("coordinator unhealthy")],
    ids=["executor-error", "deferrals-exhausted"],
)
async def test_live_input_command_terminal_failure_broadcasts_returned(
    monkeypatch: pytest.MonkeyPatch, task: Any, error: BaseException
) -> None:
    frames: list[dict[str, Any]] = []

    async def publish(frame: dict[str, Any], task_id: int) -> None:
        frames.append(frame)

    monkeypatch.setattr(task_command_execution, "publish_task_event", publish)
    await task_command_execution._broadcast_terminal_command_error(command(task), error)
    assert frames == [
        {
            "type": "live_input_returned",
            "task_id": task.id,
            "turn_id": TURN,
            "reason": "unavailable",
        }
    ]
