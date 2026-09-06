"""Exception text must not reach chat clients (PR #1472 review finding N3)."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.web.api.client_safe_ast_guard import (
    ALLOWED_RAW_MESSAGES,
    SAFE_MESSAGE_BUILDERS,
    SAFE_MESSAGE_CONSTANTS,
    _scan,
)
from tests.web.api.client_safe_ast_guard import guard_offenders as _guard_offenders
from xagent.web.api import websocket as websocket_api
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.mcp_runtime import (
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator

from .conftest import _direct_db_session

SECRET = "/srv/xagent/secrets/prod.key"


def _client_payloads(connection_manager: MagicMock) -> list[dict]:
    return [
        call.args[0]
        for call in (
            connection_manager.send_personal_message.await_args_list
            + connection_manager.broadcast_to_task.await_args_list
        )
        if call.args and isinstance(call.args[0], dict)
    ]


def _sent_text_payloads(websocket: MagicMock) -> list[dict]:
    """Payloads written straight to the socket, bypassing ``manager``.

    ``handle_builder_chat`` uses this sink, so a helper that only reads the
    manager mock cannot see it - which is why that handler's leak survived
    until the AST sweep learned to recognize ``send_text``.
    """
    payloads = []
    for call in websocket.send_text.await_args_list:
        if not call.args:
            continue
        try:
            decoded = json.loads(call.args[0])
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            payloads.append(decoded)
    return payloads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised",
    [
        ValueError(f"invalid payload while reading {SECRET}"),
        KeyError(f"missing key near {SECRET}"),
        TypeError(f"bad type from {SECRET}"),
        MCPBuiltinOAuthActorPolicyRequiredError(
            f"actor task policy loaded from {SECRET}"
        ),
    ],
    ids=["value", "key", "type", "actor-policy"],
)
async def test_execute_task_redacts_an_incidental_validation_error(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
) -> None:
    """A widget visitor gets the fixed string, never the exception's text."""
    db = _direct_db_session()
    try:
        user = User(
            username=f"safe-error-owner-{type(raised).__name__}",
            password_hash="hash",
        )
        db.add(user)
        db.commit()
        task = Task(
            user_id=int(user.id),
            title="Client safe errors",
            description="Run the existing task",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        user_id = int(user.id)
    finally:
        db.close()

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    async def raise_at_schedule(**_kwargs: object) -> asyncio.Task:
        raise raised

    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "schedule_existing_task_execution",
        AsyncMock(side_effect=raise_at_schedule),
    )

    await websocket_api.handle_execute_task(
        MagicMock(),
        task_id,
        {"user": SimpleNamespace(id=user_id, is_admin=False)},
    )

    payloads = _client_payloads(connection_manager)
    assert payloads, "the handler must tell the client something"
    serialized = repr(payloads)
    assert SECRET not in serialized
    assert str(raised) not in serialized
    assert any(
        payload.get("message") == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
        for payload in payloads
    )


@pytest.mark.asyncio
async def test_execute_task_uses_the_authentication_error_contract(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A curated authentication failure uses its fixed code and fallback."""
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    # No authenticated actor: the handler raises its own curated message.
    await websocket_api.handle_execute_task(MagicMock(), 1, {})

    payloads = _client_payloads(connection_manager)
    assert any(
        payload.get("message") == "Authentication is required to send this message."
        and payload.get("error_code") == "authentication_required"
        for payload in payloads
    )


def test_no_delivery_producer_can_bypass_the_client_safe_message() -> None:
    """Exception text may not reach a client through the *recognized* shapes.

    Scope, stated honestly: this walks the direct producers and error-payload
    sinks used by this module. It understands the task-error and stream-event
    helpers, explicit overrides on dict-spread payloads, and the listed
    deferred-delivery wrapper.

    It is not general interprocedural data-flow analysis. Dynamic payload types
    fail closed unless they come from the listed module helpers, and the type
    set is maintained rather than derived. Do not read a passing run as proof
    that arbitrary Python data flow cannot reach a client.
    """
    # Explicit encoding: this module carries non-ASCII prose, and the
    # platform default would decode it as cp1252/GBK on a Windows runner.
    source = Path(websocket_api.__file__).read_text(encoding="utf-8")
    result = _scan(ast.parse(source))

    for builder in SAFE_MESSAGE_BUILDERS:
        assert callable(getattr(websocket_api, builder, None)), (
            f"SAFE_MESSAGE_BUILDERS blesses {builder!r}, which does not exist"
        )
    for constant in SAFE_MESSAGE_CONSTANTS:
        assert isinstance(getattr(websocket_api, constant, None), str), (
            f"SAFE_MESSAGE_CONSTANTS blesses {constant!r}, which is not text"
        )

    # These are deliberate exact baselines. If a producer is added or removed,
    # inspect the changed site and bump the corresponding count in this test.
    assert result.producers == 29, (
        f"expected exactly 29 producers, matched {result.producers}; "
        "review the changed sites and bump deliberately"
    )
    # #1658 removed ``_resync_client_to_running_task``'s stale-client ``error``
    # frame in favour of the control-only ``task_resumed`` shape. The inner
    # RuntimeError arm now also reuses ``answer_durable_turn_failure`` instead
    # of spelling out three error payloads locally, bringing the census to 50.
    # ``_broadcast_terminal_command_error`` gained a third payload literal for
    # external-scope non-cancel commands, mirroring the persisted-event
    # identity rule for the live frame too, bringing the census to 51.
    assert result.error_payloads == 51, (
        f"expected exactly 51 error payloads, matched {result.error_payloads}; "
        "review the changed sites and bump deliberately"
    )
    # Every allowlist entry must be earned by a live call site: a stale entry
    # is a standing exemption nothing uses, and an unused closure entry is
    # exactly what a reverted parameter-rebinding fix would leave behind.
    assert result.used_allowlist == ALLOWED_RAW_MESSAGES, (
        "stale allowlist entries: "
        f"{sorted(ALLOWED_RAW_MESSAGES - result.used_allowlist)}"
    )
    assert not result.offenders, (
        "raw text can reach a chat client; route it through "
        "client_safe_error_message: " + "; ".join(sorted(set(result.offenders)))
    )


WEBSOCKET_LOGGER = "xagent.web.api.websocket"

# Every handler that turns an enqueue refusal into client-visible text. Each
# entry is (handler, extra message_data) - the ack in handle_chat_message only
# fires when the client supplied an id, so that one needs the extra key.
ENQUEUE_FAILURE_HANDLERS = [
    ("handle_chat_message", {"client_message_id": "cmid-1"}),
    ("handle_pause_task", {}),
    ("handle_resume_task", {}),
]


def test_missing_task_keeps_its_wording_for_the_sender(_test_db: None) -> None:
    """A missing task is the sender's own answer, so redaction must spare it.

    ``execute_task_background`` already raises this as client-visible; the
    pause/resume enqueue path raised a bare ``ValueError``, which the redaction
    turned into the generic string and left the sender with nothing to act on.
    """
    with pytest.raises(ValueError) as raised:
        websocket_api._enqueue_websocket_task_command_sync(
            task_id=424242,
            actor_user_id=1,
            actor_is_admin=False,
            command_id="pause:missing-task",
            kind=websocket_api.TaskCommandKind.PAUSE,
            payload={},
            allow_missing_task=False,
        )

    assert isinstance(raised.value, websocket_api.ClientVisibleValidationError)
    assert (
        websocket_api.client_safe_error_message(raised.value) == "Task 424242 not found"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_name", ["handle_pause_task", "handle_resume_task"], ids=["pause", "resume"]
)
async def test_missing_task_control_uses_stable_error_code(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
) -> None:
    connection_manager = MagicMock(
        send_personal_message=AsyncMock(),
        broadcast_to_task=AsyncMock(),
    )
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    await getattr(websocket_api, handler_name)(
        MagicMock(),
        424242,
        {"user": SimpleNamespace(id=1, is_admin=False)},
    )

    payloads = _client_payloads(connection_manager)
    assert payloads == [
        {
            "type": "error",
            "message": "Task is no longer available.",
            "error_code": "task_unavailable",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_type", ["pause_task", "resume_task"], ids=["pause", "resume"]
)
async def test_private_endpoint_closes_foreign_task_before_control_dispatch(
    _test_db: None,
    message_type: str,
) -> None:
    """Foreign sockets never join or dispatch; missing ids retain recovery."""
    from fastapi import WebSocketDisconnect

    db = _direct_db_session()
    try:
        owner = User(username=f"oracle-owner-{message_type}", password_hash="hash")
        intruder = User(
            username=f"oracle-intruder-{message_type}", password_hash="hash"
        )
        db.add_all([owner, intruder])
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Private task",
            description="Must not be discoverable",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        foreign_task_id = int(task.id)
        intruder_id = int(intruder.id)
    finally:
        db.close()

    async def send_control(task_id: int) -> tuple[list[dict], list[tuple[int, str]]]:
        websocket = MagicMock()
        websocket.accept = AsyncMock()
        closed: list[tuple[int, str]] = []
        websocket.close = AsyncMock(
            side_effect=lambda *, code, reason: closed.append((code, reason))
        )
        websocket.receive_text = AsyncMock(
            side_effect=[json.dumps({"type": message_type}), WebSocketDisconnect()]
        )
        connection_manager = MagicMock(
            register_connection=MagicMock(),
            disconnect=MagicMock(),
            send_personal_message=AsyncMock(),
            broadcast_to_task=AsyncMock(),
        )
        with (
            patch.object(websocket_api, "manager", connection_manager),
            patch.object(
                websocket_api,
                "get_authenticated_user",
                AsyncMock(return_value=SimpleNamespace(id=intruder_id, is_admin=False)),
            ),
        ):
            await websocket_api.websocket_chat_endpoint(websocket, task_id, None)
        return (
            [
                call.args[0]
                for call in connection_manager.send_personal_message.await_args_list
            ],
            closed,
        )

    missing_payloads, missing_closes = await send_control(foreign_task_id + 424242)
    foreign_payloads, foreign_closes = await send_control(foreign_task_id)

    expected = [
        {
            "type": "error",
            "message": "Task is no longer available.",
            "error_code": "task_unavailable",
        }
    ]
    assert missing_payloads == expected
    assert missing_closes == []
    assert foreign_payloads == []
    assert foreign_closes == [(4003, "Task is no longer available.")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "extra_message_data"),
    ENQUEUE_FAILURE_HANDLERS,
    ids=[name for name, _ in ENQUEUE_FAILURE_HANDLERS],
)
async def test_redacted_enqueue_failure_still_reaches_the_log(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    handler_name: str,
    extra_message_data: dict,
) -> None:
    """Redacting the client's copy must not delete the operator's copy.

    These handlers previously leaked ``str(exc)`` to the client and logged
    nothing; the leak was the only record. With the text redacted, an
    incidental failure would otherwise vanish without a trace.
    """
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "_enqueue_websocket_task_command",
        AsyncMock(side_effect=ValueError(f"enqueue failed reading {SECRET}")),
    )

    handler = getattr(websocket_api, handler_name)
    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        await handler(
            MagicMock(),
            7,
            {"user": SimpleNamespace(id=1, is_admin=False), **extra_message_data},
        )

    payloads = _client_payloads(connection_manager)
    assert payloads, "the handler must tell the client something"
    assert SECRET not in repr(payloads)
    assert any(
        payload.get("message") == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
        for payload in payloads
    )

    records = [record for record in caplog.records if record.name == WEBSOCKET_LOGGER]
    assert records, f"{handler_name} redacted the failure without logging it"
    assert any(SECRET in record.getMessage() for record in records)
    assert any(record.exc_info is not None for record in records)


@pytest.mark.parametrize(
    ("error", "expected_level", "expects_traceback"),
    [
        (
            websocket_api.ClientVisibleValidationError("User authentication required"),
            logging.WARNING,
            False,
        ),
        (ValueError(f"incidental fault at {SECRET}"), logging.ERROR, True),
    ],
    ids=["curated", "incidental"],
)
def test_log_level_follows_the_marker_not_the_call_site(
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    expected_level: int,
    expects_traceback: bool,
) -> None:
    """A curated refusal is routine; only an incidental fault earns a traceback.

    Without the split, any visitor could make the server dump a stack on
    demand by sending an unauthenticated frame in a loop.
    """
    with caplog.at_level(logging.DEBUG, logger=WEBSOCKET_LOGGER):
        websocket_api.log_client_facing_failure(
            error, "Pause command rejected for task %s: %s", 7
        )

    (record,) = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert record.levelno == expected_level
    assert (record.exc_info is not None) is expects_traceback
    assert "task 7" in record.getMessage()


def test_malformed_curated_failure_remains_a_warning_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    websocket_api.log_client_facing_failure(
        websocket_api.ClientVisibleValidationError("Authentication required"),
        "Pause command rejected",
    )

    (record,) = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert (record.levelno, record.exc_info) == (logging.WARNING, None)
    assert "malformed client-facing log template" in record.getMessage().lower()


@pytest.mark.parametrize(
    ("template", "args"),
    [
        ("Pause command rejected", ()),
        ("Task %s failed: %s", ()),
        ("Task %d failed: %s", ()),
        ("Task %(task_id)s failed: %s", ()),
        ("%%s", ()),
    ],
    ids=[
        "missing-placeholder",
        "count-mismatch",
        "integer-placeholder",
        "mapping-placeholder",
        "terminal-escaped-percent-s",
    ],
)
def test_log_helper_rejects_every_malformed_percent_template(
    caplog: pytest.LogCaptureFixture,
    template: str,
    args: tuple[object, ...],
) -> None:
    try:
        raise ValueError("operator detail")
    except ValueError as error:
        with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
            websocket_api.log_client_facing_failure(error, template, *args)

    records = [record for record in caplog.records if record.name == WEBSOCKET_LOGGER]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is ValueError
    assert "malformed client-facing log template" in records[0].getMessage().lower()
    assert "operator detail" in records[0].getMessage()


def test_log_helper_accepts_int_enum_for_native_integer_placeholder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class TaskNumber(IntEnum):
        FIRST = 7

    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        websocket_api.log_client_facing_failure(
            ValueError("operator detail"),
            "Task %d failed: %s",
            TaskNumber.FIRST,
        )

    records = [record for record in caplog.records if record.name == WEBSOCKET_LOGGER]
    assert len(records) == 1
    assert records[0].getMessage() == "Task 7 failed: operator detail"
    assert "malformed client-facing log template" not in records[0].getMessage().lower()


@pytest.mark.parametrize(
    ("template", "args", "expected_message"),
    [
        ("V=%r: %s", (SimpleNamespace(x="é"),), "V=namespace(x='é'): e"),
        ("V=%a: %s", (SimpleNamespace(x="é"),), "V=namespace(x='\\xe9'): e"),
        ("100%%: %s", (), "100%: e"),
    ],
)
def test_log_helper_preserves_native_percent_formatting(
    caplog: pytest.LogCaptureFixture,
    template: str,
    args: tuple[object, ...],
    expected_message: str,
) -> None:
    websocket_api.log_client_facing_failure(ValueError("e"), template, *args)

    (record,) = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert record.getMessage() == expected_message


def test_log_helper_does_not_raise_for_unprintable_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenText:
        def __str__(self) -> str:
            raise RuntimeError("broken text")

    class BrokenStr(str):
        __str__ = BrokenText.__str__

        def __repr__(self) -> str:
            raise RuntimeError("broken repr")

    class BadFallback(str):
        def __str__(self) -> str:
            return self

        def __repr__(self) -> str:
            raise RuntimeError("bad repr")

    class StatefulText(str):
        calls = 0

        def __str__(self) -> str:
            type(self).calls += 1
            if type(self).calls >= 3:
                raise RuntimeError("rendered repeatedly")
            return self

    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        websocket_api.log_client_facing_failure(
            ValueError(BrokenText()), "Data validation error: %s"
        )
        websocket_api.log_client_facing_failure(
            ValueError("operator detail"), "Task %s failed: %s", BrokenStr("x")
        )
        websocket_api.log_client_facing_failure(
            ValueError("operator detail"), "Malformed template", BadFallback("x")
        )
        websocket_api.log_client_facing_failure(
            ValueError("operator detail"),
            "Task %s failed: %s",
            StatefulText("stable"),
        )

    records = [record for record in caplog.records if record.name == WEBSOCKET_LOGGER]
    assert len(records) == 4
    assert "unprintable ValueError" in records[0].getMessage()
    assert "unprintable BrokenStr" in records[1].getMessage()
    assert "malformed client-facing log template" in records[2].getMessage().lower()
    assert "Task stable failed" in records[3].getMessage()


def test_client_visible_error_is_a_subclass_only_marker() -> None:
    """The marker base cannot escape handlers that catch its typed subclasses."""
    with pytest.raises(TypeError, match="must be subclassed"):
        websocket_api.ClientVisibleError("bare marker")

    assert str(websocket_api.ClientVisibleValidationError("curated")) == "curated"


@pytest.mark.parametrize("message", ["", "   ", "\t\n"])
def test_empty_client_visible_message_falls_back_to_the_generic_text(
    message: str,
) -> None:
    error = websocket_api.ClientVisibleValidationError(message)

    assert (
        websocket_api.client_safe_error_message(error)
        == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
    )


def test_client_visible_message_preserves_non_ascii_text() -> None:
    message = "请求无效：缺少步骤标识"

    assert (
        websocket_api.client_safe_error_message(
            websocket_api.ClientVisibleValidationError(message)
        )
        == message
    )


def test_incidental_exception_uses_the_requested_safe_fallback() -> None:
    fallback = "The requested operation could not be completed."

    rendered = websocket_api.client_safe_error_message(
        RuntimeError(SECRET),
        fallback=fallback,
    )

    assert rendered == fallback
    assert SECRET not in rendered


def test_empty_client_visible_outer_error_does_not_expose_its_cause() -> None:
    cause = RuntimeError(SECRET)
    error = websocket_api.ClientVisibleValidationError("")
    error.__cause__ = cause

    rendered = websocket_api.client_safe_error_message(error)

    assert rendered == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
    assert SECRET not in rendered


@pytest.mark.asyncio
async def test_builder_chat_redacts_through_its_own_socket_sink(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The leak found by self-review, pinned at runtime rather than by AST alone.

    ``handle_builder_chat`` answers on ``websocket.send_text`` instead of going
    through ``manager``, which is why it escaped both the original sweep and
    every behavioural test in this file.
    """
    from xagent.web.services import builder_chat_runtime

    monkeypatch.setattr(
        builder_chat_runtime,
        "load_builder_chat_runtime_inputs",
        AsyncMock(side_effect=ValueError(f"builder fault at {SECRET}")),
    )

    websocket = MagicMock()
    websocket.send_text = AsyncMock()
    websocket.state = SimpleNamespace()

    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        await websocket_api.handle_builder_chat(
            websocket,
            {"message": "build me an agent"},
            SimpleNamespace(id=1, is_admin=False),
        )

    records = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert any(SECRET in r.getMessage() for r in records), (
        "the operator half of the contract: redacting the client's copy must "
        "not delete the server's"
    )

    payloads = _sent_text_payloads(websocket)
    errors = [p for p in payloads if p.get("type") == "error"]
    assert errors, "the handler must answer the builder client"
    assert SECRET not in repr(payloads)
    assert errors[-1]["message"] == websocket_api.CLIENT_SAFE_VALIDATION_ERROR


@pytest.mark.asyncio
async def test_actor_policy_rejection_returns_chat_delivery_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = MagicMock()
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    send_delivery = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(websocket_api, "send_message_delivery", send_delivery)
    monkeypatch.setattr(
        websocket_api,
        "_enqueue_websocket_task_command",
        AsyncMock(
            side_effect=MCPBuiltinOAuthActorPolicyRequiredError(
                "actor-marked task does not support generic control"
            )
        ),
    )

    await websocket_api.handle_chat_message(
        websocket,
        42,
        {"client_message_id": "command-1"},
    )

    send_delivery.assert_awaited_once_with(
        websocket,
        client_message_id="command-1",
        turn_id="command-1",
        accepted=False,
        message=websocket_api.CLIENT_SAFE_VALIDATION_ERROR,
        error_code=ClientErrorCode.MESSAGE_PROCESSING_FAILED.value,
        rejection_outcome="not_accepted",
    )
    connection_manager.send_personal_message.assert_awaited_once_with(
        {
            "type": "error",
            "message": websocket_api.CLIENT_SAFE_VALIDATION_ERROR,
            "error_code": ClientErrorCode.MESSAGE_PROCESSING_FAILED.value,
        },
        websocket,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_name", ["handle_pause_task", "handle_resume_task"], ids=["pause", "resume"]
)
async def test_actor_policy_rejection_returns_websocket_error(
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
) -> None:
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "_enqueue_websocket_task_command",
        AsyncMock(
            side_effect=MCPBuiltinOAuthActorPolicyRequiredError(
                "actor-marked task does not support generic control"
            )
        ),
    )

    await getattr(websocket_api, handler_name)(
        MagicMock(),
        42,
        {"user": SimpleNamespace(id=7, is_admin=False)},
    )

    connection_manager.send_personal_message.assert_awaited_once()
    payload = connection_manager.send_personal_message.await_args.args[0]
    assert payload == {
        "type": "error",
        "message": websocket_api.CLIENT_SAFE_VALIDATION_ERROR,
        "error_code": ClientErrorCode.MESSAGE_PROCESSING_FAILED.value,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_name", ["handle_pause_task", "handle_resume_task"], ids=["pause", "resume"]
)
async def test_permission_rejection_uses_neutral_unavailable_code(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
) -> None:
    """Pause and resume do not reveal that another user's task exists."""
    db = _direct_db_session()
    try:
        owner = User(username=f"owner-{handler_name}", password_hash="hash")
        intruder = User(username=f"intruder-{handler_name}", password_hash="hash")
        db.add_all([owner, intruder])
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Someone else's task",
            description="Not yours",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id, intruder_id = int(task.id), int(intruder.id)
    finally:
        db.close()

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    await getattr(websocket_api, handler_name)(
        MagicMock(),
        task_id,
        {"user": SimpleNamespace(id=intruder_id, is_admin=False)},
    )

    payloads = _client_payloads(connection_manager)
    assert payloads, "the handler must refuse the intruder out loud"
    assert any(
        payload.get("message") == "Task is no longer available."
        and payload.get("error_code") == "task_unavailable"
        for payload in payloads
    ), payloads


# Each entry mirrors a real leak shape fixed by #1696 rather than a minimal
# synthetic repro: dict-spread copies
# ``execute_task_background`` (text under ``error``, type inherited from the
# spread), helper-built copies ``send_historical_data_as_stream``, and
# wrapper-forwarded copies ``notify_deferred_delivery``. These cases must stay
# visible to the production sweep so the three mechanisms cannot regress.
BYPASS_SHAPES = [
    pytest.param(
        """
def _terminal_task_error_payload(task_id, message):
    return {"type": "agent_error", "message": message}


async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        terminal_payload = _terminal_task_error_payload(task_id, str(e))
        message = str(e)
        await manager.broadcast_to_task(
            {
                **terminal_payload,
                "task_id": task_id,
                "error": message,
                "timestamp": 0,
            },
            task_id,
        )
""",
        id="dict-spread",
    ),
    pytest.param(
        """
def create_stream_event(event_type, task_id, data):
    return {"type": event_type, "task_id": task_id, **data}


async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        error_event = create_stream_event(
            "error", task_id, {"message": str(e)}
        )
        await manager.send_personal_message(error_event, websocket)
""",
        id="helper-built-then-passed-by-name",
    ),
    pytest.param(
        """
async def notify_deferred_delivery(accepted, raw):
    await send_message_delivery(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message=raw,
        rejection_outcome="not_accepted",
    )


async def leak(websocket):
    try:
        pass
    except Exception as e:
        await notify_deferred_delivery(False, str(e))
""",
        id="wrapper-forwarded",
    ),
    pytest.param(
        """
async def leak(task_id):
    try:
        pass
    except Exception as e:
        await manager.broadcast_to_task(
            {"type": "task_error", "message": str(e)}, task_id
        )
""",
        id="task-error-message",
    ),
    pytest.param(
        """
async def leak(task_id):
    try:
        pass
    except Exception as e:
        await manager.broadcast_to_task(
            {"type": "task_error", "error": str(e)}, task_id
        )
""",
        id="task-error-error-field",
    ),
]

# Deliberately NOT listed above: ``message_data["_durable_command_error"] =
# str(e)``. Earlier rounds of this PR described that as reaching clients via
# _broadcast_terminal_command_error, which is wrong - TaskCommandRejected is
# re-raised without broadcasting (websocket.py, execute_durable_task_command),
# and the two branches that do broadcast now go through the chokepoint.
#
# The text lands in the TaskExecutionCommand.error column. That column IS
# read back to a client - a2a.py returns it verbatim as a 500 internal_error
# body - so the reason this particular channel is not a client leak is
# narrower than "nothing reads the column": a2a only ever enqueues CANCEL, so
# the pause/resume text written here cannot reach that read path. Widening a2a
# to another command kind would turn this into a real leak, which is why the
# dependency is written down rather than left implicit.


@pytest.mark.parametrize("source", BYPASS_SHAPES)
def test_known_bypass_shapes_are_rejected_by_the_guard(source: str) -> None:
    """The guard rejects every producer shape fixed for #1696."""
    assert _guard_offenders(source), (
        "the guard missed a client-facing raw exception shape fixed for #1696"
    )


@pytest.mark.asyncio
async def test_unresolvable_task_answers_the_sender_instead_of_dropping_them(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence is a worse answer than redaction.

    This raise site was a bare ``Exception`` while its sibling a few lines
    above already carried the marker. Being untyped, it escaped every typed
    handler, reached the connection-level ``finally: manager.disconnect`` and
    left the client with nothing at all - not the text, not even the generic
    string.
    """
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    missing_task_id = 987654
    with caplog.at_level(logging.DEBUG, logger=WEBSOCKET_LOGGER):
        await websocket_api.handle_execute_task(
            MagicMock(),
            missing_task_id,
            {"user": SimpleNamespace(id=1, is_admin=False)},
        )

    payloads = _client_payloads(connection_manager)
    assert any(
        payload.get("message") == "Task is no longer available."
        and payload.get("error_code") == "task_unavailable"
        for payload in payloads
    ), payloads

    # Marking this raise made it reachable by a typed handler, which is the
    # point - but that handler must not hand an anonymous visitor a way to
    # make the server dump a stack for every task id they guess.
    records = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert records, "the refusal still has to be recorded"
    assert all(r.exc_info is None for r in records), (
        "a curated refusal is routine; only an incidental fault earns a stack"
    )
    assert all(r.levelno <= logging.WARNING for r in records)


# --- Round 5: the parameter short-circuit must not hide rebound names -------

REBOUND_PARAMETER_SHAPES = [
    pytest.param(
        """
async def leak(websocket, message):
    try:
        pass
    except Exception as e:
        message = str(e)
        await send_message_delivery(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=message,
            rejection_outcome="not_accepted",
        )
""",
        id="same-scope-rebind",
    ),
    pytest.param(
        """
async def outer(websocket, message):
    async def inner(e):
        message = str(e)
        await send_message_delivery(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=message,
            rejection_outcome="not_accepted",
        )
""",
        id="nested-shadow",
    ),
    pytest.param(
        """
async def leak(websocket, message):
    try:
        pass
    except Exception as e:
        message: str = str(e)
        await send_message_delivery(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=message,
            rejection_outcome="not_accepted",
        )
""",
        id="annotated-rebind",
    ),
    pytest.param(
        """
async def leak(websocket, message):
    try:
        pass
    except Exception as e:
        message += str(e)
        await send_message_delivery(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=message,
            rejection_outcome="not_accepted",
        )
""",
        id="augmented-rebind",
    ),
    pytest.param(
        """
async def leak(websocket, message):
    try:
        pass
    except Exception as e:
        (message := str(e))
        await send_message_delivery(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=message,
            rejection_outcome="not_accepted",
        )
""",
        id="walrus-rebind",
    ),
    pytest.param(
        """
async def leak(websocket, message):
    try:
        pass
    except Exception as e:
        message, ignored = str(e), None
        await send_message_delivery(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=message,
            rejection_outcome="not_accepted",
        )
""",
        id="tuple-rebind",
    ),
]


@pytest.mark.parametrize("source", REBOUND_PARAMETER_SHAPES)
def test_guard_catches_a_rebound_parameter(source: str) -> None:
    """A name is only "vetted at the caller" while nothing in scope rebinds it.

    The short-circuit used to fire on the bare parameter match, before local
    assignments were even collected, so ``message = str(e)`` shadowing a
    ``message`` argument sailed through.
    """
    assert _guard_offenders(source), "the rebound parameter must be flagged"


def test_guard_still_trusts_a_genuinely_forwarded_parameter() -> None:
    """The wrapper shape stays clean; only its callers are judged."""
    source = """
async def forward(websocket, message):
    await send_message_delivery(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message=message,
        rejection_outcome="not_accepted",
    )
"""
    assert not _guard_offenders(source)


def test_allowlist_is_scoped_to_the_runtime_error_handler() -> None:
    source = """
async def handle_intervention(websocket):
    try:
        pass
    except ValueError as e:
        await manager.send_personal_message(
            {"type": "error", "message": f"Runtime error: {str(e)}"},
            websocket,
        )
"""

    assert _guard_offenders(source), "a validation branch cannot reuse the carve-out"


def test_allowlist_cannot_flow_from_runtime_into_a_validation_handler() -> None:
    source = """
async def handle_intervention(websocket):
    try:
        pass
    except RuntimeError as e:
        message = f"Runtime error: {str(e)}"

    try:
        pass
    except ValueError:
        await manager.send_personal_message(
            {"type": "error", "message": message},
            websocket,
        )
"""

    assert _guard_offenders(source), "the carve-out cannot cross except handlers"


def test_allowlist_cannot_flow_into_a_nested_runtime_error_handler() -> None:
    source = """
async def handle_intervention(websocket):
    try: first_operation()
    except RuntimeError as e:
        message = f"Runtime error: {str(e)}"
        try: second_operation()
        except RuntimeError as other:
            await manager.send_personal_message({"type": "error", "message": message})
"""

    assert _guard_offenders(source), "a nested handler cannot borrow the carve-out"


def test_conditional_allowlisted_assignment_keeps_the_incoming_parameter() -> None:
    source = """
async def handle_intervention(websocket, message, flag):
    try:
        pass
    except RuntimeError as e:
        if flag:
            message = f"Runtime error: {str(e)}"
        await manager.send_personal_message(
            {"type": "error", "message": message},
            websocket,
        )
"""

    assert _guard_offenders(source), "the false branch still forwards raw input"


def test_conditional_assignment_keeps_the_outer_local_definition() -> None:
    source = """
async def handle_intervention(websocket, error, flag):
    message = str(error)
    try: pass
    except RuntimeError as e:
        if flag: message = f"Runtime error: {str(e)}"
        else:
            await manager.send_personal_message({"type": "error", "message": message})
"""

    assert _guard_offenders(source), "the else branch still uses the outer local"


def test_allowlisted_assignment_after_sink_cannot_rewrite_history() -> None:
    source = """
async def handle_intervention(websocket, message):
    try:
        pass
    except RuntimeError as e:
        await manager.send_personal_message(
            {"type": "error", "message": message},
            websocket,
        )
        message = f"Runtime error: {str(e)}"
"""

    assert _guard_offenders(source), "a later binding cannot sanitize an earlier send"


@pytest.mark.parametrize(
    "loop_header",
    ["for item in items:", "async for item in items:", "while items:"],
    ids=["for", "async-for", "while"],
)
def test_loop_backedge_rebinding_reaches_an_earlier_sink(loop_header: str) -> None:
    source = f"""
async def leak(items, error):
    message = "safe"
    {loop_header}
        await send_message_delivery(message=message)
        message = str(error)
"""

    assert _guard_offenders(source), "the next iteration sends the rebound value"


def test_loop_backedge_binding_is_unavailable_on_the_first_iteration() -> None:
    source = """
async def leak(items, message):
    for item in items:
        await send_message_delivery(message=message)
        message = "safe"
"""

    assert _guard_offenders(source), "the first iteration still sends the input"


def test_while_backedge_rebinding_reaches_the_next_condition() -> None:
    source = """
async def leak(error):
    message = "safe"
    while await send_message_delivery(message=message):
        message = str(error)
"""

    assert _guard_offenders(source), "the next condition sees the rebound value"


def test_while_condition_rebinding_reaches_the_next_condition() -> None:
    source = """
async def leak(error):
    message = "safe"
    while (await send_message_delivery(message=message)) or (message := str(error)):
        pass
"""

    assert _guard_offenders(source), "the next condition sees the rebound value"


def test_safe_while_condition_backedge_keeps_the_sink_clean() -> None:
    source = """
async def deliver():
    message = "safe"
    while (await send_message_delivery(message=message)) or (message := "still safe"):
        pass
"""

    assert not _guard_offenders(source)


def test_loop_backedge_augassign_reaches_an_earlier_sink() -> None:
    source = """
async def leak(items, error):
    message = "safe"
    for item in items:
        await send_message_delivery(message=message)
        message += str(error)
"""

    assert _guard_offenders(source), "the augmented value reaches iteration two"


def test_safe_loop_backedge_keeps_the_sink_clean() -> None:
    source = """
async def deliver(items):
    message = "safe"
    for item in items:
        await send_message_delivery(message=message)
        message = "still safe"
"""

    assert not _guard_offenders(source)


@pytest.mark.parametrize(
    "control_flow",
    [
        'for item in items:\n            message = f"Runtime error: {str(e)}"',
        'async for item in items:\n            message = f"Runtime error: {str(e)}"',
        'while items:\n            message = f"Runtime error: {str(e)}"\n            break',
        'match items:\n            case [item]:\n                message = f"Runtime error: {str(e)}"',
    ],
    ids=["for", "async-for", "while", "match"],
)
def test_conditional_control_flow_keeps_the_incoming_parameter(
    control_flow: str,
) -> None:
    source = f"""
async def handle_intervention(websocket, message, items):
    try:
        pass
    except RuntimeError as e:
        {control_flow}
        await manager.send_personal_message(
            {{"type": "error", "message": message}},
            websocket,
        )
"""

    assert _guard_offenders(source), "the control flow can skip the safe binding"


def test_short_circuit_walrus_keeps_the_incoming_parameter() -> None:
    source = """
async def handle_intervention(websocket, message, flag):
    try:
        pass
    except RuntimeError as e:
        flag and (message := f"Runtime error: {str(e)}")
        await manager.send_personal_message(
            {"type": "error", "message": message},
            websocket,
        )
"""

    assert _guard_offenders(source), "the short-circuited walrus may never bind"


@pytest.mark.parametrize(
    "tail",
    [
        "except Exception: await send_message_delivery(message=message)",
        "except* Exception: await send_message_delivery(message=message)",
        "finally: await send_message_delivery(message=message)",
        "except Exception: pass\n    await send_message_delivery(message=message)",
    ],
    ids=["handler", "exception-group-handler", "finally", "after-handled-try"],
)
def test_try_reaching_definitions_keep_the_skipped_value(tail: str) -> None:
    source = f"""
async def leak(message):
    try: may_raise(); message = "safe"
    {tail}
"""
    assert _guard_offenders(source)


def test_try_assignment_reaches_a_later_sink_in_the_same_body() -> None:
    source = """
async def reject(message):
    try: message = "safe"; await send_message_delivery(message=message)
    except Exception: pass
"""
    assert not _guard_offenders(source)


def test_lambda_bindings_do_not_reach_the_enclosing_scope() -> None:
    source = """
async def leak(websocket, error):
    message = "safe"; build_detail = lambda: (message := error.detail)
    await send_message_delivery(message=message)
"""

    assert not _guard_offenders(source), "the lambda has its own lexical scope"


@pytest.mark.parametrize(
    "source",
    [
        """async def outer(message, raw):
    maker = lambda message: send_message_delivery(message=message)
    await maker(raw)""",
        """async def outer(message, raw):
    class Pending:
        message = raw
        delivery = send_message_delivery(message=message)
    await Pending.delivery""",
    ],
    ids=["lambda", "class"],
)
def test_guard_rejects_sinks_inside_unsupported_scopes(source: str) -> None:
    assert _guard_offenders(source)


def test_conditional_expression_assignment_keeps_the_incoming_value() -> None:
    source = """
async def leak(message, flag):
    "safe" if flag else (message := "safe")
    await send_message_delivery(message=message)
"""
    assert _guard_offenders(source)


def test_guard_catches_augassign_that_keeps_a_forwarded_parameter() -> None:
    source = """
async def leak(websocket, message):
    message += "!"
    await send_message_delivery(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message=message,
        rejection_outcome="not_accepted",
    )
"""

    assert _guard_offenders(source)


def test_guard_catches_a_nested_unpack_rebinding() -> None:
    source = """
async def leak(websocket, message, source):
    ((message, other), final) = source
    await send_message_delivery(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message=message,
        rejection_outcome="not_accepted",
    )
"""

    assert _guard_offenders(source)


@pytest.mark.parametrize(
    "body",
    [
        "for message in items: await send_message_delivery(message=message)",
        "await gather(*(send_message_delivery(message=message) for message in items))",
        """match items:
        case {"raw": message}: await send_message_delivery(message=message)""",
        "with resource as message: await send_message_delivery(message=message)",
        """try: pass
    except Exception as message: await send_message_delivery(message=message)""",
    ],
    ids=["loop", "comprehension", "match", "with", "except"],
)
def test_guard_catches_non_assignment_rebindings(body: str) -> None:
    source = f"async def leak(message, items, resource):\n    {body}\n"
    assert _guard_offenders(source), "the binding is not a forwarded parameter"


@pytest.mark.parametrize(
    "expression",
    [
        'message_data.get("error", "fallback")',
        'message_data.get("error")',
        'error.__dict__.get("detail", "fallback")',
    ],
)
def test_guard_rejects_get_calls_from_untrusted_receivers(expression: str) -> None:
    source = f"""
async def leak(websocket, message_data, error):
    await send_message_delivery(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message={expression},
        rejection_outcome="not_accepted",
    )
"""

    assert _guard_offenders(source)


def test_guard_accepts_the_curated_rejection_table_lookup() -> None:
    source = """
async def reject(websocket, reason):
    await send_message_delivery(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message=_TURN_REJECTION_MESSAGES.get(reason, "Task is busy"),
        rejection_outcome="not_accepted",
    )
"""

    assert not _guard_offenders(source)


@pytest.mark.parametrize(
    ("parameters", "binding", "fallback"),
    [
        ("reason, error", "", "error.detail"),
        ("_TURN_REJECTION_MESSAGES, reason", "", '"Task is busy"'),
        ("reason, **_TURN_REJECTION_MESSAGES", "", '"fallback"'),
        ("reason", "from evil import _TURN_REJECTION_MESSAGES", '"fallback"'),
        ("reason", "class _TURN_REJECTION_MESSAGES: get = evil_get", '"fallback"'),
    ],
    ids=["nonliteral-fallback", "parameter", "kwargs", "import", "class"],
)
def test_guard_rejects_unsafe_curated_lookups(
    parameters: str, binding: str, fallback: str
) -> None:
    source = f"""
async def leak({parameters}):
    {binding}
    await send_message_delivery(message=_TURN_REJECTION_MESSAGES.get(reason, {fallback}))
"""
    assert _guard_offenders(source)


def test_guard_rejects_a_rebound_module_curated_table() -> None:
    source = """
_TURN_REJECTION_MESSAGES = attacker_controlled
async def leak(reason):
    await send_message_delivery(message=_TURN_REJECTION_MESSAGES.get(reason, "fallback"))
"""
    assert _guard_offenders(source)


def test_guard_rejects_a_conditionally_rebound_module_curated_table() -> None:
    source = """
_TURN_REJECTION_MESSAGES = {"busy": "safe"}
if flag:
    _TURN_REJECTION_MESSAGES = attacker_controlled
async def leak(reason):
    await send_message_delivery(message=_TURN_REJECTION_MESSAGES.get(reason, "fallback"))
"""
    assert _guard_offenders(source)


@pytest.mark.parametrize(
    "binding",
    [
        """def configure(value=(_TURN_REJECTION_MESSAGES := attacker_controlled)):
    pass""",
        "configure = lambda value=(_TURN_REJECTION_MESSAGES := attacker_controlled): value",
        """class Configure((_TURN_REJECTION_MESSAGES := attacker_controlled)):
    pass""",
    ],
    ids=["function-default", "lambda-default", "class-base"],
)
def test_guard_rejects_a_curated_table_rebound_during_definition(
    binding: str,
) -> None:
    source = f"""
_TURN_REJECTION_MESSAGES = {{"busy": "safe"}}
{binding}
async def leak(reason):
    await send_message_delivery(message=_TURN_REJECTION_MESSAGES.get(reason, "fallback"))
"""
    assert _guard_offenders(source)


def test_guard_rejects_curated_table_comprehension_and_star_import_shadows() -> None:
    source = """
_TURN_REJECTION_MESSAGES = {"busy": "safe"}
from evil import *
async def leak(reason, tables):
    await gather(*(send_message_delivery(
        message=_TURN_REJECTION_MESSAGES.get(reason, "fallback")
    ) for _TURN_REJECTION_MESSAGES in tables))
"""
    assert _guard_offenders(source)


def test_guard_resolves_a_single_name_producer_alias() -> None:
    source = """
async def leak(websocket):
    producer = send_message_delivery
    try:
        pass
    except Exception as error:
        await producer(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=f"leaked: {str(error)}",
            rejection_outcome="not_accepted",
        )
"""

    assert _guard_offenders(source)


def test_guard_resolves_a_module_level_producer_alias() -> None:
    source = """
producer = send_message_delivery

async def leak(websocket, error):
    await producer(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message=str(error),
        rejection_outcome="not_accepted",
    )
"""

    assert _guard_offenders(source)


@pytest.mark.parametrize(
    "source",
    [
        """async def leak(error, flag):
    if flag:
        producer = send_message_delivery
    else:
        producer = audit
    await producer(message=str(error))""",
        """producer = send_message_delivery
producer = send_message_delivery
async def leak(error):
    await producer(message=str(error))""",
    ],
    ids=["conditional-local", "duplicate-module"],
)
def test_guard_treats_any_preceding_producer_alias_as_egress(source: str) -> None:
    assert _guard_offenders(source)


@pytest.mark.parametrize(
    "source",
    [
        """async def leak(error):
    await producer(message=str(error))
producer = send_message_delivery""",
        """async def leak(error):
    await producer(message=str(error))
if flag:
    producer = send_message_delivery
else:
    producer = audit""",
        """async def leak(error):
    first = send_message_delivery
    producer = first
    await producer(message=str(error))""",
        """first = send_message_delivery
producer = first
async def leak(error):
    await producer(message=str(error))""",
        """async def outer(error):
    async def leak():
        await producer(message=str(error))
    producer = send_message_delivery
    return leak""",
    ],
    ids=[
        "module-binding-after-function",
        "conditional-module-binding-after-function",
        "chained-local",
        "chained-module",
        "outer-closure-binding-after-function",
    ],
)
def test_guard_resolves_deferred_and_chained_producer_aliases(source: str) -> None:
    assert _guard_offenders(source)


def test_guard_uses_the_last_unconditional_producer_alias_binding() -> None:
    source = """
producer = send_message_delivery
producer = audit
async def record(error):
    await producer(message=str(error))
"""
    assert not _guard_offenders(source)


def test_guard_does_not_treat_a_short_circuit_rebind_as_a_direct_alias() -> None:
    source = """
async def record(error):
    producer = send_message_delivery
    producer = producer and audit
    await producer(message=str(error))
"""
    assert not _guard_offenders(source)


@pytest.mark.parametrize(
    ("source", "is_egress"),
    [
        (
            """async def leak(error):
    first = send_message_delivery
    producer = first
    first = audit
    await producer(message=str(error))""",
            True,
        ),
        (
            """first = send_message_delivery
producer = first
first = audit
async def leak(error):
    await producer(message=str(error))""",
            True,
        ),
        (
            """async def record(error):
    first = audit
    producer = first
    first = send_message_delivery
    await producer(message=str(error))""",
            False,
        ),
        (
            """first = audit
producer = first
first = send_message_delivery
async def record(error):
    await producer(message=str(error))""",
            False,
        ),
    ],
    ids=[
        "local-captures-egress",
        "module-captures-egress",
        "local-captures-audit",
        "module-captures-audit",
    ],
)
def test_guard_resolves_chained_aliases_at_assignment_time(
    source: str, is_egress: bool
) -> None:
    assert bool(_guard_offenders(source)) is is_egress


def test_guard_keeps_outer_aliases_that_reach_an_early_nested_call() -> None:
    source = """
async def outer(error):
    producer = send_message_delivery
    async def leak():
        await producer(message=str(error))
    await leak()
    producer = audit
"""
    assert _guard_offenders(source)


@pytest.mark.parametrize(
    ("seed", "is_egress"), [("send_message_delivery", True), ("audit", False)]
)
@pytest.mark.parametrize("scope", ["local", "module"])
def test_guard_preserves_seeded_self_aliases(
    seed: str, is_egress: bool, scope: str
) -> None:
    module_binding = (
        f"producer = {seed}\nproducer = producer\n" if scope == "module" else ""
    )
    local_binding = (
        "" if scope == "module" else f"    producer = {seed}\n    producer = producer\n"
    )
    source = f"""{module_binding}async def record(error):
{local_binding}    await producer(message=str(error))
"""
    assert bool(_guard_offenders(source)) is is_egress


@pytest.mark.parametrize("scope", ["local", "module"])
def test_guard_resolves_a_producer_alias_rebound_after_the_sink(scope: str) -> None:
    binding = "producer = send_message_delivery\n" if scope == "module" else ""
    local_binding = (
        "" if scope == "module" else "    producer = send_message_delivery\n"
    )
    source = f"""{binding}async def leak(error):
{local_binding}    await producer(message=str(error))
    producer = audit
"""
    assert _guard_offenders(source)


def test_guard_resolves_a_producer_alias_on_the_next_loop_iteration() -> None:
    source = """
async def leak(items, error):
    producer = audit
    for item in items:
        await producer(message=str(error))
        producer = send_message_delivery
"""

    assert _guard_offenders(source), "the second iteration calls the producer"


@pytest.mark.parametrize(
    "message",
    [
        "client_error_message(ClientErrorCode.MESSAGE_PROCESSING_FAILED)",
        "CLIENT_SAFE_VALIDATION_ERROR",
    ],
)
def test_guard_accepts_the_exact_client_error_contract_import(message: str) -> None:
    source = f"""
from ..services.client_error_messages import (
    CLIENT_SAFE_VALIDATION_ERROR,
    ClientErrorCode,
    client_error_message,
)
async def safe():
    await send_message_delivery(message={message})
"""
    assert not _guard_offenders(source)


@pytest.mark.parametrize(
    "source",
    [
        """from attacker import client_error_message
async def leak(error):
    await send_message_delivery(message=client_error_message(error))""",
        """from attacker import CLIENT_SAFE_VALIDATION_ERROR
async def leak():
    await send_message_delivery(message=CLIENT_SAFE_VALIDATION_ERROR)""",
        """async def leak(client_error_message, error):
    await send_message_delivery(message=client_error_message(error))""",
        """async def leak(error):
    client_error_message = raw_message
    await send_message_delivery(message=client_error_message(error))""",
    ],
    ids=["wrong-builder-import", "wrong-constant-import", "parameter", "local"],
)
def test_guard_rejects_untrusted_client_error_contract_names(source: str) -> None:
    assert _guard_offenders(source)


@pytest.mark.parametrize(
    "source",
    [
        """async def leak(client_safe_error_message, error):
    await send_message_delivery(message=client_safe_error_message(error))""",
        """async def leak(error):
    client_safe_error_message = raw_message
    await send_message_delivery(message=client_safe_error_message(error))""",
        """async def leak(error, builders):
    await gather(*(send_message_delivery(message=client_safe_error_message(error))
        for client_safe_error_message in builders))""",
    ],
    ids=["parameter", "local", "comprehension"],
)
def test_guard_rejects_shadowed_safe_message_builders(source: str) -> None:
    assert _guard_offenders(source)


def test_guard_rejects_a_conditionally_rebound_safe_message_builder() -> None:
    source = """
def client_safe_error_message(error):
    return "safe"
if flag:
    client_safe_error_message = raw_message
async def leak(error):
    await send_message_delivery(message=client_safe_error_message(error))
"""
    assert _guard_offenders(source)


@pytest.mark.parametrize(
    ("fallback", "imported_constant", "accepted"),
    [
        ('f"raw: {error}"', "", False),
        ("raw_fallback", "", False),
        ('"fixed safe text"', "", True),
        (
            "CLIENT_SAFE_TASK_FAILURE",
            "from ..services.client_error_messages import CLIENT_SAFE_TASK_FAILURE\n",
            True,
        ),
    ],
    ids=["formatted-exception", "unresolved-name", "literal", "trusted-constant"],
)
def test_guard_checks_an_explicit_client_safe_fallback(
    fallback: str, imported_constant: str, accepted: bool
) -> None:
    source = f"""{imported_constant}
def client_safe_error_message(error, *, fallback="safe"):
    return fallback
async def safe(websocket, error):
    await manager.send_personal_message(
        {{
            "type": "error",
            "message": client_safe_error_message(error, fallback={fallback}),
        }},
        websocket,
    )
"""

    assert bool(_guard_offenders(source)) is not accepted


def test_guard_rejects_a_stream_builder_rebound_through_global() -> None:
    source = """
def create_stream_event(event_type, task_id, data):
    return {"type": event_type, "task_id": task_id, **data}
def configure(raw_builder):
    global create_stream_event
    create_stream_event = raw_builder
async def send(websocket, task_id):
    await manager.send_personal_message(
        create_stream_event("error", task_id, {"message": "fixed text"}),
        websocket,
    )
"""

    assert _guard_offenders(source)


def test_guard_rejects_a_safe_builder_rebound_in_a_decorator() -> None:
    source = """
def client_safe_error_message(error):
    return "safe"
@(client_safe_error_message := attacker_controlled)
def configure():
    pass
async def leak(error):
    await send_message_delivery(message=client_safe_error_message(error))
"""
    assert _guard_offenders(source)


def test_allowlist_does_not_apply_to_a_same_named_nested_handler() -> None:
    source = """
async def outer(websocket):
    async def handle_intervention():
        try:
            pass
        except RuntimeError as e:
            await manager.send_personal_message(
                {"type": "error", "message": f"Runtime error: {str(e)}"},
                websocket,
            )
"""

    assert _guard_offenders(source)


def test_guard_ignores_a_same_named_method_on_an_unrelated_receiver() -> None:
    source = """
async def audit_failure(audit, error):
    await audit.send_text(
        json.dumps({"type": "error", "message": str(error)})
    )
"""

    assert not _guard_offenders(source)


# The concurrent-delete race (TaskCommandTaskMissing between lookup and
# enqueue) is pinned in tests/web/services/test_task_command_transport.py:
# recovery-allowed returns None, and the strict path converts to
# ClientVisibleValidationError with "Task N not found" preserved.
# --- Round 5: runtime payload contracts for the changed egresses ------------


@pytest.mark.asyncio
async def test_terminal_command_failure_keeps_context_and_redacts_detail() -> None:
    """The kind prefix is ours; the exception text is not.

    The frontend renders ``message`` verbatim for ``agent_error``, so the
    redaction must not also delete the command context the client used to
    get, and the secret must not ride along in any field.
    """
    connection_manager = MagicMock()
    connection_manager.broadcast_to_task = AsyncMock()
    command = SimpleNamespace(
        kind=websocket_api.TaskCommandKind.PAUSE,
        task_id=7,
        command_id="cmd-7",
        payload={},
        target_run_id="run-7",
        attempt_count=1,
    )
    with patch.object(websocket_api, "manager", connection_manager):
        await websocket_api._broadcast_terminal_command_error(
            command, RuntimeError(f"lease lost at {SECRET}")
        )
    (payload, task_id) = connection_manager.broadcast_to_task.await_args.args
    assert task_id == 7
    assert SECRET not in repr(payload)
    assert payload["message"] == (
        f"Task command pause failed: {websocket_api.CLIENT_SAFE_VALIDATION_ERROR}"
    )
    assert payload["command_kind"] == "pause"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_name",
    ["_handle_pause_task_unserialized", "_handle_resume_task_unserialized"],
    ids=["pause", "resume"],
)
async def test_inner_command_validation_redacts_the_client_payload(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
) -> None:
    """The unserialized handlers' validation branch is a client egress too."""
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "run_db_io_cancellation_safe",
        AsyncMock(side_effect=ValueError(f"snapshot fault at {SECRET}")),
    )

    await getattr(websocket_api, handler_name)(
        MagicMock(),
        7,
        {"user": SimpleNamespace(id=1, is_admin=False)},
    )

    payloads = _client_payloads(connection_manager)
    assert payloads, "the handler must answer the client"
    assert SECRET not in repr(payloads)
    assert any(
        payload.get("message") == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
        for payload in payloads
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_name",
    ["_handle_pause_task_unserialized", "_handle_resume_task_unserialized"],
    ids=["pause", "resume"],
)
async def test_inner_command_runtime_failure_uses_safe_localizable_error(
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
) -> None:
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "run_db_io_cancellation_safe",
        AsyncMock(side_effect=RuntimeError(f"snapshot fault at {SECRET}")),
    )

    with pytest.raises(RuntimeError):
        await getattr(websocket_api, handler_name)(
            MagicMock(),
            7,
            {"user": SimpleNamespace(id=1, is_admin=False)},
        )

    payloads = _client_payloads(connection_manager)
    assert SECRET not in repr(payloads)
    assert any(
        payload.get("error_code") == "message_processing_failed"
        and payload.get("message") == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
        for payload in payloads
    )


@pytest.mark.asyncio
async def test_intervention_validation_redacts_the_client_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intervention validation branch is asserted, not just swept."""
    connection_manager = MagicMock()
    connection_manager.broadcast_to_task = AsyncMock(
        side_effect=ValueError(f"intervention fault at {SECRET}")
    )
    connection_manager.send_personal_message = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    await websocket_api.handle_intervention(
        MagicMock(), 7, {"step_id": "s1", "action": "approve"}
    )

    sent = [c.args[0] for c in connection_manager.send_personal_message.await_args_list]
    assert sent, "the handler must answer the sender"
    assert SECRET not in repr(sent)
    assert sent[-1]["message"] == websocket_api.CLIENT_SAFE_VALIDATION_ERROR


@pytest.mark.asyncio
async def test_intervention_runtime_failure_uses_a_safe_localizable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MagicMock(
        broadcast_to_task=AsyncMock(
            side_effect=RuntimeError(f"provider response at {SECRET}")
        ),
        send_personal_message=AsyncMock(),
    )
    monkeypatch.setattr(websocket_api, "manager", manager)

    await websocket_api.handle_intervention(
        MagicMock(),
        7,
        {"step_id": "step-1", "action": "continue"},
    )

    payload = manager.send_personal_message.await_args.args[0]
    assert SECRET not in repr(payload)
    assert payload == {
        "type": "error",
        "error_code": "message_processing_failed",
        "message": websocket_api.CLIENT_SAFE_VALIDATION_ERROR,
    }


@pytest.mark.asyncio
async def test_unexpected_execute_error_keeps_its_traceback(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """handle_execute_task re-raises into callers that log no stack."""

    class _Unexpected(Exception):
        pass

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    db = _direct_db_session()
    try:
        user = User(username="unexpected-owner", password_hash="hash")
        db.add(user)
        db.commit()
        task = Task(
            user_id=int(user.id),
            title="Unexpected",
            description="generic branch",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id, user_id = int(task.id), int(user.id)
    finally:
        db.close()
    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "schedule_existing_task_execution",
        AsyncMock(side_effect=_Unexpected("boom")),
    )

    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        with pytest.raises(_Unexpected):
            await websocket_api.handle_execute_task(
                MagicMock(),
                task_id,
                {"user": SimpleNamespace(id=user_id, is_admin=False)},
            )

    records = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert any(r.exc_info is not None for r in records), (
        "the traceback must be recorded where the detail is known"
    )


@pytest.mark.asyncio
async def test_unexpected_intervention_error_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """handle_intervention re-raises into public endpoints that swallow."""

    class _Unexpected(Exception):
        pass

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock(side_effect=_Unexpected("boom"))
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        with pytest.raises(_Unexpected):
            await websocket_api.handle_intervention(
                MagicMock(), 7, {"step_id": "s1", "action": "approve"}
            )

    records = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert any(r.exc_info is not None for r in records), (
        "the traceback must be recorded where the detail is known"
    )


@pytest.mark.asyncio
async def test_chat_validation_redacts_both_the_ack_and_the_broadcast(
    _test_db: None,
) -> None:
    """The inner chat validation branch answers on two sinks; assert both.

    The rejection ack goes to the sender and the task broadcast goes to every
    subscriber through a dict-spread payload. The AST guard now recognizes
    that shape; this runtime test additionally pins the actual audience and
    serialized values on both sinks.
    """
    db = _direct_db_session()
    try:
        owner = User(username="chat-validation-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Live control",
            description="validation branch",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task.runner_id = "rt5-runner"
        task.run_id = "rt5-run"
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        side_effect=ValueError(f"validation fault at {SECRET}")
    )
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = (
        websocket_api.ResumeReservationOutcome.RESERVED
    )

    def _fake_error_payload(task_id: int, message: str, **kwargs: object) -> dict:
        payload = {"type": "agent_error", "message": message, "task_id": task_id}
        if isinstance(kwargs.get("error_code"), str):
            payload["error_code"] = kwargs["error_code"]
        return payload

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            MagicMock(),
        ),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            MagicMock(side_effect=_fake_error_payload),
        ),
    ):
        await websocket_api._handle_chat_message_unserialized(
            MagicMock(),
            task_id,
            {
                "message": "trip the validation branch",
                "client_message_id": "chat-validation-secret",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
            },
        )

    personal = [c.args[0] for c in ws_manager.send_personal_message.await_args_list]
    broadcast = [c.args[0] for c in ws_manager.broadcast_to_task.await_args_list]
    everything = personal + broadcast
    assert everything, "the failure must be answered somewhere"
    assert SECRET not in repr(everything), everything

    rejected = [p for p in personal if p.get("type") == "message_rejected"]
    assert rejected and rejected[0]["message"] == (
        websocket_api.CLIENT_SAFE_VALIDATION_ERROR
    )
    task_errors = [b for b in broadcast if b.get("type") == "agent_error"]
    assert task_errors and task_errors[0]["message"] == (
        websocket_api.CLIENT_SAFE_VALIDATION_ERROR
    )


def _chat_runtime_error_harness(secret_error: Exception):
    """Shared live-control setup that raises ``secret_error`` at injection."""
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(side_effect=secret_error)
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = (
        websocket_api.ResumeReservationOutcome.RESERVED
    )

    def _fake_error_payload(task_id: int, message: str, **kwargs: object) -> dict:
        payload = {"type": "agent_error", "message": message, "task_id": task_id}
        if isinstance(kwargs.get("error_code"), str):
            payload["error_code"] = kwargs["error_code"]
        return payload

    return mgr, ws_manager, bg_mgr, _fake_error_payload


@pytest.mark.asyncio
async def test_runtime_error_is_redacted_and_coded_for_every_audience(
    _test_db: None,
) -> None:
    """Neither the initiator nor task subscribers may receive exception text."""
    db = _direct_db_session()
    try:
        owner = User(username="runtime-boundary-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Audience boundary",
            description="runtime branch",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task.runner_id = "rt6-runner"
        task.run_id = "rt6-run"
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    raised = RuntimeError(f"durable object scope={SECRET}")
    mgr, ws_manager, bg_mgr, fake_payload = _chat_runtime_error_harness(raised)

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            MagicMock(),
        ),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            MagicMock(side_effect=fake_payload),
        ),
    ):
        await websocket_api._handle_chat_message_unserialized(
            MagicMock(),
            task_id,
            {
                "message": "trip the runtime branch",
                "client_message_id": "runtime-boundary",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
            },
        )

    broadcast = [c.args[0] for c in ws_manager.broadcast_to_task.await_args_list]
    assert broadcast, "the task-wide notification must still go out"
    assert SECRET not in repr(broadcast), broadcast
    task_errors = [b for b in broadcast if b.get("type") == "agent_error"]
    assert task_errors and task_errors[0]["message"] == (
        websocket_api.CLIENT_SAFE_TASK_FAILURE
    )
    assert task_errors[0]["error_code"] == "task_execution_failed"

    personal = [c.args[0] for c in ws_manager.send_personal_message.await_args_list]
    assert SECRET not in repr(personal), personal
    rejected = [p for p in personal if p.get("type") == "message_rejected"]
    assert rejected == [
        {
            "type": "message_rejected",
            "client_message_id": "runtime-boundary",
            "turn_id": "runtime-boundary",
            "timestamp": rejected[0]["timestamp"],
            "message": websocket_api.CLIENT_SAFE_VALIDATION_ERROR,
            "error_code": "message_processing_failed",
            "rejection_outcome": "not_accepted",
        }
    ]


@pytest.mark.asyncio
async def test_execute_runtime_error_broadcast_is_redacted(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same audience boundary through handle_execute_task's runtime branch."""
    db = _direct_db_session()
    try:
        owner = User(username="exec-runtime-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Exec audience boundary",
            description="runtime branch",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "_read_task_error_payload_isolated",
        MagicMock(
            side_effect=lambda task_id, message, **kwargs: {
                "type": "agent_error",
                "message": message,
                "task_id": task_id,
                **(
                    {"error_code": kwargs["error_code"]}
                    if isinstance(kwargs.get("error_code"), str)
                    else {}
                ),
            }
        ),
    )
    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "schedule_existing_task_execution",
        AsyncMock(side_effect=RuntimeError(f"storage prefix {SECRET}")),
    )

    await websocket_api.handle_execute_task(
        MagicMock(),
        task_id,
        {"user": SimpleNamespace(id=owner_id, is_admin=False)},
    )

    broadcast = [
        c.args[0] for c in connection_manager.broadcast_to_task.await_args_list
    ]
    assert broadcast, "the task-wide notification must still go out"
    assert SECRET not in repr(broadcast), broadcast
    assert any(
        b.get("message") == websocket_api.CLIENT_SAFE_TASK_FAILURE for b in broadcast
    )
    assert any(b.get("error_code") == "task_execution_failed" for b in broadcast)
    personal = [
        c.args[0] for c in connection_manager.send_personal_message.await_args_list
    ]
    assert SECRET not in repr(personal), personal
    assert any(
        payload.get("type") == "error"
        and payload.get("message") == websocket_api.CLIENT_SAFE_TASK_FAILURE
        and payload.get("error_code") == "task_execution_failed"
        for payload in personal
    )


# --- Round 6: the durable command origin registry ---------------------------
#
# Personal detail from durable execution goes to the exact socket that
# submitted the command, verified to still be connected to that task - or
# nowhere. Origin is never inferred from task membership, actor id, or
# connection order.


@pytest.fixture()
def _clean_origins() -> Iterator[None]:
    saved = dict(websocket_api._command_origins._origins)
    websocket_api._command_origins._origins.clear()
    yield
    websocket_api._command_origins._origins.clear()
    websocket_api._command_origins._origins.update(saved)


def _pause_command(
    task_id: int = 7, command_id: str = "pause:origin"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        task_id=task_id,
        actor_user_id=1,
        command_id=command_id,
        kind=websocket_api.TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
        target_run_id=None,
        attempt_count=1,
        failure_count=0,
        defer_count=0,
    )


def _origin_test_manager(registered: set) -> MagicMock:
    """A manager mock whose registration check is membership in ``registered``."""
    m = MagicMock()
    m.send_personal_message = AsyncMock()
    m.broadcast_to_task = AsyncMock()
    m.is_connection_registered = MagicMock(
        side_effect=lambda ws, task_id: ws in registered
    )
    return m


async def _run_pause_to_runtime_error(
    monkeypatch: pytest.MonkeyPatch, manager_mock: MagicMock, command
) -> None:
    monkeypatch.setattr(websocket_api, "manager", manager_mock)
    monkeypatch.setattr(
        websocket_api,
        "_load_command_actor",
        lambda actor_user_id: SimpleNamespace(id=actor_user_id or 1, is_admin=False),
    )
    monkeypatch.setattr(
        websocket_api, "task_has_live_foreign_runner", lambda task_id: False
    )
    import xagent.web.services.task_setup_snapshot as snapshot_module

    monkeypatch.setattr(
        snapshot_module,
        "load_task_setup_snapshot_sync",
        MagicMock(side_effect=RuntimeError(f"storage fault at {SECRET}")),
    )
    with pytest.raises(RuntimeError):
        await websocket_api._execute_durable_task_command(command)


def _personal_targets(manager_mock: MagicMock) -> list[tuple[dict, object]]:
    return [
        (c.args[0], c.args[1])
        for c in manager_mock.send_personal_message.await_args_list
        if c.args and isinstance(c.args[0], dict)
    ]


@pytest.mark.asyncio
async def test_durable_runtime_error_is_safe_for_the_verified_origin(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
) -> None:
    """The verified command origin receives only the stable safe contract."""
    public, owner_origin, sse = (
        MagicMock(name="public"),
        MagicMock(name="origin"),
        MagicMock(name="sse"),
    )
    sse.is_broadcast_only = True
    manager_mock = _origin_test_manager(registered={public, owner_origin, sse})
    command = _pause_command()
    websocket_api._command_origins.register(
        command.command_id, owner_origin, command.task_id
    )

    await _run_pause_to_runtime_error(monkeypatch, manager_mock, command)

    personal = _personal_targets(manager_mock)
    assert SECRET not in repr(personal)
    safe_sends = [
        (payload, ws)
        for payload, ws in personal
        if payload.get("error_code") == "message_processing_failed"
    ]
    assert safe_sends, "the verified origin must receive the safe error"
    assert all(ws is owner_origin for _, ws in safe_sends), safe_sends
    assert not any(ws is public for _, ws in _personal_targets(manager_mock)), (
        "the public socket must receive nothing personal"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "degrade_case",
    ["no-registration", "origin-disconnected", "wrong-task"],
)
async def test_durable_safe_error_degrades_when_origin_is_unverifiable(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
    degrade_case: str,
) -> None:
    """Worker restart/handoff, disconnect, and task mismatch all degrade safely.

    No registration entry (a different worker executed the command), an
    origin that has since disconnected, or an entry recorded for another
    task: in every case the raw text reaches no socket at all.
    """
    public, owner_origin = MagicMock(name="public"), MagicMock(name="origin")
    registered = {public, owner_origin}
    command = _pause_command()
    if degrade_case == "origin-disconnected":
        websocket_api._command_origins.register(
            command.command_id, owner_origin, command.task_id
        )
        registered = {public}
    elif degrade_case == "wrong-task":
        websocket_api._command_origins.register(
            command.command_id, owner_origin, command.task_id + 1
        )
    manager_mock = _origin_test_manager(registered=registered)

    await _run_pause_to_runtime_error(monkeypatch, manager_mock, command)

    personal = _personal_targets(manager_mock)
    assert SECRET not in repr(personal)
    safe_targets = [
        ws
        for payload, ws in personal
        if payload.get("error_code") == "message_processing_failed"
    ]
    assert all(
        isinstance(ws, websocket_api._DiscardingCommandWebSocket) for ws in safe_targets
    ), f"durable error rerouted to a real socket: {safe_targets}"


@pytest.mark.asyncio
async def test_origin_entry_dies_with_its_command_or_socket(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
) -> None:
    """Lifecycle: terminal outcomes and disconnects both clear the entry."""
    origins = websocket_api._command_origins

    # Terminal rejection clears it.
    command = _pause_command(command_id="pause:cleanup")
    socket = MagicMock(name="origin")
    origins.register(command.command_id, socket, command.task_id)
    monkeypatch.setattr(
        websocket_api,
        "_execute_durable_task_command",
        AsyncMock(side_effect=websocket_api.TaskCommandRejected("done")),
    )
    with pytest.raises(websocket_api.TaskCommandRejected):
        await websocket_api.execute_durable_task_command(command)
    assert origins.resolve(command.command_id, command.task_id) is None
    assert not origins.has(command.command_id, command.task_id)

    # A deferral that will retry keeps it; exhaustion clears it.
    origins.register(command.command_id, socket, command.task_id)
    monkeypatch.setattr(
        websocket_api,
        "_execute_durable_task_command",
        AsyncMock(side_effect=websocket_api.ClientVisibleTaskCommandDeferred("wait")),
    )
    monkeypatch.setattr(websocket_api, "manager", _origin_test_manager({socket}))
    with pytest.raises(websocket_api.TaskCommandDeferred):
        await websocket_api.execute_durable_task_command(command)
    assert origins.has(command.command_id, command.task_id), (
        "retrying deferral keeps the origin"
    )
    exhausted = _pause_command(command_id="pause:cleanup")
    exhausted.defer_count = websocket_api.max_command_defers()
    with pytest.raises(websocket_api.TaskCommandDeferred):
        await websocket_api.execute_durable_task_command(exhausted)
    assert not origins.has(command.command_id, command.task_id)

    # Disconnect clears every entry for that socket.
    origins.register("a:1", socket, 7)
    origins.register("b:2", socket, 8)
    real_manager = websocket_api.ConnectionManager()
    real_manager.disconnect(socket)
    assert not origins.has("a:1", 7) and not origins.has("b:2", 8), (
        "disconnect must clear the socket's entries"
    )


@pytest.mark.asyncio
async def test_durable_chat_runtime_error_is_safe_for_verified_origin(
    _test_db: None,
) -> None:
    """Durable chat sends one safe, coded bubble to its verified origin."""
    db = _direct_db_session()
    try:
        owner = User(username="durable-detail-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Durable detail",
            description="runtime branch, durable path",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task.runner_id = "rt7-runner"
        task.run_id = "rt7-run"
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    raised = RuntimeError(f"durable object scope={SECRET}")
    mgr, ws_manager, bg_mgr, fake_payload = _chat_runtime_error_harness(raised)
    origin_socket = MagicMock(name="verified-origin")

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            MagicMock(),
        ),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            MagicMock(side_effect=fake_payload),
        ),
    ):
        await websocket_api._handle_chat_message_unserialized(
            origin_socket,
            task_id,
            {
                "message": "durable runtime failure",
                "client_message_id": "durable-detail",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
                "_durable_ack_sent": True,
            },
        )

    broadcast = [c.args[0] for c in ws_manager.broadcast_to_task.await_args_list]
    assert broadcast and SECRET not in repr(broadcast), broadcast
    personal = [
        (c.args[0], c.args[1]) for c in ws_manager.send_personal_message.await_args_list
    ]
    # The suppressed ack means no message_rejected; the safe error bubble is
    # the sender's only copy and goes to the socket the executor resolved.
    assert not any(p.get("type") == "message_rejected" for p, _ in personal)
    assert SECRET not in repr(personal)
    safe = [
        (p, ws)
        for p, ws in personal
        if p.get("error_code") == "message_processing_failed"
    ]
    assert safe, "the verified origin must receive the safe error bubble"
    assert all(ws is origin_socket for _, ws in safe), safe


@pytest.mark.asyncio
async def test_execute_safe_error_reaches_the_ingress_socket(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute keeps its real ingress socket but never exposes raw detail."""
    db = _direct_db_session()
    try:
        owner = User(username="exec-detail-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Exec detail",
            description="runtime branch",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "_read_task_error_payload_isolated",
        MagicMock(
            side_effect=lambda task_id, message, **kwargs: {
                "type": "agent_error",
                "message": message,
                "task_id": task_id,
                **(
                    {"error_code": kwargs["error_code"]}
                    if isinstance(kwargs.get("error_code"), str)
                    else {}
                ),
            }
        ),
    )
    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "schedule_existing_task_execution",
        AsyncMock(side_effect=RuntimeError(f"storage prefix {SECRET}")),
    )

    ingress = MagicMock(name="ingress")
    await websocket_api.handle_execute_task(
        ingress,
        task_id,
        {"user": SimpleNamespace(id=owner_id, is_admin=False)},
    )

    broadcast = [
        c.args[0] for c in connection_manager.broadcast_to_task.await_args_list
    ]
    assert broadcast and SECRET not in repr(broadcast), broadcast
    personal = [
        (c.args[0], c.args[1])
        for c in connection_manager.send_personal_message.await_args_list
    ]
    assert personal and SECRET not in repr(personal)
    assert all(ws is ingress for _, ws in personal), personal
    assert any(
        payload.get("error_code") == "task_execution_failed"
        and payload.get("message") == websocket_api.CLIENT_SAFE_TASK_FAILURE
        for payload, _ in personal
    )


@pytest.mark.asyncio
async def test_preview_unknown_message_answers_on_the_wire_without_echo() -> None:
    """G16: endpoint-level coverage of the unknown-message response.

    Feeds an unknown type through the real receive loop and decodes the
    actual send_text JSON, so a deleted branch, wrong wiring, or a
    reintroduced echo of the client's message type all fail here.
    """
    from fastapi import WebSocketDisconnect

    mock_websocket = AsyncMock()
    mock_websocket.state = MagicMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    hostile_type = f"nope-{SECRET}"
    mock_websocket.receive_text.side_effect = [
        json.dumps({"type": hostile_type}),
        WebSocketDisconnect(),
    ]

    with patch(
        "xagent.web.api.websocket.get_authenticated_user", return_value=mock_user
    ):
        await websocket_api.websocket_build_preview_endpoint(mock_websocket)

    sent = [json.loads(c.args[0]) for c in mock_websocket.send_text.call_args_list]
    errors = [p for p in sent if p.get("type") == "error"]
    assert errors == [{"type": "error", "message": "Unknown message type"}] or (
        len(errors) == 1 and errors[0]["message"] == "Unknown message type"
    ), errors
    assert hostile_type not in repr(sent), "the client's type must not echo back"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "extra"),
    [
        ("handle_chat_message", {"client_message_id": "reg-1"}),
        ("handle_pause_task", {}),
        ("handle_resume_task", {}),
    ],
    ids=["chat", "pause", "resume"],
)
async def test_ingress_handlers_register_the_command_origin(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
    handler_name: str,
    extra: dict,
) -> None:
    """The creating ingress binds; deleting the line degrades senders silently."""
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    connection_manager.is_connection_registered = MagicMock(return_value=True)
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    enqueued = SimpleNamespace(
        command_id=41,
        client_command_id="cmd:origin-reg",
        payload_matches=True,
        status="claimed",
        created=True,
    )
    monkeypatch.setattr(
        websocket_api,
        "_enqueue_websocket_task_command",
        AsyncMock(return_value=enqueued),
    )
    monkeypatch.setattr(websocket_api, "dispatch_task_command_promptly", AsyncMock())

    ingress = MagicMock(name="ingress")
    await getattr(websocket_api, handler_name)(
        ingress,
        7,
        {"user": SimpleNamespace(id=1, is_admin=False), **extra},
    )

    assert websocket_api._command_origins.has("cmd:origin-reg", 7), (
        f"{handler_name} must register its origin"
    )
    assert websocket_api._command_origins.resolve("cmd:origin-reg", 7) is ingress


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "extra"),
    [
        ("handle_chat_message", {"client_message_id": "dup-1"}),
        ("handle_pause_task", {}),
        ("handle_resume_task", {}),
    ],
    ids=["chat", "pause", "resume"],
)
async def test_a_duplicate_enqueue_never_binds_the_origin(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
    handler_name: str,
    extra: dict,
) -> None:
    """A payload-matching duplicate (created=False) must not acquire origin.

    This is the P1 blocker: a duplicate - a co-tenant resubmission, one after
    the creator disconnected, or one handled on another worker - reaches these
    handlers with a valid command_id but created=False. It still dispatches
    (idempotent), but it must never register, or the durable executor could
    resolve it and send the creator's raw detail to the wrong socket.
    """
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    connection_manager.is_connection_registered = MagicMock(return_value=True)
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    dispatch = AsyncMock()
    monkeypatch.setattr(websocket_api, "dispatch_task_command_promptly", dispatch)
    monkeypatch.setattr(
        websocket_api,
        "_enqueue_websocket_task_command",
        AsyncMock(
            return_value=SimpleNamespace(
                command_id=41,
                client_command_id="dup:cmd",
                payload_matches=True,
                status="claimed",
                created=False,
            )
        ),
    )

    duplicate = MagicMock(name="duplicate-ingress")
    await getattr(websocket_api, handler_name)(
        duplicate,
        7,
        {"user": SimpleNamespace(id=1, is_admin=False), **extra},
    )

    assert not websocket_api._command_origins.has("dup:cmd", 7), (
        "a duplicate must never bind the origin"
    )
    assert dispatch.await_count == 1, "the duplicate still dispatches idempotently"


def test_a_resubmitted_command_id_cannot_capture_another_senders_origin(
    _clean_origins: None,
) -> None:
    """First registration wins (preflight PoC: co-tenant origin hijack).

    On a public/share task every visitor carries the owner principal, so the
    enqueue dedupe returns the in-flight row for a resubmission of the same
    command_id and the second connection would otherwise reach `register` and
    overwrite the origin - redirecting the first sender's error detail to the
    attacker. The registry must keep the original.
    """
    origins = websocket_api._command_origins
    victim, attacker = MagicMock(name="victim"), MagicMock(name="attacker")

    origins.register("shared-cmd", victim, 7)
    origins.register("shared-cmd", attacker, 7)  # resubmission on the same task

    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("shared-cmd", 7) is victim, (
            "the attacker must not capture the victim's origin"
        )

    # Re-registering the same socket stays idempotent.
    origins.register("shared-cmd", victim, 7)
    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("shared-cmd", 7) is victim


def test_same_command_id_on_two_tasks_is_isolated(_clean_origins: None) -> None:
    """command_id is unique only per task, so the key is (task_id, command_id).

    A shared id must not let one task's registration void or answer the
    other's - the DB carries a (task_id, command_id) uniqueness constraint,
    not command_id alone.
    """
    origins = websocket_api._command_origins
    sock_a, sock_b = MagicMock(name="task-a"), MagicMock(name="task-b")
    origins.register("1", sock_a, 100)
    origins.register("1", sock_b, 200)

    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("1", 100) is sock_a
        assert origins.resolve("1", 200) is sock_b

    # Discarding one task's entry leaves the other intact.
    origins.discard_command("1", 100)
    assert not origins.has("1", 100)
    assert origins.has("1", 200)


@pytest.mark.asyncio
async def test_live_chat_runtime_error_sends_one_safe_rejection(
    _test_db: None,
) -> None:
    """The live path returns one coded rejection and never exposes detail."""
    db = _direct_db_session()
    try:
        owner = User(username="no-double-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="No double",
            description="live runtime branch",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task.runner_id = "rt8-runner"
        task.run_id = "rt8-run"
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    raised = RuntimeError(f"live fault {SECRET}")
    mgr, ws_manager, bg_mgr, fake_payload = _chat_runtime_error_harness(raised)
    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch("xagent.web.api.websocket.mark_user_message_delivery_sync", MagicMock()),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            MagicMock(side_effect=fake_payload),
        ),
    ):
        await websocket_api._handle_chat_message_unserialized(
            MagicMock(name="live-sender"),
            task_id,
            {
                "message": "live runtime failure",
                "client_message_id": "no-double",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
                # no _durable_ack_sent: this is the live path
            },
        )

    personal = [
        c.args[0]
        for c in ws_manager.send_personal_message.await_args_list
        if isinstance(c.args[0], dict)
    ]
    assert SECRET not in repr(personal)
    safe_rejections = [
        p
        for p in personal
        if p.get("type") == "message_rejected"
        and p.get("error_code") == "message_processing_failed"
    ]
    assert len(safe_rejections) == 1, safe_rejections


def test_a_later_duplicate_cannot_rebind_after_the_creator_disconnects(
    _clean_origins: None,
) -> None:
    """P1 disconnect/rebind: once the creator's entry is gone, no bind at all.

    First-registration-wins protects a live entry, but the sharper case is the
    creator disconnecting (its entry cleared) and a later duplicate arriving.
    Because only the creating ingress registers, the duplicate never calls
    register, so resolve stays empty rather than pointing at the late arrival.
    """
    origins = websocket_api._command_origins
    creator = MagicMock(name="creator")
    origins.register("cmd", creator, 7)

    # creator disconnects
    real_manager = websocket_api.ConnectionManager()
    real_manager.disconnect(creator)
    assert not origins.has("cmd", 7)

    # a later duplicate is created=False at the handler, so it never registers;
    # the registry stays empty and the executor will safe-discard.
    assert not origins.has("cmd", 7)
    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("cmd", 7) is None


def test_registry_is_bounded_by_lru_eviction(_clean_origins: None) -> None:
    """P2: entries whose commands run on another worker cannot grow unbounded.

    A long-lived socket whose commands are always claimed elsewhere would never
    get local cleanup. The store is an LRU capped at _MAX_ORIGINS; the oldest
    entry is evicted on overflow, which only makes resolve miss (safe discard),
    never reroutes detail.
    """
    origins = websocket_api._command_origins
    cap = websocket_api._CommandOriginRegistry._MAX_ORIGINS
    socket = MagicMock(name="long-lived")

    for i in range(cap + 50):
        origins.register(f"cmd-{i}", socket, 7)

    assert len(origins._origins) == cap, "the store must not grow past the cap"
    # oldest evicted, newest retained
    assert not origins.has("cmd-0", 7)
    assert origins.has(f"cmd-{cap + 49}", 7)
    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("cmd-0", 7) is None  # evicted -> safe discard


@pytest.mark.parametrize(
    ("types", "reassignment", "blocked"),
    [
        ('"final_answer_start", "final_answer_error"', "", False),
        ('"final_answer_start", "error"', "", True),
        ('"final_answer_start"', 'kind = "error"', True),
    ],
)
def test_assigned_final_answer_envelope_type_narrowing(types, reassignment, blocked):
    source = f"""
def create_final_answer_stream_event(event_type, task_id, data):
    return {{"type": event_type, **data}}

async def send(kind, raw):
    if kind in {{{types}}}:
        {reassignment or "pass"}
        envelope = create_final_answer_stream_event(kind, 1, {{"message": str(raw)}})
        await manager.broadcast_to_task(envelope, 1)
"""
    assert bool(_guard_offenders(source)) is blocked
