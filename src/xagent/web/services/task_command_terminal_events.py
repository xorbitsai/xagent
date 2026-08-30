"""Durable, cross-worker delivery of terminal task-command outcomes.

Command disposition code stages an append-only event in its own transaction.
Each web worker independently tails that shared log for its local sockets, so
delivery never depends on which process happened to claim the command.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import uuid
import weakref
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.database import get_session_local
from ..models.task import Task
from ..models.task_command import TaskExecutionCommand
from ..models.task_command_terminal_event import TaskCommandTerminalEvent
from ..utils.db_timezone import format_datetime_for_api
from .db_runtime import await_task_settlement, run_db_io_cancellation_safe

logger = logging.getLogger(__name__)

MAX_TERMINAL_EVENT_CURSOR = (1 << 31) - 1


class TerminalTaskEventMessageCode(str, enum.Enum):
    """Closed vocabulary for rendering terminal outcomes without stored text."""

    TASK_COMMAND_FAILED = "task_command_failed"
    TASK_COMMAND_DEFERRED = "task_command_deferred"
    EXTERNAL_CANCEL_NOT_APPLIED = "external_cancel_not_applied"
    EXTERNAL_TURN_INTERRUPTED = "external_turn_interrupted"


@dataclass(frozen=True)
class TerminalTaskEventDraft:
    """Client-safe terminal outcome staged by a command disposition."""

    message_code: TerminalTaskEventMessageCode | None
    resend_safe: bool
    include_command_identity: bool = True


@dataclass(frozen=True)
class TerminalTaskEventPrincipal:
    """Authenticated owner identity used to authorize an event subscription."""

    user_id: int
    is_admin: bool


@dataclass(frozen=True)
class TerminalTaskEvent:
    """Detached event safe to move from a DB worker to an async sink."""

    cursor: int
    event_id: str
    task_id: int
    task_run_id: str | None
    task_state_version: int | None
    command_id: str
    command_kind: str
    actor_user_id: int | None
    task_owner_user_id: int
    outcome_version: int
    outcome: str
    message_code: TerminalTaskEventMessageCode | None
    resend_safe: bool
    include_command_identity: bool
    created_at: datetime


TerminalTaskEventSink = Callable[[TerminalTaskEvent], Awaitable[None]]
_DRAFT_ATTRIBUTE = "_xagent_terminal_task_event_draft"
_CANCEL_COMMAND_KIND = "cancel"
_EXTERNAL_COMMAND_SCOPE = "external"


def is_external_cancel_command(*, kind: str, scope: object) -> bool:
    """Return whether strict command kind/scope values name an external cancel.

    The helper accepts normalized primitives so durable ORM rows and live
    command snapshots can share one disclosure-policy classifier without
    importing each other's modules.
    """

    return kind == _CANCEL_COMMAND_KIND and scope == _EXTERNAL_COMMAND_SCOPE


class TerminalTaskEventAccessDenied(PermissionError):
    """The principal may not subscribe to this task's terminal outcomes."""


def bind_terminal_event_draft(
    error: BaseException,
    draft: TerminalTaskEventDraft,
) -> None:
    """Attach client-safe presentation metadata without performing delivery."""

    setattr(error, _DRAFT_ATTRIBUTE, draft)


def terminal_event_draft_for_error(
    error: BaseException,
) -> TerminalTaskEventDraft | None:
    """Read presentation metadata previously attached by an executor adapter."""

    draft = getattr(error, _DRAFT_ATTRIBUTE, None)
    return draft if isinstance(draft, TerminalTaskEventDraft) else None


def stage_terminal_event(
    db: Session,
    *,
    command_db_id: int,
    draft: TerminalTaskEventDraft | None = None,
) -> TaskCommandTerminalEvent:
    """Stage one idempotent event without committing the caller's transaction.

    The command must already have a terminal disposition in this transaction.
    The caller owns the commit, which makes disposition and event one recovery
    boundary instead of two best-effort operations. Run correlation is copied
    exclusively from the immutable command-acceptance snapshot; reading the
    task's current run or state version here could associate an old command
    outcome with a newer interaction.

    The insert runs in a savepoint so a concurrent natural-key winner can be
    adopted without poisoning the caller's transaction. Other integrity
    failures still propagate after the savepoint has been rolled back.
    """

    snapshot = (
        db.query(TaskExecutionCommand, Task)
        .join(Task, Task.id == TaskExecutionCommand.task_id)
        .filter(TaskExecutionCommand.id == command_db_id)
        .populate_existing()
        .one_or_none()
    )
    if snapshot is None:
        raise ValueError(f"Task command {command_db_id} does not exist")
    command, task = snapshot
    if command.status not in {"completed", "failed"}:
        raise ValueError(
            f"Task command {command_db_id} is not terminal: {command.status}"
        )
    outcome_version = int(command.attempt_count or 0)
    outcome = str(command.status)
    if draft is None:
        failed = command.status == "failed"
        draft = TerminalTaskEventDraft(
            message_code=(
                TerminalTaskEventMessageCode.TASK_COMMAND_FAILED if failed else None
            ),
            resend_safe=False,
        )
    scope = command.payload.get("scope") if isinstance(command.payload, dict) else None

    event = TaskCommandTerminalEvent(
        event_id=str(uuid.uuid4()),
        task_command_id=int(command.id),
        task_id=int(command.task_id),
        task_run_id=command.target_run_id,
        task_state_version=(
            int(command.target_state_version)
            if command.target_state_version is not None
            else None
        ),
        command_id=str(command.command_id),
        command_kind=str(command.kind),
        actor_user_id=(
            int(command.actor_user_id) if command.actor_user_id is not None else None
        ),
        task_owner_user_id=int(task.user_id),
        outcome_version=outcome_version,
        outcome=outcome,
        message_code=(draft.message_code.value if draft.message_code else None),
        resend_safe=bool(draft.resend_safe),
        include_command_identity=bool(
            draft.include_command_identity
            and not is_external_cancel_command(kind=str(command.kind), scope=scope)
        ),
    )
    try:
        with db.begin_nested():
            db.add(event)
            db.flush()
    except IntegrityError:
        existing = (
            db.query(TaskCommandTerminalEvent)
            .filter(
                TaskCommandTerminalEvent.task_command_id == command_db_id,
                TaskCommandTerminalEvent.outcome_version == outcome_version,
            )
            .one_or_none()
        )
        if existing is None:
            raise
        return existing
    return event


def _authorize_and_resolve_cursor(
    principal: TerminalTaskEventPrincipal,
    task_id: int,
    after_event_id: int | None,
    *,
    allow_missing_task: bool = False,
) -> int | None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        owner_id = db.query(Task.user_id).filter(Task.id == task_id).scalar()
        if owner_id is None:
            if allow_missing_task:
                return None
            raise TerminalTaskEventAccessDenied(
                f"Task {task_id} is not available to this principal"
            )
        if not principal.is_admin and int(owner_id) != principal.user_id:
            raise TerminalTaskEventAccessDenied(
                f"Task {task_id} is not available to this principal"
            )
        if after_event_id is not None:
            if after_event_id < 0:
                raise ValueError("after_event_id must be non-negative")
            return min(after_event_id, MAX_TERMINAL_EVENT_CURSOR)
        latest = (
            db.query(TaskCommandTerminalEvent.id)
            .filter(TaskCommandTerminalEvent.task_id == task_id)
            .order_by(TaskCommandTerminalEvent.id.desc())
            .limit(1)
            .scalar()
        )
        return int(latest or 0)


async def resolve_terminal_task_event_cursor(
    *,
    principal: TerminalTaskEventPrincipal,
    task_id: int,
    after_event_id: int | None,
    allow_missing_task: bool = False,
) -> int | None:
    """Authorize a task and fix the replay baseline before other awaits."""

    return await run_db_io_cancellation_safe(
        lambda: _authorize_and_resolve_cursor(
            principal,
            task_id,
            after_event_id,
            allow_missing_task=allow_missing_task,
        )
    )


def _decode_message_code(
    raw_code: str | None,
    *,
    outcome: str,
) -> TerminalTaskEventMessageCode | None:
    if raw_code is None:
        return None
    try:
        return TerminalTaskEventMessageCode(raw_code)
    except ValueError:
        logger.error("Ignoring unknown terminal task event message code: %s", raw_code)
        return (
            TerminalTaskEventMessageCode.TASK_COMMAND_FAILED
            if outcome == "failed"
            else None
        )


def _load_events(after_cursor_by_task: dict[int, int]) -> list[TerminalTaskEvent]:
    if not after_cursor_by_task:
        return []
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        # SQLite rejects deeply nested expressions around 1000 terms. Keep
        # each cursor query comfortably below that bound while still polling
        # every subscribed task in this cycle.
        rows: list[TaskCommandTerminalEvent] = []
        cursor_items = list(after_cursor_by_task.items())
        for offset in range(0, len(cursor_items), 200):
            cursor_predicates = [
                and_(
                    TaskCommandTerminalEvent.task_id == task_id,
                    TaskCommandTerminalEvent.id > after_cursor,
                )
                for task_id, after_cursor in cursor_items[offset : offset + 200]
            ]
            rows.extend(
                db.query(TaskCommandTerminalEvent)
                .filter(or_(*cursor_predicates))
                .order_by(TaskCommandTerminalEvent.id.asc())
                .limit(1000)
                .all()
            )
        rows.sort(key=lambda row: int(row.id))
        return [
            TerminalTaskEvent(
                cursor=int(row.id),
                event_id=str(row.event_id),
                task_id=int(row.task_id),
                task_run_id=(
                    str(row.task_run_id) if row.task_run_id is not None else None
                ),
                task_state_version=(
                    int(row.task_state_version)
                    if row.task_state_version is not None
                    else None
                ),
                command_id=str(row.command_id),
                command_kind=str(row.command_kind),
                actor_user_id=(
                    int(row.actor_user_id) if row.actor_user_id is not None else None
                ),
                task_owner_user_id=int(row.task_owner_user_id),
                outcome_version=int(row.outcome_version),
                outcome=str(row.outcome),
                message_code=_decode_message_code(
                    str(row.message_code) if row.message_code is not None else None,
                    outcome=str(row.outcome),
                ),
                resend_safe=bool(row.resend_safe),
                include_command_identity=bool(row.include_command_identity),
                created_at=row.created_at,
            )
            for row in rows
        ]


def render_terminal_task_event_message(event: TerminalTaskEvent) -> str | None:
    """Render only server-owned message codes and validated command kinds."""

    code = event.message_code
    if code is None:
        return None
    if code == TerminalTaskEventMessageCode.EXTERNAL_CANCEL_NOT_APPLIED:
        return "Stopping this response didn't go through — please try again."
    if code == TerminalTaskEventMessageCode.EXTERNAL_TURN_INTERRUPTED:
        return "This response was interrupted."
    kind = (
        event.command_kind
        if event.command_kind
        in {
            "message",
            "pause",
            "resume",
            "cancel",
        }
        else "unknown"
    )
    if code == TerminalTaskEventMessageCode.TASK_COMMAND_DEFERRED:
        detail = "The command could not be handed to the active task worker."
    else:
        detail = "Task execution failed."
    return f"Task command {kind} failed: {detail}"


def terminal_task_event_payload(event: TerminalTaskEvent) -> dict[str, Any]:
    """Project a durable event onto the client protocol without stored text."""

    message = render_terminal_task_event_message(event)
    timestamp = format_datetime_for_api(event.created_at)
    if timestamp is None:
        raise ValueError("Terminal task event is missing its creation timestamp")
    correlated = event.task_state_version is not None
    payload: dict[str, Any] = {
        "type": (
            "agent_error"
            if correlated and event.outcome == "failed" and message is not None
            else "task_command_outcome"
        ),
        "terminal_event_id": event.event_id,
        "terminal_event_cursor": event.cursor,
        "terminal_event_schema_version": 1,
        "outcome": event.outcome,
        "outcome_version": event.outcome_version,
        "resend_safe": event.resend_safe,
        "task_id": event.task_id,
        "timestamp": timestamp,
    }
    if correlated:
        payload["run_id"] = event.task_run_id
        payload["state_version"] = event.task_state_version
    if message is not None and payload["type"] == "agent_error":
        payload["message"] = message
    if event.include_command_identity:
        payload["command_id"] = event.command_id
        payload["command_kind"] = event.command_kind
    return payload


@dataclass
class _Subscriber:
    task_id: int
    sink: TerminalTaskEventSink
    cursor: int
    principal_user_id: int
    principal_is_admin: bool


class TerminalTaskEventSubscription:
    """Handle whose close operation detaches one local event sink."""

    def __init__(self, hub: "TerminalTaskEventHub", subscription_id: int) -> None:
        self._hub = hub
        self._subscription_id = subscription_id
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        if self._closed:
            return
        cleanup = asyncio.create_task(self._hub._detach(self._subscription_id))
        _, cancellation = await await_task_settlement(cleanup)
        self._closed = True
        if cancellation is not None:
            raise cancellation


class TerminalTaskEventHub:
    """One per-worker poller that fans durable events into local sinks."""

    def __init__(self, *, poll_interval_seconds: float = 0.5) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._poll_interval_seconds = poll_interval_seconds
        self._subscribers: dict[int, _Subscriber] = {}
        self._next_subscription_id = 1
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._poller: asyncio.Task[None] | None = None
        self._delivery_tasks: dict[int, asyncio.Task[None]] = {}
        self._closed = False

    async def attach_terminal_events(
        self,
        *,
        principal: TerminalTaskEventPrincipal,
        task_id: int,
        sink: TerminalTaskEventSink,
        after_event_id: int | None = None,
    ) -> TerminalTaskEventSubscription:
        """Authorize and attach a local sink, optionally replaying after a cursor."""

        cursor = await resolve_terminal_task_event_cursor(
            principal=principal,
            task_id=task_id,
            after_event_id=after_event_id,
        )
        assert cursor is not None
        async with self._lock:
            if self._closed:
                raise RuntimeError("Terminal task event hub is closed")
            subscription_id = self._next_subscription_id
            self._next_subscription_id += 1
            self._subscribers[subscription_id] = _Subscriber(
                task_id=task_id,
                sink=sink,
                cursor=cursor,
                principal_user_id=principal.user_id,
                principal_is_admin=principal.is_admin,
            )
            if self._poller is None or self._poller.done():
                self._poller = asyncio.create_task(self._run())
            self._wake.set()
        return TerminalTaskEventSubscription(self, subscription_id)

    async def _detach(self, subscription_id: int) -> None:
        async with self._lock:
            self._subscribers.pop(subscription_id, None)
            delivery = self._delivery_tasks.pop(subscription_id, None)
            self._wake.set()
        if delivery is not None and delivery is not asyncio.current_task():
            delivery.cancel()
            await asyncio.gather(delivery, return_exceptions=True)

    async def _run(self) -> None:
        try:
            while True:
                async with self._lock:
                    if self._closed or not self._subscribers:
                        return
                    after_cursor_by_task: dict[int, int] = {}
                    for subscription_id, item in self._subscribers.items():
                        if subscription_id in self._delivery_tasks:
                            continue
                        current = after_cursor_by_task.get(item.task_id)
                        if current is None or item.cursor < current:
                            after_cursor_by_task[item.task_id] = item.cursor
                try:
                    events = await run_db_io_cancellation_safe(
                        lambda: _load_events(after_cursor_by_task)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "Terminal task event database poll failed; retrying",
                        exc_info=True,
                    )
                    events = []
                await self._schedule_deliveries(events)
                self._wake.clear()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self._poll_interval_seconds,
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Terminal task event poller failed unexpectedly")

    async def _schedule_deliveries(
        self,
        events: list[TerminalTaskEvent],
    ) -> None:
        events_by_task: dict[int, list[TerminalTaskEvent]] = {}
        for event in events:
            events_by_task.setdefault(event.task_id, []).append(event)
        async with self._lock:
            candidates = [
                (
                    subscription_id,
                    subscriber,
                    [
                        event
                        for event in events_by_task.get(subscriber.task_id, [])
                        if subscriber.cursor < event.cursor
                    ],
                )
                for subscription_id, subscriber in self._subscribers.items()
                if subscription_id not in self._delivery_tasks
                and any(
                    subscriber.cursor < event.cursor
                    for event in events_by_task.get(subscriber.task_id, [])
                )
            ]
        for subscription_id, subscriber, pending_events in candidates:
            async with self._lock:
                current = self._subscribers.get(subscription_id)
                if (
                    current is subscriber
                    and subscription_id not in self._delivery_tasks
                    and pending_events
                ):
                    self._delivery_tasks[subscription_id] = asyncio.create_task(
                        self._deliver_batch(
                            subscription_id,
                            subscriber,
                            pending_events,
                        )
                    )

    async def _deliver_batch(
        self,
        subscription_id: int,
        subscriber: _Subscriber,
        events: list[TerminalTaskEvent],
    ) -> None:
        try:
            for event in events:
                authorized = (
                    subscriber.principal_is_admin
                    or subscriber.principal_user_id == event.task_owner_user_id
                )
                while authorized:
                    try:
                        await subscriber.sink(event)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.warning(
                            "Terminal task event delivery failed; retrying "
                            "event_id=%s task_id=%s",
                            event.event_id,
                            event.task_id,
                            exc_info=True,
                        )
                        await asyncio.sleep(self._poll_interval_seconds)
                    else:
                        break
                async with self._lock:
                    current = self._subscribers.get(subscription_id)
                    if current is not subscriber:
                        return
                    if current.cursor < event.cursor:
                        current.cursor = event.cursor
        finally:
            current_task = asyncio.current_task()
            async with self._lock:
                if self._delivery_tasks.get(subscription_id) is current_task:
                    del self._delivery_tasks[subscription_id]
                self._wake.set()

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._subscribers.clear()
            poller = self._poller
            self._poller = None
            deliveries = list(self._delivery_tasks.values())
            self._delivery_tasks.clear()
            self._wake.set()
        for delivery in deliveries:
            delivery.cancel()
        if deliveries:
            await asyncio.gather(*deliveries, return_exceptions=True)
        if poller is not None and not poller.done():
            poller.cancel()
            try:
                await poller
            except asyncio.CancelledError:
                pass


class TerminalTaskEventLoopRegistry:
    """Own one hub per event loop and one subscription per local connection."""

    def __init__(self) -> None:
        self._hubs: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            TerminalTaskEventHub,
        ] = weakref.WeakKeyDictionary()
        self._subscriptions: dict[object, TerminalTaskEventSubscription] = {}

    def has_subscription(self, connection: object) -> bool:
        return connection in self._subscriptions

    async def attach(
        self,
        *,
        connection: object,
        principal: TerminalTaskEventPrincipal,
        task_id: int,
        sink: TerminalTaskEventSink,
        after_event_id: int | None,
    ) -> TerminalTaskEventSubscription:
        prior = self._subscriptions.get(connection)
        if prior is not None:
            try:
                await prior.close()
            finally:
                if prior.closed and self._subscriptions.get(connection) is prior:
                    del self._subscriptions[connection]
        loop = asyncio.get_running_loop()
        hub = self._hubs.get(loop)
        if hub is None:
            hub = TerminalTaskEventHub()
            self._hubs[loop] = hub
        subscription = await hub.attach_terminal_events(
            principal=principal,
            task_id=task_id,
            sink=sink,
            after_event_id=after_event_id,
        )
        self._subscriptions[connection] = subscription
        return subscription

    async def detach(self, connection: object) -> None:
        subscription = self._subscriptions.get(connection)
        if subscription is not None:
            try:
                await subscription.close()
            finally:
                if (
                    subscription.closed
                    and self._subscriptions.get(connection) is subscription
                ):
                    del self._subscriptions[connection]
