"""Shared batch scheduling and cooperative stop signaling for bot ingress."""

import asyncio
import logging
from collections.abc import Coroutine
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from ...config import get_shared_task_execution_enabled
from ..services.channel_delivery import ChannelDelivery, deliver_channel_result
from ..services.channel_runtime import (
    ChannelAuthorizationError,
    ChannelConfigurationError,
)
from ..services.channel_stop import ChannelStopOutcome, stop_channel_task
from ..services.db_runtime import cancel_and_drain_async_task
from ..services.shared_channel_execution import SharedChannelTurn

logger = logging.getLogger(__name__)
_Key = TypeVar("_Key", int, str)


class StopReply(str, Enum):
    CANCELLED = "cancelled"  # the paused run will not resume automatically
    STOPPED = "stopped"  # today's "Stopped the current run..."
    NOTHING = "nothing"  # "No active run to stop."


def stop_reply_text(stop: StopReply, *, new_hint: str) -> str:
    """The /stop reply; ``new_hint`` is how the bot names /new."""
    return {
        StopReply.CANCELLED: "Stopped. The interrupted run will not resume "
        f"automatically. Send another message to continue here, or {new_hint} "
        "for a fresh task.",
        StopReply.STOPPED: "Stopped the current run. Send another message to "
        f"continue here, or {new_hint} for a fresh task.",
        StopReply.NOTHING: "No active run to stop.",
    }[stop]


def selected_task_id(selected: object) -> int | None:
    """The persisted task selection as a task id; ``None`` for none or a
    fresh conversation (``-1``)."""
    try:
        task_id = int(cast(Any, selected))
    except (TypeError, ValueError):
        return None
    return task_id if task_id > 0 else None


class BatchChannelControl(Generic[_Key]):
    queue_flush_delay_seconds = 1.0
    control_label: str
    _accepting: bool
    user_message_queues: dict[_Key, list[Any]]
    user_message_tasks: dict[_Key, asyncio.Task]
    user_active_executions: dict[_Key, tuple[int, object]]
    user_preparing_executions: set[_Key]
    user_stop_events: dict[_Key, asyncio.Event]
    user_conversation_generations: dict[_Key, int]
    channel_id: int | None = None

    def _initialize_batch_control(self) -> None:
        self.user_message_queues = {}
        self.user_message_tasks = {}
        self.user_active_executions = {}
        self.user_preparing_executions = set()
        self.user_stop_events = {}
        self.user_conversation_generations = {}

    async def _process_queued_batch(self, user_id: _Key, messages: list[Any]) -> None:
        raise NotImplementedError

    def _active_trace_handlers(self) -> dict[_Key, Any]:
        raise NotImplementedError

    async def _deliver_shared_result(
        self, delivery: ChannelDelivery, result: dict[str, Any]
    ) -> None:
        raise NotImplementedError

    def _prune_idle_user_state(self, user_id: _Key) -> None:
        event = self.user_stop_events.get(user_id)
        if event is not None and not event.is_set():
            self.user_stop_events.pop(user_id, None)

    def _enqueue_user_message(self, user_id: _Key, message: Any) -> bool:
        if not self._accepting:
            return False

        self.user_message_queues.setdefault(user_id, []).append(message)
        task = self.user_message_tasks.get(user_id)
        if task is None or task.done():
            self._schedule_user_queue(user_id)
        return True

    def _schedule_user_queue(self, user_id: _Key) -> bool:
        if not self._accepting:
            return False
        self.user_message_tasks[user_id] = asyncio.create_task(
            self._process_user_queue(user_id)
        )
        return True

    def _conversation_generation(self, user_id: _Key) -> int:
        return self.user_conversation_generations.get(user_id, 0)

    def _stop_current_conversation(self, user_id: _Key) -> bool:
        # discard_output=False: /stop pauses the run but keeps the user in this
        # conversation, so the partial answer must still be delivered.
        return self._request_current_conversation_stop(
            user_id,
            reason=f"{self.control_label} stop requested",
            discard_output=False,
        )

    def _drop_pending_input(self, user_id: _Key) -> bool:
        """Drop queued input and stop a turn being prepared; leave accepted runs."""
        queued = self.user_message_queues.pop(user_id, None)
        preparing = user_id in self.user_preparing_executions
        if preparing:
            self._request_user_stop(user_id)
        return bool(queued) or preparing

    async def _handle_stop_request(
        self, user_id: _Key, *, external_user_id: str, task_id: int | None
    ) -> tuple[StopReply, int | None]:
        """/stop: the database first, then the in-memory stop when it still
        has something to stop. Returns the reply and the START whose held
        reply the stop released (deliver it after replying)."""
        result = None
        if (
            get_shared_task_execution_enabled()
            and self.channel_id is not None
            and task_id is not None
        ):
            try:
                result = await stop_channel_task(
                    channel_id=self.channel_id,
                    external_user_id=external_user_id,
                    task_id=task_id,
                    # Only a tracked turn is stopped in memory; queued or
                    # preparing input is no turn of the awaited run.
                    pause_awaited_run=user_id not in self.user_active_executions,
                )
            except (ChannelAuthorizationError, ChannelConfigurationError):
                result = None
            except Exception:
                logger.exception("Channel stop could not read task %s", task_id)
                result = None
        if result is not None:
            if result.outcome is ChannelStopOutcome.CANCELLED:
                # A retained turn's stop would queue a PAUSE the paused run
                # ignores; only the input not yet accepted is dropped.
                self._drop_pending_input(user_id)
                return StopReply.CANCELLED, result.held_command_db_id
            if result.outcome is ChannelStopOutcome.IDLE:
                dropped = self._drop_pending_input(user_id)
                return (StopReply.STOPPED if dropped else StopReply.NOTHING), None
            if result.outcome is ChannelStopOutcome.PAUSE_QUEUED:
                self._stop_current_conversation(user_id)
                return StopReply.STOPPED, None
        stopped = self._stop_current_conversation(user_id)
        return (StopReply.STOPPED if stopped else StopReply.NOTHING), None

    async def _deliver_released_reply(self, command_db_id: int | None) -> None:
        """Best effort; the recovery loop delivers it otherwise."""
        if command_db_id is None:
            return
        try:
            await deliver_channel_result(command_db_id, self._deliver_shared_result)
        except Exception:
            logger.exception(
                "Released channel reply %s not delivered now", command_db_id
            )

    def _request_current_conversation_stop(
        self, user_id: _Key, *, reason: str, discard_output: bool = True
    ) -> bool:
        queued_messages = self.user_message_queues.pop(user_id, None)
        active_trace_handler = self._active_trace_handlers().get(user_id)
        if active_trace_handler is not None:
            active_trace_handler.cancel(discard_output=discard_output)
        active_execution = self.user_active_executions.get(user_id)
        if active_execution is not None and isinstance(
            active_execution[1], SharedChannelTurn
        ):
            active_execution[1].discard_output = discard_output
        stopped = self._stop_user_active_execution(user_id, reason=reason)
        preparing = user_id in self.user_preparing_executions
        # A retained previous turn may accept stop while a newer turn prepares.
        # Record the preparation stop independently of signaling that handle.
        if preparing:
            self._request_user_stop(user_id)
        return bool(queued_messages) or stopped or preparing

    def _stop_user_active_execution(self, user_id: _Key, *, reason: str) -> bool:
        active_execution = self.user_active_executions.get(user_id)
        if active_execution is None:
            return False

        task_id, agent_service = active_execution
        if isinstance(agent_service, SharedChannelTurn):
            return agent_service.request_stop()
        pause_execution_by_id = getattr(agent_service, "pause_execution_by_id", None)
        if not callable(pause_execution_by_id):
            logger.warning(
                "Channel active task %s for user %s does not support pause",
                task_id,
                user_id,
            )
            return False

        try:
            return bool(pause_execution_by_id(str(task_id), reason=reason))
        except Exception as e:
            logger.warning(
                "Failed to pause Channel active task %s for user %s: %s",
                task_id,
                user_id,
                e,
            )
            return False

    def _get_user_stop_event(self, user_id: _Key) -> asyncio.Event:
        event = self.user_stop_events.get(user_id)
        if event is None:
            event = asyncio.Event()
            self.user_stop_events[user_id] = event
        return event

    def _request_user_stop(self, user_id: _Key) -> None:
        self._get_user_stop_event(user_id).set()

    def _consume_user_stop_request(self, user_id: _Key) -> bool:
        event = self.user_stop_events.get(user_id)
        if event is None or not event.is_set():
            return False
        event.clear()
        return True

    def _clear_user_stop_request(self, user_id: _Key) -> None:
        event = self.user_stop_events.get(user_id)
        if event is not None:
            event.clear()

    async def _await_execution_with_stop_monitor(
        self,
        user_id: _Key,
        execution: Coroutine[Any, Any, dict[str, Any]],
        *,
        reason: str,
    ) -> dict[str, Any]:
        execution_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(execution)
        stop_event = self._get_user_stop_event(user_id)

        try:
            while True:
                if execution_task.done():
                    return await execution_task

                if stop_event.is_set():
                    while not execution_task.done():
                        if self._stop_user_active_execution(user_id, reason=reason):
                            stop_event.clear()
                            break
                        await asyncio.sleep(0.05)
                    continue

                done, _ = await asyncio.wait(
                    {execution_task},
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if execution_task in done:
                    return await execution_task
        finally:
            if not execution_task.done():
                await cancel_and_drain_async_task(execution_task)

    async def _process_user_queue(self, user_id: _Key) -> None:
        while True:
            await asyncio.sleep(self.queue_flush_delay_seconds)
            messages = self.user_message_queues.pop(user_id, [])
            if messages:
                try:
                    await self._process_queued_batch(user_id, messages)
                except Exception:
                    logger.exception("Channel batch failed for user %s", user_id)

            if self.user_message_queues.get(user_id):
                continue

            current_task = cast(asyncio.Task, asyncio.current_task())
            if self.user_message_tasks.get(user_id) is current_task:
                self.user_message_tasks.pop(user_id, None)

            if not self.user_message_queues.get(user_id):
                self._prune_idle_user_state(user_id)
                return

            self.user_message_tasks[user_id] = current_task
