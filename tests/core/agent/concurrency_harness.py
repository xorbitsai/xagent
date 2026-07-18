"""Reusable test harness for ReAct in-turn tool-concurrency development.

Shared by the concurrency increments (segmentation, concurrent batch, ledger
ordering, segment-loop integration, resume). Provides controllable fakes so
order/concurrency assertions are deterministic:

- ``FakeTool`` — minimal tool face with a configurable ``execute`` (delay/gate/
  raises/result) plus concurrency-safety metadata; records its own calls and,
  via a shared ``ConcurrencyTracker``, the live concurrency peak.
- ``ConcurrencyTracker`` — tracks active/peak overlap and enter/leave order to
  prove a batch ran concurrently (not serially) and respected the Semaphore cap.
- ``RecordingContext`` — drop-in for ``ExecutionContext`` message methods that
  records ``add_tool_result`` calls in order with their ``tool_call_id`` (I1/I2).
- ``FakeRuntime`` — stubs the ``PatternRuntime`` surface ReAct touches
  (checkpoint / on_tool_* / should_interrupt / interrupt_reason) and records the
  call sequence for checkpoint/span assertions.
- ``make_tool_call`` / ``make_react`` — small constructors for tool-call dicts
  and a minimal ``ReActPattern`` wired with the concurrency knobs.

This is test-support code (no production logic); it is exercised by
``test_concurrency_harness.py``'s sanity test.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
from typing import Any

from xagent.core.agent import ReActPattern

_tool_call_ids = itertools.count(1)


class ConcurrencyTracker:
    """Records live concurrency so tests can assert real parallelism + cap.

    ``enter`` is called before a fake tool awaits its delay/gate, so when N
    tools run concurrently the active count reaches N (and ``peak`` records it)
    before any of them leaves. A purely serial run never exceeds 1.
    """

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.enter_order: list[str] = []
        self.leave_order: list[str] = []

    def enter(self, name: str) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.enter_order.append(name)

    def leave(self, name: str) -> None:
        self.active -= 1
        self.leave_order.append(name)


class FakeToolMetadata:
    """Lightweight stand-in for ``ToolMetadata``.

    Carries only what the scheduler reads (``name`` / ``concurrency_safe`` /
    ``read_only`` / ``category`` / ``decision_group``). ``concurrency_safe`` is
    implied by ``read_only`` to mirror the production implication rule.
    """

    def __init__(
        self,
        name: str,
        *,
        concurrency_safe: bool = False,
        read_only: bool = False,
        category: str = "basic",
        decision_group: str | None = None,
        description: str | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"Fake tool {name}"
        self.read_only = bool(read_only)
        self.concurrency_safe = bool(concurrency_safe) or bool(read_only)
        self.category = category
        self.decision_group = decision_group


class FakeTool:
    """Controllable fake tool for concurrency and ordering tests."""

    def __init__(
        self,
        name: str,
        *,
        concurrency_safe: bool = False,
        read_only: bool = False,
        delay: float = 0.0,
        gate: asyncio.Event | None = None,
        raises: BaseException | None = None,
        result: Any = None,
        decision_group: str | None = None,
        tracker: ConcurrencyTracker | None = None,
    ) -> None:
        self.metadata = FakeToolMetadata(
            name,
            concurrency_safe=concurrency_safe,
            read_only=read_only,
            decision_group=decision_group,
        )
        self._delay = delay
        self._gate = gate
        self._raises = raises
        self._result = result
        self.tracker = tracker
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self.metadata.name

    async def execute(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        tracker = self.tracker
        if tracker is not None:
            tracker.enter(self.metadata.name)
        try:
            if self._gate is not None:
                await self._gate.wait()
            elif self._delay:
                await asyncio.sleep(self._delay)
            if self._raises is not None:
                raise self._raises
            if self._result is not None:
                return self._result
            return {"success": True, "tool": self.metadata.name, "args": dict(kwargs)}
        finally:
            if tracker is not None:
                tracker.leave(self.metadata.name)


class _RecordedMessage:
    __slots__ = ("role", "content", "tool_call_id", "metadata")

    def __init__(
        self,
        role: str,
        content: Any,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id
        self.metadata = metadata or {}


class RecordingContext:
    """Records message-producing calls in order (drop-in for ExecutionContext).

    ``tool_results`` preserves the exact order and ``tool_call_id`` of every
    ``add_tool_result`` call, which is what the ordered-backfill (I1) and
    pairing (I2) assertions inspect.
    """

    def __init__(self, execution_id: str | None = "task-test") -> None:
        self.execution_id = execution_id
        self.messages: list[_RecordedMessage] = []
        self.tool_results: list[dict[str, Any]] = []

    def add_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> _RecordedMessage:
        self.tool_results.append(
            {
                "tool_name": tool_name,
                "result": result,
                "tool_call_id": tool_call_id,
            }
        )
        message = _RecordedMessage("tool", result, tool_call_id=tool_call_id)
        self.messages.append(message)
        return message

    def add_assistant_message(self, content: str, **kwargs: Any) -> _RecordedMessage:
        message = _RecordedMessage("assistant", content, metadata=dict(kwargs))
        self.messages.append(message)
        return message

    def add_system_message(self, content: str, **kwargs: Any) -> _RecordedMessage:
        message = _RecordedMessage("system", content, metadata=dict(kwargs))
        self.messages.append(message)
        return message


class FakeRuntime:
    """Stubs the PatternRuntime surface ReAct touches and records the sequence.

    ``events`` is an ordered log of ``(kind, payload)`` tuples so tests can
    assert checkpoint sequencing and per-call span pairing. ``should_interrupt``
    is driven by ``interrupt_at``: a set of labels/statuses (or the literal
    ``True`` for always) used by interrupt tests.
    """

    def __init__(
        self,
        *,
        execution_id: str | None = "task-test",
        interrupt: bool = False,
        interrupt_reason: str | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.active_react_step_id: str | None = None
        self._interrupt = interrupt
        self.interrupt_reason = interrupt_reason
        self.events: list[tuple[str, Any]] = []

    @staticmethod
    def _tool_call_id(tool_call: dict[str, Any]) -> str | None:
        value = tool_call.get("id")
        return str(value) if value is not None else None

    async def should_interrupt(self) -> bool:
        return bool(self._interrupt)

    async def send_message(self, **kwargs: Any) -> None:
        self.events.append(("send_message", dict(kwargs)))

    def request_interrupt(self, reason: str | None = None) -> None:
        self._interrupt = True
        self.interrupt_reason = reason

    async def run_tool_call(self, invoke: Any) -> Any:
        result = invoke()
        if inspect.isawaitable(result):
            return await result
        return result

    async def checkpoint(
        self,
        status: str,
        *,
        context: Any = None,
        pattern: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(("checkpoint", {"status": status, "metadata": metadata}))

    async def on_tool_start(self, *, tool_call: dict[str, Any]) -> None:
        self.events.append(
            (
                "on_tool_start",
                {
                    "tool_name": tool_call.get("name"),
                    "tool_call_id": self._tool_call_id(tool_call),
                },
            )
        )

    async def on_tool_end(self, *, tool_call: dict[str, Any], result: Any) -> None:
        self.events.append(
            (
                "on_tool_end",
                {
                    "tool_name": tool_call.get("name"),
                    "tool_call_id": self._tool_call_id(tool_call),
                    "result": result,
                },
            )
        )

    async def on_tool_error(
        self, *, tool_call: dict[str, Any], error: BaseException, result: Any
    ) -> None:
        self.events.append(
            (
                "on_tool_error",
                {
                    "tool_name": tool_call.get("name"),
                    "tool_call_id": self._tool_call_id(tool_call),
                    "error": str(error),
                    "result": result,
                },
            )
        )

    async def on_tool_cancelled(
        self, *, tool_call: dict[str, Any], reason: str | None = None
    ) -> None:
        self.events.append(
            (
                "on_tool_cancelled",
                {
                    "tool_name": tool_call.get("name"),
                    "tool_call_id": self._tool_call_id(tool_call),
                    "reason": reason,
                },
            )
        )

    def events_of(self, kind: str) -> list[Any]:
        return [payload for event_kind, payload in self.events if event_kind == kind]


def make_tool_call(
    name: str,
    args: dict[str, Any] | None = None,
    id: str | None = None,
) -> dict[str, Any]:
    """Build a normalized tool-call dict; auto-assigns a unique id if omitted."""
    return {
        "name": name,
        "args": dict(args or {}),
        "id": id if id is not None else f"call_{next(_tool_call_ids)}",
    }


def make_react(
    *,
    parallel: bool = False,
    max_concurrency: int = 3,
    **kwargs: Any,
) -> ReActPattern:
    """Construct a minimal ReActPattern wired with the concurrency knobs.

    The concurrency attributes are set defensively via ``setattr`` so the
    harness works both before Inc.2 introduces them on ``__init__`` and after.
    """
    pattern = ReActPattern(**kwargs)
    setattr(pattern, "tool_parallel_enabled", parallel)
    setattr(pattern, "tool_max_concurrency", max_concurrency)
    return pattern
