"""
Output Filter Tool Wrapper

Wraps any tool with output length filtering capabilities.
"""

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

from pydantic import BaseModel

from ....agent.result import normalize_tool_failure_code
from ...tool_result_spill import (
    SPILL_RESERVED_RESULT_KEY,
    SpillRunBudget,
    SpillTarget,
    is_classified_tool_failure,
    spill_oversized_values,
    strip_reserved_spill_key,
)
from ...user_interaction import (
    WAITING_FOR_USER_STATUS,
    tool_result_waits_for_user,
)
from .base import AbstractBaseTool
from .output_filter import OutputValueFilter

if TYPE_CHECKING:
    from .base import ToolCategory

logger = logging.getLogger(__name__)

_INTERACTION_DISPLAY_KEYS = frozenset(
    {
        "description",
        "help_text",
        "label",
        "message",
        "placeholder",
        "prompt",
        "title",
    }
)


def _accepts_kwarg(func: Any, name: str) -> bool:
    """Return whether ``func`` accepts ``name`` as a keyword argument."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == name
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
        for parameter in signature.parameters.values()
    )


class OutputFilteredToolWrapper(AbstractBaseTool):
    """
    Wrapper that applies output filtering to any tool.

    This wrapper intercepts the return value from run_json_sync/async.
    Every result has the engine's reserved spill report key stripped
    first, regardless of configuration. When this wrapper is given a
    spill target, it then stores any oversized value in that target
    before applying length limiting, and on the asynchronous paths that
    storing step runs in a worker thread. With no target, the reserved-key
    strip is the only extra step and the observed behavior is the output
    filter alone. When a spill did happen, the spill report is carried
    past the length limiting unchanged.
    """

    def __init__(
        self,
        target_tool: AbstractBaseTool,
        max_chars: int,
        max_fields: int,
        max_recursion: int,
        spill_target: SpillTarget | None = None,
        spill_run_budget: SpillRunBudget | None = None,
    ):
        """
        Initialize output filter wrapper.

        Args:
            target_tool: Tool to wrap
            max_chars: Maximum output length in characters.
            max_fields: Maximum number of fields/items in dict/list.
            max_recursion: Maximum recursion depth.
            spill_target: Where oversized values get written instead of
                truncated. None disables spilling entirely -- when no spill
                target was given, this wrapper's behavior is identical to
                before spilling existed.
            spill_run_budget: Shared file-count budget across every wrapper
                built in the same tool-set construction. None gives this
                wrapper its own budget, which only matters when a single
                wrapper spills more than once.
        """
        self._target = target_tool
        self._spill_target = spill_target
        self._spill_run_budget = spill_run_budget or SpillRunBudget()

        # Create output filter
        self._filter = OutputValueFilter(max_chars, max_fields, max_recursion)

    @property
    def is_sandboxed(self) -> bool:
        return getattr(self._target, "is_sandboxed", False)

    @property
    def name(self) -> str:
        return self._target.name

    @property
    def description(self) -> str:
        return self._target.description

    @property
    def tags(self) -> list[str]:
        return self._target.tags

    @property
    def category(self) -> "ToolCategory":
        """Get tool category (delegates to target tool)."""
        return getattr(self._target, "category", None)  # type: ignore[return-value]

    @property
    def metadata(self) -> Any:  # ToolMetadata (avoid circular import)
        return self._target.metadata

    def args_type(self) -> Type[BaseModel]:
        return self._target.args_type()

    def return_type(self) -> Type[BaseModel]:
        return self._target.return_type()

    def state_type(self) -> Optional[Type[BaseModel]]:
        return self._target.state_type()

    def is_async(self) -> bool:
        return self._target.is_async()

    def return_value_as_string(self, value: Any) -> str:
        """Convert return value to string (delegates to target tool)."""
        return self._target.return_value_as_string(value)

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Execute tool synchronously with output filtering."""
        result = self._target.run_json_sync(args)
        return self._filter_result(result)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute tool asynchronously with output filtering."""
        result = await self._target.run_json_async(args)
        return await self._filter_result_async(result)

    async def save_state_json(self) -> Mapping[str, Any]:
        """Save state (delegates to target tool)."""
        return await self._target.save_state_json()

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        """Load state (delegates to target tool)."""
        await self._target.load_state_json(state)

    async def setup(self, task_id: Optional[str] = None) -> None:
        """Setup tool (delegates to target tool)."""
        if hasattr(self._target, "setup"):
            await self._target.setup(task_id)

    async def teardown(
        self,
        task_id: Optional[str] = None,
        execution_status: Optional[str] = None,
    ) -> None:
        """Teardown the target without hiding the execution's final state."""

        teardown = getattr(self._target, "teardown", None)
        if teardown is None:
            return
        kwargs: dict[str, Any] = {"task_id": task_id}
        if execution_status is not None and _accepts_kwarg(
            teardown, "execution_status"
        ):
            kwargs["execution_status"] = execution_status
        result = teardown(**kwargs)
        if inspect.isawaitable(result):
            await result

    def __getattr__(self, name: str) -> Any:
        """Delegate optional runtime capabilities to the wrapped tool."""

        if name.startswith("_"):
            raise AttributeError(name)
        try:
            target = object.__getattribute__(self, "_target")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(target, name)

    @property
    def func(self) -> Any:
        """Get the underlying function, wrapped with output filtering."""
        if not hasattr(self, "_wrapped_func"):
            func_obj = getattr(self._target, "func", None)
            if func_obj is None:
                raise AttributeError(
                    f"Tool '{self._target.name}' has no 'func' attribute"
                )

            # Create wrapper based on function type
            if inspect.iscoroutinefunction(func_obj):
                self._wrapped_func = self._make_async_wrapper(func_obj)
            else:
                self._wrapped_func = self._make_sync_wrapper(func_obj)

        return self._wrapped_func

    def _make_sync_wrapper(self, original_func: Any) -> Any:
        """Create a sync wrapper that applies output filtering."""

        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            result = original_func(*args, **kwargs)
            return self._filter_result(result)

        return wrapped_func

    def _make_async_wrapper(self, original_func: Any) -> Any:
        """Create an async wrapper that applies output filtering."""

        async def wrapped_func_async(*args: Any, **kwargs: Any) -> Any:
            result = await original_func(*args, **kwargs)
            return await self._filter_result_async(result)

        return wrapped_func_async

    def _filter_result(self, result: Any) -> Any:
        """Filter one result on this thread; only for callers off the loop.

        The spill step under _spill_only is synchronous and CPU-bound -- the
        module's entry point can run for seconds on a collection of tiny
        items -- so a caller on an asyncio event loop must use
        _filter_result_async instead of this method. The two synchronous
        callers (run_json_sync and the closure _make_sync_wrapper returns)
        are already off the loop, and wrapping them in a worker thread would
        only add a hop.
        """

        return self._filter_after_spill(self._spill_only(result))

    async def _filter_result_async(self, result: Any) -> Any:
        """Filter one result without holding the event loop for the spill.

        The whole spill entry point goes to a worker thread, not one inner
        function of it: every CPU cost on this path -- each measuring pass,
        the per-item prefix scan that enforces the file cap, and the writes
        -- sits under that one call, so a boundary drawn anywhere inside it
        leaves part of the cost on the loop. What stays here is dict work
        and the pre-existing output filter, which this change does not move.

        The run budget crosses the boundary by reference: to_thread receives
        the bound method, so self._spill_run_budget is the same object in
        the worker. It has to be -- it holds a lock and cannot be copied --
        and its reserve() is what makes two workers landing on one budget
        safe.
        """

        if self._spill_target is None:
            # Nothing to offload. Without a target the spill step is a
            # dictionary-key strip, and the worker-thread hop would cost
            # more than the work it moves. This is the same None test
            # _spill_oversized_values already makes, not a switch: this
            # path always hops when a spill target was given.
            return self._filter_result(result)
        spilled = await asyncio.to_thread(self._spill_only, result)
        return self._filter_after_spill(spilled)

    def _spill_only(self, result: Any) -> Any:
        """Strip a forged report key, then spill oversized values.

        The CPU-bound half of filtering, kept in one method so the async
        path has exactly one thing to offload.
        """

        return self._spill_oversized_values(strip_reserved_spill_key(result))

    def _filter_after_spill(self, spilled: Any) -> Any:
        """Filter the tool's own payload; carry the engine's spill report past it.

        A reserved report key present here was written by the spill step:
        _spill_only strips any tool-supplied one before spilling, on both the
        sync and the async path. The report is engine metadata, not tool
        output, so none of the output filter's limits may apply to it -- a
        per-string cap shorter than a generated relative_path would cut the
        path, and a field-count cap would drop the key itself, which the
        spill step appends after every tool key. It is taken off before
        filtering and put back unchanged; ExecutionContext validates it when
        registering.
        """
        if not isinstance(spilled, dict) or SPILL_RESERVED_RESULT_KEY not in spilled:
            return self._filter_tool_payload(spilled)
        records = spilled[SPILL_RESERVED_RESULT_KEY]
        payload = {k: v for k, v in spilled.items() if k != SPILL_RESERVED_RESULT_KEY}
        filtered = self._filter_tool_payload(payload)
        if isinstance(filtered, dict):
            filtered[SPILL_RESERVED_RESULT_KEY] = records
        return filtered

    def _filter_tool_payload(self, spilled: Any) -> Any:
        """Filter output without dropping a control or classification envelope."""

        filtered = self._filter.filter(spilled, self._target.name)
        if not isinstance(filtered, dict) or not isinstance(spilled, dict):
            return filtered

        if tool_result_waits_for_user(spilled):
            filtered["status"] = WAITING_FOR_USER_STATUS
            for key in ("interaction_id", "message_type"):
                if key in spilled:
                    filtered[key] = spilled[key]
            if "message" in spilled:
                filtered["message"] = self._filter.filter(
                    spilled["message"], self._target.name
                )
            if "interactions" in spilled:
                filtered["interactions"] = self._filter_interactions(
                    spilled["interactions"]
                )
            return filtered

        if is_classified_tool_failure(spilled):
            # ``success``/``is_error`` were matched by identity above, so
            # they are literally ``False``/``True``; the two caller-supplied
            # classification values are re-checked before bypassing the
            # filter. Only ``"error"`` can reach this branch for ``status``:
            # both producers of the ``success=False``/``is_error=True`` pair
            # hardcode it (agent_tool._classified_failure,
            # mcp_adapter._run_unavailable), and a waiting result is handled
            # above. Exact plain-string match keeps a ``str`` subclass from
            # writing itself back unfiltered.
            filtered["success"] = spilled["success"]
            filtered["is_error"] = spilled["is_error"]
            status = spilled.get("status")
            if type(status) is str and status == "error":
                filtered["status"] = status
            normalized_failure_code = normalize_tool_failure_code(
                spilled.get("failure_code")
            )
            if normalized_failure_code is not None:
                filtered["failure_code"] = normalized_failure_code
            for key in ("error", "output", "response"):
                if key in spilled:
                    filtered[key] = self._filter.filter(spilled[key], self._target.name)
            return filtered

        return filtered

    def _spill_oversized_values(self, result: Any) -> Any:
        """Replace oversized values with a file-backed placeholder, if wired.

        When no spill target was given, this is a no-op, returning result
        unchanged -- the same behavior this wrapper had before spilling
        existed.

        Any failure of the spill step lands here and degrades to that same
        no-op, so a tool call that succeeds without this layer keeps
        succeeding with it.
        """
        if self._spill_target is None:
            return result
        try:
            spilled, _records = spill_oversized_values(
                result,
                self._spill_target,
                tool_name=self._target.name,
                max_recursion=self._filter.max_recursion,
                run_budget=self._spill_run_budget,
            )
        except Exception as exc:
            # A deliberately broad boundary, and the only one on this path.
            # Spilling is an optional optimization layered in front of the
            # output filter, and this is the one place where "it did not
            # work" has a real, correct answer: hand the untouched result to
            # the same filter that handled it before spilling existed. That
            # is a genuine degradation with a log line, not a bug folded
            # into "resource unavailable" -- the result the caller gets is
            # identical to the one this wrapper's own output filter would
            # have produced for the same input with the reserved report key
            # already removed (that strip runs before this method is ever
            # called, so it is not part of what this boundary changes).
            #
            # It has to be broad because the failures are not ours. The
            # entry point measures values by serializing them, json.dumps
            # falls back to str() for a type it has no rule for, and the
            # module folds only ValueError, TypeError and RecursionError
            # into "leave this one alone" (see its own docstring). A value
            # whose __str__ raises RuntimeError, AttributeError or KeyError
            # therefore reaches here, and without this boundary a tool
            # result carrying one such object -- which does not fail a call
            # today -- would start failing it.
            #
            # asyncio.CancelledError and KeyboardInterrupt derive from
            # BaseException, not Exception, so neither is caught here:
            # cancelling a tool call still cancels it, and Ctrl-C still
            # interrupts.
            logger.warning(
                "Tool %s: storing oversized values failed (%s); falling back "
                "to ordinary output truncation for this result.",
                self._target.name,
                type(exc).__name__,
                exc_info=True,
            )
            return result
        return spilled

    def _filter_interactions(self, interactions: Any) -> Any:
        """Filter display text without changing interaction cardinality."""

        if not isinstance(interactions, list):
            return self._filter.filter(interactions, self._target.name)

        return [
            self._filter_interaction_item(item) if isinstance(item, dict) else item
            for item in interactions
        ]

    def _filter_interaction_item(self, item: dict[str, Any]) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        for key, value in item.items():
            if key in _INTERACTION_DISPLAY_KEYS:
                filtered[key] = self._filter.filter(value, self._target.name)
            elif key in {"actions", "options"}:
                filtered[key] = self._filter_interaction_options(value)
            elif key == "properties" and isinstance(value, dict):
                filtered[key] = self._filter_interaction_item(value)
            else:
                # Control, routing, cardinality, and submitted-value properties
                # must remain byte-for-byte equivalent to the tool result.
                filtered[key] = value
        return filtered

    def _filter_interaction_options(self, options: Any) -> Any:
        if not isinstance(options, list):
            return options

        return [
            self._filter_interaction_item(option)
            if isinstance(option, dict)
            else option
            for option in options
        ]
