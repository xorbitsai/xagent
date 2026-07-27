"""
Output Filter Tool Wrapper

Wraps any tool with output length filtering capabilities.
"""

import inspect
import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

from pydantic import BaseModel

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

    This wrapper intercepts the return value from run_json_sync/async
    and applies length limiting before returning to the caller.
    """

    def __init__(
        self,
        target_tool: AbstractBaseTool,
        max_chars: int,
        max_fields: int,
        max_recursion: int,
    ):
        """
        Initialize output filter wrapper.

        Args:
            target_tool: Tool to wrap
            max_chars: Maximum output length in characters.
            max_fields: Maximum number of fields/items in dict/list.
            max_recursion: Maximum recursion depth.
        """
        self._target = target_tool

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
        return self._filter_result(result)

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
            return self._filter_result(result)

        return wrapped_func_async

    def _filter_result(self, result: Any) -> Any:
        """Filter output without dropping the user-interaction control envelope."""

        filtered = self._filter.filter(result, self._target.name)
        if not tool_result_waits_for_user(result) or not isinstance(filtered, dict):
            return filtered

        filtered["status"] = WAITING_FOR_USER_STATUS
        assert isinstance(result, dict)
        for key in ("interaction_id", "message_type"):
            if key in result:
                filtered[key] = result[key]
        if "message" in result:
            filtered["message"] = self._filter.filter(
                result["message"], self._target.name
            )
        if "interactions" in result:
            filtered["interactions"] = self._filter_interactions(result["interactions"])
        return filtered

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
