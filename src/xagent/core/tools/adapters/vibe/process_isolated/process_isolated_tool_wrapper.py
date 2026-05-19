"""
Process-isolated tool wrapper.

Execute tool's run_json_sync/async methods in isolated processes using xoscar.
xoscar automatically handles serialization - no need for manual pickle/json encoding.
"""

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

from pydantic import BaseModel

from .....execution.service.manager import get_process_service
from ..base import AbstractBaseTool, ToolMetadata

if TYPE_CHECKING:
    from ..base import ToolCategory

logger = logging.getLogger(__name__)


class ProcessIsolatedToolWrapper(AbstractBaseTool):
    """Process-isolated tool wrapper.

    Wrap any AbstractBaseTool to execute in isolated processes using xoscar.
    xoscar automatically serializes the tool instance and arguments.
    """

    def __init__(
        self,
        target_tool: AbstractBaseTool,
        timeout: int = 300,
    ):
        """Initialize process-isolated tool wrapper.

        Args:
            target_tool: Target tool to wrap
            timeout: Execution timeout in seconds (default: 300)
        """
        self._target = target_tool
        self._timeout = timeout

        # Proxy target tool attributes
        self._visibility = getattr(target_tool, "_visibility", None)
        self._allow_users = getattr(target_tool, "_allow_users", None)

    @property
    def is_isolated(self) -> bool:
        """Marker for process-isolated."""
        return True

    @property
    def supports_process_isolation(self) -> bool:
        """Marker for process-isolation-capable tools."""
        return True

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
        return getattr(self._target, "category", None)  # type: ignore[return-value]

    @property
    def metadata(self) -> ToolMetadata:
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
        return self._target.return_value_as_string(value)

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Synchronous execution."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_json_async(args))

        result: dict[str, Any] = {}

        def _run_in_thread() -> None:
            try:
                result["value"] = asyncio.run(self.run_json_async(args))
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()
        thread.join()

        if "error" in result:
            raise result["error"]
        return result.get("value")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute tool asynchronously in isolated process.

        Args:
            args: Tool arguments

        Returns:
            Tool execution result
        """
        process_service = get_process_service()
        if not process_service:
            # ProcessService not available, fall back to direct execution
            logger.warning(
                f"ProcessService not available for {self._target.name}, "
                "falling back to direct execution"
            )
            return await self._target.run_json_async(args)

        try:
            # xoscar automatically serializes tool and args
            result = await process_service.execute_tool(
                tool=self._target,  # Tool instance - xoscar serializes it
                args=dict(args),  # Arguments - xoscar serializes them
                timeout=self._timeout,
            )

            if not result.success:
                raise RuntimeError(
                    f"Process-isolated execution failed for {self._target.name}: "
                    f"{result.error}"
                )

            return result.output

        except Exception as e:
            logger.error(
                f"Error executing tool {self._target.name} in isolated process: {e}",
                exc_info=True,
            )
            raise

    async def save_state_json(self) -> Mapping[str, Any]:
        """Save state (delegates to target tool)."""
        return await self._target.save_state_json()

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        """Load state (delegates to target tool)."""
        await self._target.load_state_json(state)

    async def setup(self, task_id: Optional[str] = None) -> None:
        """Setup tool (delegates to target tool)."""
        await self._target.setup(task_id)

    async def teardown(self, task_id: Optional[str] = None) -> None:
        """Teardown tool (delegates to target tool)."""
        await self._target.teardown(task_id)


def create_process_isolated_tool(
    tool: AbstractBaseTool,
    timeout: int = 300,
) -> ProcessIsolatedToolWrapper:
    """Create process-isolated tool instance.

    Args:
        tool: Tool to wrap
        timeout: Execution timeout in seconds

    Returns:
        Process-isolated tool wrapper
    """
    wrapper = ProcessIsolatedToolWrapper(
        target_tool=tool,
        timeout=timeout,
    )

    return wrapper
