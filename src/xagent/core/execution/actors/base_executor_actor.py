"""
Base executor actor class.

Provides common functionality for all executor actors using xoscar framework.
"""

import inspect
import time
import traceback
from typing import Any

import xoscar as xo


class BaseExecutorActor(xo.Actor):  # type: ignore[misc]
    """Base executor actor.

    Provides common execution functionality for Python, JavaScript, and command executors.
    All executors inherit from this class to get consistent error handling and result formatting.
    """

    async def __post_create__(self) -> None:
        """Called after actor creation.

        Can be overridden by subclasses for initialization.
        """
        pass

    async def __pre_destroy__(self) -> None:
        """Called before actor destruction.

        Can be overridden by subclasses for cleanup.
        """
        pass

    def _format_result(self, result: dict[str, Any], execution_time: float) -> dict:
        return {
            "success": result.get("success", result.get("return_code", 0) == 0),
            "output": result.get("output", ""),
            "error": result.get("error", ""),
            "return_code": result.get("return_code", 0),
            "metadata": result.get("metadata", {}),
            "execution_time": execution_time,
        }

    def _format_exception(self, exc: Exception, execution_time: float) -> dict:
        error_message = f"{type(exc).__name__}: {str(exc)}"
        error_traceback = traceback.format_exc()

        return {
            "success": False,
            "output": "",
            "error": error_message,
            "return_code": -1,
            "metadata": {"traceback": error_traceback},
            "execution_time": execution_time,
        }

    def _execute_with_tracking(self, func: Any, *args: Any, **kwargs: Any) -> dict:
        """Execute function with time tracking.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Execution result dictionary
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return self._format_result(result, execution_time)
        except Exception as e:
            execution_time = time.time() - start_time
            return self._format_exception(e, execution_time)

    async def _execute_async_with_tracking(
        self, func: Any, *args: Any, **kwargs: Any
    ) -> dict:
        """Execute async-capable function with time tracking."""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            execution_time = time.time() - start_time
            return self._format_result(result, execution_time)
        except Exception as e:
            execution_time = time.time() - start_time
            return self._format_exception(e, execution_time)
