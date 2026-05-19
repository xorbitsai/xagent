"""
Process-isolated tool wrapper.

Provides process-level isolation for tools using xoscar.
xoscar automatically handles serialization - no manual encoding needed.
"""

from .integration import (
    maybe_wrap_tool,
    should_use_process_isolation,
    supports_process_isolation,
    wrap_tools,
)
from .process_isolated_tool_wrapper import (
    ProcessIsolatedToolWrapper,
    create_process_isolated_tool,
)

__all__ = [
    "ProcessIsolatedToolWrapper",
    "create_process_isolated_tool",
    "supports_process_isolation",
    "should_use_process_isolation",
    "maybe_wrap_tool",
    "wrap_tools",
]
