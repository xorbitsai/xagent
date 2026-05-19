"""
Tests for ProcessIsolatedToolWrapper.
"""

import pytest
from pydantic import BaseModel

from xagent.core.execution.service import ProcessService
from xagent.core.execution.service.manager import (
    clear_process_service,
    set_process_service,
)
from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolCategory
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)
from xagent.core.tools.adapters.vibe.process_isolated import (
    ProcessIsolatedToolWrapper,
    create_process_isolated_tool,
    maybe_wrap_tool,
    should_use_process_isolation,
    wrap_tools,
)


# Simple test tool
class SimpleCalculatorTool:
    """A simple calculator tool for testing."""

    supports_process_isolation = True

    def __init__(self, precision: int = 2):
        self.precision = precision

    @property
    def name(self) -> str:
        return "simple_calculator"

    @property
    def description(self) -> str:
        return "A simple calculator"

    @property
    def tags(self) -> list[str]:
        return ["calculator", "test"]

    @property
    def metadata(self):
        return {"version": "1.0"}

    def args_type(self):
        from typing import Optional

        from pydantic import BaseModel

        class CalculatorArgs(BaseModel):
            expression: str
            scale: Optional[float] = 1.0

        return CalculatorArgs

    def return_type(self):
        from pydantic import BaseModel

        class CalculatorResult(BaseModel):
            result: float
            scaled: float

        return CalculatorResult

    def state_type(self):
        return None

    async def run_json_async(self, args: dict) -> dict:
        """Calculate expression."""
        try:
            expression = args["expression"]
            scale = args.get("scale", 1.0)

            # Safe evaluation
            result = eval(expression, {"__builtins__": {}}, {})

            # Apply precision
            rounded = round(result, self.precision)
            scaled = round(result * scale, self.precision)

            return {
                "result": rounded,
                "scaled": scaled,
            }
        except Exception as e:
            raise RuntimeError(f"Calculation failed: {e}")


class LifecycleArgs(BaseModel):
    value: str


class LifecycleResult(BaseModel):
    value: str


class LifecycleState(BaseModel):
    value: str = "initial"


class LifecycleTool(AbstractBaseTool):
    supports_process_isolation = True
    category = ToolCategory.BASIC

    def __init__(self) -> None:
        self.setup_calls: list[str | None] = []
        self.teardown_calls: list[str | None] = []
        self.state = "initial"

    @property
    def name(self) -> str:
        return "lifecycle_tool"

    @property
    def description(self) -> str:
        return "Lifecycle test tool"

    def args_type(self) -> type[BaseModel]:
        return LifecycleArgs

    def return_type(self) -> type[BaseModel]:
        return LifecycleResult

    def state_type(self) -> type[BaseModel]:
        return LifecycleState

    def is_async(self) -> bool:
        return False

    def return_value_as_string(self, value):
        return f"formatted:{value}"

    def run_json_sync(self, args):
        return {"value": args["value"]}

    async def run_json_async(self, args):
        return self.run_json_sync(args)

    async def save_state_json(self):
        return {"value": self.state}

    async def load_state_json(self, state):
        self.state = state["value"]

    async def setup(self, task_id=None):
        self.setup_calls.append(task_id)

    async def teardown(self, task_id=None):
        self.teardown_calls.append(task_id)


class UnsupportedTool(LifecycleTool):
    supports_process_isolation = False

    @property
    def name(self) -> str:
        return "unsupported_tool"


@pytest.mark.asyncio
async def test_process_isolated_wrapper_basic():
    """Test basic functionality of ProcessIsolatedToolWrapper."""
    # Create and start ProcessService
    service = ProcessService(n_workers=2, address="localhost:12360")
    await service.start()
    set_process_service(service)

    try:
        # Create tool
        tool = SimpleCalculatorTool(precision=4)

        # Wrap with process isolation
        wrapped_tool = create_process_isolated_tool(tool, timeout=30)

        # Verify wrapper properties
        assert wrapped_tool.name == "simple_calculator"
        assert wrapped_tool.description == "A simple calculator"
        assert wrapped_tool.is_isolated is True

        # Execute tool
        result = await wrapped_tool.run_json_async(
            {
                "expression": "2 + 3 * 4",
                "scale": 2.0,
            }
        )

        assert result["result"] == 14.0
        assert result["scaled"] == 28.0

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_process_isolated_wrapper_with_init_params():
    """Test wrapper preserves init params via xoscar serialization."""
    service = ProcessService(n_workers=2, address="localhost:12361")
    await service.start()
    set_process_service(service)

    try:
        # Create tool with custom init param
        # xoscar automatically serializes the tool instance
        tool = SimpleCalculatorTool(precision=6)
        wrapped_tool = create_process_isolated_tool(tool)

        # Execute and verify precision is preserved
        # xoscar serializes tool with its state (precision=6)
        result = await wrapped_tool.run_json_async(
            {
                "expression": "1 / 3",
            }
        )

        # Should be rounded to 6 decimal places (tool state preserved)
        assert result["result"] == 0.333333

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_should_use_process_isolation():
    """Test should_use_process_isolation logic."""
    import xagent.web.sandbox_manager as sandbox_manager_module
    from xagent.core.execution.service.manager import clear_process_service

    # Clear both services
    clear_process_service()
    # Temporarily clear sandbox manager
    sandbox_manager_module._sandbox_manager = None
    sandbox_manager_module._sandbox_manager_initialized = False

    # No services available
    assert should_use_process_isolation("test_tool") is False

    # Enable ProcessService
    service = ProcessService(n_workers=2, address="localhost:12362")
    await service.start()
    set_process_service(service)

    try:
        # ProcessService available, no sandbox
        assert should_use_process_isolation("test_tool") is True

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_maybe_wrap_tool():
    """Test maybe_wrap_tool function."""
    service = ProcessService(n_workers=2, address="localhost:12363")
    await service.start()
    set_process_service(service)

    try:
        tool = SimpleCalculatorTool(precision=2)

        # Should wrap when ProcessService is available
        wrapped = maybe_wrap_tool(tool)
        assert isinstance(wrapped, ProcessIsolatedToolWrapper)

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_maybe_wrap_tool_fallback():
    """Test maybe_wrap_tool falls back to original tool when ProcessService unavailable."""
    from xagent.core.execution.service.manager import clear_process_service

    # Ensure no ProcessService
    clear_process_service()

    tool = SimpleCalculatorTool(precision=2)

    # Should return original tool when no ProcessService
    wrapped = maybe_wrap_tool(tool)
    assert wrapped is tool  # Same object
    assert not isinstance(wrapped, ProcessIsolatedToolWrapper)


@pytest.mark.asyncio
async def test_maybe_wrap_tool_requires_explicit_support():
    """Test process isolation only wraps explicitly supported tools."""
    service = ProcessService(n_workers=2, address="localhost:12365")
    await service.start()
    set_process_service(service)

    try:
        tool = UnsupportedTool()

        wrapped = maybe_wrap_tool(tool)
        assert wrapped is tool
        assert not isinstance(wrapped, ProcessIsolatedToolWrapper)

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_wrap_tools_list():
    """Test wrap_tools function with multiple tools."""
    service = ProcessService(n_workers=2, address="localhost:12364")
    await service.start()
    set_process_service(service)

    try:
        tools = [
            SimpleCalculatorTool(precision=2),
            SimpleCalculatorTool(precision=4),
        ]

        wrapped_tools = wrap_tools(tools)

        assert len(wrapped_tools) == 2
        assert all(isinstance(t, ProcessIsolatedToolWrapper) for t in wrapped_tools)

    finally:
        await service.stop()
        clear_process_service()


@pytest.mark.asyncio
async def test_process_isolated_wrapper_delegates_tool_contract():
    """Test lifecycle, state, formatting, and marker properties delegate."""
    tool = LifecycleTool()
    wrapped = create_process_isolated_tool(tool)

    assert wrapped.is_async() is False
    assert wrapped.category == ToolCategory.BASIC
    assert wrapped.supports_process_isolation is True
    assert (
        wrapped.return_value_as_string({"value": "ok"}) == "formatted:{'value': 'ok'}"
    )

    await wrapped.setup(task_id="task-1")
    assert tool.setup_calls == ["task-1"]

    await wrapped.load_state_json({"value": "restored"})
    assert await wrapped.save_state_json() == {"value": "restored"}

    await wrapped.teardown(task_id="task-1")
    assert tool.teardown_calls == ["task-1"]


@pytest.mark.asyncio
async def test_tool_factory_applies_process_isolation_for_supported_tools(monkeypatch):
    """Test ToolFactory wraps supported tools through the production path."""
    clear_process_service()

    async def fake_create_registered_tools(config):
        return [LifecycleTool(), UnsupportedTool()]

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        fake_create_registered_tools,
    )
    set_process_service(ProcessService(n_workers=1, address="localhost:12366"))

    try:
        tools = await ToolFactory.create_all_tools(ToolConfig({}))

        assert len(tools) == 2
        assert all(isinstance(tool, OutputFilteredToolWrapper) for tool in tools)
        assert tools[0].is_isolated is True
        assert tools[1].is_isolated is False
    finally:
        clear_process_service()
