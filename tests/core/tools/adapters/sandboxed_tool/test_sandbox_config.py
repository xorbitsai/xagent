"""Unit tests for sandbox config metadata helpers."""

from typing import Any

from tests.core.tools.adapters.sandboxed_tool.conftest import (
    FakeBaseTool,
)
from xagent.core.tools.adapters.vibe.function import FunctionTool
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    SandboxConfig,
    extract_bound_method_target,
    get_sandbox_config,
    resolve_sandbox_config,
    sandbox_config,
    set_instance_sandbox_config,
)


@sandbox_config(packages=["sqlalchemy"], env_vars=["DB_URL"])
class _ConfiguredTool(FakeBaseTool):
    """Fake tool with sandbox config metadata."""

    @property
    def name(self) -> str:
        return "configured_tool"


class _UnconfiguredTool(FakeBaseTool):
    """Fake tool without sandbox config metadata."""

    @property
    def name(self) -> str:
        return "unconfigured_tool"


@sandbox_config(packages=["sqlalchemy"])
class _PlainOwner:
    """Regular class with sandbox config metadata."""

    def say(self, code: str = "") -> dict[str, Any]:
        return {"output": code}


@sandbox_config(enabled=False)
class _DisabledTool(FakeBaseTool):
    """Fake tool with sandbox explicitly disabled."""

    @property
    def name(self) -> str:
        return "disabled_tool"


def _make_bound_method_tool() -> FunctionTool:
    """Create a FunctionTool backed by a bound method."""
    owner = _PlainOwner()
    return FunctionTool(owner.say, name="bound_method_tool")


def _make_closure_tool() -> FunctionTool:
    """Create a FunctionTool backed by a closure."""
    tool = _ConfiguredTool()

    def fake_closure(code: str = "") -> dict[str, Any]:
        return tool.run_json_sync({"code": code})

    return FunctionTool(fake_closure, name="closure_tool")


class TestSandboxConfig:
    """Tests for sandbox config helper functions."""

    def test_get_sandbox_config(self):
        """Configured classes should expose sandbox metadata."""
        config = get_sandbox_config(_ConfiguredTool())
        assert config is not None
        assert config.enabled is True
        assert config.packages == ("sqlalchemy",)
        assert config.env_vars == ("DB_URL",)

    def test_get_sandbox_config_without_decorator(self):
        """Undecorated classes should not expose sandbox metadata."""
        assert get_sandbox_config(_UnconfiguredTool()) is None

    def test_extract_bound_method_target(self):
        """Bound-method FunctionTools should expose owner and method name."""
        target = extract_bound_method_target(_make_bound_method_tool())
        assert target is not None
        instance, method_name = target
        assert isinstance(instance, _PlainOwner)
        assert method_name == "say"

    def test_resolve_sandbox_config_for_tool(self):
        """Direct tools should resolve sandbox config from their class."""
        config = resolve_sandbox_config(_ConfiguredTool())
        assert config is not None
        assert config.enabled is True

    def test_resolve_sandbox_config_for_bound_method(self):
        """Bound methods should resolve sandbox config from their owner class."""
        config = resolve_sandbox_config(_make_bound_method_tool())
        assert config is not None
        assert config.enabled is True

    def test_resolve_sandbox_config_for_closure(self):
        """Closure-backed FunctionTools should not resolve sandbox metadata."""
        config = resolve_sandbox_config(_make_closure_tool())
        assert config is None

    def test_resolve_sandbox_config_enabled_false(self):
        """Explicitly disabled tools should return config with enabled=False."""
        config = resolve_sandbox_config(_DisabledTool())
        assert config is not None
        assert config.enabled is False

    def test_set_sandbox_config(self):
        """Set sandbox config."""
        tool = _UnconfiguredTool()
        set_instance_sandbox_config(tool, SandboxConfig(packages=("mcp>=1.12.4",)))
        config = get_sandbox_config(tool)
        assert config is not None
        assert config.packages == ("mcp>=1.12.4",)
