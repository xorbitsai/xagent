"""Inc.1 — tool concurrency-safety metadata (design §3).

Pins the contract that drives the ReAct concurrent scheduler:
- ``ToolMetadata`` carries ``read_only`` / ``concurrency_safe`` (default False).
- ``read_only=True`` implies ``concurrency_safe=True``.
- A tool may opt into ``concurrency_safe`` without being ``read_only``.
- The fields flow through the tool wrappers unchanged.
- The v1 minimal safe set (read-only web/file tools) is concurrency-safe while
  writing / process-stateful tools are not.

``calculator`` from the original design table is intentionally NOT sampled here:
the core ``calculator`` function is not wired into the vibe tool layer in this
codebase, so there is no tool object to classify. The realized v1 safe set is
web search + read-only web fetch + read-only file tools.
"""

from __future__ import annotations

from typing import Any, Mapping, Type
from unittest.mock import MagicMock

from pydantic import BaseModel

from tests.core.tools.adapters.sandboxed_tool.conftest import FakeBaseTool
from xagent.core.tools.adapters.vibe.base import ToolMetadata
from xagent.core.tools.adapters.vibe.fetch_web_content import FetchWebContentTool
from xagent.core.tools.adapters.vibe.file_tool import (
    read_file_tool,
    write_file_tool,
)
from xagent.core.tools.adapters.vibe.function import FunctionTool
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)
from xagent.core.tools.adapters.vibe.python_executor import PythonExecutorTool
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    sandbox_config,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    SandboxedToolWrapper,
)
from xagent.core.tools.adapters.vibe.web_search import WebSearchTool


def _noop(x: int = 0) -> dict[str, Any]:
    """A trivial function for FunctionTool-based tests."""
    return {"x": x}


def test_tool_metadata_defaults_are_not_concurrency_safe() -> None:
    metadata = ToolMetadata(name="anything")
    assert metadata.read_only is False
    assert metadata.concurrency_safe is False


def test_read_only_implies_concurrency_safe_on_function_tool() -> None:
    tool = FunctionTool(_noop, name="ro", read_only=True)
    assert tool.metadata.read_only is True
    assert tool.metadata.concurrency_safe is True


def test_read_only_implies_concurrency_safe_on_class_attribute() -> None:
    class ReadOnlyTool(FakeBaseTool):
        read_only = True

        @property
        def name(self) -> str:
            return "class_attr_ro"

    assert ReadOnlyTool().metadata.concurrency_safe is True


def test_concurrency_safe_can_be_declared_without_read_only() -> None:
    tool = FunctionTool(_noop, name="cs", concurrency_safe=True)
    assert tool.metadata.read_only is False
    assert tool.metadata.concurrency_safe is True


def test_default_function_tool_is_not_concurrency_safe() -> None:
    tool = FunctionTool(_noop, name="plain")
    assert tool.metadata.read_only is False
    assert tool.metadata.concurrency_safe is False


def test_output_filter_wrapper_passthrough() -> None:
    safe = FunctionTool(_noop, name="ro", read_only=True)
    unsafe = FunctionTool(_noop, name="plain")

    wrapped_safe = OutputFilteredToolWrapper(safe, 1000, 50, 5)
    wrapped_unsafe = OutputFilteredToolWrapper(unsafe, 1000, 50, 5)

    assert wrapped_safe.metadata.concurrency_safe is True
    assert wrapped_unsafe.metadata.concurrency_safe is False


def test_sandboxed_wrapper_passthrough() -> None:
    @sandbox_config()
    class SandboxReadOnlyTool(FakeBaseTool):
        read_only = True

        def args_type(self) -> Type[BaseModel]:
            return BaseModel

        def run_json_sync(self, args: Mapping[str, Any]) -> Any:
            return {}

        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            return {}

        @property
        def name(self) -> str:
            return "sandbox_ro"

    # The sandbox is irrelevant to metadata delegation; a stub suffices.
    wrapped = SandboxedToolWrapper(SandboxReadOnlyTool(), MagicMock())
    assert wrapped.metadata.concurrency_safe is True


def test_builtin_read_only_tools_are_concurrency_safe() -> None:
    assert WebSearchTool().metadata.concurrency_safe is True
    assert FetchWebContentTool().metadata.concurrency_safe is True
    assert read_file_tool.metadata.concurrency_safe is True


def test_builtin_writing_or_stateful_tools_are_not_concurrency_safe() -> None:
    assert write_file_tool.metadata.concurrency_safe is False
    assert PythonExecutorTool().metadata.concurrency_safe is False


def test_abstract_base_tool_metadata_exposes_concurrency_fields() -> None:
    # The metadata object must always carry the fields (even when False) so the
    # scheduler can read them uniformly across every tool.
    assert "read_only" in PythonExecutorTool().metadata.model_dump()
    assert "concurrency_safe" in PythonExecutorTool().metadata.model_dump()


def test_non_idempotent_marker_reaches_metadata_on_a_real_base_tool() -> None:
    # The duplicate-write guard (#2217) reads its enrollment off metadata, so
    # a first-party tool's class-level marker has to survive the base
    # property's construction.
    class SubmitFormTool(FakeBaseTool):
        non_idempotent = True

        @property
        def name(self) -> str:
            return "submit_form"

    assert SubmitFormTool().metadata.non_idempotent is True
    assert FunctionTool(_noop, name="plain").metadata.non_idempotent is False


def test_sandboxed_wrapper_forwards_the_non_idempotent_marker() -> None:
    # Sandboxed npx/uvx MCP tools reach the ReAct loop as wrappers that
    # forward only .metadata; an enrollment signal read off the concrete tool
    # object would vanish for every one of them.
    @sandbox_config()
    class SandboxSubmitTool(FakeBaseTool):
        non_idempotent = True

        def args_type(self) -> Type[BaseModel]:
            return BaseModel

        def run_json_sync(self, args: Mapping[str, Any]) -> Any:
            return {}

        async def run_json_async(self, args: Mapping[str, Any]) -> Any:
            return {}

        @property
        def name(self) -> str:
            return "sandbox_submit"

    wrapped = SandboxedToolWrapper(SandboxSubmitTool(), MagicMock())

    assert wrapped.metadata.non_idempotent is True
    # And the guard enrolls it through that forwarded metadata alone.
    from xagent.core.agent.pattern.react.duplicate_write_guard import (
        tool_requires_duplicate_write_guard,
    )

    assert tool_requires_duplicate_write_guard(wrapped) is True
