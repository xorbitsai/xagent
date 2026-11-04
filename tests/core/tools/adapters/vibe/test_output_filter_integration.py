"""
Integration tests for output filter with tool factory.
"""

import asyncio
import inspect
import logging
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from xagent.config import TOOL_MAX_OUTPUT_LENGTH
from xagent.core.agent.context import ExecutionContext
from xagent.core.tools import tool_result_spill
from xagent.core.tools.adapters.vibe import output_filter_wrapper
from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolCategory
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.function import FunctionTool
from xagent.core.tools.adapters.vibe.output_filter import DEFAULT_TRUNCATION_MESSAGE
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool import sandbox_config
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    extract_bound_method_target,
)
from xagent.core.tools.adapters.vibe.workspace_file_tool import WorkspaceFileTools
from xagent.core.tools.core.workspace_file_tool import WorkspaceFileOperations
from xagent.core.tools.tool_result_spill import (
    SPILL_PLACEHOLDER_TEXT,
    SPILL_READ_TOOL_NAME,
    SPILL_RESERVED_RESULT_KEY,
    SpillRunBudget,
    SpillTarget,
    spill_dir_for_workspace,
    spill_record_shape_is_valid,
)
from xagent.core.tools.user_interaction import WAITING_FOR_USER_STATUS
from xagent.core.workspace import MockWorkspace, TaskWorkspace


@pytest.mark.asyncio
async def test_tool_factory_applies_filters():
    """Test that tools created by factory have output filtering."""
    config = ToolConfig(
        {
            "workspace": None,
            "max_output_length": 100,
        }
    )

    tools = await ToolFactory.create_all_tools(config)

    # Check that tools were created
    assert len(tools) > 0

    # Find any wrapped tool (all tools should be wrapped with _filter)
    wrapped_tools = [t for t in tools if hasattr(t, "_filter")]
    assert len(wrapped_tools) > 0, "No tools with output filter found"

    # Check that the filter has the correct configuration
    tool = wrapped_tools[0]
    assert hasattr(tool, "_filter")
    assert tool._filter.max_chars == 100


@pytest.mark.asyncio
async def test_filtered_tool_execution():
    """Test that filtered tools truncate output correctly when executed."""
    from langchain_core.tools.structured import StructuredTool
    from pydantic import BaseModel, Field

    from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolMetadata
    from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
        OutputFilteredToolWrapper,
    )

    # Create a simple test tool that returns predictable long output
    class TestInput(BaseModel):
        text: str = Field(description="Text to repeat")

    def test_long_output_func(text: str) -> str:
        """Return the input text repeated 100 times for testing output filtering."""
        return text * 100

    # Create a StructuredTool
    langchain_tool = StructuredTool.from_function(
        func=test_long_output_func,
        name="test_long_output",
        description="Test tool that returns long output",
        args_schema=TestInput,
    )

    # Create AbstractBaseTool wrapper
    class TestTool(AbstractBaseTool):
        @property
        def name(self) -> str:
            return "test_long_output"

        @property
        def description(self) -> str:
            return "Test tool that returns long output"

        @property
        def metadata(self) -> ToolMetadata:
            return ToolMetadata(
                name="test_long_output",
                description="Test tool that returns long output",
                category="BASIC",
            )

        def args_type(self):
            return TestInput

        def return_type(self):
            return str

        def state_type(self):
            return None

        def is_async(self):
            return False

        def run_json_sync(self, args):
            result = langchain_tool.invoke(args)
            return result

        async def run_json_async(self, args):
            return self.run_json_sync(args)

    # Wrap it with the same wrapper used by ToolFactory
    wrapped = OutputFilteredToolWrapper(
        target_tool=TestTool(),
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
    )

    # Execute the tool and verify truncation
    result = wrapped.run_json_sync({"text": "abcdefghij" * 10})  # 100 chars

    # Result should be truncated to 50 chars + message
    assert len(result) <= 50 + len(DEFAULT_TRUNCATION_MESSAGE)
    assert result.endswith(DEFAULT_TRUNCATION_MESSAGE)
    assert result.startswith("abcdefghij")


@pytest.mark.asyncio
async def test_default_max_output_length():
    """Test that default max output length is 50K characters."""
    config = ToolConfig({"workspace": None})

    tools = await ToolFactory.create_all_tools(config)

    # Check that at least one tool was created
    assert len(tools) > 0

    # Check that tools have the default limit
    for tool in tools:
        if hasattr(tool, "_filter"):
            assert tool._filter.max_chars == 50 * 1024


@pytest.mark.asyncio
async def test_hardcoded_truncation_message():
    """Test that truncation message uses the hardcoded default from output_filter.py."""
    config = ToolConfig(
        {
            "workspace": None,
            "max_output_length": 10,
        }
    )

    tools = await ToolFactory.create_all_tools(config)

    # Find a tool and verify truncation message is used
    for tool in tools:
        if hasattr(tool, "_filter"):
            # The filter uses the hardcoded message from output_filter.py
            assert tool._filter.max_chars == 10
            break


# --- wrapper integration (strip / spill / bypass branches) ----------------


def _wrapper(spill_target=None, max_chars=50, max_fields=1000, max_recursion=20):
    return OutputFilteredToolWrapper(
        target_tool=SimpleNamespace(name="acme"),
        max_chars=max_chars,
        max_fields=max_fields,
        max_recursion=max_recursion,
        spill_target=spill_target,
    )


def test_wrapper_spills_oversized_dict_result_instead_of_truncating(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    big_text = "x" * 100
    result = {
        "content": [{"type": "text", "text": big_text}],
        "structured_content": None,
        "is_error": False,
    }
    filtered = wrapper._filter_result(result)

    assert DEFAULT_TRUNCATION_MESSAGE not in str(filtered)
    assert filtered["content"][0]["text"] == SPILL_PLACEHOLDER_TEXT
    assert filtered["is_error"] is False
    records = filtered[SPILL_RESERVED_RESULT_KEY]
    assert len(records) == 1
    written = spill_dir / records[0]["relative_path"].split("/")[-1]
    assert written.read_text(encoding="utf-8") == big_text


def test_a_real_spill_is_invisible_to_get_output_files(tmp_path):
    """The file a real spill writes must not turn into a deliverable the
    model can hand back to the user. It lands inside the workspace's
    engine-owned tool-results directory, which get_output_files() already
    excludes from every listing -- this pins that the two mechanisms
    actually meet, using a real TaskWorkspace rather than a bare tmp_path
    and manually planted files the way test_workspace_engine_owned_dir.py
    does."""
    workspace = TaskWorkspace("task-spill", str(tmp_path))
    spill_dir = workspace.output_dir / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    big_text = "x" * 100
    result = {
        "content": [{"type": "text", "text": big_text}],
        "structured_content": None,
        "is_error": False,
    }
    filtered = wrapper._filter_result(result)
    records = filtered[SPILL_RESERVED_RESULT_KEY]
    written = spill_dir / records[0]["relative_path"].split("/")[-1]
    assert written.exists()  # the spill really happened

    listed_paths = {entry["file_path"] for entry in workspace.get_output_files()}
    assert str(written) not in listed_paths


def test_a_real_spill_refuses_file_tool_writes_but_reads_back_whole(tmp_path):
    """The same real spill file, seen from the two sides of the workspace
    file tools: writing into the engine's directory is refused, while
    read_tool_result returns exactly the value that was stored in place of
    the placeholder."""
    workspace = TaskWorkspace("task-spill", str(tmp_path))
    spill_dir = workspace.output_dir / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    big_text = "x" * 100
    result = {
        "content": [{"type": "text", "text": big_text}],
        "structured_content": None,
        "is_error": False,
    }
    filtered = wrapper._filter_result(result)
    relative_path = filtered[SPILL_RESERVED_RESULT_KEY][0]["relative_path"]
    assert filtered["content"][0]["text"] == SPILL_PLACEHOLDER_TEXT
    file_tools = WorkspaceFileTools(workspace)

    with pytest.raises(ValueError):
        file_tools.write_file("output/tool-results/x.json", "[1]")
    assert not (spill_dir / "x.json").exists()

    assert file_tools.read_tool_result(relative_path) == {
        "relative_path": relative_path,
        "output": big_text,
    }


def test_wrapper_without_spill_target_truncates_as_before(tmp_path):
    wrapper = _wrapper(spill_target=None)
    result = {"content": [{"type": "text", "text": "x" * 100}]}
    filtered = wrapper._filter_result(result)
    assert DEFAULT_TRUNCATION_MESSAGE in filtered["content"][0]["text"]
    assert SPILL_RESERVED_RESULT_KEY not in filtered


def test_wrapper_small_results_are_byte_identical_with_a_spill_target(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=50))
    result = {"output": "small value", "count": 3}
    filtered = wrapper._filter_result(result)
    assert filtered == result
    assert not spill_dir.exists()


def test_a_waiting_for_user_result_is_never_spilled(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    result = {
        "status": "waiting_for_user",
        "interaction_id": "i1",
        "message_type": "question",
        "message": "please answer",
        "interactions": [{"prompt": "pick one"}],
        "records": "r" * 100,  # oversized sibling, not part of the card
    }
    filtered = wrapper._filter_result(result)
    assert SPILL_RESERVED_RESULT_KEY not in filtered
    assert not spill_dir.exists()
    assert DEFAULT_TRUNCATION_MESSAGE in filtered["records"]
    assert filtered["status"] == WAITING_FOR_USER_STATUS
    assert filtered["message"] == "please answer"
    assert filtered["interactions"] == [{"prompt": "pick one"}]


def test_a_classified_failure_result_is_never_spilled(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    result = {
        "success": False,
        "is_error": True,
        "status": "error",
        "error": "short error",
        "output": "o" * 100,  # oversized, not part of the failure signal
    }
    filtered = wrapper._filter_result(result)
    assert SPILL_RESERVED_RESULT_KEY not in filtered
    assert not spill_dir.exists()
    assert filtered["success"] is False
    assert filtered["is_error"] is True
    assert filtered["error"] == "short error"
    assert DEFAULT_TRUNCATION_MESSAGE in filtered["output"]


def test_the_bypass_branches_read_the_post_spill_object(tmp_path):
    """Neither the waiting-for-user nor the classified-failure envelope, but
    an oversized ``output`` sibling: the bypass branch's final `return
    filtered` still reads off the post-spill object, not the original one."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    result = {"output": "o" * 100}
    filtered = wrapper._filter_result(result)
    assert filtered["output"] == SPILL_PLACEHOLDER_TEXT
    assert filtered[SPILL_RESERVED_RESULT_KEY][0]["value_path"] == "output"


def test_wrapper_strips_a_forged_reserved_key_even_without_a_spill_target(caplog):
    wrapper = _wrapper(spill_target=None)
    forged = [{"relative_path": "tool-results/evil.json"}]
    result = {"output": "ok", SPILL_RESERVED_RESULT_KEY: forged}
    with caplog.at_level("WARNING"):
        filtered = wrapper._filter_result(result)
    assert filtered.get(SPILL_RESERVED_RESULT_KEY) != forged
    assert SPILL_RESERVED_RESULT_KEY not in filtered


@pytest.mark.asyncio
async def test_the_async_path_also_strips_a_forged_reserved_key():
    async def run_json_async(args):
        forged = [{"relative_path": "tool-results/evil.json"}]
        return {"output": "ok", SPILL_RESERVED_RESULT_KEY: forged}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=None,
    )
    filtered = await wrapper.run_json_async({})
    assert SPILL_RESERVED_RESULT_KEY not in filtered


def test_a_real_report_key_survives_while_a_forged_one_is_replaced(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80)
    )
    forged = [{"relative_path": "tool-results/evil.json"}]
    result = {
        "content": [{"type": "text", "text": "x" * 100}],
        "is_error": False,
        SPILL_RESERVED_RESULT_KEY: forged,
    }
    filtered = wrapper._filter_result(result)
    records = filtered[SPILL_RESERVED_RESULT_KEY]
    assert records != forged
    assert records[0]["value_path"] == "content[0].text"


def test_the_wrapper_uses_the_spill_module_s_only_failure_classifier():
    assert (
        output_filter_wrapper.is_classified_tool_failure
        is tool_result_spill.is_classified_tool_failure
    )


def test_the_wrapper_module_keeps_no_private_failure_classifier():
    assert not hasattr(output_filter_wrapper, "_is_classified_tool_failure")


# --- the spill entry point runs off the event loop -------------------------


def _thread_recording_stub(sink):
    """Stand in for spill_oversized_values, recording only which thread
    called it and the run_budget it was handed -- not doing any real spill
    work, so thread identity is the one thing these tests measure."""

    def _record(result, target, *, tool_name, max_recursion, run_budget):
        sink["thread"] = threading.get_ident()
        sink["run_budget"] = run_budget
        return result, []

    return _record


@pytest.mark.asyncio
async def test_run_json_async_offloads_the_spill_entry_point_to_a_worker_thread(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    async def run_json_async(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    caller_thread = threading.get_ident()
    await wrapper.run_json_async({})
    assert sink["thread"] != caller_thread


@pytest.mark.asyncio
async def test_async_func_wrapper_offloads_the_spill_entry_point_to_a_worker_thread(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    async def original(*args, **kwargs):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", func=original)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    caller_thread = threading.get_ident()
    await wrapper.func()
    assert sink["thread"] != caller_thread


def test_run_json_sync_keeps_the_spill_entry_point_on_the_calling_thread(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    def run_json_sync(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_sync=run_json_sync)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    caller_thread = threading.get_ident()
    wrapper.run_json_sync({})
    assert sink["thread"] == caller_thread


def test_sync_func_wrapper_keeps_the_spill_entry_point_on_the_calling_thread(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    def original(*args, **kwargs):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", func=original)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    caller_thread = threading.get_ident()
    wrapper.func()
    assert sink["thread"] == caller_thread


@pytest.mark.asyncio
async def test_a_wrapper_without_a_target_does_not_hop_to_a_worker_thread(
    monkeypatch,
):
    hop_calls = []
    real_to_thread = output_filter_wrapper.asyncio.to_thread

    async def _spy_to_thread(func, *args, **kwargs):
        hop_calls.append(True)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(output_filter_wrapper.asyncio, "to_thread", _spy_to_thread)

    async def run_json_async(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=None,
    )
    await wrapper.run_json_async({})
    assert hop_calls == []


@pytest.mark.asyncio
async def test_the_run_budget_inside_the_worker_thread_is_the_same_object(
    monkeypatch, tmp_path
):
    sink = {}
    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _thread_recording_stub(sink)
    )

    async def run_json_async(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    await wrapper.run_json_async({})
    assert sink["run_budget"] is wrapper._spill_run_budget


@pytest.mark.asyncio
async def test_two_wrappers_accumulate_on_one_shared_budget_across_the_thread_hop(
    tmp_path,
):
    spill_dir = tmp_path / "output" / "tool-results"
    budget = SpillRunBudget()

    async def run_json_async(args):
        return {"content": [{"type": "text", "text": "x" * 100}]}

    wrapper_one = OutputFilteredToolWrapper(
        target_tool=SimpleNamespace(name="one", run_json_async=run_json_async),
        max_chars=80,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80),
        spill_run_budget=budget,
    )
    wrapper_two = OutputFilteredToolWrapper(
        target_tool=SimpleNamespace(name="two", run_json_async=run_json_async),
        max_chars=80,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80),
        spill_run_budget=budget,
    )
    await wrapper_one.run_json_async({})
    await wrapper_two.run_json_async({})
    assert budget.files_written == 2


# --- a failed spill degrades to plain filtering -----------------------------


class _SpillHostileValue:
    """Filtered without str(), but json.dumps can only render it with str()."""

    def model_dump(self) -> dict[str, str]:
        return {"kind": "unrenderable"}

    def __str__(self) -> str:
        raise RuntimeError("this value cannot be rendered")

    __repr__ = __str__


def test_a_value_that_cannot_be_rendered_falls_back_to_plain_filtering(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    no_target = _wrapper(spill_target=None, max_chars=80)
    with_target = _wrapper(
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80), max_chars=80
    )
    baseline = no_target._filter_result(
        {"payload": _SpillHostileValue(), "note": "n" * 200}
    )
    degraded = with_target._filter_result(
        {"payload": _SpillHostileValue(), "note": "n" * 200}
    )
    assert degraded == baseline


def test_a_value_that_cannot_be_rendered_leaves_no_spill_file(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80), max_chars=80
    )
    wrapper._filter_result({"payload": _SpillHostileValue(), "note": "n" * 200})
    assert not spill_dir.exists()


def test_a_failed_spill_logs_one_warning_naming_the_tool_and_the_exception_type(
    tmp_path, caplog
):
    """Scoped to output_filter_wrapper's own logger: rendering the same
    hostile value also trips an unrelated, pre-existing warning inside the
    output filter's own Pydantic-model fallback (it tries to reconstruct
    the value from its filtered model_dump() and that constructor call
    fails too) -- a fact about that filter, not about the spill boundary
    this test is pinning."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80), max_chars=80
    )
    with caplog.at_level("WARNING"):
        wrapper._filter_result({"payload": _SpillHostileValue(), "note": "n" * 200})
    warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and r.name == output_filter_wrapper.__name__
    ]
    assert len(warnings) == 1
    assert "acme" in warnings[0].getMessage()
    assert "RuntimeError" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_cancelling_an_async_tool_call_is_not_swallowed_by_the_spill_boundary(
    monkeypatch, tmp_path, caplog
):
    def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _raise_cancelled
    )

    async def run_json_async(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    with caplog.at_level("WARNING"):
        with pytest.raises(asyncio.CancelledError):
            await wrapper.run_json_async({})
    assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_a_keyboard_interrupt_is_not_swallowed_by_the_spill_boundary(
    monkeypatch, tmp_path, caplog
):
    def _raise_keyboard_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        output_filter_wrapper, "spill_oversized_values", _raise_keyboard_interrupt
    )

    def run_json_sync(args):
        return {"output": "value"}

    target = SimpleNamespace(name="acme", run_json_sync=run_json_sync)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(
            spill_dir=str(tmp_path / "output" / "tool-results"), max_chars=50
        ),
    )
    with caplog.at_level("WARNING"):
        with pytest.raises(KeyboardInterrupt):
            wrapper.run_json_sync({})
    assert not any(r.levelname == "WARNING" for r in caplog.records)


# --- the factory resolves a tool set's spill target from its reader -------

FACTORY_LOGGER = "xagent.core.tools.adapters.vibe.factory"


def _only_tool_named(tools, name):
    matches = [tool for tool in tools if tool.name == name]
    assert len(matches) == 1, [tool.name for tool in tools]
    return matches[0]


def _bound_workspace(wrapper):
    """The workspace object the wrapped file tool's bound instance holds."""
    target = extract_bound_method_target(wrapper._target)
    assert target is not None
    instance, _ = target
    return instance.workspace


@pytest.mark.asyncio
async def test_factory_wires_a_spill_target_when_read_tool_result_is_workspace_bound(
    tmp_path,
):
    config = ToolConfig(
        {"workspace": {"task_id": "spill-target-test", "base_dir": str(tmp_path)}}
    )
    tools = await ToolFactory.create_all_tools(config)
    workspace = _bound_workspace(_only_tool_named(tools, SPILL_READ_TOOL_NAME))
    assert isinstance(workspace, TaskWorkspace)
    read_file = _only_tool_named(tools, "read_file")

    target = read_file._spill_target

    assert target is not None
    assert Path(target.spill_dir).parts[-2:] == (
        tool_result_spill.SPILL_WORKSPACE_OUTPUT_DIR_NAME,
        tool_result_spill.SPILL_DIR_NAME,
    )
    assert target.spill_dir == spill_dir_for_workspace(workspace.workspace_dir)
    assert target.max_chars == read_file._filter.max_chars


@pytest.mark.asyncio
async def test_factory_leaves_spill_target_none_without_workspace(caplog):
    config = ToolConfig({"workspace": None})
    with caplog.at_level(logging.INFO, logger=FACTORY_LOGGER):
        tools = await ToolFactory.create_all_tools(config)

    assert not any(tool.name == SPILL_READ_TOOL_NAME for tool in tools)
    wrappers = [tool for tool in tools if isinstance(tool, OutputFilteredToolWrapper)]
    assert wrappers
    assert all(wrapper._spill_target is None for wrapper in wrappers)
    disabled = [
        record.getMessage()
        for record in caplog.records
        if "spill disabled" in record.getMessage()
    ]
    assert disabled
    assert all("no read_tool_result in the tool set" in message for message in disabled)


@pytest.mark.asyncio
async def test_factory_leaves_spill_target_none_for_a_mock_workspace(tmp_path, caplog):
    """The tool-listing endpoint builds its tool set with task_id
    "tools_list", which binds the file tools, read_tool_result included, to
    a MockWorkspace: a workspace that never exists on disk must not be
    given a spill target."""
    config = ToolConfig(
        {"workspace": {"task_id": "tools_list", "base_dir": str(tmp_path)}}
    )
    with caplog.at_level(logging.INFO, logger=FACTORY_LOGGER):
        tools = await ToolFactory.create_all_tools(config)
    workspace = _bound_workspace(_only_tool_named(tools, SPILL_READ_TOOL_NAME))
    assert isinstance(workspace, MockWorkspace)

    wrappers = [tool for tool in tools if isinstance(tool, OutputFilteredToolWrapper)]
    assert wrappers
    assert all(wrapper._spill_target is None for wrapper in wrappers)
    disabled = [
        record.getMessage()
        for record in caplog.records
        if "spill disabled" in record.getMessage()
    ]
    assert disabled
    assert all(
        "read_tool_result is not bound to a task workspace" in message
        for message in disabled
    )
    assert not Path(spill_dir_for_workspace(workspace.workspace_dir)).exists()


class _ExtensionReader(AbstractBaseTool):
    """A tool named read_tool_result that is not a FunctionTool: no func."""

    category = ToolCategory.OTHER

    class _Args(BaseModel):
        path: str = ""

    @property
    def name(self) -> str:
        return SPILL_READ_TOOL_NAME

    @property
    def description(self) -> str:
        return "Read a stored result through a task runtime extension."

    def args_type(self):
        return self._Args

    def return_type(self):
        return self._Args

    def run_json_sync(self, args):
        return {"output": "ok"}

    async def run_json_async(self, args):
        return {"output": "ok"}


@pytest.mark.asyncio
async def test_factory_leaves_spill_target_none_for_a_read_tool_result_that_is_not_a_function_tool(
    tmp_path, caplog
):
    """With the file tools disabled, a task runtime extension may contribute
    a tool under the reader's name. Having no bound method, it cannot name
    a workspace: the tool set is still built, with no spill target
    anywhere."""
    config = ToolConfig(
        {
            "workspace": {"task_id": "extension-reader", "base_dir": str(tmp_path)},
            "file_tools_enabled": False,
        }
    )
    extension_reader = _ExtensionReader()
    with caplog.at_level(logging.INFO, logger=FACTORY_LOGGER):
        tools = await ToolFactory.create_all_tools(
            config, additional_tools=[extension_reader]
        )

    assert _only_tool_named(tools, SPILL_READ_TOOL_NAME)._target is extension_reader
    wrappers = [tool for tool in tools if isinstance(tool, OutputFilteredToolWrapper)]
    assert wrappers
    assert all(wrapper._spill_target is None for wrapper in wrappers)
    disabled = [
        record.getMessage()
        for record in caplog.records
        if "spill disabled" in record.getMessage()
    ]
    assert disabled
    assert all(
        "read_tool_result is not bound to a task workspace" in message
        for message in disabled
    )


@pytest.mark.asyncio
async def test_factory_leaves_spill_target_none_for_a_closure_read_tool_result(
    tmp_path, caplog, monkeypatch
):
    """A FunctionTool under the reader's name whose function is a closure,
    not a bound method, may well read from a real workspace, but the factory
    has no instance to find that workspace on: it passes the FunctionTool
    check and stops at the bound-method one, and the tool set gets no spill
    target anywhere."""
    config = ToolConfig(
        {
            "workspace": {"task_id": "closure-reader", "base_dir": str(tmp_path)},
            "file_tools_enabled": False,
        }
    )
    closed_over = TaskWorkspace("closure-reader", str(tmp_path))

    def read_tool_result(path: str = "") -> dict:
        """Read a stored result through a closure over a workspace."""
        return {"output": str(closed_over.workspace_dir / path)}

    closure_reader = FunctionTool(read_tool_result, name=SPILL_READ_TOOL_NAME)
    extracted = []
    real_extract = sandbox_config.extract_bound_method_target

    def recording_extract(tool):
        target = real_extract(tool)
        extracted.append((tool, target))
        return target

    monkeypatch.setattr(
        sandbox_config, "extract_bound_method_target", recording_extract
    )
    with caplog.at_level(logging.INFO, logger=FACTORY_LOGGER):
        tools = await ToolFactory.create_all_tools(
            config, additional_tools=[closure_reader]
        )

    assert _only_tool_named(tools, SPILL_READ_TOOL_NAME)._target is closure_reader
    assert (closure_reader, None) in extracted
    wrappers = [tool for tool in tools if isinstance(tool, OutputFilteredToolWrapper)]
    assert wrappers
    assert all(wrapper._spill_target is None for wrapper in wrappers)
    disabled = [
        record for record in caplog.records if "spill disabled" in record.getMessage()
    ]
    assert disabled
    assert all(record.levelno == logging.INFO for record in disabled)
    assert all(
        "read_tool_result is not bound to a task workspace" in record.getMessage()
        for record in disabled
    )


# --- a production tool set wires a spill target ---------------------------


@pytest.mark.asyncio
async def test_a_production_tool_set_now_wires_a_spill_target(tmp_path):
    """A workspace-bound tool set gives every wrapper but the stored-result
    reader the same SpillTarget and every wrapper the same SpillRunBudget,
    pointed at this workspace's own spill directory. Building the tool set
    does not create that directory; the first spill does. The tool-listing
    construction, bound to a MockWorkspace, wires no target at all."""
    config = ToolConfig(
        {"workspace": {"task_id": "spill-wiring-test", "base_dir": str(tmp_path)}}
    )
    tools = await ToolFactory.create_all_tools(config)
    workspace = _bound_workspace(_only_tool_named(tools, "read_file"))
    reader = _only_tool_named(tools, SPILL_READ_TOOL_NAME)
    wrappers = [tool for tool in tools if isinstance(tool, OutputFilteredToolWrapper)]
    others = [wrapper for wrapper in wrappers if wrapper is not reader]

    assert reader._spill_target is None
    assert others
    assert all(wrapper._spill_target is not None for wrapper in others)
    assert len({id(wrapper._spill_target) for wrapper in others}) == 1
    assert len({id(wrapper._spill_run_budget) for wrapper in wrappers}) == 1
    spill_dir = spill_dir_for_workspace(workspace.workspace_dir)
    assert others[0]._spill_target.spill_dir == spill_dir
    assert not Path(spill_dir).exists()

    listing = await ToolFactory.create_all_tools(
        ToolConfig({"workspace": {"task_id": "tools_list", "base_dir": str(tmp_path)}})
    )
    listed = [tool for tool in listing if isinstance(tool, OutputFilteredToolWrapper)]
    assert listed
    assert all(wrapper._spill_target is None for wrapper in listed)


@pytest.mark.asyncio
async def test_all_wrappers_from_one_construction_share_one_run_budget_object(
    tmp_path,
):
    config = ToolConfig(
        {"workspace": {"task_id": "spill-budget-test", "base_dir": str(tmp_path)}}
    )
    tools = await ToolFactory.create_all_tools(config)
    wrappers = [tool for tool in tools if hasattr(tool, "_spill_run_budget")]

    assert len(wrappers) > 1
    first = wrappers[0]._spill_run_budget
    assert isinstance(first, SpillRunBudget)
    assert all(wrapper._spill_run_budget is first for wrapper in wrappers)


def _store_for_reading(spill_dir, value, max_chars):
    """Spill ``value`` into ``spill_dir`` as one file; return its relative path.

    ``max_chars`` must sit below ``value`` and at or above each of its
    items, so that ``value`` itself is the one spill point."""
    _, records = tool_result_spill.spill_oversized_values(
        {"value": value},
        SpillTarget(spill_dir=str(spill_dir), max_chars=max_chars),
        tool_name="acme",
        max_recursion=20,
    )
    assert len(records) == 1
    return records[0]["relative_path"]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_chars", [12_000, 8_000])
async def test_the_read_back_tool_gets_no_spill_target(tmp_path, max_chars):
    """What read_tool_result returns is never spilled again, in either of its
    success shapes: the whole shape carries the text in output, and the
    preview shape for one item longer than a page carries it in
    content_preview, which is not an envelope field. Each shape here
    carries one full page, which is 12,000 characters at a 12,000 limit
    and 8,000 at an 8,000 limit, and with the output limit lowered that far
    a spill target on this wrapper would write a file for either shape or
    put the placeholder where the text was. Without one, the reply is
    exactly what the output filter alone makes of it, and since a page is
    never longer than the limit, that is the reply unchanged."""
    config = ToolConfig(
        {
            "workspace": {
                "task_id": f"read-back-{max_chars}",
                "base_dir": str(tmp_path),
            },
            "max_output_length": max_chars,
        }
    )
    tools = await ToolFactory.create_all_tools(config)
    reader = _only_tool_named(tools, SPILL_READ_TOOL_NAME)
    assert reader._spill_target is None
    spill_dir = Path(
        spill_dir_for_workspace(
            _bound_workspace(_only_tool_named(tools, "read_file")).workspace_dir
        )
    )
    page_chars = tool_result_spill.spill_read_page_chars(max_chars)
    whole_text = "x" * page_chars
    whole_path = _store_for_reading(spill_dir, whole_text, max_chars=100)
    long_items = ["y" * 15_000, "z" * 15_000]
    preview_path = _store_for_reading(spill_dir, long_items, max_chars=20_000)
    stored_files = sorted(spill_dir.iterdir())
    plain = OutputFilteredToolWrapper(
        target_tool=reader._target,
        max_chars=max_chars,
        max_fields=config.get_max_field_count(),
        max_recursion=config.get_max_recursion_depth(),
    )

    for args, text_key in (
        ({"path": whole_path}, "output"),
        ({"path": preview_path, "start": 1, "end": 1}, "content_preview"),
    ):
        raw = await reader._target.run_json_async(args)
        assert len(raw[text_key]) == page_chars
        reply = await reader.run_json_async(args)

        assert SPILL_RESERVED_RESULT_KEY not in reply
        assert reply[text_key] != SPILL_PLACEHOLDER_TEXT
        assert reply == plain._filter_result(raw)
        # plain is built like reader, so the equality above holds by
        # construction; these pin that the filter leaves the text alone.
        assert reply == raw
        assert DEFAULT_TRUNCATION_MESSAGE not in reply[text_key]
        assert sorted(spill_dir.iterdir()) == stored_files
    assert set(raw) >= {
        "content_preview",
        "content_truncated",
        "original_chars",
        "item_count",
        "offset",
        "instruction",
    }


async def _read_back_tool(tmp_path, task_id, max_output_length=None):
    """The factory-built, filter-wrapped read_tool_result and its spill
    directory, for a workspace under ``tmp_path``."""
    settings = {"workspace": {"task_id": task_id, "base_dir": str(tmp_path)}}
    if max_output_length is not None:
        settings["max_output_length"] = max_output_length
    tools = await ToolFactory.create_all_tools(ToolConfig(settings))
    reader = _only_tool_named(tools, SPILL_READ_TOOL_NAME)
    spill_dir = spill_dir_for_workspace(_bound_workspace(reader).workspace_dir)
    return reader, spill_dir


def _stated_page_chars(reader):
    """The per-call character count the read tool's description states."""
    stated = re.search(r"returns at most ([\d,]+) characters", reader.description)
    assert stated is not None, reader.description
    return int(stated.group(1).replace(",", ""))


@pytest.mark.asyncio
async def test_read_back_pages_at_a_lower_output_limit_join_to_the_stored_text(
    tmp_path,
):
    """With the output limit at 8,000, one 15,000-character line read
    through the factory's wrapped read_tool_result, stepping offset by the
    page size the description states, comes back whole: each page fits the
    limit, so the output filter cuts nothing and no stretch is skipped."""
    reader, spill_dir = await _read_back_tool(tmp_path, "read-back-pages", 8_000)
    step = _stated_page_chars(reader)
    line = "".join(chr(ord("a") + index % 26) for index in range(15_000))
    path = _store_for_reading(spill_dir, line, max_chars=100)

    pages = []
    offset = 0
    while True:
        reply = await reader.run_json_async(
            {"path": path, "start": 1, "end": 1, "offset": offset}
        )
        assert DEFAULT_TRUNCATION_MESSAGE not in reply["content_preview"]
        pages.append(reply["content_preview"])
        if not reply["content_truncated"]:
            break
        offset += step

    assert step == 8_000
    assert [len(page) for page in pages] == [8_000, 7_000]
    assert "".join(pages) == line


@pytest.mark.asyncio
async def test_read_back_of_an_item_between_the_page_size_and_the_cap_is_paged(
    tmp_path,
):
    """A 10,000-character line is shorter than SPILL_READ_MAX_CHARS but
    longer than an 8,000-character output limit. Through the factory's
    wrapped read_tool_result it comes back as a preview, not as one whole
    reply the filter would then cut: two pages of 8,000 and 2,000 that join
    to the stored text, neither carrying the truncation marker."""
    reader, spill_dir = await _read_back_tool(tmp_path, "read-back-between", 8_000)
    line = "".join(chr(ord("a") + index % 26) for index in range(10_000))
    path = _store_for_reading(spill_dir, line, max_chars=100)

    first = await reader.run_json_async({"path": path, "start": 1, "end": 1})
    assert "output" not in first
    assert first["content_truncated"] is True
    assert DEFAULT_TRUNCATION_MESSAGE not in first["content_preview"]
    second = await reader.run_json_async(
        {"path": path, "start": 1, "end": 1, "offset": _stated_page_chars(reader)}
    )
    assert second["content_truncated"] is False
    assert DEFAULT_TRUNCATION_MESSAGE not in second["content_preview"]
    assert [len(first["content_preview"]), len(second["content_preview"])] == [
        8_000,
        2_000,
    ]
    assert first["content_preview"] + second["content_preview"] == line


@pytest.mark.asyncio
async def test_read_back_page_size_is_stated_in_the_description_not_an_argument(
    tmp_path,
):
    """The page size is fixed when the tool is built and stated in its
    description; the model's argument list, generated from the method
    signature, stays path, start, end and offset."""
    reader, _ = await _read_back_tool(tmp_path, "read-back-args", 8_000)

    assert "One call returns at most 8,000 characters" in reader.description
    assert set(reader.args_type().model_fields) == {"path", "start", "end", "offset"}


@pytest.mark.asyncio
async def test_read_back_page_size_follows_the_config_override_not_the_environment(
    tmp_path, monkeypatch
):
    """The output filter takes its limit from the config object, and a
    ToolConfig max_output_length overrides XAGENT_TOOL_MAX_OUTPUT_LENGTH
    there; the page size is read from the same object, so it follows the
    override. Without an override both follow the environment."""
    monkeypatch.setenv(TOOL_MAX_OUTPUT_LENGTH, "5000")

    overridden, _ = await _read_back_tool(tmp_path, "read-back-override", 8_000)
    from_env, _ = await _read_back_tool(tmp_path, "read-back-env")

    for reader, limit in ((overridden, 8_000), (from_env, 5_000)):
        instance, _ = extract_bound_method_target(reader._target)
        assert reader._filter.max_chars == limit
        assert instance.read_page_chars == limit
        assert instance.inner.read_page_chars == limit
        assert _stated_page_chars(reader) == limit


def test_read_tool_result_pages_at_the_size_its_operations_object_was_built_with(
    tmp_path,
):
    """The page size is a constructor argument of WorkspaceFileOperations,
    not a read_tool_result argument. Without one the page is
    SPILL_READ_MAX_CHARS, so a 10,000-character line comes back whole;
    built with 8,000 it comes back as an 8,000-character preview.
    WorkspaceFileTools hands its page size to its inner operations object,
    so its read_tool_result takes the same parameters as the core method
    and returns the same reply."""
    workspace = TaskWorkspace(id="page-size-ctor", base_dir=str(tmp_path))
    spill_dir = spill_dir_for_workspace(workspace.workspace_dir)
    line = "".join(chr(ord("a") + index % 26) for index in range(10_000))
    path = _store_for_reading(spill_dir, line, max_chars=100)

    assert WorkspaceFileOperations(workspace).read_tool_result(path) == {
        "relative_path": path,
        "output": line,
    }
    paged = WorkspaceFileOperations(workspace, page_chars=8_000).read_tool_result(path)
    assert paged["content_preview"] == line[:8_000]
    assert paged["content_truncated"] is True

    tools = WorkspaceFileTools(workspace, page_chars=8_000)
    assert tools.read_page_chars == tools.inner.read_page_chars == 8_000
    assert tools.read_tool_result(path) == paged
    assert list(inspect.signature(WorkspaceFileTools.read_tool_result).parameters) == (
        list(inspect.signature(WorkspaceFileOperations.read_tool_result).parameters)
    )


@pytest.mark.asyncio
async def test_the_factory_the_context_and_the_read_tool_agree_on_the_spill_directory(
    tmp_path,
):
    """The factory's target, the execution context that registers a spill,
    and read_tool_result all take the directory from spill_dir_for_workspace.
    One workspace object feeds all three here, so this pins the shared
    spelling; test_the_factory_and_the_runner_resolve_one_spill_directory
    (tests/core/agent/test_agent_service_spill_directory.py) covers two
    separately built workspace objects agreeing."""
    config = ToolConfig(
        {"workspace": {"task_id": "spill-dir-agree", "base_dir": str(tmp_path)}}
    )
    tools = await ToolFactory.create_all_tools(config)
    read_file = _only_tool_named(tools, "read_file")
    reader = _only_tool_named(tools, SPILL_READ_TOOL_NAME)
    workspace = _bound_workspace(read_file)
    ctx = ExecutionContext()
    ctx.attach_workspace(workspace.id, str(workspace.workspace_dir))

    expected = spill_dir_for_workspace(workspace.workspace_dir)
    assert read_file._spill_target.spill_dir == ctx._spill_dir() == expected

    spilled, records = tool_result_spill.spill_oversized_values(
        {"content": "x" * (read_file._spill_target.max_chars + 1)},
        read_file._spill_target,
        tool_name="acme",
        max_recursion=20,
    )
    relative_path = records[0]["relative_path"]
    ctx.add_tool_result("acme", spilled)
    assert [record["relative_path"] for record in ctx.spilled_results] == [
        relative_path
    ]
    listing = await reader.run_json_async({})
    assert [entry["relative_path"] for entry in listing["stored_results"]] == [
        relative_path
    ]


# --- the engine's own spill report bypasses ordinary output filtering ------


def test_a_short_character_limit_leaves_the_spill_report_intact(tmp_path):
    """A generated relative_path (about 55 characters for a short tool name)
    is longer than max_chars=50. The engine's report is not subject to that
    cap: the record stays well-formed, ExecutionContext registers it and its
    notice is rendered."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=50, spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=50)
    )
    filtered = wrapper._filter_result({"content": "x" * 200})

    record = filtered[SPILL_RESERVED_RESULT_KEY][0]
    assert spill_record_shape_is_valid(record)
    written = spill_dir / record["relative_path"].split("/")[-1]
    assert written.exists()

    ctx = ExecutionContext()
    ctx.attach_workspace("ws-1", str(tmp_path))
    tool = ctx.add_tool_result("acme", filtered)
    assert len(ctx.spilled_results) == 1
    assert record["relative_path"] in tool.content


def test_a_field_count_limit_does_not_drop_the_spill_report(tmp_path):
    """Regression for the report key being appended after every tool key
    and then counted against max_fields: a root that already has
    max_fields keys pushed the report past the cutoff and the field-count
    truncation marker took its place, even though nothing about the
    result's own size called for it."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(
        max_chars=80,
        max_fields=2,
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=80),
    )
    filtered = wrapper._filter_result({"a": "x" * 200, "b": "small"})

    assert SPILL_RESERVED_RESULT_KEY in filtered
    assert spill_record_shape_is_valid(filtered[SPILL_RESERVED_RESULT_KEY][0])
    assert filtered["a"] != "x" * 200  # the oversized value was spilled
    assert filtered["b"] == "small"
    assert not any(str(key).endswith("more keys") for key in filtered)


@pytest.mark.asyncio
async def test_the_async_path_also_keeps_the_spill_report_intact(tmp_path):
    spill_dir = tmp_path / "output" / "tool-results"

    async def run_json_async(args):
        return {"content": "x" * 200}

    target = SimpleNamespace(name="acme", run_json_async=run_json_async)
    wrapper = OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=50,
        max_fields=1000,
        max_recursion=20,
        spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=50),
    )
    filtered = await wrapper.run_json_async({})

    record = filtered[SPILL_RESERVED_RESULT_KEY][0]
    assert spill_record_shape_is_valid(record)
    written = spill_dir / record["relative_path"].split("/")[-1]
    assert written.exists()

    ctx = ExecutionContext()
    ctx.attach_workspace("ws-1", str(tmp_path))
    tool = ctx.add_tool_result("acme", filtered)
    assert len(ctx.spilled_results) == 1
    assert record["relative_path"] in tool.content


def test_a_forged_report_key_is_stripped_with_a_target_when_nothing_spills(tmp_path):
    """With a real report bypassing the output filter, stripping a
    tool-supplied report key in _spill_only is the only guard against a
    forged key. It strips even when a spill target is configured and the
    result is too small to spill."""
    spill_dir = tmp_path / "output" / "tool-results"
    wrapper = _wrapper(spill_target=SpillTarget(spill_dir=str(spill_dir), max_chars=50))
    forged = [{"relative_path": "tool-results/evil.json"}]
    result = {"output": "small value", SPILL_RESERVED_RESULT_KEY: forged}
    filtered = wrapper._filter_result(result)

    assert SPILL_RESERVED_RESULT_KEY not in filtered
    assert not spill_dir.exists()
