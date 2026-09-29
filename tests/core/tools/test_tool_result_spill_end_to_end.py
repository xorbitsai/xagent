"""A tool set built by ToolFactory stores an oversized result end to end.

Each test runs the chain a deployed tool call runs: the tool set comes from
ToolFactory.create_all_tools with a real TaskWorkspace, an oversized result
passes through that set's own OutputFilteredToolWrapper, the execution
context registers it and renders the observation, and the stored file is
read back through the same set's read_tool_result tool. Compacting that
context through PatternRuntime still tells the model which file it can read.
"""

import asyncio
import json
import logging
import re
import threading
from pathlib import Path

import pytest

from xagent.core.agent.context import ExecutionContext
from xagent.core.agent.runtime import PatternRuntime
from xagent.core.tools import tool_result_spill
from xagent.core.tools.adapters.vibe.config import ToolConfig
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.function import FunctionTool
from xagent.core.tools.adapters.vibe.output_filter import DEFAULT_TRUNCATION_MESSAGE
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
    extract_bound_method_target,
)
from xagent.core.tools.tool_result_spill import (
    SPILL_MAX_FILES_PER_RUN,
    SPILL_PLACEHOLDER_TEXT,
    SPILL_READ_MAX_CHARS,
    SPILL_READ_TOOL_NAME,
    SPILL_RESERVED_RESULT_KEY,
    render_spill_notice,
    spill_dir_for_workspace,
)
from xagent.core.workspace import TaskWorkspace

SPILL_LOGGER = "xagent.core.tools.tool_result_spill"
STORED_PATH_RE = re.compile(r"tool-results/[A-Za-z0-9_-]+\.(?:json|txt)")

# About 62,000 characters as a JSON array: over the default 51,200-character
# limit as a whole, while every one of its 700 items is small.
CLIENTS = [
    {"id": index, "name": f"client-{index:05d}", "note": "n" * 40}
    for index in range(700)
]


def _returning(name, produce_result):
    """An asynchronous tool that returns ``produce_result()`` on every call."""

    async def produce() -> dict:
        """Return this tool's fixed result."""
        return produce_result()

    return FunctionTool(produce, name=name, description=f"Returns {name} data.")


def _only_tool_named(tools, name):
    matches = [tool for tool in tools if tool.name == name]
    assert len(matches) == 1, [tool.name for tool in tools]
    return matches[0]


def _bound_workspace(tools):
    """The workspace object the tool set's read_file is bound to."""
    target = extract_bound_method_target(_only_tool_named(tools, "read_file")._target)
    assert target is not None
    instance, _ = target
    assert isinstance(instance.workspace, TaskWorkspace)
    return instance.workspace


async def _tool_set(
    tmp_path, *extra_tools, task_id="spill-e2e", config_cls=ToolConfig, **config
):
    return await ToolFactory.create_all_tools(
        config_cls(
            {"workspace": {"task_id": task_id, "base_dir": str(tmp_path)}, **config}
        ),
        additional_tools=list(extra_tools),
    )


def _context_for(workspace):
    ctx = ExecutionContext()
    ctx.attach_workspace(workspace.id, str(workspace.workspace_dir))
    return ctx


def _stored_files(spill_dir):
    directory = Path(spill_dir)
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir())


async def _store_the_client_list(tmp_path):
    tools = await _tool_set(
        tmp_path, _returning("acme_clients", lambda: {"clients": CLIENTS})
    )
    workspace = _bound_workspace(tools)
    ctx = _context_for(workspace)
    result = await _only_tool_named(tools, "acme_clients").run_json_async({})
    message = ctx.add_tool_result("acme_clients", result)
    return tools, workspace, ctx, message


@pytest.mark.asyncio
async def test_an_oversized_list_result_is_stored_and_the_model_is_told_where(
    tmp_path,
):
    source_chars = len(json.dumps(CLIENTS, ensure_ascii=False))
    assert source_chars > 51_200

    tools, workspace, ctx, message = await _store_the_client_list(tmp_path)

    spill_dir = spill_dir_for_workspace(workspace.workspace_dir)
    stored = _stored_files(spill_dir)
    assert len(stored) == 1
    assert (Path(spill_dir) / stored[0]).is_file()
    assert [record["relative_path"] for record in ctx.spilled_results] == [
        f"tool-results/{stored[0]}"
    ]
    body, _, notice = message.content.partition("\n")
    assert SPILL_PLACEHOLDER_TEXT in body
    assert "tool-results/" not in body
    assert f"tool-results/{stored[0]}" in notice
    assert f"a JSON array of {len(CLIENTS)} items" in notice
    assert f"{source_chars} source characters" in notice
    # The whole point: about 62,000 characters of context become a
    # placeholder and a notice.
    assert len(message.content) < 2_000


@pytest.mark.asyncio
async def test_the_model_can_read_the_stored_result_back_by_the_listed_path(
    tmp_path,
):
    tools, _, _, message = await _store_the_client_list(tmp_path)
    listed_paths = STORED_PATH_RE.findall(message.content)
    assert len(listed_paths) == 1
    reader = _only_tool_named(tools, SPILL_READ_TOOL_NAME)

    for start, end in ((1, 20), (350, 360), (690, 700)):
        reply = await reader.run_json_async(
            {"path": listed_paths[0], "start": start, "end": end}
        )
        assert reply["relative_path"] == listed_paths[0]
        assert json.loads(reply["output"]) == CLIENTS[start - 1 : end]


@pytest.mark.asyncio
async def test_listing_without_a_path_reports_the_stored_file(tmp_path):
    tools, workspace, ctx, _ = await _store_the_client_list(tmp_path)
    spill_dir = Path(spill_dir_for_workspace(workspace.workspace_dir))
    (relative_path,) = [record["relative_path"] for record in ctx.spilled_results]

    listing = await _only_tool_named(tools, SPILL_READ_TOOL_NAME).run_json_async({})

    assert listing["count"] == len(_stored_files(spill_dir)) == 1
    assert (listing["start"], listing["end"], listing["omitted"]) == (1, 1, 0)
    assert listing["stored_results"] == [
        {
            "relative_path": relative_path,
            "bytes": (spill_dir / relative_path.split("/")[-1]).stat().st_size,
        }
    ]


@pytest.mark.asyncio
async def test_an_oversized_single_item_is_read_back_with_offset(tmp_path):
    rows = ["a" * 30_000, "b" * 30_000]
    tools = await _tool_set(tmp_path, _returning("acme_rows", lambda: {"rows": rows}))
    ctx = _context_for(_bound_workspace(tools))
    result = await _only_tool_named(tools, "acme_rows").run_json_async({})
    ctx.add_tool_result("acme_rows", result)
    (relative_path,) = [record["relative_path"] for record in ctx.spilled_results]
    reader = _only_tool_named(tools, SPILL_READ_TOOL_NAME)
    expected = json.dumps(rows[:1], ensure_ascii=False)

    first = await reader.run_json_async({"path": relative_path, "start": 1, "end": 1})
    assert "output" not in first
    assert first["content_truncated"] is True
    assert first["offset"] == 0
    assert first["original_chars"] == len(expected)
    assert len(first["content_preview"]) == SPILL_READ_MAX_CHARS

    pieces = [first["content_preview"]]
    reply = first
    while reply["content_truncated"]:
        reply = await reader.run_json_async(
            {
                "path": relative_path,
                "start": 1,
                "end": 1,
                "offset": reply["offset"] + len(reply["content_preview"]),
            }
        )
        pieces.append(reply["content_preview"])
    assert "instruction" not in reply
    assert "".join(pieces) == expected
    assert json.loads("".join(pieces)) == rows[:1]


@pytest.mark.asyncio
async def test_a_result_at_exactly_the_threshold_is_not_stored(tmp_path):
    """Both spill tiers store only what serializes to more than max_chars.
    The output filter's own <= applies to one string at a time, while the
    whole-result tier compares the serialized result as a whole, so the two
    limits are not complements: a single string of exactly max_chars
    characters passes the filter but makes its result too long. This cell
    pins the case that is left alone: a result whose whole serialization is
    exactly 51,200 characters has no child over the limit and is not over
    it itself, so nothing is written and the result is exactly what the
    output filter alone makes of it."""
    overhead = len(json.dumps({"output": ""}, ensure_ascii=False))
    result = {"output": "x" * (51_200 - overhead)}
    assert len(json.dumps(result, ensure_ascii=False)) == 51_200
    tool = _returning("acme_exact", lambda: dict(result))
    tools = await _tool_set(tmp_path, tool)
    wrapper = _only_tool_named(tools, "acme_exact")
    assert wrapper._spill_target is not None
    assert wrapper._spill_target.max_chars == 51_200
    plain = OutputFilteredToolWrapper(
        target_tool=tool,
        max_chars=51_200,
        max_fields=wrapper._filter.max_fields,
        max_recursion=wrapper._filter.max_recursion,
    )

    reply = await wrapper.run_json_async({})

    assert SPILL_RESERVED_RESULT_KEY not in reply
    assert _stored_files(wrapper._spill_target.spill_dir) == []
    assert json.dumps(reply) == json.dumps(await plain.run_json_async({}))
    assert reply == result


async def _assert_left_to_truncation(tmp_path, caplog, bound_tool_name, **config):
    """Build a tool set whose policy removes read_tool_result and check that
    an oversized value reaches the model exactly as it does without the
    spill layer: no target anywhere, no file, and the same observation."""
    big = _returning("acme_big", lambda: {"output": "x" * 60_000})
    with caplog.at_level(
        logging.INFO, logger="xagent.core.tools.adapters.vibe.factory"
    ):
        tools = await _tool_set(tmp_path, big, **config)
    assert not any(tool.name == SPILL_READ_TOOL_NAME for tool in tools)
    assert all(tool._spill_target is None for tool in tools)
    assert any(
        "spill disabled" in record.getMessage()
        and "no read_tool_result in the tool set" in record.getMessage()
        for record in caplog.records
    )
    bound = extract_bound_method_target(
        _only_tool_named(tools, bound_tool_name)._target
    )
    assert bound is not None
    workspace = bound[0].workspace
    assert isinstance(workspace, TaskWorkspace)
    wrapper = _only_tool_named(tools, "acme_big")
    plain = OutputFilteredToolWrapper(
        target_tool=big,
        max_chars=wrapper._filter.max_chars,
        max_fields=wrapper._filter.max_fields,
        max_recursion=wrapper._filter.max_recursion,
    )

    reply = await wrapper.run_json_async({})
    observation = _context_for(workspace).add_tool_result("acme_big", reply)
    today = _context_for(workspace).add_tool_result(
        "acme_big", await plain.run_json_async({})
    )

    assert SPILL_RESERVED_RESULT_KEY not in reply
    assert DEFAULT_TRUNCATION_MESSAGE in reply["output"]
    assert _stored_files(spill_dir_for_workspace(workspace.workspace_dir)) == []
    assert observation.content == today.content


class _ReaderDisabledConfig(ToolConfig):
    """A per-user disabled-tools table that turns off only the reader."""

    def get_user_tool_overrides(self):
        return {SPILL_READ_TOOL_NAME: {"enabled": False}}


# Three tool policies that remove read_tool_result while a tool bound to the
# task workspace stays in the set:
# - write_file-only: the policy leaves write_file, still bound to the task
#   workspace, but none of the reading tools.
# - legacy-allowlist: a concrete allowed_tools list written before
#   read_tool_result existed keeps read_file and drops the reader.
# - per-user-disabled: a user disables read_tool_result on its own from the
#   tool list, which keeps read_file.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bound_tool_name", "config"),
    [
        ("write_file", {"allowed_tools": ["write_file", "acme_big"]}),
        ("read_file", {"allowed_tools": ["read_file", "acme_big"]}),
        ("read_file", {"config_cls": _ReaderDisabledConfig}),
    ],
    ids=["write_file-only", "legacy-allowlist", "per-user-disabled"],
)
async def test_a_tool_set_without_a_workspace_bound_read_tool_result_stores_nothing(
    tmp_path, caplog, bound_tool_name, config
):
    """A stored value could not be read back, so nothing in the set gets a
    spill target, nothing is stored, and an oversized value is truncated
    exactly as before."""
    await _assert_left_to_truncation(tmp_path, caplog, bound_tool_name, **config)


@pytest.mark.asyncio
async def test_the_run_budget_caps_files_across_several_tool_calls(tmp_path, caplog):
    """The 64-file budget belongs to the tool set, not to one run or one
    tool: every wrapper the factory built in one construction draws on the
    same budget, for as long as that set of tools is in use. Two tools
    alternate here and their files add up to one cap."""
    calls = {"count": 0}

    def distinct_result():
        calls["count"] += 1
        return {"output": f"{calls['count']:06d}" + "x" * 60_000}

    tools = await _tool_set(
        tmp_path,
        _returning("acme_one", distinct_result),
        _returning("acme_two", distinct_result),
    )
    spill_dir = spill_dir_for_workspace(_bound_workspace(tools).workspace_dir)
    alternating = [
        _only_tool_named(tools, "acme_one"),
        _only_tool_named(tools, "acme_two"),
    ]

    for index in range(SPILL_MAX_FILES_PER_RUN):
        reply = await alternating[index % 2].run_json_async({})
        assert SPILL_RESERVED_RESULT_KEY in reply
    assert len(_stored_files(spill_dir)) == SPILL_MAX_FILES_PER_RUN

    with caplog.at_level(logging.WARNING, logger=SPILL_LOGGER):
        over_cap = [await tool.run_json_async({}) for tool in alternating]

    for reply in over_cap:
        assert SPILL_RESERVED_RESULT_KEY not in reply
        assert DEFAULT_TRUNCATION_MESSAGE in reply["output"]
    assert len(_stored_files(spill_dir)) == SPILL_MAX_FILES_PER_RUN
    messages = [record.getMessage() for record in caplog.records]
    budget_messages = [
        message
        for message in messages
        if message.startswith(f"Spill run budget of {SPILL_MAX_FILES_PER_RUN} files")
    ]
    # One warning per over-cap result, from the exhausted-budget branch;
    # none from the branch where a build that was admitted produced no record.
    assert len(budget_messages) == len(over_cap)
    assert not any("Failed to write spill file" in message for message in messages)


@pytest.mark.asyncio
async def test_cancelling_during_the_write_consumes_a_slot_and_does_not_crash(
    tmp_path, monkeypatch
):
    """Cancelling a tool call cancels only the wait for the worker thread
    that stores its result; the thread runs to completion. The file is
    written, the slot it reserved stays taken, and no record reaches the
    execution context. The tool set keeps working: a later oversized result
    is stored and registered as usual, and listing the stored results shows
    both files, because the listing reads the directory, not the registry.
    """
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_write = tool_result_spill._write_spill_file

    def slow_write(*args, **kwargs):
        started.set()
        try:
            release.wait(5)
            return original_write(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(tool_result_spill, "_write_spill_file", slow_write)
    tools = await _tool_set(
        tmp_path,
        _returning("acme_first", lambda: {"output": "f" * 60_000}),
        _returning("acme_second", lambda: {"output": "s" * 60_000}),
    )
    workspace = _bound_workspace(tools)
    ctx = _context_for(workspace)
    spill_dir = spill_dir_for_workspace(workspace.workspace_dir)
    first = _only_tool_named(tools, "acme_first")
    budget = first._spill_run_budget

    call = asyncio.create_task(first.run_json_async({}))
    assert await asyncio.to_thread(started.wait, 5)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert budget.files_written == 1
    release.set()
    assert await asyncio.to_thread(finished.wait, 5)

    orphan = _stored_files(spill_dir)
    assert len(orphan) == 1
    assert budget.files_written == 1
    assert ctx.spilled_results == ()

    reply = await _only_tool_named(tools, "acme_second").run_json_async({})
    ctx.add_tool_result("acme_second", reply)

    assert budget.files_written == 2
    (registered,) = [record["relative_path"] for record in ctx.spilled_results]
    assert registered != f"tool-results/{orphan[0]}"
    listing = await _only_tool_named(tools, SPILL_READ_TOOL_NAME).run_json_async({})
    assert listing["count"] == 2
    assert {entry["relative_path"] for entry in listing["stored_results"]} == {
        registered,
        f"tool-results/{orphan[0]}",
    }


@pytest.mark.asyncio
async def test_default_limits_keep_the_report_on_a_wide_whole_root_spill(tmp_path):
    """With the default limits (51,200 characters, 1,000 fields) a result of
    1,001 short top-level entries is stored whole. The in-context copy
    still meets the field limit, so it ends in "... and 1 more keys", but
    the engine's report is carried past that limit and registered."""
    wide = {f"k{index:04d}": "v" * 60 for index in range(1_001)}
    assert len(json.dumps(wide, ensure_ascii=False)) > 51_200
    tools = await _tool_set(tmp_path, _returning("acme_wide", lambda: dict(wide)))
    wrapper = _only_tool_named(tools, "acme_wide")
    assert (wrapper._filter.max_chars, wrapper._filter.max_fields) == (51_200, 1_000)
    ctx = _context_for(_bound_workspace(tools))

    reply = await wrapper.run_json_async({})
    message = ctx.add_tool_result("acme_wide", reply)

    assert len(reply[SPILL_RESERVED_RESULT_KEY]) == 1
    assert "... and 1 more keys" in reply
    (relative_path,) = [record["relative_path"] for record in ctx.spilled_results]
    assert relative_path in message.content
    first_two = await _only_tool_named(tools, SPILL_READ_TOOL_NAME).run_json_async(
        {"path": relative_path, "start": 1, "end": 2}
    )
    assert json.loads(first_two["output"]) == {"k0000": "v" * 60, "k0001": "v" * 60}


@pytest.mark.asyncio
async def test_a_stored_file_is_not_listed_as_a_user_deliverable_and_refuses_writes(
    tmp_path,
):
    tools, workspace, ctx, _ = await _store_the_client_list(tmp_path)
    (relative_path,) = [record["relative_path"] for record in ctx.spilled_results]
    stored = (
        Path(spill_dir_for_workspace(workspace.workspace_dir))
        / (relative_path.split("/")[-1])
    )
    assert stored.is_file()

    output_paths = {entry["file_path"] for entry in workspace.get_output_files()}
    all_paths = {
        entry["file_path"]
        for entries in workspace.get_all_files().values()
        for entry in entries
    }
    assert not any("tool-results" in path for path in output_paths | all_paths)

    with pytest.raises(ValueError):
        await _only_tool_named(tools, "write_file").run_json_async(
            {"file_path": "output/tool-results/x.json", "content": "[1]"}
        )
    assert not (stored.parent / "x.json").exists()


class _SummaryLLM:
    """A compaction model that always writes the same short summary."""

    context_window = 64_000

    async def chat(self, **_):
        return {"content": "The client list was fetched."}


@pytest.mark.asyncio
@pytest.mark.parametrize("summarize", [True, False], ids=["summary", "drop_oldest"])
async def test_compaction_through_the_runtime_lists_the_stored_file(
    tmp_path, summarize
):
    """Compaction removes the observation that named the stored file; the
    messages the model is sent next still name it, in the list compaction
    builds from the registry, and that path reads the file back."""
    tools = await _tool_set(
        tmp_path, _returning("acme_clients", lambda: {"clients": CLIENTS})
    )
    ctx = _context_for(_bound_workspace(tools))
    ctx.compact_config.threshold = 1
    # One message of window: the drop-oldest path keeps only the latest user
    # message, so the observation is gone on both paths.
    ctx.compact_config.max_messages = 1
    ctx.add_user_message("List our clients")
    ctx.add_assistant_message(
        "",
        tool_calls=[
            {"id": "call-1", "type": "function", "function": {"name": "acme_clients"}}
        ],
    )
    result = await _only_tool_named(tools, "acme_clients").run_json_async({})
    observation = ctx.add_tool_result("acme_clients", result, "call-1")
    (relative_path,) = [record["relative_path"] for record in ctx.spilled_results]
    assert relative_path in observation.content
    ctx.add_user_message("Which of them are new?")

    compacted = await PatternRuntime().compact_context_if_needed(
        context=ctx, llm=_SummaryLLM() if summarize else None
    )

    assert compacted.compacted
    assert compacted.strategy == ("llm_summary" if summarize else "truncate")
    assert not any(message is observation for message in ctx.messages)
    sent = ctx.get_messages_for_llm()
    naming = [message for message in sent if relative_path in str(message["content"])]
    assert naming == [
        {
            "role": "user",
            "content": render_spill_notice(
                list(ctx.spilled_results), style="compaction"
            ),
        }
    ]
    assert (
        f"- {relative_path}: a JSON array of {len(CLIENTS)} items"
        in (naming[0]["content"])
    )
    assert sent[-1] == {"role": "user", "content": "Which of them are new?"}
    (listed_path,) = STORED_PATH_RE.findall(naming[0]["content"])
    reply = await _only_tool_named(tools, SPILL_READ_TOOL_NAME).run_json_async(
        {"path": listed_path, "start": 1, "end": 2}
    )
    assert json.loads(reply["output"]) == CLIENTS[:2]
