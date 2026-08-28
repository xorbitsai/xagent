import asyncio
import json
import os
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.core.tools.adapters.vibe.agent_tool as mod
import xagent.core.tools.adapters.vibe.db_session as db_session_module
from xagent.core.agent.result import tool_result_succeeded
from xagent.core.tools.adapters.vibe.agent_tool import (
    _CHILD_NO_ANSWER_MESSAGE,
    AgentTool,
)
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User
from xagent.web.services.llm_utils import UserAwareModelStorage


class _Stop(Exception):
    """Halt the run before the sub-agent executes."""


class _DelegatedQuery:
    def __init__(self, agent):
        self._agent = agent

    def filter(self, *_args):
        return self

    def first(self):
        return self._agent


class _DelegatedSession:
    def __init__(self, agent):
        self._agent = agent

    def query(self, *_args):
        return _DelegatedQuery(self._agent)

    def commit(self):
        return None

    def close(self):
        return None


class _FailingCloseConfig:
    def close(self):
        raise ValueError("cleanup sentinel")


class _SucceedingCloseConfig:
    def close(self):
        return None


def _delegated_agent_tool() -> AgentTool:
    return AgentTool(
        agent_id=1,
        agent_name="Delegated",
        agent_description="d",
        session_factory=lambda: _DelegatedSession(
            SimpleNamespace(
                id=1,
                name="Delegated",
                instructions=None,
                knowledge_bases=None,
                skills=None,
                tool_categories=[],
                models={"general": 1},
                execution_mode=None,
            )
        ),
        user_id=1,
        tool_name="delegated",
        tool_description="d",
    )


def _patch_delegated_runtime(
    monkeypatch, execute_task, *, close_config=_FailingCloseConfig
):
    import xagent.core.agent.service as service_module
    import xagent.core.tools.adapters.vibe.agent_model_resolution as resolution

    class FakeAgentService:
        workspace = None

        def __init__(self, **_kwargs):
            return None

        async def execute_task(self, **_kwargs):
            return await execute_task()

    monkeypatch.setattr(mod, "WebToolConfig", lambda **_kwargs: close_config())
    monkeypatch.setattr(service_module, "AgentService", FakeAgentService)
    monkeypatch.setattr(
        resolution,
        "resolve_agent_model_llms",
        lambda *_args: (object(), None, None, None),
    )


@pytest.mark.asyncio
async def test_agent_tool_maps_successful_body_cleanup_failure_to_boundary_error(
    monkeypatch,
):
    async def execute_task():
        return {"output": "completed"}

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["response"].endswith("Tool runtime cleanup could not be completed.")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_agent_tool_preserves_body_failure_when_cleanup_also_fails(
    monkeypatch, caplog
):
    primary = RuntimeError("body sentinel")

    async def execute_task():
        raise primary

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["response"].endswith("body sentinel")
    assert result["success"] is False
    assert "Failed to close delegated agent tool runtime after execution" in caplog.text


@pytest.mark.asyncio
async def test_agent_tool_preserves_cancelled_error_identity_when_cleanup_fails(
    monkeypatch,
):
    primary = asyncio.CancelledError("cancelled sentinel")

    async def execute_task():
        raise primary

    _patch_delegated_runtime(monkeypatch, execute_task)
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    with pytest.raises(asyncio.CancelledError) as caught:
        await tool.run_json_async({"task": "run"})

    assert caught.value is primary


@pytest.mark.asyncio
async def test_agent_tool_child_waiting_returns_classified_nested_failure(
    monkeypatch,
):
    """A child that paused for user input must not surface as a success.

    Even when the child left a non-empty partial ``output`` behind, the
    delegated call cannot forward the interactive prompt one level up, so the
    parent must see a classified failure rather than a half-finished answer.
    """

    async def execute_task():
        return {"status": "waiting_for_user", "output": "Here is what I have so far"}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["success"] is False
    assert result["is_error"] is True
    assert result["status"] == "error"
    assert result["failure_code"] == "unsupported_nested_interaction"
    assert isinstance(result["error"], str) and result["error"]
    assert isinstance(result["output"], str) and result["output"]
    assert isinstance(result["response"], str) and result["response"]
    assert list(result.keys()) == [
        "success",
        "is_error",
        "status",
        "failure_code",
        "error",
        "output",
        "response",
    ]
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "child_result",
    [
        {"success": True, "status": "failed"},
        {"success": False, "status": "completed"},
    ],
)
async def test_agent_tool_or_precedence_disagreement_classifies_as_generic_failure(
    monkeypatch, child_result
):
    """``success`` and ``status`` disagreeing must still fail closed.

    Whichever field says "not done" wins; neither can veto the other back to
    a happy-path result.
    """

    async def execute_task():
        return dict(child_result)

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["success"] is False
    assert result["is_error"] is True
    assert result["status"] == "error"
    assert "failure_code" not in result
    # Exact key list: the classified dict must be freshly constructed, never a
    # mutated copy of the child result carrying live objects into trace data.
    assert list(result.keys()) == [
        "success",
        "is_error",
        "status",
        "error",
        "output",
        "response",
    ]
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
async def test_agent_tool_interrupt_uses_error_text_over_placeholder_output(
    monkeypatch,
):
    """The interrupted child's real diagnostic must win over its placeholder text.

    The execution adapter replaces ``output`` with a user-facing placeholder
    message for interrupted runs, but keeps the real diagnostic in ``error``.
    The classified failure must surface the diagnostic, not the placeholder.
    """

    async def execute_task():
        return {
            "status": "interrupted",
            "success": False,
            "error": "child tool call was cancelled mid-flight",
            "output": "The assistant was interrupted before finishing.",
        }

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["success"] is False
    assert "failure_code" not in result
    assert result["error"] == "child tool call was cancelled mid-flight"
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
async def test_agent_tool_falls_back_to_output_when_error_is_empty(
    monkeypatch,
):
    """The generic classified failure falls back to output when error is empty.

    Mirrors ``test_agent_tool_interrupt_uses_error_text_over_placeholder_output``
    above: that test pins ``error`` winning over ``output`` when both are
    non-empty, this one pins the fallback direction — an empty ``error``
    must not surface as the message, so the classifier reads ``output``
    instead, and every message-shaped field in the envelope carries it.
    """

    async def execute_task():
        return {
            "status": "failed",
            "success": False,
            "error": "",
            "output": "the child's own diagnostic text",
        }

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["success"] is False
    assert "failure_code" not in result
    assert result["error"] == "the child's own diagnostic text"
    assert result["output"] == "the child's own diagnostic text"
    assert result["response"] == "the child's own diagnostic text"
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
async def test_agent_tool_normal_child_result_unchanged(monkeypatch):
    """A plain completed child result must keep its exact legacy shape.

    This also pins the "absent status/success is inert" rule: fakes that omit
    both fields (as this one does) must stay on the happy path. The
    missing-output branch is also gated on an explicit completed status, so
    it does not disturb this case either.
    """

    async def execute_task():
        return {"output": "all good"}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result == {"response": "all good"}


def test_classifier_leaves_statusless_results_untouched():
    """Absent status/success is inert even when output is missing or empty.

    The missing-output branch fires only on an explicit completed status;
    a result carrying neither status nor usable output stays unclassified
    rather than being reinterpreted as a failure.
    """

    assert mod._classify_delegated_child_failure({}) is None
    assert mod._classify_delegated_child_failure({"output": ""}) is None


def test_classify_delegated_child_failure_relays_the_childs_own_pause_message():
    """A delegated child that paused (e.g. UnavailableMCPTool naming a specific
    app needing reconnection) must still surface that diagnostic to the
    parent - the generic 'nested calls cannot forward prompts' framing alone
    throws away the only actionable part of the failure."""

    result = {
        "status": "waiting_for_user",
        "success": False,
        "message": "I need access to Gmail to continue.",
    }

    classified = mod._classify_delegated_child_failure(result)

    assert classified is not None
    assert classified["failure_code"] == "unsupported_nested_interaction"
    assert "I need access to Gmail to continue." in classified["error"]
    assert "cannot forward" in classified["error"]


def test_classify_delegated_child_failure_reads_raw_output_over_backfill():
    """The classifier reads the child's own raw answer, not the backfilled one.

    A completed result whose normalized ``output`` was backfilled from an
    earlier assistant message must still classify as missing when the raw
    ``agent_result`` carries no answer of its own.
    """

    result = {
        "status": "completed",
        "success": True,
        "output": "stale preamble",
        "agent_result": {
            "status": "completed",
            "success": True,
            "output": "",
            "response": "",
        },
    }

    classified = mod._classify_delegated_child_failure(result)

    assert classified is not None
    assert classified["failure_code"] == "missing_delegated_output"


def test_classify_delegated_child_failure_passes_real_raw_output():
    """A real raw output passes even when the normalized layer agrees with it."""

    result = {
        "status": "completed",
        "success": True,
        "output": "real answer",
        "agent_result": {
            "status": "completed",
            "success": True,
            "output": "real answer",
        },
    }

    assert mod._classify_delegated_child_failure(result) is None


def test_classify_delegated_child_failure_falls_back_without_agent_result():
    """Absent ``agent_result`` degrades to the normalized-output check, both ways."""

    assert (
        mod._classify_delegated_child_failure(
            {"status": "completed", "success": True, "output": "all good"}
        )
        is None
    )
    classified = mod._classify_delegated_child_failure(
        {"status": "completed", "success": True, "output": ""}
    )
    assert classified is not None
    assert classified["failure_code"] == "missing_delegated_output"


def test_classify_delegated_child_failure_passes_raw_placeholder_text():
    """A raw answer equal to a placeholder string is still a real answer.

    The placeholder sentinels only apply to the normalized fallback surface;
    raw pre-backfill content from ``agent_result`` is judged on emptiness
    alone, so a child whose task was to reply with that exact text must not
    be classified as missing.
    """

    result = {
        "status": "completed",
        "success": True,
        "output": "No output provided",
        "agent_result": {
            "status": "completed",
            "success": True,
            "output": "No output provided",
        },
    }

    assert mod._classify_delegated_child_failure(result) is None


def test_agent_tool_result_declares_every_classified_failure_key():
    """The declared return contract covers the failure envelope.

    ``AgentTool.return_type()`` is ``AgentToolResult``; a consumer that
    model-validates and re-dumps a classified failure must not be able to
    silently strip the classification keys. Asserting the actual round trip
    (not just field declaration) also guards against a future ``exclude``
    or alias on a field, which would strip keys while still passing a
    declared-fields subset check.
    """

    plain = mod._classified_failure("boom")
    with_code = mod._classified_failure("boom", failure_code="missing_delegated_output")

    for envelope in (plain, with_code):
        round_tripped = mod.AgentToolResult.model_validate(envelope).model_dump(
            exclude_none=True
        )
        assert round_tripped == envelope


@pytest.mark.asyncio
async def test_agent_tool_catchall_does_not_rewrap_classified_failure(monkeypatch):
    """The classified failure must return before the catch-all can touch it.

    Using a close config that does not raise proves this end to end: the
    classified dict built ahead of the ``except Exception`` block must reach
    the caller unrewrapped, not folded into the generic
    ``Error executing agent ...`` message.
    """

    async def execute_task():
        return {"status": "waiting_for_user", "output": "partial"}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "unsupported_nested_interaction"
    assert not str(result["response"]).startswith("Error executing agent")
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "child_result",
    [
        {"status": "waiting_for_user", "output": "partial"},
        {"success": False, "status": "failed", "error": "child blew up"},
    ],
)
async def test_agent_tool_classified_failure_still_traces_delegation_error(
    monkeypatch, child_result
):
    """Both classified paths must emit the delegation terminal event.

    A delegated run's public outcome is derived from the
    ``workforce_delegation_start``/``_end``/``_error`` trace events, so a
    classified failure that returned without tracing one would leave the
    child showing as still running. Emitting the terminal event is part of
    the classified-failure contract, not an optional extra.
    """

    traced = []

    async def execute_task():
        return dict(child_result)

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    async def _record(status, **kwargs):
        traced.append((status, kwargs))

    monkeypatch.setattr(tool, "_trace_delegation", _record)

    result = await tool.run_json_async({"task": "run"})

    assert tool_result_succeeded(result) is False
    assert [status for status, _ in traced] == ["start", "error"]
    assert traced[-1][1]["error"] == result["error"]
    assert traced[-1][1]["execution_task_id"] is not None


@pytest.mark.asyncio
async def test_agent_tool_classified_failure_skips_file_registration(monkeypatch):
    """The classified branch performs no file bookkeeping.

    It cannot durably attach a failed child's artifacts to the parent task,
    because it never opens a session or registers anything on this path. A
    raising second session proves the branch never asks for one: if it did,
    this test would blow up instead of asserting a clean single-session
    count.
    """

    async def execute_task():
        return {
            "status": "waiting_for_user",
            "output": "partial",
            "file_outputs": [{"filename": "a.txt", "file_path": "a.txt"}],
        }

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    traced = []

    async def _record(status, **kwargs):
        traced.append((status, kwargs))

    monkeypatch.setattr(tool, "_trace_delegation", _record)

    class _RaisingSession:
        def query(self, *_args):
            raise AssertionError("file bookkeeping must not query the DB here")

        def commit(self):
            raise RuntimeError("db down")

        def close(self):
            return None

    real_scope = db_session_module.tool_session_scope
    calls = {"n": 0}

    @contextmanager
    def _counting_scope(factory):
        calls["n"] += 1
        if calls["n"] == 1:
            with real_scope(factory) as db:
                yield db
        else:
            db = _RaisingSession()
            try:
                yield db
            finally:
                db.close()

    monkeypatch.setattr(db_session_module, "tool_session_scope", _counting_scope)

    result = await tool.run_json_async({"task": "run"})

    assert calls["n"] == 1
    assert result["failure_code"] == "unsupported_nested_interaction"
    assert result["status"] == "error"
    assert "file_outputs" not in result
    assert [status for status, _ in traced] == ["start", "error"]
    assert tool_result_succeeded(result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["  waiting_for_user ", "WAITING_FOR_USER"])
async def test_agent_tool_normalizes_waiting_status_variants(monkeypatch, status):
    """Whitespace/case variants of the wait status must still classify.

    The classifier now runs on the shared ``tool_result_waits_for_user``
    predicate instead of an exact string compare, so a child that reports
    its status with incidental whitespace or casing still fails closed as
    the unsupported-nested-interaction case rather than the generic one.
    """

    async def execute_task():
        return {"status": status, "output": "partial"}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "unsupported_nested_interaction"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "child_output",
    ["", "   ", "No output provided", "No response generated"],
)
async def test_agent_tool_completed_child_without_usable_output_fails_closed(
    monkeypatch, child_output
):
    """A completed child that answered nothing must not surface as a success.

    Covers both a genuinely empty/whitespace answer and the two placeholder
    strings the execution layers substitute when a run produced no text of
    its own — a child that returns one of those completed without
    answering, and the parent must not launder that into a plain response.
    """

    async def execute_task():
        return {"status": "completed", "success": True, "output": child_output}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "missing_delegated_output"
    assert result["status"] == "error"
    assert result["success"] is False
    assert tool_result_succeeded(result) is False
    assert list(result.keys()) == [
        "success",
        "is_error",
        "status",
        "failure_code",
        "error",
        "output",
        "response",
    ]


@pytest.mark.asyncio
async def test_agent_tool_completed_child_with_output_key_absent_fails_closed(
    monkeypatch,
):
    """A completed child that omits ``output`` entirely must also fail closed.

    Without this branch, an absent ``output`` key would fall through to the
    ``"No response generated"`` default at the happy-path return and be
    laundered into a success.
    """

    async def execute_task():
        return {"status": "completed", "success": True}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "missing_delegated_output"
    assert result["status"] == "error"
    assert result["success"] is False
    assert tool_result_succeeded(result) is False
    assert list(result.keys()) == [
        "success",
        "is_error",
        "status",
        "failure_code",
        "error",
        "output",
        "response",
    ]


@pytest.mark.asyncio
async def test_agent_tool_realistic_completed_child_result_unchanged(monkeypatch):
    """A completed child with a real answer keeps the plain happy-path shape."""

    async def execute_task():
        return {"status": "completed", "success": True, "output": "all good"}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result == {"response": "all good"}


@pytest.mark.asyncio
async def test_agent_tool_waiting_child_without_output_stays_nested_interaction(
    monkeypatch,
):
    """The wait check must run before the missing-output check.

    A waiting child that also happens to carry no ``output`` must still
    classify as the unsupported-nested-interaction failure, not the
    missing-output one — the wait status is the more specific diagnosis.
    """

    async def execute_task():
        return {"status": "waiting_for_user", "success": True}

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "unsupported_nested_interaction"


class _MissingAgentSession:
    """A session whose agent lookup always misses."""

    def query(self, *_args):
        return _DelegatedQuery(None)

    def commit(self):
        return None

    def close(self):
        return None


@pytest.mark.asyncio
async def test_agent_tool_missing_agent_fails_closed(monkeypatch):
    """The agent-not-found preflight exit must be a classified failure.

    Before N2 this returned a bare ``{"response": ...}`` shape that
    ``tool_result_succeeded`` read as a success, so a missing agent could
    reach the ReAct loop disguised as a completed delegation.
    """

    traced_statuses: list[str] = []

    async def _trace_delegation(self, status, **_kwargs):
        traced_statuses.append(status)

    monkeypatch.setattr(AgentTool, "_trace_delegation", _trace_delegation)

    tool = AgentTool(
        agent_id=1,
        agent_name="Delegated",
        agent_description="d",
        session_factory=lambda: _MissingAgentSession(),
        user_id=1,
        tool_name="delegated",
        tool_description="d",
    )

    result = await tool.run_json_async({"task": "run"})

    assert tool_result_succeeded(result) is False
    assert result["status"] == "error"
    assert "failure_code" not in result
    assert result["response"] == "Error: Agent 1 not found"
    assert list(result.keys()) == [
        "success",
        "is_error",
        "status",
        "error",
        "output",
        "response",
    ]
    assert traced_statuses == ["error"]


@pytest.mark.asyncio
async def test_agent_tool_without_resolved_model_fails_closed(monkeypatch):
    """The no-valid-model preflight exit must be a classified failure.

    ``agent_models`` is falsy so model resolution is skipped entirely and
    ``default_llm`` stays ``None``, driving the same preflight exit a
    resolution failure would.
    """

    traced_statuses: list[str] = []

    async def _trace_delegation(self, status, **_kwargs):
        traced_statuses.append(status)

    monkeypatch.setattr(AgentTool, "_trace_delegation", _trace_delegation)

    tool = AgentTool(
        agent_id=1,
        agent_name="Delegated",
        agent_description="d",
        session_factory=lambda: _DelegatedSession(
            SimpleNamespace(
                id=1,
                name="Delegated",
                instructions=None,
                knowledge_bases=None,
                skills=None,
                tool_categories=[],
                models=None,
                execution_mode=None,
            )
        ),
        user_id=1,
        tool_name="delegated",
        tool_description="d",
    )

    result = await tool.run_json_async({"task": "run"})

    assert tool_result_succeeded(result) is False
    assert result["status"] == "error"
    assert "failure_code" not in result
    assert result["response"] == "Error: No valid model configured for agent Delegated"
    assert list(result.keys()) == [
        "success",
        "is_error",
        "status",
        "error",
        "output",
        "response",
    ]
    assert traced_statuses == ["start", "error"]


class _StubSingleCallLLM:
    """Minimal LLM stub returning one fixed tool call from ``chat()``.

    ``ask_user_question`` ends the ReAct loop immediately once handled, so
    those scenarios only need a single turn. An empty ``final_answer`` is
    rejected as a tool-protocol violation and re-requested once, so those
    scenarios take two turns and get this same fixed response both times.
    """

    model_name = "stub-model"

    def __init__(self, response: dict[str, object]) -> None:
        self._response = response
        self.calls = 0

    async def chat(self, **_kwargs: object) -> dict[str, object]:
        self.calls += 1
        return dict(self._response)


def _real_delegated_agent_tool(
    monkeypatch, tmp_path, llm, *, execution_mode: str | None = None
) -> AgentTool:
    """Build an ``AgentTool`` that runs a real ``AgentService``/``ReActPattern``.

    Unlike ``_patch_delegated_runtime`` (which fakes ``AgentService``
    entirely), this drives the real execution stack so the classifier is
    exercised against the real ``_normalize_result`` output shape, not a
    hand-built stand-in for it. Three seams are still faked to keep the test
    hermetic and fast: the child's tool config (no MCP/DB-backed tool
    building), model resolution (returns the stub LLM), and tool discovery
    (skips the Node/parser-dependent tool build that makes ``tests/core/tools``
    flaky in this environment).

    ``execution_mode`` defaults to ``None`` (mapped to the ``react`` pattern,
    ``max_iterations=200``) to match every existing caller. Passing
    ``"flash"`` selects the ``single_call`` pattern instead
    (``max_iterations=2``, ``execution_adapter.py:310-319``), letting a
    scenario that must run out the clock reach ``max_iterations`` in two
    turns instead of two hundred.
    """
    import xagent.core.tools.adapters.vibe.agent_model_resolution as resolution
    import xagent.core.tools.adapters.vibe.factory as factory_module

    monkeypatch.setattr(
        mod, "WebToolConfig", lambda **_kwargs: _SucceedingCloseConfig()
    )
    monkeypatch.setattr(
        resolution,
        "resolve_agent_model_llms",
        lambda *_args: (llm, None, None, None),
    )

    async def _no_tools(*_args, **_kwargs):
        return []

    monkeypatch.setattr(factory_module.ToolFactory, "create_all_tools", _no_tools)

    tool = AgentTool(
        agent_id=1,
        agent_name="Delegated",
        agent_description="d",
        session_factory=lambda: _DelegatedSession(
            SimpleNamespace(
                id=1,
                name="Delegated",
                instructions=None,
                knowledge_bases=None,
                skills=None,
                tool_categories=[],
                models={"general": 1},
                execution_mode=execution_mode,
            )
        ),
        user_id=1,
        tool_name="delegated",
        tool_description="d",
        workspace_base_dir=str(tmp_path),
    )
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)
    return tool


@pytest.mark.asyncio
async def test_agent_tool_real_pausing_child_fails_closed(monkeypatch, tmp_path):
    """A real ReAct child that asks the user a question fails closed end to end.

    The classifier must work against the actual shape
    ``AgentExecutionAdapter._normalize_result`` produces from a real pattern
    run, not just against hand-built dicts.
    """

    llm = _StubSingleCallLLM(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "ask_user_question",
                        "arguments": json.dumps(
                            {"message": "Which format do you want?"}
                        ),
                    },
                }
            ],
            "done": False,
        }
    )
    tool = _real_delegated_agent_tool(monkeypatch, tmp_path, llm)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "unsupported_nested_interaction"
    assert result["status"] == "error"
    assert result["success"] is False


@pytest.mark.asyncio
async def test_agent_tool_real_child_with_empty_final_answer_fails_closed(
    monkeypatch, tmp_path
):
    """A real ReAct child that answers with an empty ``final_answer`` fails closed.

    ReAct now rejects an empty ``final_answer`` as a tool-protocol violation and
    spends its one repair retry on it, so the child fails inside its own run
    instead of reaching the parent as a ``completed`` result. It still classifies
    as ``missing_delegated_output`` — the child never produced an answer either
    way — but via the no-answer-status branch rather than the completed-but-empty
    one, and the parent sees an actionable message instead of the runtime's
    "invalid tool protocol" diagnostic.

    ``test_agent_tool_completed_child_without_usable_output_fails_closed``
    covers the completed-but-empty branch, and its parametrized rows serve as
    the drift detector for ``NO_OUTPUT_PLACEHOLDER`` /
    ``NO_RESPONSE_PLACEHOLDER``.
    """

    llm = _StubSingleCallLLM(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps({"answer": ""}),
                    },
                }
            ],
            "done": False,
        }
    )
    tool = _real_delegated_agent_tool(monkeypatch, tmp_path, llm)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "missing_delegated_output"
    assert result["status"] == "error"
    assert result["success"] is False
    assert tool_result_succeeded(result) is False
    # The child re-requested an answer once before giving up, instead of
    # finalizing the empty one.
    assert llm.calls == 2
    # The parent gets an actionable message, not the child's internal
    # "invalid tool protocol" diagnostic.
    assert "tool protocol" not in result["output"]
    assert result["output"] == _CHILD_NO_ANSWER_MESSAGE


@pytest.mark.asyncio
async def test_agent_tool_real_child_with_stale_assistant_preamble_fails_closed(
    monkeypatch, tmp_path
):
    """A child with an empty final answer never surfaces a stale preamble.

    The single LLM turn carries both a reasoning preamble (assistant
    ``content``) and an empty ``final_answer``. The preamble is exactly what
    ``AgentExecutionAdapter`` would backfill as ``output`` via
    ``_latest_assistant_message`` when the pattern's own answer is falsy, so
    the invariant under test is that it must never reach the parent as the
    child's answer. ReAct now rejects the empty ``final_answer`` before the run
    can complete, so the failure is reported instead of the preamble.
    """

    llm = _StubSingleCallLLM(
        {
            "content": "Let me look into that for you.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps({"answer": ""}),
                    },
                }
            ],
            "done": False,
        }
    )
    tool = _real_delegated_agent_tool(monkeypatch, tmp_path, llm)

    result = await tool.run_json_async({"task": "run"})

    assert result["failure_code"] == "missing_delegated_output"
    assert result["status"] == "error"
    assert result["success"] is False
    assert tool_result_succeeded(result) is False
    # The invariant: the preamble must not be laundered into the child's answer.
    assert "Let me look into that for you." not in json.dumps(
        {key: value for key, value in result.items() if key != "agent_result"},
        default=str,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("marker", "expects_no_answer_classification"),
    [
        pytest.param(True, True, id="empty-answer"),
        pytest.param(False, False, id="other-protocol-violation"),
    ],
)
async def test_agent_tool_child_protocol_failure_keeps_its_diagnostic(
    monkeypatch, marker, expects_no_answer_classification
):
    """Only a genuinely unanswered child is classified as missing output.

    ``invalid_tool_protocol`` also covers provider protocol errors, mixed
    control calls, and a non-``final_answer`` tool on a forced turn, which can
    follow a child that did produce text. For those the child's own error is the
    parent's only way to tell "the child model misbehaved" from "the child had
    nothing to say", so it must survive classification.
    """

    child_error = "The model returned an invalid tool protocol response."

    async def execute_task():
        return {
            "status": "invalid_tool_protocol",
            "success": False,
            "error": child_error,
            "empty_final_answer": marker,
        }

    _patch_delegated_runtime(
        monkeypatch, execute_task, close_config=_SucceedingCloseConfig
    )
    tool = _delegated_agent_tool()
    monkeypatch.setattr(tool, "_create_child_execution_tracer", lambda **_kwargs: None)

    result = await tool.run_json_async({"task": "run"})

    assert result["status"] == "error"
    assert result["success"] is False
    if expects_no_answer_classification:
        assert result["failure_code"] == "missing_delegated_output"
        assert child_error not in result["output"]
    else:
        assert "failure_code" not in result
        assert result["output"] == child_error


@pytest.mark.asyncio
async def test_agent_tool_real_child_answering_with_placeholder_text_succeeds(
    monkeypatch, tmp_path
):
    """A real child whose final answer equals a placeholder string succeeds.

    The placeholder sentinels only apply to the normalized fallback surface,
    not to the child's own raw final answer. A child asked to answer with
    the literal text "No output provided" completed its task and must reach
    the parent as a normal response, not be classified as missing.
    """

    llm = _StubSingleCallLLM(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "final_answer",
                        "arguments": json.dumps({"answer": "No output provided"}),
                    },
                }
            ],
            "done": False,
        }
    )
    tool = _real_delegated_agent_tool(monkeypatch, tmp_path, llm)

    result = await tool.run_json_async({"task": "run"})

    assert result == {"response": "No output provided"}


@pytest.mark.asyncio
async def test_agent_tool_real_child_exhausting_iterations_classifies_as_generic_failure(
    monkeypatch, tmp_path
):
    """A real ReAct child that never reaches a final answer fails closed generically.

    The stub LLM always calls a tool name that doesn't exist in the (empty)
    tool list handed to the child, so every turn's ``_execute_tool_safely``
    converts the ``Tool not found`` lookup into a failed tool result
    (react.py:2759-2764) and the pattern keeps looping. With
    ``execution_mode="flash"`` the child runs the ``single_call`` pattern
    (``max_iterations=2``, ``execution_adapter.py:310-319``), so it exhausts
    its iteration budget after two turns and returns
    ``PatternResult(success=False, error="ReActPattern reached max
    iterations...", metadata={"status": "max_iterations"})``
    (react.py:679-685) instead of pausing or completing.

    That status is neither ``waiting_for_user`` nor ``completed``, so
    ``_classify_delegated_child_failure`` takes its generic branch
    (agent_tool.py:1647-1657): a classified failure with no ``failure_code``,
    read against the real ``AgentExecutionAdapter._normalize_result`` output
    shape rather than a hand-built stand-in for it.
    """

    llm = _StubSingleCallLLM(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "tool_that_does_not_exist",
                        "arguments": "{}",
                    },
                }
            ],
            "done": False,
        }
    )
    tool = _real_delegated_agent_tool(
        monkeypatch, tmp_path, llm, execution_mode="flash"
    )

    result = await tool.run_json_async({"task": "run"})

    assert result["success"] is False
    assert result["is_error"] is True
    assert result["status"] == "error"
    assert "failure_code" not in result
    assert tool_result_succeeded(result) is False


def _create_factory() -> tuple[sessionmaker, str]:
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    db_url = f"sqlite:///{temp_db.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal, temp_db.name


def test_agent_tool_does_not_share_a_live_session_with_child_config(monkeypatch):
    """The child WebToolConfig must be built with a factory, never a live session."""
    SessionLocal, db_path = _create_factory()
    try:
        seed = SessionLocal()
        try:
            user = User(username="iso_owner", password_hash="x", is_admin=False)
            seed.add(user)
            seed.commit()
            seed.refresh(user)

            model = Model(
                model_id="general-model",
                model_provider="openai",
                model_name="General Model",
                api_key="x",
            )
            seed.add(model)
            seed.commit()
            seed.refresh(model)

            agent = Agent(
                user_id=user.id,
                name="Iso Worker",
                status=AgentStatus.PUBLISHED,
                models={"general": model.id},
            )
            seed.add(agent)
            seed.commit()
            seed.refresh(agent)

            agent_id = agent.id
            user_id = user.id
        finally:
            seed.close()

        # Make model resolution succeed so we reach the WebToolConfig build.
        monkeypatch.setattr(
            UserAwareModelStorage,
            "get_llm_by_name_with_access",
            lambda self, model_id, uid: object(),
        )

        captured: dict = {}

        def spy(*args, **kwargs):
            captured["db"] = kwargs.get("db")
            captured["db_factory"] = kwargs.get("db_factory")
            raise _Stop()

        monkeypatch.setattr(mod, "WebToolConfig", spy)

        tool = AgentTool(
            agent_id=agent_id,
            agent_name="Iso Worker",
            agent_description="d",
            session_factory=SessionLocal,
            user_id=user_id,
            tool_name="t",
            tool_description="d",
        )

        try:
            asyncio.run(tool.run_json_async({"task": "hi"}))
        except _Stop:
            pass

        assert captured["db"] is None
        assert captured["db_factory"] is SessionLocal
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass
