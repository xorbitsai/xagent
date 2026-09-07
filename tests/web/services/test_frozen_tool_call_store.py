"""The approved payload is what runs -- and it runs at most once.

These tests drive the real adapter, the real hooks and a real database.
The incident this mechanism exists to prevent is specific: a person
approved a LinkedIn post and a different post went out, because the
arguments were written again after the approval. So the assertions are on
the arguments the connector actually received, never on the pause alone.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp.types import Tool as MCPTool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.mcp_adapter import _build_mcp_tool_adapter
from xagent.core.tools.adapters.vibe.write_gate import (
    get_write_gate_hook,
    set_write_gate_hook,
    set_write_gate_resume_hook,
)
from xagent.core.tools.core.mcp.tools import attach_raw_annotations
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.frozen_tool_call import FrozenToolCall
from xagent.web.services import frozen_tool_call_store as store
from xagent.web.services.task_lease_service import TaskLease, bind_task_lease_context

from .task_interaction_schema_shared import make_task, make_user

APPROVED_ARGS = {
    "text": "The copy the user actually approved.",
    "image_path": "/workspace/tasks/248032/output/generated_image_19bb735e.png",
}


@pytest.fixture
def session_factory(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'frozen.db'}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def task_id(session_factory) -> int:
    db = session_factory()
    tid = make_task(db, user_id=make_user(db))
    db.close()
    return tid


@pytest.fixture(autouse=True)
def _uninstall_gate():
    yield
    store.uninstall_write_gate()


@pytest.fixture
def gate(session_factory, monkeypatch):
    """Install the gate against this test's database."""
    monkeypatch.setattr(store, "get_session_local", lambda: session_factory)
    return store


def _adapter(raw_annotations=None, name: str = "create_post"):
    tool = MCPTool.model_validate(
        {
            "name": name,
            "description": "d",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "image_path": {"type": "string"},
                },
            },
        }
    )
    attach_raw_annotations(tool, raw_annotations)
    return _build_mcp_tool_adapter("linkedin", SimpleNamespace(), tool)


class _Executions(list):
    """Records the arguments each execution actually received.

    ``as_patch()`` returns a plain function so ``patch.object`` installs
    something that binds ``self`` the way the real method does; a callable
    instance would be treated as an already-bound attribute and swallow the
    adapter argument.
    """

    def as_patch(self):
        recorded = self

        async def _execute(_self, _connection, args, _meta):
            recorded.append(dict(args))
            return {"success": True, "executed_with": dict(args)}

        return _execute


def _pause(adapter, task_id, args=APPROVED_ARGS):
    lease = TaskLease(task_id=task_id, runner_id="runner", run_id="run-1")
    with bind_task_lease_context(lease):
        return asyncio.run(adapter.run_json_async(dict(args)))


def test_a_gated_call_pauses_without_executing(gate, task_id, session_factory):
    """The whole point: the call does not run when it is gated."""
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        result = _pause(adapter, task_id)

    assert result["status"] == "waiting_for_user"
    assert executions == []
    # Exactly one confirm, which is the shape the approval buttons render.
    assert [i["type"] for i in result["interactions"]] == ["confirm"]
    assert [o["value"] for o in result["interactions"][0]["options"]] == [
        "approve",
        "reject",
    ]

    with session_factory() as db:
        row = db.get(FrozenToolCall, result["interaction_id"])
    assert row.status == "pending"
    assert row.arguments == APPROVED_ARGS
    assert row.task_id == task_id


def test_approval_executes_the_frozen_arguments(gate, task_id):
    """The regression this whole mechanism exists for.

    A model that rewrote its arguments after the approval would show up
    here as an execution that does not equal what was frozen.
    """
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        paused = _pause(adapter, task_id)
        asyncio.run(
            adapter.resume_user_interaction(
                interaction_id=paused["interaction_id"], response="approve"
            )
        )

    assert executions == [APPROVED_ARGS]


def test_rejection_never_executes(gate, task_id, session_factory):
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        paused = _pause(adapter, task_id)
        result = asyncio.run(
            adapter.resume_user_interaction(
                interaction_id=paused["interaction_id"], response="reject"
            )
        )

    assert executions == []
    assert result["status"] == "cancelled"
    with session_factory() as db:
        assert db.get(FrozenToolCall, paused["interaction_id"]).status == "voided"


def test_a_second_answer_does_not_execute_again(gate, task_id):
    """Two clicks on one approval must publish once."""
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        paused = _pause(adapter, task_id)
        first = asyncio.run(
            adapter.resume_user_interaction(
                interaction_id=paused["interaction_id"], response="approve"
            )
        )
        second = asyncio.run(
            adapter.resume_user_interaction(
                interaction_id=paused["interaction_id"], response="approve"
            )
        )

    assert len(executions) == 1
    assert first["success"] is True
    assert second["success"] is False


def test_an_expired_approval_does_not_execute(gate, task_id, session_factory):
    """A yes that arrives after the deadline must not publish."""
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        paused = _pause(adapter, task_id)
        with session_factory() as db:
            row = db.get(FrozenToolCall, paused["interaction_id"])
            row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            db.commit()
        result = asyncio.run(
            adapter.resume_user_interaction(
                interaction_id=paused["interaction_id"], response="approve"
            )
        )

    assert executions == []
    assert "expired" in result["error"]
    with session_factory() as db:
        assert db.get(FrozenToolCall, paused["interaction_id"]).status == "voided"


@pytest.mark.parametrize(
    "response", ["approve please", "yes", "", "rejected", "APPROVE "]
)
def test_only_the_exact_option_value_grants(gate, task_id, response):
    """Free text must not approve.

    The pause offers two option values, so anything else reaching the
    callback did not come from those buttons. Only an exact "approve"
    (case and surrounding space aside) runs the payload.
    """
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        paused = _pause(adapter, task_id)
        asyncio.run(
            adapter.resume_user_interaction(
                interaction_id=paused["interaction_id"], response=response
            )
        )

    assert executions == (
        [APPROVED_ARGS] if response.strip().lower() == "approve" else []
    )


def test_a_read_only_declaration_is_not_gated(gate, task_id):
    """A server that declares read-only runs without asking."""
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter({"readOnlyHint": True})
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        result = _pause(adapter, task_id)

    assert executions == [APPROVED_ARGS]
    assert result.get("status") != "waiting_for_user"


@pytest.mark.parametrize(
    "raw",
    [{"readOnlyHint": "true"}, {"readOnlyHint": 1}, {"destructiveHint": True}, None],
)
def test_anything_short_of_a_read_only_declaration_is_gated(gate, task_id, raw):
    """Silence and coercible non-booleans are writes, not promises.

    ``"true"`` and ``1`` are what an untrusted server can send to look
    read-only after the SDK's non-strict validation flattens them; neither
    may skip the gate.
    """
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter(raw)
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        result = _pause(adapter, task_id)

    assert result["status"] == "waiting_for_user"
    assert executions == []


def test_no_policy_means_nothing_is_gated(gate, task_id):
    """Installed but inert: an approval pending from before can still settle."""
    gate.install_write_gate(policy=None)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        result = _pause(adapter, task_id)

    assert executions == [APPROVED_ARGS]
    assert result.get("status") != "waiting_for_user"


def test_nothing_installed_leaves_execution_untouched(task_id):
    """The off switch: no hook, no gate, no row, no behavior change."""
    assert get_write_gate_hook() is None
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        result = _pause(adapter, task_id)

    assert executions == [APPROVED_ARGS]
    assert result.get("status") != "waiting_for_user"


def test_a_failing_hook_lets_the_call_through(task_id):
    """Deliberately fail-open; the module docstring argues why.

    This seam makes an approved call faithful. It is not what stops a
    dangerous one, and failing closed here would strand every connector in
    a workspace behind an approval nobody can grant.
    """

    def boom(_call):
        raise RuntimeError("policy lookup failed")

    set_write_gate_hook(boom)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        result = _pause(adapter, task_id)

    assert executions == [APPROVED_ARGS]
    assert result.get("status") != "waiting_for_user"


def test_an_unknown_interaction_never_executes(gate, task_id):
    """A forged or stale id must not run anything."""
    gate.install_write_gate(policy=lambda call: True)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        result = asyncio.run(
            adapter.resume_user_interaction(
                interaction_id="ftc_does_not_exist", response="approve"
            )
        )

    assert executions == []
    assert result["success"] is False


def test_resume_without_a_host_hook_reports_rather_than_executes(task_id):
    """A host that unregistered mid-flight leaves nothing runnable."""
    set_write_gate_resume_hook(None)
    adapter = _adapter()
    executions = _Executions()

    with patch.object(type(adapter), "_execute_mcp_call", new=executions.as_patch()):
        result = asyncio.run(
            adapter.resume_user_interaction(interaction_id="ftc_x", response="approve")
        )

    assert executions == []
    assert result["success"] is False
