"""Tests for ``task_setup_snapshot`` — the off-loop loader that batches
``get_agent_for_task``'s synchronous DB block.

Two invariants the snapshot must hold under all paths:

    1. **No ORM leak.** Every field returned by
       ``load_task_setup_snapshot_sync`` must be either a primitive,
       an enum value, a frozen dataclass, or a fully-constructed
       application-layer object (``BaseLLM``). A downstream caller
       reading any field after the loader's session closes must not
       trip ``DetachedInstanceError``. The ``test_*_no_orm_leak``
       cases enforce this with ``isinstance`` assertions.

    2. **Agent-builder override semantics.** When ``task.agent_id``
       resolves to an ``Agent`` row, the snapshot's resolved LLMs and
       ``task_pattern`` must reflect the agent-builder configuration,
       not the per-task fields. ``excluded_agent_id`` is only set for
       ``PUBLISHED`` agents.

Tests use SQLite in a temp directory + direct ORM, in line with the
existing ``test_task_orchestrator`` fixture style.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from xagent.core.agent.checkpoint import CHECKPOINT_TYPE
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import DAGExecution, Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
    load_task_setup_snapshot_sync,
)
from xagent.web.services.trace_message_storage import (
    MESSAGE_REFS_ENCODING,
    encode_checkpoint_data_for_storage,
)
from xagent.web.services.workforce_runtime import WorkforceTaskRuntime


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'snapshot.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _create_user(db) -> User:
    user = User(username="snap-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_task(db, user_id: int, **overrides) -> Task:
    defaults = dict(
        user_id=user_id,
        title="Snapshot test",
        description="snapshot",
        status=TaskStatus.PENDING,
        execution_mode="flash",
        source="sdk",
    )
    defaults.update(overrides)
    task = Task(**defaults)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _create_agent(db, user_id: int, **overrides) -> Agent:
    defaults = dict(
        user_id=user_id,
        name="snap-agent",
        instructions="be terse",
        status=AgentStatus.PUBLISHED,
        execution_mode="balanced",
        models={},
        knowledge_bases=[],
        skills=[],
        tool_categories=["basic"],
    )
    defaults.update(overrides)
    agent = Agent(**defaults)
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def test_returns_none_when_task_missing(db_session) -> None:
    """A task_id with no matching row must yield ``None``, mirroring
    the legacy ``else`` branch's "Task not found" fallback."""
    snapshot = load_task_setup_snapshot_sync(task_id=99999, task_owner_user_id=1)
    assert snapshot is None


def test_owner_mismatch_raises_distinct_from_missing(db_session) -> None:
    """Identity guard: a task that exists but is owned by a different user
    must raise ``TaskOwnerMismatchError`` -- NOT return ``None`` (which means
    "task missing"). Keeps runtime model/tool resolution from running as the
    wrong user, and lets callers tell a vanished task apart from an identity
    inconsistency."""
    from xagent.web.services.task_setup_snapshot import TaskOwnerMismatchError

    user = _create_user(db_session)
    task = _create_task(db_session, user_id=int(user.id))

    with pytest.raises(TaskOwnerMismatchError):
        load_task_setup_snapshot_sync(
            task_id=int(task.id), task_owner_user_id=int(user.id) + 9999
        )

    # None owner skips the check (backward compat for callers without an owner).
    assert (
        load_task_setup_snapshot_sync(task_id=int(task.id), task_owner_user_id=None)
        is not None
    )


def test_actor_authorization_is_applied_inside_snapshot_session(db_session) -> None:
    owner = _create_user(db_session)
    stranger = User(username="stranger", password_hash="hash", is_admin=False)
    admin = User(username="admin", password_hash="hash", is_admin=True)
    db_session.add_all([stranger, admin])
    db_session.commit()
    task = _create_task(db_session, user_id=int(owner.id))

    assert (
        load_task_setup_snapshot_sync(
            task_id=int(task.id),
            task_owner_user_id=None,
            actor_user_id=int(stranger.id),
            actor_is_admin=False,
        )
        is None
    )
    admin_snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id),
        task_owner_user_id=None,
        actor_user_id=int(admin.id),
        actor_is_admin=True,
    )
    assert admin_snapshot is not None
    assert admin_snapshot.task.user_id == int(owner.id)


def test_basic_task_no_agent_builder(db_session) -> None:
    """Happy path: standalone task with no ``agent_id``. Snapshot
    populates ``task`` and resolves LLMs from task fields only;
    ``agent`` / ``agent_config`` / ``excluded_agent_id`` stay None."""
    user = _create_user(db_session)
    task = _create_task(db_session, user_id=int(user.id))

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert isinstance(snapshot, TaskSetupSnapshot)
    assert is_dataclass(snapshot)

    # _TaskFields primitives
    assert isinstance(snapshot.task, _TaskFields)
    assert snapshot.task.id == int(task.id)
    assert snapshot.task.user_id == int(user.id)
    assert snapshot.task.status == TaskStatus.PENDING
    assert snapshot.task.source == "sdk"
    assert snapshot.task.run_id == task.run_id
    assert snapshot.task.state_version == int(task.state_version or 0)
    assert snapshot.task.control_state == task.control_state
    assert snapshot.task.agent_id is None
    assert snapshot.task.execution_mode == "flash"
    assert snapshot.runtime_user == RuntimeUserFields(id=int(user.id), is_admin=False)
    assert snapshot.has_reconstructable_history is False

    # No agent-builder branch fired
    assert snapshot.agent is None
    assert snapshot.agent_config is None
    assert snapshot.excluded_agent_id is None

    # task_pattern derived from execution_mode
    assert snapshot.task_pattern == "single_call"  # "flash" -> single_call


def test_runtime_user_voice_reduced_from_preferences(db_session) -> None:
    """`RuntimeUserFields.voice` is reduced from the owner's preferences
    JSON by the SAME query that fetches id/is_admin - see
    api/agents.py's apply_user_voice, which relies on this to avoid a
    second query against a request session that may already be
    released by the time a system prompt is assembled."""
    user = _create_user(db_session)
    user.preferences = {"voice": "warm", "department": "Sales"}
    db_session.commit()
    task = _create_task(db_session, user_id=int(user.id))

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.runtime_user == RuntimeUserFields(
        id=int(user.id), is_admin=False, voice="warm"
    )


def test_runtime_user_voice_is_none_without_preferences(db_session) -> None:
    user = _create_user(db_session)
    task = _create_task(db_session, user_id=int(user.id))

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.runtime_user.voice is None


@pytest.mark.parametrize("source", ["sdk", "trigger", None])
def test_task_source_is_preserved(db_session, source: str | None) -> None:
    """The setup policy owner receives the persisted task origin unchanged."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user_id=int(user.id),
        source=source if source is not None else "internal",
    )
    if source is None:
        task.source = None
        db_session.commit()

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.task.source == source


def test_reconstructable_history_is_captured_in_snapshot(db_session) -> None:
    """The retry pre-check is read in the worker-owned snapshot session."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user_id=int(user.id),
        status=TaskStatus.RUNNING,
    )

    build_event = TraceEvent(
        task_id=int(task.id),
        build_id="builder-session",
        event_id="build-only",
        event_type="build_step",
        timestamp=datetime.now(timezone.utc),
        data={},
    )
    db_session.add(build_event)
    db_session.commit()

    without_runtime_history = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )
    assert without_runtime_history is not None
    assert without_runtime_history.has_reconstructable_history is False

    runtime_event = TraceEvent(
        task_id=int(task.id),
        build_id=None,
        event_id="runtime-event",
        event_type="agent_step",
        timestamp=datetime.now(timezone.utc),
        data={},
    )
    db_session.add(runtime_event)
    db_session.commit()

    with_trace_history = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )
    assert with_trace_history is not None
    assert with_trace_history.has_reconstructable_history is True

    db_session.delete(runtime_event)
    db_session.add(DAGExecution(task_id=int(task.id), current_plan={"steps": []}))
    db_session.commit()

    with_dag_history = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )
    assert with_dag_history is not None
    assert with_dag_history.has_reconstructable_history is True


def test_snapshot_captures_turn_transcript_recovery_and_decoded_reconstruction(
    db_session,
) -> None:
    """Every retry read is completed inside the snapshot's worker session."""
    user = _create_user(db_session)
    task = _create_task(
        db_session,
        user_id=int(user.id),
        status=TaskStatus.RUNNING,
    )
    prior_message = TaskChatMessage(
        task_id=int(task.id),
        user_id=int(user.id),
        role="user",
        content="prior turn",
        message_type="user_message",
    )
    current_message = TaskChatMessage(
        task_id=int(task.id),
        user_id=int(user.id),
        role="user",
        content="current turn",
        message_type="user_message",
    )
    db_session.add_all([prior_message, current_message])
    db_session.flush()

    checkpoint_messages = [{"role": "user", "content": "checkpoint input"}]
    checkpoint_data = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "root_execution_id": "execution-1",
        "execution_id": "execution-1",
        "label": "after_llm",
        "snapshot": {
            "type": "checkpoint",
            "label": "after_llm",
            "execution_id": "execution-1",
            "context": {"messages": checkpoint_messages},
            "pattern": "ReActPattern",
            "pattern_state": {"current_iteration": 1},
        },
    }
    encoded_checkpoint = encode_checkpoint_data_for_storage(
        db_session,
        task_id=int(task.id),
        data=checkpoint_data,
        use_v2=False,
    )
    assert (
        encoded_checkpoint["snapshot"]["context"]["messages"]["__encoding"]
        == MESSAGE_REFS_ENCODING
    )
    now = datetime.now(timezone.utc)
    db_session.add_all(
        [
            TraceEvent(
                task_id=int(task.id),
                build_id=None,
                event_id="checkpoint-event",
                event_type="pattern_checkpoint",
                timestamp=now,
                data=encoded_checkpoint,
            ),
            TraceEvent(
                task_id=int(task.id),
                build_id=None,
                event_id="tool-event",
                event_type="tool_execution_end",
                timestamp=now,
                data={
                    "tool_name": "web_search",
                    "success": True,
                    "result": {"title": "prior evidence"},
                },
            ),
            TraceEvent(
                task_id=int(task.id),
                build_id=None,
                event_id="skill-event",
                event_type="skill_select_end",
                timestamp=now,
                data={"selected": True, "skill_name": "translator"},
            ),
            DAGExecution(
                task_id=int(task.id),
                current_plan={"steps": [{"id": "step-1"}]},
            ),
        ]
    )
    db_session.commit()

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id),
        task_owner_user_id=int(user.id),
        before_message_id=int(current_message.id),
    )

    assert snapshot is not None
    assert snapshot.conversation_history == ({"role": "user", "content": "prior turn"},)
    assert snapshot.execution_recovery.selected_skill_name == "translator"
    assert len(snapshot.execution_recovery.messages) == 1
    assert "prior evidence" in snapshot.execution_recovery.messages[0]["content"]
    assert snapshot.reconstruction.plan_state == {"steps": [{"id": "step-1"}]}
    checkpoint_event = next(
        event
        for event in snapshot.reconstruction.tracer_events
        if event["id"] == "checkpoint-event"
    )
    assert checkpoint_event["data"]["snapshot"]["context"]["messages"] == (
        checkpoint_messages
    )


def test_snapshot_carries_verified_workforce_runtime(db_session) -> None:
    """The runtime resolved by llm_utils is not discarded and re-queried later."""
    user = _create_user(db_session)
    task = _create_task(db_session, user_id=int(user.id), agent_config={})
    manager = _create_agent(
        db_session,
        user_id=int(user.id),
        id=51,
        name="Research manager",
    )
    workforce = Workforce(
        id=41,
        owner_user_id=int(user.id),
        scope_type="user",
        scope_id=str(user.id),
        name="Research team",
        description="Coordinates research tasks",
        manager_agent_id=int(manager.id),
        status="active",
    )
    db_session.add(workforce)
    db_session.flush()
    workforce_snapshot = {
        "version": 1,
        "workforce": {"id": 41, "name": "Research team"},
        "manager": {"agent_id": 51, "runtime_prompt": "Coordinate workers"},
        "workers": [
            {
                "agent_id": 61,
                "name": "Researcher",
                "alias": "Researcher",
                "description": "Find evidence",
                "assignment_instructions": "Find evidence",
                "enabled": True,
            }
        ],
    }
    run = WorkforceRun(
        workforce_id=int(workforce.id),
        task_id=int(task.id),
        user_id=int(user.id),
        status="pending",
        snapshot=workforce_snapshot,
    )
    db_session.add(run)
    db_session.flush()
    task.agent_config = {"workforce_run_id": int(run.id)}
    db_session.commit()

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert isinstance(snapshot.workforce_runtime, WorkforceTaskRuntime)
    assert snapshot.workforce_runtime.workforce_run_id == int(run.id)
    assert snapshot.workforce_runtime.allowed_agent_ids == [61]


def test_inline_published_preview_agent_is_excluded_in_snapshot(db_session) -> None:
    """Preview exclusion uses the same owner/team visibility rule off-loop."""
    user = _create_user(db_session)
    preview_agent = _create_agent(db_session, user_id=int(user.id))
    task = _create_task(
        db_session,
        user_id=int(user.id),
        agent_id=None,
        agent_config={
            "instructions": "preview instructions",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": [],
            "preview_agent_id": int(preview_agent.id),
        },
    )

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.agent is None
    assert snapshot.excluded_agent_id == int(preview_agent.id)


def test_inline_preview_agent_with_noncanonical_id_is_not_excluded_in_snapshot(
    db_session,
) -> None:
    user = _create_user(db_session)
    preview_agent = _create_agent(db_session, user_id=int(user.id))
    task = _create_task(
        db_session,
        user_id=int(user.id),
        agent_config={
            "instructions": "preview instructions",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": [],
            "preview_agent_id": f"0{int(preview_agent.id)}",
        },
    )

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.excluded_agent_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_excluded"),
    [
        (AgentStatus.PUBLISHED, True),
        (AgentStatus.DRAFT, False),
    ],
)
async def test_build_tools_inline_preview_uses_persisted_task_owner(
    db_session,
    monkeypatch,
    status,
    expected_excluded,
) -> None:
    """The live DB reader authorizes preview IDs as the persisted task owner."""
    from xagent.web.api import chat as chat_module
    from xagent.web.api.chat import AgentServiceManager

    owner = _create_user(db_session)
    actor = User(username="preview-actor", password_hash="hash", is_admin=False)
    db_session.add(actor)
    db_session.commit()
    db_session.refresh(actor)
    preview_agent = _create_agent(db_session, int(owner.id), status=status)
    task = _create_task(
        db_session,
        user_id=int(owner.id),
        agent_config={
            "instructions": "preview instructions",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": [],
            "preview_agent_id": str(int(preview_agent.id)),
        },
    )

    resolved_owner_ids: list[int] = []
    real_resolver = chat_module.resolve_authorized_agent

    def observe_resolver(session, owner_user_id, candidate_id):
        resolved_owner_ids.append(owner_user_id)
        return real_resolver(session, owner_user_id, candidate_id)

    observed_excluded_ids: list[int | None] = []

    async def capture_tools(*_args, **kwargs):
        observed_excluded_ids.append(kwargs["excluded_agent_id"])
        return [], object()

    monkeypatch.setattr(chat_module, "resolve_authorized_agent", observe_resolver)
    monkeypatch.setattr(chat_module, "create_default_tools", capture_tools)
    monkeypatch.setattr("xagent.web.sandbox_manager.get_sandbox_manager", lambda: None)

    manager = AgentServiceManager()
    await manager._build_tools_for_task(
        task_id=int(task.id),
        task=task,
        db=db_session,
        user=actor,
        agent_config=task.agent_config,
        task_llm=None,
        task_vision_llm=None,
    )

    assert resolved_owner_ids == [int(owner.id)]
    assert int(actor.id) != int(owner.id)
    assert observed_excluded_ids == [
        int(preview_agent.id) if expected_excluded else None
    ]


def test_agent_builder_published_sets_excluded_agent_id(db_session) -> None:
    """Task pointing at a PUBLISHED agent: excluded_agent_id matches
    agent.id, agent_config is populated, agent.status flows through."""
    user = _create_user(db_session)
    agent = _create_agent(
        db_session,
        user_id=int(user.id),
        status=AgentStatus.PUBLISHED,
        execution_mode="think",
        tool_categories=["basic", "mcp:Gmail"],
    )
    task = _create_task(
        db_session,
        user_id=int(user.id),
        agent_id=int(agent.id),
        execution_mode=None,  # let agent-builder execution_mode take over downstream
    )

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.agent is not None
    assert isinstance(snapshot.agent, AgentRuntimeFields)
    assert snapshot.agent.id == int(agent.id)
    assert snapshot.agent.name == "snap-agent"
    assert snapshot.agent.status == AgentStatus.PUBLISHED
    assert snapshot.excluded_agent_id == int(agent.id)

    assert snapshot.agent_config is not None
    assert snapshot.agent_config["execution_mode"] == "think"
    assert snapshot.agent_config["instructions"] == "be terse"
    assert snapshot.agent_config["tool_categories"] == ["basic", "mcp:Gmail"]
    # llms tuple shape (all None because no DBModel rows seeded)
    assert "llms" in snapshot.agent_config
    assert len(snapshot.agent_config["llms"]) == 4


def test_agent_builder_draft_no_excluded_agent_id(db_session) -> None:
    """A DRAFT agent must still load config (so the task can run for
    its owner) but must NOT be added to ``excluded_agent_id``: only
    PUBLISHED agents exclude themselves from the tool list."""
    user = _create_user(db_session)
    agent = _create_agent(
        db_session,
        user_id=int(user.id),
        status=AgentStatus.DRAFT,
    )
    task = _create_task(db_session, user_id=int(user.id), agent_id=int(agent.id))

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )

    assert snapshot is not None
    assert snapshot.agent is not None
    assert snapshot.agent.status == AgentStatus.DRAFT
    assert snapshot.excluded_agent_id is None
    # agent_config still populated -- excluded_agent_id is orthogonal.
    assert snapshot.agent_config is not None


def test_task_pattern_derived_from_execution_mode(db_session) -> None:
    """Spot-check each execution_mode -> pattern mapping the snapshot
    inherits from ``get_agent_pattern_for_execution_mode``."""
    user = _create_user(db_session)
    cases = [
        ("flash", "single_call"),
        ("balanced", "react"),
        ("think", "dag_plan_execute"),
        ("auto", "auto"),
    ]
    for mode, expected_pattern in cases:
        task = _create_task(db_session, user_id=int(user.id), execution_mode=mode)
        snapshot = load_task_setup_snapshot_sync(
            task_id=int(task.id), task_owner_user_id=int(user.id)
        )
        assert snapshot is not None
        assert snapshot.task_pattern == expected_pattern, (
            f"execution_mode={mode!r} expected pattern={expected_pattern!r}, "
            f"got {snapshot.task_pattern!r}"
        )


def test_no_orm_leak_in_returned_fields(db_session) -> None:
    """Strict primitive-only invariant. A future refactor that
    accidentally puts an ORM row in the snapshot (e.g. ``return
    TaskSetupSnapshot(task=task_row, ...)``) would fail here -- the
    loader's session has already closed by the time these assertions
    run.

    This is the load-bearing test for cross-thread safety: when
    ``get_agent_for_task`` calls ``asyncio.to_thread(load_...)``, the
    returned object must be usable on the loop thread without lazy-
    loading anything against the now-closed session.
    """
    user = _create_user(db_session)
    agent = _create_agent(db_session, user_id=int(user.id))
    task = _create_task(db_session, user_id=int(user.id), agent_id=int(agent.id))

    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )
    assert snapshot is not None

    # The frozen container.
    assert is_dataclass(snapshot)
    # _TaskFields: every visible attr must be primitive / enum.
    assert isinstance(snapshot.task, _TaskFields)
    assert isinstance(snapshot.task.id, int)
    assert isinstance(snapshot.task.user_id, int)
    assert isinstance(snapshot.task.status, TaskStatus)
    assert snapshot.task.source is None or isinstance(snapshot.task.source, str)
    assert snapshot.task.agent_id is None or isinstance(snapshot.task.agent_id, int)
    assert snapshot.task.execution_mode is None or isinstance(
        snapshot.task.execution_mode, str
    )
    assert snapshot.task.agent_type is None or isinstance(snapshot.task.agent_type, str)
    assert isinstance(snapshot.runtime_user, RuntimeUserFields)
    assert isinstance(snapshot.runtime_user.id, int)
    assert isinstance(snapshot.runtime_user.is_admin, bool)
    assert isinstance(snapshot.has_reconstructable_history, bool)
    # agent_config is JSON column -- dict or None, never an ORM proxy.
    assert snapshot.task.agent_config is None or isinstance(
        snapshot.task.agent_config, dict
    )

    # AgentRuntimeFields (when present): same invariant.
    assert isinstance(snapshot.agent, AgentRuntimeFields)
    assert isinstance(snapshot.agent.id, int)
    assert isinstance(snapshot.agent.name, str)
    assert isinstance(snapshot.agent.status, AgentStatus)
    assert snapshot.agent.instructions is None or isinstance(
        snapshot.agent.instructions, str
    )

    # agent_config is a plain dict whose JSON-column values are plain
    # Python collections (lists / dict / str / None).
    cfg = snapshot.agent_config
    assert isinstance(cfg, dict)
    assert isinstance(cfg["skills"], list)
    assert isinstance(cfg["knowledge_bases"], list)
    assert isinstance(cfg["tool_categories"], list)
    # llms tuple: each slot is None or a BaseLLM (not an ORM).
    from xagent.core.model.chat.basic.base import BaseLLM

    for slot in cfg["llms"]:
        assert slot is None or isinstance(slot, BaseLLM), (
            f"llms slot leaked non-BaseLLM type: {type(slot).__name__}"
        )

    # task_pattern is a plain string.
    assert isinstance(snapshot.task_pattern, str)

    # excluded_agent_id either None or int.
    assert snapshot.excluded_agent_id is None or isinstance(
        snapshot.excluded_agent_id, int
    )


def test_snapshot_frozen_dataclass(db_session) -> None:
    """A frozen dataclass prevents accidental mutation by downstream
    code that mistakes the snapshot for a config dict it can amend.
    Mutating any field must raise ``FrozenInstanceError``."""
    from dataclasses import FrozenInstanceError

    user = _create_user(db_session)
    task = _create_task(db_session, user_id=int(user.id))
    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(user.id)
    )
    assert snapshot is not None

    with pytest.raises(FrozenInstanceError):
        snapshot.task_pattern = "react"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        snapshot.task.user_id = 999  # type: ignore[misc]


def test_session_closes_even_when_loader_raises(db_session) -> None:
    """The loader opens its own session and must close it in a
    ``finally`` even when an inner query raises -- otherwise a leaked
    connection eventually exhausts the pool under load.

    We simulate the failure by patching
    ``resolve_task_runtime_config_core`` (the shared helper now
    invoked by the snapshot loader) to raise mid-load, then verify
    the snapshot session closed by issuing a fresh query against
    ``db_session``. This is a structural test of the ``try/finally``,
    not of the error message.
    """
    user = _create_user(db_session)
    task = _create_task(db_session, user_id=int(user.id))

    boom = RuntimeError("simulated llm-resolve failure")
    # The snapshot loader does a lazy
    # ``from .llm_utils import resolve_task_runtime_config_core``
    # inside its body, so the patch must target the source module.
    with patch(
        "xagent.web.services.llm_utils.resolve_task_runtime_config_core",
        side_effect=boom,
    ):
        with pytest.raises(RuntimeError, match="simulated llm-resolve failure"):
            load_task_setup_snapshot_sync(
                task_id=int(task.id), task_owner_user_id=int(user.id)
            )

    # If the session leaked, this fresh query would block / fail on
    # SQLite (single-writer) or exhaust the pool elsewhere. A clean
    # round-trip here confirms the finally branch did its job.
    from sqlalchemy import text

    result = db_session.execute(text("SELECT 1")).scalar()
    assert result == 1
