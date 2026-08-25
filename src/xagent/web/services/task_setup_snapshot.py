"""Read-only snapshot of the synchronous DB state required to bootstrap
a task-bound ``AgentService``.

Background:
    ``AgentServiceManager.get_agent_for_task`` runs a contiguous block
    of synchronous DB queries (Task row + per-task LLM resolution +
    optional Agent Builder lookup with up to 4 ``DBModel`` queries and
    4 user-aware LLM access checks) on the main asyncio event loop. On
    a fully-configured Agent Builder task that adds up to 8-12 DB
    round-trips. Under load the block measures 20+ seconds of asyncio
    slow-callback time and blocks every other request on the same
    worker (issue #427 — ``_schedule_bg._runner took 23.371s``
    observed locally on 2026-05-20).

    This module batches those reads into a single function intended to
    be invoked through ``asyncio.to_thread``. The function opens its
    own ``SessionLocal``, eagerly reads everything, closes the session,
    and returns a frozen primitive snapshot. ORM rows MUST NOT escape
    the loader -- a downstream caller that mistakenly held an ORM
    reference past the close would hit ``DetachedInstanceError`` on
    its next attribute access.

Out of scope:
    * ``UploadedFile`` selected-files binding -- contains writes and is
      therefore handled by its own worker-owned Session during agent
      bootstrap instead of being folded into this read-only snapshot.
    * MCP server configs -- async + OAuth refresh path.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, cast

from sqlalchemy.orm import Session

from ...core.model.chat.basic.base import BaseLLM
from ..models.database import get_session_local
from ..models.task import DAGExecution, Task, TraceEvent
from ..models.user import User
from .llm_utils import AgentRuntimeFields
from .task_execution_context_service import (
    TaskExecutionRecoverySnapshot,
    load_task_execution_recovery_snapshot_sync,
)

if TYPE_CHECKING:
    from .workforce_runtime import WorkforceTaskRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TaskFields:
    """Primitive subset of the ``Task`` row needed past the snapshot."""

    id: int
    user_id: int
    status: Any  # ``TaskStatus`` enum value (frozen, not ORM).
    agent_id: Optional[int]
    agent_config: Any  # JSON column -- ``dict | None`` in practice.
    model_name: Optional[str]
    compact_model_name: Optional[str]
    execution_mode: Optional[str]
    agent_type: Optional[str]
    source: str | None = None
    run_id: str | None = None
    state_version: int = 0
    control_state: str | None = None


@dataclass(frozen=True)
class RuntimeUserFields:
    """Primitive runtime identity required while an agent is constructed."""

    id: int
    is_admin: bool
    # Onboarding "Launch" step voice choice, verbatim from the raw
    # preferences JSON (not validated against VALID_VOICES here - an
    # unrecognized value is stored as-is and only becomes an inert no-op
    # later, inside apply_output_voice's own isinstance/lookup guard in
    # api/agents.py), or None if the key is unset.
    voice: Optional[str] = None


def detach_runtime_user_fields(user: User) -> RuntimeUserFields:
    """Reduce a live ORM ``User`` row to the primitives ``RuntimeUserFields``
    needs, read now while the row is still attached and unexpired.

    A caller holding a live ``User`` past the point its Session's
    connection gets released (``release_db_connection_if_clean``'s
    rollback unconditionally expires every object that Session loaded)
    would otherwise force an implicit reload on the next attribute
    access - synchronously, on the event loop, in the same window this
    whole snapshot mechanism exists to keep query-free. Call this as
    soon as a live ``User`` is in hand instead of holding onto it."""
    preferences = user.preferences
    return RuntimeUserFields(
        id=int(user.id),
        is_admin=bool(user.is_admin),
        voice=(
            cast(dict, preferences).get("voice")
            if isinstance(preferences, dict)
            else None
        ),
    )


@dataclass(frozen=True)
class TaskReconstructionSnapshot:
    """Detached historical state consumed by agent reconstruction."""

    tracer_events: tuple[dict[str, Any], ...] = ()
    plan_state: Optional[dict[str, Any]] = None
    # Preserve the legacy retry pre-check: any runtime TraceEvent or
    # DAGExecution row attempts reconstruction, even when current_plan is empty.
    has_history: bool = False


@dataclass(frozen=True)
class TaskSetupSnapshot:
    """All synchronous DB state that ``get_agent_for_task`` needs to
    bootstrap a task-bound ``AgentService``.

    Strict invariant: every field is a primitive, an enum, a frozen
    dataclass, or a fully-constructed application-layer object
    (``BaseLLM``) that is safe to read off the loop thread. ORM rows
    must not leak.
    """

    task: _TaskFields
    runtime_user: Optional[RuntimeUserFields]
    has_reconstructable_history: bool
    task_pattern: str
    # Final resolved LLMs after the agent-builder override (if any).
    task_llm: Optional[BaseLLM]
    task_fast_llm: Optional[BaseLLM]
    task_vision_llm: Optional[BaseLLM]
    task_compact_llm: Optional[BaseLLM]
    # Agent Builder configuration -- only populated when
    # ``task.agent_id`` resolves to an existing ``Agent`` row. The
    # frozen dataclass lives in ``llm_utils`` because the same shape
    # is also produced by ``resolve_task_runtime_config_core`` for
    # the main-loop reconstruct path; one definition, one home.
    agent: Optional[AgentRuntimeFields]
    agent_config: Optional[dict]
    excluded_agent_id: Optional[int]
    conversation_history: tuple[dict[str, Any], ...] = ()
    execution_recovery: TaskExecutionRecoverySnapshot = TaskExecutionRecoverySnapshot()
    reconstruction: TaskReconstructionSnapshot = TaskReconstructionSnapshot()
    # Resolved by ``resolve_task_runtime_config_core`` using this same Session.
    # Kept as the service-layer frozen dataclass; no ORM row is retained.
    workforce_runtime: WorkforceTaskRuntime | None = None


# NOTE: All LLM resolution + agent-builder merge + execution-mode →
# pattern logic lives in ``llm_utils.resolve_task_runtime_config_core``.
# This loader is the off-loop wrapper that:
#   1. opens its own ``SessionLocal``,
#   2. calls the shared core to do the actual resolution,
#   3. wraps the resulting ORM ``Task`` row in a frozen
#      ``_TaskFields`` so nothing escapes the loader's session
#      (``Agent`` primitives are already provided by the core as an
#      ``AgentRuntimeFields`` instance).
# Reconstruction and normal creation both consume this detached snapshot;
# neither path re-resolves runtime configuration from a request Session.


class TaskOwnerMismatchError(Exception):
    """The task row's owner does not match the expected ``task_owner_user_id``.

    Distinct from "task missing" (which returns ``None``): a missing task is a
    benign race (deleted before the bg run), whereas an owner mismatch is an
    identity/authorization inconsistency. Raising keeps the runtime from
    silently resolving models / tools as the wrong user.
    """

    def __init__(self, task_id: int, expected: int, actual: int):
        super().__init__(f"task {task_id} owner {actual} != expected {expected}")
        self.task_id = task_id
        self.expected = expected
        self.actual = actual


def load_task_reconstruction_snapshot_sync(
    session: Session,
    task_id: int,
) -> TaskReconstructionSnapshot:
    """Load and decode reconstruction rows before the worker Session closes."""
    from .trace_message_storage import decode_trace_events_data

    trace_rows = (
        session.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
        )
        .all()
    )
    decoded_data = decode_trace_events_data(
        session,
        task_id=task_id,
        data_items=[row.data for row in trace_rows],
        strict=False,
    )
    tracer_events = tuple(
        {
            "id": str(row.event_id),
            "event_type": str(row.event_type),
            "task_id": str(row.task_id),
            "step_id": str(row.step_id) if row.step_id is not None else None,
            "timestamp": row.timestamp.timestamp() if row.timestamp else None,
            "data": deepcopy(data),
            "parent_id": (
                str(row.parent_event_id) if row.parent_event_id is not None else None
            ),
        }
        for row, data in zip(trace_rows, decoded_data)
    )

    dag_row = (
        session.query(DAGExecution).filter(DAGExecution.task_id == task_id).first()
    )
    plan_state = (
        deepcopy(cast(dict[str, Any], dag_row.current_plan))
        if dag_row is not None and dag_row.current_plan
        else None
    )
    return TaskReconstructionSnapshot(
        tracer_events=tracer_events,
        plan_state=plan_state,
        has_history=bool(trace_rows) or dag_row is not None,
    )


def _resolve_inline_preview_excluded_agent_id(
    session: Session,
    task_row: Task,
    agent_config: Optional[dict],
) -> Optional[int]:
    """Resolve the published preview agent with the legacy owner/team rule."""
    if not agent_config or not agent_config.get("preview_agent_id"):
        return None

    from ..models.agent import AgentStatus
    from .agent_team_scope import resolve_authorized_agent

    owner_user_id = int(task_row.user_id)
    preview_agent = resolve_authorized_agent(
        session,
        owner_user_id,
        agent_config.get("preview_agent_id"),
    )
    if preview_agent is None or preview_agent.status != AgentStatus.PUBLISHED:
        return None
    return int(preview_agent.id)


def load_task_setup_snapshot_sync(
    task_id: int,
    task_owner_user_id: Optional[int],
    *,
    before_message_id: Optional[int] = None,
    actor_user_id: Optional[int] = None,
    actor_is_admin: bool = False,
) -> Optional[TaskSetupSnapshot]:
    """Open a dedicated ``SessionLocal``, read every synchronous field
    ``get_agent_for_task`` needs for normal (non-reconstruct) creation,
    close the session, and return a primitive snapshot.

    Designed to be called from the event loop via
    ``await asyncio.to_thread(load_task_setup_snapshot_sync, ...)`` so
    the main loop stays responsive during the read (issue #427).

    ``task_owner_user_id`` is the runtime identity the task runs as. When
    provided it MUST match the row's ``user_id``; a mismatch raises
    :class:`TaskOwnerMismatchError` (never silently splits "snapshot owner"
    from "resolution user"). Model / runtime resolution always uses the
    row's owner, not the passed value.

    Returns ``None`` when the task row is missing -- callers fall back
    to whatever behaviour the legacy in-line code already implements
    for that case (default LLM, no agent-builder override).
    """
    from .llm_utils import resolve_task_runtime_config_core

    session_factory = get_session_local()
    session: Session = session_factory()
    try:
        task_query = session.query(Task).filter(Task.id == task_id)
        if actor_user_id is not None and not actor_is_admin:
            task_query = task_query.filter(Task.user_id == actor_user_id)
        task_row = task_query.first()
        if task_row is None:
            return None

        owner_user_id = int(task_row.user_id)
        if task_owner_user_id is not None and owner_user_id != task_owner_user_id:
            raise TaskOwnerMismatchError(task_id, task_owner_user_id, owner_user_id)

        runtime_user_row = (
            session.query(User.id, User.is_admin, User.preferences)
            .filter(User.id == owner_user_id)
            .first()
        )
        runtime_user = (
            RuntimeUserFields(
                id=int(runtime_user_row[0]),
                is_admin=bool(runtime_user_row[1]),
                voice=(
                    cast(dict, runtime_user_row[2] or {}).get("voice")
                    if isinstance(runtime_user_row[2], dict)
                    else None
                ),
            )
            if runtime_user_row is not None
            else None
        )

        task_fields = _TaskFields(
            id=int(task_row.id),
            user_id=int(task_row.user_id),
            status=task_row.status,
            source=str(task_row.source) if task_row.source is not None else None,
            agent_id=int(task_row.agent_id) if task_row.agent_id is not None else None,
            agent_config=(
                dict(task_row.agent_config)
                if isinstance(task_row.agent_config, dict)
                else task_row.agent_config
            ),
            model_name=(
                str(task_row.model_name) if task_row.model_name is not None else None
            ),
            compact_model_name=(
                str(task_row.compact_model_name)
                if task_row.compact_model_name is not None
                else None
            ),
            execution_mode=getattr(task_row, "execution_mode", None),
            agent_type=(
                str(task_row.agent_type) if task_row.agent_type is not None else None
            ),
            run_id=str(task_row.run_id) if task_row.run_id is not None else None,
            state_version=int(task_row.state_version or 0),
            control_state=(
                str(task_row.control_state)
                if task_row.control_state is not None
                else None
            ),
        )

        # Runtime resolution always uses the row's OWNER, never the passed
        # value (which is validated to match above when provided).
        core = resolve_task_runtime_config_core(
            task_row, session, user_id=owner_user_id
        )
        task_llm, task_fast_llm, task_vision_llm, task_compact_llm = core.llms
        excluded_agent_id = core.excluded_agent_id
        if excluded_agent_id is None:
            excluded_agent_id = _resolve_inline_preview_excluded_agent_id(
                session,
                task_row,
                core.agent_config,
            )

        from .chat_history_service import load_task_transcript

        conversation_history = tuple(
            load_task_transcript(
                session,
                task_id,
                before_message_id=before_message_id,
            )
        )
        execution_recovery = load_task_execution_recovery_snapshot_sync(
            session,
            task_id,
        )
        reconstruction = load_task_reconstruction_snapshot_sync(session, task_id)

        # ``core.agent_fields`` is already an ``AgentRuntimeFields``
        # frozen dataclass; pass it through directly.
        return TaskSetupSnapshot(
            task=task_fields,
            runtime_user=runtime_user,
            has_reconstructable_history=reconstruction.has_history,
            task_pattern=core.task_pattern,
            task_llm=task_llm,
            task_fast_llm=task_fast_llm,
            task_vision_llm=task_vision_llm,
            task_compact_llm=task_compact_llm,
            agent=core.agent_fields,
            agent_config=core.agent_config,
            excluded_agent_id=excluded_agent_id,
            conversation_history=conversation_history,
            execution_recovery=execution_recovery,
            reconstruction=reconstruction,
            workforce_runtime=deepcopy(core.workforce),
        )
    finally:
        session.close()
