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

Out of scope (first cut, by design):
    * ``UploadedFile`` selected-files loop -- contains writes
      (``UploadedFile.task_id`` assignment + ``db.flush()``), so it
      stays on the main loop with the request session.
    * ``_load_persisted_conversation_history`` /
      ``_load_persisted_execution_context`` -- already separate async
      helpers; can be migrated in a follow-up.
    * ToolFactory inner DB I/O -- tool subclasses hold ``self._db``;
      threading the factory requires a session-factory refactor of
      every tool subclass first.
    * MCP server configs -- async + OAuth refresh path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.orm import Session

from ...core.model.chat.basic.base import BaseLLM
from ..models.database import get_session_local
from ..models.task import Task
from .llm_utils import AgentRuntimeFields

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
    computer_runtime_kind: str | None = None


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


# NOTE: All LLM resolution + agent-builder merge + execution-mode →
# pattern logic lives in ``llm_utils.resolve_task_runtime_config_core``.
# This loader is the off-loop wrapper that:
#   1. opens its own ``SessionLocal``,
#   2. calls the shared core to do the actual resolution,
#   3. wraps the resulting ORM ``Task`` row in a frozen
#      ``_TaskFields`` so nothing escapes the loader's session
#      (``Agent`` primitives are already provided by the core as an
#      ``AgentRuntimeFields`` instance).
# The main-loop reconstruct path (``_resolve_task_runtime_config`` in
# chat.py) calls the same core directly, since it doesn't need the
# primitive wrapping and runs inside the request session's lifetime.


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


def load_task_setup_snapshot_sync(
    task_id: int,
    task_owner_user_id: Optional[int],
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
        task_row = session.query(Task).filter(Task.id == task_id).first()
        if task_row is None:
            return None

        owner_user_id = int(task_row.user_id)
        if task_owner_user_id is not None and owner_user_id != task_owner_user_id:
            raise TaskOwnerMismatchError(task_id, task_owner_user_id, owner_user_id)

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
            computer_runtime_kind=(
                str(task_row.computer_runtime_kind)
                if task_row.computer_runtime_kind is not None
                else None
            ),
        )

        # Runtime resolution always uses the row's OWNER, never the passed
        # value (which is validated to match above when provided).
        core = resolve_task_runtime_config_core(
            task_row, session, user_id=owner_user_id
        )
        task_llm, task_fast_llm, task_vision_llm, task_compact_llm = core.llms

        # ``core.agent_fields`` is already an ``AgentRuntimeFields``
        # frozen dataclass; pass it through directly.
        return TaskSetupSnapshot(
            task=task_fields,
            task_pattern=core.task_pattern,
            task_llm=task_llm,
            task_fast_llm=task_fast_llm,
            task_vision_llm=task_vision_llm,
            task_compact_llm=task_compact_llm,
            agent=core.agent_fields,
            agent_config=core.agent_config,
            excluded_agent_id=core.excluded_agent_id,
        )
    finally:
        session.close()
