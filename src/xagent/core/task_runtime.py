"""Provider-facing types for task-scoped runtime extensions.

The web layer owns provider registration and lifecycle dispatch.  These types
live in ``core`` so an out-of-tree provider can describe its contribution
without importing ORM models, request objects, or a live database session.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

PREFERRED_INPUT_MODALITIES_METADATA_KEY = "preferred_input_modalities"


@dataclass(frozen=True)
class TaskRuntimeContext:
    """Detached inputs shared with one task runtime extension.

    ``session_factory`` is intentionally a factory instead of a checked-out
    SQLAlchemy ``Session``.  Providers must open short, operation-local
    sessions so a task-bound runtime never retains a request session across
    tool calls or async boundaries.
    """

    task_id: int
    user_id: int
    source: str | None
    session_factory: Callable[[], Any]
    workspace: Any | None = None

    def with_workspace(self, workspace: Any | None) -> "TaskRuntimeContext":
        """Return a build context bound to the task workspace."""

        return replace(self, workspace=workspace)


@dataclass(frozen=True)
class TaskRuntimeContribution:
    """Resources contributed to an ``AgentService`` for one task.

    ``environment`` is non-secret system context describing the selected
    resource and how the agent should use it. ``preferred_input_modalities`` is
    a routing preference, not a hard model requirement.
    """

    tools: tuple[Any, ...] = ()
    environment: str | None = None
    preferred_input_modalities: tuple[str, ...] = ()


EMPTY_TASK_RUNTIME_CONTRIBUTION = TaskRuntimeContribution()


class TaskRuntimeExtensionProvider(Protocol):
    """Lifecycle contract implemented by an out-of-tree task extension."""

    def on_task_created(
        self,
        context: TaskRuntimeContext,
        configuration: Mapping[str, Any],
    ) -> Awaitable[None] | None: ...

    def build_runtime(
        self,
        context: TaskRuntimeContext,
    ) -> TaskRuntimeContribution | None | Awaitable[TaskRuntimeContribution | None]: ...

    def public_metadata(
        self,
        context: TaskRuntimeContext,
    ) -> Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None]: ...

    def on_task_deleted(
        self,
        context: TaskRuntimeContext,
    ) -> Awaitable[None] | None: ...


def normalize_input_modalities(values: Any) -> tuple[str, ...]:
    """Normalize a provider or execution metadata modality sequence."""

    if isinstance(values, (str, bytes)) or values is None:
        values = (values,) if values else ()
    try:
        candidates = tuple(values)
    except TypeError:
        candidates = (values,)
    return tuple(
        dict.fromkeys(
            normalized
            for item in candidates
            if item is not None and (normalized := str(item).strip().lower())
        )
    )


def normalize_task_runtime_contribution(
    value: TaskRuntimeContribution | None,
) -> TaskRuntimeContribution:
    """Validate and detach one provider contribution."""

    if value is None:
        return EMPTY_TASK_RUNTIME_CONTRIBUTION
    if not isinstance(value, TaskRuntimeContribution):
        raise TypeError("build_runtime must return TaskRuntimeContribution or None")

    environment = (
        value.environment.strip()
        if isinstance(value.environment, str) and value.environment.strip()
        else None
    )
    modalities = normalize_input_modalities(value.preferred_input_modalities)
    return TaskRuntimeContribution(
        tools=tuple(value.tools),
        environment=environment,
        preferred_input_modalities=modalities,
    )


def merge_task_runtime_contributions(
    contributions: Mapping[str, TaskRuntimeContribution],
) -> TaskRuntimeContribution:
    """Merge provider contributions in registry order."""

    tools: list[Any] = []
    environments: list[str] = []
    modalities: list[str] = []
    for contribution in contributions.values():
        normalized = normalize_task_runtime_contribution(contribution)
        tools.extend(normalized.tools)
        if normalized.environment:
            environments.append(normalized.environment)
        modalities.extend(normalized.preferred_input_modalities)

    return TaskRuntimeContribution(
        tools=tuple(tools),
        environment="\n\n".join(environments) or None,
        preferred_input_modalities=tuple(dict.fromkeys(modalities)),
    )
