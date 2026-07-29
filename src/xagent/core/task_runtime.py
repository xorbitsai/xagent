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
MAX_TASK_RUNTIME_EXTENSIONS = 16
MAX_TASK_RUNTIME_JSON_BYTES = 64 * 1024
MAX_TASK_RUNTIME_ENVIRONMENT_BYTES = 64 * 1024
MAX_TASK_RUNTIME_TOOLS = 64
MAX_TASK_RUNTIME_PUBLIC_METADATA_BYTES = 256 * 1024
TASK_RUNTIME_PUBLIC_METADATA_STATUS_KEY = "_runtime"


class TaskRuntimeClientError(Exception):
    """Provider-approved error detail that may be returned to a client.

    Provider implementations should raise this only for expected request or
    authorization failures. Unexpected provider exceptions remain private and
    are mapped to generic server errors by the web layer.
    """

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        normalized_detail = detail.strip()
        if not normalized_detail:
            raise ValueError("Task runtime client error detail must not be empty")
        if status_code not in (400, 403):
            raise ValueError("Task runtime client error status must be 400 or 403")
        super().__init__(normalized_detail)
        self.detail = normalized_detail
        self.status_code = status_code


@dataclass(frozen=True)
class TaskRuntimeContext:
    """Detached inputs shared with one task runtime extension.

    ``user_id`` is always the task owner's stable identity, never the acting
    admin or runtime caller. ``session_factory`` is intentionally a factory
    instead of a checked-out SQLAlchemy ``Session``. Providers own every
    session they open and must close it within the current hook; the framework
    deliberately does not retain or instrument provider sessions.
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
    # Populated by the registry merge for filtering diagnostics. Providers do
    # not need to set this field themselves.
    tool_origins: tuple[tuple[str, str], ...] = ()
    # Detached per-provider contributions retained by the registry merge so
    # policy filtering can remove a provider's prompt context when none of its
    # tools survive. Providers do not set this field themselves.
    provider_contributions: tuple[tuple[str, "TaskRuntimeContribution"], ...] = ()


EMPTY_TASK_RUNTIME_CONTRIBUTION = TaskRuntimeContribution()


class TaskRuntimeExtensionProvider(Protocol):
    """Lifecycle contract implemented by an out-of-tree task extension.

    ``on_task_deleted`` must be idempotent: core deletion can fail after
    provider cleanup, and a retry will dispatch the hook again. Providers that
    release an external lease or sandbox should persist a provider-side
    "release requested" state and reconcile it safely on repeated calls instead
    of treating the first release attempt as an irreversible one-shot action.
    """

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

    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    elif isinstance(values, (bytes, bytearray, Mapping)):
        raise TypeError("Input modalities must be a string sequence")
    try:
        candidates = tuple(values)
    except TypeError as exc:
        raise TypeError("Input modalities must be a string sequence") from exc

    normalized_values: list[str] = []
    for item in candidates:
        if item is None:
            continue
        if not isinstance(item, str):
            raise TypeError("Input modality items must be strings")
        normalized = item.strip().lower()
        if normalized:
            normalized_values.append(normalized)
    return tuple(dict.fromkeys(normalized_values))


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
    if environment is not None:
        environment_bytes = len(environment.encode("utf-8"))
        if environment_bytes > MAX_TASK_RUNTIME_ENVIRONMENT_BYTES:
            raise ValueError(
                "Task runtime environment exceeds the "
                f"{MAX_TASK_RUNTIME_ENVIRONMENT_BYTES}-byte limit"
            )
    tools = tuple(value.tools)
    if len(tools) > MAX_TASK_RUNTIME_TOOLS:
        raise ValueError(
            f"Task runtime contribution exceeds the {MAX_TASK_RUNTIME_TOOLS}-tool limit"
        )
    modalities = normalize_input_modalities(value.preferred_input_modalities)
    return TaskRuntimeContribution(
        tools=tools,
        environment=environment,
        preferred_input_modalities=modalities,
        tool_origins=tuple(value.tool_origins),
    )


def merge_task_runtime_contributions(
    contributions: Mapping[str, TaskRuntimeContribution | None],
) -> TaskRuntimeContribution:
    """Merge provider contributions in registry order."""

    tools: list[Any] = []
    environments: list[str] = []
    modalities: list[str] = []
    tool_origins: list[tuple[str, str]] = []
    provider_contributions: list[tuple[str, TaskRuntimeContribution]] = []
    for provider_name, contribution in contributions.items():
        normalized = normalize_task_runtime_contribution(contribution)
        provider_contributions.append((provider_name, normalized))
        tools.extend(normalized.tools)
        if len(tools) > MAX_TASK_RUNTIME_TOOLS:
            raise ValueError(
                "Merged task runtime contributions exceed the "
                f"{MAX_TASK_RUNTIME_TOOLS}-tool limit"
            )
        tool_origins.extend(
            (name.strip(), provider_name)
            for tool in normalized.tools
            if isinstance((name := getattr(tool, "name", None)), str) and name.strip()
        )
        if normalized.environment:
            environments.append(normalized.environment)
        modalities.extend(normalized.preferred_input_modalities)

    environment = "\n\n".join(environments) or None
    if (
        environment is not None
        and len(environment.encode("utf-8")) > MAX_TASK_RUNTIME_ENVIRONMENT_BYTES
    ):
        raise ValueError(
            "Merged task runtime environments exceed the "
            f"{MAX_TASK_RUNTIME_ENVIRONMENT_BYTES}-byte limit"
        )

    return TaskRuntimeContribution(
        tools=tuple(tools),
        environment=environment,
        preferred_input_modalities=tuple(dict.fromkeys(modalities)),
        tool_origins=tuple(tool_origins),
        provider_contributions=tuple(provider_contributions),
    )


def filter_task_runtime_contribution_tools(
    contribution: TaskRuntimeContribution,
    available_tool_names: set[str],
) -> TaskRuntimeContribution:
    """Reconcile provider context with runtime tools that survived policy.

    Providers that contribute no tools retain their environment and modality
    preferences. When a provider does contribute tools, its entire contribution
    is removed only if none survive; otherwise its prompt context is retained
    and its tool list is narrowed to the surviving names.
    """

    if not contribution.provider_contributions:
        return contribution

    retained: dict[str, TaskRuntimeContribution] = {}
    for provider_name, provider_contribution in contribution.provider_contributions:
        provider_tools = tuple(provider_contribution.tools)
        if not provider_tools:
            retained[provider_name] = provider_contribution
            continue
        surviving_tools = tuple(
            tool
            for tool in provider_tools
            if isinstance((name := getattr(tool, "name", None)), str)
            and name in available_tool_names
        )
        if surviving_tools:
            retained[provider_name] = replace(
                provider_contribution,
                tools=surviving_tools,
                tool_origins=(),
                provider_contributions=(),
            )

    return (
        merge_task_runtime_contributions(retained)
        if retained
        else EMPTY_TASK_RUNTIME_CONTRIBUTION
    )
