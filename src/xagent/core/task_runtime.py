"""Provider-facing types for task-scoped runtime extensions.

The web layer owns provider registration and lifecycle dispatch.  These types
live in ``core`` so an out-of-tree provider can describe its contribution
without importing ORM models, request objects, or a live database session.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

PREFERRED_INPUT_MODALITIES_METADATA_KEY = "preferred_input_modalities"
# Server-owned marker and workspace-config key for the minimal new-public-task
# File Operation rollout. The namespaced spelling reduces collision risk with
# historical free-form Task.agent_config values.
FILE_OPERATION_ACCESS_VERSION_KEY = "__xagent_file_operation_access_version"
FILE_OPERATION_ACCESS_VERSION = 1
SUPPORTED_FILE_OPERATION_ACCESS_VERSIONS = frozenset({1})
MAX_TASK_RUNTIME_EXTENSIONS = 16
MAX_TASK_RUNTIME_JSON_BYTES = 64 * 1024
MAX_TASK_RUNTIME_ENVIRONMENT_BYTES = 64 * 1024
MAX_TASK_RUNTIME_TOOLS = 64
MAX_TASK_RUNTIME_PUBLIC_METADATA_BYTES = 256 * 1024
# Aggregate cap on one create request's runtime-extension configurations. The
# per-extension MAX_TASK_RUNTIME_JSON_BYTES cap alone leaves the total
# unbounded (MAX_TASK_RUNTIME_EXTENSIONS x 64 KiB is ~1 MiB), so the create
# path bounds the whole payload the way the public-metadata read path bounds
# the whole response.
MAX_TASK_RUNTIME_REQUEST_BYTES = 256 * 1024


class FileOperationAccessPolicyError(RuntimeError):
    """A marked task cannot prove its exact File Operation authority."""


def requires_exact_file_operation_scope(task: Any) -> bool:
    """Return whether one persisted task opts into exact File Operation scope.

    Marker absence deliberately preserves private and historical behavior for
    the focused #803 rollout. Once a marker exists, every field is strict: an
    unknown version or inconsistent public identity fails closed instead of
    falling back to creator-wide access.
    """

    config = getattr(task, "agent_config", None)
    if not isinstance(config, Mapping):
        return False
    marker = config.get(FILE_OPERATION_ACCESS_VERSION_KEY)
    if marker is None:
        return False
    if (
        isinstance(marker, bool)
        or not isinstance(marker, int)
        or marker not in SUPPORTED_FILE_OPERATION_ACCESS_VERSIONS
    ):
        raise FileOperationAccessPolicyError(
            "File Operation access policy version is unsupported"
        )

    task_id = getattr(task, "id", None)
    owner_user_id = getattr(task, "user_id", None)
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or isinstance(owner_user_id, bool)
        or not isinstance(owner_user_id, int)
        or owner_user_id <= 0
    ):
        raise FileOperationAccessPolicyError(
            "Marked File Operation task has no authoritative identity"
        )

    source = getattr(task, "source", None)
    auth_mode = config.get("auth_mode")
    expected_source = "shared_link" if auth_mode == "share" else "widget"
    if auth_mode not in {"share", "widget"} or source != expected_source:
        raise FileOperationAccessPolicyError(
            "Marked File Operation task has inconsistent public identity"
        )
    return True


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
    a routing preference, not a hard model requirement: a router that cannot
    honour it degrades by routing without it. Only modalities the conversation
    itself carries are enforced as hard requirements.

    ``tools``, ``environment`` and ``preferred_input_modalities`` are the only
    provider-owned fields. The remaining three are registry-internal
    bookkeeping: a provider *may* set them because ``frozen=True`` only blocks
    post-init mutation, but ``normalize_task_runtime_contribution`` strips them
    from every provider-returned value, so setting them has no effect and
    cannot be used to misattribute tools to another provider.
    """

    tools: tuple[Any, ...] = ()
    environment: str | None = None
    preferred_input_modalities: tuple[str, ...] = ()
    # Populated by the registry merge for filtering diagnostics. Registry
    # bookkeeping: a provider-set value is discarded by normalization.
    tool_origins: tuple[tuple[str, str], ...] = ()
    # Detached per-provider contributions retained by the registry merge so
    # policy filtering can remove a provider's prompt context when none of its
    # tools survive. Registry bookkeeping: a provider-set value is discarded
    # by normalization.
    provider_contributions: tuple[tuple[str, "TaskRuntimeContribution"], ...] = ()
    # Registry-internal back-reference: on a policy-narrowed view this is the
    # full pre-policy contribution the view was derived from. Tool policy can
    # widen again on a later turn, so every rebuild must re-derive from the
    # full contribution instead of re-narrowing an already-narrowed value,
    # which would lose filtered tools permanently. Excluded from equality and
    # repr so a narrowed view still compares as the plain contribution it is.
    # Registry bookkeeping: a provider-set value is discarded by normalization.
    source_contribution: "TaskRuntimeContribution | None" = field(
        default=None, compare=False, repr=False
    )


EMPTY_TASK_RUNTIME_CONTRIBUTION = TaskRuntimeContribution()


@dataclass(frozen=True)
class TaskRuntimeToolConflict:
    """One provider dropped because its surviving tools collide by name."""

    provider: str
    tool_names: tuple[str, ...]


class TaskRuntimeExtensionProvider(Protocol):
    """Lifecycle contract implemented by an out-of-tree task extension.

    ``on_task_deleted`` must be idempotent. A provider failure preserves the
    core task so the caller can retry; after provider cleanup succeeds, core
    deletion can still fail and a retry will dispatch the hook again. Providers
    that release an external lease or sandbox should persist a provider-side
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
    # ``tool_origins``, ``provider_contributions`` and ``source_contribution``
    # are registry-internal bookkeeping. ``frozen=True`` only blocks post-init
    # mutation, so nothing at the type level stops an out-of-tree provider from
    # passing fabricated entries to the constructor -- entries that would
    # misattribute its tools to another provider and mislead the attribution
    # guard in ``ToolFactory._create_all_tools_prepared``. Normalization is the
    # trust boundary: they are dropped here unconditionally and re-derived from
    # the tools the provider actually handed over by
    # ``merge_task_runtime_contributions``.
    return TaskRuntimeContribution(
        tools=tools,
        environment=environment,
        preferred_input_modalities=modalities,
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


def full_task_runtime_contribution(
    contribution: TaskRuntimeContribution,
) -> TaskRuntimeContribution:
    """Return the pre-policy contribution a possibly narrowed view came from.

    Tool policy can widen again between turns, so callers that rebuild tools
    must start from the full contribution. Re-narrowing an already-narrowed
    value would make every policy filter permanent for the lifetime of the
    cached configuration.
    """

    source = contribution.source_contribution
    return source if isinstance(source, TaskRuntimeContribution) else contribution


def reconcile_task_runtime_contribution_tools(
    contribution: TaskRuntimeContribution,
    *,
    available_tools: Sequence[Any],
    reserved_tool_names: set[str] | None = None,
) -> tuple[TaskRuntimeContribution, tuple[TaskRuntimeToolConflict, ...]]:
    """Apply tool policy and isolate providers whose surviving names collide.

    ``available_tools`` is the accepted tool *occurrences* that survived the
    task's selection and user policy, matched back to their owning provider by
    object identity. Identity matching — rather than matching by name — is
    required because a tool the caller already rejected (for example a
    contribution without a usable tool category) may share its ``name`` with a
    different provider's accepted tool; name matching would then let the
    rejected object claim the name and evict the tool that actually survived.

    ``reserved_tool_names`` contains core tool names that survived the same
    policy. Providers are considered in registry order; a provider with any
    collision is removed as one unit so its prompt environment and modality
    preference cannot describe tools that were discarded. A provider left with
    no surviving tool is removed for the same reason.

    ``contribution`` may itself be a previously narrowed view; reconciliation
    always re-derives from its full pre-policy source so a policy that widens
    again restores the tools, prompt text and provider entries it had removed.
    The returned view carries that full contribution forward.
    """

    full_contribution = full_task_runtime_contribution(contribution)

    if not full_contribution.provider_contributions:
        return full_contribution, ()

    # Occurrence counts, not a bare identity set: two providers may legitimately
    # contribute the same tool object, and each accepted occurrence belongs to
    # exactly one of them.
    remaining_available_occurrences = Counter(id(tool) for tool in available_tools)

    retained: dict[str, TaskRuntimeContribution] = {}
    conflicts: list[TaskRuntimeToolConflict] = []
    claimed_names = set(reserved_tool_names or ())
    for (
        provider_name,
        provider_contribution,
    ) in full_contribution.provider_contributions:
        provider_tools = tuple(provider_contribution.tools)
        if not provider_tools:
            retained[provider_name] = provider_contribution
            continue
        accepted: list[Any] = []
        for tool in provider_tools:
            if remaining_available_occurrences[id(tool)] > 0:
                remaining_available_occurrences[id(tool)] -= 1
                accepted.append(tool)
        surviving_tools = tuple(accepted)
        if not surviving_tools:
            continue

        surviving_names = tuple(str(tool.name).strip() for tool in surviving_tools)
        seen_names: set[str] = set()
        duplicate_names: set[str] = set()
        for name in surviving_names:
            if name in seen_names:
                duplicate_names.add(name)
            seen_names.add(name)
        conflicting_names = duplicate_names | (set(surviving_names) & claimed_names)
        if conflicting_names:
            conflicts.append(
                TaskRuntimeToolConflict(
                    provider=provider_name,
                    tool_names=tuple(sorted(conflicting_names)),
                )
            )
            continue

        retained[provider_name] = replace(
            provider_contribution,
            tools=surviving_tools,
            tool_origins=(),
            provider_contributions=(),
        )
        claimed_names.update(surviving_names)

    reconciled = (
        merge_task_runtime_contributions(retained)
        if retained
        else EMPTY_TASK_RUNTIME_CONTRIBUTION
    )
    # Keep the full contribution reachable from the narrowed view so a later,
    # more permissive policy can restore what this pass filtered out.
    return replace(reconciled, source_contribution=full_contribution), tuple(conflicts)
