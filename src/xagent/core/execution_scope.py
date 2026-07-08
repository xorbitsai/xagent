"""Execution scope context for scoping sandbox, workspace, and memory.

An :class:`ExecutionScope` is a **cooperative namespace, not a security
boundary**: it partitions sandbox lifecycle keys, workspace/storage paths,
and memory metadata *within* a single platform user. File records, RAG/KB
isolation, and tool credentials remain keyed by the platform ``user_id``
only — a scope must never be relied on to keep one principal's data from
another principal.

Scope fields are consumed **independently** by each subsystem: a consumer
reads exactly the field(s) it needs (``sandbox_key_suffix``,
``workspace_segments``, ``memory_dimensions``, ``strict_memory_isolation``,
``isolate_external_dirs``) and must never gate on "a scope is active" as an
all-or-nothing switch — a scope may set any subset of its fields.

Two activation mechanisms:

1. **Resolver hook** (primary): the embedding application registers a
   resolver via :func:`set_execution_scope_resolver`; the task orchestrator
   enters :func:`turn_execution_scope` at the start of every turn — the same
   place the acting user is resolved — so process restart and task
   resumption re-derive the scope from the embedder's own persistent data
   keyed by ``task_id`` rather than from a long-gone request context.
2. **Contextvar helpers** (secondary): :func:`set_execution_scope` /
   :func:`reset_execution_scope` / :class:`ExecutionScopeContext`, for
   synchronous paths that run inside the request that established the
   scope, mirroring the existing user-context pattern.
"""

from __future__ import annotations

import contextvars
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

_SCOPE_COMPONENT_RE = re.compile(r"[a-zA-Z0-9_-]{1,63}")


class InvalidScopeComponentError(ValueError):
    """A scope component failed validation.

    Raised instead of sanitizing: silently rewriting an invalid component
    could collapse two distinct inputs into one namespace.
    """


def validate_scope_component(value: Any, *, field_name: str = "scope component") -> str:
    """Validate a single scope component against ``[a-zA-Z0-9_-]{1,63}``.

    No ``:``, ``/``, ``..``, whitespace, or empty strings — components are
    embedded verbatim in sandbox lifecycle keys, filesystem paths, and
    storage keys. Invalid input is rejected with a logged error, never
    silently sanitized.

    Args:
        value: The candidate component.
        field_name: Name used in the log/error message.

    Returns:
        ``value`` unchanged, if valid.

    Raises:
        InvalidScopeComponentError: if ``value`` is not a string matching
            ``[a-zA-Z0-9_-]{1,63}``.
    """
    if not isinstance(value, str) or not _SCOPE_COMPONENT_RE.fullmatch(value):
        logger.error(
            "Invalid %s %r: must be a string matching [a-zA-Z0-9_-]{1,63}",
            field_name,
            value,
        )
        raise InvalidScopeComponentError(
            f"invalid {field_name} {value!r}: "
            "must be a string matching [a-zA-Z0-9_-]{1,63}"
        )
    return value


@dataclass(frozen=True)
class ExecutionScope:
    """Immutable execution scope. All fields default to current behavior.

    Attributes:
        sandbox_key_suffix: Appended to the sandbox lifecycle key
            (``user:{owner_id}`` becomes ``user:{owner_id}:{suffix}``).
        workspace_segments: Extra path segments inserted after the user root
            in workspace paths and storage keys.
        sandbox_mount_segments: When set, the sandbox bind-mount root covers
            only this **prefix** of ``workspace_segments`` instead of the full
            tuple. Two scopes that share ``sandbox_key_suffix`` and this prefix
            then produce an identical mount and can share one container, while
            their deeper ``workspace_segments`` keep them in disjoint subtrees
            of that shared mount (isolation stays at the tool/path-check
            layer). Must be a prefix of ``workspace_segments``. ``None`` (the
            default) means the mount covers the full ``workspace_segments`` —
            byte-identical to pre-existing behavior. Consumed only by the
            sandbox-mount composition; workspace paths and storage keys always
            use the full ``workspace_segments``.
        memory_dimensions: Extra metadata stamped on memory notes on add and
            filtered on scoped search.
        strict_memory_isolation: When True, unscoped searches also exclude
            any note carrying scope dimensions (default is one-way
            visibility: scoped searches are isolated, unscoped searches see
            everything under the user). Consumed even when every other field
            is empty.
        isolate_external_dirs: When True, KB/upload external dirs become
            scope-local instead of shared across the user's scopes.
    """

    sandbox_key_suffix: Optional[str] = None
    workspace_segments: tuple[str, ...] = ()
    sandbox_mount_segments: Optional[tuple[str, ...]] = None
    memory_dimensions: Mapping[str, str] = field(default_factory=dict)
    strict_memory_isolation: bool = False
    isolate_external_dirs: bool = False

    @property
    def effective_mount_segments(self) -> tuple[str, ...]:
        """Segments the sandbox bind-mount root covers.

        Defaults to the full ``workspace_segments`` (mount root == workspace
        root), so an unset prefix reproduces today's behavior exactly. When
        ``sandbox_mount_segments`` is set, the mount root covers only that
        prefix and scopes sharing ``sandbox_key_suffix`` + this prefix share
        one container.
        """
        if self.sandbox_mount_segments is None:
            return self.workspace_segments
        return self.sandbox_mount_segments

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot (see :func:`ExecutionScope.from_dict`).

        Used to persist a scope into a task's ``agent_config`` so internally
        created tasks (workforce runs) stay scoped across process restarts
        without the embedder's resolver knowing their task ids.
        """
        return {
            "sandbox_key_suffix": self.sandbox_key_suffix,
            "workspace_segments": list(self.workspace_segments),
            "sandbox_mount_segments": (
                None
                if self.sandbox_mount_segments is None
                else list(self.sandbox_mount_segments)
            ),
            "memory_dimensions": dict(self.memory_dimensions),
            "strict_memory_isolation": self.strict_memory_isolation,
            "isolate_external_dirs": self.isolate_external_dirs,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionScope":
        """Rebuild a scope from :meth:`to_dict` output (re-validated)."""
        raw_mount = data.get("sandbox_mount_segments")
        return cls(
            sandbox_key_suffix=data.get("sandbox_key_suffix"),
            workspace_segments=tuple(data.get("workspace_segments") or ()),
            sandbox_mount_segments=(None if raw_mount is None else tuple(raw_mount)),
            memory_dimensions=dict(data.get("memory_dimensions") or {}),
            strict_memory_isolation=bool(data.get("strict_memory_isolation", False)),
            isolate_external_dirs=bool(data.get("isolate_external_dirs", False)),
        )

    def __post_init__(self) -> None:
        if self.sandbox_key_suffix is not None:
            validate_scope_component(
                self.sandbox_key_suffix, field_name="sandbox_key_suffix"
            )

        if self.workspace_segments is None:
            raise ValueError(
                "workspace_segments cannot be None; pass () for a scope "
                "without workspace segments"
            )
        if self.memory_dimensions is None:
            raise ValueError(
                "memory_dimensions cannot be None; pass {} for a scope "
                "without memory dimensions"
            )

        segments = tuple(self.workspace_segments)
        for segment in segments:
            validate_scope_component(segment, field_name="workspace_segments entry")
        object.__setattr__(self, "workspace_segments", segments)

        if self.sandbox_mount_segments is not None:
            mount_segments = tuple(self.sandbox_mount_segments)
            for segment in mount_segments:
                validate_scope_component(
                    segment, field_name="sandbox_mount_segments entry"
                )
            # The mount root must be a prefix of the workspace root: the
            # workspace subtree (full segments) has to live *inside* the
            # mounted directory to be visible in the container, and a
            # non-prefix mount could expose an unrelated subtree.
            if mount_segments != segments[: len(mount_segments)]:
                logger.error(
                    "sandbox_mount_segments %r is not a prefix of "
                    "workspace_segments %r",
                    mount_segments,
                    segments,
                )
                raise InvalidScopeComponentError(
                    f"sandbox_mount_segments {mount_segments!r} must be a "
                    f"prefix of workspace_segments {segments!r}"
                )
            object.__setattr__(self, "sandbox_mount_segments", mount_segments)

        dimensions = dict(self.memory_dimensions)
        for key, dim_value in dimensions.items():
            validate_scope_component(key, field_name="memory_dimensions key")
            if not isinstance(dim_value, str) or not dim_value:
                logger.error(
                    "Invalid memory_dimensions value %r for key %r: "
                    "must be a non-empty string",
                    dim_value,
                    key,
                )
                raise InvalidScopeComponentError(
                    f"invalid memory_dimensions value {dim_value!r} for key "
                    f"{key!r}: must be a non-empty string"
                )
        object.__setattr__(self, "memory_dimensions", MappingProxyType(dimensions))


# Reserved key under which a task's ``agent_config`` JSON carries a
# persisted scope snapshot (ExecutionScope.to_dict()). Internally created
# tasks (workforce runs) have task ids the embedder's resolver cannot map;
# the snapshot is written at task creation and preferred over the resolver.
EXECUTION_SCOPE_AGENT_CONFIG_KEY = "execution_scope"

# Metadata-key prefix under which ExecutionScope.memory_dimensions are
# stamped onto memory notes (flat, string-valued entries — the memory
# backends apply plain string-equality filters). The prefix keeps dimension
# keys from colliding with system metadata such as ``user_id``.
MEMORY_DIMENSION_METADATA_PREFIX = "execution_scope_"


def memory_dimension_metadata(scope: Optional[ExecutionScope]) -> dict[str, str]:
    """Prefixed metadata entries for a scope's memory dimensions.

    Empty when unscoped or when the scope carries no dimensions — fields
    are consumed independently.
    """
    if scope is None:
        return {}
    return {
        f"{MEMORY_DIMENSION_METADATA_PREFIX}{key}": value
        for key, value in scope.memory_dimensions.items()
    }


def metadata_carries_scope_dimensions(metadata: Mapping[str, Any]) -> bool:
    """True when a note's metadata was stamped with any scope dimension.

    Used by ``strict_memory_isolation`` post-filters to exclude scoped
    notes from unscoped searches.
    """
    return any(key.startswith(MEMORY_DIMENSION_METADATA_PREFIX) for key in metadata)


# Hashable identity of a scope's namespace-affecting fields:
# (sandbox_key_suffix, workspace_segments, sorted memory_dimensions items).
ScopeFingerprint = tuple[Optional[str], tuple[str, ...], tuple[tuple[str, str], ...]]


def scope_fingerprint(scope: Optional[ExecutionScope]) -> Optional[ScopeFingerprint]:
    """Hashable fingerprint of the namespaces a scope selects.

    Per-task caches that bake scope-derived state in at build time (sandbox
    keys, workspace paths, memory dimensions) key their eviction checks on
    this. ``None`` is the sentinel for unscoped, distinct from an empty
    scope's fingerprint.
    """
    if scope is None:
        return None
    return (
        scope.sandbox_key_suffix,
        scope.workspace_segments,
        tuple(sorted(scope.memory_dimensions.items())),
    )


current_execution_scope: contextvars.ContextVar[Optional[ExecutionScope]] = (
    contextvars.ContextVar("current_execution_scope", default=None)
)


def get_execution_scope() -> Optional[ExecutionScope]:
    """Get the execution scope active in the current context, if any."""
    return current_execution_scope.get()


def set_execution_scope(scope: Optional[ExecutionScope]) -> contextvars.Token:
    """Set the current execution scope.

    Args:
        scope: Scope to activate, or None for explicitly-unscoped.

    Returns:
        Context token for :func:`reset_execution_scope`.
    """
    return current_execution_scope.set(scope)


def reset_execution_scope(token: contextvars.Token) -> None:
    """Reset the execution scope to its previous state.

    Args:
        token: Context token from :func:`set_execution_scope`.
    """
    current_execution_scope.reset(token)


class ExecutionScopeContext:
    """Context manager for setting the execution scope."""

    def __init__(self, scope: Optional[ExecutionScope]) -> None:
        self.scope = scope
        self.token: Optional[contextvars.Token] = None

    def __enter__(self) -> "ExecutionScopeContext":
        self.token = set_execution_scope(self.scope)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object],
    ) -> None:
        if self.token is not None:
            reset_execution_scope(self.token)


# The embedding application injects a scope resolver via
# set_execution_scope_resolver() (same injection pattern as
# set_user_tool_overrides_hook in the web layer).
ExecutionScopeResolver = Callable[[str], Optional[ExecutionScope]]

_execution_scope_resolver: Optional[ExecutionScopeResolver] = None


def set_execution_scope_resolver(resolver: Optional[ExecutionScopeResolver]) -> None:
    """Register the resolver that maps a ``task_id`` to its ExecutionScope.

    Resolver contract:

    - **Idempotent per task**: called at the start of every turn of a task
      (including resumed turns after a process restart), it must return an
      equal scope for the same ``task_id`` every time. Reassigning a task to
      a different scope between turns is possible but expensive (per-task
      caches rebuild); a resolver that flaps A -> B -> A is a bug.
    - Return ``None`` for tasks that run unscoped. No registered resolver
      means every task runs unscoped.
    - Scope fields are consumed independently by subsystems; the resolver
      may populate any subset.
    - An exception from the resolver fails the turn: falling back to
      unscoped on error would silently merge namespaces.
    """
    global _execution_scope_resolver
    _execution_scope_resolver = resolver


# Loader for persisted scope snapshots (EXECUTION_SCOPE_AGENT_CONFIG_KEY in
# a task's agent_config). The web layer registers an implementation backed
# by the Task table; None means no snapshot support.
ExecutionScopeSnapshotLoader = Callable[[str], Optional[ExecutionScope]]

_execution_scope_snapshot_loader: Optional[ExecutionScopeSnapshotLoader] = None


def set_execution_scope_snapshot_loader(
    loader: Optional[ExecutionScopeSnapshotLoader],
) -> None:
    """Register the loader for persisted per-task scope snapshots.

    The loader returns the snapshot persisted at task creation, or None for
    tasks without one. A persisted snapshot is preferred over the resolver:
    internally created tasks (workforce runs) have ids the embedder's
    resolver cannot map, and the snapshot is what keeps them scoped across
    process restarts. Loader exceptions fail the turn — falling back to the
    resolver (or unscoped) on error could silently switch namespaces.
    """
    global _execution_scope_snapshot_loader
    _execution_scope_snapshot_loader = loader


def resolve_execution_scope(task_id: str | int) -> Optional[ExecutionScope]:
    """Resolve the scope for ``task_id``.

    A persisted snapshot (see :func:`set_execution_scope_snapshot_loader`)
    is preferred over the resolver; with neither registered, or both
    returning None, the task runs unscoped. Loader/resolver exceptions
    propagate to the caller.

    Raises:
        ValueError: ``task_id`` is None — ``str(None)`` would silently
            query the loader/resolver for the literal string ``"None"``.
            Callers that legitimately have no task identity must treat
            that as unscoped themselves instead of passing None.
    """
    if task_id is None:
        raise ValueError(
            "task_id cannot be None; a caller without a task identity "
            "must treat the execution as unscoped instead"
        )
    if _execution_scope_snapshot_loader is not None:
        snapshot = _execution_scope_snapshot_loader(str(task_id))
        if snapshot is not None:
            return snapshot
    if _execution_scope_resolver is None:
        return None
    return _execution_scope_resolver(str(task_id))


@contextmanager
def turn_execution_scope(task_id: str | int) -> Iterator[Optional[ExecutionScope]]:
    """Resolve and activate the execution scope for one turn of ``task_id``.

    The task orchestrator enters this at the start of every turn, at the
    same place the acting user is resolved, so restart/resume re-derive the
    scope correctly. The scope (or explicit None) is set for the duration of
    the turn and restored on exit.
    """
    scope = resolve_execution_scope(task_id)
    token = set_execution_scope(scope)
    try:
        yield scope
    finally:
        reset_execution_scope(token)
