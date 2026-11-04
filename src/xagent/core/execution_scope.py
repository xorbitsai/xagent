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
   resolver via :func:`set_execution_scope_resolver`; callers resolve the
   scope with :func:`resolve_execution_scope` and activate it via
   :class:`ExecutionScopeContext` at the start of every turn — the same
   place the acting user is resolved — so process restart and task
   resumption re-derive the scope from the embedder's own persistent data
   keyed by ``task_id`` rather than from a long-gone request context.
2. **Contextvar helpers** (secondary): :func:`set_execution_scope` /
   :func:`reset_execution_scope` / :class:`ExecutionScopeContext`, for
   synchronous paths that run inside the request that established the
   scope, mirroring the existing user-context pattern.

Precedence contract, adjudicated in exactly one place
(:func:`resolve_execution_scope`), between the resolver and a snapshot
persisted in ``Task.agent_config`` (see
:data:`EXECUTION_SCOPE_AGENT_CONFIG_KEY`): with a resolver registered, the
resolver is authoritative and the snapshot is at most a corroborating
candidate — a namespace-affecting disagreement fails the turn
(:class:`ExecutionScopeAuthorityError`) rather than letting the
client-influenceable snapshot silently win. A resolver that does not own a
task (e.g. a workforce/delegated sub-task) abstains via a
:class:`DeferToSnapshot` carrier, which gives the snapshot a bounded say: by
default only as a *narrowing* of the resolver's own ``fallback``, or,
opted into via ``snapshot_defines_namespace=True``, as the namespace
authority outright. Registering a non-``None`` resolver requires passing
``acknowledges_snapshot_candidate_contract=True`` to
:func:`set_execution_scope_resolver` — a one-time acknowledgment that the
caller has read this precedence contract. With **no resolver registered at
all**, the persisted snapshot is the sole answer and is returned exactly
as loaded, with none of the checks below applied to it.

Snapshots carry a shape tag, :data:`EXECUTION_SCOPE_SHAPE_VERSION`
(``ExecutionScope.version``), and the gate reading it is **asymmetric**. A
snapshot built under an *older* shape — whose absent fields would otherwise
look like a deliberate disagreement — is only refused where a field-by-field
comparison actually happens: the resolver-authoritative branch and the
default (non-opted-in) resolver-abstention branch. The opted-in
(``snapshot_defines_namespace=True``) abstention branch compares nothing and
uses such a snapshot verbatim. A snapshot stamped *newer* than this process,
by contrast, is refused on **every** resolver-registered branch including
that opt-in: ``from_dict`` drops keys the current shape does not know, so a
newer snapshot arrives partially decoded and may have lost a
namespace-narrowing field. The no-resolver branch described above applies
neither direction — it returns the snapshot exactly as loaded.
"""

from __future__ import annotations

import contextvars
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast

logger = logging.getLogger(__name__)

_SCOPE_COMPONENT_RE = re.compile(r"[a-zA-Z0-9_-]{1,63}")


class ExecutionScopeNotProvided:
    """Marker distinguishing an omitted scope from an explicit ``None``."""

    __slots__ = ()


EXECUTION_SCOPE_NOT_PROVIDED = ExecutionScopeNotProvided()

ExecutionScopeInput = Union[
    "ExecutionScope",
    None,
    ExecutionScopeNotProvided,
]

# What a caller may hand :func:`resolve_execution_scope` in place of letting
# the registered loader read the persisted snapshot: the task's raw
# ``agent_config`` mapping, NOT an already-decoded scope. Decoding here rather
# than at the caller is what keeps each resolution branch's tolerance for a
# malformed snapshot correct by construction -- see
# :func:`_decode_execution_scope_snapshot`.
ExecutionScopeSnapshotSource = Union[
    Mapping[str, Any],
    None,
    ExecutionScopeNotProvided,
]


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


def _coerce_scope_sequence(value: Any, *, field_name: str) -> tuple[Any, ...]:
    """Coerce a raw ``from_dict`` sequence field, never silently misreading it.

    A falsy value (``None``, ``""``, ``[]``, ``0``, ``False``) means "absent"
    and becomes ``()``. Anything else must already be a ``list``/``tuple``:
    ``tuple(value)`` on a bare string or mapping does not raise, it silently
    iterates characters/keys into single-character "segments" that then pass
    per-segment validation, and on a non-iterable it raises a bare
    ``TypeError`` outside this module's error taxonomy. Both failure modes are
    closed here instead of deferring to ``tuple()``.

    Raises:
        InvalidScopeComponentError: ``value`` is truthy and not a list/tuple.
    """
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    logger.error("Invalid %s %r: must be a list or tuple", field_name, value)
    raise InvalidScopeComponentError(
        f"invalid {field_name} {value!r}: must be a list or tuple"
    )


def _coerce_scope_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Coerce a raw ``from_dict`` mapping field, never silently misreading it.

    Mirrors :func:`_coerce_scope_sequence` for the one mapping-shaped field
    (``memory_dimensions``): a falsy value means "absent" and becomes ``{}``;
    anything else must already be a ``Mapping``, since ``dict(value)`` on an
    iterable of 2-character strings (e.g. ``["ab"]``) silently reinterprets
    each as a key/value pair instead of raising.

    Raises:
        InvalidScopeComponentError: ``value`` is truthy and not a ``Mapping``.
    """
    if not value:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    logger.error("Invalid %s %r: must be a mapping", field_name, value)
    raise InvalidScopeComponentError(
        f"invalid {field_name} {value!r}: must be a mapping"
    )


def _coerce_snapshot_version(raw_version: Any) -> int:
    """Read ``from_dict``'s ``version`` field, or reject it as malformed.

    ``version`` reaches here from a snapshot that remains untrusted input: a
    runtime-extension provider holding a ``session_factory``
    (:data:`xagent.core.task_runtime.TaskRuntimeContext.session_factory`) can
    write ``Task.agent_config`` (see :data:`EXECUTION_SCOPE_AGENT_CONFIG_KEY`)
    directly, and a row persisted before ``execution_scope`` was reserved at
    the request boundary (``CLIENT_RESERVED_AGENT_CONFIG_KEYS``,
    ``xagent.web.services.task_runtime``) still carries whatever a request
    seeded at the time (issue #1135). One value is the pre-versioning shape
    sentinel: ``0``, what an absent key decodes to. It says nothing about who
    wrote the snapshot -- omitting the field produces the same value, so it is
    a statement about shape, not provenance. It still decides handling -- the
    *older* half of
    ``_validate_execution_scope_snapshot_candidate``'s asymmetric gate is
    relaxed on the ``snapshot_defines_namespace=True`` branch, where such a
    snapshot's namespace fields are then used verbatim -- so a value that
    cannot be read as a shape version must not be folded into it. An
    unreadable version means the persisted snapshot is malformed, which this
    module classifies as :class:`ExecutionScopeResolverContractError`:
    deliberately outside ``(ValueError, KeyError, TypeError)``, so the
    websocket handlers catching that tuple cannot report a persisted-data
    fault as a malformed client message.

    Readable, and nothing else:

    - a missing key or ``None`` -> ``0``, the pre-versioning marker;
    - a string of digits -> its ``int`` value, the shape a JSON-decoded
      ``version`` can arrive as (``str.isdigit`` admits no sign, so this
      never yields a negative);
    - a non-negative ``int`` -> itself, ``bool`` excluded: ``True`` is an
      ``int`` subclass but no writer ever stamps a boolean version.

    Everything else raises, including two values a lenient read would have
    turned into plausible-looking version claims: a ``float``, which a bare
    ``int()`` truncates (``1.5`` -> ``1``, the current shape), and a negative
    ``int``, which no writer emits -- ``to_dict`` always stamps
    :data:`EXECUTION_SCOPE_SHAPE_VERSION` -- and which "older than
    everything" would hand the pre-versioning marker's verbatim-namespace
    trust. The returned version is therefore always ``>= 0``, which is what
    makes the downstream gate's two comparisons total.

    Raises:
        ExecutionScopeResolverContractError: ``raw_version`` is present and
            is not a readable shape version.
    """
    if raw_version is None:
        return 0
    if isinstance(raw_version, str) and raw_version.isdigit():
        return int(raw_version)
    if (
        isinstance(raw_version, int)
        and not isinstance(raw_version, bool)
        and raw_version >= 0
    ):
        return raw_version
    # The value stays in this server-side log line and out of the raised
    # message: it comes from a client-influenceable snapshot and that message
    # travels to the task's error column and a client-facing event.
    logger.error(
        "Persisted execution scope snapshot carries an unreadable shape version %r",
        raw_version,
    )
    raise ExecutionScopeResolverContractError(
        "persisted execution scope snapshot carries an unreadable shape "
        f"version of type {type(raw_version).__name__}; see the logged value"
    )


# Shape version stamped by ``to_dict`` / read back by ``from_dict``. Bump
# whenever a namespace-affecting field is added so ``resolve_execution_scope``
# can tell a snapshot written against a different shape (``version`` missing,
# lower, or -- during a mixed-version rollout -- higher than this constant)
# from one written against the current shape.
# ``from_dict`` cannot distinguish "field absent because pre-dates it" from
# "field explicitly at its default" any other way -- it fills both the same.
EXECUTION_SCOPE_SHAPE_VERSION = 1


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
            their deeper ``workspace_segments`` place them in distinct subtrees
            of that shared mount. **Security note:** those subtrees are *not*
            an isolation boundary. The mount is read-write and the
            code-execution tools (shell/python executors) run directly in the
            sandbox with no ``scoped_user_root`` path check, so code in one
            scope's task can read and write a co-mounted sibling's subtree.
            Only the orchestrator-side file/workspace API enforces
            ``scoped_user_root``. Therefore this field must only group scopes
            that already share one **runtime trust domain**; never use it to
            co-mount scopes across distinct runtime trust domains. Must be a prefix
            of ``workspace_segments``. ``None`` (the default) means the mount
            covers the full ``workspace_segments`` — byte-identical to
            pre-existing behavior. Consumed only by the sandbox-mount
            composition; workspace paths and storage keys always use the full
            ``workspace_segments``.
        memory_dimensions: Extra metadata stamped on memory notes on add and
            filtered on scoped search.
        strict_memory_isolation: When True, unscoped searches also exclude
            any note carrying scope dimensions (default is one-way
            visibility: scoped searches are isolated, unscoped searches see
            everything under the user). Consumed even when every other field
            is empty.
        isolate_external_dirs: When True, KB/upload external dirs become
            scope-local instead of shared across the user's scopes.
        version: Shape version this instance was constructed against
            (see :data:`EXECUTION_SCOPE_SHAPE_VERSION`). Excluded from
            equality (``compare=False``): it is bookkeeping for
            :func:`resolve_execution_scope`'s snapshot-vs-resolver
            comparison, not a namespace-affecting field, and two otherwise
            identical scopes built at different times must still compare
            equal. Not intended for callers to set explicitly.
    """

    sandbox_key_suffix: Optional[str] = None
    workspace_segments: tuple[str, ...] = ()
    sandbox_mount_segments: Optional[tuple[str, ...]] = None
    memory_dimensions: Mapping[str, str] = field(default_factory=dict)
    strict_memory_isolation: bool = False
    isolate_external_dirs: bool = False
    version: int = field(default=EXECUTION_SCOPE_SHAPE_VERSION, compare=False)

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

    @property
    def durable_storage_segments(self) -> tuple[str, ...]:
        """Segments a durable-storage handle should confine to.

        Mirrors the filesystem external-dir allowlist
        (``_build_allowed_external_dirs``): a ``ScopedFileStorage`` handle is
        narrowed to the scope subtree only when ``isolate_external_dirs`` is
        set, so a scoped-but-not-isolated execution keeps its legitimate
        shared owner-level reads. When the flag is off this returns ``()`` — a
        handle bound to ``users/{owner}`` exactly as before.
        """
        return self.workspace_segments if self.isolate_external_dirs else ()

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
            # Always the current constant, never ``self.version``: this
            # dict is being built *now*, in the current shape, regardless of
            # whether ``self`` was itself decoded from an older persisted
            # snapshot (which would carry a stale/low ``self.version``).
            # Propagating that stale value here would let a
            # decoded-then-re-persisted scope masquerade as pre-dating
            # fields it actually has, permanently ignoring it as a
            # candidate in resolve_execution_scope.
            "version": EXECUTION_SCOPE_SHAPE_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionScope":
        """Rebuild a scope from :meth:`to_dict` output (re-validated).

        ``data`` reaches here from a client-influenceable source (a
        snapshot embedded in ``Task.agent_config``, see
        :data:`EXECUTION_SCOPE_AGENT_CONFIG_KEY`), so every field is coerced
        defensively rather than trusted to already have the shape
        ``to_dict`` produces: ``sandbox_key_suffix`` and the segment/mapping
        fields are re-validated by the constructor
        (:meth:`__post_init__`) or by :func:`_coerce_scope_sequence` /
        :func:`_coerce_scope_mapping` before reaching it, and ``version`` is
        coerced by :func:`_coerce_snapshot_version` -- never a raw
        ``int()``/``tuple()``/``dict()`` call that could raise a bare
        stdlib error outside this module's error taxonomy or silently
        misread/truncate the value.

        ``version`` defaults to ``0`` (not
        :data:`EXECUTION_SCOPE_SHAPE_VERSION`) when the key is absent: a
        snapshot persisted before this field existed must decode as
        distinguishably older rather than silently passing as current-shape.
        A version that is present but unreadable is a malformed snapshot
        rather than an old one, and raises instead of borrowing that
        marker's trust.

        Raises:
            InvalidScopeComponentError: a segment/mapping field, or
                ``sandbox_key_suffix``, fails validation.
            ExecutionScopeResolverContractError: ``version`` is present and
                unreadable (see :func:`_coerce_snapshot_version`).
        """
        raw_mount = data.get("sandbox_mount_segments")
        return cls(
            sandbox_key_suffix=data.get("sandbox_key_suffix"),
            workspace_segments=_coerce_scope_sequence(
                data.get("workspace_segments"), field_name="workspace_segments"
            ),
            sandbox_mount_segments=(
                None
                if raw_mount is None
                else _coerce_scope_sequence(
                    raw_mount, field_name="sandbox_mount_segments"
                )
            ),
            memory_dimensions=_coerce_scope_mapping(
                data.get("memory_dimensions"), field_name="memory_dimensions"
            ),
            strict_memory_isolation=bool(data.get("strict_memory_isolation", False)),
            isolate_external_dirs=bool(data.get("isolate_external_dirs", False)),
            version=_coerce_snapshot_version(data.get("version")),
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
# the snapshot is written at task creation and read back as a corroborating
# candidate when a resolver is registered (see resolve_execution_scope), or
# as the sole answer when no resolver is registered.
EXECUTION_SCOPE_AGENT_CONFIG_KEY = "execution_scope"


def execution_scope_from_agent_config(
    agent_config: Any,
) -> Optional[ExecutionScope]:
    """Decode the persisted scope snapshot owned by a task config.

    ``None`` means the task has no persisted snapshot. The canonical
    resolution flow (:func:`resolve_execution_scope`) always calls a
    registered resolver first; this snapshot is only a corroborating
    candidate for the resolver's answer, or the sole answer when no
    resolver is registered -- it is never consulted ahead of the resolver.
    Invalid snapshots propagate instead of silently degrading to an
    unscoped namespace.

    Every decode failure leaves here as
    :class:`ExecutionScopeResolverContractError`, never as the
    :class:`InvalidScopeComponentError` (a ``ValueError``) the field coercions
    raise: the field coercions' error is re-raised as that class below, and
    the version reader (:func:`_coerce_snapshot_version`) already raises it
    directly. This function is the boundary where *persisted* data is read: the
    snapshot loaders registered by the web layer call it from inside
    :func:`resolve_execution_scope`, which several websocket handlers wrap in
    ``except (ValueError, KeyError, TypeError)`` clauses that answer the
    client with a message-format validation error. A snapshot that no longer
    decodes is a persisted-data/contract fault, not a malformed client
    message, and must not be reported as one. The original error is chained
    as ``__cause__`` and its field-level detail is already logged by the
    coercion that raised it, so nothing diagnosable is lost -- while the
    message here names no snapshot value, since it can carry end-user
    identifiers and travels further than the log does.

    Raises:
        ExecutionScopeResolverContractError: the persisted snapshot is
            present but fails field validation, or carries an unreadable
            shape version.
    """

    if not isinstance(agent_config, Mapping):
        return None
    if EXECUTION_SCOPE_AGENT_CONFIG_KEY not in agent_config:
        return None
    scope_data = agent_config.get(EXECUTION_SCOPE_AGENT_CONFIG_KEY)
    if scope_data is None:
        return None
    if not isinstance(scope_data, Mapping):
        # Present but the wrong shape is corrupt, not absent. Reporting "no
        # candidate" here would make the same silent substitution the
        # invalid-field case below already refuses to make: on the branches
        # where the snapshot is the only namespace authority, "absent" resolves
        # to unscoped, or to the abstention's fallback -- which is the widest
        # value the narrowing check would ever admit.
        logger.error(
            "Persisted execution scope snapshot for a task is a %s, not a "
            "mapping; treating it as corrupt rather than absent",
            type(scope_data).__name__,
        )
        raise ExecutionScopeResolverContractError(
            "persisted execution scope snapshot is not a mapping "
            f"({type(scope_data).__name__}); it cannot be decoded"
        )
    try:
        return ExecutionScope.from_dict(scope_data)
    except InvalidScopeComponentError as exc:
        # The field-level detail names the offending value, which can carry an
        # end-user identifier, so it stays in this server-side log line; the
        # raised message names only the failure's type, because that message
        # reaches the task's error column and a client-facing event.
        logger.error("Persisted execution scope snapshot failed validation: %s", exc)
        raise ExecutionScopeResolverContractError(
            "persisted execution scope snapshot failed validation "
            f"({type(exc).__name__}); see the logged field-level detail"
        ) from exc


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
# (sandbox_key_suffix, workspace_segments, effective_mount_segments,
#  sorted memory_dimensions items, isolate_external_dirs).
ScopeFingerprint = tuple[
    Optional[str],
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    bool,
]


def scope_fingerprint(scope: Optional[ExecutionScope]) -> Optional[ScopeFingerprint]:
    """Hashable fingerprint of the namespaces a scope selects.

    Per-task caches that bake scope-derived state in at build time (sandbox
    keys, workspace paths, sandbox mount root, memory dimensions,
    ``allowed_external_dirs``) key their eviction checks on this. Each
    namespace field is read through
    :data:`_EXECUTION_SCOPE_NAMESPACE_FIELD_READERS`, the same table
    ``_execution_scope_field_diff`` and ``_execution_scope_narrowing_violations``
    use, rather than a raw attribute access here -- the mount slot in
    particular reads ``effective_mount_segments`` (not the raw
    ``sandbox_mount_segments`` attribute) so a changed mount prefix
    invalidates the cache instead of silently reusing a stale ``base_dir``
    (which a later rebuild would then reject in
    ``SandboxManager._ensure_config_equivalent``). ``isolate_external_dirs``
    is included for the same reason: it is baked
    into the cached ``AgentService``'s ``Workspace.allowed_external_dirs``
    at build time (``_build_allowed_external_dirs`` -> ``AgentService.
    __init__`` -> ``WorkspaceManager.get_or_create_workspace``) rather than
    read fresh per call, so an isolate_external_dirs-only change across
    turns must evict the cache or the stale allowed-dirs list (shared root
    vs. scope-local) keeps being enforced. ``strict_memory_isolation`` is
    intentionally excluded: it is read fresh from the contextvar on every
    memory operation (``UserIsolatedMemoryStore``), so nothing cached here
    goes stale when only that flag changes. ``None`` is the sentinel for
    unscoped, distinct from an empty scope's fingerprint.
    """
    if scope is None:
        return None
    reader = _EXECUTION_SCOPE_NAMESPACE_FIELD_READERS
    return (
        reader["sandbox_key_suffix"](scope),
        reader["workspace_segments"](scope),
        reader["sandbox_mount_segments"](scope),
        tuple(sorted(reader["memory_dimensions"](scope).items())),
        reader["isolate_external_dirs"](scope),
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


class DeferToSnapshot:
    """Resolver return value meaning "defer to the persisted snapshot".

    Not an :class:`ExecutionScope`: the carrier itself never enters
    :func:`resolve_execution_scope`'s return value or reaches the
    ``current_execution_scope`` contextvar, and carries no scope attributes
    of its own. What the carrier *resolves to* -- a validated snapshot, or
    ``fallback`` itself when none is persisted (or the persisted one fails
    validation) -- is exactly what gets returned and activated instead, the
    same as any other resolved scope.

    ``fallback`` is mandatory (not ``Optional``, no default): it is the scope
    used when the task carries no persisted snapshot -- a task id the
    resolver's embedder does not itself recognize, with no Task-table
    snapshot either. :func:`resolve_execution_scope` cannot enforce that it
    is a *meaningful* answer (a resolver may pass anything); supplying a real
    one for the no-snapshot case is the resolver author's obligation. An
    implicit ``None`` fallback would silently mean "authoritative unscoped on
    a snapshot miss" and must instead be spelled out by the caller.

    ``snapshot_defines_namespace`` picks which of two abstention shapes a
    carrier is:

    - ``False`` (the default): the resolver has its own opinion of this
      task's namespace -- expressed as ``fallback`` -- and a persisted
      snapshot may only *narrow* it further (see
      :func:`_execution_scope_narrowing_violations`). ``fallback`` is
      therefore also the mandatory floor the snapshot must narrow, and must
      be the resolver's own most conservative answer for the task (e.g.
      today's creator-direct scope), so the resolved value never ends up
      wider than that answer would have been. Use this when the resolver
      *owns* the task but wants a corroborating/narrowing snapshot to have a
      say.
    - ``True``: the resolver does not own this task and does not know its
      namespace -- the workforce/delegated-task pattern, where a task is
      created and scoped entirely by the persisted snapshot and the
      resolver's embedder has no record mapping that task id to a namespace
      of its own. In that shape ``fallback`` **must not** pre-commit to the
      namespace (if it could, the resolver would return an authoritative
      ``ExecutionScope`` instead of deferring), and that precondition is
      enforced at construction (below). Given a conforming fallback, the
      narrowing check is skipped for namespace fields and a present, type-
      and shape-valid snapshot's namespace fields are used verbatim. This
      opt-in declares the snapshot is the namespace authority for tasks this
      resolver does not own; it does not make an untrusted snapshot safe, and
      it does not skip the type check, the *newer*-shape half of the
      shape-version gate (see
      ``_validate_execution_scope_snapshot_candidate``), the mandatory
      ``fallback`` for a snapshot miss, or policy-field symmetry (every field
      in :data:`_EXECUTION_SCOPE_POLICY_FIELDS` that differs is logged and
      ``fallback``'s value still wins) -- only namespace narrowing and the
      *older*-shape half of the version gate are skipped.

    Neither value distinguishes a snapshot that arrived inside a
    client-supplied ``Task.agent_config`` from one written server-side out of
    an already-resolved scope -- those are different trust tiers, and this
    contract cannot tell them apart at read time. It cannot be fixed here
    either: any provenance marker would live in the same client-writable
    field as the snapshot it vouches for. The actual fix is at the input
    boundary instead: a task-create request body can no longer seed this key
    at all -- ``execution_scope`` is one of
    ``CLIENT_RESERVED_AGENT_CONFIG_KEYS`` (``xagent.web.services.task_runtime``),
    stripped from every client-supplied ``agent_config`` before it reaches
    ``Task.agent_config`` (issue #1016).

    That makes ``snapshot_defines_namespace=True`` unshippable for now, not
    merely delicate: with the namespace taken verbatim, any writer that can
    put a client-seeded value into ``Task.agent_config`` chooses the
    namespace. The request-body boundary now refuses ``execution_scope``, but
    the column is not closed under that refusal: closed-source distributions
    register runtime-extension providers that receive a raw
    ``session_factory`` (``TaskRuntimeContext.session_factory``,
    ``xagent.core.task_runtime``) and can write this column directly, and a
    row persisted before the request-body refusal existed still carries
    whatever a request seeded at the time -- those rows are not disarmed by
    that refusal and cannot be judged from the column itself, since a value
    stored then could name any version or marker (issue #1135 owns clearing
    them). No resolver in this repository sets
    ``snapshot_defines_namespace=True``, and none should until the snapshot is
    server-owned end to end.
    The flag exists so the abstention shape a workforce sub-task needs is
    expressible and reviewable; it is inert until something opts in.

    Construction enforces that opt-in's precondition: with
    ``snapshot_defines_namespace=True`` the ``fallback`` must claim no
    namespace at all -- every field in
    :data:`_EXECUTION_SCOPE_NAMESPACE_FIELDS` sitting at its own no-scoping
    identity. A fallback that already commits to a namespace has an answer
    the snapshot would *replace* rather than narrow, which is the one thing
    no abstention shape may do; such a resolver must return an authoritative
    :class:`ExecutionScope` instead of deferring. Policy fields are
    unconstrained -- the fallback still owns those on both shapes.

    Raises:
        TypeError: ``fallback`` is not an ``ExecutionScope``.
        ExecutionScopeResolverContractError:
            ``snapshot_defines_namespace=True`` with a ``fallback`` that
            already commits to a namespace.
    """

    __slots__ = ("fallback", "snapshot_defines_namespace")

    def __init__(
        self,
        fallback: ExecutionScope,
        *,
        snapshot_defines_namespace: bool = False,
    ) -> None:
        if not isinstance(fallback, ExecutionScope):
            raise TypeError(
                "DeferToSnapshot(fallback) requires an ExecutionScope, got "
                f"{type(fallback).__name__}"
            )
        if snapshot_defines_namespace:
            committed = _namespace_committed_fields(fallback)
            if committed:
                # Machine-checks the opt-in's precondition rather than leaving
                # it to prose: resolve_execution_scope hands the namespace to
                # the snapshot verbatim on this shape, so a fallback that had
                # pre-committed to one would be silently overwritten instead
                # of acting as a floor.
                logger.error(
                    "DeferToSnapshot(snapshot_defines_namespace=True) got a "
                    "namespace-committed fallback %r; committed fields: %s",
                    fallback,
                    sorted(committed),
                )
                raise ExecutionScopeResolverContractError(
                    "DeferToSnapshot(snapshot_defines_namespace=True) "
                    "requires a fallback that claims no namespace, but this "
                    f"one commits to {sorted(committed)!r}; a resolver that "
                    "already knows the namespace must return an authoritative "
                    "ExecutionScope instead of deferring"
                )
        self.fallback = fallback
        self.snapshot_defines_namespace = snapshot_defines_namespace


# The embedding application injects a scope resolver via
# set_execution_scope_resolver() (same injection pattern as
# set_user_tool_overrides_hook in the web layer). Three return values:
#
# - ``ExecutionScope``: authoritative for this task.
# - ``None``: authoritative unscoped for this task (not "abstain").
# - ``DeferToSnapshot`` (see that class): abstain in favor of
#   the persisted snapshot, with a fallback for when none is persisted.
ExecutionScopeResolver = Callable[[str], Union[ExecutionScope, None, DeferToSnapshot]]

_execution_scope_resolver: Optional[ExecutionScopeResolver] = None


def execution_scope_resolver_registered() -> bool:
    """Whether an embedder has registered a scope resolver.

    Used at startup to log which authority mode is active: with a resolver
    registered, persisted snapshots are a corroborating candidate (see
    :func:`resolve_execution_scope`); with none, they are the sole answer.
    """
    return _execution_scope_resolver is not None


def set_execution_scope_resolver(
    resolver: Optional[ExecutionScopeResolver],
    *,
    acknowledges_snapshot_candidate_contract: bool = False,
) -> None:
    """Register the resolver that maps a ``task_id`` to its ExecutionScope.

    Resolver contract:

    - **Idempotent per task**: called at the start of every turn of a task
      (including resumed turns after a process restart), it must return an
      equal scope for the same ``task_id`` every time. Reassigning a task to
      a different scope between turns is possible but expensive (per-task
      caches rebuild); a resolver that flaps A -> B -> A is a bug.
    - **Three-valued return**: an ``ExecutionScope`` is authoritative for
      the task; ``None`` is authoritative *unscoped* for the task (not an
      abstention); a :class:`DeferToSnapshot` carrier abstains in favor of the
      persisted snapshot, with a mandatory fallback for tasks that carry
      none. No registered resolver means every task runs unscoped.
    - **Provenance, not existence**: whether to defer must be decided from
      the task's own provenance (e.g. whether the embedder's own records
      show it as one of its tasks), never from whether a snapshot happens
      to exist for it. Deferring merely because a snapshot is present
      degrades this contract back into "snapshot always wins".
    - Scope fields are consumed independently by subsystems; the resolver
      may populate any subset.
    - An exception from the resolver fails the turn: falling back to
      unscoped on error would silently merge namespaces.
    - When a persisted snapshot also exists for the task,
      :func:`resolve_execution_scope` treats it as a corroborating
      candidate, not an override: a namespace-affecting disagreement with
      an authoritative (non-defer) answer fails the turn instead of
      silently picking one side.

    Args:
        resolver: The resolver, or ``None`` to clear it.
        acknowledges_snapshot_candidate_contract: Must be ``True`` to
            register a non-``None`` resolver.

    Raises:
        TypeError: a non-``None`` resolver is registered without passing
            ``acknowledges_snapshot_candidate_contract=True``. This fails at
            registration time (e.g. embedder import/startup) instead of the
            first turn silently treating a persisted snapshot as an override
            by an embedder that has not read the contract above.
    """
    if resolver is not None and not acknowledges_snapshot_candidate_contract:
        raise TypeError(
            "set_execution_scope_resolver(resolver) requires "
            "acknowledges_snapshot_candidate_contract=True: a persisted "
            "scope snapshot is now a corroborating candidate rather than an "
            "override, and callers must confirm they have read the "
            "resolver's three-valued / provenance contract in this "
            "function's docstring before registering one"
        )
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
    tasks without one. With a resolver registered, the snapshot is a
    corroborating candidate for the resolver's authoritative answer (see
    :func:`resolve_execution_scope`); with no resolver registered, it is the
    sole answer -- this is what keeps internally created tasks (workforce
    runs, whose ids the embedder's resolver cannot map) scoped across
    process restarts. Loader exceptions fail the turn, except on the
    branch where the resolver already returned a real ``ExecutionScope``:
    there, a broken candidate is logged and ignored rather than vetoing an
    authoritative answer that already exists. With no resolver registered,
    or when the resolver abstained with a :class:`DeferToSnapshot`, nobody has
    an authoritative answer yet, so a broken loader fails the turn instead
    of silently proceeding unscoped or under a stale fallback.
    """
    global _execution_scope_snapshot_loader
    _execution_scope_snapshot_loader = loader


class ExecutionScopeResolverContractError(Exception):
    """A resolver, a snapshot loader, or persisted snapshot data broke its contract.

    Deliberately not a subclass of ``RuntimeError``, ``ValueError``, or
    ``TypeError``: several websocket handlers catch
    ``except (ValueError, KeyError, TypeError)`` around the turn-execution
    path and fold anything in that tuple into a generic "client message
    format error" response. A resolver or loader author's bug at the
    ``resolve_execution_scope`` boundary, and a persisted snapshot that no
    longer decodes, are both server-side/persisted-data faults rather than
    malformed client messages, and must not be swallowed by that handler as
    if they were.

    Raised in five places:

    - the resolver's three-valued return type
      (:func:`resolve_execution_scope`);
    - a loaded snapshot that is neither ``None`` nor an ``ExecutionScope``
      (``_validate_execution_scope_snapshot_candidate``) -- a snapshot loader
      is held to the same discipline as the resolver, since its output
      reaches the turn contextvar on both branches;
    - a persisted snapshot that fails field validation while being decoded
      (:func:`execution_scope_from_agent_config`);
    - a persisted snapshot whose ``version`` is present but unreadable
      (:func:`_coerce_snapshot_version`) -- malformed persisted data, not a
      snapshot old enough to pre-date the field;
    - a ``DeferToSnapshot(snapshot_defines_namespace=True)`` carrier whose
      ``fallback`` already commits to a namespace
      (:class:`DeferToSnapshot`) -- that opt-in's precondition.

    Messages name the offending *type*, never the offending value: this
    error escapes to a generic handler and can reach the client, and a
    resolver bug that returns an internal object must not publish its
    ``repr()``. The value is available in the server-side log at the raise
    site.

    The ``TypeError`` that :func:`set_execution_scope_resolver` raises for a
    missing acknowledgment token is unrelated -- that happens at
    registration time (embedder import/startup, before any request handler
    exists) and is intentionally left as a plain ``TypeError``.
    """


class ExecutionScopeAuthorityError(Exception):
    """A persisted snapshot disagrees with the resolver's authoritative scope.

    Raised directly by :func:`resolve_execution_scope`'s
    resolver-authoritative branch (the resolver returned a real
    ``ExecutionScope``): there, ``resolver_scope`` genuinely is that
    authoritative answer, which is why :func:`resolve_execution_scope_off_turn`
    may downgrade to it. The resolver-abstention branch raises the
    :class:`ExecutionScopeAbstentionMismatchError` subclass instead of this
    class directly — never this class itself — precisely so a caller cannot
    conflate the two: on that branch ``resolver_scope`` is not an
    authoritative answer (see the subclass docstring for why it must not be
    downgraded off-turn).

    Deliberately not a subclass of ``RuntimeError`` or ``ValueError``:
    ``resolve_execution_scope`` already raises plain ``ValueError`` for a
    ``None`` task_id, and callers/tests must be able to tell an authority
    conflict apart from that structural error instead of both being folded
    by the same ``except ValueError`` (or a broad ``except RuntimeError``)
    clause.

    Carries the resolver's answer (``resolver_scope``) so a caller that must
    not fail a turn that has already ended or never started (see
    :func:`resolve_execution_scope_off_turn`) can log a structured warning
    and continue with the authoritative value instead of raising. Whether
    that downgrade is sanctioned is stated by the raise site, not inferred
    from the exception's class: ``resolver_scope_is_authoritative`` is a
    required constructor argument, so every raise site — including one added
    by a future subclass — must declare whether ``resolver_scope`` is a real
    authority to fall back on. :func:`resolve_execution_scope_off_turn`
    downgrades only when that flag is ``True``, which is fail-closed for
    anything that has not said so.

    ``str()`` on this exception deliberately carries only the ``task_id``
    and the names of the mismatched fields, never the scope values: it
    ends up in ``task.error_message`` and in the client's terminal error
    event. Both surfacing sites format that durable/broadcast string
    straight from ``str(exc)``: ``task_orchestrator``'s per-turn background
    runner (``_schedule_bg``'s ``_runner``, which resolves the scope for a
    fresh turn before calling ``execute_task_background``) and
    ``websocket.execute_resume_background`` (which resolves its own scope
    for a resumed turn instead of receiving one pre-resolved). The scope
    values include ``sandbox_key_suffix``, ``workspace_segments``, and
    ``memory_dimensions`` -- namespace components that can carry
    end-user/client identifiers. The full scopes and the field-level value
    diff are logged via ``logger.error`` at the raise site instead, which
    stays server-side.
    """

    def __init__(
        self,
        task_id: str,
        *,
        resolver_scope: ExecutionScope,
        snapshot_scope: ExecutionScope,
        mismatched_fields: Mapping[str, tuple[Any, Any]],
        resolver_scope_is_authoritative: bool,
    ) -> None:
        """Record the conflict.

        Args:
            task_id: The task whose resolution conflicted.
            resolver_scope: What the resolver contributed -- an authoritative
                answer on the authoritative branch, the abstention carrier's
                ``fallback`` on the abstention branch.
            snapshot_scope: The persisted candidate that disagreed.
            mismatched_fields: ``{field_name: (snapshot_value,
                resolver_value)}`` for the namespace fields that conflicted.
            resolver_scope_is_authoritative: Whether ``resolver_scope`` is a
                sanctioned answer an off-turn caller may continue with.
                Keyword-only and required: it is the raise site, not the
                exception's class, that knows whether an authority exists,
                and :func:`resolve_execution_scope_off_turn` downgrades only
                on ``True``.
        """
        self.task_id = task_id
        self.resolver_scope = resolver_scope
        self.snapshot_scope = snapshot_scope
        self.mismatched_fields = dict(mismatched_fields)
        self.resolver_scope_is_authoritative = resolver_scope_is_authoritative
        super().__init__(
            f"execution scope authority mismatch for task {task_id!r}: "
            f"mismatched_fields={sorted(self.mismatched_fields)!r}"
        )


class ExecutionScopeAbstentionMismatchError(ExecutionScopeAuthorityError):
    """A snapshot fails the resolver-abstention branch's narrowing check.

    Raised by :func:`resolve_execution_scope` when the resolver returned
    a :class:`DeferToSnapshot` carrier (it does not own this task's scope) and
    the persisted snapshot is neither absent/shape-gated nor a valid narrowing
    of the carrier's ``fallback``. It inherits ``resolver_scope`` from the
    base class, but that attribute means something different here: the
    resolver never produced an authoritative scope on this branch, so
    ``resolver_scope`` is ``fallback`` — the value the resolver supplied
    *while declining to answer*, typically all-default/unscoped for the
    delegated/workforce case this branch exists for.

    This is why a caller must never downgrade this exception the way
    :func:`resolve_execution_scope_off_turn` downgrades the authoritative
    branch's error: a downgrade needs an authority to downgrade *to*, and
    this branch has none. Handing out ``fallback`` as if it were one would be
    indistinguishable from "no scope at all" to a consumer like workspace
    cleanup. Its raise site therefore passes
    ``resolver_scope_is_authoritative=False``, which is what actually blocks
    the downgrade; being a distinct type on top of that keeps the two
    branches separable for callers that want to handle only one of them,
    without any caller having to enumerate which subclasses are unsafe.
    """


# Namespace-affecting fields: a disagreement here changes which sandbox key,
# workspace path, mount root, or memory-dimension notes a task's execution
# actually touches, so resolve_execution_scope fails the turn rather than
# silently picking one side.
_EXECUTION_SCOPE_NAMESPACE_FIELDS: tuple[str, ...] = (
    "sandbox_key_suffix",
    "workspace_segments",
    "sandbox_mount_segments",
    "memory_dimensions",
    "isolate_external_dirs",
)

# Policy fields: change post-filter behavior on an otherwise-identical
# namespace, never which key/path is touched. A disagreement here is logged
# and the resolver's value wins, but does not fail the turn.
_EXECUTION_SCOPE_POLICY_FIELDS: tuple[str, ...] = ("strict_memory_isolation",)

# How to read each namespace field's *namespace value* off a scope. Every
# field reads its own attribute raw except ``sandbox_mount_segments``, whose
# namespace value is the derived ``effective_mount_segments`` property
# (``None`` means "the full workspace segments", so the raw attribute alone
# does not say what the mount actually covers). All three consumers that
# compare or fingerprint namespace fields -- ``_execution_scope_field_diff``,
# ``_execution_scope_narrowing_violations``, and ``scope_fingerprint`` --
# read through this table instead of picking their own attribute, so "what
# value does this field contribute to the namespace" has exactly one answer
# in this module: a future derived field cannot be read raw by one consumer
# and derived by another.
_EXECUTION_SCOPE_NAMESPACE_FIELD_READERS: Mapping[
    str, Callable[[ExecutionScope], Any]
] = {
    "sandbox_key_suffix": lambda scope: scope.sandbox_key_suffix,
    "workspace_segments": lambda scope: scope.workspace_segments,
    "sandbox_mount_segments": lambda scope: scope.effective_mount_segments,
    "memory_dimensions": lambda scope: scope.memory_dimensions,
    "isolate_external_dirs": lambda scope: scope.isolate_external_dirs,
}


def _narrows_by_equality(snapshot_value: Any, fallback_value: Any) -> bool:
    """``snapshot_value`` narrows ``fallback_value`` only by being equal.

    Used for fields with no partial order beyond equality: an opaque
    validated string (``sandbox_key_suffix``) or a boolean that has no
    state narrower than ``True`` (``isolate_external_dirs``).
    """
    return bool(snapshot_value == fallback_value)


def _narrows_by_prefix(
    snapshot_value: tuple[str, ...], fallback_value: tuple[str, ...]
) -> bool:
    """``snapshot_value`` narrows ``fallback_value`` by extending it.

    Used for path-shaped fields (``workspace_segments``, and
    ``sandbox_mount_segments`` via its ``effective_mount_segments`` reader):
    extra trailing segments only ever place a scope deeper inside
    ``fallback_value``'s already-claimed tree, never outside it.
    """
    return snapshot_value[: len(fallback_value)] == fallback_value


def _narrows_by_superset(
    snapshot_value: Mapping[str, str], fallback_value: Mapping[str, str]
) -> bool:
    """``snapshot_value`` narrows ``fallback_value`` by extending it.

    Used for ``memory_dimensions``: extra dimensions only ever narrow which
    notes a scoped search sees; a missing or changed dimension from
    ``fallback_value`` would widen it.
    """
    return all(
        snapshot_value.get(key) == value for key, value in fallback_value.items()
    )


# Per-field narrowing metadata: (identity_value, narrows_relation).
# ``identity_value`` is the field's own no-scoping default -- when
# ``fallback`` sits at it, the resolver has claimed no authority in that
# dimension for this task, so only an exact match is accepted regardless of
# what ``narrows_relation`` would otherwise say (see
# ``_execution_scope_narrowing_violations``'s docstring for why). When
# ``fallback`` is not at its identity, ``narrows_relation(snapshot_value,
# fallback_value)`` decides. Keyed by the same names as
# :data:`_EXECUTION_SCOPE_NAMESPACE_FIELDS` so a namespace field added there
# without an entry here fails the completeness test in
# tests/core/test_execution_scope.py instead of being silently skipped by
# ``_execution_scope_narrowing_violations``'s loop.
_EXECUTION_SCOPE_NAMESPACE_FIELD_NARROWING: Mapping[
    str, tuple[Any, Callable[[Any, Any], bool]]
] = {
    "sandbox_key_suffix": (None, _narrows_by_equality),
    "workspace_segments": ((), _narrows_by_prefix),
    "sandbox_mount_segments": ((), _narrows_by_prefix),
    "memory_dimensions": ({}, _narrows_by_superset),
    "isolate_external_dirs": (False, _narrows_by_equality),
}


def _namespace_committed_fields(scope: ExecutionScope) -> set[str]:
    """Namespace fields on which ``scope`` claims scoping of its own.

    A field is committed when its namespace value -- read through
    :data:`_EXECUTION_SCOPE_NAMESPACE_FIELD_READERS`, so a derived field like
    the mount is judged on what it actually covers -- differs from that
    field's own no-scoping identity in
    :data:`_EXECUTION_SCOPE_NAMESPACE_FIELD_NARROWING`. An empty result means
    the scope partitions nothing: the shape a
    ``DeferToSnapshot(snapshot_defines_namespace=True)`` fallback must have,
    since that opt-in hands namespace authority to the snapshot outright (see
    :class:`DeferToSnapshot`). Driven by the same two tables as the narrowing
    check, so a namespace field added there is covered here without another
    edit.
    """
    return {
        name
        for name in _EXECUTION_SCOPE_NAMESPACE_FIELDS
        if _EXECUTION_SCOPE_NAMESPACE_FIELD_READERS[name](scope)
        != _EXECUTION_SCOPE_NAMESPACE_FIELD_NARROWING[name][0]
    }


def _execution_scope_field_diff(
    snapshot: ExecutionScope, resolver: ExecutionScope
) -> dict[str, tuple[Any, Any]]:
    """Per-field ``(snapshot_value, resolver_value)`` for differing fields.

    Compares every namespace/policy field (excludes ``version``, which is
    shape bookkeeping, not a scope field). Namespace fields are read through
    :data:`_EXECUTION_SCOPE_NAMESPACE_FIELD_READERS` rather than a raw
    ``getattr`` so a field whose namespace value is derived (see that
    table's docstring) is compared on its namespace value, not its raw
    attribute -- reading ``sandbox_mount_segments`` raw here would flag a
    conflict between a resolver-built scope (mount left at its default
    ``None``) and a snapshot with the mount explicitly set equal to the
    workspace segments, even though the two select the identical mount.
    """
    diff: dict[str, tuple[Any, Any]] = {}
    for name in _EXECUTION_SCOPE_NAMESPACE_FIELDS:
        reader = _EXECUTION_SCOPE_NAMESPACE_FIELD_READERS[name]
        snapshot_value = reader(snapshot)
        resolver_value = reader(resolver)
        if snapshot_value != resolver_value:
            diff[name] = (snapshot_value, resolver_value)
    for name in _EXECUTION_SCOPE_POLICY_FIELDS:
        snapshot_value = getattr(snapshot, name)
        resolver_value = getattr(resolver, name)
        if snapshot_value != resolver_value:
            diff[name] = (snapshot_value, resolver_value)
    return diff


def _execution_scope_narrowing_violations(
    snapshot: ExecutionScope, fallback: ExecutionScope
) -> dict[str, tuple[Any, Any]]:
    """Namespace fields where ``snapshot`` is not a narrowing of ``fallback``.

    Used only on :func:`resolve_execution_scope`'s resolver-abstention
    branch, where the resolver has supplied a mandatory *fallback* instead
    of an authoritative answer: a persisted snapshot remains untrusted input
    -- a runtime-extension provider can write ``Task.agent_config`` directly
    through its ``session_factory`` (``TaskRuntimeContext.session_factory``,
    ``xagent.core.task_runtime``), bypassing every request-body sanitizer,
    and a row persisted before ``execution_scope`` was reserved at the
    request boundary (``CLIENT_RESERVED_AGENT_CONFIG_KEYS``,
    ``xagent.web.services.task_runtime``) still carries whatever value a
    request seeded at the time -- so an unchecked snapshot here would let a
    caller widen its own namespace past the resolver's own most conservative
    answer for the task. Returns an empty dict when ``snapshot`` narrows (or
    matches) ``fallback`` on every field below.

    The governing rule, per field: narrowing can only extend scoping the
    fallback *already committed to*. If ``fallback``'s value for a field is
    that field's own no-scoping identity (``None`` for ``sandbox_key_suffix``,
    ``()`` for ``workspace_segments``/the mount, ``False`` for
    ``isolate_external_dirs``, ``{}`` for ``memory_dimensions``), the
    resolver has claimed *no* authority in that dimension for this task, so
    ``snapshot`` may not introduce any there either -- only an exact match is
    accepted, regardless of whether the introduced value would, taken alone,
    look like it "only narrows". Concretely, this means an all-default
    ``fallback`` (the natural value for a resolver with "no opinion" on this
    task) accepts nothing but an all-default snapshot: there is no scoping
    to narrow into, so every field falls back to equality. This is what
    closes the case a purely-relative, per-field prefix/superset test cannot:
    such a test is vacuously satisfied by *any* snapshot value once the
    fallback's own field sits at its identity element.

    When ``fallback``'s value for a field is *not* at that identity, the
    field's specific narrowing relation applies:

    - ``sandbox_key_suffix``: no partial order beyond equality -- a suffix is
      an opaque, validated string, not something one value can be "deeper
      inside" another. Equal is the only accepted relation once ``fallback``
      has committed to a specific value.
    - ``workspace_segments``: ``snapshot``'s tuple starts with ``fallback``'s
      -- extra trailing path segments only ever place a scope deeper inside
      ``fallback``'s already-claimed directory tree, never outside it.
    - ``sandbox_mount_segments``: the same prefix test, read via
      :data:`_EXECUTION_SCOPE_NAMESPACE_FIELD_READERS` (``effective_mount_segments``)
      so an unset field (mount covers the full ``workspace_segments``)
      compares like its expanded form rather than bypassing the check.
    - ``isolate_external_dirs``: a boolean has no state narrower than
      ``True``, so once ``fallback`` is already ``True`` only ``True`` is
      accepted -- equality, same as ``sandbox_key_suffix``.
    - ``memory_dimensions``: ``snapshot``'s mapping must be a superset of
      ``fallback``'s -- extra dimensions only ever narrow which notes a
      scoped search sees; a missing or changed dimension from ``fallback``
      would widen it.

    ``strict_memory_isolation`` (policy, not namespace) is intentionally
    excluded, matching :data:`_EXECUTION_SCOPE_POLICY_FIELDS` -- narrowing
    is about the namespace a scope selects, not post-filter policy on it. A
    policy-only disagreement between ``snapshot`` and ``fallback`` is
    handled separately by the caller (see the abstention branch of
    :func:`resolve_execution_scope`), not by this function.

    Enforced generically: this loops over
    :data:`_EXECUTION_SCOPE_NAMESPACE_FIELDS`, reading each field through
    :data:`_EXECUTION_SCOPE_NAMESPACE_FIELD_READERS` and applying its
    identity/relation pair from
    :data:`_EXECUTION_SCOPE_NAMESPACE_FIELD_NARROWING`, rather than five
    hardcoded per-field checks -- a namespace field added to the first table
    without a matching entry in the second raises a ``KeyError`` here
    instead of silently passing through unchecked.
    """
    violations: dict[str, tuple[Any, Any]] = {}
    reader = _EXECUTION_SCOPE_NAMESPACE_FIELD_READERS

    for name in _EXECUTION_SCOPE_NAMESPACE_FIELDS:
        identity, narrows = _EXECUTION_SCOPE_NAMESPACE_FIELD_NARROWING[name]
        fallback_value = reader[name](fallback)
        snapshot_value = reader[name](snapshot)
        ok = (
            snapshot_value == fallback_value
            if fallback_value == identity
            else narrows(snapshot_value, fallback_value)
        )
        if not ok:
            recorded_snapshot = (
                dict(snapshot_value)
                if isinstance(snapshot_value, Mapping)
                else snapshot_value
            )
            recorded_fallback = (
                dict(fallback_value)
                if isinstance(fallback_value, Mapping)
                else fallback_value
            )
            violations[name] = (recorded_snapshot, recorded_fallback)

    return violations


def _decode_execution_scope_snapshot(
    task_id: str | int,
    persisted_agent_config: ExecutionScopeSnapshotSource,
) -> Optional[ExecutionScope]:
    """Produce the snapshot candidate, decoding it here rather than upstream.

    A database owner that already read the task row in its own Session may
    hand over the raw ``agent_config`` via ``persisted_agent_config`` to skip
    the registered loader (see :func:`resolve_execution_scope`).

    The decode deliberately happens inside this function, so a snapshot that
    fails field validation raises from the same place the registered loader
    would raise: whether that is tolerated is then decided by which branch of
    :func:`resolve_execution_scope` is calling, exactly as it is for the
    loader. A caller that decoded first and passed the result would have to
    reimplement that per-branch policy, and passing ``None`` after a failed
    decode would read as "no snapshot exists" -- which on the branches that
    must fail closed is indistinguishable from an authoritative
    "unscoped".
    """
    if persisted_agent_config is EXECUTION_SCOPE_NOT_PROVIDED:
        return (
            _execution_scope_snapshot_loader(str(task_id))
            if _execution_scope_snapshot_loader is not None
            else None
        )
    return execution_scope_from_agent_config(
        cast(Optional[Mapping[str, Any]], persisted_agent_config)
    )


def _validate_execution_scope_snapshot_candidate(
    task_id: str | int,
    snapshot: Optional[ExecutionScope],
    diff_target: ExecutionScope,
    *,
    enforce_shape_version: bool = True,
) -> Optional[ExecutionScope]:
    """Shared candidate gate a loaded snapshot must pass before either branch uses it.

    Both the authoritative and resolver-abstention branches of
    :func:`resolve_execution_scope` route a freshly loaded snapshot through
    this one check rather than duplicating it. ``_decode_execution_scope_snapshot``
    only ``cast()``s the loader's return (a static type hint, not a runtime
    check), so nothing upstream of this function enforces that the loader
    actually honored its contract -- without this gate, a loader returning a
    raw dict, a stray :class:`DeferToSnapshot`, or any other object would
    reach the turn contextvar unchecked, or (on the branch that compares
    fields) raise an unrelated ``AttributeError`` at the first ``.version``
    access instead of a diagnosable contract error. The type check below
    always applies, regardless of ``enforce_shape_version``.

    The shape-version gate is **asymmetric**, because the two directions of
    version skew fail differently:

    - *Older* (lower version, or the key missing entirely): every field such
      a snapshot carries is still decodable here, and fields it lacks take
      their current defaults. The only thing it breaks is a *comparison* --
      a field the current shape added or changed always looks "different"
      from a scope built under the older shape, which is not a real
      conflict. So this direction is refused only where a comparison
      actually happens, and ``enforce_shape_version=False`` is the caller's
      statement that ``snapshot``, once it passes the type check, will be
      used verbatim rather than compared, so nothing needs protecting and
      gating would discard real data for nothing.
    - *Newer* (higher version, e.g. a mixed-version rollout where some
      workers already run the next shape): refused on **every** path,
      ``enforce_shape_version=False`` included. ``ExecutionScope.from_dict``
      drops keys this shape does not know, so a newer snapshot arrives
      partially decoded -- potentially with a namespace-narrowing field gone
      -- and "used verbatim" would mean using that truncated scope.
      ``to_dict`` then always stamps the current constant, and the workforce
      snapshot writer re-persists the resolved scope, so accepting one also
      re-saves the truncation under this process's version, erasing the
      evidence that anything was ever dropped.

    Concretely: the authoritative branch and the default (non-opted-in)
    resolver-abstention branch both compare the snapshot (against the
    resolver's answer, or against the carrier's fallback for narrowing) and
    pass the default ``True``; the ``snapshot_defines_namespace=True``
    abstention branch uses the snapshot verbatim and passes ``False``,
    relaxing the older direction only. A gated-away snapshot is logged with a
    field-level diff against ``diff_target`` and treated as absent (``None``)
    rather than raised on; that diff is only computed when the warning will
    actually be emitted, since this gate runs on every turn for any snapshot
    that predates a shape bump.

    Returns:
        The snapshot unchanged when it is an ``ExecutionScope``, its version
        does not exceed :data:`EXECUTION_SCOPE_SHAPE_VERSION`, and either
        ``enforce_shape_version`` is ``False`` or its version matches that
        constant exactly; ``None`` when there was no snapshot to begin with,
        or it was shape-gated away.

    Raises:
        ExecutionScopeResolverContractError: ``snapshot`` is neither
            ``None`` nor an ``ExecutionScope``.
    """
    if snapshot is None:
        return None
    if not isinstance(snapshot, ExecutionScope):
        raise ExecutionScopeResolverContractError(
            "execution scope snapshot loader returned a "
            f"{type(snapshot).__name__} for task {task_id!r}; expected an "
            "ExecutionScope or None"
        )
    written_by_a_newer_shape = snapshot.version > EXECUTION_SCOPE_SHAPE_VERSION
    if written_by_a_newer_shape or (
        enforce_shape_version and snapshot.version != EXECUTION_SCOPE_SHAPE_VERSION
    ):
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "Ignoring execution scope snapshot candidate for task %s: shape "
                "version %s does not match the current version %s; diff=%s",
                task_id,
                snapshot.version,
                EXECUTION_SCOPE_SHAPE_VERSION,
                _execution_scope_field_diff(snapshot, diff_target),
            )
        return None
    return snapshot


def resolve_execution_scope(
    task_id: str | int,
    *,
    persisted_agent_config: ExecutionScopeSnapshotSource = (
        EXECUTION_SCOPE_NOT_PROVIDED
    ),
) -> Optional[ExecutionScope]:
    """Resolve the scope for ``task_id``.

    With no resolver registered, the persisted snapshot (see
    :func:`set_execution_scope_snapshot_loader`) is the sole answer,
    byte-identical to the pre-authority behavior standalone/workforce
    deployments rely on for restart/resume -- returned exactly as loaded,
    with **no type check and no shape-version gate**: there is nothing to
    compare it to on this branch, and this is a known, pre-existing gap
    rather than a new contract (see
    ``test_no_resolver_registered_non_scope_loader_output_passes_through``).
    With a resolver registered, the resolver is authoritative and runs
    first; the snapshot (when the resolver's answer needs one) is a
    corroborating candidate only. Whenever a snapshot is loaded on a
    resolver-registered branch, both branches below route it through the
    same :func:`_validate_execution_scope_snapshot_candidate` gate before
    using it -- neither branch compares against, or activates, a snapshot
    that hasn't passed that check:

    - Resolver raises: propagates immediately: the snapshot is not consulted.
    - Resolver returns ``None``: authoritative unscoped; the snapshot is not
      consulted -- not even loaded, so the ``INFO`` line this branch emits
      records that any persisted snapshot was bypassed unread rather than
      whether one existed.
    - Resolver returns a :class:`DeferToSnapshot` carrier: the resolver does
      not know this task's scope, so a snapshot-loader exception here
      propagates instead of being swallowed -- "the resolver abstained"
      means nobody has an authoritative answer for this turn, and turn
      callers must fail closed rather than silently proceed unscoped or
      under a stale fallback (contrast the authoritative branch below,
      which already has a real answer and can afford to ignore a broken
      candidate). An absent or shape-gated snapshot falls back to the
      carrier's ``fallback``. Otherwise, whether a present, shape-valid
      snapshot's namespace fields are checked against ``fallback`` depends
      on the carrier's ``snapshot_defines_namespace`` (see
      :class:`DeferToSnapshot`):

      - ``False`` (the default): the shape-version gate applies in both
        directions (the snapshot is about to be compared), and the snapshot
        is used only if it is a *narrowing* of the carrier's ``fallback``
        (see :func:`_execution_scope_narrowing_violations`) -- a snapshot
        remains untrusted input regardless of the request-body refusal at
        task-create time: a runtime-extension provider can write
        ``Task.agent_config`` directly through its ``session_factory``, and
        a row persisted before that refusal existed still carries whatever a
        request seeded at the time, so an unchecked snapshot here would let
        a caller widen its own namespace past the resolver's own
        conservative fallback. A non-narrowing snapshot raises
        :class:`ExecutionScopeAbstentionMismatchError` (a subclass of
        :class:`ExecutionScopeAuthorityError` -- see that class for why the
        distinction matters to :func:`resolve_execution_scope_off_turn`).
      - ``True``: the resolver has stated it does not own this task and the
        snapshot is the namespace authority; the narrowing check is skipped
        and the snapshot's namespace fields are used verbatim once it passes
        the type check and the *newer*-shape half of the version gate (an
        older-shape snapshot is used as-is here, a newer-shape one is still
        refused -- see
        :func:`_validate_execution_scope_snapshot_candidate`). The carrier's
        own construction guarantees ``fallback`` claims no namespace on this
        shape, so nothing of the resolver's is being replaced.

      Once the namespace is settled either way, every field in
      :data:`_EXECUTION_SCOPE_POLICY_FIELDS` on which the snapshot and the
      fallback differ is logged and taken from the fallback -- symmetric with
      the authoritative branch below, where the resolver's policy values win
      the same way. The snapshot's settled namespace is preserved while those
      policy values are overlaid onto it.
    - Resolver returns an ``ExecutionScope``: authoritative. A snapshot
      loader exception is logged and ignored (the candidate is corrupt, but
      an authoritative answer already exists). Otherwise: a
      namespace-affecting difference (see
      :data:`_EXECUTION_SCOPE_NAMESPACE_FIELDS`) raises
      :class:`ExecutionScopeAuthorityError`; a policy-only difference (see
      :data:`_EXECUTION_SCOPE_POLICY_FIELDS`) is logged and the resolver's
      value still wins.

    A database owner that already read the task row may hand over its raw
    ``agent_config`` via ``persisted_agent_config``; this skips the registered
    loader while preserving the same precedence, and the snapshot inside it is
    decoded here so that a malformed one is tolerated or fatal according to
    the branch below rather than according to the caller. Explicit ``None``
    means "this task carries no agent_config", not "force an unscoped task".

    Raises:
        ValueError: ``task_id`` is None — ``str(None)`` would silently
            query the loader/resolver for the literal string ``"None"``.
            Callers that legitimately have no task identity must treat
            that as unscoped themselves instead of passing None.
        ExecutionScopeAuthorityError: raised by the resolver-authoritative
            branch when a persisted snapshot disagrees with the resolver's
            scope on a namespace-affecting field. ``resolver_scope`` on
            this exception is a genuine authoritative answer.
        ExecutionScopeAbstentionMismatchError: (subclass of the above)
            raised by the resolver-abstention branch when, unless the
            carrier opted into ``snapshot_defines_namespace=True``, a
            persisted snapshot is not a narrowing of the resolver's
            fallback. ``resolver_scope`` on this exception is ``fallback``,
            not an authoritative answer -- see the subclass docstring.
        ExecutionScopeResolverContractError: the resolver, or a registered
            snapshot loader, returned something outside its contract, or a
            persisted snapshot failed to decode (see
            :func:`execution_scope_from_agent_config`).
    """
    if task_id is None:
        raise ValueError(
            "task_id cannot be None; a caller without a task identity "
            "must treat the execution as unscoped instead"
        )
    if isinstance(persisted_agent_config, ExecutionScope):
        # Checked here, before any branch dispatch, and deliberately not where
        # the value is consumed: the authoritative branch wraps its load in a
        # tolerant except so a bad *candidate* cannot veto a real answer, and a
        # raise from inside that wrapper would be absorbed and reported as a
        # loader failure. This is a caller bug, not a data problem, so which
        # branch happens to be active must not decide whether it is heard.
        #
        # The value itself would otherwise fail silently rather than loudly: an
        # already-decoded scope is not a Mapping, so it decodes to "this task
        # carries no snapshot", which on the two fail-closed branches is
        # indistinguishable from an authoritative unscoped answer.
        raise ExecutionScopeResolverContractError(
            "persisted_agent_config takes the task's raw agent_config mapping, "
            "not an already-decoded ExecutionScope; the snapshot inside it is "
            "decoded during resolution so each branch keeps its own tolerance"
        )

    resolver = _execution_scope_resolver
    if resolver is None:
        return _decode_execution_scope_snapshot(task_id, persisted_agent_config)

    resolved = resolver(str(task_id))

    if isinstance(resolved, DeferToSnapshot):
        # No try/except here: the resolver has just said it does not know
        # this task's scope, so a broken loader means nobody knows, and the
        # turn faces that reach this branch must fail closed rather than
        # silently proceed unscoped or under a stale fallback.
        snapshot = _decode_execution_scope_snapshot(task_id, persisted_agent_config)
        snapshot = _validate_execution_scope_snapshot_candidate(
            task_id,
            snapshot,
            resolved.fallback,
            enforce_shape_version=not resolved.snapshot_defines_namespace,
        )
        if snapshot is None:
            return resolved.fallback

        if not resolved.snapshot_defines_namespace:
            violations = _execution_scope_narrowing_violations(
                snapshot, resolved.fallback
            )
            if violations:
                logger.error(
                    "Execution scope snapshot for task %s widens the resolver's "
                    "fallback instead of narrowing it: %s",
                    task_id,
                    violations,
                )
                # Not the base ExecutionScopeAuthorityError: the resolver
                # abstained on this branch, so resolved.fallback below is
                # not an authoritative value -- see
                # ExecutionScopeAbstentionMismatchError's docstring for why
                # that distinction must survive as far as the exception type.
                raise ExecutionScopeAbstentionMismatchError(
                    str(task_id),
                    resolver_scope=resolved.fallback,
                    snapshot_scope=snapshot,
                    mismatched_fields=violations,
                    # The resolver declined to answer, so resolved.fallback is
                    # not an authority an off-turn caller may continue with.
                    resolver_scope_is_authoritative=False,
                )
        policy_diff = {
            name: (getattr(snapshot, name), getattr(resolved.fallback, name))
            for name in _EXECUTION_SCOPE_POLICY_FIELDS
            if getattr(snapshot, name) != getattr(resolved.fallback, name)
        }
        if policy_diff:
            # Symmetric with the authoritative branch's policy-only-mismatch
            # handling below: the namespace is settled (validated above as a
            # narrowing of resolved.fallback, or handed to the snapshot by the
            # opt-in), but policy fields are not part of that check, so an
            # unlogged pass-through here would silently let a
            # client-influenceable snapshot flip one. The fallback plays the
            # authoritative role on this branch, so its policy values win, the
            # same way the resolver's win on the authoritative branch -- while
            # the settled namespace stays the snapshot's, which is why this
            # overlays onto ``snapshot`` instead of returning the fallback.
            # Both the comparison and the overlay are driven by
            # _EXECUTION_SCOPE_POLICY_FIELDS, so a policy field added there is
            # honoured here without another edit.
            logger.warning(
                "Execution scope policy-only mismatch for task %s on the "
                "resolver-abstention branch (fallback wins): %s",
                task_id,
                policy_diff,
            )
            return replace(
                snapshot,
                **{name: getattr(resolved.fallback, name) for name in policy_diff},
            )
        return snapshot

    if resolved is None:
        # INFO, not WARNING: a resolver claiming a task as unscoped is the
        # three-valued contract working as designed, not a fault. It is still
        # logged on every such turn, ungated, because this is the branch that
        # de-scopes a task whose persisted snapshot may well have named a
        # namespace -- a resolver that does not yet recognise older tasks
        # de-scopes every one of them, and that has to leave a per-task trace
        # the way every sibling branch that overrides or ignores a candidate
        # does. The snapshot is deliberately not loaded here (nothing on this
        # branch consults it), so the line says the snapshot was bypassed
        # unread rather than claiming to know whether one exists: loading one
        # purely to log it would give the resolver's cheapest answer the cost
        # of the branches that actually need a snapshot.
        logger.info(
            "Execution scope resolver returned None for task %s: authoritative "
            "unscoped; any persisted snapshot is bypassed without being loaded",
            task_id,
        )
        return None

    if not isinstance(resolved, ExecutionScope):
        # Names the type, not the value: this message reaches a generic
        # handler and can surface to the client, and a resolver bug that
        # returns an internal config object must not publish its repr.
        logger.error(
            "Execution scope resolver returned an unsupported value for task %s: %r",
            task_id,
            resolved,
        )
        raise ExecutionScopeResolverContractError(
            f"execution scope resolver returned a {type(resolved).__name__}; "
            "expected an ExecutionScope, None, or a DeferToSnapshot"
        )

    try:
        snapshot = _decode_execution_scope_snapshot(task_id, persisted_agent_config)
    except Exception:
        logger.warning(
            "Snapshot loader failed while the resolver returned an "
            "authoritative scope for task %s; ignoring the candidate",
            task_id,
            exc_info=True,
        )
        return resolved

    snapshot = _validate_execution_scope_snapshot_candidate(task_id, snapshot, resolved)
    if snapshot is None:
        return resolved

    diff = _execution_scope_field_diff(snapshot, resolved)
    if not diff:
        return resolved

    namespace_diff = {
        name: values
        for name, values in diff.items()
        if name in _EXECUTION_SCOPE_NAMESPACE_FIELDS
    }
    if namespace_diff:
        # Names the remediation, not just the fault: this mismatch is stable,
        # so every turn of this task fails the same way until the persisted
        # snapshot is removed or corrected. Without the recovery spelled out
        # here, the only signal an operator gets is a repeating failure with
        # no stated way out, and the snapshot lives in a column no request
        # path offers to clear.
        logger.error(
            "Execution scope authority mismatch for task %s: %s. Every turn of "
            "this task fails until the persisted snapshot agrees with the "
            "resolver: clear the %r key from this task's agent_config to let "
            "the resolver's answer stand alone, or correct it to match. "
            "Clearing relocates the task to the resolver's namespace, so any "
            "workspace or sandbox subtree it already owns under the snapshot's "
            "namespace is left behind -- correct the snapshot instead where "
            "that subtree still matters.",
            task_id,
            diff,
            EXECUTION_SCOPE_AGENT_CONFIG_KEY,
        )
        # mismatched_fields carries only namespace_diff, not the full diff:
        # this error reaches task.error_message and a client-facing event
        # (see the class docstring), and the raise is gated on
        # namespace_diff alone -- including policy-only fields here would
        # misleadingly list them as "mismatched" next to the real conflict.
        # The full diff (namespace + policy) is still logged above, which
        # stays server-side.
        raise ExecutionScopeAuthorityError(
            str(task_id),
            resolver_scope=resolved,
            snapshot_scope=snapshot,
            mismatched_fields=namespace_diff,
            # The resolver produced a real scope, so an off-turn caller that
            # cannot fail may continue with it.
            resolver_scope_is_authoritative=True,
        )

    logger.warning(
        "Execution scope policy-only mismatch for task %s (resolver wins): %s",
        task_id,
        diff,
    )
    return resolved


def resolve_execution_scope_off_turn(task_id: str | int) -> Optional[ExecutionScope]:
    """Resolve scope for a consumer outside the turn lifecycle.

    Used by off-turn storage-key/workspace-segment composition (legacy
    preview backfill, a not-yet-persisted durable object) that cannot fail a
    turn that has already ended or never started.

    Downgrades an :class:`ExecutionScopeAuthorityError` to
    ``exc.resolver_scope`` plus a structured warning, instead of raising,
    **only** when the raise site declared that value to be a real authority
    (``resolver_scope_is_authoritative``). That holds for
    ``resolve_execution_scope``'s resolver-authoritative branch, where the
    resolver returned a real ``ExecutionScope``; an unhandled mismatch there
    would otherwise surface as a misleading "file not found" or a bulk
    endpoint's 500, at a point where an authoritative answer already exists
    to fall back on.

    Anything that has *not* declared an authoritative value propagates
    unchanged -- the condition is the positive one, so a mismatch raised by
    some future branch is fail-closed here by default rather than needing to
    be enumerated as an exclusion. Today that covers
    :class:`ExecutionScopeAbstentionMismatchError`, raised by the
    resolver-abstention branch: a downgrade needs an authority to downgrade
    *to*, and that branch never produced one -- its ``resolver_scope`` is
    ``fallback``, the value the resolver supplied *while declining to
    answer*, typically all-default/unscoped for the delegated/workforce
    case. Handing that out here would be indistinguishable from "no scope at
    all" to a caller such as workspace cleanup, so this off-turn face stays
    fail-closed exactly like the in-turn one.

    Every other exception (resolver/loader failure, ``task_id`` None) also
    propagates unchanged.
    """
    try:
        return resolve_execution_scope(task_id)
    except ExecutionScopeAuthorityError as exc:
        if not exc.resolver_scope_is_authoritative:
            raise
        logger.warning(
            "Execution scope authority mismatch resolved off-turn for task "
            "%s (using the resolver's answer): %s",
            exc.task_id,
            exc.mismatched_fields,
        )
        return exc.resolver_scope


@contextmanager
def turn_execution_scope(task_id: str | int) -> Iterator[Optional[ExecutionScope]]:
    """Resolve and activate the execution scope for one turn of ``task_id``.

    A convenience wrapper combining :func:`resolve_execution_scope` with
    :class:`ExecutionScopeContext`: entering it at the start of a turn, at
    the same place the acting user is resolved, makes restart/resume
    re-derive the scope correctly. The scope (or explicit None) is set for
    the duration of the turn and restored on exit.
    """
    scope = resolve_execution_scope(task_id)
    token = set_execution_scope(scope)
    try:
        yield scope
    finally:
        reset_execution_scope(token)
