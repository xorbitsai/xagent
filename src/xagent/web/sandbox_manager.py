"""
Sandbox management in application layer.
"""

import asyncio
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import (
    SANDBOX_VOLUMES,
    get_boxlite_home_dir,
    get_external_upload_dirs,
    get_sandbox_cpus,
    get_sandbox_env,
    get_sandbox_host_storage_root,
    get_sandbox_idle_ttl,
    get_sandbox_image,
    get_sandbox_max_concurrency,
    get_sandbox_max_containers,
    get_sandbox_memory,
    get_sandbox_sweep_interval,
    get_sandbox_volumes,
    get_storage_root,
    get_uploads_dir,
)
from ..core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    build_code_mount_volumes,
)
from ..core.workspace import scoped_user_root
from ..sandbox import SandboxService
from ..sandbox.base import (
    ResolvedSandboxRuntimeSpec,
    Sandbox,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxContractError,
    SandboxInfo,
    SandboxInspection,
    SandboxMountIntent,
    SandboxNotFoundError,
    SandboxRecoveryRequiredError,
    SandboxRuntimeConflictError,
    SandboxTemplate,
    SpecVerdict,
    canonical_sandbox_path,
    spec_matches_inspection,
)
from .sandbox_keys import USER_LIFECYCLE_TYPE, parse_user_lifecycle_id

logger = logging.getLogger(__name__)

_WORKER_LIFECYCLE_MARKER = "::worker::"

# Bound on the graceful-stop wait for a sandbox container the manager itself
# stops (mismatch rebuild, quiesce): the same seconds budget for both call
# sites, rather than each falling back to the backend's own (backend-specific,
# potentially unbounded) default.
_SANDBOX_STOP_TIMEOUT_SECONDS = 30


class SandboxCapacityError(RuntimeError):
    """The sandbox container cap is reached and no idle sandbox is evictable.

    Distinct from sandbox-service unavailability: by default the web layer
    rejects the task with this error instead of falling back to local
    execution (see XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY).
    """

    def __init__(self, *, cap: int, in_use: int) -> None:
        super().__init__(
            f"Sandbox capacity limit reached ({in_use} containers, cap {cap}) "
            "and all sandboxes are busy. Please retry when a running task "
            "finishes, or raise XAGENT_SANDBOX_MAX_CONTAINERS."
        )
        self.cap = cap
        self.in_use = in_use


@dataclass
class _SandboxActivity:
    """Per-lifecycle activity state used for reclamation decisions."""

    ref_count: int = 0
    last_activity: float = 0.0


@dataclass
class _LifecycleLockEntry:
    """Per-lifecycle lock with holder/waiter tracking for safe eviction."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    waiters: int = 0


class SandboxLease:
    """Async context manager for one leased sandbox execution slot."""

    def __init__(
        self,
        provider: "SandboxLeaseProvider",
        *,
        concurrency_safe: bool,
    ) -> None:
        self._provider = provider
        self._concurrency_safe = concurrency_safe
        self._slot: int | None = None
        self._sandbox: Sandbox | None = None

    async def __aenter__(self) -> Sandbox:
        if not self._concurrency_safe:
            self._sandbox = self._provider.primary_sandbox
            return self._sandbox

        self._slot = await self._provider.acquire_worker_slot()
        try:
            self._sandbox = await self._provider.get_worker_sandbox(self._slot)
            return self._sandbox
        except Exception:
            await self._provider.release_worker_slot(self._slot)
            self._slot = None
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._slot is not None:
            await self._provider.release_worker_slot(self._slot)
            self._slot = None
        self._sandbox = None


class SandboxLeaseProvider:
    """Lease primary or worker sandboxes for sandboxed tool execution."""

    def __init__(
        self,
        *,
        manager: "SandboxManager",
        lifecycle_type: str,
        lifecycle_id: str,
        primary_sandbox: Sandbox,
        mount_intent: SandboxMountIntent | None,
        max_concurrency: int,
    ) -> None:
        self._manager = manager
        self._lifecycle_type = lifecycle_type
        self._lifecycle_id = lifecycle_id
        self._mount_intent = mount_intent
        self._available_slots: asyncio.Queue[int] = asyncio.Queue()
        self._worker_locks: dict[int, asyncio.Lock] = {}
        self._workers: dict[int, Sandbox] = {}
        self.primary_sandbox = primary_sandbox
        for slot in range(max(1, max_concurrency)):
            self._available_slots.put_nowait(slot)

    def lease(self, *, concurrency_safe: bool) -> SandboxLease:
        """Return an async context manager for the requested execution mode."""
        return SandboxLease(self, concurrency_safe=concurrency_safe)

    async def acquire_worker_slot(self) -> int:
        """Reserve one worker slot, waiting when all workers are busy."""
        return await self._available_slots.get()

    async def release_worker_slot(self, slot: int) -> None:
        """Return a worker slot to the provider."""
        self._available_slots.put_nowait(slot)

    async def get_worker_sandbox(self, slot: int) -> Sandbox:
        """Get or lazily create a worker sandbox for a slot.

        The worker's desired spec is derived from the same ``mount_intent``
        the primary sandbox was built from (same physical mount set, a
        different name) — routed through the manager's per-lifecycle-key
        gate (``get_or_create_sandbox``), the same gate provider creation and
        release-to-zero use, so a worker can never be created while its
        primary is mid-release or mid-reconcile.

        When the container cap leaves no room for a worker, or the worker's
        own reconciliation is rejected (e.g. a same-key runtime-config
        conflict, or a sandbox that needs recovery before use), the lease
        degrades to the primary sandbox instead of failing the tool
        mid-task — the same sharing semantics non-concurrency-safe leases
        already have, trading isolation for availability. The degraded
        result is not cached, so a later lease retries worker creation
        once the underlying condition clears.
        """
        if slot in self._workers:
            return self._workers[slot]

        if slot not in self._worker_locks:
            self._worker_locks[slot] = asyncio.Lock()

        async with self._worker_locks[slot]:
            if slot in self._workers:
                return self._workers[slot]
            try:
                # Resolved through the manager's locked entry point, not the
                # unlocked resolver: it takes this worker's *primary*
                # lifecycle gate -- the same lock provider creation and
                # release-to-zero use -- so a worker cannot be created
                # while its primary is mid-release or mid-reconcile.
                worker = await self._manager.get_or_create_sandbox(
                    self._lifecycle_type,
                    f"{self._lifecycle_id}::worker::{slot}",
                    mount_intent=self._mount_intent,
                )
            except (SandboxCapacityError, SandboxContractError) as exc:
                logger.warning(
                    "Worker sandbox %s::%s::worker::%d unavailable (%s); "
                    "degrading to the primary sandbox: %s",
                    self._lifecycle_type,
                    self._lifecycle_id,
                    slot,
                    type(exc).__name__,
                    exc,
                )
                return self.primary_sandbox
            self._workers[slot] = worker
            return worker

    async def cleanup_worker_sandboxes(self) -> None:
        """Delete worker sandboxes while keeping the primary sandbox cached."""
        await self._manager.delete_worker_sandboxes(
            self._lifecycle_type,
            self._lifecycle_id,
        )
        self._workers.clear()


def absolute_backend_mount_path(path: str | Path) -> Path:
    """Absolutize one backend-domain path: env vars, then ``~``, then cwd.

    The backend domain is this process's own filesystem; translation to the
    machine actually running the container backend happens afterwards, in
    ``SandboxPathMapper``. The configuration values that reach it
    (``XAGENT_UPLOADS_DIR``, ``XAGENT_EXTERNAL_UPLOAD_DIRS``) are raw strings
    and may be relative or ``~``-prefixed, while every downstream mount
    consumer -- the volume tuples and ``SandboxMountIntent``'s lexical
    classification alike -- requires an absolute path. Every producer of a
    backend-domain mount path goes through this one function so a relative
    spelling resolves to the same directory everywhere.
    """
    backend_path = Path(os.path.expandvars(str(path))).expanduser()
    if not backend_path.is_absolute():
        backend_path = Path.cwd() / backend_path
    return backend_path


def resolve_backend_mount_path(path: str | Path) -> str:
    """Absolutize a backend-domain path and resolve its symlinks.

    ``SandboxMountIntent``'s covered/covering/disjoint split is purely
    lexical, so it cannot tell a directory that is lexically inside a mount
    root from a symlink at that same lexical position pointing somewhere
    else entirely. Only the second one is *not* exposed by the root's bind,
    so folding decisions must be taken against resolved paths -- this is the
    resolver that answers that question.

    The paths that end up mounted deliberately keep their unresolved
    spelling: the guest mount point has to stay the path the rest of the
    system (file tools, the workspace allowlist) already refers to.
    """
    return os.path.realpath(absolute_backend_mount_path(path))


class SandboxPathMapper:
    """Translate backend-visible workspace paths into sandbox volume tuples."""

    def __init__(
        self,
        *,
        backend_storage_root: Path,
        host_storage_root: Path | None,
        sandbox_storage_root: Path | None = None,
    ) -> None:
        self.backend_storage_root = self._as_backend_path(backend_storage_root)
        self.host_storage_root = host_storage_root
        self.sandbox_storage_root = self._as_backend_path(
            sandbox_storage_root or self.backend_storage_root
        )

    @classmethod
    def from_env(cls) -> "SandboxPathMapper":
        return cls(
            backend_storage_root=get_storage_root(),
            host_storage_root=get_sandbox_host_storage_root(),
        )

    @property
    def uses_host_storage_root(self) -> bool:
        return self.host_storage_root is not None

    @staticmethod
    def _as_backend_path(path: str | Path) -> Path:
        return absolute_backend_mount_path(path)

    def _relative_to_backend_storage(self, backend_path: Path) -> Path | None:
        try:
            return backend_path.relative_to(self.backend_storage_root)
        except ValueError:
            return None

    def to_host_bind_source(self, backend_path: str | Path) -> Path:
        path = self._as_backend_path(backend_path)
        if self.host_storage_root is None:
            return path

        relative_path = self._relative_to_backend_storage(path)
        if relative_path is None:
            return path
        return self.host_storage_root / relative_path

    def to_sandbox_target(self, backend_path: str | Path) -> Path:
        path = self._as_backend_path(backend_path)
        if self.host_storage_root is None:
            return path

        relative_path = self._relative_to_backend_storage(path)
        if relative_path is None:
            return path
        return self.sandbox_storage_root / relative_path

    def volume_for_backend_path(
        self, backend_path: str | Path, mode: str = "rw"
    ) -> tuple[str, str, str]:
        return (
            str(self.to_host_bind_source(backend_path)),
            str(self.to_sandbox_target(backend_path)),
            mode,
        )


class SandboxManager:
    """Manages sandbox instances, their activity state, and reclamation.

    Primitives (what each dict/lock is the single source of truth for):

    - ``_cache`` / ``_config_cache``: the live ``Sandbox`` instance and the
      ``ResolvedSandboxRuntimeSpec`` it was built from, per exact sandbox
      name (primary or worker). Always written and popped together. A
      ``_config_cache`` hit for a name is the first gate any route (legacy
      or reconciliation) must pass: a mismatching freshly-desired spec is a
      loud ``SandboxRuntimeConflictError``, never a silent adoption.
    - ``_lease_providers``: the cached ``SandboxLeaseProvider``, per primary
      (base) lifecycle name only — never per worker name.
    - ``_activity``: ref-count + last-activity, per base lifecycle name.
      The single source of truth reclamation decisions are made from.
    - ``_reconcile_budget``: remaining mismatch-triggered rebuilds, per base
      lifecycle name. Only the reconciliation route's mismatch branch
      consumes it (never the absent-\\>create branch, and never the delete
      *inside* a mismatch rebuild — see ``_reconcile_delete`` — since that
      would erase the decrement this same rebuild just made); ``cleanup()``
      resets it wholesale, and ``_delete_sandbox_names`` drops the one entry
      for a base name once its delete for that name has actually succeeded
      — a still-live container (delete failed) keeps its exhausted budget,
      so a later occupant of the key only starts with a fresh budget once
      the earlier container is confirmed gone. A worker-only disposal
      (``delete_worker_sandboxes`` / release-to-zero) never touches it: the
      primary the budget is scoped to is still live.
    - ``_locks`` / ``_locks_guard``: per-exact-name creation lock used only
      by the legacy (non-reconciling) backend route — see
      ``_legacy_get_or_create``. The reconciliation route never inserts
      into this dict: its own per-key exclusion is ``_lifecycle_locked``.
    - ``_lifecycle_locks``: per-base-name lock with waiter tracking. The
      single per-key gate every route now funnels through — provider
      creation, release-to-zero, worker creation, and (for the
      reconciling backend) the entire inspect-then-act reconciliation
      sequence. Entries are garbage-collected when the last holder/waiter
      leaves.
    - ``_capacity_gate`` (global): serializes the cap check + eviction +
      container creation so concurrent creations for different names cannot
      all pass the count check.

    Synchronization primitives, from innermost to outermost:

    - ``_activity_guard`` (one asyncio.Lock): makes compound check-then-act
      decisions on activity state atomic — attach's provider-existence
      check + ref-count increment, release's decrement + provider pop, and
      the eviction claim (ref-count re-check + provider pop + instance
      cache purge). Critical sections must stay fully synchronous: never
      ``await`` while holding it, and never acquire it inside a ``finally``
      (a cancellation delivered at that await point would skip the cleanup).
      Independent single dict operations do NOT need it.
    - ``_lifecycle_locks`` (per lifecycle key, waiter-counted): serialize
      lease provider creation, worker creation, release-to-zero worker
      cleanup, and (reconciling backend) the full reconcile sequence, with
      the idle sweep, per key. Entries are garbage-collected when the last
      holder/waiter leaves.
    - ``_locks`` + ``_locks_guard`` (per sandbox name): serialize container
      creation per name inside the legacy route only.
    - ``_capacity_gate`` (global): see above.

    Ordering rules:

    - lifecycle lock -> (legacy route: per-name lock) -> capacity gate is
      the only nesting direction; never acquire a lifecycle lock while
      holding the gate (a same-key creator holds its lifecycle lock while
      waiting for the gate, so gate -> lifecycle closes a deadlock cycle).
      Capacity eviction therefore does NOT lock the victim's lifecycle: it
      relies on the gate plus the atomic claim purging the instance cache,
      which forces any concurrent same-key re-creation to cache-miss and
      queue behind the gate until the deletion finished, AND on
      ``_pick_eviction_victim`` skipping any base name whose lifecycle lock
      is currently held or awaited — which is what keeps a reconciliation
      in progress for key X safe from being picked as an eviction victim by
      an unrelated key Y's capacity check.

    Destruction paths and what each one pops (identity-only ABA depends on
    every destructive path using this same pop set for ``_lease_providers``
    — a fresh object always replaces the old one, never a mutation):

    - ``_delete_sandbox_names`` (explicit ``delete_sandbox`` /
      ``delete_worker_sandboxes`` / sweep / capacity eviction): pops
      ``_cache``, ``_config_cache``, ``_locks``, ``_lease_providers``,
      ``_activity`` for every name whose backend delete was attempted
      (even on failure — the instance may be in an unknown state), and
      additionally ``_reconcile_budget`` for a name that is itself a base
      (primary) name, but ONLY once that name's own delete succeeded — a
      failed delete leaves the budget exhausted, since the same
      mismatching container is still there. Never popped for a
      worker-only ``delete_worker_sandboxes`` call, where the primary the
      budget is scoped to is untouched regardless of outcome.
    - ``_claim_idle_sandbox`` (idle sweep / capacity eviction, ahead of the
      ``_delete_sandbox_names`` call that follows): pops the primary's and
      its workers' ``_cache``/``_config_cache`` entries plus
      ``_lease_providers``. It does not touch ``_reconcile_budget`` — the
      physical delete has not run yet at this point, so whether the
      container is actually gone is still unknown; that decision belongs
      to ``_delete_sandbox_names`` above.
    - ``release`` (ref-count reaches zero): pops ``_lease_providers`` only
      (under ``_lifecycle_locked`` + ``_activity_guard``), then deletes
      worker sandboxes (which pops their ``_cache``/``_config_cache``
      through the path above). The primary's own ``_cache``/
      ``_config_cache``/``_activity``/``_reconcile_budget`` entries are
      deliberately left alone — the primary container itself is not deleted
      on release, only its provider and workers.
    - ``_reconcile_delete`` (mismatch rebuild, reconciliation route only):
      pops ``_cache``, ``_config_cache``, ``_lease_providers``,
      ``_activity`` for the one name being rebuilt — never ``_locks``
      (the reconciliation route never inserts into it in the first place)
      and never ``_reconcile_budget``: this delete is immediately followed,
      in the same call, by ``_reconcile_create`` re-populating the same
      name, so popping the budget here would erase the very decrement
      ``_reconcile_mismatch`` just made and let the same base name rebuild
      without limit.
    - ``cleanup()`` (both routes): clears ``_cache``, ``_config_cache``,
      ``_lease_providers``, ``_activity`` wholesale; the reconciling
      route's ``_quiesce`` additionally clears ``_reconcile_budget``
      (the only place that dict is ever reset) and never deletes
      containers (only stops running ones); the legacy route's
      ``_legacy_cleanup`` also clears ``_locks`` and may delete containers
      whose config has drifted (see its own docstring).

    Safety contract:

    - A lifecycle with a non-zero ref-count is never deleted, stopped, or
      evicted — by the sweep, by capacity eviction, by the reconciliation
      matrix's mismatch handling, or by any race between them. The idle
      sweep and capacity eviction go through ``_evict_idle_sandbox``, whose
      claim re-validates the ref-count under ``_activity_guard``; the
      reconciliation matrix's mismatch branch re-checks the same
      ``_activity`` ref-count before ever calling ``stop_existing`` or
      deleting, and rejects the new caller instead when it is non-zero.
    """

    def __init__(self, service: SandboxService):
        """
        Initialize sandbox manager.

        Args:
            service: SandboxService instance for creating sandboxes
        """
        self._service: SandboxService = service
        self._cache: dict[str, Sandbox] = {}
        self._config_cache: dict[str, ResolvedSandboxRuntimeSpec] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        # Activity tracking: lease providers, active-task ref-counts, and
        # last-activity timestamps keyed by the primary sandbox name. This is
        # the single source of truth reclamation decisions are made from.
        self._lease_providers: dict[str, SandboxLeaseProvider] = {}
        self._activity: dict[str, _SandboxActivity] = {}
        self._activity_guard = asyncio.Lock()
        self._lifecycle_locks: dict[str, _LifecycleLockEntry] = {}
        self._startup_monotonic = time.monotonic()
        # Global gate serializing the capacity check with container creation:
        # per-name locks cannot stop two concurrent creations for different
        # names from both passing the count check.
        self._capacity_gate = asyncio.Lock()
        # Remaining mismatch-triggered rebuilds per base lifecycle name (the
        # reconciliation route only). See the class docstring's destruction
        # matrix for exactly which paths do (and do not) touch this.
        self._reconcile_budget: dict[str, int] = {}
        # Cached result of ``await service.supports_runtime_spec()``: None
        # until the first caller resolves it (see ``_resolve_backend_probe``).
        # The app-startup readiness step resolves this eagerly in a later
        # stage; every entry point here resolves it lazily on first use so
        # this manager is correct with or without that wiring.
        self._backend_probe: bool | None = None

    @staticmethod
    def make_sandbox_name(lifecycle_type: str, lifecycle_id: str) -> str:
        """Build a sandbox name from lifecycle type and id."""
        return f"{lifecycle_type}::{lifecycle_id}"

    @staticmethod
    def parse_sandbox_name(name: str) -> tuple[str, str]:
        """Parse a sandbox name into (lifecycle_type, lifecycle_id).

        Raises:
            ValueError: Invalid sandbox name format.
        """
        parts = name.split("::", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid sandbox name format: {name!r}")
        return parts[0], parts[1]

    @staticmethod
    def _base_lifecycle_id(lifecycle_id: str) -> str:
        """Return the owner lifecycle id for primary and worker sandboxes."""
        return lifecycle_id.split(_WORKER_LIFECYCLE_MARKER, 1)[0]

    @classmethod
    def _worker_sandbox_prefix(cls, lifecycle_type: str, lifecycle_id: str) -> str:
        return (
            cls.make_sandbox_name(lifecycle_type, lifecycle_id)
            + _WORKER_LIFECYCLE_MARKER
        )

    @classmethod
    def _base_sandbox_name(cls, lifecycle_type: str, lifecycle_id: str) -> str:
        """Primary sandbox name owning activity state for a lifecycle key."""
        return cls.make_sandbox_name(
            lifecycle_type, cls._base_lifecycle_id(lifecycle_id)
        )

    @asynccontextmanager
    async def _lifecycle_locked(self, base_name: str) -> AsyncIterator[None]:
        """Serialize provider creation and release-to-zero cleanup per key.

        Entries are dropped once no holder or waiter remains, so the dict
        does not grow with every lifecycle key ever seen.

        The waiter bookkeeping is deliberately not guarded by
        ``_activity_guard``: each step is a single synchronous operation
        with no compound invariant, and awaiting a lock inside ``finally``
        could leak the waiter count if the task were cancelled at that
        await point.
        """
        entry = self._lifecycle_locks.get(base_name)
        if entry is None:
            entry = _LifecycleLockEntry()
            self._lifecycle_locks[base_name] = entry
        entry.waiters += 1

        try:
            await entry.lock.acquire()
        except BaseException:
            entry.waiters -= 1
            self._drop_lifecycle_lock_if_unused(base_name, entry)
            raise

        try:
            yield
        finally:
            entry.lock.release()
            entry.waiters -= 1
            self._drop_lifecycle_lock_if_unused(base_name, entry)

    def _drop_lifecycle_lock_if_unused(
        self, base_name: str, entry: _LifecycleLockEntry
    ) -> None:
        if entry.waiters > 0:
            return
        if self._lifecycle_locks.get(base_name) is entry:
            self._lifecycle_locks.pop(base_name, None)

    def _assert_lifecycle_locked(self, base_name: str) -> None:
        """Assert the caller already holds ``_lifecycle_locked(base_name)``.

        Mirrors ``DockerSandboxService._get_live_control``'s pattern: this
        can only check that *some* task holds the lock right now, not that
        it is the caller (``asyncio.Lock`` has no owner concept), so it is a
        guard against the lock not being held at all, not a full runtime
        proof. The structural half of this contract — that
        ``_get_or_create_sandbox_locked`` and ``_create_lease_provider_locked``
        are themselves only ever called from inside a ``_lifecycle_locked``
        block — is pinned at the source level by
        ``test_sandbox_manager_reconcile.py``'s
        ``test_gate_helper_call_sites_are_structurally_pinned``.
        """
        entry = self._lifecycle_locks.get(base_name)
        assert entry is not None and entry.lock.locked(), (
            f"called without holding _lifecycle_locked({base_name!r})"
        )

    def _touch_locked(self, base_name: str) -> _SandboxActivity:
        """Bump last-activity for a key; caller must hold ``_activity_guard``."""
        activity = self._activity.get(base_name)
        if activity is None:
            activity = _SandboxActivity()
            self._activity[base_name] = activity
        activity.last_activity = time.monotonic()
        return activity

    async def attach(self, lifecycle_type: str, lifecycle_id: str) -> bool:
        """Mark one task as actively using the lifecycle's lease provider.

        Returns False when no lease provider is cached for the key — nothing
        is attached and the caller must not release. A sandbox with a
        non-zero ref-count is never reclaimed.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._activity_guard:
            if base_name not in self._lease_providers:
                return False
            self._touch_locked(base_name).ref_count += 1
        return True

    async def attach_provider(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        provider: SandboxLeaseProvider,
    ) -> bool:
        """Attach one active task to a *specific* provider object (ABA-safe).

        Unlike ``attach()`` (existence-only: does the key have *a* cached
        provider), this additionally verifies the caller's ``provider`` is
        identically the object currently cached for this lifecycle key.
        Identity, not equality, is the guard: every path that replaces a
        provider (mismatch rebuild, sweep, capacity eviction,
        release-to-zero) always installs a genuinely new object rather than
        mutating the old one in place, so a stale handle obtained before
        such a replacement is caught here even though the *key* still
        resolves to a live (different) provider. No integer generation
        counter is needed: object identity already encodes "did this
        exact handle survive the last replacement".

        Returns False when the caller's handle is stale or nothing is
        cached; the caller must treat that exactly like ``attach()``
        returning False — evict its own cache and rebuild — rather than
        proceeding with a provider that may already be torn down.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._activity_guard:
            current = self._lease_providers.get(base_name)
            if current is None or current is not provider:
                return False
            self._touch_locked(base_name).ref_count += 1
        return True

    async def release(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        on_last_release: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Release one active task for a lifecycle key.

        When the last task releases, the cached lease provider is dropped,
        ``on_last_release`` is invoked (still under the per-key lifecycle
        lock, before worker deletion, so callers can evict their own caches
        exactly once), and the lifecycle's worker sandboxes are deleted.

        A release without a matching attach (ref-count already zero) is
        ignored with a warning: running the cleanup path anyway could tear
        down a freshly created, not-yet-attached provider.

        Returns True when this call released the last active task.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._lifecycle_locked(base_name):
            async with self._activity_guard:
                activity = self._touch_locked(base_name)
                if activity.ref_count == 0:
                    logger.warning(
                        "Ignoring sandbox release without matching attach for %s",
                        base_name,
                    )
                    return False
                if activity.ref_count > 1:
                    activity.ref_count -= 1
                    return False
                activity.ref_count = 0
                self._lease_providers.pop(base_name, None)

            if on_last_release is not None:
                on_last_release()

            await self.delete_worker_sandboxes(
                lifecycle_type, self._base_lifecycle_id(lifecycle_id)
            )
        return True

    def ref_count(self, lifecycle_type: str, lifecycle_id: str) -> int:
        """Number of active tasks attached to a lifecycle key."""
        activity = self._activity.get(
            self._base_sandbox_name(lifecycle_type, lifecycle_id)
        )
        return activity.ref_count if activity is not None else 0

    def last_activity_at(self, lifecycle_type: str, lifecycle_id: str) -> float:
        """Monotonic timestamp of the last recorded activity for a key.

        Keys with no recorded activity (e.g. containers discovered after a
        backend restart) report idle since manager startup.
        """
        activity = self._activity.get(
            self._base_sandbox_name(lifecycle_type, lifecycle_id)
        )
        if activity is None:
            return self._startup_monotonic
        return activity.last_activity

    def _get_sandbox_image_and_config(self) -> tuple[str, SandboxConfig]:
        """Get sandbox image and configuration from centralized config module."""
        image = get_sandbox_image()
        config = SandboxConfig()
        path_mapper = SandboxPathMapper.from_env()

        # CPU
        cpus = get_sandbox_cpus()
        if cpus is not None:
            config.cpus = cpus

        # MEM
        memory = get_sandbox_memory()
        if memory is not None:
            config.memory = memory

        # ENV
        env = get_sandbox_env()
        if env:
            config.env = env

        # VOL
        volumes = get_sandbox_volumes(
            host_side_sources=path_mapper.uses_host_storage_root
        )
        if volumes:
            config.volumes = volumes

        return image, config

    @staticmethod
    def _append_unique_volume(
        volumes: list[tuple[str, str, str]], volume: tuple[str, str, str]
    ) -> None:
        if volume not in volumes:
            volumes.append(volume)

    @staticmethod
    def _workspace_mount_paths(
        lifecycle_type: str,
        lifecycle_id: str,
        mount_intent: SandboxMountIntent | None,
    ) -> list[tuple[Path, bool]]:
        paths: list[tuple[Path, bool]] = []

        if mount_intent is not None:
            if mount_intent.mount_root:
                paths.append((Path(mount_intent.mount_root), True))
            for extra in mount_intent.extra_mounts:
                paths.append((Path(extra), False))
        elif lifecycle_type == USER_LIFECYCLE_TYPE:
            owner_lifecycle_id = SandboxManager._base_lifecycle_id(lifecycle_id)
            # A scope-suffixed lifecycle id ("7:tenant-a") still mounts the
            # user-level upload dir: the scope suffix namespaces the
            # container, not this default mount (scope-local dirs come from
            # mount_intent). Unparsable ids keep the historical verbatim-path
            # behavior.
            try:
                owner_id, _suffix = parse_user_lifecycle_id(owner_lifecycle_id)
                mount_root = scoped_user_root(get_uploads_dir(), owner_id)
            except ValueError:
                # Legacy/non-standard lifecycle id: keep the historical
                # verbatim-path behavior.
                mount_root = get_uploads_dir() / f"user_{owner_lifecycle_id}"
            paths.append((mount_root, True))

        return paths

    def _prepare_workspace_mounts(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        mount_intent: SandboxMountIntent | None,
        *,
        prepare_root: str | None = None,
    ) -> None:
        """Create on-host directories for a sandbox's workspace mount paths.

        The mount root is always created; each extra/allowlist mount is
        created only if it already exists (never freshly created) — same
        split as the historical base_dir(True)/allowed_external_dirs(False)
        behavior. Called explicitly right before a creation attempt (never
        on a cache hit or a plain reuse/start_existing), independent of
        spec/volume-list construction: this is not a side effect threaded
        through volume-list building.

        ``prepare_root``, when given, overrides *only* the mount root's
        creation target (see ``ChatWorkspaceBinding.prepare_root``): folding
        can re-root ``mount_intent.mount_root`` onto a covering ancestor
        shared by several scopes, but this task's own files live at the
        deeper, pre-fold ``prepare_root`` — that is the directory that must
        exist on disk, not necessarily the (possibly shallower) directory
        actually bind-mounted. Extra/allowlist mounts are unaffected.
        """
        for backend_path, should_create in self._workspace_mount_paths(
            lifecycle_type, lifecycle_id, mount_intent
        ):
            target = backend_path
            if (
                should_create
                and prepare_root is not None
                and mount_intent is not None
                # Identifies the mount-root entry (there is at most one) so
                # ``prepare_root`` only redirects that path, never an
                # extra/allowlist mount. ``backend_path`` here is exactly
                # ``Path(mount_intent.mount_root)`` (see
                # ``_workspace_mount_paths``), so this comparison reduces to
                # ``str(Path(c)) == c`` for ``c = mount_intent.mount_root``.
                # That holds because ``mount_intent.mount_root`` always
                # passed through ``canonical_sandbox_path``, which is a
                # fixed point of ``Path``: ``str(Path(canonical_sandbox_path(
                # x))) == canonical_sandbox_path(x)`` — not because its
                # normalization is a subset of ``Path``'s own (it is not:
                # e.g. ``canonical_sandbox_path("/a/b/..") == "/a"`` while
                # ``str(Path("/a/b/.."))`` leaves the ``..`` unresolved). If
                # the normalizer ever stops being idempotent under ``Path``,
                # this comparison can start missing the mount root.
                and str(backend_path) == mount_intent.mount_root
            ):
                target = Path(prepare_root)
            try:
                if should_create or target.exists():
                    os.makedirs(target, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "Failed to prepare sandbox workspace mount %s: %s",
                    target,
                    exc,
                )

    @staticmethod
    def _config_equivalent(
        left: ResolvedSandboxRuntimeSpec, right: ResolvedSandboxRuntimeSpec
    ) -> bool:
        return left == right

    @staticmethod
    def _ensure_config_equivalent(
        sandbox_name: str,
        cached_spec: ResolvedSandboxRuntimeSpec | None,
        desired_spec: ResolvedSandboxRuntimeSpec,
    ) -> None:
        if cached_spec is None:
            return
        if SandboxManager._config_equivalent(cached_spec, desired_spec):
            return
        raise SandboxRuntimeConflictError(
            f"Sandbox {sandbox_name!r} already exists with different runtime "
            "configuration. Use a distinct lifecycle id for different workspace "
            "mounts."
        )

    @staticmethod
    def _spec_from_stored_info(info: SandboxInfo) -> ResolvedSandboxRuntimeSpec:
        """Rebuild the desired spec a backend store record represents.

        Used only for the UNVERIFIED-with-a-store-row branch: since a live
        container's environment cannot be reliably reconstructed from
        inspection facts alone (see ``ObservedRuntimeFacts``), the
        previously-recorded intent is compared against instead — blind to
        drift in the actual running container, but the best available
        signal for a container with no fingerprint attestation.
        """
        return ResolvedSandboxRuntimeSpec.from_parts(
            template_type=info.template.type or "image",
            image=info.template.image,
            snapshot_id=info.template.snapshot_id,
            working_dir=info.config.working_dir,
            cpus=info.config.cpus,
            memory=info.config.memory,
            env=info.config.env,
            volumes=info.config.volumes,
            network_isolated=bool(info.config.network_isolated),
            ports=info.config.ports,
        )

    def _build_runtime_spec(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        mount_intent: SandboxMountIntent | None = None,
    ) -> ResolvedSandboxRuntimeSpec:
        """Build the single canonical desired-state spec for one sandbox name.

        The sole normalizer from environment config + code mounts + intent
        mounts to ``ResolvedSandboxRuntimeSpec``: both the legacy and the
        reconciliation routes compare against this same spec via the
        process-local ``_config_cache``, and the reconciliation route also
        hands it straight to ``service.create()`` / ``spec_matches_inspection()``.
        Pure — no directory creation or other side effects; see
        ``_prepare_workspace_mounts`` for that.
        """
        image, config = self._get_sandbox_image_and_config()
        volumes = list(config.volumes) if config.volumes else []
        volumes += self._make_default_volumes(
            lifecycle_type, lifecycle_id, mount_intent=mount_intent
        )
        return ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image",
            image=image,
            working_dir=config.working_dir,
            cpus=config.cpus,
            memory=config.memory,
            env=config.env,
            volumes=volumes,
            network_isolated=bool(config.network_isolated),
            ports=config.ports,
        )

    def _make_default_volumes(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        mount_intent: SandboxMountIntent | None = None,
    ) -> list[tuple[str, str, str]]:
        """
        Build default volume mounts.

        Code directories are always mounted read-only.
        User workspace is additionally mounted read-write for user lifecycle type.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
            mount_intent: Actual sandbox mount intent, when known
        """
        # Code mounts are always present (at least src/)
        volumes: list[tuple[str, str, str]] = list(build_code_mount_volumes())
        path_mapper = SandboxPathMapper.from_env()

        for backend_path, _should_create in self._workspace_mount_paths(
            lifecycle_type,
            lifecycle_id,
            mount_intent,
        ):
            self._append_unique_volume(
                volumes, path_mapper.volume_for_backend_path(backend_path, "rw")
            )

        return volumes

    async def _resolve_backend_probe(self) -> bool:
        """Resolve and cache ``await service.supports_runtime_spec()``.

        Resolved once per manager instance. App startup resolves this
        eagerly during the readiness step in a later stage; every entry
        point below resolves it lazily on first use so this manager behaves
        correctly with or without that wiring (tests included).
        """
        if self._backend_probe is None:
            self._backend_probe = await self._service.supports_runtime_spec()
        return self._backend_probe

    # --- Primary/worker sandbox resolution ---
    #
    # Entry points, all funneling through ``_lifecycle_locked(base_name)``
    # exactly once before reaching the unlocked internal resolver:
    # ``get_or_create_sandbox`` (one exact name, primary or worker: the
    # lease provider's worker path uses it, and it is otherwise a test seam)
    # and ``get_or_create_lease_provider`` (the production entry point).

    async def get_or_create_sandbox(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        mount_intent: SandboxMountIntent | None = None,
    ) -> Sandbox:
        """Get or create the sandbox for one exact lifecycle name.

        Acquires this name's per-key gate (shared with provider creation,
        worker creation, and release-to-zero) and delegates to the unlocked
        internal resolver. Direct callers of this method do not get a
        ``SandboxLeaseProvider`` — most production code goes through
        ``get_or_create_lease_provider`` instead, which additionally caches
        the provider object itself.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._lifecycle_locked(base_name):
            async with self._activity_guard:
                self._touch_locked(base_name)
            return await self._get_or_create_sandbox_locked(
                lifecycle_type, lifecycle_id, mount_intent=mount_intent
            )

    async def _get_or_create_sandbox_locked(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        mount_intent: SandboxMountIntent | None = None,
        prepare_root: str | None = None,
    ) -> Sandbox:
        """Resolve (create/reuse/recover) the sandbox for one exact name.

        Callers must already hold ``_lifecycle_locked(base_name)`` for this
        name's lifecycle base — this function takes no lock of its own.
        This used to be the public ``get_or_create_sandbox`` body,
        independently locked per exact name via ``_locks``; every caller
        now already holds the broader per-key gate, and the reconciliation
        route below additionally needs that same gate held across its
        inspect-then-act sequence, which a second, narrower lock could not
        provide on its own.

        First gate, common to both routes: an in-process spec-cache hit for
        this exact name must match the freshly-desired spec exactly, or the
        new caller is rejected outright with ``SandboxRuntimeConflictError``
        — regardless of which route (legacy or reconciliation) is about to
        run. On a miss, routes on the cached backend-capability probe: the
        full spec-reconciliation matrix when the backend supports it
        (``_reconcile_sandbox``), else the untouched legacy
        ``service.get_or_create()`` path (``_legacy_get_or_create``).

        ``prepare_root``, when given, overrides the on-host mkdir target for
        the mount root on a creation attempt (see
        ``ChatWorkspaceBinding.prepare_root`` and ``_prepare_workspace_mounts``);
        ``None`` keeps the historical behavior of creating
        ``mount_intent.mount_root`` itself.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        self._assert_lifecycle_locked(base_name)
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)

        desired = self._build_runtime_spec(
            lifecycle_type, lifecycle_id, mount_intent=mount_intent
        )

        cached_spec = self._config_cache.get(sandbox_name)
        if cached_spec is not None:
            self._ensure_config_equivalent(sandbox_name, cached_spec, desired)
            return self._cache[sandbox_name]

        probe = await self._resolve_backend_probe()
        if not probe:
            return await self._legacy_get_or_create(
                lifecycle_type,
                lifecycle_id,
                sandbox_name,
                mount_intent=mount_intent,
                prepare_root=prepare_root,
            )
        return await self._reconcile_sandbox(
            lifecycle_type,
            lifecycle_id,
            sandbox_name,
            base_name,
            desired,
            mount_intent,
            prepare_root=prepare_root,
        )

    async def _legacy_get_or_create(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        sandbox_name: str,
        *,
        mount_intent: SandboxMountIntent | None,
        prepare_root: str | None = None,
    ) -> Sandbox:
        """Untouched legacy path: adopts any existing container silently via
        ``service.get_or_create()``. Used only when
        ``service.supports_runtime_spec()`` is False (Boxlite today) — #296's
        reconciliation guarantee does not extend to this backend.

        Caller (``_get_or_create_sandbox_locked``) already holds
        ``_lifecycle_locked(base_name)`` and has already checked the spec
        cache for a hit; this only runs on that cache miss. Also acquires
        this exact name's ``_locks`` entry, mirroring the historical
        per-name double-checked-locking structure byte for byte: the
        broader per-key gate the caller holds already excludes any
        concurrent call for this same name, so this inner lock is provably
        uncontended and kept only so this path's own structure — and
        ``_locks``' role in it — matches the legacy (non-reconciling)
        backend byte for byte.
        """
        async with self._locks_guard:
            if sandbox_name not in self._locks:
                self._locks[sandbox_name] = asyncio.Lock()
            lock = self._locks[sandbox_name]

        async with lock:
            cached_spec = self._config_cache.get(sandbox_name)
            if cached_spec is not None:
                desired = self._build_runtime_spec(
                    lifecycle_type, lifecycle_id, mount_intent=mount_intent
                )
                self._ensure_config_equivalent(sandbox_name, cached_spec, desired)
                return self._cache[sandbox_name]

            self._prepare_workspace_mounts(
                lifecycle_type,
                lifecycle_id,
                mount_intent,
                prepare_root=prepare_root,
            )
            desired = self._build_runtime_spec(
                lifecycle_type, lifecycle_id, mount_intent=mount_intent
            )
            template, config = desired.to_backend_config()

            logger.info(
                "Getting/creating sandbox: image=%r, cpus=%r, memory=%r, volumes=%r, env_count=%r",
                desired.image,
                desired.cpus,
                desired.memory,
                config.volumes,
                len(desired.env),
            )

            logger.debug(f"Getting or creating sandbox for: {sandbox_name}")
            cap = get_sandbox_max_containers()
            if cap is None:
                sandbox = await self._service.get_or_create(
                    sandbox_name,
                    template=template,
                    config=config,
                )
            else:
                async with self._capacity_gate:
                    await self._ensure_capacity_for(sandbox_name, cap)
                    sandbox = await self._service.get_or_create(
                        sandbox_name,
                        template=template,
                        config=config,
                    )

            self._cache[sandbox_name] = sandbox
            self._config_cache[sandbox_name] = desired
            return sandbox

    # --- Spec-reconciliation route (backends with supports_runtime_spec()) ---

    async def _reconcile_sandbox(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        sandbox_name: str,
        base_name: str,
        desired: ResolvedSandboxRuntimeSpec,
        mount_intent: SandboxMountIntent | None,
        *,
        prepare_root: str | None = None,
    ) -> Sandbox:
        """Reconciliation matrix entry point for one exact sandbox name.

        Caller already holds ``_lifecycle_locked(base_name)`` and has
        already ruled out an in-process spec-cache hit.
        """
        inspection = await self._service.inspect(sandbox_name)
        if inspection is None:
            return await self._reconcile_create(
                lifecycle_type,
                lifecycle_id,
                sandbox_name,
                base_name,
                desired,
                mount_intent,
                prepare_root=prepare_root,
                retry_on_already_exists=True,
            )
        return await self._reconcile_existing(
            lifecycle_type,
            lifecycle_id,
            sandbox_name,
            base_name,
            desired,
            mount_intent,
            inspection,
            prepare_root=prepare_root,
        )

    async def _reconcile_create(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        sandbox_name: str,
        base_name: str,
        desired: ResolvedSandboxRuntimeSpec,
        mount_intent: SandboxMountIntent | None,
        *,
        prepare_root: str | None = None,
        retry_on_already_exists: bool,
    ) -> Sandbox:
        """Create a brand-new sandbox for the absent (or just-deleted) row.

        ``retry_on_already_exists`` covers the cross-process race where
        another process created the same name between our ``inspect()``
        and this ``create()``: a single re-inspect-then-act retry, not a
        fail-closed error — the alternative would spuriously reject a
        legitimate concurrent creator.
        """
        self._prepare_workspace_mounts(
            lifecycle_type, lifecycle_id, mount_intent, prepare_root=prepare_root
        )
        template, config = desired.to_backend_config()

        cap = get_sandbox_max_containers()
        try:
            if cap is None:
                sandbox = await self._service.create(sandbox_name, template, config)
            else:
                async with self._capacity_gate:
                    await self._ensure_capacity_for(sandbox_name, cap)
                    sandbox = await self._service.create(sandbox_name, template, config)
        except SandboxAlreadyExistsError:
            if not retry_on_already_exists:
                raise
            inspection = await self._service.inspect(sandbox_name)
            if inspection is None:
                raise
            return await self._reconcile_existing(
                lifecycle_type,
                lifecycle_id,
                sandbox_name,
                base_name,
                desired,
                mount_intent,
                inspection,
                prepare_root=prepare_root,
            )

        async with self._activity_guard:
            self._cache[sandbox_name] = sandbox
            self._config_cache[sandbox_name] = desired
        return sandbox

    async def _reconcile_existing(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        sandbox_name: str,
        base_name: str,
        desired: ResolvedSandboxRuntimeSpec,
        mount_intent: SandboxMountIntent | None,
        inspection: SandboxInspection,
        *,
        prepare_root: str | None = None,
    ) -> Sandbox:
        """Dispatch on the matcher verdict plus store-row presence.

        | verdict    | store row | action                                    |
        |------------|-----------|-------------------------------------------|
        | MATCH      | present   | reuse (``start_existing``)                 |
        | MATCH      | absent    | backfill the row, then reuse               |
        | UNVERIFIED | present   | t0(row)==t1(desired) -> reuse; else mismatch |
        | UNVERIFIED | absent    | mismatch (rebuild is the only convergence) |
        | MISMATCH   | either    | mismatch handling (see ``_reconcile_mismatch``) |
        """
        verdict = spec_matches_inspection(desired, inspection)
        stored = await self._service.get_store_record(sandbox_name)

        if verdict is SpecVerdict.MATCH:
            if stored is None:
                await self._backfill_store_record(sandbox_name, desired, inspection)
            return await self._reconcile_reuse(
                lifecycle_type,
                lifecycle_id,
                sandbox_name,
                base_name,
                desired,
                mount_intent,
                prepare_root=prepare_root,
            )

        if verdict is SpecVerdict.UNVERIFIED and stored is not None:
            recorded = self._spec_from_stored_info(stored)
            if recorded == desired:
                return await self._reconcile_reuse(
                    lifecycle_type,
                    lifecycle_id,
                    sandbox_name,
                    base_name,
                    desired,
                    mount_intent,
                    prepare_root=prepare_root,
                )

        return await self._reconcile_mismatch(
            lifecycle_type,
            lifecycle_id,
            sandbox_name,
            base_name,
            desired,
            mount_intent,
            inspection,
            prepare_root=prepare_root,
        )

    async def _backfill_store_record(
        self,
        sandbox_name: str,
        desired: ResolvedSandboxRuntimeSpec,
        inspection: SandboxInspection,
    ) -> None:
        """Persist a store row for a MATCH-verified container that has none.

        The container's live facts and fingerprint label already attest to
        ``desired``, so writing the missing row is a pure convergence
        action, never a rebuild: destroying an already-verified, healthy
        container over a persistence-layer gap would be pure waste and
        would turn a row-write failure into a container-destruction event.
        """
        template, config = desired.to_backend_config()
        info = SandboxInfo(
            name=sandbox_name,
            state=inspection.state,
            template=template,
            config=config,
            created_at=inspection.facts.created_at,
        )
        try:
            await self._service.persist_store_record(sandbox_name, info)
        except Exception as exc:
            logger.warning(
                "Failed to backfill store record for %s after MATCH verification: %s",
                sandbox_name,
                exc,
            )

    async def _reconcile_reuse(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        sandbox_name: str,
        base_name: str,
        desired: ResolvedSandboxRuntimeSpec,
        mount_intent: SandboxMountIntent | None,
        *,
        prepare_root: str | None = None,
    ) -> Sandbox:
        """Publish a reused sandbox via the idempotent ``start_existing``.

        Works uniformly whether the container is currently running or
        stopped. The publish itself runs under ``_capacity_gate`` exactly
        like ``_reconcile_create``/``_legacy_get_or_create`` (only when a
        cap is configured — with no cap there is no eviction to race in the
        first place), the same gate capacity eviction's claim-then-delete
        of an unrelated key always runs under (see ``_ensure_capacity_for``).

        A *same-process* eviction pass can never pick this call's own
        ``base_name`` as its LRU victim while this runs: the caller
        (``get_or_create_sandbox`` / ``_create_lease_provider_locked``) has
        held ``_lifecycle_locked(base_name)`` since before ``_reconcile_sandbox``
        was ever entered, so ``_pick_eviction_victim``'s lock-skip already
        excludes this key for this call's entire duration, not just from
        some later point once reuse is "underway". What the gate and the
        ``SandboxNotFoundError`` handling below actually guard against is
        an actor the in-process lock cannot see at all: another process
        sharing the same backend (its own independent capacity eviction, or
        a cross-process mismatch rebuild) or a manual/administrative
        removal against the backend directly. If the container vanishes
        between the inspect that chose this reuse and this
        ``start_existing`` call, the raised ``SandboxNotFoundError`` is not
        surfaced to this call's caller as a failure — it converges to the
        ordinary absent-\\>create path instead, re-inspecting under the
        now-free gate and creating the (confirmed-absent) sandbox fresh.
        """
        cap = get_sandbox_max_containers()
        try:
            if cap is None:
                sandbox = await self._service.start_existing(sandbox_name)
            else:
                async with self._capacity_gate:
                    sandbox = await self._service.start_existing(sandbox_name)
        except SandboxNotFoundError:
            inspection = await self._service.inspect(sandbox_name)
            if inspection is not None:
                return await self._reconcile_existing(
                    lifecycle_type,
                    lifecycle_id,
                    sandbox_name,
                    base_name,
                    desired,
                    mount_intent,
                    inspection,
                    prepare_root=prepare_root,
                )
            return await self._reconcile_create(
                lifecycle_type,
                lifecycle_id,
                sandbox_name,
                base_name,
                desired,
                mount_intent,
                prepare_root=prepare_root,
                retry_on_already_exists=True,
            )

        async with self._activity_guard:
            self._cache[sandbox_name] = sandbox
            self._config_cache[sandbox_name] = desired
        return sandbox

    async def _reject_if_now_active(self, base_name: str, sandbox_name: str) -> None:
        """Re-validate the ref-count immediately before a destructive step.

        ``_lifecycle_locked`` only excludes a concurrent *reconcile* for
        this base name; ``attach``/``attach_provider`` never wait for it
        (only ``_activity_guard``, see the class docstring), so a caller
        holding an already-cached lease provider for this base name can
        attach a new task while a mismatch rebuild is mid-flight here.
        Re-reading right before each destructive action — not only once at
        this function's entry — is what actually closes that window.
        """
        activity = self._activity.get(base_name)
        if activity is not None and activity.ref_count > 0:
            raise SandboxRuntimeConflictError(
                f"Sandbox {sandbox_name!r} became active while a mismatch "
                f"rebuild was in flight (lifecycle {base_name!r}); "
                "rejecting instead of tearing down an in-use sandbox."
            )

    async def _reconcile_mismatch(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        sandbox_name: str,
        base_name: str,
        desired: ResolvedSandboxRuntimeSpec,
        mount_intent: SandboxMountIntent | None,
        inspection: SandboxInspection,
        *,
        prepare_root: str | None = None,
    ) -> Sandbox:
        """Handle a MISMATCH verdict (or an UNVERIFIED verdict that a
        store-record comparison could not reconcile).

        A non-zero ref-count on the base lifecycle rejects the new caller
        outright — no destructive action is ever taken against a sandbox
        still in use, regardless of state. With a zero ref-count: a running
        mismatch is stopped first (idempotent no-op if already stopped by
        the time this runs) and, once genuinely stopped, converges exactly
        like a stopped mismatch — delete + create, gated by the per-key
        reconcile rebuild budget. The ref-count is re-read (see
        ``_reject_if_now_active``) immediately before each destructive step,
        not only here at entry, since attach can land in between.
        """
        await self._reject_if_now_active(base_name, sandbox_name)

        if inspection.state == "running":
            try:
                await self._service.stop_existing(
                    sandbox_name, timeout=_SANDBOX_STOP_TIMEOUT_SECONDS
                )
            except Exception as exc:
                raise SandboxRecoveryRequiredError(
                    f"Failed to stop mismatched sandbox {sandbox_name!r} for "
                    f"rebuild: {exc}"
                ) from exc

            post_stop = await self._service.inspect(sandbox_name)
            if post_stop is not None and post_stop.state == "running":
                raise SandboxRecoveryRequiredError(
                    f"Sandbox {sandbox_name!r} did not stop; needs recovery "
                    "before it can be rebuilt."
                )

        budget = self._reconcile_budget.get(base_name, 1)
        if budget <= 0:
            logger.warning(
                "Reconcile rebuild budget exhausted for lifecycle %s (sandbox "
                "%s); rejecting the new caller instead of rebuilding again",
                base_name,
                sandbox_name,
            )
            raise SandboxRuntimeConflictError(
                f"Sandbox {sandbox_name!r} needs a rebuild but its per-key "
                "reconciliation budget is exhausted."
            )
        self._reconcile_budget[base_name] = budget - 1

        await self._reject_if_now_active(base_name, sandbox_name)
        await self._reconcile_delete(sandbox_name)
        return await self._reconcile_create(
            lifecycle_type,
            lifecycle_id,
            sandbox_name,
            base_name,
            desired,
            mount_intent,
            prepare_root=prepare_root,
            retry_on_already_exists=True,
        )

    async def _reconcile_delete(self, sandbox_name: str) -> None:
        """Delete primitive used only by reconciliation mismatch rebuilds.

        Unlike ``_delete_sandbox_names`` (the legacy/lifecycle delete path),
        this never touches ``_locks``: the reconciliation route never
        inserts into that dict in the first place, since its own per-key
        exclusion is ``_lifecycle_locked`` (held by the caller throughout
        this whole sequence). Cache/activity bookkeeping otherwise matches
        ``_delete_sandbox_names`` exactly, so the identity-only ABA contract
        (a fresh provider/instance after a rebuild) holds regardless of
        which path triggered the delete.
        """
        try:
            await self._service.delete(sandbox_name)
            logger.debug("Sandbox deleted for rebuild: %s", sandbox_name)
        finally:
            # Plain synchronous pops on purpose, same as
            # ``_delete_sandbox_names``: each is a single dict operation with
            # no compound invariant, and awaiting ``_activity_guard`` inside
            # ``finally`` would risk skipping the eviction entirely if the
            # task were cancelled at that await point (see the class
            # docstring's ``_activity_guard`` rule).
            self._cache.pop(sandbox_name, None)
            self._config_cache.pop(sandbox_name, None)
            self._lease_providers.pop(sandbox_name, None)
            self._activity.pop(sandbox_name, None)

    async def create_lease_provider(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        mount_intent: SandboxMountIntent | None = None,
    ) -> SandboxLeaseProvider:
        """Create a lease provider for primary and worker sandboxes.

        Acquires this lifecycle's per-key gate (shared with worker creation
        and release-to-zero) before delegating to the unlocked internal
        builder. Direct callers get a *fresh* provider every time — no
        caching; ``get_or_create_lease_provider`` is the caching entry
        point most production code should use instead.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._lifecycle_locked(base_name):
            return await self._create_lease_provider_locked(
                lifecycle_type, lifecycle_id, mount_intent=mount_intent
            )

    async def _create_lease_provider_locked(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        mount_intent: SandboxMountIntent | None = None,
        prepare_root: str | None = None,
    ) -> SandboxLeaseProvider:
        """Unlocked internal builder for a lease provider.

        Caller must already hold ``_lifecycle_locked(base_name)`` for this
        lifecycle key.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        self._assert_lifecycle_locked(base_name)
        primary = await self._get_or_create_sandbox_locked(
            lifecycle_type,
            lifecycle_id,
            mount_intent=mount_intent,
            prepare_root=prepare_root,
        )
        return SandboxLeaseProvider(
            manager=self,
            lifecycle_type=lifecycle_type,
            lifecycle_id=lifecycle_id,
            primary_sandbox=primary,
            mount_intent=mount_intent,
            max_concurrency=get_sandbox_max_concurrency(),
        )

    async def _list_managed_sandbox_names(self) -> set[str]:
        """Names of existing managed containers (warmup/unparsable excluded)."""
        names: set[str] = set()
        listed_sandboxes = await self._service.list_sandboxes()
        for sb in listed_sandboxes or []:
            if not isinstance(sb.name, str):
                continue
            try:
                self.parse_sandbox_name(sb.name)
            except ValueError:
                continue
            names.add(sb.name)
        return names

    async def _pick_eviction_victim(
        self, existing: set[str], protected_base: str, skip: set[str]
    ) -> Optional[str]:
        """Pick the LRU idle primary from ``existing`` (no claim).

        Skips primaries with active tasks, the protected key, and keys whose
        lifecycle lock is currently held or awaited (in-flight creation or
        release-to-zero cleanup — including an in-flight reconciliation
        sequence, which holds the same lock for its whole duration).
        """
        async with self._activity_guard:
            candidates: list[tuple[float, str]] = []
            for name in existing:
                try:
                    lifecycle_type, lifecycle_id = self.parse_sandbox_name(name)
                except ValueError:
                    continue
                base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
                if name != base_name:
                    # Workers are deleted with their primary.
                    continue
                if base_name == protected_base or base_name in skip:
                    continue
                lock_entry = self._lifecycle_locks.get(base_name)
                if lock_entry is not None and (
                    lock_entry.lock.locked() or lock_entry.waiters > 0
                ):
                    continue
                activity = self._activity.get(base_name)
                if activity is not None and activity.ref_count > 0:
                    continue
                last_activity = (
                    activity.last_activity
                    if activity is not None
                    else self._startup_monotonic
                )
                candidates.append((last_activity, base_name))

            if not candidates:
                return None
            return min(candidates)[1]

    async def _claim_idle_sandbox(self, base_name: str) -> bool:
        """Atomically claim an idle lifecycle for deletion.

        Under the activity guard: re-validates that no task is attached,
        then drops the lease provider (new attaches fail) and the cached
        sandbox/config instances for the primary and its workers. Purging
        the instance cache is what makes eviction safe against a concurrent
        same-key re-creation: with the cache empty, ``get_or_create_sandbox``
        cannot short-circuit and hand out the doomed container — it falls
        through to the capacity gate and recreates only after the deletion
        has finished.

        This only claims the instance cache and lease provider; the base
        name's reconcile budget is left untouched here. The physical
        delete has not run yet, so the container this budget is scoped to
        may still be there if that delete goes on to fail — see
        ``_delete_sandbox_names``, which is what actually pops the budget,
        and only once its own delete for the base name succeeds.

        Returns False when the lifecycle became active since selection.
        """
        worker_prefix = base_name + _WORKER_LIFECYCLE_MARKER
        async with self._activity_guard:
            activity = self._activity.get(base_name)
            if activity is not None and activity.ref_count > 0:
                return False
            self._lease_providers.pop(base_name, None)
            for name in [
                n for n in self._cache if n == base_name or n.startswith(worker_prefix)
            ]:
                self._cache.pop(name, None)
                self._config_cache.pop(name, None)
            return True

    async def _evict_idle_sandbox(self, base_name: str, *, reason: str) -> bool:
        """Claim and delete one idle primary together with its workers.

        Shared primitive for the idle sweep and capacity eviction. The
        caller must hold the context that excludes a concurrent same-key
        re-creation from completing against the old container: the sweep
        holds the victim's per-key lifecycle lock; capacity eviction holds
        the global capacity gate (which every post-claim re-creation must
        pass through, because the claim purged the instance cache).

        Returns False when the lifecycle became active and must be spared.
        """
        if not await self._claim_idle_sandbox(base_name):
            return False

        logger.info("Reclaiming idle sandbox %s (%s)", base_name, reason)
        lifecycle_type, lifecycle_id = self.parse_sandbox_name(base_name)
        await self.delete_sandbox(lifecycle_type, lifecycle_id)
        return True

    async def _ensure_capacity_for(self, sandbox_name: str, cap: int) -> None:
        """Make room under the container cap for one new sandbox.

        Caller must hold ``_capacity_gate``. Evicts LRU idle primaries (with
        their workers) until the new container fits; raises
        ``SandboxCapacityError`` when nothing is evictable. If listing the
        service fails, enforcement is skipped for this creation (fail-open:
        the daemon being unreachable will fail the creation itself anyway).
        """
        try:
            existing = await self._list_managed_sandbox_names()
        except Exception as exc:
            logger.warning(
                "Failed to list sandboxes for capacity check; "
                "skipping enforcement for %s: %s",
                sandbox_name,
                exc,
            )
            return

        if sandbox_name in existing:
            return

        lifecycle_type, lifecycle_id = self.parse_sandbox_name(sandbox_name)
        protected_base = self._base_sandbox_name(lifecycle_type, lifecycle_id)

        # Victims whose deletion was already attempted this pass: a failed
        # delete leaves the container listed and would otherwise be re-picked
        # forever.
        tried_victims: set[str] = set()
        while len(existing) >= cap:
            victim = await self._pick_eviction_victim(
                existing, protected_base, tried_victims
            )
            if victim is None:
                raise SandboxCapacityError(cap=cap, in_use=len(existing))

            if not await self._evict_idle_sandbox(
                victim, reason=f"LRU eviction under container cap {cap}"
            ):
                # Became active between selection and claim; the picker's
                # ref-count check will exclude it on the next round.
                continue
            tried_victims.add(victim)

            try:
                existing = await self._list_managed_sandbox_names()
            except Exception as exc:
                logger.warning(
                    "Failed to re-list sandboxes after eviction; "
                    "skipping further enforcement for %s: %s",
                    sandbox_name,
                    exc,
                )
                return

    async def get_or_create_lease_provider(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        mount_intent: SandboxMountIntent | None = None,
        prepare_root: str | None = None,
    ) -> SandboxLeaseProvider:
        """Get the cached lease provider for a lifecycle key or create one.

        Creation is serialized per key with release-to-zero cleanup, so a new
        provider can never create worker sandboxes while an old provider's
        workers are still being deleted. Holds ``_lifecycle_locked`` exactly
        once for the whole check-then-create sequence; the internal builder
        it calls on a cache miss takes no lock of its own.

        A cached provider is handed out only after the requested mount
        intent passes the same spec-cache gate a cached *sandbox* must pass
        (``_get_or_create_sandbox_locked``'s first gate): the provider owns
        the mount intent its primary container was built from and hands it
        to every worker it later creates, so returning it to a caller
        wanting different mounts would silently serve the first caller's
        container — the exact silent adoption that gate exists to reject.
        A cached provider always implies a cached spec for its primary
        (every pop site drops both together), so the comparison never
        degrades to "no cached spec, allow".

        ``prepare_root``, when given, is only consulted on that cache miss
        (a fresh creation attempt) — see ``ChatWorkspaceBinding.prepare_root``
        and ``_prepare_workspace_mounts``; a cache hit never creates
        directories at all.
        """
        base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        async with self._lifecycle_locked(base_name):
            async with self._activity_guard:
                provider = self._lease_providers.get(base_name)
                if provider is not None:
                    self._ensure_config_equivalent(
                        sandbox_name,
                        self._config_cache.get(sandbox_name),
                        self._build_runtime_spec(
                            lifecycle_type, lifecycle_id, mount_intent=mount_intent
                        ),
                    )
                    self._touch_locked(base_name)
                    return provider

            provider = await self._create_lease_provider_locked(
                lifecycle_type,
                lifecycle_id,
                mount_intent=mount_intent,
                prepare_root=prepare_root,
            )

            async with self._activity_guard:
                self._lease_providers[base_name] = provider
                self._touch_locked(base_name)
            return provider

    async def delete_sandbox(self, lifecycle_type: str, lifecycle_id: str) -> None:
        """
        Delete sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
        """
        sandbox_names = await self._find_lifecycle_sandbox_names(
            lifecycle_type,
            lifecycle_id,
            include_primary=True,
            include_workers=True,
        )
        await self._delete_sandbox_names(sandbox_names)

    async def delete_worker_sandboxes(
        self, lifecycle_type: str, lifecycle_id: str
    ) -> None:
        """Delete worker sandboxes for a lifecycle while preserving the primary."""
        sandbox_names = await self._find_lifecycle_sandbox_names(
            lifecycle_type,
            lifecycle_id,
            include_primary=False,
            include_workers=True,
        )
        await self._delete_sandbox_names(sandbox_names)

    async def _find_lifecycle_sandbox_names(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        include_primary: bool,
        include_workers: bool,
    ) -> set[str]:
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        worker_prefix = self._worker_sandbox_prefix(lifecycle_type, lifecycle_id)
        sandbox_names = {
            name
            for name in self._cache
            if (include_primary and name == sandbox_name)
            or (include_workers and name.startswith(worker_prefix))
        }
        if include_primary:
            sandbox_names.add(sandbox_name)

        try:
            listed_sandboxes = await self._service.list_sandboxes()
        except Exception as exc:
            logger.warning("Failed to list sandboxes for cleanup: %s", exc)
            return sandbox_names

        for sb in listed_sandboxes or []:
            name = sb.name
            if not isinstance(name, str):
                continue
            if include_primary and name == sandbox_name:
                sandbox_names.add(name)
            elif include_workers and name.startswith(worker_prefix):
                sandbox_names.add(name)

        return sandbox_names

    async def _delete_sandbox_names(self, sandbox_names: set[str]) -> None:
        for name in sorted(sandbox_names):
            deleted = False
            try:
                await self._service.delete(name)
                logger.debug(f"Sandbox deleted: {name}")
                deleted = True
            except Exception as e:
                logger.error(f"Failed to delete sandbox {name}: {e}")
            finally:
                # Always evict from cache — even on failure the instance
                # may be in an unknown state and should be recreated.
                self._cache.pop(name, None)
                self._config_cache.pop(name, None)
                self._locks.pop(name, None)
                # Only primary names appear in these maps; worker names no-op.
                # Plain pops on purpose: each is a single synchronous
                # operation with no compound invariant, and awaiting
                # ``_activity_guard`` inside ``finally`` would risk skipping
                # the eviction entirely if the task were cancelled at that
                # await point.
                self._lease_providers.pop(name, None)
                self._activity.pop(name, None)
            if not deleted:
                # The backend delete failed, so the container this budget
                # is scoped to is still live: leave the entry exhausted
                # rather than granting the next mismatch a fresh rebuild
                # allowance for a container that never actually went away.
                continue
            # The reconcile budget is keyed by base (primary) name and
            # shared with every worker under it, so it is only dropped
            # here when the primary itself is the name just deleted
            # (``delete_sandbox``'s full-lifecycle set) — never on a
            # worker-only delete (``delete_worker_sandboxes`` / release-
            # to-zero), where the primary, and the budget scoped to it,
            # is still live. Every name reaching this point came from
            # ``make_sandbox_name``/a worker-prefix filter, so it always
            # contains ``"::"`` and ``parse_sandbox_name`` cannot raise.
            lifecycle_type, lifecycle_id = self.parse_sandbox_name(name)
            base_name = self._base_sandbox_name(lifecycle_type, lifecycle_id)
            if name == base_name:
                self._reconcile_budget.pop(base_name, None)

    async def sweep_idle_sandboxes(self, idle_ttl: float) -> list[str]:
        """Delete sandboxes with no attached tasks that are idle past the TTL.

        Candidates come from both the in-memory activity map and the sandbox
        service listing, so containers surviving a backend restart are also
        reclaimed: with no recorded activity they report idle since manager
        startup and get one TTL grace period.

        Each deletion re-checks ref-count and idle time under the per-key
        lifecycle lock, and the eviction decision plus provider removal are
        atomic under the activity guard, so a sweep can never delete a
        sandbox a task is concurrently attaching or recreating. Workspace data lives on bind
        mounts and survives; the next use recreates the sandbox.

        Args:
            idle_ttl: Idle threshold in seconds (> 0).

        Returns:
            Primary sandbox names that were reclaimed.
        """
        try:
            listed_sandboxes = await self._service.list_sandboxes()
        except Exception as exc:
            logger.warning("Failed to list sandboxes for idle sweep: %s", exc)
            listed_sandboxes = []

        # Only keys with an existing container (or cached instance) are
        # candidates; activity entries alone have nothing left to reclaim.
        candidates: set[str] = set()
        listed_names = [
            sb.name for sb in listed_sandboxes or [] if isinstance(sb.name, str)
        ]
        for name in [*listed_names, *self._cache]:
            try:
                lifecycle_type, lifecycle_id = self.parse_sandbox_name(name)
            except ValueError:
                continue
            candidates.add(self._base_sandbox_name(lifecycle_type, lifecycle_id))

        reclaimed: list[str] = []
        for base_name in sorted(candidates):
            try:
                lifecycle_type, lifecycle_id = self.parse_sandbox_name(base_name)
            except ValueError:
                continue

            async with self._lifecycle_locked(base_name):
                idle_for = time.monotonic() - self.last_activity_at(
                    lifecycle_type, lifecycle_id
                )
                if idle_for <= idle_ttl:
                    continue

                # _evict_idle_sandbox re-validates the ref-count and drops
                # the provider in one atomic step under the activity guard:
                # an attach can never land between the check and the pop
                # that makes attaches fail.
                if await self._evict_idle_sandbox(
                    base_name,
                    reason=f"idle for {idle_for:.0f}s, TTL {idle_ttl:.0f}s",
                ):
                    reclaimed.append(base_name)

        return reclaimed

    async def run_idle_sweep_loop(self) -> None:
        """Periodically reclaim idle sandboxes until cancelled.

        Reads XAGENT_SANDBOX_IDLE_TTL / XAGENT_SANDBOX_SWEEP_INTERVAL; when
        no TTL is configured the loop exits immediately and behavior is
        identical to deployments without idle reclamation.
        """
        idle_ttl = get_sandbox_idle_ttl()
        if idle_ttl is None:
            logger.debug("Sandbox idle reclamation disabled (no TTL configured)")
            return

        sweep_interval = get_sandbox_sweep_interval()
        logger.info(
            "Sandbox idle reclamation enabled: TTL %.0fs, sweep interval %.0fs",
            idle_ttl,
            sweep_interval,
        )
        while True:
            await asyncio.sleep(sweep_interval)
            try:
                reclaimed = await self.sweep_idle_sandboxes(idle_ttl)
                if reclaimed:
                    logger.info(
                        "Idle sweep reclaimed %d sandbox(es): %s",
                        len(reclaimed),
                        ", ".join(reclaimed),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Idle sandbox sweep failed: %s", exc)

    async def warmup(self) -> None:
        """
        Warmup default image.
        Uses empty config for warmup to avoid unnecessary volume mounts.
        """
        image = get_sandbox_image()
        warmup_name = "__warmup__"
        try:
            template = SandboxTemplate(type="image", image=image)
            # Use empty config for warmup - no need for volumes/env
            warmup_config = SandboxConfig()
            async with await self._service.get_or_create(
                warmup_name, template=template, config=warmup_config
            ):
                pass
            await self._service.delete(warmup_name)
            logger.info(f"Sandbox image warmup completed: {image}")
        except Exception as e:
            logger.error(f"Failed to warmup sandbox image: {e}")

    async def cleanup(self) -> None:
        """Stop all running sandboxes and reset process-local caches.

        Routes on the cached backend-capability probe: backends that
        support spec reconciliation (Docker today) get ``_quiesce`` — no
        config-diff guessing, since first-use reconciliation now owns that
        decision entirely (running both would be two competing judges of
        the same question). Backends that do not (Boxlite today) keep
        ``_legacy_cleanup``, unchanged.
        """
        if await self._resolve_backend_probe():
            await self._quiesce()
            return
        await self._legacy_cleanup()

    async def _quiesce(self) -> None:
        """Stop every managed running sandbox; reset process-local caches.

        Deliberately does not delete or otherwise inspect any container's
        configuration: whether a stopped container's desired spec still
        matches is the reconciliation matrix's job on next use, not this
        method's. This intentionally replaces the legacy config-diff
        delete-guessing for this backend (see ``_legacy_cleanup``).
        """
        try:
            sandboxes = await self._service.list_sandboxes()
        except Exception as exc:
            logger.error(f"Failed to list sandboxes for quiesce: {exc}")
            sandboxes = []

        for sb in sandboxes or []:
            if sb.state != "running":
                continue
            try:
                await self._service.stop_existing(
                    sb.name, timeout=_SANDBOX_STOP_TIMEOUT_SECONDS
                )
                logger.debug(f"Stopped sandbox: {sb.name}")
            except Exception as exc:
                logger.error(f"Failed to stop sandbox {sb.name} during quiesce: {exc}")

        self._cache.clear()
        self._config_cache.clear()
        self._lease_providers.clear()
        self._activity.clear()
        self._reconcile_budget.clear()
        logger.info("Sandbox quiesce completed")

    async def _legacy_cleanup(self) -> None:
        """Stop all running sandboxes.

        Delete sandboxes whose config (image, cpus, memory, volumes)
        differs from the current environment so they get recreated
        with the correct settings next time.

        Used only when the backend does not support spec reconciliation
        (Boxlite today): config-diff delete-guessing, as opposed to
        ``_quiesce``'s stop-only behavior for backends that do.

        Note:
            If ``get_uploads_dir()`` (via ``XAGENT_UPLOADS_DIR`` env var) changes
            between deployments, all user sandboxes will be detected as
            having stale volume mounts and will be deleted for recreation.
        """
        try:
            sandboxes = await self._service.list_sandboxes()
            if not sandboxes:
                logger.info("No sandboxes to clean up")
                return

            image, config = self._get_sandbox_image_and_config()

            for sb in sandboxes:
                try:
                    lifecycle_type, lifecycle_id = None, None
                    try:
                        lifecycle_type, lifecycle_id = self.parse_sandbox_name(sb.name)
                    except ValueError:
                        # Not a normal managed sandbox name, stop
                        if sb.state == "running":
                            box = await self._service.get_or_create(
                                sb.name, template=sb.template, config=sb.config
                            )
                            await box.stop()
                            logger.debug(f"Stopped sandbox: {sb.name}")
                        continue

                    # Delete sandbox if config changed (force recreate on next start)
                    image_changed = sb.template.image != image
                    cpus_changed = sb.config.cpus != config.cpus
                    memory_changed = sb.config.memory != config.memory

                    # volumes comparison: None and empty list are treated as equal, ignore order
                    old_volumes = sb.config.volumes or []

                    default_volumes = self._make_default_volumes(
                        lifecycle_type, lifecycle_id
                    )
                    config_volumes = list(config.volumes) if config.volumes else []
                    # Merge volumes
                    new_volumes = config_volumes + default_volumes

                    volumes_changed = set(old_volumes) != set(new_volumes)

                    # env comparison: None and empty dict are treated as equal
                    old_env = sb.config.env or {}
                    new_env = config.env or {}
                    env_changed = old_env != new_env

                    if (
                        image_changed
                        or cpus_changed
                        or memory_changed
                        or volumes_changed
                        or env_changed
                    ):
                        changes = []
                        if image_changed:
                            changes.append(f"image: {sb.template.image} -> {image}")
                        if cpus_changed:
                            changes.append(f"cpus: {sb.config.cpus} -> {config.cpus}")
                        if memory_changed:
                            changes.append(
                                f"memory: {sb.config.memory} -> {config.memory}"
                            )
                        if env_changed:
                            old_env_str = (
                                ";".join([f"{k}={v}" for k, v in old_env.items()])
                                if old_env
                                else "none"
                            )
                            new_env_str = (
                                ";".join([f"{k}={v}" for k, v in new_env.items()])
                                if new_env
                                else "none"
                            )
                            changes.append(f"env: {old_env_str} -> {new_env_str}")
                        if volumes_changed:
                            old_vol_str = (
                                ";".join([f"{h}:{g}:{m}" for h, g, m in old_volumes])
                                if old_volumes
                                else "none"
                            )
                            new_vol_str = (
                                ";".join([f"{h}:{g}:{m}" for h, g, m in new_volumes])
                                if new_volumes
                                else "none"
                            )
                            changes.append(f"volumes: {old_vol_str} -> {new_vol_str}")
                        logger.info(
                            f"Config changed for sandbox [{sb.name}]: "
                            f"{', '.join(changes)}, deleting"
                        )
                        await self._service.delete(sb.name)
                        continue

                    # Stop running sandboxes with matching image
                    if sb.state == "running":
                        box = await self._service.get_or_create(
                            sb.name, template=sb.template, config=sb.config
                        )
                        await box.stop()
                        logger.debug(f"Stopped sandbox: {sb.name}")
                except Exception as e:
                    logger.error(f"Failed to handle sandbox {sb.name}: {e}")

            self._cache.clear()
            self._config_cache.clear()
            self._locks.clear()
            self._lease_providers.clear()
            self._activity.clear()
            logger.info("Sandbox cleanup completed")
        except Exception as e:
            logger.error(f"Failed to cleanup sandboxes: {e}")


def _check_no_conflicting_readiness_volumes(
    volumes: list[tuple[str, str, str]],
) -> None:
    """Reject readiness volumes that disagree at a shared host or guest path.

    Two directions, both against the same normalized triple set:

    - host conflict: two entries share a host path but disagree on guest
      path or mode (the backend indexes bind mounts by host path and would
      silently drop one of them).
    - guest conflict ("guest crash"): two entries share a guest path but
      disagree on host source (Docker rejects the duplicate mount point at
      container creation; failing here at startup surfaces the bad static
      configuration before any task is accepted).

    An exactly identical triple repeated across sources (e.g. the same
    directory named twice) collapses onto itself in both directions and is
    legal. Paths are canonicalized through ``canonical_sandbox_path``, the
    same owner the desired spec uses, so this check groups exactly the
    spellings the backend itself would collapse onto one mount point.
    """
    seen_by_host: dict[str, tuple[str, str]] = {}
    seen_by_guest: dict[str, tuple[str, str]] = {}
    for host_path, guest_path, mode in volumes:
        norm_host = canonical_sandbox_path(host_path)
        norm_guest = canonical_sandbox_path(guest_path)

        host_key = (norm_guest, mode)
        prior_for_host = seen_by_host.get(norm_host)
        if prior_for_host is not None and prior_for_host != host_key:
            raise SandboxRuntimeConflictError(
                f"Conflicting sandbox volume mounts for host path "
                f"{norm_host!r}: {prior_for_host} vs {host_key}"
            )
        seen_by_host[norm_host] = host_key

        guest_key = (norm_host, mode)
        prior_for_guest = seen_by_guest.get(norm_guest)
        if prior_for_guest is not None and prior_for_guest != guest_key:
            raise SandboxRuntimeConflictError(
                f"Conflicting sandbox volume mounts for guest path "
                f"{norm_guest!r}: {prior_for_guest} vs {guest_key}"
            )
        seen_by_guest[norm_guest] = guest_key


def _reserved_uploads_user_subtree(
    path_mapper: "SandboxPathMapper",
) -> tuple[str, str]:
    """Return ``(guest_uploads_root, reserved_prefix)`` for the per-user subtree.

    Every task's default volumes reserve
    ``<guest_uploads_root>/<reserved_prefix><user_id>`` for that user's own
    workspace mount (see ``_workspace_mount_paths`` / ``_make_default_volumes``,
    both built on ``scoped_user_root``); the id is a runtime fact, not
    enumerable at startup, so readiness instead treats the whole
    ``<reserved_prefix>\\d+`` shape under the uploads guest root as reserved.
    ``reserved_prefix`` is read off ``scoped_user_root``'s own output for two
    sentinel ids (never a hardcoded ``"user_"`` literal), so a rename of that
    naming scheme cannot silently desync this check from the path it
    protects: a single sentinel cannot distinguish "the id is the trailing
    token" from "the id happens to end in a digit that looks like part of
    the prefix" (e.g. a zero-padded ``user_0000`` would strip only one
    trailing ``"0"``), so two sentinels are compared and the derived prefix
    is verified to actually reproduce both before it is trusted. The uploads
    root is mapped into the same host/guest domain the runtime mount-
    building path uses: ``absolute_backend_mount_path`` then
    ``SandboxPathMapper``.

    Raises:
        SandboxContractError: the derived prefix does not reproduce both
            sentinel names, meaning the naming scheme no longer fits the
            ``<prefix><id>`` shape this check assumes — better to fail
            startup than silently protect nothing.
    """
    uploads_root = get_uploads_dir()
    backend_uploads_root = absolute_backend_mount_path(uploads_root)
    _host, guest_uploads_root, _mode = path_mapper.volume_for_backend_path(
        backend_uploads_root, "rw"
    )
    sample_0 = scoped_user_root(uploads_root, 0).name
    sample_1 = scoped_user_root(uploads_root, 1).name
    common_len = 0
    for a, b in zip(sample_0, sample_1):
        if a != b:
            break
        common_len += 1
    reserved_prefix = sample_0[:common_len]
    # An empty prefix satisfies the reproduction check below trivially and
    # leaves a bare ``\d+`` pattern that claims every numeric directory name
    # under the uploads root -- the silent mismatch this derivation exists to
    # prevent, arising at its own boundary.
    if (
        not reserved_prefix
        or reserved_prefix + "0" != sample_0
        or reserved_prefix + "1" != sample_1
    ):
        raise SandboxContractError(
            "Cannot derive a reserved per-user uploads prefix from "
            f"scoped_user_root output {sample_0!r} / {sample_1!r}: they do not "
            "reproduce from a non-empty common prefix plus their sentinel id. "
            "The per-user naming scheme no longer fits the <prefix><id> shape "
            "the reserved-uploads readiness check assumes."
        )
    return canonical_sandbox_path(guest_uploads_root), reserved_prefix


def _check_no_reserved_uploads_conflict(
    volumes: list[tuple[str, str, str]],
    guest_uploads_root: str,
    reserved_prefix: str,
) -> None:
    """Reject a configured mount whose guest path IS a reserved per-user dir.

    The per-user workspace mount every task adds by default cannot be
    enumerated here — user ids are runtime facts — so the exact
    ``<guest_uploads_root>/<reserved_prefix><id>`` guest path is treated as
    reserved: a static mount claiming that exact guest path collides with
    the mount every future task for that user needs there, which today only
    surfaces as ``SandboxRuntimeConflictError`` at that user's first task
    rather than here at startup.

    A mount nested *under* a reserved directory (e.g.
    ``<uploads>/user_1/models``) is a distinct guest path and is NOT
    rejected: nested bind mounts at different guest paths are legal —
    ``_check_no_conflicting_volumes`` (the per-create check) and Docker
    itself only flag exact guest-path collisions, never parent/child
    nesting — and a deployment-named ``XAGENT_EXTERNAL_UPLOAD_DIRS`` entry
    landing there is exactly the exception ``_fold_mount_paths`` implements
    (see ``_MountCandidate``'s ``"deployment"`` provenance in
    ``workspace_binding.py``): an operator-named mount keeps its own bind
    regardless of where it falls relative to the reserved subtree. A mount
    elsewhere under the uploads root (e.g. a shared knowledge-base
    directory) does not match the reserved shape either and still passes.
    """
    reserved_re = re.compile(rf"^{re.escape(reserved_prefix)}\d+$")
    prefix_with_sep = guest_uploads_root + "/"
    for _host_path, guest_path, _mode in volumes:
        norm_guest = canonical_sandbox_path(guest_path)
        if not norm_guest.startswith(prefix_with_sep):
            continue
        remainder = norm_guest[len(prefix_with_sep) :]
        if "/" in remainder:
            # Nested under a reserved directory, not the directory itself.
            continue
        if reserved_re.match(remainder):
            raise SandboxRuntimeConflictError(
                f"Configured sandbox mount with guest path {guest_path!r} "
                "collides with the reserved per-user uploads path "
                f"{guest_uploads_root!r}/{reserved_prefix}<id>: every task's "
                "default workspace mount for that user needs that exact "
                "guest path."
            )


async def check_sandbox_static_readiness(sandbox_mgr: "SandboxManager") -> None:
    """Validate static sandbox mount configuration before serving traffic.

    Called once at app startup, before ``cleanup()``/``warmup()``: a
    misconfigured deployment fails to start instead of only surfacing as a
    per-task ``SandboxRuntimeConflictError`` once real workloads land. Also
    resolves and caches ``sandbox_mgr``'s backend-capability probe as a side
    effect (via ``_resolve_backend_probe()``), so the cleanup step that runs
    right after reads the cached value instead of resolving it again.

    Only meaningful for backends that support spec reconciliation (Docker
    today): the legacy Boxlite route never reconciles a desired spec against
    a live container, so there is nothing here for it to protect. Skipped
    entirely when neither ``SANDBOX_VOLUMES`` nor
    ``XAGENT_EXTERNAL_UPLOAD_DIRS`` is configured — code mounts alone never
    conflict with themselves, and they never land under the uploads guest
    root either (``build_code_mount_volumes`` mounts fixed ``/app/src`` and
    ``/app/tests`` paths), so with nothing operator-configured the per-user
    reserved-subtree check below has nothing to reject; that is also why the
    reserved-subtree check runs after this early return rather than before
    it. When the check does run, code mounts are still part of the scanned
    volume list along with everything else — they are not specially
    exempted, they simply do not match the reserved shape.

    Domain discipline: ``SANDBOX_VOLUMES`` (via ``get_sandbox_volumes``,
    using the same ``host_side_sources`` flag the runtime build path uses)
    and code mounts (``build_code_mount_volumes``) are already host-domain
    triples and are compared as-is. External upload dirs are backend-domain
    paths: they are absolutized through ``absolute_backend_mount_path`` (the
    same owner the runtime mount-building path uses, so a relative
    ``XAGENT_EXTERNAL_UPLOAD_DIRS`` entry is checked as the directory it
    actually names) and folded (normalized + deduplicated, the same
    backend-domain normalization ``SandboxMountIntent`` itself applies)
    before being converted to host domain through the same
    ``SandboxPathMapper`` the runtime mount-building path uses. Conflict
    detection itself always runs in the post-mapper host domain, over the
    combined triple set from all three sources.

    Beyond mutual conflicts among the configured mounts, one configured
    mount can also collide with a mount that does not exist yet: every
    task's per-user uploads mount (see ``_workspace_mount_paths`` /
    ``_make_default_volumes``), whose user id is a runtime fact this
    startup step cannot enumerate. ``_check_no_reserved_uploads_conflict``
    catches that case instead by treating the whole per-user subtree shape
    under the uploads guest root as reserved, so a configured mount landing
    there fails now instead of at that user's first task.

    Raises:
        SandboxRuntimeConflictError: Two configured mounts disagree at a
            shared host or guest path, or a configured mount lands inside
            the reserved per-user uploads subtree.
    """
    if not await sandbox_mgr._resolve_backend_probe():
        return

    external_dirs = get_external_upload_dirs()
    if not os.getenv(SANDBOX_VOLUMES, "").strip() and not external_dirs:
        return

    path_mapper = SandboxPathMapper.from_env()
    volumes = list(
        get_sandbox_volumes(host_side_sources=path_mapper.uses_host_storage_root)
    )
    volumes.extend(build_code_mount_volumes())

    folded_external = SandboxMountIntent(
        extra_mounts=tuple(str(absolute_backend_mount_path(d)) for d in external_dirs)
    ).extra_mounts
    for backend_dir in folded_external:
        volumes.append(path_mapper.volume_for_backend_path(backend_dir, "rw"))

    _check_no_conflicting_readiness_volumes(volumes)
    guest_uploads_root, reserved_prefix = _reserved_uploads_user_subtree(path_mapper)
    _check_no_reserved_uploads_conflict(volumes, guest_uploads_root, reserved_prefix)


# Global sandbox manager instance
_sandbox_manager: Optional[SandboxManager] = None
_sandbox_manager_lock = threading.Lock()
_sandbox_manager_initialized = False


def _create_sandbox_service() -> Optional[SandboxService]:
    """
    Create sandbox service based on environment configuration.

    Environment variables:
    - SANDBOX_ENABLED: Enable/disable sandbox (default: true)
    - SANDBOX_IMPLEMENTATION: Implementation type (default: docker)
      - docker: Use Docker sandbox
      - boxlite: Use Boxlite sandbox
    - BOXLITE_HOME_DIR: Boxlite home directory (optional)

    Returns:
        SandboxService instance or None if disabled
    """
    # Check if sandbox is enabled
    sandbox_enabled = os.getenv("SANDBOX_ENABLED", "false").lower() == "true"
    if not sandbox_enabled:
        logger.info("Sandbox is disabled via SANDBOX_ENABLED environment variable")
        return None

    # Get implementation type
    implementation = os.getenv("SANDBOX_IMPLEMENTATION", "docker")

    if implementation == "boxlite":
        return _create_boxlite_service()
    elif implementation == "docker":
        return _create_docker_service()
    else:
        logger.warning(
            f"Unknown sandbox implementation: {implementation}, falling back to docker"
        )
        return _create_docker_service()


def _create_boxlite_service() -> Optional[SandboxService]:
    """Create Boxlite sandbox service."""
    try:
        from ..sandbox import BoxliteSandboxService
    except ImportError:
        logger.error("boxlite is not installed.")
        return None

    from .sandbox_store import DBBoxliteStore

    store = DBBoxliteStore()
    # Get home directory
    home_dir = get_boxlite_home_dir()

    service = None
    try:
        service = BoxliteSandboxService(
            store=store, home_dir=None if home_dir is None else str(home_dir)
        )
        logger.info(
            f"Created Boxlite sandbox service (home_dir={home_dir or 'default'})"
        )
    except Exception as e:
        logger.error(f"Failed to create Boxlite sandbox service: {e}")

    return service


def _create_docker_service() -> Optional[SandboxService]:
    """Create Docker sandbox service."""
    try:
        from ..sandbox import DockerSandboxService
    except ImportError:
        logger.error("docker sandbox dependencies are not installed.")
        return None

    from .sandbox_store import DBDockerStore

    store = DBDockerStore()

    service = None
    try:
        service = DockerSandboxService(store=store)
        logger.info("Created Docker sandbox service")
    except Exception as e:
        logger.error(f"Failed to create Docker sandbox service: {e}")

    return service


def get_sandbox_manager() -> Optional[SandboxManager]:
    """
    Get or create global sandbox manager instance.

    Thread-safe singleton pattern with double-checked locking.

    Returns:
        SandboxManager instance or None if sandbox is disabled
    """
    global _sandbox_manager, _sandbox_manager_initialized

    # Fast path: already initialized (either successfully or service was None)
    if _sandbox_manager_initialized:
        return _sandbox_manager

    # Slow path: need to initialize
    with _sandbox_manager_lock:
        # Double-check after acquiring lock
        if _sandbox_manager_initialized:
            return _sandbox_manager

        # Get sandbox service
        service = _create_sandbox_service()
        if service is None:
            _sandbox_manager_initialized = True
            return None

        # Create sandbox manager
        _sandbox_manager = SandboxManager(service)
        _sandbox_manager_initialized = True
        logger.info("Created global sandbox manager")

        return _sandbox_manager
