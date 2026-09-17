"""Shared fake ``SandboxService`` for ``SandboxManager`` unit tests.

Before this module existed, manager tests each rolled their own stand-in for
the injected ``SandboxService``: a bare ``AsyncMock()``, a bare
``MagicMock()``, or a hand-written ``_FakeService`` class that implemented
only the legacy methods a given test file happened to touch. None of those
three shapes are safe once ``SandboxManager`` starts routing on
``await service.supports_runtime_spec()``: an unconfigured ``AsyncMock()``
returns a truthy child mock for that call (never the real default ``False``),
a bare ``MagicMock()`` cannot be awaited at all, and a hand-written fake
without the method simply raises ``AttributeError``. All three would
silently misroute or crash under that gate.

``FakeSandboxService`` fixes this by actually inheriting ``SandboxService``.
With ``runtime_spec_supported=False`` (the default) it carries the exact
same production defaults a real legacy-only backend would: awaiting
``supports_runtime_spec()`` returns ``False`` (a real ``bool``, not a mock),
and ``inspect``/``create``/``start_existing``/``stop_existing``/
``get_store_record``/``persist_store_record`` are left un-overridden, so
they inherit the base class's own defaults (raise
``SandboxReconcileUnsupportedError`` for the first four; return
``None``/no-op for the store methods).

With ``runtime_spec_supported=True`` this instance additionally gets a
small in-memory reconciliation backend: ``create()`` records a container
with an attested fingerprint/version label pair (mirroring
``DockerSandboxService.create()``), ``inspect()`` reads it back as a
``SandboxInspection``, ``start_existing``/``stop_existing`` flip its
state, and ``get_store_record``/``persist_store_record`` read/write an
independent store dict — independent on purpose, so a test can construct
"label present, store row absent" (and vice versa) scenarios directly by
poking ``_containers``/``_store`` without going through ``create()`` at
all (e.g. to simulate a legacy-created container with no fingerprint
label but a store row).

All eleven methods are wrapped in ``AsyncMock(wraps=...)`` on the instance,
so tests keep full ``unittest.mock`` call-tracking
(``assert_awaited_once_with``, ``await_args_list``, ``.side_effect =``,
``.return_value =``) while the default behavior (when a test does not
override return_value/side_effect) calls through to this class's own
small in-memory implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
from unittest.mock import AsyncMock, MagicMock

from xagent.sandbox.base import (
    SPEC_CONTRACT_VERSION,
    ObservedRuntimeFacts,
    ResolvedSandboxRuntimeSpec,
    Sandbox,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxInfo,
    SandboxInspection,
    SandboxNotFoundError,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
)


@dataclass
class _FakeReconcileContainer:
    """One in-memory container tracked by the reconciliation-lifecycle fake."""

    state: str  # "running" | "stopped"
    spec: ResolvedSandboxRuntimeSpec
    fingerprint_label: Optional[str] = None
    version_label: Optional[str] = None


class FakeSandboxService(SandboxService):
    """In-memory ``SandboxService`` stand-in for manager-level tests."""

    def __init__(
        self,
        initial: Iterable[str] = (),
        *,
        runtime_spec_supported: bool = False,
    ) -> None:
        self.containers: set[str] = set(initial)
        self.peak = len(self.containers)
        self.deleted: list[str] = []
        self.snapshots: dict[str, SandboxSnapshot] = {}

        # Wrap the concrete bodies below in AsyncMock so tests get the full
        # unittest.mock spy/override API on the instance while default
        # behavior still calls through to this class's own implementation.
        self.get_or_create = AsyncMock(wraps=self.get_or_create)
        self.list_sandboxes = AsyncMock(wraps=self.list_sandboxes)
        self.delete = AsyncMock(wraps=self.delete)
        self.supports_snapshots = AsyncMock(wraps=self.supports_snapshots)
        self.create_snapshot = AsyncMock(wraps=self.create_snapshot)
        self.list_snapshots = AsyncMock(wraps=self.list_snapshots)
        self.delete_snapshot = AsyncMock(wraps=self.delete_snapshot)

        # Reconciliation-lifecycle state. Only meaningful (and only
        # populated) when runtime_spec_supported=True; independent of
        # `containers`/`deleted` above (the legacy-route bookkeeping),
        # which `create`/`delete` below also keep updated so a single test
        # can exercise both routes' assertions if it needs to.
        self._containers: dict[str, _FakeReconcileContainer] = {}
        self._store: dict[str, SandboxInfo] = {}

        # Bound only per-instance, never at class level: FakeSandboxService
        # must not shadow SandboxService's own raise-by-default bodies for
        # any other (runtime_spec_supported=False) instance — see
        # test_fake_service_contract.py's class-level identity pin. The
        # concrete implementations live on this class as `_*_impl` methods
        # precisely so they can be wrapped and bound here without ever
        # being reachable as `FakeSandboxService.inspect` etc.
        if runtime_spec_supported:
            self.supports_runtime_spec = AsyncMock(return_value=True)
            self.inspect = AsyncMock(wraps=self._inspect_impl)
            self.create = AsyncMock(wraps=self._create_impl)
            self.start_existing = AsyncMock(wraps=self._start_existing_impl)
            self.stop_existing = AsyncMock(wraps=self._stop_existing_impl)
            self.get_store_record = AsyncMock(wraps=self._get_store_record_impl)
            self.persist_store_record = AsyncMock(wraps=self._persist_store_record_impl)

    # --- legacy lifecycle: concrete bodies (also satisfies SandboxService's
    # abstract methods so this class is instantiable) ---

    async def get_or_create(
        self,
        name: str,
        template: Optional[SandboxTemplate] = None,
        config: Optional[SandboxConfig] = None,
    ) -> Sandbox:
        self.containers.add(name)
        self.peak = max(self.peak, len(self.containers))
        sandbox = MagicMock()
        sandbox.name = name
        return sandbox

    async def list_sandboxes(self) -> list[SandboxInfo]:
        return [
            SandboxInfo(
                name=name,
                # Reconciliation-lifecycle containers (populated via
                # create()/_start_existing_impl/_stop_existing_impl) report
                # their real, current state; legacy-route names never
                # tracked in ``_containers`` keep the historical "stopped"
                # stub, since the legacy tests that rely on this fake never
                # touch real container state either.
                state=(
                    self._containers[name].state
                    if name in self._containers
                    else "stopped"
                ),
                template=SandboxTemplate(type="image", image="img:v1"),
                config=SandboxConfig(),
            )
            for name in sorted(self.containers)
        ]

    async def delete(self, name: str) -> None:
        self.containers.discard(name)
        self.deleted.append(name)
        self._containers.pop(name, None)
        self._store.pop(name, None)

    async def supports_snapshots(self) -> bool:
        return False

    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        snapshot = SandboxSnapshot(snapshot_id=snapshot_id)
        self.snapshots[snapshot_id] = snapshot
        return snapshot

    async def list_snapshots(self) -> list[SandboxSnapshot]:
        return list(self.snapshots.values())

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self.snapshots.pop(snapshot_id, None)

    # --- spec-reconciliation lifecycle: concrete `_*_impl` bodies. Bound to
    # the public names only per-instance (see __init__) when
    # runtime_spec_supported=True; a plain FakeSandboxService() inherits
    # SandboxService's own raise-by-default (or None/no-op) bodies for the
    # public names, keeping the class-level identity pin in
    # test_fake_service_contract.py true.

    def _inspection_for(self, container: _FakeReconcileContainer) -> SandboxInspection:
        spec = container.spec
        facts = ObservedRuntimeFacts(
            raw_status=container.state,
            image_ref=spec.image,
            image_digest=spec.image,
            raw_nano_cpus=int(spec.cpus * 1_000_000_000),
            raw_memory_bytes=int(spec.memory * 1024 * 1024),
            env=dict(spec.env),
            volumes=spec.volumes,
            ports=spec.ports,
            network_isolated=spec.network_isolated,
            runtime_networks=(),
            labels={},
            created_at=None,
            working_dir=spec.working_dir,
        )
        return SandboxInspection(
            state="running" if container.state == "running" else "stopped",
            facts=facts,
            fingerprint_label=container.fingerprint_label,
            version_label=container.version_label,
        )

    async def _inspect_impl(self, name: str) -> Optional[SandboxInspection]:
        container = self._containers.get(name)
        if container is None:
            return None
        return self._inspection_for(container)

    async def _create_impl(
        self, name: str, template: SandboxTemplate, config: SandboxConfig
    ) -> Sandbox:
        if name in self._containers:
            raise SandboxAlreadyExistsError(f"Sandbox already exists: {name!r}")

        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type=template.type or "image",
            image=template.image,
            snapshot_id=template.snapshot_id,
            working_dir=config.working_dir,
            cpus=config.cpus,
            memory=config.memory,
            env=config.env,
            volumes=config.volumes,
            network_isolated=bool(config.network_isolated),
            ports=config.ports,
        )
        self._containers[name] = _FakeReconcileContainer(
            state="running",
            spec=spec,
            fingerprint_label=spec.fingerprint(),
            version_label=str(SPEC_CONTRACT_VERSION),
        )
        self._store[name] = SandboxInfo(
            name=name, state="running", template=template, config=config
        )

        self.containers.add(name)
        self.peak = max(self.peak, len(self.containers))
        sandbox = MagicMock()
        sandbox.name = name
        return sandbox

    async def _start_existing_impl(self, name: str) -> Sandbox:
        container = self._containers.get(name)
        if container is None:
            raise SandboxNotFoundError(f"Sandbox not found: {name}")
        container.state = "running"
        self.containers.add(name)
        sandbox = MagicMock()
        sandbox.name = name
        return sandbox

    async def _stop_existing_impl(
        self, name: str, *, timeout: Optional[int] = None
    ) -> None:
        container = self._containers.get(name)
        if container is None:
            raise SandboxNotFoundError(f"Sandbox not found: {name}")
        container.state = "stopped"

    async def _get_store_record_impl(self, name: str) -> Optional[SandboxInfo]:
        return self._store.get(name)

    async def _persist_store_record_impl(self, name: str, info: SandboxInfo) -> None:
        self._store[name] = info
