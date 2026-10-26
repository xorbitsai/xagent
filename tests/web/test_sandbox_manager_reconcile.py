"""The spec-reconciliation matrix in ``SandboxManager`` (#296).

Exercises the routing gate (``_resolve_backend_probe``), the reconciliation
matrix cell by cell (absent / MATCH / UNVERIFIED with and without a store
row / MISMATCH with each ref-count and state combination), the per-key
rebuild budget, the ``SandboxAlreadyExistsError`` cross-process retry, and
provider ABA (``attach_provider``). All tests use ``FakeSandboxService(
runtime_spec_supported=True)`` so the manager takes the reconciliation route
end to end against the fake's small in-memory container/store model (see
``tests/web/sandbox_fakes.py``).
"""

from __future__ import annotations

import ast
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import xagent.web.sandbox_manager as sandbox_manager_module
import xagent.web.services.workspace_binding as workspace_binding_module
from tests.web.sandbox_fakes import FakeSandboxService, _FakeReconcileContainer
from xagent.core.execution_scope import ExecutionScope
from xagent.sandbox.base import (
    SPEC_CONTRACT_VERSION,
    ResolvedSandboxRuntimeSpec,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxContractError,
    SandboxInfo,
    SandboxMountIntent,
    SandboxRecoveryRequiredError,
    SandboxRuntimeConflictError,
    SandboxTemplate,
    grant_sandbox_runtime_access,
)
from xagent.web.sandbox_keys import USER_LIFECYCLE_TYPE, make_user_lifecycle_id
from xagent.web.sandbox_manager import SandboxManager, check_sandbox_static_readiness
from xagent.web.services.workspace_binding import build_chat_workspace_binding


@pytest.fixture
def _env(tmp_path):
    """Isolate sandbox env config and neutralize code mounts."""
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/repo/src", "/app/src", "ro")],
        ),
        patch("xagent.web.sandbox_manager.grant_sandbox_runtime_access"),
    ):
        yield


def _intent(tmp_path, name: str = "workspace") -> SandboxMountIntent:
    return SandboxMountIntent(mount_root=str(tmp_path / name))


def _make_manager() -> tuple[SandboxManager, FakeSandboxService]:
    service = FakeSandboxService(runtime_spec_supported=True)
    return SandboxManager(service), service


class TestBackendProbeRouting:
    def test_desired_spec_omits_limits_not_enforced_by_backend(
        self, _env, tmp_path
    ) -> None:
        manager, _service = _make_manager()

        desired = manager._build_runtime_spec(
            "user", "1", mount_intent=_intent(tmp_path)
        )

        assert desired.pids_limit is None
        assert desired.security_profile is None

    @pytest.mark.asyncio
    async def test_probe_resolved_once_and_cached(self, _env, tmp_path) -> None:
        manager, service = _make_manager()

        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))
        await manager.get_or_create_sandbox(
            "user", "2", mount_intent=_intent(tmp_path, "workspace2")
        )

        service.supports_runtime_spec.assert_awaited_once()
        assert manager._backend_probe is True

    @pytest.mark.asyncio
    async def test_legacy_backend_never_calls_reconcile_methods(
        self, _env, tmp_path
    ) -> None:
        """probe=False routes through service.get_or_create(), never
        inspect/create/start_existing/stop_existing."""
        service = FakeSandboxService()  # runtime_spec_supported=False (default)
        manager = SandboxManager(service)

        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )

        assert sandbox.name == "user::1"
        service.get_or_create.assert_awaited_once()


class TestReconcileMatrixAbsent:
    @pytest.mark.asyncio
    async def test_absent_creates(self, _env, tmp_path) -> None:
        manager, service = _make_manager()

        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )

        assert sandbox.name == "user::1"
        service.create.assert_awaited_once()
        assert "user::1" in service._containers
        assert "user::1" in service._store

    @pytest.mark.asyncio
    async def test_already_exists_retries_once_via_reinspect(
        self, _env, tmp_path
    ) -> None:
        """A cross-process race: another process created the name between
        our inspect() and create(). A single re-inspect must recover it
        instead of failing closed."""
        manager, service = _make_manager()

        async def racy_create(name, template, config):
            # Simulate a concurrent process winning the create() race: the
            # container actually gets created (so the re-inspect finds it),
            # but this call itself still reports the conflict.
            await service._create_impl(name, template, config)
            raise SandboxAlreadyExistsError("raced")

        service.create.side_effect = racy_create

        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )

        assert sandbox is not None
        assert service.create.await_count == 1


class TestReconcileMatrixMatch:
    @pytest.mark.asyncio
    async def test_match_with_store_row_reuses(self, _env, tmp_path) -> None:
        manager, service = _make_manager()
        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))
        service.create.reset_mock()
        manager._cache.clear()
        manager._config_cache.clear()

        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )

        assert sandbox.name == "user::1"
        service.create.assert_not_awaited()
        service.start_existing.assert_awaited_once_with("user::1")

    @pytest.mark.asyncio
    async def test_match_without_store_row_backfills_then_reuses(
        self, _env, tmp_path
    ) -> None:
        manager, service = _make_manager()
        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))
        # Simulate create()'s store write having failed independently of
        # the container (label is immutable and unaffected).
        service._store.pop("user::1", None)
        manager._cache.clear()
        manager._config_cache.clear()

        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )

        assert sandbox.name == "user::1"
        service.persist_store_record.assert_awaited_once()
        assert "user::1" in service._store
        service.create.assert_awaited_once()  # only the original creation

    @pytest.mark.asyncio
    async def test_stopped_container_with_matching_label_is_started_not_rebuilt(
        self, _env, tmp_path
    ) -> None:
        """The shape a process restart leaves behind: containers stopped,
        specs unchanged. The first task after the restart must start the
        container it finds, because rebuilding it would discard the sandbox
        state a restart is not supposed to touch — and would do so for every
        lifecycle at once, on the one code path where that is most expensive.
        """
        manager, service = _make_manager()
        intent = _intent(tmp_path)
        desired = manager._build_runtime_spec("user", "1", mount_intent=intent)
        template, config = desired.to_backend_config()
        service._containers["user::1"] = _FakeReconcileContainer(
            state="stopped",
            spec=desired,
            fingerprint_label=desired.fingerprint(),
            version_label=str(SPEC_CONTRACT_VERSION),
        )
        service._store["user::1"] = SandboxInfo(
            name="user::1", state="stopped", template=template, config=config
        )

        sandbox = await manager.get_or_create_sandbox("user", "1", mount_intent=intent)

        assert sandbox.name == "user::1"
        service.start_existing.assert_awaited_once_with("user::1")
        service.create.assert_not_awaited()
        service.delete.assert_not_awaited()
        # Reuse means the container is running afterwards, not merely spared.
        assert service._containers["user::1"].state == "running"


class TestReconcileMatrixUnverified:
    @pytest.mark.asyncio
    async def test_unverified_with_matching_store_row_reuses(
        self, _env, tmp_path
    ) -> None:
        """A legacy-created container (no fingerprint label) whose store
        row's recorded spec matches the freshly desired one converges by
        reuse, staying UNVERIFIED (no label is ever backfilled)."""
        manager, service = _make_manager()
        intent = _intent(tmp_path)
        desired = manager._build_runtime_spec("user", "1", mount_intent=intent)
        template, config = desired.to_backend_config()

        service._containers["user::1"] = _FakeReconcileContainer(
            state="stopped", spec=desired, fingerprint_label=None, version_label=None
        )
        service._store["user::1"] = SandboxInfo(
            name="user::1", state="stopped", template=template, config=config
        )

        sandbox = await manager.get_or_create_sandbox("user", "1", mount_intent=intent)

        assert sandbox.name == "user::1"
        service.start_existing.assert_awaited_once_with("user::1")
        service.create.assert_not_awaited()
        service.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unverified_with_diverging_store_row_rebuilds(
        self, _env, tmp_path
    ) -> None:
        manager, service = _make_manager()
        intent = _intent(tmp_path)
        desired = manager._build_runtime_spec("user", "1", mount_intent=intent)

        service._containers["user::1"] = _FakeReconcileContainer(
            state="stopped", spec=desired, fingerprint_label=None, version_label=None
        )
        service._store["user::1"] = SandboxInfo(
            name="user::1",
            state="stopped",
            template=SandboxTemplate(type="image", image="stale:v0"),
            config=SandboxConfig(),
        )

        sandbox = await manager.get_or_create_sandbox("user", "1", mount_intent=intent)

        assert sandbox.name == "user::1"
        service.delete.assert_awaited_once_with("user::1")
        service.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_v1_store_row_without_hardening_fields_rebuilds(
        self, _env, tmp_path
    ) -> None:
        manager, service = _make_manager()
        service.runtime_pids_limit = lambda: 1024
        service.runtime_security_profile = lambda: "fake-hardening-v1"
        intent = _intent(tmp_path)
        desired = manager._build_runtime_spec("user", "1", mount_intent=intent)
        template, config = desired.to_backend_config()
        service._containers["user::1"] = _FakeReconcileContainer(
            state="stopped", spec=desired, fingerprint_label=None, version_label=None
        )
        service._store["user::1"] = SandboxInfo(
            name="user::1",
            state="stopped",
            template=template,
            config=config.model_copy(
                update={"pids_limit": None, "security_profile": None}
            ),
        )

        await manager.get_or_create_sandbox("user", "1", mount_intent=intent)

        service.delete.assert_awaited_once_with("user::1")
        service.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unverified_with_no_store_row_rebuilds(self, _env, tmp_path) -> None:
        manager, service = _make_manager()
        intent = _intent(tmp_path)
        desired = manager._build_runtime_spec("user", "1", mount_intent=intent)

        service._containers["user::1"] = _FakeReconcileContainer(
            state="stopped", spec=desired, fingerprint_label=None, version_label=None
        )
        # No store row at all.

        sandbox = await manager.get_or_create_sandbox("user", "1", mount_intent=intent)

        assert sandbox.name == "user::1"
        service.delete.assert_awaited_once_with("user::1")
        service.create.assert_awaited_once()


class TestReconcileMatrixMismatch:
    @staticmethod
    def _seed_mismatch(
        service: FakeSandboxService,
        manager: SandboxManager,
        name: str,
        *,
        state: str,
        tmp_path,
    ) -> None:
        stale_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="stale:v0"
        )
        service._containers[name] = _FakeReconcileContainer(
            state=state,
            spec=stale_spec,
            fingerprint_label=stale_spec.fingerprint(),
            version_label="1",
        )

    @pytest.mark.asyncio
    async def test_stopped_mismatch_ref_zero_rebuilds(self, _env, tmp_path) -> None:
        manager, service = _make_manager()
        self._seed_mismatch(
            service, manager, "user::1", state="stopped", tmp_path=tmp_path
        )

        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )

        assert sandbox.name == "user::1"
        service.delete.assert_awaited_once_with("user::1")
        service.create.assert_awaited_once()
        service.stop_existing.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stopped_mismatch_ref_nonzero_rejects_new_caller(
        self, _env, tmp_path
    ) -> None:
        manager, service = _make_manager()
        self._seed_mismatch(
            service, manager, "user::1", state="stopped", tmp_path=tmp_path
        )
        manager._lease_providers["user::1"] = object()
        assert await manager.attach("user", "1")

        with pytest.raises(SandboxRuntimeConflictError):
            await manager.get_or_create_sandbox(
                "user", "1", mount_intent=_intent(tmp_path)
            )

        service.delete.assert_not_awaited()
        service.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_running_mismatch_ref_zero_stops_then_rebuilds(
        self, _env, tmp_path
    ) -> None:
        manager, service = _make_manager()
        self._seed_mismatch(
            service, manager, "user::1", state="running", tmp_path=tmp_path
        )

        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )

        assert sandbox.name == "user::1"
        service.stop_existing.assert_awaited_once_with(
            "user::1", timeout=sandbox_manager_module._SANDBOX_STOP_TIMEOUT_SECONDS
        )
        service.delete.assert_awaited_once_with("user::1")
        service.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_running_mismatch_ref_nonzero_rejects_and_never_deletes(
        self, _env, tmp_path
    ) -> None:
        """The safety contract's strongest form: a mismatched RUNNING
        sandbox still in use is never stopped or deleted — the new caller
        is rejected instead."""
        manager, service = _make_manager()
        self._seed_mismatch(
            service, manager, "user::1", state="running", tmp_path=tmp_path
        )
        manager._lease_providers["user::1"] = object()
        assert await manager.attach("user", "1")

        with pytest.raises(SandboxRuntimeConflictError):
            await manager.get_or_create_sandbox(
                "user", "1", mount_intent=_intent(tmp_path)
            )

        service.stop_existing.assert_not_awaited()
        service.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_failure_raises_recovery_required(self, _env, tmp_path) -> None:
        manager, service = _make_manager()
        self._seed_mismatch(
            service, manager, "user::1", state="running", tmp_path=tmp_path
        )
        service.stop_existing.side_effect = RuntimeError("docker daemon unreachable")

        with pytest.raises(SandboxRecoveryRequiredError):
            await manager.get_or_create_sandbox(
                "user", "1", mount_intent=_intent(tmp_path)
            )

        service.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_that_does_not_take_raises_recovery_required(
        self, _env, tmp_path
    ) -> None:
        """stop_existing() returns without error but the container is
        still observably running on re-inspect: fail closed rather than
        delete a running container."""
        manager, service = _make_manager()
        self._seed_mismatch(
            service, manager, "user::1", state="running", tmp_path=tmp_path
        )

        async def fake_stop_existing(name, *, timeout=None):
            return None  # no-op: state stays "running"

        service.stop_existing.side_effect = fake_stop_existing

        with pytest.raises(SandboxRecoveryRequiredError):
            await manager.get_or_create_sandbox(
                "user", "1", mount_intent=_intent(tmp_path)
            )

        service.delete.assert_not_awaited()


class TestReconcileBudget:
    @pytest.mark.asyncio
    async def test_second_mismatch_rebuild_is_rejected(self, _env, tmp_path) -> None:
        """Default budget is 1 rebuild per base lifecycle name."""
        manager, service = _make_manager()
        stale_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="stale:v0"
        )

        def _seed_stale() -> None:
            service._containers["user::1"] = _FakeReconcileContainer(
                state="stopped",
                spec=stale_spec,
                fingerprint_label=stale_spec.fingerprint(),
                version_label=str(SPEC_CONTRACT_VERSION),
            )

        _seed_stale()
        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))
        assert manager._reconcile_budget["user::1"] == 0

        # Force a second mismatch for the same base name.
        manager._cache.pop("user::1", None)
        manager._config_cache.pop("user::1", None)
        _seed_stale()

        with pytest.raises(SandboxRuntimeConflictError):
            await manager.get_or_create_sandbox(
                "user", "1", mount_intent=_intent(tmp_path)
            )
        service.delete.assert_called_once()  # only the first rebuild deleted

    @pytest.mark.asyncio
    async def test_absent_create_never_consumes_budget(self, _env, tmp_path) -> None:
        manager, service = _make_manager()

        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))

        assert "user::1" not in manager._reconcile_budget

    @pytest.mark.asyncio
    async def test_full_lifecycle_delete_clears_budget_and_allows_rebuild_again(
        self, _env, tmp_path
    ) -> None:
        """``delete_sandbox`` disposes of the whole lifecycle (primary +
        workers), so its exhausted budget must not linger to reject a
        later, unrelated occupant of the same base name — only a restart
        used to clear it, forcing every subsequent mismatch for that key to
        fail even though the earlier container is long gone."""
        manager, service = _make_manager()
        stale_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="stale:v0"
        )

        def _seed_stale() -> None:
            service._containers["user::1"] = _FakeReconcileContainer(
                state="stopped",
                spec=stale_spec,
                fingerprint_label=stale_spec.fingerprint(),
                version_label="1",
            )

        _seed_stale()
        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))
        assert manager._reconcile_budget["user::1"] == 0

        await manager.delete_sandbox("user", "1")
        assert "user::1" not in manager._reconcile_budget

        # A fresh mismatch for the same base name after the delete gets its
        # own rebuild allowance rather than inheriting the deleted
        # lifecycle's exhausted one.
        manager._cache.pop("user::1", None)
        manager._config_cache.pop("user::1", None)
        _seed_stale()

        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )
        assert sandbox.name == "user::1"
        assert service.create.await_count == 2  # both mismatches actually rebuilt

    @pytest.mark.asyncio
    async def test_worker_only_delete_does_not_touch_primary_budget(
        self, _env, tmp_path
    ) -> None:
        """``delete_worker_sandboxes`` (also release-to-zero's own cleanup)
        never disposes of the primary itself, so the base name's budget —
        still scoped to a live primary — must survive it."""
        manager, service = _make_manager()
        stale_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="stale:v0"
        )
        service._containers["user::1"] = _FakeReconcileContainer(
            state="stopped",
            spec=stale_spec,
            fingerprint_label=stale_spec.fingerprint(),
            version_label=str(SPEC_CONTRACT_VERSION),
        )
        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))
        assert manager._reconcile_budget["user::1"] == 0

        await manager.get_or_create_sandbox(
            "user", "1::worker::0", mount_intent=_intent(tmp_path, "worker0")
        )
        assert "user::1::worker::0" in manager._cache

        await manager.delete_worker_sandboxes("user", "1")

        assert "user::1::worker::0" not in manager._cache
        assert manager._reconcile_budget["user::1"] == 0

    @pytest.mark.asyncio
    async def test_idle_eviction_clears_budget_and_allows_rebuild_again(
        self, _env, tmp_path
    ) -> None:
        """Idle reclamation claims (``_claim_idle_sandbox``) the whole
        lifecycle group atomically before the underlying delete even runs,
        so the claim itself must drop the base name's budget too — the same
        unbounded-key / stale-rejection defect as an explicit delete, just
        reached through the sweep instead."""
        manager, service = _make_manager()
        stale_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="stale:v0"
        )

        def _seed_stale() -> None:
            service._containers["user::1"] = _FakeReconcileContainer(
                state="stopped",
                spec=stale_spec,
                fingerprint_label=stale_spec.fingerprint(),
                version_label="1",
            )

        _seed_stale()
        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))
        assert manager._reconcile_budget["user::1"] == 0

        # Force the lifecycle to read as idle since the dawn of time (the
        # mismatch rebuild above already dropped its own activity entry —
        # see ``_reconcile_delete`` — so idleness falls back to this
        # startup timestamp).
        manager._startup_monotonic = 0.0

        reclaimed = await manager.sweep_idle_sandboxes(idle_ttl=0)

        assert reclaimed == ["user::1"]
        assert "user::1" not in manager._reconcile_budget

        # A fresh mismatch for the same base name after reclamation gets its
        # own rebuild allowance rather than inheriting the reclaimed
        # lifecycle's exhausted one.
        _seed_stale()
        sandbox = await manager.get_or_create_sandbox(
            "user", "1", mount_intent=_intent(tmp_path)
        )
        assert sandbox.name == "user::1"

    @pytest.mark.asyncio
    async def test_failed_eviction_delete_leaves_budget_exhausted(
        self, _env, tmp_path
    ) -> None:
        """A container that keeps mismatching survives its eviction attempt
        (the backend delete raises), so its exhausted budget must survive
        too -- popping it regardless of delete outcome would hand the same
        still-live container a fresh rebuild allowance on every idle-sweep
        pass instead of ever refusing it. The idle sweep evicts by
        idleness alone, independent of the container's own match state, so
        this pins the budget straight (an earlier mismatch rebuild already
        spent it) rather than needing a second live mismatch to exercise
        the eviction path."""
        manager, service = _make_manager()
        await manager.get_or_create_sandbox("user", "1", mount_intent=_intent(tmp_path))
        manager._reconcile_budget["user::1"] = 0

        # Idle since the dawn of time, same as the sibling eviction test.
        manager._startup_monotonic = 0.0
        service.delete.side_effect = RuntimeError("backend unreachable")

        reclaimed = await manager.sweep_idle_sandboxes(idle_ttl=0)

        assert reclaimed == ["user::1"]  # claim succeeded; the delete failed
        assert manager._reconcile_budget["user::1"] == 0

        # The failed delete left the container in place (the fake never ran
        # its real delete body) and the instance cache was purged. Seed a
        # fresh mismatch and retry: it must still be refused, not handed a
        # fresh rebuild allowance for a container that never actually went
        # away.
        stale_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="stale:v0"
        )
        service._containers["user::1"] = _FakeReconcileContainer(
            state="stopped",
            spec=stale_spec,
            fingerprint_label=stale_spec.fingerprint(),
            version_label=str(SPEC_CONTRACT_VERSION),
        )
        with pytest.raises(SandboxRuntimeConflictError):
            await manager.get_or_create_sandbox(
                "user", "1", mount_intent=_intent(tmp_path)
            )

    @pytest.mark.asyncio
    async def test_cleanup_quiesce_resets_budget(self, _env, tmp_path) -> None:
        manager, service = _make_manager()
        manager._reconcile_budget["user::1"] = 0

        await manager.cleanup()

        assert manager._reconcile_budget == {}


class TestProviderABA:
    @pytest.mark.asyncio
    async def test_attach_provider_true_for_current_object(self, _env) -> None:
        manager = SandboxManager(FakeSandboxService(runtime_spec_supported=True))
        provider = object()
        manager._lease_providers["user::1"] = provider

        assert await manager.attach_provider("user", "1", provider) is True
        assert manager.ref_count("user", "1") == 1

    @pytest.mark.asyncio
    async def test_attach_provider_false_for_stale_handle(self, _env) -> None:
        """A provider replaced by a rebuild fails attach_provider for a
        caller still holding the old (now-stale) handle, even though the
        key itself resolves to a live (different) provider."""
        manager = SandboxManager(FakeSandboxService(runtime_spec_supported=True))
        old_provider = object()
        new_provider = object()
        manager._lease_providers["user::1"] = old_provider

        manager._lease_providers["user::1"] = new_provider

        assert await manager.attach_provider("user", "1", old_provider) is False
        assert manager.ref_count("user", "1") == 0

    @pytest.mark.asyncio
    async def test_attach_provider_false_when_nothing_cached(self, _env) -> None:
        manager = SandboxManager(FakeSandboxService(runtime_spec_supported=True))

        assert await manager.attach_provider("user", "1", object()) is False


class TestWorkerReconcileDegradation:
    @pytest.mark.asyncio
    async def test_worker_reuses_primary_workspace_preparation_root(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        """A folded external bind must prepare the managed scope, not its
        operator-owned ancestor, for both the primary and every worker."""
        operator_root = tmp_path / "data"
        uploads_root = operator_root / "xagent" / "uploads"
        uploads_root.mkdir(parents=True)
        prepare_root = uploads_root / "user_42" / "scope-a"
        monkeypatch.setattr(
            sandbox_manager_module, "get_uploads_dir", lambda: uploads_root
        )
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        grants: list[tuple[Path, Path, Path | None]] = []

        def record_grant(
            path: Path, *, trusted_root: Path, preserve_through: Path | None = None
        ) -> None:
            grants.append((path, trusted_root, preserve_through))

        monkeypatch.setattr(
            sandbox_manager_module, "grant_sandbox_runtime_access", record_grant
        )
        provider = await manager.get_or_create_lease_provider(
            "user",
            "42",
            mount_intent=SandboxMountIntent(mount_root=str(operator_root)),
            prepare_root=str(prepare_root),
        )

        worker = await provider.get_worker_sandbox(0)

        assert worker is not provider.primary_sandbox
        assert worker.name == "user::42::worker::0"
        assert grants == [
            (prepare_root, operator_root, uploads_root),
            (prepare_root, operator_root, uploads_root),
        ]

    @pytest.mark.asyncio
    async def test_worker_mismatch_degrades_to_primary(self, _env, tmp_path) -> None:
        """A worker whose reconciliation hits a runtime-config conflict
        degrades to the primary sandbox instead of failing the tool
        mid-task (the same sharing semantics a capacity failure gets)."""
        manager, service = _make_manager()
        intent = _intent(tmp_path)
        provider = await manager.get_or_create_lease_provider(
            "user", "1", mount_intent=intent
        )

        stale_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="stale:v0"
        )
        service._containers["user::1::worker::0"] = _FakeReconcileContainer(
            state="stopped",
            spec=stale_spec,
            fingerprint_label=stale_spec.fingerprint(),
            version_label="1",
        )
        # Exhaust the base lifecycle's rebuild budget (shared by primary and
        # workers) so reconciliation rejects instead of silently rebuilding,
        # forcing the degrade path.
        manager._reconcile_budget["user::1"] = 0

        worker = await provider.get_worker_sandbox(0)

        assert worker is provider.primary_sandbox


class TestGateCallSiteStructuralPin:
    """Source-level pin for ``_assert_lifecycle_locked``'s runtime check:
    ``_get_or_create_sandbox_locked`` and ``_create_lease_provider_locked``
    may only be called from the lifecycle-locked entry points, never from
    anywhere else. Mirrors ``tests/sandbox/test_docker_lock_infra.py``'s
    ``TestSandboxControlSingleConstructionPoint`` pattern, but resolves each
    call site's *enclosing method* via the AST instead of a raw substring
    count, since both names also appear in prose inside docstrings."""

    ALLOWED_CALLERS = {
        "_get_or_create_sandbox_locked": {
            "get_or_create_sandbox",
            "_create_lease_provider_locked",
        },
        "_create_lease_provider_locked": {
            "create_lease_provider",
            "get_or_create_lease_provider",
        },
    }

    def test_gate_helper_call_sites_are_structurally_pinned(self) -> None:
        source = Path(sandbox_manager_module.__file__).read_text()
        tree = ast.parse(source)
        class_node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "SandboxManager"
        )

        callers: dict[str, set[str]] = {name: set() for name in self.ALLOWED_CALLERS}
        for method in class_node.body:
            if not isinstance(method, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            for node in ast.walk(method):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in self.ALLOWED_CALLERS
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    callers[node.func.attr].add(method.name)

        for callee, allowed in self.ALLOWED_CALLERS.items():
            assert callers[callee] == allowed, (
                f"{callee} must only be called from {sorted(allowed)}, "
                f"found {sorted(callers[callee])}"
            )


class TestPrepareRootMkdir:
    """``prepare_root`` (``ChatWorkspaceBinding.prepare_root``) must be the
    on-host mkdir target on creation, not ``mount_intent.mount_root`` -- for
    an isolate=False scoped task the two diverge: folding re-roots the
    mount onto the shared, already-existing unscoped user root, but the
    scope's own (deeper, not-yet-existing) subtree is where this task's
    files actually live and must be created."""

    @pytest.mark.asyncio
    async def test_isolate_false_scoped_task_creates_scope_subtree(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            workspace_binding_module, "get_uploads_dir", lambda: tmp_path
        )
        monkeypatch.setattr(
            workspace_binding_module, "get_external_upload_dirs", lambda: []
        )
        owner_id = 42
        scope = ExecutionScope(
            sandbox_key_suffix="tenantA", workspace_segments=("proj1",)
        )
        binding = build_chat_workspace_binding(owner_id, scope)
        # The fixture this scenario relies on: folding re-roots onto the
        # (already shared) unscoped ancestor, distinct from the scope's own
        # subtree that must still be created on disk.
        assert binding.prepare_root != binding.mount_intent.mount_root
        assert not Path(binding.prepare_root).exists()

        manager, _service = _make_manager()
        lifecycle_id = make_user_lifecycle_id(owner_id, scope.sandbox_key_suffix)

        await manager.get_or_create_lease_provider(
            USER_LIFECYCLE_TYPE,
            lifecycle_id,
            mount_intent=binding.mount_intent,
            prepare_root=binding.prepare_root,
        )

        assert Path(binding.prepare_root).is_dir()


class TestProviderCacheMountIntentGate:
    """A cached lease provider owns the mount intent its primary container
    was built from, and hands that same intent to every worker it creates.
    So the provider cache must pass the same spec gate the sandbox cache
    does: without it a second caller on the same lifecycle key silently
    receives the first caller's mounts, and ``attach_provider`` then
    succeeds by object identity with nothing left to catch the divergence.
    """

    @pytest.mark.asyncio
    async def test_second_mount_intent_is_rejected_on_cache_hit(
        self, _env, tmp_path
    ) -> None:
        manager, service = _make_manager()
        first = await manager.get_or_create_lease_provider(
            "user", "1", mount_intent=_intent(tmp_path, "actor-a")
        )

        with pytest.raises(SandboxRuntimeConflictError):
            await manager.get_or_create_lease_provider(
                "user", "1", mount_intent=_intent(tmp_path, "actor-b")
            )

        # The rejected caller leaves no trace: same provider, same container.
        assert manager._lease_providers["user::1"] is first
        assert first._mount_intent == _intent(tmp_path, "actor-a")
        assert set(service._containers) == {"user::1"}

    @pytest.mark.asyncio
    async def test_matching_mount_intent_still_shares_the_provider(
        self, _env, tmp_path
    ) -> None:
        """The #296 sharing case must keep working: two callers whose
        intents fold to the same desired spec get the one provider."""
        manager, service = _make_manager()
        first = await manager.get_or_create_lease_provider(
            "user", "1", mount_intent=_intent(tmp_path, "ca-root")
        )
        second = await manager.get_or_create_lease_provider(
            "user", "1", mount_intent=_intent(tmp_path, "ca-root")
        )

        assert second is first
        assert set(service._containers) == {"user::1"}

    @pytest.mark.asyncio
    async def test_cached_provider_does_not_repeat_mount_preparation(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        prepared: list[Path] = []

        def record_grant(path: Path, **_kwargs) -> None:
            prepared.append(path)

        monkeypatch.setattr(
            sandbox_manager_module, "grant_sandbox_runtime_access", record_grant
        )
        monkeypatch.setattr(sandbox_manager_module, "get_uploads_dir", lambda: tmp_path)
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        intent = _intent(tmp_path, "user-root")
        user_root = tmp_path / "user-root"

        first = await manager.get_or_create_lease_provider(
            "user", "1", mount_intent=intent, prepare_root=str(user_root)
        )
        second = await manager.get_or_create_lease_provider(
            "user", "1", mount_intent=intent, prepare_root=str(user_root)
        )

        assert second is first
        assert prepared == [user_root]

    def test_folded_mount_root_is_the_trusted_grant_anchor(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        grants: list[tuple[Path, Path]] = []

        def record_grant(
            path: Path, *, trusted_root: Path, preserve_through: Path | None = None
        ) -> None:
            grants.append((path, trusted_root))

        monkeypatch.setattr(
            sandbox_manager_module, "grant_sandbox_runtime_access", record_grant
        )
        monkeypatch.setattr(sandbox_manager_module, "get_uploads_dir", lambda: tmp_path)
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        mount_root = tmp_path / "user-root"
        prepare_root = mount_root / "scopes" / "actor-a"

        manager._prepare_workspace_mounts(
            "user",
            "1",
            SandboxMountIntent(mount_root=str(mount_root)),
            prepare_root=str(prepare_root),
        )

        assert grants == [
            (mount_root, tmp_path),
            (prepare_root, mount_root),
        ]

    def test_folded_ancestor_preserves_operator_owned_uploads_ancestry(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        operator_root = tmp_path / "data"
        operator_parent = operator_root / "xagent"
        uploads_root = operator_parent / "uploads"
        uploads_root.mkdir(parents=True)
        operator_root.chmod(0o775)
        operator_parent.chmod(0o771)
        uploads_root.chmod(0o777)
        original_modes = {
            path: stat.S_IMODE(path.stat().st_mode)
            for path in (operator_root, operator_parent, uploads_root)
        }
        monkeypatch.setattr(
            sandbox_manager_module, "get_uploads_dir", lambda: uploads_root
        )
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        monkeypatch.setattr(
            sandbox_manager_module,
            "grant_sandbox_runtime_access",
            grant_sandbox_runtime_access,
        )
        monkeypatch.setattr(
            "xagent.sandbox.base.os.fchown", lambda *_args, **_kwargs: None
        )
        prepare_root = uploads_root / "user_42" / "scope-a"

        manager._prepare_workspace_mounts(
            "user",
            "42",
            SandboxMountIntent(mount_root=str(operator_root)),
            prepare_root=str(prepare_root),
        )

        assert {
            path: stat.S_IMODE(path.stat().st_mode)
            for path in (operator_root, operator_parent, uploads_root)
        } == original_modes
        for managed_path in (uploads_root / "user_42", prepare_root):
            assert stat.S_IMODE(managed_path.stat().st_mode) & 0o777 == 0o750

    def test_folded_ancestor_uses_physical_suffix_across_symlink(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        operator_root = tmp_path / "data"
        physical_parent = operator_root / "real"
        physical_uploads = physical_parent / "uploads"
        physical_uploads.mkdir(parents=True)
        uploads_alias_parent = operator_root / "link"
        uploads_alias_parent.symlink_to(physical_parent, target_is_directory=True)
        uploads_root = uploads_alias_parent / "uploads"
        prepare_root = uploads_root / "user_42" / "scope-a"
        monkeypatch.setattr(
            sandbox_manager_module, "get_uploads_dir", lambda: uploads_root
        )
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        monkeypatch.setattr(
            sandbox_manager_module,
            "grant_sandbox_runtime_access",
            grant_sandbox_runtime_access,
        )
        monkeypatch.setattr(
            "xagent.sandbox.base.os.fchown", lambda *_args, **_kwargs: None
        )

        manager._prepare_workspace_mounts(
            "user",
            "42",
            SandboxMountIntent(mount_root=str(operator_root)),
            prepare_root=str(prepare_root),
        )

        assert (physical_uploads / "user_42" / "scope-a").is_dir()
        assert uploads_alias_parent.is_symlink()
        assert stat.S_IMODE(operator_root.stat().st_mode) == 0o755

    def test_folded_mount_creates_and_prepares_missing_anchor(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(sandbox_manager_module, "get_uploads_dir", lambda: tmp_path)
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        monkeypatch.setattr(
            sandbox_manager_module,
            "grant_sandbox_runtime_access",
            grant_sandbox_runtime_access,
        )
        requested_modes: list[int] = []
        real_fchmod = os.fchmod

        def record_fchmod(fd: int, mode: int) -> None:
            requested_modes.append(mode)
            real_fchmod(fd, mode)

        monkeypatch.setattr(
            "xagent.sandbox.base.os.fchown", lambda *_args, **_kwargs: None
        )
        monkeypatch.setattr("xagent.sandbox.base.os.fchmod", record_fchmod)
        mount_root = tmp_path / "user-root"
        prepare_root = mount_root / "scopes" / "actor-a"

        manager._prepare_workspace_mounts(
            "user",
            "1",
            SandboxMountIntent(mount_root=str(mount_root)),
            prepare_root=str(prepare_root),
        )

        assert prepare_root.is_dir()
        for path in (mount_root, mount_root / "scopes", prepare_root):
            assert stat.S_IMODE(path.stat().st_mode) & 0o777 == 0o750
        assert requested_modes == [0o2750, 0o2750, 0o2750]

    def test_folded_mount_symlink_alias_prepares_physical_workspace(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        physical_uploads = tmp_path / "physical-uploads"
        physical_uploads.mkdir(mode=0o755)
        uploads_alias = tmp_path / "uploads-alias"
        uploads_alias.symlink_to(physical_uploads, target_is_directory=True)
        prepare_root = physical_uploads / "user_42" / "scope-a"
        monkeypatch.setattr(
            sandbox_manager_module, "get_uploads_dir", lambda: physical_uploads
        )
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        monkeypatch.setattr(
            sandbox_manager_module,
            "grant_sandbox_runtime_access",
            grant_sandbox_runtime_access,
        )
        monkeypatch.setattr(
            "xagent.sandbox.base.os.fchown", lambda *_args, **_kwargs: None
        )

        manager._prepare_workspace_mounts(
            "user",
            "42",
            SandboxMountIntent(mount_root=str(uploads_alias)),
            prepare_root=str(prepare_root),
        )

        assert prepare_root.is_dir()
        assert stat.S_IMODE(physical_uploads.stat().st_mode) == 0o755
        for managed_path in (physical_uploads / "user_42", prepare_root):
            managed_mode = stat.S_IMODE(managed_path.stat().st_mode)
            assert managed_mode & 0o777 == 0o750
            if sys.platform == "linux":
                assert managed_mode & stat.S_ISGID

    def test_folded_mount_supports_distinct_aliases_to_same_uploads_root(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        physical_uploads = tmp_path / "physical-uploads"
        physical_uploads.mkdir(mode=0o755)
        uploads_alias = tmp_path / "uploads-alias"
        external_alias = tmp_path / "external-alias"
        uploads_alias.symlink_to(physical_uploads, target_is_directory=True)
        external_alias.symlink_to(physical_uploads, target_is_directory=True)
        monkeypatch.setattr(
            sandbox_manager_module, "get_uploads_dir", lambda: uploads_alias
        )
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        monkeypatch.setattr(
            sandbox_manager_module,
            "grant_sandbox_runtime_access",
            grant_sandbox_runtime_access,
        )
        monkeypatch.setattr(
            "xagent.sandbox.base.os.fchown", lambda *_args, **_kwargs: None
        )
        prepare_root = uploads_alias / "user_42" / "scope-a"

        manager._prepare_workspace_mounts(
            "user",
            "42",
            SandboxMountIntent(mount_root=str(external_alias)),
            prepare_root=str(prepare_root),
        )

        assert (physical_uploads / "user_42" / "scope-a").is_dir()

    def test_folded_mount_rejects_symlink_in_managed_suffix(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        physical_uploads = tmp_path / "physical-uploads"
        physical_uploads.mkdir(mode=0o755)
        uploads_alias = tmp_path / "uploads-alias"
        uploads_alias.symlink_to(physical_uploads, target_is_directory=True)
        protected = physical_uploads / "protected"
        protected.mkdir(mode=0o755)
        redirected_component = physical_uploads / "user_42"
        redirected_component.symlink_to(protected, target_is_directory=True)
        prepare_root = redirected_component / "scope-a"
        monkeypatch.setattr(
            sandbox_manager_module, "get_uploads_dir", lambda: physical_uploads
        )
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        monkeypatch.setattr(
            sandbox_manager_module,
            "grant_sandbox_runtime_access",
            grant_sandbox_runtime_access,
        )

        with pytest.raises(SandboxContractError):
            manager._prepare_workspace_mounts(
                "user",
                "42",
                SandboxMountIntent(mount_root=str(uploads_alias)),
                prepare_root=str(prepare_root),
            )

        assert stat.S_IMODE(protected.stat().st_mode) == 0o755
        assert not (protected / "scope-a").exists()

    def test_folded_mount_pins_physical_uploads_root_across_symlink(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        physical_root = tmp_path / "physical-root"
        physical_root.mkdir(mode=0o755)
        trusted_alias = tmp_path / "trusted-alias"
        trusted_alias.symlink_to(physical_root, target_is_directory=True)
        protected = physical_root / "protected"
        protected.mkdir(mode=0o755)
        uploads_root = physical_root / "uploads"
        uploads_root.symlink_to(protected, target_is_directory=True)
        prepare_root = uploads_root / "user_42" / "scope-a"
        monkeypatch.setattr(
            sandbox_manager_module, "get_uploads_dir", lambda: uploads_root
        )
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        monkeypatch.setattr(
            sandbox_manager_module,
            "grant_sandbox_runtime_access",
            grant_sandbox_runtime_access,
        )
        monkeypatch.setattr(
            "xagent.sandbox.base.os.fchown", lambda *_args, **_kwargs: None
        )

        manager._prepare_workspace_mounts(
            "user",
            "42",
            SandboxMountIntent(mount_root=str(trusted_alias)),
            prepare_root=str(prepare_root),
        )

        assert stat.S_IMODE(protected.stat().st_mode) == 0o755
        assert (protected / "user_42" / "scope-a").is_dir()
        assert uploads_root.is_symlink()

    def test_legacy_mount_exists_error_remains_best_effort(
        self, _env, tmp_path, monkeypatch, caplog
    ) -> None:
        manager, service = _make_manager()
        blocked = tmp_path / "blocked"
        monkeypatch.setattr(service, "host_workspace_gid", lambda: None)
        monkeypatch.setattr(
            manager,
            "_workspace_mount_paths",
            lambda *_args, **_kwargs: [(blocked, False)],
        )
        real_exists = Path.exists

        def restricted_exists(path: Path) -> bool:
            if path == blocked:
                raise PermissionError("restricted")
            return real_exists(path)

        monkeypatch.setattr(Path, "exists", restricted_exists)

        manager._prepare_workspace_mounts("user", "42", None)

        assert "Failed to prepare sandbox workspace mount" in caplog.text

    @pytest.mark.asyncio
    async def test_reused_container_still_prepares_managed_mount(
        self, _env, tmp_path, monkeypatch
    ) -> None:
        grants: list[tuple[Path, Path]] = []

        def record_grant(
            path: Path, *, trusted_root: Path, preserve_through: Path | None = None
        ) -> None:
            grants.append((path, trusted_root))

        monkeypatch.setattr(
            sandbox_manager_module, "grant_sandbox_runtime_access", record_grant
        )
        monkeypatch.setattr(sandbox_manager_module, "get_uploads_dir", lambda: tmp_path)
        manager, service = _make_manager()
        monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
        intent = _intent(tmp_path, "mount-root")
        prepare_root = tmp_path / "mount-root" / "task"
        desired = manager._build_runtime_spec("user", "1", mount_intent=intent)
        template, config = desired.to_backend_config()
        service._containers["user::1"] = _FakeReconcileContainer(
            state="stopped",
            spec=desired,
            fingerprint_label=desired.fingerprint(),
            version_label=str(SPEC_CONTRACT_VERSION),
        )
        service._store["user::1"] = SandboxInfo(
            name="user::1", state="stopped", template=template, config=config
        )

        await manager.get_or_create_lease_provider(
            "user",
            "1",
            mount_intent=intent,
            prepare_root=str(prepare_root),
        )

        assert grants == [
            (tmp_path / "mount-root", tmp_path),
            (prepare_root, tmp_path / "mount-root"),
        ]
        service.start_existing.assert_awaited_once_with("user::1")


class TestReadinessReservedUploadsSubtree:
    """``check_sandbox_static_readiness`` must also reject a static mount
    landing in the per-user uploads subtree every task's default workspace
    mount reserves (see ``_workspace_mount_paths`` / ``_make_default_volumes``),
    even though no single user id is enumerable at startup. Uses the same
    manager fixture as the rest of this module (real
    ``_resolve_backend_probe()`` against ``FakeSandboxService(
    runtime_spec_supported=True)``) rather than duplicating the ``_ProbeStub``
    from ``test_sandbox_manager_readiness.py``.
    """

    @pytest.mark.asyncio
    async def test_rejects_mount_exactly_at_a_reserved_user_dir(self, tmp_path) -> None:
        uploads_dir = tmp_path / "uploads"
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "SANDBOX_VOLUMES": f"{tmp_path / 'host'}:{uploads_dir / 'user_5'}:rw",
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            with pytest.raises(SandboxRuntimeConflictError):
                await check_sandbox_static_readiness(manager)

    @pytest.mark.asyncio
    async def test_rejects_mount_nested_inside_a_reserved_user_dir(
        self, tmp_path
    ) -> None:
        """A nested static bind would shadow the managed workspace bind."""
        uploads_dir = tmp_path / "uploads"
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "SANDBOX_VOLUMES": (
                        f"{tmp_path / 'host' / 'models'}:"
                        f"{uploads_dir / 'user_1' / 'models'}:ro"
                    ),
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            with pytest.raises(SandboxRuntimeConflictError):
                await check_sandbox_static_readiness(manager)

    @pytest.mark.parametrize("guest_path", ["/user_0", "/user_0/nested"])
    @pytest.mark.asyncio
    async def test_root_uploads_dir_still_rejects_reserved_user_subtree(
        self, tmp_path, guest_path
    ) -> None:
        """The root separator must not turn the reserved prefix into ``//``."""
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": "/",
                    "SANDBOX_VOLUMES": f"{tmp_path / 'host'}:{guest_path}:rw",
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            with pytest.raises(SandboxRuntimeConflictError):
                await check_sandbox_static_readiness(manager)

    @pytest.mark.asyncio
    async def test_root_uploads_dir_allows_descendant_mount(self, tmp_path) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": "/",
                    "SANDBOX_VOLUMES": f"{tmp_path / 'host'}:/shared:rw",
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            await check_sandbox_static_readiness(manager)

    @pytest.mark.asyncio
    async def test_rejects_external_upload_dir_nested_inside_a_reserved_user_dir(
        self, tmp_path
    ) -> None:
        """Deployment external mounts cannot shadow managed workspaces either."""
        uploads_dir = tmp_path / "uploads"
        external_dir = uploads_dir / "user_5" / "shared"
        external_dir.mkdir(parents=True)
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "XAGENT_EXTERNAL_UPLOAD_DIRS": str(external_dir),
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            with pytest.raises(SandboxRuntimeConflictError):
                await check_sandbox_static_readiness(manager)

    @pytest.mark.asyncio
    async def test_allows_mount_elsewhere_under_uploads_root(self, tmp_path) -> None:
        """A shared, non-per-user directory under the uploads root is not
        reserved and must still pass."""
        uploads_dir = tmp_path / "uploads"
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "SANDBOX_VOLUMES": (
                        f"{tmp_path / 'host'}:{uploads_dir / 'shared-kb'}:rw"
                    ),
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            await check_sandbox_static_readiness(manager)

    @pytest.mark.asyncio
    async def test_allows_mount_with_non_numeric_suffix(self, tmp_path) -> None:
        """``user_5abc`` is not a per-user directory ``scoped_user_root``
        would ever produce (ids are always ``int``), so it is not reserved."""
        uploads_dir = tmp_path / "uploads"
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "SANDBOX_VOLUMES": (
                        f"{tmp_path / 'host'}:{uploads_dir / 'user_5abc'}:rw"
                    ),
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            await check_sandbox_static_readiness(manager)

    @pytest.mark.asyncio
    async def test_rejects_writable_mount_covering_uploads_root(self, tmp_path) -> None:
        uploads_dir = tmp_path / "uploads"
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "SANDBOX_VOLUMES": f"{tmp_path}:{tmp_path / 'guest'}:rw",
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            with pytest.raises(SandboxRuntimeConflictError, match="covers"):
                await check_sandbox_static_readiness(manager)

    def test_rejects_covering_mount_before_workspace_permission_grant(
        self, tmp_path, monkeypatch
    ) -> None:
        uploads_dir = tmp_path / "uploads"
        prepared: list[Path] = []
        with patch.dict(
            "os.environ",
            {
                "XAGENT_UPLOADS_DIR": str(uploads_dir),
                "SANDBOX_VOLUMES": f"{tmp_path}:{tmp_path / 'guest'}:rw",
            },
            clear=True,
        ):
            manager, service = _make_manager()
            monkeypatch.setattr(service, "host_workspace_gid", lambda: 1010)
            monkeypatch.setattr(
                sandbox_manager_module,
                "grant_sandbox_runtime_access",
                lambda path, **_kwargs: prepared.append(path),
            )

            with pytest.raises(SandboxRuntimeConflictError, match="covers"):
                manager._prepare_workspace_mounts(
                    "user",
                    "42",
                    SandboxMountIntent(mount_root=str(uploads_dir)),
                    prepare_root=str(uploads_dir / "user_42"),
                )

        assert prepared == []
        assert not (uploads_dir / "user_42").exists()

    @pytest.mark.asyncio
    async def test_allows_read_only_mount_covering_uploads_root(self, tmp_path) -> None:
        uploads_dir = tmp_path / "uploads"
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "SANDBOX_VOLUMES": f"{tmp_path}:{tmp_path / 'guest'}:ro",
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            await check_sandbox_static_readiness(manager)

    @pytest.mark.asyncio
    async def test_allows_unrelated_source_with_guest_target_covering_uploads(
        self, tmp_path
    ) -> None:
        uploads_dir = tmp_path / "uploads"
        unrelated_source = tmp_path / "unrelated"
        unrelated_source.mkdir()
        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "SANDBOX_VOLUMES": (f"{unrelated_source}:{tmp_path}:rw"),
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
        ):
            manager, _service = _make_manager()
            await check_sandbox_static_readiness(manager)

    @pytest.mark.asyncio
    async def test_prefix_derivation_raises_on_incompatible_naming_scheme(
        self, tmp_path
    ) -> None:
        """If the per-user naming scheme is ever changed so the id is not
        the trailing token, deriving the reserved prefix from a single
        sentinel silently produces a prefix that matches no real directory
        -- protecting nothing without anyone noticing. Deriving it from two
        sentinels and verifying the result actually reproduces both must
        instead fail startup."""
        uploads_dir = tmp_path / "uploads"

        def _fake_scoped_user_root(base_dir, user_id, scope_segments=()):
            root = Path(base_dir) if base_dir is not None else uploads_dir
            return root / f"u{int(user_id)}-workspace"

        with (
            patch.dict(
                "os.environ",
                {
                    "XAGENT_UPLOADS_DIR": str(uploads_dir),
                    "SANDBOX_VOLUMES": (
                        f"{tmp_path / 'host'}:{uploads_dir / 'u5-workspace'}:rw"
                    ),
                },
                clear=True,
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[],
            ),
            patch(
                "xagent.web.sandbox_manager.scoped_user_root",
                side_effect=_fake_scoped_user_root,
            ),
        ):
            manager, _service = _make_manager()
            with pytest.raises(SandboxContractError):
                await check_sandbox_static_readiness(manager)
