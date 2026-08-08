"""End-to-end spec reconciliation against a real Docker daemon (#296).

Unlike ``test_sandbox_manager_reconcile.py`` (``FakeSandboxService``, no
Docker), every test here drives the real ``SandboxManager`` on top of a real
``DockerSandboxService`` + ``MemDockerStore`` and a real Docker daemon --
proving the #296 fix and the legacy-container migration path against actual
containers, not a mock's approximation of them.

Covers:

- #296 itself: two different Actors under the same CA fold to a
  byte-identical mount intent and share exactly one container, sequentially
  and concurrently, across independent ``SandboxManager`` instances (so
  reuse is provably coming from the reconciliation matrix converging on one
  resolved spec, not from a single manager's in-process provider cache).
- The legacy-container migration: a container built the way the pre-1b
  sandbox layer built it (managed/name labels, no spec attestation, the
  Actor subtree mounted as a second bind) converges to the new folded intent
  by exactly one rebuild, keeps its bind-mounted host data, and never
  rebuilds again.
- The UNVERIFIED-with-a-matching-store-row case: a legacy-shaped container
  that never actually drifted from the new desired spec reuses without
  ever gaining a fingerprint label (UNVERIFIED is a valid permanent state).
- The per-key reconcile rebuild budget: a second drift for the same
  lifecycle key is rejected instead of rebuilding again.
- Provider ABA: a handle obtained before release-to-zero must never
  re-attach once a fresh provider has replaced it.

Parallel discipline (mirrors ``tests/sandbox/test_docker_lifecycle_api.py``):
uuid-suffixed lifecycle keys so concurrent test workers never collide,
try/finally cleanup per test, no ``pytest.mark.docker``. Module-scoped
``event_loop``/``docker_service`` (real Docker, shared across this module's
tests) means this file must run under ``--dist=loadscope`` when parallelized
so it stays on one worker.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import xagent.sandbox.docker_sandbox as docker_sandbox_module
import xagent.web.services.workspace_binding as workspace_binding
from xagent.core.execution_scope import ExecutionScope
from xagent.core.workspace import scoped_user_root
from xagent.sandbox.base import (
    SPEC_CONTRACT_VERSION,
    SandboxInfo,
    SandboxMountIntent,
    SandboxRuntimeConflictError,
)
from xagent.sandbox.docker_sandbox import (
    DockerSandboxService,
    MemDockerStore,
    is_docker_available,
)
from xagent.web.sandbox_keys import USER_LIFECYCLE_TYPE, make_user_lifecycle_id
from xagent.web.sandbox_manager import SandboxManager
from xagent.web.services.workspace_binding import build_chat_workspace_binding

requires_docker = pytest.mark.skipif(
    not is_docker_available(), reason="Requires reachable Docker daemon"
)

OWNER_ID = 4242


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def docker_service():
    """Shared real Docker service/store for this module's tests."""
    return DockerSandboxService(MemDockerStore(), namespace="test")


@pytest.fixture
def _env(tmp_path):
    """Isolate sandbox env config; code mounts point at a real, existing dir.

    Real Docker rejects (or silently misbehaves on) bind sources that don't
    exist on the host, so the code-mount stand-in must be a real directory,
    unlike the FakeSandboxService tests that can use a bogus path.
    """
    code_src = tmp_path / "code-src"
    code_src.mkdir()
    with (
        patch.dict("os.environ", {}, clear=True),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[(str(code_src), "/app/src", "ro")],
        ),
    ):
        yield


@pytest.fixture
def _uploads(tmp_path, monkeypatch):
    """Point the workspace-binding builder's uploads dir at an isolated tree.

    No deployment-level external dirs: keeps the folded physical mount set
    to exactly the mount root plus the isolate-driven allowlist entry (the
    Actor subtree), which is what the CA-scoped rows below exercise.
    """
    monkeypatch.setattr(workspace_binding, "get_uploads_dir", lambda: tmp_path)
    monkeypatch.setattr(workspace_binding, "get_external_upload_dirs", lambda: [])
    return tmp_path


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _ca_scope(ca_segment: str, actor_segment: str) -> ExecutionScope:
    """A CA-scoped Actor: shared sandbox key + mount prefix, own Actor subtree.

    Two scopes built with the same ``ca_segment`` but a different
    ``actor_segment`` are exactly the #296 shape: same lifecycle key, same
    mount root, different (and, pre-fix, separately-mounted) Actor subtree.
    """
    return ExecutionScope(
        sandbox_key_suffix=ca_segment,
        workspace_segments=(ca_segment, actor_segment),
        sandbox_mount_segments=(ca_segment,),
        isolate_external_dirs=True,
    )


async def _delete_lifecycle(service, lifecycle_type: str, lifecycle_id: str) -> None:
    name = SandboxManager.make_sandbox_name(lifecycle_type, lifecycle_id)
    try:
        await service.delete(name)
    except Exception:
        pass


@requires_docker
class TestSameCAConflictFreeReuse:
    """#296's fix, asserted directly against real containers: two Actors
    under the same CA fold to a byte-identical mount intent and share
    exactly one container, whether the two lease-provider calls land
    sequentially or concurrently."""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_sequential_reuse_across_manager_instances(
        self, docker_service, _env, _uploads
    ) -> None:
        ca_segment = _unique("ca-seq")
        lifecycle_type = USER_LIFECYCLE_TYPE
        lifecycle_id = make_user_lifecycle_id(OWNER_ID, ca_segment)
        sandbox_name = SandboxManager.make_sandbox_name(lifecycle_type, lifecycle_id)

        binding_a = build_chat_workspace_binding(
            OWNER_ID, _ca_scope(ca_segment, "actor7")
        )
        binding_b = build_chat_workspace_binding(
            OWNER_ID, _ca_scope(ca_segment, "actor9")
        )
        assert binding_a.mount_intent == binding_b.mount_intent, (
            "two Actors under the same CA must fold to a byte-identical intent"
        )

        # Two independent manager instances sharing only the service/store:
        # proves reuse comes from the reconciliation matrix converging on
        # the same resolved spec, not from one manager's provider cache.
        manager_a = SandboxManager(docker_service)
        manager_b = SandboxManager(docker_service)
        try:
            provider_a = await manager_a.get_or_create_lease_provider(
                lifecycle_type, lifecycle_id, mount_intent=binding_a.mount_intent
            )
            container_a = await docker_service._find_container(sandbox_name)
            assert container_a is not None

            provider_b = await manager_b.get_or_create_lease_provider(
                lifecycle_type, lifecycle_id, mount_intent=binding_b.mount_intent
            )
            container_b = await docker_service._find_container(sandbox_name)
            assert container_b is not None

            assert container_b.id == container_a.id, (
                "Actor b's request must reuse Actor a's container, never build a "
                "second one"
            )
            assert provider_a.primary_sandbox.name == provider_b.primary_sandbox.name
        finally:
            await _delete_lifecycle(docker_service, lifecycle_type, lifecycle_id)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_concurrent_reuse_across_manager_instances(
        self, docker_service, _env, _uploads
    ) -> None:
        """Same fold-to-one-container guarantee under genuine concurrency.

        Two ``SandboxManager`` instances have independent per-key gates
        (``_lifecycle_locks`` lives on ``self``), so this races both
        requests all the way down to ``DockerSandboxService.create()``,
        which serializes on its own ``_named_lock`` and raises
        ``SandboxAlreadyExistsError`` for the loser -- the reconciliation
        route's single re-inspect retry must recover that into a MATCH
        reuse rather than surfacing a conflict.
        """
        ca_segment = _unique("ca-conc")
        lifecycle_type = USER_LIFECYCLE_TYPE
        lifecycle_id = make_user_lifecycle_id(OWNER_ID, ca_segment)
        sandbox_name = SandboxManager.make_sandbox_name(lifecycle_type, lifecycle_id)

        binding_a = build_chat_workspace_binding(
            OWNER_ID, _ca_scope(ca_segment, "actor1")
        )
        binding_b = build_chat_workspace_binding(
            OWNER_ID, _ca_scope(ca_segment, "actor2")
        )
        assert binding_a.mount_intent == binding_b.mount_intent

        manager_a = SandboxManager(docker_service)
        manager_b = SandboxManager(docker_service)
        try:
            provider_a, provider_b = await asyncio.gather(
                manager_a.get_or_create_lease_provider(
                    lifecycle_type, lifecycle_id, mount_intent=binding_a.mount_intent
                ),
                manager_b.get_or_create_lease_provider(
                    lifecycle_type, lifecycle_id, mount_intent=binding_b.mount_intent
                ),
            )

            container = await docker_service._find_container(sandbox_name)
            assert container is not None
            assert provider_a.primary_sandbox.name == sandbox_name
            assert provider_b.primary_sandbox.name == sandbox_name
        finally:
            await _delete_lifecycle(docker_service, lifecycle_type, lifecycle_id)


@requires_docker
class TestLegacyContainerMigration:
    """A container built the way the pre-1b sandbox layer built it (managed
    + name labels only, no spec attestation, the Actor subtree mounted as a
    second bind) converges to the new folded intent by exactly one rebuild,
    keeps its host-side bind data, and never rebuilds again."""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_legacy_container_rebuilds_once_and_keeps_bind_data(
        self, docker_service, _env, _uploads, tmp_path
    ) -> None:
        ca_segment = _unique("legacy-ca")
        actor_segment = "actor-legacy"
        lifecycle_type = USER_LIFECYCLE_TYPE
        lifecycle_id = make_user_lifecycle_id(OWNER_ID, ca_segment)
        sandbox_name = SandboxManager.make_sandbox_name(lifecycle_type, lifecycle_id)
        base_name = sandbox_name

        scope = _ca_scope(ca_segment, actor_segment)
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        new_intent = binding.mount_intent
        ca_root = binding.prepare_root
        actor_child = str(
            scoped_user_root(tmp_path, OWNER_ID, scope.workspace_segments)
        )
        assert new_intent.mount_root == ca_root
        assert new_intent.extra_mounts == (), "the Actor subtree must fold away"

        os.makedirs(ca_root, exist_ok=True)
        os.makedirs(actor_child, exist_ok=True)
        (Path(ca_root) / "ca-marker.txt").write_text("ca-data")
        (Path(actor_child) / "actor-marker.txt").write_text("actor-data")

        # Pre-1b physical mount set: root + Actor subtree as a second,
        # unfolded bind (today's fixed-up builder folds this away).
        legacy_intent = SandboxMountIntent(
            mount_root=ca_root, extra_mounts=(actor_child,)
        )

        manager = SandboxManager(docker_service)
        try:
            t0_spec = manager._build_runtime_spec(
                lifecycle_type, lifecycle_id, mount_intent=legacy_intent
            )
            desired_spec = manager._build_runtime_spec(
                lifecycle_type, lifecycle_id, mount_intent=new_intent
            )
            assert t0_spec != desired_spec, "fixture must exercise a genuine drift"

            # Raw construction, deliberately bypassing both create() (which
            # would attest a fingerprint label) and get_or_create() (which
            # would also record a store row from this same config): builds
            # exactly the managed+name-labeled, unattested container the
            # pre-1b get_or_create() path produced, then seeds the store row
            # by hand so the row's recorded config is under this test's
            # control (t0, including the Actor-child bind).
            template, config = t0_spec.to_backend_config()
            container = await docker_sandbox_module._create_container(
                docker_service._client,
                sandbox_name,
                docker_service._namespace,
                t0_spec.image,
                template,
                config,
            )
            await asyncio.to_thread(container.start)
            await asyncio.to_thread(container.reload)
            await docker_service.persist_store_record(
                sandbox_name,
                SandboxInfo(
                    name=sandbox_name, state="running", template=template, config=config
                ),
            )

            # Positive precondition (anti-false-green): a genuinely
            # legacy-shaped container is live, with no spec attestation.
            pre_inspection = await docker_service.inspect(sandbox_name)
            assert pre_inspection is not None
            assert pre_inspection.fingerprint_label is None
            id_before = container.id

            sandbox = await manager.get_or_create_sandbox(
                lifecycle_type, lifecycle_id, mount_intent=new_intent
            )
            assert sandbox.name == sandbox_name
            assert manager._reconcile_budget.get(base_name) == 0, (
                "the one-time legacy rebuild must consume the per-key budget"
            )

            rebuilt = await docker_service._find_container(sandbox_name)
            assert rebuilt is not None
            assert rebuilt.id != id_before, (
                "legacy container must be rebuilt exactly once"
            )

            post_inspection = await docker_service.inspect(sandbox_name)
            assert post_inspection is not None
            assert post_inspection.fingerprint_label == desired_spec.fingerprint()
            assert post_inspection.version_label == str(SPEC_CONTRACT_VERSION)

            # Bind-mounted host data is untouched by the container rebuild:
            # only the container was destroyed and recreated, never the
            # host-side directories the bind mounts point at.
            assert (Path(ca_root) / "ca-marker.txt").read_text() == "ca-data"
            assert (Path(actor_child) / "actor-marker.txt").read_text() == "actor-data"

            # A later request (fresh manager instance, as after a redeploy)
            # converges by MATCH-reuse and never rebuilds again.
            manager2 = SandboxManager(docker_service)
            sandbox2 = await manager2.get_or_create_sandbox(
                lifecycle_type, lifecycle_id, mount_intent=new_intent
            )
            assert sandbox2.name == sandbox_name
            assert base_name not in manager2._reconcile_budget

            unchanged = await docker_service._find_container(sandbox_name)
            assert unchanged is not None
            assert unchanged.id == rebuilt.id
        finally:
            await _delete_lifecycle(docker_service, lifecycle_type, lifecycle_id)


@requires_docker
class TestLegacyMatchingStoreRowStaysUnverified:
    """A legacy-shaped container whose store row's recorded spec already
    equals the new desired spec reuses via UNVERIFIED-equal convergence: no
    rebuild, and the container never gains a spec-attestation label
    (UNVERIFIED is a valid permanent terminal state for it)."""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_unverified_with_matching_store_row_never_rebuilds(
        self, docker_service, _env, _uploads, tmp_path
    ) -> None:
        ca_segment = _unique("legacy-match")
        actor_segment = "actor-match"
        lifecycle_type = USER_LIFECYCLE_TYPE
        lifecycle_id = make_user_lifecycle_id(OWNER_ID, ca_segment)
        sandbox_name = SandboxManager.make_sandbox_name(lifecycle_type, lifecycle_id)
        base_name = sandbox_name

        scope = _ca_scope(ca_segment, actor_segment)
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        new_intent = binding.mount_intent
        os.makedirs(new_intent.mount_root, exist_ok=True)

        manager = SandboxManager(docker_service)
        try:
            desired_spec = manager._build_runtime_spec(
                lifecycle_type, lifecycle_id, mount_intent=new_intent
            )
            template, config = desired_spec.to_backend_config()

            # Legacy-shaped (no spec labels), but this one was already built
            # with today's post-fold mount set -- a container that predates
            # the fingerprint label yet never drifted from what the new
            # spec would build.
            container = await docker_sandbox_module._create_container(
                docker_service._client,
                sandbox_name,
                docker_service._namespace,
                desired_spec.image,
                template,
                config,
            )
            await asyncio.to_thread(container.start)
            await asyncio.to_thread(container.reload)
            await docker_service.persist_store_record(
                sandbox_name,
                SandboxInfo(
                    name=sandbox_name, state="running", template=template, config=config
                ),
            )

            pre_inspection = await docker_service.inspect(sandbox_name)
            assert pre_inspection is not None
            assert pre_inspection.fingerprint_label is None
            id_before = container.id

            sandbox = await manager.get_or_create_sandbox(
                lifecycle_type, lifecycle_id, mount_intent=new_intent
            )
            assert sandbox.name == sandbox_name

            unchanged = await docker_service._find_container(sandbox_name)
            assert unchanged is not None
            assert unchanged.id == id_before, (
                "UNVERIFIED-equal must reuse, never rebuild"
            )

            post_inspection = await docker_service.inspect(sandbox_name)
            assert post_inspection is not None
            assert post_inspection.fingerprint_label is None, (
                "UNVERIFIED-equal reuse must never backfill a label onto an "
                "unattested container"
            )
            assert base_name not in manager._reconcile_budget
        finally:
            await _delete_lifecycle(docker_service, lifecycle_type, lifecycle_id)


@requires_docker
class TestReconcileBudgetExhaustion:
    """The per-key reconcile rebuild budget (default 1) rejects a second
    drift for the same lifecycle key instead of rebuilding again."""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_second_drift_is_rejected_after_budget_exhausted(
        self, docker_service, _env, _uploads, tmp_path, monkeypatch, caplog
    ) -> None:
        ca_segment = _unique("budget")
        lifecycle_type = USER_LIFECYCLE_TYPE
        lifecycle_id = make_user_lifecycle_id(OWNER_ID, ca_segment)
        sandbox_name = SandboxManager.make_sandbox_name(lifecycle_type, lifecycle_id)
        base_name = sandbox_name

        scope = ExecutionScope(
            sandbox_key_suffix=ca_segment, workspace_segments=(ca_segment,)
        )
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        intent = binding.mount_intent
        os.makedirs(intent.mount_root, exist_ok=True)

        # A mutable env value the manager's own config reader picks up fresh
        # on every call, simulating successive redeploys that each change
        # the desired sandbox env -- without needing os.environ (already
        # cleared by _env for the rest of this test's config surface).
        env_holder = {"value": {"BUDGET_TEST": "v1"}}
        monkeypatch.setattr(
            "xagent.web.sandbox_manager.get_sandbox_env",
            lambda: dict(env_holder["value"]),
        )

        manager = SandboxManager(docker_service)
        try:
            await manager.get_or_create_sandbox(
                lifecycle_type, lifecycle_id, mount_intent=intent
            )
            container_v1 = await docker_service._find_container(sandbox_name)
            assert container_v1 is not None
            id_v1 = container_v1.id

            # First drift: process cache evicted (a fresh manager would also
            # cache-miss on the next request; clearing here instead keeps
            # this test on the same _reconcile_budget instance so the
            # exhaustion below is observable).
            env_holder["value"] = {"BUDGET_TEST": "v2"}
            manager._cache.pop(sandbox_name, None)
            manager._config_cache.pop(sandbox_name, None)

            await manager.get_or_create_sandbox(
                lifecycle_type, lifecycle_id, mount_intent=intent
            )
            container_v2 = await docker_service._find_container(sandbox_name)
            assert container_v2 is not None
            id_v2 = container_v2.id
            assert id_v2 != id_v1, "the first drift must rebuild"
            assert manager._reconcile_budget[base_name] == 0

            # Second drift for the same key: budget is exhausted, so the new
            # caller is rejected instead of rebuilding again.
            env_holder["value"] = {"BUDGET_TEST": "v3"}
            manager._cache.pop(sandbox_name, None)
            manager._config_cache.pop(sandbox_name, None)

            caplog.set_level(logging.WARNING)
            with pytest.raises(SandboxRuntimeConflictError):
                await manager.get_or_create_sandbox(
                    lifecycle_type, lifecycle_id, mount_intent=intent
                )

            assert any(
                "budget exhausted" in record.getMessage().lower()
                for record in caplog.records
            ), "rejection must log a structured warning naming the exhausted budget"

            container_after_reject = await docker_service._find_container(sandbox_name)
            assert container_after_reject is not None
            assert container_after_reject.id == id_v2, (
                "a rejected rebuild must never delete or replace the container"
            )
        finally:
            await _delete_lifecycle(docker_service, lifecycle_type, lifecycle_id)


@requires_docker
class TestProviderABAAcrossReleaseToZero:
    """A provider handle obtained before release-to-zero must never
    re-attach once release installed no provider (and a later
    ``get_or_create_lease_provider`` installed a fresh one)."""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_stale_provider_handle_fails_attach_after_release_to_zero(
        self, docker_service, _env, _uploads, tmp_path
    ) -> None:
        ca_segment = _unique("aba")
        lifecycle_type = USER_LIFECYCLE_TYPE
        lifecycle_id = make_user_lifecycle_id(OWNER_ID, ca_segment)

        scope = ExecutionScope(
            sandbox_key_suffix=ca_segment, workspace_segments=(ca_segment,)
        )
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        intent = binding.mount_intent
        os.makedirs(intent.mount_root, exist_ok=True)

        manager = SandboxManager(docker_service)
        try:
            provider1 = await manager.get_or_create_lease_provider(
                lifecycle_type, lifecycle_id, mount_intent=intent
            )
            assert await manager.attach_provider(
                lifecycle_type, lifecycle_id, provider1
            )
            assert manager.ref_count(lifecycle_type, lifecycle_id) == 1

            released = await manager.release(lifecycle_type, lifecycle_id)
            assert released is True
            assert manager.ref_count(lifecycle_type, lifecycle_id) == 0

            provider2 = await manager.get_or_create_lease_provider(
                lifecycle_type, lifecycle_id, mount_intent=intent
            )
            assert provider2 is not provider1, (
                "release-to-zero must always install a fresh provider object"
            )

            assert (
                await manager.attach_provider(lifecycle_type, lifecycle_id, provider1)
                is False
            ), "a handle obtained before release-to-zero must never re-attach"
            assert (
                await manager.attach_provider(lifecycle_type, lifecycle_id, provider2)
                is True
            )
            assert manager.ref_count(lifecycle_type, lifecycle_id) == 1
        finally:
            await _delete_lifecycle(docker_service, lifecycle_type, lifecycle_id)
