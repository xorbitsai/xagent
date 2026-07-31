"""
Tests for DockerSandboxService's explicit lifecycle API: inspect(), create(),
start_existing(), stop_existing(), and supports_runtime_spec().

Parallel discipline: most tests here run against a real Docker daemon and
share a module-scoped ``docker_service``/event loop (same pattern as
test_docker_sandbox.py). Each test uses a uuid-suffixed sandbox name so
concurrent test workers never collide on the same name; ``list_sandboxes()``
assertions are membership-only (never exact count/content), and every test
cleans up its own sandbox in a ``finally`` block. A handful of tests that
exercise failure paths not practically reachable via a real daemon (start
failure, publish-verification mismatch) use a fake Docker client/container
instead and do not require Docker.
"""

from __future__ import annotations

import asyncio
import inspect as std_inspect
import threading
import uuid
from contextlib import asynccontextmanager

import pytest
from docker.errors import APIError

import xagent.sandbox.docker_sandbox as docker_sandbox_module
from xagent.sandbox import DEFAULT_SANDBOX_IMAGE
from xagent.sandbox.base import (
    SPEC_CONTRACT_VERSION,
    ResolvedSandboxRuntimeSpec,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxInfo,
    SandboxInspection,
    SandboxNotFoundError,
    SandboxRuntimeConflictError,
    SandboxService,
    SandboxTemplate,
    SpecVerdict,
    spec_matches_inspection,
)
from xagent.sandbox.boxlite_sandbox import BoxliteSandboxService
from xagent.sandbox.docker_sandbox import (
    DockerSandboxService,
    MemDockerStore,
    is_docker_available,
)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


requires_docker = pytest.mark.skipif(
    not is_docker_available(), reason="Requires reachable Docker daemon"
)


@pytest.fixture(scope="module")
def docker_service():
    """Provide a shared Docker sandbox service for integration-style tests."""
    return DockerSandboxService(MemDockerStore())


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# --- Fakes for mock-layer tests (no Docker required) ---


class _FakeContainerCollection:
    def __init__(self, containers=()):
        self._containers = list(containers)

    def list(self, *args, **kwargs):
        return list(self._containers)


class _FakeDockerClient:
    def __init__(self, containers=()):
        self.containers = _FakeContainerCollection(containers)

    def ping(self):
        return True


class _FakeCreatedContainer:
    """Fake container returned by a monkeypatched ``_create_container``."""

    def __init__(self, start_exc: Exception | None = None):
        self._start_exc = start_exc
        self.remove_calls: list[bool] = []
        self.start_calls = 0
        self.reload_calls = 0
        self.name = "fake-container"
        self.attrs: dict = {
            "Config": {"WorkingDir": "/home"},
            "State": {"Status": "running"},
            "Created": "2026-01-01T00:00:00Z",
        }
        self.labels: dict = {}

    def start(self):
        self.start_calls += 1
        if self._start_exc is not None:
            raise self._start_exc

    def reload(self):
        self.reload_calls += 1

    def remove(self, force: bool = False):
        self.remove_calls.append(force)


# --- Signature / async pins ---


class TestLifecycleApiSignatures:
    """Pin that DockerSandboxService's overrides keep the base's async
    signature shape (not the return annotation)."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "supports_runtime_spec",
            "inspect",
            "create",
            "start_existing",
            "stop_existing",
            "get_store_record",
            "persist_store_record",
        ],
    )
    def test_docker_override_matches_base_signature(self, method_name):
        base_method = getattr(SandboxService, method_name)
        docker_method = getattr(DockerSandboxService, method_name)

        assert std_inspect.iscoroutinefunction(docker_method)

        base_params = list(std_inspect.signature(base_method).parameters)
        docker_params = list(std_inspect.signature(docker_method).parameters)
        assert docker_params == base_params


class TestLabelConstants:
    def test_spec_fingerprint_label_name(self):
        assert (
            docker_sandbox_module.LABEL_SPEC_FINGERPRINT
            == "xagent.sandbox.spec.fingerprint"
        )

    def test_spec_version_label_name(self):
        assert docker_sandbox_module.LABEL_SPEC_VERSION == "xagent.sandbox.spec.version"


class TestCheckNoConflictingPorts:
    """Pin both directions of _check_no_conflicting_ports: a guest-side
    collision (silently dropped by _create_container's guest-keyed dict) and
    a host-side collision (only ever caught by Docker at container start, as
    a raw 500)."""

    def test_rejects_same_host_port_for_different_guest_ports(self):
        with pytest.raises(SandboxRuntimeConflictError, match="host port"):
            docker_sandbox_module._check_no_conflicting_ports(
                [(18080, 80), (18080, 81)]
            )

    def test_accepts_multiple_ephemeral_host_ports_for_different_guests(self):
        # host_port == 0 means "let Docker assign an ephemeral port"; many
        # guest ports may legitimately share it.
        docker_sandbox_module._check_no_conflicting_ports([(0, 80), (0, 81), (0, 82)])


class TestCheckNoConflictingVolumes:
    """Pin both directions of _check_no_conflicting_volumes: a host-side
    collision (silently dropped by _create_container's host-keyed dict) and a
    guest-side collision (only ever caught by Docker at container create, as
    a raw 400 'Duplicate mount point')."""

    def test_rejects_host_path_spelled_with_a_leading_double_slash(self):
        # Docker collapses the leading slash run, so both entries land on the
        # same host key and one guest path would be silently dropped.
        with pytest.raises(SandboxRuntimeConflictError, match="host path"):
            docker_sandbox_module._check_no_conflicting_volumes(
                [("//data", "/guest/a", "rw"), ("/data", "/guest/b", "rw")]
            )

    def test_rejects_same_guest_path_for_different_host_sources(self):
        with pytest.raises(SandboxRuntimeConflictError, match="guest path"):
            docker_sandbox_module._check_no_conflicting_volumes(
                [("/data/a", "/guest", "rw"), ("/data/b", "/guest", "rw")]
            )

    def test_rejects_guest_path_spelled_with_a_leading_double_slash(self):
        # One mount point to the backend, so canonicalization has to file both
        # entries under the same guest key for the conflict to be caught.
        with pytest.raises(SandboxRuntimeConflictError, match="guest path"):
            docker_sandbox_module._check_no_conflicting_volumes(
                [("/data/a", "//guest", "rw"), ("/data/b", "/guest", "rw")]
            )

    def test_accepts_duplicate_triples_spelled_differently(self):
        docker_sandbox_module._check_no_conflicting_volumes(
            [("//data", "//guest/a", "rw"), ("/data", "/guest/a", "rw")]
        )


# --- supports_runtime_spec ---


class TestSupportsRuntimeSpec:
    @pytest.mark.asyncio
    async def test_docker_supports_runtime_spec(self):
        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())
        assert await service.supports_runtime_spec() is True

    @pytest.mark.asyncio
    async def test_boxlite_does_not_support_runtime_spec(self):
        # BoxliteSandboxService.__init__ talks to a real boxlite runtime;
        # the probe itself needs no runtime access, so we bypass __init__.
        service = object.__new__(BoxliteSandboxService)
        assert await service.supports_runtime_spec() is False


# --- create(): mock-layer tests for paths that are impractical against a
# real daemon (start failure compensation, publish-verification mismatch) ---


class TestCreateStartFailureCompensation:
    @pytest.mark.asyncio
    async def test_create_removes_container_via_raw_remove_when_start_fails(
        self, monkeypatch
    ):
        created = _FakeCreatedContainer(start_exc=RuntimeError("port conflict"))

        async def fake_create_container(*args, **kwargs):
            return created

        monkeypatch.setattr(
            docker_sandbox_module, "_create_container", fake_create_container
        )

        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())

        async def delete_must_not_be_called(name):
            raise AssertionError(
                "create()'s start-failure compensation must call container.remove"
                " directly, never self.delete() (self-deadlock risk)"
            )

        monkeypatch.setattr(service, "delete", delete_must_not_be_called)

        with pytest.raises(RuntimeError, match="port conflict"):
            await service.create(
                "start-failure",
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(),
            )

        assert created.remove_calls == [True]


class TestCreateCompensationHoldsLockUnderRepeatedCancellation:
    """Pin the fix for _await_shielded: create()'s compensating remove()
    must run to completion, and _named_lock(name) must stay held the whole
    time, even if the caller's task is cancelled more than once while the
    remove is in flight. Before this fix, a second cancellation landing on
    the bare ``await asyncio.to_thread(container.remove, ...)`` would let
    _named_lock's ``finally: entry.lock.release()`` run while the remove was
    still executing in its worker thread, so a same-name create() started
    right after could race that in-flight remove and see a transient
    AlreadyExists error from Docker.
    """

    @pytest.mark.asyncio
    async def test_lock_stays_held_until_slow_remove_settles_across_two_cancels(
        self, monkeypatch
    ):
        remove_started = threading.Event()
        release_remove = threading.Event()
        remove_finished = threading.Event()

        class _SlowRemoveContainer:
            def __init__(self):
                self.remove_calls: list[bool] = []

            def start(self):
                raise RuntimeError("start failed, forcing compensation")

            def remove(self, force: bool = False):
                remove_started.set()
                release_remove.wait(timeout=5)
                self.remove_calls.append(force)
                remove_finished.set()

        created = _SlowRemoveContainer()

        async def fake_create_container(*args, **kwargs):
            return created

        monkeypatch.setattr(
            docker_sandbox_module, "_create_container", fake_create_container
        )

        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())
        name = "slow-remove-compensation"

        task = asyncio.ensure_future(
            service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(),
            )
        )

        while not remove_started.is_set():
            await asyncio.sleep(0.01)

        # First cancellation lands while the compensating remove() is
        # still blocked in its worker thread.
        task.cancel()
        await asyncio.sleep(0.01)
        assert not remove_finished.is_set()
        assert name in service._locks, (
            "the name lock entry must not be evicted while remove() is still in flight"
        )
        assert service._locks[name].lock.locked(), (
            "the name lock must still be held while remove() is still in flight"
        )

        # A second cancellation lands on top of the first, before remove()
        # has settled.
        task.cancel()
        await asyncio.sleep(0.01)
        assert not remove_finished.is_set()
        assert service._locks[name].lock.locked()

        # Only now let the compensating remove() actually finish.
        release_remove.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert remove_finished.is_set()
        assert created.remove_calls == [True]
        assert name not in service._locks


class TestCreatePublishVerification:
    @pytest.mark.asyncio
    async def test_create_removes_container_and_raises_on_mismatch(self, monkeypatch):
        created = _FakeCreatedContainer()

        async def fake_create_container(*args, **kwargs):
            return created

        monkeypatch.setattr(
            docker_sandbox_module, "_create_container", fake_create_container
        )

        def fake_build_inspection(container):
            from xagent.sandbox.base import ObservedRuntimeFacts

            return SandboxInspection(
                state="running",
                facts=ObservedRuntimeFacts(
                    raw_status="running",
                    image_ref=DEFAULT_SANDBOX_IMAGE,
                    image_digest="sha256:deadbeef",
                    raw_nano_cpus=1_000_000_000,
                    raw_memory_bytes=512 * 1024 * 1024,
                    env={},
                    # Deliberately wrong: does not match the desired (empty)
                    # volume set, to force a MISMATCH on publish verification.
                    volumes=(("/tampered/host", "/tampered/guest", "rw"),),
                    ports=(),
                    network_isolated=False,
                    runtime_networks=("bridge",),
                    labels={},
                    created_at="2026-01-01T00:00:00Z",
                    working_dir="/home",
                ),
                fingerprint_label="whatever",
                version_label="1",
            )

        monkeypatch.setattr(
            docker_sandbox_module, "_build_inspection", fake_build_inspection
        )

        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())

        async def delete_must_not_be_called(name):
            raise AssertionError(
                "publish-verification failure must call container.remove"
                " directly, never self.delete()"
            )

        monkeypatch.setattr(service, "delete", delete_must_not_be_called)

        with pytest.raises(SandboxRuntimeConflictError, match="volumes"):
            await service.create(
                "publish-mismatch",
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(),
            )

        assert created.remove_calls == [True]
        assert service._store.get_info("publish-mismatch") is None

    @pytest.mark.asyncio
    async def test_conflict_error_survives_a_failing_compensating_remove(
        self, monkeypatch
    ):
        """A remove() that itself fails must not mask the verification verdict.

        The mismatched fields are the caller-actionable fact; a raw docker
        error from the cleanup says nothing about why the sandbox was
        rejected, and the leaked container is reported to the log instead.
        """
        created = _FakeCreatedContainer()

        def remove_that_fails(force: bool = False):
            created.remove_calls.append(force)
            raise APIError("daemon refused to remove")

        monkeypatch.setattr(created, "remove", remove_that_fails)

        async def fake_create_container(*args, **kwargs):
            return created

        monkeypatch.setattr(
            docker_sandbox_module, "_create_container", fake_create_container
        )
        monkeypatch.setattr(
            docker_sandbox_module,
            "_build_inspection",
            lambda container: _observed_inspection(volumes=(("/x", "/y", "rw"),)),
        )

        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())

        with pytest.raises(SandboxRuntimeConflictError, match="volumes"):
            await service.create(
                "publish-mismatch-remove-fails",
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(),
            )

        assert created.remove_calls == [True], "the remove must still be attempted"
        assert service._store.get_info("publish-mismatch-remove-fails") is None, (
            "a container that failed verification must never reach the store"
        )


class TestCreatePersistsCanonicalDesiredState:
    """Pin that the store row, the created container and the fingerprint
    label all come from one canonical source.

    If the row kept the caller's raw spelling instead, the row and the label
    would be two different spellings of one spec, and every future reader
    would have to re-normalize the row through ``from_parts`` before
    comparing it -- an obligation nothing can enforce.
    """

    @pytest.mark.asyncio
    async def test_row_matches_the_canonical_spec_the_label_attests(self, monkeypatch):
        created = _FakeCreatedContainer()
        captured: dict = {}

        async def fake_create_container(
            client, name, image, template, config, extra_labels=None
        ):
            captured["template"] = template
            captured["config"] = config
            captured["extra_labels"] = dict(extra_labels or {})
            return created

        monkeypatch.setattr(
            docker_sandbox_module, "_create_container", fake_create_container
        )

        # Non-canonical spellings that from_parts() normalizes: a trailing
        # slash on working_dir, and a host source carrying both a reserved
        # POSIX '//' prefix and a '.' segment.
        raw_config = SandboxConfig(
            working_dir="/home/",
            volumes=[("//data/./sub/", "/mnt/x/", "rw")],
        )
        canonical_volumes = (("/data/sub", "/mnt/x", "rw"),)

        monkeypatch.setattr(
            docker_sandbox_module,
            "_build_inspection",
            lambda container: _observed_inspection(
                volumes=canonical_volumes, working_dir="/home"
            ),
        )

        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())
        await service.create(
            "canonical-row",
            SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
            raw_config,
        )

        row = service._store.get_info("canonical-row")
        assert row is not None

        # The row is the canonical form, not what the caller typed.
        assert raw_config.working_dir == "/home/", "test input must be non-canonical"
        assert row.config.working_dir == "/home"
        assert row.config.volumes == [tuple(v) for v in canonical_volumes]

        # Same objects the container itself was built from.
        assert row.template == captured["template"]
        assert row.config == captured["config"]

        # And the row reproduces the fingerprint that was stamped on the
        # container, so a reader comparing the row against the label needs no
        # renormalization step to get a MATCH.
        spec_from_row = ResolvedSandboxRuntimeSpec.from_parts(
            template_type=row.template.type,
            image=row.template.image,
            snapshot_id=row.template.snapshot_id,
            working_dir=row.config.working_dir,
            cpus=row.config.cpus,
            memory=row.config.memory,
            env=row.config.env,
            volumes=row.config.volumes,
            network_isolated=bool(row.config.network_isolated),
            ports=row.config.ports,
        )
        assert (
            spec_from_row.fingerprint()
            == captured["extra_labels"][docker_sandbox_module.LABEL_SPEC_FINGERPRINT]
        )
        assert spec_from_row.to_backend_config() == (row.template, row.config), (
            "the row must already be a fixed point of canonicalization"
        )

    @pytest.mark.asyncio
    async def test_legacy_get_or_create_still_persists_the_raw_request(
        self, monkeypatch
    ):
        """The canonical-row contract belongs to create(), not to the legacy path.

        ``get_or_create()`` writes no attestation label, so its row has no
        label to agree with and it keeps recording the caller's request
        verbatim. Pinned so the two paths' rows are not assumed identical.
        """
        created = _FakeCreatedContainer()

        async def fake_create_container(*args, **kwargs):
            return created

        monkeypatch.setattr(
            docker_sandbox_module, "_create_container", fake_create_container
        )

        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())
        raw_config = SandboxConfig(working_dir="/home/")
        await service.get_or_create(
            "legacy-row",
            template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
            config=raw_config,
        )

        row = service._store.get_info("legacy-row")
        assert row is not None
        assert row.config.working_dir == "/home/"


def _observed_facts(**overrides):
    """Observed facts that agree with ``SandboxConfig()``'s resolved spec."""
    from xagent.sandbox.base import ObservedRuntimeFacts

    defaults = dict(
        raw_status="running",
        image_ref=DEFAULT_SANDBOX_IMAGE,
        image_digest="sha256:deadbeef",
        raw_nano_cpus=1_000_000_000,
        raw_memory_bytes=512 * 1024 * 1024,
        env={},
        volumes=(),
        ports=(),
        network_isolated=False,
        runtime_networks=("bridge",),
        labels={},
        created_at="2026-01-01T00:00:00Z",
        working_dir="/home",
    )
    defaults.update(overrides)
    return ObservedRuntimeFacts(**defaults)


def _observed_inspection(**overrides):
    return SandboxInspection(
        state="running",
        facts=_observed_facts(**overrides),
        fingerprint_label="whatever",
        version_label="1",
    )


class TestFindPublishMismatchesEveryBranch:
    """Cover all seven comparison branches, in both directions.

    Publish verification is the mechanism whose *false positives* destroy a
    just-created container, so every branch needs a negative control (an
    agreeing value must not be reported) alongside the positive case.
    """

    @staticmethod
    def _desired():
        return ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image",
            image=DEFAULT_SANDBOX_IMAGE,
            working_dir="/home",
            cpus=1,
            memory=512,
        )

    def test_agreeing_observation_reports_no_mismatch(self):
        assert (
            docker_sandbox_module._find_publish_mismatches(
                self._desired(), DEFAULT_SANDBOX_IMAGE, _observed_inspection()
            )
            == []
        )

    @pytest.mark.parametrize(
        "field,overrides",
        [
            ("image", {"image_ref": "someone/else:latest"}),
            ("volumes", {"volumes": (("/host", "/guest", "rw"),)}),
            ("ports", {"ports": ((18080, 80),)}),
            ("working_dir", {"working_dir": "/somewhere/else"}),
            ("network_isolated", {"network_isolated": True}),
            ("cpus", {"raw_nano_cpus": 2_000_000_000}),
            ("memory", {"raw_memory_bytes": 1024 * 1024 * 1024}),
        ],
    )
    def test_each_field_is_reported_when_it_disagrees(self, field, overrides):
        mismatches = docker_sandbox_module._find_publish_mismatches(
            self._desired(), DEFAULT_SANDBOX_IMAGE, _observed_inspection(**overrides)
        )
        assert mismatches == [field]

    def test_reports_every_disagreeing_field_at_once(self):
        mismatches = docker_sandbox_module._find_publish_mismatches(
            self._desired(),
            "someone/else:latest",
            _observed_inspection(
                working_dir="/elsewhere",
                network_isolated=True,
                raw_memory_bytes=1024 * 1024 * 1024,
            ),
        )
        assert mismatches == ["image", "working_dir", "network_isolated", "memory"]

    def test_missing_raw_cpu_and_memory_read_as_zero_not_as_a_match(self):
        """``None`` from a backend that omits the field must not pass as equal."""
        mismatches = docker_sandbox_module._find_publish_mismatches(
            self._desired(),
            DEFAULT_SANDBOX_IMAGE,
            _observed_inspection(raw_nano_cpus=None, raw_memory_bytes=None),
        )
        assert mismatches == ["cpus", "memory"]

    def test_volume_and_port_comparison_ignores_ordering(self):
        desired = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image",
            image=DEFAULT_SANDBOX_IMAGE,
            working_dir="/home",
            volumes=[("/h1", "/g1", "rw"), ("/h2", "/g2", "ro")],
            ports=[(18080, 80), (18081, 81)],
        )
        inspection = _observed_inspection(
            volumes=(("/h2", "/g2", "ro"), ("/h1", "/g1", "rw")),
            ports=((18081, 81), (18080, 80)),
        )
        assert (
            docker_sandbox_module._find_publish_mismatches(
                desired, DEFAULT_SANDBOX_IMAGE, inspection
            )
            == []
        )


# --- inspect(): real-container tests ---


@requires_docker
class TestDockerInspect:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_inspect_missing_sandbox_returns_none(self, docker_service):
        name = _unique_name("inspect-missing")
        assert await docker_service.inspect(name) is None

    @pytest.mark.asyncio(loop_scope="module")
    async def test_inspect_reflects_created_sandbox(self, docker_service):
        name = _unique_name("inspect-created")
        service = docker_service
        try:
            await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.state == "running"
            assert inspection.facts.image_ref == DEFAULT_SANDBOX_IMAGE
            assert inspection.facts.raw_nano_cpus == 1_000_000_000
            assert inspection.facts.raw_memory_bytes == 256 * 1024 * 1024
            # get_or_create() is the legacy path: it never writes the new
            # spec-attestation labels.
            assert inspection.fingerprint_label is None
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_inspect_has_no_side_effects_on_stopped_container(
        self, docker_service
    ):
        name = _unique_name("inspect-no-side-effects")
        service = docker_service
        try:
            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )
            await sandbox.stop()

            before = await service.inspect(name)
            assert before is not None
            assert before.state == "stopped"

            # Calling inspect() again must not have started/changed anything.
            after = await service.inspect(name)
            assert after is not None
            assert after.state == "stopped"
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_inspect_never_acquires_name_lock(self, docker_service, monkeypatch):
        """Liveness pin: inspect() must not go through _named_lock at all,
        neither when no sandbox exists nor when one already does."""
        missing_name = _unique_name("inspect-no-lock-missing")
        existing_name = _unique_name("inspect-no-lock-existing")
        service = docker_service

        await service.get_or_create(
            existing_name,
            template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
            config=SandboxConfig(cpus=1, memory=256),
        )

        @asynccontextmanager
        async def failing_named_lock(_name):
            raise AssertionError("inspect() must never acquire the per-name lock")
            yield  # pragma: no cover

        try:
            monkeypatch.setattr(service, "_named_lock", failing_named_lock)

            # Missing sandbox: still must not touch the lock.
            assert await service.inspect(missing_name) is None

            # Existing sandbox: must return a real inspection, still without
            # touching the lock.
            inspection = await service.inspect(existing_name)
            assert inspection is not None
        finally:
            monkeypatch.undo()
            try:
                await service.delete(existing_name)
            except Exception:
                pass


# --- create(): real-container tests ---


@requires_docker
class TestDockerCreate:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_happy_path(self, docker_service):
        name = _unique_name("create-happy")
        service = docker_service
        config = SandboxConfig(cpus=1, memory=256)
        try:
            sandbox = await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )

            assert sandbox.name == name

            result = await sandbox.exec("echo", "hello")
            assert result.success
            assert result.stdout.strip() == "hello"

            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.fingerprint_label is not None
            assert inspection.version_label == str(SPEC_CONTRACT_VERSION)

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image=DEFAULT_SANDBOX_IMAGE,
                working_dir=config.working_dir,
                cpus=config.cpus,
                memory=config.memory,
                env=config.env,
                volumes=config.volumes,
                network_isolated=bool(config.network_isolated),
                ports=config.ports,
            )
            assert inspection.fingerprint_label == desired.fingerprint()
            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH

            stored = service._store.get_info(name)
            assert stored is not None
            assert stored.name == name
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_rejects_name_already_used_by_get_or_create(
        self, docker_service
    ):
        name = _unique_name("create-conflict-legacy")
        service = docker_service
        try:
            await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            with pytest.raises(SandboxAlreadyExistsError):
                await service.create(
                    name,
                    SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                    SandboxConfig(cpus=1, memory=256),
                )
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_rejects_name_already_used_by_create(self, docker_service):
        name = _unique_name("create-conflict-new")
        service = docker_service
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(cpus=1, memory=256),
            )

            with pytest.raises(SandboxAlreadyExistsError):
                await service.create(
                    name,
                    SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                    SandboxConfig(cpus=1, memory=256),
                )
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_rejects_conflicting_volume_host_paths(self, docker_service):
        import tempfile

        name = _unique_name("create-volume-conflict")
        service = docker_service
        host_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(SandboxRuntimeConflictError):
                await service.create(
                    name,
                    SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                    SandboxConfig(
                        cpus=1,
                        memory=256,
                        volumes=[
                            (host_dir, "/mnt/a", "rw"),
                            (host_dir, "/mnt/b", "rw"),
                        ],
                    ),
                )

            sandboxes = await service.list_sandboxes()
            assert name not in {s.name for s in sandboxes}
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_accepts_identical_duplicate_volumes(self, docker_service):
        import tempfile

        name = _unique_name("create-volume-dup")
        service = docker_service
        host_dir = tempfile.mkdtemp()
        try:
            sandbox = await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(
                    cpus=1,
                    memory=256,
                    volumes=[
                        (host_dir, "/mnt/a", "rw"),
                        (host_dir, "/mnt/a", "rw"),
                    ],
                ),
            )
            assert sandbox.name == name

            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.facts.volumes == ((host_dir, "/mnt/a", "rw"),)
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_rejects_conflicting_port_guest_ports(self, docker_service):
        name = _unique_name("create-port-conflict")
        service = docker_service
        try:
            with pytest.raises(SandboxRuntimeConflictError):
                await service.create(
                    name,
                    SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                    SandboxConfig(
                        cpus=1,
                        memory=256,
                        ports=[(18080, 80), (18081, 80)],
                    ),
                )

            sandboxes = await service.list_sandboxes()
            assert name not in {s.name for s in sandboxes}
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_accepts_identical_duplicate_ports(self, docker_service):
        name = _unique_name("create-port-dup")
        service = docker_service
        try:
            sandbox = await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(
                    cpus=1,
                    memory=256,
                    ports=[(18082, 80), (18082, 80)],
                ),
            )
            assert sandbox.name == name

            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.facts.ports == ((18082, 80),)
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    # --- create() builds the container from the same normalized desired
    # spec that publish-verification compares against, so a raw input that
    # from_parts() normalizes differently from its literal form (a trailing
    # slash, a `..` segment, an explicit None) still ends up
    # MATCH-verifiable. ---

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_normalizes_trailing_slash_working_dir(self, docker_service):
        name = _unique_name("create-norm-workdir")
        service = docker_service
        config = SandboxConfig(cpus=1, memory=256, working_dir="/home/")
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )
            inspection = await service.inspect(name)
            assert inspection is not None

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image=DEFAULT_SANDBOX_IMAGE,
                working_dir=config.working_dir,
                cpus=config.cpus,
                memory=config.memory,
            )
            assert desired.working_dir == "/home"
            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_normalizes_volume_trailing_slash(self, docker_service):
        import tempfile

        name = _unique_name("create-norm-vol-slash")
        service = docker_service
        host_dir = tempfile.mkdtemp()
        config = SandboxConfig(
            cpus=1, memory=256, volumes=[(host_dir + "/", "/mnt/a/", "rw")]
        )
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )
            inspection = await service.inspect(name)
            assert inspection is not None

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image=DEFAULT_SANDBOX_IMAGE,
                cpus=config.cpus,
                memory=config.memory,
                volumes=config.volumes,
            )
            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_normalizes_volume_dot_dot_segment(self, docker_service):
        import tempfile

        name = _unique_name("create-norm-vol-dotdot")
        service = docker_service
        host_dir = tempfile.mkdtemp()
        config = SandboxConfig(
            cpus=1,
            memory=256,
            volumes=[(host_dir + "/nested/..", "/mnt/a", "rw")],
        )
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )
            inspection = await service.inspect(name)
            assert inspection is not None

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image=DEFAULT_SANDBOX_IMAGE,
                cpus=config.cpus,
                memory=config.memory,
                volumes=config.volumes,
            )
            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_with_none_cpus_and_memory_succeeds_and_matches(
        self, docker_service
    ):
        name = _unique_name("create-none-cpus-memory")
        service = docker_service
        config = SandboxConfig(cpus=None, memory=None)
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )
            inspection = await service.inspect(name)
            assert inspection is not None
            # The backend default (cpus or 1, memory or 512) took effect.
            assert inspection.facts.raw_nano_cpus == 1_000_000_000
            assert inspection.facts.raw_memory_bytes == 512 * 1024 * 1024

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image=DEFAULT_SANDBOX_IMAGE,
                cpus=config.cpus,
                memory=config.memory,
            )
            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_env_fact_contains_desired_key(self, docker_service):
        """Docstring-documented fact: Config.Env mixes in image-defined ENV
        values alongside the ones we injected, so the assertion here is
        deliberately a subset check (``desired.items() <= actual.items()``)
        rather than an exact-match on the full env mapping."""
        name = _unique_name("create-env-fact")
        service = docker_service
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(
                    cpus=1, memory=256, env={"XAGENT_TEST_KEY": "test-value"}
                ),
            )

            inspection = await service.inspect(name)
            assert inspection is not None
            assert {
                "XAGENT_TEST_KEY": "test-value"
            }.items() <= inspection.facts.env.items()
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass


# --- create() from a snapshot template: real-container tests ---


@requires_docker
class TestDockerCreateFromSnapshot:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_from_snapshot_matches_and_uses_snapshot_image(
        self, docker_service
    ):
        service = docker_service
        source_name = _unique_name("snapshot-source")
        target_name = _unique_name("snapshot-target")
        snapshot_id = _unique_name("snap")
        try:
            await service.get_or_create(
                source_name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )
            snapshot = await service.create_snapshot(source_name, snapshot_id)

            config = SandboxConfig(cpus=1, memory=256)
            sandbox = await service.create(
                target_name,
                SandboxTemplate(type="snapshot", snapshot_id=snapshot_id),
                config,
            )
            assert sandbox.name == target_name

            inspection = await service.inspect(target_name)
            assert inspection is not None
            assert inspection.facts.image_ref == snapshot.metadata["image_tag"]
            assert inspection.fingerprint_label is not None

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type="snapshot",
                snapshot_id=snapshot_id,
                working_dir=config.working_dir,
                cpus=config.cpus,
                memory=config.memory,
                env=config.env,
                volumes=config.volumes,
                network_isolated=bool(config.network_isolated),
                ports=config.ports,
            )
            assert inspection.fingerprint_label == desired.fingerprint()
            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(target_name)
            except Exception:
                pass
            try:
                await service.delete(source_name)
            except Exception:
                pass
            try:
                await service.delete_snapshot(snapshot_id)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_from_missing_snapshot_raises_file_not_found(
        self, docker_service
    ):
        name = _unique_name("snapshot-missing")
        service = docker_service
        with pytest.raises(FileNotFoundError):
            await service.create(
                name,
                SandboxTemplate(type="snapshot", snapshot_id="does-not-exist"),
                SandboxConfig(cpus=1, memory=256),
            )


# --- start_existing() / stop_existing(): real-container tests ---


@requires_docker
class TestStartStopExisting:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_start_existing_missing_raises_not_found(self, docker_service):
        name = _unique_name("start-missing")
        with pytest.raises(SandboxNotFoundError):
            await docker_service.start_existing(name)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_stop_existing_missing_raises_not_found(self, docker_service):
        name = _unique_name("stop-missing")
        with pytest.raises(SandboxNotFoundError):
            await docker_service.stop_existing(name)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_stop_existing_then_start_existing_round_trip(self, docker_service):
        name = _unique_name("start-stop-roundtrip")
        service = docker_service
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(cpus=1, memory=256),
            )

            await service.stop_existing(name)
            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.state == "stopped"
            assert service._store.get_info(name).state == "stopped"

            sandbox = await service.start_existing(name)
            assert sandbox.name == name
            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.state == "running"
            assert service._store.get_info(name).state == "running"
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_stop_existing_is_idempotent(self, docker_service):
        name = _unique_name("stop-idempotent")
        service = docker_service
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(cpus=1, memory=256),
            )
            await service.stop_existing(name)
            # Second stop on an already-stopped sandbox must be a no-op, not
            # an error.
            await service.stop_existing(name)

            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.state == "stopped"
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_start_existing_is_idempotent(self, docker_service):
        name = _unique_name("start-idempotent")
        service = docker_service
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(cpus=1, memory=256),
            )
            # Already running: start_existing() must return happily rather
            # than erroring on a redundant start.
            sandbox = await service.start_existing(name)
            assert sandbox.name == name

            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.state == "running"
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_stop_existing_accepts_timeout_kwarg(self, docker_service):
        """An explicit timeout is passed straight through to
        docker-py's own container.stop(timeout=...) bound."""
        name = _unique_name("stop-timeout")
        service = docker_service
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(cpus=1, memory=256),
            )
            await service.stop_existing(name, timeout=1)

            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.state == "stopped"
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass


# --- get_store_record() / persist_store_record() ---


@requires_docker
class TestStoreRecordAccess:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_get_store_record_returns_none_for_absent_sandbox(
        self, docker_service
    ):
        name = _unique_name("store-absent")
        assert await docker_service.get_store_record(name) is None

    @pytest.mark.asyncio(loop_scope="module")
    async def test_get_store_record_reflects_create(self, docker_service):
        name = _unique_name("store-created")
        service = docker_service
        try:
            config = SandboxConfig(cpus=1, memory=256)
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )

            record = await service.get_store_record(name)

            assert record is not None
            assert record.name == name
            assert record.template.image == DEFAULT_SANDBOX_IMAGE
            assert record.config.cpus == 1
            assert record.config.memory == 256
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_persist_store_record_backfills_a_missing_row(self, docker_service):
        """Simulates reconciliation's MATCH-with-no-row backfill: the store
        row was independently lost (label is immutable, unaffected)."""
        name = _unique_name("store-backfill")
        service = docker_service
        try:
            template = SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE)
            config = SandboxConfig(cpus=1, memory=256)
            await service.create(name, template, config)
            service._store.delete_info(name)
            assert await service.get_store_record(name) is None

            info = SandboxInfo(
                name=name, state="running", template=template, config=config
            )
            await service.persist_store_record(name, info)

            record = await service.get_store_record(name)
            assert record is not None
            assert record.config.cpus == 1
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass


# --- _build_inspection(): direct unit coverage ---


class _FakeInspectableContainer:
    """Container stub whose ``attrs``/``labels`` are supplied verbatim."""

    def __init__(self, attrs: dict, labels: dict | None = None):
        self.attrs = attrs
        self.labels = labels if labels is not None else {}


def _attrs(**overrides) -> dict:
    base = {
        "Config": {
            "Image": "img:tag",
            "Env": ["A=1", "B=2", "MALFORMED"],
            "WorkingDir": "/home",
            "NetworkDisabled": False,
        },
        "HostConfig": {
            "NanoCpus": 2_000_000_000,
            "Memory": 256 * 1024 * 1024,
            "PortBindings": {"80/tcp": [{"HostPort": "18080"}]},
        },
        "State": {"Status": "running"},
        "Image": "sha256:imagedigest",
        "Created": "2026-01-01T00:00:00Z",
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/host/rw",
                "Destination": "/guest/rw",
                "RW": True,
            },
            {
                "Type": "bind",
                "Source": "/host/ro",
                "Destination": "/guest/ro",
                "RW": False,
            },
            # Non-bind mounts are not desired-state bind volumes and must be
            # skipped rather than reported as one.
            {
                "Type": "volume",
                "Source": "/var/lib/docker/volumes/x",
                "Destination": "/guest/vol",
                "RW": True,
            },
        ],
        "NetworkSettings": {"Networks": {"bridge": {}, "extra": {}}},
    }
    base.update(overrides)
    return base


class TestBuildInspectionFacts:
    """Assert every field ``_build_inspection`` derives, not just the ones the
    publish check happens to compare."""

    def test_reports_image_ref_and_digest_separately(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(_attrs())
        )
        assert inspection.facts.image_ref == "img:tag"
        assert inspection.facts.image_digest == "sha256:imagedigest"

    def test_keeps_raw_backend_units_for_cpus_and_memory(self):
        """Raw units are what make a live ``docker update`` observable."""
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(_attrs())
        )
        assert inspection.facts.raw_nano_cpus == 2_000_000_000
        assert inspection.facts.raw_memory_bytes == 256 * 1024 * 1024

    def test_parses_env_and_drops_entries_without_a_separator(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(_attrs())
        )
        assert inspection.facts.env == {"A": "1", "B": "2"}

    def test_derives_volume_mode_from_the_rw_flag_and_skips_non_binds(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(_attrs())
        )
        assert inspection.facts.volumes == (
            ("/host/rw", "/guest/rw", "rw"),
            ("/host/ro", "/guest/ro", "ro"),
        )

    def test_reads_ports_from_host_config_port_bindings(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(_attrs())
        )
        assert inspection.facts.ports == ((18080, 80),)

    def test_reports_runtime_networks_and_labels(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(_attrs(), labels={"a": "1"})
        )
        assert set(inspection.facts.runtime_networks) == {"bridge", "extra"}
        assert inspection.facts.labels == {"a": "1"}

    def test_network_isolated_comes_from_config_network_disabled(self):
        attrs = _attrs()
        attrs["Config"]["NetworkDisabled"] = True
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(attrs)
        )
        assert inspection.facts.network_isolated is True

    def test_network_isolated_is_false_when_network_disabled_is_absent(self):
        attrs = _attrs()
        del attrs["Config"]["NetworkDisabled"]
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(attrs)
        )
        assert inspection.facts.network_isolated is False

    def test_working_dir_and_created_at_are_passed_through_unnormalized(self):
        """Observed facts stay backend-native; normalizing here would hide a
        genuine backend rewrite."""
        attrs = _attrs()
        attrs["Config"]["WorkingDir"] = "//home//sub/"
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(attrs)
        )
        assert inspection.facts.working_dir == "//home//sub/"
        assert inspection.facts.created_at == "2026-01-01T00:00:00Z"

    def test_tolerates_completely_empty_attrs(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer({})
        )
        assert inspection.state == "stopped"
        assert inspection.facts.raw_status == ""
        assert inspection.facts.volumes == ()
        assert inspection.facts.ports == ()


class TestBuildInspectionStateMapping:
    """``state`` is a two-value reduction; ``facts.raw_status`` keeps the rest."""

    @pytest.mark.parametrize(
        "raw_status,expected_state",
        [
            ("running", "running"),
            ("created", "stopped"),
            ("exited", "stopped"),
            ("dead", "stopped"),
            ("restarting", "stopped"),
            ("paused", "stopped"),
            ("", "stopped"),
        ],
    )
    def test_reduces_every_status_to_running_or_stopped(
        self, raw_status, expected_state
    ):
        attrs = _attrs(State={"Status": raw_status})
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(attrs)
        )
        assert inspection.state == expected_state

    @pytest.mark.parametrize("raw_status", ["created", "exited", "dead", "restarting"])
    def test_raw_status_stays_available_for_states_state_cannot_express(
        self, raw_status
    ):
        """A caller needing "never started" vs "exited" reads raw_status."""
        attrs = _attrs(State={"Status": raw_status})
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(attrs)
        )
        assert inspection.state == "stopped"
        assert inspection.facts.raw_status == raw_status


class TestBuildInspectionAttestationLabels:
    """A blank attestation label means "no attestation", not "a fingerprint
    that fails to match".

    ``_create_container`` writes both keys blank when it has no attestation to
    make, so that a container cannot silently present a fingerprint inherited
    from its base image. Docker cannot remove an inherited label, only
    overwrite it, so blank is the only "absent" form available on the wire --
    and reading blank as a real value would turn UNVERIFIED into MISMATCH,
    making a reconciler rebuild a container it should have adopted.
    """

    def test_reads_a_real_attestation(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(
                _attrs(),
                labels={
                    docker_sandbox_module.LABEL_SPEC_FINGERPRINT: "abc123",
                    docker_sandbox_module.LABEL_SPEC_VERSION: "1",
                },
            )
        )
        assert inspection.fingerprint_label == "abc123"
        assert inspection.version_label == "1"

    def test_absent_labels_read_as_none(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(_attrs(), labels={})
        )
        assert inspection.fingerprint_label is None
        assert inspection.version_label is None

    def test_blank_labels_read_as_none(self):
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(
                _attrs(),
                labels={
                    docker_sandbox_module.LABEL_SPEC_FINGERPRINT: "",
                    docker_sandbox_module.LABEL_SPEC_VERSION: "",
                },
            )
        )
        assert inspection.fingerprint_label is None
        assert inspection.version_label is None

    def test_a_blank_attestation_yields_unverified_not_mismatch(self):
        desired = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="img:tag", working_dir="/home"
        )
        inspection = docker_sandbox_module._build_inspection(
            _FakeInspectableContainer(
                _attrs(),
                labels={
                    docker_sandbox_module.LABEL_SPEC_FINGERPRINT: "",
                    docker_sandbox_module.LABEL_SPEC_VERSION: "",
                },
            )
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED


class TestCreateContainerOwnsAttestationLabels:
    """``_create_container`` is the single writer of both attestation label
    keys for every container it creates, on either lifecycle path."""

    @staticmethod
    async def _captured_labels(monkeypatch, extra_labels=None):
        captured: dict = {}

        class _Collection:
            def create(self, **kwargs):
                captured.update(kwargs)
                return object()

        class _Client:
            containers = _Collection()

        async def no_pull(client, image):
            return None

        monkeypatch.setattr(docker_sandbox_module, "_ensure_image", no_pull)
        await docker_sandbox_module._create_container(
            _Client(),
            "box",
            "img:tag",
            SandboxTemplate(type="image", image="img:tag"),
            SandboxConfig(),
            extra_labels=extra_labels,
        )
        return captured["labels"]

    @pytest.mark.asyncio
    async def test_blanks_both_keys_when_there_is_no_attestation(self, monkeypatch):
        labels = await self._captured_labels(monkeypatch)
        assert labels[docker_sandbox_module.LABEL_SPEC_FINGERPRINT] == ""
        assert labels[docker_sandbox_module.LABEL_SPEC_VERSION] == ""

    @pytest.mark.asyncio
    async def test_extra_labels_override_the_blanks(self, monkeypatch):
        labels = await self._captured_labels(
            monkeypatch,
            extra_labels={
                docker_sandbox_module.LABEL_SPEC_FINGERPRINT: "fp",
                docker_sandbox_module.LABEL_SPEC_VERSION: "1",
            },
        )
        assert labels[docker_sandbox_module.LABEL_SPEC_FINGERPRINT] == "fp"
        assert labels[docker_sandbox_module.LABEL_SPEC_VERSION] == "1"


class TestLifecycleDoesNotLeakControlEntries:
    """``_controls`` is keyed by sandbox name, a name space with no bound
    (``ssh::{task_id}``), and only ``delete()`` ever evicts from it. Probing a
    name that has no container must therefore not install an entry, or every
    such probe grows the dict for the life of the process."""

    @pytest.mark.asyncio
    async def test_start_existing_on_a_missing_sandbox_installs_no_control(self):
        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())
        with pytest.raises(SandboxNotFoundError):
            await service.start_existing("never-existed")
        assert service._controls == {}

    @pytest.mark.asyncio
    async def test_stop_existing_on_a_missing_sandbox_installs_no_control(self):
        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())
        with pytest.raises(SandboxNotFoundError):
            await service.stop_existing("never-existed")
        assert service._controls == {}

    @pytest.mark.asyncio
    async def test_repeated_probes_of_distinct_missing_names_stay_bounded(self):
        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())
        for i in range(25):
            with pytest.raises(SandboxNotFoundError):
                await service.stop_existing(f"ssh::task-{i}")
        assert service._controls == {}
        assert len(service._locks) == 0, "the lock registry must also stay empty"

    @pytest.mark.asyncio
    async def test_inspect_of_a_missing_sandbox_installs_no_control(self):
        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())
        assert await service.inspect("never-existed") is None
        assert service._controls == {}


@requires_docker
class TestCreateNetworkIsolatedSandbox:
    """``network_isolated`` is the one security-relevant boolean in the spec and
    had no real-daemon coverage: nothing proved Docker round-trips
    ``NetworkDisabled`` well enough for publish verification to pass."""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_isolated_sandbox_passes_verification_and_matches(
        self, docker_service
    ):
        service = docker_service
        name = _unique_name("net-isolated")
        config = SandboxConfig(cpus=1, memory=256, network_isolated=True)
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )

            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.facts.network_isolated is True

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image=DEFAULT_SANDBOX_IMAGE,
                working_dir=config.working_dir,
                cpus=config.cpus,
                memory=config.memory,
                network_isolated=True,
            )
            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_a_non_isolated_sandbox_reports_isolation_false(self, docker_service):
        """Negative control: the observation must actually track the request."""
        service = docker_service
        name = _unique_name("net-open")
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                SandboxConfig(cpus=1, memory=256, network_isolated=False),
            )
            inspection = await service.inspect(name)
            assert inspection is not None
            assert inspection.facts.network_isolated is False
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass
