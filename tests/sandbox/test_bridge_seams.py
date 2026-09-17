"""
Bridge tests between the legacy get_or_create()/create() lifecycle paths and
spec_matches_inspection(): the three seams the reconciliation matcher must
handle correctly once both paths can produce containers for the same
service.

Real Docker required (label/attestation behavior is only meaningful against
real container attrs). Parallel discipline: uuid-suffixed names, membership-
only list_sandboxes() assertions (none used here), try/finally cleanup per
test.
"""

from __future__ import annotations

import uuid

import pytest

from xagent.sandbox import DEFAULT_SANDBOX_IMAGE
from xagent.sandbox.base import (
    ResolvedSandboxRuntimeSpec,
    SandboxConfig,
    SandboxTemplate,
    SpecVerdict,
    spec_matches_inspection,
)
from xagent.sandbox.docker_sandbox import (
    DockerSandboxService,
    MemDockerStore,
    is_docker_available,
)

requires_docker = pytest.mark.skipif(
    not is_docker_available(), reason="Requires reachable Docker daemon"
)


@pytest.fixture(scope="module")
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def docker_service():
    return DockerSandboxService(MemDockerStore(), namespace="test")


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _spec_for(config: SandboxConfig, image: str = DEFAULT_SANDBOX_IMAGE):
    return ResolvedSandboxRuntimeSpec.from_parts(
        template_type="image",
        image=image,
        working_dir=config.working_dir,
        cpus=config.cpus,
        memory=config.memory,
        env=config.env,
        volumes=config.volumes,
        network_isolated=bool(config.network_isolated),
        ports=config.ports,
    )


@requires_docker
class TestBridgeSeams:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_legacy_get_or_create_container_is_unverified(self, docker_service):
        """A container created by the old get_or_create() path carries no
        spec-attestation label, so the matcher must report UNVERIFIED rather
        than MISMATCH (which would force-rebuild every pre-existing
        container the moment reconciliation is turned on)."""
        service = docker_service
        name = _unique_name("bridge-legacy")
        config = SandboxConfig(cpus=1, memory=256)
        try:
            await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=config,
            )

            inspection = await service.inspect(name)
            assert inspection is not None
            desired = _spec_for(config)

            assert (
                spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED
            )
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_new_create_container_is_match(self, docker_service):
        """A container created by the new create() lifecycle carries a
        verified fingerprint/version label pair, so the same desired spec
        used to create it must compare as MATCH."""
        service = docker_service
        name = _unique_name("bridge-new")
        config = SandboxConfig(cpus=1, memory=256)
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )

            inspection = await service.inspect(name)
            assert inspection is not None
            desired = _spec_for(config)

            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_label_present_but_store_row_missing_still_matches(
        self, docker_service
    ):
        """The matcher itself is blind to the store: it only ever looks at
        the live label/facts, so a container with a verified label but no
        store row still reports MATCH here. Recognizing that this
        specific combination (label present, store row absent) is the one
        case reconciliation must always treat as needing a store-row
        backfill — never a rebuild, since the label already attests the
        live container matches desired — is a consumer-side contract
        documented on spec_matches_inspection() in base.py, not something
        this matcher call enforces on its own.
        """
        service = docker_service
        name = _unique_name("bridge-no-store-row")
        config = SandboxConfig(cpus=1, memory=256)
        try:
            await service.create(
                name,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )
            # Simulate the store row having been lost independently of the
            # container (the label is immutable and unaffected).
            service._store.delete_info(name)
            assert service._store.get_info(name) is None

            inspection = await service.inspect(name)
            assert inspection is not None
            desired = _spec_for(config)

            assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass


@requires_docker
class TestSnapshotDoesNotLaunderAttestation:
    """A snapshot must not carry its source sandbox's attestation forward.

    Docker copies a container's labels into the image produced by
    ``commit()``, and then merges an image's labels into any container
    created from it for every key the create request does not specify. So
    ``create(A, spec_A)`` -> ``create_snapshot(A)`` -> a container built from
    that snapshot would inherit ``spec_A``'s fingerprint and present it as
    its own attestation -- reporting MISMATCH where it owes UNVERIFIED, or a
    false MATCH when the specs happen to agree, on a container that never
    went through the verified create() path at all.

    ``_create_container`` closes this by always writing both attestation keys
    itself, blank when there is nothing to attest.
    """

    @pytest.mark.asyncio(loop_scope="module")
    async def test_legacy_container_from_a_snapshot_is_unverified(self, docker_service):
        service = docker_service
        source = _unique_name("snap-src")
        derived = _unique_name("snap-derived")
        snapshot_id = _unique_name("snapid")
        config = SandboxConfig(cpus=1, memory=256)
        try:
            # 1. A verified sandbox: its container carries a real fingerprint.
            await service.create(
                source,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )
            source_inspection = await service.inspect(source)
            assert source_inspection is not None
            assert source_inspection.fingerprint_label, (
                "the source sandbox must actually be attested for this test to "
                "be meaningful"
            )

            # 2. Snapshot it. The committed image inherits those labels.
            await service.create_snapshot(source, snapshot_id)

            # 3. Build a container from that snapshot via the legacy path,
            #    which makes no attestation of its own.
            await service.get_or_create(
                derived,
                template=SandboxTemplate(type="snapshot", snapshot_id=snapshot_id),
                config=config,
            )

            derived_inspection = await service.inspect(derived)
            assert derived_inspection is not None
            assert derived_inspection.fingerprint_label is None, (
                "a legacy container built from a snapshot must present no "
                "attestation, not the source sandbox's inherited fingerprint"
            )

            # The verdict, which is what reconciliation actually consumes.
            assert (
                spec_matches_inspection(_spec_for(config), derived_inspection)
                is SpecVerdict.UNVERIFIED
            ), (
                "an unattested container must be UNVERIFIED: MATCH would be a "
                "forged attestation and MISMATCH would force a needless rebuild"
            )
        finally:
            for name in (derived, source):
                try:
                    await service.delete(name)
                except Exception:
                    pass
            try:
                await service.delete_snapshot(snapshot_id)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_snapshot_derived_container_with_the_same_spec_is_not_a_false_match(
        self, docker_service
    ):
        """The dangerous direction: identical specs must still not report MATCH.

        With an inherited fingerprint this would compare equal and pass the
        live cpus/memory re-check, so the container would be adopted as
        verified without ever having been verified.
        """
        service = docker_service
        source = _unique_name("snap-fm-src")
        derived = _unique_name("snap-fm-derived")
        snapshot_id = _unique_name("snapid-fm")
        config = SandboxConfig(cpus=1, memory=256)
        try:
            await service.create(
                source,
                SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config,
            )
            await service.create_snapshot(source, snapshot_id)
            await service.get_or_create(
                derived,
                template=SandboxTemplate(type="snapshot", snapshot_id=snapshot_id),
                config=config,
            )

            derived_inspection = await service.inspect(derived)
            assert derived_inspection is not None
            verdict = spec_matches_inspection(_spec_for(config), derived_inspection)
            assert verdict is not SpecVerdict.MATCH, (
                "an unverified container must never report MATCH even when the "
                "desired spec is identical to the snapshot source's"
            )
            assert verdict is SpecVerdict.UNVERIFIED
        finally:
            for name in (derived, source):
                try:
                    await service.delete(name)
                except Exception:
                    pass
            try:
                await service.delete_snapshot(snapshot_id)
            except Exception:
                pass
