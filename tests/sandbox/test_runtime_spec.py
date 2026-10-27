"""
Unit tests for the sandbox runtime spec types, matcher, and lifecycle
contract additions in ``xagent.sandbox.base``.

These are pure unit tests with no Docker/Boxlite dependency: SandboxInspection
instances are hand-constructed to drive the matcher through every verdict
path.
"""

from __future__ import annotations

import dataclasses
import os
import stat

import pytest

from xagent.sandbox.base import (
    SPEC_CONTRACT_VERSION,
    ObservedRuntimeFacts,
    ResolvedSandboxRuntimeSpec,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxContractError,
    SandboxInspection,
    SandboxReconcileUnsupportedError,
    SandboxRecoveryRequiredError,
    SandboxRuntimeConflictError,
    SandboxService,
    SandboxTemplate,
    SpecVerdict,
    canonical_sandbox_path,
    grant_sandbox_runtime_access,
    spec_matches_inspection,
)


def _make_spec(**overrides) -> ResolvedSandboxRuntimeSpec:
    kwargs = dict(
        template_type="image",
        image="python:3.12-slim",
        working_dir="/home",
        cpus=1,
        memory=512,
        pids_limit=1024,
        security_profile="nonroot-public-egress-v1",
        env={"FOO": "bar", "BAZ": "qux"},
        volumes=[("/host/a", "/guest/a", "ro")],
        network_isolated=False,
        ports=[(8080, 80)],
    )
    kwargs.update(overrides)
    return ResolvedSandboxRuntimeSpec.from_parts(**kwargs)


def _make_facts(**overrides) -> ObservedRuntimeFacts:
    kwargs = dict(
        raw_status="running",
        image_ref="python:3.12-slim",
        image_digest="sha256:deadbeef",
        raw_nano_cpus=1_000_000_000,
        raw_memory_bytes=512 * 1024 * 1024,
        raw_pids_limit=1024,
        env={"FOO": "bar"},
        volumes=(("/host/a", "/guest/a", "ro"),),
        ports=((8080, 80),),
        network_isolated=False,
        runtime_networks=("bridge",),
        labels={},
        created_at="2026-01-01T00:00:00Z",
        working_dir="/home",
    )
    kwargs.update(overrides)
    return ObservedRuntimeFacts(**kwargs)


def _make_inspection(**overrides) -> SandboxInspection:
    kwargs = dict(
        state="running",
        facts=_make_facts(),
        fingerprint_label=None,
        version_label=str(SPEC_CONTRACT_VERSION),
    )
    kwargs.update(overrides)
    return SandboxInspection(**kwargs)


def test_workspace_access_grants_only_the_mount_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o777)
    nested = workspace / "nested"
    nested.mkdir(mode=0o700)
    source = nested / "source.txt"
    source.write_text("data")
    source.chmod(0o600)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    outside.chmod(0o600)
    (workspace / "outside-link").symlink_to(outside)

    requested_modes: list[int] = []
    real_fchmod = os.fchmod

    def record_fchmod(fd: int, mode: int) -> None:
        requested_modes.append(mode)
        real_fchmod(fd, mode)

    monkeypatch.setattr("xagent.sandbox.base.os.fchown", lambda *args, **kwargs: None)
    monkeypatch.setattr("xagent.sandbox.base.os.fchmod", record_fchmod)
    grant_sandbox_runtime_access(workspace, trusted_root=tmp_path)

    workspace_mode = stat.S_IMODE(workspace.stat().st_mode)
    nested_mode = stat.S_IMODE(nested.stat().st_mode)
    source_mode = stat.S_IMODE(source.stat().st_mode)
    assert workspace_mode & 0o777 == 0o750
    assert requested_modes == [0o2750]
    assert nested_mode == 0o700
    assert source_mode == 0o600
    assert workspace_mode & 0o007 == 0
    assert nested_mode & 0o007 == 0
    assert source_mode & 0o006 == 0
    assert stat.S_IMODE(outside.stat().st_mode) == 0o600


def test_workspace_access_fails_closed_when_gid_cannot_be_assigned(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)

    def refuse_chown(*args, **kwargs):
        raise PermissionError("not permitted")

    monkeypatch.setattr("xagent.sandbox.base.os.fchown", refuse_chown)
    with pytest.raises(SandboxContractError, match="run as root"):
        grant_sandbox_runtime_access(workspace, trusted_root=tmp_path)
    assert stat.S_IMODE(workspace.stat().st_mode) == 0o700


def test_workspace_access_accepts_trusted_symlinked_ancestor(tmp_path, monkeypatch):
    real_root = tmp_path / "real-root"
    workspace = real_root / "workspace"
    workspace.mkdir(parents=True, mode=0o700)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    requested_modes: list[int] = []
    real_fchmod = os.fchmod

    def record_fchmod(fd: int, mode: int) -> None:
        requested_modes.append(mode)
        real_fchmod(fd, mode)

    monkeypatch.setattr("xagent.sandbox.base.os.fchown", lambda *args: None)
    monkeypatch.setattr("xagent.sandbox.base.os.fchmod", record_fchmod)

    grant_sandbox_runtime_access(linked_root / "workspace", trusted_root=linked_root)

    assert stat.S_IMODE(workspace.stat().st_mode) & 0o777 == 0o750
    assert requested_modes == [0o2750]


@pytest.mark.skipif(not hasattr(os, "O_PATH"), reason="O_PATH is Linux-specific")
def test_workspace_access_traverses_execute_only_ancestor(tmp_path, monkeypatch):
    execute_only = tmp_path / "execute-only"
    trusted_root = execute_only / "uploads"
    workspace = trusted_root / "workspace"
    workspace.mkdir(parents=True, mode=0o700)
    execute_only.chmod(0o111)
    monkeypatch.setattr("xagent.sandbox.base.os.fchown", lambda *args: None)

    try:
        grant_sandbox_runtime_access(workspace, trusted_root=trusted_root)
    finally:
        execute_only.chmod(0o755)

    assert stat.S_IMODE(workspace.stat().st_mode) & 0o777 == 0o750


def test_workspace_access_rejects_symlink_in_managed_suffix(tmp_path, monkeypatch):
    trusted_root = tmp_path / "uploads"
    trusted_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (trusted_root / "workspace").symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr("xagent.sandbox.base.os.fchown", lambda *args: None)

    with pytest.raises(SandboxContractError, match="safely open"):
        grant_sandbox_runtime_access(
            trusted_root / "workspace", trusted_root=trusted_root
        )

    assert stat.S_IMODE(outside.stat().st_mode) == 0o700


# --- ResolvedSandboxRuntimeSpec: field sensitivity ---


class TestSpecFieldSensitivity:
    def test_baseline_fingerprint_is_stable(self):
        a = _make_spec()
        b = _make_spec()
        assert a == b
        assert a.fingerprint() == b.fingerprint()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"working_dir": "/other"},
            {"cpus": 2},
            {"memory": 1024},
            {"pids_limit": 2048},
            {"security_profile": "different-profile"},
            {"env": {"FOO": "different", "BAZ": "qux"}},
            {"volumes": [("/host/a", "/guest/a", "rw")]},
            {"network_isolated": True},
            {"ports": [(9090, 90)]},
        ],
    )
    def test_perturbing_any_field_changes_fingerprint(self, overrides):
        baseline = _make_spec()
        perturbed = _make_spec(**overrides)
        assert perturbed != baseline
        assert perturbed.fingerprint() != baseline.fingerprint()

    def test_perturbing_image_changes_fingerprint(self):
        baseline = _make_spec()
        perturbed = _make_spec(image="python:3.11-slim")
        assert perturbed != baseline
        assert perturbed.fingerprint() != baseline.fingerprint()

    def test_template_type_and_snapshot_id_are_covered(self):
        """The two remaining fingerprint fields, which the matrix above cannot
        perturb because ``__post_init__`` constrains them against each other."""
        image_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", working_dir="/home"
        )
        snapshot_spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="snapshot", snapshot_id="snap-1", working_dir="/home"
        )
        assert image_spec != snapshot_spec
        assert image_spec.fingerprint() != snapshot_spec.fingerprint()

        other_snapshot = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="snapshot", snapshot_id="snap-2", working_dir="/home"
        )
        assert other_snapshot != snapshot_spec
        assert other_snapshot.fingerprint() != snapshot_spec.fingerprint()

    def test_every_spec_field_is_covered_by_the_fingerprint(self):
        """Guard against a new field arriving without fingerprint coverage.

        A field that does not move the fingerprint is a field two different
        desired states can silently share one attestation for.
        """
        covered = {
            "template_type",
            "snapshot_id",
            "image",
            "working_dir",
            "cpus",
            "memory",
            "pids_limit",
            "security_profile",
            "env",
            "volumes",
            "network_isolated",
            "ports",
        }
        actual = {f.name for f in dataclasses.fields(ResolvedSandboxRuntimeSpec)}
        assert actual == covered, (
            "a spec field was added or removed; fingerprint() and the "
            "field-sensitivity tests above must be updated to match"
        )


class TestSpecRejectsUnrealizableValues:
    """The spec's constructible domain must equal what the backend can realize.

    ``SandboxConfig`` constrains ``cpus`` (ge=1) and ``memory`` (ge=128), so a
    spec outside those bounds used to construct fine and then fail pydantic
    validation later inside ``create()`` -- surfacing at the backend boundary
    rather than at the desired-state boundary that owns the value.
    """

    @pytest.mark.parametrize("memory", [1, 64, 127])
    def test_memory_below_the_backend_minimum_is_rejected(self, memory):
        with pytest.raises(ValueError, match="memory must be >= 128"):
            _make_spec(memory=memory)

    @pytest.mark.parametrize("cpus", [-1, -8])
    def test_cpus_below_the_backend_minimum_is_rejected(self, cpus):
        with pytest.raises(ValueError, match="cpus must be >= 1"):
            _make_spec(cpus=cpus)

    def test_the_boundary_values_are_accepted(self):
        spec = _make_spec(cpus=1, memory=128)
        assert (spec.cpus, spec.memory) == (1, 128)

    @pytest.mark.parametrize("unset", [None, 0])
    def test_unset_falls_back_to_the_backend_defaults(self, unset):
        """``from_parts`` treats both ``None`` and ``0`` as "not specified",
        matching the backend's own ``cpus or 1`` / ``memory or 512``, so
        neither reaches the bounds check as an out-of-range value."""
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", cpus=unset, memory=unset
        )
        assert (spec.cpus, spec.memory) == (1, 512)

    def test_anything_constructible_converts_to_a_backend_config(self):
        """The property the bounds exist to guarantee."""
        _make_spec(cpus=1, memory=128).to_backend_config()

    def test_direct_construction_is_validated_too(self):
        """``__post_init__`` covers the constructor, not only ``from_parts``."""
        with pytest.raises(ValueError, match="memory must be >= 128"):
            ResolvedSandboxRuntimeSpec(
                template_type="image",
                image="python:3.12-slim",
                snapshot_id=None,
                working_dir="/home",
                cpus=1,
                memory=64,
                env=(),
                volumes=(),
                network_isolated=False,
                ports=(),
            )


# --- ResolvedSandboxRuntimeSpec: order insensitivity ---


class TestSpecOrderInsensitivity:
    def test_env_key_order_does_not_affect_equality_or_fingerprint(self):
        a = _make_spec(env={"FOO": "bar", "BAZ": "qux"})
        b = _make_spec(env={"BAZ": "qux", "FOO": "bar"})
        assert a == b
        assert a.fingerprint() == b.fingerprint()

    def test_volume_order_and_duplicates_do_not_affect_equality(self):
        a = _make_spec(
            volumes=[("/host/a", "/guest/a", "ro"), ("/host/b", "/guest/b", "rw")]
        )
        b = _make_spec(
            volumes=[
                ("/host/b", "/guest/b", "rw"),
                ("/host/a", "/guest/a", "ro"),
                ("/host/a", "/guest/a", "ro"),
            ]
        )
        assert a == b
        assert a.fingerprint() == b.fingerprint()
        assert len(a.volumes) == 2

    def test_port_order_and_duplicates_do_not_affect_equality(self):
        a = _make_spec(ports=[(8080, 80), (9090, 90)])
        b = _make_spec(ports=[(9090, 90), (9090, 90), (8080, 80)])
        assert a == b
        assert a.fingerprint() == b.fingerprint()
        assert len(a.ports) == 2

    def test_volume_paths_are_normalized(self):
        a = _make_spec(volumes=[("/host/a", "/guest/a", "ro")])
        b = _make_spec(volumes=[("/host/x/../a", "/guest/./a", "ro")])
        assert a == b
        assert a.fingerprint() == b.fingerprint()


# --- canonical_sandbox_path: the leading-// case normpath alone keeps ---


class TestCanonicalSandboxPath:
    """A desired path must be spelled the way the backend reports it back.

    ``posixpath.normpath`` preserves exactly two leading slashes (POSIX
    reserves ``//`` for implementation-defined interpretation) while Docker
    collapses them in ``Mounts.Destination`` and ``Config.WorkingDir``, so a
    ``//``-prefixed desired path would fail create()'s
    publish-before-verify byte comparison against a container that is in
    fact exactly what was asked for.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("//data/uploads", "/data/uploads"),
            ("///data/uploads", "/data/uploads"),
            ("//data/./uploads/../uploads", "/data/uploads"),
            ("//", "/"),
            ("/data/uploads", "/data/uploads"),
            ("/data//uploads", "/data/uploads"),
        ],
    )
    def test_collapses_leading_slash_run(self, raw, expected):
        assert canonical_sandbox_path(raw) == expected

    def test_double_slash_spec_equals_single_slash_spec(self):
        a = _make_spec(volumes=[("/host/a", "/guest/a", "ro")], working_dir="/home")
        b = _make_spec(volumes=[("//host/a", "//guest/a", "ro")], working_dir="//home")
        assert a == b
        assert a.fingerprint() == b.fingerprint()

    def test_spec_paths_match_what_the_backend_reports_back(self):
        """The observed side stays backend-native; the desired side is the
        one that has to be canonical."""
        spec = _make_spec(
            volumes=[("//data/uploads", "//workspace", "rw")],
            working_dir="//home",
        )
        # What Docker echoes for that request (leading run collapsed).
        assert spec.volumes == (("/data/uploads", "/workspace", "rw"),)
        assert spec.working_dir == "/home"


# --- ResolvedSandboxRuntimeSpec: repr redaction ---


class TestSpecReprRedaction:
    def test_repr_does_not_leak_env_values_or_paths(self):
        spec = _make_spec(
            env={"SECRET_TOKEN": "sup3r-s3cr3t-value"},
            volumes=[("/very/private/host/path", "/guest/a", "ro")],
        )
        text = repr(spec)
        assert "sup3r-s3cr3t-value" not in text
        assert "SECRET_TOKEN" not in text
        assert "/very/private/host/path" not in text
        assert "/guest/a" not in text
        assert spec.fingerprint()[:12] in text

    def test_repr_reports_field_counts(self):
        spec = _make_spec(
            env={"A": "1", "B": "2"},
            volumes=[("/host/a", "/guest/a", "ro")],
            ports=[(1, 1), (2, 2)],
        )
        text = repr(spec)
        assert "env_count=2" in text
        assert "volumes_count=1" in text
        assert "ports_count=2" in text


# --- ResolvedSandboxRuntimeSpec: image/snapshot mutual exclusion ---


class TestSpecTemplateMutualExclusion:
    def test_image_type_requires_image(self):
        with pytest.raises(ValueError):
            ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image=None,
                snapshot_id=None,
            )

    def test_image_type_forbids_snapshot_id(self):
        with pytest.raises(ValueError):
            ResolvedSandboxRuntimeSpec.from_parts(
                template_type="image",
                image="python:3.12-slim",
                snapshot_id="snap-1",
            )

    def test_snapshot_type_requires_snapshot_id(self):
        with pytest.raises(ValueError):
            ResolvedSandboxRuntimeSpec.from_parts(
                template_type="snapshot",
                image=None,
                snapshot_id=None,
            )

    def test_snapshot_type_forbids_image(self):
        with pytest.raises(ValueError):
            ResolvedSandboxRuntimeSpec.from_parts(
                template_type="snapshot",
                image="python:3.12-slim",
                snapshot_id="snap-1",
            )

    def test_snapshot_type_is_valid_on_its_own(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="snapshot",
            image=None,
            snapshot_id="snap-1",
        )
        assert spec.snapshot_id == "snap-1"


# --- ResolvedSandboxRuntimeSpec: from_parts sentinel defaults ---


class TestFromPartsSentinelDefaults:
    """A resolved spec must never carry a 0/None cpus/memory or an
    un-normalized working_dir: from_parts applies the same defaults the
    Docker backend applies at container creation, so both sides of a later
    publish-verification comparison are computed from one canonical form."""

    def test_cpus_none_defaults_to_one(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", cpus=None
        )
        assert spec.cpus == 1

    def test_memory_none_defaults_to_512(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", memory=None
        )
        assert spec.memory == 512

    def test_working_dir_none_defaults_to_sandbox_home(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", working_dir=None
        )
        assert spec.working_dir == "/home/sandbox"

    def test_working_dir_trailing_slash_is_normalized(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="python:3.12-slim", working_dir="/home/"
        )
        assert spec.working_dir == "/home"


# --- ResolvedSandboxRuntimeSpec: contract version is out-of-band ---


class TestContractVersionNotInFingerprint:
    def test_contract_version_is_not_a_spec_field(self):
        field_names = {f.name for f in dataclasses.fields(ResolvedSandboxRuntimeSpec)}
        assert "contract_version" not in field_names

    def test_module_constant_exists(self):
        assert SPEC_CONTRACT_VERSION == 2


# --- ResolvedSandboxRuntimeSpec: to_backend_config round trip ---


class TestToBackendConfig:
    def test_round_trip_image_template(self):
        spec = _make_spec()
        template, config = spec.to_backend_config()
        assert isinstance(template, SandboxTemplate)
        assert isinstance(config, SandboxConfig)
        assert template.type == "image"
        assert template.image == "python:3.12-slim"
        assert template.snapshot_id is None
        assert config.working_dir == "/home"
        assert config.cpus == 1
        assert config.memory == 512
        assert config.pids_limit == 1024
        assert config.security_profile == "nonroot-public-egress-v1"
        assert config.env == {"FOO": "bar", "BAZ": "qux"}
        assert config.volumes == [("/host/a", "/guest/a", "ro")]
        assert config.network_isolated is False
        assert config.ports == [(8080, 80)]

    def test_round_trip_snapshot_template(self):
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="snapshot",
            image=None,
            snapshot_id="snap-1",
        )
        template, _config = spec.to_backend_config()
        assert template.type == "snapshot"
        assert template.snapshot_id == "snap-1"
        assert template.image is None


# --- Exceptions: never RuntimeError ---


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            SandboxContractError,
            SandboxAlreadyExistsError,
            SandboxRuntimeConflictError,
            SandboxRecoveryRequiredError,
            SandboxReconcileUnsupportedError,
        ],
    )
    def test_not_a_runtime_error(self, exc_cls):
        assert not issubclass(exc_cls, RuntimeError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            SandboxAlreadyExistsError,
            SandboxRuntimeConflictError,
            SandboxRecoveryRequiredError,
            SandboxReconcileUnsupportedError,
        ],
    )
    def test_subclasses_of_contract_error(self, exc_cls):
        assert issubclass(exc_cls, SandboxContractError)


# --- SandboxService: default implementations ---


class _MinimalSandboxService(SandboxService):
    """Smallest possible concrete SandboxService for pinning default bodies."""

    async def get_or_create(self, name, template=None, config=None):
        raise NotImplementedError

    async def list_sandboxes(self):
        raise NotImplementedError

    async def delete(self, name):
        raise NotImplementedError

    async def supports_snapshots(self):
        return False

    async def create_snapshot(self, name, snapshot_id):
        raise NotImplementedError

    async def list_snapshots(self):
        raise NotImplementedError

    async def delete_snapshot(self, snapshot_id):
        raise NotImplementedError


class TestSandboxServiceDefaults:
    async def test_supports_runtime_spec_defaults_false(self):
        service = _MinimalSandboxService()
        assert await service.supports_runtime_spec() is False

    async def test_inspect_defaults_to_unsupported(self):
        service = _MinimalSandboxService()
        with pytest.raises(SandboxReconcileUnsupportedError):
            await service.inspect("some-name")

    async def test_create_defaults_to_unsupported(self):
        service = _MinimalSandboxService()
        with pytest.raises(SandboxReconcileUnsupportedError):
            await service.create("some-name", SandboxTemplate(), SandboxConfig())

    async def test_start_existing_defaults_to_unsupported(self):
        service = _MinimalSandboxService()
        with pytest.raises(SandboxReconcileUnsupportedError):
            await service.start_existing("some-name")

    async def test_stop_existing_defaults_to_unsupported(self):
        service = _MinimalSandboxService()
        with pytest.raises(SandboxReconcileUnsupportedError):
            await service.stop_existing("some-name")


# --- Matcher: spec_matches_inspection ---


class TestSpecMatchesInspection:
    def test_missing_fingerprint_label_is_unverified(self):
        desired = _make_spec()
        inspection = _make_inspection(fingerprint_label=None)
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED

    def test_missing_version_label_is_unverified(self):
        desired = _make_spec()
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(), version_label=None
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED

    def test_older_version_label_is_unverified(self):
        desired = _make_spec()
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            version_label=str(SPEC_CONTRACT_VERSION - 1),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.UNVERIFIED

    def test_matching_label_and_matching_raw_units_is_match(self):
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(
                raw_nano_cpus=1_000_000_000, raw_memory_bytes=512 * 1024 * 1024
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH

    def test_docker_update_cpus_drift_is_mismatch(self):
        # `docker update --cpus 0.5` after creation with desired cpus=1.
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(
                raw_nano_cpus=500_000_000, raw_memory_bytes=512 * 1024 * 1024
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MISMATCH

    def test_memory_drift_is_mismatch(self):
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(
                raw_nano_cpus=1_000_000_000, raw_memory_bytes=1024 * 1024 * 1024
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MISMATCH

    @pytest.mark.parametrize("raw_pids_limit", [2048, -1, 0, None])
    def test_pids_limit_drift_is_mismatch(self, raw_pids_limit):
        desired = _make_spec(pids_limit=1024)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(raw_pids_limit=raw_pids_limit),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MISMATCH

    @pytest.mark.parametrize("raw_pids_limit", [-1, 0, None])
    def test_unlimited_pids_values_match_unlimited_desired(self, raw_pids_limit):
        desired = _make_spec(pids_limit=None)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(raw_pids_limit=raw_pids_limit),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH

    def test_label_mismatch_is_mismatch_regardless_of_raw_units(self):
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label="not-the-real-fingerprint",
            facts=_make_facts(
                raw_nano_cpus=1_000_000_000, raw_memory_bytes=512 * 1024 * 1024
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MISMATCH

    def test_env_volumes_ports_are_not_directly_compared(self):
        # Label + raw cpu/memory agree; env/volumes/ports on facts differ from
        # desired entirely. Verdict must still be MATCH because these fields
        # are covered by label attestation, not live re-comparison.
        desired = _make_spec(cpus=1, memory=512)
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(),
            facts=_make_facts(
                raw_nano_cpus=1_000_000_000,
                raw_memory_bytes=512 * 1024 * 1024,
                env={"UNRELATED": "value"},
                volumes=(("/totally/different", "/x", "rw"),),
                ports=((1, 1),),
            ),
        )
        assert spec_matches_inspection(desired, inspection) is SpecVerdict.MATCH

    def test_current_contract_version_override(self):
        desired = _make_spec()
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(), version_label="5"
        )
        assert (
            spec_matches_inspection(desired, inspection, current_contract_version=5)
            is SpecVerdict.MATCH
        )
        assert (
            spec_matches_inspection(desired, inspection, current_contract_version=6)
            is SpecVerdict.UNVERIFIED
        )

    def test_newer_version_label_is_unverified(self):
        # A label written by a newer contract version than this code
        # currently implements: the old `<` comparison would fall through
        # to a fingerprint/raw-unit comparison this code cannot safely make
        # sense of; `!=` correctly reports UNVERIFIED for either direction
        # of version mismatch.
        desired = _make_spec()
        inspection = _make_inspection(
            fingerprint_label=desired.fingerprint(), version_label="2"
        )
        assert (
            spec_matches_inspection(desired, inspection, current_contract_version=1)
            is SpecVerdict.UNVERIFIED
        )


# --- ObservedRuntimeFacts / SandboxInspection: identity equality only ---


class TestObservedFactsAndInspectionIdentityEquality:
    def test_facts_equality_is_identity_not_structural(self):
        facts = _make_facts()
        assert (facts == dataclasses.replace(facts)) is False

    def test_inspection_equality_is_identity_not_structural(self):
        inspection = _make_inspection()
        assert (inspection == dataclasses.replace(inspection)) is False
