"""
Abstract interface for Sandbox Service.
"""

from __future__ import annotations

import abc
import hashlib
import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

TemplateType = Literal["image", "snapshot"]
"""Supported template types."""

CodeType = Literal["python", "javascript"]
"""Supported code execution types."""


class SandboxNotFoundError(Exception):
    """Raised when a requested sandbox resource no longer exists."""


class SandboxContractError(Exception):
    """Base class for violations of the explicit sandbox lifecycle contract.

    Deliberately does not inherit from ``RuntimeError``: some call sites
    catch bare ``RuntimeError`` for unrelated recovery paths, and inheriting
    it would silently fold these lifecycle contract violations into that
    generic handling instead of surfacing as their own explicit error type.
    """


class SandboxAlreadyExistsError(SandboxContractError):
    """Raised when a create() targets a name that already has a container.

    The new lifecycle contract's create() is not idempotent: an existing
    container in any state (created, running, or stopped) raises this error
    rather than being silently adopted.
    """


class SandboxRuntimeConflictError(SandboxContractError):
    """Raised when a sandbox's desired state conflicts with itself or with
    what the backend actually materialized.

    Covers two distinct checks against the same ``create()`` desired spec:

    - Static pre-create validation: two volume mounts that share a host path
      but disagree on guest path or mode, or two port mappings that share a
      guest port but disagree on host port (or vice versa). The backend's
      internal container-creation path indexes mounts by host path and ports
      by guest port, so it would otherwise silently drop one of them.
    - Publish-before-verify: after the container is created and started, its
      observed runtime facts disagree with the desired spec it was created
      from (see ``create()``'s step 6). The container is removed and this is
      raised rather than publishing a store record for a container that
      isn't what was asked for.
    """


class SandboxMountEscapeError(SandboxContractError):
    """Raised when a mount candidate escapes the root that must contain it.

    ``SandboxMountIntent``'s covered/covering/disjoint split is lexical, so
    a caller that needs a candidate to be *physically* inside (or to
    physically contain) the mount root owns the resolved view as well. When
    such a candidate's lexical containment and its ``realpath`` containment
    disagree, no fold verdict is safe: dropping it loses access to a path
    the surviving bind does not expose, promoting it re-roots onto a
    directory that does not contain the old root, and granting it a separate
    bind exposes whatever the symlink points at.

    Callers raise this for candidates whose containment is a precondition
    rather than an observation -- a path derived from a workspace root that
    the mount root is required to cover. A candidate that is an
    independently declared mount in its own right (an operator-configured
    external directory) has no such precondition and keeps its own bind
    instead.
    """


class SandboxRecoveryRequiredError(SandboxContractError):
    """Raised when a sandbox is in a state that needs recovery before use.

    Signals that the caller (or a reconciliation pass) must resolve the
    sandbox's state before it can be treated as available.
    """


class SandboxReconcileUnsupportedError(SandboxContractError):
    """Raised by the default body of a spec-reconciliation lifecycle method.

    Backends that have not implemented ``inspect``/``create``/
    ``start_existing``/``stop_existing`` inherit this default; callers
    should gate on ``supports_runtime_spec()`` rather than relying on this
    exception for control flow.
    """


class SandboxTemplate(BaseModel):
    """
    Template for creating a sandbox.

    `type="image"` creates a sandbox from a container image.

    `type="snapshot"` creates a sandbox from a previously committed filesystem
    snapshot. A snapshot is only a creation template for the new sandbox's
    initial filesystem contents; runtime configuration such as working
    directory, environment variables, volume mounts, network isolation, and
    port mappings still comes from `SandboxConfig` on the current
    `get_or_create()` call.
    """

    type: Optional[TemplateType] = Field(default="image", description="Template type")

    image: Optional[str] = Field(
        default=None, description="Container image, required when type=image"
    )

    snapshot_id: Optional[str] = Field(
        default=None, description="Snapshot ID, required when type=snapshot"
    )


class SandboxConfig(BaseModel):
    """
    Configuration parameters for creating a sandbox.
    """

    working_dir: Optional[str] = Field(default="/home", description="Working dir")

    cpus: Optional[int] = Field(default=1, ge=1, description="CPU core limit")

    memory: Optional[int] = Field(default=512, ge=128, description="Memory limit in MB")

    env: Optional[dict[str, str]] = Field(
        default=None, description="Environment variables to inject"
    )

    volumes: Optional[list[tuple[str, str, str]]] = Field(
        default=None,
        description="Volume mounts as (host_path, guest_path, mode). Mode: 'ro' (read-only) or 'rw' (read-write)",
    )

    network_isolated: Optional[bool] = Field(
        default=False,
        description="Network isolation. True blocks external network access",
    )

    ports: Optional[list[tuple[int, int]]] = Field(
        default=None, description="Port mappings as [(host_port, guest_port)]"
    )


class SandboxInfo(BaseModel):
    """Sandbox status information."""

    name: str = Field(description="Sandbox name")

    state: str = Field(description="Sandbox state: 'running', 'stopped', or 'unknown'")

    template: SandboxTemplate = Field(
        description="Template used to create this sandbox"
    )

    config: SandboxConfig = Field(
        description="Configuration used to create this sandbox"
    )

    created_at: Optional[str] = Field(
        default=None, description="Creation time in ISO 8601 format"
    )


class SandboxSnapshot(BaseModel):
    """Sandbox snapshot information."""

    snapshot_id: str = Field(description="Snapshot ID")

    metadata: dict = Field(default_factory=dict, description="Snapshot metadata")

    created_at: Optional[str] = Field(
        default=None, description="Creation time in ISO 8601 format"
    )


class ExecResult(BaseModel):
    """Execution result of a command or code."""

    exit_code: int = Field(
        description="Exit code. 0 indicates success, non-zero indicates failure"
    )

    stdout: str = Field(description="Standard output")

    stderr: str = Field(description="Standard error output")

    truncated: bool = Field(
        default=False,
        description="True if output was capped at max_output_bytes and cut short",
    )

    error_message: Optional[str] = Field(default=None, description="Error message")

    @property
    def success(self) -> bool:
        return self.exit_code == 0


# Contract version for the fingerprint+label attestation written by create().
# Bumped when the set of fields covered by fingerprint()/create()'s
# publish-before-verify step changes in a way that makes an old label
# untrustworthy for matching against a newly-desired spec. Not part of the
# fingerprint itself (see ResolvedSandboxRuntimeSpec.fingerprint), so bumping
# it does not change any existing fingerprint value.
SPEC_CONTRACT_VERSION = 1

# Realizable bounds for a resolved spec, mirroring the `ge=` constraints
# SandboxConfig declares for the same two fields. Kept here so an
# unrealizable desired state fails at spec construction rather than later,
# inside the backend conversion.
_MIN_SPEC_CPUS = 1
_MIN_SPEC_MEMORY_MB = 128


def canonical_sandbox_path(path: str) -> str:
    """Canonicalize one sandbox-domain path for desired-state comparison.

    The sandbox path domain is POSIX on both sides of a mount: guest paths
    are container paths, and host bind sources are paths on the machine
    running the container backend (in Docker sibling mode, the Docker
    host — not necessarily this process's filesystem, see
    ``XAGENT_SANDBOX_HOST_STORAGE_ROOT``). Normalization therefore belongs
    to ``posixpath``, never to ``os.path``.

    ``posixpath.normpath`` alone is not a canonical form: POSIX reserves a
    leading ``//`` for implementation-defined interpretation, so normpath
    keeps exactly two leading slashes (``'//data'`` stays ``'//data'``)
    while Docker collapses them in the paths it reports back
    (``Mounts.Destination``, ``Config.WorkingDir``). A desired-state path
    that keeps ``//`` can therefore never byte-match what the backend
    echoes, which turns a correctly-created container into a
    publish-verification mismatch, and lets ``'//x'`` and ``'/x'`` pass the
    pre-create conflict checks as if they were different mount points when
    the backend would collapse them onto the same one. Collapsing the
    leading slash run is what makes the desired form the form the backend
    reports.
    """
    normalized = posixpath.normpath(path)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    return normalized


@dataclass(frozen=True, repr=False)
class ResolvedSandboxRuntimeSpec:
    """Fully-resolved, canonical desired runtime configuration for a sandbox.

    This is the single authoritative expression of "what should this sandbox
    look like": structural equality between two instances is authoritative
    equivalence, and ``fingerprint()`` is a stable hash of that same
    structure. All collection fields are normalized primitive tuples (sorted,
    deduplicated) so that construction order never affects equality or the
    fingerprint.

    Exactly one of ``image`` / ``snapshot_id`` must be set, matching
    ``template_type``. The image reference is a tag, not a content digest:
    the fingerprint intentionally does not cover what the tag currently
    resolves to.

    This type has no ``from_backend_info`` counterpart: desired state and
    observed state are different types (``ObservedRuntimeFacts``) because
    they are not interchangeable and must never be silently coerced into one
    another.

    Construct via ``from_parts()`` rather than the constructor directly
    unless the caller has already normalized every field: ``working_dir``,
    ``cpus`` and ``memory`` are declared non-optional because a resolved spec
    always carries the backend defaults applied by ``from_parts``.
    """

    template_type: TemplateType
    image: Optional[str]
    snapshot_id: Optional[str]
    working_dir: str
    cpus: int
    memory: int
    env: tuple[tuple[str, str], ...]
    volumes: tuple[tuple[str, str, str], ...]
    network_isolated: bool
    ports: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.template_type == "image":
            if not self.image or self.snapshot_id:
                raise ValueError(
                    "template_type='image' requires 'image' and forbids 'snapshot_id'"
                )
        elif self.template_type == "snapshot":
            if not self.snapshot_id or self.image:
                raise ValueError(
                    "template_type='snapshot' requires 'snapshot_id' and forbids 'image'"
                )
        else:
            raise ValueError(f"unknown template_type: {self.template_type!r}")

        # Keep this type's constructible domain identical to the domain
        # to_backend_config() can actually realize. SandboxConfig declares
        # cpus ge=1 and memory ge=128, so a spec outside those bounds used to
        # construct fine and then fail pydantic validation later, inside
        # create(). Since this type is the canonical desired-state
        # expression, an unrealizable desired state must be rejected at its
        # own boundary rather than at the backend conversion.
        if self.cpus < _MIN_SPEC_CPUS:
            raise ValueError(
                f"cpus must be >= {_MIN_SPEC_CPUS} to be realizable, got {self.cpus!r}"
            )
        if self.memory < _MIN_SPEC_MEMORY_MB:
            raise ValueError(
                f"memory must be >= {_MIN_SPEC_MEMORY_MB} MB to be realizable, "
                f"got {self.memory!r}"
            )

    @classmethod
    def from_parts(
        cls,
        *,
        template_type: TemplateType,
        image: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        working_dir: Optional[str] = None,
        cpus: Optional[int] = None,
        memory: Optional[int] = None,
        env: Optional[Mapping[str, str]] = None,
        volumes: Optional[Sequence[tuple[str, str, str]]] = None,
        network_isolated: bool = False,
        ports: Optional[Sequence[tuple[int, int]]] = None,
    ) -> "ResolvedSandboxRuntimeSpec":
        """Build a spec, normalizing all collection fields.

        Paths are canonicalized with ``canonical_sandbox_path``; env/volumes/
        ports are sorted and exactly deduplicated so that input order and
        duplicate entries never affect equality or the fingerprint.
        ``cpus``/``memory`` fall back to the same defaults the Docker backend
        applies at container creation (``cpus or 1``, ``memory or 512``) so a
        resolved spec never carries a 0/None value the backend would silently
        have upgraded on its own. ``working_dir`` defaults to ``"/home"`` and
        is canonicalized the same way.
        """
        normalized_env = tuple(sorted((env or {}).items()))
        normalized_volumes = tuple(
            sorted(
                {
                    (
                        canonical_sandbox_path(host),
                        canonical_sandbox_path(guest),
                        mode,
                    )
                    for host, guest, mode in (volumes or [])
                }
            )
        )
        normalized_ports = tuple(
            sorted({(host, guest) for host, guest in (ports or [])})
        )
        return cls(
            template_type=template_type,
            image=image,
            snapshot_id=snapshot_id,
            working_dir=(
                canonical_sandbox_path(working_dir) if working_dir else "/home"
            ),
            cpus=cpus or 1,
            memory=memory or 512,
            env=normalized_env,
            volumes=normalized_volumes,
            network_isolated=network_isolated,
            ports=normalized_ports,
        )

    def to_backend_config(self) -> tuple["SandboxTemplate", "SandboxConfig"]:
        """Produce the backend-private (template, config) pair for this spec.

        This conversion is one-way: backends consume the result to create or
        describe a sandbox, but nothing reconstructs a
        ``ResolvedSandboxRuntimeSpec`` from a ``SandboxTemplate``/
        ``SandboxConfig`` pair.
        """
        template = SandboxTemplate(
            type=self.template_type, image=self.image, snapshot_id=self.snapshot_id
        )
        config = SandboxConfig(
            working_dir=self.working_dir,
            cpus=self.cpus,
            memory=self.memory,
            env=dict(self.env) or None,
            volumes=[tuple(volume) for volume in self.volumes] or None,
            network_isolated=self.network_isolated,
            ports=[tuple(port) for port in self.ports] or None,
        )
        return template, config

    def fingerprint(self) -> str:
        """Stable sha256 hash of this spec's full canonical structure.

        Every field is covered, including ``env``. ``SPEC_CONTRACT_VERSION``
        is not covered: it is written and checked as an independent label
        alongside the fingerprint (see ``spec_matches_inspection``).
        """
        canonical = repr(
            (
                self.template_type,
                self.image,
                self.snapshot_id,
                self.working_dir,
                self.cpus,
                self.memory,
                self.env,
                self.volumes,
                self.network_isolated,
                self.ports,
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        # Deliberately excludes env values and paths: this repr can end up in
        # logs and error messages, and env/volumes routinely carry secrets or
        # sensitive host paths.
        return (
            f"{type(self).__name__}(fingerprint={self.fingerprint()[:12]}, "
            f"env_count={len(self.env)}, volumes_count={len(self.volumes)}, "
            f"ports_count={len(self.ports)})"
        )


@dataclass(eq=False)
class ObservedRuntimeFacts:
    """Raw facts read back from a live backend for a single sandbox.

    Values are kept in their observed, backend-native form rather than
    normalized to match ``ResolvedSandboxRuntimeSpec``: this type intentionally
    has no ``fingerprint()`` and no promised equality semantics, because
    observed state is not a desired-state expression and must never be
    compared as if it were one. ``eq=False`` makes this explicit at the type
    level: two instances compare by identity, never structurally, so nothing
    can accidentally treat one as a desired-state stand-in for the other.

    ``raw_nano_cpus`` / ``raw_memory_bytes`` are the untouched values from the
    backend (e.g. Docker's ``HostConfig.NanoCpus`` / ``HostConfig.Memory``),
    not the divided-and-clamped values a display-facing parser would produce;
    a live edit such as ``docker update --cpus 0.5`` must be observable here.

    ``runtime_networks`` reflects network attachments at inspection time; it
    is not part of any spec comparison because network attachment can change
    at runtime independent of the sandbox's declared configuration.
    """

    raw_status: str
    image_ref: Optional[str]
    image_digest: Optional[str]
    raw_nano_cpus: Optional[int]
    raw_memory_bytes: Optional[int]
    env: Mapping[str, str]
    volumes: tuple[tuple[str, str, str], ...]
    ports: tuple[tuple[int, int], ...]
    network_isolated: bool
    runtime_networks: tuple[str, ...]
    labels: Mapping[str, str]
    created_at: Optional[str]
    working_dir: Optional[str]


@dataclass(eq=False)
class SandboxInspection:
    """Point-in-time observation of a sandbox: state, facts, and labels.

    ``eq=False`` for the same reason as ``ObservedRuntimeFacts``: this is a
    point-in-time observation, not a desired-state value, so two instances
    compare by identity rather than structurally.

    ``state`` is a deliberate two-value reduction for the reconcile
    decision, which only ever branches on "is it running or not". It cannot
    express the difference between ``created`` (never started),  ``exited``,
    ``dead`` and ``restarting`` — all four reduce to ``"stopped"``. A caller
    that needs to treat a never-started container specially must read
    ``facts.raw_status``, which carries the backend's own status string
    unreduced for exactly that purpose.
    """

    state: Literal["running", "stopped"]
    facts: ObservedRuntimeFacts
    fingerprint_label: Optional[str]
    version_label: Optional[str]


def _require_absolute_mount_path(path: str, field: str) -> None:
    """Reject a mount path that lexical prefix comparison cannot classify.

    ``SandboxMountIntent``'s covered/covering/disjoint split is a pure
    string-prefix operation, so only absolute POSIX paths are comparable.
    An empty or relative value would normalize to something like ``"."`` and
    then read as disjoint from every absolute path.
    """
    if not path or not path.startswith("/"):
        raise ValueError(
            f"{field} must be a non-empty absolute POSIX path, got {path!r}"
        )


def _is_root_or_descendant(path: str, root: str) -> bool:
    """Whether ``path`` equals ``root`` or is a lexical descendant of it."""
    if path == root:
        return True
    prefix = root if root.endswith("/") else root + "/"
    return path.startswith(prefix)


@dataclass(frozen=True)
class SandboxMountIntent:
    """A desired mount root plus a set of independently-declared extra mounts.

    Inputs are canonicalized with ``canonical_sandbox_path``, sorted, and
    exactly deduplicated. Every path must be absolute: the three
    classification properties below are pure lexical prefix comparisons, so
    a relative path silently classifies as ``disjoint`` against every
    absolute extra. Because that verdict feeds callers' allow-list
    decisions, and "disjoint" is the direction that grants a separate mount
    rather than folding it into an already-approved root, a non-absolute
    input is rejected here instead of being normalized into a wrong answer.
    (``canonical_sandbox_path("")`` is ``"."``, which is exactly such a
    silently-wrong root.)

    Classification into ``covered_extras`` / ``covering_extras`` /
    ``disjoint_extras`` is purely lexical string comparison over the
    normalized paths — it does not touch the filesystem and cannot detect
    symlinks, bind-mount aliasing, or host-side path mappings, and this type
    performs no resolution of its own. A caller acting on the verdict owns
    the resolved view: both folding directions (dropping a covered extra,
    promoting a covering one) assume the surviving mount physically contains
    the dropped path, and a symlink breaks that in either direction. So
    either pass backend-side paths that are already resolved (e.g. via
    ``realpath``), or classify the resolved paths as well and treat a
    disagreement between the two verdicts as disjoint. Deciding what to do
    with the disjoint set (e.g. fail-closed against an allow-list) is left
    to the caller.
    """

    mount_root: Optional[str] = None
    extra_mounts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mount_root is not None:
            _require_absolute_mount_path(self.mount_root, "mount_root")
        for path in self.extra_mounts:
            _require_absolute_mount_path(path, "extra_mounts entry")
        normalized_root = (
            canonical_sandbox_path(self.mount_root)
            if self.mount_root is not None
            else None
        )
        normalized_extras = tuple(
            sorted({canonical_sandbox_path(path) for path in self.extra_mounts})
        )
        object.__setattr__(self, "mount_root", normalized_root)
        object.__setattr__(self, "extra_mounts", normalized_extras)

    @property
    def covered_extras(self) -> tuple[str, ...]:
        """Extra mounts that are the mount root itself or its descendants."""
        if self.mount_root is None:
            return ()
        return tuple(
            path
            for path in self.extra_mounts
            if _is_root_or_descendant(path, self.mount_root)
        )

    @property
    def covering_extras(self) -> tuple[str, ...]:
        """Extra mounts that are proper ancestors of the mount root."""
        if self.mount_root is None:
            return ()
        return tuple(
            path
            for path in self.extra_mounts
            if path != self.mount_root and _is_root_or_descendant(self.mount_root, path)
        )

    @property
    def disjoint_extras(self) -> tuple[str, ...]:
        """Extra mounts that are neither covered by nor covering the root."""
        covered = set(self.covered_extras)
        covering = set(self.covering_extras)
        return tuple(
            path
            for path in self.extra_mounts
            if path not in covered and path not in covering
        )


class SpecVerdict(Enum):
    """Result of comparing a desired spec against a live inspection."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNVERIFIED = "unverified"


def spec_matches_inspection(
    desired: ResolvedSandboxRuntimeSpec,
    inspection: SandboxInspection,
    *,
    current_contract_version: int = SPEC_CONTRACT_VERSION,
) -> SpecVerdict:
    """Compare a desired spec against a live inspection's attestation.

    This is advisory, not an execution instruction: callers decide what to
    do with each verdict rather than treating this function as authorizing
    an action by itself.

    - No fingerprint label, or a version label that is missing or does not
      exactly match ``current_contract_version``, yields ``UNVERIFIED`` —
      never ``MISMATCH``. An older label is a stale attestation that cannot
      be trusted against today's fingerprint rules; a newer label means this
      code does not yet know that version's fingerprint rules either, so
      neither direction of mismatch can be safely compared. Treating an
      unlabeled or version-mismatched container as a mismatch would force a
      rebuild of every pre-existing container the first time this contract
      is enabled or bumped, which is not an acceptable outcome; the label is
      also immutable once written, so there is no backfill path that could
      turn a legacy container into a verified one.
    - A matching fingerprint label is followed by a live re-check of the
      fields Docker allows to drift after creation (cpus/memory, compared in
      their raw backend units) to catch e.g. ``docker update --cpus 0.5``.
      Disagreement there is ``MISMATCH`` even though the label matched.
    - A non-matching fingerprint label is always ``MISMATCH``.
    - ``env`` / ``volumes`` / ``ports`` / ``image`` are not independently
      re-compared against live facts: they are immutable after creation, so
      the fingerprint label attests to them once and for all as long as
      the label was written by a verified create() call. ``env`` in
      particular cannot be reliably reconstructed from an inspection at all,
      since a container's observed environment mixes in image-defined
      values that were never part of the desired spec.

    Callers that receive ``UNVERIFIED`` for a sandbox that has a
    corresponding store record should fall back to a full desired-state
    comparison against the record — recognizing that this is blind to
    drift in the actual running container and does not verify it, only the
    previously-recorded intent. A sandbox with a matching fingerprint label
    but no store record (``MATCH`` here, row missing) is the one case
    reconciliation must always treat as needing a store-row backfill, not a
    rebuild: the label already attests the live container matches
    ``desired``, so destroying it over a persistence-layer gap would be
    pure waste and would turn a row-write failure into an unnecessary
    container-destruction event; writing the missing row is the only
    action needed to bring label and record into agreement. Regardless of
    verdict, any destructive action still must go through the
    reference-count check that applies to all sandbox teardown; the
    fallback for a non-zero-reference-count sandbox is to reject new
    callers, not to tear down one already in use.
    """
    if inspection.fingerprint_label is None or inspection.version_label is None:
        return SpecVerdict.UNVERIFIED
    try:
        observed_version = int(inspection.version_label)
    except (TypeError, ValueError):
        return SpecVerdict.UNVERIFIED
    if observed_version != current_contract_version:
        return SpecVerdict.UNVERIFIED
    if inspection.fingerprint_label != desired.fingerprint():
        return SpecVerdict.MISMATCH

    facts = inspection.facts
    desired_nano_cpus = int(desired.cpus * 1_000_000_000)
    desired_memory_bytes = int(desired.memory * 1024 * 1024)
    observed_nano_cpus = facts.raw_nano_cpus or 0
    observed_memory_bytes = facts.raw_memory_bytes or 0
    if observed_nano_cpus != desired_nano_cpus:
        return SpecVerdict.MISMATCH
    if observed_memory_bytes != desired_memory_bytes:
        return SpecVerdict.MISMATCH
    return SpecVerdict.MATCH


class Sandbox(abc.ABC):
    """
    Abstract interface for a sandbox instance.

    Supports two usage patterns:

        # Manual stop
        try:
            result = await sandbox.exec("echo hello")
        finally:
            await sandbox.stop()

        # Auto-stop with async context manager
        async with sandbox:
            result = await sandbox.exec("echo hello")
    """

    async def __aenter__(self) -> "Sandbox":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()

    # --- Properties ---

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Sandbox name (unique identifier)."""

    # --- Lifecycle ---

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the sandbox, preserving its state."""

    @abc.abstractmethod
    async def info(self) -> SandboxInfo:
        """Get sandbox status information."""

    # --- Execution ---

    @abc.abstractmethod
    async def exec(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]] = None,
        max_output_bytes: Optional[int] = None,
    ) -> ExecResult:
        """Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute.
            args: Command arguments.
            env: Additional environment variables (merged with existing).
            max_output_bytes: When set, cap each of stdout/stderr at this many
                bytes, reading no more once the cap is hit (so a flood of output
                can't grow the host client unbounded). Sets ``truncated`` on the
                result. ``None`` (default) keeps the unbounded one-shot read.

        Returns:
            ExecResult: Execution result with exit code, stdout, and stderr.
        """

    @abc.abstractmethod
    async def run_code(
        self,
        code: str,
        code_type: CodeType = "python",
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute code in the sandbox.

        Args:
            code: Code string to execute.
            code_type: Code type.
            env: Additional environment variables (merged with existing).

        Returns:
            ExecResult: Execution result with exit code, stdout, and stderr.
        """

    # --- File Operations ---

    @abc.abstractmethod
    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        """Upload a local file to the sandbox.

        Args:
            local_path: Local file path.
            remote_path: Target path in sandbox (including filename).
            overwrite: Whether to overwrite if target exists. Default False.

        Raises:
            FileNotFoundError: Local file not found.
            FileExistsError: Target exists and overwrite=False.
        """

    @abc.abstractmethod
    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
        """Download a file from the sandbox.

        Args:
            remote_path: Source path in sandbox.
            local_path: Local target path (including filename).
            overwrite: Whether to overwrite if local file exists. Default False.

        Raises:
            FileNotFoundError: Source file not found in sandbox.
            FileExistsError: Local file exists and overwrite=False.
        """

    @abc.abstractmethod
    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        """Write string content directly to a sandbox file.

        Args:
            content: Text content to write.
            remote_path: Target path in sandbox (including filename).
            overwrite: Whether to overwrite if target exists. Default False.

        Raises:
            FileExistsError: Target exists and overwrite=False.
        """

    @abc.abstractmethod
    async def read_file(self, remote_path: str) -> str:
        """Read file content from the sandbox.

        Args:
            remote_path: File path in sandbox.

        Raises:
            FileNotFoundError: File not found in sandbox.
        """


class SandboxService(abc.ABC):
    """
    Abstract interface for sandbox lifecycle management.

    Typical usage:

        service = BoxliteService()

        # Get or create sandbox
        async with await service.get_or_create("my-box") as sandbox:
            result = await sandbox.exec("python train.py")
            print(sandbox.name)  # "my-box"

        # List all sandboxes
        boxes = await service.list_sandboxes()
        print(boxes)

        # Delete sandbox
        await service.delete("my-box")

        # Create snapshot
        await service.create_snapshot("my-box", "my-box-v1.0")

        # Create from snapshot
        await service.get_or_create("my-box", template=SandboxTemplate(_type="snapshot", snapshot_id="my-box-v1.0"))
    """

    @abc.abstractmethod
    async def get_or_create(
        self,
        name: str,
        template: Optional[SandboxTemplate] = None,
        config: Optional[SandboxConfig] = None,
    ) -> Sandbox:
        """Get or create a sandbox, handling resume automatically.

        Behavior:
        - Exists and running → return directly
        - Exists and stopped → resume and return
        - Does not exist → create and return

        Args:
            name: Sandbox name (unique identifier).
            template: Template for creation only. Ignored for existing sandboxes.
            config: Configuration for creation only. Ignored for existing sandboxes.

        Returns:
            Sandbox: Operational sandbox instance.

        Note:
            Managed lifecycle names are owned by the explicit lifecycle API
            (inspect/create/start_existing/stop_existing); get_or_create
            does not validate an existing sandbox's configuration.
        """

    @abc.abstractmethod
    async def list_sandboxes(self) -> list[SandboxInfo]:
        """List all sandboxes (both running and stopped).

        Returns:
            list[SandboxInfo]: List of sandbox status information.
        """

    @abc.abstractmethod
    async def delete(self, name: str) -> None:
        """Permanently delete a sandbox and release all resources.

        Args:
            name: Sandbox name to delete.
        """

    @abc.abstractmethod
    async def supports_snapshots(self) -> bool:
        """Check if this sandbox service supports snapshot operations.

        Returns:
            bool: True if snapshots are supported, False otherwise.
        """

    @abc.abstractmethod
    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        """Create a sandbox snapshot.

        Args:
            name: Sandbox name.
            snapshot_id: Unique snapshot identifier.
        """

    @abc.abstractmethod
    async def list_snapshots(self) -> list[SandboxSnapshot]:
        """List all sandbox snapshots.

        Returns:
            list[SandboxSnapshot]: List of snapshot information.
        """

    @abc.abstractmethod
    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Permanently delete a sandbox snapshot.

        Args:
            snapshot_id: Unique snapshot identifier.
        """

    # --- Spec-based reconciliation lifecycle (optional per backend) ---
    #
    # These four methods are deliberately concrete, not abstract: this class
    # is a public, externally-consumed interface, and making them abstract
    # would be a breaking change for any existing subclass with no
    # transition period. Backends that do not implement spec-based
    # reconciliation simply inherit the default bodies below, which raise
    # SandboxReconcileUnsupportedError. Readiness checks must call
    # supports_runtime_spec() rather than probing with hasattr or a
    # try/except around a call.

    async def supports_runtime_spec(self) -> bool:
        """Whether this backend implements the reconciliation lifecycle.

        Mirrors ``supports_snapshots()``: callers gate on this probe rather
        than using hasattr/try-call detection. Defaults to False; backends
        that implement inspect/create/start_existing/stop_existing override
        this to return True.
        """
        return False

    async def inspect(self, name: str) -> Optional[SandboxInspection]:
        """Observe a sandbox's current state and runtime facts.

        Returns None if no sandbox with this name exists. This call has no
        side effects and must not acquire any lock or control handle: each
        call re-reads backend state directly rather than reusing a cached
        handle. Any lock this method takes internally only guarantees the
        atomicity of that single call; a caller that needs to act on the
        result must hold its own critical section spanning the
        inspect-then-act sequence.

        A returned inspection with ``state`` reflecting a container that
        exists but has never been started must not, by itself, be treated as
        grounds for destroying or reusing the sandbox.

        When multiple containers carry the same sandbox name label (only
        reachable via out-of-band operations against the backend, not
        through this class's own API), which one this returns is undefined;
        callers must not depend on it.

        Raises:
            SandboxReconcileUnsupportedError: This backend does not
                implement spec-based reconciliation.
        """
        raise SandboxReconcileUnsupportedError(
            f"{type(self).__name__} does not support inspect()"
        )

    async def create(
        self, name: str, template: SandboxTemplate, config: SandboxConfig
    ) -> Sandbox:
        """Create a new sandbox under the explicit, verified lifecycle contract.

        Conceptually an eight-step sequence: (1) reject if a container with
        this name already exists, (2) resolve a snapshot template to its
        backing image, (3) validate the desired volumes and ports for
        host-path/guest-port conflicts, (4) create the container — from the
        same normalized desired spec this validation ran against — with a
        fingerprint label and a contract-version label written at creation
        time, (5) start it, compensating with a forced container removal if
        start fails, (6) verify the started container's observed facts
        against the desired spec before publishing, removing the container
        and raising if verification fails, (7) persist the store record only
        after verification passes, recording the same canonical desired state
        the container was built from and the fingerprint label attests rather
        than the caller's raw input, so a reader never has to re-normalize the
        row before comparing it, and (8) return a live Sandbox handle.

        Not idempotent: an existing container in any state (created,
        running, or stopped) raises SandboxAlreadyExistsError rather than
        being adopted. Callers that want get-or-create semantics must call
        inspect() first under their own critical section.

        Raises:
            SandboxAlreadyExistsError: A container with this name exists.
            SandboxRuntimeConflictError: The desired volumes conflict at the
                same host path with a different guest path or mode, the
                desired ports conflict at the same guest port with a
                different host port, or the container's observed facts
                disagreed with the desired spec at publish-before-verify
                (step 6).
            SandboxReconcileUnsupportedError: This backend does not
                implement spec-based reconciliation.
        """
        raise SandboxReconcileUnsupportedError(
            f"{type(self).__name__} does not support create()"
        )

    async def start_existing(self, name: str) -> Sandbox:
        """Start a previously-created sandbox.

        Idempotent: starting an already-running sandbox returns it
        unchanged.

        Raises:
            SandboxNotFoundError: No sandbox with this name exists.
            SandboxReconcileUnsupportedError: This backend does not
                implement spec-based reconciliation.
        """
        raise SandboxReconcileUnsupportedError(
            f"{type(self).__name__} does not support start_existing()"
        )

    async def stop_existing(self, name: str, *, timeout: Optional[int] = None) -> None:
        """Stop an existing sandbox, preserving its state.

        Idempotent: stopping an already-stopped sandbox is a no-op.

        Args:
            timeout: Seconds to wait for a graceful stop before a forced
                kill (backend-native bound, e.g. docker-py's own
                ``container.stop(timeout=...)``). ``None`` uses the
                backend's own default.

        Raises:
            SandboxNotFoundError: No sandbox with this name exists.
            SandboxReconcileUnsupportedError: This backend does not
                implement spec-based reconciliation.
        """
        raise SandboxReconcileUnsupportedError(
            f"{type(self).__name__} does not support stop_existing()"
        )

    async def get_store_record(self, name: str) -> Optional[SandboxInfo]:
        """Return the backend's own persistent store record for this name.

        Unlike ``list_sandboxes()``'s merged view (store-row-augmented when
        a row exists, reconstructed-from-live-facts otherwise —
        indistinguishable from the caller's side), this exposes the store
        row itself so reconciliation can tell "no row" apart from "row
        happens to equal live facts", and can rebuild the previously
        desired spec from what ``create()`` (or the legacy
        ``get_or_create()``) actually persisted rather than from live
        inspection facts alone — env in particular cannot be reliably
        reconstructed from live facts (see ``ObservedRuntimeFacts``).

        Returns None if the backend has no row for this name, including
        backends that keep no persistent store at all.

        Defaults to None; backends that implement the reconciliation
        lifecycle override this alongside inspect/create/start_existing/
        stop_existing.
        """
        return None

    async def persist_store_record(self, name: str, info: SandboxInfo) -> None:
        """Write (or overwrite) the backend's persistent store row for this name.

        Used only by reconciliation to backfill a store row for a
        container whose live facts and fingerprint label already verify a
        MATCH against the desired spec but whose store write did not land
        (``create()``'s own store write is best-effort after publish
        verification passes — see its docstring, step 7). Not used by any
        other lifecycle method; a normal ``create()``/``get_or_create()``
        persists its own row itself.

        Defaults to a no-op; backends that implement the reconciliation
        lifecycle override this alongside inspect/create/start_existing/
        stop_existing.
        """
        return None
