"""
Docker sandbox implementation.
"""

from __future__ import annotations

import abc
import asyncio
import io
import logging
import os
import posixpath
import re
import shutil
import tarfile
import tempfile
import textwrap
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from hashlib import sha1
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncContextManager,
    AsyncIterator,
    Coroutine,
    Optional,
    cast,
)

from docker.errors import APIError, ImageNotFound, NotFound

import docker

from ..config import (
    get_sandbox_image,
    get_sandbox_namespace,
    validate_sandbox_namespace,
)
from .base import (
    SPEC_CONTRACT_VERSION,
    CodeType,
    ExecResult,
    ObservedRuntimeFacts,
    ResolvedSandboxRuntimeSpec,
    Sandbox,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxInfo,
    SandboxInspection,
    SandboxNotFoundError,
    SandboxRuntimeConflictError,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
    canonical_sandbox_path,
)
from .keyed_lock import KeyedLockRegistry

if TYPE_CHECKING:
    from docker.models.containers import Container

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = get_sandbox_image()

LABEL_MANAGED = "xagent.managed"
LABEL_SANDBOX_NAME = "xagent.sandbox.name"
LABEL_NAMESPACE = "xagent.sandbox.namespace"
LABEL_TEMPLATE_TYPE = "xagent.sandbox.template.type"
LABEL_SNAPSHOT_ID = "xagent.sandbox.snapshot_id"
# Written only by create() (the new explicit lifecycle API), never by the
# legacy get_or_create() path: their presence is the attestation that
# spec_matches_inspection() keys off of. Immutable once written.
LABEL_SPEC_FINGERPRINT = "xagent.sandbox.spec.fingerprint"
LABEL_SPEC_VERSION = "xagent.sandbox.spec.version"
# Ownership-scheme version. v2 resources carry ``xagent.managed=v2`` plus an
# exact ``xagent.sandbox.namespace`` value; legacy resources used
# ``xagent.managed=true`` with no namespace. The value change (not just a
# new key) is what keeps a pre-namespace backend -- which lists
# ``xagent.managed=true`` globally -- from ever adopting v2 containers
# during a mixed-version rollout.
MANAGED_LABEL_VALUE = "v2"
CONTAINER_NAME_PREFIX = "xagent_sandbox_"
SNAPSHOT_REPOSITORY = "xagent-sandbox-snapshot"
_CPU_NANOS = 1_000_000_000


def _ownership_label_filters(namespace: str) -> list[str]:
    """Return the load-bearing ownership filters for one deployment."""
    return [
        f"{LABEL_MANAGED}={MANAGED_LABEL_VALUE}",
        f"{LABEL_NAMESPACE}={namespace}",
    ]


class DockerStore(abc.ABC):
    """Store for persisting Docker sandbox metadata."""

    @abc.abstractmethod
    def get_info(self, name: str) -> Optional[SandboxInfo]:
        """Get sandbox info."""

    @abc.abstractmethod
    def add_info(self, name: str, info: SandboxInfo) -> None:
        """Add sandbox info."""

    @abc.abstractmethod
    def update_info_state(self, name: str, state: str) -> None:
        """Update sandbox state."""

    @abc.abstractmethod
    def delete_info(self, name: str) -> None:
        """Delete sandbox info."""

    @abc.abstractmethod
    def get_snapshot(self, snapshot_id: str) -> Optional[SandboxSnapshot]:
        """Get snapshot info."""

    @abc.abstractmethod
    def add_snapshot(self, snapshot: SandboxSnapshot) -> None:
        """Add snapshot info."""

    @abc.abstractmethod
    def list_snapshots(self) -> list[SandboxSnapshot]:
        """List snapshot info."""

    @abc.abstractmethod
    def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete snapshot info."""


class MemDockerStore(DockerStore):
    """In-memory implementation of DockerStore."""

    def __init__(self) -> None:
        self._metadata: dict[str, SandboxInfo] = {}
        self._snapshots: dict[str, SandboxSnapshot] = {}

    def get_info(self, name: str) -> Optional[SandboxInfo]:
        return self._metadata.get(name)

    def add_info(self, name: str, info: SandboxInfo) -> None:
        self._metadata[name] = info

    def update_info_state(self, name: str, state: str) -> None:
        if name in self._metadata:
            self._metadata[name].state = state

    def delete_info(self, name: str) -> None:
        self._metadata.pop(name, None)

    def get_snapshot(self, snapshot_id: str) -> Optional[SandboxSnapshot]:
        return self._snapshots.get(snapshot_id)

    def add_snapshot(self, snapshot: SandboxSnapshot) -> None:
        self._snapshots[snapshot.snapshot_id] = snapshot

    def list_snapshots(self) -> list[SandboxSnapshot]:
        return list(self._snapshots.values())

    def delete_snapshot(self, snapshot_id: str) -> None:
        self._snapshots.pop(snapshot_id, None)


def _create_docker_client() -> Any:
    """Create a Docker SDK client using the standard Docker environment config.

    The Docker SDK can also talk to Docker-compatible runtimes such as Podman
    when ``DOCKER_HOST`` points at a compatible socket/service.
    """
    return cast(Any, docker.from_env())


def is_docker_available() -> bool:
    """Return whether Docker is reachable."""
    try:
        client = _create_docker_client()
        client.ping()
    except Exception as e:
        logger.exception(
            "No Docker-compatible runtime API is reachable. "
            "For Podman or other non-default runtimes, start the service/socket and "
            "set DOCKER_HOST to the compatible endpoint. error=%s",
            e,
        )
        return False
    return True


def _make_safe_name(name: str) -> str:
    """Convert an arbitrary sandbox identifier into a Docker-safe name."""
    # Convert to safe name
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-.") or "sandbox"
    # Add a sha1 suffix to prevent duplicate names
    digest = sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"{base.lower()}-{digest}"


def _container_name(name: str, namespace: str) -> str:
    """Build the managed Docker container name for a sandbox.

    The namespace is part of the hashed identity, so two deployments that
    use the same logical sandbox name get distinct physical container names
    on a shared Docker daemon."""
    return f"{CONTAINER_NAME_PREFIX}{_make_safe_name(f'{namespace}::{name}')}"


#: Max length of the complete namespace token embedded in the snapshot repository
#: name, including the separator and 10-character digest. The full Docker
#: reference (repository:tag) must stay under 255 characters: the fixed prefix
#: (24 chars) plus this token plus the tag (<= 128 chars) must fit.
_SNAPSHOT_NAMESPACE_TOKEN_MAX = 100
_SNAPSHOT_NAMESPACE_DIGEST_LENGTH = 10
_SNAPSHOT_NAMESPACE_PREFIX_MAX = (
    _SNAPSHOT_NAMESPACE_TOKEN_MAX - _SNAPSHOT_NAMESPACE_DIGEST_LENGTH - 1
)


def _snapshot_repository(namespace: str) -> str:
    """Map a deployment namespace to a Docker-repository-safe repository name.

    The Compose project-name grammar enforced by ``validate_sandbox_namespace``
    is not the Docker reference grammar: namespaces may end in ``-``/``_``,
    mix separators (``a_-b``), and be arbitrarily long, all of which produce
    invalid or overlong repository names. Separator runs are collapsed to a
    single ``-`` and the readable prefix is length-capped. Every mapping
    appends a SHA-1 digest of the raw namespace, including already-safe names,
    so no valid namespace can impersonate another namespace's encoded token.
    """
    prefix = re.sub(r"[-_]+", "-", namespace).strip("-")
    prefix = prefix[:_SNAPSHOT_NAMESPACE_PREFIX_MAX].strip("-")
    digest = sha1(namespace.encode("utf-8")).hexdigest()[
        :_SNAPSHOT_NAMESPACE_DIGEST_LENGTH
    ]
    return f"{SNAPSHOT_REPOSITORY}-{prefix}-{digest}"


def _snapshot_tag(snapshot_id: str, namespace: str) -> str:
    """Build the managed Docker image tag for a snapshot.

    The namespace is part of the repository name so two deployments that
    use the same snapshot id get distinct images on a shared Docker daemon:
    a sibling deployment can neither retag, consume, nor delete this
    deployment's snapshot image. The namespace is mapped through
    ``_snapshot_repository`` because the Compose grammar is not a valid
    Docker repository grammar."""
    safe = _make_safe_name(snapshot_id)
    return f"{_snapshot_repository(namespace)}:{safe}"


def _get_state(status: str | None) -> str:
    """Map Docker container status to the sandbox state model."""
    if not status:
        return "unknown"
    lowered = status.lower()
    if lowered == "running":
        return "running"
    if lowered in {"created", "exited", "paused", "dead", "restarting"}:
        return "stopped"
    return "unknown"


def _parse_container_config(container: Container) -> SandboxInfo:
    """Reconstruct SandboxInfo from Docker inspect data."""
    attrs = cast(dict[str, Any], container.attrs)
    config_data = cast(dict[str, Any], attrs.get("Config") or {})
    host_config = cast(dict[str, Any], attrs.get("HostConfig") or {})
    state = cast(dict[str, Any], attrs.get("State") or {})

    env_map: dict[str, str] = {}
    # Docker stores env vars as ["KEY=value", ...]
    for item in cast(list[str], config_data.get("Env") or []):
        if "=" in item:
            key, value = item.split("=", 1)
            env_map[key] = value

    volumes: list[tuple[str, str, str]] = []
    # Only bind mounts
    for mount in cast(list[dict[str, Any]], attrs.get("Mounts") or []):
        if mount.get("Type") != "bind":
            continue
        source = str(mount.get("Source") or "")
        target = str(mount.get("Destination") or "")
        mode = "ro" if bool(mount.get("RW")) is False else "rw"
        if source and target:
            volumes.append((source, target, mode))

    ports: list[tuple[int, int]] = []
    port_bindings = cast(
        dict[str, list[dict[str, str]]], host_config.get("PortBindings") or {}
    )
    for guest_port, host_bindings in port_bindings.items():
        container_port = int(str(guest_port).split("/", 1)[0])
        for binding in host_bindings or []:
            host_port = binding.get("HostPort")
            if host_port:
                ports.append((int(host_port), container_port))

    nano_cpus = int(host_config.get("NanoCpus") or 0)
    cpus = nano_cpus // _CPU_NANOS if nano_cpus else 1
    memory_bytes = int(host_config.get("Memory") or 0)
    memory = memory_bytes // (1024 * 1024) if memory_bytes else 512

    labels = container.labels
    template_type = labels.get(LABEL_TEMPLATE_TYPE, "image")
    if template_type == "snapshot" and labels.get(LABEL_SNAPSHOT_ID):
        template = SandboxTemplate(
            type="snapshot", snapshot_id=labels[LABEL_SNAPSHOT_ID]
        )
    else:
        template = SandboxTemplate(
            type="image", image=str(config_data.get("Image") or "")
        )
    config = SandboxConfig(
        working_dir=str(config_data.get("WorkingDir") or "/home"),
        cpus=max(1, cpus),
        memory=max(128, memory),
        env=env_map or None,
        volumes=volumes or None,
        network_isolated=bool(
            attrs.get("NetworkSettings", {}).get("Networks") == {}
            or host_config.get("NetworkMode") == "none"
        ),
        ports=ports or None,
    )
    return SandboxInfo(
        name=str(labels.get(LABEL_SANDBOX_NAME, container.name)),
        state=_get_state(str(state.get("Status"))),
        template=template,
        config=config,
        created_at=str(attrs.get("Created") or ""),
    )


def _merge_info(
    runtime_info: SandboxInfo, stored_info: Optional[SandboxInfo]
) -> SandboxInfo:
    """Merge runtime info and stored info."""
    if stored_info is None:
        return runtime_info
    return SandboxInfo(
        name=stored_info.name,
        state=runtime_info.state,
        template=stored_info.template,
        config=stored_info.config,
        created_at=runtime_info.created_at,
    )


def _build_inspection(container: Container) -> SandboxInspection:
    """Build a point-in-time SandboxInspection directly from Docker inspect data.

    Unlike ``_parse_container_config``, this keeps raw backend units
    (``HostConfig.NanoCpus`` / ``HostConfig.Memory``) rather than the
    divided-and-clamped ``SandboxConfig`` values, so a live edit such as
    ``docker update --cpus 0.5`` remains observable in the returned facts.
    Shared by ``inspect()`` (no side effects, no lock held) and ``create()``'s
    publish-before-verify step; the caller is responsible for reloading the
    container beforehand so ``container.attrs`` reflects current state.
    """
    attrs = cast(dict[str, Any], container.attrs)
    config_data = cast(dict[str, Any], attrs.get("Config") or {})
    host_config = cast(dict[str, Any], attrs.get("HostConfig") or {})
    state = cast(dict[str, Any], attrs.get("State") or {})

    env_map: dict[str, str] = {}
    for item in cast(list[str], config_data.get("Env") or []):
        if "=" in item:
            key, value = item.split("=", 1)
            env_map[key] = value

    volumes: list[tuple[str, str, str]] = []
    for mount in cast(list[dict[str, Any]], attrs.get("Mounts") or []):
        if mount.get("Type") != "bind":
            continue
        source = str(mount.get("Source") or "")
        target = str(mount.get("Destination") or "")
        mode = "ro" if bool(mount.get("RW")) is False else "rw"
        if source and target:
            volumes.append((source, target, mode))

    ports: list[tuple[int, int]] = []
    port_bindings = cast(
        dict[str, list[dict[str, str]]], host_config.get("PortBindings") or {}
    )
    for guest_port, host_bindings in port_bindings.items():
        container_port = int(str(guest_port).split("/", 1)[0])
        for binding in host_bindings or []:
            host_port = binding.get("HostPort")
            if host_port:
                ports.append((int(host_port), container_port))

    labels = dict(container.labels)
    raw_status = str(state.get("Status") or "")
    network_settings = cast(dict[str, Any], attrs.get("NetworkSettings") or {})
    runtime_networks = tuple(
        cast(dict[str, Any], network_settings.get("Networks") or {})
    )

    facts = ObservedRuntimeFacts(
        raw_status=raw_status,
        image_ref=cast(Optional[str], config_data.get("Image")),
        image_digest=cast(Optional[str], attrs.get("Image")),
        raw_nano_cpus=cast(Optional[int], host_config.get("NanoCpus")),
        raw_memory_bytes=cast(Optional[int], host_config.get("Memory")),
        env=env_map,
        volumes=tuple(volumes),
        ports=tuple(ports),
        network_isolated=bool(config_data.get("NetworkDisabled")),
        runtime_networks=runtime_networks,
        labels=labels,
        created_at=cast(Optional[str], attrs.get("Created")),
        working_dir=cast(Optional[str], config_data.get("WorkingDir")),
    )
    return SandboxInspection(
        state="running" if _get_state(raw_status) == "running" else "stopped",
        facts=facts,
        fingerprint_label=_attestation_label(labels.get(LABEL_SPEC_FINGERPRINT)),
        version_label=_attestation_label(labels.get(LABEL_SPEC_VERSION)),
    )


def _attestation_label(value: Optional[str]) -> Optional[str]:
    """Read one spec-attestation label, mapping a blank value to ``None``.

    ``None`` is the contract's "no attestation" value, which
    ``spec_matches_inspection`` answers with ``UNVERIFIED``. A blank string
    means the same thing and must not be treated as a fingerprint that simply
    fails to match, which would report ``MISMATCH`` and make a reconciler
    rebuild a container it should have adopted.

    Blank is a value this code writes on purpose: ``_create_container``
    stamps both keys blank when it has no attestation to make, so that a
    container never silently presents a fingerprint inherited from its base
    image. Docker has no way to remove an inherited label, only to overwrite
    it, so blank is the only available "absent" form on the wire.
    """
    if value is None:
        return None
    return value or None


def _check_no_conflicting_volumes(
    volumes: Optional[list[tuple[str, str, str]]],
) -> None:
    """Reject desired volumes that collide on either side of the mount.

    Two failure modes share this check, the same way
    ``_check_no_conflicting_ports`` covers both sides of a port mapping:

    - Host-side: ``_create_container`` builds its Docker ``volumes`` dict
      keyed by host path, so two entries with the same host path but a
      different guest path or mode would silently drop one of them — a dict
      collapsing entries with no error.
    - Guest-side: two entries with different host sources on the same guest
      path pass straight through to Docker, which rejects the pair at
      container *create* time with a raw ``APIError``/400
      ``Duplicate mount point``.

    Both become the same pre-create, typed ``SandboxRuntimeConflictError``
    instead of surfacing later as a silent drop or a raw daemon error.
    Exactly identical triples (duplicates) are accepted and simply
    collapse; this only rejects a real disagreement, canonicalizing paths
    first (via ``canonical_sandbox_path``, the same owner the desired spec
    uses) so equivalent-but-differently-spelled paths are treated as the
    same key on both sides.
    """
    if not volumes:
        return
    seen_by_host: dict[str, tuple[str, str]] = {}
    seen_by_guest: dict[str, tuple[str, str]] = {}
    for host_path, guest_path, mode in volumes:
        normalized_host = canonical_sandbox_path(host_path)
        normalized_guest = canonical_sandbox_path(guest_path)

        host_key = (normalized_guest, mode)
        prior_for_host = seen_by_host.get(normalized_host)
        if prior_for_host is not None and prior_for_host != host_key:
            raise SandboxRuntimeConflictError(
                f"Conflicting desired volume mounts for host path "
                f"{normalized_host!r}: {prior_for_host} vs {host_key}"
            )
        seen_by_host[normalized_host] = host_key

        guest_key = (normalized_host, mode)
        prior_for_guest = seen_by_guest.get(normalized_guest)
        if prior_for_guest is not None and prior_for_guest != guest_key:
            raise SandboxRuntimeConflictError(
                f"Conflicting desired volume mounts for guest path "
                f"{normalized_guest!r}: {prior_for_guest} vs {guest_key}"
            )
        seen_by_guest[normalized_guest] = guest_key


def _check_no_conflicting_ports(
    ports: Optional[list[tuple[int, int]]],
) -> None:
    """Reject desired ports that collide on either side of the mapping.

    Two failure modes share this check:

    - Guest-side: ``_create_container`` builds its Docker ``ports`` dict
      keyed by guest port, so two entries with the same guest port but a
      different host port would silently drop one of them — a dict
      collapsing entries with no error.
    - Host-side: two entries with different guest ports but the same
      nonzero host port pass straight through to Docker, which only rejects
      the overlap when the container *starts* (a raw ``APIError``/500), not
      at create time.

    Both are turned into the same pre-create, typed
    ``SandboxRuntimeConflictError`` here instead of surfacing later as a
    silent drop or a raw daemon error. Exactly identical pairs (duplicates)
    are accepted and simply collapse; a host port of ``0`` (meaning "let
    Docker assign an ephemeral port") is excluded from the host-side check
    since many guest ports may legitimately share it.
    """
    if not ports:
        return
    seen_by_guest: dict[int, int] = {}
    seen_by_host: dict[int, int] = {}
    for host_port, guest_port in ports:
        prior_host = seen_by_guest.get(guest_port)
        if prior_host is not None and prior_host != host_port:
            raise SandboxRuntimeConflictError(
                f"Conflicting desired port mappings for guest port "
                f"{guest_port}: host {prior_host} vs {host_port}"
            )
        seen_by_guest[guest_port] = host_port

        if host_port != 0:
            prior_guest = seen_by_host.get(host_port)
            if prior_guest is not None and prior_guest != guest_port:
                raise SandboxRuntimeConflictError(
                    f"Conflicting desired port mappings for host port "
                    f"{host_port}: guest {prior_guest} vs {guest_port}"
                )
            seen_by_host[host_port] = guest_port


def _find_publish_mismatches(
    desired: ResolvedSandboxRuntimeSpec,
    resolved_image: str,
    inspection: SandboxInspection,
) -> list[str]:
    """Return the field names whose observed value disagrees with ``desired``.

    Used only by create()'s publish-before-verify step. Returns field names
    only (never the actual values) since this feeds directly into a raised
    error message and desired/observed values may carry sensitive paths.

    The image check applies identically to both template types: for a
    snapshot-based create, ``resolved_image`` is the snapshot's own image
    tag, and ``facts.image_ref`` (Docker's ``Config.Image``) equals that tag
    once the container has actually been created from it, so there is no
    need for a separate label-based check on the snapshot leg.

    cpus/memory are re-checked here (immediately after start, in raw backend
    units) in addition to the live re-check ``spec_matches_inspection`` does
    later: this is the first opportunity to catch e.g. Docker silently
    clamping an out-of-range request, before the container is ever
    published.

    Failure policy: any single mismatch fails the whole publish and the
    container is destroyed. This is deliberate and cannot be softened into a
    per-field "hard-fail on security-relevant fields, log-and-degrade on
    cosmetic ones" policy, because the fingerprint attestation is
    all-or-nothing and immutable. ``create()`` stamps
    ``LABEL_SPEC_FINGERPRINT`` over the *entire* desired spec before the
    container starts -- it has to, since Docker offers no way to add or
    change a label on an existing container -- and ``spec_matches_inspection``
    trusts that one label for every field except cpus/memory, which it
    re-reads live. Publishing a container whose ``working_dir`` was quietly
    accepted as different would therefore leave a label positively attesting
    a spec the container does not implement, and every later reconcile would
    read that as MATCH. A lying attestation is a worse failure than a refused
    create, so the tiering has to live upstream of the label: fields whose
    exact value cannot be guaranteed must be constrained where the desired
    spec is built, not waived after the fact.

    Two such constraints already exist upstream, which is what keeps this
    exact-equality check from being brittle in practice: paths pass through
    ``canonical_sandbox_path`` so a desired path is spelled the way the
    backend echoes it, and ``get_sandbox_volumes()`` clamps every volume mode
    to exactly ``ro``/``rw``, so a mode string the daemon would rewrite (an
    SELinux ``rw,z``, say) cannot reach this comparison from configuration.
    """
    mismatches: list[str] = []
    facts = inspection.facts

    if facts.image_ref != resolved_image:
        mismatches.append("image")
    if set(facts.volumes) != set(desired.volumes):
        mismatches.append("volumes")
    if set(facts.ports) != set(desired.ports):
        mismatches.append("ports")
    if facts.working_dir != desired.working_dir:
        mismatches.append("working_dir")
    if facts.network_isolated != desired.network_isolated:
        mismatches.append("network_isolated")
    if (facts.raw_nano_cpus or 0) != int(desired.cpus * _CPU_NANOS):
        mismatches.append("cpus")
    if (facts.raw_memory_bytes or 0) != int(desired.memory * 1024 * 1024):
        mismatches.append("memory")
    return mismatches


def _write_tar_from_local_path(
    local_path: str, arcname: str, file_obj: io.BufferedRandom
) -> None:
    """Pack a local file into a tar stream for Docker put_archive."""
    with tarfile.open(fileobj=file_obj, mode="w") as tar:
        tar.add(local_path, arcname=arcname)
    file_obj.seek(0)


def _write_tar_from_content(
    content: str, arcname: str, file_obj: io.BufferedRandom
) -> None:
    """Pack in-memory text content into a tar stream for Docker put_archive."""
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    with tarfile.open(fileobj=file_obj, mode="w") as tar:
        tar.addfile(info, io.BytesIO(data))
    file_obj.seek(0)


def _exec_capped(
    container: Any, cmd: list[str], env: Optional[dict[str, str]], cap: int
) -> ExecResult:
    """Exec ``cmd`` streaming stdout/stderr and keeping at most ``cap`` bytes of
    each, so the host docker client never buffers an unbounded remote flood
    (#3). ``container.exec_run(demux=True)`` reads the whole stream into host
    memory before returning; the low-level exec API streams instead, letting us
    drop everything past the cap while still draining to EOF (the caller's
    ``timeout`` bounds how long that can take). Blocking — run in a thread."""
    api = container.client.api
    exec_id = api.exec_create(
        container.id, cmd, environment=env, stdout=True, stderr=True
    )["Id"]
    out = bytearray()
    err = bytearray()
    out_truncated = False
    err_truncated = False
    for stdout_chunk, stderr_chunk in api.exec_start(exec_id, stream=True, demux=True):
        if stdout_chunk:
            room = cap - len(out)
            if room > 0:
                out.extend(stdout_chunk[:room])
            if len(stdout_chunk) > max(room, 0):
                out_truncated = True
        if stderr_chunk:
            room = cap - len(err)
            if room > 0:
                err.extend(stderr_chunk[:room])
            if len(stderr_chunk) > max(room, 0):
                err_truncated = True
    exit_code = api.exec_inspect(exec_id).get("ExitCode")
    return ExecResult(
        exit_code=exit_code if exit_code is not None else -1,
        stdout=bytes(out).decode("utf-8", errors="replace"),
        stderr=bytes(err).decode("utf-8", errors="replace"),
        truncated=out_truncated or err_truncated,
        error_message=None,
    )


def _write_stream_to_file(
    stream: Any, file_obj: io.BufferedRandom | io.BufferedWriter
) -> None:
    """Copy a streamed Docker archive into a local file object."""
    for chunk in stream:
        file_obj.write(chunk)
    file_obj.flush()
    file_obj.seek(0)


def _extract_single_file_from_tar(
    tar_file_obj: io.BufferedRandom | io.BufferedReader,
    output_file_obj: io.BufferedWriter | io.BytesIO,
) -> None:
    """Extract the first regular file from a Docker get_archive tar stream."""
    with tarfile.open(fileobj=tar_file_obj, mode="r:*") as tar:
        member = next((item for item in tar if item.isfile()), None)
        if member is None:
            raise FileNotFoundError("No file found in archive")
        fileobj = tar.extractfile(member)
        if fileobj is None:
            raise FileNotFoundError(f"Could not read file from archive: {member.name}")
        shutil.copyfileobj(fileobj, output_file_obj)
        output_file_obj.flush()


def _archive_path_exists(container: Container, remote_path: str) -> bool:
    """Check file existence."""
    try:
        container.get_archive(remote_path)
        return True
    except NotFound:
        return False


@dataclass
class _SandboxControl:
    """Shared concurrency guard for operations targeting the same sandbox."""

    name: str
    active_ops: int = 0
    new_operations_paused: bool = False
    deleted: bool = False
    file_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    exec_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def acquire_operation(self) -> None:
        """Register a new sandbox operation, blocking while new operations are paused."""
        async with self.cond:
            while self.new_operations_paused and not self.deleted:
                await self.cond.wait()
            if self.deleted:
                raise RuntimeError(f"Sandbox {self.name!r} has been deleted")
            self.active_ops += 1

    async def release_operation(self) -> None:
        """Mark a sandbox operation as finished."""
        async with self.cond:
            self.active_ops -= 1
            if self.active_ops == 0:
                self.cond.notify_all()

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        """Track a sandbox operation and always release it on cancellation."""
        await self.acquire_operation()
        try:
            yield
        finally:
            await asyncio.shield(self.release_operation())

    async def pause_new_operations(self, mark_deleted: bool) -> None:
        """Block new operations and wait for in-flight work to finish."""
        async with self.cond:
            self.new_operations_paused = True
            while self.active_ops > 0:
                await self.cond.wait()
            if mark_deleted:
                self.deleted = True

    async def resume_new_operations(self) -> None:
        """Allow new operations again after a non-destructive pause."""
        async with self.cond:
            if not self.deleted:
                self.new_operations_paused = False
            self.cond.notify_all()

    @asynccontextmanager
    async def exclusive_access(self, *, mark_deleted: bool) -> AsyncIterator[None]:
        """Block new operations and wait for exclusive lifecycle access."""
        await self.pause_new_operations(mark_deleted=mark_deleted)
        try:
            yield
        finally:
            await asyncio.shield(self.resume_new_operations())


class DockerSandbox(Sandbox):
    """Runtime sandbox implementation backed by a managed Docker container."""

    def __init__(
        self,
        sandbox_name: str,
        container: Container,
        info: SandboxInfo,
        store: DockerStore,
        control: _SandboxControl,
        locks: KeyedLockRegistry,
    ) -> None:
        """Bind a handle to one container plus the registries that guard it.

        ``locks`` is the owning service's per-name lifecycle registry, passed
        in so ``stop()`` can take the *same* mutex the service's own lifecycle
        methods take (see ``stop()``). It is the registry object itself rather
        than the service, so this handle keeps no reverse dependency on
        ``DockerSandboxService``.
        """
        self._container = container
        self._name = sandbox_name
        self._info = info
        self._store = store
        self._control = control
        self._locks = locks

    @property
    def name(self) -> str:
        """Sandbox name (unique identifier)."""
        return self._name

    async def _require_container(self) -> Container:
        """Return the managed container or raise if it has been deleted."""
        try:
            await asyncio.to_thread(self._container.reload)
        except NotFound as e:
            raise SandboxNotFoundError(
                f"Sandbox container not found: {self._name}"
            ) from e
        return self._container

    async def _exec_in_container(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]] = None,
        max_output_bytes: Optional[int] = None,
    ) -> ExecResult:
        """Execute a command directly against the current container instance."""
        container = await self._require_container()
        cmd: list[str] = [command, *args]
        try:
            if max_output_bytes is not None:
                return await asyncio.to_thread(
                    _exec_capped, container, cmd, env, max_output_bytes
                )
            result = await asyncio.to_thread(
                container.exec_run,
                cmd,
                environment=env,
                demux=True,
                stdout=True,
                stderr=True,
            )
        except Exception as exc:
            return ExecResult(
                exit_code=1,
                stdout="",
                stderr="",
                error_message=str(exc),
            )

        output = cast(tuple[bytes | None, bytes | None] | None, result.output)
        stdout_bytes, stderr_bytes = output if output is not None else (b"", b"")
        return ExecResult(
            exit_code=cast(int, result.exit_code),
            stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
            error_message=None,
        )

    async def stop(self) -> None:
        """Stop the sandbox container while preserving filesystem state.

        Takes two guards, in the same order every lifecycle method on
        ``DockerSandboxService`` takes them: first the owning service's
        per-container-name lock, then this sandbox's own
        ``exclusive_access`` barrier. Acquiring the named lock is what makes
        a stop mutually exclusive with ``start_existing``/``stop_existing``/
        ``delete``/``create_snapshot`` for the same name -- the
        ``exclusive_access`` barrier alone cannot do that, because it drains
        in-flight ``operation()`` work and tracks no exclusive holder, and
        neither ``container.stop()`` nor ``container.start()`` registers as
        an ``operation()``. Because every holder of a named lock that also
        takes ``exclusive_access`` acquires them in this order, the two can
        never deadlock against each other.

        The named lock is not reentrant, so no code already holding
        ``name``'s entry may call this method: that is why
        ``stop_existing()`` stops its container through the raw
        ``Container.stop`` API instead of routing through a handle, the same
        rule that keeps ``create()``'s compensating cleanup on raw
        ``container.remove()`` rather than ``self.delete()``.
        """
        async with self._locks.locked(self._name):
            async with self._control.exclusive_access(mark_deleted=False):
                container = await self._require_container()
                await asyncio.to_thread(container.stop)
                self._store.update_info_state(self._name, "stopped")

    async def info(self) -> SandboxInfo:
        """Return current sandbox metadata derived from Docker inspect."""
        container = await self._require_container()

        runtime_info = _parse_container_config(container)
        self._info.state = runtime_info.state

        return self._info

    async def exec(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]] = None,
        max_output_bytes: Optional[int] = None,
    ) -> ExecResult:
        """Execute a shell command inside the sandbox.

        Exec calls are serialized per sandbox to avoid Docker SDK stream
        corruption when concurrent execs read from the same container socket.
        """
        async with self._control.operation():
            async with self._control.exec_lock:
                return await self._exec_in_container(
                    command, *args, env=env, max_output_bytes=max_output_bytes
                )

    async def run_code(
        self,
        code: str,
        code_type: CodeType = "python",
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute code snippet."""
        code = textwrap.dedent(code)
        if code_type == "python":
            return await self.exec("python", "-c", code, env=env)
        elif code_type == "javascript":
            return await self.exec("node", "-e", code, env=env)
        raise ValueError(f"Unsupported code type: {code_type}")

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        """Upload a local file into the sandbox filesystem."""
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        async with self._control.operation():
            async with self._control.file_lock:
                # Serialize tar-based file transfers so concurrent writes do not produce partially-overwritten archives at the destination path.
                if not overwrite:
                    container = await self._require_container()
                    exists = await asyncio.to_thread(
                        _archive_path_exists, container, remote_path
                    )
                    if exists:
                        raise FileExistsError(
                            f"Remote file already exists: {remote_path}"
                        )

                remote_dir = posixpath.dirname(remote_path) or "/"
                mkdir = await self._exec_in_container("mkdir", "-p", remote_dir)
                if mkdir.exit_code != 0:
                    raise RuntimeError(f"Failed to create remote dir: {mkdir.stderr}")

                container = await self._require_container()
                with tempfile.TemporaryFile() as archive_file:
                    _write_tar_from_local_path(
                        local_path, posixpath.basename(remote_path), archive_file
                    )
                    ok = await asyncio.to_thread(
                        container.put_archive, remote_dir, archive_file
                    )
                    if not ok:
                        raise RuntimeError(f"Failed to upload file to {remote_path}")

    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
        """Download a file from the sandbox to the local filesystem."""
        if not overwrite and os.path.exists(local_path):
            raise FileExistsError(f"Local file already exists: {local_path}")

        async with self._control.operation():
            async with self._control.file_lock:
                container = await self._require_container()
                try:
                    stream, _ = await asyncio.to_thread(
                        container.get_archive, remote_path
                    )
                except NotFound as e:
                    raise FileNotFoundError(
                        f"Remote file not found: {remote_path}"
                    ) from e

                local_dir = os.path.dirname(local_path)
                if local_dir:
                    os.makedirs(local_dir, exist_ok=True)
                with tempfile.TemporaryFile() as archive_file:
                    await asyncio.to_thread(_write_stream_to_file, stream, archive_file)
                    with open(local_path, "wb") as file_obj:
                        _extract_single_file_from_tar(archive_file, file_obj)

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        """Write text content directly to a file inside the sandbox."""
        async with self._control.operation():
            async with self._control.file_lock:
                if not overwrite:
                    container = await self._require_container()
                    exists = await asyncio.to_thread(
                        _archive_path_exists, container, remote_path
                    )
                    if exists:
                        raise FileExistsError(
                            f"Remote file already exists: {remote_path}"
                        )

                remote_dir = posixpath.dirname(remote_path) or "/"
                mkdir = await self._exec_in_container("mkdir", "-p", remote_dir)
                if mkdir.exit_code != 0:
                    raise RuntimeError(f"Failed to create remote dir: {mkdir.stderr}")

                container = await self._require_container()
                with tempfile.TemporaryFile() as archive_file:
                    _write_tar_from_content(
                        content, posixpath.basename(remote_path), archive_file
                    )
                    ok = await asyncio.to_thread(
                        container.put_archive, remote_dir, archive_file
                    )
                    if not ok:
                        raise RuntimeError(f"Failed to write file to {remote_path}")

    async def read_file(self, remote_path: str) -> str:
        """Read text content from a sandbox file."""
        async with self._control.operation():
            async with self._control.file_lock:
                container = await self._require_container()
                try:
                    stream, _ = await asyncio.to_thread(
                        container.get_archive, remote_path
                    )
                except NotFound as e:
                    raise FileNotFoundError(
                        f"Remote file not found: {remote_path}"
                    ) from e
                with tempfile.TemporaryFile() as archive_file:
                    await asyncio.to_thread(_write_stream_to_file, stream, archive_file)
                    with io.BytesIO() as file_bytes:
                        _extract_single_file_from_tar(archive_file, file_bytes)
                        return file_bytes.getvalue().decode("utf-8")


async def _ensure_image(client: Any, image: str) -> None:
    """Ensure the requested image exists locally before container creation."""
    try:
        await asyncio.to_thread(client.images.get, image)
    except ImageNotFound:
        logger.info("Start pulling sandbox image: %s", image)
        await asyncio.to_thread(client.images.pull, image)
        logger.info("Finish pulling sandbox image: %s", image)


async def _create_container(
    client: Any,
    name: str,
    namespace: str,
    image: str,
    template: SandboxTemplate,
    config: SandboxConfig,
    extra_labels: Optional[dict[str, str]] = None,
) -> Container:
    """Create a managed Docker container from sandbox template and config.

    ``extra_labels``, when given, is merged into the container's labels. This
    parameter exists so the ``create()`` lifecycle method can attach the spec
    fingerprint/version attestation labels without this shared helper (also
    used by the legacy ``get_or_create()`` path) needing to know anything
    about that contract.

    Both spec-attestation label keys are always written, blank when no
    attestation was supplied, because Docker merges an *image's* labels into
    a new container's label set for every key the create request does not
    itself specify. Committing a sandbox to a snapshot image copies that
    container's labels into the image (verified on Docker 29.4.0), so a
    container created from a snapshot of a ``create()``-made sandbox would
    otherwise inherit that sandbox's fingerprint and present it as its own
    attestation. Writing the keys unconditionally makes this function the
    single owner of their value for every container it creates, regardless of
    what the base image carries. A blank value is the wire form of "no
    attestation": Docker offers no way to *remove* an inherited label (both
    an empty ``labels`` value and a commit-time ``LABEL key=`` land on the
    empty string, also verified on 29.4.0), which is why ``_build_inspection``
    reads blank as absent rather than as a fingerprint that cannot match.

    The ownership labels (``LABEL_MANAGED``, ``LABEL_NAMESPACE``,
    ``LABEL_SANDBOX_NAME``) are written unconditionally for the same
    reason: a snapshot image inherits its source container's labels, so a
    container created from it would otherwise present the source
    deployment's owner. This function is the single owner of their value
    for every container it creates."""
    await _ensure_image(client, image)

    volumes: dict[str, dict[str, str]] | None = None
    if config.volumes:
        volumes = {
            host_path: {"bind": guest_path, "mode": mode}
            for host_path, guest_path, mode in config.volumes
        }

    ports: dict[str, int] | None = None
    if config.ports:
        ports = {f"{guest}/tcp": host for host, guest in config.ports}

    labels = {
        LABEL_MANAGED: MANAGED_LABEL_VALUE,
        LABEL_SANDBOX_NAME: name,
        LABEL_NAMESPACE: namespace,
        LABEL_TEMPLATE_TYPE: template.type or "image",
        # Shadow any spec attestation inherited from the base image; see the
        # docstring. Overwritten below when the caller supplies a real one.
        LABEL_SPEC_FINGERPRINT: "",
        LABEL_SPEC_VERSION: "",
    }
    if template.type == "snapshot" and template.snapshot_id:
        labels[LABEL_SNAPSHOT_ID] = template.snapshot_id
    if extra_labels:
        labels.update(extra_labels)

    kwargs: dict[str, Any] = {
        "image": image,
        "name": _container_name(name, namespace),
        # Keep the container alive
        "command": ["tail", "-f", "/dev/null"],
        # PID-1 tini forwards SIGTERM so tail exits fast, not on a SIGKILL timeout.
        "init": True,
        "detach": True,
        # Run as root to match the file access behavior of Boxlite.
        "user": "root",
        "working_dir": config.working_dir,
        "environment": config.env,
        "volumes": volumes,
        "ports": ports,
        "nano_cpus": int((config.cpus or 1) * _CPU_NANOS),
        "mem_limit": (config.memory or 512) * 1024 * 1024,
        "network_disabled": bool(config.network_isolated),
        # Security config
        "security_opt": ["no-new-privileges:true"],
        "labels": labels,
    }
    return cast(
        "Container", await asyncio.to_thread(client.containers.create, **kwargs)
    )


class DockerSandboxService(SandboxService):
    """SandboxService implementation backed by Docker containers."""

    def __init__(
        self,
        store: DockerStore,
        client: Optional[Any] = None,
        *,
        namespace: Optional[str] = None,
    ) -> None:
        """Initialize the Docker sandbox service and validate daemon access.

        Args:
            store: Storage for persisting sandbox metadata.
            client: Docker SDK client override (tests). Its position is retained
                for compatibility with existing direct service callers.
            namespace: Stable per-deployment namespace (e.g. the Docker
                Compose project name). When omitted, resolve it from
                ``XAGENT_SANDBOX_NAMESPACE``. Every container this service
                creates is namespaced by it, and every lookup/list operation
                is restricted to it, so multiple deployments sharing one
                Docker daemon can never discover or mutate each other's
                sandboxes.
        """
        resolved_namespace = (
            namespace if namespace is not None else get_sandbox_namespace()
        )
        if resolved_namespace is None:
            raise RuntimeError(
                "XAGENT_SANDBOX_NAMESPACE is required when constructing "
                "DockerSandboxService without an explicit namespace"
            )
        validate_sandbox_namespace(resolved_namespace)
        self._namespace = resolved_namespace
        self._client = client or _create_docker_client()
        self._client.ping()
        self._store = store
        # Per-name lifecycle locks, one entry per sandbox name currently held
        # or waited on. The registry evicts unreferenced entries so it does
        # not grow with every sandbox name ever seen (names such as
        # ``ssh::{task_id}`` come from an unbounded namespace).
        self._locks = KeyedLockRegistry()
        # Sandbox shared runtime control
        self._controls: dict[str, _SandboxControl] = {}

    def _named_lock(self, name: str) -> AsyncContextManager[None]:
        """Serialize lifecycle operations (create/delete/snapshot) for one name.

        Thin alias for the shared ``KeyedLockRegistry`` primitive, which owns
        the waiter counting and identity-checked eviction; see
        ``sandbox/keyed_lock.py`` for why that bookkeeping is deliberately
        synchronous and unguarded.

        This lock is per *container name* and is private to this service. It
        does not span a caller's inspect->create sequence: a caller needing
        get-or-create semantics must hold its own critical section over the
        whole sequence, keyed by whatever identity it owns (``SandboxManager``
        does this with its per-lifecycle-key lock, which is a coarser key than
        a container name).
        """
        return self._locks.locked(name)

    @staticmethod
    async def _await_shielded(coro: Coroutine[Any, Any, Any]) -> Any:
        """Run ``coro`` to completion even under repeated cancellation.

        Used by create() for its two compensating ``container.remove()``
        calls, both of which execute inside ``_named_lock(name)``. That
        lock's ``finally: entry.lock.release()`` runs as soon as a
        ``CancelledError`` propagates out of the ``async with`` body, so a
        bare ``await asyncio.to_thread(container.remove, ...)`` releases the
        lock the instant a cancellation lands on it — while the remove is
        still running in its background thread (``to_thread`` cancellation
        stops waiting, it does not stop the thread). A same-name create()
        that then acquires the freed lock can race that in-flight remove and
        observe a transient "already exists" error from Docker.

        Mirrors the shield-loop in
        ``TaskTurnOrchestrator.begin_turn`` (web/services/task_orchestrator.py):
        wrap the coroutine in its own task and await it under
        ``asyncio.shield`` in a loop, so any number of cancellations delivered
        to *this* await is absorbed without cancelling the underlying task.
        Once the task settles, the original cancellation is re-raised
        regardless of how the task itself settled (the task's own exception,
        if any, is logged but not raised) so the caller's cancellation is
        still honored; if no cancellation occurred, the task's result or
        exception is returned/propagated unchanged.
        """
        task = asyncio.ensure_future(coro)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task_error = task.exception()
                if task_error is not None:
                    logger.error(
                        "Compensating container operation failed while a"
                        " cancelled caller was waiting for it to settle",
                        exc_info=(
                            type(task_error),
                            task_error,
                            task_error.__traceback__,
                        ),
                    )
            raise

    def _get_control(self, name: str) -> _SandboxControl:
        """Get the shared runtime control object for a sandbox."""
        if name not in self._controls:
            self._controls[name] = _SandboxControl(name=name)
        return self._controls[name]

    def _get_live_control(self, name: str) -> _SandboxControl:
        """Return the live control object for a sandbox, replacing it if deleted.

        This is the sole construction point for a fresh ``_SandboxControl``
        used by the create path: if no control exists yet, or the existing
        one was marked deleted by a prior ``delete()``, a new one is
        installed and returned; otherwise the existing live control is
        returned as-is so that in-flight callers sharing it are not split
        across two control objects. Must only be called while holding
        ``_named_lock(name)`` — that per-name mutual exclusion is what makes
        the deleted-check-then-replace race-free. The assertion below can
        only check that *some* task holds ``name``'s lock right now, not
        that it is the caller: ``asyncio.Lock`` has no owner concept, so
        this cannot be a full runtime proof of the contract, only a guard
        against the lock not being held at all.
        """
        entry = self._locks.get(name)
        assert entry is not None and entry.lock.locked(), (
            f"_get_live_control({name!r}) called without holding _named_lock(name)"
        )
        existing = self._controls.get(name)
        if existing is None or existing.deleted:
            existing = _SandboxControl(name=name)
            self._controls[name] = existing
        return existing

    async def _find_container(self, name: str) -> Optional[Container]:
        """Find this service's managed Docker container for a sandbox name.

        The lookup is scoped to the service's deployment namespace: a
        container must carry the v2 managed marker, this service's exact
        namespace, and the logical sandbox name. Containers owned by other
        deployments (or by the legacy pre-namespace scheme) are never
        visible."""
        filters: dict[str, str | list[str] | bool] = {
            "label": [
                *_ownership_label_filters(self._namespace),
                f"{LABEL_SANDBOX_NAME}={name}",
            ]
        }
        containers = await asyncio.to_thread(
            self._client.containers.list, all=True, filters=filters
        )
        if not containers:
            return None
        return cast("Container", containers[0])

    async def get_or_create(
        self,
        name: str,
        template: Optional[SandboxTemplate] = None,
        config: Optional[SandboxConfig] = None,
    ) -> DockerSandbox:
        """Get, resume, or create a Docker-backed sandbox."""
        async with self._named_lock(name):
            control = self._get_live_control(name)

            container = await self._find_container(name)
            if container is not None:
                await asyncio.to_thread(container.reload)
                state = _get_state(str(container.attrs.get("State", {}).get("Status")))
                if state != "running":
                    await asyncio.to_thread(container.start)
                    await asyncio.to_thread(container.reload)
                runtime_info = _parse_container_config(container)
                info = _merge_info(runtime_info, self._store.get_info(name))
                self._store.update_info_state(name, "running")
                return DockerSandbox(
                    name, container, info, self._store, control, self._locks
                )

            template = template or SandboxTemplate(
                type="image", image=DEFAULT_SANDBOX_IMAGE
            )
            cfg = config or SandboxConfig()
            image = template.image or DEFAULT_SANDBOX_IMAGE
            if template.type == "snapshot":
                snapshot = self._store.get_snapshot(cast(str, template.snapshot_id))
                if snapshot is None:
                    raise FileNotFoundError(
                        f"Snapshot not found: {template.snapshot_id}"
                    )
                image = cast(str, snapshot.metadata.get("image_tag"))

            container = await _create_container(
                self._client,
                name,
                self._namespace,
                image,
                template,
                cfg,
            )
            try:
                await asyncio.to_thread(container.start)
            except Exception:
                await asyncio.to_thread(container.remove, force=True)
                raise
            await asyncio.to_thread(container.reload)
            runtime_info = _parse_container_config(container)
            stored_info = SandboxInfo(
                name=name,
                state=runtime_info.state,
                template=template,
                config=cfg,
                created_at=runtime_info.created_at,
            )
            info = _merge_info(runtime_info, stored_info)
            self._store.add_info(name, info)
            return DockerSandbox(
                name, container, info, self._store, control, self._locks
            )

    # --- Spec-based reconciliation lifecycle ---
    #
    # supports_runtime_spec/inspect/create/start_existing/stop_existing do
    # not touch the get_or_create()/list_sandboxes()/delete()/
    # create_snapshot() code paths above, aside from sharing
    # `_create_container` (which owns the spec-attestation label keys for
    # every container either path creates, blanking them when there is no
    # attestation to make -- see its docstring) and the
    # `_named_lock`/`_get_live_control` infrastructure.
    #
    # All of it assumes a single process is the sole owner of the containers
    # it manages. `_named_lock` and `_get_live_control` are in-process
    # registries with no cross-process counterpart, and create()'s
    # check-then-create sequence is only atomic against other tasks in this
    # event loop. Two processes pointed at one Docker daemon with an
    # overlapping name space would race each other; the guard against that is
    # deployment topology, not this code.
    #
    # The deployment namespace (``XAGENT_SANDBOX_NAMESPACE``) is what makes
    # that topology real for co-located deployments: every physical name and
    # every lookup/list filter is scoped to it, so two backend processes
    # from different Compose projects never operate on the same container.
    # Two processes that *share* a namespace remain unsupported: within one
    # namespace the in-process locks above are still the only mutex.

    async def supports_runtime_spec(self) -> bool:
        """Docker backs the explicit spec-reconciliation lifecycle."""
        return True

    async def inspect(self, name: str) -> Optional[SandboxInspection]:
        """Observe a sandbox's current state and runtime facts.

        No side effects and no lock/control acquisition: re-finds the
        container and reloads it fresh on every call rather than reusing a
        cached handle, so the only atomicity guaranteed is within this one
        call.

        When multiple containers carry the same sandbox name label (only
        reachable via out-of-band operations against the Docker daemon),
        ``_find_container`` picks whichever one the Docker API lists first;
        which container that is is undefined and callers must not depend
        on it.
        """
        container = await self._find_container(name)
        if container is None:
            return None
        await asyncio.to_thread(container.reload)
        return _build_inspection(container)

    async def create(
        self, name: str, template: SandboxTemplate, config: SandboxConfig
    ) -> DockerSandbox:
        """Create a new sandbox under the explicit, verified lifecycle contract.

        See ``SandboxService.create`` for the full eight-step contract this
        implements: existence check, snapshot resolution, volume-conflict
        validation, labeled container creation, start with raw-remove
        compensation on failure, publish-before-verify against the desired
        spec, store persistence, and returning a live handle.

        Reference-count-blind, like the rest of this service: it knows nothing
        about how many actors hold a sandbox and enforces no lease. The
        existence check makes it refuse to overwrite a container that is
        already there, but a caller that shares sandboxes between actors owns
        the reference counting and must not route a destructive decision
        through this service on the assumption that it will second-guess one.

        ``_named_lock(name)`` is held for the whole method, so create() is
        atomic against another create/delete/snapshot *for the same name* in
        this process. It does not extend to a caller's own
        inspect-then-create sequence: that window belongs to the caller's
        critical section (see ``_named_lock``).
        """
        async with self._named_lock(name):
            existing = await self._find_container(name)
            if existing is not None:
                raise SandboxAlreadyExistsError(f"Sandbox already exists: {name!r}")

            template_type = template.type or "image"
            if template_type == "snapshot":
                snapshot = self._store.get_snapshot(cast(str, template.snapshot_id))
                if snapshot is None:
                    raise FileNotFoundError(
                        f"Snapshot not found: {template.snapshot_id}"
                    )
                resolved_image = cast(str, snapshot.metadata.get("image_tag"))
                spec_image, spec_snapshot_id = None, template.snapshot_id
            else:
                resolved_image = template.image or DEFAULT_SANDBOX_IMAGE
                spec_image, spec_snapshot_id = resolved_image, None

            _check_no_conflicting_volumes(config.volumes)
            _check_no_conflicting_ports(config.ports)

            desired = ResolvedSandboxRuntimeSpec.from_parts(
                template_type=template_type,
                image=spec_image,
                snapshot_id=spec_snapshot_id,
                working_dir=config.working_dir,
                cpus=config.cpus,
                memory=config.memory,
                env=config.env,
                volumes=config.volumes,
                network_isolated=bool(config.network_isolated),
                ports=config.ports,
            )
            extra_labels = {
                LABEL_SPEC_FINGERPRINT: desired.fingerprint(),
                LABEL_SPEC_VERSION: str(SPEC_CONTRACT_VERSION),
            }

            # Build the container from the same normalized desired spec that
            # publish-before-verify below compares against, so both sides of
            # that check are computed from one canonical source instead of
            # two independent normalizers that could silently diverge (e.g.
            # on a volume's trailing slash or a `..` segment).
            backend_template, backend_config = desired.to_backend_config()

            try:
                container = await _create_container(
                    self._client,
                    name,
                    self._namespace,
                    resolved_image,
                    backend_template,
                    backend_config,
                    extra_labels=extra_labels,
                )
            except APIError as exc:
                if exc.status_code == 409 or "already in use" in str(exc):
                    raise SandboxAlreadyExistsError(
                        f"Sandbox already exists: {name!r}"
                    ) from exc
                raise

            try:
                await asyncio.to_thread(container.start)
            except BaseException as start_exc:
                # Compensate with a raw container removal, never
                # self.delete(): delete() acquires this same _named_lock
                # entry, so calling it here would self-deadlock. The
                # original start failure (including a cancellation) is
                # preserved as the raised exception regardless of whether
                # the compensating remove itself succeeds. The remove runs
                # through _await_shielded so that a cancellation landing
                # here (e.g. a second cancel on top of the one that failed
                # start) cannot cut the remove short and let this method's
                # `_named_lock(name)` release while it is still in flight —
                # that would let a same-name create() race an in-flight
                # remove. If such a cancellation occurs, it is re-raised
                # once the remove settles, taking priority over start_exc.
                try:
                    await self._await_shielded(
                        asyncio.to_thread(container.remove, force=True)
                    )
                except Exception as remove_exc:
                    raise start_exc from remove_exc
                raise start_exc

            await asyncio.to_thread(container.reload)
            inspection = _build_inspection(container)
            mismatches = _find_publish_mismatches(desired, resolved_image, inspection)
            if mismatches:
                # A container whose observed facts disagree with what we
                # asked for must never be published: remove it directly
                # (never self.delete(), for the same self-deadlock reason as
                # the start-failure path above) and fail loudly rather than
                # let a lying label reach the store. Routed through
                # _await_shielded for the same reason as the start-failure
                # compensation above: a cancellation here must not cut the
                # remove short and release `_named_lock(name)` while it is
                # still in flight. If a cancellation occurs, it is re-raised
                # once the remove settles, taking priority over the
                # SandboxRuntimeConflictError below.
                #
                # A failure of the remove itself is logged and swallowed so
                # the caller still gets SandboxRuntimeConflictError naming the
                # mismatched fields, rather than a raw docker APIError that
                # says nothing about why the sandbox was rejected. The
                # verification verdict is the caller-actionable fact; the
                # leaked container is an operator-actionable one, so it goes
                # to the log. `except Exception` deliberately excludes
                # CancelledError, which must keep propagating.
                try:
                    await self._await_shielded(
                        asyncio.to_thread(container.remove, force=True)
                    )
                except Exception as remove_exc:
                    logger.error(
                        "Failed to remove sandbox %r after it failed publish "
                        "verification (mismatched fields: %s); the container "
                        "is left on the host and must be removed manually. "
                        "error=%s",
                        name,
                        ", ".join(mismatches),
                        remove_exc,
                    )
                raise SandboxRuntimeConflictError(
                    f"Sandbox {name!r} failed publish verification; "
                    f"mismatched fields: {', '.join(mismatches)}"
                )

            runtime_info = _parse_container_config(container)
            stored_info = SandboxInfo(
                name=name,
                state=runtime_info.state,
                # The row records the *canonical* desired state, not the
                # caller's raw input: `backend_template`/`backend_config` are
                # the same `desired.to_backend_config()` products the
                # container was built from and the fingerprint label attests.
                # Persisting the raw input instead would make the row and the
                # label two different spellings of one spec, leaving every
                # future reader obliged to re-normalize the row through
                # `from_parts` before comparing it to anything -- an
                # obligation nothing in the type system can enforce, and one
                # that a reconciler falling back to a desired-state
                # comparison on UNVERIFIED (see `spec_matches_inspection`)
                # would silently violate. One canonical source for the
                # container, the label and the row removes that class of bug
                # instead of documenting it. The conversion drops no
                # information: `SandboxTemplate` is exactly
                # (type, image, snapshot_id) and `SandboxConfig` exactly the
                # seven fields the spec carries, so the only difference is
                # canonical spelling.
                template=backend_template,
                config=backend_config,
                created_at=runtime_info.created_at,
            )
            info = _merge_info(runtime_info, stored_info)
            # Persisted only after verification passes; a failure here does
            # not roll back the container — it is left running with a
            # verified label and no store row, which the next reconcile pass
            # converges by observing running+MATCH and recreating the row.
            self._store.add_info(name, info)

            control = self._get_live_control(name)
            return DockerSandbox(
                name, container, info, self._store, control, self._locks
            )

    async def start_existing(self, name: str) -> DockerSandbox:
        """Start a previously-created sandbox, idempotent if already running.

        Adopts whatever container currently answers to ``name``: this method
        performs no spec verification of any kind, and neither reads nor
        writes the fingerprint attestation. A caller that cares whether the
        running container still matches a desired spec must call ``inspect()``
        and ``spec_matches_inspection()`` itself before adopting the result.

        Like every method on this service, this is also reference-count-blind
        (see ``create()``).

        Mutual exclusion is per container name and *is* shared with
        ``DockerSandbox.stop()``: that method acquires this same keyed lock
        entry before its own ``exclusive_access`` barrier, so a caller may
        issue a stop and a lifecycle transition for one name concurrently
        without serializing them itself.
        """
        async with self._named_lock(name):
            container = await self._find_container(name)
            if container is None:
                # Resolved before _get_live_control so that probing a name
                # with no container does not install a control entry that
                # nothing would ever evict.
                raise SandboxNotFoundError(f"Sandbox not found: {name}")
            control = self._get_live_control(name)
            async with control.exclusive_access(mark_deleted=False):
                await asyncio.to_thread(container.reload)
                state = _get_state(str(container.attrs.get("State", {}).get("Status")))
                if state != "running":
                    await asyncio.to_thread(container.start)
                    await asyncio.to_thread(container.reload)
                runtime_info = _parse_container_config(container)
                info = _merge_info(runtime_info, self._store.get_info(name))
                self._store.update_info_state(name, "running")
                return DockerSandbox(
                    name, container, info, self._store, control, self._locks
                )

    async def stop_existing(self, name: str, *, timeout: Optional[int] = None) -> None:
        """Stop an existing sandbox, idempotent if already stopped.

        Reference-count-blind (see ``create()``) and, like
        ``start_existing()``, mutually exclusive with ``DockerSandbox.stop()``
        through the shared per-name lock.

        Stops the container through the raw ``Container.stop`` API rather than
        through a ``DockerSandbox`` handle on purpose: the handle's ``stop()``
        acquires this same non-reentrant lock entry, so delegating to it from
        inside this critical section would self-deadlock.

        ``timeout`` is passed straight through to docker-py's own
        ``container.stop(timeout=...)`` (seconds to wait for a graceful
        stop before SIGKILL); omitted, docker-py applies its own default
        (10s). This is a blocking call bounded by that timeout, not an
        externally-cancellable wait: the caller observes the outcome by
        re-inspecting afterward rather than racing this call with a
        separate deadline.
        """
        async with self._named_lock(name):
            container = await self._find_container(name)
            if container is None:
                # Resolved before _get_live_control for the same reason as in
                # start_existing().
                raise SandboxNotFoundError(f"Sandbox not found: {name}")
            control = self._get_live_control(name)
            async with control.exclusive_access(mark_deleted=False):
                await asyncio.to_thread(container.reload)
                state = _get_state(str(container.attrs.get("State", {}).get("Status")))
                if state == "running":
                    if timeout is None:
                        await asyncio.to_thread(container.stop)
                    else:
                        await asyncio.to_thread(container.stop, timeout=timeout)
                self._store.update_info_state(name, "stopped")

    async def get_store_record(self, name: str) -> Optional[SandboxInfo]:
        """Return the store's own row for this name, or None (see base class)."""
        return self._store.get_info(name)

    async def persist_store_record(self, name: str, info: SandboxInfo) -> None:
        """Write/overwrite the store row for this name (see base class)."""
        self._store.add_info(name, info)

    async def list_sandboxes(self) -> list[SandboxInfo]:
        """List v2 sandboxes owned by this deployment namespace.

        Containers owned by other deployments and legacy pre-namespace
        containers are deliberately invisible, so callers such as capacity
        accounting, idle sweep, and quiesce act only on this owner domain.
        """
        containers = await asyncio.to_thread(
            lambda: self._client.containers.list(
                all=True,
                filters={"label": _ownership_label_filters(self._namespace)},
            )
        )
        result: list[SandboxInfo] = []
        for container in containers:
            runtime_info = _parse_container_config(container)
            stored_info = self._store.get_info(runtime_info.name)
            info = _merge_info(runtime_info, stored_info)
            result.append(info)
        return result

    def count_legacy_containers(self) -> tuple[int, int]:
        """Count running and inactive legacy managed containers.

        Legacy ``xagent.managed=true`` containers are deliberately invisible
        to every namespaced operation; this is the discovery aid operators
        need to complete the documented manual removal after an upgrade.

        Returns:
            A ``(running, inactive)`` count pair. Listing failures return
            ``(0, 0)`` because this startup diagnostic is best-effort.
        """
        try:
            containers = self._client.containers.list(
                all=True,
                filters={"label": f"{LABEL_MANAGED}=true"},
            )
            running = sum(container.status == "running" for container in containers)
            return running, len(containers) - running
        except Exception as exc:
            logger.warning("Failed to list legacy sandbox containers: %s", exc)
            return 0, 0

    async def delete(self, name: str) -> None:
        """Permanently delete a sandbox container and its metadata."""
        async with self._named_lock(name):
            control = self._get_control(name)
            async with control.exclusive_access(mark_deleted=True):
                container = await self._find_container(name)
                if container is not None:
                    await asyncio.to_thread(container.remove, force=True)
                self._store.delete_info(name)
                if self._controls.get(name) is control:
                    self._controls.pop(name)

    async def supports_snapshots(self) -> bool:
        """Return whether snapshot operations are supported."""
        return True

    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        """Create a snapshot by committing the current container filesystem."""
        async with self._named_lock(name):
            control = self._get_control(name)
            async with control.exclusive_access(mark_deleted=False):
                container = await self._find_container(name)
                if container is None:
                    raise SandboxNotFoundError(f"Sandbox not found: {name}")
                if self._store.get_snapshot(snapshot_id) is not None:
                    raise FileExistsError(f"Snapshot already exists: {snapshot_id}")

                tag = _snapshot_tag(snapshot_id, self._namespace)
                repository, _, tag_part = tag.partition(":")
                await asyncio.to_thread(
                    container.commit,
                    repository=repository,
                    tag=tag_part,
                    changes=None,
                )
                image_info = await asyncio.to_thread(self._client.images.get, tag)
                snapshot = SandboxSnapshot(
                    snapshot_id=snapshot_id,
                    metadata={
                        "image_id": image_info.id,
                        "image_tag": tag,
                        "source_sandbox": name,
                    },
                    created_at=str(image_info.attrs.get("Created") or ""),
                )
                self._store.add_snapshot(snapshot)
                return snapshot

    async def list_snapshots(self) -> list[SandboxSnapshot]:
        """List snapshots tracked by the sandbox store."""
        return self._store.list_snapshots()

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot image and its stored metadata."""
        snapshot = self._store.get_snapshot(snapshot_id)
        if snapshot is None:
            return
        image_tag = cast(Optional[str], snapshot.metadata.get("image_tag"))
        if image_tag:
            try:
                await asyncio.to_thread(self._client.images.remove, image=image_tag)
            except (ImageNotFound, NotFound):
                logger.info(
                    "Snapshot image already absent during delete: snapshot_id=%s tag=%s",
                    snapshot_id,
                    image_tag,
                )
            except APIError as exc:
                raise RuntimeError(
                    f"Failed to delete snapshot {snapshot_id}: {exc}"
                ) from exc
        self._store.delete_snapshot(snapshot_id)
