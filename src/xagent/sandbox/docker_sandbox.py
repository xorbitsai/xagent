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
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, cast

from docker.errors import APIError, ImageNotFound, NotFound

import docker

from ..config import get_sandbox_image
from .base import (
    CodeType,
    ExecResult,
    Sandbox,
    SandboxConfig,
    SandboxInfo,
    SandboxNotFoundError,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
)

if TYPE_CHECKING:
    from docker.models.containers import Container

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = get_sandbox_image()

LABEL_MANAGED = "xagent.managed"
LABEL_SANDBOX_NAME = "xagent.sandbox.name"
LABEL_TEMPLATE_TYPE = "xagent.sandbox.template.type"
LABEL_SNAPSHOT_ID = "xagent.sandbox.snapshot_id"
CONTAINER_NAME_PREFIX = "xagent_sandbox_"
SNAPSHOT_REPOSITORY = "xagent-sandbox-snapshot"
_CPU_NANOS = 1_000_000_000


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


def _container_name(name: str) -> str:
    """Build the managed Docker container name for a sandbox."""
    return f"{CONTAINER_NAME_PREFIX}{_make_safe_name(name)}"


def _snapshot_tag(snapshot_id: str) -> str:
    """Build the managed Docker image tag for a snapshot."""
    safe = _make_safe_name(snapshot_id)
    return f"{SNAPSHOT_REPOSITORY}:{safe}"


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
    result = container.exec_run(["test", "-e", remote_path], stdout=False, stderr=False)
    return cast(int, result.exit_code) == 0


@dataclass
class _SandboxControl:
    """Shared concurrency guard for operations targeting the same sandbox."""

    name: str
    active_ops: int = 0
    new_operations_paused: bool = False
    deleted: bool = False
    file_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
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
    ) -> None:
        self._container = container
        self._name = sandbox_name
        self._info = info
        self._store = store
        self._control = control

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
    ) -> ExecResult:
        """Execute a command directly against the current container instance."""
        container = await self._require_container()
        cmd: list[str] = [command, *args]
        try:
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
        """Stop the sandbox container while preserving filesystem state."""
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
    ) -> ExecResult:
        """Execute a shell command inside the sandbox."""
        async with self._control.operation():
            return await self._exec_in_container(command, *args, env=env)

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
    image: str,
    template: SandboxTemplate,
    config: SandboxConfig,
) -> Container:
    """Create a managed Docker container from sandbox template and config."""
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
        LABEL_MANAGED: "true",
        LABEL_SANDBOX_NAME: name,
        LABEL_TEMPLATE_TYPE: template.type or "image",
    }
    if template.type == "snapshot" and template.snapshot_id:
        labels[LABEL_SNAPSHOT_ID] = template.snapshot_id

    kwargs: dict[str, Any] = {
        "image": image,
        "name": _container_name(name),
        # Keep the container alive
        "command": ["tail", "-f", "/dev/null"],
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
    ) -> None:
        """Initialize the Docker sandbox service and validate daemon access."""
        self._client = client or _create_docker_client()
        self._client.ping()
        self._store = store
        # Lock for protecting concurrent creation, one lock per name
        self._locks: dict[str, asyncio.Lock] = {}
        # Lock for protecting the _locks dict itself
        self._locks_lock = asyncio.Lock()
        # Sandbox shared runtime control
        self._controls: dict[str, _SandboxControl] = {}

    async def _get_name_lock(self, name: str) -> asyncio.Lock:
        """Get the per-sandbox lifecycle lock, creating it on demand."""
        async with self._locks_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            return self._locks[name]

    def _get_control(self, name: str) -> _SandboxControl:
        """Get the shared runtime control object for a sandbox."""
        if name not in self._controls:
            self._controls[name] = _SandboxControl(name=name)
        return self._controls[name]

    async def _find_container(self, name: str) -> Optional[Container]:
        """Find the managed Docker container for a sandbox name."""
        filters: dict[str, str | list[str] | bool] = {
            "label": [f"{LABEL_MANAGED}=true", f"{LABEL_SANDBOX_NAME}={name}"]
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
        lock = await self._get_name_lock(name)
        async with lock:
            control = self._get_control(name)
            if control.deleted:
                # Recreate the control object after delete() so the same sandbox name can be safely reused in a later lifecycle.
                self._controls[name] = _SandboxControl(name=name)
                control = self._controls[name]

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
                return DockerSandbox(name, container, info, self._store, control)

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
            return DockerSandbox(name, container, info, self._store, control)

    async def list_sandboxes(self) -> list[SandboxInfo]:
        """List all managed Docker sandboxes."""
        containers = await asyncio.to_thread(
            lambda: self._client.containers.list(
                all=True,
                filters={"label": f"{LABEL_MANAGED}=true"},
            )
        )
        result: list[SandboxInfo] = []
        for container in containers:
            runtime_info = _parse_container_config(container)
            stored_info = self._store.get_info(runtime_info.name)
            info = _merge_info(runtime_info, stored_info)
            result.append(info)
        return result

    async def delete(self, name: str) -> None:
        """Permanently delete a sandbox container and its metadata."""
        lock = await self._get_name_lock(name)
        async with lock:
            control = self._get_control(name)
            async with control.exclusive_access(mark_deleted=True):
                container = await self._find_container(name)
                if container is not None:
                    await asyncio.to_thread(container.remove, force=True)
                self._store.delete_info(name)
                self._controls.pop(name, None)
                async with self._locks_lock:
                    self._locks.pop(name, None)

    async def supports_snapshots(self) -> bool:
        """Return whether snapshot operations are supported."""
        return True

    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        """Create a snapshot by committing the current container filesystem."""
        lock = await self._get_name_lock(name)
        async with lock:
            control = self._get_control(name)
            async with control.exclusive_access(mark_deleted=False):
                container = await self._find_container(name)
                if container is None:
                    raise SandboxNotFoundError(f"Sandbox not found: {name}")
                if self._store.get_snapshot(snapshot_id) is not None:
                    raise FileExistsError(f"Snapshot already exists: {snapshot_id}")

                tag = _snapshot_tag(snapshot_id)
                await asyncio.to_thread(
                    container.commit,
                    repository=SNAPSHOT_REPOSITORY,
                    tag=tag.split(":", 1)[1],
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
