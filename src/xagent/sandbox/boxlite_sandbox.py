"""
Boxlite sandbox implementation.
"""

from __future__ import annotations

import abc
import asyncio
import codecs
import hashlib
import ipaddress
import logging
import os
import shlex
import tempfile
import textwrap
import uuid
from typing import Any, Optional, TypeVar

import boxlite  # type: ignore[import-not-found,import-untyped,unused-ignore]
from boxlite import SimpleBox

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

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = get_sandbox_image()
_EXECUTION_TIMEOUT_SECONDS = 60 * 60
_EXECUTION_KILL_TIMEOUT_SECONDS = 5.0

# BoxLite's allow_net is enforced against the resolved connection destination.
# Build an allowlist that is the IPv4 Internet minus IANA special-use ranges.
# IPv6 entries are included for forward compatibility, but BoxLite 0.10.7's
# native filter rejects non-IPv4 destinations before CIDR matching, so IPv6
# egress currently remains fail-closed. This blocks loopback, link-local,
# RFC1918/ULA, the standard link-local metadata endpoint, and multicast
# destinations even when untrusted code uses literal addresses or a hostname
# is rebound after validation. Host/public control-plane addresses require an
# additional deployment firewall because a guest filter cannot identify them.
_NON_PUBLIC_IPV4 = tuple(
    ipaddress.IPv4Network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)
_NON_PUBLIC_IPV6 = tuple(
    ipaddress.IPv6Network(network)
    for network in (
        # The IETF protocol-assignment block contains transition, benchmarking,
        # discard and other special-purpose allocations. Deny it as a whole;
        # false negatives are safer than exposing a non-public control plane.
        "2001::/23",
        "2001:db8::/32",
        "2002::/16",
        "3fff::/20",
    )
)


_Network = TypeVar("_Network", ipaddress.IPv4Network, ipaddress.IPv6Network)


async def _kill_execution_bounded(execution: Any) -> None:
    """Best-effort guest termination that cannot replace one hang with another."""
    try:
        await asyncio.wait_for(
            execution.kill(), timeout=_EXECUTION_KILL_TIMEOUT_SECONDS
        )
    except Exception:  # noqa: BLE001 - preserve the original failure/cancellation
        logger.exception("Failed to kill BoxLite execution")


def _subtract_networks(
    allowed: list[_Network],
    blocked: tuple[_Network, ...],
) -> list[_Network]:
    for denied in blocked:
        remaining: list[_Network] = []
        for candidate in allowed:
            if not candidate.overlaps(denied):
                remaining.append(candidate)
            elif denied.subnet_of(candidate):
                remaining.extend(candidate.address_exclude(denied))
        allowed = list(ipaddress.collapse_addresses(remaining))
    return allowed


PUBLIC_EGRESS_NETWORKS = tuple(
    str(network)
    for network in (
        _subtract_networks([ipaddress.IPv4Network("0.0.0.0/0")], _NON_PUBLIC_IPV4)
        + _subtract_networks([ipaddress.IPv6Network("2000::/3")], _NON_PUBLIC_IPV6)
    )
)

_BOXLITE_POLICY_DIGEST = hashlib.sha256(
    "\n".join(
        ("outbound=public-only", "inbound=disabled", *PUBLIC_EGRESS_NETWORKS)
    ).encode("ascii")
).hexdigest()[:12]
_BOXLITE_SECURITY_PROFILE = f"boxlite-public-egress-v1-{_BOXLITE_POLICY_DIGEST}"


class BoxliteStore(abc.ABC):
    """
    Store for persisting Boxlite data.
    """

    @abc.abstractmethod
    def get_info(self, name: str) -> Optional[SandboxInfo]:
        """Get sandbox info."""

    @abc.abstractmethod
    def add_info(self, name: str, info: SandboxInfo) -> None:
        """Add sandbox info."""

    @abc.abstractmethod
    def update_info_state(self, name: str, state: str) -> None:
        """Update sandbox info state."""

    @abc.abstractmethod
    def delete_info(self, name: str) -> None:
        """Delete sandbox info."""


class MemBoxliteStore(BoxliteStore):
    """
    In-memory implementation of BoxliteStore.
    """

    def __init__(self) -> None:
        self._metadata: dict[str, SandboxInfo] = {}

    def get_info(self, name: str) -> Optional[SandboxInfo]:
        return self._metadata.get(name)

    def add_info(self, name: str, info: SandboxInfo) -> None:
        self._metadata[name] = info

    def update_info_state(self, name: str, state: str) -> None:
        if name in self._metadata:
            self._metadata[name].state = state

    def delete_info(self, name: str) -> None:
        if name in self._metadata:
            del self._metadata[name]


def _get_state(raw_info: boxlite.BoxInfo) -> str:  # type: ignore[no-any-unimported]
    """Map boxlite BoxInfo.state to 'running' / 'stopped' / 'unknown'."""
    state = raw_info.state.status.lower()
    if "running" in state:
        return "running"
    if any(key in state for key in ("stopped", "paused", "exited", "created")):
        return "stopped"
    return "unknown"


def _get_info_from_box_info(raw_info: boxlite.BoxInfo) -> SandboxInfo:  # type: ignore[no-any-unimported]
    tpl = SandboxTemplate(
        type="image",
        image=raw_info.image,
    )
    cfg = SandboxConfig(
        cpus=raw_info.cpus,
        memory=raw_info.memory_mib,
        env=None,  # Cannot retrieve
        volumes=None,  # Cannot retrieve
        network_isolated=None,  # Cannot retrieve
        ports=None,  # Cannot retrieve
    )
    info = SandboxInfo(
        name=raw_info.name,
        state=_get_state(raw_info),
        template=tpl,
        config=cfg,
        created_at=raw_info.created_at,
    )
    return info


async def _collect_capped_stream(stream: Any, cap: int) -> tuple[str, bool]:
    """Drain a BoxLite output stream while retaining at most ``cap`` bytes."""
    if stream is None:
        return "", False

    retained = bytearray()
    truncated = False
    async for chunk in stream:
        encoded = chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8")
        remaining = cap - len(retained)
        if remaining > 0:
            retained.extend(encoded[:remaining])
        if len(encoded) > max(remaining, 0):
            truncated = True
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    # If the byte cap splits a multibyte codepoint, final=False omits only the
    # incomplete suffix while still replacing malformed bytes in the prefix.
    return decoder.decode(bytes(retained), final=not truncated), truncated


class BoxliteSandbox(Sandbox):
    """
    Boxlite implementation.
    """

    def __init__(  # type: ignore[no-any-unimported]
        self,
        sandbox_name: str,
        box: SimpleBox,
        info: SandboxInfo,
        store: BoxliteStore,
    ) -> None:
        self._box = box
        self._name = sandbox_name
        self._info = info
        self._store = store

    # --- Properties ---

    @property
    def name(self) -> str:
        """Sandbox name (unique identifier)."""
        return self._name

    # --- Lifecycle ---

    async def stop(self) -> None:
        """Stop the sandbox, preserving its state."""
        await self._box.stop()
        self._store.update_info_state(self._name, "stopped")

    async def info(self) -> SandboxInfo:
        """Get sandbox status information."""
        # Update state in real-time
        # box.info() will throw an error when the box has been stopped, so we need to use the box._runtime.get_info(name) method instead.
        box_info = await self._box._runtime.get_info(self._name)
        self._info.state = _get_state(box_info)

        return self._info

    # --- Execution ---

    async def _exec_with_output_cap(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]],
        max_output_bytes: int,
    ) -> ExecResult:
        """Use BoxLite's streaming handle so retained output is bounded.

        ``SimpleBox.exec`` joins every output chunk before returning. The
        locked BoxLite API exposes the same execution streams on its native
        box handle; consume and drain them concurrently here so a noisy guest
        cannot grow backend memory without bound.
        """
        if max_output_bytes < 0:
            raise ValueError("max_output_bytes must be non-negative")

        native_box = getattr(self._box, "_box", None)
        if native_box is None:
            raise RuntimeError("BoxLite box is not started")
        execution = await native_box.exec(
            command,
            list(args) if args else None,
            list(env.items()) if env else None,
            user=None,
            timeout_secs=_EXECUTION_TIMEOUT_SECONDS,
            cwd=None,
        )
        try:
            stdout_stream = execution.stdout()
        except Exception:  # noqa: BLE001 - native stream acquisition
            logger.exception("Failed to acquire BoxLite stdout stream")
            stdout_stream = None
        try:
            stderr_stream = execution.stderr()
        except Exception:  # noqa: BLE001 - native stream acquisition
            logger.exception("Failed to acquire BoxLite stderr stream")
            stderr_stream = None

        stdout = stderr = ""
        stdout_truncated = stderr_truncated = False
        error_message = None
        collectors = (
            asyncio.create_task(
                _collect_capped_stream(stdout_stream, max_output_bytes)
            ),
            asyncio.create_task(
                _collect_capped_stream(stderr_stream, max_output_bytes)
            ),
        )
        try:
            (
                (stdout, stdout_truncated),
                (stderr, stderr_truncated),
            ) = await asyncio.gather(*collectors)
        except BaseException:
            for collector in collectors:
                if not collector.done():
                    collector.cancel()
            await asyncio.gather(*collectors, return_exceptions=True)
            await _kill_execution_bounded(execution)
            raise

        try:
            raw_result = await execution.wait()
            exit_code = raw_result.exit_code
            error_message = raw_result.error_message
        except asyncio.CancelledError:
            await _kill_execution_bounded(execution)
            raise
        except Exception:  # noqa: BLE001 - native wait failure becomes exec failure
            logger.exception("Failed to wait for BoxLite execution")
            exit_code = -1

        return ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=stdout_truncated or stderr_truncated,
            error_message=error_message,
        )

    async def exec(
        self,
        command: str,
        *args: str,
        env: Optional[dict[str, str]] = None,
        max_output_bytes: Optional[int] = None,
    ) -> ExecResult:
        if max_output_bytes is None:
            res = await self._box.exec(command, *args, env=env)
            truncated = False
        else:
            res = await self._exec_with_output_cap(
                command,
                *args,
                env=env,
                max_output_bytes=max_output_bytes,
            )
            truncated = res.truncated

        # Filter out seccomp warnings (known issue on macOS, usually appears on first line)
        stderr = res.stderr
        if stderr and "seccomp not available" in stderr:
            lines = stderr.split("\n", 1)  # Split only the first line
            if len(lines) > 0 and "seccomp not available" in lines[0]:
                # Remove first line, keep the rest
                stderr = lines[1] if len(lines) > 1 else ""

        return ExecResult(
            exit_code=res.exit_code,
            stdout=res.stdout,
            stderr=stderr,
            truncated=truncated,
            error_message=res.error_message,
        )

    async def run_code(
        self,
        code: str,
        code_type: CodeType = "python",
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """
        Execute code snippet.
        """
        code = textwrap.dedent(code)
        if code_type == "python":
            return await self.exec("python", "-c", code, env=env)
        elif code_type == "javascript":
            return await self.exec("node", "-e", code, env=env)
        raise ValueError(f"Unsupported code type: {code_type}")

    # --- File Operations ---

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if not overwrite:
            check = await self.exec("test", "-e", remote_path)
            if check.exit_code == 0:
                raise FileExistsError(f"Remote file already exists: {remote_path}")

        # First copy to temp directory (if copying directly to mount directory, it won't be readable on host)
        # Note: Cannot use /tmp as it's a tmpfs mount, copy_in will fail
        # Use /var/tmp or other non-tmpfs directory

        temp_filename = f"_upload_{uuid.uuid4().hex}"
        temp_remote = f"/var/tmp/{temp_filename}"

        # Ensure /var/tmp exists
        await self.exec("mkdir", "-p", "/var/tmp")

        # copy_in to temporary location
        await self._box.copy_in(local_path, temp_remote, overwrite=overwrite)

        # Verify temporary file exists
        check = await self.exec("test", "-f", temp_remote)
        if check.exit_code != 0:
            raise RuntimeError(
                f"Failed to copy file to temporary location: {temp_remote}"
            )

        # Create target directory
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            await self.exec("mkdir", "-p", remote_dir)

        temp_remote_quoted = shlex.quote(temp_remote)
        remote_path_quoted = shlex.quote(remote_path)

        if overwrite:
            result = await self.exec(
                "sh",
                "-c",
                f"mv {temp_remote_quoted} {remote_path_quoted}",
            )
        else:
            result = await self.exec(
                "sh",
                "-c",
                f"if [ -e {remote_path_quoted} ]; then exit 100; fi; mv {temp_remote_quoted} {remote_path_quoted}",
            )
            if result.exit_code == 100:
                await self.exec("rm", "-f", temp_remote)
                raise FileExistsError(f"Remote file already exists: {remote_path}")

        if result.exit_code != 0:
            # Clean up temporary file
            await self.exec("rm", "-f", temp_remote)
            raise RuntimeError(f"Failed to move file to {remote_path}: {result.stderr}")

    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
        check = await self.exec("test", "-e", remote_path)
        if check.exit_code != 0:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")

        if not overwrite and os.path.exists(local_path):
            raise FileExistsError(f"Local file already exists: {local_path}")

        # First copy to temp directory (avoid volume mount issues)

        temp_filename = f"_download_{uuid.uuid4().hex}"
        temp_remote = f"/var/tmp/{temp_filename}"

        # Ensure /var/tmp exists
        await self.exec("mkdir", "-p", "/var/tmp")

        # Use cp command to copy to temporary location (supports volume mounts)
        result = await self.exec("cp", remote_path, temp_remote)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to copy file to temporary location: {result.stderr}"
            )

        # Create local directory
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        # copy_out from temporary location to local
        try:
            await self._box.copy_out(temp_remote, local_path, overwrite=overwrite)
        finally:
            # Clean up temporary file
            await self.exec("rm", "-f", temp_remote)

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        # Write to local temp file
        fd, tmp = tempfile.mkstemp(suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            # Use upload_file to upload (supports volume mounts)
            await self.upload_file(tmp, remote_path, overwrite=overwrite)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    async def read_file(self, remote_path: str) -> str:
        # Use download_file to read (supports volume mounts)
        fd, tmp = tempfile.mkstemp(suffix=".tmp")
        os.close(fd)
        try:
            await self.download_file(remote_path, tmp, overwrite=True)
            with open(tmp, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


async def _create_or_reuse_box(  # type: ignore[no-any-unimported]
    name: str,
    template: SandboxTemplate,
    config: SandboxConfig,
    runtime: boxlite.Boxlite,
) -> SimpleBox:
    """Create a new Box."""
    if config.ports and config.network_isolated:
        raise ValueError(
            "Published ports cannot be combined with isolated guest networking"
        )
    # Build SimpleBox parameters
    kwargs: dict = {
        "image": template.image,
        "cpus": config.cpus,
        "memory_mib": config.memory,
        "disk_size_gb": 10,  # Increased to accommodate packages + workspace files
        "auto_remove": False,  # Don't auto-remove, preserve state
        "runtime": runtime,
        "name": name,
        "reuse_existing": True,  # Allow reusing existing box
        "working_dir": config.working_dir,
    }

    if config.env:
        kwargs["env"] = list(config.env.items())

    if config.volumes:
        kwargs["volumes"] = [
            (h, g, mode.lower() == "ro") for h, g, mode in config.volumes
        ]

    if config.ports:
        kwargs["ports"] = config.ports

    if config.network_isolated:
        # SecurityOptions.network_enabled controls the host-side sandbox, not
        # guest networking. BoxLite 0.10 requires an explicit guest policy.
        kwargs["network"] = boxlite.NetworkSpec(
            outbound=boxlite.OutboundNetworkSpec(mode="disabled"),
            inbound=boxlite.InboundNetworkSpec(mode="disabled"),
        )
    else:
        kwargs["network"] = boxlite.NetworkSpec(
            outbound=boxlite.OutboundNetworkSpec(
                mode="enabled", allow_net=list(PUBLIC_EGRESS_NETWORKS)
            ),
            inbound=boxlite.InboundNetworkSpec(mode="disabled"),
        )

    # Create SimpleBox
    box = SimpleBox(**kwargs)
    await box.start()
    return box


class BoxliteSandboxService(SandboxService):
    """
    Boxlite implementation.
    """

    def __init__(self, store: BoxliteStore, home_dir: Optional[str] = None) -> None:
        """
        Initialize BoxliteSandboxService。

        Args:
            store: Storage for persisting sandbox information
            home_dir: Boxlite's home directory, used to store data such as mirroring and VMs.
                    If None, use the default directory (usually ~/.boxlite)
        """
        if home_dir:
            self._runtime = boxlite.Boxlite(boxlite.Options(home_dir=home_dir))
        else:
            self._runtime = boxlite.Boxlite.default()
        self._store = store
        # Lock for protecting concurrent creation, one lock per name
        self._locks: dict[str, asyncio.Lock] = {}
        # Lock for protecting the _locks dict itself
        self._locks_lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        template: Optional[SandboxTemplate] = None,
        config: Optional[SandboxConfig] = None,
    ) -> BoxliteSandbox:
        # Snapshot creation not supported
        if template is not None and template.type == "snapshot":
            raise NotImplementedError("Unsupported")

        # Get or create lock for this name
        async with self._locks_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            lock = self._locks[name]

        # Use lock to protect entire get_or_create process
        async with lock:
            # Check if box already exists
            raw_box = await self._runtime.get(name)
            info = self._store.get_info(name) if raw_box else None
            desired_template = template or (
                info.template
                if info is not None
                else SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE)
            )
            desired_config = (
                config or (info.config if info is not None else SandboxConfig())
            ).model_copy(update={"security_profile": _BOXLITE_SECURITY_PROFILE})
            if raw_box:
                # BoxLite has no immutable labels. Persisted desired state is
                # therefore the only way to detect a network-policy upgrade.
                # Always require the current profile, including for callers
                # that omit template/config and after in-memory metadata loss;
                # otherwise an old unrestricted VM could be reused fail-open.
                stored_config = (
                    info.config.model_copy(update={"pids_limit": None})
                    if info is not None
                    else None
                )
                effective_config = desired_config.model_copy(
                    update={"pids_limit": None}
                )
                if (
                    info is None
                    or info.template != desired_template
                    or stored_config != effective_config
                ):
                    await self._runtime.remove(name, force=True)
                    self._store.delete_info(name)
                    raw_box = None
            if not raw_box:
                # Box doesn't exist, create new one
                info = SandboxInfo(
                    name=name,
                    state="running",
                    template=desired_template,
                    config=desired_config,
                )

            assert info is not None
            # Create or reuse box
            box = await _create_or_reuse_box(
                name, info.template, info.config, self._runtime
            )

            # Update created_at if this is a new box
            if box.created:
                info.created_at = (await box.info()).created_at
                self._store.add_info(name, info)

            # Update state
            self._store.update_info_state(name, "running")
            return BoxliteSandbox(
                sandbox_name=name, box=box, info=info, store=self._store
            )

    async def stop_existing(self, name: str, *, timeout: Optional[int] = None) -> None:
        """Stop a native BoxLite handle without validating or recreating it."""
        del timeout  # BoxLite 0.10.7 does not expose a graceful-stop timeout.
        async with self._locks_lock:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()
            lock = self._locks[name]

        async with lock:
            raw_box = await self._runtime.get(name)
            if raw_box is None:
                raise SandboxNotFoundError(f"Sandbox not found: {name}")
            await raw_box.stop()
            self._store.update_info_state(name, "stopped")

    async def list_sandboxes(self) -> list[SandboxInfo]:
        # Use boxlite as source of truth
        raw_list: list[boxlite.BoxInfo] = await self._runtime.list_info()  # type: ignore[no-any-unimported]
        result: list[SandboxInfo] = []
        for raw_info in raw_list:
            box_name = raw_info.name
            info = self._store.get_info(box_name)
            if info:
                result.append(info)
                info.state = _get_state(raw_info)  # Refresh state
            else:
                # DB data lost, read from boxlite
                info = _get_info_from_box_info(raw_info)
                result.append(info)
        return result

    async def delete(self, name: str) -> None:
        box = await self._runtime.get(name)
        if not box:
            self._store.delete_info(name)
            # clear lock
            async with self._locks_lock:
                self._locks.pop(name, None)
            return

        try:
            await self._runtime.remove(name, force=True)
            self._store.delete_info(name)
            # clear lock
            async with self._locks_lock:
                self._locks.pop(name, None)
        except Exception as e:
            raise RuntimeError(f"delete {name!r} error: {e}") from e

    async def supports_snapshots(self) -> bool:
        return False

    async def supports_runtime_spec(self) -> bool:
        """Boxlite has no label carrier, hardcodes reuse_existing=True (a
        fail-open get-or-create), cannot retrieve env/volumes/ports back
        from a running box, and has no control/name-lock infrastructure of
        its own. Inspect/create/start remain unsupported; stop_existing is
        implemented only so startup cleanup can quiesce a native handle
        without entering get_or_create migration. This probe therefore stays
        false until the complete reconciliation contract is implemented.
        """
        return False

    def runtime_security_profile(self) -> str | None:
        return _BOXLITE_SECURITY_PROFILE

    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        raise NotImplementedError("Unsupported")

    async def list_snapshots(self) -> list[SandboxSnapshot]:
        raise NotImplementedError("Unsupported")

    async def delete_snapshot(self, snapshot_id: str) -> None:
        raise NotImplementedError("Unsupported")
