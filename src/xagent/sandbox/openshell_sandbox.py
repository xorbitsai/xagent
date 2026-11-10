"""
OpenShell sandbox implementation (experimental).

Backs xagent's tool-execution sandbox with an NVIDIA OpenShell gateway
instead of a Docker daemon. The backend implements the basic, non-reconciling
contract: get_or_create / exec / file transfer / stop / delete, no snapshots
and no spec reconciliation (SandboxManager's legacy route).

What OpenShell adds over the Docker backend:

- Egress is default-deny. ``network_isolated=False`` no longer means "full
  internet"; outbound access is whatever the policy file allows, matched per
  calling binary and destination.
- Credentials attached as OpenShell providers reach the sandbox only as
  placeholders and are substituted by the gateway-side proxy.
- The workload runs non-root under Landlock + seccomp, with the trusted
  supervisor in a separate container.

Configuration (environment):

- ``XAGENT_OPENSHELL_WORKSPACE``: OpenShell workspace (default ``default``).
- ``XAGENT_OPENSHELL_GATEWAY``: gateway name (default: the active gateway).
- ``XAGENT_OPENSHELL_POLICY_FILE``: optional OpenShell policy YAML whose
  ``network_policies`` (and extra filesystem paths) are merged into every
  sandbox this backend creates.
- ``XAGENT_OPENSHELL_PROVIDERS``: comma-separated provider names to attach.
- ``XAGENT_SANDBOX_NAMESPACE`` (required, via the web factory): the
  deployment's ownership boundary on a shared gateway, as for Docker.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import posixpath
import shlex
import textwrap
import uuid
from typing import Any, Protocol

import grpc  # type: ignore[import-untyped]
from google.protobuf import json_format, struct_pb2  # type: ignore[import-untyped]
from openshell import SandboxClient  # type: ignore[import-not-found]
from openshell._proto import (  # type: ignore[import-not-found]
    openshell_pb2,
    sandbox_pb2,
)

from ..config import get_sandbox_image, validate_sandbox_namespace
from .base import (
    CodeType,
    ExecResult,
    Sandbox,
    SandboxConfig,
    SandboxInfo,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
)

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_IMAGE = get_sandbox_image()

MANAGED_LABEL = "xagent.io/managed"
# Digest of the deployment/worker namespace: label values only allow
# [a-z0-9._-] and must start and end alphanumeric, which namespaces need not.
NAMESPACE_LABEL = "xagent.io/namespace"
# The xagent name rides base32hex-encoded in NAME_LABEL-0..N-1: one label
# value holds at most 63 characters, and names such as
# ``chrome-execution::<64-hex digest>`` need several.
NAME_LABEL = "xagent.io/name"
_LABEL_VALUE_MAX = 63
_NAME_LABEL_CHUNKS_MAX = 16

# Paths the workload may read even though Landlock is default-deny. OpenShell
# adds its own baseline on top when network rules are present.
_BASE_READ_ONLY = ["/usr", "/lib", "/bin", "/sbin", "/etc", "/opt", "/proc", "/dev"]
_BASE_READ_WRITE = ["/tmp", "/var/tmp", "/dev/null"]

# Each exec stdin chunk travels as one gRPC message, capped at 1 MiB.
_UPLOAD_CHUNK_BYTES = 768 * 1024
# The gateway rejects exec requests with any single argument over this size
# ("command argument N exceeds 32768 byte limit").
_GATEWAY_ARG_LIMIT = 32768

_PHASE_STATE = {
    openshell_pb2.SANDBOX_PHASE_READY: "running",
    openshell_pb2.SANDBOX_PHASE_PROVISIONING: "running",
    openshell_pb2.SANDBOX_PHASE_STARTING: "running",
    openshell_pb2.SANDBOX_PHASE_STOPPED: "stopped",
    openshell_pb2.SANDBOX_PHASE_STOPPING: "stopped",
    openshell_pb2.SANDBOX_PHASE_SUSPENDED: "stopped",
    openshell_pb2.SANDBOX_PHASE_SUSPENDING: "stopped",
    openshell_pb2.SANDBOX_PHASE_COMPLETED: "stopped",
}


class OpenShellStore(Protocol):
    """Persists what the gateway cannot give back (env, volumes, template)."""

    def get_info(self, name: str) -> SandboxInfo | None: ...

    def add_info(self, name: str, info: SandboxInfo) -> None: ...

    def update_info_state(self, name: str, state: str) -> None: ...

    def delete_info(self, name: str) -> None: ...


class MemOpenShellStore(OpenShellStore):
    def __init__(self) -> None:
        self._metadata: dict[str, SandboxInfo] = {}

    def get_info(self, name: str) -> SandboxInfo | None:
        return self._metadata.get(name)

    def add_info(self, name: str, info: SandboxInfo) -> None:
        self._metadata[name] = info

    def update_info_state(self, name: str, state: str) -> None:
        if name in self._metadata:
            self._metadata[name].state = state

    def delete_info(self, name: str) -> None:
        self._metadata.pop(name, None)


def openshell_name(name: str, namespace: str) -> str:
    """Map an xagent sandbox name to a gateway name, scoped to a namespace.

    The gateway caps names at 19 characters, too short for a readable slug
    of ``user::{owner}:{suffix}``; the original name rides in name labels.
    The namespace is part of the hashed identity (as for the Docker backend's
    container names), so two deployments or workers sharing one gateway
    workspace never resolve the same xagent name to the same sandbox.
    """
    digest = hashlib.sha256(f"{namespace}\0{name}".encode()).hexdigest()
    return "xa-" + digest[:16]


def namespace_label_value(namespace: str) -> str:
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:32]


def encode_name_labels(name: str) -> dict[str, str]:
    """Carry the original name across as many labels as it needs.

    Raises ValueError for a name too long to carry, instead of creating a
    sandbox whose listing could not map back to its xagent name.
    """
    value = base64.b32hexencode(name.encode("utf-8")).decode().rstrip("=").lower()
    chunks = [
        value[i : i + _LABEL_VALUE_MAX] for i in range(0, len(value), _LABEL_VALUE_MAX)
    ]
    if not chunks or len(chunks) > _NAME_LABEL_CHUNKS_MAX:
        raise ValueError(
            f"Sandbox name of {len(name.encode('utf-8'))} bytes cannot be "
            "carried in OpenShell labels"
        )
    return {f"{NAME_LABEL}-{i}": chunk for i, chunk in enumerate(chunks)}


def decode_name_labels(labels: Any) -> str | None:
    """Reassemble a name from its label chunks; None if absent or corrupt."""
    chunks: list[str] = []
    for i in range(_NAME_LABEL_CHUNKS_MAX):
        chunk = labels.get(f"{NAME_LABEL}-{i}")
        if not chunk:
            break
        chunks.append(chunk)
    if not chunks:
        return None
    value = "".join(chunks)
    try:
        padded = value.upper() + "=" * (-len(value) % 8)
        return base64.b32hexdecode(padded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _is_not_found(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    return (
        isinstance(exc, grpc.RpcError)
        and callable(code)
        and (code() == grpc.StatusCode.NOT_FOUND)
    )


def _phase_state(ref: Any) -> str:
    phase = getattr(getattr(ref, "status", None), "phase", 0)
    return _PHASE_STATE.get(phase, "unknown")


# CLI YAML spells endpoint enums as ``enforce`` / ``read-only``; the proto
# JSON mapping wants the full enum value names.
_ENUM_PREFIXES = {
    "tls": "NETWORK_TLS_MODE_",
    "enforcement": "NETWORK_ENFORCEMENT_MODE_",
    "access": "NETWORK_ACCESS_PRESET_",
}


def load_policy_file(path: str) -> sandbox_pb2.SandboxPolicy:  # type: ignore[no-any-unimported]
    """Parse an OpenShell policy YAML (CLI schema) into the proto message."""
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # The CLI schema names the filesystem section ``filesystem_policy``; the
    # proto field is ``filesystem``.
    if "filesystem_policy" in raw:
        raw["filesystem"] = raw.pop("filesystem_policy")
    for rule_name, rule in (raw.get("network_policies") or {}).items():
        rule.setdefault("name", rule_name)
        for endpoint in rule.get("endpoints") or []:
            for field, prefix in _ENUM_PREFIXES.items():
                if isinstance(endpoint.get(field), str):
                    value = endpoint[field].upper().replace("-", "_")
                    endpoint[field] = f"{prefix}{value}"
        rule["binaries"] = [
            b if isinstance(b, dict) else {"path": b}
            for b in rule.get("binaries") or []
        ]
    policy = sandbox_pb2.SandboxPolicy()
    json_format.ParseDict(raw, policy)
    return policy


def build_policy(  # type: ignore[no-any-unimported]
    config: SandboxConfig, base: sandbox_pb2.SandboxPolicy | None = None
) -> sandbox_pb2.SandboxPolicy:
    """Build the per-sandbox policy from xagent's config and the base file."""
    policy = sandbox_pb2.SandboxPolicy()
    if base is not None:
        policy.CopyFrom(base)
    policy.version = policy.version or 1

    read_only = list(_BASE_READ_ONLY)
    read_write = list(_BASE_READ_WRITE)
    if config.working_dir:
        read_write.append(config.working_dir)
    for _host, guest, mode in config.volumes or []:
        (read_only if mode.lower() == "ro" else read_write).append(guest)

    fs = policy.filesystem
    fs.include_workdir = True
    fs.read_only.extend(p for p in read_only if p not in fs.read_only)
    fs.read_write.extend(p for p in read_write if p not in fs.read_write)

    if config.network_isolated:
        policy.ClearField("network_policies")
    return policy


def build_spec(  # type: ignore[no-any-unimported]
    template: SandboxTemplate,
    config: SandboxConfig,
    *,
    base_policy: sandbox_pb2.SandboxPolicy | None = None,
    providers: list[str] | None = None,
) -> openshell_pb2.SandboxSpec:
    spec = openshell_pb2.SandboxSpec()
    spec.template.image = template.image or DEFAULT_SANDBOX_IMAGE
    spec.template.labels[MANAGED_LABEL] = "true"
    if config.env:
        spec.environment.update(config.env)

    resources: dict[str, Any] = {"limits": {}}
    if config.cpus:
        resources["limits"]["cpu"] = str(config.cpus)
    if config.memory:
        resources["limits"]["memory"] = f"{config.memory}Mi"
    spec.template.resources.CopyFrom(_struct(resources))

    if config.volumes:
        mounts = [
            {
                "type": "bind",
                "source": host,
                "target": guest,
                "read_only": mode.lower() == "ro",
            }
            for host, guest, mode in config.volumes
        ]
        spec.template.driver_config.CopyFrom(_struct({"docker": {"mounts": mounts}}))

    if config.ports:
        logger.warning(
            "OpenShell backend ignores port mappings %s; use service exposure",
            config.ports,
        )

    spec.policy.CopyFrom(build_policy(config, base_policy))
    spec.providers.extend(providers or [])
    return spec


def _struct(value: dict[str, Any]) -> struct_pb2.Struct:  # type: ignore[no-any-unimported]
    s = struct_pb2.Struct()
    s.update(value)
    return s


def _stage_oversized_args(argv: list[str]) -> tuple[list[str], bytes | None]:
    """Route argv elements over the gateway's per-argument cap through stdin.

    The gateway rejects any single exec argument over ``_GATEWAY_ARG_LIMIT``
    bytes (the sandboxed tool runtime ships a larger base64 bundle in argv).
    Oversized values travel as one stdin payload instead: a POSIX ``sh`` reads
    it into a private temp file, slices each value back into a variable,
    removes the file, then ``exec``s the original command with the original
    argv. Small arguments ride along as ``sh`` positional parameters, so the
    script itself never carries caller data.

    Returns the argv to send and the stdin payload (None when nothing is
    oversized, in which case argv is returned unchanged).
    """
    encoded = [arg.encode("utf-8") for arg in argv]
    if all(len(value) <= _GATEWAY_ARG_LIMIT for value in encoded):
        return argv, None

    payload = bytearray()
    slices: list[str] = []
    refs: list[str] = []
    positional: list[str] = []
    fail = '|| { rm -f "$f"; exit 125; }'
    for index, value in enumerate(encoded):
        position = index + 1
        if len(value) <= _GATEWAY_ARG_LIMIT:
            positional.append(argv[index])
            refs.append(f'"${{{position}}}"')
            continue
        positional.append("")
        var = f"a{position}"
        # `; printf x` + `${var%x}` keeps trailing newlines that $(...) strips.
        slices.append(
            f'{var}=$(tail -c +{len(payload) + 1} "$f" | head -c {len(value)}; '
            f"printf x); {var}=${{{var}%x}}"
        )
        # The substitution's exit status is printf's, so a failed tail/head
        # would otherwise pass a short value: verify the byte count instead.
        slices.append(
            f'[ $(( $(printf %s "${var}" | wc -c) )) -eq {len(value)} ] {fail}'
        )
        refs.append(f'"${var}"')
        payload += value
    lines = [
        "f=$(mktemp) || exit 125",
        f'cat > "$f" {fail}',
        f'[ $(( $(wc -c < "$f") )) -eq {len(payload)} ] {fail}',
        *slices,
        'rm -f "$f"',
        "exec " + " ".join(refs),
    ]
    return ["sh", "-c", "\n".join(lines), "sh", *positional], bytes(payload)


def _truncate_utf8(text: str, cap: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= cap:
        return text, False
    return encoded[:cap].decode("utf-8", errors="ignore"), True


class OpenShellSandbox(Sandbox):
    """One OpenShell sandbox driven through the gateway's exec API."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        sandbox_name: str,
        os_name: str,
        client: SandboxClient,
        workspace: str,
        info: SandboxInfo,
        store: OpenShellStore,
    ) -> None:
        self._name = sandbox_name
        self._os_name = os_name
        self._client = client
        self._workspace = workspace
        self._info = info
        self._store = store

    @property
    def name(self) -> str:
        return self._name

    async def stop(self) -> None:
        await asyncio.to_thread(
            self._client.stop, self._os_name, workspace=self._workspace
        )
        self._store.update_info_state(self._name, "stopped")

    async def info(self) -> SandboxInfo:
        ref = await asyncio.to_thread(
            self._client.get, self._os_name, workspace=self._workspace
        )
        self._info.state = _phase_state(ref)
        return self._info

    async def _exec(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self._client.exec,
            self._os_name,
            argv,
            workspace=self._workspace,
            workdir=self._info.config.working_dir,
            env=env,
            stdin=stdin,
            no_login_shell=True,
            timeout_seconds=600,
        )

    async def exec(
        self,
        command: str,
        *args: str,
        env: dict[str, str] | None = None,
        max_output_bytes: int | None = None,
    ) -> ExecResult:
        argv, stdin = _stage_oversized_args([command, *args])
        res = await self._exec(argv, env=env, stdin=stdin)
        stdout, stderr, truncated = res.stdout, res.stderr, False
        if max_output_bytes is not None:
            # The SDK buffers the whole output before returning, so this trims
            # after the fact.
            stdout, out_t = _truncate_utf8(stdout, max_output_bytes)
            stderr, err_t = _truncate_utf8(stderr, max_output_bytes)
            truncated = out_t or err_t
        return ExecResult(
            exit_code=res.exit_code, stdout=stdout, stderr=stderr, truncated=truncated
        )

    async def run_code(
        self,
        code: str,
        code_type: CodeType = "python",
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        code = textwrap.dedent(code)
        if code_type == "python":
            return await self.exec("python", "-c", code, env=env)
        elif code_type == "javascript":
            return await self.exec("node", "-e", code, env=env)
        raise ValueError(f"Unsupported code type: {code_type}")

    # The SDK has no file-transfer call, so files move over exec stdin/stdout.

    async def upload_file(
        self, local_path: str, remote_path: str, overwrite: bool = False
    ) -> None:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        with open(local_path, "rb") as f:
            data = f.read()
        await self._write_bytes(data, remote_path, overwrite)

    async def _write_bytes(
        self, data: bytes, remote_path: str, overwrite: bool
    ) -> None:
        target = shlex.quote(remote_path)
        # Unique per call: concurrent uploads to one path must not append into
        # the same temp file and then move an interleaved result into place.
        tmp_path = f"{remote_path}.xagent-upload-{uuid.uuid4().hex[:12]}"
        tmp = shlex.quote(tmp_path)
        remote_dir = shlex.quote(posixpath.dirname(remote_path) or "/")
        guard = "" if overwrite else f"if [ -e {target} ]; then exit 100; fi; "
        prep = await self._exec(
            ["sh", "-c", f"{guard}mkdir -p {remote_dir} && : > {tmp}"]
        )
        if prep.exit_code == 100:
            raise FileExistsError(f"Remote file already exists: {remote_path}")
        if prep.exit_code != 0:
            raise RuntimeError(f"Failed to prepare {remote_path}: {prep.stderr}")
        for start in range(0, max(len(data), 1), _UPLOAD_CHUNK_BYTES):
            chunk = data[start : start + _UPLOAD_CHUNK_BYTES]
            res = await self._exec(["sh", "-c", f"cat >> {tmp}"], stdin=chunk)
            if res.exit_code != 0:
                await self._exec(["rm", "-f", tmp_path])
                raise RuntimeError(f"Failed to upload {remote_path}: {res.stderr}")
        res = await self._exec(["mv", "-f", tmp_path, remote_path])
        if res.exit_code != 0:
            await self._exec(["rm", "-f", tmp_path])
            raise RuntimeError(f"Failed to move file to {remote_path}: {res.stderr}")

    async def download_file(
        self, remote_path: str, local_path: str, overwrite: bool = False
    ) -> None:
        if not overwrite and os.path.exists(local_path):
            raise FileExistsError(f"Local file already exists: {local_path}")
        res = await self._exec(["base64", "-w0", remote_path])
        if res.exit_code != 0:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(base64.b64decode(res.stdout))

    async def write_file(
        self, content: str, remote_path: str, overwrite: bool = False
    ) -> None:
        await self._write_bytes(content.encode("utf-8"), remote_path, overwrite)

    async def read_file(self, remote_path: str) -> str:
        res = await self._exec(["cat", remote_path])
        if res.exit_code != 0:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")
        return str(res.stdout)


class OpenShellSandboxService(SandboxService):
    """Sandbox lifecycle on an OpenShell gateway."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        store: OpenShellStore,
        *,
        namespace: str,
        client: SandboxClient | None = None,
        workspace: str | None = None,
        policy_file: str | None = None,
        providers: list[str] | None = None,
    ) -> None:
        # Ownership boundary on a shared gateway, like the Docker backend's
        # namespace: names, labels and listing are all scoped to it.
        validate_sandbox_namespace(namespace)
        self._namespace = namespace
        self._client = client or SandboxClient.from_active_cluster(
            cluster=os.getenv("XAGENT_OPENSHELL_GATEWAY") or None, timeout=60.0
        )
        self._workspace: str = (
            workspace or os.getenv("XAGENT_OPENSHELL_WORKSPACE") or "default"
        )
        policy_file = policy_file or os.getenv("XAGENT_OPENSHELL_POLICY_FILE")
        self._base_policy = load_policy_file(policy_file) if policy_file else None
        if providers is None:
            raw = os.getenv("XAGENT_OPENSHELL_PROVIDERS", "")
            providers = [p.strip() for p in raw.split(",") if p.strip()]
        self._providers = providers
        self._store = store
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

    async def _lock(self, name: str) -> asyncio.Lock:
        async with self._locks_lock:
            return self._locks.setdefault(name, asyncio.Lock())

    async def _get_ref(self, os_name: str) -> Any | None:
        try:
            return await asyncio.to_thread(
                self._client.get, os_name, workspace=self._workspace
            )
        except grpc.RpcError as exc:
            if _is_not_found(exc):
                return None
            raise

    async def get_or_create(
        self,
        name: str,
        template: SandboxTemplate | None = None,
        config: SandboxConfig | None = None,
    ) -> OpenShellSandbox:
        if template is not None and template.type == "snapshot":
            raise NotImplementedError("Unsupported")

        os_name = openshell_name(name, self._namespace)
        async with await self._lock(name):
            ref = await self._get_ref(os_name)
            info = self._store.get_info(name)
            if ref is None:
                tpl = template or SandboxTemplate(
                    type="image", image=DEFAULT_SANDBOX_IMAGE
                )
                cfg = config or SandboxConfig()
                spec = build_spec(
                    tpl, cfg, base_policy=self._base_policy, providers=self._providers
                )
                ref = await asyncio.to_thread(
                    self._client.create,
                    workspace=self._workspace,
                    spec=spec,
                    name=os_name,
                    labels={
                        MANAGED_LABEL: "true",
                        NAMESPACE_LABEL: namespace_label_value(self._namespace),
                        **encode_name_labels(name),
                    },
                )
                info = SandboxInfo(name=name, state="running", template=tpl, config=cfg)
                self._store.add_info(name, info)
            elif _phase_state(ref) == "stopped":
                await asyncio.to_thread(
                    self._client.start, os_name, workspace=self._workspace
                )
            await asyncio.to_thread(
                self._client.wait_ready,
                os_name,
                workspace=self._workspace,
                timeout_seconds=300,
            )
            if info is None:
                # Store row lost: the gateway cannot give back env/volumes.
                info = SandboxInfo(
                    name=name,
                    state="running",
                    template=SandboxTemplate(type="image", image=None),
                    config=config or SandboxConfig(),
                )
            info.state = "running"
            self._store.update_info_state(name, "running")
            return OpenShellSandbox(
                name, os_name, self._client, self._workspace, info, self._store
            )

    async def list_sandboxes(self) -> list[SandboxInfo]:
        refs = await asyncio.to_thread(
            self._client.list_all,
            workspace=self._workspace,
            label_selector=(
                f"{MANAGED_LABEL}=true,"
                f"{NAMESPACE_LABEL}={namespace_label_value(self._namespace)}"
            ),
        )
        result: list[SandboxInfo] = []
        for ref in refs:
            name = decode_name_labels(ref.labels)
            # Report only sandboxes whose labels map back to their own gateway
            # name: SandboxManager drives probes, worker discovery and cleanup
            # from these names, and a raw ``xa-<hash>`` would be re-hashed
            # into a different sandbox.
            if name is None or openshell_name(name, self._namespace) != ref.name:
                logger.warning(
                    "Skipping OpenShell sandbox %s: its name labels do not map "
                    "back to it",
                    ref.name,
                )
                continue
            info = self._store.get_info(name) or SandboxInfo(
                name=name,
                state="unknown",
                template=SandboxTemplate(type="image", image=None),
                config=SandboxConfig(),
            )
            info.state = _phase_state(ref)
            result.append(info)
        return result

    async def delete(self, name: str) -> None:
        os_name = openshell_name(name, self._namespace)
        result = await asyncio.to_thread(
            self._client.delete, os_name, workspace=self._workspace, allow_missing=True
        )
        if result.sandbox_id:
            await asyncio.to_thread(
                self._client.wait_deleted,
                os_name,
                workspace=self._workspace,
                expected_sandbox_id=result.sandbox_id,
            )
        self._store.delete_info(name)
        async with self._locks_lock:
            self._locks.pop(name, None)

    async def supports_snapshots(self) -> bool:
        return False

    async def supports_runtime_spec(self) -> bool:
        return False

    async def create_snapshot(self, name: str, snapshot_id: str) -> SandboxSnapshot:
        raise NotImplementedError("Unsupported")

    async def list_snapshots(self) -> list[SandboxSnapshot]:
        raise NotImplementedError("Unsupported")

    async def delete_snapshot(self, snapshot_id: str) -> None:
        raise NotImplementedError("Unsupported")
