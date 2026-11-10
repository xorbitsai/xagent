"""Unit tests for the OpenShell sandbox backend (no gateway needed).

The SDK client is replaced by ``FakeSandboxClient``: an in-memory gateway
with a tiny shell emulator for the commands the backend's file transfer
issues, so every service and sandbox code path runs without OpenShell.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

# Skip only when the optional extra is absent. Any other import failure (e.g. a
# protobuf runtime older than the SDK's 6.31 gencode raising VersionError) is a
# broken environment and must fail collection rather than hide behind a skip.
pytest.importorskip("openshell", exc_type=ImportError)

import grpc  # noqa: E402
from openshell._proto import openshell_pb2, sandbox_pb2  # noqa: E402

from xagent.sandbox import openshell_sandbox as mod  # noqa: E402
from xagent.sandbox.base import (  # noqa: E402
    SandboxConfig,
    SandboxInfo,
    SandboxTemplate,
)

MANAGED_LABEL = mod.MANAGED_LABEL
NAME_LABEL = mod.NAME_LABEL
MemOpenShellStore = mod.MemOpenShellStore
OpenShellSandboxService = mod.OpenShellSandboxService
build_policy = mod.build_policy
build_spec = mod.build_spec
decode_name_label = mod.decode_name_label
encode_name_label = mod.encode_name_label
load_policy_file = mod.load_policy_file
openshell_name = mod.openshell_name

READY = openshell_pb2.SANDBOX_PHASE_READY
STOPPED = openshell_pb2.SANDBOX_PHASE_STOPPED


# ---------------------------------------------------------------------------
# Fake gateway
# ---------------------------------------------------------------------------


class _RpcError(grpc.RpcError):
    def __init__(self, status: grpc.StatusCode) -> None:
        super().__init__(status.name)
        self._status = status

    def code(self) -> grpc.StatusCode:
        return self._status


@dataclass
class _Box:
    name: str
    labels: dict[str, str]
    spec: Any
    phase: int = READY
    sandbox_id: str = "sb-1"
    files: dict[str, bytes] = field(default_factory=dict)

    @property
    def ref(self) -> SimpleNamespace:
        return SimpleNamespace(
            name=self.name,
            labels=dict(self.labels),
            status=SimpleNamespace(phase=self.phase),
        )


def _res(exit_code: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(exit_code=exit_code, stdout=stdout, stderr=stderr)


class FakeSandboxClient:
    """In-memory stand-in for ``openshell.SandboxClient``.

    Calls are recorded as ``(method, args, kwargs)``. ``exec`` runs a small
    emulator over each box's ``files`` dict; set ``exec_hook`` to intercept a
    command (return a result to short-circuit, or None to fall through).
    """

    def __init__(self) -> None:
        self.boxes: dict[str, _Box] = {}
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.get_error: grpc.RpcError | None = None
        self.create_delay = 0.0
        self.exec_hook: Callable[[list[str], bytes | None], Any] | None = None
        self._lock = threading.Lock()

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            self.calls.append((method, args, kwargs))

    def count(self, method: str) -> int:
        return sum(1 for m, _, _ in self.calls if m == method)

    def last(self, method: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for m, args, kwargs in reversed(self.calls):
            if m == method:
                return args, kwargs
        raise AssertionError(f"{method} was never called")

    def execs(self) -> list[dict[str, Any]]:
        return [
            dict(kwargs, argv=args[1]) for m, args, kwargs in self.calls if m == "exec"
        ]

    # -- lifecycle ---------------------------------------------------------

    def get(self, name: str, *, workspace: str) -> SimpleNamespace:
        self._record("get", name, workspace=workspace)
        if self.get_error is not None:
            raise self.get_error
        if name not in self.boxes:
            raise _RpcError(grpc.StatusCode.NOT_FOUND)
        return self.boxes[name].ref

    def create(
        self, *, workspace: str, spec: Any, name: str, labels: dict[str, str]
    ) -> SimpleNamespace:
        self._record("create", workspace=workspace, spec=spec, name=name, labels=labels)
        if self.create_delay:
            time.sleep(self.create_delay)
        box = _Box(name=name, labels=dict(labels), spec=spec)
        self.boxes[name] = box
        return box.ref

    def start(self, name: str, *, workspace: str) -> None:
        self._record("start", name, workspace=workspace)
        self.boxes[name].phase = READY

    def stop(self, name: str, *, workspace: str) -> None:
        self._record("stop", name, workspace=workspace)
        self.boxes[name].phase = STOPPED

    def wait_ready(self, name: str, *, workspace: str, timeout_seconds: int) -> None:
        self._record(
            "wait_ready", name, workspace=workspace, timeout_seconds=timeout_seconds
        )

    def list_all(self, *, workspace: str, label_selector: str) -> list[SimpleNamespace]:
        self._record("list_all", workspace=workspace, label_selector=label_selector)
        key, _, value = label_selector.partition("=")
        return [b.ref for b in self.boxes.values() if b.labels.get(key) == value]

    def delete(
        self, name: str, *, workspace: str, allow_missing: bool
    ) -> SimpleNamespace:
        self._record("delete", name, workspace=workspace, allow_missing=allow_missing)
        box = self.boxes.pop(name, None)
        return SimpleNamespace(sandbox_id=box.sandbox_id if box else "")

    def wait_deleted(
        self, name: str, *, workspace: str, expected_sandbox_id: str
    ) -> None:
        self._record(
            "wait_deleted",
            name,
            workspace=workspace,
            expected_sandbox_id=expected_sandbox_id,
        )

    # -- exec --------------------------------------------------------------

    def exec(self, name: str, argv: list[str], **kwargs: Any) -> SimpleNamespace:
        self._record("exec", name, list(argv), **kwargs)
        if self.exec_hook is not None:
            hooked = self.exec_hook(list(argv), kwargs.get("stdin"))
            if hooked is not None:
                return hooked
        return self._emulate(self.boxes[name].files, list(argv), kwargs.get("stdin"))

    @staticmethod
    def _emulate(
        fs: dict[str, bytes], argv: list[str], stdin: bytes | None
    ) -> SimpleNamespace:
        if argv[:2] == ["sh", "-c"]:
            script = argv[2]
            if script.startswith("cat >> "):
                (tmp,) = shlex.split(script[len("cat >> ") :])
                fs[tmp] = fs.get(tmp, b"") + (stdin or b"")
                return _res()
            if script.startswith("if [ -e "):
                guard, _, rest = script.partition("; then exit 100; fi; ")
                (target,) = shlex.split(guard[len("if [ -e ") : -len(" ]")])
                if target in fs:
                    return _res(100)
                script = rest
            _mkdir, _, truncate = script.partition(" && : > ")
            (tmp,) = shlex.split(truncate)
            fs[tmp] = b""
            return _res()
        if argv[0] == "mv":
            fs[argv[3]] = fs.pop(argv[2])
            return _res()
        if argv[0] == "rm":
            fs.pop(argv[2], None)
            return _res()
        if argv[0] == "base64":
            if argv[2] not in fs:
                return _res(1, stderr="No such file")
            return _res(stdout=base64.b64encode(fs[argv[2]]).decode())
        if argv[0] == "cat":
            if argv[1] not in fs:
                return _res(1, stderr="No such file")
            return _res(stdout=fs[argv[1]].decode())
        return _res(stdout=" ".join(argv))


@pytest.fixture
def client() -> FakeSandboxClient:
    return FakeSandboxClient()


@pytest.fixture
def store() -> MemOpenShellStore:
    return MemOpenShellStore()


@pytest.fixture
def service(
    client: FakeSandboxClient, store: MemOpenShellStore
) -> OpenShellSandboxService:
    return OpenShellSandboxService(store, client=client, workspace="ws", providers=[])


TEMPLATE = SandboxTemplate(type="image", image="img:1")
CONFIG = SandboxConfig(working_dir="/home/sandbox", env={"A": "1"})
NAME = "user::7:client-3"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_openshell_name_fits_gateway_limit_and_is_stable() -> None:
    name = "user::12345678:client-987654321"
    assert len(openshell_name(name)) <= 19
    assert openshell_name(name) == openshell_name(name)
    assert openshell_name(name) != openshell_name(name + "x")


@pytest.mark.parametrize("name", ["user::7", "user::7:client-3", "task::a_b-c"])
def test_name_label_round_trip(name: str) -> None:
    labels = encode_name_label(name)
    assert decode_name_label(labels[NAME_LABEL]) == name


def test_name_label_skipped_when_too_long() -> None:
    assert encode_name_label("user::" + "9" * 60) == {}


def test_decode_empty_label_is_none() -> None:
    assert decode_name_label("") is None


def test_load_policy_file_maps_cli_yaml_to_proto(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "policy.yaml"
    path.write_text(
        textwrap.dedent(
            """
            version: 1
            filesystem_policy:
              read_write: [/data]
            network_policies:
              pypi:
                endpoints:
                  - host: pypi.org
                    port: 443
                    protocol: rest
                    enforcement: enforce
                    access: read-only
                binaries:
                  - /usr/bin/python3.11
            """
        )
    )
    policy = load_policy_file(str(path))
    rule = policy.network_policies["pypi"]
    assert rule.name == "pypi"
    assert rule.binaries[0].path == "/usr/bin/python3.11"
    endpoint = rule.endpoints[0]
    assert endpoint.enforcement == sandbox_pb2.NETWORK_ENFORCEMENT_MODE_ENFORCE
    assert endpoint.access == sandbox_pb2.NETWORK_ACCESS_PRESET_READ_ONLY
    assert list(policy.filesystem.read_write) == ["/data"]


def test_load_policy_file_accepts_empty_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_policy_file(str(path)) == sandbox_pb2.SandboxPolicy()


def test_build_policy_network_isolated_drops_base_rules() -> None:
    base = sandbox_pb2.SandboxPolicy()
    base.network_policies["pypi"].name = "pypi"
    assert build_policy(SandboxConfig(network_isolated=False), base).network_policies
    assert not build_policy(SandboxConfig(network_isolated=True), base).network_policies


def test_build_policy_does_not_mutate_base() -> None:
    base = sandbox_pb2.SandboxPolicy()
    base.network_policies["pypi"].name = "pypi"
    build_policy(SandboxConfig(network_isolated=True, working_dir="/w"), base)
    assert "pypi" in base.network_policies
    assert "/w" not in base.filesystem.read_write


def test_build_policy_maps_volume_modes_to_landlock_paths() -> None:
    policy = build_policy(
        SandboxConfig(
            working_dir="/home/sandbox",
            volumes=[("/h/src", "/app/src", "ro"), ("/h/ws", "/ws", "RW")],
        )
    )
    assert "/app/src" in policy.filesystem.read_only
    assert "/ws" in policy.filesystem.read_write
    assert "/home/sandbox" in policy.filesystem.read_write


def test_build_policy_does_not_duplicate_paths_from_base() -> None:
    base = sandbox_pb2.SandboxPolicy()
    base.filesystem.read_write.append("/tmp")
    policy = build_policy(SandboxConfig(working_dir="/tmp"), base)
    assert list(policy.filesystem.read_write).count("/tmp") == 1


def test_build_spec_carries_image_mounts_and_limits() -> None:
    spec = build_spec(
        SandboxTemplate(type="image", image="img:1"),
        SandboxConfig(
            cpus=2, memory=1024, env={"A": "1"}, volumes=[("/h", "/g", "rw")]
        ),
        providers=["p1"],
    )
    assert spec.template.image == "img:1"
    assert spec.template.labels[MANAGED_LABEL] == "true"
    assert dict(spec.environment) == {"A": "1"}
    assert list(spec.providers) == ["p1"]
    limits = spec.template.resources["limits"]
    assert limits["cpu"] == "2" and limits["memory"] == "1024Mi"
    mount = spec.template.driver_config["docker"]["mounts"][0]
    assert mount["source"] == "/h" and mount["target"] == "/g"
    assert mount["read_only"] is False


def test_build_spec_defaults_image_and_skips_empty_sections() -> None:
    config = SandboxConfig()
    spec = build_spec(SandboxTemplate(type="image"), config)
    assert spec.template.image == mod.DEFAULT_SANDBOX_IMAGE
    assert not spec.environment
    # SandboxConfig validation guarantees cpus >= 1 and memory >= 128.
    limits = spec.template.resources["limits"]
    assert limits["cpu"] == str(config.cpus)
    assert limits["memory"] == f"{config.memory}Mi"
    assert not spec.template.HasField("driver_config")
    assert not spec.providers


def test_build_spec_warns_that_ports_are_ignored(caplog) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level("WARNING", logger=mod.__name__):
        build_spec(TEMPLATE, SandboxConfig(ports=[(8080, 80)]))
    assert "ignores port mappings" in caplog.text


@pytest.mark.parametrize(
    ("text", "cap", "expected"),
    [
        ("abc", 3, ("abc", False)),
        ("abcd", 3, ("abc", True)),
        # A multi-byte character cut in half is dropped, not mangled.
        ("a中b", 2, ("a", True)),
    ],
)
def test_truncate_utf8(text: str, cap: int, expected: tuple[str, bool]) -> None:
    assert mod._truncate_utf8(text, cap) == expected


def test_phase_state_maps_unknown_phase() -> None:
    assert (
        mod._phase_state(SimpleNamespace(status=SimpleNamespace(phase=READY)))
        == "running"
    )
    assert (
        mod._phase_state(SimpleNamespace(status=SimpleNamespace(phase=STOPPED)))
        == "stopped"
    )
    assert (
        mod._phase_state(SimpleNamespace(status=SimpleNamespace(phase=-1))) == "unknown"
    )
    assert mod._phase_state(object()) == "unknown"


def test_is_not_found_only_matches_grpc_not_found() -> None:
    assert mod._is_not_found(_RpcError(grpc.StatusCode.NOT_FOUND))
    assert not mod._is_not_found(_RpcError(grpc.StatusCode.UNAVAILABLE))
    assert not mod._is_not_found(KeyError("NOT_FOUND"))


# Echo the exact argv the command received, so a staged round trip can be
# compared byte for byte against what the caller asked for.
_ARGV_ECHO = "import json, sys; print(json.dumps(sys.argv[1:]))"


def _run_staged(argv: list[str], tmpdir) -> subprocess.CompletedProcess[bytes]:  # type: ignore[no-untyped-def]
    staged, stdin = mod._stage_oversized_args(argv)
    assert all(len(arg.encode()) <= mod._GATEWAY_ARG_LIMIT for arg in staged)
    return subprocess.run(  # noqa: S603 - fixed local interpreter, test-only
        staged,
        input=stdin,
        capture_output=True,
        env={**os.environ, "TMPDIR": str(tmpdir)},
        check=False,
    )


def test_stage_oversized_args_leaves_small_argv_alone() -> None:
    argv = ["python", "-c", "x" * mod._GATEWAY_ARG_LIMIT]
    assert mod._stage_oversized_args(argv) == (argv, None)


def test_stage_oversized_args_round_trips_through_real_sh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    big_ascii = base64.b64encode(os.urandom(40_000)).decode()
    # Multibyte characters, shell metacharacters and trailing newlines must
    # all survive: $(...) would otherwise strip the newlines.
    big_tricky = ("中文 $HOME `id` \"q\" 'q' \\ " * 3000) + "\n\n"
    argv = [
        sys.executable,
        "-c",
        _ARGV_ECHO,
        big_ascii,
        "small arg with 'quotes' and $VARS",
        big_tricky,
        *[f"p{i}" for i in range(12)],  # positions past $9 need ${N}
    ]
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()

    proc = _run_staged(argv, tmpdir)

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == argv[3:]
    assert list(tmpdir.iterdir()) == []  # the staging file is removed


def test_stage_oversized_args_preserves_exit_code(tmp_path) -> None:  # type: ignore[no-untyped-def]
    argv = [sys.executable, "-c", "import sys; sys.exit(7)", "y" * 50_000]
    assert _run_staged(argv, tmp_path).returncode == 7


async def test_exec_stages_oversized_args_over_stdin(service, client) -> None:  # type: ignore[no-untyped-def]
    seen: list[tuple[list[str], bytes | None]] = []
    client.exec_hook = lambda argv, stdin: seen.append((argv, stdin)) or _res(0)
    sb = await service.get_or_create(NAME, TEMPLATE, CONFIG)
    big = "b" * (mod._GATEWAY_ARG_LIMIT + 1)

    await sb.exec("python", "-c", "pass", big)

    argv, stdin = seen[-1]
    assert argv[:2] == ["sh", "-c"]
    assert argv[3:] == ["sh", "python", "-c", "pass", ""]
    assert stdin == big.encode()


# ---------------------------------------------------------------------------
# Service construction
# ---------------------------------------------------------------------------


def test_service_reads_environment(monkeypatch, tmp_path, client) -> None:  # type: ignore[no-untyped-def]
    policy = tmp_path / "p.yaml"
    policy.write_text(
        "network_policies:\n  x:\n    endpoints: [{host: a.b, port: 443}]\n"
    )
    monkeypatch.setenv("XAGENT_OPENSHELL_WORKSPACE", "team-a")
    monkeypatch.setenv("XAGENT_OPENSHELL_POLICY_FILE", str(policy))
    monkeypatch.setenv("XAGENT_OPENSHELL_PROVIDERS", " p1, ,p2 ")
    svc = OpenShellSandboxService(MemOpenShellStore(), client=client)
    assert svc._workspace == "team-a"
    assert svc._providers == ["p1", "p2"]
    assert svc._base_policy is not None and "x" in svc._base_policy.network_policies


def test_service_defaults_without_environment(monkeypatch, client) -> None:  # type: ignore[no-untyped-def]
    for var in (
        "XAGENT_OPENSHELL_WORKSPACE",
        "XAGENT_OPENSHELL_POLICY_FILE",
        "XAGENT_OPENSHELL_PROVIDERS",
    ):
        monkeypatch.delenv(var, raising=False)
    svc = OpenShellSandboxService(MemOpenShellStore(), client=client)
    assert svc._workspace == "default"
    assert svc._providers == []
    assert svc._base_policy is None


def test_service_builds_client_from_active_gateway(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, Any] = {}

    def fake_from_active_cluster(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "client"

    monkeypatch.setenv("XAGENT_OPENSHELL_GATEWAY", "gw-1")
    monkeypatch.setattr(
        mod.SandboxClient, "from_active_cluster", fake_from_active_cluster
    )
    svc = OpenShellSandboxService(MemOpenShellStore(), providers=[])
    assert svc._client == "client"
    assert seen["cluster"] == "gw-1"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_get_or_create_creates_new_sandbox(service, client, store) -> None:  # type: ignore[no-untyped-def]
    sb = await service.get_or_create(NAME, TEMPLATE, CONFIG)

    assert sb.name == NAME
    os_name = openshell_name(NAME)
    _, kwargs = client.last("create")
    assert kwargs["name"] == os_name
    assert kwargs["workspace"] == "ws"
    assert kwargs["labels"][MANAGED_LABEL] == "true"
    assert decode_name_label(kwargs["labels"][NAME_LABEL]) == NAME
    assert kwargs["spec"].template.image == "img:1"
    assert dict(kwargs["spec"].environment) == {"A": "1"}
    assert client.count("wait_ready") == 1
    info = store.get_info(NAME)
    assert info is not None and info.state == "running" and info.config is CONFIG


async def test_get_or_create_passes_base_policy_and_providers(
    client, store, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    policy = tmp_path / "p.yaml"
    policy.write_text(
        "network_policies:\n  x:\n    endpoints: [{host: a.b, port: 443}]\n"
    )
    svc = OpenShellSandboxService(
        store, client=client, workspace="ws", policy_file=str(policy), providers=["p1"]
    )
    await svc.get_or_create(NAME, TEMPLATE, CONFIG)
    spec = client.last("create")[1]["spec"]
    assert "x" in spec.policy.network_policies
    assert list(spec.providers) == ["p1"]


async def test_get_or_create_defaults_template_and_config(
    service, client, store
) -> None:  # type: ignore[no-untyped-def]
    await service.get_or_create(NAME)
    assert client.last("create")[1]["spec"].template.image == mod.DEFAULT_SANDBOX_IMAGE
    info = store.get_info(NAME)
    assert info is not None and info.config == SandboxConfig()


async def test_get_or_create_reuses_running_sandbox(service, client, store) -> None:  # type: ignore[no-untyped-def]
    await service.get_or_create(NAME, TEMPLATE, CONFIG)
    sb = await service.get_or_create(NAME)

    assert client.count("create") == 1
    assert client.count("start") == 0
    # The stored config (not the defaults) drives later execs.
    await sb.exec("true")
    assert client.execs()[-1]["workdir"] == "/home/sandbox"


async def test_get_or_create_starts_stopped_sandbox(service, client, store) -> None:  # type: ignore[no-untyped-def]
    sb = await service.get_or_create(NAME, TEMPLATE, CONFIG)
    await sb.stop()
    assert store.get_info(NAME).state == "stopped"  # type: ignore[union-attr]

    await service.get_or_create(NAME)
    assert client.count("start") == 1
    assert client.count("create") == 1
    assert store.get_info(NAME).state == "running"  # type: ignore[union-attr]


async def test_get_or_create_survives_lost_store_row(service, client, store) -> None:  # type: ignore[no-untyped-def]
    await service.get_or_create(NAME, TEMPLATE, CONFIG)
    store.delete_info(NAME)

    sb = await service.get_or_create(NAME)
    info = await sb.info()
    assert client.count("create") == 1
    # The gateway cannot give back env/volumes, so the config falls back.
    assert info.config == SandboxConfig()
    assert info.template.image is None


async def test_get_or_create_rejects_snapshot_template(service, client) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(NotImplementedError):
        await service.get_or_create(
            NAME, SandboxTemplate(type="snapshot", snapshot_id="s")
        )
    assert client.calls == []


async def test_get_or_create_propagates_non_not_found_errors(service, client) -> None:  # type: ignore[no-untyped-def]
    client.get_error = _RpcError(grpc.StatusCode.UNAVAILABLE)
    with pytest.raises(grpc.RpcError):
        await service.get_or_create(NAME, TEMPLATE, CONFIG)
    assert client.count("create") == 0


async def test_concurrent_get_or_create_creates_once(service, client) -> None:  # type: ignore[no-untyped-def]
    client.create_delay = 0.05
    boxes = await asyncio.gather(
        *(service.get_or_create(NAME, TEMPLATE, CONFIG) for _ in range(5))
    )
    assert client.count("create") == 1
    assert {b.name for b in boxes} == {NAME}


async def test_concurrent_get_or_create_different_names_do_not_serialize(
    service, client
) -> None:  # type: ignore[no-untyped-def]
    client.create_delay = 0.2
    started = time.perf_counter()
    await asyncio.gather(
        *(service.get_or_create(f"user::{i}", TEMPLATE, CONFIG) for i in range(4))
    )
    assert client.count("create") == 4
    assert time.perf_counter() - started < 0.6


async def test_list_sandboxes_maps_names_back(service, client, store) -> None:  # type: ignore[no-untyped-def]
    await service.get_or_create(NAME, TEMPLATE, CONFIG)
    sb2 = await service.get_or_create("user::8", TEMPLATE, CONFIG)
    await sb2.stop()
    # A managed sandbox without the name label (name too long to encode) and
    # without a store row still shows up under its gateway name.
    client.boxes["xa-orphan"] = _Box(
        name="xa-orphan", labels={MANAGED_LABEL: "true"}, spec=None
    )
    # Unmanaged sandboxes are filtered out by label.
    client.boxes["other"] = _Box(name="other", labels={}, spec=None)

    infos = {i.name: i for i in await service.list_sandboxes()}

    assert client.last("list_all")[1]["label_selector"] == f"{MANAGED_LABEL}=true"
    assert set(infos) == {NAME, "user::8", "xa-orphan"}
    assert infos[NAME].state == "running"
    assert infos[NAME].config is CONFIG
    assert infos["user::8"].state == "stopped"
    assert infos["xa-orphan"].config == SandboxConfig()


async def test_delete_waits_for_removal_and_clears_store(
    service, client, store
) -> None:  # type: ignore[no-untyped-def]
    await service.get_or_create(NAME, TEMPLATE, CONFIG)
    await service.delete(NAME)

    _, kwargs = client.last("delete")
    assert kwargs["allow_missing"] is True
    assert client.last("wait_deleted")[1]["expected_sandbox_id"] == "sb-1"
    assert store.get_info(NAME) is None
    assert NAME not in service._locks


async def test_delete_missing_sandbox_skips_wait(service, client, store) -> None:  # type: ignore[no-untyped-def]
    store.add_info(
        NAME, SandboxInfo(name=NAME, state="stopped", template=TEMPLATE, config=CONFIG)
    )
    await service.delete(NAME)
    assert client.count("wait_deleted") == 0
    assert store.get_info(NAME) is None


async def test_snapshot_and_runtime_spec_are_unsupported(service) -> None:  # type: ignore[no-untyped-def]
    assert await service.supports_snapshots() is False
    assert await service.supports_runtime_spec() is False
    with pytest.raises(NotImplementedError):
        await service.create_snapshot(NAME, "s")
    with pytest.raises(NotImplementedError):
        await service.list_snapshots()
    with pytest.raises(NotImplementedError):
        await service.delete_snapshot("s")


async def test_info_refreshes_state_from_gateway(service, client) -> None:  # type: ignore[no-untyped-def]
    sb = await service.get_or_create(NAME, TEMPLATE, CONFIG)
    client.boxes[openshell_name(NAME)].phase = STOPPED
    assert (await sb.info()).state == "stopped"


# ---------------------------------------------------------------------------
# Exec
# ---------------------------------------------------------------------------


async def test_exec_passes_workdir_env_and_flags(service, client) -> None:  # type: ignore[no-untyped-def]
    sb = await service.get_or_create(NAME, TEMPLATE, CONFIG)
    res = await sb.exec("echo", "a b", env={"X": "1"})

    call = client.execs()[-1]
    assert call["argv"] == ["echo", "a b"]
    assert call["workspace"] == "ws"
    assert call["workdir"] == "/home/sandbox"
    assert call["env"] == {"X": "1"}
    assert call["no_login_shell"] is True
    assert call["timeout_seconds"] == 600
    assert (res.exit_code, res.stdout, res.truncated) == (0, "echo a b", False)


async def test_exec_truncates_each_stream_after_the_fact(service, client) -> None:  # type: ignore[no-untyped-def]
    client.exec_hook = lambda argv, stdin: _res(3, stdout="x" * 10, stderr="中文")
    sb = await service.get_or_create(NAME, TEMPLATE, CONFIG)

    res = await sb.exec("noisy", max_output_bytes=4)
    assert (res.exit_code, res.stdout, res.stderr, res.truncated) == (
        3,
        "xxxx",
        "中",
        True,
    )

    res = await sb.exec("noisy", max_output_bytes=100)
    assert (res.stdout, res.truncated) == ("x" * 10, False)


async def test_run_code_dispatches_by_language(service, client) -> None:  # type: ignore[no-untyped-def]
    sb = await service.get_or_create(NAME, TEMPLATE, CONFIG)
    await sb.run_code("    print(1)\n    print(2)\n", env={"E": "1"})
    call = client.execs()[-1]
    assert call["argv"] == ["python", "-c", "print(1)\nprint(2)\n"]
    assert call["env"] == {"E": "1"}

    await sb.run_code("console.log(1)", "javascript")
    assert client.execs()[-1]["argv"] == ["node", "-e", "console.log(1)"]

    with pytest.raises(ValueError, match="Unsupported code type"):
        await sb.run_code("x", "ruby")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# File transfer
# ---------------------------------------------------------------------------


@pytest.fixture
async def sandbox(service):  # type: ignore[no-untyped-def]
    return await service.get_or_create(NAME, TEMPLATE, CONFIG)


def _fs(client: FakeSandboxClient) -> dict[str, bytes]:
    return client.boxes[openshell_name(NAME)].files


async def test_upload_round_trip_in_chunks(
    sandbox, client, tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(mod, "_UPLOAD_CHUNK_BYTES", 4)
    data = bytes(range(10))
    src = tmp_path / "src.bin"
    src.write_bytes(data)

    await sandbox.upload_file(str(src), "/w/dir with space/f.bin")

    chunks = [
        c["stdin"]
        for c in client.execs()
        if c["argv"][2:3] and "cat >>" in c["argv"][2]
    ]
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert _fs(client) == {"/w/dir with space/f.bin": data}

    dst = tmp_path / "out" / "back.bin"
    await sandbox.download_file("/w/dir with space/f.bin", str(dst))
    assert dst.read_bytes() == data


async def test_upload_empty_file_still_creates_it(sandbox, client, tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "empty"
    src.write_bytes(b"")
    await sandbox.upload_file(str(src), "/w/empty")
    assert _fs(client) == {"/w/empty": b""}


async def test_upload_missing_local_file(sandbox, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        await sandbox.upload_file(str(tmp_path / "nope"), "/w/x")


async def test_write_file_respects_overwrite_flag(sandbox, client) -> None:  # type: ignore[no-untyped-def]
    await sandbox.write_file("one", "/w/f.txt")
    with pytest.raises(FileExistsError):
        await sandbox.write_file("two", "/w/f.txt")
    assert _fs(client)["/w/f.txt"] == b"one"

    await sandbox.write_file("three", "/w/f.txt", overwrite=True)
    assert await sandbox.read_file("/w/f.txt") == "three"


async def test_upload_prepare_failure(sandbox, client) -> None:  # type: ignore[no-untyped-def]
    client.exec_hook = lambda argv, stdin: (
        _res(1, stderr="read-only") if ": >" in " ".join(argv) else None
    )
    with pytest.raises(RuntimeError, match="Failed to prepare"):
        await sandbox.write_file("x", "/etc/f")


async def test_upload_chunk_failure_removes_partial_file(sandbox, client) -> None:  # type: ignore[no-untyped-def]
    client.exec_hook = lambda argv, stdin: (
        _res(1, stderr="disk full") if "cat >>" in " ".join(argv) else None
    )
    with pytest.raises(RuntimeError, match="Failed to upload"):
        await sandbox.write_file("x", "/w/f")
    assert client.execs()[-1]["argv"] == ["rm", "-f", "/w/f.xagent-upload"]
    assert _fs(client) == {}


async def test_upload_move_failure(sandbox, client) -> None:  # type: ignore[no-untyped-def]
    client.exec_hook = (
        lambda argv, stdin: _res(1, stderr="busy") if argv[0] == "mv" else None
    )
    with pytest.raises(RuntimeError, match="Failed to move"):
        await sandbox.write_file("x", "/w/f")


async def test_download_refuses_existing_local_file(sandbox, tmp_path) -> None:  # type: ignore[no-untyped-def]
    dst = tmp_path / "exists"
    dst.write_text("keep")
    with pytest.raises(FileExistsError):
        await sandbox.download_file("/w/f", str(dst))
    assert dst.read_text() == "keep"


async def test_download_overwrites_when_asked(sandbox, client, tmp_path) -> None:  # type: ignore[no-untyped-def]
    await sandbox.write_file("new", "/w/f")
    dst = tmp_path / "exists"
    dst.write_text("old")
    await sandbox.download_file("/w/f", str(dst), overwrite=True)
    assert dst.read_text() == "new"


async def test_download_and_read_missing_remote_file(sandbox, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        await sandbox.download_file("/w/missing", str(tmp_path / "x"))
    with pytest.raises(FileNotFoundError):
        await sandbox.read_file("/w/missing")
