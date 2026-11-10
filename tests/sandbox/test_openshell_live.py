"""Live tests for the OpenShell backend against a real gateway.

Skipped unless ``XAGENT_OPENSHELL_LIVE=1``. Prerequisites:

- a running OpenShell gateway (``openshell status``) with the Docker driver
  and bind mounts enabled (``allow_driver_config`` / ``enable_bind_mounts``);
- the sandbox image (``XAGENT_OPENSHELL_LIVE_IMAGE``, default
  ``xprobe/xagent-sandbox:latest``) with ``/usr/bin/python3.11`` and node;
- outbound access to pypi.org from the gateway host.

Bind-mount sources live under ``<repo>/.poc-data`` rather than pytest's
tmp_path, because a VM-backed Docker (colima, Docker Desktop) only shares the
user's home directory with the daemon.

Credential checks additionally need a provider (see
``poc/openshell/httpbin-profile.yaml``) named by
``XAGENT_OPENSHELL_LIVE_PROVIDER`` whose credential value is in
``XAGENT_OPENSHELL_LIVE_SECRET``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import textwrap
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

if os.getenv("XAGENT_OPENSHELL_LIVE") != "1":
    pytest.skip("set XAGENT_OPENSHELL_LIVE=1 to run", allow_module_level=True)

import pytest_asyncio

from xagent.sandbox.base import SandboxConfig, SandboxTemplate
from xagent.sandbox.openshell_sandbox import MemOpenShellStore, OpenShellSandboxService

pytestmark = pytest.mark.asyncio(loop_scope="module")

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "poc" / "openshell" / "policy.yaml"
IMAGE = os.getenv("XAGENT_OPENSHELL_LIVE_IMAGE", "xprobe/xagent-sandbox:latest")
WORKDIR = "/home/sandbox"
RUN_ID = uuid.uuid4().hex[:8]
# Per-run namespace: also proves the gateway honours namespace scoping, since
# a leftover from another run (or deployment) must never show up here.
NAMESPACE = f"xagent-live-{RUN_ID}"


def _http_probe(url: str) -> str:
    return textwrap.dedent(
        f"""
        import urllib.request
        try:
            r = urllib.request.urlopen({url!r}, timeout=15)
            print("HTTP", r.status)
        except Exception as e:
            print("ERR", type(e).__name__, str(e)[:160])
        """
    )


def _node_probe(url: str) -> str:
    return (
        f"fetch({url!r}).then(r=>console.log('HTTP',r.status))"
        ".catch(e=>console.log('ERR',e.cause?e.cause.code||e.cause.message:e.message))"
    )


@pytest.fixture(scope="module")
def host_dir() -> Iterator[Path]:
    parent = ROOT / ".poc-data"
    parent.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="openshell-live-", dir=parent))
    (path / "from_host.txt").write_text("hello from host\n")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def service() -> AsyncIterator[OpenShellSandboxService]:
    svc = OpenShellSandboxService(
        MemOpenShellStore(), namespace=NAMESPACE, policy_file=str(POLICY), providers=[]
    )
    created: list[str] = []
    svc._live_created = created  # type: ignore[attr-defined]
    yield svc
    for name in created:
        await svc.delete(name)


async def _create(
    svc: OpenShellSandboxService, suffix: str, config: SandboxConfig
) -> Any:
    name = f"user::live-{RUN_ID}:{suffix}"
    svc._live_created.append(name)  # type: ignore[attr-defined]
    return await svc.get_or_create(
        name, SandboxTemplate(type="image", image=IMAGE), config
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sandbox(service: OpenShellSandboxService, host_dir: Path) -> Any:
    return await _create(
        service,
        "main",
        SandboxConfig(
            working_dir=WORKDIR,
            cpus=1,
            memory=1024,
            env={"XAGENT_LIVE": "1"},
            volumes=[
                (str(ROOT / "src"), "/app/src", "ro"),
                (str(host_dir), "/workspace", "rw"),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


async def test_runs_non_root_with_env_and_workdir(sandbox) -> None:  # type: ignore[no-untyped-def]
    res = await sandbox.exec("sh", "-c", 'id -u; pwd; echo "$XAGENT_LIVE"')
    uid, cwd, env = res.stdout.split()
    assert res.exit_code == 0
    assert uid != "0"
    assert (cwd, env) == (WORKDIR, "1")


async def test_no_capabilities_and_no_new_privs(sandbox) -> None:  # type: ignore[no-untyped-def]
    res = await sandbox.exec(
        "sh", "-c", "grep -E '^(CapEff|NoNewPrivs)' /proc/self/status"
    )
    status = dict(line.split(":\t") for line in res.stdout.strip().splitlines())
    assert int(status["CapEff"], 16) == 0
    assert status["NoNewPrivs"] == "1"


async def test_landlock_blocks_writes_outside_policy(sandbox) -> None:  # type: ignore[no-untyped-def]
    res = await sandbox.exec(
        "sh", "-c", "echo x > /etc/xagent-live 2>/dev/null; echo rc=$?"
    )
    assert "rc=0" not in res.stdout


async def test_run_code_python_and_javascript(sandbox) -> None:  # type: ignore[no-untyped-def]
    assert "42" in (await sandbox.run_code("print(6 * 7)")).stdout
    assert "42" in (await sandbox.run_code("console.log(6 * 7)", "javascript")).stdout


async def test_exec_latency_is_low(sandbox) -> None:  # type: ignore[no-untyped-def]
    started = time.perf_counter()
    for _ in range(10):
        await sandbox.exec("true")
    assert (time.perf_counter() - started) / 10 < 0.5


async def test_exec_argument_over_gateway_limit(sandbox) -> None:  # type: ignore[no-untyped-def]
    # The gateway caps each exec argument at 32 KiB; the backend stages larger
    # ones over stdin (the sandboxed tool runtime bundle is ~35 KB).
    big = "x" * 100_000 + "\n"
    code = "import sys, hashlib; print(len(sys.argv[1]), hashlib.sha256(sys.argv[1].encode()).hexdigest())"
    res = await sandbox.exec("python", "-c", code, big)
    expected = f"{len(big)} {hashlib.sha256(big.encode()).hexdigest()}"
    assert res.exit_code == 0, res.stderr
    assert res.stdout.strip() == expected
    leftovers = await sandbox.exec("sh", "-c", "ls /tmp | grep -c '^tmp\\.' || true")
    assert leftovers.stdout.strip() == "0"


# ---------------------------------------------------------------------------
# Mounts and file transfer
# ---------------------------------------------------------------------------


async def test_ro_mount_is_readable_but_not_writable(sandbox) -> None:  # type: ignore[no-untyped-def]
    listing = await sandbox.exec("ls", "/app/src/xagent/sandbox")
    assert "base.py" in listing.stdout
    res = await sandbox.exec("sh", "-c", "touch /app/src/x 2>/dev/null; echo rc=$?")
    assert "rc=0" not in res.stdout


async def test_rw_mount_is_shared_with_host(sandbox, host_dir) -> None:  # type: ignore[no-untyped-def]
    assert "hello from host" in await sandbox.read_file("/workspace/from_host.txt")
    await sandbox.write_file(
        "from sandbox\n", "/workspace/from_sandbox.txt", overwrite=True
    )
    assert (host_dir / "from_sandbox.txt").read_text() == "from sandbox\n"


async def test_large_file_round_trip(sandbox, tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Larger than one gRPC message (1 MiB), so it exercises chunking.
    data = os.urandom(3 * 1024 * 1024 + 123)
    src, dst = tmp_path / "big.bin", tmp_path / "back.bin"
    src.write_bytes(data)
    await sandbox.upload_file(str(src), f"{WORKDIR}/big.bin", overwrite=True)
    await sandbox.download_file(f"{WORKDIR}/big.bin", str(dst))
    assert dst.read_bytes() == data


async def test_upload_refuses_overwrite_by_default(sandbox) -> None:  # type: ignore[no-untyped-def]
    await sandbox.write_file("one", f"{WORKDIR}/once.txt", overwrite=True)
    with pytest.raises(FileExistsError):
        await sandbox.write_file("two", f"{WORKDIR}/once.txt")


# ---------------------------------------------------------------------------
# Egress policy
# ---------------------------------------------------------------------------


async def test_egress_allowed_for_listed_binary_and_host(sandbox) -> None:  # type: ignore[no-untyped-def]
    res = await sandbox.run_code(_http_probe("https://pypi.org/simple/pip/"))
    assert "HTTP 200" in res.stdout, res.stdout + res.stderr


async def test_egress_denied_for_unlisted_host(sandbox) -> None:  # type: ignore[no-untyped-def]
    res = await sandbox.run_code(_http_probe("https://example.com/"))
    assert "HTTP 200" not in res.stdout


async def test_egress_denied_for_unlisted_binary(sandbox) -> None:  # type: ignore[no-untyped-def]
    res = await sandbox.run_code(
        _node_probe("https://pypi.org/simple/pip/"), "javascript"
    )
    assert "HTTP 200" not in res.stdout


async def test_pip_user_install_through_allowed_egress(sandbox) -> None:  # type: ignore[no-untyped-def]
    res = await sandbox.exec(
        "sh",
        "-c",
        "pip install --user -q --no-cache-dir tomli-w >/dev/null 2>&1; "
        "python -c 'import tomli_w; print(\"ok\")'",
    )
    assert res.stdout.strip() == "ok", res.stdout + res.stderr


async def test_network_isolated_drops_all_egress(service) -> None:  # type: ignore[no-untyped-def]
    iso = await _create(
        service, "iso", SandboxConfig(working_dir=WORKDIR, network_isolated=True)
    )
    res = await iso.run_code(_http_probe("https://pypi.org/simple/pip/"))
    assert "HTTP 200" not in res.stdout


# ---------------------------------------------------------------------------
# Lifecycle (runs last: it stops and resumes the shared sandbox)
# ---------------------------------------------------------------------------


async def test_list_maps_names_back(service, sandbox) -> None:  # type: ignore[no-untyped-def]
    names = {i.name for i in await service.list_sandboxes()}
    assert sandbox.name in names


async def test_long_names_survive_the_listing(service) -> None:  # type: ignore[no-untyped-def]
    # Chrome sandboxes are named chrome-execution::<64-hex digest> (82 bytes),
    # which needs several label chunks; SandboxManager's durable probe and
    # worker discovery match on the listed name.
    name = f"chrome-execution::{RUN_ID}" + "a" * 56
    worker = f"{name}::worker::1"
    for n in (name, worker):
        service._live_created.append(n)  # type: ignore[attr-defined]
        await service.get_or_create(
            n,
            SandboxTemplate(type="image", image=IMAGE),
            SandboxConfig(working_dir=WORKDIR),
        )
    names = {i.name for i in await service.list_sandboxes()}
    assert {name, worker} <= names
    assert not any(n.startswith("xa-") for n in names)


async def test_namespaces_sharing_the_gateway_are_isolated(service) -> None:  # type: ignore[no-untyped-def]
    other = OpenShellSandboxService(
        MemOpenShellStore(),
        namespace=f"{NAMESPACE}-other",
        policy_file=str(POLICY),
        providers=[],
    )
    name = f"user::live-{RUN_ID}:shared"
    template = SandboxTemplate(type="image", image=IMAGE)
    config = SandboxConfig(working_dir=WORKDIR)
    service._live_created.append(name)  # type: ignore[attr-defined]
    mine = await service.get_or_create(name, template, config)
    try:
        theirs = await other.get_or_create(name, template, config)
        await mine.write_file("mine", f"{WORKDIR}/owner.txt", overwrite=True)
        # Same xagent name, different deployment: a different sandbox.
        assert (await theirs.exec("cat", f"{WORKDIR}/owner.txt")).exit_code != 0
        assert [i.name for i in await other.list_sandboxes()] == [name]
        assert name in {i.name for i in await service.list_sandboxes()}
    finally:
        await other.delete(name)
    # Deleting the other deployment's sandbox left ours alone.
    assert (await mine.read_file(f"{WORKDIR}/owner.txt")) == "mine"


async def test_stop_and_resume_keeps_writable_layer(service, sandbox) -> None:  # type: ignore[no-untyped-def]
    await sandbox.exec("sh", "-c", f"echo keep > {WORKDIR}/state.txt")
    await sandbox.stop()
    resumed = await service.get_or_create(sandbox.name)
    assert (await resumed.read_file(f"{WORKDIR}/state.txt")).strip() == "keep"


async def test_delete_removes_sandbox(service) -> None:  # type: ignore[no-untyped-def]
    gone = await _create(service, "gone", SandboxConfig(working_dir=WORKDIR))
    await service.delete(gone.name)
    assert gone.name not in {i.name for i in await service.list_sandboxes()}


# ---------------------------------------------------------------------------
# Real xagent tools through SandboxedToolWrapper
# ---------------------------------------------------------------------------


async def test_xagent_tools_run_through_wrapper(service, host_dir) -> None:  # type: ignore[no-untyped-def]
    from xagent.core.tools.adapters.vibe.command_executor import (
        CommandExecutorToolForBasic,
    )
    from xagent.core.tools.adapters.vibe.javascript_executor import (
        JavaScriptExecutorToolForBasic,
    )
    from xagent.core.tools.adapters.vibe.python_executor import (
        PythonExecutorToolForBasic,
    )
    from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
        SandboxedToolWrapper,
    )
    from xagent.core.workspace import TaskWorkspace

    base = host_dir / "uploads"
    base.mkdir()
    workspace = TaskWorkspace(id=f"live-{RUN_ID}", base_dir=str(base))
    sandbox = await _create(
        service,
        "tools",
        SandboxConfig(
            working_dir=WORKDIR,
            memory=1024,
            # No platform source mount: the wrapper ships its own minimal
            # runtime bundle with each call (see sandbox/tool_runtime/bundle.py).
            volumes=[(str(base), str(base), "rw")],
        ),
    )
    report = Path(workspace.resolve_path("report.csv"))
    py_code = textwrap.dedent(
        f"""
        import pandas as pd
        df = pd.DataFrame({{"x": [1, 2, 3], "y": [10, 20, 30]}})
        df.assign(z=df.x * df.y).to_csv({str(report)!r}, index=False)
        print("sum_z", int((df.x * df.y).sum()))
        """
    )

    py = await SandboxedToolWrapper(
        PythonExecutorToolForBasic(workspace), sandbox
    ).run_json_async({"code": py_code})
    js = await SandboxedToolWrapper(
        JavaScriptExecutorToolForBasic(workspace), sandbox
    ).run_json_async({"code": "console.log('js', [1,2,3].map(x => x * 2).join(','))"})
    cmd = await SandboxedToolWrapper(
        CommandExecutorToolForBasic(workspace), sandbox
    ).run_json_async({"command": "echo done"})

    assert "sum_z 140" in str(py)
    assert "js 2,4,6" in str(js)
    assert "done" in str(cmd)
    assert report.read_text().splitlines()[-1] == "3,30,90"


async def test_sandbox_manager_lease_runs_tools(service, host_dir) -> None:  # type: ignore[no-untyped-def]
    """The production route: SandboxManager's legacy (non-reconciling) path
    hands tools a lease provider, exactly as agent task execution does."""
    from xagent.core.tools.adapters.vibe.command_executor import (
        CommandExecutorToolForBasic,
    )
    from xagent.core.tools.adapters.vibe.python_executor import (
        PythonExecutorToolForBasic,
    )
    from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
        SandboxedToolWrapper,
    )
    from xagent.core.workspace import TaskWorkspace
    from xagent.sandbox.base import SandboxMountIntent
    from xagent.web.sandbox_manager import SandboxManager

    base = os.path.realpath(host_dir / "manager")
    os.makedirs(base)
    lifecycle_id = f"live-{RUN_ID}-mgr"
    workspace = TaskWorkspace(id=lifecycle_id, base_dir=base)
    manager = SandboxManager(service)
    try:
        provider = await manager.get_or_create_lease_provider(
            "user",
            lifecycle_id,
            mount_intent=SandboxMountIntent(mount_root=base),
            prepare_root=base,
        )
        py = await SandboxedToolWrapper(
            PythonExecutorToolForBasic(workspace), provider
        ).run_json_async(
            {"code": "open('out.txt', 'w').write('from sandbox'); print('ok')"}
        )
        cmd = await SandboxedToolWrapper(
            CommandExecutorToolForBasic(workspace), provider
        ).run_json_async({"command": "id -u"})
    finally:
        await manager.cleanup()
        await manager.delete_sandbox("user", lifecycle_id)

    assert py["success"], py
    assert "ok" in py["output"]
    assert cmd["success"], cmd
    assert cmd["output"].strip() != "0"  # never root inside the workload
    assert lifecycle_id not in str([s.name for s in await service.list_sandboxes()])


# ---------------------------------------------------------------------------
# Credential placeholders
# ---------------------------------------------------------------------------

_CRED_PROBE = """
import json, os, urllib.request
key = os.environ.get("HTTPBIN_API_KEY", "")
def get(url):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
    try:
        r = urllib.request.urlopen(req, timeout=20)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return type(e).__name__, ""
bound = get("https://httpbin.org/headers")
unbound = get("https://pypi.org/simple/pip/")
print(json.dumps({"env": key, "bound": bound, "unbound": unbound[0]}))
"""


async def test_provider_credentials_are_placeholders(service) -> None:  # type: ignore[no-untyped-def]
    import json

    provider = os.getenv("XAGENT_OPENSHELL_LIVE_PROVIDER")
    secret = os.getenv("XAGENT_OPENSHELL_LIVE_SECRET")
    if not provider or not secret:
        pytest.skip(
            "set XAGENT_OPENSHELL_LIVE_PROVIDER and XAGENT_OPENSHELL_LIVE_SECRET"
        )
    svc = OpenShellSandboxService(
        MemOpenShellStore(),
        namespace=NAMESPACE,
        policy_file=str(POLICY),
        providers=[provider],
    )
    name = f"user::live-{RUN_ID}:cred"
    sb = await svc.get_or_create(
        name,
        SandboxTemplate(type="image", image=IMAGE),
        SandboxConfig(working_dir=WORKDIR),
    )
    try:
        out = json.loads((await sb.run_code(_CRED_PROBE)).stdout)
    finally:
        await svc.delete(name)

    assert out["env"].startswith("openshell:resolve:")
    assert secret not in out["env"]
    # A placeholder sent anywhere but the bound host fails closed.
    assert out["unbound"] == 403
    status, body = out["bound"]
    if status != 200:
        pytest.skip(f"bound endpoint unreachable from the gateway: {status}")
    # httpbin echoes request headers: the gateway substituted the real key.
    assert secret in body
