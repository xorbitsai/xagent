# OpenShell sandbox backend POC

Goal: find out whether NVIDIA OpenShell (v0.1.7) can back xagent's
tool-execution sandbox in place of the Docker backend, and what
it buys us (egress allowlists, credential placeholders) versus what breaks.

Scope: a second `SandboxService` implementation on SandboxManager's legacy
(non-reconciling) route
(`get_or_create` / exec / file transfer / stop / delete). No snapshots and
no spec reconciliation (`supports_runtime_spec() == False`).

## What was built

| File | Purpose |
|---|---|
| `src/xagent/sandbox/openshell_sandbox.py` | `OpenShellSandboxService` / `OpenShellSandbox` over the `openshell` Python SDK |
| `src/xagent/web/sandbox_manager.py` | `SANDBOX_IMPLEMENTATION=openshell` wiring (fails closed like the other backends) |
| `src/xagent/web/sandbox_store.py` | `DBOpenShellStore` (same table, `sandbox_type="openshell"`) |
| `tests/sandbox/test_openshell_sandbox.py` | 55 unit tests over an in-memory fake gateway: helpers, lifecycle, concurrency, exec (incl. oversized-argument staging run through a real `sh`), chunked file transfer, error paths (no gateway). Run in CI via the `openshell` extra |
| `tests/sandbox/test_openshell_live.py` | 21 live tests: isolation, mounts, Landlock, file transfer, egress, lifecycle, oversized exec arguments, real xagent tools (directly and through `SandboxManager` lease providers), credential placeholders. Skipped unless `XAGENT_OPENSHELL_LIVE=1` |
| `tests/web/test_openshell_sandbox_store.py` | `DBOpenShellStore` round trip and isolation from Docker rows |
| `tests/web/test_sandbox_manager.py` | `SANDBOX_IMPLEMENTATION=openshell` dispatch and `_create_openshell_service` |
| `poc/openshell/policy.yaml` | base egress policy (pypi only, python binary only) |
| `poc/openshell/httpbin-profile.yaml` | provider profile used by the live credential test |

Config: `XAGENT_OPENSHELL_WORKSPACE`, `XAGENT_OPENSHELL_GATEWAY`,
`XAGENT_OPENSHELL_POLICY_FILE`, `XAGENT_OPENSHELL_PROVIDERS`.

## Results (live, macOS arm64 + colima Docker 29.5, kernel 6.8)

All 21 live tests pass (~75 s), including the credential checks, the
real tools, and the `SandboxManager` lease-provider route that agent tasks use.

| Area | Result |
|---|---|
| Create from `xprobe/xagent-sandbox` | ~1 s; runs as image `USER` (uid 1100), `net=none`, no caps, not privileged |
| exec latency | ~7-8 ms per call through the gateway |
| CPU / memory limits | applied (`NanoCpus`, `Memory` set on the workload container) |
| Bind mounts (`/app/src` ro, workspace rw) | work; host sees sandbox writes; ro is enforced |
| Landlock | writes outside the policy (e.g. `/etc`) fail |
| File transfer | 5 MiB upload/download round-trip identical in ~0.2 s (exec stdin/stdout, chunked) |
| Egress allowlist | python -> pypi.org 200; python -> example.com denied; **node -> pypi.org denied** (binary not listed); `pip install --user` works |
| `network_isolated=True` | all egress denied |
| stop / resume | `get_or_create` restarts a stopped sandbox; writable layer survives |
| Credential provider | env holds `openshell:resolve:env:...` only; bound host receives the real key; placeholder sent to an unbound host -> **403** (fail closed) |
| `SandboxedToolWrapper` e2e | python_executor (pandas), javascript_executor, command_executor all succeed, directly and through `SandboxManager.get_or_create_lease_provider`; output file lands on host via the rw mount |

## Gaps and gotchas found

1. **Default-deny egress changes semantics.** Docker backend's
   `network_isolated=False` means full internet. Here it means "whatever the
   policy file allows". Every tool that needs network (pip, npx/uvx MCP
   servers, web fetchers) must be enumerated by host *and* real binary path
   (`/usr/bin/python3.11`, not `/usr/local/bin/python`).
2. **Non-root only.** OpenShell rejects root. The Docker backend
   deliberately runs the sandbox as root. Anything relying on that (writing
   to `/home` - xagent's default `working_dir` - or system site-packages)
   breaks; POC uses `working_dir=/home/sandbox` and `pip --user`.
3. **Name limit 19 chars.** Sandbox names are hashed (`xa-<16 hex>`); the
   original xagent name rides in a base32 label so `list_sandboxes` can map
   back.
4. **No SDK file-transfer API.** Done over exec stdin/stdout. Each stdin chunk
   is one gRPC message capped at 1 MiB (the documented 4 MiB is the total),
   so chunks are 768 KiB. Output is buffered fully by the SDK, so
   `max_output_bytes` only trims after the fact.
5. **No snapshots.** `supports_snapshots() == False`; snapshot templates are
   rejected. The Docker backend's commit-to-template flow has no equivalent.
6. **No spec reconciliation.** The SandboxManager reconcile/fingerprint path
   (`inspect` / `create` / `start_existing` / `stop_existing`) is not
   implemented; the gateway cannot give back env or mounts, so the store row
   is the only record of them.
7. **Bind mounts are an operator opt-in** (`allow_driver_config`,
   `enable_bind_mounts`, resource admission off) and OpenShell's docs warn
   they bypass workspace isolation. Named volumes are the supported path.
8. **Echoing endpoints leak credentials.** A bound host that reflects
   request headers (httpbin `/headers`) hands the real key back to the
   sandbox. Bind credentials only to endpoints that never echo them.
9. **Credential rewrite is HTTP-only.** SSH tools / DB connections get no
   placeholder protection.
10. **Ports** are ignored (OpenShell uses service exposure instead).
11. **SDK is synchronous**; every call goes through `asyncio.to_thread`.
12. **Pre-1.0 churn.** `workspace=` became mandatory in 0.1, the managed
    inference route was removed; pin SDK and gateway to the same release.
13. **Per-argument exec cap of 32 KiB.** The gateway rejects any single
    exec argument over 32768 bytes, and the sandboxed tool runtime ships a
    ~35 KB base64 bundle in argv. `OpenShellSandbox.exec` stages oversized
    arguments over stdin into a private temp file, rebuilds them in `sh`,
    deletes the file, then `exec`s the original argv (still bounded by the
    kernel's 128 KiB per-argument limit, like the other backends).
14. **protobuf 6 required.** openshell 0.1.x ships protobuf 6.31 gencode but
    only declares `protobuf>=4.25`; the `openshell` extra states the real
    floor. On an older runtime the import raises `VersionError`: the factory
    then fails closed (`RuntimeError`) and the unit tests fail collection
    instead of skipping.

## Local gateway setup used (colima)

Docker Desktop is the documented macOS runtime. With colima the supervisor
container cannot reach a loopback-only gateway in the default host network
mode; this `~/.config/openshell/gateway.toml` works:

```toml
[openshell]
version = 2

[openshell.gateway]
compute_driver = "docker"

[openshell.drivers.docker]
socket_path = "/Users/<you>/.colima/default/docker.sock"
supervisor_network_mode = "bridge"
host_gateway_ip = "192.168.5.2"          # Lima host address, forwards to host loopback
grpc_endpoint = "https://host.docker.internal:17670"  # in the gateway cert SANs
allow_driver_config = true
enable_bind_mounts = true

[openshell.drivers.docker.resource_admission]
enabled = false
```

Telemetry off: `OPENSHELL_TELEMETRY_ENABLED=false` in
`~/.config/openshell/gateway.env`. Restart with `brew services restart openshell`.

## Running

The SDK is the `openshell` extra (included in `--all-extras`, as CI installs):

```bash
uv sync --all-extras
uv run pytest tests/sandbox/test_openshell_sandbox.py \
  tests/web/test_openshell_sandbox_store.py tests/web/test_sandbox_manager.py

# Live, against the local gateway. The credential test also needs the provider:
#   openshell profile import -f poc/openshell/httpbin-profile.yaml
#   openshell provider create --name poc-httpbin --type poc-httpbin \
#       --credential HTTPBIN_API_KEY=<any test value>
XAGENT_OPENSHELL_LIVE=1 \
XAGENT_OPENSHELL_LIVE_PROVIDER=poc-httpbin \
XAGENT_OPENSHELL_LIVE_SECRET=<same test value> \
  uv run pytest tests/sandbox/test_openshell_live.py -v
```

## Takeaway

The backend swap itself is cheap: ~600 lines, and xagent's real tool path
runs unchanged. The value is real (per-binary egress allowlist, fail-closed
credential placeholders, non-root + Landlock by default). The cost is
operational: egress must be enumerated per tool, root-dependent behavior must
go, snapshots and the reconcile lifecycle need replacements, and the project
is still pre-1.0.
