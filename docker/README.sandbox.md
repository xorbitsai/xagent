# Xagent Sandbox

Sandbox runtime image for [Xagent](https://github.com/xorbitsai/xagent), an open-source framework for building and running AI agents. With sandboxing enabled, Xagent delegates untrusted work — generated Python and JavaScript, shell commands, `npx`/`uvx` MCP servers — to a sandbox built from this image rather than running it in the backend process.

A sandbox is a **named, stateful workspace**, not a throwaway container. Xagent creates one on demand, keeps it alive, and execs into it for each subsequent tool call. Stopping a sandbox preserves its filesystem; the next request resumes it. Xagent can also commit a sandbox's filesystem to a snapshot and use that snapshot — rather than this image — as the template for a new one.

This image is that starting template. It has no service of its own: its `CMD` is a plain `bash`, which the Docker backend replaces with a long-running idle process.

## What's inside

- **Node.js 22** (`node:22-slim` base) and **Python 3.11**, both on `PATH` as `node` and `python`
- **`uv` / `uvx`**, for sandboxed `uvx` MCP server connections
- A deliberately small, lockfile-pinned Python set: `pydantic`, `pydantic-settings`, `cloudpickle`, `mcp`, `pandas`, `numpy`, `matplotlib`, `openpyxl`, `python-docx`, `fsspec`
- `ca-certificates`, `tzdata`, `netbase`, and `openssh-client`
- user-site `pip install` support, so tools can pull their own declared dependencies without writing system packages

The Python set comes only from the dedicated `sandbox` dependency group in Xagent's `pyproject.toml`, exported from `uv.lock` at build time. It intentionally does not inherit the backend image's much larger dependency set, and the build fails if any supported package is missing from the result.

## Isolation

Sandboxing is opt-in: Xagent runs eligible tool calls in the backend process unless `SANDBOX_ENABLED=true`. What follows applies to work that does reach a sandbox.

The isolation comes from how the sandbox is run, not from this image. Xagent's Docker backend applies `no-new-privileges`, drops all Linux capabilities, and enforces CPU, memory, and PID limits. Docker does not itself filter private/metadata destinations when networking is enabled, so deployments using Docker must enforce public-only egress with host or network firewall policy. The backend stamps this policy into sandbox desired-state matching, so an existing sandbox created under a different profile is rebuilt before reuse. See [docker/README.md](https://github.com/xorbitsai/xagent/blob/main/docker/README.md) for deployment details.

The image defines an unprivileged `sandbox` user (uid 1100, gid 1010). The
Docker backend enforces uid 1100/gid 1010. Replacement images must provide that identity, a writable
`/home/sandbox` working directory, and writable cache directories.

### File upload wire contract

The Docker backend sends `upload_file` and `write_file` through Docker's
archive endpoint using a versioned, single-file PAX archive. The wire stream
contains exactly one local PAX header followed by one regular-file member and
zero-block padding. The PAX header always contains
`Xagent.archive.version=1`; `path` is the only other permitted key and appears
only when tar's legacy name field cannot represent the UTF-8 basename. Global
PAX headers, directories, links, devices, sparse files, multiple members, and
path components are never emitted. The local PAX header is named
`././@PaxHeader`, uses type `x`, and has mode, uid, gid, and mtime set to zero;
the regular-file member also has mtime zero and a size below `8**11` bytes.
Basenames must be valid UTF-8 and at most 255 bytes; empty names, `.`, `..`,
NUL, and names containing path components are rejected.

This deliberately narrow format is the upload boundary consumed by the
restricted Docker broker maintained in
[`xagent-saas`](https://github.com/xorbitsai/xagent-saas/blob/main/docs/sandbox-broker.md).
Change it only together with that parser and its real-Docker upload smoke test.

## Platform source isolation

Task sandboxes do not mount the backend `src/` or `tests/` trees. Each call
receives a small, fixed execution runtime containing only the shell, Python,
JavaScript, stdio MCP transport and Chrome daemon implementations. Authorization, connector
bindings, application configuration and file registration stay in the backend.
The guest receives JSON arguments, never pickled backend application objects.
Custom `@sandbox_config` tools that require arbitrary application imports are
not supported by this runtime. They must be adapted to an execution-only
protocol; unsupported tools fail rather than falling back to backend execution.

Custom volume mounts that expose known backend source, tests, or Git metadata
and object stores are rejected, including parent paths and aliases resolvable
in the backend's filesystem. In sibling-Docker deployments, operators must
audit host-only aliases and Git storage paths that the backend cannot inspect;
see [Sandbox Runtime Overlay](README.md#sandbox-runtime-overlay).
Do not install the full Xagent package in a custom sandbox image: execution
refuses an environment where that package or the former `/app/src` and
`/app/tests` directories are visible.

On upgrade, drain running work and restart the backend. Docker reconciles the
changed mount spec when resolving a sandbox. Managed bind-mounted task
files survive recreation, but the disposable writable layer does not.
Do not restore old snapshots without auditing them for copied platform source.
Source copies already made into task files or disclosed to users need separate
incident handling; changing mounts cannot erase them.

The obsolete `XAGENT_SANDBOX_ALLOW_LOCAL_FALLBACK_ON_CAPACITY` flag no longer
permits task execution on the backend. When sandbox-required tools are selected,
outages and capacity exhaustion fail their execution for all entry points,
including ordinary chat.

## Tags

- `latest` — the most recently published build
- `X.Y.Z` — published by pushing the matching Git tag, though a manual run can publish an arbitrary tag; see [Xagent releases](https://github.com/xorbitsai/xagent/releases)

Built for `linux/amd64` and `linux/arm64`.

Pin an explicit version in production; Xagent's sandbox Compose overlays pin one. Changing that pin is not a free rollback: Xagent reconciles running sandboxes against the new image spec, stopping, deleting and recreating them. Bind-mounted workspace and upload data survives; the container's writable layer — `/tmp`, `$HOME`, packages a tool installed at run time — does not. Drain or back up that state before switching tags.

## Usage

Xagent selects the image through the `SANDBOX_IMAGE` environment variable, which defaults to `xprobe/xagent-sandbox:latest`:

```yaml
environment:
  - SANDBOX_IMAGE=xprobe/xagent-sandbox:latest
```

To inspect the image directly:

```bash
docker run --rm -it xprobe/xagent-sandbox:latest bash
```

## Building a custom sandbox image

A replacement image has to stay runtime-compatible with [`docker/Dockerfile.sandbox`](https://github.com/xorbitsai/xagent/blob/main/docker/Dockerfile.sandbox), which is the easiest starting point. Every sandbox needs, on `PATH`:

- `python` and `node` — tool code runs as `python -c ...` and `node -e ...`
- `pip` — tools install their declared dependencies after the container starts
- `cat`, `rm`, `mkdir`, `/bin/sh`, and a writable `/tmp` — staging input, reading results back, cleanup
- `tail` — the Docker backend replaces the image's `CMD` with `tail -f /dev/null` to hold the container open

The image must define uid 1100 and gid 1010, provide a writable home directory
at `/home/sandbox`, and allow that identity to use Python's user site-packages.
The Docker backend enforces this identity even if the image declares another
`USER`.

Only if you use the matching feature:

- `npx` — sandboxed `npx` MCP servers
- `uvx` — sandboxed `uvx` MCP servers; Xagent no longer installs `uv` dynamically

## Links

- [Source and documentation](https://github.com/xorbitsai/xagent)
- [Dockerfile](https://github.com/xorbitsai/xagent/blob/main/docker/Dockerfile.sandbox)
- [Issue tracker](https://github.com/xorbitsai/xagent/issues)
