# Xagent Sandbox

Sandbox runtime image for [Xagent](https://github.com/xorbitsai/xagent), an open-source framework for building and running AI agents. With sandboxing enabled, Xagent delegates untrusted work — generated Python and JavaScript, shell commands, `npx`/`uvx` MCP servers — to a sandbox built from this image rather than running it in the backend process.

A sandbox is a **named, stateful workspace**, not a throwaway container. Xagent creates one on demand, keeps it alive, and execs into it for each subsequent tool call. Stopping a sandbox preserves its filesystem; the next request resumes it. On the Docker backend, Xagent can also commit a sandbox's filesystem to a snapshot and use that snapshot — rather than this image — as the template for a new one; the Boxlite backend does not support snapshots.

This image is that starting template. It has no service of its own: its `CMD` is a plain `bash`, which the Docker backend replaces with a long-running idle process.

## What's inside

- **Node.js 22** (`node:22-slim` base) and **Python 3.11**, both on `PATH` as `node` and `python`
- **`uv` / `uvx`**, for sandboxed `uvx` MCP server connections
- A deliberately small, lockfile-pinned Python set: `pydantic`, `pydantic-settings`, `cloudpickle`, `mcp`, `pandas`, `numpy`, `matplotlib`, `openpyxl`, `python-docx`, `fsspec`
- `ca-certificates`, `tzdata`, `netbase`, and `openssh-client`
- `pip install` permitted without `--break-system-packages`, so tools can pull their own declared dependencies after the container starts

The Python set comes only from the dedicated `sandbox` dependency group in Xagent's `pyproject.toml`, exported from `uv.lock` at build time. It intentionally does not inherit the backend image's much larger dependency set, and the build fails if any supported package is missing from the result.

## Isolation

Sandboxing is opt-in: Xagent runs tool calls in the backend process unless `SANDBOX_ENABLED=true`, and falls back to the backend when no sandbox backend is available or a tool cannot be wrapped. What follows applies to work that does reach a sandbox.

The isolation comes from how the sandbox is run, not from this image. Xagent's Docker backend applies `no-new-privileges`, CPU and memory limits, and optional network isolation; its Boxlite backend runs the sandbox inside a KVM microVM. See [docker/README.md](https://github.com/xorbitsai/xagent/blob/main/docker/README.md) for the two modes and their trade-offs.

The image itself defines an unprivileged `sandbox` user (uid 1100, gid 1010) as its default. Note that Xagent's Docker backend deliberately overrides this and runs the container as root, to match the file access behavior of the Boxlite backend.

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
- `test`, `cp`, `mv`, and a writable `/var/tmp` — the Boxlite backend stages every file transfer there before moving it into place, because `/tmp` is a tmpfs mount it cannot copy into

Only if you use the matching feature:

- `npx` — sandboxed `npx` MCP servers
- `uvx` — sandboxed `uvx` MCP servers; Xagent no longer installs `uv` dynamically

## Links

- [Source and documentation](https://github.com/xorbitsai/xagent)
- [Dockerfile](https://github.com/xorbitsai/xagent/blob/main/docker/Dockerfile.sandbox)
- [Issue tracker](https://github.com/xorbitsai/xagent/issues)
