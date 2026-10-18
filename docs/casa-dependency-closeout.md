# CASA dependency remediation scope

The frozen Python graph includes compatible official security fixes and retains
dependencies without a compatible official remediation. A lockfile is not a
clean vulnerability scan or certification of a deployed image.

## Dependency policy

- Direct requirements, optional extras, sandbox groups and runtime sandbox
  installers carry reviewed security floors. Direct imports such as LangChain
  Core and Starlette have explicit requirements. The floor contract accepts
  higher minimums and exact audited pins; it is not an advisory database.
- `BaseTool` comes from `langchain_core.tools`. Unused top-level LangChain,
  CopilotKit and LangGraph-prebuilt requirements are absent; application
  features do not import them.
- JSON repair is pinned to `0.64.0` to stabilize tool-argument repair behavior.
  Updating that pin requires the malformed/streamed tool-argument regressions,
  not just a successful dependency resolution.
- OpenAI 2.x, Xinference 2.x and SQLAlchemy 2.0 contracts remain supported.
- Resolution requires wheels on Linux x86_64 and macOS ARM. BoxLite is bounded
  to `>=0.10.7,<0.11`; 0.10.6 lacks a Linux/Python 3.12 wheel.

### Larger version transitions

- Unstructured `0.27.28` is the reviewed security baseline, not a transitive
  resolver requirement or a claim about the smallest fix. The old 0.18 line
  predates the official [URL-partition SSRF fix in 0.24.0](https://github.com/Unstructured-IO/unstructured/security/advisories/GHSA-4mvj-m6j5-pmf7)
  and [HTML-output XSS fix in 0.24.1](https://github.com/Unstructured-IO/unstructured/security/advisories/GHSA-v5mq-3xhg-98m9).
  Its use still requires application-level input/output boundaries.
- Wrapt `2.5.1` satisfies Unstructured 0.27.28's `>=2.1.1,<3.0.0` requirement;
  the former 1.17 selection cannot satisfy that constraint. 2.5.1 is the locked
  selection, not the minimum version required by Unstructured.
- LangSmith `0.14.11` satisfies LangChain Core 1.6.12's `>=0.14.9,<1.0.0`
  requirement; the former 0.3 selection cannot satisfy it.

The exact dependency constraints can be reproduced from the published metadata:
[Unstructured 0.27.28](https://pypi.org/pypi/unstructured/0.27.28/json),
[LangChain Core 1.6.12](https://pypi.org/pypi/langchain-core/1.6.12/json).

## Lockfile scope

The lock is generated and checked by uv. Security upgrades include required
framework/parser/SDK dependency chains and platform wheel metadata; unrelated
tooling and SDK selections are not blanket-refreshed. The mypy hook has its
own pinned SDK environment and uses NumPy 1.26.4 for Python 3.11 stub parsing,
not the runtime's NumPy 2.4.6. That type-checker exception is not a runtime
dependency downgrade.

The declared floors cover the reviewed direct requirements, not a promise
that every future independent pip resolution has a clean transitive graph.
Use the frozen lock for the audited deployment graph; wheel consumers must
audit their resolved environment as well. Retained findings below still apply.

Linux/Python 3.12 frozen installation planning does not replace a Linux
runtime/image test. Frontend dependencies need their own audit; Python scans
do not cover them.

## Runtime acceptance and follow-ups

BoxLite 0.10 metadata APIs are asynchronous. Isolated guests use `NetworkSpec`
with inbound and outbound disabled, not host-side `SecurityOptions`.
Published ports plus isolated guest networking fail before VM creation.
Creation, reuse and metadata recovery have SDK contract tests, but those do
not prove native VM execution, file transfer or network isolation. Compare
the base and upgraded SDK with the same image digests on a working VM host;
registry/network failures are not compatibility evidence. Deployment requires
the native suite and an actual isolated-guest egress check to pass.

BoxLite honors the sandbox image's non-root `USER`. Guest file operations must
target an authorized writable directory or workspace mount, not `/root`.
Standalone callers overriding the image working directory (including the
shared `SandboxConfig` default `/home`) must provide a writable mount/directory
for relative writes. Do not add a root-user fallback to restore root-only paths.

MCP's `streamablehttp_client` remains a supported deprecated compatibility API
in 1.30. Its migration to `streamable_http_client` is a tracked follow-up:
preserve custom client-factory ownership, auth/headers, connect/read timeouts
and cancellation/close semantics in transport tests before removing it. It
is not a remaining security-floor exception.

## Retained findings

The full extras/groups audit on the local macOS Python 3.12 environment still
reports six Python package names. Advisory entries can contain duplicates;
they must not be counted as distinct exploitable vulnerabilities.

| Package | Remaining condition |
| --- | --- |
| chromadb 1.5.1 | The remaining reported advisories have no official fixed version. Preserve the original version. |
| nltk 3.10.3 | Earlier fixable findings are upgraded; the remaining model artifact path finding has no reported fixed version. |
| PyPDF2 3.0.1 | Preserve the existing declaration under the decision not to change dependencies without an official fix. |
| setuptools 75.9.1 | Official fixed versions exist, but deepdoc-lib 0.2.4 requires `>=75.2,<76`. A conventional compatible upgrade cannot select them. |
| torch 2.12.0 | Official fixed CPU releases exist. They require `setuptools>=77.0.3`, which conflicts with deepdoc-lib's hard upper bound. This is an upstream compatibility block, not absence of a Torch fix. |
| transformers 4.57.6 | Official fixes for some findings require 5.x and a newer huggingface-hub. deepdoc-lib requires `huggingface-hub<1`; Docling also has platform-specific exclusions. Other reported findings have no fixed version. |

The dependency resolver rejects both a Torch security-floor probe and a
Linux-only Transformers security-floor probe without overriding those upstream
bounds. Do not force an incompatible override, substitute a library, or claim
these findings fixed. They require a future compatible upstream release or a
separately approved maintained replacement/backport.

Primary package metadata:
[deepdoc-lib](https://pypi.org/pypi/deepdoc-lib/json),
[Torch 2.13](https://pypi.org/pypi/torch/2.13.0/json),
[official CPU wheels](https://download.pytorch.org/whl/cpu/torch/).

## Reproduction and acceptance

Use a separate Python 3.12 environment; do not upgrade an existing developer or
production virtual environment in place.

```sh
uv lock --check
uv sync --frozen --all-extras --all-groups --no-install-project
uv export --frozen --all-extras --all-groups --no-hashes --no-emit-project --format requirements-txt
```

Audit the exported requirements without ignoring advisory IDs. Exporting all
groups does not make a host-platform scan a Linux image scan: package markers,
system packages, and separately installed router/image dependencies must also
be scanned in the actual build. SaaS must pin the merged core revision and
regenerate its own lockfile; its separate dependency update does not embed an
unmerged core commit. Preserve the existing advisory evidence until deployment
and retesting justify a changed status.
