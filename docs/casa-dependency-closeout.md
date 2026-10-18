# CASA dependency remediation scope

This round upgrades the compatible official fixes in the Python lockfile and
retains dependencies without a compatible official remediation. It does not
claim a clean vulnerability scan or certify a deployed image.

## Changes

- Upgrade the resolved FastAPI/Starlette, MCP, LangChain, document-processing,
  cryptography and other affected dependency chains. Raise the declared
  runtime, optional-extra and sandbox/test-group security floors as well, so
  wheel metadata does not continue advertising the old vulnerable direct
  versions. Declare the directly imported LangChain Core and Starlette with
  their tested floors instead of relying on permissive transitive bounds.
- Remove unused test-only CopilotKit and LangGraph prebuilt declarations that
  constrained the web framework and LangChain upgrades. Application features
  do not import either package.
- Import `BaseTool` directly from `langchain_core.tools`; remove the unused
  top-level `langchain` dependency and the LangGraph runtime chain it pulled in.
  This import is a dependency cleanup, not a claim that LangChain 1.x stopped
  exporting `BaseTool`.
- Keep the supported OpenAI 2.x and Xinference 2.x client contracts, and retain
  SQLAlchemy 2.0.47 rather than introducing an unrelated database major change.
- Align the type-checking MCP dependency with the runtime security floor.
- Require wheel availability on Linux x86_64 and macOS ARM during resolution.
  BoxLite 0.10.6 lacks a Linux/Python 3.12 wheel; select the complete official
  0.10.7 release and check the frozen Linux installation plan too.
- Support the BoxLite 0.10 async `SimpleBox.info()` / native `Box.info()` API.
  Use `NetworkSpec` with inbound and outbound networking disabled for isolated
  guests, not `SecurityOptions.network_enabled` (a host-side sandbox setting).
  Bound BoxLite to `>=0.10.7,<0.11` and cover creation, reuse, metadata recovery
  and native network-option construction with tests that do not require KVM.
  These SDK contract tests do not replace a real KVM/VM isolation test.
- Requests' floor follows the patched Docling Core requirement, not an adapter
  introduced by another PR. Websockets remains at the original locked 16.0;
  the temporary broad-refresh downgrade to 15.0.1 is not part of this graph.

## Lockfile scope

The initial broad resolution changed 235 existing package names and produced
6,285 added / 4,132 removed lockfile lines. The narrowed resolution starts from
the original committed lock and targets the 45 security-affected packages
that were upgraded in that initial candidate, retaining their previously
audited fixed versions. Other original selections remain preferred unless the
patched graph requires a compatible change. This is not a blanket upgrade.

After removing the unused LangChain/LangGraph chain, the narrowed graph changes
64 existing package names and produces 1,969 added / 1,360 removed lockfile
lines. Mypy, Ruff, pre-commit, Google GenAI, boto3 and
SQLAlchemy retain their original locked versions. The remaining changes also
include required framework/parser/SDK dependency chains and platform wheel
metadata; the lock is still generated and checked by uv, not edited by hand.
These are package-name and generated-line counts, not vulnerability counts or
a claim that every selected security release is the oldest possible fix.

The four reviewed SDK pins in the mypy hook now match the narrowed runtime:
Slack SDK 3.43.0, Google GenAI 1.66.0, ElevenLabs 2.56.0 and Google Auth 2.48.0.
The hook deliberately retains NumPy 1.26.4 for Python 3.11 stub parsing; it is
not the runtime's NumPy 2.4.6. That existing type-checker exception is not a
runtime dependency downgrade.

The declared floors cover the reviewed direct requirements, not a promise
that every future independent pip resolution has a clean transitive graph.
Use the frozen lock for the audited deployment graph; wheel consumers must
audit their resolved environment as well. Retained findings below still apply.

The all-extras/groups Python 3.12 audit retains exactly the same six affected
package names and advisory IDs as the broader candidate, without advisory
ignores. Linux/Python 3.12 frozen installation planning also passes; a dry run
does not replace a Linux runtime/image test. Each changed graph still needs
the normal CI and deployed acceptance checks below.

The separate frontend dependency work retains the existing braces chain:
there is no official patched release in the reviewed advisory. It does not
replace the library, add a local backport, delete functionality or suppress
audit output.

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
