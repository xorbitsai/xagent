# CASA dependency remediation scope

This round upgrades the compatible official fixes in the Python lockfile and
retains dependencies without a compatible official remediation. It does not
claim a clean vulnerability scan or certify a deployed image.

## Changes

- Upgrade the resolved FastAPI/Starlette, MCP, LangChain, document-processing,
  cryptography and other affected dependency chains. Raise the MCP,
  LangChain OpenAI and JSON repair floors to their selected security fixes.
- Remove unused test-only CopilotKit and LangGraph prebuilt declarations that
  constrained the web framework and LangChain upgrades. Application features
  do not import either package.
- Import `BaseTool` from `langchain_core.tools` for LangChain 1.x compatibility.
- Keep the supported OpenAI 2.x and Xinference 2.x client contracts, and retain
  SQLAlchemy 2.0.47 rather than introducing an unrelated database major change.
- Align the type-checking MCP dependency with the runtime security floor.
- Require wheel availability on Linux x86_64 and macOS ARM during resolution.
  BoxLite 0.10.6 lacks a Linux/Python 3.12 wheel; select the complete official
  0.10.7 release and check the frozen Linux installation plan too.

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
