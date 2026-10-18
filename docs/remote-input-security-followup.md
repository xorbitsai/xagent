# Caller-supplied media inputs (CASA 15 follow-up)

This change extends the provider-output download protection in PR #3183 to two
additional server-side input downloads: remote JSON in speech batches and
remote reference images in the Gemini image-editing model. It does not claim
that every outbound request is now isolated.

## Enforced boundaries

`download_public_input` uses the existing shared HTTP downloader to validate
all resolved addresses and connect to the checked IP, preserving logical Host
and TLS SNI/certificate checks. Each redirect is independently validated and
the chain is bounded. Literal, mixed-DNS, metadata, private, non-global,
userinfo, non-HTTP(S), and port-zero targets fail closed. Port zero is also
rejected by the shared URL validator, rather than becoming the default port.

The new download client attaches no model API credentials, ignores environment
proxies, and supplies no proxy override. A total deadline includes DNS,
redirects and streaming the decoded body. Declared and observed sizes are
checked, empty bodies are rejected, and response contexts close on failure or
cancellation. Cancellation still propagates to the caller. Public download
errors omit raw URLs, redirect Locations and third-party response text.

- Speech JSON: 30-second deadline, 10 MiB default maximum. A positive integer
  `XAGENT_REMOTE_JSON_DOWNLOAD_MAX_BYTES` overrides the size; invalid explicit
  settings fail closed. JSON parsing and TTS happen only after download checks.
- Gemini reference image: 60-second deadline, existing
  `XAGENT_IMAGE_DOWNLOAD_MAX_BYTES` limit (64 MiB default). Only the subsequent
  configured model API request gets the model credential. A rejected reference
  never reaches that potentially billed model call.

The initial input-download patch did not restrict configured model endpoints.
The subsequent managed-client changes now reject private model and MCP
destinations by default; see [managed public egress](managed-public-egress.md)
for the current policy and compatibility impact. Gemini data URLs and trusted
SDK local-file inputs are not changed here.

## Workspace authorization

Speech-batch JSON paths and per-segment voice-cloning reference files must use
the workspace resolver when a workspace exists. A failed workspace check no
longer falls back to reading an arbitrary absolute or current-directory host
file. Registered file IDs and explicitly authorized external workspace roots
retain the resolver's existing permissions. Standalone SDK calls without a
workspace keep their existing trusted local-file behavior.

Application-level success messages no longer print the speech JSON URL, and
download errors do not echo signed query strings. Keep production HTTPX logging
at its existing WARNING level; enabling third-party wire/request logging in
debug mode can still expose request URLs.

## Verification and remaining work

The regression suite exercises real HTTPX clients, shared DNS validation,
pinning, redirects and caller integration. Socket transport and DNS are fake;
no live internal or metadata endpoints are used. It also covers streaming and
declared size limits, decoding, deadlines, cancellation, preserved configured
model requests, and workspace path/symlink rejection.

The updated branch incorporates #3183, #3186's generic API policy, and the
dependency changes in #3185. Merge those parent PRs first. Managed MCP/model
clients and proxy-backed fetches now use connection-level public destination
checks, as described in [managed public egress](managed-public-egress.md).
This does not change production infrastructure, pin the final SaaS core
version, or complete CASA item 15. Arbitrary user code and third-party programs
are outside the managed-client boundary; independent network isolation was
deferred. Build, deployment and acceptance testing are still required.
