# Caller-supplied media inputs (CASA 15 follow-up)

Remote speech-batch JSON and Gemini image-editing reference downloads use a
bounded public-input policy. This policy does not isolate every outbound
request or replace provider-output and generic API destination policies.

## Enforced boundaries

`download_public_input` uses the existing shared HTTP downloader to validate
all resolved addresses and connect to the checked IP, preserving logical Host
and TLS SNI/certificate checks. Each redirect is independently validated and
the chain is bounded. Literal, mixed-DNS, metadata, private, non-global,
userinfo, non-HTTP(S), and port-zero targets fail closed. Port zero is also
rejected by the shared URL validator, rather than becoming the default port.

The download client attaches no model API credentials and disables HTTPX's
implicit environment/OS proxy lookup. A configured proxy is accepted only with
the existing `XAGENT_TRUSTED_EGRESS_PROXY=1` opt-in: the operator must ensure
that proxy enforces private-range destination/DNS policy at connection time.
Proxy mode preserves the target hostname for CONNECT/SNI; it cannot pin the
proxy's independently resolved IP. An untrusted ambient proxy fails closed.

A total deadline includes DNS, redirects and streaming. HTTPS-to-HTTP
redirects are rejected. Only HTTP 200 is accepted, before reading the body or
making a billed provider call. Requests advertise `Accept-Encoding: identity`;
non-identity response encodings are rejected before decompression, avoiding
compressed-input expansion. Declared and observed sizes are checked, empty
bodies are rejected, and response contexts close on failure or cancellation.
Cancellation propagates to the caller. Public errors omit raw URLs, redirect
Locations and third-party response text.

- Speech JSON: 30-second deadline, 10 MiB default maximum.
  `XAGENT_REMOTE_JSON_DOWNLOAD_MAX_BYTES` accepts positive bytes or size suffixes
  such as `2K`/`1.5M`, like the media-size settings. Blank means default;
  invalid explicit settings fail closed. Parsing and TTS follow download checks.
- Gemini reference image: 60-second deadline, existing
  `XAGENT_IMAGE_DOWNLOAD_MAX_BYTES` limit (64 MiB default). Only the subsequent
  configured model API request gets the model credential. A rejected reference
  never reaches that potentially billed model call.

The input-download policy is separate from configured model API requests.
The managed MCP/model client migrations in #3192/#3193 reject private endpoints
on their covered paths once merged and deployed; see
[managed public egress](managed-public-egress.md) for the complete-stack policy
and compatibility impact. This input helper alone does not enforce that policy.
Gemini data URLs and trusted SDK local-file inputs are not changed here.

## Workspace authorization

Speech-batch JSON paths and per-segment voice-cloning reference files must use
the workspace resolver when a workspace exists. A failed workspace check no
longer falls back to reading an arbitrary absolute or current-directory host
file. Registered file IDs and explicitly authorized external workspace roots
retain the resolver's existing permissions. Batch and single synthesis share
the reference resolver, including provider-side HTTP(S) reference URLs. Those
provider-fetched references are not downloads by this helper and still need
provider-specific policy. Standalone SDK calls without a workspace keep their
trusted local-file behavior; nonexistent references fail explicitly.

Application-level success messages no longer print the speech JSON URL, and
download errors do not echo signed query strings. Keep production HTTPX logging
at its existing WARNING level; enabling third-party wire/request logging in
debug mode can still expose request URLs.

## Verification and remaining work

The regression suite exercises real HTTPX clients, shared DNS validation,
pinning, redirects and caller integration. Socket transport and DNS are fake;
no live internal or metadata endpoints are used. It also covers streaming and
declared size limits, decoding, deadlines, cancellation, preserved configured
model requests, rejection before compressed-body reads, HTTP 206 rejection,
authorized workspace-relative paths and workspace path/symlink rejection.

The input policy does not change production infrastructure, pin the SaaS core
version, or complete CASA item 15. The managed MCP/model, connector and asset
migrations are separate PRs; their connection-level checks and remaining limits
are documented in [managed public egress](managed-public-egress.md). Each
migration still requires merge, deployment and acceptance. Arbitrary user code,
third-party programs and independent sandbox egress remain outside that boundary.
In particular, existing fetch paths configured with an explicit
outbound proxy need the proxy to enforce destination/DNS policy at connection
time; preflight DNS validation by the application is not sufficient against
rebinding through a separately resolving proxy. A universal production egress
policy must account for authorized private integrations, not add a
model-controlled allow-private flag.
