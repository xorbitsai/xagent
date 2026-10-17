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

These restrictions deliberately do not apply to configured model API endpoints:
legitimate private model services retain their previous configuration behavior.
Gemini data URLs and trusted SDK local-file inputs are not changed here.

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

This patch is based on #3183 and is complementary to #3186's generic API policy.
It does not change production infrastructure, pin the SaaS core version, or
complete CASA item 15. Configured MCP/model destinations, other SDK/provider
URL consumers, and sandbox egress still require scoped policies and deployment
acceptance. In particular, existing fetch paths configured with an explicit
outbound proxy need the proxy to enforce destination/DNS policy at connection
time; preflight DNS validation by the application is not sufficient against
rebinding through a separately resolving proxy. A universal production egress
policy must account for authorized private integrations, not add a
model-controlled allow-private flag.
