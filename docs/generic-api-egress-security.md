# Public-network boundary for the generic API tool

## Scope

The model-facing `api_call` tool (`APIClientCore`) accepts arbitrary URLs. It
now permits only public HTTP(S) targets. This is one application-level layer of
CASA #15, not a statement that all outbound access is isolated.

`CustomApiTool` delegates to this same client: configured private/internal
endpoints and URL userinfo are therefore also rejected, not exempted. This is
an intentional compatibility break, covered through the real adapter. There is
no model-controlled or configured-default-URL bypass. Dedicated MCP connections,
model endpoints, browser tools, and other URL consumers require their own
policies; production egress controls are still necessary.

## Request and redirect policy

- Reject localhost, private, loopback, link-local, non-global, and mixed
  public/private DNS answers, including IPv4-mapped IPv6 addresses.
- Resolve and validate each hop, then connect to one of those validated IPs.
  HTTP Host and TLS SNI/certificate verification retain the logical hostname.
  HTTPX retains the logical URL for redirect, cookie, and response metadata.
- Ignore environment proxies by default. The existing operator-only
  `XAGENT_TRUSTED_EGRESS_PROXY=1` opt-in permits the configured HTTP(S) proxy;
  every target is still checked locally, but the trusted proxy must enforce
  connect-time private-range restrictions because it performs its own DNS.
  HTTPS proxies use the configured `SSL_CERT_FILE`/`SSL_CERT_DIR` CA bundle
  for the proxy TLS hop as well as the destination TLS hop.
- Reject URL userinfo, non-HTTP(S) URLs, port zero, and CONNECT tunnels.
- Follow at most five redirects when redirects are enabled. A cross-origin hop
  is allowed only for a proven empty-body GET/HEAD. Even GET/HEAD bodies are
  rejected instead of being forwarded to another origin. All caller headers
  and cookies are removed,
  including unrecognized credential header names; only a canonical Host and
  fixed Accept/User-Agent values remain. POST/other body replays across origins
  are rejected.
  A same-host default-port HTTP-to-HTTPS upgrade preserves authentication and
  body semantics. TLS connections are not reused across pinned hostnames.
- Bound each complete attempt (DNS, redirects, headers, body) by the request
  timeout. Keep the existing decoded-response size limit (10 MiB by default).
- Retry GET/HEAD/OPTIONS failures within the existing retry count. Never
  automatically replay writes after a network error: even a connection failure
  at a later redirect hop can follow a successfully accepted write.

## Compatibility and rollout

The generic tool and configured Custom APIs no longer access private APIs or
accept `https://user:password@host/`. Use the existing explicit authentication
fields for public APIs. Do not re-enable private destinations with a
model-controlled bypass flag. If a private integration is needed, design a
tenant-scoped configured connector and a separately enforced deployment policy.

The public policy may reject split-horizon hosts or public hostnames that resolve
to a mixture of public and private IPs. It deliberately fails closed. It uses
validated public addresses in order, falling back only for connect failures
before the API body is sent, never for read/write failures. Network routing, NAT, and firewall policy cannot be
proven safe by Python IP classification alone.
The remaining connect budget is divided among untried addresses, including
timeout failures, so a black-holed first address does not consume the entire
attempt deadline. DNS time is charged to that same budget.

## Verification

`tests/core/tools/test_api_tool_public_network.py` runs the real API client,
redirect event hook, DNS validator, and pinning transport against a fake socket
transport. It checks DNS rebinding, mixed answers, Host/SNI pinning, ambient
proxy bypass, every-hop redirects, credential stripping, non-replayed writes,
response limits, and DNS timeout. No live internal or metadata endpoint is
contacted. Existing API/auth/connector-hint tests remain part of the regression
run.

Before closing CASA #15, inventory the other URL entry points and verify actual
production network egress rules and tenant-scoped exception handling. Merging
this change alone does not complete that work.
