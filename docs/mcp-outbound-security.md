# Managed MCP network migration

The default HTTP/SSE/streamable-HTTP and WebSocket clients require public
addresses and bind requests to the configured origin. Configure the final URL:
even a same-host HTTP to HTTPS upgrade must not inherit MCP credentials/bodies.

This is a breaking change for remote endpoints on localhost, Docker/LAN hosts
and private cloud addresses, including servers marked `managed: "internal"`.
That label describes lifecycle/ownership, not permission to bypass SSRF policy.
The model-only private allowlist does not authorize MCP endpoints.

For an internal server, migrate to a trusted stdio connector when the server
supports that transport, or provide a public HTTPS endpoint with authentication
and appropriate network controls. Stdio programs are trusted executable code,
not a public-egress sandbox; review the command and deployment permissions.
An HTTP-only private server that cannot use either option is not supported by
this default transport. A future private MCP allowlist needs a separate design;
do not disable validation or use the OAuth development flag as a runtime bypass.

`XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS` is a development-only exception for MCP
OAuth discovery/token exchange, MCP OAuth grant refresh (including runtime
refresh), and the custom OAuth transport used by the MCP connection-test API.
That test transport can accept private endpoints and does not bind requests to
one origin. Passing the test is not evidence of runtime compatibility. The flag
does not relax the default managed MCP sessions or integration OAuth refresh.
Keep it disabled in production.

Single-origin MCP HTTP clients reuse connections to the checked IP. DNS is still
validated for every request; changed origins, private answers and changed DNS
addresses cannot reuse a connection as authorization. Origin binding runs at
HTTPX's public transport entrypoint, independently of the internal DNS hook.
Generic multi-origin clients retain disabled keep-alive to prevent cross-host
TLS identity reuse on a shared IP.

Trusted HTTP(S) egress proxies require `XAGENT_TRUSTED_EGRESS_PROXY`. WebSocket
uses an owned local gate and literal-IP CONNECT targets. DNS, gate startup and
handshakes share one connection budget; the established session is not limited
to that short connection budget. Proxy policy, deployment isolation and arbitrary
stdio/user code remain separate responsibilities. CASA #15 is not fully closed
by these application-client changes alone.
