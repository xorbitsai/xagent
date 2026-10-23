# Managed MCP network migration

The default HTTP/SSE/streamable-HTTP and WebSocket clients require public
addresses and bind requests to the configured origin. Configure the final URL:
even a same-host HTTP to HTTPS upgrade must not inherit MCP credentials/bodies.

This is a breaking change for remote endpoints on localhost, Docker/LAN hosts
and private cloud addresses, including servers marked `managed: "internal"`.
That label describes lifecycle/ownership, not permission to bypass SSRF policy.
The model-only private allowlist does not authorize MCP endpoints.

For an internal server, migrate to a reviewed stdio connector when the server
supports that transport, or provide a public HTTPS endpoint with authentication
and appropriate network controls. Custom stdio programs run in the configured
code sandbox as its unprivileged user and fail closed when no sandbox is
available. Their commands, working directories, dependencies, and any private
network access must therefore exist explicitly inside that sandbox.
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

## Signed uploads

An `sse` or `streamable_http` connector can carry `signed_uploads`, set with
`POST /api/mcp/servers` or `PUT /api/mcp/servers/{server_id}`. The value is a
list with one object per tool that takes a workspace file, each with exactly
these keys; `null` and `[]` mean none:

```json
[{"tool_name": "attach_document", "file_argument": "document_file_id",
  "issuing_tool_name": "request_upload",
  "allowed_upload_hosts": ["uploads.example.test"],
  "upload_path_prefix": "/staged/", "content_types": {".pdf": "application/pdf"}}]
```

When the model passes `file:<id>` in `file_argument`, the host calls
`issuing_tool_name` itself (the model never sees that tool) and PUTs the bytes
of that workspace file to the signed URL it returns. The setting therefore
decides which outside hosts receive task files, so only the connector's owner
or a platform administrator can change it. The URL must be
HTTPS without a port or user info, on one of `allowed_upload_hosts`, under
`upload_path_prefix`, and the request uses the public-address egress policy of
remote MCP sessions.

Settings that cannot be used fail closed: the web runtime marks the whole
connector unavailable with reason `invalid_configuration`, and
`signed_uploads_error` in the API response says why. A remote OAuth catalog
connector run under an actor's authorization policy uses only its catalog
definition, so if its row stores any settings it is unavailable with the same
reason, even when `signed_uploads_error` is `null` because the settings
themselves are valid.
