# Managed MCP network migration

The default HTTP/SSE/streamable-HTTP and WebSocket clients require public
addresses and bind requests to the configured origin. Configure the final URL:
even a same-host HTTP to HTTPS upgrade must not inherit MCP credentials/bodies.

This is a breaking change for remote endpoints on localhost, Docker/LAN hosts
and private cloud addresses, including servers marked `managed: "internal"`.
That label describes lifecycle/ownership, not permission to bypass SSRF policy.
The model-only private allowlist does not authorize MCP endpoints.

## Operator-only private MCP endpoints

An operator can approve specific private HTTP/SSE/streamable-HTTP MCP servers
by configuring BOTH XAGENT_MCP_PRIVATE_ENDPOINTS (exact HTTP(S) origins,
including ports) and XAGENT_MCP_PRIVATE_CIDRS (approved DNS address ranges):

    XAGENT_MCP_PRIVATE_ENDPOINTS=http://mcp-tools:8000,http://10.0.3.7:9000
    XAGENT_MCP_PRIVATE_CIDRS=172.20.0.0/16,10.0.3.7/32

The rules match the model exceptions in `model-outbound-security.md`: paths,
query strings, userinfo and wildcard origins are rejected; every DNS answer
must be public or inside an approved range for a listed origin; a different
protocol or port does not inherit approval; metadata, link-local, multicast,
unspecified and reserved addresses stay forbidden. Invalid entries fail web and
worker startup with the variable name. The settings are read only from process
environment, so users who create MCP servers cannot enable them. Requests stay
bound to the configured origin and redirects to another origin are refused.

The approval is deployment-wide: any user allowed to create an MCP server can
point one at an approved origin, under any path. List only origins that every
such user may reach, and do not use broad CIDRs without that review.

Dual-stack localhost commonly resolves to BOTH 127.0.0.1 and ::1, so approving
`http://localhost:8000` needs both `127.0.0.1/32` and `::1/128`, or use an exact
literal origin such as `http://127.0.0.1:8000`. Well-known NAT64 answers in
64:ff9b::/96 are classified by their embedded IPv4 destination, as described in
`model-outbound-security.md`. Origins that are not listed keep the default
public-only behavior unchanged.

These settings cover only the HTTP transports used to load and call tools.
`POST /api/mcp/test-connection` still rejects an approved private origin even
though loading the server works, and private servers that need MCP OAuth
discovery or refresh are not supported. WebSocket MCP servers keep their
existing policy, and the model allowlist still does not apply to MCP. Do not
enable `XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS` to work around either. With a
trusted egress proxy enabled, the proxy must also allow the private origin.

For an internal server, migrate to a reviewed stdio connector when the server
supports that transport, or provide a public HTTPS endpoint with authentication
and appropriate network controls. Custom stdio programs run in the configured
code sandbox as its unprivileged user and fail closed when no sandbox is
available. Their commands, working directories, dependencies, and any private
network access must therefore exist explicitly inside that sandbox.
An HTTP-only private server that cannot use either option needs the operator
allowlist above; do not disable validation or use the OAuth development flag
as a runtime bypass.

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
list of at most 20 objects; `null` and `[]` mean none. An object takes one of
two forms, told apart by its keys, and the two can share a list.

A replacement entry is for a tool that takes a workspace file. It has exactly
these keys:

```json
[{"tool_name": "attach_document", "file_argument": "document_file_id",
  "issuing_tool_name": "request_upload",
  "allowed_upload_hosts": ["uploads.example.test"],
  "upload_path_prefix": "/staged/", "content_types": {".pdf": "application/pdf"}}]
```

When the model passes `file:<id>` in `file_argument`, the host calls
`issuing_tool_name` itself (the model never sees that tool's reply) and PUTs the
bytes of that workspace file to the signed URL it returns.

An issuer entry is for a connector whose issuing tool returns a file id that the
connector's receiving tools accept. It has exactly these keys:

```json
[{"issuing_tool_name": "request_upload",
  "allowed_upload_hosts": ["uploads.example.test"],
  "upload_path_prefix": "/staged/", "content_types": {".pdf": "application/pdf"}}]
```

The host takes the issuing tool over. The model sees a host tool of the same
name with one argument, `file`, written as `file:<id>`. When the model calls it,
the host calls `issuing_tool_name`, PUTs the bytes of that workspace file to the
signed URL it returns, and gives the model only the connector's `file_id`, plus
its `expires_at` in UTC when the connector reports one. The model never sees
the upload URL or the headers. It then puts that `file_id` into whichever
connector tool takes the file, also when that tool is called through another
tool's arguments. With an issuer entry the bytes leave when the model asks for
a file id, before any later tool call, and a file id that is never used stays in
the connector's storage until the connector removes it. If the connector does
not list the issuing tool for a user, that user gets no upload tool and the
host logs a warning that names the connector and the tool.

An approval gate that pauses the tool which takes the file does not hold the
upload back, because the bytes leave when the file id is requested; a gate
decides on the upload tool by its own name.

One issuing tool cannot be used by both forms, and two issuer entries cannot
name the same issuing tool.

The setting decides which outside hosts receive task files, so only the
connector's owner or a platform administrator can change it. The URL must be
HTTPS without a port or user info, on one of `allowed_upload_hosts`, under
`upload_path_prefix`, and the request uses the public-address egress policy of
remote MCP sessions.

Both forms have the host call the issuing tool itself. At log level DEBUG the
MCP SDK can log server messages in full, which includes the issuing tool's
reply and so its signed URL. The default level, INFO, does not. And if the
connector also accepts its issuing tool through a dispatcher tool (for example
`call_tool(name="request_upload", ...)`), the model can still reach the issuing
tool that way and see its reply, because the host does not know the
dispatcher's argument shape and does not block it.

Save an issuer entry only after every instance of the deployment runs this
version.

Before rolling a deployment back to a version that does not know issuer
entries, rewrite each issuer entry as a replacement entry. Give it a
`tool_name` that the connector does not list at the top level and a
`file_argument` of any name, and keep its `issuing_tool_name`, hosts, path
prefix and content types. The older version accepts that entry and keeps hiding
the issuing tool, while an issuer entry would make it mark the connector
unavailable. Do not clear the settings instead: without them the older version
offers the issuing tool to the model again. After rolling forward, restore the
issuer entry.

Settings that cannot be used fail closed: the web runtime marks the whole
connector unavailable with reason `invalid_configuration`, and
`signed_uploads_error` in the API response says why. A remote OAuth catalog
connector run under an actor's authorization policy uses only its catalog
definition, so if its row stores any settings it is unavailable with the same
reason, even when `signed_uploads_error` is `null` because the settings
themselves are valid.
