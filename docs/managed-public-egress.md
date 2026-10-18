# Managed application outbound connections

This follow-up tightens the managed HTTP clients, remote MCP sessions and media
input paths used by the web application. It depends on the dependency closeout
and the earlier media download and generic API changes. It does not install a
new network service or change a running deployment.

## Connection boundaries

The public-only HTTPX, Requests, aiohttp and botocore transports validate all
resolved addresses and use the validated addresses for the actual connection.
Private, loopback, metadata, non-global and mixed DNS targets fail closed.
Literal IPv6 and IPv4-mapped IPv6 are checked too. Logical HTTP Host and TLS
SNI/certificate hostname verification remain intact.

Managed direct clients ignore ambient proxies. Requests and aiohttp clients
reject per-request proxy overrides. Browser/webpage and SVG downloads that use
a configured upstream proxy go through the local public-network gate: the
upstream receives a checked literal CONNECT destination rather than resolving
the untrusted hostname independently. Closing or cancelling the operation also
closes the gate and accepted tunnels.

HTTPX and Requests redirects validate the next connection, strip all caller
headers on cross-origin empty GET/HEAD redirects, and reject cross-origin body
replays. aiohttp model clients reject cross-origin redirects. MCP HTTP clients
stay on the configured origin, including negotiated SSE write URLs; legacy
WebSocket sessions pin the connection, ignore environment proxies, and reject
handshake redirects. Redirect chains are bounded.

## Managed callers

- Default remote MCP SSE, streamable HTTP and WebSocket sessions. No private
  MCP allowlist is introduced; existing private records and bindings are not
  deleted, but will fail to connect after deployment.
- OpenAI/Azure, Anthropic, Gemini and Zhipu managed chat clients and their
  catalog readers; Gemini/OpenAI image clients; public model-list requests.
- Xinference client discovery, auth probing, and returned synchronous or
  asynchronous model handles. The async path retains its nonblocking,
  explicitly configured key behavior and does not issue a synchronous probe.
- ElevenLabs ASR, TTS, music and sound effects, including explicit ownership
  and closing of injected HTTP clients.
- Bedrock runtime connections, retaining AWS streaming connection classes,
  configured timeouts and CA verification while disabling proxy bypasses.
- Application Requests-based embedding/rerank clients, OAuth HTTP calls,
  built-in SaaS connectors and skill registry requests.
- Custom-endpoint Zhipu web search and serialized A2A tool configurations.
  A caller-controlled `allow_private_networks` field cannot approve an A2A
  internal destination.
- OpenAI reference-image downloads join Gemini reference images and speech
  JSON on the bounded public-input downloader. Provider credentials are not
  attached to those downloads, and rejected references do not reach the billed
  image-edit request.

Configured private model endpoints on these managed paths are no longer
supported by default. Public DNS names resolving to private/split-horizon
addresses are rejected too. There is no fallback to an unrestricted client.

## Scope outside the managed boundary

This is not an operating-system firewall and must not be represented as a
universal egress sandbox. Python code executed by a user, stdio/third-party MCP
programs, and arbitrary third-party SDK code can open their own sockets. An
independent restricted outlet/deployment component was explicitly deferred for
this round. Existing Docker/network settings are unchanged.

Trusted programmatic SDK entrypoints are not equivalent to serialized web tool
settings. The legacy optional LangChain adapter and explicitly supplied custom
client factories are not covered by the managed-client guarantee. The current
web/RAG resolver uses the basic adapter by default; no application call site
selects `use_langchain_adapter=True`. A future web exposure of that legacy
adapter must first supply an equally guarded transport. A2A's direct Python
constructor retains its trusted-code private-network option; serialized tool
configs cannot set it.

Deployment-owned database, Redis, Docker, registry and remote document-parser
connections are not blanket-blocked: their destinations must remain trusted
server configuration rather than caller/model-controlled URLs. Existing MCP
OAuth discovery uses its separate pinned transport; the server-only
`XAGENT_MCP_OAUTH_ALLOW_PRIVATE_HOSTS` development bypass must remain disabled
for CASA acceptance. No production configuration or database rows are changed.

## Acceptance

Regression tests exercise real client/SDK construction, actual pool targets,
Host/SNI, DNS rebinding/mixed answers, literal bypasses, redirects and credential
isolation. Socket destinations are mocked; no live metadata/internal service
is contacted. Separate local-proxy tests exercise loopback sockets only.

SaaS must pin the merged core and build/deploy it before these protections can
be considered live. Verify the reported CASA URLs and redirect/DNS cases in
that deployment. Retain the deferred arbitrary-code/network risk explicitly;
merging this application patch alone is not evidence that all of CASA #15 or
all possible outbound network access is closed.
