# Managed model outbound policy

## Upgrade notes

Managed model clients now refuse localhost, LAN/Docker and private cloud
endpoints by default, including previously saved configurations. Before
upgrading a deployment using those endpoints, configure BOTH private-origin
and private-address settings below. Invalid entries fail web/worker startup
and client construction with the relevant environment variable name.

Public model calls ignore untrusted ambient proxies; Zhipu web search and
third-party reference downloads instead reject a configured but untrusted
HTTP(S) proxy. These features no longer implicitly trust a corporate proxy.
Proxy-only deployments must audit the proxy's destination enforcement and
explicitly enable XAGENT_TRUSTED_EGRESS_PROXY before using it.

## Operator-only private exceptions

Default: managed model endpoints must resolve entirely to public addresses.
To use localhost, a Docker/LAN inference server or a private cloud endpoint,
configure BOTH XAGENT_MODEL_PRIVATE_ENDPOINTS (exact HTTP(S) origins, including
ports) and XAGENT_MODEL_PRIVATE_CIDRS (approved DNS address ranges). For example:

    XAGENT_MODEL_PRIVATE_ENDPOINTS=http://localhost:9997,http://xinference:9997
    XAGENT_MODEL_PRIVATE_CIDRS=127.0.0.1/32,::1/128,172.20.0.10/32

Paths, query strings, userinfo and wildcard origins are rejected. Every address
from DNS is checked: listing a hostname alone cannot permit arbitrary private
answers. A different protocol or port does not inherit approval. Metadata,
link-local, multicast, unspecified and reserved addresses remain forbidden,
including IPv4-mapped spellings and the IPv6 AWS metadata address. Invalid
operator configuration fails closed. Do not use broad CIDRs without reviewing
what model users would be able to reach.

Dual-stack localhost commonly resolves to BOTH 127.0.0.1 and ::1. Approve
both address ranges (as above), or use an exact literal origin for just one
family; approving only 127.0.0.1/32 does not permit an ::1 DNS answer.
Well-known DNS64/NAT64 addresses in 64:ff9b::/96 are classified by their embedded
IPv4 destination, while connections retain IPv6. Metadata and unsafe embedded
IPv4 destinations remain forbidden. Other translation prefixes are not inferred.

These settings are read only from process environment through config.py.
Serialized model/tool parameters cannot enable exceptions. They apply only to
the migrated model clients, not MCP (see `mcp-outbound-security.md`), SaaS
connectors, web fetches or third-party reference-image/audio downloads. Those
inputs use the bounded public downloader.

## Proxy and redirect compatibility

Managed clients use an explicitly trusted HTTP(S) egress proxy only when the
operator enables XAGENT_TRUSTED_EGRESS_PROXY. The trusted proxy must enforce
destination/DNS policy at connect time, including the exact model exceptions
when intentionally routing approved private models. Client preflight alone is
not a substitute for the proxy policy. No ambient/OS proxy or per-request proxy
override can silently disable the selected transport. Direct connections are
pinned to checked addresses; normal TLS verification and CA settings remain.

Upgrade note: all migrated model clients (HTTPX, aiohttp, Botocore and Requests)
ignore untrusted ambient HTTP(S)_PROXY and connect directly when the trust switch
is off. They do not fail construction merely because a corporate proxy is set,
and they never implicitly trust it. Proxy-only deployments must explicitly set
XAGENT_TRUSTED_EGRESS_PROXY=1 after checking the proxy's enforcement policy.
NO_PROXY does not bypass that selected route, even for an approved local model.
A deployment combining a corporate proxy with localhost/Xinference must use a
trusted proxy that can reach and authorize that model origin, or run those
clients in a separate process without the proxy enabled. No implicit direct
private-network bypass is added. Web fetchers/MCP/tools retain their stricter
configured-but-untrusted proxy rejection policy. Zhipu web search and reference
downloads do not receive private model exceptions.

HTTPX-backed SDK clients follow redirects with a five-hop limit and the shared
credential/body guard. Changed-origin body replay is refused. aiohttp model
requests retain their stricter same-origin redirect policy. No model credentials
are attached to third-party reference input downloads.

HTTPX model clients pool connections in separate scheme/host/port buckets, so
different TLS names sharing an IP cannot reuse one another's connection. Each
request still validates every current DNS answer before accessing its pool.
There are at most 16 pooled origins per client; additional origins use an
unpooled fallback until the client closes. Generic multi-origin transports
remain unpooled. HTTP/2 remains disabled.

## Timeouts and ownership

Zhipu's injected HTTPX client now consistently uses the configured model timeout
(default 180 seconds) for connect/read/write/pool phases, rather than inheriting
the SDK's ZAI_DEFAULT_TIMEOUT. Gemini's official and custom endpoints both use
the configured model timeout (SDK units are milliseconds). These are intentional
timeout normalization changes, not claims to retain the SDK defaults.

OpenAI chat/Azure/Anthropic/image constructors close their injected client if SDK
construction fails; OpenAI image exposes aclose/close. ElevenLabs retains explicit
ownership/cleanup, including constructor failure. Xinference SDK state and
Botocore transport swapping are version-sensitive and have real-SDK contracts.
Policy denials are non-retryable even if an SDK wraps them as connection errors.
OpenAI model-catalog reads explicitly disable SDK retries too: unlike chat, this
one-shot catalog path has no outer retry wrapper, so transient listing failures
now return to the caller without SDK backoff. This avoids SDK connection-error
retry logic retrying terminal destination-policy refusals.

## Explicitly deferred paths

The optional LangChain adapter in chat/langchain.py is NOT covered. Its
use_langchain_adapter resolver entrypoint must not be exposed as guarded merely
because the basic adapter is the default. The optional Volcengine/BytePlus Ark
video SDK in video/ark.py is also NOT migrated: it needs its own pinned SDK
contract and supported HTTP-client injection. Do not enable those paths for
untrusted endpoint configuration while relying on this managed-client guarantee.

User code, stdio MCP programs and arbitrary SDKs remain outside application
transport control. CASA #15 still needs deployment verification and the deferred
network/egress architecture; this PR does not claim universal isolation.
