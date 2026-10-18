# Managed model outbound policy

This model migration (#3193) requires the transport API in #3191. It is not
independently runnable before that prerequisite lands. No deployment is changed.

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

These settings are read only from process environment through config.py.
Serialized model/tool parameters cannot enable exceptions. They apply only to
the migrated model clients, not MCP, SaaS connectors, web fetches or third-party
reference-image/audio downloads. Those inputs use the bounded public downloader.

## Proxy and redirect compatibility

Managed clients use an explicitly trusted HTTP(S) egress proxy only when the
operator enables XAGENT_TRUSTED_EGRESS_PROXY. The trusted proxy must enforce
destination/DNS policy at connect time, including the exact model exceptions
when intentionally routing approved private models. Client preflight alone is
not a substitute for the proxy policy. No ambient/OS proxy or per-request proxy
override can silently disable the selected transport. Direct connections are
pinned to checked addresses; normal TLS verification and CA settings remain.

HTTPX-backed SDK clients follow redirects with a five-hop limit and the shared
credential/body guard. Changed-origin body replay is refused. aiohttp model
requests retain their stricter same-origin redirect policy. No model credentials
are attached to third-party reference input downloads.

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
