# Managed application outbound connections

## What this PR changes

This assets/A2A patch (#3195) ships these changes on current main:

- Webpage fetches, remote SVG inspection, web-asset downloads and bounded
  remote-input downloads route trusted proxy traffic through an owned local
  public-network gate. It checks connect-time DNS and sends the upstream a
  validated literal-IP CONNECT target. Redirects and mixed/private DNS cannot
  make that upstream independently resolve an attacker-controlled hostname.
- Gate connect/response timeouts follow the caller's timeout. Cancellation
  and normal completion close the gate and accepted tunnels.
- Serialized A2A entries cannot enable private-network access. Invalid entries
  are skipped individually, with a credential-free warning, so valid public
  A2A tools in the same list are retained.

It does not deploy a new network service or modify running infrastructure.

## Prerequisites and pending protections

#3191 provides the general public-only HTTPX, Requests, aiohttp and Botocore
transports. #3192 migrates managed MCP HTTP/SSE/WebSocket and OAuth refresh.
#3193 migrates managed model SDKs, model catalogs, custom Zhipu search and
OpenAI reference-image downloads. #3194 migrates Requests connectors, OAuth
HTTP calls and skill registries. Those protections are not implemented by
#3195 and must not be described as already shipped when only this PR is merged.

The prior media/generic-API and remote-input work is already on main. The
bounded downloader already serves audio/Gemini inputs; the OpenAI reference
image caller migration remains in #3193. This PR adds the proxy gate to the
shared bounded downloader, not the pending model callers themselves.

Follow the split sequence and synchronize/retest after prerequisites land.
Do not deploy an incomplete stack or treat this document as deployment evidence.
Private model endpoint policy and any operator exceptions belong to the model
PR, not caller-controlled A2A settings.

## Trusted upstream requirements

Set XAGENT_TRUSTED_EGRESS_PROXY only for an operator-approved HTTP(S) proxy.
The gate sends CONNECT <literal-ip>:<port> for **all** upstream fetches,
including plain HTTP destinations on port 80. The upstream must permit CONNECT
on the required destination ports and permit the checked IP destinations;
hostname-only allowlists and SSL-port-only CONNECT policies need adjustment.
Host and TLS SNI/certificate verification still use the logical destination.

SOCKS upstreams are not supported by these gated fetchers and fail with an
explicit error, without a direct-network fallback. Ambient/OS proxy discovery
and NO_PROXY cannot bypass the selected gate. SSL_CERT_FILE/SSL_CERT_DIR remain
supported for CA trust; TLS verification is not disabled.

The gate provides connect-time pinning. Merely setting via_proxy=True on the
bounded body helper does not: any other caller that bypasses the gate must use
a trusted proxy enforcing its own public-only connect-time DNS policy.

## Remaining boundary

This is application-client hardening, not an operating-system firewall or
universal egress sandbox. User Python code, stdio/third-party MCP programs and
SDK paths not explicitly migrated can open independent sockets. The separate
restricted-outlet/deployment architecture remains deferred. Database, Redis,
Docker and parser-service destinations remain trusted deployment configuration.

The optional LangChain adapter is not covered until explicitly migrated;
its configurable resolver entrypoint must not be represented as guarded.
A2A's direct Python constructor retains a trusted-code private-network option,
but serialized web tool configurations cannot enable it. The separate MCP
OAuth development private-host bypass must remain disabled for CASA acceptance.

## Verification and acceptance

Tests in this PR exercise the real webpage and bounded-input proxy paths,
literal CONNECT destinations, DNS rebinding rejection, timeout propagation,
cancellation, and a verified HTTPS tunnel preserving logical Host and TLS SNI.
Only loopback test sockets are used; no live metadata/internal service is called.
Broader transport/model guarantees require the tests in the prerequisite PRs.

SaaS must pin merged core, rebuild/deploy and retest the reported CASA URLs.
Retain the deferred arbitrary-code/network risk. Merging application changes
alone does not fully close CASA #15.
