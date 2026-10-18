"""Real public Requests adapter coverage, not connector-level request mocks."""

import importlib
import io
import socket

import pytest
from urllib3.response import HTTPResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connector", ["linkedin", "freshdesk", "zendesk", "magento", "posthog"]
)
@pytest.mark.parametrize("private_dns", [False, True])
async def test_connectors_use_trusted_proxy_and_still_reject_private_dns(
    monkeypatch, connector, private_dns
):
    module = importlib.import_module(f"xagent.web.tools.mcp.{connector}")
    for key, value in {
        "LINKEDIN_ACCESS_TOKEN": "token",
        "FRESHDESK_API_KEY": "key",
        "FRESHDESK_SUBDOMAIN": "tenant",
        "ZENDESK_ACCESS_TOKEN": "token",
        "ZENDESK_SUBDOMAIN": "tenant",
        "MAGENTO_ACCESS_TOKEN": "token",
        "MAGENTO_BASE_URL": "https://store.example.com",
        "POSTHOG_API_KEY": "key",
        "POSTHOG_HOST": "https://us.posthog.com",
        "HTTPS_PROXY": "http://trusted.example:3128",
        "XAGENT_TRUSTED_EGRESS_PROXY": "1",
    }.items():
        monkeypatch.setenv(key, value)
    address = "10.0.0.1" if private_dns else "93.184.216.34"
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))],
    )
    seen = []

    def urlopen(pool, method, url, **kwargs):
        seen.append((pool, dict(kwargs["headers"])))
        return HTTPResponse(body=io.BytesIO(b"{}"), status=200, preload_content=False)

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)
    try:
        if connector == "linkedin":
            await module.call_tool("get_profile", {})
        else:
            module._request("GET", "/test")
    except (ValueError, RuntimeError):
        if not private_dns:
            raise
    if private_dns:
        assert not seen
    else:
        assert len(seen) == 1
        pool, headers = seen[0]
        assert pool.proxy.host == "trusted.example"
        assert pool.host == headers["Host"]
        assert pool.assert_hostname == headers["Host"]


@pytest.mark.asyncio
async def test_linkedin_does_not_implicitly_trust_ambient_proxy(monkeypatch):
    from xagent.web.tools.mcp import linkedin

    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted.example:3128")
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)

    def no_request(*args, **kwargs):
        pytest.fail("Untrusted proxy reached HTTP dispatch")

    monkeypatch.setattr(linkedin.requests, "get", no_request)
    result = await linkedin.call_tool("get_profile", {})
    assert "trusted" in result[0].text
