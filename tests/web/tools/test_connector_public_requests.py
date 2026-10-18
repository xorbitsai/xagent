"""Real public adapter coverage: direct pinning, proxies and DNS rebinding."""

import importlib
import io
import socket

import pytest
import urllib3.util.connection
from urllib3.response import HTTPResponse

from xagent.core.utils import public_requests

CONNECTORS = ["linkedin", "freshdesk", "zendesk", "magento", "posthog"]
PUBLIC_IP = "93.184.216.34"
DENIAL = "Host must not resolve to a private network"


@pytest.fixture
def connector_wire(monkeypatch):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "XAGENT_TRUSTED_EGRESS_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
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
    }.items():
        monkeypatch.setenv(key, value)
    seen = []
    original_connection = urllib3.util.connection.create_connection

    def urlopen(pool, method, url, **kwargs):
        assert urllib3.util.connection.create_connection is original_connection
        seen.append((pool, dict(kwargs["headers"])))
        return HTTPResponse(body=io.BytesIO(b"{}"), status=200, preload_content=False)

    monkeypatch.setattr("urllib3.connectionpool.HTTPConnectionPool.urlopen", urlopen)

    def load(connector, via_proxy):
        if via_proxy:
            monkeypatch.setenv("HTTPS_PROXY", "http://trusted.example:3128")
            monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
        module = importlib.import_module(f"xagent.web.tools.mcp.{connector}")
        if hasattr(module, "_session"):
            # Mirror a fresh stdio process, including import-time proxy state.
            session = module.requests.Session()
            monkeypatch.setattr(module, "_session", session)
        return module

    return load, seen


def _dns(address):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]


async def _call(module, connector):
    if connector == "linkedin":
        return await module.call_tool("get_profile", {})
    return module._request("GET", "/test")


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", CONNECTORS)
@pytest.mark.parametrize("via_proxy", [False, True])
async def test_connectors_use_checked_direct_ip_or_trusted_proxy(
    monkeypatch, connector_wire, connector, via_proxy
):
    load, seen = connector_wire
    module = load(connector, via_proxy)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args: _dns(PUBLIC_IP))
    await _call(module, connector)
    assert len(seen) == 1
    pool, headers = seen[0]
    hostname = headers["Host"]
    assert pool.host == (hostname if via_proxy else PUBLIC_IP)
    assert pool.assert_hostname == hostname
    assert pool.conn_kw["server_hostname"] == hostname
    if via_proxy:
        assert pool.proxy.host == "trusted.example"
    else:
        assert pool.proxy is None


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", CONNECTORS)
@pytest.mark.parametrize("via_proxy", [False, True])
async def test_connectors_reject_private_dns(
    monkeypatch, connector_wire, connector, via_proxy
):
    load, seen = connector_wire
    module = load(connector, via_proxy)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args: _dns("10.0.0.1"))
    if connector == "linkedin":
        result = await _call(module, connector)
        assert DENIAL in result[0].text
    else:
        with pytest.raises((ValueError, RuntimeError), match=DENIAL):
            await _call(module, connector)
    assert not seen


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", CONNECTORS)
@pytest.mark.parametrize("via_proxy", [False, True])
async def test_adapter_rechecks_dns_after_public_connector_preflight(
    monkeypatch, connector_wire, connector, via_proxy
):
    load, seen = connector_wire
    module = load(connector, via_proxy)
    lookups = []

    def dns(*args):
        lookups.append(args[0])
        return _dns(PUBLIC_IP if len(lookups) == 1 else "10.0.0.1")

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    if connector == "linkedin":
        # LinkedIn has no connector DNS preflight: each real facade request
        # must obtain its own checked snapshot, rather than caching permission.
        await _call(module, connector)
        assert len(seen) == 1
        seen.clear()
        result = await _call(module, connector)
        assert DENIAL in result[0].text
    else:
        with pytest.raises(RuntimeError, match=DENIAL) as error:
            await _call(module, connector)
        # Prove refusal came from the real adapter, not connector preflight.
        assert isinstance(error.value.__cause__, public_requests.exceptions.InvalidURL)
    assert len(lookups) == 2
    assert not seen


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
