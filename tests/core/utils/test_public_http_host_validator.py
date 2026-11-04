"""Optional host restrictions for the shared public HTTP fetcher."""

import inspect
import socket

import httpx
import pytest

from xagent.core.utils import security


@pytest.mark.parametrize("rejected_hop", [0, 1, None])
async def test_host_validator_runs_before_dns_on_every_hop(monkeypatch, rejected_hop):
    assert (
        "host_validator"
        in inspect.signature(security.fetch_public_http_bytes).parameters
    )
    validated, resolved, requested = [], [], []

    def validate(host):
        validated.append(host)
        if len(validated) - 1 == rejected_hop:
            raise ValueError("host rejected")

    def resolve(host, port, *args):
        resolved.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    def respond(request):
        requested.append(request)
        if len(requested) == 1:
            return httpx.Response(
                302, headers={"Location": "https://second.example/image"}
            )
        return httpx.Response(200, content=b"image")

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:

        async def fetch():
            return await security.fetch_public_http_bytes(
                client,
                "https://first.example/image",
                max_content_bytes=50,
                timeout=5,
                host_validator=validate,
            )

        if rejected_hop is None:
            assert (await fetch()).content == b"image"
        else:
            with pytest.raises(ValueError, match="host rejected"):
                await fetch()
    assert (
        validated
        == ["first.example", "second.example"][: 1 if rejected_hop == 0 else 2]
    )
    count = 2 if rejected_hop is None else rejected_hop
    assert len(resolved) == len(requested) == count


async def test_default_host_policy_still_allows_unrelated_public_hosts(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"ok")
        )
    ) as client:
        result = await security.fetch_public_http_bytes(
            client, "https://unrelated.example/image", max_content_bytes=10, timeout=5
        )
    assert result.content == b"ok"
