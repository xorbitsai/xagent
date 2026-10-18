"""Exercise the actual aiohttp connector, not a replaced ClientSession."""

import socket

import pytest
from aiohttp import web

from xagent.core.utils.media_download import PublicMediaResolver, download_public_media

UNSAFE_HOSTS = [
    "2130706433",
    "127.1",
    "0",
    "127.0.0.1",
    "127。0。0。1",
    "127．0．0．1",
    "１２７．０．０．１",
    "127.0.0.1．",
    "0x7f.1",
    "0177.0.0.1",
]


@pytest.fixture
async def media_server():
    paths = []
    redirect = {"location": ""}

    async def handle(request):
        paths.append(request.path)
        if request.path == "/redirect":
            # Preserve Unicode in the wire header; HTTPFound normalizes it
            # before the client can exercise its own URL normalization.
            return web.Response(status=302, headers={"Location": redirect["location"]})
        return web.Response(body=b"must not be downloaded")

    app = web.Application()
    app.router.add_get("/{tail:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        yield port, paths, redirect
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("host", [*UNSAFE_HOSTS, "private.test"])
async def test_real_connector_never_contacts_private_host(
    host, media_server, tmp_path, monkeypatch
):
    port, paths, _ = media_server
    original = socket.getaddrinfo

    def lookup(hostname, *args, **kwargs):
        return original(
            "127.0.0.1" if hostname == "private.test" else hostname, *args, **kwargs
        )

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    with pytest.raises(ValueError):
        await download_public_media(
            f"http://{host}:{port}/private",
            tmp_path / "output",
            timeout=2,
            max_bytes=100,
            media_kind="image",
        )
    assert paths == []
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".media-*"))


@pytest.mark.parametrize("host", UNSAFE_HOSTS)
async def test_real_connector_redirect_never_contacts_private_target(
    host, media_server, tmp_path, monkeypatch
):
    port, paths, redirect = media_server
    redirect["location"] = f"http://{host}:{port}/private"
    original = PublicMediaResolver.resolve

    async def resolve(self, hostname, requested_port=0, family=socket.AF_INET):
        if hostname == "public.test":
            # Route only the fixture's initial "public" request to our server.
            # All redirect validation and actual TCP connections remain real.
            return [
                {
                    "hostname": hostname,
                    "host": "127.0.0.1",
                    "port": port,
                    "family": socket.AF_INET,
                    "proto": socket.IPPROTO_TCP,
                    "flags": 0,
                }
            ]
        return await original(self, hostname, requested_port, family)

    monkeypatch.setattr(PublicMediaResolver, "resolve", resolve)
    with pytest.raises(ValueError):
        await download_public_media(
            f"http://public.test:{port}/redirect",
            tmp_path / "output",
            timeout=2,
            max_bytes=100,
            media_kind="image",
        )
    assert paths == ["/redirect"]
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".media-*"))
