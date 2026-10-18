"""Security regressions for the actual media downloader, without external IO."""

import asyncio
import socket
from unittest.mock import AsyncMock

import aiohttp
import pytest

from xagent.core.utils.media_download import PublicMediaResolver, download_public_media


class Response:
    def __init__(self, *, status=200, headers=None, chunks=(b"image",)):
        self.status = status
        self.headers = headers or {}
        self.chunks = chunks
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


@pytest.fixture
def session(monkeypatch):
    class Session:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.calls = []
            self.responses = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.options["connector"].close()

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return self.responses.pop(0)

    instances = []

    def create(**kwargs):
        instance = Session(**kwargs)
        instance.responses = list(responses)
        instances.append(instance)
        return instance

    responses = []
    monkeypatch.setattr(aiohttp, "ClientSession", create)
    return responses, instances


async def download(url, destination):
    await download_public_media(
        url, destination, timeout=10, max_bytes=10, media_kind="image"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a",
        "http://2130706433/a",
        "http://127.1/a",
        "http://0/a",
        "http://169.254.43518/latest/meta-data/",
        "http://127.1./a",
        "http://169.254.169.254/latest/meta-data/",
        "http://169。254。169。254/latest/meta-data/",
        "http://[::1]/a",
        "http://[::ffff:127.0.0.1]/a",
        "http://10.0.0.1/a",
        "http://localhost/a",
        "file:///etc/passwd",
        "ftp://example.com/a",
        "https://user:password@example.com/a",
        "https://example.com:0/a",
    ],
)
async def test_unsafe_initial_url_never_opens_a_session(url, tmp_path, session):
    _, instances = session
    with pytest.raises(ValueError):
        await download(url, tmp_path / "output")
    assert instances == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["169.254.169.254"],
        ["::ffff:10.0.0.1"],
        ["93.184.216.34", "192.168.1.1"],
    ],
)
async def test_resolver_rejects_every_nonpublic_answer(monkeypatch, addresses):
    results = [{"host": ip} for ip in addresses]
    resolver = PublicMediaResolver()
    underlying = AsyncMock(return_value=results)
    monkeypatch.setattr(PublicMediaResolver.__bases__[0], "resolve", underlying)
    try:
        with pytest.raises(ValueError):
            await resolver.resolve("example.com", 443, socket.AF_UNSPEC)
        underlying.assert_awaited_once()
    finally:
        await resolver.close()


@pytest.mark.asyncio
async def test_public_dns_answers_are_returned_to_the_connector_unchanged(monkeypatch):
    results = [{"host": "93.184.216.34"}, {"host": "2606:4700:4700::1111"}]
    resolver = PublicMediaResolver()
    underlying = AsyncMock(return_value=results)
    monkeypatch.setattr(PublicMediaResolver.__bases__[0], "resolve", underlying)
    try:
        assert await resolver.resolve("example.com", 443) is results
        underlying.assert_awaited_once()
    finally:
        await resolver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/secret",
        "http://2130706433/secret",
        "http://127.1/secret",
        "http://0/secret",
        "http://169.254.43518/secret",
        "http://example.com/downgrade",
        "http://169.254.169.254/",
        "file:///etc/passwd",
        "https://user:secret@example.com/a",
        "//[::1]/a",
    ],
)
async def test_redirect_target_is_checked_before_second_request(
    location, status, tmp_path, session
):
    responses, instances = session
    responses.append(Response(status=status, headers={"Location": location}))
    with pytest.raises(ValueError):
        await download("https://example.com/a", tmp_path / "output")
    assert len(instances[0].calls) == 1
    assert not [path for path in tmp_path.iterdir() if path.is_file()]


@pytest.mark.asyncio
async def test_public_relative_redirect_and_proxy_bypass(
    tmp_path, session, monkeypatch
):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1234")
    responses, instances = session
    responses.extend([Response(status=302, headers={"Location": "/b"}), Response()])
    target = tmp_path / "output"
    await download("https://example.com/a", target)
    assert target.read_bytes() == b"image"
    instance = instances[0]
    assert instance.options["trust_env"] is False
    assert instance.options["connector"]._use_dns_cache is False
    assert isinstance(instance.options["connector"]._resolver, PublicMediaResolver)
    assert instance.calls == [
        ("https://example.com/a", {"allow_redirects": False}),
        ("https://example.com/b", {"allow_redirects": False}),
    ]
    assert [path for path in tmp_path.iterdir() if path.is_file()] == [target]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        Response(headers={"Content-Length": "11"}),
        Response(headers={"Content-Length": "-1"}),
        Response(headers={"Content-Length": "invalid"}),
        Response(chunks=(b"123456", b"78901")),
        Response(chunks=()),
        Response(chunks=(b"partial", OSError("broken stream"))),
        Response(status=404),
    ],
)
async def test_failed_download_preserves_destination_and_removes_partial_file(
    response, tmp_path, session
):
    responses, _ = session
    responses.append(response)
    target = tmp_path / "output"
    target.write_bytes(b"original")
    with pytest.raises((ValueError, RuntimeError, OSError)):
        await download("https://example.com/a", target)
    assert target.read_bytes() == b"original"
    assert [path for path in tmp_path.iterdir() if path.is_file()] == [target]


@pytest.mark.asyncio
async def test_redirect_loop_is_bounded(tmp_path, session):
    responses, instances = session
    responses.extend(
        Response(status=302, headers={"Location": "/loop"}) for _ in range(6)
    )
    with pytest.raises(ValueError, match="redirects"):
        await download("https://example.com/a", tmp_path / "output")
    assert len(instances[0].calls) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout,max_bytes",
    [(0, 10), (-1, 10), (float("inf"), 10), (float("nan"), 10), (1, 0)],
)
async def test_invalid_budgets_fail_before_opening_session(
    timeout, max_bytes, tmp_path, session
):
    _, instances = session
    with pytest.raises(ValueError):
        await download_public_media(
            "https://example.com/a",
            tmp_path / "output",
            timeout=timeout,
            max_bytes=max_bytes,
            media_kind="image",
        )
    assert instances == []


@pytest.mark.asyncio
async def test_total_timeout_cleans_partial_file(tmp_path, session):
    responses, _ = session
    response = Response()

    async def slow_chunks(size):
        yield b"first"
        await asyncio.sleep(1)
        yield b"last"

    response.iter_chunked = slow_chunks
    responses.append(response)
    target = tmp_path / "output"
    target.write_bytes(b"original")
    with pytest.raises(TimeoutError):
        await download_public_media(
            "https://example.com/a",
            target,
            timeout=0.05,
            max_bytes=10,
            media_kind="image",
        )
    assert target.read_bytes() == b"original"
    assert [path for path in tmp_path.iterdir() if path.is_file()] == [target]


@pytest.mark.asyncio
async def test_cancelled_download_cleans_partial_file(tmp_path, session):
    responses, _ = session
    response = Response()
    waiting = asyncio.Event()

    async def paused_chunks(size):
        yield b"first"
        waiting.set()
        await asyncio.Event().wait()

    response.iter_chunked = paused_chunks
    responses.append(response)
    target = tmp_path / "output"
    task = asyncio.create_task(download("https://example.com/a", target))
    await asyncio.wait_for(waiting.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not [path for path in tmp_path.iterdir() if path.is_file()]
