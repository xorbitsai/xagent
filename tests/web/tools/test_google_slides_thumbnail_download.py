"""Shared transport and publication contracts for slide thumbnails."""

import asyncio
import inspect
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from xagent.core.utils import security
from xagent.web.tools.mcp import google_slides as slides

HOST = "lh3.googleusercontent.com"
URL = f"https://{HOST}/signed-image?arbitrary=synthetic-private-query"
_buffer = io.BytesIO()
Image.new("RGB", (2, 3), (10, 20, 30)).save(_buffer, format="PNG")
PNG = _buffer.getvalue()


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", str(tmp_path))
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "XAGENT_TRUSTED_EGRESS_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture
def api(monkeypatch):
    service = Mock()
    presentations = service.presentations.return_value
    presentations.get.return_value.execute.return_value = {
        "slides": [{"objectId": "s1"}]
    }
    thumbnail = presentations.pages.return_value.getThumbnail
    thumbnail.return_value.execute.return_value = {"contentUrl": URL}
    monkeypatch.setattr(slides, "get_slides_service", lambda: service)
    return presentations, thumbnail


async def invoke(**kwargs):
    result = slides.google_slides_get_slide_thumbnail(presentation_id="pres1", **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return json.loads(result)


async def test_unknown_slide_is_rejected_before_thumbnail_request(api):
    result = await invoke(slide_id="missing-slide")
    assert result["status"] == "error"
    assert "slide_id" in result["message"] and "not found" in result["message"]
    api[1].assert_not_called()


async def test_locator_guidance_link_is_preserved():
    result = slides.google_slides_get_slide_thumbnail(
        presentation_id="file:synthetic", slide_id="s1"
    )
    if inspect.isawaitable(result):
        result = await result
    result = json.loads(result)
    assert result["status"] == "error"
    assert "https://docs.google.com/presentation/d/..." in result["message"]


def test_failed_write_removes_only_its_new_file(monkeypatch, tmp_path):
    output, fd = slides._open_thumbnail_output_directory()
    original = os.fdopen
    existing = output / "existing.png"
    existing.write_bytes(b"existing output")

    class BrokenWrite:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, data):
            self.handle.write(data[:8])
            self.handle.flush()
            raise OSError("write interrupted")

    monkeypatch.setattr(os, "fdopen", lambda *a, **kw: BrokenWrite(original(*a, **kw)))
    try:
        with pytest.raises(Exception, match="could not save"):
            slides._write_thumbnail_png(
                PNG,
                output_dir=output,
                dir_fd=fd,
                filename="slide.png",
            )
    finally:
        os.close(fd)
    assert list(output.iterdir()) == [existing]
    assert existing.read_bytes() == b"existing output"


@pytest.fixture
def transport(monkeypatch):
    assert inspect.iscoroutinefunction(slides.google_slides_get_slide_thumbnail)
    state = {
        "requests": [],
        "response": lambda request: httpx.Response(200, content=PNG),
    }

    async def resolve(url):
        return ["93.184.216.34"]

    def respond(request):
        state["requests"].append(request)
        return state["response"](request)

    @asynccontextmanager
    async def client(proxy, **kwargs):
        state["proxy"] = proxy
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as value:
            yield value

    monkeypatch.setattr(security, "validate_public_http_url", resolve)
    monkeypatch.setattr(slides, "public_fetch_client", client)
    return state


async def test_identity_encoding_is_requested_and_compression_rejected(
    api, transport, tmp_path
):
    transport["response"] = lambda request: httpx.Response(
        200,
        headers={"content-encoding": "gzip"},
        stream=httpx.ByteStream(b"compressed"),
    )
    result = await invoke(slide_id="s1")
    assert transport["requests"][0].headers["accept-encoding"] == "identity"
    assert result["status"] == "error"
    assert "encoding" in result["message"]
    assert not (tmp_path / "output").exists()


async def test_api_guidance_is_not_redacted(api):
    guidance = "https://docs.google.com/presentation/d/..."
    api[1].return_value.execute.side_effect = ValueError(f"Open {guidance}")
    result = await invoke(slide_id="s1")
    assert guidance in result["message"]


async def test_redirect_cookies_are_removed(api, transport):
    def respond(request):
        if len(transport["requests"]) == 1:
            return httpx.Response(
                302,
                headers={
                    "location": URL + "&next=1",
                    "set-cookie": "session=synthetic; Path=/",
                },
            )
        return httpx.Response(200, content=PNG)

    transport["response"] = respond
    result = await invoke(slide_id="s1")
    assert result["status"] == "success"
    assert len(transport["requests"]) == 2
    for request in transport["requests"]:
        assert "cookie" not in request.headers
        assert "authorization" not in request.headers


@pytest.fixture
async def loopback(monkeypatch):
    assert inspect.iscoroutinefunction(slides.google_slides_get_slide_thumbnail)
    seen, pinned, tasks = [], [], set()
    release = asyncio.Event()
    state = {
        "seen": seen,
        "pinned": pinned,
        "release": release,
    }

    async def serve(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            seen.append(request)
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(PNG)}\r\nConnection: close\r\n\r\n".encode()
            )
            await release.wait()
            writer.write(PNG)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    class LocalTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            pinned.append(str(request.url))
            local = httpx.Request(
                request.method,
                request.url.copy_with(scheme="http", host="127.0.0.1", port=port),
                headers=request.headers,
                extensions=request.extensions,
            )
            return await self.inner.handle_async_request(local)

        async def aclose(self):
            await self.inner.aclose()

    async def resolve(url):
        return ["93.184.216.34"]

    @asynccontextmanager
    async def client(proxy, **kwargs):
        async with httpx.AsyncClient(transport=LocalTransport()) as value:
            yield value

    monkeypatch.setattr(security, "validate_public_http_url", resolve)
    monkeypatch.setattr(slides, "public_fetch_client", client)
    try:
        yield state
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)


async def test_default_logging_hides_signed_request_url(api, loopback, caplog):
    caplog.set_level(logging.INFO)
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING
    loopback["release"].set()
    result = await invoke(slide_id="s1")
    assert result["status"] == "success"
    assert any(
        "93.184.216.34" in url and "synthetic-private-query" in url
        for url in loopback["pinned"]
    )
    assert b"synthetic-private-query" in loopback["seen"][0]
    assert "synthetic-private-query" not in caplog.text


@pytest.mark.parametrize("length", ["+77", "²"])
async def test_declared_length_requires_ascii_digits(api, transport, length):
    transport["response"] = lambda request: httpx.Response(
        200,
        headers=[(b"content-length", length.encode("utf-8"))],
        stream=httpx.ByteStream(PNG),
    )
    result = await invoke(slide_id="s1")
    assert result["status"] == "error"
    assert "invalid Content-Length" in result["message"]


async def test_total_deadline_includes_dns_wait(api, transport, monkeypatch):
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def resolve(url):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    monkeypatch.setattr(security, "validate_public_http_url", resolve)
    monkeypatch.setattr(slides, "_THUMBNAIL_TIMEOUT_SECONDS", 0.1)
    result = await asyncio.wait_for(invoke(slide_id="s1"), 5)
    assert entered.is_set() and exited.is_set()
    assert "timed out" in result["message"]
    assert transport["requests"] == []


async def test_shared_fetch_options_and_trusted_proxy(api, transport, monkeypatch):
    proxy = "http://127.0.0.1:9876"
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "true")
    from unittest.mock import AsyncMock

    fetch = AsyncMock(wraps=slides.fetch_public_http_bytes)
    monkeypatch.setattr(slides, "fetch_public_http_bytes", fetch)
    result = await invoke(slide_id="s1")
    assert result["status"] == "success"
    options = fetch.call_args.kwargs
    assert options["max_content_bytes"] == 10 * 1024 * 1024
    assert options["timeout"] == 30
    assert options["via_proxy"] is True
    assert options["require_identity_encoding"] is True
    assert options["allow_https_downgrade"] is False
    assert options["require_non_empty"] is True
    assert options["required_status_code"] == 200
    assert options["host_validator"] is slides._validate_thumbnail_host
    assert transport["proxy"] == proxy
    assert transport["requests"][0].url.host == HOST


async def test_api_failure_does_not_create_output_directory(api, tmp_path):
    api[1].return_value.execute.side_effect = ValueError("presentation unavailable")
    result = await invoke(slide_id="s1")
    assert result["status"] == "error"
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("http", "thumbnail image download failed: HTTP 403"),
        ("network", "thumbnail image download failed: network error"),
        ("validation", "Thumbnail image response was empty"),
    ],
)
async def test_download_error_messages_are_useful_without_signed_urls(
    api, transport, caplog, failure, expected
):
    def respond(request):
        if failure == "network":
            raise httpx.ConnectError(f"Cannot connect to {URL}", request=request)
        return httpx.Response(403 if failure == "http" else 200, content=b"")

    transport["response"] = respond
    result = await invoke(slide_id="s1")
    assert result["status"] == "error"
    assert result["message"] == expected
    assert expected in caplog.text
    assert "synthetic-private-query" not in json.dumps(result) + caplog.text


async def test_https_port_uses_shared_url_policy(api, transport):
    api[1].return_value.execute.return_value = {
        "contentUrl": f"https://{HOST}:8443/image"
    }
    result = await invoke(slide_id="s1")
    assert result["status"] == "success"
    assert transport["requests"][0].url.port == 8443
