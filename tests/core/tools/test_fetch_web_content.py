"""Tests for FetchWebContent tool."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from xagent.core.tools.adapters.vibe.fetch_web_content import (
    FetchWebContentArgs,
    FetchWebContentResult,
    FetchWebContentTool,
)
from xagent.core.tools.core.web_content import WebContentFetcher, get_trusted_proxy_url
from xagent.core.utils.security import PrivateNetworkHostError


@pytest.fixture
def fetch_tool():
    return FetchWebContentTool()


@pytest.fixture(autouse=True)
def allow_public_test_hosts():
    with patch(
        "xagent.core.utils.security.validate_public_http_url",
        new=AsyncMock(return_value=["93.184.216.34"]),
    ) as validate:
        yield validate


class _MockStreamResponse:
    def __init__(
        self,
        *,
        body: bytes = b"",
        chunks: list[bytes] | None = None,
        headers: dict[str, str] | None = None,
        status_code: int = 200,
        url: str = "https://example.com/page",
        reason_phrase: str = "OK",
        encoding: str | None = "utf-8",
        raise_status: bool = False,
    ) -> None:
        self._chunks = chunks if chunks is not None else [body]
        self.headers = headers or {}
        self.status_code = status_code
        self.url = url
        self.reason_phrase = reason_phrase
        self.encoding = encoding
        self._raise_status = raise_status

    def raise_for_status(self) -> None:
        if self._raise_status:
            raise httpx.HTTPStatusError(
                f"{self.status_code} {self.reason_phrase}",
                request=Mock(),
                response=self,
            )

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _MockStreamContext:
    def __init__(self, response: _MockStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _MockStreamResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class TestFetchWebContentTool:
    def test_tool_properties(self, fetch_tool):
        assert fetch_tool.name == "fetch_web_content"
        assert "web" in fetch_tool.tags
        assert fetch_tool.args_type() == FetchWebContentArgs
        assert fetch_tool.return_type() == FetchWebContentResult

    def test_sync_not_implemented(self, fetch_tool):
        with pytest.raises(NotImplementedError):
            fetch_tool.run_json_sync({"url": "https://example.com"})

    @pytest.mark.asyncio
    async def test_fetch_webpage_content(self, fetch_tool):
        html = """
        <html>
          <head><title>Example Title</title></head>
          <body>
            <script>console.log("remove me")</script>
            <h1>Readable Heading</h1>
            <p>Useful body text.</p>
            <a href="/about">About</a>
          </body>
        </html>
        """
        response = _MockStreamResponse(
            body=html.encode("utf-8"),
            headers={"content-type": "text/html; charset=utf-8"},
            url="https://example.com/page",
        )

        with patch(
            "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
        ):
            result = await fetch_tool.run_json_async(
                {"url": "https://example.com/page"}
            )

        assert result["success"] is True
        assert result["url"] == "https://example.com/page"
        assert result["title"] == "Example Title"
        assert "Readable Heading" in result["content"]
        assert "Useful body text." in result["content"]
        assert "console.log" not in result["content"]
        assert "https://example.com/about" in result["content"]
        assert result["status_code"] == 200
        assert result["content_type"] == "text/html; charset=utf-8"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_fetch_discovers_exact_html_assets_when_requested(self, fetch_tool):
        html = """
        <html>
          <head>
            <title>Brand</title>
            <link rel="icon" href="/favicon.png">
            <script defer src="/static/js/main.abc123.js"></script>
          </head>
          <body>
            <img src="/assets/brand-logo.svg" alt="Brand logo">
            <img src="/assets/product.png" alt="Product">
          </body>
        </html>
        """
        response = _MockStreamResponse(
            body=html.encode("utf-8"),
            headers={"content-type": "text/html"},
            url="https://example.com/campaign",
        )

        with (
            patch(
                "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
            ),
            patch("httpx.AsyncClient.get") as mock_get,
        ):
            result = await fetch_tool.run_json_async(
                {
                    "url": "https://example.com/campaign",
                    "include_assets": True,
                    "asset_query": "logo",
                }
            )

        assert result["success"] is True
        assert result["assets"] == [
            {
                "url": "https://example.com/assets/brand-logo.svg",
                "kind": "image",
                "name": "",
                "alt": "Brand logo",
                "source": "html",
            }
        ]
        mock_get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_discovers_asset_name_joins_multi_valued_class(
        self, fetch_tool
    ):
        html = """
        <html>
          <body>
            <img src="/assets/brand-logo.svg" class="logo primary" alt="Brand logo">
          </body>
        </html>
        """
        response = _MockStreamResponse(
            body=html.encode("utf-8"),
            headers={"content-type": "text/html"},
            url="https://example.com/campaign",
        )

        with (
            patch(
                "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
            ),
            patch("httpx.AsyncClient.get"),
        ):
            result = await fetch_tool.run_json_async(
                {
                    "url": "https://example.com/campaign",
                    "include_assets": True,
                }
            )

        assert result["success"] is True
        assert result["assets"] == [
            {
                "url": "https://example.com/assets/brand-logo.svg",
                "kind": "image",
                "name": "logo primary",
                "alt": "Brand logo",
                "source": "html",
            }
        ]

    @pytest.mark.asyncio
    async def test_fetch_follows_redirects(
        self, fetch_tool, allow_public_test_hosts: AsyncMock
    ):
        html = "<html><body><p>Redirect target</p></body></html>"
        redirect = _MockStreamResponse(
            headers={"location": "https://example.com/final"},
            status_code=302,
            url="https://example.com/start",
        )
        response = _MockStreamResponse(
            body=html.encode("utf-8"),
            headers={"content-type": "text/html"},
            url="https://example.com/final",
        )

        with patch(
            "httpx.AsyncClient.stream",
            side_effect=[
                _MockStreamContext(redirect),
                _MockStreamContext(response),
            ],
        ) as mock_stream:
            result = await fetch_tool.run_json_async(
                {"url": "https://example.com/start"}
            )

        assert result["success"] is True
        assert result["url"] == "https://example.com/final"
        assert "Redirect target" in result["content"]
        assert allow_public_test_hosts.await_args_list == [
            (("https://example.com/start",),),
            (("https://example.com/final",),),
        ]
        assert all(
            call.kwargs["follow_redirects"] is False
            for call in mock_stream.call_args_list
        )

    @pytest.mark.asyncio
    async def test_fetch_rejects_private_network_before_request(
        self, fetch_tool, allow_public_test_hosts: AsyncMock
    ):
        allow_public_test_hosts.side_effect = PrivateNetworkHostError(
            "Host must not resolve to a private network."
        )

        with patch("httpx.AsyncClient.stream") as mock_stream:
            result = await fetch_tool.run_json_async(
                {"url": "http://169.254.169.254/latest/meta-data/"}
            )

        assert result["success"] is False
        assert "private network" in result["error"]
        mock_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_plain_text_content(self, fetch_tool):
        response = _MockStreamResponse(
            body=b"plain text body",
            headers={"content-type": "text/plain; charset=utf-8"},
            url="https://example.com/plain.txt",
        )

        with patch(
            "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
        ):
            result = await fetch_tool.run_json_async(
                {"url": "https://example.com/plain.txt"}
            )

        assert result["success"] is True
        assert result["title"] == ""
        assert result["content"] == "plain text body"
        assert result["content_type"] == "text/plain; charset=utf-8"

    @pytest.mark.asyncio
    async def test_fetch_rejects_unsupported_binary_content(self, fetch_tool):
        response = _MockStreamResponse(
            body=b"%PDF-1.7",
            headers={"content-type": "application/pdf"},
            url="https://example.com/file.pdf",
        )

        with patch(
            "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
        ):
            result = await fetch_tool.run_json_async(
                {"url": "https://example.com/file.pdf"}
            )

        assert result["success"] is False
        assert result["content"] == ""
        assert result["content_type"] == "application/pdf"
        assert "Unsupported non-text content type" in result["error"]

    @pytest.mark.asyncio
    async def test_fetch_rejects_large_content_length(self, fetch_tool):
        response = _MockStreamResponse(
            body=b"",
            headers={
                "content-type": "text/html",
                "content-length": str(10 * 1024 * 1024 + 1),
            },
        )

        with patch(
            "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
        ):
            result = await fetch_tool.run_json_async(
                {"url": "https://example.com/large"}
            )

        assert result["success"] is False
        assert result["content"] == ""
        assert "exceeds maximum" in result["error"]

    @pytest.mark.asyncio
    async def test_fetch_rejects_stream_larger_than_limit(self):
        response = _MockStreamResponse(
            chunks=[b"1234", b"5678", b"9"],
            headers={"content-type": "text/html"},
        )

        with patch(
            "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
        ):
            result = await WebContentFetcher(max_content_bytes=8).fetch(
                "https://example.com/large"
            )

        assert result.success is False
        assert result.content == ""
        assert "exceeds maximum" in result.error

    @pytest.mark.asyncio
    async def test_fetch_webpage_http_error(self, fetch_tool):
        response = _MockStreamResponse(
            status_code=404,
            reason_phrase="Not Found",
            raise_status=True,
        )

        with patch(
            "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
        ):
            result = await fetch_tool.run_json_async(
                {"url": "https://example.com/missing"}
            )

        assert result["success"] is False
        assert result["content"] == ""
        assert result["status_code"] == 404
        assert "HTTP 404 error" in result["error"]

    def test_args_validation(self):
        args = FetchWebContentArgs(url="https://example.com")
        assert args.url == "https://example.com"
        assert args.include_assets is False
        assert args.asset_query is None


class TestGetTrustedProxyUrl:
    """An ambient HTTP(S)_PROXY must not silently reopen the DNS-rebinding
    TOCTOU window that ``via_proxy`` pinning gives up on: proxied requests
    only get IP pinning's protection if the proxy is explicitly marked
    trusted to enforce its own private-range egress policy."""

    def test_untrusted_proxy_raises(self, monkeypatch):
        monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
        monkeypatch.delenv("HTTP_PROXY", raising=False)

        with pytest.raises(PrivateNetworkHostError):
            get_trusted_proxy_url()

    def test_trusted_proxy_returns_url(self, monkeypatch):
        monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
        monkeypatch.delenv("HTTP_PROXY", raising=False)

        assert get_trusted_proxy_url() == "http://proxy.example.com:8080"

    def test_no_proxy_configured_returns_none_regardless_of_trust_flag(
        self, monkeypatch
    ):
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("http_proxy", raising=False)
        monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)

        assert get_trusted_proxy_url() is None

        monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
        assert get_trusted_proxy_url() is None

    @pytest.mark.asyncio
    async def test_fetch_web_content_surfaces_untrusted_proxy_as_tool_error(
        self, fetch_tool, monkeypatch
    ):
        """End-to-end: an ambient but untrusted proxy must fail the tool
        call with a clear error instead of silently fetching through it
        with DNS pinning disabled."""
        monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")
        monkeypatch.delenv("HTTP_PROXY", raising=False)

        with patch("httpx.AsyncClient.stream") as mock_stream:
            result = await fetch_tool.run_json_async({"url": "https://example.com"})

        assert result["success"] is False
        assert "XAGENT_TRUSTED_EGRESS_PROXY" in result["error"]
        mock_stream.assert_not_called()
