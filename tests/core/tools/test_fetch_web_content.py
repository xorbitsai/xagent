"""Tests for FetchWebContent tool."""

from unittest.mock import Mock, patch

import httpx
import pytest

from xagent.core.tools.adapters.vibe.fetch_web_content import (
    FetchWebContentArgs,
    FetchWebContentResult,
    FetchWebContentTool,
)
from xagent.core.tools.core.web_content import WebContentFetcher


@pytest.fixture
def fetch_tool():
    return FetchWebContentTool()


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
    async def test_fetch_follows_redirects(self, fetch_tool):
        html = "<html><body><p>Redirect target</p></body></html>"
        response = _MockStreamResponse(
            body=html.encode("utf-8"),
            headers={"content-type": "text/html"},
            url="https://example.com/final",
        )

        with patch(
            "httpx.AsyncClient.stream", return_value=_MockStreamContext(response)
        ) as mock_stream:
            result = await fetch_tool.run_json_async(
                {"url": "https://example.com/start"}
            )

        assert result["success"] is True
        assert result["url"] == "https://example.com/final"
        assert "Redirect target" in result["content"]
        assert mock_stream.call_args.kwargs["follow_redirects"] is True

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
