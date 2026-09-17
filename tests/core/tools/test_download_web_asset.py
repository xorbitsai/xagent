"""Tests for downloading exact remote image assets into a workspace."""

from __future__ import annotations

import io
import ssl
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from PIL import Image

from xagent.core.tools.adapters.vibe.download_web_asset import (
    DownloadWebAssetArgs,
    DownloadWebAssetResult,
    DownloadWebAssetTool,
)
from xagent.core.workspace import TaskWorkspace


class _ResponseContext:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def __aenter__(self) -> httpx.Response:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (12, 4), (146, 0, 186, 255)).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def workspace(tmp_path: Path) -> TaskWorkspace:
    return TaskWorkspace("asset-test", base_dir=str(tmp_path))


@pytest.fixture(autouse=True)
def allow_public_test_hosts():
    with patch(
        "xagent.core.utils.security.validate_public_http_url",
        new=AsyncMock(return_value=["93.184.216.34"]),
    ) as validate:
        yield validate


def test_download_web_asset_tool_contract(workspace: TaskWorkspace) -> None:
    tool = DownloadWebAssetTool(workspace)

    assert tool.name == "download_web_asset"
    assert tool.args_type() is DownloadWebAssetArgs
    assert tool.return_type() is DownloadWebAssetResult
    assert tool.metadata.category.value == "web_search"
    assert tool.metadata.read_only is False


@pytest.mark.asyncio
async def test_download_web_asset_registers_exact_image(
    workspace: TaskWorkspace,
) -> None:
    content = _png_bytes()
    response = httpx.Response(
        200,
        content=content,
        headers={"content-type": "image/png", "content-length": str(len(content))},
        request=httpx.Request("GET", "https://brand.example/logo.png"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch("httpx.AsyncClient.stream", return_value=_ResponseContext(response)):
        result = await tool.run_json_async({"url": "https://brand.example/logo.png"})

    assert result["success"] is True
    assert result["source_url"] == "https://brand.example/logo.png"
    assert result["filename"] == "logo.png"
    assert result["file_id"]
    assert result["markdown_link"] == f"[logo.png](file:{result['file_id']})"
    saved_path = workspace.output_dir / "logo.png"
    assert saved_path.read_bytes() == content
    assert result["file_ref"]["file_path"] == str(saved_path.resolve())


@pytest.mark.asyncio
async def test_download_web_asset_validates_every_redirect_hop(
    workspace: TaskWorkspace,
    allow_public_test_hosts: AsyncMock,
) -> None:
    content = _png_bytes()
    redirect = httpx.Response(
        302,
        headers={"location": "https://cdn.example/logo.png"},
        request=httpx.Request("GET", "https://brand.example/logo"),
    )
    final = httpx.Response(
        200,
        content=content,
        headers={"content-type": "image/png"},
        request=httpx.Request("GET", "https://cdn.example/logo.png"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch(
        "httpx.AsyncClient.stream",
        side_effect=[_ResponseContext(redirect), _ResponseContext(final)],
    ) as stream:
        result = await tool.run_json_async({"url": "https://brand.example/logo"})

    assert result["success"] is True
    assert result["source_url"] == "https://cdn.example/logo.png"
    assert allow_public_test_hosts.await_args_list == [
        (("https://brand.example/logo",),),
        (("https://cdn.example/logo.png",),),
    ]
    assert all(
        call.kwargs["follow_redirects"] is False for call in stream.call_args_list
    )


@pytest.mark.asyncio
async def test_download_web_asset_infers_extension_from_image_bytes(
    workspace: TaskWorkspace,
) -> None:
    content = _png_bytes()
    response = httpx.Response(
        200,
        content=content,
        headers={"content-type": "image/jpeg"},
        request=httpx.Request("GET", "https://brand.example/logo"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch("httpx.AsyncClient.stream", return_value=_ResponseContext(response)):
        result = await tool.run_json_async({"url": "https://brand.example/logo"})

    assert result["success"] is True
    assert result["filename"] == "logo.png"
    assert result["file_ref"]["mime_type"] == "image/png"
    assert result["content_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_download_web_asset_uses_exclusive_unique_filename(
    workspace: TaskWorkspace,
) -> None:
    content = _png_bytes()
    existing = workspace.output_dir / "logo.png"
    existing.write_bytes(b"existing")
    response = httpx.Response(
        200,
        content=content,
        headers={"content-type": "image/png"},
        request=httpx.Request("GET", "https://brand.example/logo.png"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch("httpx.AsyncClient.stream", return_value=_ResponseContext(response)):
        result = await tool.run_json_async({"url": "https://brand.example/logo.png"})

    assert result["success"] is True
    assert result["filename"] == "logo_1.png"
    assert existing.read_bytes() == b"existing"
    assert (workspace.output_dir / "logo_1.png").read_bytes() == content


@pytest.mark.asyncio
async def test_download_web_asset_preserves_official_svg(
    workspace: TaskWorkspace,
) -> None:
    content = b'<svg xmlns="http://www.w3.org/2000/svg" width="146" height="32" />'
    response = httpx.Response(
        200,
        content=content,
        headers={"content-type": "image/svg+xml"},
        request=httpx.Request("GET", "https://brand.example/simba-logo.svg"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch("httpx.AsyncClient.stream", return_value=_ResponseContext(response)):
        result = await tool.run_json_async(
            {"url": "https://brand.example/simba-logo.svg"}
        )

    assert result["success"] is True
    assert result["content_type"] == "image/svg+xml"
    assert (workspace.output_dir / "simba-logo.svg").read_bytes() == content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
        b'<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg />',
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        b'<image href="https://evil.example/x" /></svg>',
    ],
)
async def test_download_web_asset_rejects_svg_with_script(
    workspace: TaskWorkspace,
    content: bytes,
) -> None:
    response = httpx.Response(
        200,
        content=content,
        headers={"content-type": "image/svg+xml"},
        request=httpx.Request("GET", "https://brand.example/logo.svg"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch("httpx.AsyncClient.stream", return_value=_ResponseContext(response)):
        result = await tool.run_json_async({"url": "https://brand.example/logo.svg"})

    assert result["success"] is False
    assert list(workspace.output_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_download_web_asset_accepts_clean_svg(
    workspace: TaskWorkspace,
) -> None:
    content = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" />'
    response = httpx.Response(
        200,
        content=content,
        headers={"content-type": "image/svg+xml"},
        request=httpx.Request("GET", "https://brand.example/logo.svg"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch("httpx.AsyncClient.stream", return_value=_ResponseContext(response)):
        result = await tool.run_json_async({"url": "https://brand.example/logo.svg"})

    assert result["success"] is True
    assert (workspace.output_dir / "logo.svg").read_bytes() == content


def test_resolve_filename_corrects_mismatched_suffix() -> None:
    filename = DownloadWebAssetTool._resolve_filename(
        "x.html", "https://brand.example/x.html", ".png"
    )
    assert filename == "x.png"


def test_resolve_filename_keeps_jpg_jpeg_alias() -> None:
    filename = DownloadWebAssetTool._resolve_filename(
        "photo.jpeg", "https://brand.example/photo.jpeg", ".jpg"
    )
    assert filename == "photo.jpeg"


@pytest.mark.asyncio
async def test_download_web_asset_rejects_non_image_content(
    workspace: TaskWorkspace,
) -> None:
    response = httpx.Response(
        200,
        content=b"<html>not an image</html>",
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://brand.example/logo.svg"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch("httpx.AsyncClient.stream", return_value=_ResponseContext(response)):
        result = await tool.run_json_async({"url": "https://brand.example/logo.svg"})

    assert result["success"] is False
    assert "Unsupported web asset content type" in result["error"]
    assert list(workspace.output_dir.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_length", ["not-a-number", "-1"])
async def test_download_web_asset_rejects_invalid_declared_length(
    workspace: TaskWorkspace,
    declared_length: str,
) -> None:
    response = httpx.Response(
        200,
        content=_png_bytes(),
        headers={
            "content-type": "image/png",
            "content-length": declared_length,
        },
        request=httpx.Request("GET", "https://brand.example/logo.png"),
    )
    tool = DownloadWebAssetTool(workspace)

    with patch("httpx.AsyncClient.stream", return_value=_ResponseContext(response)):
        result = await tool.run_json_async({"url": "https://brand.example/logo.png"})

    assert result["success"] is False
    assert result["error"] == f"Invalid remote asset content length: {declared_length}"
    assert list(workspace.output_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_download_disables_trust_env_but_honors_explicit_proxy(
    workspace: TaskWorkspace,
) -> None:
    # httpx.AsyncClient defaults to trust_env=True, which falls back to the
    # OS's own proxy configuration (getproxies_macosx_sysconf()/
    # getproxies_registry()) once no HTTP(S)_PROXY env var is set --
    # reopening the DNS-rebinding TOCTOU window get_trusted_proxy_url()
    # exists to close. trust_env=False must be set explicitly, while the
    # proxy this connector *does* trust must still be passed through.
    content = _png_bytes()
    response = httpx.Response(
        200,
        content=content,
        headers={"content-type": "image/png", "content-length": str(len(content))},
        request=httpx.Request("GET", "https://brand.example/logo.png"),
    )
    tool = DownloadWebAssetTool(workspace)

    with (
        patch(
            "xagent.core.tools.adapters.vibe.download_web_asset.get_trusted_proxy_url",
            return_value="http://trusted-proxy.internal:8080",
        ),
        patch(
            "xagent.core.tools.adapters.vibe.download_web_asset.httpx.AsyncClient"
        ) as async_client,
    ):
        client = async_client.return_value.__aenter__.return_value
        client.stream = Mock(return_value=_ResponseContext(response))
        result = await tool.run_json_async({"url": "https://brand.example/logo.png"})

    assert result["success"] is True
    async_client.assert_called_once()
    client_kwargs = async_client.call_args.kwargs
    assert client_kwargs["proxy"] == "http://trusted-proxy.internal:8080"
    assert client_kwargs["trust_env"] is False
    assert isinstance(client_kwargs["verify"], ssl.SSLContext)
