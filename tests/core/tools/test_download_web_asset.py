"""Tests for downloading exact remote image assets into a workspace."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

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
