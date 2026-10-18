from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.tools.core.image_tool import ImageGenerationToolCore
from xagent.core.tools.core.video_tool import VideoGenerationToolCore


@pytest.fixture
def media_tools(tmp_path):
    root = tmp_path / "workspace"
    output = root / "output"
    output.mkdir(parents=True)
    workspace = SimpleNamespace(workspace_dir=root, output_dir=output)
    return (
        ImageGenerationToolCore({}, {}, workspace),
        VideoGenerationToolCore({}, workspace=workspace),
        workspace,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("via_symlink", [False, True])
async def test_provider_local_results_cannot_escape_workspace(
    kind, via_symlink, media_tools, tmp_path
):
    image, video, workspace = media_tools
    secret = tmp_path / "outside-secret"
    secret.write_bytes(b"private host data")
    source = secret
    if via_symlink:
        source = workspace.workspace_dir / "link"
        source.symlink_to(secret)
    method = image._download_image if kind == "image" else video._download_video
    with pytest.raises(ValueError, match="outside the workspace"):
        await method(source.as_uri())
    assert not list(workspace.output_dir.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "video"])
async def test_workspace_local_media_is_preserved(kind, media_tools):
    image, video, workspace = media_tools
    source = workspace.workspace_dir / "local media.png"
    source.write_bytes(b"provider result")
    method = image._download_image if kind == "image" else video._download_video
    result = await method(source.as_uri())
    from pathlib import Path

    assert Path(result).read_bytes() == b"provider result"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "video"])
async def test_provider_metadata_url_rejected_without_a_connection(
    kind, media_tools, monkeypatch
):
    session = Mock()
    monkeypatch.setattr("aiohttp.ClientSession", session)
    image, video, workspace = media_tools
    method = image._download_image if kind == "image" else video._download_video
    with pytest.raises(ValueError):
        await method("http://169.254.169.254/latest/meta-data/")
    assert not list(workspace.output_dir.iterdir())
    session.assert_not_called()


@pytest.mark.parametrize("kind", ["image", "video"])
async def test_relative_local_media_is_resolved_from_workspace(
    kind, media_tools, monkeypatch, tmp_path
):
    image, video, workspace = media_tools
    source = workspace.workspace_dir / "relative.png"
    source.write_bytes(b"provider result")
    monkeypatch.chdir(tmp_path)
    method = image._download_image if kind == "image" else video._download_video
    from pathlib import Path

    assert Path(await method("relative.png")).read_bytes() == b"provider result"


@pytest.mark.parametrize("kind", ["image", "video"])
async def test_configured_budget_reaches_result_download(
    kind, media_tools, monkeypatch
):
    image, video, _ = media_tools
    monkeypatch.setenv(f"XAGENT_{kind.upper()}_DOWNLOAD_MAX_BYTES", "1K")
    downloader = AsyncMock()
    monkeypatch.setattr(
        f"xagent.core.tools.core.{kind}_tool.download_public_media", downloader
    )
    method = image._download_image if kind == "image" else video._download_video
    await method("https://example.com/result")
    assert downloader.await_args.kwargs["max_bytes"] == 1024


@pytest.mark.parametrize("method", ["generate_image", "edit_image"])
async def test_failed_image_save_reports_redacted_download_error(
    method, media_tools, monkeypatch
):
    image, _, workspace = media_tools
    model = Mock()
    model.model_name = "fixture"
    model.generate_image = AsyncMock(
        return_value={"image_url": "http://10.0.0.1/image"}
    )
    model.edit_image = AsyncMock(return_value={"image_url": "http://10.0.0.1/image"})
    monkeypatch.setattr(image, "_get_model", lambda _: model)
    monkeypatch.setattr(image, "_get_edit_model", lambda _: model)
    workspace.auto_register_files = nullcontext
    monkeypatch.setattr(
        image,
        "_download_image",
        AsyncMock(side_effect=ValueError("token=sensitive-provider-credential")),
    )
    kwargs = (
        {"image_url": "data:image/png;base64,aQ=="} if method == "edit_image" else {}
    )
    result = await getattr(image, method)(prompt="fixture", **kwargs)
    assert result["success"] is True  # Provider succeeded; saving did not.
    assert result["saved_to_workspace"] is False
    assert result["image_path"] is None
    assert "token=" in result["download_error"]
    assert "sensitive-provider-credential" not in result["download_error"]
