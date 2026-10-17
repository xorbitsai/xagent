from types import SimpleNamespace

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
async def test_provider_metadata_url_rejected_without_a_connection(kind, media_tools):
    image, video, workspace = media_tools
    method = image._download_image if kind == "image" else video._download_video
    with pytest.raises(ValueError):
        await method("http://169.254.169.254/latest/meta-data/")
    assert not list(workspace.output_dir.iterdir())
