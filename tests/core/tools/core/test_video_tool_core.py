import base64
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.config import get_max_upload_size_bytes
from xagent.core.model.video import xinference as xinference_video
from xagent.core.model.video.ark import ArkVideoModel
from xagent.core.model.video.base import BaseVideoModel
from xagent.core.model.video.xinference import XinferenceVideoModel
from xagent.core.tools.adapters.vibe.video_tool import (
    VideoGenerationFunctionTool,
    create_video_tool,
)
from xagent.core.tools.core.video_tool import VideoGenerationToolCore


@pytest.fixture
def mock_video_model():
    model = Mock(spec=BaseVideoModel)
    model.has_ability = Mock(return_value=True)
    model.abilities = ["generate"]
    model.generate_video = AsyncMock(
        return_value={
            "task_id": "task-1",
            "status": "succeeded",
            "video_url": "",
            "seed": 123,
            "duration": 5,
            "ratio": "16:9",
        }
    )
    return model


@pytest.fixture
def mock_workspace(tmp_path):
    workspace = Mock()
    workspace.workspace_dir = tmp_path / "workspace"
    workspace.output_dir = workspace.workspace_dir / "output"
    workspace.output_dir.mkdir(parents=True)

    @contextmanager
    def auto_register_files():
        yield workspace

    workspace.auto_register_files = auto_register_files
    workspace.get_file_id_from_path = Mock(return_value="video-file-id")
    workspace.register_file = Mock(return_value="registered-video-file-id")
    return workspace


def test_generate_video_schema_uses_plain_seconds_field():
    tool_core = VideoGenerationToolCore({}, workspace=None)
    tool = VideoGenerationFunctionTool(
        tool_core.generate_video,
        name="generate_video",
    )

    properties = tool.args_type().model_json_schema()["properties"]

    assert "seconds" in properties
    assert "duration" in properties
    assert "seconds_str" not in properties
    assert "seconds_int" not in properties
    assert "duration_str" not in properties
    seconds_types = {
        option.get("type")
        for option in properties["seconds"].get("anyOf", [])
        if isinstance(option, dict)
    }
    duration_types = {
        option.get("type")
        for option in properties["duration"].get("anyOf", [])
        if isinstance(option, dict)
    }
    assert seconds_types == {"string", "null"}
    assert duration_types == {"string", "null"}


@pytest.mark.asyncio
async def test_generate_video_accepts_model_generated_typed_arg_names(
    mock_video_model,
    mock_workspace,
):
    tool_core = VideoGenerationToolCore(
        {"video-model": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )
    tool = VideoGenerationFunctionTool(
        tool_core.generate_video,
        name="generate_video",
    )

    result = await tool.run_json_async(
        {
            "prompt": "A scenic time-lapse",
            "size": "854x480",
            "seconds_str": "4-5",
            "ratio_str": "16:9",
            "seconds_int": None,
            "duration_int": None,
            "duration_str": None,
        }
    )

    assert result["success"] is True
    assert result["model_used"] == "video-model"
    mock_video_model.generate_video.assert_awaited_once()
    generate_kwargs = mock_video_model.generate_video.await_args.kwargs
    assert generate_kwargs["seconds"] == "4-5"
    assert generate_kwargs["duration"] == "4-5"
    assert generate_kwargs["ratio"] == "16:9"
    assert generate_kwargs["size"] == "854x480"
    assert generate_kwargs["resolution"] == "480p"


@pytest.mark.asyncio
async def test_generate_video_downloads_and_registers_local_video(
    tmp_path: Path, mock_video_model, mock_workspace
):
    source_video = mock_workspace.workspace_dir / "provider-result.mp4"
    source_video.write_bytes(b"fake video data")
    mock_video_model.generate_video.return_value["video_url"] = str(source_video)

    tool = VideoGenerationToolCore(
        {"seedance": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    result = await tool.generate_video(
        "A fast camera push through a neon city",
        duration=5,
        watermark=False,
    )

    assert result["success"] is True
    assert result["task_id"] == "task-1"
    assert result["model_used"] == "seedance"
    assert result["file_id"] == "video-file-id"
    assert result["saved_to_workspace"] is True
    assert result["video_path"]
    assert Path(result["video_path"]).read_bytes() == b"fake video data"
    assert result["artifacts"] == [
        {
            "type": "video",
            "file_id": "video-file-id",
            "filename": Path(result["video_path"]).name,
            "mime_type": "video/mp4",
            "display": "inline",
        }
    ]
    assert result["file_ref"]["file_id"] == "video-file-id"

    mock_video_model.generate_video.assert_awaited_once_with(
        prompt="A fast camera push through a neon city",
        n=1,
        ratio="16:9",
        generate_audio=True,
        watermark=False,
        return_last_frame=False,
        poll_interval=30.0,
        seconds=5,
        duration=5,
    )


@pytest.mark.asyncio
async def test_generate_video_reports_default_wrapper_model_id(mock_workspace):
    registry_model = Mock(spec=BaseVideoModel)
    registry_model.has_ability = Mock(return_value=True)
    registry_model.abilities = ["generate"]
    registry_model.model_name = "raw-provider-name"
    registry_model.generate_video = AsyncMock()

    default_model = Mock(spec=BaseVideoModel)
    default_model.has_ability = Mock(return_value=True)
    default_model.abilities = ["generate"]
    default_model.model_id = "seedance-default"
    default_model.generate_video = AsyncMock(
        return_value={
            "task_id": "task-default",
            "status": "succeeded",
            "video_url": "",
        }
    )

    tool = VideoGenerationToolCore(
        {"seedance-default": registry_model},
        workspace=mock_workspace,
        default_video_model=default_model,
    )

    result = await tool.generate_video("A short test video")

    assert result["success"] is True
    assert result["task_id"] == "task-default"
    assert result["model_used"] == "seedance-default"
    default_model.generate_video.assert_awaited_once()
    registry_model.generate_video.assert_not_called()


@pytest.mark.asyncio
async def test_generate_video_converts_workspace_image_ref_to_data_url_for_ark(
    mock_workspace,
):
    image_path = mock_workspace.output_dir / "frame.jpg"
    image_path.write_bytes(b"image bytes")
    mock_workspace.stage_path_for_external_upload = Mock(return_value=image_path)
    mock_workspace.discard_staged_external_upload = Mock()

    ark_model = ArkVideoModel(api_key="test-key")
    ark_model.generate_video = AsyncMock(
        return_value={
            "task_id": "task-ark",
            "status": "succeeded",
            "video_url": "",
        }
    )

    tool = VideoGenerationToolCore(
        {"ark-video": ark_model},
        workspace=mock_workspace,
        default_video_model=ark_model,
    )

    result = await tool.generate_video(
        "Animate this frame",
        image_url="file:image-file-id",
        seconds=4,
    )

    assert result["success"] is True
    ark_model.generate_video.assert_awaited_once()
    generate_kwargs = ark_model.generate_video.await_args.kwargs
    data_url = generate_kwargs["first_frame_image_url"]
    assert generate_kwargs["input_reference"] == data_url
    assert data_url.startswith("data:image/jpeg;base64,")
    encoded = data_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == b"image bytes"
    mock_workspace.stage_path_for_external_upload.assert_called_once_with(
        "file:image-file-id",
        max_bytes=get_max_upload_size_bytes(),
        host_private=True,
    )
    mock_workspace.discard_staged_external_upload.assert_called_once_with(image_path)


@pytest.mark.asyncio
async def test_generate_video_keeps_workspace_image_ref_for_xinference(mock_workspace):
    xinference_model = XinferenceVideoModel(model_name="Wan2.1-i2v-14B-480p")
    xinference_model.generate_video = AsyncMock(
        return_value={
            "task_id": "task-xinference",
            "status": "succeeded",
            "video_url": "",
        }
    )

    tool = VideoGenerationToolCore(
        {"xinference-video": xinference_model},
        workspace=mock_workspace,
        default_video_model=xinference_model,
    )

    result = await tool.generate_video(
        "Animate this frame",
        image_url="file:image-file-id",
        seconds=4,
    )

    assert result["success"] is True
    xinference_model.generate_video.assert_awaited_once()
    generate_kwargs = xinference_model.generate_video.await_args.kwargs
    assert generate_kwargs["input_reference"] == "file:image-file-id"
    assert generate_kwargs["first_frame_image_url"] == "file:image-file-id"
    assert generate_kwargs["allowed_local_media_roots"] == [
        mock_workspace.workspace_dir
    ]


@pytest.mark.asyncio
async def test_list_video_models_reports_lightning_by_running_uid(
    monkeypatch, mock_workspace
):
    catalog = AsyncMock(
        return_value={
            "custom/fast": {"model_name": "MiniMax-H3", "lightning_inference_steps": 8},
            "custom/four": {"model_name": "MiniMax-H3", "lightning_inference_steps": 4},
            "custom/base": {"model_name": "MiniMax-H3"},
        }
    )
    monkeypatch.setattr(xinference_video, "fetch_running_xinference_models", catalog)
    models: dict[str, BaseVideoModel] = {
        name: XinferenceVideoModel(model_name="MiniMax-H3", model_uid=f"custom/{name}")
        for name in ("fast", "four", "base", "missing")
    }
    models["other"] = ArkVideoModel(model_name="seedance")
    _, list_tool = create_video_tool(models, workspace=mock_workspace)

    result = await list_tool.run_json_async({})

    assert result["success"] is True
    assert {
        model["model_id"]: (
            model["supports_lightning"],
            model["lightning_inference_steps"],
        )
        for model in result["models"]
    } == {
        "fast": (True, 8),
        "four": (True, 4),
        "base": (False, None),
        "missing": (None, None),
        "other": (False, None),
    }
    catalog.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options, path_suffix, expected_mode",
    [
        ({}, "", None),
        ({"use_lightning": True}, "", True),
        (
            {"use_lightning": False, "input_reference": "data:image/png;base64,aW1n"},
            "/image",
            False,
        ),
        (
            {
                "use_lightning_bool": False,
                "first_frame_image_url": "data:image/png;base64,aW1n",
                "last_frame_image_url": "data:image/png;base64,aW1n",
            },
            "/flf",
            False,
        ),
    ],
)
async def test_generate_video_forwards_lightning_without_overriding_steps(
    monkeypatch, mock_workspace, options, path_suffix, expected_mode
):
    catalog = AsyncMock(return_value={"h3": {"lightning_inference_steps": 8}})
    monkeypatch.setattr(xinference_video, "fetch_running_xinference_models", catalog)
    model = XinferenceVideoModel(model_name="MiniMax-H3", model_uid="h3")
    post = AsyncMock(return_value={"data": [{"b64_json": "dmlkZW8="}]})
    monkeypatch.setattr(
        model, "_post_video_form" if path_suffix else "_post_video_json", post
    )
    generate_tool, _ = create_video_tool({"h3": model}, workspace=mock_workspace)

    result = await generate_tool.run_json_async(
        {"prompt": "Animate this scene", **options}
    )

    assert result["success"] is True
    post.assert_awaited_once()
    _, path, body = post.await_args.args[:3]
    assert path == f"/v1/video/generations{path_suffix}"
    assert body["model"] == "h3"
    kwargs = json.loads(body["kwargs"])
    assert "num_inference_steps" not in kwargs
    if expected_mode is None:
        assert "use_lightning" not in kwargs
        catalog.assert_not_awaited()
    else:
        assert kwargs["use_lightning"] is expected_mode
        catalog.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "catalog, error_category",
    [
        ({"h3": {"lightning_inference_steps": True}}, "invalid_request"),
        ({"h3": {}}, "invalid_request"),
        ({}, "provider_unavailable"),
    ],
)
async def test_generate_video_requires_advertised_lightning_support(
    monkeypatch, mock_workspace, catalog, error_category
):
    monkeypatch.setattr(
        xinference_video,
        "fetch_running_xinference_models",
        AsyncMock(return_value=catalog),
    )
    model = XinferenceVideoModel(model_name="h3")
    generate = AsyncMock()
    monkeypatch.setattr(model, "generate_video", generate)
    generate_tool, _ = create_video_tool({"h3": model}, workspace=mock_workspace)

    result = await generate_tool.run_json_async(
        {"prompt": "Animate this scene", "use_lightning": False}
    )

    assert result["success"] is False
    assert result["error_category"] == error_category
    generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_download_video_rejects_local_video_outside_workspace(
    tmp_path: Path, mock_video_model, mock_workspace
):
    source_video = tmp_path / "outside.mp4"
    source_video.write_bytes(b"fake video data")
    tool = VideoGenerationToolCore(
        {"seedance": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    with pytest.raises(ValueError, match="outside the workspace"):
        await tool._download_video(str(source_video))


@pytest.mark.asyncio
async def test_download_video_accepts_workspace_file_uri_with_escaped_path(
    mock_video_model, mock_workspace
):
    source_video = mock_workspace.workspace_dir / "provider result.mp4"
    source_video.write_bytes(b"fake video data")
    tool = VideoGenerationToolCore(
        {"seedance": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    video_path = await tool._download_video(source_video.as_uri())

    assert Path(video_path).read_bytes() == b"fake video data"


@pytest.mark.asyncio
async def test_download_video_resolves_relative_local_path_against_workspace(
    mock_video_model,
    mock_workspace,
):
    source_video = mock_workspace.workspace_dir / "provider-result.mp4"
    source_video.write_bytes(b"fake video data")
    tool = VideoGenerationToolCore(
        {"seedance": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    video_path = await tool._download_video("provider-result.mp4")

    assert Path(video_path).read_bytes() == b"fake video data"


@pytest.mark.asyncio
async def test_download_video_skips_copy_when_source_is_destination(
    mock_video_model, mock_workspace
):
    source_video = mock_workspace.output_dir / "generated_video.mp4"
    source_video.write_bytes(b"fake video data")
    tool = VideoGenerationToolCore(
        {"seedance": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    video_path = await tool._download_video(
        str(source_video), filename=source_video.name
    )

    assert Path(video_path) == source_video
    assert source_video.read_bytes() == b"fake video data"


@pytest.mark.asyncio
async def test_generate_video_supports_openai_compatible_parameters(
    mock_video_model, mock_workspace
):
    tool = VideoGenerationToolCore(
        {"video-model": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    result = await tool.generate_video(
        "A product reveal",
        seconds=5,
        size="720x1280",
        input_reference="https://example.com/frame.png",
        negative_prompt="blur",
        n=1,
    )

    assert result["success"] is True
    mock_video_model.generate_video.assert_awaited_once_with(
        prompt="A product reveal",
        n=1,
        ratio="9:16",
        generate_audio=True,
        watermark=True,
        return_last_frame=False,
        poll_interval=30.0,
        seconds=5,
        duration=5,
        size="720x1280",
        resolution="720p",
        negative_prompt="blur",
        input_reference="https://example.com/frame.png",
        first_frame_image_url="https://example.com/frame.png",
    )


@pytest.mark.asyncio
async def test_generate_video_preserves_explicit_ratio_with_size(
    mock_video_model, mock_workspace
):
    tool = VideoGenerationToolCore(
        {"video-model": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    result = await tool.generate_video(
        "A landscape clip",
        seconds=5,
        size="720x1280",
        ratio="16:9",
    )

    assert result["success"] is True
    mock_video_model.generate_video.assert_awaited_once()
    generate_kwargs = mock_video_model.generate_video.await_args.kwargs
    assert generate_kwargs["ratio"] == "16:9"
    assert generate_kwargs["size"] == "720x1280"
    assert generate_kwargs["resolution"] == "720p"


@pytest.mark.asyncio
async def test_generate_video_saves_base64_video(mock_video_model, mock_workspace):
    video_payload = base64.b64encode(b"video bytes").decode()
    mock_video_model.generate_video.return_value["video_url"] = (
        f"data:video/mp4;base64,{video_payload}"
    )

    tool = VideoGenerationToolCore(
        {"video-model": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    result = await tool.generate_video("A short clip")

    assert result["success"] is True
    assert result["saved_to_workspace"] is True
    assert Path(result["video_path"]).read_bytes() == b"video bytes"


@pytest.mark.asyncio
async def test_download_video_uses_data_url_mime_for_generated_filename(
    mock_video_model,
    mock_workspace,
):
    video_payload = base64.b64encode(b"video bytes").decode()
    tool = VideoGenerationToolCore(
        {"video-model": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    video_path = await tool._download_video(f"data:video/webm;base64,{video_payload}")

    assert Path(video_path).suffix == ".webm"
    assert Path(video_path).read_bytes() == b"video bytes"


@pytest.mark.asyncio
async def test_generate_video_returns_error_without_models(mock_workspace):
    tool = VideoGenerationToolCore({}, workspace=mock_workspace)

    result = await tool.generate_video("A product teaser")

    assert result["success"] is False
    assert result["error"] == "No available video models configured"
    assert result["video_path"] is None
    assert result["model_used"] == "default"


@pytest.mark.asyncio
async def test_generate_video_reports_explicit_model_without_generate_ability(
    mock_video_model,
    mock_workspace,
):
    incapable_model = Mock(spec=BaseVideoModel)
    incapable_model.has_ability = Mock(return_value=False)
    incapable_model.abilities = ["embed"]
    incapable_model.generate_video = AsyncMock()
    tool = VideoGenerationToolCore(
        {"incapable": incapable_model, "seedance": mock_video_model},
        workspace=mock_workspace,
        default_video_model=mock_video_model,
    )

    result = await tool.generate_video("A product teaser", model_id="incapable")

    assert result["success"] is False
    assert result["model_used"] == "incapable"
    assert (
        result["error"] == "Video model 'incapable' does not support video generation; "
        "available video generation models: seedance"
    )
    incapable_model.generate_video.assert_not_called()
    mock_video_model.generate_video.assert_not_called()


def test_size_from_resolution_ratio_tolerates_non_string_resolution():
    resolution: Any = 720
    assert (
        VideoGenerationToolCore._size_from_resolution_ratio(resolution, "16:9") is None
    )


def test_normalize_seconds_preserves_fractional_numeric_values():
    assert VideoGenerationToolCore._normalize_seconds("4.5") == 4.5
    assert VideoGenerationToolCore._normalize_seconds(4.5) == 4.5
    assert VideoGenerationToolCore._normalize_seconds("5") == 5


def _ark_video_tool(mock_workspace, model_name=None, result=None):
    ark_model = ArkVideoModel(
        model_name=model_name or "doubao-seedance-2-0-fast-260128",
        api_key="test-key",
    )
    if result is None:
        ark_model.generate_video = AsyncMock(
            return_value={"task_id": "task-ark", "status": "succeeded", "video_url": ""}
        )
    else:
        ark_model.generate_video = AsyncMock(return_value=result)
    tool = VideoGenerationToolCore(
        {"ark-video": ark_model},
        workspace=mock_workspace,
        default_video_model=ark_model,
    )
    return tool, ark_model


@pytest.mark.asyncio
async def test_seconds_below_the_shortest_length_are_raised_to_it(mock_workspace):
    tool, ark_model = _ark_video_tool(
        mock_workspace,
        result={"task_id": "task-ark", "status": "succeeded", "video_url": ""},
    )

    result = await tool.generate_video("A three-second idea", seconds=3)

    assert result["success"] is True
    generate_kwargs = ark_model.generate_video.await_args.kwargs
    assert generate_kwargs["seconds"] == 4
    assert generate_kwargs["duration"] == 4
    # The result says which length was actually made, and why, so the model can
    # tell the user instead of implying the 3 seconds they asked for.
    assert result["duration"] == 4
    assert result["duration_adjusted"] == {"requested": 3, "used": 4}
    assert "shortest supported clip is 4 seconds" in result["duration_note"]


@pytest.mark.asyncio
async def test_seconds_above_the_longest_length_are_refused_not_truncated(
    mock_workspace,
):
    tool, ark_model = _ark_video_tool(mock_workspace)

    result = await tool.generate_video("A twenty-second idea", seconds=20)

    assert result["success"] is False
    assert result["error_category"] == "invalid_request"
    assert result["allowed_values"] == {"duration_seconds": "4-15"}
    ark_model.generate_video.assert_not_called()


@pytest.mark.asyncio
async def test_the_1_5_pro_family_reports_its_own_longest_length(mock_workspace):
    tool, ark_model = _ark_video_tool(
        mock_workspace, model_name="doubao-seedance-1-5-pro-250528"
    )

    result = await tool.generate_video("A fourteen-second idea", seconds=14)

    assert result["success"] is False
    assert result["error_category"] == "invalid_request"
    assert result["allowed_values"] == {"duration_seconds": "4-12"}
    ark_model.generate_video.assert_not_called()


@pytest.mark.asyncio
async def test_intelligent_duration_is_forwarded_untouched(mock_workspace):
    """``-1`` means "let the provider choose", not "one second short"."""
    tool, ark_model = _ark_video_tool(mock_workspace)

    result = await tool.generate_video("Let the model choose", seconds=-1)

    assert result["success"] is True
    assert ark_model.generate_video.await_args.kwargs["seconds"] == -1
    assert "duration_adjusted" not in result


@pytest.mark.asyncio
async def test_other_video_models_keep_the_length_they_are_given(mock_workspace):
    """Only Seedance states a supported range, and a guess is worse than none."""
    xinference_model = XinferenceVideoModel(model_name="Wan2.1-t2v-14B-480p")
    xinference_model.generate_video = AsyncMock(
        return_value={"task_id": "task-x", "status": "succeeded", "video_url": ""}
    )
    tool = VideoGenerationToolCore(
        {"xinference-video": xinference_model},
        workspace=mock_workspace,
        default_video_model=xinference_model,
    )

    result = await tool.generate_video("A three-second idea", seconds=3)

    assert result["success"] is True
    assert xinference_model.generate_video.await_args.kwargs["seconds"] == 3
    assert "duration_adjusted" not in result


@pytest.mark.asyncio
async def test_a_failed_generation_does_not_carry_the_internal_model_id(
    mock_workspace, caplog
):
    """The model id is a diagnostic: it goes to the log, not to the model."""
    tool, ark_model = _ark_video_tool(mock_workspace)
    ark_model.generate_video = AsyncMock(side_effect=RuntimeError("provider exploded"))

    with caplog.at_level(logging.INFO):
        result = await tool.generate_video("A short clip", seconds=4)

    assert result["success"] is False
    assert "model_used" not in result
    assert "ark-video" in caplog.text


@pytest.mark.asyncio
async def test_the_provider_reported_length_wins_over_the_adjusted_request(
    mock_workspace,
):
    """A clip asked for at 4 seconds can come back at 5; the note says 5."""
    tool, ark_model = _ark_video_tool(
        mock_workspace,
        result={
            "task_id": "task-ark",
            "status": "succeeded",
            "video_url": "",
            "duration": 5,
        },
    )

    result = await tool.generate_video("A three-second idea", seconds=3)

    assert ark_model.generate_video.await_args.kwargs["seconds"] == 4
    assert result["duration"] == 5
    assert result["duration_adjusted"] == {"requested": 3, "used": 5}
    assert "shortest supported clip is 4 seconds" in result["duration_note"]
    assert result["duration_note"].endswith("this clip is 5 seconds long.")


@pytest.mark.asyncio
async def test_a_seedance_version_without_a_recorded_range_keeps_the_length(
    mock_workspace,
):
    """Seedance 2.5 accepts 4-30 seconds, so the 2.0 bounds must not clamp it."""
    tool, ark_model = _ark_video_tool(
        mock_workspace, model_name="doubao-seedance-2-5-pro-260101"
    )

    result = await tool.generate_video("A short clip", seconds=2)

    assert result["success"] is True
    assert ark_model.generate_video.await_args.kwargs["seconds"] == 2
    assert "duration_adjusted" not in result


@pytest.mark.asyncio
async def test_a_seedance_version_without_a_recorded_range_is_not_refused(
    mock_workspace,
):
    """No known range means no local refusal: the provider decides."""
    tool, ark_model = _ark_video_tool(
        mock_workspace, model_name="doubao-seedance-2-5-pro-260101"
    )

    result = await tool.generate_video("A long clip", seconds=20)

    assert result["success"] is True
    assert ark_model.generate_video.await_args.kwargs["seconds"] == 20
    assert "allowed_values" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [0, -2, 4.5])
async def test_a_length_that_cannot_be_a_clip_is_refused_locally(
    mock_workspace, length
):
    tool, ark_model = _ark_video_tool(mock_workspace)

    result = await tool.generate_video("A clip", seconds=length)

    assert result["success"] is False
    assert result["error_category"] == "invalid_request"
    assert result["allowed_values"] == {"duration_seconds": "4-15"}
    ark_model.generate_video.assert_not_called()


@pytest.mark.asyncio
async def test_a_zero_length_is_refused_for_a_model_with_no_known_range(
    mock_workspace,
):
    xinference_model = XinferenceVideoModel(model_name="Wan2.1-t2v-14B-480p")
    xinference_model.generate_video = AsyncMock(
        return_value={"task_id": "task-x", "status": "succeeded", "video_url": ""}
    )
    tool = VideoGenerationToolCore(
        {"xinference-video": xinference_model},
        workspace=mock_workspace,
        default_video_model=xinference_model,
    )

    result = await tool.generate_video("A clip", seconds=0)

    assert result["success"] is False
    assert result["error_category"] == "invalid_request"
    assert "allowed_values" not in result
    xinference_model.generate_video.assert_not_called()
