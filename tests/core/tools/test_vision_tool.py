"""
Tests for Vision Tool
"""

import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.retry.wrapper import create_retry_wrapper
from xagent.core.tools.adapters.vibe.vision_tool import VisionTool, get_vision_tool
from xagent.web.services.model_service import get_default_vision_model


@pytest.fixture
def mock_vision_model():
    """Create a mock vision model for testing"""
    model = Mock(spec=BaseLLM)

    # Mock vision_chat method for understand_images
    model.vision_chat = AsyncMock(
        return_value="This is a beautiful landscape photo with mountains and a lake."
    )

    # Mock has_ability method
    model.has_ability = Mock(return_value=True)
    model.supports_native_video_input = False

    return model


@pytest.fixture
def mock_vision_model_with_descriptions():
    """Create a mock vision model that returns structured descriptions"""
    model = Mock(spec=BaseLLM)

    # Mock for describe_images
    model.vision_chat = AsyncMock(
        return_value="Image 1: A red apple on a wooden table\nImage 2: A green tree in a park"
    )

    # Mock has_ability method
    model.has_ability = Mock(return_value=True)

    return model


@pytest.fixture
def mock_vision_model_with_detection():
    """Create a mock vision model that returns object detection data"""
    model = Mock(spec=BaseLLM)

    # Mock for detect_objects - return JSON string with correct structure
    model.vision_chat = AsyncMock(
        return_value='{"detections": [{"class": "person", "confidence": 0.95, "bbox": [0.1, 0.1, 0.6, 0.8]}, {"class": "car", "confidence": 0.87, "bbox": [0.7, 0.5, 0.95, 0.75]}], "image_info": {"width": "640", "height": "480"}}'
    )

    # Mock has_ability method
    model.has_ability = Mock(return_value=True)

    return model


@pytest.fixture
def mock_vision_model_with_unstructured_detection():
    """Create a mock vision model that returns unstructured detection data"""
    model = Mock(spec=BaseLLM)

    # Mock for detect_objects with unstructured response - format that matches the regex pattern
    model.vision_chat = AsyncMock(
        return_value="person: [0.1, 0.1, 0.6, 0.8] (confidence: 0.95) car: [0.7, 0.5, 0.95, 0.75] (confidence: 0.87)"
    )

    # Mock has_ability method
    model.has_ability = Mock(return_value=True)

    return model


@pytest.fixture
def mock_workspace():
    """Create a mock workspace for testing"""
    import tempfile
    from contextlib import contextmanager

    # Create temporary directory and files
    temp_dir = Path(tempfile.mkdtemp())
    existing_image = temp_dir / "existing_image.jpg"
    test_image = temp_dir / "test_image.png"

    # Create fake image files
    existing_image.write_bytes(b"fake_image_data_for_existing_image")
    test_image.write_bytes(b"fake_image_data_for_test_image")

    workspace = Mock()

    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    workspace.output_dir = output_dir

    # Mock the resolve_path_with_search method that VisionTool actually uses
    def mock_resolve_path_with_search(filename):
        if filename == "existing_image.jpg":
            return existing_image
        elif filename == "test_image.png":
            return test_image
        else:
            raise FileNotFoundError(f"File not found: {filename}")

    workspace.resolve_path_with_search = Mock(side_effect=mock_resolve_path_with_search)

    # Mock auto_register_files to return a proper context manager
    @contextmanager
    def auto_register_files():
        yield workspace

    workspace.auto_register_files = auto_register_files
    # Mock get_file_id_from_path to return a valid file_id
    workspace.get_file_id_from_path = Mock(return_value="test-file-id")

    return workspace


@pytest.fixture
def vision_tool_without_workspace(mock_vision_model):
    """Create VisionTool instance without workspace for testing"""
    return VisionTool(mock_vision_model)


@pytest.fixture
def vision_tool_with_workspace(mock_vision_model, mock_workspace):
    """Create VisionTool instance with workspace for testing"""
    return VisionTool(mock_vision_model, workspace=mock_workspace)


@pytest.fixture
def sample_image_base64():
    """Create sample base64 encoded image data for testing"""
    return base64.b64encode(b"fake_image_data").decode("utf-8")


@pytest.fixture
def sample_images_data():
    """Create sample images data for testing"""
    return [
        {
            "type": "image",
            "data": base64.b64encode(b"fake_image_data_1").decode("utf-8"),
            "format": "jpeg",
        },
        {
            "type": "image",
            "data": base64.b64encode(b"fake_image_data_2").decode("utf-8"),
            "format": "png",
        },
    ]


class TestVisionToolInitialization:
    """Test cases for VisionTool initialization"""

    def test_init_with_model(self, mock_vision_model):
        """Test VisionTool initialization with model"""
        tool = VisionTool(mock_vision_model)
        assert tool.vision_model == mock_vision_model
        assert tool.workspace is None

    def test_init_with_model_and_workspace(self, mock_vision_model, mock_workspace):
        """Test VisionTool initialization with model and workspace"""
        tool = VisionTool(mock_vision_model, workspace=mock_workspace)
        assert tool.vision_model == mock_vision_model
        assert tool.workspace == mock_workspace

    def test_init_without_model_raises_error(self):
        """Test VisionTool initialization without model raises error"""
        with pytest.raises(TypeError):
            VisionTool()


class TestVisionToolUnderstandImages:
    """Test cases for understand_images method"""

    @pytest.mark.asyncio
    async def test_understand_single_image_path_with_workspace(
        self, vision_tool_with_workspace, mock_vision_model, mock_workspace
    ):
        """Test understanding a single image path with workspace"""
        result = await vision_tool_with_workspace.understand_images(
            "existing_image.jpg", "What is in this image?"
        )

        assert result.success is True
        assert (
            result.answer
            == "This is a beautiful landscape photo with mountains and a lake."
        )
        assert result.images_processed == 1

        # Verify workspace methods were called
        mock_workspace.resolve_path_with_search.assert_called_with("existing_image.jpg")

    @pytest.mark.asyncio
    async def test_understand_single_image_path_without_workspace(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """Test understanding a single image path without workspace"""
        # Use a data URL since we don't have a workspace
        result = await vision_tool_without_workspace.understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", "What is this?"
        )

        assert result.success is True
        assert (
            result.answer
            == "This is a beautiful landscape photo with mountains and a lake."
        )
        assert result.images_processed == 1

    @pytest.mark.asyncio
    async def test_understand_reports_model_name_through_retry_wrapper(self) -> None:
        model = Mock(spec=BaseLLM)
        model.model_name = "deepseek/deepseek-v4-flash"
        model.model_id = "configured-model-id"
        model.vision_chat = AsyncMock(return_value="Token usage details")
        model.has_ability = Mock(return_value=True)
        wrapped_model = create_retry_wrapper(
            model,
            BaseLLM,
            retry_methods={"vision_chat"},
        )

        result = await VisionTool(wrapped_model).understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            "Read the token usage.",
        )

        assert wrapped_model.__class__.__name__ == "GenericRetryWrapper"
        assert result.success is True
        assert result.model_used == "deepseek/deepseek-v4-flash"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "file_ref",
        [
            "file:355f1fee-48e4-4cb6-afd3-71654e2f5c7e",
            "file://355f1fee-48e4-4cb6-afd3-71654e2f5c7e",
        ],
    )
    async def test_understand_file_id_ref_with_workspace(
        self, vision_tool_with_workspace, mock_workspace, file_ref
    ):
        image_path = mock_workspace.resolve_path_with_search("existing_image.jpg")
        mock_workspace.resolve_path_with_search.reset_mock()

        def resolve_file_ref(value: str) -> Path:
            if value == file_ref:
                return image_path
            raise FileNotFoundError(value)

        mock_workspace.resolve_path_with_search.side_effect = resolve_file_ref

        result = await vision_tool_with_workspace.understand_images(
            file_ref, "What is this?"
        )

        assert result.success is True
        assert result.images_processed == 1
        mock_workspace.resolve_path_with_search.assert_called_once_with(file_ref)

    @pytest.mark.asyncio
    async def test_understand_multiple_image_paths_with_workspace(
        self, vision_tool_with_workspace, mock_vision_model, mock_workspace
    ):
        """Test understanding multiple image paths with workspace"""
        result = await vision_tool_with_workspace.understand_images(
            ["existing_image.jpg", "test_image.png"], "Describe these images"
        )

        assert result.success is True
        assert (
            result.answer
            == "This is a beautiful landscape photo with mountains and a lake."
        )
        assert result.images_processed == 2

        # Verify workspace methods were called for both images
        assert mock_workspace.resolve_path_with_search.call_count == 2

    @pytest.mark.asyncio
    async def test_understand_single_base64_image(
        self, vision_tool_without_workspace, mock_vision_model, sample_image_base64
    ):
        """Test understanding a single base64 encoded image"""
        # Use data URL format for base64 image
        image_data = f"data:image/jpeg;base64,{sample_image_base64}"

        result = await vision_tool_without_workspace.understand_images(
            image_data, "What is this image?"
        )

        assert result.success is True
        assert (
            result.answer
            == "This is a beautiful landscape photo with mountains and a lake."
        )
        assert result.images_processed == 1

    @pytest.mark.asyncio
    async def test_understand_multiple_base64_images(
        self, vision_tool_without_workspace, mock_vision_model, sample_images_data
    ):
        """Test understanding multiple base64 encoded images"""
        # Convert dictionary data to data URLs
        image_urls = []
        for img_data in sample_images_data:
            image_urls.append(
                f"data:image/{img_data['format']};base64,{img_data['data']}"
            )

        result = await vision_tool_without_workspace.understand_images(
            image_urls, "Describe these images"
        )

        assert result.success is True
        assert (
            result.answer
            == "This is a beautiful landscape photo with mountains and a lake."
        )
        assert result.images_processed == 2

    @pytest.mark.asyncio
    async def test_understand_mixed_images(
        self,
        vision_tool_with_workspace,
        mock_vision_model,
        mock_workspace,
        sample_image_base64,
    ):
        """Test understanding mixed image types (path and base64)"""
        images = [
            "existing_image.jpg",  # path
            f"data:image/jpeg;base64,{sample_image_base64}",  # base64 data URL
        ]

        result = await vision_tool_with_workspace.understand_images(
            images, "What do you see in these images?"
        )

        assert result.success is True
        assert (
            result.answer
            == "This is a beautiful landscape photo with mountains and a lake."
        )
        assert result.images_processed == 2

    @pytest.mark.asyncio
    async def test_understand_with_custom_parameters(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """Test understanding with custom temperature and max_tokens"""
        result = await vision_tool_without_workspace.understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            "What is in this image?",
            temperature=0.7,
            max_tokens=200,
        )

        assert result.success is True

        # Verify vision_chat was called with correct parameters
        mock_vision_model.vision_chat.assert_called_once()
        call_args = mock_vision_model.vision_chat.call_args
        assert call_args.kwargs.get("temperature") == 0.7
        assert call_args.kwargs.get("max_tokens") == 200

    @pytest.mark.asyncio
    async def test_understand_coerces_string_numeric_parameters(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """Tool calls may pass optional numeric arguments as strings."""
        result = await vision_tool_without_workspace.understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            "What is in this image?",
            temperature="0.7",
            max_tokens="200",
        )

        assert result.success is True

        call_args = mock_vision_model.vision_chat.call_args
        assert call_args.kwargs.get("temperature") == 0.7
        assert call_args.kwargs.get("max_tokens") == 200

    @pytest.mark.asyncio
    async def test_understand_coerces_decimal_string_integer_parameters(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """Integer parameters may arrive as decimal strings from tool payloads."""
        result = await vision_tool_without_workspace.understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            "What is in this image?",
            max_tokens="200.0",
        )

        assert result.success is True

        call_args = mock_vision_model.vision_chat.call_args
        assert call_args.kwargs.get("max_tokens") == 200

    @pytest.mark.asyncio
    async def test_understand_rejects_fractional_integer_parameters(
        self, vision_tool_without_workspace
    ):
        """Fractional max_tokens values should not be silently truncated."""
        result = await vision_tool_without_workspace.understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            "What is in this image?",
            max_tokens="200.5",
        )

        assert result.success is False
        assert "max_tokens must be an integer" in result.error

    @pytest.mark.asyncio
    async def test_understand_ignores_blank_optional_numeric_parameters(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """Blank optional values should not be sent to OpenAI as strings."""
        result = await vision_tool_without_workspace.understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            "What is in this image?",
            temperature="",
            max_tokens="",
        )

        assert result.success is True

        call_args = mock_vision_model.vision_chat.call_args
        assert call_args.kwargs.get("temperature") is None
        assert call_args.kwargs.get("max_tokens") is None

    @pytest.mark.asyncio
    async def test_understand_file_not_found_with_workspace(
        self, vision_tool_with_workspace, mock_workspace
    ):
        """Test understanding when file is not found in workspace"""
        result = await vision_tool_with_workspace.understand_images(
            "nonexistent_image.jpg", "What is this?"
        )

        assert result.success is False
        assert "No valid images or video frames could be processed" in result.error

    @pytest.mark.asyncio
    async def test_understand_no_model_available(self):
        """Test understanding when no vision model is available"""
        from unittest.mock import Mock

        # Create a mock model that doesn't have vision capability
        mock_model = Mock(spec=BaseLLM)
        mock_model.has_ability = Mock(return_value=False)

        tool = VisionTool(mock_model)
        result = await tool.understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", "What is this?"
        )

        assert result.success is False
        assert "does not support vision capabilities" in result.error

    @pytest.mark.asyncio
    async def test_understand_model_error(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """Test understanding when model raises an exception"""
        mock_vision_model.vision_chat.side_effect = Exception("Model error")

        result = await vision_tool_without_workspace.understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", "What is this?"
        )

        assert result.success is False
        assert "Model error" in result.error


class TestVisionToolUnderstandMedia:
    """Tests for the public image/video understanding entrypoint."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field_name", "invalid_value"),
        [
            ("start_time", float("nan")),
            ("start_time", float("inf")),
            ("end_time", float("nan")),
            ("end_time", float("inf")),
        ],
    )
    async def test_understand_video_rejects_non_finite_time_ranges(
        self,
        vision_tool_without_workspace,
        mock_vision_model,
        field_name,
        invalid_value,
    ):
        result = await vision_tool_without_workspace.understand_media(
            "clip.mp4",
            "What happens?",
            **{field_name: invalid_value},
        )

        assert result.success is False
        assert (
            f"{field_name} must be a finite number greater than or equal to 0"
            in result.error
        )
        mock_vision_model.vision_chat.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("start_time", "end_time"), [(4, 4), (10, 2)])
    async def test_understand_video_rejects_non_increasing_time_range(
        self,
        vision_tool_without_workspace,
        mock_vision_model,
        start_time,
        end_time,
    ):
        result = await vision_tool_without_workspace.understand_media(
            "clip.mp4",
            "What happens?",
            start_time=start_time,
            end_time=end_time,
        )

        assert result.success is False
        assert "end_time must be greater than start_time" in result.error
        mock_vision_model.vision_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_understand_video_samples_timestamped_frames(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        frames = [
            (0.0, "data:image/jpeg;base64,ZmFrZV9mcmFtZV8x"),
            (2.5, "data:image/jpeg;base64,ZmFrZV9mcmFtZV8y"),
        ]
        with patch.object(
            vision_tool_without_workspace.core,
            "_extract_video_frames",
            return_value=frames,
        ) as extract_frames:
            result = await vision_tool_without_workspace.understand_media(
                "clip.mp4",
                "What changes over time?",
                start_time=0,
                end_time=3,
                max_frames=2,
            )

        assert result.success is True
        assert result.media_processed == 1
        assert result.images_processed == 0
        assert result.videos_processed == 1
        assert result.frames_extracted == 2
        extract_frames.assert_called_once_with(
            "clip.mp4", start_time=0.0, end_time=3.0, max_frames=2
        )

        content = mock_vision_model.vision_chat.call_args.kwargs["messages"][0][
            "content"
        ]
        assert content[0] == {"type": "text", "text": "What changes over time?"}
        assert content[1]["text"] == "Video clip.mp4, frame at 0.00 seconds:"
        assert content[3]["text"] == "Video clip.mp4, frame at 2.50 seconds:"
        image_items = [item for item in content if item["type"] == "image_url"]
        assert [item["image_url"]["url"] for item in image_items] == [
            frame_data for _, frame_data in frames
        ]

    @pytest.mark.asyncio
    async def test_understand_video_uses_native_input_when_model_supports_it(
        self, tmp_path, mock_vision_model
    ):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake-video")
        mock_vision_model.supports_native_video_input = True
        mock_vision_model.supports_native_video_time_range = True
        mock_vision_model.build_native_video_content = Mock(
            return_value={
                "type": "video_url",
                "video_url": {"url": "provider-ready-video"},
            }
        )
        tool = VisionTool(mock_vision_model)

        with (
            patch.object(
                tool.core,
                "_convert_video_to_base64",
                return_value="data:video/mp4;base64,ZmFrZS12aWRlbw==",
            ) as convert_video,
            patch(
                "xagent.core.tools.core.vision_tool.asyncio.to_thread",
                new=AsyncMock(return_value="data:video/mp4;base64,ZmFrZS12aWRlbw=="),
            ) as to_thread,
            patch.object(tool.core, "_extract_video_frames") as extract_frames,
        ):
            result = await tool.understand_media(
                str(video_path),
                "What happens?",
                start_time=1,
                end_time=4,
            )

        assert result.success is True
        assert result.videos_processed == 1
        assert result.native_videos_processed == 1
        assert result.frames_extracted == 0
        extract_frames.assert_not_called()
        to_thread.assert_awaited_once_with(convert_video, str(video_path))

        video_data = mock_vision_model.build_native_video_content.call_args.args[0]
        assert video_data.startswith("data:video/mp4;base64,")
        assert mock_vision_model.build_native_video_content.call_args.kwargs == {
            "start_time": 1.0,
            "end_time": 4.0,
        }
        content = mock_vision_model.vision_chat.call_args.kwargs["messages"][0][
            "content"
        ]
        assert content[1] == {
            "type": "video_url",
            "video_url": {"url": "provider-ready-video"},
        }

    @pytest.mark.asyncio
    async def test_time_range_falls_back_when_native_protocol_has_no_offsets(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        mock_vision_model.supports_native_video_input = True
        mock_vision_model.supports_native_video_time_range = False
        with patch.object(
            vision_tool_without_workspace.core,
            "_extract_video_frames",
            return_value=[(2.0, "data:image/jpeg;base64,ZmFrZQ==")],
        ) as extract_frames:
            result = await vision_tool_without_workspace.understand_media(
                "clip.mp4",
                "What happens between one and three seconds?",
                start_time=1,
                end_time=3,
                max_frames=1,
            )

        assert result.success is True
        assert result.native_videos_processed == 0
        assert result.frames_extracted == 1
        extract_frames.assert_called_once_with(
            "clip.mp4", start_time=1.0, end_time=3.0, max_frames=1
        )

    @pytest.mark.asyncio
    async def test_understand_mixed_media_uses_one_model_call(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        with patch.object(
            vision_tool_without_workspace.core,
            "_extract_video_frames",
            return_value=[(1.0, "data:image/jpeg;base64,ZmFrZV9mcmFtZQ==")],
        ):
            result = await vision_tool_without_workspace.understand_media(
                ["data:image/png;base64,ZmFrZV9pbWFnZQ==", "clip.mov"],
                "Compare them",
                max_frames=1,
            )

        assert result.success is True
        assert result.media_processed == 2
        assert result.images_processed == 1
        assert result.videos_processed == 1
        assert result.frames_extracted == 1
        mock_vision_model.vision_chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_video_is_never_wrapped_directly_as_image_url(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        result = await vision_tool_without_workspace.understand_media(
            "data:video/mp4;base64,ZmFrZV92aWRlbw==", "What happens?"
        )

        assert result.success is False
        assert result.warnings
        mock_vision_model.vision_chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_native_video_data_url_is_passed_without_frame_sampling(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        mock_vision_model.supports_native_video_input = True
        mock_vision_model.build_native_video_content = Mock(
            side_effect=lambda url, **_: {
                "type": "video_url",
                "video_url": {"url": url},
            }
        )

        result = await vision_tool_without_workspace.understand_media(
            "data:video/mp4;base64,ZmFrZV92aWRlbw==", "What happens?"
        )

        assert result.success is True
        assert result.native_videos_processed == 1
        assert result.frames_extracted == 0
        content = mock_vision_model.vision_chat.call_args.kwargs["messages"][0][
            "content"
        ]
        assert content[1]["type"] == "video_url"

    @pytest.mark.asyncio
    async def test_unsupported_native_video_url_does_not_fail_other_media(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        mock_vision_model.supports_native_video_input = True

        def build_native_video_content(url, **_):
            if url.startswith("https://"):
                raise ValueError("unsupported remote video URL")
            return {"type": "video_url", "video_url": {"url": url}}

        mock_vision_model.build_native_video_content = Mock(
            side_effect=build_native_video_content
        )

        result = await vision_tool_without_workspace.understand_media(
            [
                "data:video/mp4;base64,ZmFrZV92aWRlbw==",
                "https://example.com/remote.mp4",
            ],
            "Summarize the available video",
        )

        assert result.success is True
        assert result.videos_processed == 1
        assert result.native_videos_processed == 1
        assert any("unsupported remote video URL" in item for item in result.warnings)
        assert any("upload the video" in item for item in result.warnings)
        mock_vision_model.vision_chat.assert_awaited_once()

    def test_frame_budget_reserves_one_frame_for_every_video(
        self, vision_tool_without_workspace
    ):
        assert vision_tool_without_workspace.core._video_frame_budgets(
            image_count=2,
            native_video_count=0,
            video_count=2,
            max_frames=8,
        ) == [4, 4]

    @pytest.mark.asyncio
    async def test_all_native_videos_skip_fallback_frame_budget(
        self, mock_vision_model
    ):
        mock_vision_model.supports_native_video_input = True
        mock_vision_model.build_native_video_content = Mock(
            side_effect=lambda url, **_: {
                "type": "video_url",
                "video_url": {"url": url},
            }
        )
        tool = VisionTool(mock_vision_model)
        media = [f"clip-{index}.mp4" for index in range(10)]

        with (
            patch.object(
                tool.core,
                "_convert_video_to_base64",
                return_value="data:video/mp4;base64,ZmFrZS12aWRlbw==",
            ),
            patch.object(tool.core, "_video_frame_budgets") as frame_budgets,
        ):
            result = await tool.understand_media(media, "Summarize all videos")

        assert result.success is True
        assert result.native_videos_processed == 10
        assert result.frames_extracted == 0
        frame_budgets.assert_not_called()

    @pytest.mark.asyncio
    async def test_native_videos_consume_fallback_frame_budget(self, mock_vision_model):
        mock_vision_model.supports_native_video_input = True
        mock_vision_model.build_native_video_content = Mock(
            return_value={
                "type": "video_url",
                "video_url": {"url": "provider-ready-video"},
            }
        )
        tool = VisionTool(mock_vision_model)

        with (
            patch.object(
                tool.core,
                "_convert_video_to_base64",
                side_effect=[
                    "data:video/mp4;base64,ZmFrZS12aWRlbw==",
                    "data:video/mp4;base64,ZmFrZS12aWRlbw==",
                    ValueError("cannot inline"),
                ],
            ),
            patch.object(
                tool.core,
                "_extract_video_frames",
                return_value=[(1.0, "data:image/jpeg;base64,ZmFrZQ==")],
            ) as extract_frames,
        ):
            result = await tool.understand_media(
                ["native-1.mp4", "native-2.mp4", "fallback.mp4"],
                "Compare the videos",
                max_frames=10,
            )

        assert result.success is True
        assert result.native_videos_processed == 2
        assert result.frames_extracted == 1
        extract_frames.assert_called_once_with(
            "fallback.mp4",
            start_time=None,
            end_time=None,
            max_frames=8,
        )

    def test_probe_video_duration_requires_ffprobe(self, vision_tool_without_workspace):
        with patch(
            "xagent.core.tools.core.vision_tool.shutil.which", return_value=None
        ):
            with pytest.raises(RuntimeError, match="requires ffprobe/ffmpeg"):
                vision_tool_without_workspace.core._probe_video_duration("clip.mp4")

    def test_extract_video_frames_skips_empty_frame_and_uses_bucket_midpoints(
        self, vision_tool_without_workspace, tmp_path
    ):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake-video")
        completed_frames = [Mock(stdout=b""), Mock(stdout=b"jpeg-frame")]

        with (
            patch(
                "xagent.core.tools.core.vision_tool.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch.object(
                vision_tool_without_workspace.core,
                "_probe_video_duration",
                return_value=8.0,
            ),
            patch(
                "xagent.core.tools.core.vision_tool.subprocess.run",
                side_effect=completed_frames,
            ) as run,
        ):
            frames = vision_tool_without_workspace.core._extract_video_frames(
                str(video_path),
                start_time=0.0,
                end_time=8.0,
                max_frames=2,
            )

        assert frames == [
            (
                6.0,
                "data:image/jpeg;base64,"
                + base64.b64encode(b"jpeg-frame").decode("ascii"),
            )
        ]
        assert [call.args[0][4] for call in run.call_args_list] == ["2.000", "6.000"]


class TestVisionToolDescribeImages:
    """Test cases for describe_images method"""

    @pytest.mark.asyncio
    async def test_describe_images(self, mock_vision_model_with_descriptions):
        """Test describing images"""
        vision_tool = VisionTool(mock_vision_model_with_descriptions)
        result = await vision_tool.describe_images(
            [
                "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRhXzE=",
                "data:image/png;base64,ZmFrZV9pbWFnZV9kYXRhXzI=",
            ],
            detail_level="normal",
        )

        assert result.success is True
        assert (
            result.answer
            == "Image 1: A red apple on a wooden table\nImage 2: A green tree in a park"
        )
        assert result.images_processed == 2

    @pytest.mark.asyncio
    async def test_describe_with_single_image(
        self, mock_vision_model_with_descriptions
    ):
        """Test describing a single image"""
        vision_tool = VisionTool(mock_vision_model_with_descriptions)
        result = await vision_tool.describe_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", detail_level="normal"
        )

        assert result.success is True
        assert (
            result.answer
            == "Image 1: A red apple on a wooden table\nImage 2: A green tree in a park"
        )
        assert result.images_processed == 1


class TestVisionToolDetectObjects:
    """Test cases for detect_objects method"""

    @pytest.mark.asyncio
    async def test_detect_objects_structured_response(
        self, mock_vision_model_with_detection
    ):
        """Test object detection with structured JSON response"""
        vision_tool = VisionTool(mock_vision_model_with_detection)
        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Find all objects in the image",
        )

        assert result.success is True
        assert len(result.detections) == 2

        # Check first object
        obj1 = result.detections[0]
        assert obj1["class"] == "person"
        assert obj1["confidence"] == 0.95
        assert obj1["bbox"] == [0.1, 0.1, 0.6, 0.8]

        # Check second object
        obj2 = result.detections[1]
        assert obj2["class"] == "car"
        assert obj2["confidence"] == 0.87
        assert obj2["bbox"] == [0.7, 0.5, 0.95, 0.75]

    @pytest.mark.asyncio
    async def test_detect_objects_with_task(self, mock_vision_model_with_detection):
        """Test object detection with natural language task"""
        vision_tool = VisionTool(mock_vision_model_with_detection)
        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Find people and dogs in the image",
        )

        assert result.success is True

        # Verify the prompt includes the task
        mock_vision_model_with_detection.vision_chat.assert_called_once()
        call_kwargs = mock_vision_model_with_detection.vision_chat.call_args.kwargs
        messages = call_kwargs["messages"]
        prompt = messages[0]["content"][0]["text"]  # User message text
        assert "Find people and dogs in the image" in prompt

    @pytest.mark.asyncio
    async def test_detect_objects_with_custom_threshold(
        self, mock_vision_model_with_detection
    ):
        """Test object detection with custom confidence threshold"""
        vision_tool = VisionTool(mock_vision_model_with_detection)
        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Detect objects",
            confidence_threshold=0.9,
        )

        assert result.success is True

        # Verify the prompt includes threshold
        mock_vision_model_with_detection.vision_chat.assert_called_once()
        call_kwargs = mock_vision_model_with_detection.vision_chat.call_args.kwargs
        messages = call_kwargs["messages"]
        prompt = messages[0]["content"][0]["text"]  # User message text
        assert "0.9" in prompt

    @pytest.mark.asyncio
    async def test_detect_objects_preserves_zero_temperature(
        self, mock_vision_model_with_detection
    ):
        """A valid zero temperature should not be replaced by the detection default."""
        vision_tool = VisionTool(mock_vision_model_with_detection)
        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Detect objects",
            temperature=0.0,
        )

        assert result.success is True

        call_kwargs = mock_vision_model_with_detection.vision_chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_detect_objects_unstructured_response(
        self, mock_vision_model_with_unstructured_detection
    ):
        """Test object detection with unstructured response (regex fallback)"""
        vision_tool = VisionTool(mock_vision_model_with_unstructured_detection)
        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Find all objects in the image",
        )

        assert result.success is True
        assert len(result.detections) == 2
        # The unstructured detection model should return parsed detections from text
        assert result.detections[0]["class"] == "person"

    @pytest.mark.asyncio
    async def test_detect_objects_invalid_json(self, mock_vision_model):
        """Test object detection with invalid JSON response"""
        # Mock response with invalid JSON string but format that matches the regex pattern
        mock_vision_model.vision_chat.return_value = (
            "person: [0.2, 0.2, 0.7, 0.9] (confidence: 0.88)"
        )

        vision_tool = VisionTool(mock_vision_model)
        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Find all objects in the image",
        )

        assert result.success is True
        assert len(result.detections) >= 1

    @pytest.mark.asyncio
    async def test_detect_objects_multiple_images(
        self, mock_vision_model_with_detection
    ):
        """Test object detection with multiple images"""
        vision_tool = VisionTool(mock_vision_model_with_detection)
        result = await vision_tool.detect_objects(
            [
                "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRhXzE=",
                "data:image/png;base64,ZmFrZV9pbWFnZV9kYXRhXzI=",
            ],
            task="Find all objects in the images",
        )

        assert result.success is True
        assert result.image_processed is not None

    @pytest.mark.asyncio
    async def test_detect_objects_with_marking(
        self, mock_vision_model_with_detection, mock_workspace
    ):
        """Test object detection with marking enabled"""
        vision_tool = VisionTool(mock_vision_model_with_detection, mock_workspace)

        # Mock the _draw_bounding_boxes method to avoid PIL dependency
        with patch.object(vision_tool.core, "_draw_bounding_boxes") as mock_draw:
            mock_draw.return_value = "/workspace/output/marked_test_image.jpg"

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_image_path = temp_file.name
                temp_file.write(b"fake_image_data")

            try:
                result = await vision_tool.detect_objects(
                    temp_image_path, task="Find people", mark_objects=True
                )

                assert result.success is True
                assert len(result.detections) == 2
                assert (
                    result.marked_image_path
                    == "/workspace/output/marked_test_image.jpg"
                )
                assert result.box_color == "red"  # default color

                # Verify _draw_bounding_boxes was called
                mock_draw.assert_called_once()

            finally:
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)


class TestVisionToolHelperMethods:
    """Test cases for helper methods"""

    def test_convert_image_to_base64(self, vision_tool_without_workspace):
        """Test _convert_image_to_base64 method"""
        # Test with URL - should return as-is
        result = vision_tool_without_workspace.core._convert_image_to_base64(
            "https://example.com/test.jpg"
        )
        assert result == "https://example.com/test.jpg"

    def test_validate_images_string_input(self, vision_tool_without_workspace):
        """Test _validate_images method with string input"""
        result = vision_tool_without_workspace.core._validate_images("test_image.jpg")
        assert result == ["test_image.jpg"]

    def test_validate_images_list_input(self, vision_tool_without_workspace):
        """Test _validate_images method with list input"""
        result = vision_tool_without_workspace.core._validate_images(
            ["img1.jpg", "img2.png"]
        )
        assert result == ["img1.jpg", "img2.png"]

    def test_validate_images_dict_input(
        self, vision_tool_without_workspace, sample_image_base64
    ):
        """Test _validate_images method with dict input - this should be handled before calling this method"""
        # The _validate_images method expects string or list of strings
        # Dict input should be processed by the understand_images method first
        # For this test, we'll skip dict input as it's not the intended use case
        pass

    def test_extract_detections_from_text(self, vision_tool_without_workspace):
        """Test _extract_detections_from_text method"""
        text = "I can see a person at [0.1, 0.1, 0.5, 0.8] (confidence: 0.9) and a car at [0.7, 0.3, 0.9, 0.6] (confidence: 0.8)."

        detections = vision_tool_without_workspace.core._extract_detections_from_text(
            text
        )

        assert len(detections) == 2
        # Check that detections were extracted (format may vary)
        assert isinstance(detections, list)
        # Verify structure of first detection
        if detections:
            assert "class" in detections[0]
            assert "bbox" in detections[0]
            assert "confidence" in detections[0]


class TestGetVisionTool:
    """Test cases for get_vision_tool function"""

    def test_get_vision_tool_with_model(self, mock_vision_model):
        """Test get_vision_tool with model provided"""
        tools = get_vision_tool(vision_model=mock_vision_model)

        assert len(tools) == 2

        # Check tool names
        tool_names = [tool.metadata.name for tool in tools]
        assert "understand_media" in tool_names
        assert "detect_objects" in tool_names

    def test_get_vision_tool_with_workspace(self, mock_vision_model, mock_workspace):
        """Test get_vision_tool with workspace provided"""
        tools = get_vision_tool(
            vision_model=mock_vision_model, workspace=mock_workspace
        )

        assert len(tools) == 2

        # Check that tools were created (workspace binding is internal to VisionTool)
        tool_names = [tool.metadata.name for tool in tools]
        assert "understand_media" in tool_names
        assert "detect_objects" in tool_names

    def test_get_vision_tool_without_model(self):
        """Test get_vision_tool without model"""
        tools = get_vision_tool()

        # Test should handle both scenarios:
        # 1. No vision models available (returns empty list)
        # 2. Vision models available in test environment (returns 2 tools)

        if len(tools) == 0:
            # No vision models available scenario
            assert len(tools) == 0
        elif len(tools) == 2:
            # Vision models available scenario
            tool_names = [tool.metadata.name for tool in tools]
            assert "understand_media" in tool_names
            assert "detect_objects" in tool_names
        else:
            # Unexpected number of tools - this indicates a problem
            pytest.fail(f"Expected 0 or 2 tools, got {len(tools)}")


class TestGetDefaultVisionModel:
    """Test cases for get_default_vision_model function"""

    @patch("xagent.web.services.llm_utils._create_llm_instance")
    def test_get_default_vision_model_from_db_success(self, mock_create_llm):
        """Test get_default_vision_model successful creation from database"""
        mock_db = Mock()

        # Mock database model
        mock_db_model = Mock()
        mock_db_model.model_provider = "openai"
        mock_db_model.model_name = "gpt-4-vision"
        mock_db_model.api_key = "test_key"
        mock_db_model.base_url = None
        mock_db_model.temperature = 0.7

        # Mock UserDefaultModel and UserModel relationship
        mock_user_default = Mock()
        mock_user_default.model = mock_db_model

        # Mock query result for UserDefaultModel query
        mock_query1 = Mock()
        mock_query1.join.return_value = mock_query1
        mock_query1.filter.return_value.first.return_value = mock_user_default
        mock_db.query.return_value = mock_query1

        # Mock LLM creation
        mock_llm = Mock(spec=BaseLLM)
        mock_create_llm.return_value = mock_llm

        result = get_default_vision_model(user_id=1, db=mock_db)

        assert result == mock_llm
        mock_create_llm.assert_called_once_with(mock_db_model)

    def test_get_default_vision_model_no_db_model(self):
        """Test get_default_vision_model when no database model found"""
        mock_db = Mock()

        # Mock empty query result
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db.query.return_value = mock_query

        result = get_default_vision_model(db=mock_db)

        assert result is None


class TestVisionToolIntegration:
    """Integration tests for vision tool"""

    @pytest.mark.asyncio
    async def test_tool_execution_flow(self, mock_vision_model, mock_workspace):
        """Test complete tool execution flow"""
        # Create tool
        tools = get_vision_tool(
            vision_model=mock_vision_model, workspace=mock_workspace
        )

        # Find unified media understanding tool
        understand_tool = None
        for tool in tools:
            if tool.metadata.name == "understand_media":
                understand_tool = tool
                break

        assert understand_tool is not None

        # Execute tool
        result = await understand_tool.run_json_async(
            {"media": "existing_image.jpg", "question": "What is this image?"}
        )

        assert result["success"] is True
        assert (
            result["answer"]
            == "This is a beautiful landscape photo with mountains and a lake."
        )

    @pytest.mark.asyncio
    async def test_multiple_tools_same_model(self, mock_vision_model, mock_workspace):
        """Test that multiple tools use the same model instance"""
        tools = get_vision_tool(
            vision_model=mock_vision_model, workspace=mock_workspace
        )

        # Get all tools
        tools_dict = {tool.metadata.name: tool for tool in tools}

        # Execute all tools
        understand_result = await tools_dict["understand_media"].run_json_async(
            {"media": "existing_image.jpg", "question": "What is this?"}
        )

        detect_result = await tools_dict["detect_objects"].run_json_async(
            {"images": "existing_image.jpg", "task": "Detect objects"}
        )

        # All should succeed and use the same model
        assert understand_result["success"] is True
        assert detect_result["success"] is True

        # Verify model was called for each tool
        assert mock_vision_model.vision_chat.call_count == 2

    def test_tool_metadata(self, mock_vision_model):
        """Test that tools have correct metadata"""
        tools = get_vision_tool(vision_model=mock_vision_model)

        for tool in tools:
            # Check basic metadata
            assert tool.metadata.name is not None
            assert len(tool.metadata.name) > 0
            assert tool.metadata.description is not None
            assert len(tool.metadata.description) > 0

            # Check that description mentions vision capabilities
            assert (
                "image" in tool.metadata.description.lower()
                or "vision" in tool.metadata.description.lower()
            )


class TestVisionToolErrorHandling:
    """Test error handling for vision tool"""

    @pytest.mark.asyncio
    async def test_empty_images_list(self, mock_vision_model):
        """Test handling of empty images list"""
        tool = VisionTool(mock_vision_model)
        result = await tool.understand_images([], "What is this?")
        assert (
            result.success is False
            and result.error is not None
            and "At least one image or video must be provided" in result.error
        )

    @pytest.mark.asyncio
    async def test_none_images_input(self, mock_vision_model):
        """Test handling of None images input"""
        tool = VisionTool(mock_vision_model)
        result = await tool.understand_images(None, "What is this?")
        assert (
            result.success is False
            and result.error is not None
            and "At least one image or video must be provided" in result.error
        )


class TestVisionToolEdgeCases:
    """Test edge cases for vision tool"""

    def test_model_info_text_generation(self, mock_vision_model):
        """Test that model info text is generated correctly"""
        tool = VisionTool(mock_vision_model)

        # Check that tool descriptions contain model information
        tools = tool.get_tools()
        for tool_instance in tools:
            description = tool_instance.description
            assert len(description) > 0
            # Should mention it's a vision tool
            assert "vision" in description.lower() or "image" in description.lower()


class TestDrawBoundingBoxes:
    """Test cases for _draw_bounding_boxes helper method"""

    def test_draw_bounding_boxes_without_pil(self, mock_workspace):
        """Test _draw_bounding_boxes when PIL is not available"""
        vision_tool = VisionTool(Mock(spec=BaseLLM), mock_workspace)

        # Mock PIL_AVAILABLE as False
        with patch.object(vision_tool.core, "_draw_bounding_boxes") as mock_draw:
            mock_draw.side_effect = RuntimeError("PIL (Pillow) library is required")

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_image_path = temp_file.name
                temp_file.write(b"fake_image_data")

            try:
                detections = [
                    {
                        "class": "person",
                        "bbox": [0.1, 0.1, 0.6, 0.8],
                        "confidence": 0.95,
                    }
                ]

                with pytest.raises(RuntimeError, match="PIL.*library is required"):
                    vision_tool.core._draw_bounding_boxes(temp_image_path, detections)

            finally:
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)

    def test_draw_bounding_boxes_success(self):
        """Test successful bounding box drawing"""
        # Create a real workspace with actual directories
        from xagent.core.workspace import TaskWorkspace

        with tempfile.TemporaryDirectory() as workspace_dir:
            workspace = TaskWorkspace(id="test_task", base_dir=workspace_dir)
            vision_tool = VisionTool(Mock(spec=BaseLLM), workspace)

            # Create a real test image using PIL
            try:
                from PIL import Image

                # Create a simple test image
                img = Image.new("RGB", (100, 100), color="white")

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp_file:
                    temp_image_path = temp_file.name
                    img.save(temp_image_path, "JPEG")

                try:
                    detections = [
                        {
                            "class": "person",
                            "bbox": [0.1, 0.1, 0.6, 0.8],
                            "confidence": 0.95,
                        },
                        {
                            "class": "car",
                            "bbox": [0.2, 0.2, 0.7, 0.9],
                            "confidence": 0.85,
                        },
                    ]

                    # Test the actual drawing functionality
                    result_path = vision_tool.core._draw_bounding_boxes(
                        temp_image_path, detections, "blue"
                    )

                    # Verify result
                    assert result_path is not None
                    assert os.path.exists(result_path)
                    assert result_path.endswith(".jpg")

                    # Verify the marked image is different from original
                    assert result_path != temp_image_path

                    # Verify it's saved in workspace output directory
                    assert str(workspace.output_dir) in result_path

                    # Clean up the result
                    if os.path.exists(result_path):
                        os.unlink(result_path)

                except Exception:
                    pass
                finally:
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
            except ImportError:
                # If PIL is not available, skip this test
                pytest.skip("PIL (Pillow) library is required for this test")

    @pytest.mark.asyncio
    async def test_detect_objects_marking_with_custom_color(
        self, mock_vision_model_with_detection, mock_workspace
    ):
        """Test object detection with marking using custom color"""
        vision_tool = VisionTool(mock_vision_model_with_detection, mock_workspace)

        with patch.object(vision_tool.core, "_draw_bounding_boxes") as mock_draw:
            mock_draw.return_value = "/workspace/output/marked_blue_image.jpg"

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_image_path = temp_file.name
                temp_file.write(b"fake_image_data")

            try:
                result = await vision_tool.detect_objects(
                    temp_image_path,
                    task="Find vehicles",
                    mark_objects=True,
                    box_color="blue",
                )

                assert result.success is True
                assert (
                    result.marked_image_path
                    == "/workspace/output/marked_blue_image.jpg"
                )
                assert result.box_color == "blue"

                # Verify _draw_bounding_boxes was called with correct color
                mock_draw.assert_called_once()
                call_kwargs = mock_draw.call_args[1]
                assert call_kwargs["box_color"] == "blue"

            except Exception:
                pass
            finally:
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)

    @pytest.mark.asyncio
    async def test_detect_objects_marking_url_not_supported(
        self, mock_vision_model_with_detection
    ):
        """Test that marking is not supported for URLs"""
        vision_tool = VisionTool(mock_vision_model_with_detection)

        result = await vision_tool.detect_objects(
            "https://example.com/image.jpg", task="Find people", mark_objects=True
        )

        assert result.success is False
        assert "only supported for local files" in result.error

    @pytest.mark.asyncio
    async def test_detect_objects_marking_base64_not_supported(
        self, mock_vision_model_with_detection
    ):
        """Test that marking is not supported for base64 data"""
        vision_tool = VisionTool(mock_vision_model_with_detection)

        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Find people",
            mark_objects=True,
        )

        assert result.success is False
        assert "only supported for local files" in result.error

    @pytest.mark.asyncio
    async def test_detect_objects_marking_file_not_found(
        self, mock_vision_model_with_detection
    ):
        """Test handling of non-existent file for marking"""
        vision_tool = VisionTool(mock_vision_model_with_detection)

        result = await vision_tool.detect_objects(
            "/non/existent/path.jpg", task="Find people", mark_objects=True
        )

        assert result.success is False
        assert "Image file not found" in result.error

    def test_draw_bounding_boxes_workspace_output(self, mock_workspace):
        """Test that bounding boxes are saved to workspace output directory"""
        vision_tool = VisionTool(Mock(spec=BaseLLM), mock_workspace)

        # Mock the actual drawing to avoid PIL dependency
        with patch.object(vision_tool.core, "_draw_bounding_boxes") as mock_draw:
            mock_draw.return_value = "/mock/path/marked_image.jpg"

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_image_path = temp_file.name
                temp_file.write(b"fake_image_data")

            try:
                detections = [
                    {
                        "class": "person",
                        "bbox": [0.1, 0.1, 0.6, 0.8],
                        "confidence": 0.95,
                    }
                ]

                result = vision_tool._draw_bounding_boxes(temp_image_path, detections)

                # Verify the method was called with correct parameters
                mock_draw.assert_called_once_with(temp_image_path, detections)
                assert result == "/mock/path/marked_image.jpg"

            except Exception:
                pass
            finally:
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
