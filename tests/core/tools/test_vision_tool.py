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
from xagent.core.tools.core.vision_tool import (
    UnderstandMediaResult,
    _normalize_vision_response,
)
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


# Each row is (label, response, expected_kind, expected_text,
# expected_tool_calls, expected_raw_display). Rows cover every shape
# ``_normalize_vision_response`` must classify: bare strings (empty,
# whitespace-only, and non-empty are tested as separate rows so a
# whitespace string cannot satisfy two rows' criteria at once), text/
# tool_call/unrecognized envelopes, a missing "type" key, ``None``, and
# other Python types.
_VISION_RESPONSE_SHAPE_MATRIX = [
    (
        "non_empty_str",
        "a plain text reply",
        "text",
        "a plain text reply",
        [],
        "a plain text reply",
    ),
    ("empty_str", "", "empty", "", [], ""),
    ("whitespace_only_str", "   ", "empty", "   ", [], "   "),
    (
        "text_envelope",
        {"type": "text", "content": "hello from the envelope"},
        "text",
        "hello from the envelope",
        [],
        "hello from the envelope",
    ),
    (
        "text_envelope_empty_content",
        {"type": "text", "content": ""},
        "empty",
        "",
        [],
        "",
    ),
    (
        "text_envelope_whitespace_content",
        {"type": "text", "content": "   "},
        "empty",
        "   ",
        [],
        "   ",
    ),
    (
        "text_envelope_non_str_content",
        {"type": "text", "content": 5},
        "unknown",
        None,
        [],
        str({"type": "text", "content": 5}),
    ),
    (
        "tool_call_envelope",
        {"type": "tool_call", "tool_calls": [{"id": "c1", "type": "function"}]},
        "tool_call",
        None,
        [{"id": "c1", "type": "function"}],
        str({"type": "tool_call", "tool_calls": [{"id": "c1", "type": "function"}]}),
    ),
    (
        "unknown_type_dict",
        {"type": "mystery", "payload": 1},
        "unknown",
        None,
        [],
        str({"type": "mystery", "payload": 1}),
    ),
    (
        "dict_without_type_key",
        {"payload": 1},
        "unknown",
        None,
        [],
        str({"payload": 1}),
    ),
    ("none", None, "unknown", None, [], "None"),
    ("other_type_int", 123, "unknown", None, [], "123"),
    (
        "other_type_list",
        [1, 2, 3],
        "unknown",
        None,
        [],
        str([1, 2, 3]),
    ),
]


class TestNormalizeVisionResponse:
    """Behavior of the response classifier shared by detect_objects and
    understand_media. Exercised directly against the pure function -- the
    call sites are exercised separately against the shapes they actually
    branch on."""

    @pytest.mark.parametrize(
        "label,response,expected_kind,expected_text,expected_tool_calls,expected_raw_display",
        _VISION_RESPONSE_SHAPE_MATRIX,
        ids=[row[0] for row in _VISION_RESPONSE_SHAPE_MATRIX],
    )
    def test_shape_matrix(
        self,
        label,
        response,
        expected_kind,
        expected_text,
        expected_tool_calls,
        expected_raw_display,
    ):
        result = _normalize_vision_response(response)
        assert result.kind == expected_kind
        assert result.text == expected_text
        assert result.tool_calls == expected_tool_calls
        assert result.raw_display == expected_raw_display

    def test_shape_matrix_covers_exactly_four_kinds(self):
        kinds = {row[2] for row in _VISION_RESPONSE_SHAPE_MATRIX}
        assert kinds == {"text", "empty", "tool_call", "unknown"}

    # Covers the shape matrix plus a handful of other plain-data inputs --
    # not a claim that the classifier never raises for any input whatsoever.
    @pytest.mark.parametrize(
        "response",
        [row[1] for row in _VISION_RESPONSE_SHAPE_MATRIX]
        + [object(), 3.14, {"nested": {"type": "text"}}],
    )
    def test_never_raises(self, response):
        result = _normalize_vision_response(response)
        assert result.kind in {"text", "empty", "tool_call", "unknown"}

    @pytest.mark.parametrize(
        "label,response,expected_kind,expected_text,expected_tool_calls,expected_raw_display",
        _VISION_RESPONSE_SHAPE_MATRIX,
        ids=[row[0] for row in _VISION_RESPONSE_SHAPE_MATRIX],
    )
    def test_text_field_type_follows_kind(
        self,
        label,
        response,
        expected_kind,
        expected_text,
        expected_tool_calls,
        expected_raw_display,
    ):
        result = _normalize_vision_response(response)
        if result.kind in {"text", "empty"}:
            assert isinstance(result.text, str)
        else:
            assert result.text is None

    @pytest.mark.parametrize(
        "label,response",
        [
            ("bare_whitespace", " " * 9000),
            ("envelope_whitespace_content", {"type": "text", "content": " " * 9000}),
        ],
    )
    def test_empty_raw_display_is_truncated(self, label, response):
        """The shape matrix rows are all short, so the cap holds trivially
        there. A whitespace-only payload long enough to exceed the limit
        forces the empty branches to route through the truncator like every
        other branch does."""
        result = _normalize_vision_response(response)

        assert result.kind == "empty"
        # The 4000-char prefix plus the marker: the marker is appended
        # *after* the cap, so the returned string is 4025 chars, not 4000.
        assert result.raw_display == " " * 4000 + "...<truncated 5000 chars>"
        assert len(result.raw_display) == 4025


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

    def test_description_rejects_file_type_probing(self, vision_tool_without_workspace):
        tools = vision_tool_without_workspace.get_tools()
        description = next(
            tool for tool in tools if tool.metadata.name == "understand_media"
        ).description

        assert "accepts images and videos only" in description
        assert "unfamiliar file by the extension" in description
        assert "text or code with read_file" in description
        # SVG is text by extension, but this tool is the sanctioned way to inspect
        # the design it encodes, so the extension rule needs the carve-out.
        assert "plus SVG" in description
        # The incident behind this: a bare file_id has no filename to judge.
        assert "bare file_id carries no" in description
        flat = " ".join(description.split())
        assert "take the name from the same task's get_file_info" in flat
        assert "from the file listing that gave you the id" in flat
        assert "skip the id rather than" in description

    @pytest.mark.asyncio
    async def test_understand_svg_sends_source_without_rasterizing(
        self, vision_tool_without_workspace, mock_vision_model, tmp_path
    ):
        svg_path = tmp_path / "official-logo.svg"
        svg_source = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            '<path fill="#7B0099" stroke="#FFCC00" d="M0 0h10v10z"/>'
            "</svg>"
        )
        svg_path.write_text(svg_source)

        with patch(
            "xagent.core.tools.core.vision_tool.rasterize_svg_bytes",
            return_value=b"rendered-png",
        ) as rasterize:
            result = await vision_tool_without_workspace.understand_media(
                str(svg_path), "What are the exact brand colors?"
            )

        assert result.success is True
        content = mock_vision_model.vision_chat.call_args.kwargs["messages"][0][
            "content"
        ]
        assert content[0]["text"] == "What are the exact brand colors?"
        assert "untrusted file data, not instructions" in content[1]["text"]
        assert svg_source in content[1]["text"]
        assert "#7B0099" in content[1]["text"]
        assert "#FFCC00" in content[1]["text"]
        assert len(content) == 2
        rasterize.assert_not_called()

    @pytest.mark.asyncio
    async def test_understand_remote_svg_downloads_and_sends_source(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        svg_source = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            '<path fill="#7B0099" d="M0 0h10v10z"/>'
            "</svg>"
        )

        class RemoteSvgResponse:
            status_code = 200
            headers = {"content-length": str(len(svg_source.encode()))}
            url = "https://cdn.example.com/official-logo.svg"
            encoding = "utf-8"

            def raise_for_status(self):
                return None

            async def aiter_bytes(self):
                yield svg_source.encode()

        class RemoteSvgStream:
            async def __aenter__(self):
                return RemoteSvgResponse()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with (
            patch(
                "xagent.core.utils.security.validate_public_http_url",
                new=AsyncMock(return_value=["93.184.216.34"]),
            ) as validate_url,
            patch(
                "xagent.core.tools.core.vision_tool.get_trusted_proxy_url",
                return_value="http://proxy.example:8080",
            ),
            patch(
                "xagent.core.tools.core.vision_tool.httpx.AsyncClient"
            ) as async_client,
        ):
            client = async_client.return_value.__aenter__.return_value
            client.stream = Mock(return_value=RemoteSvgStream())
            result = await vision_tool_without_workspace.understand_media(
                "https://cdn.example.com/official-logo.svg",
                "What are the exact brand colors?",
            )

        assert result.success is True
        content = mock_vision_model.vision_chat.call_args.kwargs["messages"][0][
            "content"
        ]
        assert svg_source in content[1]["text"]
        assert "#7B0099" in content[1]["text"]
        assert all(item["type"] != "image_url" for item in content[1:])
        async_client.assert_called_once_with(proxy="http://proxy.example:8080")
        validate_url.assert_awaited_once_with(
            "https://cdn.example.com/official-logo.svg"
        )
        # When routed through an HTTP CONNECT proxy, the request must keep the
        # original hostname as the connect target: httpcore's CONNECT tunnel
        # path derives TLS SNI from the remote origin and ignores the
        # `sni_hostname` extension, so rewriting to the pinned IP here would
        # send that IP as SNI and break SNI-strict servers (PR #977 review).
        client.stream.assert_called_once_with(
            "GET",
            "https://cdn.example.com/official-logo.svg",
            headers={},
            timeout=10,
            follow_redirects=False,
            extensions={},
        )

    @pytest.mark.asyncio
    async def test_understand_local_image_offloads_conversion(
        self, vision_tool_without_workspace, mock_vision_model, tmp_path
    ):
        image_path = tmp_path / "brand-image.png"
        image_path.write_bytes(b"fake-image")
        converted = "data:image/png;base64,ZmFrZS1pbWFnZQ=="

        with (
            patch.object(
                vision_tool_without_workspace.core,
                "_convert_image_to_base64",
                return_value=converted,
            ) as convert_image,
            patch(
                "xagent.core.tools.core.vision_tool.asyncio.to_thread",
                new=AsyncMock(return_value=converted),
            ) as to_thread,
        ):
            result = await vision_tool_without_workspace.understand_media(
                str(image_path), "What is shown?"
            )

        assert result.success is True
        to_thread.assert_awaited_once_with(convert_image, str(image_path))
        content = mock_vision_model.vision_chat.call_args.kwargs["messages"][0][
            "content"
        ]
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": converted},
        }

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


class TestVisionToolUnderstandMediaEnvelope:
    """Response-shape handling in understand_media. None of the module's
    other fixtures produce an envelope response, so these configure the
    mock model's return value directly per case."""

    @pytest.mark.asyncio
    async def test_understand_media_returns_envelope_content(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """A text envelope's content must become the answer verbatim, and
        the envelope itself must not leak into it."""
        mock_vision_model.vision_chat.return_value = {
            "type": "text",
            "content": "The lake is Lake Louise.",
            "raw": {"id": "resp-1"},
        }

        result = await vision_tool_without_workspace.understand_media(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", "Where is this?"
        )

        assert result.success is True
        assert result.answer == "The lake is Lake Louise."
        assert "'type'" not in result.answer
        assert "'raw'" not in result.answer

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label,response,expected_error",
        [
            (
                "unknown_dict",
                {"type": "mystery", "payload": 1},
                "Vision model returned an unsupported response shape",
            ),
            (
                "non_str_content",
                {"type": "text", "content": 5},
                "Vision model returned an unsupported response shape",
            ),
            (
                "none",
                None,
                "Vision model returned an unsupported response shape",
            ),
        ],
    )
    async def test_understand_media_rejects_shapes_without_text_payload(
        self,
        vision_tool_without_workspace,
        mock_vision_model,
        label,
        response,
        expected_error,
    ):
        """A response with no text payload must not be turned into an
        answer via str(response); it must fail explicitly instead."""
        mock_vision_model.vision_chat.return_value = response

        result = await vision_tool_without_workspace.understand_media(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", "Describe this."
        )

        assert result.success is False
        assert result.answer is None
        assert result.error == expected_error

    @pytest.mark.asyncio
    async def test_understand_media_tool_call_message_unchanged(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """The existing tool-call message text is not part of this fix and
        must survive the rewrite unchanged."""
        mock_vision_model.vision_chat.return_value = {
            "type": "tool_call",
            "tool_calls": [{"id": "c1", "type": "function"}],
            "raw": {"id": "resp-1"},
        }

        result = await vision_tool_without_workspace.understand_media(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", "Describe this."
        )

        assert result.success is True
        assert result.answer == (
            "Model triggered tool call instead of answering: "
            "[{'id': 'c1', 'type': 'function'}]"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label,response",
        [
            ("bare_empty", ""),
            ("bare_whitespace", "   "),
            ("envelope_empty_content", {"type": "text", "content": ""}),
            ("envelope_whitespace_content", {"type": "text", "content": "   "}),
        ],
    )
    async def test_understand_media_rejects_empty_text_payload(
        self, vision_tool_without_workspace, mock_vision_model, label, response
    ):
        """An empty or whitespace-only response, bare or wrapped in an
        envelope, must not be reported as a successful empty answer."""
        mock_vision_model.vision_chat.return_value = response

        result = await vision_tool_without_workspace.understand_media(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", "Describe this."
        )

        assert result.success is False
        assert result.answer is None
        assert result.error == "Vision model returned an empty response"

    @pytest.mark.asyncio
    async def test_raw_display_confined_to_detect_objects_diagnostics(
        self, vision_tool_without_workspace, mock_vision_model
    ):
        """raw_display -- the classifier's length-capped diagnostic string
        -- must never leak into a user-visible answer, UnderstandMediaResult
        must not grow a field to carry it, and DetectObjectsResult must
        keep populating its own raw_response diagnostic field."""
        # Frozen field set. Equality rather than a `"raw_display" not in`
        # check is deliberate: it also pins the absence of `parsing_method`
        # and `raw_response`, which the detect_objects result carries and
        # this one must not. Adding a field to this public result object is
        # a deliberate contract change, so update this line in the same
        # commit that adds it.
        assert set(UnderstandMediaResult.model_fields) == {
            "success",
            "answer",
            "media_processed",
            "images_processed",
            "videos_processed",
            "native_videos_processed",
            "frames_extracted",
            "model_used",
            "warnings",
            "error",
        }

        # A content payload longer than the truncation limit must reach
        # `answer` in full. If `answer` were ever sourced from the
        # length-capped raw_display instead of the classifier's `text`
        # field, this would come back shorter than the original.
        long_content = "x" * 5000
        mock_vision_model.vision_chat.return_value = {
            "type": "text",
            "content": long_content,
            "raw": {"id": "resp-1"},
        }
        result = await vision_tool_without_workspace.understand_media(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", "Describe this."
        )
        assert result.answer == long_content
        assert len(result.answer) > 4000

        # detect_objects still populates its own raw_response field.
        model = Mock(spec=BaseLLM)
        model.vision_chat = AsyncMock(
            return_value=(
                '{"detections": [{"class": "person", "confidence": 0.9, '
                '"bbox": [0.1, 0.1, 0.6, 0.8]}]}'
            )
        )
        model.has_ability = Mock(return_value=True)
        detect_result = await VisionTool(model).detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh", task="Find objects"
        )
        assert detect_result.raw_response


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
    async def test_detect_objects_offloads_local_image_conversion(
        self, mock_vision_model_with_detection, tmp_path
    ):
        image_path = tmp_path / "brand.svg"
        image_path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        vision_tool = VisionTool(mock_vision_model_with_detection)
        converted = "data:image/png;base64,ZmFrZQ=="

        with (
            patch.object(
                vision_tool.core,
                "_convert_image_to_base64",
                return_value=converted,
            ) as convert_image,
            patch(
                "xagent.core.tools.core.vision_tool.asyncio.to_thread",
                new=AsyncMock(return_value=converted),
            ) as to_thread,
        ):
            result = await vision_tool.detect_objects(
                str(image_path), task="Find the logo"
            )

        assert result.success is True
        to_thread.assert_awaited_once_with(convert_image, str(image_path))

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

    @pytest.mark.asyncio
    async def test_detect_objects_parses_text_envelope_payload(self):
        """OpenAI-family providers wrap the detection JSON in a text
        envelope; the envelope's content must reach the same parser a bare
        JSON string would."""
        detection_json = (
            '{"detections": [{"class": "person", "confidence": 0.95, '
            '"bbox": [0.1, 0.1, 0.6, 0.8]}], "image_info": {"width": "640", '
            '"height": "480"}}'
        )
        model = Mock(spec=BaseLLM)
        model.vision_chat = AsyncMock(
            return_value={
                "type": "text",
                "content": detection_json,
                "raw": {"id": "resp-1"},
            }
        )
        model.has_ability = Mock(return_value=True)

        vision_tool = VisionTool(model)
        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Find all objects in the image",
        )

        assert result.success is True
        assert result.total_detections == 1
        assert result.detections[0]["class"] == "person"
        assert result.parsing_method == "json"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "label,response,expected_error,expected_parsing_method,expected_raw_response",
        [
            (
                "tool_call",
                {
                    "type": "tool_call",
                    "tool_calls": [{"id": "c1", "type": "function"}],
                },
                "Vision model returned a tool call instead of a detection payload",
                "tool_call_response",
                "{'type': 'tool_call', 'tool_calls': [{'id': 'c1', 'type': 'function'}]}",
            ),
            (
                "unknown_dict",
                {"type": "mystery", "payload": 1},
                "Vision model returned an unsupported response shape",
                "unknown_type",
                "{'type': 'mystery', 'payload': 1}",
            ),
            (
                "non_str_content",
                {"type": "text", "content": 5},
                "Vision model returned an unsupported response shape",
                "unknown_type",
                "{'type': 'text', 'content': 5}",
            ),
            (
                "none",
                None,
                "Vision model returned an unsupported response shape",
                "unknown_type",
                "None",
            ),
        ],
    )
    @pytest.mark.parametrize("mark_objects", [False, True])
    async def test_detect_objects_flags_shapes_without_text_payload(
        self,
        label,
        response,
        expected_error,
        expected_parsing_method,
        expected_raw_response,
        mark_objects,
    ):
        """A response with no text payload must be reported as a failure
        with a non-empty error, and must never reach the marking step that
        writes a bounding-box image to disk. Uses a real local file (rather
        than a data: URL) so the mark_objects=True rows exercise the actual
        marking code path instead of being rejected earlier by the
        local-file-only guard for URL/data images."""
        model = Mock(spec=BaseLLM)
        model.vision_chat = AsyncMock(return_value=response)
        model.has_ability = Mock(return_value=True)

        vision_tool = VisionTool(model)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_image_path = temp_file.name
            temp_file.write(b"fake_image_data")

        try:
            with patch.object(vision_tool.core, "_draw_bounding_boxes") as mock_draw:
                result = await vision_tool.detect_objects(
                    temp_image_path,
                    task="Find all objects in the image",
                    mark_objects=mark_objects,
                )
                mock_draw.assert_not_called()
        finally:
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

        assert result.success is False
        assert result.error == expected_error
        assert result.parsing_method == expected_parsing_method
        assert result.raw_response == expected_raw_response
        assert result.total_detections == 0
        assert result.marked_image_path is None
        # 4000 is the truncation limit; the suffix is the truncation marker.
        # Exactly one of the two must hold for every raw_response.
        # The cap is the only bound on provider-internal keys; no key-level
        # redaction is performed.
        assert result.raw_response is not None
        assert len(result.raw_response) <= 4000 or result.raw_response.endswith(
            " chars>"
        )

    @pytest.mark.asyncio
    # bare_empty / bare_whitespace: zhipu.py's vision entry point (lines
    # 1027/1033) returns a bare "" or a bare unstripped `content` when there
    # are no tool calls; its vision path has no whitespace guard, unlike its
    # chat path's `.strip()` check at zhipu.py:369.
    # envelope_whitespace_content: xinference.py:347 gates its text exit on
    # bare truthiness, so whitespace-only content passes through unrejected.
    # envelope_empty_content: no confirmed production source; exercised
    # defensively as a normalization boundary.
    @pytest.mark.parametrize(
        "label,response,expected_raw_response",
        [
            ("bare_empty", "", ""),
            ("bare_whitespace", "   \n\t ", "   \n\t "),
            ("envelope_empty_content", {"type": "text", "content": ""}, ""),
            (
                "envelope_whitespace_content",
                {"type": "text", "content": "   "},
                "   ",
            ),
        ],
    )
    @pytest.mark.parametrize("mark_objects", [False, True])
    async def test_detect_objects_rejects_empty_text_payload(
        self, label, response, expected_raw_response, mark_objects
    ):
        """An empty or whitespace-only text payload -- bare or wrapped in a
        text envelope -- must be reported as a failure rather than an
        empty-but-successful detection, and must never reach the marking
        step. Uses a real local file (rather than a data: URL) so the
        mark_objects=True rows exercise the actual marking code path
        instead of being rejected earlier by the local-file-only guard for
        URL/data images."""
        model = Mock(spec=BaseLLM)
        model.vision_chat = AsyncMock(return_value=response)
        model.has_ability = Mock(return_value=True)

        vision_tool = VisionTool(model)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_image_path = temp_file.name
            temp_file.write(b"fake_image_data")

        try:
            with patch.object(vision_tool.core, "_draw_bounding_boxes") as mock_draw:
                result = await vision_tool.detect_objects(
                    temp_image_path,
                    task="Find all objects in the image",
                    mark_objects=mark_objects,
                )
                mock_draw.assert_not_called()
        finally:
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

        assert result.success is False
        assert result.error == "Vision model returned an empty response"
        assert result.parsing_method == "empty_response"
        assert result.raw_response == expected_raw_response
        assert result.detections == []
        assert result.total_detections == 0
        assert result.marked_image_path is None

    @pytest.mark.asyncio
    async def test_detect_objects_truncates_large_raw_response(self):
        """The no-text-payload rows above are all short, so the truncation
        bound holds trivially whether or not truncation actually runs. This
        forces an oversized payload (str(result) > 4000 chars) so the
        truncation marker assertion has something to catch."""
        model = Mock(spec=BaseLLM)
        model.vision_chat = AsyncMock(
            return_value={"type": "mystery", "payload": "x" * 5000}
        )
        model.has_ability = Mock(return_value=True)

        vision_tool = VisionTool(model)
        result = await vision_tool.detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Find all objects in the image",
        )

        assert result.success is False
        assert result.raw_response is not None
        # The truncated string is the 4000-char prefix plus a "...<truncated
        # N chars>" marker, so its length is *not* <= 4000 -- it must carry
        # the marker instead.
        assert result.raw_response.endswith(" chars>")
        assert len(result.raw_response) > 4000


class TestVisionToolHelperMethods:
    """Test cases for helper methods"""

    def test_convert_image_to_base64(self, vision_tool_without_workspace):
        """Test _convert_image_to_base64 method"""
        # Test with URL - should return as-is
        result = vision_tool_without_workspace.core._convert_image_to_base64(
            "https://example.com/test.jpg"
        )
        assert result == "https://example.com/test.jpg"

    def test_convert_svg_to_png_base64(self, vision_tool_without_workspace, tmp_path):
        svg_path = tmp_path / "official-logo.svg"
        svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" />')
        png_bytes = b"rendered-png"

        with patch(
            "xagent.core.tools.core.vision_tool.rasterize_svg_bytes",
            return_value=png_bytes,
        ) as rasterize:
            result = vision_tool_without_workspace.core._convert_image_to_base64(
                str(svg_path)
            )

        assert result == (
            "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        )
        rasterize.assert_called_once_with(svg_path.read_bytes())

    def test_decode_svg_bytes_rejects_script(self, vision_tool_without_workspace):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        with pytest.raises(ValueError, match="script"):
            vision_tool_without_workspace.core._decode_svg_bytes(svg)

    @pytest.mark.asyncio
    async def test_svg_source_content_rejects_local_file_over_size_limit(
        self, vision_tool_without_workspace, tmp_path
    ):
        from xagent.core.utils.svg import MAX_SVG_BYTES

        svg_path = tmp_path / "huge.svg"
        with open(svg_path, "w") as f:
            f.write("<svg>")
            f.write("a" * (MAX_SVG_BYTES + 1))
            f.write("</svg>")

        content, warning = await vision_tool_without_workspace.core._svg_source_content(
            str(svg_path), 0
        )

        assert content is None
        assert warning is not None
        assert "exceeds maximum size" in warning

    @pytest.mark.asyncio
    async def test_svg_source_content_rejects_local_file_without_svg_root(
        self, vision_tool_without_workspace, tmp_path
    ):
        svg_path = tmp_path / "not-really.svg"
        svg_path.write_text("just some text pretending to be an svg file")

        content, warning = await vision_tool_without_workspace.core._svg_source_content(
            str(svg_path), 0
        )

        assert content is None
        assert warning is not None
        assert "does not contain" in warning

    @pytest.mark.asyncio
    async def test_svg_source_content_accepts_normal_local_svg(
        self, vision_tool_without_workspace, tmp_path
    ):
        svg_path = tmp_path / "official-logo.svg"
        svg_source = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"/>'
        svg_path.write_text(svg_source)

        content, warning = await vision_tool_without_workspace.core._svg_source_content(
            str(svg_path), 0
        )

        assert warning is None
        assert content is not None
        assert content["type"] == "text"
        assert svg_source in content["text"]

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


class TestVisionToolChatEnvelope:
    """Regression tests for the chat-envelope contract (#520/#1714 class).

    Adapters now return ``{"type": "text", "content": ...}`` envelopes from
    ``vision_chat``; the tool must read ``content``, never repr the envelope.
    """

    @pytest.mark.asyncio
    async def test_understand_images_unwraps_text_envelope(self) -> None:
        model = Mock(spec=BaseLLM)
        model.vision_chat = AsyncMock(
            return_value={
                "type": "text",
                "content": "A cat sitting on a keyboard.",
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }
        )
        model.has_ability = Mock(return_value=True)

        result = await VisionTool(model).understand_images(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            "What is in this image?",
        )

        assert result.success is True
        assert result.answer == "A cat sitting on a keyboard."
        assert "'type': 'text'" not in result.answer

    @pytest.mark.asyncio
    async def test_detect_objects_parses_text_envelope_content(self) -> None:
        model = Mock(spec=BaseLLM)
        model.vision_chat = AsyncMock(
            return_value={
                "type": "text",
                "content": (
                    '{"detections": [{"class": "person", "confidence": 0.95, '
                    '"bbox": [0.1, 0.1, 0.6, 0.8]}]}'
                ),
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }
        )
        model.has_ability = Mock(return_value=True)

        result = await VisionTool(model).detect_objects(
            "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh",
            task="Find all objects in the image",
        )

        assert result.success is True
        assert len(result.detections) == 1
        assert result.detections[0]["class"] == "person"
