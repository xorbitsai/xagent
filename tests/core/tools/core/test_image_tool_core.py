"""
Tests for ImageGenerationToolCore class
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from xagent.core.model.image.base import BaseImageModel
from xagent.core.tools.core.image_tool import ImageGenerationToolCore


@pytest.fixture
def mock_image_model():
    """Create a mock image model for testing"""
    model = Mock(spec=BaseImageModel)
    model.generate_image = AsyncMock(
        return_value={
            "image_url": "https://example.com/test_image.jpg",
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "task_metric": {"total_time": 2.5},
            "request_id": "test_request_id",
        }
    )
    return model


@pytest.fixture
def mock_image_models():
    """Create multiple mock image models for testing"""
    model1 = Mock(spec=BaseImageModel)
    model1.generate_image = AsyncMock(
        return_value={
            "image_url": "https://example.com/image1.jpg",
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "request_id": "req1",
        }
    )

    model2 = Mock(spec=BaseImageModel)
    model2.generate_image = AsyncMock(
        return_value={
            "image_url": "https://example.com/image2.jpg",
            "usage": {"input_tokens": 15, "output_tokens": 25},
            "request_id": "req2",
        }
    )

    return {"model1": model1, "model2": model2}


@pytest.fixture
def mock_workspace():
    """Create a mock workspace for testing"""
    from contextlib import contextmanager
    from pathlib import Path
    from unittest.mock import Mock

    workspace = Mock()
    workspace.output_dir = Path("/tmp/test_workspace/output")
    workspace.output_dir.mkdir(parents=True, exist_ok=True)

    # Mock auto_register_files to return a proper context manager
    @contextmanager
    def auto_register_files():
        yield workspace

    workspace.auto_register_files = auto_register_files
    # Mock get_file_id_from_path to return a valid file_id
    workspace.get_file_id_from_path = Mock(return_value="test-file-id")

    return workspace


@pytest.fixture
def image_tool_core(mock_image_models, mock_workspace):
    """Create ImageGenerationToolCore instance for testing"""
    return ImageGenerationToolCore(
        mock_image_models,
        {"model1": "Test model 1", "model2": "Test model 2"},
        mock_workspace,
    )


@pytest.fixture
def edit_image_tool(mock_image_models, mock_workspace):
    """Create ImageGenerationToolCore configured for edit operations."""
    mock_image_models["model1"].edit_image = AsyncMock(
        return_value={
            "image_url": "https://example.com/edited_image.jpg",
            "usage": {"input_tokens": 15, "output_tokens": 25},
            "request_id": "edit_req1",
        }
    )
    mock_image_models["model1"].has_ability = Mock(return_value=True)
    return ImageGenerationToolCore(
        mock_image_models, {"model1": "Test model 1"}, mock_workspace
    )


def _setup_edit_http_mock(mock_get):
    """Configure mock HTTP response for image download in edit tests."""
    mock_response = Mock()
    mock_response.status = 200

    async def mock_iter_chunked(chunk_size):
        for chunk in [b"fake_edited_image_data"]:
            yield chunk

    mock_response.content.iter_chunked = mock_iter_chunked
    mock_get.return_value.__aenter__.return_value = mock_response


class TestImageGenerationToolCore:
    """Test cases for ImageGenerationToolCore class"""

    def test_init_with_models(self, mock_image_models, mock_workspace):
        """Test ImageGenerationToolCore initialization with models"""
        tool = ImageGenerationToolCore(mock_image_models, workspace=mock_workspace)
        assert tool._image_models == mock_image_models
        assert len(tool._image_models) == 2

    def test_init_with_empty_models(self, mock_workspace):
        """Test ImageGenerationToolCore initialization with empty models"""
        tool = ImageGenerationToolCore({}, workspace=mock_workspace)
        assert tool._image_models == {}
        assert len(tool._image_models) == 0

    def test_init_without_workspace(self, mock_image_models):
        """Test ImageGenerationToolCore initialization without workspace"""
        tool = ImageGenerationToolCore(mock_image_models)
        assert tool._image_models == mock_image_models
        assert tool._workspace is None

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_generate_image_with_default_model(
        self, mock_get, image_tool_core, mock_image_models
    ):
        """Test image generation with default model"""
        # Mock HTTP response for image download
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await image_tool_core.generate_image("A test prompt")

        assert result["success"] is True
        assert result["image_path"] is not None
        assert result["model_used"] == "default"
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 20}
        assert result["request_id"] == "req1"
        assert result["saved_to_workspace"] is True

        # Verify the first model was used (default behavior)
        mock_image_models["model1"].generate_image.assert_called_once_with(
            prompt="A test prompt", size="1024*1024", negative_prompt=""
        )

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_generate_image_with_specific_model(
        self, mock_get, image_tool_core, mock_image_models
    ):
        """Test image generation with specific model"""
        # Mock HTTP response for image download
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await image_tool_core.generate_image(
            "A test prompt", model_id="model2"
        )

        assert result["success"] is True
        assert result["image_path"] is not None
        assert result["model_used"] == "model2"
        assert result["saved_to_workspace"] is True

        # Verify the specified model was used
        mock_image_models["model2"].generate_image.assert_called_once_with(
            prompt="A test prompt", size="1024*1024", negative_prompt=""
        )
        mock_image_models["model1"].generate_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_image_with_no_models(self, mock_workspace):
        """Test image generation with no models available"""
        tool = ImageGenerationToolCore({}, workspace=mock_workspace)
        result = await tool.generate_image("A test prompt")

        assert result["success"] is False
        assert result["error"] == "No available image models configured"
        assert result["image_path"] is None

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_generate_image_with_workspace(
        self, mock_get, mock_image_models, mock_workspace
    ):
        """Test image generation with workspace (should download and save)"""
        # Create tool with workspace
        tool = ImageGenerationToolCore(
            mock_image_models, {"model1": "Test model 1"}, mock_workspace
        )

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await tool.generate_image("A test prompt")

        assert result["success"] is True
        assert result["image_path"] is not None
        assert result["saved_to_workspace"] is True
        assert "generated_image_" in result["image_path"]

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_download_image_with_default_timeout(self, mock_get, mock_workspace):
        """Test image download with default timeout (30 seconds)"""
        # Create tool with workspace
        tool = ImageGenerationToolCore({}, {}, mock_workspace)

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await tool._download_image("https://example.com/test.png")

        assert result is not None
        assert "generated_image_" in result
        assert result.endswith(".png")

        # Verify timeout was set to default 30 seconds
        mock_get.assert_called_once()
        # The timeout is passed as aiohttp.ClientTimeout object in the session

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_download_image_with_custom_timeout(self, mock_get, mock_workspace):
        """Test image download with custom timeout"""
        # Create tool with workspace
        tool = ImageGenerationToolCore({}, {}, mock_workspace)

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        # Test with custom timeout
        result = await tool._download_image("https://example.com/test.png", timeout=60)

        assert result is not None
        assert "generated_image_" in result
        assert result.endswith(".png")

        # Verify timeout was set to 60 seconds
        mock_get.assert_called_once()
        # The timeout is passed as aiohttp.ClientTimeout object in the session

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_download_image_with_custom_timeout_parameter(
        self, mock_get, mock_workspace
    ):
        """Test that core class accepts custom timeout parameter"""
        # Create tool with workspace
        tool = ImageGenerationToolCore({}, {}, mock_workspace)

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        # Test with custom timeout parameter
        result = await tool._download_image("https://example.com/test.png", timeout=120)

        assert result is not None
        assert "generated_image_" in result
        assert result.endswith(".png")

        # Verify the call was made
        mock_get.assert_called_once()

    def test_list_available_models(self, image_tool_core, mock_image_models):
        """Test listing available models"""
        result = image_tool_core.list_available_models()

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["models"]) == 2

        # Check that all model IDs are present
        model_ids = [model["model_id"] for model in result["models"]]
        assert "model1" in model_ids
        assert "model2" in model_ids

        # Check model availability and descriptions
        for model in result["models"]:
            assert model["available"] is True
            assert "description" in model
            assert len(model["description"]) > 0

        # Check specific descriptions
        model1_info = next(m for m in result["models"] if m["model_id"] == "model1")
        model2_info = next(m for m in result["models"] if m["model_id"] == "model2")
        assert model1_info["description"] == "Test model 1"
        assert model2_info["description"] == "Test model 2"

    def test_list_available_models_empty(self, mock_workspace):
        """Test listing available models when no models are configured"""
        tool = ImageGenerationToolCore({}, workspace=mock_workspace)
        result = tool.list_available_models()

        assert result["success"] is True
        assert result["count"] == 0
        assert result["models"] == []

    def test_get_model_with_id(self, image_tool_core, mock_image_models):
        """Test _get_model method with specific model ID"""
        model = image_tool_core._get_model("model2")
        assert model == mock_image_models["model2"]

    def test_get_model_with_default(self, image_tool_core, mock_image_models):
        """Test _get_model method with default model"""
        model = image_tool_core._get_model()
        assert model == mock_image_models["model1"]  # First model

    def test_get_model_with_nonexistent_id(self, image_tool_core, mock_image_models):
        """Test _get_model method with non-existent model ID"""
        model = image_tool_core._get_model("nonexistent")
        assert model == mock_image_models["model1"]  # Should return default

    def test_get_model_with_empty_models(self, mock_workspace):
        """Test _get_model method with no models"""
        tool = ImageGenerationToolCore({}, workspace=mock_workspace)
        model = tool._get_model()
        assert model is None

    def test_model_info_text_generation(self, mock_workspace):
        """Test that model info text is generated correctly"""
        # Create mock image models with descriptions
        mock_model1 = Mock(spec=BaseImageModel)
        mock_model2 = Mock(spec=BaseImageModel)

        image_models = {
            "model1": mock_model1,
            "model2": mock_model2,
        }

        model_descriptions = {
            "model1": "Test model 1 description",
            "model2": "Test model 2 description",
        }

        image_tool = ImageGenerationToolCore(
            image_models,  # pyright: ignore[reportArgumentType]
            model_descriptions,
            workspace=mock_workspace,
        )

        # Check that model info text was generated during initialization
        assert hasattr(image_tool, "_model_info_text")

        # Verify the format
        model_info = image_tool._model_info_text
        lines = model_info.split("\n")

        # Should have one line per model
        assert len(lines) == 2
        assert "- model1: Test model 1 description ✎" in lines
        assert "- model2: Test model 2 description ✎" in lines

    def test_model_info_text_generation_without_descriptions(self, mock_workspace):
        """Test model info text generation when models have no descriptions"""
        mock_models = {"model1": Mock(spec=BaseImageModel)}

        # Create tool without descriptions
        image_tool = ImageGenerationToolCore(
            mock_models,  # pyright: ignore[reportArgumentType]
            {},
            workspace=mock_workspace,
        )

        # Check that it handles missing descriptions gracefully
        model_info = image_tool._model_info_text
        assert "- model1: No description available" in model_info

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_edit_image_success(
        self, mock_get, mock_image_models, mock_workspace
    ):
        """Test successful image editing"""
        # Configure mock model to support editing
        mock_image_models["model1"].edit_image = AsyncMock(
            return_value={
                "image_url": "https://example.com/edited_image.jpg",
                "usage": {"input_tokens": 15, "output_tokens": 25},
                "request_id": "edit_req1",
            }
        )
        # Add has_ability method to indicate edit capability
        mock_image_models["model1"].has_ability = Mock(return_value=True)

        tool = ImageGenerationToolCore(
            mock_image_models, {"model1": "Test model 1"}, mock_workspace
        )

        # Mock HTTP response for image download
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_edited_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await tool.edit_image(
            prompt="Make it look like a painting",
            image_url="https://example.com/original.jpg",
        )

        assert result["success"] is True
        assert result["image_path"] is not None
        assert result["model_used"] == "default_edit_model"
        assert result["usage"] == {"input_tokens": 15, "output_tokens": 25}
        assert result["request_id"] == "edit_req1"
        assert result["saved_to_workspace"] is True

        # Verify the model's edit_image was called
        mock_image_models["model1"].edit_image.assert_called_once_with(
            prompt="Make it look like a painting",
            image_url="https://example.com/original.jpg",
            size="1024*1024",
            negative_prompt="",
        )

    @pytest.mark.asyncio
    async def test_edit_image_with_no_edit_models(
        self, mock_image_models, mock_workspace
    ):
        """Test image editing when no models support editing"""
        # Models don't have has_ability method or don't support editing
        for model in mock_image_models.values():
            model.has_ability = Mock(return_value=False)

        tool = ImageGenerationToolCore(mock_image_models, workspace=mock_workspace)

        result = await tool.edit_image(
            prompt="Make it look like a painting",
            image_url="https://example.com/original.jpg",
        )

        assert result["success"] is False
        assert "No available image models with edit capabilities" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_image_with_model_error(self, mock_image_models, mock_workspace):
        """Test image editing when model raises an exception"""
        # Configure mock model to support editing but raise error
        mock_image_models["model1"].edit_image = AsyncMock(
            side_effect=Exception("Edit model error")
        )
        # Add has_ability method to indicate edit capability
        mock_image_models["model1"].has_ability = Mock(return_value=True)

        tool = ImageGenerationToolCore(mock_image_models, workspace=mock_workspace)

        result = await tool.edit_image(
            prompt="Make it look like a painting",
            image_url="https://example.com/original.jpg",
        )

        assert result["success"] is False
        assert result["error"] == "Edit model error"
        assert result["model_used"] == "default_edit_model"

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_edit_image_with_multiple_images(
        self, mock_get, mock_image_models, mock_workspace
    ):
        """Test image editing with multiple input images"""
        # Configure mock model to support editing
        mock_image_models["model1"].edit_image = AsyncMock(
            return_value={
                "image_url": "https://example.com/edited_image.jpg",
                "usage": {"input_tokens": 20, "output_tokens": 30},
                "request_id": "edit_req2",
            }
        )
        # Add has_ability method to indicate edit capability
        mock_image_models["model1"].has_ability = Mock(return_value=True)

        tool = ImageGenerationToolCore(
            mock_image_models, {"model1": "Test model 1"}, mock_workspace
        )

        # Mock HTTP response for image download
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_edited_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await tool.edit_image(
            prompt="Combine these images",
            image_url=[
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
            ],
        )

        assert result["success"] is True
        assert result["image_path"] is not None
        assert result["saved_to_workspace"] is True

        # Verify the model's edit_image was called with list
        mock_image_models["model1"].edit_image.assert_called_once_with(
            prompt="Combine these images",
            image_url=[
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
            ],
            size="1024*1024",
            negative_prompt="",
        )

    @pytest.mark.parametrize(
        "edit_kwargs,expected_kwargs",
        [
            ({"size": "2048*2048"}, {"size": "2048*2048"}),
            ({"width": 1920, "height": 1080}, {"width": 1920, "height": 1080}),
            ({"resolution": "1920x1080"}, {"resolution": "1920x1080"}),
            ({"aspect_ratio": "16:9"}, {"aspect_ratio": "16:9"}),
            (
                {"size": "2048*2048", "aspect_ratio": "1:1"},
                {"size": "2048*2048", "aspect_ratio": "1:1"},
            ),
        ],
        ids=["size", "width_height", "resolution", "aspect_ratio", "size_and_aspect"],
    )
    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_edit_image_size_parameters(
        self, mock_get, edit_image_tool, mock_image_models, edit_kwargs, expected_kwargs
    ):
        """Test image editing with various size parameter combinations."""
        _setup_edit_http_mock(mock_get)

        result = await edit_image_tool.edit_image(
            prompt="Make it look like a painting",
            image_url="https://example.com/original.jpg",
            **edit_kwargs,
        )

        assert result["success"] is True
        mock_image_models["model1"].edit_image.assert_called_once()
        call_kwargs = mock_image_models["model1"].edit_image.call_args.kwargs
        for key, value in expected_kwargs.items():
            assert call_kwargs[key] == value

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_edit_image_default_size_is_passed(
        self, mock_get, edit_image_tool, mock_image_models
    ):
        """Test that default size is passed to the model (consistent with generate_image)."""
        _setup_edit_http_mock(mock_get)

        result = await edit_image_tool.edit_image(
            prompt="Make it look like a painting",
            image_url="https://example.com/original.jpg",
        )

        assert result["success"] is True
        mock_image_models["model1"].edit_image.assert_called_once()
        call_kwargs = mock_image_models["model1"].edit_image.call_args.kwargs
        assert "size" in call_kwargs
        assert call_kwargs["size"] == "1024*1024"
