"""
Tests for Image Generation tool
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from xagent.core.model.image.base import BaseImageModel
from xagent.core.tools.adapters.vibe.image_tool import (
    ImageGenerationTool,
    create_image_tool,
)


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
def image_tool(mock_image_models, mock_workspace):
    """Create ImageGenerationTool instance for testing"""
    return ImageGenerationTool(
        mock_image_models,
        {"model1": "Test model 1", "model2": "Test model 2"},
        mock_workspace,
    )


@pytest.fixture
def model_descriptions():
    """Create model descriptions for testing"""
    return {"model1": "Test model 1", "model2": "Test model 2"}


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


class TestImageGenerationTool:
    """Test cases for ImageGenerationTool class"""

    def test_init_with_models(self, mock_image_models, mock_workspace):
        """Test ImageGenerationTool initialization with models"""
        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)
        assert tool._image_models == mock_image_models
        assert len(tool._image_models) == 2

    def test_init_with_empty_models(self, mock_workspace):
        """Test ImageGenerationTool initialization with empty models"""
        tool = ImageGenerationTool({}, workspace=mock_workspace)
        assert tool._image_models == {}
        assert len(tool._image_models) == 0

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_generate_image_with_default_model(
        self, mock_get, image_tool, mock_image_models
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

        result = await image_tool.generate_image("A test prompt")

        assert result["success"] is True
        assert result["image_path"] is not None
        assert result["model_used"] == "model1"
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
        self, mock_get, image_tool, mock_image_models
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

        result = await image_tool.generate_image("A test prompt", model_id="model2")

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
    @patch("aiohttp.ClientSession.get")
    async def test_generate_image_with_custom_parameters(
        self, mock_get, image_tool, mock_image_models
    ):
        """Test image generation with custom parameters"""
        # Mock HTTP response for image download
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await image_tool.generate_image(
            "A test prompt",
            size="512*512",
            negative_prompt="blurry, low quality",
            model_id="model1",
        )

        assert result["success"] is True

        # Verify all parameters were passed correctly
        mock_image_models["model1"].generate_image.assert_called_once_with(
            prompt="A test prompt",
            size="512*512",
            negative_prompt="blurry, low quality",
        )

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_generate_image_with_nonexistent_model(
        self, mock_get, image_tool, mock_image_models
    ):
        """Test image generation with non-existent model"""
        # Mock HTTP response for image download
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        result = await image_tool.generate_image(
            "A test prompt", model_id="nonexistent_model"
        )

        assert result["success"] is True
        assert result["model_used"] == "model1"

        # Should fall back to first available model
        mock_image_models["model1"].generate_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_image_with_no_models(self, mock_workspace):
        """Test image generation with no models available"""
        tool = ImageGenerationTool({}, workspace=mock_workspace)
        result = await tool.generate_image("A test prompt")

        assert result["success"] is False
        assert "No available image models with generate capabilities" in result["error"]
        assert "Configured image models:" in result["error"]
        assert result["image_path"] is None

    @pytest.mark.asyncio
    async def test_generate_image_without_workspace_raises_error(
        self, image_tool, mock_image_models
    ):
        """Test that image generation without workspace raises an error"""
        # Creating tool without workspace should raise ValueError
        with pytest.raises(ValueError, match="Workspace is required"):
            ImageGenerationTool(mock_image_models, {"model1": "Test model 1"})

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_generate_image_with_workspace(
        self, mock_get, image_tool, mock_image_models, mock_workspace
    ):
        """Test image generation with workspace (should download and save)"""
        # Create tool with workspace
        tool = ImageGenerationTool(
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
    async def test_download_image_success(self, mock_get, mock_workspace):
        """Test successful image download"""
        # Create tool with workspace

        tool = ImageGenerationTool({}, {}, mock_workspace)

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

    @pytest.mark.asyncio
    async def test_download_image_no_workspace(self):
        """Test that creating tool without workspace raises error"""
        # Creating tool without workspace should raise ValueError
        from xagent.core.tools.adapters.vibe.image_tool import ImageGenerationTool

        with pytest.raises(ValueError, match="Workspace is required"):
            ImageGenerationTool({}, {})

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_download_image_http_error(self, mock_get, mock_workspace):
        """Test image download with HTTP error"""
        # Create tool with workspace

        tool = ImageGenerationTool({}, {}, mock_workspace)

        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status = 404

        # Make response.text() async
        async def mock_text():
            return "404 Not Found"

        mock_response.text = mock_text
        mock_get.return_value.__aenter__.return_value = mock_response

        with pytest.raises(RuntimeError, match="Failed to download image: HTTP 404"):
            await tool._download_image("https://example.com/test.png")

    @pytest.mark.asyncio
    async def test_generate_image_with_model_error(
        self, mock_image_models, mock_workspace
    ):
        """Test image generation when model raises an exception"""
        # Configure mock to raise an exception
        mock_image_models["model1"].generate_image.side_effect = Exception(
            "Model error"
        )

        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)
        result = await tool.generate_image("A test prompt")

        assert result["success"] is False
        assert result["error"] == "Model error"
        assert result["image_path"] is None
        assert result["model_used"] == "model1"

    def test_list_available_models(self, image_tool, mock_image_models):
        """Test listing available models"""
        result = image_tool.list_available_models()

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
        tool = ImageGenerationTool({}, workspace=mock_workspace)
        result = tool.list_available_models()

        assert result["success"] is True
        assert result["count"] == 0
        assert result["models"] == []

    def test_get_model_with_id(self, image_tool, mock_image_models):
        """Test _get_model method with specific model ID"""
        model = image_tool._get_model("model2")
        assert model == mock_image_models["model2"]

    def test_get_model_with_default(self, image_tool, mock_image_models):
        """Test _get_model method with default model"""
        model = image_tool._get_model()
        assert model == mock_image_models["model1"]  # First model

    def test_get_model_with_nonexistent_id(self, image_tool, mock_image_models):
        """Test _get_model method with non-existent model ID"""
        model = image_tool._get_model("nonexistent")
        assert model == mock_image_models["model1"]  # Should return default

    def test_get_model_with_empty_models(self, mock_workspace):
        """Test _get_model method with no models"""
        tool = ImageGenerationTool({}, workspace=mock_workspace)
        model = tool._get_model()
        assert model is None

    def test_generate_image_tool_description_with_models(self, mock_workspace):
        """Test that generate_image tool description includes model information"""
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

        image_tool = ImageGenerationTool(
            image_models,  # pyright: ignore[reportArgumentType]
            model_descriptions,
            workspace=mock_workspace,
        )
        tools = image_tool.get_tools()

        # Find the generate_image tool
        generate_tool = None
        for tool in tools:
            if tool.metadata.name == "generate_image":
                generate_tool = tool
                break

        assert generate_tool is not None

        # Check that the description contains model information
        description = generate_tool.description
        assert "Available models" in description
        assert "⭐[DEFAULT]" in description
        assert "model1:" in description
        assert "model2:" in description
        assert "Test model 1 description" in description
        assert "Test model 2 description" in description
        assert "generate complete designed graphics" in description

    def test_generate_image_tool_schema_includes_reference_images(self, image_tool):
        """Test that generate_image exposes reference images to the model."""
        tools = image_tool.get_tools()
        generate_tool = next(
            tool for tool in tools if tool.metadata.name == "generate_image"
        )

        schema = generate_tool.args_type().model_json_schema()
        assert "images" in schema["properties"]
        assert "images (optional): source/reference image" in generate_tool.description

    def test_no_generate_tool_without_models(self, mock_workspace):
        """Test that no generate_image tool is offered when no model can generate"""
        image_tool = ImageGenerationTool({}, {}, workspace=mock_workspace)

        assert "generate_image" not in {t.metadata.name for t in image_tool.get_tools()}

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

        image_tool = ImageGenerationTool(
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
        image_tool = ImageGenerationTool(
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

        tool = ImageGenerationTool(
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
        assert result["model_used"] == "model1"
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

        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)

        result = await tool.edit_image(
            prompt="Make it look like a painting",
            image_url="https://example.com/original.jpg",
        )

        assert result["success"] is False
        assert "No available image models with edit capabilities" in result["error"]

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

        tool = ImageGenerationTool(
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

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_download_image_with_vibe_timeout(self, mock_get, mock_workspace):
        """Test that vibe adapter uses 3600 second timeout"""
        from xagent.core.tools.adapters.vibe.image_tool import ImageGenerationTool

        # Create vibe adapter tool
        tool = ImageGenerationTool({}, {}, mock_workspace)

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        # Call _download_image (should use 3600s timeout via override)
        result = await tool._download_image("https://example.com/test.png")

        assert result is not None
        assert "generated_image_" in result
        assert result.endswith(".png")

        # Verify the session was created (timeout is set in ClientSession constructor)
        mock_get.assert_called_once()


class TestCreateImageTool:
    """Test cases for create_image_tool function"""

    def test_create_image_tool_with_models(
        self, mock_image_models, model_descriptions, mock_workspace
    ):
        """Test create_image_tool function with models"""
        tools = create_image_tool(
            image_models=mock_image_models,
            model_descriptions=model_descriptions,
            workspace=mock_workspace,
        )

        assert len(tools) == 3

        # Check that tools have the correct names
        tool_names = [tool.metadata.name for tool in tools]
        assert "generate_image" in tool_names
        assert "list_image_models" in tool_names

    def test_create_image_tool_with_empty_models(self, mock_workspace):
        """Test create_image_tool function with empty models"""
        tools = create_image_tool({}, workspace=mock_workspace)

        # Nothing can generate or edit, so only the read-only listing remains.
        assert {t.metadata.name for t in tools} == {"list_image_models"}

    def test_create_image_tool_with_descriptions(
        self, mock_image_models, model_descriptions, mock_workspace
    ):
        """Test create_image_tool function with model descriptions"""
        tools = create_image_tool(
            image_models=mock_image_models,
            model_descriptions=model_descriptions,
            workspace=mock_workspace,
        )

        # Find the list_image_models tool
        list_tool = None
        for tool in tools:
            if tool.metadata.name == "list_image_models":
                list_tool = tool
                break

        assert list_tool is not None

        # Test that the tool works and returns descriptions
        result = list_tool.run_json_sync({})
        assert result["success"] is True
        assert result["count"] == 2

        # Check that descriptions are included
        for model in result["models"]:
            assert "description" in model
            assert len(model["description"]) > 0

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_generated_tools_functionality(
        self, mock_get, mock_image_models, mock_workspace
    ):
        """Test that generated tools actually work"""
        tools = create_image_tool(
            image_models=mock_image_models, workspace=mock_workspace
        )

        # Find the generate_image tool
        generate_tool = None
        for tool in tools:
            if tool.metadata.name == "generate_image":
                generate_tool = tool
                break

        assert generate_tool is not None

        # Mock HTTP response for image download
        mock_response = Mock()
        mock_response.status = 200

        # Create async iterator for chunks
        async def mock_iter_chunked(chunk_size):
            for chunk in [b"fake_image_data"]:
                yield chunk

        mock_response.content.iter_chunked = mock_iter_chunked
        mock_get.return_value.__aenter__.return_value = mock_response

        # Test the tool functionality using run_json_async
        result = await generate_tool.run_json_async(
            {"prompt": "test prompt", "kwargs": {}}
        )

        assert result["success"] is True
        assert result["image_path"] is not None
        assert result["saved_to_workspace"] is True

    def test_list_models_tool_functionality(self, mock_image_models, mock_workspace):
        """Test that list_models tool works"""
        tools = create_image_tool(
            image_models=mock_image_models, workspace=mock_workspace
        )

        # Find the list_image_models tool
        list_tool = None
        for tool in tools:
            if tool.metadata.name == "list_image_models":
                list_tool = tool
                break

        assert list_tool is not None

        # Test the tool functionality using run_json_sync (sync method)
        result = list_tool.run_json_sync({})

        assert result["success"] is True
        assert result["count"] == 2


class TestImageToolCapabilityGating:
    """A tool no configured model can serve must leave the schema entirely."""

    @staticmethod
    def _set_abilities(models, abilities):
        for model in models.values():
            model.abilities = list(abilities)
            model.has_ability = Mock(side_effect=lambda a: a in abilities)

    def test_edit_image_withheld_when_no_model_can_edit(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, ["generate"])

        tools = ImageGenerationTool(
            mock_image_models, workspace=mock_workspace
        ).get_tools()
        names = {tool.metadata.name for tool in tools}

        assert "edit_image" not in names
        assert "generate_image" in names
        generate = next(t for t in tools if t.metadata.name == "generate_image")
        assert "IMAGE EDITING IS UNAVAILABLE" in generate.description

    def test_edit_image_offered_when_a_model_can_edit(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, ["generate", "edit"])

        tools = ImageGenerationTool(
            mock_image_models, workspace=mock_workspace
        ).get_tools()
        names = {tool.metadata.name for tool in tools}

        assert "edit_image" in names
        generate = next(t for t in tools if t.metadata.name == "generate_image")
        assert "IMAGE EDITING IS UNAVAILABLE" not in generate.description

    @pytest.mark.asyncio
    async def test_error_lists_abilities_and_points_at_generate_image(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, ["generate"])
        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)

        result = await tool.edit_image(prompt="add a headline", image_url="file:abc123")

        assert result["success"] is False
        assert "model1 (abilities: generate)" in result["error"]
        assert "retry generate_image without images" in result["error"]

    def test_default_edit_model_that_denies_edit_is_not_trusted(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, ["generate"])
        tool = ImageGenerationTool(
            mock_image_models,
            workspace=mock_workspace,
            default_edit_model=mock_image_models["model1"],
        )

        assert tool._get_edit_model() is None
        assert "edit_image" not in {t.metadata.name for t in tool.get_tools()}

    def test_generate_image_withheld_when_only_editing_is_available(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, ["edit"])

        tools = ImageGenerationTool(
            mock_image_models, workspace=mock_workspace
        ).get_tools()
        names = {tool.metadata.name for tool in tools}

        assert "generate_image" not in names
        assert "edit_image" in names

    def test_no_tools_when_configured_models_can_do_neither(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, [])

        tools = ImageGenerationTool(
            mock_image_models, workspace=mock_workspace
        ).get_tools()

        # list_image_models is read-only and cannot fail, so it stays as the
        # only way to answer "why is there nothing here".
        assert {t.metadata.name for t in tools} == {"list_image_models"}

    def test_default_generate_model_that_denies_generate_is_not_trusted(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, ["edit"])
        tool = ImageGenerationTool(
            mock_image_models,
            workspace=mock_workspace,
            default_generate_model=mock_image_models["model1"],
        )

        assert tool._get_model() is None
        assert "generate_image" not in {t.metadata.name for t in tool.get_tools()}

    @pytest.mark.asyncio
    async def test_delegated_edit_does_not_recommend_the_tool_that_was_called(
        self, mock_image_models, mock_workspace
    ):
        # generate_image(images=...) delegates into the edit path, so answering it
        # with "use generate_image" would send the model back where it started.
        self._set_abilities(mock_image_models, ["generate"])
        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)

        result = await tool.generate_image("add a headline", images="file:abc123")

        assert result["success"] is False
        assert "retry generate_image without images" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_error_stops_retrying_when_nothing_can_generate_either(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, [])
        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)

        result = await tool.edit_image(prompt="add a headline", image_url="file:abc")

        assert result["success"] is False
        assert "stop retrying image tools" in result["error"]
        assert "retry generate_image" not in result["error"]
        assert "model1 (abilities: none)" in result["error"]

    @pytest.mark.asyncio
    async def test_wrong_model_id_points_at_the_models_that_can_serve_it(
        self, mock_image_models, mock_workspace
    ):
        # A model_id that cannot edit must not be reported as "editing is
        # unavailable" while another configured model can edit.
        mock_image_models["model1"].abilities = ["generate"]
        mock_image_models["model1"].has_ability = Mock(
            side_effect=lambda a: a == "generate"
        )
        mock_image_models["model2"].abilities = ["generate", "edit"]
        mock_image_models["model2"].has_ability = Mock(side_effect=lambda a: True)
        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)

        edit = await tool.edit_image(prompt="x", image_url="file:a", model_id="model1")

        assert "Model model1 cannot edit" in edit["error"]
        assert "retry edit_image without model_id" in edit["error"]
        assert "editing is unavailable" not in edit["error"].lower()

    def test_listing_reports_what_each_model_can_actually_serve(
        self, mock_image_models, mock_workspace
    ):
        # list_image_models is the only tool left in this state, so it has to
        # explain the absence rather than claim everything is available.
        self._set_abilities(mock_image_models, [])
        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)

        listed = tool.list_available_models()["models"]

        assert [m["available"] for m in listed] == [False, False]
        assert [m["abilities"] for m in listed] == [[], []]

        mock_image_models["model1"].has_ability = Mock(
            side_effect=lambda a: a == "edit"
        )
        edit_only = ImageGenerationTool(
            mock_image_models, workspace=mock_workspace
        ).list_available_models()["models"]

        assert edit_only[0] == {
            "model_id": "model1",
            "available": True,
            "abilities": ["edit"],
            "description": "",
        }

    def test_a_model_without_has_ability_is_not_trusted(self, mock_workspace):
        # fail closed: an object that never declares the ability cannot serve it,
        # whether it arrives as a default or from the configured mapping.
        opaque = SimpleNamespace(abilities=["generate", "edit"])
        tool = ImageGenerationTool(
            {"opaque": opaque},
            workspace=mock_workspace,
            default_generate_model=opaque,
            default_edit_model=opaque,
        )

        assert tool._get_model() is None
        assert tool._get_edit_model() is None

    @pytest.mark.asyncio
    async def test_generate_error_lists_abilities(
        self, mock_image_models, mock_workspace
    ):
        self._set_abilities(mock_image_models, ["edit"])
        tool = ImageGenerationTool(mock_image_models, workspace=mock_workspace)

        result = await tool.generate_image("a poster", model_id="model1")

        assert result["success"] is False
        assert "model1 (abilities: edit)" in result["error"]


@pytest.mark.asyncio
async def test_creator_propagates_a_broken_model_instead_of_hiding_it(tmp_path) -> None:
    # A blanket handler here would turn a broken model class into "this
    # deployment has no image tools", indistinguishable from having none.
    from xagent.core.tools.adapters.vibe.config import ToolConfig
    from xagent.core.tools.adapters.vibe.image_tool import (
        create_image_tools_from_config,
    )

    broken = Mock(spec=BaseImageModel)
    broken.has_ability = Mock(side_effect=RuntimeError("ability probe exploded"))

    class _Config(ToolConfig):
        def get_image_models(self):
            return {"broken": broken}

        def get_workspace_config(self):
            return {"task_id": "creator-propagation", "base_dir": str(tmp_path)}

    with pytest.raises(RuntimeError, match="ability probe exploded"):
        await create_image_tools_from_config(_Config({}))
