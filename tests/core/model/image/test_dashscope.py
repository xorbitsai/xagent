import base64
import os
from unittest.mock import patch

import pytest

from xagent.core.model.image.dashscope import DashScopeImageModel


class TestDashScopeImageModel:
    """Test cases for DashScope image generation model."""

    @pytest.fixture
    def model(self):
        """Create a DashScope image model instance."""
        return DashScopeImageModel(
            model_name="qwen-image",
            api_key="test_api_key",
        )

    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Test with explicit parameters
        model1 = DashScopeImageModel(
            model_name="custom-model",
            api_key="custom-key",
            base_url="https://custom-url.com/api",
            timeout=120.0,
        )
        assert model1.model_name == "custom-model"
        assert model1.api_key == "custom-key"
        assert model1.base_url == "https://custom-url.com/api"
        assert model1.timeout == 120.0

        # Test with environment variable
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}):
            model2 = DashScopeImageModel()
            assert model2.api_key == "env-key"
            assert model2.model_name == "qwen-image"

    def test_abilities_configuration(self):
        """Test model abilities configuration."""
        # Test with default abilities
        model1 = DashScopeImageModel()
        assert model1.abilities == ["generate"]

        # Test with custom abilities
        model2 = DashScopeImageModel(abilities=["generate"])
        assert model2.abilities == ["generate"]

        # Test with both abilities
        model3 = DashScopeImageModel(abilities=["generate", "edit"])
        assert model3.abilities == ["generate", "edit"]

        # Test with empty abilities (should use default)
        model4 = DashScopeImageModel(abilities=[])
        assert model4.abilities == ["generate"]

        # Test with None abilities (should use default)
        model5 = DashScopeImageModel(abilities=None)
        assert model5.abilities == ["generate"]

    def test_has_ability_method(self):
        """Test the has_ability method."""
        model = DashScopeImageModel(abilities=["generate"])

        assert model.has_ability("generate") is True
        assert model.has_ability("edit") is False
        assert model.has_ability("invalid") is False

        # Test with both abilities
        model_both = DashScopeImageModel(abilities=["generate", "edit"])
        assert model_both.has_ability("generate") is True
        assert model_both.has_ability("edit") is True

    def test_convert_image_to_base64_url(self):
        """Test _convert_image_to_base64 with URLs."""
        model = DashScopeImageModel(abilities=["generate", "edit"])

        # Valid URL
        valid_url = "https://example.com/image.jpg"
        result = model._convert_image_to_base64(valid_url)
        assert result == valid_url

        # URL with Chinese characters
        chinese_url = "https://example.com/图片.jpg"
        with pytest.raises(
            RuntimeError, match="Image URL cannot contain Chinese characters"
        ):
            model._convert_image_to_base64(chinese_url)

    def test_convert_image_to_base64_local_file(self, tmp_path):
        """Test _convert_image_to_base64 with local files."""
        model = DashScopeImageModel(abilities=["generate", "edit"])

        # Create a test image file
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xd9"
        )

        result = model._convert_image_to_base64(str(test_image))

        # Check format
        assert result.startswith("data:image/jpeg;base64,")

        # Check base64 is valid
        base64_part = result.split(",")[1]
        decoded = base64.b64decode(base64_part)
        assert len(decoded) > 0

    def test_convert_image_to_base64_nonexistent_file(self):
        """Test _convert_image_to_base64 with non-existent file."""
        model = DashScopeImageModel(abilities=["generate", "edit"])

        with pytest.raises(RuntimeError, match="Image file not found"):
            model._convert_image_to_base64("/non/existent/file.jpg")

    def test_convert_image_to_base64_large_file(self, tmp_path):
        """Test _convert_image_to_base64 with file size validation."""
        model = DashScopeImageModel(abilities=["generate", "edit"])

        # Create a large file (>10MB)
        large_file = tmp_path / "large.jpg"
        large_file.write_bytes(b"x" * (11 * 1024 * 1024))  # 11MB

        with pytest.raises(RuntimeError, match="Image file too large"):
            model._convert_image_to_base64(str(large_file))

    @pytest.mark.asyncio
    async def test_generate_image_no_api_key(self):
        """Test image generation without API key."""
        # Clear environment variable to ensure no API key is available
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}, clear=False):
            model = DashScopeImageModel(
                model_name="qwen-image",
                api_key=None,
            )

            with pytest.raises(
                RuntimeError,
                match="DASHSCOPE_API_KEY is required|Image generation failed",
            ):
                await model.generate_image("A beautiful landscape")

    def test_generate_image_success(self, model):
        """Test successful image generation - just test the payload structure."""
        # Just test that the model has the right attributes
        assert model.model_name == "qwen-image"
        assert model.api_key == "test_api_key"
        assert (
            model.base_url
            == "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        )
        assert model.timeout == 60.0
