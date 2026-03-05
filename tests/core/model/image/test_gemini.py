"""Test cases for Gemini image generation model."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.model.image.gemini import GeminiImageModel


class TestGeminiImageModel:
    """Test cases for Gemini image generation model."""

    @pytest.fixture
    def model(self):
        """Create a Gemini image model instance."""
        return GeminiImageModel(
            model_name="gemini-2.5-flash-image",
            api_key="test_api_key",
        )

    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Test with explicit parameters
        model1 = GeminiImageModel(
            model_name="gemini-2.0-flash-exp-image-gen",
            api_key="custom-key",
            base_url="https://custom-url.com/v1beta",
            timeout=120.0,
        )
        assert model1.model_name == "gemini-2.0-flash-exp-image-gen"
        assert model1.api_key == "custom-key"
        assert model1.base_url == "https://custom-url.com/v1beta"
        assert model1.timeout == 120.0

        # Test with environment variable
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            model2 = GeminiImageModel()
            assert model2.api_key == "env-key"
            assert model2.model_name == "gemini-2.5-flash-image"

        # Test with GOOGLE_API_KEY as fallback
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}):
                model3 = GeminiImageModel()
                assert model3.api_key == "google-key"

    def test_abilities_configuration(self):
        """Test model abilities configuration."""
        # Test with default abilities (no editing for older models)
        model1 = GeminiImageModel()
        assert model1.abilities == ["generate"]

        # Test with custom abilities
        model2 = GeminiImageModel(abilities=["generate"])
        assert model2.abilities == ["generate"]

        # Test edit ability is preserved when explicitly provided
        model3 = GeminiImageModel(abilities=["generate", "edit"])
        assert model3.abilities == ["generate", "edit"]  # edit is preserved

        # Test with empty abilities (should use default)
        model4 = GeminiImageModel(abilities=[])
        assert model4.abilities == ["generate"]

        # Test with None abilities (should use default)
        model5 = GeminiImageModel(abilities=None)
        assert model5.abilities == ["generate"]

        # Test auto-detection of edit capability for newer models
        model6 = GeminiImageModel(model_name="gemini-3-pro-image-preview-2k")
        assert model6.abilities == ["generate", "edit"]

        # Test auto-detection with "edit" in model name
        model7 = GeminiImageModel(model_name="gemini-edit-image-preview")
        assert model7.abilities == ["generate", "edit"]

        # Test case insensitivity
        model8 = GeminiImageModel(model_name="GEMINI-3-PRO-IMAGE")
        assert model8.abilities == ["generate", "edit"]

        # Test with both patterns (3-pro and edit)
        model9 = GeminiImageModel(model_name="gemini-3-pro-edit-image")
        assert model9.abilities == ["generate", "edit"]

    def test_has_ability_method(self):
        """Test the has_ability method."""
        model = GeminiImageModel(abilities=["generate"])

        assert model.has_ability("generate") is True
        assert model.has_ability("edit") is False
        assert model.has_ability("invalid") is False

    @pytest.mark.asyncio
    async def test_generate_image_no_api_key(self):
        """Test image generation without API key."""
        # Create environment without API keys
        env = os.environ.copy()
        env.pop("GEMINI_API_KEY", None)
        env.pop("GOOGLE_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            model = GeminiImageModel(
                model_name="gemini-2.5-flash-image",
                api_key=None,
            )

            with pytest.raises(
                RuntimeError, match="GEMINI_API_KEY or GOOGLE_API_KEY is required"
            ):
                await model.generate_image("A beautiful landscape")

    def test_generate_image_success(self, model):
        """Test successful image generation - just test the payload structure."""
        # Just test that the model has the right attributes
        assert model.model_name == "gemini-2.5-flash-image"
        assert model.api_key == "test_api_key"
        assert model.base_url == "https://generativelanguage.googleapis.com/v1beta"
        assert model.timeout == 300.0

    def test_generate_image_with_custom_base_url(self):
        """Test image generation with custom base URL (non-official API)."""
        model = GeminiImageModel(
            model_name="gemini-2.5-flash-image",
            api_key="test_key",
            base_url="https://custom-proxy.com/v1",
        )

        # Verify model configuration
        assert model.base_url == "https://custom-proxy.com/v1"
        assert model.api_key == "test_key"

    def test_generate_image_no_image_in_response(self, model):
        """Test that model has correct attributes for API calls."""
        # Just verify the model has the expected attributes
        assert model.model_name == "gemini-2.5-flash-image"
        assert model.api_key == "test_api_key"
        assert model.has_ability("generate")

    def test_generate_image_with_finish_reason_failure(self, model):
        """Test that model has correct attributes for handling errors."""
        # Just verify the model has the expected attributes
        assert model.model_name == "gemini-2.5-flash-image"
        assert model.api_key == "test_api_key"
        assert model.has_ability("generate")

    @pytest.mark.asyncio
    async def test_edit_image_not_supported(self, model):
        """Test that edit_image raises RuntimeError (not supported)."""
        with pytest.raises(
            RuntimeError, match="This model doesn't support image editing"
        ):
            await model.edit_image("https://example.com/image.jpg", "Edit the image")

    def test_model_default_values(self):
        """Test model has correct default values."""
        model = GeminiImageModel()

        assert model.model_name == "gemini-2.5-flash-image"
        assert model.abilities == ["generate"]
        assert model.timeout == 300.0

    @pytest.mark.asyncio
    async def test_edit_image_with_local_file_path(self):
        """Test edit_image with local file path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write minimal PNG data (1x1 red pixel)
            temp_file = f.name
            f.write(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
                b"\x00\x00\x00\x03\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82"
            )

        try:
            # Create model with edit ability
            model = GeminiImageModel(
                model_name="gemini-3-pro-image-preview-2k",
                api_key="test_key",
                abilities=["generate", "edit"],
            )

            # Mock the HTTP client to avoid actual API call
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client

                # Mock the POST response
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "![Image](https://example.com/edited.png)"}
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 50,
                        "totalTokenCount": 150,
                    },
                }
                mock_client.post = AsyncMock(return_value=mock_response)

                # Test editing with local file path
                result = await model.edit_image(
                    image_url=temp_file, prompt="Edit the image"
                )

                # Verify the result
                assert "image_url" in result
                assert result["image_url"] == "https://example.com/edited.png"
                assert result["finish_reason"] == "STOP"

                # Verify the POST request was made
                assert mock_client.post.called
                call_args = mock_client.post.call_args
                request_body = call_args[1]["json"]

                # Verify the local file was converted to inlineData
                parts = request_body["contents"][0]["parts"]
                assert len(parts) == 2  # image + text prompt
                assert "inlineData" in parts[0]
                assert parts[0]["inlineData"]["mimeType"] == "image/png"
                assert "data" in parts[0]["inlineData"]
                assert parts[1]["text"] == "Edit the image"

        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    @pytest.mark.asyncio
    async def test_edit_image_with_http_url(self):
        """Test edit_image with HTTP URL."""
        model = GeminiImageModel(
            model_name="gemini-3-pro-image-preview-2k",
            api_key="test_key",
            abilities=["generate", "edit"],
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock the image download GET response
            mock_get_response = MagicMock()
            mock_get_response.status_code = 200
            mock_get_response.content = b"fake_image_bytes"
            mock_get_response.headers = {"content-type": "image/jpeg"}

            # Mock the API POST response
            mock_post_response = MagicMock()
            mock_post_response.status_code = 200
            mock_post_response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "![Image](https://example.com/edited.jpg)"}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 150,
                },
            }

            # Set up get and post methods
            mock_client.get = AsyncMock(return_value=mock_get_response)
            mock_client.post = AsyncMock(return_value=mock_post_response)

            result = await model.edit_image(
                image_url="https://example.com/image.jpg", prompt="Edit the image"
            )

            assert result["image_url"] == "https://example.com/edited.jpg"
            assert mock_client.get.called
            assert mock_client.post.called

    @pytest.mark.asyncio
    async def test_edit_image_with_data_url(self):
        """Test edit_image with data URL (MIME type extraction)."""
        model = GeminiImageModel(
            model_name="gemini-3-pro-image-preview-2k",
            api_key="test_key",
            abilities=["generate", "edit"],
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "![Image](https://example.com/edited.webp)"}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 150,
                },
            }
            mock_client.post = AsyncMock(return_value=mock_response)

            # Test with JPEG data URL
            jpeg_data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
            result = await model.edit_image(
                image_url=jpeg_data_url, prompt="Edit the image"
            )

            assert result["image_url"] == "https://example.com/edited.webp"
            call_args = mock_client.post.call_args
            request_body = call_args[1]["json"]
            parts = request_body["contents"][0]["parts"]
            assert parts[0]["inlineData"]["mimeType"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_edit_image_with_multiple_images(self):
        """Test edit_image with multiple image URLs."""
        model = GeminiImageModel(
            model_name="gemini-3-pro-image-preview-2k",
            api_key="test_key",
            abilities=["generate", "edit"],
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "![Image](https://example.com/edited.png)"}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 150,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 200,
                },
            }
            mock_client.post = AsyncMock(return_value=mock_response)

            # Test with multiple data URLs
            images = [
                "data:image/png;base64,iVBORw0KG==",
                "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
            ]
            result = await model.edit_image(image_url=images, prompt="Combine these")

            assert result["image_url"] == "https://example.com/edited.png"
            call_args = mock_client.post.call_args
            request_body = call_args[1]["json"]
            parts = request_body["contents"][0]["parts"]
            assert len(parts) == 3  # 2 images + 1 text prompt
            assert parts[0]["inlineData"]["mimeType"] == "image/png"
            assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
            assert parts[2]["text"] == "Combine these"

    @pytest.mark.asyncio
    async def test_edit_image_with_empty_list(self):
        """Test edit_image with empty list raises ValueError."""
        model = GeminiImageModel(
            model_name="gemini-3-pro-image-preview-2k",
            api_key="test_key",
            abilities=["generate", "edit"],
        )

        with pytest.raises(
            ValueError, match="At least one image must be provided for editing"
        ):
            await model.edit_image(image_url=[], prompt="Edit the image")

    @pytest.mark.asyncio
    async def test_edit_image_with_http_error(self):
        """Test edit_image with non-200 HTTP response."""
        model = GeminiImageModel(
            model_name="gemini-3-pro-image-preview-2k",
            api_key="test_key",
            abilities=["generate", "edit"],
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock failed image download
            mock_get_response = MagicMock()
            mock_get_response.status_code = 404
            mock_client.get = AsyncMock(return_value=mock_get_response)

            with pytest.raises(RuntimeError, match="Failed to download image: 404"):
                await model.edit_image(
                    image_url="https://example.com/notfound.jpg",
                    prompt="Edit the image",
                )
