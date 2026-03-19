"""Unit tests for Xinference TTS model."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Handle optional xinference dependency
try:
    import xinference.client.restful.async_restful_client  # noqa: F401

    _ASYNC_CLIENT_PATH = "xinference.client.restful.async_restful_client.AsyncClient"
except ImportError:
    try:
        import xinference_client.client.restful.async_restful_client  # noqa: F401

        _ASYNC_CLIENT_PATH = (
            "xinference_client.client.restful.async_restful_client.AsyncClient"
        )
    except ImportError:
        pytest.skip(
            "Neither xinference nor xinference_client is installed",
            allow_module_level=True,
        )

from xagent.core.model.tts import TTSResult, XinferenceTTS


class TestXinferenceTTS:
    """Test cases for Xinference TTS model."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        tts = XinferenceTTS()
        assert tts.model == "chat-tts"
        assert tts.base_url == "http://localhost:9997"
        assert tts.format == "mp3"
        assert tts.sample_rate == 24000
        assert tts._client is None
        assert tts._model_handle is None

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        tts = XinferenceTTS(model="edge-tts")
        assert tts.model == "edge-tts"

    def test_init_with_base_url(self) -> None:
        """Test initialization with custom base URL."""
        tts = XinferenceTTS(base_url="http://remote-server:9997")
        assert tts.base_url == "http://remote-server:9997"

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key."""
        tts = XinferenceTTS(api_key="test-api-key")
        assert tts.api_key == "test-api-key"

    def test_init_with_voice(self) -> None:
        """Test initialization with voice."""
        tts = XinferenceTTS(voice="female")
        assert tts.voice == "female"

    def test_init_with_language(self) -> None:
        """Test initialization with language."""
        tts = XinferenceTTS(language="zh")
        assert tts.language == "zh"

    def test_init_with_format(self) -> None:
        """Test initialization with format."""
        tts = XinferenceTTS(format="wav")
        assert tts.format == "wav"

    def test_init_with_sample_rate(self) -> None:
        """Test initialization with sample rate."""
        tts = XinferenceTTS(sample_rate=48000)
        assert tts.sample_rate == 48000

    def test_init_with_model_uid(self) -> None:
        """Test initialization with custom model UID."""
        tts = XinferenceTTS(model="chat-tts", model_uid="custom-uid-123")
        assert tts.model == "chat-tts"
        assert tts._model_uid == "custom-uid-123"

    def test_init_with_base_url_trailing_slash(self) -> None:
        """Test that trailing slash is removed from base URL."""
        tts = XinferenceTTS(base_url="http://localhost:9997/")
        assert tts.base_url == "http://localhost:9997"

    def test_init_with_all_parameters(self) -> None:
        """Test initialization with all parameters."""
        tts = XinferenceTTS(
            model="fish-speech",
            model_uid="fish-123",
            base_url="http://remote:9997",
            api_key="test-key",
            voice="male",
            language="en",
            format="wav",
            sample_rate=48000,
        )
        assert tts.model == "fish-speech"
        assert tts._model_uid == "fish-123"
        assert tts.base_url == "http://remote:9997"
        assert tts.api_key == "test-key"
        assert tts.voice == "male"
        assert tts.language == "en"
        assert tts.format == "wav"
        assert tts.sample_rate == 48000

    def test_abilities(self) -> None:
        """Test that abilities property returns expected list."""
        tts = XinferenceTTS()
        abilities = tts.abilities
        assert isinstance(abilities, list)
        assert "tts" in abilities
        assert "text_to_speech" in abilities
        assert "audio" in abilities
        assert "audio_generation" in abilities
        assert "multilingual" in abilities
        assert "real_time" in abilities

    def test_abilities_with_voice(self) -> None:
        """Test that abilities include multiple_voices when voice is set."""
        tts = XinferenceTTS(voice="female")
        abilities = tts.abilities
        assert "multiple_voices" in abilities

    def test_supports_multiple_voices(self) -> None:
        """Test that supports_multiple_voices property works."""
        tts = XinferenceTTS()
        assert tts.supports_multiple_voices is False

        tts_with_voice = XinferenceTTS(voice="female")
        assert tts_with_voice.supports_multiple_voices is True

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_simple_audio_only(self, mock_client_class: Mock) -> None:
        """Test simple synthesis returning audio bytes only (verbose=False)."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("Hello, world!", verbose=False)

        # Verify
        assert isinstance(result, bytes)
        assert result == b"fake audio data"
        mock_model_handle.speech.assert_called_once()

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_verbose_with_metadata(
        self, mock_client_class: Mock
    ) -> None:
        """Test verbose synthesis returning TTSResult with metadata."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("Hello, world!", verbose=True)

        # Verify
        assert isinstance(result, TTSResult)
        assert result.audio == b"fake audio data"
        assert result.format == "mp3"
        assert result.sample_rate == 24000

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_with_custom_voice(self, mock_client_class: Mock) -> None:
        """Test synthesis with custom voice."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("Hello", voice="female", verbose=False)

        # Verify voice was passed to the API
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("voice") == "female"
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_with_language(self, mock_client_class: Mock) -> None:
        """Test synthesis with language parameter."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("Hello", language="en", verbose=False)

        # Verify language was passed
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("language") == "en"
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_with_custom_format(self, mock_client_class: Mock) -> None:
        """Test synthesis with custom format."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("Hello", format="wav", verbose=False)

        # Verify format was passed
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("output_audio_format") == "wav"
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_with_custom_sample_rate(
        self, mock_client_class: Mock
    ) -> None:
        """Test synthesis with custom sample rate."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("Hello", sample_rate=48000, verbose=False)

        # Verify sample_rate was passed
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("sample_rate") == 48000
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_with_speed_parameter(
        self, mock_client_class: Mock
    ) -> None:
        """Test synthesis with speed parameter."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("Hello", speed=1.2, verbose=False)

        # Verify speed was passed
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("speed") == 1.2
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_with_volume_parameter(
        self, mock_client_class: Mock
    ) -> None:
        """Test synthesis with volume parameter."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("Hello", volume=1.5, verbose=False)

        # Verify volume was passed
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("volume") == 1.5
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    @patch("builtins.open", new_callable=MagicMock)
    async def test_synthesize_with_reference_audio(
        self, mock_open: Mock, mock_client_class: Mock
    ) -> None:
        """Test synthesis with reference audio for voice cloning."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        # Mock file reading
        mock_file = MagicMock()
        mock_file.read.return_value = b"reference audio data"
        mock_open.return_value.__enter__.return_value = mock_file

        tts = XinferenceTTS()
        result = await tts.synthesize(
            "Hello, this is a cloned voice",
            reference_audio="/path/to/reference.wav",
            verbose=False,
        )

        # Verify reference_audio was read and passed as prompt_speech
        mock_open.assert_called_once_with("/path/to/reference.wav", "rb")
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("prompt_speech") == b"reference audio data"
        assert "reference_audio" not in call_kwargs  # Should be removed from kwargs
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_with_multiple_parameters(
        self, mock_client_class: Mock
    ) -> None:
        """Test synthesis with multiple custom parameters."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize(
            "Hello",
            voice="female",
            language="en",
            format="wav",
            sample_rate=48000,
            speed=1.3,
            volume=1.2,
            verbose=False,
        )

        # Verify all parameters were passed
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("voice") == "female"
        assert call_kwargs.get("language") == "en"
        assert call_kwargs.get("output_audio_format") == "wav"
        assert call_kwargs.get("sample_rate") == 48000
        assert call_kwargs.get("speed") == 1.3
        assert call_kwargs.get("volume") == 1.2
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_uses_init_defaults(self, mock_client_class: Mock) -> None:
        """Test that synthesis uses initialization defaults when not overridden."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS(
            voice="male", language="zh", format="wav", sample_rate=48000
        )
        result = await tts.synthesize("Hello", verbose=False)

        # Verify init defaults were used
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("voice") == "male"
        assert call_kwargs.get("language") == "zh"
        assert call_kwargs.get("output_audio_format") == "wav"
        assert call_kwargs.get("sample_rate") == 48000
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_override_init_defaults(
        self, mock_client_class: Mock
    ) -> None:
        """Test that synthesis parameters override initialization defaults."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS(
            voice="male", language="zh", format="wav", sample_rate=48000
        )
        result = await tts.synthesize(
            "Hello", voice="female", language="en", format="mp3", verbose=False
        )

        # Verify override parameters were used
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("voice") == "female"
        assert call_kwargs.get("language") == "en"
        assert call_kwargs.get("output_audio_format") == "mp3"
        # sample_rate should use init default since not overridden
        assert call_kwargs.get("sample_rate") == 48000
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_empty_text(self, mock_client_class: Mock) -> None:
        """Test synthesis with empty text."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("", verbose=False)

        # Verify empty text was passed through with default parameters
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("input") == ""
        assert call_kwargs.get("output_audio_format") == "mp3"
        assert call_kwargs.get("sample_rate") == 24000
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_unicode_text(self, mock_client_class: Mock) -> None:
        """Test synthesis with unicode text."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize("你好，世界！🎉", verbose=False)

        # Verify unicode text was passed through
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert call_kwargs.get("input") == "你好，世界！🎉"
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_long_text(self, mock_client_class: Mock) -> None:
        """Test synthesis with long text."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        long_text = "This is a very long text. " * 100
        tts = XinferenceTTS()
        result = await tts.synthesize(long_text, verbose=False)

        # Verify long text was passed through
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert len(call_kwargs.get("input", "")) == len(long_text)
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_special_characters(self, mock_client_class: Mock) -> None:
        """Test synthesis with special characters."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        result = await tts.synthesize(
            "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?", verbose=False
        )

        # Verify special characters were preserved
        mock_model_handle.speech.assert_called_once()
        call_args, call_kwargs = mock_model_handle.speech.call_args
        assert "@#$%^&*()" in call_kwargs.get("input", "")
        assert result == b"fake audio data"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_verbose_includes_all_metadata(
        self, mock_client_class: Mock
    ) -> None:
        """Test that verbose mode includes all metadata."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS(voice="female", language="en")
        result = await tts.synthesize("Hello", verbose=True)

        # Verify all metadata is included
        assert isinstance(result, TTSResult)
        assert result.audio == b"fake audio data"
        assert result.format == "mp3"
        assert result.sample_rate == 24000
        assert result.language == "en"
        assert result.raw_response is not None
        assert result.raw_response["model"] == "chat-tts"
        assert result.raw_response["voice"] == "female"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_error_handling(self, mock_client_class: Mock) -> None:
        """Test error handling when synthesis fails."""
        # Setup mock to return None
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=None)
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()

        with pytest.raises(RuntimeError, match="Unexpected audio data type"):
            await tts.synthesize("Hello")

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_exception_handling(self, mock_client_class: Mock) -> None:
        """Test exception handling when API call raises exception."""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(side_effect=Exception("API Error"))
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()

        with pytest.raises(RuntimeError, match="Xinference TTS failed"):
            await tts.synthesize("Hello")

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_invalid_response_type(
        self, mock_client_class: Mock
    ) -> None:
        """Test error handling when response is not bytes."""
        # Setup mock to return invalid type
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value={"audio": "not bytes"})
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()

        with pytest.raises(RuntimeError, match="Unexpected audio data type"):
            await tts.synthesize("Hello")

    @patch(_ASYNC_CLIENT_PATH)
    async def test_synthesize_get_model_failure(self, mock_client_class: Mock) -> None:
        """Test error handling when get_model raises exception."""
        # Setup mock to raise exception on get_model
        mock_client = MagicMock()
        mock_client.get_model.side_effect = Exception("Model not found")
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()

        # The exception from get_model is not caught by synthesize's try-except
        # which only wraps the speech() call
        with pytest.raises(Exception, match="Model not found"):
            await tts.synthesize("Hello")

    def test_get_sample_rates(self) -> None:
        """Test getting supported sample rates."""
        sample_rates = XinferenceTTS.get_sample_rates()

        assert isinstance(sample_rates, list)
        assert 22050 in sample_rates
        assert 24000 in sample_rates
        assert 48000 in sample_rates

    def test_get_supported_formats(self) -> None:
        """Test getting supported formats."""
        formats = XinferenceTTS.get_supported_formats()

        assert isinstance(formats, list)
        assert "mp3" in formats
        assert "wav" in formats
        assert "pcm" in formats
        assert "opus" in formats

    @patch("xinference_client.RESTfulClient")
    def test_list_available_models_success(self, mock_client_class: Mock) -> None:
        """Test listing available models successfully."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            "model-1": {
                "model_name": "chat-tts",
                "model_type": "audio",
                "model_ability": ["audio-generation"],
                "model_description": "Chat TTS model",
            },
            "model-2": {
                "model_name": "fish-speech",
                "model_type": "audio",
                "model_ability": "tts",
                "model_description": "Fish Speech model",
            },
            "model-3": {
                "model_name": "embedding-model",
                "model_type": "embedding",
                "model_ability": ["embedding"],
                "model_description": "Embedding model",
            },
        }
        mock_client_class.return_value = mock_client

        models = XinferenceTTS.list_available_models(
            base_url="http://localhost:9997", api_key="test-key"
        )

        # Verify
        assert len(models) == 2  # Only audio-generation/tts models
        assert models[0]["id"] == "chat-tts"
        assert models[0]["model_uid"] == "model-1"
        assert models[0]["model_ability"] == ["audio-generation"]
        assert models[1]["id"] == "fish-speech"
        assert models[1]["model_uid"] == "model-2"
        # The code converts string abilities to lists
        assert models[1]["model_ability"] == ["tts"]
        mock_client.close.assert_called_once()

    @patch("xinference_client.RESTfulClient")
    def test_list_available_models_empty(self, mock_client_class: Mock) -> None:
        """Test listing models when no TTS models available."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            "model-1": {
                "model_name": "embedding-model",
                "model_type": "embedding",
                "model_ability": ["embedding"],
            }
        }
        mock_client_class.return_value = mock_client

        models = XinferenceTTS.list_available_models(base_url="http://localhost:9997")

        # Verify
        assert len(models) == 0
        mock_client.close.assert_called_once()

    @patch("xinference_client.RESTfulClient")
    def test_list_available_models_error_handling(
        self, mock_client_class: Mock
    ) -> None:
        """Test error handling in list_available_models."""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_client.list_models.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        models = XinferenceTTS.list_available_models(base_url="http://localhost:9997")

        # Verify returns empty list on error
        assert models == []
        mock_client.close.assert_called_once()

    @patch("xinference_client.RESTfulClient")
    def test_list_available_models_with_speech_ability(
        self, mock_client_class: Mock
    ) -> None:
        """Test listing models with 'speech' ability."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            "model-1": {
                "model_name": "speech-model",
                "model_type": "audio",
                "model_ability": ["speech"],
            }
        }
        mock_client_class.return_value = mock_client

        models = XinferenceTTS.list_available_models(base_url="http://localhost:9997")

        # Verify speech ability is recognized
        assert len(models) == 1
        assert models[0]["id"] == "speech-model"

    @patch(_ASYNC_CLIENT_PATH)
    def test_context_manager(self, mock_client_class: Mock) -> None:
        """Test that context manager properly closes resources."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        # Test context manager
        with XinferenceTTS() as tts:
            assert tts._model_handle is None  # Not initialized yet

        # After exiting context, close should have been called if model was used
        # (In this case model was never used, so _model_handle is still None)

    @patch(_ASYNC_CLIENT_PATH)
    async def test_context_manager_with_model_used(
        self, mock_client_class: Mock
    ) -> None:
        """Test context manager closes resources when model is used."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_model_handle.speech = AsyncMock(return_value=b"fake audio data")
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        # Test context manager with model usage
        with XinferenceTTS() as tts:
            await tts.synthesize("Hello", verbose=False)
            assert tts._model_handle is not None

        # After exiting context, close should have been called
        mock_model_handle.close.assert_called_once()
        mock_client.close.assert_called_once()
        assert tts._model_handle is None
        assert tts._client is None

    def test_close(self) -> None:
        """Test close method."""
        # Create mock model handle and client
        mock_model_handle = MagicMock()
        mock_client = MagicMock()

        tts = XinferenceTTS()
        tts._model_handle = mock_model_handle
        tts._client = mock_client

        # Call close
        tts.close()

        # Verify close was called
        mock_model_handle.close.assert_called_once()
        mock_client.close.assert_called_once()

        # Verify cleanup
        assert tts._model_handle is None
        assert tts._client is None

    @patch(_ASYNC_CLIENT_PATH)
    def test_close_without_model_handle(self, mock_client_class: Mock) -> None:
        """Test close when model handle was never initialized."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        # Client is not initialized yet, so _model_handle and _client are None

        # Should not raise exception
        tts.close()
        assert tts._model_handle is None
        assert tts._client is None

    @patch(_ASYNC_CLIENT_PATH)
    def test_close_exception_handling(self, mock_client_class: Mock) -> None:
        """Test that close handles exceptions gracefully."""
        # Setup mock that raises exception
        mock_model_handle = MagicMock()
        mock_model_handle.close.side_effect = Exception("Close error")
        mock_client = MagicMock()

        tts = XinferenceTTS()
        tts._model_handle = mock_model_handle
        tts._client = mock_client

        # Should not raise exception
        tts.close()

        # Verify cleanup happened despite exception
        assert tts._model_handle is None
        assert tts._client is None

    @patch(_ASYNC_CLIENT_PATH)
    async def test_get_session_lazy_initialization(
        self, mock_client_class: Mock
    ) -> None:
        """Test that client is lazily initialized."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        assert tts._client is None

        # First call creates client
        client1 = await tts._get_session()
        assert client1 is not None
        mock_client_class.assert_called_once_with(
            base_url="http://localhost:9997", api_key=None
        )

        # Second call returns same client
        client2 = await tts._get_session()
        assert client1 is client2
        assert mock_client_class.call_count == 1

    @patch(_ASYNC_CLIENT_PATH)
    async def test_ensure_model_handle_lazy_initialization(
        self, mock_client_class: Mock
    ) -> None:
        """Test that model handle is lazily initialized."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS()
        assert tts._model_handle is None

        # First call creates model handle
        handle1 = await tts._ensure_model_handle()
        assert handle1 is not None
        mock_client.get_model.assert_called_once_with("chat-tts")

        # Second call returns same handle
        handle2 = await tts._ensure_model_handle()
        assert handle1 is handle2
        assert mock_client.get_model.call_count == 1

    @patch(_ASYNC_CLIENT_PATH)
    async def test_ensure_model_handle_with_custom_model_uid(
        self, mock_client_class: Mock
    ) -> None:
        """Test _ensure_model_handle with custom model UID."""
        # Setup mock
        mock_client = MagicMock()
        mock_model_handle = MagicMock()
        mock_client.get_model = AsyncMock(return_value=mock_model_handle)
        mock_client_class.return_value = mock_client

        tts = XinferenceTTS(model="chat-tts", model_uid="custom-uid-123")
        handle = await tts._ensure_model_handle()

        assert handle is not None
        mock_client.get_model.assert_called_once_with("custom-uid-123")


class TestTTSResult:
    """Test cases for TTSResult dataclass."""

    def test_tts_result_creation(self) -> None:
        """Test TTSResult creation."""
        result = TTSResult(
            audio=b"test audio data",
            format="mp3",
            sample_rate=24000,
            language="en",
        )

        assert result.audio == b"test audio data"
        assert result.format == "mp3"
        assert result.sample_rate == 24000
        assert result.language == "en"
        assert result.raw_response is None

    def test_tts_result_with_raw_response(self) -> None:
        """Test TTSResult with raw response."""
        raw_data = {"request_id": "test-id", "model": "chat-tts"}
        result = TTSResult(
            audio=b"audio",
            format="wav",
            sample_rate=48000,
            raw_response=raw_data,
        )

        assert result.raw_response == raw_data
        assert result.raw_response["model"] == "chat-tts"

    def test_tts_result_optional_fields(self) -> None:
        """Test TTSResult with optional fields."""
        result = TTSResult(
            audio=b"audio",
            format="mp3",
        )

        assert result.sample_rate is None
        assert result.language is None
        assert result.raw_response is None

    def test_tts_result_empty_raw_response(self) -> None:
        """Test TTSResult with empty raw response dict."""
        result = TTSResult(audio=b"audio", format="mp3", raw_response={})

        assert result.raw_response == {}

    def test_tts_result_all_fields(self) -> None:
        """Test TTSResult with all fields populated."""
        raw_data = {
            "request_id": "req-123",
            "model": "chat-tts",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        result = TTSResult(
            audio=b"full audio data",
            format="wav",
            sample_rate=48000,
            language="zh",
            raw_response=raw_data,
        )

        assert result.audio == b"full audio data"
        assert result.format == "wav"
        assert result.sample_rate == 48000
        assert result.language == "zh"
        assert result.raw_response == raw_data
        assert result.raw_response["request_id"] == "req-123"
