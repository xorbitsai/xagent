"""Unit tests for Xinference ASR model."""

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

from xagent.core.model.asr import ASRResult, ASRSegment, XinferenceASR


class TestXinferenceASR:
    """Test cases for Xinference ASR model."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        asr = XinferenceASR()
        assert asr.model == "whisper-base"
        assert asr.base_url == "http://localhost:9997"
        assert asr.api_key is None

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        asr = XinferenceASR(model="seaco-paraformer-zh")
        assert asr.model == "seaco-paraformer-zh"

    def test_init_with_model_uid(self) -> None:
        """Test initialization with model UID."""
        asr = XinferenceASR(model="whisper-base", model_uid="custom-whisper")
        assert asr._model_uid == "custom-whisper"

    def test_init_with_base_url(self) -> None:
        """Test initialization with custom base URL."""
        asr = XinferenceASR(base_url="http://localhost:8080")
        assert asr.base_url == "http://localhost:8080"

    def test_init_with_base_url_trailing_slash(self) -> None:
        """Test initialization removes trailing slash from base URL."""
        asr = XinferenceASR(base_url="http://localhost:9997/")
        assert asr.base_url == "http://localhost:9997"

    def test_init_with_language(self) -> None:
        """Test initialization with language."""
        asr = XinferenceASR(language="zh")
        assert asr.language == "zh"

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key."""
        asr = XinferenceASR(api_key="test-api-key")
        assert asr.api_key == "test-api-key"

    def test_abilities(self) -> None:
        """Test that abilities property returns expected list."""
        asr = XinferenceASR()
        abilities = asr.abilities
        assert isinstance(abilities, list)
        assert "asr" in abilities
        assert "speech_recognition" in abilities
        assert "audio" in abilities
        assert "timestamps" in abilities
        assert "speaker_diarization" in abilities
        assert "verbose_json" in abilities
        assert "hotword" in abilities

    def test_supports_speaker_diarization(self) -> None:
        """Test that supports_speaker_diarization property works."""
        asr = XinferenceASR()
        assert asr.supports_speaker_diarization is True

    def test_supports_timestamps(self) -> None:
        """Test that supports_timestamps property works."""
        asr = XinferenceASR()
        assert asr.supports_timestamps is True

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_simple_text_only(self, mock_client_class: Mock) -> None:
        """Test simple transcription returning text only (verbose=False)."""
        # Setup mock
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.transcriptions = AsyncMock(return_value={"text": "test text"})
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        # Use audio bytes to avoid file system operations
        audio_bytes = b"fake audio data"
        result = await asr.transcribe(audio_bytes, format="wav", verbose=False)

        # Verify
        assert isinstance(result, str)
        assert result == "test text"
        mock_model.transcriptions.assert_called_once()

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_verbose_with_segments(
        self, mock_client_class: Mock
    ) -> None:
        """Test verbose transcription returning ASRResult with segments."""
        # Setup mock
        mock_client = MagicMock()
        mock_model = MagicMock()
        segments_data = [
            {"id": 0, "start": 0.0, "end": 2.5, "text": "test text", "speaker": 0}
        ]
        mock_result = {"text": "test text", "segments": segments_data}
        mock_model.transcriptions = AsyncMock(return_value=mock_result)
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        audio_bytes = b"fake audio data"
        result = await asr.transcribe(audio_bytes, format="wav", verbose=True)

        # Verify
        assert isinstance(result, ASRResult)
        assert result.text == "test text"
        assert result.segments is not None
        assert len(result.segments) == 1
        assert result.segments[0].text == "test text"
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 2.5
        assert result.segments[0].speaker == "0"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_with_speaker_zero(self, mock_client_class: Mock) -> None:
        """Test that speaker ID 0 is correctly parsed (not treated as falsy)."""
        # Setup mock
        mock_client = MagicMock()
        mock_model = MagicMock()
        segments_data = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "speech", "speaker": 0}
        ]
        mock_result = {"text": "speech", "segments": segments_data}
        mock_model.transcriptions = AsyncMock(return_value=mock_result)
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        audio_bytes = b"fake audio data"
        result = await asr.transcribe(audio_bytes, format="wav", verbose=True)

        # Verify speaker ID is preserved
        assert result.segments[0].speaker == "0"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_with_multiple_speakers(
        self, mock_client_class: Mock
    ) -> None:
        """Test transcription with multiple speakers."""
        # Setup mock
        mock_client = MagicMock()
        mock_model = MagicMock()
        segments_data = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "hello", "speaker": 0},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "world", "speaker": 1},
        ]
        mock_result = {"text": "hello world", "segments": segments_data}
        mock_model.transcriptions = AsyncMock(return_value=mock_result)
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        audio_bytes = b"fake audio data"
        result = await asr.transcribe(audio_bytes, format="wav", verbose=True)

        # Verify
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "0"
        assert result.segments[1].speaker == "1"
        assert result.segments[0].text == "hello"
        assert result.segments[1].text == "world"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_with_hotword(self, mock_client_class: Mock) -> None:
        """Test transcription with hotword parameter."""
        # Setup mock
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.transcriptions = AsyncMock(return_value={"text": "test"})
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        audio_bytes = b"fake audio data"
        result = await asr.transcribe(
            audio_bytes, format="wav", hotword="test word", verbose=False
        )

        # Verify hotword was passed to the API
        mock_model.transcriptions.assert_called_once()
        call_kwargs = mock_model.transcriptions.call_args.kwargs
        assert "hotword" in call_kwargs
        assert call_kwargs["hotword"] == "test word"
        assert result == "test"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_with_bytes(self, mock_client_class: Mock) -> None:
        """Test transcription with audio bytes."""
        # Setup mock
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.transcriptions = AsyncMock(return_value={"text": "test"})
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        audio_bytes = b"fake audio data"
        result = await asr.transcribe(audio_bytes, format="wav", verbose=False)

        # Verify
        assert result == "test"
        mock_model.transcriptions.assert_called_once()

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_with_language_parameter(
        self, mock_client_class: Mock
    ) -> None:
        """Test transcription with language parameter."""
        # Setup mock
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.transcriptions = AsyncMock(return_value={"text": "test"})
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        audio_bytes = b"fake audio data"
        await asr.transcribe(audio_bytes, format="wav", language="zh", verbose=False)

        # Verify language was passed
        call_kwargs = mock_model.transcriptions.call_args.kwargs
        assert "language" in call_kwargs
        assert call_kwargs["language"] == "zh"

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_response_format_verbose_json(
        self, mock_client_class: Mock
    ) -> None:
        """Test that verbose=True sets response_format to verbose_json."""
        # Setup mock
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.transcriptions = AsyncMock(return_value={"text": "test"})
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        audio_bytes = b"fake audio data"
        await asr.transcribe(audio_bytes, format="wav", verbose=True)

        # Verify response_format was set
        call_kwargs = mock_model.transcriptions.call_args.kwargs
        assert call_kwargs.get("response_format") == "verbose_json"

    def test_parse_segments_with_dict(self) -> None:
        """Test _parse_segments with dict input."""
        asr = XinferenceASR()
        segments_data = [
            {
                "text": "hello world",
                "start": 0.0,
                "end": 2.5,
                "speaker": "spk0",
                "confidence": 0.95,
            },
            {"text": "test completed", "start": 2.5, "end": 4.0, "speaker": "spk1"},
        ]

        segments = asr._parse_segments(segments_data)
        assert len(segments) == 2
        assert segments[0].text == "hello world"
        assert segments[0].start == 0.0
        assert segments[0].end == 2.5
        assert segments[0].speaker == "spk0"
        assert segments[0].confidence == 0.95
        assert segments[1].speaker == "spk1"

    def test_parse_segments_preserves_speaker_zero(self) -> None:
        """Test that speaker ID 0 is preserved (not treated as falsy)."""
        asr = XinferenceASR()
        segments_data = [
            {
                "text": "test",
                "start": 0.0,
                "end": 1.0,
                "speaker": 0,  # This should be preserved
            }
        ]

        segments = asr._parse_segments(segments_data)
        assert len(segments) == 1
        assert segments[0].speaker == "0"  # Should be string "0", not None

    def test_parse_segments_with_object(self) -> None:
        """Test _parse_segments with object input."""
        asr = XinferenceASR()

        # Create a mock segment object
        class MockSegment:
            def __init__(self, text, start, end, speaker=None):
                self.text = text
                self.start = start
                self.end = end
                self.speaker = speaker

        segments_data = [MockSegment("Test text", 1.0, 3.0, "spk0")]
        segments = asr._parse_segments(segments_data)

        assert len(segments) == 1
        assert segments[0].text == "Test text"
        assert segments[0].start == 1.0
        assert segments[0].end == 3.0
        assert segments[0].speaker == "spk0"

    def test_parse_segments_empty_list(self) -> None:
        """Test _parse_segments with empty list."""
        asr = XinferenceASR()
        segments = asr._parse_segments([])
        assert len(segments) == 0

    def test_parse_segments_without_text_field(self) -> None:
        """Test _parse_segments ignores segments without text field."""
        asr = XinferenceASR()
        segments_data = [
            {
                "start": 0.0,
                "end": 1.0,
                # No "text" field - should be ignored
            }
        ]

        segments = asr._parse_segments(segments_data)
        assert len(segments) == 0

    @patch(_ASYNC_CLIENT_PATH)
    def test_context_manager(self, mock_client_class: Mock) -> None:
        """Test context manager functionality."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        with asr:
            assert asr is not None

        # After exiting, resources should be cleaned up
        assert asr._client is None
        assert asr._model_handle is None

    @patch(_ASYNC_CLIENT_PATH)
    async def test_close(self, mock_client_class: Mock) -> None:
        """Test close method cleans up resources."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        # Create a client by accessing _get_session
        await asr._get_session()
        assert asr._client is not None

        # Close should clean up resources
        asr.close()
        assert asr._client is None
        assert asr._model_handle is None

    @patch(_ASYNC_CLIENT_PATH)
    async def test_transcribe_error_handling(self, mock_client_class: Mock) -> None:
        """Test error handling when transcription fails."""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.transcriptions = AsyncMock(side_effect=Exception("API Error"))
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        audio_bytes = b"fake audio data"

        with pytest.raises(RuntimeError, match="Xinference ASR failed"):
            await asr.transcribe(audio_bytes, format="wav")

    @patch(_ASYNC_CLIENT_PATH)
    async def test_get_session_lazy_initialization(
        self, mock_client_class: Mock
    ) -> None:
        """Test that client is lazily initialized."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        assert asr._client is None

        # Access _get_session should create client
        client = await asr._get_session()
        assert asr._client is not None
        assert client is not None

    @patch(_ASYNC_CLIENT_PATH)
    async def test_ensure_model_handle(self, mock_client_class: Mock) -> None:
        """Test that model handle is created and cached."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_client.get_model = AsyncMock(return_value=mock_model)
        mock_client_class.return_value = mock_client

        asr = XinferenceASR()
        assert asr._model_handle is None

        # First call should create model handle
        handle1 = await asr._ensure_model_handle()
        assert handle1 is not None
        assert asr._model_handle is not None

        # Second call should return cached handle
        handle2 = await asr._ensure_model_handle()
        assert handle1 is handle2

        # Should only call get_model once
        assert mock_client.get_model.call_count == 1


class TestASRResult:
    """Test cases for ASRResult dataclass."""

    def test_asr_result_creation(self) -> None:
        """Test ASRResult creation."""
        result = ASRResult(text="test text", segments=None, language="zh")

        assert result.text == "test text"
        assert result.segments is None
        assert result.language == "zh"
        assert result.raw_response is None

    def test_asr_result_str_representation(self) -> None:
        """Test ASRResult __str__ returns text."""
        result = ASRResult(text="test text")
        assert str(result) == "test text"

    def test_asr_result_with_segments(self) -> None:
        """Test ASRResult with segments."""
        segments = [
            ASRSegment(text="hello", start=0.0, end=1.0, speaker="0"),
            ASRSegment(text="world", start=1.0, end=2.0, speaker="1"),
        ]

        result = ASRResult(text="hello world", segments=segments, language="en")

        assert len(result.segments) == 2
        assert result.segments[0].text == "hello"
        assert result.segments[1].speaker == "1"


class TestASRSegment:
    """Test cases for ASRSegment dataclass."""

    def test_segment_creation(self) -> None:
        """Test ASRSegment creation."""
        segment = ASRSegment(
            text="test", start=0.0, end=2.5, speaker="spk0", confidence=0.95
        )

        assert segment.text == "test"
        assert segment.start == 0.0
        assert segment.end == 2.5
        assert segment.speaker == "spk0"
        assert segment.confidence == 0.95

    def test_segment_with_speaker_str(self) -> None:
        """Test ASRSegment __str__ with speaker."""
        segment = ASRSegment(text="hello", start=0.0, end=1.0, speaker="0")

        assert str(segment) == "[0] hello"

    def test_segment_without_speaker_str(self) -> None:
        """Test ASRSegment __str__ without speaker."""
        segment = ASRSegment(text="test", start=0.0, end=1.0)

        assert str(segment) == "test"

    def test_segment_optional_fields(self) -> None:
        """Test ASRSegment with optional fields."""
        segment = ASRSegment(text="test", start=0.0, end=1.0)

        assert segment.speaker is None
        assert segment.confidence is None

    def test_segment_speaker_zero_preserved(self) -> None:
        """Test that speaker ID 0 is preserved in string representation."""
        segment = ASRSegment(text="speech", start=0.0, end=1.0, speaker="0")

        # Should show [0] not [] (preserving speaker 0)
        assert "[0]" in str(segment)
