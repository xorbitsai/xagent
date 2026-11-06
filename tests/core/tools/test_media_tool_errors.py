"""What a media tool tells the model when a provider fails.

The production failure behind these tests: Ark answered a video request with

    Error code: 404 - {'error': {'code': 'ModelNotOpen', 'message': 'Your
    account 2100...4339 has not activated the model doubao-seedance-2-0-fast-
    260128. Please activate the model service in the Ark Console. Request id:
    0217...f8', ...}}, request_id: 2026...zEj

the tool passed that string through as ``error``, and the model repeated it to
the user -- account id, request id, provider model id and console name
included. Every media tool now reports a category and one plain sentence
instead, and keeps the provider's words in the log with the ids removed.

The exception shapes below mirror what the provider SDKs actually raise: an
OpenAI-style ``APIStatusError`` (``status_code``, ``body['error']['code']``,
``request_id``), which is also what the Ark SDK raises.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.model.asr.base import ASRResult, BaseASR
from xagent.core.model.asr.elevenlabs import ElevenLabsASR
from xagent.core.model.asr.xinference import XinferenceASR
from xagent.core.model.image import dashscope as dashscope_image
from xagent.core.model.image.dashscope import DashScopeImageModel
from xagent.core.model.image.xinference import XinferenceImageModel
from xagent.core.model.music.elevenlabs import ElevenLabsMusicModel
from xagent.core.model.sound_effect.elevenlabs import ElevenLabsSoundEffectModel
from xagent.core.model.tts.base import BaseTTS, TTSResult
from xagent.core.model.tts.elevenlabs import ElevenLabsTTS
from xagent.core.model.tts.xinference import XinferenceTTS
from xagent.core.model.video.ark import ArkVideoModel
from xagent.core.model.video.base import BilledVideoTaskTimeout
from xagent.core.tools.core.audio_tool import AudioToolCore
from xagent.core.tools.core.image_tool import ImageGenerationToolCore
from xagent.core.tools.core.media_errors import (
    INVALID_REQUEST,
    MODEL_NOT_ENABLED,
    PROVIDER_UNAVAILABLE,
    UNKNOWN,
    MediaParameterError,
    classify_media_failure,
    redact_provider_detail,
)
from xagent.core.tools.core.music_tool import MusicToolCore
from xagent.core.tools.core.sound_effect_tool import SoundEffectToolCore
from xagent.core.tools.core.video_tool import VideoGenerationToolCore

ACCOUNT_ID = "2100123456784339"
PROVIDER_REQUEST_ID = "021760000000000f8"
SDK_REQUEST_ID = "20261106050247zEj8Qk"
PROVIDER_MODEL = "doubao-seedance-2-0-fast-260128"


class ProviderStatusError(Exception):
    """An OpenAI-style SDK error: status, body and request id as attributes."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        request_id: str = SDK_REQUEST_ID,
    ) -> None:
        self.status_code = status_code
        self.request_id = request_id
        self.body = {
            "error": {
                "code": code,
                "message": message,
                "param": "",
                "type": "Not Found",
            }
        }
        super().__init__(
            f"Error code: {status_code} - {json.dumps(self.body)}, "
            f"request_id: {request_id}"
        )


def model_not_open_error() -> ProviderStatusError:
    return ProviderStatusError(
        status_code=404,
        code="ModelNotOpen",
        message=(
            f"Your account {ACCOUNT_ID} has not activated the model "
            f"{PROVIDER_MODEL}. Please activate the model service in the Ark "
            f"Console. Request id: {PROVIDER_REQUEST_ID}"
        ),
    )


def invalid_parameter_error() -> ProviderStatusError:
    return ProviderStatusError(
        status_code=400,
        code="InvalidParameter",
        message="The parameter `duration` specified in the request is not valid.",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Mock:
    workspace = Mock()
    workspace.workspace_dir = tmp_path / "workspace"
    workspace.output_dir = workspace.workspace_dir / "output"
    workspace.output_dir.mkdir(parents=True)
    workspace.get_file_id_from_path = Mock(return_value="file-id")
    return workspace


def _ark_video_tool(
    workspace: Mock, error: Optional[Exception] = None
) -> tuple[VideoGenerationToolCore, ArkVideoModel]:
    ark_model = ArkVideoModel(api_key="test-key")
    if error is None:
        ark_model.generate_video = AsyncMock(
            return_value={"task_id": "task-1", "status": "succeeded", "video_url": ""}
        )
    else:
        ark_model.generate_video = AsyncMock(side_effect=error)
    tool = VideoGenerationToolCore(
        {"ark-video": ark_model},
        workspace=workspace,
        default_video_model=ark_model,
    )
    return tool, ark_model


@pytest.mark.asyncio
async def test_model_not_open_reports_a_category_and_no_provider_detail(
    workspace: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    tool, _ = _ark_video_tool(workspace, model_not_open_error())

    with caplog.at_level(logging.ERROR):
        result = await tool.generate_video("A short clip", seconds=4)

    assert result["success"] is False
    assert result["error_category"] == MODEL_NOT_ENABLED
    # The result is what the model reads and can repeat, so nothing the
    # provider said about its own account, request or model may be in it.
    readable = json.dumps(result, default=str)
    for leaked in (
        ACCOUNT_ID,
        PROVIDER_REQUEST_ID,
        SDK_REQUEST_ID,
        PROVIDER_MODEL,
        "ModelNotOpen",
        "Ark Console",
        "ProviderStatusError",
    ):
        assert leaked not in readable


@pytest.mark.asyncio
async def test_model_not_open_leaves_a_searchable_marker_in_the_log(
    workspace: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    tool, _ = _ark_video_tool(workspace, model_not_open_error())

    with caplog.at_level(logging.ERROR):
        await tool.generate_video("A short clip", seconds=4)

    assert (
        f"media model not enabled: provider=volcengine-ark model={PROVIDER_MODEL}"
        in caplog.text
    )
    # The operator keeps the provider's own words, but not the account or the
    # request id: a log search must not be able to turn either one up.
    assert "has not activated the model" in caplog.text
    assert ACCOUNT_ID not in caplog.text
    assert PROVIDER_REQUEST_ID not in caplog.text
    assert SDK_REQUEST_ID not in caplog.text


@pytest.mark.asyncio
async def test_a_transient_provider_failure_is_marked_retryable(
    workspace: Mock,
) -> None:
    image_model = Mock()
    image_model.has_ability = Mock(return_value=True)
    image_model.abilities = ["generate"]
    image_model.generate_image = AsyncMock(
        side_effect=ProviderStatusError(
            status_code=503, code="ServiceUnavailable", message="upstream is busy"
        )
    )
    tool = ImageGenerationToolCore(
        {"image-model": image_model},
        workspace=workspace,
        default_generate_model=image_model,
    )

    result = await tool.generate_image("A product shot")

    assert result["success"] is False
    assert result["error_category"] == PROVIDER_UNAVAILABLE
    assert "upstream is busy" not in result["error"]


@pytest.mark.asyncio
async def test_a_successful_video_carries_no_provider_response(workspace: Mock) -> None:
    tool, ark_model = _ark_video_tool(workspace)
    ark_model.generate_video = AsyncMock(
        return_value={
            "task_id": "task-1",
            "status": "succeeded",
            "video_url": "",
            "duration": 4,
            "raw_response": {
                "id": "task-1",
                "usage": {"completion_tokens": 108900},
                "error": {"message": f"account {ACCOUNT_ID}"},
            },
        }
    )

    result = await tool.generate_video("A short clip", seconds=4)

    assert result["success"] is True
    assert "raw_response" not in result
    assert ACCOUNT_ID not in json.dumps(result, default=str)


class FailingTTS(BaseTTS):
    provider_name = "fake-tts"

    def __init__(self, error: Exception) -> None:
        self.model_name = "fake-tts"
        self._error = error

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[bytes, TTSResult]:
        raise self._error

    @property
    def abilities(self) -> list[str]:
        return ["tts", "text_to_speech"]


@pytest.mark.asyncio
async def test_a_rejected_parameter_reports_the_values_the_caller_can_change(
    workspace: Mock,
) -> None:
    tool, _ = _ark_video_tool(workspace, invalid_parameter_error())

    result = await tool.generate_video("A short clip", seconds=6)

    assert result["success"] is False
    assert result["error_category"] == INVALID_REQUEST
    assert result["allowed_values"] == {"duration_seconds": "4-15"}
    assert "InvalidParameter" not in json.dumps(result, default=str)


@pytest.mark.asyncio
async def test_a_rejected_speech_request_lists_the_supported_formats(
    workspace: Mock,
) -> None:
    tts_model = FailingTTS(invalid_parameter_error())
    tool = AudioToolCore(
        tts_models={"tts": tts_model},
        default_tts_model=tts_model,
        workspace=workspace,
    )

    result = await tool.synthesize_speech("Hello there", audio_format="opus")

    assert result["success"] is False
    assert result["error_category"] == INVALID_REQUEST
    assert result["allowed_values"] == {"audio_format": ["mp3", "wav", "pcm"]}


class FailingASR(BaseASR):
    def __init__(self, error: Exception) -> None:
        self.model_name = "fake-asr"
        self._error = error

    async def transcribe(
        self,
        audio: Union[str, bytes],
        language: Optional[str] = None,
        format: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Union[str, ASRResult]:
        raise self._error

    @property
    def abilities(self) -> list[str]:
        return ["asr"]


@pytest.mark.asyncio
async def test_a_failed_transcription_keeps_the_provider_text_out(
    workspace: Mock,
) -> None:
    audio_path = workspace.output_dir / "voice.mp3"
    audio_path.write_bytes(b"audio")
    asr_model = FailingASR(model_not_open_error())
    tool = AudioToolCore(
        asr_models={"asr": asr_model},
        default_asr_model=asr_model,
        workspace=workspace,
    )

    result = await tool.transcribe_audio(str(audio_path))

    assert result["success"] is False
    assert result["error_category"] == MODEL_NOT_ENABLED
    assert ACCOUNT_ID not in json.dumps(result, default=str)


class TestClassification:
    """The category is what an alert fires on, so pin the four it can say."""

    def test_a_missing_credential_is_not_reported_as_a_retryable_outage(self) -> None:
        """401/402/403 need a human; calling them "unavailable" would promise
        a retry that cannot work."""
        error = ProviderStatusError(
            status_code=401, code="AuthenticationError", message="invalid api key"
        )

        assert classify_media_failure(error) == UNKNOWN

    def test_a_timeout_is_an_outage(self) -> None:
        assert classify_media_failure(TimeoutError("read timed out")) == (
            PROVIDER_UNAVAILABLE
        )

    def test_a_wrapped_provider_error_keeps_its_category(self) -> None:
        """Adapters that re-raise with their own prefix still leave the SDK
        exception in the chain."""
        try:
            try:
                raise model_not_open_error()
            except Exception as inner:
                raise RuntimeError(f"Image generation failed: {inner}") from inner
        except RuntimeError as wrapped:
            assert classify_media_failure(wrapped) == MODEL_NOT_ENABLED

    def test_redaction_keeps_the_message_and_drops_the_identifiers(self) -> None:
        detail = redact_provider_detail(
            f"Your account {ACCOUNT_ID} has not activated {PROVIDER_MODEL}. "
            f"Request id: {PROVIDER_REQUEST_ID}, "
            "key sk-live-0123456789abcdef0123"
        )

        assert "has not activated" in detail
        assert PROVIDER_MODEL in detail
        assert ACCOUNT_ID not in detail
        assert PROVIDER_REQUEST_ID not in detail
        assert "sk-live" not in detail


class FailingVoiceTTS(BaseTTS):
    """A voice-capable TTS client whose provider calls all fail."""

    provider_name = "elevenlabs"
    supports_voice_listing = True
    supports_persistent_voice_cloning = True

    def __init__(self, error: Exception) -> None:
        self.model_name = "eleven_v3"
        self._error = error

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[bytes, TTSResult]:
        raise self._error

    async def list_available_voices(self) -> list[dict[str, Any]]:
        raise self._error

    async def clone_voice(self, **kwargs: Any) -> dict[str, Any]:
        raise self._error

    async def delete_voice(self, voice_id: str) -> None:
        raise self._error

    @property
    def abilities(self) -> list[str]:
        return ["tts", "voice_listing", "voice_cloning"]


def _voice_tool(error: Exception) -> tuple[AudioToolCore, FailingVoiceTTS]:
    model = FailingVoiceTTS(error)
    tool = AudioToolCore(
        tts_models={"elevenlabs-tts": model},
        default_tts_model=model,
    )
    return tool, model


class TestAdapterLogs:
    """The adapters log before a tool can redact anything, so they share the
    redactor. These tests read the whole logger hierarchy, not one logger."""

    @pytest.mark.asyncio
    async def test_a_polling_failure_writes_no_account_or_request_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = ArkVideoModel(api_key="test-key")
        model.create_video_task = AsyncMock(return_value={"task_id": "cgt-1"})
        model.get_video_task = AsyncMock(side_effect=model_not_open_error())

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(BilledVideoTaskTimeout):
                # A deadline that has already passed ends the loop after the
                # first failed poll: the warning is written either way.
                await model.generate_video(prompt="A clip", timeout=-1)

        assert "Transient error polling video task" in caplog.text
        assert ACCOUNT_ID not in caplog.text
        assert PROVIDER_REQUEST_ID not in caplog.text
        assert SDK_REQUEST_ID not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("adapter_line", "call"),
        [
            (
                "ElevenLabs voice cloning failed",
                lambda tool: tool.clone_tts_voice("My voice", ["reference.wav"]),
            ),
            (
                "ElevenLabs voice deletion failed",
                lambda tool: tool.delete_tts_voice("voice-1"),
            ),
        ],
        ids=["clone", "delete"],
    )
    async def test_a_failed_voice_request_writes_no_account_or_request_id(
        self,
        caplog: pytest.LogCaptureFixture,
        adapter_line: str,
        call: Callable[[AudioToolCore], Any],
    ) -> None:
        """Cloning and deleting go through the SDK and log before the tool
        sees the exception, so these two lines are the whole hierarchy."""
        model = ElevenLabsTTS(api_key="test-key")
        sdk = Mock()
        sdk.voices.ivc.create = AsyncMock(side_effect=model_not_open_error())
        sdk.voices.delete = AsyncMock(side_effect=model_not_open_error())
        model._async_client = sdk
        model._reference_audio_file = Mock(
            return_value=("reference.wav", b"audio", "audio/wav")
        )
        tool = AudioToolCore(tts_models={"elevenlabs": model}, default_tts_model=model)

        with caplog.at_level(logging.DEBUG):
            result = await call(tool)

        assert result["success"] is False
        assert adapter_line in caplog.text
        assert ACCOUNT_ID not in caplog.text
        assert PROVIDER_REQUEST_ID not in caplog.text
        assert SDK_REQUEST_ID not in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_class", "section", "method", "call", "adapter_line"),
        [
            (
                ElevenLabsTTS,
                "text_to_speech",
                "convert",
                lambda model: model.synthesize("Hello there"),
                "ElevenLabs TTS failed",
            ),
            (
                ElevenLabsMusicModel,
                "music",
                "compose",
                lambda model: model.generate_music("A slow piano theme"),
                "ElevenLabs music generation failed",
            ),
            (
                ElevenLabsSoundEffectModel,
                "text_to_sound_effects",
                "convert",
                lambda model: model.generate_sound_effect("A door slams"),
                "ElevenLabs sound effect generation failed",
            ),
            (
                ElevenLabsASR,
                "speech_to_text",
                "convert",
                lambda model: model.transcribe(b"audio"),
                "ElevenLabs ASR failed",
            ),
        ],
        ids=["speech", "music", "sound-effect", "transcription"],
    )
    async def test_a_failed_generation_writes_no_account_or_request_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        model_class: Any,
        section: str,
        method: str,
        call: Callable[[Any], Any],
        adapter_line: str,
    ) -> None:
        client = SimpleNamespace(
            **{
                section: SimpleNamespace(
                    **{method: AsyncMock(side_effect=model_not_open_error())}
                )
            }
        )
        monkeypatch.setattr(
            model_class,
            "_create_async_client",
            staticmethod(lambda *args, **kwargs: client),
        )

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(RuntimeError):
                await call(model_class(api_key="test-key"))

        assert adapter_line in caplog.text
        assert ACCOUNT_ID not in caplog.text
        assert PROVIDER_REQUEST_ID not in caplog.text
        assert SDK_REQUEST_ID not in caplog.text

    @pytest.mark.parametrize(
        ("client_path", "list_models"),
        [
            (
                "xagent.core.model.image.xinference.XinferenceClient",
                XinferenceImageModel.list_available_models,
            ),
            (
                "xagent.core.model.public_xinference.PublicXinferenceClient",
                XinferenceASR.list_available_models,
            ),
            (
                "xagent.core.model.public_xinference.PublicXinferenceClient",
                XinferenceTTS.list_available_models,
            ),
        ],
        ids=["image", "asr", "tts"],
    )
    def test_a_failed_model_list_writes_no_account_or_request_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        client_path: str,
        list_models: Callable[[str], Any],
    ) -> None:
        client = Mock()
        client.list_models.side_effect = model_not_open_error()
        monkeypatch.setattr(client_path, lambda *args, **kwargs: client)

        with caplog.at_level(logging.DEBUG):
            assert list_models("http://localhost:9997") == []

        assert "Failed to fetch" in caplog.text
        assert ACCOUNT_ID not in caplog.text
        assert PROVIDER_REQUEST_ID not in caplog.text
        assert SDK_REQUEST_ID not in caplog.text


class TestStructuredRedaction:
    """Ids in the shapes providers actually print: JSON, repr and plain text."""

    @pytest.mark.parametrize(
        ("detail", "hidden"),
        [
            ("request_id=abc", "abc"),
            ("account_id=xyz", "xyz"),
            ("request_id=req_short_id", "req_short_id"),
            ("request_id: req_short_id", "req_short_id"),
            ("{'request_id': 'req_short_id'}", "req_short_id"),
            ('{"request_id": "req_short_id"}', "req_short_id"),
            ('{"error": {"api_key": "review_key_alpha"}}', "review_key_alpha"),
            ("api_key=review_key_alpha", "review_key_alpha"),
            ('{"voice_id": "21m00Tcm4TlvDq8ikWAM"}', "21m00Tcm4TlvDq8ikWAM"),
        ],
    )
    def test_a_labelled_identifier_is_masked_in_its_own_shape(
        self, detail: str, hidden: str
    ) -> None:
        assert hidden not in redact_provider_detail(detail)

    def test_masking_a_repr_keeps_its_delimiters(self) -> None:
        """The value goes, the punctuation around it stays: consuming the
        quote and the colon instead left the id itself in the log."""
        detail = redact_provider_detail(
            "{'request_id': 'req_short_id', 'account_id': 'acct_abc'}"
        )

        assert detail == "{'request_id': <redacted>, 'account_id': <redacted>}"

    @pytest.mark.parametrize(
        ("detail", "hidden"),
        [
            ("Request id req_short_id", "req_short_id"),
            ("Your account acct_abc has not activated it", "acct_abc"),
            ('{"Authorization": "Bearer abc"}', "Bearer abc"),
            ('{"request_ids": ["req_a", "req_b"]}', "req_a"),
            ('{"request_id": "req_\\"_after", "account_id": "acct_b"}', "_after"),
        ],
        ids=[
            "no-separator",
            "account-prose",
            "short-auth",
            "id-array",
            "escaped-quote",
        ],
    )
    def test_a_provider_shape_without_a_plain_field_is_masked(
        self, detail: str, hidden: str
    ) -> None:
        """A label written as prose, a value that is a list, and a quote
        escaped inside one: none of them are a ``name=value`` pair."""
        assert hidden not in redact_provider_detail(detail)

    def test_prose_about_an_account_keeps_its_words(self) -> None:
        """Only a value that looks like an id is masked: the sentence itself
        stays readable in the log."""
        detail = "your account is not activated yet"

        assert redact_provider_detail(detail) == detail

    def test_an_identifier_inside_a_json_message_is_masked(self) -> None:
        """SDKs nest the real error in `"message": "..."`; scanning only the
        outer field skipped everything inside it."""
        detail = redact_provider_detail(
            '{"error": {"code": "ModelNotOpen", "message": "Your account '
            f"{ACCOUNT_ID} has not activated it. Request id: "
            f'{PROVIDER_REQUEST_ID}"}}}}'
        )

        assert "has not activated" in detail
        assert ACCOUNT_ID not in detail
        assert PROVIDER_REQUEST_ID not in detail


class TestNotFoundClassification:
    """A 404 is only a missing model when the provider says it is."""

    def test_a_voice_that_does_not_exist_is_the_callers_to_fix(self) -> None:
        error = ProviderStatusError(
            status_code=404, code="voice_not_found", message="voice not found"
        )

        assert classify_media_failure(error) == INVALID_REQUEST

    def test_a_missing_resource_is_not_a_missing_model(self) -> None:
        error = ProviderStatusError(
            status_code=404,
            code="resource_not_found",
            message="the requested resource was not found",
        )

        assert classify_media_failure(error) == UNKNOWN

    def test_a_404_that_names_the_model_as_its_target_is_not_enabled(self) -> None:
        error = ProviderStatusError(
            status_code=404,
            code="invalid_request_error",
            message="no endpoint serves this request",
        )
        error.body["error"]["param"] = "model"

        assert classify_media_failure(error) == MODEL_NOT_ENABLED

    @pytest.mark.asyncio
    async def test_listing_voices_through_a_404_writes_no_not_enabled_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool, _ = _voice_tool(
            ProviderStatusError(
                status_code=404, code="voice_not_found", message="voice not found"
            )
        )

        with caplog.at_level(logging.ERROR):
            result = await tool.list_tts_voices()

        assert result["error_category"] == INVALID_REQUEST
        assert "media model not enabled" not in caplog.text
        assert ACCOUNT_ID not in json.dumps(result, default=str)


class TestParameterErrorsFromTheAdapters:
    """An adapter that validated a parameter says which one and its range."""

    @pytest.mark.asyncio
    async def test_a_rejected_speech_format_lists_the_provider_formats(
        self,
    ) -> None:
        tts_model = ElevenLabsTTS(api_key="test-key")
        tool = AudioToolCore(
            tts_models={"elevenlabs": tts_model},
            default_tts_model=tts_model,
        )

        result = await tool.synthesize_speech("Hello", audio_format="not-a-format")

        assert result["success"] is False
        assert result["error_category"] == INVALID_REQUEST
        formats = result["allowed_values"]["audio_format"]
        # The adapter knows ElevenLabs also renders opus/ulaw, which the
        # tool's own documented list does not mention.
        assert "opus" in formats
        assert "mp3" in formats
        assert "not-a-format" not in json.dumps(result, default=str)

    @pytest.mark.asyncio
    async def test_a_music_length_outside_the_range_reports_the_range(self) -> None:
        model = ElevenLabsMusicModel(api_key="test-key")
        tool = MusicToolCore(models={"elevenlabs-music": model}, default_model=model)

        result = await tool.generate_music("A slow piano theme", music_length_seconds=2)

        assert result["success"] is False
        assert result["error_category"] == INVALID_REQUEST
        assert result["allowed_values"] == {"music_length_seconds": "3-600"}
        # The plain sentence is all the model reads: the provider's own text,
        # which names the field and the range in its words, stays in the log.
        assert result["error"] == (
            "The request was rejected because a parameter is outside what this "
            "model supports."
        )

    @pytest.mark.asyncio
    async def test_an_overlong_music_prompt_reports_the_limit(self) -> None:
        model = ElevenLabsMusicModel(api_key="test-key")
        tool = MusicToolCore(models={"elevenlabs-music": model}, default_model=model)

        result = await tool.generate_music("x" * 4101)

        assert result["success"] is False
        assert result["error_category"] == INVALID_REQUEST
        assert result["allowed_values"] == {"prompt_max_characters": 4100}

    @pytest.mark.asyncio
    async def test_a_sound_effect_length_outside_the_range_reports_the_range(
        self,
    ) -> None:
        model = ElevenLabsSoundEffectModel(api_key="test-key")
        tool = SoundEffectToolCore(
            models={"elevenlabs-sound": model}, default_model=model
        )

        result = await tool.generate_sound_effect("A door slams", duration_seconds=0.4)

        assert result["success"] is False
        assert result["error_category"] == INVALID_REQUEST
        assert result["allowed_values"] == {"duration_seconds": "0.5-30"}

    @pytest.mark.asyncio
    async def test_a_parameter_with_no_known_range_carries_no_allowed_values(
        self,
    ) -> None:
        """No reliable range means no range: the failure says the request was
        rejected and stops there instead of inventing values."""
        model = ElevenLabsMusicModel(api_key="test-key")
        tool = MusicToolCore(models={"elevenlabs-music": model}, default_model=model)

        result = await tool.generate_music("   ")

        assert result["success"] is False
        assert result["error_category"] == INVALID_REQUEST
        assert "allowed_values" not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_kwargs",
        [
            # The adapter rejects both of these itself, and neither has a
            # range it could offer: one is a rate it cannot resolve, the other
            # a rate the format does not render at.
            {"audio_format": "pcm", "sample_rate": 12345},
            {"audio_format": "mp3", "sample_rate": 22050},
            {"voice": "not a voice id"},
        ],
        ids=["pcm-rate", "mp3-rate", "voice"],
    )
    async def test_a_rejected_parameter_with_no_known_range_names_no_range(
        self, request_kwargs: dict[str, Any]
    ) -> None:
        """The adapter knows the field was rejected but not which values the
        provider accepts -- so the result says "rejected" and stops, instead
        of offering the audio formats as if they were the problem."""
        tts_model = ElevenLabsTTS(api_key="test-key")
        tts_model._ensure_async_client = Mock(
            side_effect=AssertionError("a rejected request must not reach the SDK")
        )
        tool = AudioToolCore(
            tts_models={"elevenlabs": tts_model},
            default_tts_model=tts_model,
        )

        result = await tool.synthesize_speech("Hello", **request_kwargs)

        assert result["success"] is False
        assert result["error_category"] == INVALID_REQUEST
        assert "allowed_values" not in result

    def test_a_typed_parameter_error_wins_over_an_ambiguous_status(self) -> None:
        """The adapter knows more than the status does, in both directions."""
        error = MediaParameterError("bad format", field="audio_format")

        assert classify_media_failure(error) == INVALID_REQUEST


class _FakeHTTPResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def json(self) -> Any:
        raise AssertionError("a non-200 body is never decoded")


class _FakePost:
    def __init__(self, response: _FakeHTTPResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeHTTPResponse:
        return self._response

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


class _FakeSession:
    def __init__(self, response: _FakeHTTPResponse) -> None:
        self._response = response

    def post(self, *args: Any, **kwargs: Any) -> _FakePost:
        return _FakePost(self._response)

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


class TestDashScopeStatusCodes:
    """The DashScope adapter keeps the HTTP status instead of spelling it
    into the text, so a 429 or 503 stays a retryable outage."""

    @pytest.mark.parametrize("status", [429, 503])
    @pytest.mark.asyncio
    async def test_a_retryable_status_reaches_the_result(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        monkeypatch.setattr(
            dashscope_image,
            "create_public_model_aiohttp_session",
            lambda **kwargs: _FakeSession(
                _FakeHTTPResponse(status, "upstream temporarily busy")
            ),
        )
        model = DashScopeImageModel(model_name="qwen-image", api_key="test-key")
        tool = ImageGenerationToolCore(
            {"dashscope": model},
            default_generate_model=model,
        )

        result = await tool.generate_image("A product shot")

        assert result["success"] is False
        assert result["error_category"] == PROVIDER_UNAVAILABLE
        assert "upstream temporarily busy" not in json.dumps(result, default=str)


class TestVoiceTools:
    """The voice helpers leaked the provider's text like the other media
    tools did; they now report the same category and plain sentence."""

    @pytest.mark.asyncio
    async def test_listing_voices_reports_no_provider_detail(self) -> None:
        tool, _ = _voice_tool(model_not_open_error())

        result = await tool.list_tts_voices()

        assert result["success"] is False
        assert result["error_category"] == MODEL_NOT_ENABLED
        readable = json.dumps(result, default=str)
        for leaked in (ACCOUNT_ID, PROVIDER_REQUEST_ID, SDK_REQUEST_ID, "ModelNotOpen"):
            assert leaked not in readable

    @pytest.mark.asyncio
    async def test_cloning_a_voice_reports_no_provider_detail(self) -> None:
        tool, _ = _voice_tool(model_not_open_error())

        result = await tool.clone_tts_voice("My voice", ["reference.mp3"])

        assert result["success"] is False
        assert result["error_category"] == MODEL_NOT_ENABLED
        assert ACCOUNT_ID not in json.dumps(result, default=str)
        assert "ModelNotOpen" not in json.dumps(result, default=str)

    @pytest.mark.asyncio
    async def test_deleting_a_voice_reports_no_provider_detail(self) -> None:
        tool, _ = _voice_tool(model_not_open_error())

        result = await tool.delete_tts_voice("voice-1")

        assert result["success"] is False
        assert result["error_category"] == MODEL_NOT_ENABLED
        assert ACCOUNT_ID not in json.dumps(result, default=str)
        assert "ModelNotOpen" not in json.dumps(result, default=str)
