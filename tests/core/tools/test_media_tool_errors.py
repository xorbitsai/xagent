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
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, Mock

import pytest

from xagent.core.model.asr.base import ASRResult, BaseASR
from xagent.core.model.tts.base import BaseTTS, TTSResult
from xagent.core.model.video.ark import ArkVideoModel
from xagent.core.tools.core.audio_tool import AudioToolCore
from xagent.core.tools.core.image_tool import ImageGenerationToolCore
from xagent.core.tools.core.media_errors import (
    INVALID_REQUEST,
    MODEL_NOT_ENABLED,
    PROVIDER_UNAVAILABLE,
    UNKNOWN,
    classify_media_failure,
    redact_provider_detail,
)
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
