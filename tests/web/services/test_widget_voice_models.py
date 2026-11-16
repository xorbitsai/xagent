"""Deployment-chosen widget voice models, resolved for the entity owner."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import logging
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import OperationalError

from tests.shared.widget_voice_env import clear_widget_voice_env
from tests.web.api.test_speech_streaming import FakeASR, FakeTTS, api  # noqa: F401
from xagent.core.model.tts import xinference as xinference_tts
from xagent.web.api import speech
from xagent.web.models.model import Model
from xagent.web.services import widget_voice


@pytest.fixture(autouse=True)
def _widget_voice_env(monkeypatch):
    clear_widget_voice_env(monkeypatch)


SECRET = "synthetic-provider-secret"


class _Discovered:
    """A provider instance that knows whether it streams only once asked."""

    def __init__(self, answer: bool | BaseException | None) -> None:
        self.answer = answer  # None: the check never returns
        self._caps: dict | None = None
        self.asked = asyncio.Event()
        self.close_count = 0

    @property
    def supports_streaming(self) -> bool:
        return (self._caps or {}).get("streaming") is True

    async def get_streaming_capabilities(self) -> dict:
        self.asked.set()
        if self.answer is None:
            await asyncio.Event().wait()
        if isinstance(self.answer, BaseException):
            raise self.answer
        self._caps = {"streaming": self.answer}
        return self._caps

    async def aclose(self) -> None:
        self.close_count += 1


CASES = [
    "available",
    "tts_unset",
    "not_visible",
    "inactive",
    "tts_not_streaming",
    "tts_check_raises",
    "tts_check_times_out",
    "unusable_voice",
    "instance_rejects_config",
    "database_failure",
]


@pytest.mark.parametrize("case", CASES)
async def test_resolve_widget_voice_models(api, monkeypatch, caplog, case):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", "voice")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_VOICE", "deployment-voice")
    tts = _Discovered(True)
    tts_factory = Mock(return_value=tts)
    monkeypatch.setattr(widget_voice, "get_tts_model_instance", tts_factory)
    owner = 1
    if case == "tts_unset":
        monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", " ")
    elif case == "not_visible":
        owner = 2
    elif case == "inactive":
        with api.db() as db:
            db.query(Model).filter(Model.model_id == "voice").update(
                {"is_active": False}
            )
            db.commit()
    elif case == "tts_not_streaming":
        tts.answer = False
    elif case == "tts_check_raises":
        tts.answer = RuntimeError(f"provider said {SECRET}")
    elif case == "tts_check_times_out":
        tts.answer = TimeoutError()
    elif case == "unusable_voice":
        monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_VOICE", "not a voice!")
    elif case == "instance_rejects_config":
        tts_factory = Mock(side_effect=ValueError("bad config"))
        monkeypatch.setattr(widget_voice, "get_tts_model_instance", tts_factory)
    elif case == "database_failure":
        monkeypatch.setattr(
            speech,
            "get_session_local",
            Mock(side_effect=OperationalError("SELECT", {}, Exception("down"))),
        )
        with pytest.raises(OperationalError):
            await widget_voice.resolve_widget_voice_models(owner, request_db=None)
        tts_factory.assert_not_called()
        return

    result = await widget_voice.resolve_widget_voice_models(owner, request_db=None)

    assert SECRET not in caplog.text
    if case == "available":
        assert result is not None
        assert result.tts.model_id == "voice"
        assert result.voice == "deployment-voice"
        assert tts.close_count == 1
    else:
        assert result is None
    if case == "tts_not_streaming":
        assert tts.close_count == 1
    elif case == "tts_check_raises":
        assert tts.close_count == 1
        assert "Widget voice model voice capability check failed: RuntimeError" in (
            caplog.text
        )
    elif case == "tts_check_times_out":
        assert tts.close_count == 1
        assert "Widget voice model voice capability check failed: TimeoutError" in (
            caplog.text
        )
    elif case == "instance_rejects_config":
        assert "cannot be built from its configuration" in caplog.text


async def test_probe_instance_is_closed_when_resolution_is_cancelled(api, monkeypatch):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", "voice")
    probe = _Discovered(None)
    monkeypatch.setattr(
        widget_voice, "get_tts_model_instance", Mock(return_value=probe)
    )
    task = asyncio.create_task(
        widget_voice.resolve_widget_voice_models(1, request_db=None)
    )
    await asyncio.wait_for(probe.asked.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert probe.close_count == 1


@pytest.mark.parametrize("ability", ["asr", "tts"])
async def test_a_provider_that_never_answers_makes_voice_unavailable(
    api, monkeypatch, caplog, ability
):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(widget_voice, "_CAPABILITY_CHECK_TIMEOUT_SECONDS", 0.05)
    hung = _Discovered(None)
    asr, tts = (
        (hung, _Discovered(True)) if ability == "asr" else (_Discovered(True), hung)
    )
    _install_probes(monkeypatch, asr, tts)

    availability = await asyncio.wait_for(
        widget_voice.resolve_widget_voice_availability(1, request_db=None), 5
    )

    assert availability.available is False
    assert availability.reason == widget_voice.VOICE_MODELS_UNAVAILABLE
    assert hung.asked.is_set() and hung.close_count == 1
    assert "capability check failed: TimeoutError" in caplog.text


@pytest.mark.parametrize("streams", [True, False])
async def test_xinference_models_stream_only_after_discovery(api, monkeypatch, streams):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", "voice")
    with api.db() as db:
        db.query(Model).update(
            {"model_provider": "xinference", "base_url": "http://xinference.invalid"}
        )
        db.commit()
    tts_asked: list[object] = []

    def stub(asked: list[object]):
        async def capabilities(provider):
            asked.append(provider)
            return {"asr": {"streaming": streams}, "tts": {"streaming": streams}}

        return capabilities

    monkeypatch.setattr(xinference_tts, "fetch_audio_capabilities", stub(tts_asked))

    result = await widget_voice.resolve_widget_voice_models(1, request_db=None)

    if streams:
        assert result is not None
        assert len(tts_asked) == 1
    else:
        assert result is None
        assert len(tts_asked) == 1


def _install_probes(monkeypatch, asr: _Discovered, tts: _Discovered):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_ASR_MODEL", "live")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", "voice")
    asr_factory, tts_factory = Mock(return_value=asr), Mock(return_value=tts)
    monkeypatch.setattr(widget_voice, "get_asr_model_instance", asr_factory)
    monkeypatch.setattr(widget_voice, "get_tts_model_instance", tts_factory)
    return asr_factory, tts_factory


async def test_each_resolution_asks_only_about_the_model_it_runs_on(api, monkeypatch):
    asr, tts = _Discovered(True), _Discovered(True)
    asr_factory, tts_factory = _install_probes(monkeypatch, asr, tts)

    transcription = await widget_voice.resolve_widget_transcription_model(
        1, request_db=None
    )
    assert transcription is not None
    assert transcription.asr.model_id == "live"
    # ElevenLabs commits a segment after half a second of silence.
    assert transcription.vad_silence_threshold_secs == 0.5
    assert (asr_factory.call_count, tts_factory.call_count) == (1, 0)
    assert (asr.close_count, tts.close_count) == (1, 0)

    synthesis = await widget_voice.resolve_widget_voice_models(1, request_db=None)
    assert synthesis is not None
    assert synthesis.tts.model_id == "voice"
    assert (asr_factory.call_count, tts_factory.call_count) == (1, 1)
    assert (asr.close_count, tts.close_count) == (1, 1)


async def test_a_model_that_is_not_used_cannot_make_a_resolution_unavailable(
    api, monkeypatch
):
    # The other model is unset and its provider would fail if it were asked.
    broken = _Discovered(RuntimeError("down"))
    asr, tts = _Discovered(True), _Discovered(True)
    _install_probes(monkeypatch, asr, tts)
    monkeypatch.delenv("XAGENT_WIDGET_VOICE_TTS_MODEL")
    monkeypatch.setattr(
        widget_voice, "get_tts_model_instance", Mock(return_value=broken)
    )
    transcription = await widget_voice.resolve_widget_transcription_model(
        1, request_db=None
    )
    assert transcription is not None

    _install_probes(monkeypatch, broken, tts)
    monkeypatch.delenv("XAGENT_WIDGET_VOICE_ASR_MODEL")
    synthesis = await widget_voice.resolve_widget_voice_models(1, request_db=None)
    assert synthesis is not None
    assert not broken.asked.is_set()


@pytest.mark.parametrize(
    "case", ["unset", "not_visible", "not_streaming", "check_raises", "times_out"]
)
async def test_transcription_model_unavailable(api, monkeypatch, caplog, case):
    caplog.set_level(logging.WARNING)
    asr, tts = _Discovered(True), _Discovered(True)
    _install_probes(monkeypatch, asr, tts)
    owner = 1
    if case == "unset":
        monkeypatch.setenv("XAGENT_WIDGET_VOICE_ASR_MODEL", " ")
    elif case == "not_visible":
        owner = 2
    elif case == "not_streaming":
        asr.answer = False
    elif case == "check_raises":
        asr.answer = RuntimeError(f"provider said {SECRET}")
    else:
        asr.answer = TimeoutError()

    result = await widget_voice.resolve_widget_transcription_model(
        owner, request_db=None
    )

    assert SECRET not in caplog.text
    assert result is None
    assert asr.close_count == (0 if case in ("unset", "not_visible") else 1)
    assert not tts.asked.is_set()


AVAILABILITY_CASES = [
    "available",
    "asr_unset",
    "tts_unset",
    "asr_not_streaming",
    "tts_not_streaming",
]


@pytest.mark.parametrize("case", AVAILABILITY_CASES)
async def test_availability_needs_both_models_and_asks_each_once(
    api, monkeypatch, case
):
    asr, tts = _Discovered(True), _Discovered(True)
    asr_factory, tts_factory = _install_probes(monkeypatch, asr, tts)
    if case == "asr_unset":
        monkeypatch.delenv("XAGENT_WIDGET_VOICE_ASR_MODEL")
    elif case == "tts_unset":
        monkeypatch.delenv("XAGENT_WIDGET_VOICE_TTS_MODEL")
    elif case == "asr_not_streaming":
        asr.answer = False
    elif case == "tts_not_streaming":
        tts.answer = False

    result = await widget_voice.resolve_widget_voice_availability(1, request_db=None)

    if case == "available":
        assert (result.available, result.reason) == (True, None)
    elif case.endswith("_unset"):
        assert (result.available, result.reason) == (
            False,
            "voice_models_not_configured",
        )
    else:
        assert (result.available, result.reason) == (
            False,
            "voice_models_unavailable",
        )
    # A model is asked about at most once, and its probe instance is closed;
    # the synthesis model is not asked about once the transcription one failed.
    asked = {
        "available": (1, 1),
        "asr_unset": (0, 0),
        "tts_unset": (0, 0),
        "asr_not_streaming": (1, 0),
        "tts_not_streaming": (1, 1),
    }[case]
    assert (asr_factory.call_count, tts_factory.call_count) == asked
    assert (asr.close_count, tts.close_count) == asked


@pytest.mark.parametrize("raw", ["", "  "])
@pytest.mark.parametrize(
    "setting", ["XAGENT_WIDGET_VOICE_ASR_MODEL", "XAGENT_WIDGET_VOICE_TTS_MODEL"]
)
async def test_blank_model_setting_is_not_configured(monkeypatch, setting, raw):
    # A deployment that leaves the variable empty passes an empty string.
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_ASR_MODEL", "live")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", "voice")
    monkeypatch.setenv(setting, raw)
    availability = await widget_voice.resolve_widget_voice_availability(
        1, request_db=None
    )
    assert availability.available is False
    assert availability.reason == widget_voice.VOICE_MODELS_NOT_CONFIGURED
