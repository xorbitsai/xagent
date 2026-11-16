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
        assert result.reason is None and result.models is not None
        assert result.models.tts.model_id == "voice"
        assert result.models.voice == "deployment-voice"
        assert tts.close_count == 1
    elif case == "tts_unset":
        assert (result.models, result.reason) == (None, "voice_models_not_configured")
    else:
        assert (result.models, result.reason) == (None, "voice_models_unavailable")
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
        assert result.reason is None and result.models is not None
        assert len(tts_asked) == 1
    else:
        assert (result.models, result.reason) == (None, "voice_models_unavailable")
        assert len(tts_asked) == 1
