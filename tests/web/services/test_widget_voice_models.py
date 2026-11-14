"""Deployment-chosen widget voice models, resolved for the entity owner."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

from unittest.mock import Mock

import pytest
from sqlalchemy.exc import OperationalError

from tests.web.api.test_speech_streaming import FakeASR, FakeTTS, api  # noqa: F401
from xagent.web.api import speech
from xagent.web.models.model import Model
from xagent.web.services import widget_voice

CASES = [
    "available",
    "asr_unset",
    "tts_unset",
    "not_visible",
    "inactive",
    "asr_not_streaming",
    "tts_not_streaming",
    "unusable_voice",
    "instance_rejects_config",
    "database_failure",
]


@pytest.mark.parametrize("case", CASES)
def test_resolve_widget_voice_models(api, monkeypatch, case):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_ASR_MODEL", "live")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", "voice")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_VOICE", "deployment-voice")
    asr, tts = FakeASR(), FakeTTS()
    monkeypatch.setattr(widget_voice, "get_asr_model_instance", Mock(return_value=asr))
    monkeypatch.setattr(widget_voice, "get_tts_model_instance", Mock(return_value=tts))
    owner = 1
    if case == "asr_unset":
        monkeypatch.delenv("XAGENT_WIDGET_VOICE_ASR_MODEL")
    elif case == "tts_unset":
        monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", " ")
    elif case == "not_visible":
        owner = 2
    elif case == "inactive":
        with api.db() as db:
            db.query(Model).filter(Model.model_id == "voice").update(
                {"is_active": False}
            )
            db.commit()
    elif case == "asr_not_streaming":
        asr.supports_streaming = False
    elif case == "tts_not_streaming":
        tts.supports_streaming = False
    elif case == "unusable_voice":
        monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_VOICE", "not a voice!")
    elif case == "instance_rejects_config":
        monkeypatch.setattr(
            widget_voice,
            "get_tts_model_instance",
            Mock(side_effect=ValueError("bad config")),
        )
    elif case == "database_failure":
        monkeypatch.setattr(
            speech,
            "get_session_local",
            Mock(side_effect=OperationalError("SELECT", {}, Exception("down"))),
        )
        with pytest.raises(OperationalError):
            widget_voice.resolve_widget_voice_models(owner)
        return

    result = widget_voice.resolve_widget_voice_models(owner)

    if case == "available":
        assert result.reason is None and result.models is not None
        assert result.models.asr.model_id == "live"
        assert result.models.tts.model_id == "voice"
        assert result.models.voice == "deployment-voice"
        assert result.models.vad_silence_threshold_secs == 0.5
    elif case in {"asr_unset", "tts_unset"}:
        assert (result.models, result.reason) == (None, "voice_models_not_configured")
    else:
        assert (result.models, result.reason) == (None, "voice_models_unavailable")
