"""Widget guest transcription (WS /api/widget/voice/asr) and the voice flag."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.shared.widget_voice_env import clear_widget_voice_env
from tests.web.api.test_speech_streaming import FakeASR
from tests.web.api.test_widget_voice_tts import (  # noqa: F401
    SECRET,
    _set_switch,
    voice,
)
from xagent.web import jwt_codec as jwt
from xagent.web.api import widget as widget_api
from xagent.web.api.public_chat_access import create_public_chat_access_token
from xagent.web.api.speech import speech_router
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.agent import Agent
from xagent.web.models.deployment import Deployment
from xagent.web.models.model import Model
from xagent.web.models.user import UserModel
from xagent.web.services import (
    quota_hooks,
    speech_streams,
    widget_voice,
    widget_voice_budget,
)
from xagent.web.services.release_hold import release_hold
from xagent.web.services.share_rate_limit import (
    get_share_rate_limiter,
    reset_share_rate_limiter,
)
from xagent.web.services.widget_voice_budget import ASR_BUCKETS, TTS_BUCKETS

from .conftest import (
    _direct_db_session,
    _install_one_slot_queue_pool,
    app_for_tests,
    client,
)
from .test_workforce_widget import (
    _authenticate_widget_guest_by_key,
    _create_workforce,
    _enable_widget,
)

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def _widget_voice_env(monkeypatch):
    clear_widget_voice_env(monkeypatch)


PATH = "/api/widget/voice/asr"
LOGIN_PATH = "/api/models/speech/asr/stream"
START = {"type": "start"}
LIMITS = {
    "asr-visitor": "XAGENT_WIDGET_VOICE_ASR_VISITOR_DAILY_SECONDS",
    "asr-entity": "XAGENT_WIDGET_VOICE_ASR_ENTITY_DAILY_SECONDS",
    "asr-site": "XAGENT_WIDGET_VOICE_ASR_SITE_DAILY_SECONDS",
}


# Signed-in speech next to the widget routes, for the tests that use both.
_both = FastAPI()
_both.include_router(widget_api.widget_router)
_both.include_router(speech_router)
_both.dependency_overrides = app_for_tests.dependency_overrides
both_client = TestClient(_both, raise_server_exceptions=False)


def _protocols(headers: dict[str, str]) -> list[str]:
    token = headers["Authorization"].removeprefix("Bearer ")
    return ["xagent-auth-v1", f"xagent-auth-token.{token}"]


def _connect(headers: dict[str, str], via: TestClient = client):
    return via.websocket_connect(PATH, subprotocols=_protocols(headers))


def _until_closed(ws) -> tuple[list[dict[str, Any]], tuple[int, str]]:
    frames: list[dict[str, Any]] = []
    with pytest.raises(WebSocketDisconnect) as closed:
        while True:
            frames.append(ws.receive_json())
    return frames, (closed.value.code, closed.value.reason)


def _start(ws, start: dict[str, Any] | None = None) -> None:
    ws.send_json(start or START)
    assert ws.receive_json()["type"] == "session_started"


def _asr_counts() -> dict[str, int]:
    counts = widget_voice_budget._memory_store._counts
    return {
        key.split(":")[3]: value
        for key, value in counts.items()
        if key.split(":")[3].startswith("asr-")
    }


@pytest.fixture
def asr_model(voice, monkeypatch):
    """The deployment's transcription model, next to the synthesis model the
    ``voice`` fixture sets up. Returns the factory that builds the instances
    the model resolution asks about."""
    db = _direct_db_session()
    try:
        row = Model(
            model_id="live",
            model_name="live",
            model_provider="elevenlabs",
            api_key=SECRET,
            category="speech",
            abilities=["asr"],
            is_active=True,
        )
        db.add(row)
        db.flush()
        db.add(UserModel(user_id=voice.owner_id, model_id=row.id, is_owner=True))
        db.commit()
    finally:
        db.close()
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_ASR_MODEL", "live")
    probe_factory = Mock(return_value=FakeASR())
    monkeypatch.setattr(widget_voice, "get_asr_model_instance", probe_factory)
    return probe_factory


@pytest.fixture
def asr(voice, asr_model, monkeypatch):
    reset_share_rate_limiter()
    resolve = AsyncMock(wraps=widget_voice.resolve_widget_transcription_model)
    monkeypatch.setattr(widget_api, "resolve_widget_transcription_model", resolve)
    model = FakeASR()
    factory = Mock(return_value=model)
    monkeypatch.setattr(speech_streams, "get_asr_model_instance", factory)
    reserve = AsyncMock(wraps=widget_voice_budget.reserve_asr)
    monkeypatch.setattr(speech_streams, "reserve_asr", reserve)
    clock = [0.0]
    monkeypatch.setattr(speech_streams, "_now", lambda: clock[0])
    # A session the code under test fails to end must not wait the full
    # first-frame and audio timeout before the test fails.
    monkeypatch.setattr(speech_streams, "_IO_TIMEOUT", 2.0)
    scopes: list[tuple[Any, ...]] = []
    real_scope = speech_streams.usage_scope

    def scope(user_id, *, usage_source, model_ids):
        scopes.append((user_id, usage_source, tuple(model_ids)))
        return real_scope(user_id, usage_source=usage_source, model_ids=model_ids)

    monkeypatch.setattr(speech_streams, "usage_scope", scope)
    access = Mock(wraps=widget_api._widget_voice_access_sync)
    monkeypatch.setattr(widget_api, "_widget_voice_access_sync", access)
    yield SimpleNamespace(
        voice=voice,
        guest=voice.guest,
        model=model,
        factory=factory,
        probe_factory=asr_model,
        resolve=resolve,
        reserve=reserve,
        clock=clock,
        scopes=scopes,
        access=access,
    )
    reset_share_rate_limiter()
    # Every exit path hands back its session slot and write token.
    assert speech_streams._guest_sessions.total == 0
    assert speech_streams._guest_sessions.by_entity == {}
    assert speech_streams._guest_sessions.by_visitor == {}
    assert release_hold.socket_write_count == 0


def _untouched(asr) -> None:
    asr.resolve.assert_not_called()
    asr.reserve.assert_not_called()
    asr.factory.assert_not_called()
    assert asr.scopes == [] and asr.voice.usage == []
    assert widget_voice_budget._memory_store._counts == {}


def _guest_token(asr, **changes: Any) -> dict[str, str]:
    token = asr.guest["Authorization"].removeprefix("Bearer ")
    claims = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    return {
        "Authorization": "Bearer "
        + jwt.encode({**claims, **changes}, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    }


@pytest.mark.parametrize(
    "credential",
    ["none", "login", "share", "expired", "widget_disabled", "widget_key_rotated"],
)
def test_widget_voice_rejects_foreign_credentials(asr, credential):
    token = asr.guest["Authorization"].removeprefix("Bearer ")
    claims = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    headers = asr.guest
    if credential == "login":
        headers = asr.voice.admin
    elif credential == "share":
        headers = {
            "Authorization": "Bearer "
            + create_public_chat_access_token({**claims, "auth_mode": "share"})
        }
    elif credential == "expired":
        headers = _guest_token(asr, exp=datetime.now(timezone.utc) - timedelta(hours=1))
    elif credential == "widget_disabled":
        changed = client.put(
            f"/api/agents/{asr.voice.agent_id}",
            headers=asr.voice.admin,
            json={"widget_enabled": False},
        )
        assert changed.status_code == 200, changed.text
    elif credential == "widget_key_rotated":
        changed = client.post(
            f"/api/agents/{asr.voice.agent_id}/widget-key/rotate",
            headers=asr.voice.admin,
        )
        assert changed.status_code == 200, changed.text
    protocols = ["xagent-auth-v1"] if credential == "none" else _protocols(headers)
    with client.websocket_connect(PATH, subprotocols=protocols) as ws:
        frames, closed = _until_closed(ws)
    assert frames == []
    assert closed[0] == 4003
    if credential == "none":
        asr.access.assert_not_called()
    _untouched(asr)


def _switch_off_agent(asr) -> None:
    db = _direct_db_session()
    try:
        db.get(Agent, asr.voice.agent_id).widget_voice_enabled = False
        db.commit()
    finally:
        db.close()


@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_widget_voice_requires_switch(asr, entity):
    headers = asr.guest
    if entity == "agent":
        _switch_off_agent(asr)
    else:
        workforce_id = _create_workforce("Voice ASR switch workforce")
        headers = _authenticate_widget_guest_by_key(_enable_widget(workforce_id))
    with _connect(headers) as ws:
        frames, closed = _until_closed(ws)
    assert frames == []
    assert closed == (4003, "Voice is not available for this chat")
    _untouched(asr)


def test_workforce_guest_with_the_switch_on_is_served(asr):
    workforce_id = _create_workforce("Voice ASR workforce on")
    headers = _authenticate_widget_guest_by_key(_enable_widget(workforce_id))
    _set_switch(Deployment, workforce_id)
    with _connect(headers) as ws:
        _start(ws)
        ws.send_json({"type": "close"})
        _until_closed(ws)
    assert asr.reserve.call_args.args[0] == f"workforce:{workforce_id}"


@pytest.mark.parametrize(
    "start",
    [
        {"type": "start", "model_id": "live"},
        {"type": "start", "vad": False},
        {"type": "start", "language": "en"},
        {"type": "start", "vad_silence_threshold_secs": 2},
        {"type": "start", "sample_rate": 12345},
        {"type": "stop"},
    ],
)
def test_widget_voice_rejects_client_model_id(asr, start):
    with _connect(asr.guest) as ws:
        ws.send_json(start)
        frames, closed = _until_closed(ws)
    assert frames == [
        {
            "type": "error",
            "code": "invalid_request",
            "message": "Invalid speech request",
        }
    ]
    assert closed == (1008, "Invalid speech request")
    _untouched(asr)


def test_widget_voice_session_is_billed_to_the_owner(asr, caplog):
    caplog.set_level(logging.INFO, logger=speech_streams.__name__)
    with _connect(asr.guest) as ws:
        _start(ws)
        ws.send_bytes(b"\x00\x00" * 16000)
        assert ws.receive_json()["type"] == "partial"
        assert ws.receive_json()["type"] == "final"
        ws.send_json({"type": "close"})
        frames, closed = _until_closed(ws)
    assert closed[0] == 1000
    assert asr.factory.call_args.args[0].model_id == "live"
    # The server, not the guest, chose the stream options.
    assert asr.model.sessions[0].options == {
        "format": "pcm",
        "sample_rate": 16000,
        "vad": True,
        "vad_silence_threshold_secs": 0.5,
    }
    assert asr.scopes == [(asr.voice.owner_id, "speech_transcribe", ("live",))]
    assert len(asr.voice.usage) == 1
    user_id, rows, source = asr.voice.usage[0]
    assert (user_id, source) == (asr.voice.owner_id, "speech_transcribe")
    assert [row["quantity"] for row in rows] == [1.0]
    # One second of audio in no wall time settles at one second.
    assert set(_asr_counts().values()) == {1}
    lines = [
        r.getMessage()
        for r in caplog.records
        if "Speech transcription usage" in r.getMessage()
    ]
    assert len(lines) == 1
    assert (
        f"audience=guest user_id={asr.voice.owner_id} "
        f"entity=agent:{asr.voice.agent_id} visitor=testclient" in lines[0]
    )
    assert "model=live seconds=1 result=ok" in lines[0]


def test_usage_log_holds_only_the_network_of_the_visitor_address(
    asr, caplog, monkeypatch
):
    monkeypatch.setattr(
        widget_api, "remote_ip_from_request", lambda _websocket: "203.0.113.77"
    )
    caplog.set_level(logging.INFO)
    with _connect(asr.guest) as ws:
        _start(ws)
        ws.send_json({"type": "close"})
        _until_closed(ws)
    lines = [
        r.getMessage()
        for r in caplog.records
        if "Speech transcription usage" in r.getMessage()
    ]
    assert len(lines) == 1
    assert "visitor=203.0.113.0/24" in lines[0]
    # No log line of the session carries the full address.
    assert "203.0.113.77" not in caplog.text


def test_widget_voice_without_models_is_unavailable(asr, monkeypatch):
    monkeypatch.delenv("XAGENT_WIDGET_VOICE_ASR_MODEL")
    with _connect(asr.guest) as ws:
        ws.send_json(START)
        frames, closed = _until_closed(ws)
    assert frames == [
        {
            "type": "error",
            "code": "voice_unavailable",
            "message": "Voice is unavailable",
        }
    ]
    assert closed == (1008, "Voice is unavailable")
    asr.factory.assert_not_called()
    assert widget_voice_budget._memory_store._counts == {} and asr.voice.usage == []


def test_widget_voice_ip_gate_refuses_before_accept(asr, monkeypatch):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_CONNECT_IP_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()
    accept = AsyncMock(wraps=widget_api._accept_public_chat_websocket)
    monkeypatch.setattr(widget_api, "_accept_public_chat_websocket", accept)
    with _connect(asr.guest) as ws:
        _start(ws)
        ws.send_json({"type": "close"})
        _until_closed(ws)
    assert accept.call_count == 1
    with pytest.raises(WebSocketDisconnect) as refused:
        with _connect(asr.guest):
            pass
    assert refused.value.code == 4008
    # Refused before the handshake was accepted or the credential was read.
    assert accept.call_count == 1 and asr.access.call_count == 1


def test_widget_voice_entity_gate_refuses_after_the_credential(asr, monkeypatch):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_CONNECT_RATE_LIMIT", "1/minute")
    reset_share_rate_limiter()
    with _connect(asr.guest) as ws:
        _start(ws)
        ws.send_json({"type": "close"})
        _until_closed(ws)
    calls = asr.reserve.call_count
    with _connect(asr.guest) as ws:
        frames, closed = _until_closed(ws)
    assert frames == [] and closed == (1008, "Too many requests")
    assert asr.access.call_count == 2 and asr.reserve.call_count == calls


@pytest.mark.parametrize(
    "method", ["allow_widget_voice_connect_ip", "allow_widget_voice_connect_entity"]
)
def test_widget_voice_connect_gate_storage_errors_refuse(asr, monkeypatch, method):
    limiter = get_share_rate_limiter()

    def broken(*_args):
        raise ConnectionError("storage down")

    monkeypatch.setattr(limiter._limiter, "hit", broken)
    if method == "allow_widget_voice_connect_ip":
        with pytest.raises(WebSocketDisconnect) as refused:
            with _connect(asr.guest):
                pass
        assert refused.value.code == 4008
    else:
        monkeypatch.setattr(limiter, "allow_widget_voice_connect_ip", lambda _ip: True)
        with _connect(asr.guest) as ws:
            _, closed = _until_closed(ws)
        assert closed == (1008, "Too many requests")
    _untouched(asr)


@pytest.mark.parametrize("cap", ["global", "per_entity", "per_visitor"])
def test_widget_voice_concurrency_caps(asr, monkeypatch, cap):
    variables = {
        "global": "XAGENT_WIDGET_VOICE_MAX_SESSIONS",
        "per_entity": "XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_ENTITY",
        "per_visitor": "XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR",
    }
    for name in variables.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variables[cap], "2")
    with _connect(asr.guest) as first, _connect(asr.guest) as second:
        _start(first)
        _start(second)
        assert speech_streams._guest_sessions.total == 2
        with _connect(asr.guest) as third:
            third.send_json(START)
            frames, closed = _until_closed(third)
        assert frames == [
            {"type": "error", "code": "voice_busy", "message": "Voice is busy"}
        ]
        assert closed == (1008, "Voice is busy")
        # The refused session reserved no budget.
        assert asr.reserve.call_count == 2
        first.send_json({"type": "close"})
        _until_closed(first)
        with _connect(asr.guest) as again:
            _start(again)
            again.send_json({"type": "close"})
            _until_closed(again)
        second.send_json({"type": "close"})
        _until_closed(second)


def test_voice_global_concurrency_counts_widget_entries_only(asr, monkeypatch):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS", "1")
    login = _protocols(asr.voice.admin)
    with both_client.websocket_connect(LOGIN_PATH, subprotocols=login) as signed_in:
        _start(signed_in, {"type": "start", "model_id": "live"})
        # A signed-in session takes no guest slot.
        assert speech_streams._guest_sessions.total == 0
        with _connect(asr.guest, both_client) as guest:
            _start(guest)
            with _connect(asr.guest, both_client) as refused:
                refused.send_json(START)
                _, closed = _until_closed(refused)
            assert closed == (1008, "Voice is busy")
            # The guest cap does not apply to signed-in speech.
            with both_client.websocket_connect(LOGIN_PATH, subprotocols=login) as more:
                _start(more, {"type": "start", "model_id": "live"})
                more.send_json({"type": "close"})
                _until_closed(more)
            guest.send_json({"type": "close"})
            _until_closed(guest)
        signed_in.send_json({"type": "close"})
        _until_closed(signed_in)


def test_login_session_runs_no_recheck(asr, monkeypatch):
    sizes: list[int] = []
    real_race = speech_streams._race

    async def race(*operations):
        sizes.append(len(operations))
        await real_race(*operations)

    monkeypatch.setattr(speech_streams, "_race", race)
    login = _protocols(asr.voice.admin)
    with both_client.websocket_connect(LOGIN_PATH, subprotocols=login) as ws:
        _start(ws, {"type": "start", "model_id": "live"})
        ws.send_json({"type": "close"})
        _until_closed(ws)
    with _connect(asr.guest, both_client) as ws:
        _start(ws)
        ws.send_json({"type": "close"})
        _until_closed(ws)
    assert sizes == [2, 3]


@pytest.mark.parametrize("failure", ["limit", "storage"])
@pytest.mark.parametrize("bucket", [b.name for b in ASR_BUCKETS])
def test_widget_voice_budget_buckets(asr, monkeypatch, caplog, bucket, failure):
    assert len(TTS_BUCKETS + ASR_BUCKETS) == 6
    if failure == "limit":
        monkeypatch.setenv(LIMITS[bucket], "599")
    else:
        store = widget_voice_budget._memory_store
        real = store.incrby

        def incrby(key, amount):
            if f":{bucket}:" in key:
                raise ConnectionError("storage down")
            return real(key, amount)

        monkeypatch.setattr(store, "incrby", incrby)
    caplog.set_level(logging.INFO, logger=speech_streams.__name__)
    with _connect(asr.guest) as ws:
        ws.send_json(START)
        frames, closed = _until_closed(ws)
    code, message = (
        ("voice_budget_exhausted", "Voice limit reached, try again later")
        if failure == "limit"
        else ("voice_unavailable", "Voice is unavailable")
    )
    assert frames == [{"type": "error", "code": code, "message": message}]
    assert closed == (1008, message)
    assert asr.model.sessions == [] and not asr.model.opening.is_set()
    assert asr.model.close_count == 1
    assert asr.scopes == [] and asr.voice.usage == []
    assert set(_asr_counts().values()) <= {0}
    result = "budget_exhausted" if failure == "limit" else "budget_unavailable"
    lines = [
        r.getMessage()
        for r in caplog.records
        if "Speech transcription usage" in r.getMessage()
    ]
    assert len(lines) == 1 and f"seconds=0 result={result}" in lines[0]


def _flip(asr, change: str) -> None:
    if change == "switch_off":
        _switch_off_agent(asr)
    elif change == "widget_disabled":
        changed = client.put(
            f"/api/agents/{asr.voice.agent_id}",
            headers=asr.voice.admin,
            json={"widget_enabled": False},
        )
        assert changed.status_code == 200, changed.text
    else:
        changed = client.post(
            f"/api/agents/{asr.voice.agent_id}/widget-key/rotate",
            headers=asr.voice.admin,
        )
        assert changed.status_code == 200, changed.text


@pytest.mark.parametrize(
    "change", ["switch_off", "widget_disabled", "credential", "read_failure"]
)
def test_widget_voice_recheck_closes_on_switch_off(asr, monkeypatch, change):
    monkeypatch.setattr(widget_api, "_WIDGET_VOICE_RECHECK_SECONDS", 0.01)
    if change == "read_failure":
        calls = {"n": 0}
        real = widget_api._widget_voice_access_sync

        def access(token):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError(f"database down {SECRET}")
            return real(token)

        monkeypatch.setattr(widget_api, "_widget_voice_access_sync", access)
    with _connect(asr.guest) as ws:
        _start(ws)
        asr.clock[0] = 7.0
        if change != "read_failure":
            _flip(asr, change)
        frames, closed = _until_closed(ws)
    if change == "read_failure":
        assert frames == [
            {
                "type": "error",
                "code": "stream_failed",
                "message": "Speech stream failed",
            }
        ]
        assert closed == (1011, "Speech stream failed")
    else:
        # The close code is the only signal: no error frame precedes it.
        assert [frame for frame in frames if frame.get("type") == "error"] == []
        assert closed == (4003, "Voice is unavailable")
    assert set(_asr_counts().values()) == {7}


def test_widget_voice_recheck_that_passes_keeps_the_session(asr, monkeypatch):
    monkeypatch.setattr(widget_api, "_WIDGET_VOICE_RECHECK_SECONDS", 0.01)
    with _connect(asr.guest) as ws:
        _start(ws)
        for _ in range(3):
            calls = asr.access.call_count
            ws.send_bytes(b"\x00\x00" * 100)
            assert ws.receive_json()["type"] == "partial"
            assert ws.receive_json()["type"] == "final"
            for _ in range(200):
                if asr.access.call_count > calls:
                    break
                asr.model.closed.wait(0.01)
            assert asr.access.call_count > calls
        ws.send_json({"type": "close"})
        _, closed = _until_closed(ws)
    assert closed[0] == 1000
    assert asr.access.call_count >= 4


@pytest.mark.parametrize(
    "exit_path", ["normal", "disconnect", "timeout", "recheck", "provider_error"]
)
def test_widget_voice_asr_budget_settles_on_every_exit(asr, monkeypatch, exit_path):
    if exit_path == "timeout":
        monkeypatch.setattr(speech_streams, "_IO_TIMEOUT", 0.2)
    if exit_path == "recheck":
        monkeypatch.setattr(widget_api, "_WIDGET_VOICE_RECHECK_SECONDS", 0.01)
    if exit_path == "provider_error":
        asr.model.send_error = RuntimeError("provider down")
    with _connect(asr.guest) as ws:
        _start(ws)
        asr.clock[0] = 7.0
        if exit_path == "normal":
            ws.send_json({"type": "close"})
        elif exit_path == "disconnect":
            ws.close()
        elif exit_path == "recheck":
            _switch_off_agent(asr)
        elif exit_path == "provider_error":
            ws.send_bytes(b"\x00\x00")
        if exit_path != "disconnect":
            _until_closed(ws)
    if exit_path == "disconnect":
        assert asr.model.closed.wait(2)
        # The settlement runs after the model is closed.
        for _ in range(200):
            if set(_asr_counts().values()) == {7}:
                break
            asr.model.closed.wait(0.01)
    assert set(_asr_counts().values()) == {7}
    assert len(_asr_counts()) == 3


def test_widget_voice_asr_budget_full_audio_settles_at_the_limit(asr):
    with _connect(asr.guest) as ws:
        _start(ws, {"type": "start", "sample_rate": 8000})
        frame = b"\x00\x00" * 32000
        for _ in range(150):
            ws.send_bytes(frame)
        ws.close()
    assert asr.model.closed.wait(5)
    for _ in range(500):
        if set(_asr_counts().values()) == {600}:
            break
        asr.model.closed.wait(0.01)
    # 150 frames of 64000 bytes at 8 kHz is exactly 600 seconds.
    assert set(_asr_counts().values()) == {600}


def test_widget_voice_asr_budget_idle_session_settles_at_wall_time(asr):
    with _connect(asr.guest) as ws:
        _start(ws)
        ws.send_bytes(b"\x00\x00")
        assert ws.receive_json()["type"] == "partial"
        asr.clock[0] = 60.0
        ws.send_json({"type": "close"})
        _until_closed(ws)
    assert set(_asr_counts().values()) == {60}


def _gate_refusal(asr, message: str) -> dict[str, str]:
    # The wall clock moves on before the stream ever opens.
    asr.clock[0] = 30.0
    return {"message": message}


@pytest.mark.parametrize("failure", ["open_stream", "usage_gate"])
def test_widget_voice_asr_budget_failure_before_the_stream_refunds_all(
    asr, monkeypatch, failure
):
    if failure == "open_stream":

        async def broken(**_options):
            # The wall clock moves on before the stream ever opens.
            asr.clock[0] = 30.0
            raise ValueError(f"bad key {SECRET}")

        monkeypatch.setattr(asr.model, "open_stream", broken)
        expected = ("model_unavailable", "Voice is unavailable", 1008)
    else:
        monkeypatch.setattr(
            quota_hooks,
            "_standalone_gate_hook",
            lambda user_id, source: _gate_refusal(asr, f"team quota {SECRET}"),
        )
        expected = ("standalone_usage_refused", "Usage limit reached", 1008)
    with _connect(asr.guest) as ws:
        ws.send_json(START)
        frames, closed = _until_closed(ws)
    assert frames == [{"type": "error", "code": expected[0], "message": expected[1]}]
    assert closed == (expected[2], expected[1])
    asr.reserve.assert_called_once()
    assert set(_asr_counts().values()) == {0}
    assert asr.voice.usage == []


@pytest.mark.parametrize(
    "failure", ["provider", "prepare_http", "prepare_runtime", "usage_gate"]
)
def test_widget_voice_error_frames_are_fixed_text(asr, monkeypatch, failure):
    if failure == "provider":
        asr.model.send_error = RuntimeError(f"provider said {SECRET}")
        expected = ("stream_failed", "Speech stream failed", 1011)
    elif failure == "prepare_http":
        monkeypatch.setattr(
            widget_api,
            "resolve_widget_transcription_model",
            AsyncMock(side_effect=HTTPException(400, f"Incompatible {SECRET}")),
        )
        expected = ("voice_unavailable", "Voice is unavailable", 1008)
    elif failure == "prepare_runtime":
        monkeypatch.setattr(
            widget_api,
            "resolve_widget_transcription_model",
            AsyncMock(side_effect=RuntimeError(f"database said {SECRET}")),
        )
        expected = ("stream_failed", "Speech stream failed", 1011)
    else:
        monkeypatch.setattr(
            quota_hooks,
            "_standalone_gate_hook",
            lambda user_id, source: {
                "message": f"team quota {SECRET}",
                "plan": "owner-plan",
            },
        )
        expected = ("standalone_usage_refused", "Usage limit reached", 1008)
    with _connect(asr.guest) as ws:
        ws.send_json(START)
        if failure == "provider":
            assert ws.receive_json()["type"] == "session_started"
            ws.send_bytes(b"\x00\x00")
        frames, closed = _until_closed(ws)
    assert frames == [{"type": "error", "code": expected[0], "message": expected[1]}]
    assert closed == (expected[2], expected[1])
    text = repr(frames) + closed[1]
    assert SECRET not in text and "owner-plan" not in text


# ----- the voice flag on POST /api/widget/auth -----


def _auth(asr, widget_key: str | None = None) -> dict[str, Any]:
    if widget_key is None:
        db = _direct_db_session()
        try:
            widget_key = str(db.get(Agent, asr.voice.agent_id).widget_key)
        finally:
            db.close()
    response = client.post(
        "/api/widget/auth", json={"guest_id": "g", "widget_key": widget_key}
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    "state", ["on", "switch_off", "asr_unavailable", "tts_unavailable"]
)
def test_widget_auth_voice_flag(asr, monkeypatch, state):
    if state == "switch_off":
        _switch_off_agent(asr)
    elif state == "asr_unavailable":
        monkeypatch.delenv("XAGENT_WIDGET_VOICE_ASR_MODEL")
    elif state == "tts_unavailable":
        monkeypatch.delenv("XAGENT_WIDGET_VOICE_TTS_MODEL")
    resolve = AsyncMock(wraps=widget_api.resolve_widget_voice_availability)
    monkeypatch.setattr(widget_api, "resolve_widget_voice_availability", resolve)
    assert _auth(asr)["voice_enabled"] is (state == "on")
    # The switch is read first; models are resolved only when it is on.
    assert resolve.call_count == (0 if state == "switch_off" else 1)


def test_widget_voice_recheck_runs_on_a_one_slot_pool(asr, monkeypatch):
    monkeypatch.setattr(widget_api, "_WIDGET_VOICE_RECHECK_SECONDS", 0.01)
    _install_one_slot_queue_pool(monkeypatch)
    with _connect(asr.guest) as ws:
        _start(ws)
        # The handshake read the access once; each recheck reads it again.
        for _ in range(500):
            if asr.access.call_count >= 3:
                break
            asr.model.closed.wait(0.01)
        assert asr.access.call_count >= 3
        ws.send_json({"type": "close"})
        frames, closed = _until_closed(ws)
    assert closed[0] != 1011
    assert all(frame["type"] != "error" for frame in frames)


def test_widget_voice_without_a_source_address_is_unavailable(asr, monkeypatch):
    monkeypatch.setattr(widget_api, "remote_ip_from_request", lambda _websocket: None)
    with _connect(asr.guest) as ws:
        ws.send_json(START)
        frames, closed = _until_closed(ws)
    assert frames == [
        {
            "type": "error",
            "code": "voice_unavailable",
            "message": "Voice is unavailable",
        }
    ]
    assert closed == (1008, "Voice is unavailable")
    asr.resolve.assert_not_called()
    asr.reserve.assert_not_called()
    assert widget_voice_budget._memory_store._counts == {}


def test_widget_auth_with_a_hung_provider_returns_voice_off(asr, monkeypatch):
    class HungASR(FakeASR):
        async def get_streaming_capabilities(self):
            await asyncio.Event().wait()

    hung = HungASR()
    asr.probe_factory.return_value = hung
    monkeypatch.setattr(widget_voice, "_CAPABILITY_CHECK_TIMEOUT_SECONDS", 0.05)
    assert _auth(asr)["voice_enabled"] is False
    assert asr.probe_factory.call_count == 1


def test_widget_auth_voice_flag_for_a_workforce(asr):
    workforce_id = _create_workforce("Voice flag workforce")
    key = _enable_widget(workforce_id)
    assert _auth(asr, key)["voice_enabled"] is False
    _set_switch(Deployment, workforce_id)
    assert _auth(asr, key)["voice_enabled"] is True


@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_widget_auth_with_voice_on_succeeds_on_a_one_slot_pool(
    asr, monkeypatch, entity
):
    if entity == "workforce":
        workforce_id = _create_workforce("Voice pool workforce")
        key = _enable_widget(workforce_id)
        _set_switch(Deployment, workforce_id)
    else:
        db = _direct_db_session()
        try:
            key = str(db.get(Agent, asr.voice.agent_id).widget_key)
        finally:
            db.close()
    _install_one_slot_queue_pool(monkeypatch)
    assert _auth(asr, key)["voice_enabled"] is True


@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_share_auth_never_advertises_voice(asr, entity):
    if entity == "agent":
        response = client.post(
            f"/api/agents/{asr.voice.agent_id}/share-link", headers=asr.voice.admin
        )
        assert response.status_code == 200, response.text
        share_token = response.json()["share_token"]
    else:
        workforce_id = _create_workforce("Voice share workforce")
        response = client.post(
            f"/api/workforces/{workforce_id}/share-link", headers=asr.voice.admin
        )
        assert response.status_code == 200, response.text
        share_token = response.json()["share_token"]
        _set_switch(Deployment, workforce_id)
    response = client.post("/api/share/auth", json={"share_token": share_token})
    assert response.status_code == 200, response.text
    assert response.json()["voice_enabled"] is False


@pytest.mark.parametrize("failure", ["switch", "models"])
def test_widget_auth_voice_flag_read_failure_is_service_error(
    asr, monkeypatch, failure
):
    if failure == "switch":
        monkeypatch.setattr(
            widget_api,
            "_widget_voice_enabled",
            Mock(side_effect=RuntimeError("database down")),
        )
    else:
        monkeypatch.setattr(
            widget_voice,
            "_resolve",
            AsyncMock(side_effect=RuntimeError("database down")),
        )
    db = _direct_db_session()
    try:
        key = str(db.get(Agent, asr.voice.agent_id).widget_key)
    finally:
        db.close()
    response = client.post(
        "/api/widget/auth", json={"guest_id": "g", "widget_key": key}
    )
    assert response.status_code == 500


def test_widget_auth_asks_each_provider_once_per_login(asr, monkeypatch):
    """A provider that answers the streaming question over the network is
    asked once per model for one login, not once per use of the answer."""
    probes = {"asr": 0, "tts": 0}

    class Probed:
        def __init__(self, ability: str) -> None:
            self.ability = ability

        async def get_streaming_capabilities(self) -> dict[str, bool]:
            probes[self.ability] += 1
            return {"streaming": True}

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(widget_voice, "get_asr_model_instance", lambda c: Probed("asr"))
    monkeypatch.setattr(widget_voice, "get_tts_model_instance", lambda c: Probed("tts"))

    assert _auth(asr)["voice_enabled"] is True
    assert probes == {"asr": 1, "tts": 1}
    assert _auth(asr)["voice_enabled"] is True
    assert probes == {"asr": 2, "tts": 2}
