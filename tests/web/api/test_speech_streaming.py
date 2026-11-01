"""Streaming transport contracts, with real auth/visibility and fake providers."""

from __future__ import annotations

import asyncio
import json
import threading
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.websockets import WebSocket, WebSocketDisconnect

from tests.web.auth_token_cases import build_access_token
from xagent.core.model.asr.base import ASRStreamEvent, ASRStreamSession
from xagent.core.model.chat.token_context import MediaCallType
from xagent.core.model.media_metering import record_provider_usage
from xagent.core.model.tts.base import TTSChunk
from xagent.core.model.tts.elevenlabs import ElevenLabsTTS
from xagent.web.api import speech, websocket_auth
from xagent.web.models import database
from xagent.web.models.model import Model
from xagent.web.models.user import User, UserModel
from xagent.web.schemas.speech import TTSStreamRequest
from xagent.web.services import model_service, quota_hooks

ASR_PATH = "/api/models/speech/asr/stream"
TTS_PATH = "/api/models/speech/tts/stream"


class FakeASRSession(ASRStreamSession):
    def __init__(self, model, options):
        self.model = model
        self.options = options
        self.events = asyncio.Queue()
        self.audio = []
        self.closed = False
        self.commits = 0

    async def send_audio(self, audio):
        if self.model.send_error is not None:
            raise self.model.send_error
        self.audio.append(audio)
        await self.events.put(ASRStreamEvent(type="partial", text="hello"))
        if self.options["vad"]:
            await self.events.put(ASRStreamEvent(type="final", text="Hello."))

    async def commit(self):
        if self.options["vad"]:
            raise ValueError("VAD owns commits")
        self.commits += 1
        await self.events.put(ASRStreamEvent(type="final", text="Hello."))

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield ASRStreamEvent(type="session_started", session_id="test-session")
        while True:
            yield await self.events.get()

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        if self.audio:
            record_provider_usage(
                self.model, MediaCallType.ASR, sum(map(len, self.audio)) / 32000
            )


class FakeASR:
    supports_streaming = True
    api_key = "synthetic-provider-secret"
    model_id = "live"
    model_name = "scribe_v2_realtime"
    provider_name = "elevenlabs"

    def __init__(self):
        self.sessions = []
        self.opening = threading.Event()
        self.closed = threading.Event()
        self.block_open = False
        self.close_count = 0
        self.send_error = None

    async def open_stream(self, **options):
        self.opening.set()
        if self.block_open:
            await asyncio.Event().wait()
        session = FakeASRSession(self, options)
        self.sessions.append(session)
        return session

    async def aclose(self):
        self.close_count += 1
        self.closed.set()


class FakeTTS:
    supports_streaming = True
    api_key = "synthetic-provider-secret"
    model_id = "voice"
    model_name = "eleven_flash_v2_5"
    provider_name = "elevenlabs"

    def __init__(self):
        self.calls = []
        self.close_count = 0
        self.stream_closed = False
        self.error = None
        self.error_after_first = None
        self.gate = None
        self.block_first = False

    async def synthesize_stream(self, **options):
        self.calls.append(options)
        try:
            if self.error is not None:
                raise self.error
            if self.block_first:
                await self.gate.wait()
            record_provider_usage(self, MediaCallType.TTS, len(options["text"]))
            yield TTSChunk(audio=b"first", format="pcm", sample_rate=24000)
            if self.error_after_first is not None:
                raise self.error_after_first
            if self.gate is not None:
                await self.gate.wait()
            yield TTSChunk(audio=b"second", format="pcm", sample_rate=24000)
        finally:
            self.stream_closed = True

    async def aclose(self):
        self.close_count += 1


@pytest.fixture
def api(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'speech.db'}", connect_args={"check_same_thread": False}
    )
    factory = sessionmaker(bind=engine)
    database.Base.metadata.create_all(engine)
    monkeypatch.setattr(database, "_SessionLocal", factory)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(model_service, "_visible_user_ids_hook", None)
    monkeypatch.setattr(model_service, "_model_visibility_hook", None)
    with factory() as db:
        db.add_all(
            [
                User(id=1, username="owner", password_hash="unused"),
                User(id=2, username="other", password_hash="unused"),
                User(id=3, username="admin", password_hash="unused", is_admin=True),
            ]
        )
        for index, (model_id, name, ability) in enumerate(
            [
                ("live", "scribe_v2_realtime", "asr"),
                ("voice", "eleven_flash_v2_5", "tts"),
            ],
            start=1,
        ):
            db.add(
                Model(
                    id=index,
                    model_id=model_id,
                    model_name=name,
                    model_provider="elevenlabs",
                    api_key="synthetic-provider-secret",
                    category="speech",
                    abilities=[ability],
                    language="en",
                    voice="configured-voice" if ability == "tts" else None,
                    is_active=True,
                )
            )
            db.add(UserModel(user_id=1, model_id=index, is_owner=True))
        db.commit()
    asr, tts = FakeASR(), FakeTTS()
    asr_factory, tts_factory = Mock(return_value=asr), Mock(return_value=tts)
    monkeypatch.setattr(speech, "get_asr_model_instance", asr_factory)
    monkeypatch.setattr(speech, "get_tts_model_instance", tts_factory)
    usage = []
    monkeypatch.setattr(
        quota_hooks, "_standalone_usage_hook", lambda *args: usage.append(args)
    )
    app = FastAPI()
    app.include_router(speech.speech_router)
    token = build_access_token(sub="owner", user_id=1)
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            db=factory,
            asr=asr,
            tts=tts,
            asr_factory=asr_factory,
            tts_factory=tts_factory,
            headers={"Authorization": f"Bearer {token}"},
            protocols=["xagent-auth-v1", f"xagent-auth-token.{token}"],
            usage=usage,
        )
    engine.dispose()


@pytest.mark.parametrize("vad", [True, False])
def test_live_audio_partial_final_manual_commit_and_scope_owned_cleanup(api, vad):
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        assert ws.accepted_subprotocol == "xagent-auth-v1"
        ws.send_json(
            {"type": "start", "model_id": "live", "vad": vad, "language": "en"}
        )
        assert ws.receive_json()["type"] == "session_started"
        ws.send_bytes(b"\x00\x00" * 8000)
        assert ws.receive_json()["type"] == "partial"
        if not vad:
            ws.send_json({"type": "commit"})
        assert ws.receive_json()["text"] == "Hello."
        ws.send_json({"type": "close"})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    assert api.asr.sessions[0].closed
    assert api.asr.close_count == 1
    assert api.asr.sessions[0].commits == (0 if vad else 1)
    assert len(api.usage) == 1
    user_id, rows, source = api.usage[0]
    assert (user_id, source) == (1, "speech_transcribe")
    assert rows[0]["quantity"] == 0.5


def test_asr_disconnect_while_opening_cancels_the_provider(api):
    api.asr.block_open = True
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live"})
        assert api.asr.opening.wait(2)
        ws.close()
    assert api.asr.closed.wait(2)
    assert not api.usage


def test_revoked_session_closes_live_audio_and_delivers_usage(api, monkeypatch):
    monkeypatch.setattr(websocket_auth, "_SESSION_RECHECK_SECONDS", 0.01)
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 8000)
        ws.receive_json()
        ws.receive_json()
        with api.db() as db:
            db.get(User, 1).session_epoch = "revoked"
            db.commit()
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 4001
    assert api.asr.closed.wait(2)
    assert api.asr.sessions[0].closed
    assert len(api.usage) == 1


def test_revocation_before_start_never_opens_a_provider_session(api):
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        with api.db() as db:
            db.get(User, 1).session_epoch = "revoked"
            db.commit()
        ws.send_json({"type": "start", "model_id": "live"})
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 4001
    api.asr_factory.assert_not_called()


@pytest.mark.parametrize("query", ["", "?token=retired-secret"])
def test_asr_rejects_missing_or_query_credentials_before_model_access(api, query):
    protocols = api.protocols if query else []
    with pytest.raises(WebSocketDisconnect):
        with api.client.websocket_connect(ASR_PATH + query, subprotocols=protocols):
            pass
    api.asr_factory.assert_not_called()


@pytest.mark.parametrize("owner,shared,status", [(2, False, 404), (3, True, 200)])
def test_tts_uses_normal_private_and_shared_model_visibility(
    api, owner, shared, status
):
    with api.db() as db:
        link = db.query(UserModel).filter_by(model_id=2).one()
        link.user_id, link.is_shared = owner, shared
        db.commit()
    response = api.client.post(
        TTS_PATH, headers=api.headers, json={"model_id": "voice", "text": "Hello"}
    )
    assert response.status_code == status, response.text
    if status == 404:
        api.tts_factory.assert_not_called()


@pytest.mark.parametrize(
    "state,status", [("inactive", 404), ("hidden", 404), ("incompatible", 400)]
)
def test_tts_rejects_unavailable_models_before_provider_access(
    api, monkeypatch, state, status
):
    with api.db() as db:
        model = db.get(Model, 2)
        if state == "inactive":
            model.is_active = False
        elif state == "incompatible":
            model.abilities = ["asr"]
        db.commit()
    if state == "hidden":
        monkeypatch.setattr(
            model_service, "_model_visibility_hook", lambda *args: False
        )
    response = api.client.post(
        TTS_PATH, headers=api.headers, json={"model_id": "voice", "text": "Hello"}
    )
    assert response.status_code == status
    api.tts_factory.assert_not_called()


def test_asr_rejects_a_foreign_model_before_opening_the_provider(api):
    with api.db() as db:
        db.query(UserModel).filter_by(model_id=1).one().user_id = 2
        db.commit()
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live"})
        assert ws.receive_json()["code"] == "model_unavailable"
    api.asr_factory.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        b"not json",
        {"type": "commit"},
        {"type": "start", "model_id": "live", "sample_rate": 123},
    ],
)
def test_asr_validates_start_before_provider_access(api, message):
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        if isinstance(message, bytes):
            ws.send_bytes(message)
        else:
            ws.send_json(message)
        assert ws.receive_json()["code"] == "invalid_request"
    api.asr_factory.assert_not_called()


@pytest.mark.parametrize("stage", ["start", "control"])
def test_asr_frame_size_checks_precede_json_parsing(api, monkeypatch, stage):
    # Valid padded JSON ensures only the byte limit rejects this frame. The
    # parse spy also makes the test fail if the size guard is removed.
    if stage == "start":
        parse = Mock(wraps=speech.ASRStreamStart.model_validate_json)
        monkeypatch.setattr(speech.ASRStreamStart, "model_validate_json", parse)
    else:
        parse = Mock(wraps=speech.json.loads)
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        if stage == "control":
            ws.send_json({"type": "start", "model_id": "live"})
            assert ws.receive_json()["type"] == "session_started"
            monkeypatch.setattr(speech.json, "loads", parse)
        command = (
            {"type": "start", "model_id": "live"}
            if stage == "start"
            else {"type": "commit"}
        )
        oversized = json.dumps(command) + " " * 5000
        ws.send_text(oversized)
        error = ws.receive_json()
        assert error["code"] == "invalid_request"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1008
    # json.loads is shared with the test client, which decodes the error event.
    if stage == "start":
        parse.assert_not_called()
        api.asr_factory.assert_not_called()
    else:
        assert not any(call.args[0] == oversized for call in parse.call_args_list)
        assert not api.asr.sessions[0].commits
        assert api.asr.sessions[0].closed


@pytest.mark.parametrize("failure", ["factory", "open"])
def test_asr_configuration_errors_are_model_unavailable(api, failure, caplog):
    error = ValueError("api_key=synthetic-provider-secret")
    if failure == "factory":
        api.asr_factory.side_effect = error
    else:
        api.asr.open_stream = AsyncMock(side_effect=error)
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live"})
        assert ws.receive_json()["code"] == "model_unavailable"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1008
    assert "Speech stream failed" in caplog.text
    assert "synthetic-provider-secret" not in caplog.text
    assert api.asr.close_count == (failure == "open")


@pytest.mark.parametrize("failure", ["config", "factory"])
def test_asr_failure_before_model_creation_is_logged_safely(
    api, monkeypatch, caplog, failure
):
    error = RuntimeError("storage failed api_key=synthetic-provider-secret")
    if failure == "config":
        monkeypatch.setattr(speech, "_load_speech_config", Mock(side_effect=error))
    else:
        api.asr_factory.side_effect = error
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live"})
        assert ws.receive_json()["code"] == "stream_failed"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1011
    assert "storage failed" in caplog.text
    assert "synthetic-provider-secret" not in caplog.text
    assert api.asr.close_count == 0


@pytest.mark.parametrize("size", [64 * 1024 + 1, 32002])
def test_asr_rejects_oversized_audio_and_closes_the_session(api, monkeypatch, size):
    if size == 32002:
        monkeypatch.setattr(speech, "_SESSION_SECONDS", 1)
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live"})
        ws.receive_json()
        ws.send_bytes(b"\x00" * size)
        assert ws.receive_json()["code"] == "invalid_request"
    assert api.asr.sessions[0].closed
    assert not api.asr.sessions[0].audio


def test_asr_send_failure_reports_a_safe_error_and_closes_both_halves(api, caplog):
    api.asr.send_error = RuntimeError("synthetic-provider-secret")
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x00")
        error = ws.receive_json()
        assert error["code"] == "stream_failed"
        assert "synthetic-provider-secret" not in str(error) + caplog.text
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1011
    assert api.asr.sessions[0].closed
    assert api.asr.close_count == 1


@pytest.mark.parametrize(
    "message",
    [
        "not json",
        {"type": "unknown"},
        {"type": "start", "model_id": "live"},
        {"type": "commit"},
        json.dumps({"type": "commit", "pad": "x" * 5000}),
        b"\x00",
    ],
)
def test_asr_invalid_frames_close_with_1008_before_provider_input(api, message):
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live"})
        assert ws.receive_json()["type"] == "session_started"
        if isinstance(message, bytes):
            ws.send_bytes(message)
        elif isinstance(message, str):
            ws.send_text(message)
        else:
            ws.send_json(message)
        assert ws.receive_json()["code"] == "invalid_request"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1008
    assert api.asr.sessions[0].closed
    assert not api.asr.sessions[0].audio


@pytest.mark.parametrize("deadline", ["initial", "idle", "session", "provider"])
def test_asr_transport_timeouts_are_distinct_from_provider_failure(
    api, monkeypatch, caplog, deadline
):
    if deadline in {"initial", "idle"}:
        monkeypatch.setattr(speech, "_IO_TIMEOUT", 0.1)
    elif deadline == "session":
        monkeypatch.setattr(speech, "_SESSION_SECONDS", 0.1)
    else:
        api.asr.send_error = TimeoutError("synthetic-provider-secret")
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        if deadline != "initial":
            ws.send_json({"type": "start", "model_id": "live"})
            assert ws.receive_json()["type"] == "session_started"
        if deadline == "provider":
            ws.send_bytes(b"\x00\x00")
        assert ws.receive_json()["code"] == (
            "stream_failed" if deadline == "provider" else "session_timeout"
        )
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == (1011 if deadline == "provider" else 1000)
    if deadline == "initial":
        api.asr_factory.assert_not_called()
    else:
        assert api.asr.sessions[0].closed and api.asr.close_count == 1
    assert "synthetic-provider-secret" not in caplog.text
    if deadline != "provider":
        assert "Speech stream failed" not in caplog.text


def test_close_aborts_a_committed_segment_whose_final_has_not_arrived(api):
    committed = threading.Event()
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json({"type": "start", "model_id": "live", "vad": False})
        ws.receive_json()

        async def commit_without_final():
            committed.set()

        api.asr.sessions[0].commit = commit_without_final
        ws.send_json({"type": "commit"})
        assert committed.wait(2)
        ws.send_json({"type": "close"})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1000
    assert api.asr.sessions[0].closed and api.asr.close_count == 1


@pytest.mark.parametrize(
    "branch,disconnected",
    [
        ("invalid", OSError()),
        ("missing", RuntimeError()),
        ("provider", WebSocketDisconnect()),
    ],
)
async def test_asr_error_response_after_disconnect_does_not_escape(
    api, monkeypatch, branch, disconnected
):
    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "websocket.connect"})
    incoming.put_nowait(
        {
            "type": "websocket.receive",
            "text": json.dumps(
                {
                    "type": "start",
                    "model_id": "missing" if branch == "missing" else "live",
                }
            ),
        }
    )
    incoming.put_nowait(
        {"type": "websocket.receive", "text": "invalid"}
        if branch == "invalid"
        else {"type": "websocket.receive", "bytes": b"\x00\x00"}
    )
    api.asr.send_error = RuntimeError("provider failed")
    attempted = []

    async def send(message):
        if (
            message["type"] == "websocket.send"
            and json.loads(message["text"])["type"] == "error"
        ):
            attempted.append(message)
            raise disconnected

    monkeypatch.setattr(
        speech, "get_authenticated_user", AsyncMock(return_value=SimpleNamespace(id=1))
    )
    websocket = WebSocket(
        {"type": "websocket", "headers": [], "subprotocols": []}, incoming.get, send
    )
    await asyncio.wait_for(speech.stream_asr.__wrapped__(websocket), 2)
    assert len(attempted) == 1
    assert api.asr.close_count == (0 if branch == "missing" else 1)


@pytest.mark.parametrize(
    "blocked", ["transcript", "error", "error_close", "normal_close", "unauth_close"]
)
async def test_blocked_asr_terminal_writes_settle_handler_and_watcher(
    api, monkeypatch, blocked
):
    release, blocked_send = asyncio.Event(), asyncio.Event()
    messages = []
    monkeypatch.setattr(speech, "_IO_TIMEOUT", 0.03)
    monkeypatch.setattr(speech, "_TERMINAL_TIMEOUT", 0.03, raising=False)
    monkeypatch.setattr(
        speech,
        "get_authenticated_user",
        AsyncMock(
            return_value=None if blocked == "unauth_close" else SimpleNamespace(id=1)
        ),
    )
    incoming = asyncio.Queue()
    incoming.put_nowait({"type": "websocket.connect"})
    incoming.put_nowait(
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "start", "model_id": "live"}),
        }
    )
    incoming.put_nowait({"type": "websocket.receive", "bytes": b"\x00\x00"})
    if blocked != "transcript":
        incoming.put_nowait(
            {
                "type": "websocket.receive",
                "text": json.dumps({"type": "close"})
                if blocked == "normal_close"
                else "invalid",
            }
        )

    async def receive():
        if release.is_set():
            return {"type": "websocket.disconnect", "code": 1000}
        if not incoming.empty():
            return incoming.get_nowait()
        # The slow reader is still sending microphone audio, so only the
        # outbound transcript can hit its idle limit.
        await asyncio.sleep(0.005)
        return {"type": "websocket.receive", "bytes": b"\x00\x00"}

    async def send(message):
        messages.append(message)
        event = (
            json.loads(message["text"]) if message["type"] == "websocket.send" else {}
        )
        if (
            message["type"] == "websocket.close"
            or (blocked == "transcript" and event.get("type") in {"partial", "error"})
            or (blocked == "error" and event.get("type") == "error")
        ):
            blocked_send.set()
            await release.wait()

    websocket = WebSocket(
        {"type": "websocket", "headers": [], "query_string": b"", "subprotocols": []},
        receive,
        send,
    )
    task = asyncio.create_task(speech.stream_asr(websocket, token="test-token"))
    try:
        await asyncio.wait_for(blocked_send.wait(), 1)
        watchers = [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "websocket-session-monitor"
        ]
        assert len(watchers) == 1
        await asyncio.wait_for(asyncio.shield(task), 0.5)
    finally:
        # Let the unfixed handler finish even when the deadline assertion fails.
        release.set()
        await asyncio.wait_for(task, 2)
    assert watchers[0].done()
    closes = [message for message in messages if message["type"] == "websocket.close"]
    assert len(closes) == 1
    assert closes[0]["code"] == (
        4001
        if blocked == "unauth_close"
        else 1000
        if blocked in {"normal_close", "transcript"}
        else 1008
    )
    errors = [
        message
        for message in messages
        if message["type"] == "websocket.send"
        and json.loads(message["text"])["type"] == "error"
    ]
    assert len(errors) == (blocked in {"error", "error_close"})
    assert api.asr.close_count == (0 if blocked == "unauth_close" else 1)
    assert len(api.usage) == (0 if blocked == "unauth_close" else 1)
    if blocked != "unauth_close":
        assert api.asr.sessions[0].closed
        assert api.usage[0][0] == 1 and api.usage[0][2] == "speech_transcribe"


async def test_repeated_cancellation_drains_both_stream_halves():
    started = [asyncio.Event(), asyncio.Event()]
    closed = [asyncio.Event(), asyncio.Event()]
    closing, release = asyncio.Event(), asyncio.Event()
    children = []

    async def operation(index):
        children.append(asyncio.current_task())
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            if index == 0:
                closing.set()
                await release.wait()
            closed[index].set()

    task = asyncio.create_task(speech._race(operation(0), operation(1)))
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
        task.cancel()
        await asyncio.wait_for(closing.wait(), 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert all(event.is_set() for event in closed)
    finally:
        release.set()
        for child in children:
            if not child.done():
                child.cancel()
        await asyncio.gather(*children, return_exceptions=True)


def test_tts_stream_has_audio_metadata_and_exactly_one_usage_delivery(api):
    response = api.client.post(
        TTS_PATH,
        headers=api.headers,
        json={
            "model_id": "voice",
            "text": "Hello",
            "voice": "override",
            "format": "pcm",
            "sample_rate": 24000,
        },
    )
    assert response.status_code == 200, response.text
    assert response.content == b"firstsecond"
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.headers["cache-control"] == "no-store"
    assert "X-Audio-Sample-Rate" in response.headers["access-control-expose-headers"]
    assert api.tts.calls == [
        {"text": "Hello", "voice": "override", "format": "pcm", "sample_rate": 24000}
    ]
    assert api.tts_factory.call_args.args[0].voice == "configured-voice"
    assert api.tts.close_count == 1 and api.tts.stream_closed
    assert len(api.usage) == 1
    assert api.usage[0][0] == 1 and api.usage[0][2] == "speech_synthesize"
    assert api.usage[0][1][0]["quantity"] == 5


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("synthetic-provider-secret"), 502),
        (ValueError("bad format"), 400),
        (TimeoutError("timeout"), 504),
    ],
)
def test_tts_pre_audio_failure_is_an_http_error_without_secrets(
    api, caplog, error, status
):
    api.tts.error = error
    response = api.client.post(
        TTS_PATH, headers=api.headers, json={"model_id": "voice", "text": "Hello"}
    )
    assert response.status_code == status
    assert "synthetic-provider-secret" not in response.text + caplog.text
    assert api.tts.close_count == 1 and api.tts.stream_closed
    assert not api.usage


@pytest.mark.parametrize("kind", ["asr", "tts"])
def test_nonstreaming_adapters_are_rejected_and_closed(api, kind):
    model = getattr(api, kind)
    model.supports_streaming = False
    if kind == "asr":
        with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
            ws.send_json({"type": "start", "model_id": "live"})
            assert ws.receive_json()["code"] == "model_unavailable"
        assert not model.sessions
    else:
        response = api.client.post(
            TTS_PATH, headers=api.headers, json={"model_id": "voice", "text": "Hello"}
        )
        assert response.status_code == 400
        assert not model.calls
    assert model.close_count == 1


def test_tts_requires_login_and_validates_text_before_provider_access(api):
    assert api.client.post(
        TTS_PATH, json={"model_id": "voice", "text": "Hello"}
    ).status_code in {401, 403}
    for text in ("", "x" * 5001):
        response = api.client.post(
            TTS_PATH, headers=api.headers, json={"model_id": "voice", "text": text}
        )
        assert response.status_code == 422
    api.tts_factory.assert_not_called()


@pytest.mark.parametrize(
    "voice", ["../../v1/user?x=", "voice/other", "voice?x=", "%2e%2e", "x" * 129]
)
def test_shared_model_voice_cannot_redirect_owner_credentials(api, voice):
    with api.db() as db:
        link = db.query(UserModel).filter_by(model_id=2).one()
        link.user_id, link.is_shared = 2, True
        db.commit()
    response = api.client.post(
        TTS_PATH,
        headers=api.headers,
        json={"model_id": "voice", "text": "Hello", "voice": voice},
    )
    assert response.status_code == 422
    api.tts_factory.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"format": "foo"},
        {"format": "pcm_12345"},
        {"format": "pcm", "sample_rate": 12345},
    ],
)
def test_invalid_tts_output_options_are_local_400_errors(api, options):
    model = ElevenLabsTTS(api_key="synthetic-provider-secret")
    model._ensure_async_client = Mock()
    api.tts_factory.return_value = model
    response = api.client.post(
        TTS_PATH,
        headers=api.headers,
        json={"model_id": "voice", "text": "Hello", **options},
    )
    assert response.status_code == 400
    model._ensure_async_client.assert_not_called()


def test_tts_first_audio_timeout_cancels_the_provider_and_closes_its_client(
    api, monkeypatch
):
    monkeypatch.setattr(speech, "_IO_TIMEOUT", 0.01)
    api.tts.block_first = True
    api.tts.gate = asyncio.Event()
    response = api.client.post(
        TTS_PATH, headers=api.headers, json={"model_id": "voice", "text": "Hello"}
    )
    assert response.status_code == 504
    assert api.tts.stream_closed and api.tts.close_count == 1
    assert not api.usage


@pytest.mark.parametrize("before_first", [False, True])
async def test_tts_sends_first_audio_before_completion_and_cancels_on_disconnect(
    api, before_first
):
    model = api.tts
    model.gate = asyncio.Event()
    model.block_first = before_first
    response = speech._TTSAudioResponse(
        model, 1, TTSStreamRequest(model_id="voice", text="Hello")
    )
    incoming = asyncio.Queue()
    first_sent = asyncio.Event()
    messages = []

    async def send(message):
        messages.append(message)
        if message.get("body") == b"first":
            first_sent.set()

    task = asyncio.create_task(response({"type": "http"}, incoming.get, send))
    if before_first:
        while not model.calls:
            await asyncio.sleep(0)
        assert not messages
    else:
        await asyncio.wait_for(first_sent.wait(), 2)
        assert not task.done()
        assert [item.get("body") for item in messages if "body" in item] == [b"first"]
    await incoming.put({"type": "http.disconnect"})
    await asyncio.wait_for(task, 2)
    assert model.stream_closed and model.close_count == 1
    assert len(api.usage) == (0 if before_first else 1)


async def test_disconnect_during_tts_client_close_waits_for_cleanup_before_metering(
    api,
):
    closing, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def close():
        closing.set()
        await release.wait()
        closed.set()

    api.tts.aclose = close
    response = speech._TTSAudioResponse(
        api.tts, 1, TTSStreamRequest(model_id="voice", text="Hello")
    )
    incoming = asyncio.Queue()

    async def send(_message):
        pass

    task = asyncio.create_task(response({"type": "http"}, incoming.get, send))
    await asyncio.wait_for(closing.wait(), 2)
    await incoming.put({"type": "http.disconnect"})
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.01)
        assert not api.usage
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    assert closed.is_set()
    assert len(api.usage) == 1


async def test_disconnect_after_audio_waits_for_usage_delivery(api, monkeypatch):
    entered, release, delivered = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def delayed_usage(*row):
        entered.set()
        assert release.wait(5)
        api.usage.append(row)
        delivered.set()

    monkeypatch.setattr(quota_hooks, "_standalone_usage_hook", delayed_usage)
    response = speech._TTSAudioResponse(
        api.tts, 1, TTSStreamRequest(model_id="voice", text="Hello")
    )
    incoming, messages = asyncio.Queue(), []

    async def send(message):
        messages.append(message)

    task = asyncio.create_task(response({"type": "http"}, incoming.get, send))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert messages[-1] == {
            "type": "http.response.body",
            "body": b"",
            "more_body": False,
        }
        assert api.tts.close_count == 1
        await incoming.put({"type": "http.disconnect"})
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.02)
        assert not api.usage
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
        assert await asyncio.to_thread(delivered.wait, 2)
    assert len(api.usage) == 1
    assert api.usage[0][0] == 1
    assert api.usage[0][2] == "speech_synthesize"


@pytest.mark.parametrize("failure", ["provider", "chunk_timeout", "session_timeout"])
async def test_tts_late_failure_keeps_truncation_without_leaking_provider_details(
    api, monkeypatch, caplog, failure
):
    if failure == "provider":
        api.tts.error_after_first = RuntimeError(
            "provider exposed synthetic-provider-secret"
        )
    else:
        api.tts.gate = asyncio.Event()
        monkeypatch.setattr(
            speech,
            "_IO_TIMEOUT" if failure == "chunk_timeout" else "_SESSION_SECONDS",
            0.02,
        )
    response = speech._TTSAudioResponse(
        api.tts, 1, TTSStreamRequest(model_id="voice", text="Hello")
    )
    messages, incoming = [], asyncio.Queue()

    async def send(message):
        messages.append(message)

    with pytest.raises(
        RuntimeError, match="Speech synthesis stream interrupted"
    ) as error:
        await asyncio.wait_for(response({"type": "http"}, incoming.get, send), 2)
    assert messages[0]["status"] == 200
    assert any(message.get("body") == b"first" for message in messages)
    assert not any(message.get("more_body") is False for message in messages)
    formatted = "".join(traceback.format_exception(error.value))
    assert "synthetic-provider-secret" not in formatted + caplog.text
    assert api.tts.stream_closed and api.tts.close_count == 1
    assert len(api.usage) == 1


async def test_tts_session_timeout_before_audio_is_a_504(api, monkeypatch):
    api.tts.gate = asyncio.Event()
    api.tts.block_first = True
    monkeypatch.setattr(speech, "_SESSION_SECONDS", 0.02)
    response = speech._TTSAudioResponse(
        api.tts, 1, TTSStreamRequest(model_id="voice", text="Hello")
    )
    messages, incoming = [], asyncio.Queue()

    async def send(message):
        messages.append(message)

    await asyncio.wait_for(response({"type": "http"}, incoming.get, send), 2)
    assert messages[0]["status"] == 504
    assert api.tts.stream_closed and api.tts.close_count == 1
    assert not api.usage


@pytest.mark.parametrize(
    "failure", ["provider", "chunk_timeout", "session_timeout", None]
)
async def test_tts_completion_through_full_app_stack(api, monkeypatch, caplog, failure):
    from xagent.web.app import app

    if failure == "provider":
        api.tts.error_after_first = RuntimeError("synthetic-provider-secret")
    elif failure:
        api.tts.gate = asyncio.Event()
        monkeypatch.setattr(
            speech,
            "_IO_TIMEOUT" if failure == "chunk_timeout" else "_SESSION_SECONDS",
            0.05,
        )
    body = json.dumps({"model_id": "voice", "text": "Hello"}).encode()
    incoming, messages = asyncio.Queue(), []
    incoming.put_nowait({"type": "http.request", "body": body, "more_body": False})

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": TTS_PATH,
        "query_string": b"",
        "headers": [
            (b"authorization", api.headers["Authorization"].encode()),
            (b"content-type", b"application/json"),
        ],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1234),
    }
    if failure:
        with pytest.raises(
            RuntimeError, match="Speech synthesis stream interrupted"
        ) as error:
            await asyncio.wait_for(app(scope, incoming.get, send), 2)
        assert (
            "synthetic-provider-secret"
            not in "".join(traceback.format_exception(error.value)) + caplog.text
        )
    else:
        await asyncio.wait_for(app(scope, incoming.get, send), 2)
    assert messages[0]["status"] == 200
    headers = dict(messages[0]["headers"])
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert headers[b"x-frame-options"] == b"DENY"
    assert any(message.get("body") == b"first" for message in messages)
    assert any(message.get("more_body") is False for message in messages) is (
        failure is None
    )
    assert api.tts.stream_closed and api.tts.close_count == 1
    assert len(api.usage) == 1
