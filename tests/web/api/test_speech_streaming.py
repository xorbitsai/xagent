"""Streaming transport contracts, with real auth/visibility and fake providers."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.websockets import WebSocketDisconnect

from tests.web.auth_token_cases import build_access_token
from xagent.core.model.asr.base import ASRStreamEvent, ASRStreamSession
from xagent.core.model.chat.token_context import MediaCallType
from xagent.core.model.media_metering import record_provider_usage
from xagent.core.model.tts.base import TTSChunk
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
            assert ws.receive_json()["code"] == "invalid_request"
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
