"""The login ASR route's failure responses, as served by the shared runner."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from starlette.websockets import WebSocketDisconnect, WebSocketState

from tests.web.api.test_speech_streaming import (  # noqa: F401
    ASR_PATH,
    FakeASR,
    api,
)
from xagent.core.model.chat.token_context import MediaCallType
from xagent.core.model.media_metering import record_provider_usage
from xagent.web.api import speech, websocket_auth
from xagent.web.schemas.speech import TTSStreamRequest
from xagent.web.services import quota_hooks, speech_streams
from xagent.web.services.release_hold import release_hold

START = {"type": "start", "model_id": "live"}


def _serve(api, start):
    """Return the error frame (or None) and the close of one login session."""
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        if start is not None:
            ws.send_json(start)
        frame = None
        with pytest.raises(WebSocketDisconnect) as closed:
            frame = ws.receive_json()
            ws.receive_json()
    return frame, (closed.value.code, closed.value.reason)


@pytest.mark.parametrize("case", ["no_start_frame", "missing_model"])
def test_login_asr_failure_responses_unchanged(api, monkeypatch, case):
    if case == "no_start_frame":
        monkeypatch.setattr(speech_streams, "_IO_TIMEOUT", 0.1)
        start, message, code = None, "Speech session timed out", "session_timeout"
        close = (1000, "Speech session timed out")
    else:
        start = {"type": "start", "model_id": "missing"}
        message, code = "Model not found", "model_unavailable"
        close = (1008, "Model not found")
    frame, closed = _serve(api, start)
    assert frame == {"type": "error", "code": code, "message": message}
    assert closed == close
    api.asr_factory.assert_not_called()
    assert not api.usage


class _FencedSocket:
    """Records what the session writes while it waits for the first frame."""

    def __init__(self) -> None:
        self.application_state = WebSocketState.CONNECTING
        self.sent: list[object] = []
        self.closes: list[tuple[int, str]] = []
        self.waiting = asyncio.Event()

    async def receive(self):
        self.waiting.set()
        await asyncio.Event().wait()

    async def send_json(self, data) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closes.append((code, reason))
        self.application_state = WebSocketState.DISCONNECTED


async def test_login_asr_fence_while_waiting_for_start_closes_1012(monkeypatch):
    socket = _FencedSocket()

    async def authenticated(*_args, **_kwargs):
        return SimpleNamespace(id=1)

    async def accept(websocket) -> None:
        websocket.application_state = WebSocketState.CONNECTED

    monkeypatch.setattr(speech, "get_authenticated_user", authenticated)
    monkeypatch.setattr(speech, "accept_authenticated_websocket", accept)
    handler = asyncio.create_task(speech.stream_asr.__wrapped__(socket, token=None))
    try:
        await asyncio.wait_for(socket.waiting.wait(), 2)
        release_hold.mark_writers_retired()
        await asyncio.wait_for(handler, 2)
        assert not handler.cancelled()
        assert socket.sent == []
        assert socket.closes == [(1012, "Service restart")]
        assert await release_hold.drain_socket_writes(1.0)
    finally:
        if not handler.done():
            handler.cancel()
        release_hold.reset_writers_retired()


@pytest.mark.parametrize(
    "raised", ["validation", "value", "http", "authentication_terminated"]
)
def test_login_asr_prepare_failures_keep_their_responses(api, monkeypatch, raised):
    start = START
    if raised == "validation":
        start = {"type": "start", "model_id": "live", "sample_rate": 123}
    elif raised == "value":
        monkeypatch.setattr(
            speech, "_load_speech_config", Mock(side_effect=ValueError("bad"))
        )
    elif raised == "http":
        monkeypatch.setattr(
            speech,
            "_load_speech_config",
            Mock(side_effect=HTTPException(400, "Incompatible speech model")),
        )
    else:
        authenticate = speech.get_authenticated_user

        async def terminated_on_revalidate(websocket, token=None, *, revalidate=False):
            if revalidate:
                raise websocket_auth._WebSocketAuthenticationTerminated()
            return await authenticate(websocket, token)

        monkeypatch.setattr(speech, "get_authenticated_user", terminated_on_revalidate)
    frame, closed = _serve(api, start)
    expected = {
        "validation": (
            {
                "type": "error",
                "code": "invalid_request",
                "message": "Invalid speech stream request",
            },
            (1008, "Invalid speech stream request"),
        ),
        "value": (
            {
                "type": "error",
                "code": "invalid_request",
                "message": "Invalid speech stream request",
            },
            (1008, "Invalid speech stream request"),
        ),
        "http": (
            {
                "type": "error",
                "code": "model_unavailable",
                "message": "Incompatible speech model",
            },
            (1008, "Incompatible speech model"),
        ),
        # Authentication already answered the client: no frame of our own.
        "authentication_terminated": (None, (1000, "")),
    }[raised]
    assert (frame, closed) == expected
    # The release-hold write token is returned on every one of these exits.
    assert release_hold.socket_write_count == 0
    api.asr_factory.assert_not_called()
    assert not api.usage


class _ClosingMeteredASR(FakeASR):
    """Records its ASR duration when the model is closed, as providers do."""

    async def aclose(self):
        await super().aclose()
        record_provider_usage(self, MediaCallType.ASR, 7.5)


def test_asr_model_is_closed_inside_the_usage_scope(api):
    model = _ClosingMeteredASR()
    api.asr_factory.return_value = model
    with api.client.websocket_connect(ASR_PATH, subprotocols=api.protocols) as ws:
        ws.send_json(START)
        assert ws.receive_json()["type"] == "session_started"
        ws.send_json({"type": "close"})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1000
    assert model.close_count == 1
    assert len(api.usage) == 1
    user_id, details, source = api.usage[0]
    assert (user_id, source) == (1, "speech_transcribe")
    assert [row["quantity"] for row in details] == [7.5]


async def test_tts_gate_and_metering_share_the_plan_usage_source(api, monkeypatch):
    gated = []

    def gate(user_id, source):
        gated.append((user_id, source))

    monkeypatch.setattr(quota_hooks, "_standalone_gate_hook", gate)
    plan = speech_streams.SpeechStreamPlan(
        user_id=1,
        config=SimpleNamespace(model_id="voice"),
        options=TTSStreamRequest(model_id="voice", text="Hello"),
        usage_source="telegram_voice",
    )
    response = await speech_streams.tts_stream_response(plan=plan)
    incoming, messages = asyncio.Queue(), []

    async def send(message):
        messages.append(message)

    await asyncio.wait_for(response({"type": "http"}, incoming.get, send), 2)
    assert messages[0]["status"] == 200
    assert gated and all(source == "telegram_voice" for _, source in gated)
    assert [(row[0], row[2]) for row in api.usage] == [(1, "telegram_voice")]
