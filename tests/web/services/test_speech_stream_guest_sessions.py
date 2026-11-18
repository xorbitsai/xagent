"""Guest transcription sessions in the shared runner: slots and settlement."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketState

from tests.shared.widget_voice_env import clear_widget_voice_env
from tests.web.api.test_speech_streaming import api  # noqa: F401
from xagent import config
from xagent.web.schemas.speech import ASRStreamStart
from xagent.web.services import speech_streams, widget_voice_budget
from xagent.web.services.release_hold import release_hold
from xagent.web.services.speech_streams import SpeechStreamPlan, run_asr_stream


@pytest.fixture(autouse=True)
def _widget_voice_env(monkeypatch):
    clear_widget_voice_env(monkeypatch)


def _text(payload: dict) -> dict:
    return {"type": "websocket.receive", "text": json.dumps(payload)}


class _Socket:
    """An accepted socket driven from the test through its inbound queue."""

    def __init__(self) -> None:
        self.application_state = WebSocketState.CONNECTED
        self.incoming: asyncio.Queue[dict] = asyncio.Queue()
        self.incoming.put_nowait(_text({"type": "start"}))
        self.sent: list[dict] = []
        self.closes: list[tuple[int, str]] = []

    async def receive(self) -> dict:
        return await self.incoming.get()

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closes.append((code, reason))
        self.application_state = WebSocketState.DISCONNECTED

    @property
    def started(self) -> bool:
        return any(frame.get("type") == "session_started" for frame in self.sent)


def _prepare(entity_key: str, visitor_key: str):
    async def prepare(_text: str) -> SpeechStreamPlan[ASRStreamStart]:
        return SpeechStreamPlan(
            user_id=1,
            config=SimpleNamespace(model_id="live"),
            options=ASRStreamStart(type="start", model_id="live"),
            usage_source="speech_transcribe",
            audience="guest",
            entity_key=entity_key,
            visitor_key=visitor_key,
        )

    return prepare


@pytest.fixture
def guest(api, monkeypatch):
    monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    monkeypatch.setattr(
        widget_voice_budget, "_memory_store", widget_voice_budget._MemoryStore()
    )
    clock = [0.0]
    monkeypatch.setattr(speech_streams, "_now", lambda: clock[0])
    # A session the code under test fails to end must not wait the full
    # first-frame and audio timeout before the test fails.
    monkeypatch.setattr(speech_streams, "_IO_TIMEOUT", 2.0)
    yield SimpleNamespace(api=api, clock=clock)
    assert speech_streams._guest_sessions.total == 0
    assert speech_streams._guest_sessions.by_entity == {}
    assert speech_streams._guest_sessions.by_visitor == {}
    assert release_hold.socket_write_count == 0


def _asr_counts() -> list[int]:
    return sorted(widget_voice_budget._memory_store._counts.values())


async def _until(condition, seconds: float = 2.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


_CAP_VARIABLES = (
    "XAGENT_WIDGET_VOICE_MAX_SESSIONS",
    "XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_ENTITY",
    "XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR",
)


def _unset_caps(monkeypatch) -> None:
    for name in _CAP_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _open(entity_key: str, visitor_key: str) -> tuple[_Socket, asyncio.Task]:
    socket = _Socket()
    session = asyncio.create_task(
        run_asr_stream(
            socket, prepare=_prepare(entity_key, visitor_key), audience="guest"
        )
    )
    return socket, session


async def _settled(socket: _Socket) -> None:
    await _until(lambda: socket.started or bool(socket.closes))


async def _end(socket: _Socket, session: asyncio.Task) -> None:
    socket.incoming.put_nowait(_text({"type": "close"}))
    await asyncio.wait_for(session, 5)


def _slots() -> tuple[int, dict[str, int], dict[str, int]]:
    sessions = speech_streams._guest_sessions
    return sessions.total, dict(sessions.by_entity), dict(sessions.by_visitor)


@pytest.mark.parametrize("cap", ["global", "per_entity", "per_visitor"])
async def test_widget_voice_concurrency_admission_is_atomic(guest, monkeypatch, cap):
    limit = 2
    _unset_caps(monkeypatch)
    if cap == "global":
        monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS", str(limit))
        callers = [(f"agent:{index}", f"v{index}") for index in range(limit + 5)]
    elif cap == "per_entity":
        monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_ENTITY", str(limit))
        callers = [("agent:1", f"v{index}") for index in range(limit + 5)]
    else:
        monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR", str(limit))
        callers = [("agent:1", "v")] * (limit + 5)
    peak = [0]
    real_admit = speech_streams._guest_sessions.admit

    def admit(entity_key: str, visitor_key: str) -> bool:
        admitted = real_admit(entity_key, visitor_key)
        peak[0] = max(peak[0], speech_streams._guest_sessions.total)
        return admitted

    monkeypatch.setattr(speech_streams._guest_sessions, "admit", admit)
    sockets = [_Socket() for _ in callers]
    sessions = [
        asyncio.create_task(
            run_asr_stream(socket, prepare=_prepare(entity, visitor), audience="guest")
        )
        for socket, (entity, visitor) in zip(sockets, callers)
    ]
    await _until(lambda: all(s.started or s.closes for s in sockets))
    started = [socket for socket in sockets if socket.started]
    busy = [socket for socket in sockets if socket.closes]
    assert len(started) == limit and peak[0] == limit
    assert len(busy) == 5
    assert all(socket.closes == [(1008, "Voice is busy")] for socket in busy)
    for socket in started:
        socket.incoming.put_nowait(_text({"type": "close"}))
    await asyncio.wait_for(asyncio.gather(*sessions), 5)


async def test_visitor_cap_refuses_one_visitor_and_serves_another(guest, monkeypatch):
    _unset_caps(monkeypatch)
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR", "2")
    held = [_open("agent:1", "v") for _ in range(2)]
    for socket, _ in held:
        await _settled(socket)
    assert all(socket.started for socket, _ in held)
    full = _slots()
    assert full == (2, {"agent:1": 2}, {"agent:1|v": 2})

    refused, refused_session = _open("agent:1", "v")
    await asyncio.wait_for(refused_session, 5)
    assert refused.sent == [
        {"type": "error", "code": "voice_busy", "message": "Voice is busy"}
    ]
    assert refused.closes == [(1008, "Voice is busy")]
    # The refused session took no slot and reserved no budget.
    assert _slots() == full
    assert _asr_counts() == [1200, 1200, 1200]

    other = _open("agent:1", "w")
    await _settled(other[0])
    assert other[0].started
    assert _slots() == (3, {"agent:1": 3}, {"agent:1|v": 2, "agent:1|w": 1})

    await _end(*held.pop())
    again = _open("agent:1", "v")
    await _settled(again[0])
    assert again[0].started

    for socket, session in [*held, other, again]:
        await _end(socket, session)
    assert _slots() == (0, {}, {})


@pytest.mark.parametrize("visitors", ["one_visitor", "many_visitors"])
async def test_sessions_are_uncapped_by_default(guest, monkeypatch, visitors):
    _unset_caps(monkeypatch)
    monkeypatch.delenv("XAGENT_WIDGET_VOICE_ASR_VISITOR_DAILY_SECONDS", raising=False)
    held = [
        _open("agent:1", "v" if visitors == "one_visitor" else f"v{index}")
        for index in range(6)
    ]
    for socket, _ in held:
        await _settled(socket)
    assert all(socket.started for socket, _ in held)
    assert speech_streams._guest_sessions.by_entity == {"agent:1": 6}
    # Each session reserved 600 seconds of the entity's transcription budget.
    assert max(_asr_counts()) == 3600
    for socket, session in held:
        await _end(socket, session)


async def test_explicit_entity_cap_spans_its_visitors(guest, monkeypatch):
    _unset_caps(monkeypatch)
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_ENTITY", "2")
    held = [_open("agent:1", "v0"), _open("agent:1", "v1")]
    for socket, _ in held:
        await _settled(socket)
    full = _slots()
    assert full == (2, {"agent:1": 2}, {"agent:1|v0": 1, "agent:1|v1": 1})

    refused, refused_session = _open("agent:1", "v2")
    await asyncio.wait_for(refused_session, 5)
    assert refused.closes == [(1008, "Voice is busy")]
    assert _slots() == full

    other = _open("agent:2", "v2")
    await _settled(other[0])
    assert other[0].started
    for socket, session in [*held, other]:
        await _end(socket, session)


async def test_cancelled_guest_session_settles_its_budget(guest):
    socket = _Socket()
    session = asyncio.create_task(
        run_asr_stream(socket, prepare=_prepare("agent:1", "v"), audience="guest")
    )
    await _until(lambda: socket.started)
    assert _asr_counts() == [600, 600, 600]
    guest.clock[0] = 9.0
    session.cancel()
    with pytest.raises(asyncio.CancelledError):
        await session
    assert _asr_counts() == [9, 9, 9]
    assert guest.api.asr.close_count == 1


def _member_plan():
    async def prepare(_text: str) -> SpeechStreamPlan[ASRStreamStart]:
        return SpeechStreamPlan(
            user_id=1,
            config=SimpleNamespace(model_id="live"),
            options=ASRStreamStart(type="start", model_id="live"),
            usage_source="speech_transcribe",
            audience="member",
        )

    return prepare


@pytest.mark.parametrize(
    ("declared", "plan"),
    [("member", "guest"), ("guest", "member")],
    ids=["member_route_guest_plan", "guest_route_member_plan"],
)
async def test_session_refuses_a_plan_for_another_audience(
    guest, caplog, declared, plan
):
    caplog.set_level(logging.WARNING, logger=speech_streams.__name__)
    socket = _Socket()
    prepare = _prepare("agent:1", "v") if plan == "guest" else _member_plan()
    await asyncio.wait_for(
        run_asr_stream(socket, prepare=prepare, audience=declared), 5
    )
    assert socket.sent == [
        {"type": "error", "code": "stream_failed", "message": "Speech stream failed"}
    ]
    assert socket.closes[0] == (1011, "Speech stream failed")
    assert "another audience" in caplog.text
    assert _asr_counts() == []
    assert speech_streams._guest_sessions.total == 0
    assert guest.api.asr.sessions == [] and not guest.api.asr.opening.is_set()
    assert guest.api.usage == []


def test_run_asr_stream_requires_the_audience():
    parameter = inspect.signature(run_asr_stream).parameters["audience"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


async def test_guest_session_wound_down_by_the_release_fence(guest):
    socket = _Socket()
    session = asyncio.create_task(
        run_asr_stream(socket, prepare=_prepare("agent:1", "v"), audience="guest")
    )
    try:
        await _until(lambda: socket.started)
        assert speech_streams._guest_sessions.total == 1
        guest.clock[0] = 5.0
        release_hold.mark_writers_retired()
        await asyncio.wait_for(session, 5)
        assert not session.cancelled()
        assert socket.closes == [(1012, "Service restart")]
        assert speech_streams._guest_sessions.total == 0
        assert _asr_counts() == [5, 5, 5]
        assert await release_hold.drain_socket_writes(1.0)
    finally:
        if not session.done():
            session.cancel()
        release_hold.reset_writers_retired()


async def test_recheck_needs_both_arguments(guest):
    async def revalidate() -> int | None:
        return None

    with pytest.raises(TypeError):
        await run_asr_stream(
            _Socket(),
            prepare=_prepare("agent:1", "v"),
            audience="guest",
            revalidate=revalidate,
        )


@pytest.mark.parametrize("raw", ["", "  "])
def test_blank_session_caps_mean_no_cap(monkeypatch, raw):
    for name in _CAP_VARIABLES:
        monkeypatch.setenv(name, raw)
    sessions = speech_streams._GuestSessions()
    assert all(sessions.admit("agent:1", "v") for _ in range(60))
    assert all(sessions.admit("agent:1", f"v{index}") for index in range(60))
    assert all(sessions.admit(f"agent:{index}", "v") for index in range(2, 60))


def test_visitor_cap_counts_one_visitor_of_one_entity(monkeypatch):
    _unset_caps(monkeypatch)
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR", "2")
    sessions = speech_streams._GuestSessions()
    assert sessions.admit("agent:1", "v") and sessions.admit("agent:1", "v")
    assert not sessions.admit("agent:1", "v")
    assert sessions.admit("agent:1", "w") and sessions.admit("agent:2", "v")


def test_zero_visitor_cap_means_no_cap(monkeypatch):
    _unset_caps(monkeypatch)
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR", "0")
    sessions = speech_streams._GuestSessions()
    assert all(sessions.admit("agent:1", "v") for _ in range(60))


@pytest.mark.parametrize("raw", ["-1", "many"])
def test_invalid_visitor_cap_falls_back_to_no_cap_with_a_warning(
    monkeypatch, caplog, raw
):
    caplog.set_level(logging.WARNING, logger=config.__name__)
    _unset_caps(monkeypatch)
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR", raw)
    sessions = speech_streams._GuestSessions()
    assert all(sessions.admit("agent:1", "v") for _ in range(60))
    assert (
        f"Invalid XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR={raw!r}" in caplog.text
    )


def test_released_slots_leave_no_keys(monkeypatch):
    _unset_caps(monkeypatch)
    sessions = speech_streams._GuestSessions()
    assert sessions.admit("agent:1", "v") and sessions.admit("agent:1", "v")
    sessions.release("agent:1", "v")
    assert sessions.by_visitor == {"agent:1|v": 1}
    sessions.release("agent:1", "v")
    assert (sessions.total, sessions.by_entity, sessions.by_visitor) == (0, {}, {})


def test_session_caps_ignore_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS", " 2 ")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_ENTITY", " 0 ")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_MAX_SESSIONS_PER_VISITOR", " 0 ")
    sessions = speech_streams._GuestSessions()
    assert sessions.admit("agent:1", "v") and sessions.admit("agent:1", "v")
    assert not sessions.admit("agent:2", "v")


@pytest.mark.parametrize("close_code", [4003, 4408])
async def test_refused_recheck_closes_with_its_code_and_no_error_frame(
    guest, close_code
):
    socket = _Socket()

    async def revalidate() -> int | None:
        return close_code

    await asyncio.wait_for(
        run_asr_stream(
            socket,
            prepare=_prepare("agent:1", "v"),
            audience="guest",
            revalidate=revalidate,
            recheck_seconds=0.01,
        ),
        5,
    )
    # The close code is the only signal: no error frame precedes it.
    assert [frame for frame in socket.sent if frame.get("type") == "error"] == []
    assert socket.closes[0] == (close_code, "Voice is unavailable")


async def test_failing_recheck_ends_with_stream_failed(guest):
    socket = _Socket()

    async def revalidate() -> int | None:
        raise RuntimeError("database down")

    await asyncio.wait_for(
        run_asr_stream(
            socket,
            prepare=_prepare("agent:1", "v"),
            audience="guest",
            revalidate=revalidate,
            recheck_seconds=0.01,
        ),
        5,
    )
    assert socket.sent[-1] == {
        "type": "error",
        "code": "stream_failed",
        "message": "Speech stream failed",
    }
    assert socket.closes[0] == (1011, "Speech stream failed")
