"""Widget guest speech synthesis: POST /api/widget/voice/tts."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from sqlalchemy import select

from tests.shared.widget_voice_env import clear_widget_voice_env
from tests.web.api.test_speech_streaming import FakeASR, FakeTTS
from tests.web.services.test_widget_voice_models import _Discovered
from xagent.web import jwt_codec as jwt
from xagent.web.api import widget as widget_api
from xagent.web.api.public_chat_access import create_public_chat_access_token
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import get_session_local
from xagent.web.models.deployment import Deployment
from xagent.web.models.model import Model
from xagent.web.models.user import User, UserModel
from xagent.web.schemas.speech import TTSStreamRequest
from xagent.web.services import (
    model_service,
    quota_hooks,
    speech_streams,
    widget_voice,
    widget_voice_budget,
)
from xagent.web.services.speech_streams import SpeechStreamPlan, guest_refusal_detail
from xagent.web.services.widget_voice_budget import TTS_BUCKETS

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _install_one_slot_queue_pool,
    client,
)
from .test_workforce_widget import (
    _authenticate_widget_guest_by_key,
    _create_workforce,
    _enable_widget,
)

pytestmark = pytest.mark.usefixtures("_test_db")

PATH = "/api/widget/voice/tts"
SECRET = "synthetic-provider-secret"
LIMITS = {
    "tts-visitor": "XAGENT_WIDGET_VOICE_TTS_VISITOR_DAILY_CHARS",
    "tts-entity": "XAGENT_WIDGET_VOICE_TTS_ENTITY_DAILY_CHARS",
    "tts-site": "XAGENT_WIDGET_VOICE_TTS_SITE_DAILY_CHARS",
}


def _refusal(response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.json() == {"detail": guest_refusal_detail(code)}


def _set_switch(model: type, owner_id: int, column: str = "owner_id") -> None:
    db = _direct_db_session()
    try:
        row = db.query(model).filter(getattr(model, column) == owner_id).one()
        row.widget_voice_enabled = True
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _widget_voice_env(monkeypatch):
    clear_widget_voice_env(monkeypatch)


@pytest.fixture
def voice(monkeypatch):
    admin = _admin_headers()
    db = _direct_db_session()
    try:
        owner_id = int(db.query(User).filter(User.username == "admin").one().id)
        for model_id, ability in (("live", "asr"), ("voice", "tts")):
            row = Model(
                model_id=model_id,
                model_name=model_id,
                model_provider="elevenlabs",
                api_key=SECRET,
                category="speech",
                abilities=[ability],
                is_active=True,
            )
            db.add(row)
            db.flush()
            db.add(UserModel(user_id=owner_id, model_id=row.id, is_owner=True))
        agent = Agent(
            user_id=owner_id,
            name="Voice widget",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
        )
        db.add(agent)
        db.commit()
        agent_id = int(agent.id)
    finally:
        db.close()
    enabled = client.put(
        f"/api/agents/{agent_id}",
        headers=admin,
        json={"widget_enabled": True, "allowed_domains": ["*"]},
    )
    assert enabled.status_code == 200, enabled.text
    db = _direct_db_session()
    try:
        key = str(db.get(Agent, agent_id).widget_key)
    finally:
        db.close()
    _set_switch(Agent, agent_id, "id")

    monkeypatch.setenv("XAGENT_WIDGET_VOICE_ASR_MODEL", "live")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", "voice")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_VOICE", "deployment-voice")
    monkeypatch.delenv("XAGENT_REDIS_URL", raising=False)
    monkeypatch.setattr(model_service, "_visible_user_ids_hook", None)
    monkeypatch.setattr(model_service, "_model_visibility_hook", None)
    monkeypatch.setattr(
        widget_voice_budget, "_memory_store", widget_voice_budget._MemoryStore()
    )
    tts = FakeTTS()
    tts_factory = Mock(return_value=tts)
    monkeypatch.setattr(
        widget_voice, "get_asr_model_instance", Mock(return_value=FakeASR())
    )
    monkeypatch.setattr(
        widget_voice, "get_tts_model_instance", Mock(return_value=FakeTTS())
    )
    monkeypatch.setattr(speech_streams, "get_tts_model_instance", tts_factory)
    usage: list[tuple[Any, ...]] = []
    gated: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        quota_hooks, "_standalone_usage_hook", lambda *a: usage.append(a)
    )

    def gate_hook(user_id, source):
        gated.append((user_id, source))

    monkeypatch.setattr(quota_hooks, "_standalone_gate_hook", gate_hook)
    gate_models: list[tuple[str, ...]] = []
    real_gate = speech_streams.check_standalone_gate_async

    async def gate(user_id, source, *, model_ids):
        gate_models.append(tuple(model_ids))
        await real_gate(user_id, source, model_ids=model_ids)

    monkeypatch.setattr(speech_streams, "check_standalone_gate_async", gate)
    resolve = Mock(wraps=widget_voice.resolve_widget_voice_models)
    monkeypatch.setattr(widget_api, "resolve_widget_voice_models", resolve)
    reserve = Mock(wraps=widget_voice_budget.reserve_tts)
    monkeypatch.setattr(speech_streams, "reserve_tts", reserve)
    return SimpleNamespace(
        admin=admin,
        owner_id=owner_id,
        agent_id=agent_id,
        guest=_authenticate_widget_guest_by_key(key),
        tts=tts,
        tts_factory=tts_factory,
        usage=usage,
        gated=gated,
        gate_models=gate_models,
        resolve=resolve,
        reserve=reserve,
    )


def _no_side_effects(voice) -> None:
    voice.tts_factory.assert_not_called()
    assert voice.tts.calls == []
    assert voice.usage == []
    assert widget_voice_budget._memory_store._counts == {}


@pytest.mark.parametrize("credential", ["none", "login", "share", "expired"])
def test_widget_voice_rejects_foreign_credentials(voice, credential):
    guest_token = voice.guest["Authorization"].removeprefix("Bearer ")
    claims = jwt.decode(guest_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    headers = {
        "none": {},
        "login": voice.admin,
        "share": {
            "Authorization": "Bearer "
            + create_public_chat_access_token({**claims, "auth_mode": "share"})
        },
        "expired": {
            "Authorization": "Bearer "
            + jwt.encode(
                {**claims, "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
                JWT_SECRET_KEY,
                algorithm=JWT_ALGORITHM,
            )
        },
    }[credential]
    response = client.post(PATH, headers=headers, json={"text": "Hello"})
    assert response.status_code in (401, 403), response.text
    assert isinstance(response.json()["detail"], str)
    voice.resolve.assert_not_called()
    _no_side_effects(voice)


@pytest.mark.parametrize("entity", ["agent", "workforce"])
def test_widget_voice_requires_switch(voice, entity):
    headers = voice.guest
    if entity == "agent":
        db = _direct_db_session()
        try:
            db.get(Agent, voice.agent_id).widget_voice_enabled = False
            db.commit()
        finally:
            db.close()
    else:
        workforce_id = _create_workforce("Voice switch workforce")
        headers = _authenticate_widget_guest_by_key(_enable_widget(workforce_id))
    _refusal(
        client.post(PATH, headers=headers, json={"text": "Hello"}),
        403,
        "voice_disabled",
    )
    voice.resolve.assert_not_called()
    voice.reserve.assert_not_called()
    assert voice.gated == []
    _no_side_effects(voice)


def test_workforce_guest_with_the_switch_on_is_served(voice):
    workforce_id = _create_workforce("Voice workforce on")
    headers = _authenticate_widget_guest_by_key(_enable_widget(workforce_id))
    _set_switch(Deployment, workforce_id)
    response = client.post(PATH, headers=headers, json={"text": "Hello"})
    assert response.status_code == 200, response.text
    assert voice.reserve.call_args.args[0] == f"workforce:{workforce_id}"


@pytest.mark.parametrize(
    "body",
    [
        {"text": "Hello", "model_id": "live"},
        {"text": "Hello", "voice": "other"},
        {"text": ""},
        {"text": "x" * 1001},
        {"text": 5},
        "not json",
    ],
)
def test_widget_voice_rejects_client_model_id_and_bad_bodies(voice, body):
    kwargs: dict[str, Any] = (
        {"content": body} if isinstance(body, str) else {"json": body}
    )
    _refusal(client.post(PATH, headers=voice.guest, **kwargs), 422, "invalid_request")
    voice.resolve.assert_not_called()
    _no_side_effects(voice)


def test_widget_voice_usage_billed_to_owner(voice):
    response = client.post(PATH, headers=voice.guest, json={"text": "Hello"})
    assert response.status_code == 200, response.text
    assert response.content == b"firstsecond"
    assert response.headers["x-audio-format"] == "pcm"
    assert voice.tts_factory.call_args.args[0].model_id == "voice"
    assert voice.tts.calls == [
        {
            "text": "Hello",
            "voice": "deployment-voice",
            "format": "pcm",
            "sample_rate": 24000,
        }
    ]
    assert voice.gate_models == [("voice",)]
    assert [(g[0], g[1]) for g in voice.gated] == [
        (voice.owner_id, "speech_synthesize")
    ] * 2
    assert len(voice.usage) == 1
    user_id, rows, source = voice.usage[0]
    assert (user_id, source) == (voice.owner_id, "speech_synthesize")
    assert [row["quantity"] for row in rows] == [5]


def test_widget_voice_refuses_unavailable_models_and_rate_limits(voice, monkeypatch):
    monkeypatch.delenv("XAGENT_WIDGET_VOICE_TTS_MODEL")
    _refusal(
        client.post(PATH, headers=voice.guest, json={"text": "Hi"}),
        503,
        "voice_unavailable",
    )
    _no_side_effects(voice)
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_MODEL", "voice")
    monkeypatch.setenv("XAGENT_WIDGET_VOICE_TTS_IP_RATE_LIMIT", "1/minute")
    from xagent.web.services.share_rate_limit import reset_share_rate_limiter

    reset_share_rate_limiter()
    assert (
        client.post(PATH, headers=voice.guest, json={"text": "Hi"}).status_code == 200
    )
    counts = dict(widget_voice_budget._memory_store._counts)
    usage = list(voice.usage)
    _refusal(
        client.post(PATH, headers=voice.guest, json={"text": "Hi"}), 429, "rate_limited"
    )
    assert widget_voice_budget._memory_store._counts == counts
    assert voice.usage == usage and len(voice.tts.calls) == 1


@pytest.mark.parametrize("failure", ["limit", "storage"])
@pytest.mark.parametrize("bucket", [b.name for b in TTS_BUCKETS])
def test_widget_voice_budget_buckets(voice, monkeypatch, caplog, bucket, failure):
    assert len(TTS_BUCKETS) == 3
    if failure == "limit":
        monkeypatch.setenv(LIMITS[bucket], "4")
    else:
        store = widget_voice_budget._memory_store
        real = store.incrby

        def incrby(key, amount):
            if f":{bucket}:" in key:
                raise ConnectionError("storage down")
            return real(key, amount)

        monkeypatch.setattr(store, "incrby", incrby)
    caplog.set_level(logging.INFO, logger=speech_streams.__name__)
    response = client.post(PATH, headers=voice.guest, json={"text": "Hello"})
    if failure == "limit":
        _refusal(response, 429, "voice_budget_exhausted")
    else:
        _refusal(response, 503, "voice_unavailable")
    assert voice.tts.calls == [] and voice.usage == []
    assert set(widget_voice_budget._memory_store._counts.values()) <= {0}
    assert voice.tts.close_count == 1
    result = "budget_exhausted" if failure == "limit" else "budget_unavailable"
    assert f"user_id={voice.owner_id}" in caplog.text
    assert f"chars=5 result={result}" in caplog.text


@pytest.mark.parametrize("failure", ["value", "runtime", "gate", "capabilities"])
def test_widget_voice_error_bodies_are_fixed_text(voice, monkeypatch, failure):
    status, code = {
        "value": (400, "speech_synthesis_failed"),
        "runtime": (502, "speech_synthesis_failed"),
        "gate": (402, "standalone_usage_refused"),
        "capabilities": (502, "speech_synthesis_failed"),
    }[failure]
    if failure == "value":
        voice.tts.error = ValueError(f"bad key {SECRET}")
    elif failure == "runtime":
        voice.tts.error = RuntimeError(f"provider said {SECRET}")
    elif failure == "gate":
        monkeypatch.setattr(
            quota_hooks,
            "_standalone_gate_hook",
            lambda user_id, source: {
                "message": f"team quota {SECRET}",
                "plan": "owner-plan",
            },
        )
    else:

        async def broken():
            raise RuntimeError(f"probe {SECRET}")

        monkeypatch.setattr(voice.tts, "get_streaming_capabilities", broken)
    response = client.post(PATH, headers=voice.guest, json={"text": "Hello"})
    _refusal(response, status, code)
    assert SECRET not in response.text and "owner-plan" not in response.text


def test_widget_voice_usage_log(voice, caplog):
    caplog.set_level(logging.INFO, logger=speech_streams.__name__)
    assert (
        client.post(PATH, headers=voice.guest, json={"text": "Hello"}).status_code
        == 200
    )
    lines = [
        r.getMessage()
        for r in caplog.records
        if "Speech synthesis usage" in r.getMessage()
    ]
    assert len(lines) == 1
    assert (
        f"audience=guest user_id={voice.owner_id} entity=agent:{voice.agent_id}"
        in lines[0]
    )
    assert "model=voice chars=5 result=ok" in lines[0]


@pytest.mark.parametrize(
    "keys",
    [
        {"audience": "guest", "visitor_key": "1.2.3.4"},
        {"audience": "guest", "entity_key": "agent:1"},
        {"audience": "member", "entity_key": "agent:1"},
    ],
)
def test_voice_plan_validation(keys):
    with pytest.raises(ValueError):
        SpeechStreamPlan(
            user_id=1,
            config=SimpleNamespace(model_id="voice"),
            options=TTSStreamRequest(model_id="voice", text="Hello"),
            usage_source="speech_synthesize",
            **keys,
        )


def test_widget_voice_request_body_is_documented():
    from xagent.web.app import app

    operation = app.openapi()["paths"][PATH]["post"]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert set(schema["properties"]) == {"text"}
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["text"]


def test_widget_voice_without_a_visitor_address_is_unavailable(voice, monkeypatch):
    monkeypatch.setattr(widget_api, "remote_ip_from_request", lambda _request: None)
    _refusal(
        client.post(PATH, headers=voice.guest, json={"text": "Hi"}),
        503,
        "voice_unavailable",
    )
    voice.resolve.assert_not_called()
    _no_side_effects(voice)


@pytest.mark.parametrize("change", ["widget_disabled", "widget_key_rotated"])
def test_widget_voice_rejects_a_stale_guest_credential(voice, change):
    if change == "widget_disabled":
        changed = client.put(
            f"/api/agents/{voice.agent_id}",
            headers=voice.admin,
            json={"widget_enabled": False},
        )
    else:
        changed = client.post(
            f"/api/agents/{voice.agent_id}/widget-key/rotate", headers=voice.admin
        )
    assert changed.status_code == 200, changed.text
    response = client.post(PATH, headers=voice.guest, json={"text": "Hi"})
    assert response.status_code in (401, 403), response.text
    assert isinstance(response.json()["detail"], str)
    voice.resolve.assert_not_called()
    _no_side_effects(voice)


def test_widget_voice_releases_the_connection_before_reading_the_body(
    voice, monkeypatch
):
    from starlette.requests import Request

    from xagent.web.models import database

    sessions: list[Any] = []
    real_release = database.release_db_connection_if_clean

    def release(db):
        sessions.append(db)
        return real_release(db)

    states: list[bool | None] = []
    real_body = Request.body

    async def body(self):
        states.append(sessions[0].in_transaction() if sessions else None)
        return await real_body(self)

    monkeypatch.setattr(widget_api, "release_db_connection_if_clean", release)
    monkeypatch.setattr(Request, "body", body)
    response = client.post(PATH, headers=voice.guest, json={"text": "Hello"})
    assert response.status_code == 200, response.text
    # The credential check's session no longer holds a connection.
    assert states == [False]


def test_widget_voice_serves_a_model_that_streams_after_discovery(voice, monkeypatch):
    monkeypatch.setattr(
        widget_voice, "get_asr_model_instance", Mock(return_value=_Discovered(True))
    )
    monkeypatch.setattr(
        widget_voice, "get_tts_model_instance", Mock(return_value=_Discovered(True))
    )
    response = client.post(PATH, headers=voice.guest, json={"text": "Hello"})
    assert response.status_code == 200, response.text
    assert response.content == b"firstsecond"


def test_widget_voice_tts_answers_on_a_one_slot_pool(voice, monkeypatch):
    _install_one_slot_queue_pool(monkeypatch)
    response = client.post(PATH, headers=voice.guest, json={"text": "Hello"})
    assert response.status_code == 200, response.text
    assert response.content == b"firstsecond"


async def test_resolution_returns_the_request_connection_first(voice, monkeypatch):
    engine = _install_one_slot_queue_pool(monkeypatch)
    checked_out: list[int] = []

    class CountingASR(FakeASR):
        async def get_streaming_capabilities(self):
            checked_out.append(engine.pool.checkedout())
            return await super().get_streaming_capabilities()

    monkeypatch.setattr(
        widget_voice, "get_asr_model_instance", Mock(return_value=CountingASR())
    )
    db = get_session_local()()
    try:
        # A text() statement counts as a possible write; an ORM SELECT does not.
        db.execute(select(User.id)).all()
        result = await widget_voice.resolve_widget_voice_models(
            voice.owner_id, request_db=db
        )
        assert result.reason is None and result.models is not None
        assert db.in_transaction() is False
        assert checked_out == [0]
    finally:
        db.close()


async def test_resolution_refuses_a_request_session_with_uncommitted_work(
    voice, monkeypatch
):
    asr_factory = Mock(return_value=FakeASR())
    tts_factory = Mock(return_value=FakeTTS())
    monkeypatch.setattr(widget_voice, "get_asr_model_instance", asr_factory)
    monkeypatch.setattr(widget_voice, "get_tts_model_instance", tts_factory)
    db = get_session_local()()
    try:
        db.add(User(username="pending", password_hash="x"))
        with pytest.raises(RuntimeError):
            await widget_voice.resolve_widget_voice_models(
                voice.owner_id, request_db=db
            )
        asr_factory.assert_not_called()
        tts_factory.assert_not_called()
    finally:
        db.rollback()
        db.close()
