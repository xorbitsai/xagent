"""Real SDK discovery/model handles must retain guarded sessions."""

from unittest.mock import AsyncMock

import aiohttp
import pytest
import requests

from xagent.core.model.public_xinference import (
    PublicAsyncXinferenceClient,
    PublicXinferenceClient,
)
from xagent.core.model.xinference_base import BaseXinferenceModel
from xagent.core.utils.public_aiohttp import PublicTCPConnector
from xagent.core.utils.public_requests import Session
from xagent.core.utils.security import PrivateNetworkHostError

MODEL_TYPES = [
    ("LLM", "chat"),
    ("LLM", "generate"),
    ("embedding", None),
    ("image", None),
    ("rerank", None),
    ("audio", None),
    ("video", None),
    ("flexible", None),
]


@pytest.mark.parametrize(("model_type", "ability"), MODEL_TYPES)
def test_sync_sdk_auth_probe_and_every_handle_use_public_sessions(
    monkeypatch, model_type, ability
):
    calls = []

    def request(session, method, url, **kwargs):
        assert isinstance(session, Session)
        calls.append(url)
        response = requests.Response()
        response.status_code = 200
        response.json = (
            lambda: {"auth": True}
            if url.endswith("/auth")
            else {"model_type": model_type, "model_ability": [ability]}
        )
        return response

    monkeypatch.setattr(Session, "request", request)
    client = PublicXinferenceClient("https://xinference.example", api_key="test-key")
    try:
        handle = client.get_model("model-id")
        try:
            assert isinstance(handle.session, Session)
            assert handle.auth_headers["Authorization"] == "Bearer test-key"
        finally:
            handle.close()
        assert calls == [
            "https://xinference.example/v1/cluster/auth",
            "https://xinference.example/v1/models/model-id",
        ]
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("model_type", "ability"), MODEL_TYPES)
async def test_async_sdk_never_probes_sync_and_all_handles_use_public_connectors(
    monkeypatch, model_type, ability
):
    def forbidden(*args, **kwargs):
        pytest.fail("asynchronous SDK made a synchronous auth request")

    monkeypatch.setattr(requests.Session, "get", forbidden)
    response = type("Response", (), {})()
    response.status = 200
    response.json = AsyncMock(
        return_value={"model_type": model_type, "model_ability": [ability]}
    )
    response.release = lambda: None
    response.wait_for_close = AsyncMock()
    monkeypatch.setattr(aiohttp.ClientSession, "get", AsyncMock(return_value=response))
    client = PublicAsyncXinferenceClient(
        "https://xinference.example", api_key="test-key"
    )
    assert isinstance(client.session.connector, PublicTCPConnector)
    assert client._headers == {"Authorization": "Bearer test-key"}
    try:
        handle = await client.get_model("model-id")
        try:
            assert isinstance(handle.session.connector, PublicTCPConnector)
            assert handle.auth_headers == {"Authorization": "Bearer test-key"}
        finally:
            await handle.close()
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_class", [PublicXinferenceClient, PublicAsyncXinferenceClient]
)
async def test_private_xinference_endpoints_fail_before_sdk_constructor(client_class):
    with pytest.raises(PrivateNetworkHostError):
        client_class("http://192.168.1.1:9997/", api_key="test-key")


@pytest.mark.asyncio
async def test_base_model_awaits_real_async_session_cleanup():
    model = BaseXinferenceModel("model", base_url="https://xinference.example")
    client = await model._get_session()
    session = client.session
    await model.aclose()
    assert session.closed
    assert model._client is None
