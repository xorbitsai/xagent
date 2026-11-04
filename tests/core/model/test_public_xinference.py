"""Real SDK discovery/model handles must retain guarded sessions."""

from unittest.mock import AsyncMock

import aiohttp
import pytest
import requests

from xagent.core.model.public_requests import Session
from xagent.core.model.public_xinference import (
    PublicAsyncXinferenceClient,
    PublicXinferenceClient,
)
from xagent.core.model.xinference_base import BaseXinferenceModel
from xagent.core.utils.public_aiohttp import PublicTCPConnector
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
        response.json = lambda: (
            {"auth": True}
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


@pytest.mark.asyncio
async def test_real_async_discovery_and_handle_guard_dns_before_socket(monkeypatch):
    import asyncio
    import json

    seen = []

    async def serve(reader, writer):
        try:
            seen.append(await reader.readuntil(b"\r\n\r\n"))
            body = json.dumps({"model_type": "embedding"}).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    class Resolver(aiohttp.abc.AbstractResolver):
        answer = "127.0.0.1"

        async def resolve(self, host, port, family=0):
            import socket

            return [
                {
                    "hostname": host,
                    "host": self.answer,
                    "port": port,
                    "family": socket.AF_INET,
                    "proto": 0,
                    "flags": 0,
                }
            ]

        async def close(self):
            pass

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    origin = f"http://xinference.example:{server.sockets[0].getsockname()[1]}"
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_ENDPOINTS", origin)
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", "127.0.0.1/32")
    resolver = Resolver()
    async with server:
        client = PublicAsyncXinferenceClient(origin, api_key="key")
        client.session.connector._resolver = resolver
        handle = None
        try:
            handle = await client.get_model("embedding")
            assert len(seen) == 1 and b"Authorization: Bearer key" in seen[0]
            handle.session.connector._resolver = resolver
            resolver.answer = "10.0.0.1"
            with pytest.raises(PrivateNetworkHostError):
                await handle.session.get(origin + "/v1/embeddings")
            with pytest.raises(PrivateNetworkHostError):
                await client.get_model("refused")
            assert len(seen) == 1
        finally:
            if handle is not None:
                await handle.close()
            await client.close()


def test_real_sync_client_close_closes_guarded_session(monkeypatch):
    from unittest.mock import Mock

    from xagent.core.model.public_requests import Session as ModelSession

    response = requests.Response()
    response.status_code = 200
    response.json = lambda: {"auth": False}
    monkeypatch.setattr(ModelSession, "get", lambda *args, **kwargs: response)
    client = PublicXinferenceClient("https://xinference.example")
    assert isinstance(client.session, ModelSession)
    close = Mock(wraps=client.session.close)
    monkeypatch.setattr(client.session, "close", close)
    client.close()
    close.assert_called_once()
