"""Review contracts against this branch's actual, merged-main transport APIs."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from xagent.core.model import public_http
from xagent.core.model.chat.error import retry_on
from xagent.core.model.image.base import retry_image_call
from xagent.core.model.network_policy import ModelNetworkPolicy
from xagent.core.utils.public_http_transport import UnsafePublicHttpTarget
from xagent.core.utils.security import PrivateNetworkHostError


@pytest.mark.parametrize(
    "cidr",
    [
        "8.8.8.0/24",
        "169.254.0.0/16",
        "0.0.0.0/0",
        "224.0.0.0/4",
        "10.0.0.1/8",
        "not-a-cidr",
    ],
)
def test_invalid_operator_cidrs_are_terminal_policy_errors(monkeypatch, cidr):
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", cidr)
    with pytest.raises(PrivateNetworkHostError, match="configuration") as error:
        ModelNetworkPolicy()
    assert public_http.is_model_network_policy_error(error.value)
    assert retry_on(error.value) is False
    assert retry_image_call(error.value) is False


def test_wrapped_policy_denial_is_not_an_image_retry():
    refusal = UnsafePublicHttpTarget("blocked")
    wrapped = RuntimeError("SDK failure")
    wrapped.__cause__ = refusal
    assert retry_image_call(wrapped) is False
    assert retry_image_call(TimeoutError("transient")) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_config", [False, True])
async def test_xinference_listing_does_not_retry_policy_errors(
    monkeypatch, invalid_config
):
    from xagent.core.model.chat.basic import xinference

    if invalid_config:
        monkeypatch.setenv(
            "XAGENT_MODEL_PRIVATE_ENDPOINTS", "http://localhost:9997/api"
        )
    create = Mock(wraps=xinference._create_async_client)
    sleep = AsyncMock(side_effect=AssertionError("policy rejection was retried"))
    monkeypatch.setattr(xinference, "_create_async_client", create)
    monkeypatch.setattr(xinference.asyncio, "sleep", sleep)
    with pytest.raises(ValueError, match="blocked by deployment network policy"):
        await xinference.XinferenceLLM.list_available_models(
            base_url="http://127.0.0.1:9997"
        )
    assert create.call_count == 1
    sleep.assert_not_awaited()


def test_exact_ipv6_origin_and_address_can_be_approved(monkeypatch):
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_ENDPOINTS", "http://[::1]:9997")
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", "::1/128")
    policy = ModelNetworkPolicy()
    policy.validate_resolved("http://[::1]:9997/v1/models", ["::1"])
    assert policy.destination("http://[::1]:9997/v1/models") == ("::1", 9997)
    for url in ("http://[::1]:9998/", "https://[::1]:9997/"):
        with pytest.raises(PrivateNetworkHostError):
            policy.validate_resolved(url, ["::1"])


@pytest.mark.asyncio
async def test_untrusted_ambient_proxy_is_ignored_by_every_model_transport(monkeypatch):
    from xagent.core.model.chat.basic.bedrock import BedrockLLM
    from xagent.core.model.public_requests import Session

    monkeypatch.setenv("HTTPS_PROXY", "http://operator-proxy.example:3128")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "false")
    assert public_http.get_model_proxy_url() is None
    with public_http.create_public_model_sync_client() as client:
        assert client._transport._via_proxy is False
    async with public_http.create_public_model_async_client() as client:
        assert client._transport._via_proxy is False
    async with public_http.create_public_model_aiohttp_session() as session:
        assert session._default_proxy is None
    with Session() as session:
        assert session.proxies == {}
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    sdk = BedrockLLM(
        "test", region_name="us-east-1", endpoint_url="https://bedrock.example"
    )._build_client()
    try:
        assert sdk._endpoint.http_session._via_proxy is False
    finally:
        sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sdk", "elevenlabs"])
async def test_async_constructor_cleanup_task_is_retained_until_finished(
    monkeypatch, kind
):
    from xagent.core.model import public_elevenlabs

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    original_close = client.aclose
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def close():
        entered.set()
        await finish.wait()
        await original_close()

    monkeypatch.setattr(client, "aclose", close)
    owner = public_http if kind == "sdk" else public_elevenlabs
    monkeypatch.setattr(
        owner, "create_public_model_async_client", lambda **kwargs: client
    )

    def fail(**kwargs):
        raise RuntimeError("constructor failed")

    previous = set(public_http._PENDING_CLOSES)
    with pytest.raises(RuntimeError, match="constructor failed"):
        if kind == "sdk":
            public_http.create_owned_async_sdk_client(fail, timeout=1)
        else:
            public_elevenlabs.create_public_elevenlabs_client(fail, asynchronous=True)
    await asyncio.wait_for(entered.wait(), 1)
    pending = public_http._PENDING_CLOSES - previous
    assert len(pending) == 1
    try:
        assert not client.is_closed
    finally:
        finish.set()
        await asyncio.gather(*pending)
    await asyncio.sleep(0)
    assert not (public_http._PENDING_CLOSES & pending)
    assert client.is_closed


def test_zhipu_constructor_failure_closes_owned_sync_client(monkeypatch):
    from xagent.core.model.chat.basic import zhipu

    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    monkeypatch.setattr(
        public_http, "create_public_model_sync_client", lambda **kwargs: client
    )
    monkeypatch.setattr(
        zhipu, "ZhipuAiClient", Mock(side_effect=RuntimeError("constructor failed"))
    )
    with pytest.raises(RuntimeError, match="constructor failed"):
        zhipu.ZhipuLLM("test", api_key="key")._ensure_client()
    assert client.is_closed


def test_bedrock_session_constructor_failure_closes_previous_session(monkeypatch):
    import boto3

    from xagent.core.model.chat.basic import bedrock

    previous = SimpleNamespace(_verify=True, close=Mock())
    sdk = SimpleNamespace(_endpoint=SimpleNamespace(http_session=previous))
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: sdk)
    monkeypatch.setattr(
        bedrock,
        "PublicBotocoreSession",
        Mock(side_effect=RuntimeError("constructor failed")),
    )
    with pytest.raises(RuntimeError, match="constructor failed"):
        bedrock.BedrockLLM("test")._build_client()
    previous.close.assert_called_once()


def test_xinference_failed_auth_probe_closes_guarded_session(monkeypatch):
    from xagent.core.model import public_xinference

    sessions = []
    original = public_xinference.Session

    class Session(original):
        def __init__(self):
            super().__init__()
            self.closed = False
            sessions.append(self)

        def get(self, *args, **kwargs):
            raise RuntimeError("auth probe failed")

        def close(self):
            self.closed = True
            super().close()

    monkeypatch.setattr(public_xinference, "Session", Session)
    with pytest.raises(RuntimeError, match="auth probe failed"):
        public_xinference.PublicXinferenceClient("https://xinference.example")
    assert len(sessions) == 1
    assert sessions[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "claude", "zhipu"])
async def test_catalog_clients_refuse_private_endpoints_without_dispatch(
    monkeypatch, provider
):
    seen = []

    def transport(**kwargs):
        return httpx.MockTransport(
            lambda request: (
                seen.append(request) or httpx.Response(200, json={"data": []})
            )
        )

    monkeypatch.setattr(httpx, "HTTPTransport", transport)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    url = "http://127.0.0.1:9997/v1"
    if provider == "openai":
        from xagent.core.model.chat.basic import openai

        real = openai.AsyncOpenAI

        def factory(**kwargs):
            assert kwargs["max_retries"] == 0
            return real(**kwargs)

        monkeypatch.setattr(openai, "AsyncOpenAI", factory)
        fetch = openai.OpenAILLM.list_available_models
    elif provider == "claude":
        from xagent.core.model.chat.basic.claude import ClaudeLLM

        fetch = ClaudeLLM.list_available_models
    else:
        from xagent.core.model.chat.basic import zhipu

        monkeypatch.setattr(
            zhipu,
            "ZhipuAiClient",
            Mock(side_effect=AssertionError("catalog constructed SDK")),
        )
        fetch = zhipu.ZhipuLLM.list_available_models
    with pytest.raises(Exception) as error:
        await fetch("key", base_url=url, raise_on_error=True)
    assert public_http.is_model_network_policy_error(error.value)
    assert not seen


@pytest.mark.asyncio
@pytest.mark.parametrize("via_proxy", [False, True])
async def test_video_listing_uses_approved_private_model_policy(monkeypatch, via_proxy):
    from xagent.core.utils.public_aiohttp import create_public_aiohttp_session
    from xagent.web.services.model_list_service import fetch_xinference_video_models

    seen = []

    async def serve(reader, writer):
        try:
            seen.append(await reader.readuntil(b"\r\n\r\n"))
            body = json.dumps(
                {
                    "data": [
                        {
                            "id": "video-uid",
                            "model_name": "video",
                            "model_type": "video",
                            "model_ability": ["text2video"],
                        }
                    ]
                }
            ).encode()
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

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    async with server:
        with pytest.raises(PrivateNetworkHostError):
            await fetch_xinference_video_models("key", base_url)
        assert not seen
        monkeypatch.setenv("XAGENT_MODEL_PRIVATE_ENDPOINTS", base_url)
        monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", "127.0.0.1/32")
        if via_proxy:
            monkeypatch.setenv("HTTPS_PROXY", base_url)
            monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
        result = await fetch_xinference_video_models("key", base_url)
        assert result[0]["model_uid"] == "video-uid"
        assert len(seen) == 1
        assert b"Authorization: Bearer key" in seen[0]
        assert seen[0].startswith(b"GET http://" if via_proxy else b"GET /v1/models")
        # Generic clients still deny the same operator-approved model origin.
        async with create_public_aiohttp_session() as generic:
            with pytest.raises(PrivateNetworkHostError):
                await generic.get(base_url)
        assert len(seen) == 1


@pytest.mark.asyncio
async def test_web_search_does_not_inherit_private_model_exceptions(monkeypatch):
    from xagent.core.tools.core.zhipu_web_search import ZhipuWebSearchCore

    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_ENDPOINTS", "http://127.0.0.1:9997")
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", "127.0.0.1/32")
    with pytest.raises(
        ValueError, match="Host must not resolve to a private network"
    ) as error:
        await ZhipuWebSearchCore(
            api_key="key", base_url="http://127.0.0.1:9997"
        ).search("test")
    assert public_http.is_model_network_policy_error(error.value)
