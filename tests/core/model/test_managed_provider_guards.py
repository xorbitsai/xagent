"""Real SDK contracts: removing an injected transport must fail these tests."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from xagent.core.model.chat.error import retry_on
from xagent.core.model.public_http import is_model_network_policy_error


@pytest.mark.asyncio
@pytest.mark.parametrize("via_proxy", [False, True])
@pytest.mark.parametrize(
    "provider", ["openai", "azure", "claude", "gemini", "zhipu", "openai_image"]
)
async def test_real_provider_rejects_private_base_before_inner_transport(
    monkeypatch, provider, via_proxy
):
    seen = []
    options = []
    if via_proxy:
        monkeypatch.setenv("HTTPS_PROXY", "http://operator-proxy.example:3128")
        monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")

    def transport(**kwargs):
        options.append(kwargs)
        return httpx.MockTransport(
            lambda request: (
                seen.append(request) or httpx.Response(200, json={"data": []})
            )
        )

    monkeypatch.setattr(httpx, "HTTPTransport", transport)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", transport)
    if provider == "openai":
        from xagent.core.model.chat.basic.openai import OpenAILLM

        model = OpenAILLM("test", api_key="key", base_url="http://127.0.0.1:9997/v1")
    elif provider == "azure":
        from xagent.core.model.chat.basic.azure_openai import AzureOpenAILLM

        model = AzureOpenAILLM(
            "test",
            api_key="key",
            azure_endpoint="http://127.0.0.1:9997",
            api_version="2024-08-01-preview",
        )
    elif provider == "claude":
        from xagent.core.model.chat.basic.claude import ClaudeLLM

        model = ClaudeLLM("test", api_key="key", base_url="http://127.0.0.1:9997")
    elif provider == "gemini":
        from xagent.core.model.chat.basic.gemini import GeminiLLM

        model = GeminiLLM("test", api_key="key", base_url="http://127.0.0.1:9997")
    elif provider == "zhipu":
        from xagent.core.model.chat.basic.zhipu import ZhipuLLM

        model = ZhipuLLM("test", api_key="key", base_url="http://127.0.0.1:9997")
    else:
        from xagent.core.model.image.openai import OpenAIImageModel

        model = OpenAIImageModel(api_key="key", base_url="http://127.0.0.1:9997/v1")
    model._ensure_client()
    try:
        sdk_http = (
            model._client._api_client._async_httpx_client
            if provider == "gemini"
            else model._client._client
        )
        assert sdk_http.follow_redirects is True
        expected_proxy = "http://operator-proxy.example:3128" if via_proxy else None
        if provider == "zhipu":
            assert sdk_http.timeout.connect == model.timeout
        try:
            if provider == "gemini":
                assert not model._client._api_client._use_aiohttp()
                await model._client.aio.models.list()
            elif provider == "zhipu":
                model._client.chat.completions.create(
                    model="test", messages=[{"role": "user", "content": "test"}]
                )
            else:
                await model._client.models.list()
        except Exception as error:
            assert is_model_network_policy_error(error), type(error)
            assert retry_on(error) is False
        else:
            pytest.fail("SDK request escaped the destination policy")
        assert not seen
        assert options and all(item["proxy"] == expected_proxy for item in options)
    finally:
        await model.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "azure", "claude", "openai_image"])
async def test_sdk_constructor_failure_closes_injected_client(monkeypatch, provider):
    from xagent.core.model import public_http

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    monkeypatch.setattr(
        public_http, "create_public_model_async_client", lambda **kwargs: client
    )

    class ConfigurationFailure(ValueError):
        pass

    def fail(**kwargs):
        assert kwargs["http_client"] is client
        raise ConfigurationFailure

    if provider == "openai":
        from xagent.core.model.chat.basic import openai

        monkeypatch.setattr(openai, "AsyncOpenAI", fail)
        model = openai.OpenAILLM("test", api_key="key")
    elif provider == "azure":
        from xagent.core.model.chat.basic import azure_openai

        monkeypatch.setattr(azure_openai, "AsyncAzureOpenAI", fail)
        model = azure_openai.AzureOpenAILLM(
            "test",
            api_key="key",
            azure_endpoint="https://azure.example",
            api_version="test",
        )
    elif provider == "claude":
        from xagent.core.model.chat.basic import claude

        monkeypatch.setattr(claude, "AsyncAnthropic", fail)
        model = claude.ClaudeLLM("test", api_key="key")
    else:
        from xagent.core.model.image import openai

        monkeypatch.setattr(openai, "AsyncOpenAI", fail)
        model = openai.OpenAIImageModel(api_key="key")
    with pytest.raises(ConfigurationFailure):
        model._ensure_client()
    await asyncio.sleep(0)
    assert client.is_closed


@pytest.mark.asyncio
async def test_openai_image_close_is_idempotent():
    from xagent.core.model.image.openai import OpenAIImageModel

    model = OpenAIImageModel(api_key="key")
    model._client = type("SDK", (), {"close": AsyncMock()})()
    sdk = model._client
    await model.aclose()
    await model.aclose()
    sdk.close.assert_awaited_once()


def test_bedrock_real_sdk_swap_and_private_refusal(monkeypatch):
    from botocore.awsrequest import AWSPreparedRequest

    from xagent.core.model.chat.basic.bedrock import BedrockLLM
    from xagent.core.utils.public_botocore import PublicBotocoreSession
    from xagent.core.utils.public_http_transport import UnsafePublicHttpTarget

    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    model = BedrockLLM(
        "test", region_name="us-east-1", endpoint_url="http://127.0.0.1:9997", timeout=7
    )
    sdk = model._build_client()
    try:
        session = sdk._endpoint.http_session
        assert isinstance(session, PublicBotocoreSession)
        assert session._timeout == 7
        assert session._verify is True
        assert (
            session._pool_classes_by_scheme["https"].__module__ == "botocore.awsrequest"
        )
        with pytest.raises(UnsafePublicHttpTarget):
            session.send(
                AWSPreparedRequest(
                    method="POST",
                    url="http://127.0.0.1:9997/",
                    headers={},
                    body=b"{}",
                    stream_output=True,
                )
            )
    finally:
        sdk.close()
