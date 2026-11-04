"""Maintained Zai SDK and published DeepDoc dependency compatibility."""

import base64
import hmac
import json

import httpx
import jwt
import pytest
from zai import ZaiClient, ZhipuAiClient


@pytest.fixture
def sdk_http_transport():
    seen = []

    def send(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "created": 1,
                "model": "test-model",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(send)) as transport:
        yield transport, seen


@pytest.mark.parametrize("client_type", [ZhipuAiClient, ZaiClient])
@pytest.mark.parametrize("raw_key", [True, False])
def test_retained_sdk_signatures_remain_compatible(
    client_type, raw_key, sdk_http_transport
):
    secret = "sdk-test-secret-with-at-least-32-characters"
    transport, seen = sdk_http_transport
    with client_type(
        api_key=f"test-id.{secret}",
        http_client=transport,
        disable_token_cache=raw_key,
    ) as client:
        result = client.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "test"}]
        )
        assert result.choices[0].message.content == "ok"
    token = seen[0].headers["Authorization"].removeprefix("Bearer ")
    if raw_key:
        assert token == f"test-id.{secret}"
        return
    header, payload, signature = token.split(".")
    unsigned = f"{header}.{payload}".encode()
    expected = (
        base64.urlsafe_b64encode(hmac.digest(secret.encode(), unsigned, "sha256"))
        .rstrip(b"=")
        .decode()
    )
    assert signature == expected
    assert json.loads(base64.urlsafe_b64decode(payload + "==="))["api_key"] == "test-id"
    assert jwt.decode(token, secret, algorithms=["HS256"])["api_key"] == "test-id"


@pytest.mark.parametrize("gateway", [None, "https://gateway.example.com/zhipu/"])
def test_published_deepdoc_vision_uses_maintained_sdk(
    sdk_http_transport, monkeypatch, gateway
):
    from deepdoc.depend import simple_cv_model

    transport, seen = sdk_http_transport
    monkeypatch.setenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
    if gateway:
        monkeypatch.setenv("ZHIPUAI_BASE_URL", gateway)
    else:
        monkeypatch.delenv("ZHIPUAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        simple_cv_model,
        "ZhipuAiClient",
        lambda **options: ZhipuAiClient(http_client=transport, **options),
    )
    model = simple_cv_model.Zhipu4V(
        "test-id.sdk-test-secret-with-at-least-32-characters"
    )
    assert model.describe_with_prompt(b"image-bytes", "Describe") == "ok"
    (request,) = seen
    assert (
        str(request.url)
        == (gateway or "https://open.bigmodel.cn/api/paas/v4/") + "chat/completions"
    )
    assert request.headers["Authorization"]
    payload = json.loads(request.content)
    assert payload["model"] == "glm-4v"
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "Describe"},
        {
            "type": "image_url",
            "image_url": {"url": base64.b64encode(b"image-bytes").decode()},
        },
    ]
