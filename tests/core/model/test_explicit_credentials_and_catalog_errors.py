"""Explicit-credential construction, strict catalog reads, and SDK-thread context.

Hosts that run configurations owned by someone other than the deployment
(a model a user configured with their own key) need three guarantees from
the shared layer:

- the request authenticates with the configuration's own key, never an
  ambient deployment credential (``explicit_credentials_only``);
- a catalog read that failed can be told apart from an empty catalog
  (``raise_on_error``), without changing what existing callers get;
- context variables set by the caller reach the SDK thread an adapter uses.

Only the HTTP transport or the SDK client object is replaced; keys are
synthetic.
"""

import contextvars
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from openai import AsyncAzureOpenAI

from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.azure_openai import AzureOpenAILLM
from xagent.core.model.chat.basic.claude import (
    AnthropicAuthenticationError,
    ClaudeLLM,
    ModelCatalogUnavailableError,
)
from xagent.core.model.chat.basic.gemini import GeminiLLM
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.model import ChatModelConfig
from xagent.web.services.model_list_service import fetch_models_from_provider

CALLER_KEY = "sk-explicit-CALLER-0001"
PLATFORM_AD_TOKEN = "platform-entra-token-0001"
ENDPOINT = "https://my-team.openai.azure.com"


def _completion(model: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def wire(monkeypatch):
    """Record every request the SDK would send; answer with ``wire.reply``."""
    sent: list[httpx.Request] = []
    state = {"reply": lambda request: httpx.Response(200, json=_completion("m"))}

    async def handle_async_request(self, request):
        sent.append(request)
        response = state["reply"](request)
        response.request = request
        return response

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request
    )
    return SimpleNamespace(sent=sent, state=state)


def _assert_caller_key_only(request: httpx.Request) -> None:
    # Some openai SDK versions also send the api key as a Bearer header; the
    # guarantee is that every credential header carries the configured key.
    assert request.headers.get("api-key") == CALLER_KEY
    assert request.headers.get("authorization") in (None, f"Bearer {CALLER_KEY}")
    assert PLATFORM_AD_TOKEN not in str(request.headers)


class TestAzureAuthentication:
    async def test_api_key_only_never_sends_an_ambient_entra_token(
        self, wire, monkeypatch
    ):
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", PLATFORM_AD_TOKEN)
        llm = AzureOpenAILLM(
            model_name="my-deployment",
            azure_endpoint=ENDPOINT,
            api_key=CALLER_KEY,
            api_key_only=True,
        )
        await llm.chat([{"role": "user", "content": "hi"}])
        # The SDK's per-request copies keep the same authentication.
        llm._ensure_client()
        copied = llm._client.with_options(timeout=5)
        await copied.chat.completions.create(
            model="my-deployment", messages=[{"role": "user", "content": "hi"}]
        )
        assert len(wire.sent) == 2
        for request in wire.sent:
            assert request.url.host == "my-team.openai.azure.com"
            _assert_caller_key_only(request)

    async def test_default_azure_behavior_is_unchanged(self, wire, monkeypatch):
        # Deployment-owned Azure models that rely on an Entra token from the
        # environment keep doing so: the isolation is opt-in.
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", PLATFORM_AD_TOKEN)
        llm = AzureOpenAILLM(
            model_name="my-deployment", azure_endpoint=ENDPOINT, api_key=CALLER_KEY
        )
        await llm.chat([{"role": "user", "content": "hi"}])
        assert (
            wire.sent[0].headers.get("authorization") == f"Bearer {PLATFORM_AD_TOKEN}"
        )

    def test_api_key_only_requires_a_key(self):
        with pytest.raises(ValueError):
            AzureOpenAILLM(
                model_name="d", azure_endpoint=ENDPOINT, api_key="", api_key_only=True
            )

    async def test_the_factory_flag_reaches_the_azure_adapter(self, wire, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", PLATFORM_AD_TOKEN)
        llm = create_base_llm(
            ChatModelConfig(
                id="my-deployment",
                model_name="my-deployment",
                model_provider="azure_openai",
                base_url=ENDPOINT,
                api_key=CALLER_KEY,
                max_retries=1,
                explicit_credentials_only=True,
            )
        )
        await llm.chat([{"role": "user", "content": "hi"}])
        _assert_caller_key_only(wire.sent[0])

    async def test_api_key_only_never_calls_a_token_provider(self, wire, monkeypatch):
        # An ambient token would win over the provider in the control below.
        monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
        calls: list[int] = []

        def token_provider() -> str:
            calls.append(1)
            return PLATFORM_AD_TOKEN

        llm = AzureOpenAILLM(
            model_name="my-deployment",
            azure_endpoint=ENDPOINT,
            api_key=CALLER_KEY,
            api_key_only=True,
        )
        llm._ensure_client()
        # A provider handed to a per-request copy is dropped like the token.
        copied = llm._client.with_options(azure_ad_token_provider=token_provider)
        await copied.chat.completions.create(
            model="my-deployment", messages=[{"role": "user", "content": "hi"}]
        )
        assert calls == []
        _assert_caller_key_only(wire.sent[0])

        # Control: the default client does authenticate with that provider.
        default = AsyncAzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_version="2024-08-01-preview",
            api_key=CALLER_KEY,
            azure_ad_token_provider=token_provider,
        )
        await default.chat.completions.create(
            model="my-deployment", messages=[{"role": "user", "content": "hi"}]
        )
        assert calls == [1]
        assert (
            wire.sent[1].headers.get("authorization") == f"Bearer {PLATFORM_AD_TOKEN}"
        )

    @pytest.mark.parametrize(
        "member",
        ["_azure_ad_token", "_azure_ad_token_provider", "_get_azure_ad_token"],
    )
    async def test_api_key_only_fails_loudly_if_the_sdk_renames_a_member(
        self, member, wire, monkeypatch
    ):
        # Stand-in for an openai release without a private member the
        # override relies on: building the client fails instead of the
        # override silently doing nothing.
        if member == "_get_azure_ad_token":
            monkeypatch.delattr(AsyncAzureOpenAI, member)
        else:
            sdk_init = AsyncAzureOpenAI.__init__

            def renamed_init(self, *args, **kwargs):
                sdk_init(self, *args, **kwargs)
                self.__dict__[member + "_v2"] = self.__dict__.pop(member)

            monkeypatch.setattr(AsyncAzureOpenAI, "__init__", renamed_init)
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", PLATFORM_AD_TOKEN)
        llm = AzureOpenAILLM(
            model_name="my-deployment",
            azure_endpoint=ENDPOINT,
            api_key=CALLER_KEY,
            api_key_only=True,
        )
        with pytest.raises(RuntimeError, match=f"has no {member}$"):
            await llm.chat([{"role": "user", "content": "hi"}])
        assert wire.sent == []
        # The default client does not depend on those members to be built.
        AzureOpenAILLM(
            model_name="my-deployment", azure_endpoint=ENDPOINT, api_key=CALLER_KEY
        )._ensure_client()


class TestExplicitCredentialsFactoryGate:
    @pytest.mark.parametrize("api_key", [None, "", "your-deepseek-key"])
    def test_missing_or_placeholder_keys_are_refused(self, api_key, monkeypatch):
        # DeepSeek would otherwise resolve these from DEEPSEEK_API_KEY.
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-platform-deepseek")
        with pytest.raises(ValueError):
            create_base_llm(
                ChatModelConfig(
                    id="deepseek-v4-flash",
                    model_name="deepseek-v4-flash",
                    model_provider="deepseek",
                    api_key=api_key,
                    explicit_credentials_only=True,
                )
            )

    @pytest.mark.parametrize(
        ("provider", "api_key"),
        [("router", None), ("router", CALLER_KEY), ("OpenRouter", CALLER_KEY)],
    )
    def test_auto_models_are_refused(self, provider, api_key):
        # Auto runs its candidates' (or a derived) configuration, which the
        # flag does not reach; refusing it keeps the flag from being dropped.
        with pytest.raises(ValueError, match="Auto"):
            create_base_llm(
                ChatModelConfig(
                    id="auto",
                    model_name="Auto",
                    model_provider=provider,
                    api_key=api_key,
                    explicit_credentials_only=True,
                )
            )

    def test_without_the_flag_the_environment_fallback_is_unchanged(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-platform-deepseek")
        llm = create_base_llm(
            ChatModelConfig(
                id="deepseek-v4-flash",
                model_name="deepseek-v4-flash",
                model_provider="deepseek",
                api_key="your-deepseek-key",
            )
        )
        assert llm.api_key == "sk-platform-deepseek"


def _status_error(status: int) -> Exception:
    request = httpx.Request("GET", "https://api.openai.com/v1/models")
    response = httpx.Response(status, request=request)
    cls = {
        403: openai.PermissionDeniedError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
    }[status]
    return cls("failed", response=response, body=None)


class TestStrictCatalogReads:
    async def test_custom_claude_endpoint_without_catalog_is_distinguishable(
        self, monkeypatch
    ):
        requested_urls: list[str] = []

        async def not_found(self, url, **kwargs):
            requested_urls.append(str(url))
            request = httpx.Request("GET", url)
            return httpx.Response(404, request=request)

        monkeypatch.setattr(httpx.AsyncClient, "get", not_found)

        with pytest.raises(ModelCatalogUnavailableError):
            await ClaudeLLM.list_available_models(
                "synthetic-key",
                "https://bedrock.example.com/anthropic/v1",
                raise_on_error=True,
            )

        assert requested_urls == ["https://bedrock.example.com/anthropic/v1/models"]

    @pytest.mark.parametrize(
        "base_url", [None, "https://api.anthropic.com", "https://api.anthropic.com/v1/"]
    )
    async def test_official_claude_catalog_404_remains_an_error(
        self, monkeypatch, base_url
    ):
        async def not_found(self, url, **kwargs):
            request = httpx.Request("GET", url)
            return httpx.Response(404, request=request)

        monkeypatch.setattr(httpx.AsyncClient, "get", not_found)

        with pytest.raises(httpx.HTTPStatusError):
            await ClaudeLLM.list_available_models(
                "synthetic-key", base_url, raise_on_error=True
            )

    @pytest.mark.parametrize("status", [401, 403])
    async def test_claude_catalog_rejects_invalid_credentials(
        self, monkeypatch, status
    ):
        async def rejected(self, url, **kwargs):
            request = httpx.Request("GET", url)
            return httpx.Response(status, request=request)

        monkeypatch.setattr(httpx.AsyncClient, "get", rejected)

        with pytest.raises(AnthropicAuthenticationError) as caught:
            await ClaudeLLM.list_available_models(
                "synthetic-key",
                "https://compatible.example.com/anthropic/v1",
                raise_on_error=True,
            )
        assert caught.value.status_code == status

    @pytest.mark.parametrize(
        "error",
        [
            _status_error(403),
            _status_error(429),
            _status_error(500),
            openai.APIConnectionError(
                request=httpx.Request("GET", "https://api.openai.com/v1/models")
            ),
        ],
        ids=["403", "429", "500", "connection"],
    )
    async def test_a_failed_read_raises_only_when_asked(self, error, mocker):
        client = mocker.AsyncMock()
        client.models.list.side_effect = error
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI", return_value=client
        )
        # Existing callers keep the empty-list answer.
        assert await OpenAILLM.list_available_models("sk-test") == []
        assert await fetch_models_from_provider("openai", "sk-test") == []
        # Strict callers see the failure.
        with pytest.raises(type(error)):
            await OpenAILLM.list_available_models("sk-test", raise_on_error=True)
        with pytest.raises(type(error)):
            await fetch_models_from_provider(
                "openrouter", "sk-test", raise_on_error=True
            )

    async def test_a_genuinely_empty_catalog_is_still_empty(self, mocker):
        client = mocker.AsyncMock()
        client.models.list.return_value = SimpleNamespace(data=[])
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI", return_value=client
        )
        assert (
            await fetch_models_from_provider("openai", "sk-test", raise_on_error=True)
            == []
        )

    async def test_fetchers_without_an_error_mode_ignore_the_option(self):
        # Static catalogs answer locally and have nothing to swallow.
        models = await fetch_models_from_provider(
            "deepseek", "sk-test", raise_on_error=True
        )
        assert {model["id"] for model in models} >= {"deepseek-v4-flash"}

    @pytest.mark.parametrize(
        ("module", "method_path"),
        [
            ("xagent.core.model.chat.basic.claude", "httpx.AsyncClient.get"),
            ("xagent.core.model.chat.basic.zhipu", "httpx.AsyncClient.get"),
        ],
    )
    async def test_other_swallowing_readers_raise_when_asked(
        self, module, method_path, monkeypatch
    ):
        import importlib

        adapter = importlib.import_module(module)
        reader = next(
            getattr(adapter, name).list_available_models
            for name in ("ClaudeLLM", "ZhipuLLM")
            if hasattr(adapter, name)
        )

        async def refuse(*args, **kwargs):
            raise httpx.ReadTimeout("read timed out")

        monkeypatch.setattr(httpx.AsyncClient, "get", refuse)
        with patch(f"{module}.ZhipuAiClient", create=True) as sdk:
            sdk.return_value.models.list.side_effect = AttributeError("no such method")
            assert await reader("sk-test") == []
            with pytest.raises(httpx.ReadTimeout):
                await reader("sk-test", raise_on_error=True)

    async def test_kimi_for_coding_forwards_the_error_mode(self, monkeypatch):
        urls: list[str] = []

        async def refuse(self, url, **kwargs):
            urls.append(str(url))
            raise httpx.ReadTimeout("read timed out")

        monkeypatch.setattr(httpx.AsyncClient, "get", refuse)
        assert await fetch_models_from_provider("kimi-for-coding", "sk-test") == []
        with pytest.raises(httpx.ReadTimeout):
            await fetch_models_from_provider(
                "kimi_for_coding", "sk-test", raise_on_error=True
            )
        # Both reads went to Kimi through the Claude-compatible reader.
        assert urls == ["https://api.kimi.com/coding/v1/models"] * 2


def _gemini_error(code: int, reason: str = "") -> Exception:
    status = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        429: "RESOURCE_EXHAUSTED",
        503: "UNAVAILABLE",
    }[code]
    body = {"error": {"code": code, "message": "failed", "status": status}}
    if reason:
        body["error"]["details"] = [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}
        ]
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body)


@pytest.fixture
def gemini_sdk(monkeypatch):
    """Replace ``genai.Client``; ``gemini_sdk.list`` answers the catalog read."""
    client = MagicMock()
    client.aio.models.list = AsyncMock()
    sdk = MagicMock(return_value=client)
    monkeypatch.setattr("google.genai.Client", sdk)
    return SimpleNamespace(client_class=sdk, list=client.aio.models.list)


class TestStrictGeminiCatalogReads:
    @pytest.mark.parametrize(
        "error",
        [
            _gemini_error(400),
            _gemini_error(429),
            _gemini_error(503),
            TimeoutError("read timed out"),
        ],
        ids=["400", "429", "503", "timeout"],
    )
    async def test_a_failed_read_raises_only_when_asked(self, error, gemini_sdk):
        gemini_sdk.list.side_effect = error
        assert await GeminiLLM.list_available_models("AIza-test") == []
        assert await fetch_models_from_provider("gemini", "AIza-test") == []
        with pytest.raises(type(error)):
            await fetch_models_from_provider("google", "AIza-test", raise_on_error=True)

    @pytest.mark.parametrize(
        ("code", "reason"),
        [(401, ""), (403, ""), (400, "API_KEY_INVALID")],
        ids=["401", "403", "400-API_KEY_INVALID"],
    )
    async def test_a_rejected_key_is_a_value_error_when_asked(
        self, code, reason, gemini_sdk
    ):
        gemini_sdk.list.side_effect = _gemini_error(code, reason)
        assert await fetch_models_from_provider("gemini", "AIza-test") == []
        with pytest.raises(ValueError) as caught:
            await fetch_models_from_provider("gemini", "AIza-test", raise_on_error=True)
        # The SDK error stays the cause, so its status is still readable.
        assert isinstance(caught.value.__cause__, genai_errors.ClientError)
        assert caught.value.__cause__.code == code

    @pytest.mark.parametrize("api_key", ["", "  ", "your-gemini-key"])
    async def test_a_strict_read_refuses_a_missing_or_placeholder_key(
        self, api_key, gemini_sdk, monkeypatch
    ):
        # genai.Client would otherwise authenticate with the deployment's key.
        monkeypatch.setenv("GOOGLE_API_KEY", "platform-google-key")
        monkeypatch.setenv("GEMINI_API_KEY", "platform-gemini-key")
        with pytest.raises(ValueError):
            await fetch_models_from_provider("gemini", api_key, raise_on_error=True)
        gemini_sdk.client_class.assert_not_called()
        # Default callers (the connection test passes ``api_key or ""``) keep
        # handing the key to genai.Client and its environment fallback.
        assert await fetch_models_from_provider("gemini", api_key) == []
        assert gemini_sdk.client_class.call_args.kwargs["api_key"] == api_key


_CALLER_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "caller_scope", default="unset"
)


def _zhipu_text_response():
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = MagicMock(content="ok", tool_calls=None)
    return MagicMock(choices=[choice])


class TestZhipuExecutorContext:
    @pytest.fixture
    def zhipu(self):
        client = MagicMock()
        with patch(
            "xagent.core.model.chat.basic.zhipu.ZhipuAiClient", return_value=client
        ):
            llm = ZhipuLLM(api_key="sk-test", abilities=["chat", "vision"])
        llm._client = client
        return llm, client

    async def test_chat_and_vision_run_in_the_callers_context(self, zhipu):
        llm, client = zhipu
        seen: list[str] = []

        def create(**kwargs):
            seen.append(_CALLER_SCOPE.get())
            return _zhipu_text_response()

        client.chat.completions.create.side_effect = create
        token = _CALLER_SCOPE.set("caller-scope")
        try:
            await llm.chat([{"role": "user", "content": "hi"}])
            await llm.vision_chat([{"role": "user", "content": "hi"}])
        finally:
            _CALLER_SCOPE.reset(token)
        assert seen == ["caller-scope", "caller-scope"]

    async def test_the_stream_producer_runs_in_the_callers_context(self, zhipu):
        llm, client = zhipu
        seen: list[str] = []

        def create(**kwargs):
            seen.append(_CALLER_SCOPE.get())
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="ok", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            ]

        client.chat.completions.create.side_effect = create
        token = _CALLER_SCOPE.set("caller-scope")
        try:
            async for _chunk in llm.stream_chat([{"role": "user", "content": "hi"}]):
                pass
        finally:
            _CALLER_SCOPE.reset(token)
        assert seen == ["caller-scope"]
