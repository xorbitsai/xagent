from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.basic import router as router_module
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.router import RouterLLM


def test_openrouter_auto_returns_router_llm():
    # "auto" is now an OpenRouter model name (no separate router provider).
    config = ChatModelConfig(
        id="auto-model",
        model_provider="openrouter",
        model_name="auto",
    )

    llm = create_base_llm(config)

    assert isinstance(llm, RouterLLM)
    assert llm.model_name == "auto"


def test_openrouter_non_auto_is_not_router_llm():
    # A normal OpenRouter slug is dispatched directly, not via xrouter.
    config = ChatModelConfig(
        id="or-claude",
        model_provider="openrouter",
        model_name="anthropic/claude-opus-4.8",
    )
    llm = create_base_llm(config)
    assert not isinstance(getattr(llm, "_inner", llm), RouterLLM)


def test_auto_is_curated_under_openrouter():
    from xagent.core.model.providers import curated_models_for_provider

    assert "auto" in curated_models_for_provider("openrouter")


def test_router_does_not_advertise_unrouted_capabilities():
    llm = RouterLLM(
        model_name="auto",
        abilities=["chat", "tool_calling", "vision", "thinking_mode"],
    )

    assert llm.abilities == ["chat", "tool_calling"]
    assert llm.supports_thinking_mode is False


async def test_router_dispatches_chosen_slug_through_downstream_resolver():
    # The xrouter-llm registry returns canonical OpenRouter slugs, so the chosen
    # id is passed straight to the injected resolver and its LLM returned as-is.
    seen: dict[str, str] = {}

    def resolver(slug: str):
        seen["slug"] = slug
        return "DOWNSTREAM_LLM"

    llm = RouterLLM(model_name="auto", downstream_resolver=resolver)

    async def fake_select(_prompt: str) -> str:
        return "anthropic/claude-opus-4.8"

    llm._select_model = fake_select  # type: ignore[assignment]

    result = await llm._resolve([{"role": "user", "content": "hi"}])
    assert seen["slug"] == "anthropic/claude-opus-4.8"
    assert result == "DOWNSTREAM_LLM"


async def test_router_fallback_uses_openrouter_config(monkeypatch):
    # Test-connection paths may not inject a downstream resolver. The fallback
    # still has to run chosen OpenRouter slugs against OpenRouter, not OpenAI.
    from xagent.core.model.chat.basic import adapter as adapter_module

    seen: dict[str, object] = {}

    def fake_create_base_llm(config):
        seen["config"] = config
        return "FALLBACK_LLM"

    monkeypatch.setattr(adapter_module, "create_base_llm", fake_create_base_llm)

    llm = RouterLLM(
        model_name="auto",
        api_key="configured-key",
        default_temperature=0.2,
        default_max_tokens=123,
    )

    async def fake_select(_prompt: str) -> str:
        return "deepseek/deepseek-v4-flash"

    llm._select_model = fake_select  # type: ignore[assignment]

    result = await llm._resolve([{"role": "user", "content": "hi"}])

    config = seen["config"]
    assert result == "FALLBACK_LLM"
    assert config.model_provider == "openrouter"
    assert config.model_name == "deepseek/deepseek-v4-flash"
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.api_key == "configured-key"
    assert config.default_temperature == 0.2
    assert config.default_max_tokens == 123


async def test_router_selects_in_process_via_service(monkeypatch):
    # _select_model runs the in-process RoutingService (no HTTP) and returns the
    # first selected slug.
    class _FakeService:
        def route(self, prompt, *, config_name):
            assert prompt == "hello"
            assert config_name == "auto"
            return {"selected": ["openai/gpt-5.5"]}

    monkeypatch.setattr(router_module, "_get_service", lambda: _FakeService())

    llm = RouterLLM(model_name="auto")
    assert await llm._select_model("hello") == "openai/gpt-5.5"


async def test_router_routes_on_active_goal_not_scaffold(monkeypatch):
    # The agent sets the active goal (user request / DAG step); routing must use
    # it, not the scaffolded message this LLM call carries.
    from xagent.core.model.intent import goal_scope

    seen: dict[str, str] = {}

    async def fake_select(prompt: str) -> str:
        seen["prompt"] = prompt
        return "openai/gpt-5.5"

    llm = RouterLLM(model_name="auto", downstream_resolver=lambda s: "DOWNSTREAM")
    llm._select_model = fake_select  # type: ignore[assignment]

    with goal_scope("你好"):
        await llm._resolve(
            [
                {
                    "role": "user",
                    "content": "## User Task\n你好\n\n## Available Skills\n...",
                }
            ]
        )

    assert seen["prompt"] == "你好"


async def test_router_uses_fallback_when_routing_fails(monkeypatch):
    monkeypatch.setenv("XAGENT_ROUTER_FALLBACK_MODEL", "anthropic/claude-opus-4.8")

    def _boom():
        raise RuntimeError("registry missing")

    monkeypatch.setattr(router_module, "_get_service", _boom)

    llm = RouterLLM(model_name="auto")
    assert await llm._select_model("hello") == "anthropic/claude-opus-4.8"


def test_router_extract_prompt_uses_latest_user_message():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": [{"type": "text", "text": "latest question"}]},
    ]
    assert RouterLLM._extract_prompt(messages) == "latest question"
