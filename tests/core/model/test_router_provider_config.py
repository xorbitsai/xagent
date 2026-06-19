from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.basic import router as router_module
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.model.providers import default_base_url_for_provider


def test_create_base_llm_returns_router_llm():
    config = ChatModelConfig(
        id="auto-model",
        model_provider="router",
        model_name="auto",
    )

    llm = create_base_llm(config)

    assert hasattr(llm, "_inner")
    assert isinstance(llm._inner, RouterLLM)
    assert llm._inner.model_name == "auto"


def test_router_provider_has_no_base_url():
    # Routing is in-process now; the router provider needs no service URL.
    assert default_base_url_for_provider("router") is None


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
