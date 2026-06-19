from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.router import RouterLLM
from xagent.core.model.providers import default_base_url_for_provider


def test_create_base_llm_returns_router_llm():
    config = ChatModelConfig(
        id="auto-model",
        model_provider="router",
        model_name="auto",
        base_url="http://127.0.0.1:8090",
    )

    llm = create_base_llm(config)

    assert hasattr(llm, "_inner")
    assert isinstance(llm._inner, RouterLLM)
    assert llm._inner.model_name == "auto"


def test_router_default_base_url():
    assert default_base_url_for_provider("router") == "http://127.0.0.1:8080"


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


def test_router_extract_prompt_uses_latest_user_message():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": [{"type": "text", "text": "latest question"}]},
    ]
    assert RouterLLM._extract_prompt(messages) == "latest question"


def test_router_base_url_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("XAGENT_XROUTER_BASE_URL", raising=False)
    llm = RouterLLM(model_name="auto")
    assert llm._base_url == "http://127.0.0.1:8080"


def test_router_base_url_uses_env_override(monkeypatch):
    monkeypatch.setenv("XAGENT_XROUTER_BASE_URL", "http://router.example:9000/")
    llm = RouterLLM(model_name="auto")
    assert llm._base_url == "http://router.example:9000"
