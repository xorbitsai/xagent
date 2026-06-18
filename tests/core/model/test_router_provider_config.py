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


def test_router_dispatch_official_vs_openrouter():
    # Claude and DeepSeek go to official providers; everything else openai/OpenRouter.
    assert RouterLLM._dispatch("claude-opus-4-8") == ("claude", "claude-opus-4-8")
    assert RouterLLM._dispatch("claude-sonnet-4-6") == ("claude", "claude-sonnet-4-6")
    assert RouterLLM._dispatch("deepseek-v4-pro") == ("deepseek", "deepseek-v4-pro")
    assert RouterLLM._dispatch("deepseek-v4-flash") == ("deepseek", "deepseek-v4-flash")
    assert RouterLLM._dispatch("z-ai/glm-5.2") == ("openai", "z-ai/glm-5.2")
    assert RouterLLM._dispatch("nvidia/nemotron-3-ultra-550b-a55b:free") == (
        "openai",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    )


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
