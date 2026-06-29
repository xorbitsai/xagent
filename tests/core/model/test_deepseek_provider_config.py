from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.deepseek import DeepSeekLLM
from xagent.core.model.providers import (
    curated_models_for_provider,
    default_base_url_for_provider,
    provider_compatibility_for_provider,
)


def test_create_base_llm_returns_deepseek_llm():
    config = ChatModelConfig(
        id="deepseek-model",
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        api_key="test-api-key",
    )

    llm = create_base_llm(config)

    assert hasattr(llm, "_inner")
    assert isinstance(llm._inner, DeepSeekLLM)


def test_create_base_llm_accepts_canonicalized_deepseek_provider():
    config = ChatModelConfig(
        id="deepseek-model",
        model_provider=" DeepSeek ",
        model_name="deepseek-v4-flash",
        api_key="test-api-key",
    )

    llm = create_base_llm(config)

    assert hasattr(llm, "_inner")
    assert isinstance(llm._inner, DeepSeekLLM)


def test_create_base_llm_uses_deepseek_base_url_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
    config = ChatModelConfig(
        id="deepseek-model",
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        api_key="test-api-key",
    )

    llm = create_base_llm(config)

    assert hasattr(llm, "_inner")
    assert isinstance(llm._inner, DeepSeekLLM)
    assert llm._inner.base_url == "https://deepseek.example.com"


def test_deepseek_provider_has_no_openai_compatibility_marker():
    assert provider_compatibility_for_provider("deepseek") is None


def test_deepseek_default_base_url():
    assert default_base_url_for_provider("deepseek") == "https://api.deepseek.com"


def test_deepseek_curated_models_are_limited_to_v4():
    assert curated_models_for_provider("deepseek") == (
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    )
