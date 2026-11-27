"""OpenRouter saved policy must not silently disappear in LangChain."""

import pytest

from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.langchain import create_base_chat_model


@pytest.mark.parametrize(
    "policy",
    [
        {"openrouter_provider_only": ["together/fp8"]},
        {"openrouter_zdr": True},
        {"openrouter_data_collection": "deny"},
    ],
)
def test_policy_bearing_openrouter_model_fails_before_langchain_client(mocker, policy):
    client_factory = mocker.patch("xagent.core.model.chat.langchain.ChatOpenAI")
    config = ChatModelConfig(
        id="regional-deepseek",
        model_provider="openrouter",
        model_name="deepseek/deepseek-v4.1-flash-1017",
        api_key="test-key",
        **policy,
    )

    with pytest.raises(ValueError, match="does not support OpenRouter routing policy"):
        create_base_chat_model(config, None)

    client_factory.assert_not_called()


def test_unrestricted_openrouter_model_still_builds_langchain_client(mocker):
    client_factory = mocker.patch("xagent.core.model.chat.langchain.ChatOpenAI")
    config = ChatModelConfig(
        id="ordinary-openrouter",
        model_provider="openrouter",
        model_name="deepseek/deepseek-v4.1-flash-1017",
        api_key="test-key",
    )

    create_base_chat_model(config, None)

    client_factory.assert_called_once()
