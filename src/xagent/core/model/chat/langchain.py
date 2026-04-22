"""Model adapter"""

import os
from typing import Any, Callable, Optional, Sequence, Union

from langchain.tools import BaseTool
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from ...model import ChatModelConfig, ModelConfig
from ...retry import ExponentialBackoff, RetryStrategy, create_retry_wrapper
from ..providers import provider_compatibility_for_provider
from .error import retry_on


class ChatModelRetryWrapper(Runnable):
    def __init__(
        self,
        model: BaseChatModel,
        strategy: RetryStrategy,
        max_retries: int = 10,
    ):
        self._retry_wrapper = create_retry_wrapper(
            model,
            Runnable,  # type: ignore[type-abstract]
            retry_methods={"invoke", "ainvoke"},
            strategy=strategy,
            max_retries=max_retries,
            retry_on=retry_on,
        )
        self.model = model
        self.strategy = strategy
        self.max_retries = max_retries

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        return self._retry_wrapper.invoke(input, config, **kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        return await self._retry_wrapper.ainvoke(input, config, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[
            Union[dict[str, Any], type, Callable, BaseTool]  # noqa: UP006
        ],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
    ) -> Runnable:
        model = self.model.bind_tools(tools)
        return create_retry_wrapper(
            model,
            Runnable,  # type: ignore[type-abstract]
            retry_methods={"invoke", "ainvoke"},
            strategy=self.strategy,
            max_retries=self.max_retries,
            retry_on=retry_on,
        )

    def with_structured_output(
        self,
        schema: Union[dict, type],  # noqa: UP006
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:  # noqa: UP006
        model = self.model.with_structured_output(schema)
        return create_retry_wrapper(
            model,
            Runnable,  # type: ignore[type-abstract]
            retry_methods={"invoke", "ainvoke"},
            strategy=self.strategy,
            max_retries=self.max_retries,
            retry_on=retry_on,
        )


def create_base_chat_model(
    model: ModelConfig, temperature: float | None
) -> BaseChatModel:
    """
    Adapts a custom LLM instance to its corresponding LangChain Chat Model class
    """

    if not isinstance(model, ChatModelConfig):
        raise TypeError(f"Unsupported Chat model type: {type(model).__name__}")

    temp = temperature if temperature is not None else model.default_temperature

    compatibility = provider_compatibility_for_provider(model.model_provider)

    if model.model_provider == "openai" or compatibility == "openai_compatible":
        return ChatOpenAI(
            model=model.model_name,
            temperature=temp,
            max_tokens=model.default_max_tokens,
            api_key=model.api_key,
            base_url=model.base_url,
            timeout=model.timeout,
        )
    elif model.model_provider == "zhipu":
        return ChatZhipuAI(
            model=model.model_name,
            temperature=temp,
            max_tokens=model.default_max_tokens,
            api_key=model.api_key,
            api_base=model.base_url,
        )
    elif model.model_provider == "azure_openai":
        api_version = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview")
        return AzureChatOpenAI(
            deployment_name=model.model_name,
            azure_endpoint=model.base_url,
            api_key=model.api_key,
            api_version=api_version,
            temperature=temp,
            max_tokens=model.default_max_tokens,
            timeout=model.timeout,
        )
    else:
        raise TypeError(f"Unsupported LLM model provider: {model.model_provider}")


def create_base_chat_model_with_retry(
    model: ModelConfig, temperature: float | None
) -> ChatModelRetryWrapper:
    chat_model = create_base_chat_model(model, temperature)
    strategy = ExponentialBackoff()
    return ChatModelRetryWrapper(chat_model, strategy, max_retries=model.max_retries)
