"""Router LLM: a virtual model that delegates to xrouter-llm for selection.

On every call it asks the xrouter-llm decision service to pick ONE concrete
model for the prompt, then dispatches the actual completion to the matching
backend: Claude models go through the Anthropic path, everything else through
the OpenAI-compatible path. Downstream credentials come from the usual env
(`ANTHROPIC_API_KEY` for Claude, `OPENAI_API_KEY` / `OPENAI_BASE_URL` for the
rest -- point the latter at OpenRouter to serve the registry models).
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, List, Optional

import httpx

from ....model import ChatModelConfig
from ..types import StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)

_DEFAULT_ROUTER_BASE_URL = "http://127.0.0.1:8080"


class RouterLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "auto",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        timeout: float = 180.0,
        abilities: Optional[List[str]] = None,
    ) -> None:
        # model_name doubles as the xrouter-llm router config name (e.g. "auto").
        self._config_name = model_name or "auto"
        self._base_url = (
            base_url or os.getenv("XAGENT_XROUTER_BASE_URL") or _DEFAULT_ROUTER_BASE_URL
        ).rstrip("/")
        self._api_key = api_key
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self._abilities = abilities or [
            "chat",
            "tool_calling",
            "vision",
            "thinking_mode",
        ]
        self._route_timeout = float(os.getenv("XAGENT_ROUTER_TIMEOUT", "10"))
        self._fallback_model = os.getenv("XAGENT_ROUTER_FALLBACK_MODEL") or None

    # ---- BaseLLM interface --------------------------------------------------
    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._config_name

    @property
    def supports_thinking_mode(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        llm = await self._resolve(messages)
        return await llm.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )

    async def vision_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        llm = await self._resolve(messages)
        return await llm.vision_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        llm = await self._resolve(messages)
        async for chunk in llm.stream_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        ):
            yield chunk

    # ---- Routing ------------------------------------------------------------
    async def _resolve(self, messages: list[dict[str, Any]]) -> BaseLLM:
        model_id = await self._select_model(self._extract_prompt(messages))
        provider, downstream_name = self._dispatch(model_id)
        logger.info("xrouter selected %s -> provider=%s", model_id, provider)
        # Lazy import avoids a circular import (adapter imports this module).
        from .adapter import create_base_llm

        config = ChatModelConfig(
            id=f"router:{model_id}",
            model_name=downstream_name,
            model_provider=provider,
            base_url=None,  # use env defaults (OPENAI_BASE_URL / Anthropic official)
            api_key=None,
            default_temperature=self.default_temperature,
            default_max_tokens=self.default_max_tokens,
            timeout=self.timeout,
            abilities=self._abilities,
        )
        return create_base_llm(config)

    async def _select_model(self, prompt: str) -> str:
        payload = {"prompt": prompt, "config": self._config_name}
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self._route_timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/route", json=payload, headers=headers
                )
                resp.raise_for_status()
                selected = resp.json().get("selected") or []
        except Exception as exc:  # noqa: BLE001 - routing must not hard-crash the agent
            if self._fallback_model:
                logger.warning(
                    "xrouter route failed (%s); using fallback %s",
                    exc,
                    self._fallback_model,
                )
                return self._fallback_model
            raise RuntimeError(
                f"xrouter-llm routing failed against {self._base_url}: {exc}. "
                "Set XAGENT_ROUTER_FALLBACK_MODEL to degrade gracefully."
            ) from exc
        if not selected:
            if self._fallback_model:
                return self._fallback_model
            raise RuntimeError("xrouter-llm returned no selected model")
        return str(selected[0])

    @staticmethod
    def _dispatch(model_id: str) -> tuple[str, str]:
        """Map a chosen model id to (provider, downstream model name).

        Claude and DeepSeek models go to their official APIs (claude / deepseek
        providers); everything else goes through the OpenAI-compatible path
        (point OPENAI_BASE_URL at OpenRouter). The model id is passed through as
        the downstream model name, so registry ids must equal the callable model
        string (Anthropic model name, DeepSeek model name, or OpenRouter slug).
        """
        lowered = model_id.lower()
        if "claude" in lowered:
            return "claude", model_id
        if "deepseek" in lowered:
            return "deepseek", model_id
        return "openai", model_id

    @staticmethod
    def _extract_prompt(messages: list[dict[str, Any]]) -> str:
        """Use the latest user message as the routing prompt."""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if texts:
                    return "\n".join(texts)
        # Fallback: concatenate any string content.
        return "\n".join(
            m["content"] for m in messages if isinstance(m.get("content"), str)
        )
