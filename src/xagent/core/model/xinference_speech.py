"""Loaded audio capabilities from the configured Xinference deployment."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import quote

import aiohttp

from .public_http import create_public_model_aiohttp_session


def audio_model_url(provider: Any, suffix: str) -> str:
    uid = quote(provider._model_uid, safe="")
    return f"{provider.base_url}/v1/models/{uid}/{suffix}"


def audio_headers(provider: Any) -> dict[str, str]:
    return {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}


async def fetch_audio_capabilities(provider: Any) -> dict[str, Any]:
    async with create_public_model_aiohttp_session(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as client:
        async with client.get(
            audio_model_url(provider, "audio/capabilities"),
            headers=audio_headers(provider),
        ) as response:
            if response.status in {404, 405}:
                return {}
            response.raise_for_status()
            try:
                data = await response.content.readexactly(65537)
            except asyncio.IncompleteReadError as exc:
                data = exc.partial
            if len(data) > 65536:
                raise RuntimeError(
                    "Xinference audio capabilities exceeded the size limit"
                )
            try:
                result = json.loads(data)
            except (ValueError, UnicodeError) as exc:
                raise RuntimeError("Invalid Xinference audio capabilities") from exc
            if not isinstance(result, dict):
                raise RuntimeError("Invalid Xinference audio capabilities")
            if any(
                not isinstance(result.get(role, {}), dict) for role in ("asr", "tts")
            ):
                raise RuntimeError("Invalid Xinference audio capabilities")
            return result
