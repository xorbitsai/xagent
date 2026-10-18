"""Own the guarded HTTP client injected into an ElevenLabs SDK instance."""

import asyncio
from typing import Any, Callable

import httpx

from .public_http import (
    create_public_model_async_client,
    create_public_model_sync_client,
)


def create_public_elevenlabs_client(
    factory: Callable[..., Any], *, asynchronous: bool, **kwargs: Any
) -> Any:
    http_client = (
        create_public_model_async_client(timeout=60, follow_redirects=True)
        if asynchronous
        else create_public_model_sync_client(timeout=60, follow_redirects=True)
    )
    try:
        client = factory(httpx_client=http_client, **kwargs)
    except BaseException:
        if isinstance(http_client, httpx.AsyncClient):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(http_client.aclose())
            else:
                loop.create_task(http_client.aclose())
        else:
            http_client.close()
        raise
    client._xagent_public_http_client = http_client
    return client


async def close_public_elevenlabs_http_client(client: Any) -> None:
    http_client = getattr(client, "_xagent_public_http_client", None)
    if isinstance(http_client, httpx.AsyncClient):
        await http_client.aclose()
    elif isinstance(http_client, httpx.Client):
        http_client.close()


def close_public_elevenlabs_sync_http_client(client: Any) -> None:
    http_client = getattr(client, "_xagent_public_http_client", None)
    if isinstance(http_client, httpx.Client):
        http_client.close()
