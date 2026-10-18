"""Own the guarded HTTP client injected into an ElevenLabs SDK instance."""

from typing import Any, Callable

import httpx

from .public_http import (
    close_async_client_on_error,
    create_public_model_async_client,
    create_public_model_sync_client,
)


def create_public_elevenlabs_client(
    factory: Callable[..., Any], *, asynchronous: bool, **kwargs: Any
) -> Any:
    http_client = (
        create_public_model_async_client(timeout=60)
        if asynchronous
        else create_public_model_sync_client(timeout=60)
    )
    try:
        client = factory(httpx_client=http_client, **kwargs)
    except BaseException:
        if isinstance(http_client, httpx.AsyncClient):
            close_async_client_on_error(http_client)
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
