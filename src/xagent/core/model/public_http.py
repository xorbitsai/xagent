"""Operator proxy policy and ownership for managed model SDK clients."""

import asyncio
from collections.abc import Callable
from typing import Any

import aiohttp
import httpx

from ..tools.core.web_content import get_trusted_proxy_url
from ..utils.public_aiohttp import create_public_aiohttp_session
from ..utils.public_http_transport import (
    UnsafePublicHttpTarget,
    create_public_async_client,
    create_public_sync_client,
    public_http_client_options,
)
from ..utils.security import PrivateNetworkHostError
from .network_policy import ModelNetworkPolicy


def create_public_model_async_client(**kwargs: Any) -> httpx.AsyncClient:
    kwargs.setdefault("follow_redirects", True)
    return create_public_async_client(
        proxy_url=get_trusted_proxy_url(), _policy=ModelNetworkPolicy(), **kwargs
    )


def create_public_model_sync_client(**kwargs: Any) -> httpx.Client:
    kwargs.setdefault("follow_redirects", True)
    return create_public_sync_client(
        proxy_url=get_trusted_proxy_url(), _policy=ModelNetworkPolicy(), **kwargs
    )


def public_model_http_client_options(*, asynchronous: bool) -> dict[str, Any]:
    return {
        **public_http_client_options(
            asynchronous=asynchronous,
            proxy_url=get_trusted_proxy_url(),
            _policy=ModelNetworkPolicy(),
        ),
        "follow_redirects": True,
    }


def create_public_model_aiohttp_session(**kwargs: Any) -> aiohttp.ClientSession:
    return create_public_aiohttp_session(
        proxy_url=get_trusted_proxy_url(), _policy=ModelNetworkPolicy(), **kwargs
    )


def create_owned_async_sdk_client(
    factory: Callable[..., Any], *, timeout: Any, **kwargs: Any
) -> Any:
    http_client = create_public_model_async_client(timeout=timeout)
    try:
        return factory(http_client=http_client, timeout=timeout, **kwargs)
    except BaseException:
        # Construction is synchronous but close is asynchronous. Preserve
        # ownership even when a constructor rejects configuration before use.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(http_client.aclose())
        else:
            loop.create_task(http_client.aclose())
        raise


def is_model_network_policy_error(error: BaseException) -> bool:
    """Find policy refusals even when an SDK wraps them as connection errors."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (UnsafePublicHttpTarget, PrivateNetworkHostError)):
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False
