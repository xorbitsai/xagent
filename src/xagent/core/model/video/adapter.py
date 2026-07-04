from typing import Any

import aiohttp
import httpx
import requests

from ...model import VideoModelConfig
from ...model.providers import canonical_provider_name
from ...retry import create_retry_wrapper
from .ark import ArkVideoModel
from .base import BaseVideoModel
from .xinference import XinferenceVideoModel


def retry_on(exc: Exception) -> bool:
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status == 429 or 500 <= exc.status < 600
    if isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
            httpx.TimeoutException,
            httpx.NetworkError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or 500 <= status_code < 600
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        status_code = exc.response.status_code
        return status_code == 429 or 500 <= status_code < 600
    cause = exc.__cause__
    if isinstance(cause, Exception) and cause is not exc:
        return retry_on(cause)
    if isinstance(exc, (TimeoutError, ValueError, RuntimeError)):
        return False
    return False


def get_video_model_instance(db_model: Any) -> BaseVideoModel:
    provider = canonical_provider_name(str(db_model.model_provider))
    model_name = str(db_model.model_name)
    api_key = str(db_model.api_key) if db_model.api_key else None
    base_url = str(db_model.base_url) if db_model.base_url else None
    abilities = list(db_model.abilities) if db_model.abilities else ["generate"]
    timeout = getattr(db_model, "timeout", 1800.0) or 1800.0
    max_retries = getattr(db_model, "max_retries", 3) or 3

    config = VideoModelConfig(
        id=f"{model_name}-{provider}",
        model_name=model_name,
        model_provider=provider,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        abilities=abilities,
        max_retries=max_retries,
    )
    return create_video_model(config)


def create_video_model(model_config: VideoModelConfig) -> BaseVideoModel:
    if not isinstance(model_config, VideoModelConfig):
        raise TypeError(f"Invalid model type: {type(model_config).__name__}")

    provider = canonical_provider_name(model_config.model_provider)
    if provider not in {"volcengine-ark", "byteplus-ark", "xinference"}:
        raise ValueError(f"Unsupported video model provider: {provider}")

    if provider == "xinference":
        model: BaseVideoModel = XinferenceVideoModel(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            timeout=model_config.timeout,
            abilities=model_config.abilities,
        )
    else:
        model = ArkVideoModel(
            model_name=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            timeout=model_config.timeout,
            abilities=model_config.abilities,
            model_provider=provider,
        )

    return create_retry_wrapper(
        model,
        BaseVideoModel,  # type: ignore[type-abstract]
        retry_methods={"create_video_task", "get_video_task", "generate_video"},
        max_retries=model_config.max_retries,
        retry_on=retry_on,
    )
