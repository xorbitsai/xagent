"""TTS model adapter factory."""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseTTS
from .xinference import XinferenceTTS


def get_tts_model_instance(db_model: Any) -> BaseTTS:
    """
    Create a BaseTTS instance from a database model record.

    Args:
        db_model: Database model instance with fields: model_name, model_provider,
                  api_key, base_url, abilities, timeout, max_retries

    Returns:
        BaseTTS instance

    Raises:
        ValueError: If provider is not supported or required fields are missing
    """
    provider = str(db_model.model_provider).lower()
    model_name = str(db_model.model_name)
    api_key = str(db_model.api_key) if db_model.api_key else None
    base_url = str(db_model.base_url) if db_model.base_url else None

    if provider == "xinference":
        return XinferenceTTS(
            model=model_name,
            model_uid=model_name,
            base_url=base_url,
            api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unsupported TTS provider: {provider}. "
            "Currently only 'xinference' is supported."
        )


def get_tts_model(
    provider: str = "xinference",
    model: Optional[str] = None,
    **kwargs: Any,
) -> BaseTTS:
    """
    Get a TTS model instance by provider.

    Args:
        provider: TTS provider name ('xinference')
        model: Model name (provider-specific)
        **kwargs: Additional provider-specific parameters

    Returns:
        A TTS model instance

    Raises:
        ValueError: If provider is not supported

    Example:
        >>> # Get Xinference TTS model
        >>> tts = get_tts_model(
        ...     provider="xinference",
        ...     model="chat-tts",
        ...     base_url="http://localhost:9997"
        ... )
        >>> audio = tts.synthesize("Hello, world!")
    """
    if provider == "xinference":
        return XinferenceTTS(model=model or "chat-tts", **kwargs)
    else:
        raise ValueError(
            f"Unsupported TTS provider: {provider}. Supported providers: xinference"
        )


__all__ = [
    "get_tts_model_instance",
    "get_tts_model",
    "XinferenceTTS",
]
