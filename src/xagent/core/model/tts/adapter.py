"""TTS model adapter factory."""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseTTS
from .xinference import XinferenceTTS


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


__all__ = ["get_tts_model", "XinferenceTTS"]
