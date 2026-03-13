from __future__ import annotations

from typing import Any, Optional

from .base import ASRResult, ASRSegment, BaseASR
from .xinference import XinferenceASR


def get_asr_model(
    provider: str = "xinference",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseASR:
    """
    Factory function to get ASR model instance by provider.

    Args:
        provider: Model provider name (e.g., 'xinference')
        model: Model name/identifier
        api_key: API key for the provider
        **kwargs: Additional provider-specific parameters

    Returns:
        ASR model instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider == "xinference":
        return XinferenceASR(
            model=model or "whisper-base",
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported ASR provider: {provider}. "
            "Currently only 'xinference' is supported."
        )


__all__ = [
    "get_asr_model",
    "BaseASR",
    "ASRResult",
    "ASRSegment",
    "XinferenceASR",
]
