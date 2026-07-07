from __future__ import annotations

from typing import Any, Optional

from .base import ASRResult, ASRSegment, BaseASR
from .elevenlabs import ElevenLabsASR
from .xinference import XinferenceASR


def get_asr_model_instance(db_model: Any) -> BaseASR:
    """
    Create a BaseASR instance from a database model record.

    Args:
        db_model: Database model instance with fields: model_name, model_provider,
                  api_key, base_url, abilities, timeout, max_retries

    Returns:
        BaseASR instance

    Raises:
        ValueError: If provider is not supported or required fields are missing
    """
    provider = str(db_model.model_provider).lower()
    model_name = str(db_model.model_name)
    api_key = str(db_model.api_key) if db_model.api_key else None
    base_url = str(db_model.base_url) if db_model.base_url else None

    if provider == "xinference":
        return XinferenceASR(
            model=model_name,
            model_uid=model_name,
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "elevenlabs":
        return ElevenLabsASR(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unsupported ASR provider: {provider}. "
            "Supported providers: xinference, elevenlabs."
        )


def get_asr_model(
    provider: str = "xinference",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseASR:
    """
    Factory function to get ASR model instance by provider.

    Args:
        provider: Model provider name (e.g., 'xinference', 'elevenlabs')
        model: Model name/identifier
        api_key: API key for the provider
        **kwargs: Additional provider-specific parameters

    Returns:
        ASR model instance

    Raises:
        ValueError: If provider is not supported
    """
    if provider is None:
        raise ValueError("ASR provider cannot be None.")

    normalized_provider = provider.lower().strip()
    if normalized_provider == "xinference":
        return XinferenceASR(
            model=model or "whisper-base",
            api_key=api_key,
            **kwargs,
        )
    elif normalized_provider == "elevenlabs":
        return ElevenLabsASR(
            model=model or "scribe_v2",
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported ASR provider: {provider}. "
            "Supported providers: xinference, elevenlabs"
        )


__all__ = [
    "get_asr_model_instance",
    "get_asr_model",
    "BaseASR",
    "ASRResult",
    "ASRSegment",
    "XinferenceASR",
    "ElevenLabsASR",
]
