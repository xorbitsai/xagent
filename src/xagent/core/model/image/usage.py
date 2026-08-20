"""Media-usage recording helper for image models.

Image providers all return a result dict with an optional ``usage`` payload
(provider-specific shape). This helper normalises that into a single
``add_media_usage`` call so every provider records usage the same way, keeping
image generation/editing metered alongside LLM tokens.
"""

from __future__ import annotations

import logging
from typing import Any

from ..chat.token_context import MediaCallType, MediaUnit, add_media_usage

logger = logging.getLogger(__name__)


def _read(payload: Any, *names: str) -> int:
    """Best-effort int read of the first present field from a usage payload."""
    if payload is None:
        return 0
    for name in names:
        value = (
            payload.get(name)
            if isinstance(payload, dict)
            else getattr(payload, name, None)
        )
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def record_image_usage(
    result: dict[str, Any],
    *,
    model_name: str = "",
    model_id: str = "",
    call_type: MediaCallType | str = MediaCallType.GENERATE_IMAGE,
    image_count: int = 1,
    resolution: str = "",
) -> None:
    """Record one image generation/edit call on the current token context.

    Best-effort: any failure here is swallowed so accounting can never break the
    underlying image call. ``result`` is the provider return dict; ``usage`` in
    it (when present) may carry token counts some providers report (e.g. Gemini,
    OpenAI gpt-image). ``resolution`` is the size tier ("1K"/"2K"/"4K" or
    "1024x1024") so the billing layer can price by (model, resolution); the
    real image tokens (when present) let a token-based price take precedence.
    ``image_count`` must be the request's ``n`` so multi-image requests are not
    under-billed as a single image.
    """
    try:
        usage = result.get("usage") if isinstance(result, dict) else None
        input_tokens = _read(usage, "prompt_tokens", "input_tokens")
        output_tokens = _read(usage, "completion_tokens", "output_tokens")
        try:
            count = max(0, int(image_count))
        except (TypeError, ValueError):
            count = 1
        add_media_usage(
            unit=MediaUnit.IMAGES,
            quantity=count,
            model=model_name,
            model_id=model_id,
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            resolution=resolution,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to record image usage: %s", e)
