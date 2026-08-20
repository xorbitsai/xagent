"""Best-effort media-usage recording for media generation tools.

TTS/ASR/video/music/sound-effect models don't return a normalised usage
payload the way image/chat models do, and their adapters are factory-only, so
the natural metering point is the tool call site where the request params and
result are both in scope. This helper wraps ``add_media_usage`` so a failure in
accounting can never break the underlying media call.

Metering invariants
-------------------
A usage record is only worth what its identity and unit are worth, so every
producer must satisfy all of these:

1. **Metering must survive adapter unwrapping.** Record on the object callers
   actually hold. Reaching past an adapter to its inner provider silently drops
   the metering — this is how rerank shipped entirely unbilled.
2. **Metering must survive thread boundaries.** ``ThreadPoolExecutor`` does not
   copy contextvars, so a worker gets a fresh empty ``TokenUsage`` unless the
   caller's is bound explicitly.
3. **The unit is a property of the modality, never of the response.** A
   duration-billed modality always reports seconds, recording ``quantity=0``
   when unmeasured rather than switching units.
4. **Never bill a placeholder identity.** ``"default"``, ``"None"`` and ``""``
   are not models; resolve through :func:`resolve_billing_model`.

Model identity convention
-------------------------
``model`` carries the human-readable **name**, ``model_id`` the configured
**id**. Populate both when known: the aggregator groups on
``model_id or model``, so a producer that leaves ``model_id`` empty while
another sets it splits one physical model into two un-mergeable billing rows.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, TypeGuard

from ...model.chat.token_context import MediaCallType, MediaUnit, add_media_usage

logger = logging.getLogger(__name__)

# Placeholders that must never reach a usage record as a model identity.
_PLACEHOLDER_MODEL_NAMES = {"", "none", "null", "default"}


def _usable_model_name(value: Any) -> TypeGuard[str]:
    """A real model identity, not a placeholder. TypeGuard so callers narrow."""
    return (
        isinstance(value, str) and value.strip().lower() not in _PLACEHOLDER_MODEL_NAMES
    )


def resolve_billing_model(
    configured_id: Optional[str],
    model: Any = None,
    *,
    fallback: str = "default",
) -> str:
    """Best available identity for a model, never a placeholder string.

    ``_configured_model_id``-style lookups return ``Optional[str]``, and passing
    that through ``str()`` records a model literally named ``"None"``. Prefer
    the configured id, fall back to the provider's own ``model_name``/``model``
    attribute, and only then to ``fallback``.
    """

    # The placeholder filter applies to the configured id too: a config that
    # literally names the model "default" or "none" must not be billed as one.
    if _usable_model_name(configured_id):
        return configured_id
    for attr in ("model_name", "model"):
        # Guarded individually: this runs *before* record_media_usage's own
        # try/except, so a provider whose model_name is a property that raises
        # would break the user's media call over an accounting lookup.
        try:
            value = getattr(model, attr, None)
        except Exception as e:  # noqa: BLE001
            logger.warning("Reading %s for billing identity failed: %s", attr, e)
            continue
        if _usable_model_name(value):
            return value
    return fallback


def coerce_duration(value: object) -> Optional[float]:
    """A positive duration in seconds, or None when unusable.

    Distinct from ``token_context._coerce_float``, which folds bad input to
    ``0.0``: here the caller must be able to tell "provider reported no
    duration" apart from "provider reported zero", because those take
    different metering branches.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    # ``inf > 0`` is True, so non-finite values would otherwise pass as a
    # usable duration and reach the record as a non-JSON-serialisable
    # quantity. Treat them as "no duration reported" instead.
    if not math.isfinite(seconds):
        logger.warning("Ignoring non-finite duration: %r", value)
        return None
    return seconds if seconds > 0 else None


def record_media_usage(
    unit: MediaUnit | str | None,
    quantity: float,
    *,
    model: str = "",
    model_id: str = "",
    call_type: MediaCallType | str | None = "",
    resolution: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    tokens_estimated: bool = False,
) -> None:
    """Record one media model call; swallow any error.

    Includes the ``ValueError`` ``add_media_usage`` raises for an unknown
    ``unit``/``call_type``: a metering bug must never break the media call the
    user actually asked for. The record is dropped and logged rather than
    written under a bogus billing dimension, which is unrepairable once
    persisted — so a typo surfaces as a missing row plus this warning, not as a
    silently mis-billed one.
    """
    try:
        add_media_usage(
            unit=unit,
            quantity=quantity,
            model=model,
            model_id=model_id,
            call_type=call_type,
            resolution=resolution,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_estimated=tokens_estimated,
        )
    except Exception as e:  # noqa: BLE001
        # exc_info: the expected case here is the deliberate validation
        # ValueError, but anything genuinely unexpected would otherwise surface
        # as one context-free line. Persisted billing rows cannot be repaired
        # after the fact, so a metering bug needs to be diagnosable from the log
        # alone.
        logger.warning(
            "Failed to record media usage: call_type=%r unit=%r quantity=%r "
            "model=%r model_id=%r: %s",
            call_type,
            unit,
            quantity,
            model,
            model_id,
            e,
            exc_info=True,
        )


def record_media_seconds(
    seconds: Optional[float],
    *,
    model: str = "",
    model_id: str = "",
    call_type: MediaCallType | str | None = "",
) -> None:
    """Record a duration-billed media call, keeping the unit stable.

    Duration-billed modalities (video/ASR/music/sound effect) must always
    report ``MediaUnit.SECONDS``: a price table keyed on (model, unit) breaks
    if the same model sometimes reports "requests" just because the provider
    omitted a duration. When the duration is unknown the call is still recorded
    — as ``seconds`` with ``quantity=0`` and a warning — so the event is
    visible to billing as unmeasured rather than silently mis-dimensioned.
    """
    if seconds is None:
        logger.warning(
            "No duration reported for %s call on model %r; recording 0 seconds "
            "(call happened but is unmeasured)",
            call_type,
            model,
        )
    record_media_usage(
        MediaUnit.SECONDS,
        seconds or 0.0,
        model=model,
        model_id=model_id,
        call_type=call_type,
    )
