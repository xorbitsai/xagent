"""Media-usage recording helper for image models.

Image providers all return a result dict with an optional ``usage`` payload
(provider-specific shape). This helper normalises that into a single
``add_media_usage`` call so every provider records usage the same way, keeping
image generation/editing metered alongside LLM tokens.

Sanitising is deliberately *not* done here. ``TokenUsage.record_media_call`` is
the write boundary and already rejects booleans, negatives, non-finite values
and overflowing integers while still writing the row -- a zero-quantity row is
how that layer says "this provider call happened but is unmeasured", which is
strictly better than losing a billable call. Coercing first here would defeat
those guards: ``int(True)`` is ``1`` (billing a JSON boolean as one token,
which the boundary deliberately reads as ``0``) and ``int(float("inf"))``
raises ``OverflowError``, which this module's best-effort handler would catch
*before* the row was written, dropping the whole billable record to salvage one
bad field. So raw provider values are forwarded untouched.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from ..chat.token_context import MediaCallType, add_media_usage

logger = logging.getLogger(__name__)

# "Argument not supplied", distinct from any value a provider could report.
_ABSENT = object()

# Upper bound for a provider-reported count or pixel dimension. No image
# response means more than this, and the values feed a billing quantity and an
# aggregate key: an unbounded integer becomes a several-hundred-character
# resolution string that joins no price table, and a count too large for a float
# is silently folded to a quantity of 0 by the write boundary -- billing nothing
# for a call that returned a real image. Rejecting it instead falls back to the
# request's own value, which is bounded.
_MAX_REPORTED = 1_000_000

# Upper bound for a provider-reported token count. Generous -- far above any
# real image request -- but finite, so an absurd provider integer neither
# shadows a usable later alias nor lands in the persisted row as a
# several-hundred-digit number.
_MAX_TOKENS = 1_000_000_000


def _read_raw(payload: Any, *names: str) -> Any:
    """The first usable alias in a usage payload, returned uncoerced.

    The chosen value is handed on exactly as the provider sent it, so the shared
    media write boundary -- the only sanitiser -- sees the real payload rather
    than something already folded to an int.

    Aliases are still tried in order past a value the boundary would discard:
    a payload reporting ``prompt_tokens: "bad"`` alongside a valid
    ``input_tokens: 7`` must bill 7, not 0. ``_countable`` decides only whether
    to keep looking; it never replaces the value that is returned.

    ``0`` is returned when no alias yields anything usable, matching "provider
    reported no tokens".
    """
    if payload is None:
        return 0
    fallback: Any = _ABSENT
    for name in names:
        try:
            value = (
                payload.get(name)
                if isinstance(payload, dict)
                else getattr(payload, name, None)
            )
        except Exception as e:  # noqa: BLE001
            # A payload whose attribute access raises (a mock, a property) must
            # not cost the row: this runs before add_media_usage.
            logger.warning("Reading usage field %r failed: %s", name, e)
            continue
        if value is None:
            continue
        if _countable(value):
            return value
        # Present but unusable. Remember the first such value so the boundary
        # still logs what the provider actually sent, and keep looking for a
        # usable alias rather than billing 0 while one exists.
        #
        # Tested against a sentinel, not `fallback == 0`: `fallback` holds a raw
        # provider value by then, and comparing an arbitrary object to 0 can
        # raise (a hostile __eq__) or return a non-bool (an array). Either would
        # escape this function into the caller's handler and drop the whole
        # billable row -- the failure this module exists to prevent.
        if fallback is _ABSENT:
            fallback = value
    return 0 if fallback is _ABSENT else fallback


def _countable(value: Any) -> bool:
    """Whether the boundary would bill ``value`` as a nonzero token count.

    The question is not "is this a number" but "would this survive the write
    boundary" -- because a value that reaches the boundary and is discarded
    there bills 0 while a usable later alias goes unread. So this mirrors every
    reduction the boundary applies to a token field:

    * ``bool`` -- malformed metadata, not a count of one
      (``_coerce_media_tokens``).
    * anything ``int()`` refuses, including ``float("inf")``, which raises
      ``OverflowError`` (``_coerce_int``).
    * negatives -- the boundary floors them with ``max(0, ...)``, so
      ``{"prompt_tokens": -5, "input_tokens": 9}`` billed 0 instead of 9.
    * fractions of either sign -- ``int()`` truncates toward zero, so ``-0.5``
      and ``1e-9`` both become ``0`` at the boundary while passing any check
      written against the truncated result.

    It must not be *stricter* than the boundary either. A value the boundary
    would bill but this rejects loses a real provider count to a later alias,
    which is the same under-billing in the other direction -- numeric strings
    are the case that caught this.
    * absurd magnitudes -- bounded like a reported count, so a 401-digit
      provider integer cannot shadow a real later alias or land in the row.

    Used only to decide whether to keep looking; it never alters the value that
    is returned.
    """
    if isinstance(value, bool):
        return False
    try:
        # The whole predicate is evaluated in here, result included. Computing
        # part of it inside and testing outside still leaves an escape: testing
        # a comparison *result* invokes its __bool__, which a provider object
        # could raise from, costing the entire row. Nothing about this decision
        # is worth a raise.
        #
        # The test is on int(value) alone, because the single question that
        # matters is "will the boundary bill a nonzero amount for this".
        # Deriving anything else from `value` has now been wrong twice, both
        # times by being *stricter* than the boundary and so losing a real
        # provider count to a later alias:
        #
        #   * comparing `value` itself rejects numeric strings, which cannot be
        #     ordered against an int -- yet the boundary bills them correctly
        #     ("123" -> 123).
        #   * a truncation test (int(value) != value) rejects 7.9,
        #     Decimal("7.5") and Fraction(15, 2), which the boundary bills as 7.
        #
        # A zero result is usable when the provider genuinely reported an
        # integral zero, and not otherwise. An explicit `0` is a real
        # measurement -- "this call used no tokens" -- and must stop the scan,
        # or a later alias silently overrides what the provider stated. But
        # -0.5 and 1e-9 also reach 0 through int(), and those must NOT stop it:
        # the boundary bills them 0 while a usable later alias goes unread.
        # The two are told apart by whether int() changed the value, which is
        # only asked for a zero result -- for anything nonzero, truncation is
        # exactly what the boundary does too, so asking there was what rejected
        # 7.9 and Decimal("7.5").
        as_int = int(value)
        if as_int == 0:
            return bool(value == 0)
        return 0 < as_int <= _MAX_TOKENS
    except Exception:  # noqa: BLE001
        # Broader than the boundary's own tuple on purpose: this runs one frame
        # earlier, and an object with a raising __int__, comparison or __bool__
        # escaping here would cost the entire row rather than one field.
        return False


def usable_image_count(value: Any) -> Optional[int]:
    """A provider-reported image count, or ``None`` when unusable.

    Distinct from the write boundary's coercion, which folds bad input to ``0``:
    a caller here must tell "the provider reported no count" apart from "the
    provider reported zero", because those take different metering branches --
    the former falls back to the request's ``n``, the latter is authoritative.

    Mirrors ``tools.core.media_usage.coerce_duration``; the two must stay in
    step on which inputs are rejected (bool, non-finite, non-numeric, and here
    also negative and fractional counts, which no image response can mean).

    Also used for pixel dimensions, which carry the same shape constraints: a
    non-negative whole number, never a bool or a non-finite value.
    """
    if value is None or isinstance(value, bool):
        return None
    # int first, and returned without a float round-trip: float(2**53 + 1) is
    # already lossy, so routing a plain int through float would silently round a
    # large value and hand back a number the provider never sent.
    if isinstance(value, int):
        if value < 0 or value > _MAX_REPORTED:
            logger.warning("Ignoring unusable provider image count: %r", value)
            return None
        return value
    try:
        count = float(value)
    except Exception:  # noqa: BLE001
        # Not just (TypeError, ValueError, OverflowError): a provider object
        # with a raising __float__ or __index__ would otherwise escape, and a
        # caller evaluating this outside a swallow would lose the whole row.
        return None
    if (
        not math.isfinite(count)
        or count < 0
        or count > _MAX_REPORTED
        or count != int(count)
    ):
        logger.warning("Ignoring unusable provider image count: %r", value)
        return None
    return int(count)


def _first_usable_count(payload: Any, *names: str) -> Optional[int]:
    """The first usable provider-reported count in ``payload``, else ``None``.

    ``None`` means "the provider reported nothing usable", which is distinct
    from a reported ``0`` -- that is authoritative and means the call was billed
    but produced nothing, which the write boundary records as a zero-quantity
    row. Only the former falls back to the request's value.
    """
    if not isinstance(payload, dict):
        return None
    for name in names:
        try:
            reported = usable_image_count(payload.get(name))
        except Exception as e:  # noqa: BLE001
            logger.warning("Reading reported count %r failed: %s", name, e)
            continue
        if reported is not None:
            return reported
    return None


def _reported_dimensions(payload: Any) -> str:
    """``WxH`` from the provider's reported output dimensions, else ``""``.

    Both halves must be present and usable: a width with no height says nothing
    about the real dimensions. Zero is rejected by the ``and`` -- unlike a
    count, a zero-pixel dimension describes no image.
    """
    if not isinstance(payload, dict):
        return ""
    for width_key, height_key in (
        ("width", "height"),
        ("output_width", "output_height"),
    ):
        try:
            width = usable_image_count(payload.get(width_key))
            height = usable_image_count(payload.get(height_key))
        except Exception as e:  # noqa: BLE001
            logger.warning("Reading reported dimensions failed: %s", e)
            continue
        if width and height:
            return f"{width}x{height}"
    return ""


def record_image_usage(
    result: dict[str, Any],
    *,
    model_name: str = "",
    model_id: str = "",
    call_type: MediaCallType | str = MediaCallType.GENERATE_IMAGE,
    image_count: Any = _ABSENT,
    resolution: str = "",
    reported_count_from: Any = None,
    reported_size_from: Any = None,
) -> None:
    """Record one image generation/edit call on the current token context.

    Best-effort: any failure here is swallowed so accounting can never break the
    underlying image call. ``result`` is the provider return dict; ``usage`` in
    it (when present) may carry token counts some providers report (e.g. Gemini,
    OpenAI gpt-image).

    ``resolution`` is the size tier ("1K"/"2K"/"4K" or "1024x1024"), retained so
    downstream grouping can separate a model's resolutions into distinct
    billable line items. Provider tokens are forwarded as raw metadata on the
    same row. Neither is a pricing rule: nothing here or in the shared media
    row expresses "price by tokens instead of by unit" -- the row carries no
    price-basis discriminator and the aggregate groups purely by
    (model, unit, call_type, resolution). Defining that precedence is tracked
    in #1461.

    ``image_count`` is the request's own count -- the fallback. It is forwarded
    raw: the write boundary sanitises it while keeping the row, so a malformed
    value bills 0 (an unmeasured-but-billed call) instead of silently billing 1
    or losing the record. Omit it entirely when the provider neither accepts nor
    reports a count, which records 1.

    ``reported_count_from`` / ``reported_size_from`` take the provider's ``usage``
    payload, and the provider's own reported count and dimensions override the
    request values when it reports usable ones. The payload is walked **here**,
    inside this function's swallow, rather than by the caller: a provider
    payload whose ``get`` raises would otherwise take down the caller's whole
    call -- turning a successful 200 that returned a real image into a failure
    with no billing row -- because an argument expression runs in the caller's
    frame, outside this protection.
    """
    try:
        try:
            usage = result.get("usage") if isinstance(result, dict) else None
        except Exception as e:  # noqa: BLE001
            # Guarded separately from the whole body: this is the outermost
            # provider-controlled read, and letting it reach the handler below
            # would drop the row for a call that did happen. Every other read
            # here already degrades to a request value instead.
            logger.warning("Reading the usage payload failed: %s", e)
            usage = None
        # Raw, uncoerced: the boundary below is the only sanitiser. See the
        # module docstring for why pre-coercion here loses rows.
        input_tokens = _read_raw(usage, "prompt_tokens", "input_tokens")
        output_tokens = _read_raw(usage, "completion_tokens", "output_tokens")
        quantity = 1 if image_count is _ABSENT else image_count
        reported = _first_usable_count(
            reported_count_from, "image_count", "output_image_count"
        )
        if reported is not None:
            quantity = reported
        reported_resolution = _reported_dimensions(reported_size_from)
        if reported_resolution:
            resolution = reported_resolution
        add_media_usage(
            call_type=call_type,
            quantity=quantity,
            model=model_name,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            resolution=resolution,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to record image usage: %s", e)
