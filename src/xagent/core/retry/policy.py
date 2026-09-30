"""Per-error-class bounds layered on top of :class:`RetryWrapper`.

``RetryWrapper`` counts attempts and nothing else, which is not enough to bound
how long one logical call can hold an execution slot: every attempt may itself
consume a full request timeout, so a 10-attempt budget over a provider that
answers slowly is an effectively unbounded stall. Two bounds that attempt
counting cannot express live here:

* a wall-clock deadline for the whole retry loop, and
* a shorter attempt budget for provider capacity refusals, where replaying the
  identical request is precisely what the provider just asked us not to do.

Classification is message-based, the same technique
:func:`xagent.core.model.chat.error.is_context_length_error` uses and for the
same reason: providers signal capacity exhaustion in the response body, and the
adapters re-raise every provider error as a bare ``RuntimeError``, so the
originating type is frequently gone by the time the wrapper sees the failure.
A missed marker is not a regression -- it only means the failure keeps the
generic attempt budget, with the deadline still bounding it.

A budget never *grants* retries. ``RetryWrapper`` consults it only after
``retry_on`` has already accepted the failure, so an attempt limit can lower
the ceiling for a class of error and can never raise it. One consequence of
that ordering: the classifiers here walk the whole cause chain while
``retry_on`` inspects only the exception and its direct ``__cause__``, so a
capacity refusal buried two wrappers deep is never retried in the first place
and never reaches a budget. The bound is the narrower predicate, which is the
safe direction -- a failure the budget cannot see is one that already fails
fast.

The two deployment knobs live in :mod:`xagent.config` with every other
setting; this module only assembles them into a budget.
"""

import email.utils
import time
from dataclasses import dataclass

from ...config import get_llm_capacity_max_attempts, get_llm_retry_deadline_seconds

# The openai/anthropic SDKs honour a ``Retry-After`` hint only up to a minute
# before falling back to their own backoff. We take their retry budget away
# (see ``OpenAILLM._ensure_client``) so we have to reproduce that ceiling here;
# a longer hint is ignored rather than slept through, because holding an
# execution slot idle is the cost this module exists to bound.
MAX_HONORED_RETRY_AFTER_SECONDS = 60.0

# Deliberately narrower than a bare "capacity": these are the phrasings
# providers use to refuse work they have no room for, and a wider marker would
# reclassify unrelated failures into the short budget.
_CAPACITY_ERROR_MARKERS = (
    "on-demand capacity",
    "insufficient capacity",
    "capacity exceeded",
    "exceeded capacity",
    "over capacity",
    "at capacity",
    "no capacity",
    "overloaded",
)


def _causes(error: BaseException) -> list[BaseException]:
    """Walk an exception's explicit cause chain, tolerating cyclic links.

    ``__cause__`` only, deliberately unlike
    :func:`xagent.core.model.chat.error.is_context_length_error`, which also
    falls back to ``__context__``. ``__context__`` is set implicitly for any
    exception raised while another was being handled, including unrelated
    cleanup failures, so a marker found there can belong to an exception that
    has nothing to do with the request. Here that costs resilience rather
    than merely misreporting: classifying a transient fault as a capacity
    refusal cuts it from the model's full budget to two attempts. Every chat
    adapter chains its provider errors explicitly (``raise ... from e``), so
    the shapes this has to recognize all carry ``__cause__``.
    """
    seen: set[int] = set()
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
    return chain


def is_capacity_error(error: BaseException) -> bool:
    """Recognize a provider capacity/throttling refusal through wrappers."""
    for current in _causes(error):
        message = str(current).lower()
        if any(marker in message for marker in _CAPACITY_ERROR_MARKERS):
            return True
    return False


def _as_seconds(raw: str, divisor: float) -> float | None:
    """Read one header value as a delay, numeric or HTTP-date.

    A date is a legal ``Retry-After``, and both SDKs whose retry budget we
    disabled honoured it: they fall through to ``email.utils.parsedate_tz``
    and return ``date - now``, under the same ceiling applied here. Dropping
    a date would spend a capacity refusal's single retry on the local backoff
    against a provider that had named a later time.
    """
    try:
        return float(raw) / divisor
    except (TypeError, ValueError):
        pass

    parsed = email.utils.parsedate_tz(raw)
    if parsed is None:
        return None
    return float(email.utils.mktime_tz(parsed) - time.time())


def _parse_retry_after(headers: object) -> float | None:
    """Read a usable delay out of one response's ``Retry-After`` headers."""
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None

    for header, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        raw = getter(header)
        if raw is None:
            continue
        seconds = _as_seconds(str(raw).strip(), divisor)
        if seconds is not None and 0 < seconds <= MAX_HONORED_RETRY_AFTER_SECONDS:
            return seconds
    return None


def retry_after_seconds(error: BaseException) -> float | None:
    """Return the provider's own retry hint for ``error``, when it gave one."""
    for current in _causes(error):
        response = getattr(current, "response", None)
        if response is None:
            continue
        hint = _parse_retry_after(getattr(response, "headers", None))
        if hint is not None:
            return hint
    return None


@dataclass(frozen=True)
class RetryBudget:
    """Bounds a retry loop beyond its attempt count.

    ``deadline_seconds`` caps the wall clock of the whole loop.
    ``capacity_max_attempts`` caps the attempts spent on a capacity refusal.
    Both are optional; an empty budget imposes nothing, which is what every
    caller that does not pass one gets.
    """

    deadline_seconds: float | None = None
    capacity_max_attempts: int | None = None

    def attempt_limit(self, error: BaseException) -> int | None:
        """Return a lower attempt ceiling for ``error``, or ``None`` for none."""
        if self.capacity_max_attempts is None:
            return None
        if not is_capacity_error(error):
            return None
        return self.capacity_max_attempts


def chat_retry_budget() -> RetryBudget:
    """Build the chat-model budget, reading configuration per call.

    Read at call time rather than at import so a deployment override applies
    to models constructed after it is set, and so the bound is exercisable
    from tests.
    """
    return RetryBudget(
        deadline_seconds=get_llm_retry_deadline_seconds(),
        capacity_max_attempts=get_llm_capacity_max_attempts(),
    )
