"""Usage tracking using contextvars, for LLM tokens and non-LLM media calls.

Tracks usage across calls without threading it through function signatures:
contextvars let statistics be collected automatically during task execution.

Two dimensions live here. LLM tokens use ``add_token_usage`` and aggregate via
``aggregate_token_usage_by_model``. Non-LLM media calls — image, video, TTS,
ASR, music, sound effect, embedding, rerank — use the ``MediaUnit`` /
``MediaCallType`` vocabulary, record through ``add_media_usage``, and aggregate
via ``aggregate_media_usage_by_model``. Both write into the same
``TokenUsage.details`` list, discriminated by each row's ``type``, so existing
persistence and quota paths carry media rows with no schema change.
"""

import contextvars
import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel: "argument not supplied", distinct from any value a caller could pass.
_UNSET: Any = object()


class MediaUnit(str, Enum):
    """Billable dimension of a non-LLM media call.

    One unit per modality, chosen so a given (model, call_type) always reports
    the same unit regardless of how complete the provider's response was — a
    price table keyed on (model, unit) is only usable if the unit is stable.
    ``REQUESTS`` means exactly one provider call and always carries
    ``quantity=1``; use it only when the modality genuinely has no finer
    billable dimension, never as a degraded fallback for a missing measurement.
    """

    IMAGES = "images"
    SECONDS = "seconds"
    CHARACTERS = "characters"
    TEXTS = "texts"
    REQUESTS = "requests"


class MediaCallType(str, Enum):
    """Modality/operation that produced a media usage entry."""

    GENERATE_IMAGE = "generate_image"
    EDIT_IMAGE = "edit_image"
    VIDEO = "video"
    TTS = "tts"
    ASR = "asr"
    MUSIC = "music"
    SOUND_EFFECT = "sound_effect"
    EMBEDDING = "embedding"
    RERANK = "rerank"


#: The billable unit each modality reports. This is what makes a price table
#: keyed on (model, unit) usable: the unit is a property of the modality, never
#: of how complete a particular response happened to be. A duration-billed call
#: whose length is unknown records ``seconds`` with ``quantity=0``, it does not
#: fall back to ``requests``.
#:
#: Enforced at the write boundary rather than only documented — the invariant was
#: stated in three docstrings and still violated by six call sites in this
#: module's own tests (``images`` paired with ``video``).
MEDIA_UNIT_BY_CALL_TYPE: Dict[str, "MediaUnit"] = {
    MediaCallType.GENERATE_IMAGE.value: MediaUnit.IMAGES,
    MediaCallType.EDIT_IMAGE.value: MediaUnit.IMAGES,
    MediaCallType.VIDEO.value: MediaUnit.SECONDS,
    MediaCallType.ASR.value: MediaUnit.SECONDS,
    MediaCallType.MUSIC.value: MediaUnit.SECONDS,
    MediaCallType.SOUND_EFFECT.value: MediaUnit.SECONDS,
    MediaCallType.TTS.value: MediaUnit.CHARACTERS,
    MediaCallType.EMBEDDING.value: MediaUnit.TEXTS,
    MediaCallType.RERANK.value: MediaUnit.REQUESTS,
}

_MEDIA_UNIT_VALUES = frozenset(member.value for member in MediaUnit)
_MEDIA_CALL_TYPE_VALUES = frozenset(member.value for member in MediaCallType)


def _validated_media_unit(unit: "MediaUnit | str | None") -> str:
    """Normalise ``unit`` to a known :class:`MediaUnit` value.

    A typo'd unit silently mints a new billing dimension that
    :func:`aggregate_media_usage_by_model` will happily key off, and a usage
    record cannot be repaired retroactively once written. Rejecting at the
    write boundary is the only point where the error is still fixable.
    """
    if unit is None:
        raise ValueError("Media unit cannot be None")
    value = unit.value if isinstance(unit, MediaUnit) else str(unit)
    if value not in _MEDIA_UNIT_VALUES:
        raise ValueError(
            f"Unknown media unit {value!r}; expected one of "
            f"{sorted(_MEDIA_UNIT_VALUES)}"
        )
    return value


def _validated_media_call_type(call_type: "MediaCallType | str | None") -> str:
    """Normalise ``call_type`` to a known :class:`MediaCallType` value.

    Empty is allowed: ``call_type`` is optional metadata rather than a billing
    dimension on its own, and omitting it is a legitimate caller choice.
    """
    if call_type is None:
        return ""
    value = call_type.value if isinstance(call_type, MediaCallType) else str(call_type)
    if value and value not in _MEDIA_CALL_TYPE_VALUES:
        raise ValueError(
            f"Unknown media call type {value!r}; expected one of "
            f"{sorted(_MEDIA_CALL_TYPE_VALUES)}"
        )
    return value


def copy_detail_rows(raw_details: Any) -> List[Dict]:
    """Detached copies of the dict rows in ``raw_details``, non-dicts dropped.

    One helper for every place a details list is handed across a boundary:
    sharing the inner dicts lets a consumer mutate live usage state, and the
    ``isinstance`` filter keeps a malformed legacy row (``details`` is persisted
    as free-form JSON) from raising in an accounting path.
    """
    if not isinstance(raw_details, list):
        return []
    return [dict(item) for item in raw_details if isinstance(item, dict)]


@dataclass
class TokenUsage:
    """Token usage statistics for a task or operation.

    Attributes:
        input_tokens: Number of tokens in prompts sent to LLM
        output_tokens: Number of tokens generated by LLM
        llm_calls: Number of LLM API calls made
        media_calls: Number of non-LLM media model calls made (image/video/
            tts/asr/embedding/rerank/...)
        details: Detailed breakdown by model/call type
    """

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    details: List[Dict] = field(default_factory=list)
    # Appended AFTER the legacy fields, and keyword-only, so the historical
    # positional signature — TokenUsage(input, output, llm_calls, tool_calls,
    # details) — keeps binding the way existing callers expect. Inserting it
    # before tool_calls silently rebound their 4th and 5th arguments (tool
    # count became media count, details became tool count) with no error.
    media_calls: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        # Counter updates are read-modify-write, and one TokenUsage is routinely
        # shared across worker threads (RAG ingestion pools, and the executor hops
        # the producer PRs add); without it a concurrent read can observe a
        # counter that disagrees with its own detail rows.
        #
        # Deliberately NOT a dataclass field. As a field it would sit in
        # ``__dataclass_fields__``, and ``dataclasses.asdict`` walks those
        # directly (ignoring ``__getstate__``), so it would raise
        # ``TypeError: cannot pickle '_thread.lock' object`` for any generic
        # consumer. Keeping it in ``__dict__`` also leaves the public
        # positional signature — ``TokenUsage(input, output, llm_calls,
        # tool_calls, details)`` — free of a private slot.
        self.__dict__["_lock"] = threading.RLock()
        # Normalise once here so the constructor and from_dict both hand this
        # object a private, dict-only list. Storing a caller's list by reference
        # would let them mutate rows outside the lock, and a non-dict row would
        # misalign the index-based delta slice.
        self.details = copy_detail_rows(self.details)

    @property
    def _lock(self) -> "threading._RLock":
        """The per-instance mutation lock (see :meth:`__post_init__`).

        Reentrant so a future nested acquisition cannot self-deadlock: the
        current no-nesting discipline (``to_dict`` inlines the token total rather
        than calling ``total_tokens``, ``merge`` takes the two locks
        sequentially) is invisible to a later editor.
        """
        # threading.RLock is a factory, not a class, so the annotation names the
        # object it returns (typeshed exposes it as threading._RLock).
        lock: threading._RLock = self.__dict__["_lock"]
        return lock

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the lock: it is not picklable and is per-instance state."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore, giving the revived instance its own fresh lock."""
        self.__dict__.update(state)
        self.__dict__["_lock"] = threading.RLock()

    def __copy__(self) -> "TokenUsage":
        """Shallow copy is unsupported; delegate to :meth:`snapshot`.

        The default shallow copy routes through the ``__getstate__`` pair above,
        so the new instance gets a *different* lock while still referencing the
        *same* ``details`` list. Two objects each believing they hold exclusive
        access would then mutate one shared list under two different locks —
        exactly the mutual exclusion the lock exists to provide. ``snapshot()``
        is the supported way to fork a usage object, so ``copy.copy`` is routed
        there rather than left as a trap.
        """
        return self.snapshot()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TokenUsage":
        """Deep copy via :meth:`snapshot` so the copy is taken under the lock."""
        copied = self.snapshot()
        memo[id(self)] = copied
        return copied

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        with self._lock:
            return self.input_tokens + self.output_tokens

    def add_input_tokens(
        self,
        tokens: int,
        model: str = "",
        call_type: str = "",
        model_id: str = "",
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Add input tokens from a prompt.

        ``cached_tokens`` is the subset of ``tokens`` served from the provider's
        prompt cache (usually billed cheaper); 0 when unknown/unsupported.
        ``cache_write_tokens`` is the subset of ``tokens`` written to the cache
        this call (Claude bills these at a premium); 0 when unknown.
        """
        with self._lock:
            self.input_tokens += tokens
            if model or call_type:
                self.details.append(
                    {
                        "type": "input",
                        "tokens": tokens,
                        "cached_tokens": cached_tokens,
                        "cache_write_tokens": cache_write_tokens,
                        "model": model,
                        "model_id": model_id,
                        "call_type": call_type,
                    }
                )

    def add_output_tokens(
        self, tokens: int, model: str = "", call_type: str = "", model_id: str = ""
    ) -> None:
        """Add output tokens from a completion."""
        with self._lock:
            self.output_tokens += tokens
            if model or call_type:
                self.details.append(
                    {
                        "type": "output",
                        "tokens": tokens,
                        "model": model,
                        "model_id": model_id,
                        "call_type": call_type,
                    }
                )

    def record_media_call(
        self,
        unit: "MediaUnit | str | None",
        quantity: float,
        model: str = "",
        call_type: "MediaCallType | str | None" = "",
        model_id: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        resolution: str = "",
        tokens_estimated: bool = False,
        _raw_quantity: Any = _UNSET,
    ) -> None:
        """Count the call and append its detail under one lock acquisition.

        The only media write path. Bumping the counter and appending the row
        separately would take the lock twice, letting a concurrent
        ``to_dict``/``merge``/``snapshot`` land between them and observe a
        counter with no matching detail row (or the reverse). Consumers derive
        per-model rows from ``details`` and the call count from the scalar, so a
        torn pair reports mutually inconsistent billing.
        """
        unit_value = _validated_media_unit(unit)
        call_type_value = _validated_media_call_type(call_type)
        # Validated here rather than only in the module-level
        # ``add_media_usage``: this is the actual write boundary, so a direct
        # caller holding a TokenUsage gets the same guarantees instead of
        # persisting a boolean, negative, or non-finite quantity that cannot be
        # repaired later.
        # _raw_quantity lets the module-level add_media_usage, which coerces one
        # layer up, report what its own caller actually passed.
        raw_quantity = quantity if _raw_quantity is _UNSET else _raw_quantity
        quantity = _coerce_float(quantity)
        # REQUESTS is defined as exactly one provider call, so its quantity is
        # not a free variable. Letting 0/0.5/2 through would make the billable
        # quantity disagree with the media_calls/aggregate `calls` count derived
        # from the same row — quota and pricing would then read different
        # numbers off one record. Raised rather than clamped: a caller passing
        # something else has a real bug, and silently rewriting it hides that.
        # Producers route through ``media_usage.record_media_usage``, which
        # swallows this so an accounting bug still cannot break a media call.
        # Rejecting before the lock is taken means a bad call leaves no state at
        # all — neither a counter bump nor an orphan row.
        expected_unit = MEDIA_UNIT_BY_CALL_TYPE.get(call_type_value)
        if expected_unit is not None and unit_value != expected_unit.value:
            raise ValueError(
                f"call_type {call_type_value!r} bills in "
                f"{expected_unit.value!r}, not {unit_value!r}; the unit is a "
                f"property of the modality, so a (model, unit) price table "
                f"breaks if one modality reports two units"
            )
        if unit_value == MediaUnit.REQUESTS.value and quantity != 1.0:
            raise ValueError(
                f"MediaUnit.REQUESTS means exactly one call, so quantity must "
                f"be 1; got {raw_quantity!r}"
            )
        input_tokens = _coerce_int(input_tokens)
        output_tokens = _coerce_int(output_tokens)
        with self._lock:
            self.media_calls += 1
            self.details.append(
                {
                    "type": "media",
                    "unit": unit_value,
                    "quantity": quantity,
                    "provider_tokens": input_tokens + output_tokens,
                    "provider_input_tokens": input_tokens,
                    "provider_output_tokens": output_tokens,
                    "tokens_estimated": tokens_estimated,
                    "model": model,
                    "model_id": model_id,
                    "call_type": call_type_value,
                    "resolution": resolution,
                }
            )

    def detail_tail(self, start: int = 0) -> tuple[List[Dict], int]:
        """Detail rows from ``start`` onward plus the tool-call count, atomically.

        The narrow read that per-turn delta computation actually needs. Both
        values come from one lock acquisition, so they describe the same instant
        — a concurrent ``record_media_call``/``increment_tool_calls`` cannot land
        between them and yield a pair from two different logical points in time.

        Prefer this over ``snapshot()`` whenever only the tail is wanted.
        ``snapshot()`` deep-copies the whole cumulative ``details`` list, which
        grows monotonically across turns (seeds are restored from the persisted
        list), so using it to read a small tail costs O(total usage) instead of
        O(delta) — and holds the lock that serialises every LLM adapter's token
        write for the duration of that copy. On the quota-gate path, polled once
        per agent step *and* once per streamed LLM chunk, that difference is
        three orders of magnitude at a few thousand accumulated rows.

        Rows are copied so the caller cannot mutate live state through them.
        """
        with self._lock:
            return (
                copy_detail_rows(self.details[start:]),
                self.tool_calls,
            )

    def snapshot(self) -> "TokenUsage":
        """A detached copy taken under the lock.

        Reading the fields individually — as an external copier must — can
        interleave with a concurrent ``record_media_call``/``merge`` and produce
        a torn snapshot: a counter without its matching detail row, or the
        reverse. Consumers derive per-model rows from ``details`` and the call
        count from the scalar, so a torn pair reports mutually inconsistent
        billing. Taking the lock once here is the only way a copier can get a
        self-consistent view.
        """
        with self._lock:
            copy = TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                llm_calls=self.llm_calls,
                tool_calls=self.tool_calls,
                media_calls=self.media_calls,
            )
            # Assigned after construction, not passed in: __post_init__ would
            # copy the list a second time, doubling the cost of a copy that is
            # already O(total details).
            copy.details = copy_detail_rows(self.details)
        return copy

    def increment_llm_calls(self) -> None:
        """Increment the LLM call counter."""
        with self._lock:
            self.llm_calls += 1

    def increment_tool_calls(self, count: int = 1) -> None:
        """Increment the tool-call counter (one per tool invocation).

        Negative counts are ignored: they would drive the counter below the
        number of matching detail rows, and no caller has a reason to decrement
        a monotonic usage counter.
        """
        if count < 0:
            logger.warning("Ignoring negative tool call increment: %r", count)
            return
        with self._lock:
            self.tool_calls += count

    def merge(self, other: "TokenUsage") -> None:
        """Merge another TokenUsage into this one."""
        # Snapshot ``other`` under its own lock and release it before taking
        # ours: holding both at once would deadlock a concurrent ``b.merge(a)``.
        with other._lock:
            input_tokens = other.input_tokens
            output_tokens = other.output_tokens
            llm_calls = other.llm_calls
            media_calls = other.media_calls
            tool_calls = other.tool_calls
            # dict(item), not list(...): sharing the inner dicts would leave the
            # merged rows aliased to the source's, so mutating one usage object
            # would silently rewrite the other's billing rows. Same reason
            # to_dict and snapshot copy each entry.
            details = copy_detail_rows(other.details)
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.llm_calls += llm_calls
            self.media_calls += media_calls
            self.tool_calls += tool_calls
            self.details.extend(details)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        with self._lock:
            input_tokens = self.input_tokens
            output_tokens = self.output_tokens
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "llm_calls": self.llm_calls,
                "media_calls": self.media_calls,
                "tool_calls": self.tool_calls,
                # dict(item), not just list(...): a new outer list around the
                # same inner dicts lets a caller mutate the live usage object
                # through the returned payload, entirely outside this lock.
                # Matches snapshot(), which already does this.
                # copy_detail_rows also drops non-dict rows, which the
                # previous inline copy here did not. Deliberate: it aligns this
                # with snapshot/merge, and a malformed legacy row passed through
                # here would fail later at JSON serialisation rather than being
                # skipped at the boundary.
                "details": copy_detail_rows(self.details),
            }

    @classmethod
    def from_dict(cls, data: Dict) -> "TokenUsage":
        """Create from dictionary."""
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            llm_calls=data.get("llm_calls", 0),
            media_calls=data.get("media_calls", 0),
            tool_calls=data.get("tool_calls", 0),
            details=data.get("details", []),
        )


# ContextVar for thread-local token tracking. Default is None (not a shared
# TokenUsage instance): a single shared default would accumulate forever for
# untracked paths (preview/builder that never call set_token_usage), leaking
# memory. get_token_usage() lazily creates a per-context instance instead.
token_context: contextvars.ContextVar[Optional[TokenUsage]] = contextvars.ContextVar(
    "token_context", default=None
)


class TokenContextManager:
    """Manager for token context with automatic cleanup.

    Example:
        with TokenContextManager() as manager:
            # LLM calls here will be tracked
            await llm.chat(messages)
        # After exiting, usage can be retrieved
        usage = manager.get_usage()
    """

    def __init__(self, parent_usage: Optional[TokenUsage] = None):
        """Initialize the context manager.

        Args:
            parent_usage: Optional parent TokenUsage to merge into
        """
        self._token_usage = TokenUsage()
        if parent_usage:
            self._token_usage.merge(parent_usage)
        self._previous_token: Optional[TokenUsage] = None

    def __enter__(self) -> "TokenContextManager":
        """Enter the context and start tracking."""
        self._previous_token = token_context.get(None)
        token_context.set(self._token_usage)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and restore the previous state.

        Restores whatever was there before (``None`` when nothing was set), so
        leaving a manager block doesn't reintroduce a lingering shared
        TokenUsage instance — matching the None default on the ContextVar.
        """
        token_context.set(self._previous_token)

    def get_usage(self) -> TokenUsage:
        """Get the current token usage."""
        return self._token_usage

    def add_input_tokens(
        self, tokens: int, model: str = "", call_type: str = ""
    ) -> None:
        """Add input tokens (convenience method)."""
        self._token_usage.add_input_tokens(tokens, model, call_type)

    def add_output_tokens(
        self, tokens: int, model: str = "", call_type: str = ""
    ) -> None:
        """Add output tokens (convenience method)."""
        self._token_usage.add_output_tokens(tokens, model, call_type)


# Global functions for easier access


def get_token_usage() -> TokenUsage:
    """Get the current token usage, lazily creating a per-context instance."""
    usage = token_context.get()
    if usage is None:
        usage = TokenUsage()
        token_context.set(usage)
    return usage


def _coerce_int(value: Any) -> int:
    """A finite, non-negative int; 0 if the value isn't a usable token count.

    Token counts share the quantity guarantees documented on
    :func:`_coerce_float`, for the same reason: this is the boundary before the
    value is persisted into task and quota details, where it cannot be
    repaired. Specifically:

    * ``bool`` — ``True``/``False`` are a caller bug, not a count of 1 or 0.
    * negatives — a negative token count would subtract from a bill.
    * ``NaN``/``inf`` — ``int(float("inf"))`` raises ``OverflowError``, which
      would propagate out of the accounting path and drop the whole (billable)
      media row rather than just the bad field.
    """
    if isinstance(value, bool):
        logger.warning("Discarding boolean token count: %r", value)
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        # None/absent is expected (provider omitted the field) — stay quiet.
        # A present-but-malformed value signals a provider-adapter bug worth
        # surfacing rather than silently billing it as zero.
        if value is not None:
            logger.warning("Discarding non-numeric token count: %r", value)
        return 0
    if result < 0:
        logger.warning("Discarding negative token count: %r", value)
        return 0
    return result


def estimate_media_tokens(text: Any) -> int:
    """Language-aware token estimate for providers that report no usage.

    CJK characters are roughly one token each, while Latin script averages
    about four characters per token; a flat chars/4 heuristic therefore
    undercounts Chinese text by close to 4x. Accepts a string or an iterable of
    strings and ignores anything else, so a malformed input can never raise in
    an accounting path. Callers must pass ``tokens_estimated=True`` alongside
    the result so billing can tell this apart from a measured count.
    """
    if isinstance(text, str):
        items: List[str] = [text]
    else:
        # Any iterable of strings, including generators: the docstring promises
        # that, and an embedding producer passing a generator would otherwise
        # silently estimate 0 tokens for a real batch. Non-iterables and
        # non-string members are ignored rather than raising, since this runs in
        # an accounting path.
        try:
            items = [item for item in text if isinstance(item, str)]
        except Exception as e:  # noqa: BLE001
            # Not just TypeError: a custom iterable's __iter__/__next__ can
            # raise anything, and this runs in an accounting path whose
            # documented contract is that malformed input never breaks the call
            # being measured. Estimate 0 and say so.
            logger.warning("Token estimation failed for %r: %s", type(text), e)
            return 0

    cjk = 0
    other = 0
    for item in items:
        for char in item:
            # CJK Unified Ideographs, Japanese kana, and Hangul syllables —
            # all roughly one token per character.
            code = ord(char)
            if (
                0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
                or 0x3400 <= code <= 0x4DBF  # CJK Ext-A
                or 0x3000 <= code <= 0x303F  # CJK punctuation (、。「」etc.)
                or 0x3040 <= code <= 0x30FF  # Japanese kana
                or 0xAC00 <= code <= 0xD7AF  # Hangul syllables
                or 0xFF00 <= code <= 0xFFEF  # Fullwidth / halfwidth forms
            ):
                cjk += 1
            else:
                other += 1
    # Round up rather than truncate: `other // 4` alone estimates zero for any
    # 1-3 character Latin string, so short non-empty text would bill nothing.
    # A non-empty input must never estimate 0 tokens.
    latin_tokens = -(-other // 4)  # ceil division
    return cjk + latin_tokens


def _coerce_float(value: Any) -> float:
    """A finite, non-negative float; 0.0 if the value isn't a usable quantity.

    Media quantities can be fractional (e.g. audio seconds), so quantity uses
    this rather than ``_coerce_int``.

    Rejects what a billable quantity can never be, because this is the last
    boundary before the value is persisted and it cannot be repaired
    afterwards:

    * ``bool`` — ``True``/``False`` are almost certainly a caller bug, not a
      quantity of 1 or 0.
    * negatives — no modality can consume a negative amount, and a negative
      would subtract from a bill.
    * ``NaN``/``inf`` — these are not JSON-serialisable, so they would produce
      literal ``NaN``/``Infinity`` tokens in the persisted task and quota
      details that strict parsers reject.

    A rejected value records 0.0, matching the "call happened but is
    unmeasured" convention rather than dropping the record entirely.
    """
    if isinstance(value, bool):
        logger.warning("Discarding boolean media quantity: %r", value)
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError, not just TypeError/ValueError: float(10**400) raises it,
        # and an uncaught raise here would propagate out and drop the whole
        # billing row — the opposite of this function's reject-to-0.0 contract.
        # _coerce_int already catches it; these two must stay in step.
        if value is not None:
            logger.warning("Discarding non-numeric media quantity: %r", value)
        return 0.0
    if not math.isfinite(result):
        logger.warning("Discarding non-finite media quantity: %r", value)
        return 0.0
    if result < 0:
        logger.warning("Discarding negative media quantity: %r", value)
        return 0.0
    return result


def _usage_field(usage: Any, name: str) -> Any:
    """Read a field from a provider usage payload (SDK object or plain dict)."""
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)


def extract_cached_input_tokens(usage: Any) -> int:
    """Prompt-cache-hit input tokens from an OpenAI-style usage payload.

    Handles both attribute-style SDK objects and plain dicts (streaming chunks
    often arrive as dicts). DeepSeek reports ``prompt_cache_hit_tokens``;
    OpenAI/DashScope/Zhipu report ``prompt_tokens_details.cached_tokens``.
    Returns 0 when unavailable.
    """
    if usage is None:
        return 0
    hit = _usage_field(usage, "prompt_cache_hit_tokens")
    if hit is not None:
        value = max(0, _coerce_int(hit))
        if value:
            return value
        # Some proxies emit prompt_cache_hit_tokens=0 as a default while the
        # real count sits in prompt_tokens_details; fall through to it.
    details = _usage_field(usage, "prompt_tokens_details")
    if details is not None:
        return max(0, _coerce_int(_usage_field(details, "cached_tokens")))
    return 0


def aggregate_token_usage_by_model(details: Any) -> List[Dict[str, Any]]:
    """Aggregate persisted token detail entries by the actual model used.

    ``model_id`` is preferred as the identity because different configured
    models can share a provider-facing name. A legacy name-only group is
    merged into an id-backed group only when that name identifies exactly one
    configured model. Entries without either an id or name are retained as an
    unattributed group rather than silently dropping tokens from the breakdown.

    Pass a detached list. This iterates ``details`` without any lock, so handing
    in a live ``TokenUsage.details`` while another thread appends risks a
    "list changed size during iteration" RuntimeError. Use
    ``TokenUsage.snapshot().details`` (or a value read from the DB column, as
    the API path does) rather than the live list.
    """
    if not isinstance(details, list):
        return []

    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for detail in details:
        if not isinstance(detail, dict):
            continue
        token_type = detail.get("type")
        if token_type not in {"input", "output"}:
            continue
        tokens = max(0, _coerce_int(detail.get("tokens")))
        if tokens == 0:
            continue

        raw_model_id = detail.get("model_id")
        raw_model_name = detail.get("model")
        model_id = raw_model_id.strip() if isinstance(raw_model_id, str) else ""
        model_name = raw_model_name.strip() if isinstance(raw_model_name, str) else ""
        key = ("id", model_id) if model_id else ("name", model_name)
        if not model_id and not model_name:
            key = ("unknown", "")

        aggregate = grouped.setdefault(
            key,
            {
                "model_id": model_id,
                "model_name": model_name,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
        )
        # Some legacy adapters only stamped the name on one of the pair.
        if not aggregate["model_name"] and model_name:
            aggregate["model_name"] = model_name
        aggregate[f"{token_type}_tokens"] += tokens
        aggregate["total_tokens"] += tokens
        if token_type == "input":
            # Legacy entries predate cache tracking; clamp to the entry's
            # own input tokens so malformed data can't exceed the total.
            cached = min(tokens, max(0, _coerce_int(detail.get("cached_tokens"))))
            cache_write = min(
                tokens, max(0, _coerce_int(detail.get("cache_write_tokens")))
            )
            aggregate["cached_input_tokens"] += cached
            aggregate["cache_write_input_tokens"] += cache_write

    id_keys_by_name: Dict[str, List[tuple[str, str]]] = {}
    for key, aggregate in grouped.items():
        if key[0] == "id" and aggregate["model_name"]:
            id_keys_by_name.setdefault(aggregate["model_name"], []).append(key)

    for key, aggregate in list(grouped.items()):
        if key[0] != "name":
            continue
        matching_id_keys = id_keys_by_name.get(aggregate["model_name"], [])
        if len(matching_id_keys) != 1:
            continue
        target = grouped[matching_id_keys[0]]
        target["input_tokens"] += aggregate["input_tokens"]
        target["output_tokens"] += aggregate["output_tokens"]
        target["total_tokens"] += aggregate["total_tokens"]
        target["cached_input_tokens"] += aggregate["cached_input_tokens"]
        target["cache_write_input_tokens"] += aggregate["cache_write_input_tokens"]
        del grouped[key]

    sorted_groups = sorted(
        grouped.values(),
        key=lambda item: (
            -item["total_tokens"],
            item["model_name"].casefold(),
            item["model_id"].casefold(),
        ),
    )
    return [
        {
            "model_id": item["model_id"],
            "model_name": item["model_name"],
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "cached_input_tokens": item["cached_input_tokens"],
            "cache_write_input_tokens": item["cache_write_input_tokens"],
        }
        for item in sorted_groups
    ]


def aggregate_media_usage_by_model(details: Any) -> List[Dict[str, Any]]:
    """Aggregate ``type:"media"`` detail entries by model/unit/call_type/resolution.

    Companion to :func:`aggregate_token_usage_by_model`, which only keeps
    input/output token entries. Media entries are billed per non-token unit
    (images, seconds, ...), so they are grouped separately by their unit and
    modality rather than summed into a single token total. Resolution is part of
    the key because an image model's price varies by resolution, so different
    resolutions of the same model surface as separate billable line items.
    Returns one entry per (model, unit, call_type, resolution) combination with
    the summed quantity, call count and provider-reported tokens. A group is
    marked ``tokens_estimated`` when any entry in it carried estimated tokens,
    so a consumer never prices a mixed group as if it were measured.

    Pass a detached list. This iterates ``details`` without any lock, so handing
    in a live ``TokenUsage.details`` while another thread appends risks a
    "list changed size during iteration" RuntimeError. Use
    ``TokenUsage.snapshot().details`` (or a value read from the DB column, as
    the API path does) rather than the live list.
    """
    if not isinstance(details, list):
        return []

    grouped: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for detail in details:
        if not isinstance(detail, dict):
            continue
        if detail.get("type") != "media":
            continue
        quantity = max(0.0, _coerce_float(detail.get("quantity")))
        # Zero-quantity entries are deliberately KEPT. Unlike a zero-token LLM
        # entry, a zero-quantity media entry is meaningful: the duration-billed
        # tools record quantity=0 precisely to say "this provider call happened
        # but its size is unknown" (an async video with no duration yet).
        # Dropping it would hide the whole media popover and report
        # media_calls=0 for a task that really did make billable calls.
        tokens = max(0, _coerce_int(detail.get("provider_tokens")))

        raw_model_id = detail.get("model_id")
        raw_model_name = detail.get("model")
        raw_unit = detail.get("unit")
        raw_call_type = detail.get("call_type")
        raw_resolution = detail.get("resolution")
        model_id = raw_model_id.strip() if isinstance(raw_model_id, str) else ""
        model_name = raw_model_name.strip() if isinstance(raw_model_name, str) else ""
        unit = raw_unit if isinstance(raw_unit, str) else ""
        call_type = raw_call_type if isinstance(raw_call_type, str) else ""
        resolution = raw_resolution if isinstance(raw_resolution, str) else ""
        # model_id is redundant in the key: it equals identity when set, and is
        # constant "" otherwise.
        identity = model_id or model_name
        key = (identity, unit, call_type, resolution)

        aggregate = grouped.setdefault(
            key,
            {
                "model_id": model_id,
                "model_name": model_name,
                "unit": unit,
                "call_type": call_type,
                "resolution": resolution,
                "quantity": 0.0,
                "calls": 0,
                "provider_tokens": 0,
                "tokens_estimated": False,
            },
        )
        if not aggregate["model_name"] and model_name:
            aggregate["model_name"] = model_name
        if not aggregate["model_id"] and model_id:
            aggregate["model_id"] = model_id
        aggregate["quantity"] += quantity
        aggregate["calls"] += 1
        aggregate["provider_tokens"] += tokens
        if detail.get("tokens_estimated"):
            aggregate["tokens_estimated"] = True

    return sorted(
        grouped.values(),
        key=lambda item: (
            -item["quantity"],
            str(item["model_name"]).casefold(),
            str(item["unit"]).casefold(),
            str(item["call_type"]).casefold(),
            str(item["resolution"]).casefold(),
        ),
    )


def add_token_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "",
    call_type: str = "",
    model_id: str = "",
    cached_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
) -> None:
    """Add token usage to the current context.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name for tracking
        call_type: Type of call (chat, stream_chat, vision_chat, etc.)
        model_id: Unique model id (disambiguates identically-named models)
        cached_input_tokens: Subset of input_tokens served from prompt cache
        cache_write_input_tokens: Subset of input_tokens written to the cache
    """
    # Coerce defensively: a provider/response that yields a non-int token count
    # (or a mock in tests) must never crash the LLM call over accounting.
    input_tokens = _coerce_int(input_tokens)
    output_tokens = _coerce_int(output_tokens)
    cached_input_tokens = _coerce_int(cached_input_tokens)
    cache_write_input_tokens = _coerce_int(cache_write_input_tokens)

    usage = get_token_usage()
    if input_tokens or output_tokens:
        # Increment LLM call counter for each API call
        usage.increment_llm_calls()
    if input_tokens:
        usage.add_input_tokens(
            input_tokens,
            model,
            call_type,
            model_id,
            cached_input_tokens,
            cache_write_input_tokens,
        )
    if output_tokens:
        usage.add_output_tokens(output_tokens, model, call_type, model_id)

    logger.debug(
        f"Token usage added: input={input_tokens}, output={output_tokens}, "
        f"model={model}, model_id={model_id}, call_type={call_type}, "
        f"total_input={usage.input_tokens}, total_output={usage.output_tokens}, "
        f"total_calls={usage.llm_calls}"
    )


def add_media_usage(
    unit: "MediaUnit | str | None",
    quantity: float,
    model: str = "",
    call_type: "MediaCallType | str | None" = "",
    model_id: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    resolution: str = "",
    tokens_estimated: bool = False,
) -> None:
    """Record non-LLM media model usage on the current context.

    Mirrors ``add_token_usage`` for image/video/tts/asr/embedding/rerank and
    any other non-chat modality. The resulting ``type:"media"`` detail entry
    flows through the same ``TokenUsage.details`` list into DB persistence and
    the quota ``delta_details`` contract, so callers need only this one call.

    Args:
        unit: Billable dimension; see :class:`MediaUnit`. Must be stable for a
            given (model, call_type) — never vary it by response completeness.
        quantity: Amount of ``unit`` consumed (may be fractional, e.g. seconds).
            Always 1 for ``MediaUnit.REQUESTS``.
        model: Model name for tracking.
        call_type: Modality/operation; see :class:`MediaCallType`.
        model_id: Unique model id (disambiguates identically-named models).
        input_tokens: Provider-reported input tokens; 0 if none.
        output_tokens: Provider-reported output tokens; 0 if none.
        resolution: Size tier ("1K"/"2K"/"4K" or "1024x1024") for image models
            whose price varies by resolution; "" when not applicable.
            Token-reporting providers (Gemini / OpenAI gpt-image) also fill
            input/output_tokens, recorded as raw ``provider_tokens`` for a future
            consumer. Deliberately NOT claimed as a pricing rule: nothing here
            expresses or enforces "price by tokens instead of by unit", the row
            schema carries no such discriminator, and the aggregate groups purely
            by (model, unit, call_type, resolution). Defining that precedence is
            tracked in #1461, with the first pricing consumer.
        tokens_estimated: True when the token counts are a local heuristic
            rather than provider-reported, so billing can refuse to price them.

    Raises:
        ValueError: If ``unit`` or ``call_type`` is not a known
            :class:`MediaUnit` / :class:`MediaCallType` value, or if ``unit`` is
            ``MediaUnit.REQUESTS`` and ``quantity`` is not exactly 1. The first
            two are checked here, before the context is touched; the REQUESTS
            constraint is enforced inside :meth:`TokenUsage.record_media_call`,
            which still rejects before taking its lock, so no partial state is
            left either way. Producers route through
            ``media_usage.record_media_usage``, which swallows all three so an
            accounting bug can never break the underlying media call.
    """
    # Coerce defensively so a provider returning a malformed count can never
    # crash the underlying media call over accounting.
    # Keep the caller's raw value for the REQUESTS error message below:
    # record_media_call reports what it was handed, which by then is already
    # coerced, so -1 would surface as "got 0.0" without this.
    raw_quantity = quantity
    quantity = _coerce_float(quantity)
    input_tokens = _coerce_int(input_tokens)
    output_tokens = _coerce_int(output_tokens)
    # Validate before touching the context: a rejected unit must not leave
    # media_calls incremented with no matching detail entry behind it.
    # Plain strings keep the details list JSON-serialisable whether the caller
    # passed an enum member or a bare string.
    unit_value = _validated_media_unit(unit)
    call_type_value = _validated_media_call_type(call_type)

    usage = get_token_usage()
    # One locked operation: see TokenUsage.record_media_call for why the count
    # and its detail row must not be observable apart.
    usage.record_media_call(
        _raw_quantity=raw_quantity,
        unit=unit_value,
        quantity=quantity,
        model=model,
        call_type=call_type_value,
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        resolution=resolution,
        tokens_estimated=tokens_estimated,
    )

    logger.debug(
        f"Media usage added: unit={unit_value}, quantity={quantity}, "
        f"resolution={resolution}, model={model}, model_id={model_id}, "
        f"call_type={call_type_value}, "
        f"provider_tokens={input_tokens + output_tokens}"
        f"{' (estimated)' if tokens_estimated else ''}, "
        f"total_media_calls={usage.media_calls}"
    )


def add_tool_call_usage(count: int = 1) -> None:
    """Record one (or more) tool invocations on the current context."""
    get_token_usage().increment_tool_calls(count)


def reset_token_usage() -> TokenUsage:
    """Reset and return the current token usage."""
    new_usage = TokenUsage()
    token_context.set(new_usage)
    return new_usage


def set_token_usage(usage: TokenUsage) -> TokenUsage:
    token_context.set(usage)
    return usage


def get_and_reset_token_usage() -> TokenUsage:
    """Get current usage and reset the context."""
    usage = get_token_usage()
    reset_token_usage()
    return usage
