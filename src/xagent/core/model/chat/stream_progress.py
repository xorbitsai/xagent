"""Detect an LLM stream that keeps emitting chunks but stopped making progress.

Some OpenAI-compatible models (Kimi K2.5 behind Bedrock Mantle is the
measured case, #2785) degenerate into a repetition or whitespace loop and run
until the endpoint's output cap. The stream never goes quiet, so the
token-interval timeout cannot fire, and the caller waits ~100 s for output
that carries nothing. This module is the pure decision half of the fix: the
provider's raw stream loop feeds it one observation per delta and it answers
"abort, and why" or "keep going". Closing the stream and shaping what is
returned stays with the provider.

Three predicates. The thresholds are configurable; a value of ``0``
switches off the empty-delta count and the whitespace/periodicity checks.
The one rule without a knob is non-whitespace after a complete tool-call
object: that input can never parse, so there is nothing to tune.

* **empty deltas** -- N consecutive raw deltas with no content, no tool-call
  bytes, no reasoning text and no finish reason. This is what an
  unrecognized reasoning field looks like from here.
* **degenerate reasoning** -- the last W characters of recognized reasoning
  text are whitespace-only, or periodic with a period of at most P
  characters. Legitimate reasoning is neither, so a thinking model that
  reasons for minutes stays live; a loop of one phrase does not.
* **bytes after a complete tool-call object** -- once a call's arguments
  parse as a JSON object, anything further appended to *that* call is
  garbage by construction: non-whitespace can never parse (aborted at once,
  no knob), whitespace is noise (aborted once it fills the window). A new
  call at a new index is unaffected, and an object that is still open (the
  runaway-list shape) is deliberately left to ``max_tokens``.

A fourth, opt-in trigger aborts after T seconds without any content or
tool-call chunk, for models listed by name. It exists as an escape hatch for
a provider whose loop none of the above can see; it is off unless listed
because it would cut every legitimate long-reasoning call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ....config import (
    get_llm_stream_degenerate_max_period,
    get_llm_stream_degenerate_window,
    get_llm_stream_empty_delta_limit,
    get_llm_stream_no_payload_abort_models,
    get_llm_stream_no_payload_timeout_seconds,
)

# ``finish_reason`` value stamped on the chunk that ends an aborted stream, so
# the runtime's #2786 markers record the abort instead of a clean finish.
NO_PROGRESS_FINISH_REASON = "no_progress"

# ``stream_fallback`` value the runtime stamps on a non-streaming retry taken
# because the stream produced no content or tool call.
NO_PAYLOAD_STREAM_FALLBACK = "no_payload"

# Key under which the chunk that ends an aborted stream carries the abort
# reason in its ``raw`` payload, and under which the runtime lifts it onto
# the response and ``llm_call_end``, so the trace can tell which predicate
# fired rather than only that one did.
STREAM_ABORTED_KEY = "stream_aborted"

AbortReason = Literal[
    "empty_deltas",
    "reasoning_whitespace",
    "reasoning_repetition",
    "tool_call_trailing_bytes",
    "tool_call_trailing_whitespace",
    "no_payload_timeout",
]


@dataclass(frozen=True)
class StreamProgressConfig:
    """Thresholds for :class:`StreamProgressGuard`.

    ``0`` disables ``empty_delta_limit`` and ``degenerate_window``;
    ``no_payload_timeout=None`` disables the wall-clock trigger.
    """

    empty_delta_limit: int = 200
    degenerate_window: int = 256
    degenerate_max_period: int = 64
    no_payload_timeout: float | None = None

    @classmethod
    def from_env(cls, *, model_name: str) -> StreamProgressConfig:
        """Build from the ``XAGENT_LLM_STREAM_*`` settings for one model.

        The wall-clock trigger is enabled only when ``model_name`` is listed
        (exact match on the wire model name) in
        ``XAGENT_LLM_STREAM_NO_PAYLOAD_ABORT_MODELS``.
        """
        timeout: float | None = None
        if model_name in get_llm_stream_no_payload_abort_models():
            timeout = get_llm_stream_no_payload_timeout_seconds()
        return cls(
            empty_delta_limit=get_llm_stream_empty_delta_limit(),
            degenerate_window=get_llm_stream_degenerate_window(),
            degenerate_max_period=get_llm_stream_degenerate_max_period(),
            no_payload_timeout=timeout,
        )


@dataclass(frozen=True)
class StreamAbort:
    """Why the guard wants the stream ended, and how to shape the tail.

    ``reason`` is a short stable token for logs and tests; ``detail`` is the
    human-readable explanation. For the tool-call predicates,
    ``tool_call_id`` names the call whose arguments received the extra bytes
    and ``keep_length`` is that call's argument length before the rejected
    delta: the provider cuts the call back to it, so what it returns equals
    or extends the last snapshot the runtime holds (a wrapper such as the
    DeepSeek tool-protocol adapter may have streamed a shorter prefix).
    """

    reason: AbortReason
    detail: str
    tool_call_id: str | None = None
    keep_length: int | None = None


class StreamProgressGuard:
    """Per-stream state for the predicates described in the module docstring."""

    def __init__(
        self,
        config: StreamProgressConfig,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        # Resolved here, not in the signature, so a test can patch
        # ``time.monotonic`` on this module.
        self._clock = clock if clock is not None else time.monotonic
        self._empty_run = 0
        self._first_delta_at: float | None = None
        self._payload_seen = False
        # Recognized reasoning text: total length plus a bounded tail. The
        # checks only ever look at the last ``degenerate_window`` characters.
        self._reasoning_len = 0
        self._reasoning_tail = ""
        # Per call id: arguments length last seen, and the r-stripped length
        # at which the arguments first parsed as a JSON object (if ever).
        self._seen_len: dict[str, int] = {}
        self._complete_len: dict[str, int] = {}

    # ------------------------------------------------------------------ API

    def observe(
        self,
        *,
        has_content: bool = False,
        reasoning_texts: Sequence[str] = (),
        reasoning_opaque: bool = False,
        has_finish_reason: bool = False,
        has_usage: bool = False,
        accumulated_tool_calls: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> StreamAbort | None:
        """Record one raw delta; return an abort verdict or ``None``.

        Args:
            has_content: the delta carried non-empty ``content``.
            reasoning_texts: string values of every ``reasoning*`` field on
                the delta (empty strings are allowed and count as nothing).
            reasoning_opaque: a ``reasoning*`` field carried a non-empty,
                non-string value the caller cannot inspect (kept live).
            has_finish_reason: the choice reported a ``finish_reason``.
            has_usage: the chunk carried usage (the trailing usage chunk).
            accumulated_tool_calls: the provider's accumulated tool-call map
                *after* this delta was merged into it.
        """
        now = self._clock()
        if self._first_delta_at is None:
            self._first_delta_at = now

        # A provider that mirrors the same text under two ``reasoning*``
        # spellings must not count twice: doubled text halves the apparent
        # period and fills the window at twice the true rate.
        unique_texts: list[str] = []
        for text in reasoning_texts:
            if text and text not in unique_texts:
                unique_texts.append(text)
        reasoning_delta = "".join(unique_texts)
        tool_progress, tool_verdict = self._observe_tool_calls(
            accumulated_tool_calls or {}
        )
        if tool_verdict is not None:
            return tool_verdict

        if has_content or tool_progress:
            self._payload_seen = True

        if has_finish_reason or has_usage:
            # The provider is ending the stream itself. Aborting now could
            # only lose the trailing usage chunk, so the remaining predicates
            # do not run on a finishing delta (the tool-call check above
            # still does: its snapshot repair is worth more than the usage).
            self._empty_run = 0
            if reasoning_delta:
                self._append_reasoning(reasoning_delta)
            return None

        progressed = (
            has_content
            or tool_progress
            or bool(reasoning_delta)
            or reasoning_opaque
            or has_finish_reason
            or has_usage
        )
        if progressed:
            self._empty_run = 0
        else:
            self._empty_run += 1
            limit = self._config.empty_delta_limit
            if limit > 0 and self._empty_run >= limit:
                return StreamAbort(
                    reason="empty_deltas",
                    detail=(
                        f"{self._empty_run} consecutive deltas carried no content, "
                        "tool-call bytes, reasoning text or finish_reason"
                    ),
                )

        if reasoning_delta:
            verdict = self._observe_reasoning(reasoning_delta)
            if verdict is not None:
                return verdict

        timeout = self._config.no_payload_timeout
        if timeout is not None and not self._payload_seen:
            elapsed = now - self._first_delta_at
            if elapsed >= timeout:
                return StreamAbort(
                    reason="no_payload_timeout",
                    detail=(
                        f"no content or tool-call chunk {elapsed:.1f}s after the "
                        f"first delta (limit {timeout:.1f}s)"
                    ),
                )
        return None

    # ------------------------------------------------------------ predicates

    def _append_reasoning(self, delta: str) -> None:
        window = self._config.degenerate_window
        if window <= 0:
            return
        self._reasoning_len += len(delta)
        # Keep twice the window so a check never needs more than it has.
        self._reasoning_tail = (self._reasoning_tail + delta)[-(2 * window) :]

    def _observe_reasoning(self, delta: str) -> StreamAbort | None:
        window = self._config.degenerate_window
        if window <= 0:
            return None
        self._append_reasoning(delta)
        if self._reasoning_len < window:
            return None
        tail = self._reasoning_tail[-window:]
        if not tail.strip():
            return StreamAbort(
                reason="reasoning_whitespace",
                detail=f"last {window} reasoning characters are whitespace",
            )
        period = _shortest_period(tail, self._config.degenerate_max_period)
        if period is not None:
            return StreamAbort(
                reason="reasoning_repetition",
                detail=(
                    f"last {window} reasoning characters repeat with period={period}"
                ),
            )
        return None

    def _observe_tool_calls(
        self, calls: Mapping[str, Mapping[str, Any]]
    ) -> tuple[bool, StreamAbort | None]:
        """Return ``(progressed, verdict)`` for the accumulated tool calls.

        Progress means a new call id or more argument bytes on an existing
        one. The verdict fires only for bytes appended to a call whose
        arguments already parsed as a JSON object.
        """
        progressed = False
        for call_id, call in calls.items():
            function = call.get("function")
            arguments = (
                function.get("arguments") if isinstance(function, dict) else None
            )
            if not isinstance(arguments, str):
                continue
            length = len(arguments)
            previous = self._seen_len.get(call_id)
            if previous is None or length > previous:
                progressed = True
            if previous is not None and length == previous:
                continue
            self._seen_len[call_id] = length

            complete_len = self._complete_len.get(call_id)
            if complete_len is None:
                # Only a delta that brought a closing brace can have completed
                # the object; skip the parse otherwise so a large open object
                # is not re-parsed on every delta.
                if "}" in arguments[previous or 0 :]:
                    complete_len = _complete_object_length(arguments)
                    if complete_len is not None:
                        self._complete_len[call_id] = complete_len
                        # Bytes after the object may have arrived in this same
                        # delta (``}{`` is one BPE token); fall through.
                if complete_len is None:
                    continue

            extra = arguments[complete_len:]
            if extra.strip():
                return progressed, StreamAbort(
                    reason="tool_call_trailing_bytes",
                    detail=(
                        f"tool call {call_id} received {len(extra)} bytes after its "
                        "arguments already formed a complete JSON object"
                    ),
                    tool_call_id=call_id,
                    keep_length=previous,
                )
            window = self._config.degenerate_window
            if window > 0 and len(extra) >= window:
                return progressed, StreamAbort(
                    reason="tool_call_trailing_whitespace",
                    detail=(
                        f"tool call {call_id} received {len(extra)} whitespace "
                        "characters after its arguments already formed a complete "
                        "JSON object"
                    ),
                    tool_call_id=call_id,
                    keep_length=previous,
                )
        return progressed, None


# ------------------------------------------------------------------ helpers


_JSON_DECODER = json.JSONDecoder()


def _complete_object_length(arguments: str) -> int | None:
    """Offset just past the first complete JSON object in ``arguments``.

    Only a top-level object counts: a bare number or string prefix can still
    legitimately grow (``12`` -> ``123``), an object that closed cannot.
    Leading whitespace is allowed, and anything after the object (``}{``
    arriving as one delta) is left for the caller to judge, so the result
    is the object's end, not the string's length.
    """
    leading = len(arguments) - len(arguments.lstrip())
    body = arguments[leading:]
    if not body.startswith("{"):
        return None
    try:
        parsed, end = _JSON_DECODER.raw_decode(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return leading + end


def _shortest_period(text: str, max_period: int) -> int | None:
    """Smallest ``p <= max_period`` such that ``text`` is periodic with period ``p``.

    Periodic means ``text[i] == text[i + p]`` for every valid ``i``; a
    single repeated character is period 1. The period is capped at half the
    text so "periodic" always means at least two full repetitions, whatever
    ``max_period`` an operator configures. Bounded to ``max_period``
    comparisons of ``len(text)`` characters, so it is cheap at the sizes the
    guard uses.
    """
    limit = min(max_period, len(text) // 2)
    for period in range(1, limit + 1):
        if text[period:] == text[:-period]:
            return period
    return None
