"""SSE transport layer for ``GET /v1/chat/tasks/{task_id}/events``.

This module owns *moving bytes to the client and deciding when to
stop* -- registration into the shared connection ``manager``, the
outbound frame queue, the 30-second watchdog, the 1-hour absolute
duration cap, and per-task / per-principal concurrency limits -- and,
layered on top of that transport, projecting the task's ``step.*`` /
``message.*`` content onto the same stream: each live broadcast frame
is classified and folded through a per-connection
``PublicStepProjector`` (see ``project_content_frames``), so a running
task's steps and streamed message text reach an already-attached
client without polling. Four more events are lifecycle-only and
carry no step/message content of their own -- ``task.status``,
``task.completed``, ``task.input_required``, and ``stream.error``.

Each connection gets its own projector, and that projector is fed only
the frames broadcast after the connection registered. A step that was
already running at that moment therefore has no matching start in it,
so the step's eventual end event folds in as an orphan and is dropped:
that step never appears on this stream at all, since its start was
broadcast before this connection existed.
Clients attaching mid-task reconcile against
``GET /v1/chat/tasks/{task_id}/steps``, which reads the database and
so is unaffected by when the stream was opened.

No ``task.status`` frame is
ever guaranteed to be fresh or in order, at any point in the stream's
life: ``ConnectionManager.broadcast_to_task``
stamps each event with whatever ``run_id`` / ``state_version`` tuple
its producer captured, falling back to a fresh read of the task row
(status included) when the producer didn't capture one, and this sink
never compares that stamp against anything -- it just forwards each
status string it sees, deduped only against the last one it sent. A
stale or out-of-order ``task.status`` frame can therefore reach the
client at any point during the stream's life, not only right after
attach. This is accepted: the only frames this module treats as
authoritative are the three close frames -- ``task.completed``,
``task.input_required``, and ``stream.error`` -- and all three are
produced by the watchdog (or the attach-time snapshot read) reading
the task row directly, never inferred from frame ordering.
``task.input_required``'s ``prompt`` field is populated from
``_TaskInfoSnapshot.pending_question`` -- the same authoritative row
read that already backs the close decision itself, so this never
triggers a query of its own and never sniffs live agent_message frames
for question text.

Sink instances duck-type the ``websocket.ConnectionManager`` connection
contract (an object with an async ``send_text(str)`` method) so they
register into the *same* shared ``manager`` real WebSocket connections
use, and ride the same ``broadcast_to_task`` fan-out.

Size and admission bounds -- every limit this stream applies, and the
values it deliberately leaves unbounded, in one place:

  - One step's ``data`` sub-object, and one message frame's text:
    ``MAX_FRAME_CONTENT_BYTES`` (64 KiB), measured in the escaped-JSON
    domain the frame is actually serialized in, not decoded UTF-8 bytes
    (see ``_capped_text``). The two families handle an overrun
    differently on purpose: a message's text is truncated in place and
    the frame is marked ``truncated``, while an oversized step ``data``
    is *replaced* by a marker keeping the step's identifying keys --
    step ``data`` is arbitrary nested JSON, so there is no single
    string to cut (see ``_capped_step_data``).
  - One inbound broadcast frame, before it is projected at all:
    ``MAX_RAW_FRAME_TEXT_CHARS`` (256 KiB), measured excluding the
    ``task_description`` stamp every converted trace event carries (see
    ``_measured_content_frame``). Over the cap is a silent drop --
    counted and logged, never a close, and never applied to a
    lifecycle frame.
  - One sink's outbound queue: ``OUTBOUND_QUEUE_MAX_SIZE`` (256
    frames) *and* ``MAX_QUEUED_WIRE_BYTES`` (4 MiB of queued frame
    text). Whichever binds first drains the backlog and closes with
    ``resync_required``; the element cap binds on a long backlog of
    ordinary frames, the byte budget on a short backlog of large ones.
    The byte budget only engages once the queued average exceeds 16
    KiB -- one quarter of the per-frame content cap above -- and even
    then the cost of engaging is one forced ``resync_required`` close,
    identical in kind and price to the element cap's own overflow; a
    consumer that is keeping up never accumulates a backlog at all.
    Tighter would hurt: 1 MiB puts the engagement average at 4 KiB,
    inside ordinary backlog territory, so 4 MiB is the tightest value
    that leaves ordinary traffic alone -- against the ~16 MiB a full
    element-capped queue of maximum frames would otherwise hold.
  - Concurrent streams: ``PER_TASK_STREAM_CAP`` (2) and
    ``PER_PRINCIPAL_STREAM_CAP`` (32), both rejected with 429 at
    attach, before any sink exists.
  - Deliberately uncapped: ``task.completed``'s ``output`` and
    ``task.input_required``'s ``prompt`` (full argument in the comment
    above ``completed_frame``). Both are the payload of a conclusion
    frame, both are sent once per stream as that frame, and neither has
    an equivalent on ``GET .../steps`` -- their recovery channel is
    ``GET /v1/chat/tasks/{task_id}``. Neither ever meets the queue's
    byte budget: on the live path (the watchdog closing an already-open
    stream) they reach the client only as a budget-exempt close frame
    via ``enqueue_close``; on the two attach-time fast paths (terminal,
    waiting-for-user) they are ``yield``ed straight from the generator
    before any sink or queue exists for this connection at all.
  - Uncapped and bounded only by the frame they arrive in: a step's
    ``id`` and a ``message.*`` frame's ``message_id``. Neither has a
    cap of its own -- ``PublicStep.id`` is a plain ``str`` -- and an id
    sits outside the ``data`` sub-object the 64 KiB cap covers. A step
    id derives from the event's own ``tool_call_id``/``step_id``, so
    what bounds it is the 256 KiB check on the inbound frame carrying
    it, and the queue's byte budget bounds ids in aggregate. Today's
    only ``message_id`` producer emits a fixed 45 characters
    (``f"final_answer_{uuid4().hex}"``,
    ``core/agent/runtime.py``'s ``start_final_answer_stream``), but
    this path does not itself enforce that length -- the effective
    bound here is the same 256 KiB inbound raw-frame check.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from time import monotonic
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, cast

from fastapi import WebSocket
from fastapi.responses import StreamingResponse

from ...models.agent_api_key import AgentApiKey
from ...models.database import get_session_local
from ...models.task import TaskStatus
from ...schemas.v1 import PublicStep
from ...services.db_runtime import run_db_io_cancellation_safe
from ...services.task_execution_controller import TaskControlState
from ..public_trace_events import (
    DELEGATED_AGENT_TRACE_SOURCE,
    is_audit_only_trace_data,
    normalize_public_trace_event,
)
from ..websocket import _is_versioned_task_event, manager
from ..ws_trace_handlers import serialize_trace_data
from ._step_mapping import PublicStepProjector
from .deps import ApiKeyPrincipal, active_runtime_key_filters
from .errors import V1ApiError, V1ErrorCode

if TYPE_CHECKING:
    # Only for the type checker -- importing ``tasks`` at module scope
    # would cycle back here (``tasks.py`` imports this module to wire the
    # endpoint). Callers pass a snapshot-reading callable at call time
    # instead (see ``TaskSnapshotReader`` below).
    from .tasks import _TaskInfoSnapshot

logger = logging.getLogger(__name__)

# -- Tunables (all injectable per-call for tests; production always uses
# the defaults below via ``tasks.py``). ---------------------------------

HEARTBEAT_INTERVAL_SECONDS = 15.0
WATCHDOG_INTERVAL_SECONDS = 30.0
# Same shape/value as A2A's cap (``web/api/a2a.py:105``); a separate v1
# constant because the two streams' generators are structurally different
# (A2A polls every <=0.5s and re-checks the deadline for free; v1 is
# queue-consumption-based, so each wait on the outbound queue must be
# capped at the heartbeat interval so the deadline gets re-checked often
# enough).
STREAM_MAX_DURATION_SECONDS = 60.0 * 60.0
OUTBOUND_QUEUE_MAX_SIZE = 256
PER_TASK_STREAM_CAP = 2
PER_PRINCIPAL_STREAM_CAP = 32
# Bounds how large a single step.*/message.* frame's content can get --
# unlike the other tunables above, this scopes only the content-frame
# families this module projects (a tool's args/result, an agent
# delegation's input/output, or a message's text), not the whole SSE
# frame. Same
# magnitude as the repo's existing single-blob byte caps (e.g.
# ``core/task_runtime.py``'s ``MAX_TASK_RUNTIME_JSON_BYTES``), not a new
# one invented for this module.
MAX_FRAME_CONTENT_BYTES = 64 * 1024
# The queue's second bound, on the total wire bytes it is holding rather
# than on its element count. The element cap alone lets a slow consumer
# retain 256 x the largest frame this module can build: a
# ``message.completed`` whose text is capped at
# ``MAX_FRAME_CONTENT_BYTES`` measures 65,659 bytes, so 16.0 MiB per
# sink, and a frame whose *id* rather than its content is what made it
# large is bounded only by ``MAX_RAW_FRAME_TEXT_CHARS`` on the frame it
# arrived in, so it can be several times that again. Both are
# constructive upper bounds from the frame builders, not observed
# backlogs.
# Sized as 64 largest-possible content frames -- expressed against
# ``MAX_FRAME_CONTENT_BYTES`` so it tracks that cap instead of drifting
# from it. Strictly over the budget closes, exactly on it does not --
# the same boundary rule ``MAX_RAW_FRAME_TEXT_CHARS`` uses. Why this
# value and not tighter or looser is argued once, in the module
# docstring's size-bounds section, not repeated here. Frames are pure
# ASCII (``json.dumps`` runs with the default ``ensure_ascii=True``), so
# ``len(frame_text)`` is the wire byte count and no encode is needed.
# CPython's own ~41-byte-per-string header is not counted: 256 of them
# is ~10 KiB against a 4 MiB budget.
MAX_QUEUED_WIRE_BYTES = 64 * MAX_FRAME_CONTENT_BYTES
# Check on a raw broadcast frame's text length -- measured excluding
# the internal task-description stamp broadcast frames carry (see
# ``_measured_content_frame``) -- run after ``json.loads`` parses it,
# gating only the call into the projection pass -- ``MAX_FRAME_CONTENT_BYTES`` only bounds the *projected*
# content that ends up on the wire, so without this a multi-megabyte raw
# frame still pays for a full ``serialize_trace_data`` walk + projection
# just to have its content truncated at the very end. 4x
# ``MAX_FRAME_CONTENT_BYTES`` gives room for a raw frame's JSON envelope
# (field names, escaping, nested step metadata) around content that
# would itself end up right at the cap -- a frame this large is already
# far past anything a capped projection would keep, so dropping it
# outright costs nothing a client would have been able to read anyway.
# Scoped to content frames only -- a lifecycle frame is never truncated
# or projected by this module, so its raw size can't be judged against
# a cap built around projected content. ``task_completed`` in
# particular carries its whole output twice (``result`` and ``output``,
# see ``websocket.py``) and can legitimately cross this threshold well
# before anything is actually wrong; dropping it here would silently
# delay the completion signal to the watchdog's next cycle instead of
# firing it immediately.
MAX_RAW_FRAME_TEXT_CHARS = 4 * MAX_FRAME_CONTENT_BYTES
# Floor for the final wait when the 1-hour deadline is close: keeps the
# queue wait from being called with a near-zero timeout, which would spin
# the loop instead of actually waiting.
_MIN_WAIT_BUDGET_SECONDS = 0.05

_TERMINAL_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED)
_TERMINAL_STATUS_VALUES = {status.value for status in _TERMINAL_STATUSES}
# A task moving to ``waiting_for_user`` also warrants an early watchdog
# wake, same as a terminal one -- both are broadcasts that make the
# authoritative row read (by the watchdog, or the attach-time snapshot)
# worth re-running sooner than the next scheduled cycle.
_EARLY_WAKE_STATUS_VALUES = _TERMINAL_STATUS_VALUES | {
    TaskStatus.WAITING_FOR_USER.value
}


def _stream_close_reason(status: TaskStatus, control_state: str | None) -> str | None:
    """Shared close determination for a task's authoritative row state.

    Both the watchdog (``watchdog_check_once``, polling) and the
    attach-time fast path (``build_event_stream_response``, one-shot at
    open) need to reach the same conclusion from the same two fields, so
    they call this instead of duplicating the condition. Returns
    ``"terminal"`` when the task is completed/failed, ``"input_required"``
    when it's stuck waiting on user input (and not mid-resume), or
    ``None`` when the stream should keep going. Each caller still builds
    its own output from the category -- a close frame on the watchdog's
    already-open stream, or a fast-path response before one ever opens.
    """
    if status in _TERMINAL_STATUSES:
        return "terminal"
    if (
        status is TaskStatus.WAITING_FOR_USER
        and control_state != TaskControlState.RESUME_REQUESTED.value
    ):
        return "input_required"
    return None


# A sync callable: (task_id, principal) -> ``_TaskInfoSnapshot``-shaped
# object (duck-typed: ``.status`` / ``.control_state`` / ``.output`` /
# ``.error`` / ``.pending_question``), raising
# ``V1ApiError(TASK_NOT_FOUND, 404)`` when the task is missing or not
# owned. Always run through ``run_db_io_cancellation_safe`` by this module
# -- never called directly.
TaskSnapshotReader = Callable[[int, "ApiKeyPrincipal"], Any]


# -- Wire format --------------------------------------------------------


def _sse_frame(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def status_frame(status: str) -> str:
    return _sse_frame("task.status", {"status": status})


# ``completed_frame``'s ``output`` and ``input_required_frame``'s
# ``prompt`` are the two content-bearing fields on this stream with no
# size cap. Both are the payload of a conclusion frame, taken from the
# caller's own authoritative task-row read, and neither has an
# equivalent on ``GET .../steps``: that endpoint returns projected
# steps, not the task row's ``output`` column and not its pending
# question. Truncating either one would leave a client with no way to
# recover the full value from this stream or from ``steps()``.
# ``step.*``/``message.*`` content is capped precisely because it does
# have that channel -- ``steps()`` always returns a step's full,
# untruncated ``data``. The recovery channel for these two is
# ``GET /v1/chat/tasks/{task_id}``: ``TaskInfoResponse.output`` and
# ``TaskInfoResponse.pending_interaction.question``. Cost of the
# exemption is bounded by construction: each is sent once per stream, as
# its closing frame (``enqueue_close`` on the live path, the last yield
# on each attach-time fast path), never repeatedly.
def completed_frame(*, status: str, output: str | None, error: str | None) -> str:
    return _sse_frame(
        "task.completed", {"status": status, "output": output, "error": error}
    )


def input_required_frame(task_id: int, prompt: str | None) -> str:
    # ``prompt`` is uncapped for the reason stated above
    # ``completed_frame``, which covers both conclusion-frame fields.
    # ``prompt`` comes straight from ``_TaskInfoSnapshot.pending_question``
    # (``tasks.py``'s ``_load_task_info_snapshot``, itself backed by
    # ``task_interaction_read.get_pending_interaction_question`` -- see the
    # module docstring): every caller of this function already has that
    # snapshot in hand from its own authoritative row read, so this
    # never triggers a query of its
    # own and never sniffs live agent_message frames for question text.
    return _sse_frame("task.input_required", {"task_id": task_id, "prompt": prompt})


_ERROR_MESSAGES = {
    "resync_required": (
        "The output queue overflowed; call steps() to resync, then re-attach."
    ),
    "unauthorized": "The API key used to open this stream has been revoked or paused.",
    "task_deleted": "The task no longer exists.",
    "stream_expired": "This stream reached its maximum allowed duration.",
}


def error_frame(code: str) -> str:
    """Build a ``stream.error`` frame. ``code`` is always the machine-
    readable value from ``_ERROR_MESSAGES`` (clients branch on it), and
    the lookup runs unconditionally, so an unrecognized ``code`` raises
    ``KeyError`` rather than reaching the wire as a frame with no
    message."""
    return _sse_frame("stream.error", {"code": code, "message": _ERROR_MESSAGES[code]})


# -- Content projection: step.* / message.* --------------------------------
#
# Everything below turns one already-parsed broadcast dict into zero or
# more ``step.*`` / ``message.*`` SSE frame strings, by classifying which
# family it belongs to (a trace event with an ``event_type``, a
# final-answer streaming frame, or neither) and routing it accordingly.
#
# Deliberately absent from that classification: comparing the frame's own
# top-level ``task_id`` against this stream's bound task_id and dropping
# on a mismatch. That check is unreachable in practice --
# ``register_connection`` already rebinds a sink to a new task_id before
# any frame for the new task can arrive -- and on the one path where it
# could theoretically fire, it would turn "wrong frame delivered" into
# the strictly worse "silently orphaned stream still holding a connection
# slot": it would discard frames for the *new* task the sink is actually
# bound to, not frames from a foreign one. Must not be added back in any
# form, including "only for content frames".


# The broadcast ``type`` values that can ever produce a content frame --
# shared between ``project_content_frames``'s own dispatch and the sink's
# live routing decision (``send_text``) so the two can't drift: the set
# ``send_text`` hands to the projector must be exactly the set this
# dispatch acts on, or a type present in one but absent from the other
# silently drops content on one side.
_CONTENT_FRAME_TYPES = ("trace_event", "final_answer_delta", "final_answer_end")


def _byte_length(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":")).encode("utf-8"))


def _capped_text(
    text: str, *, limit: int = MAX_FRAME_CONTENT_BYTES
) -> tuple[str, bool]:
    """Truncate ``text`` to ``limit`` bytes of its own escaped-JSON wire
    representation (``_byte_length``'s domain: ``len(json.dumps(text,
    separators=(",", ":")).encode())``, the string's own surrounding
    quotes included), returning ``(possibly-truncated text, whether
    truncation happened)``.

    Measured in the escaped domain, not decoded UTF-8 bytes, because
    that's the domain this text actually goes out in: ``_sse_frame``
    serializes with ``json.dumps``'s default ``ensure_ascii=True``, so a
    non-ASCII character can cost far more than its own UTF-8 byte width
    on the wire -- an emoji (4 UTF-8 bytes) becomes a 12-byte
    ``\\uD83D\\uDE00`` surrogate-pair escape, a CJK character (3 UTF-8
    bytes) becomes a 6-byte ``\\uXXXX`` escape. Measuring decoded UTF-8
    bytes against this same ``limit`` instead would let a
    16384-character emoji string -- exactly ``MAX_FRAME_CONTENT_BYTES``
    UTF-8 bytes -- through completely untruncated, while its escaped
    wire form runs roughly 3x over the cap. This is also the same domain
    ``_capped_step_data``'s own cap checks already use via
    ``_byte_length``, so a value ``_capped_text`` returns to
    ``_capped_step_data`` never gets re-measured into a different
    budget than the one it was actually truncated against (see the
    per-key budgeting step in that function's docstring).

    ``limit`` is overridden by ``_capped_step_data``, which needs a
    *smaller* budget than the full frame cap for an individual
    identifying value -- the value shares the cap with the truncation
    marker's own JSON overhead around it, not the whole cap to itself.

    Character-sliced first, then a binary search over the remaining
    prefix's character count for the longest one whose escaped bytes
    still fit ``limit``. The character pre-slice is a performance
    guard, not a correctness one -- it bounds the character count, not
    the escaped byte length the cap is measured in: every escaped character
    costs at least 1 byte, so a text longer than ``limit`` *characters*
    is guaranteed to still need the search below even after slicing to
    that many -- the pre-slice only avoids running the search over an
    entire multi-megabyte string when almost all of it will be thrown
    away regardless. The search is what makes the cut safe; a raw
    UTF-8-byte slice plus a lenient decode is not usable in the
    escaped domain, because a raw byte slice of an *escaped* string can
    land inside a multi-byte escape sequence (e.g. half of
    ``\\uD83D\\uDE00``), and decoding that slice as UTF-8 says nothing
    about whether the *unescaped* text it came from is still valid --
    there is no lenient-decode equivalent for "half an escape
    sequence". Slicing by character count instead never has this
    problem: every prefix is a valid string by construction. The binary
    search is safe because ``_byte_length`` is monotonically
    non-decreasing in the string's character count -- ``ensure_ascii``
    emits each character's own escape sequence independently, in order,
    so a longer prefix's encoded bytes are always a shorter prefix's
    encoded bytes plus more appended, never fewer.
    """
    pre_sliced = len(text) > limit
    if pre_sliced:
        text = text[:limit]
    if _byte_length(text) <= limit:
        # ``False``, not ``pre_sliced``: this branch is only reachable
        # when no slice happened. The escaped form carries the string's
        # own two surrounding quotes and every escaped character costs
        # at least one byte, so a text sliced to ``limit`` characters
        # measures at least ``limit + 2`` and can never satisfy the
        # check above.
        return text, False
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _byte_length(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo], True


# The small, bounded fields that identify *what* a step is, as opposed
# to its content -- one per public step type, per ``_step_mapping.py``'s
# ``map_trace_events_to_public_steps`` docstring: ``thinking.phase``,
# ``tool_call.name``, ``agent_delegation.sub_agent_name``,
# ``message.role``. Kept through ``_capped_step_data``'s truncation
# below (unlike ``args``/``result``/``error``/``input``/``output``/
# ``content``, which are exactly the fields that can be arbitrarily
# large) so a client reading a truncated frame still knows which tool
# ran, which sub-agent was delegated to, or whose message it was --
# not just that *something* on this step was too big to send.
_STEP_DATA_IDENTIFYING_KEYS = ("phase", "name", "sub_agent_name", "role")


def _capped_step_data(data: dict[str, Any]) -> dict[str, Any]:
    """Bound one step's ``data`` sub-object to ``MAX_FRAME_CONTENT_BYTES``.

    Step ``data`` shapes vary by public step type -- a tool's ``args``/
    ``result`` or an agent delegation's ``input``/``output`` can be
    arbitrary nested JSON, not a single string field like a message's
    ``content``. Walking arbitrary nested structure to trim it back
    under the cap isn't worth the complexity here: once the whole
    sub-object is oversized, its content fields are replaced wholesale
    with a small truncation marker instead.

    Each pass over the surviving identifying keys does two things,
    re-measured with ``_byte_length`` against the cap in the same
    escaped-JSON byte domain ``_capped_text`` truncates in (see that
    function's docstring), so the cap is an actual bound on what this
    returns rather than on the pre-replacement content:

      1. The marker plus this step's identifying fields
         (``_STEP_DATA_IDENTIFYING_KEYS``) as they are, if that fits --
         so a client reading a truncated frame still knows which tool
         ran, which sub-agent was delegated to, or whose message it was,
         not just that *something* on this step was too big to send.
      2. Otherwise every surviving *string* value truncated
         (``_capped_text``), budgeted to leave room for the marker's own
         JSON overhead plus the other keys' shares -- truncating each to
         the full cap and only then wrapping it would overflow on the
         wrapper alone.

    A value that is not a string (a dict- or list-valued ``name``, say)
    cannot be truncated in place, so when a pass still does not fit, the
    largest surviving value that cannot be truncated in place is dropped
    (falling back to the largest overall if every survivor is a string,
    which the budgeting math makes unreachable), and the next pass
    rebuilds from the original ``data``. Rebuilding rather than
    continuing matters: an un-shrinkable oversized value counts toward
    step 2's overhead, which drives every other key's budget to zero, so
    a pass that kept the already-truncated values would return the
    identifying keys as empty strings -- which tells a client nothing.
    Dropping only the key that cannot fit keeps the rest with real
    content, and a step whose only identifying value is that one
    degrades to the bare marker, exactly as before.

    The rest of the step (id/type/status/timestamps) is always small and
    untouched either way.
    """
    size = _byte_length(data)
    if size <= MAX_FRAME_CONTENT_BYTES:
        return data
    bare_marker: dict[str, Any] = {"truncated": True, "original_bytes": size}
    surviving = [key for key in _STEP_DATA_IDENTIFYING_KEYS if key in data]
    # At most one pass per identifying key plus the final bare-marker
    # pass: each iteration that fails drops exactly one key, and the
    # bare marker on its own is always far under the cap.
    for _ in range(len(_STEP_DATA_IDENTIFYING_KEYS) + 1):
        capped = dict(bare_marker)
        for key in surviving:
            capped[key] = data[key]
        if _byte_length(capped) <= MAX_FRAME_CONTENT_BYTES:
            return capped
        string_keys = [key for key in surviving if isinstance(capped[key], str)]
        overhead = _byte_length({**capped, **{key: "" for key in string_keys}})
        per_key_budget = max(
            0, (MAX_FRAME_CONTENT_BYTES - overhead) // max(len(string_keys), 1)
        )
        for key in string_keys:
            capped[key], _ = _capped_text(capped[key], limit=per_key_budget)
        if _byte_length(capped) <= MAX_FRAME_CONTENT_BYTES:
            return capped
        candidates = [
            key for key in surviving if not isinstance(data[key], str)
        ] or surviving
        surviving.remove(max(candidates, key=lambda key: _byte_length(data[key])))
    return bare_marker


def step_started_frame(public_step: dict[str, Any]) -> str:
    return _sse_frame("step.started", {"step": public_step})


def step_completed_frame(public_step: dict[str, Any]) -> str:
    return _sse_frame("step.completed", {"step": public_step})


def message_delta_frame(message_id: str, text: str) -> str:
    capped_text, truncated = _capped_text(text)
    data: dict[str, Any] = {"message_id": message_id, "text": capped_text}
    if truncated:
        data["truncated"] = True
    return _sse_frame("message.delta", data)


def message_completed_frame(message_id: str, content: str) -> str:
    capped_content, truncated = _capped_text(content)
    data: dict[str, Any] = {"message_id": message_id, "content": capped_content}
    if truncated:
        data["truncated"] = True
    return _sse_frame("message.completed", data)


def _step_wire_frame(public_step_json: dict[str, Any]) -> str:
    """Pick ``step.started`` vs ``step.completed`` for an already-JSON-safe
    step dict, after applying the single-frame content cap to its
    ``data`` sub-object. One status -> event-name rule, and one size cap,
    in one place: the only producer of step content on this stream is
    live folding (``_step_content_frame``, below), and keeping the rule
    here rather than inline there is what lets a second producer be added
    later without a second copy of it that could disagree.
    """
    capped = {**public_step_json, "data": _capped_step_data(public_step_json["data"])}
    if capped["status"] == "running":
        return step_started_frame(capped)
    return step_completed_frame(capped)


def _step_content_frame(step: dict[str, Any]) -> str:
    """Serialize one projector-produced step dict to its SSE frame.

    Called synchronously, immediately after the ``PublicStepProjector``
    call that produced ``step`` returns -- never stored or handed off
    first. ``feed()``/``materialized_steps()`` return the projector's own
    *live* dict objects, which get mutated in place when a pending step's
    end event later arrives (see ``_step_mapping.py``'s class docstring);
    holding one past this point instead of serializing it right away
    would risk a ``step.started`` frame's queued text reflecting a status
    the step had moved past by the time it was actually read out of the
    queue. Routing every field through ``PublicStep`` here is also what
    keeps this wire shape identical to ``steps()``'s -- same validation,
    same JSON coercion (``started_at``/``completed_at`` in particular),
    same public step-type list, zero second copy of any of it.
    """
    return _step_wire_frame(PublicStep(**step).model_dump(mode="json"))


def _feed_trace_event(
    projector: "PublicStepProjector", event_type: str, message: dict[str, Any]
) -> list[str]:
    """Fold one ``type == "trace_event"`` broadcast frame into ``projector``
    and serialize whatever step(s) it changed.

    Three filters run before anything reaches the projector, all on the
    frame's *original* ``event_type``/``data``:

      1. ``data["source"] == DELEGATED_AGENT_TRACE_SOURCE``: drop trace
         events from a delegated child agent's own run. ``steps()``
         reaches the same outcome through a different check: a
         ``TraceEvent.build_id IS NULL`` column filter in its own query
         (``tasks.py``'s ``_load_task_steps_snapshot`` /
         ``_load_task_steps_version_snapshot``), not
         ``public_trace_events.public_task_trace_filter`` (that
         predicate ORs delegated-child-agent traces back *in*, for a
         different consumer -- the opposite of what this drop does).
         Reimplemented here as a data-field check, rather than reused
         directly, because the live broadcast fan-out has no SQL WHERE
         clause to piggyback on.
         The two agree because one producer stamps both the
         ``data["source"]`` field and the ``build_id`` column on a
         delegated child's events; nothing enforces that they keep
         agreeing, which is why the both-paths test persists a
         ``build_id`` row rather than only sending an in-memory frame.
      2. ``event_type == "task_info"``: ``task_info`` is never a public
         step; it only ever drives ``task.status``/``task.input_required``
         (handled elsewhere in ``send_text``, unconditionally, before this
         function is called), so it short-circuits here before it would
         otherwise fall through to the catch-all "not a known step
         family" drop below.
      3. ``is_audit_only_trace_data``: server-only RCA payloads that must
         never reach a client.
         ``steps()`` applies no equivalent filter on its own read. No
         audit-only row reaches a public step there today because the
         event types that carry audit-only data are not in the
         projector's exposed set, but that equivalence is a coincidence
         of the current producer rather than something either surface
         enforces (#1405).

    An ``ai_message`` event carrying ``data["stream_message_id"]`` --
    the trace-event mirror of a final answer already delivered
    token-by-token via ``message.delta``/``message.completed`` -- is
    deliberately NOT filtered here: it folds into a ``message`` step
    the same way any other ``ai_message`` does, exactly matching what
    ``steps()`` already shows for the same persisted row. A client that
    already rendered the delta/completed sequence sees this step as a
    duplicate of content it has, not as new information -- duplication,
    not loss, is the accepted trade-off. The alternative -- dropping it
    here -- has no persisted counterpart: ``steps()`` surfaces this same
    row as a ``message`` step regardless, so filtering it only on the
    live path would make the stream disagree with ``steps()`` about
    whether the step exists.
    """
    data = message.get("data")
    if not isinstance(data, dict):
        return []
    if data.get("source") == DELEGATED_AGENT_TRACE_SOURCE:
        return []
    if event_type == "task_info":
        return []
    if is_audit_only_trace_data(data):
        return []
    # Measured cost of this pass per sink (serialize, normalize, fold,
    # validate, serialize to wire): by direct call on a development
    # machine, median of 7, no I/O -- 6.2 ms (min 6.0 / max 6.3) for a raw
    # frame just under ``MAX_RAW_FRAME_TEXT_CHARS``, 1.6 ms at a 64 KiB
    # payload. ``PER_TASK_STREAM_CAP`` is 2, so one broadcast's sequential
    # fan-out adds at most ~12.4 ms.
    serialized_data = serialize_trace_data(data)
    normalized_type, normalized_data = normalize_public_trace_event(
        event_type, serialized_data
    )
    # Same six fields as ``tasks.py``'s ``_TraceEventSnapshot`` -- the
    # projector's ``_safe_get``/``_data_get`` accessors read both ORM rows
    # and plain dicts shaped like this one, so no adapter class is needed
    # for the live path.
    live_event = {
        "task_id": message.get("task_id"),
        "event_id": message.get("event_id"),
        "event_type": normalized_type,
        "timestamp": message.get("timestamp"),
        "step_id": message.get("step_id"),
        "data": normalized_data,
    }
    changed = projector.feed(live_event)
    return [_step_content_frame(step) for step in changed]


def _feed_final_answer(frame_type: str, message: dict[str, Any]) -> list[str]:
    """Project one ``final_answer_delta``/``final_answer_end`` broadcast
    frame directly to its ``message.*`` SSE frame -- no projector involved.

    These frames carry no ``event_type`` key and are never persisted
    individually, so there's nothing to pair or replay here; they map
    straight through. ``final_answer_start`` and ``final_answer_error``
    are both deliberately not projected: a
    ``message.delta`` sequence may therefore end with no
    ``message.completed`` -- the next lifecycle event (``task.status`` /
    ``task.completed`` / a fresh ``message.delta`` for a different
    ``message_id``) is the client's signal that it was abandoned, not a
    dedicated close event.
    """
    message_id = message.get("message_id")
    if not isinstance(message_id, str):
        return []
    if frame_type == "final_answer_delta":
        text = message.get("delta")
        if not isinstance(text, str):
            return []
        return [message_delta_frame(message_id, text)]
    content = message.get("content")
    if not isinstance(content, str):
        return []
    return [message_completed_frame(message_id, content)]


def project_content_frames(
    message: dict[str, Any], projector: "PublicStepProjector"
) -> list[str]:
    """Classify one already-parsed broadcast dict and return its
    ``step.*`` / ``message.*`` SSE frames, if any (see this section's
    header comment for the classification rules and the one deliberate
    omission from them).

    Every other frame family this module already understands --
    ``task_info``, the other versioned lifecycle types, the bare
    ``task_completed`` acceleration dict -- produces no content frames
    here; ``send_text`` handles those itself and calls this
    unconditionally alongside that handling, not instead of it.
    """
    frame_type = str(message.get("type") or "")
    if frame_type not in _CONTENT_FRAME_TYPES:
        return []
    if frame_type == "trace_event":
        return _feed_trace_event(
            projector, str(message.get("event_type") or ""), message
        )
    return _feed_final_answer(frame_type, message)


def _measured_content_frame(
    text: str, message: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Return ``(chars, frame)`` -- the length the drop check below
    compares against the cap, and the frame projection should consume.

    ``ws_trace_handlers.py``'s ``_convert_trace_event_to_stream_event``
    copies the task's ``description`` column onto
    ``data["task_description"]`` for *every* trace event it converts,
    with no event-type gate. That column has no length bound
    (``Column(Text)``, filled from the task's first user message, which
    has no ``max_length`` of its own), and it is re-stamped on each
    frame -- so counting it would let one long description push every
    subsequent ``step.*`` frame of that task past
    ``MAX_RAW_FRAME_TEXT_CHARS`` and drop the task's whole content
    stream for the life of the connection.

    Excluding it costs the client nothing, because no content this
    module puts on the wire can contain it: each of the four producers
    builds its ``data`` from explicitly named keys (``_step_mapping``'s
    ``_build_thinking_start`` / ``_build_tool_start`` /
    ``_build_message_step`` and the ``extra_data_fn`` patches, then
    ``message_delta_frame`` / ``message_completed_frame``), so the
    description was consuming a budget it could never spend.

    An under-cap frame is returned unchanged, description and all --
    the walk that projection pays over it is already bounded by the
    cap. An over-cap frame carrying a description is pruned *before*
    both measuring and projecting, not just before measuring: the raw
    frame check was this field's only per-frame CPU bound (a
    synchronous, unbounded ``clean_string`` walk during projection), so
    a frame that survives the check must not still be paying for the
    field the check just excused it from. Measured by re-serializing
    the parsed frame without that one field rather than by subtracting
    an estimate of the field's serialized length: an estimate would
    have to reproduce the producer's separators and escaping, while a
    re-dump is exact in the same character domain ``len(text)`` is
    measured in (``broadcast_to_task`` serializes with a default
    ``json.dumps``, and so does this). It runs only for a frame that is
    both over the cap and carrying a description -- the frame that
    would otherwise be dropped outright -- so an ordinary frame still
    costs one ``len()``, and the re-dump's own cost scales with the
    frame *without* the description, not with the description.
    ``json.dumps`` cannot fail here: ``message`` came out of
    ``json.loads`` in the caller.
    """
    if len(text) <= MAX_RAW_FRAME_TEXT_CHARS:
        return len(text), message
    data = message.get("data")
    if not isinstance(data, dict) or "task_description" not in data:
        return len(text), message
    pruned = {key: value for key, value in data.items() if key != "task_description"}
    frame = {**message, "data": pruned}
    return len(json.dumps(frame)), frame


# -- Sink -----------------------------------------------------------------


class V1EventStreamSink:
    """One SSE consumer's broadcast-frame filter and outbound queue.

    Bound to exactly one ``task_id`` and one ``principal`` for the life
    of the connection. The principal is used only for quota accounting
    and the watchdog's key-validity check -- it is **not** an
    authorization gate for who may act on the task; that check already
    happened once, at attach time, via ``get_principal_from_api_key`` +
    ``_resolve_task_or_404``.
    """

    # This sink only consumes the broadcast fan-out; it must never be
    # picked as the target for a personal reply to a durable command (see
    # ``websocket.py``'s ``_execute_durable_task_command``), since it only
    # understands versioned task-status events and would silently drop
    # anything else routed to it.
    is_broadcast_only = True

    def __init__(
        self, *, task_id: int, principal_key_prefix: str, initial_status: str
    ) -> None:
        self.task_id = task_id
        self.principal_key_prefix = principal_key_prefix
        self.dropped_frame_count = 0
        # Wire bytes currently sitting in ``_queue``. Maintained at the
        # three places that touch the queue and nowhere else: ``_put_counted``
        # adds a frame's length as it goes in, ``next_frame`` subtracts it
        # as it comes out, and ``enqueue_close`` zeroes it after draining.
        # No caller outside this class adjusts it.
        self._queued_bytes = 0
        self.completion_hint = asyncio.Event()
        # Each element is ``(frame_text, is_close)``. ``is_close`` travels
        # with the frame itself rather than being inferred from
        # ``self._closing`` at dequeue time -- the generator can be
        # suspended at ``yield`` on an *earlier*, non-close frame while a
        # concurrent ``enqueue_close`` flips ``_closing`` and appends the
        # close frame behind it; reading ``_closing`` after that yield
        # would wrongly treat the earlier frame as the close and return
        # without ever delivering the close frame still sitting in the
        # queue.
        self._queue: "asyncio.Queue[tuple[str, bool]]" = asyncio.Queue(
            maxsize=OUTBOUND_QUEUE_MAX_SIZE
        )
        self._closing = False
        self._last_status = initial_status
        # One incremental folding state machine per connection (never
        # shared across sinks -- see ``PublicStepProjector``'s own
        # preconditions). Starts empty, and every content-bearing frame
        # broadcast from then on is fed straight into it by ``send_text``.
        # Because it starts empty, a step already running at attach time
        # has no matching start here, so that step's end event folds in
        # as an orphan and is dropped -- see the module docstring.
        # ``retain_finished=False``: this sink acts on each ``feed()``
        # result immediately (``_step_content_frame`` serializes it on
        # the spot) and never calls ``materialized_steps()``, so
        # retaining every finalized step would hold each one's
        # untruncated ``data`` -- tool args and results, delegation
        # input/output, message content -- for as long as the connection
        # lives (up to ``STREAM_MAX_DURATION_SECONDS``) in a list nothing
        # on this path reads. The pending table the pairing rules need is
        # kept either way.
        self._projector = PublicStepProjector(retain_finished=False)
        # Recorded at construction time (always inside the endpoint's own
        # request coroutine) so later state-mutating calls -- however they
        # got here -- can be asserted to run on the same loop.
        self._owner_loop = asyncio.get_running_loop()

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def queue(self) -> "asyncio.Queue[tuple[str, bool]]":
        """The raw outbound queue, for inspecting depth and emptiness.

        Take frames off it through ``next_frame`` instead of reading this
        directly: that method is where a dequeued frame's bytes are
        discounted from ``queued_wire_bytes``, so a bare ``get`` /
        ``get_nowait`` here drains the queue while leaving the byte
        accounting reading high.
        """
        return self._queue

    @property
    def queued_wire_bytes(self) -> int:
        """Total wire bytes of the frames currently queued."""
        return self._queued_bytes

    def _assert_owner_loop(self) -> None:
        if asyncio.get_running_loop() is not self._owner_loop:
            raise RuntimeError("v1 SSE sink touched from a foreign event loop")

    def enqueue_status(self, status: str) -> None:
        """Enqueue a deduped ``task.status`` frame. Never closes the stream."""
        self._assert_owner_loop()
        if self._closing or status == self._last_status:
            return
        self._last_status = status
        self._put_or_overflow(status_frame(status))

    def enqueue_close(self, frame_text: str) -> bool:
        """Close exactly once; the first caller wins.

        Drains any queued backlog before inserting the close frame so
        the close frame always has room -- even when called *because*
        the queue just overflowed. Losing unread backlog on close is
        intentional: every close reason tells the client to resync
        (``steps()`` + re-attach) rather than trust the tail of the
        stream.
        """
        self._assert_owner_loop()
        if self._closing:
            return False
        self._closing = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Exact rather than approximate: the loop above only exits with the
        # queue empty (one event loop, no concurrent producer), so there
        # are no bytes left to account for.
        self._queued_bytes = 0
        # Deliberately not routed through the byte budget: this is the one
        # frame that must always fit. ``task.completed``'s ``output`` and
        # ``task.input_required``'s ``prompt`` are uncapped by design (see
        # the comment above ``completed_frame``); on the live path they
        # only ever reach the client as a budget-exempt close frame via
        # this method, and on the attach-time fast paths there is no sink
        # and no queue at all -- refusing one here on a budget while the
        # stream is closing anyway would drop a payload the client cannot
        # recover from this stream, with no backlog left to protect.
        self._put_counted(frame_text, is_close=True)
        return True

    def _put_counted(self, frame_text: str, *, is_close: bool) -> None:
        """The only place a frame enters the outbound queue.

        Single write point so ``_queued_bytes`` cannot fall out of step
        with the queue's actual contents: a frame that is queued is
        counted in the same statement, and if ``put_nowait`` raises
        ``QueueFull`` the count is not touched either.
        """
        self._queue.put_nowait((frame_text, is_close))
        self._queued_bytes += len(frame_text)

    async def next_frame(self) -> tuple[str, bool]:
        """Take the next queued frame and discount its bytes.

        The accounted counterpart of ``_put_counted``, and the way the
        generator consumes this queue. There is no ``await`` between the
        queue read and the subtraction, so a cancelled wait (the
        generator's per-frame ``wait_for`` timeout) can never leave a
        frame dequeued but still counted.
        """
        self._assert_owner_loop()
        frame_text, is_close = await self._queue.get()
        self._queued_bytes -= len(frame_text)
        return frame_text, is_close

    def _put_or_overflow(self, frame_text: str) -> None:
        # Defense in depth: every current caller already checks ``closing``
        # first, but a closed sink must never grow its queue again no
        # matter how it's reached, so the guard lives here too.
        if self._closing:
            return
        # Two bounds, one close path. Checked before the put rather than
        # after it so ``_queued_bytes`` never exceeds the budget even
        # transiently. Strictly over closes, exactly at the budget does
        # not -- the same boundary rule ``MAX_RAW_FRAME_TEXT_CHARS`` uses.
        if self._queued_bytes + len(frame_text) > MAX_QUEUED_WIRE_BYTES:
            self.enqueue_close(error_frame("resync_required"))
            return
        try:
            self._put_counted(frame_text, is_close=False)
        except asyncio.QueueFull:
            self.enqueue_close(error_frame("resync_required"))

    def _project_and_queue(self, message: dict[str, Any]) -> None:
        """Project one content frame and enqueue its output. Never raises --
        a frame that fails to project is dropped and counted, exactly like a
        frame that fails on the live path; see ``send_text``'s discipline note.
        """
        try:
            for frame_text in project_content_frames(message, self._projector):
                self._put_or_overflow(frame_text)
        except Exception:
            self.dropped_frame_count += 1
            logger.exception(
                "v1 SSE sink dropped a content frame for task %s", self.task_id
            )

    async def send_text(self, text: str) -> None:
        """Receive one broadcast frame. Duck-types ``WebSocket.send_text``.

        Must never raise. ``ConnectionManager.broadcast_to_task``
        catches network errors from a connection's ``send_text`` and drops
        the connection, but re-raises anything else -- that re-raise
        happens on the *task's own* outbound event path, so an uncaught
        exception here would abort the broadcast for every other
        listener on the task, not just this stream. Every drop this
        outer ``except`` catches is logged via ``logger.exception``
        below, and also counted on ``dropped_frame_count`` for tests to
        assert against (no production code reads that counter today).
        A content frame that fails to *project* is caught and counted
        one layer in, by ``_project_and_queue`` itself -- see that
        method's docstring -- so this outer ``except`` is left to catch
        everything else that can go wrong while parsing or dispatching a
        broadcast frame (bad JSON, a non-dict payload, a bug in the
        lifecycle handling above). The two never double-count the same
        failure: a projection failure never propagates up to this
        ``except`` in the first place.

        Content frames (``step.*`` / ``message.*``) are classified and
        projected alongside the lifecycle handling below, not instead of
        it -- ``project_content_frames`` returns an empty list for every
        frame shape this method already handles on its own
        (``task_completed``, the versioned lifecycle types, ``task_info``),
        so the two never fight over the same frame. A content frame is
        projected and queued via ``_project_and_queue``, which always
        routes successfully-projected output through ``_put_or_overflow``
        like any other queued frame -- never a direct
        ``queue.put_nowait`` -- so it gets the same closed guard and the
        same overflow-to-``resync_required`` handling status frames
        already get, on both of the queue's bounds: its element count
        and its total queued wire bytes.

        The raw-size check itself runs after ``json.loads`` and the
        lifecycle handling above, guarding only the call into
        ``_project_and_queue`` -- the last thing between a parsed frame
        and the projector. It applies only to frames whose parsed
        ``type`` is in ``_CONTENT_FRAME_TYPES``; dispatch into the
        projector is unchanged for that whole set regardless. What it
        compares against ``MAX_RAW_FRAME_TEXT_CHARS`` (256 KiB) is the
        frame's own character count minus the ``task_description``
        field the broadcast conversion stamps onto every trace event --
        see ``_measured_content_frame`` for why that subtraction exists
        and why it costs nothing on the ordinary path. Strictly greater
        than the cap is a drop, exactly equal is kept. On a drop,
        ``dropped_frame_count`` is incremented, one ``logger.warning``
        fires, and the method returns -- no queued frame, no close, no
        ``stream.error``: a dropped frame is invisible to the client.

        Two separate mechanisms keep lifecycle delivery out of this
        check's way, and they are not interchangeable:

          - Placement. Running after ``json.loads`` and after the
            lifecycle handling above is what makes an oversized frame
            unable to cost a client its lifecycle signal:
            ``task_completed`` returns on the acceleration branch
            before this check is reached, and ``task.status`` /
            ``completion_hint`` have already been enqueued or set by
            the time it runs. A gate placed ahead of the parse would
            drop those frames whole -- and could not tell them apart
            anyway, since a ``task_info`` frame's own top-level
            ``type`` is ``trace_event``, the same value real content
            frames carry (see ``MAX_RAW_FRAME_TEXT_CHARS``'s own
            comment for why ``task_completed`` can legitimately cross
            this threshold too).
          - The ``event_type != "task_info"`` exemption. By the time
            control reaches it, a ``task_info`` frame's status is
            already enqueued above, and ``_feed_trace_event`` returns
            ``[]`` for ``task_info`` whether the exemption is there or
            not. So the exemption decides exactly one thing: whether
            an oversized ``task_info`` increments
            ``dropped_frame_count`` and logs a drop warning. It is not
            what protects ``task.status``/``completion_hint``
            delivery. It is there because such a frame carries the
            task description with no size bound of its own, so
            counting it as a dropped content frame would be counting a
            drop that had no content to lose.

        The parse is not extra cost on the fan-out path:
        ``ConnectionManager.broadcast_to_task`` serializes the frame
        with ``json.dumps`` once per connection, inside its own send
        loop, before this method is called at all. What happens to the
        expensive half here -- ``serialize_trace_data`` plus the
        projection fold -- depends on where the frame lands: a frame
        under the cap pays it in full; a frame over the cap only
        because of its task description is pruned by
        ``_measured_content_frame`` before projection runs, not just
        before the comparison, so it pays that cost against the frame
        without the description instead; and a frame still over the
        cap after pruning is dropped outright and pays none of it.
        """
        try:
            self._assert_owner_loop()
            if self._closing:
                return
            message = json.loads(text)
            if not isinstance(message, dict):
                return
            if message.get("type") == "task_completed":
                # Acceleration signal only: the authoritative completion
                # frame still comes from the watchdog reading the task
                # row, just woken up early instead of waiting out its
                # normal cadence. This keeps the sink itself from ever
                # touching the database -- no query happens here.
                self.completion_hint.set()
                return
            if _is_versioned_task_event(message):
                status = message.get("status")
                if isinstance(status, str):
                    self.enqueue_status(status)
                    if status in _EARLY_WAKE_STATUS_VALUES:
                        # A failed task's broadcast (``task_error``) never
                        # carries ``type == "task_completed"``, and a task
                        # moving to ``waiting_for_user`` never does either,
                        # so both would otherwise miss the acceleration
                        # signal above and wait out the watchdog's normal
                        # cadence. Safe to wake early on either: the
                        # watchdog re-reads the authoritative row (via
                        # ``_stream_close_reason``) before closing anything,
                        # including the ``resume_requested`` carve-out, so
                        # an early wake never closes a stream that a fresh
                        # read wouldn't have closed anyway.
                        self.completion_hint.set()
            if str(message.get("type") or "") in _CONTENT_FRAME_TYPES:
                frame = message
                if message.get("event_type") != "task_info":
                    measured_chars, frame = _measured_content_frame(text, message)
                    if measured_chars > MAX_RAW_FRAME_TEXT_CHARS:
                        self.dropped_frame_count += 1
                        logger.warning(
                            "v1 SSE sink dropped an oversized broadcast frame "
                            "(%d chars, %d excluding the task description) for "
                            "task %s before projecting it",
                            len(text),
                            measured_chars,
                            self.task_id,
                        )
                        return
                self._project_and_queue(frame)
        except Exception:
            self.dropped_frame_count += 1
            logger.exception(
                "v1 SSE sink dropped a broadcast frame for task %s", self.task_id
            )

    async def close(self) -> None:
        """No-op. Duck-types ``WebSocket.close`` -- task deletion
        (``chat.py``'s ``_cleanup_runtime_state``) calls ``close()`` on every
        connection still registered for the task, and a missing method here
        would log a misleading "failed to close" warning for a sink that was
        never a real socket in the first place. This generator's own
        ``finally`` block is what actually tears the stream down.
        """
        return None


# -- Per-principal concurrency accounting --------------------------------

_principal_stream_counts: dict[str, int] = {}


def try_reserve_principal_slot(key_prefix: str) -> bool:
    current = _principal_stream_counts.get(key_prefix, 0)
    if current >= PER_PRINCIPAL_STREAM_CAP:
        return False
    _principal_stream_counts[key_prefix] = current + 1
    return True


def principal_slot_available(key_prefix: str) -> bool:
    """Read-only capacity check -- would ``try_reserve_principal_slot``
    currently succeed for this key. Used at response-construction time
    (``build_event_stream_response``) so a 429 is raised before any
    stream opens, without mutating the counter yet: the actual
    reservation happens once the generator starts running (see
    ``_generate``), so a response that's constructed but never iterated
    never touches this counter."""
    return _principal_stream_counts.get(key_prefix, 0) < PER_PRINCIPAL_STREAM_CAP


def release_principal_slot(key_prefix: str) -> None:
    current = _principal_stream_counts.get(key_prefix, 0)
    if current <= 1:
        _principal_stream_counts.pop(key_prefix, None)
    else:
        _principal_stream_counts[key_prefix] = current - 1


def reset_principal_stream_counts_for_testing() -> None:
    _principal_stream_counts.clear()


def count_task_sinks(task_id: int) -> int:
    """Count only v1 SSE sinks for a task -- WebSocket connections on the
    same task_id don't share this concurrency cap."""
    return sum(
        1
        for connection in manager.connections_for_task(task_id)
        if isinstance(connection, V1EventStreamSink)
    )


# -- Watchdog --------------------------------------------------------------


def _is_runtime_key_active(key_prefix: str) -> bool:
    """One indexed lookup, no bcrypt (the handshake already verified the
    secret; this only re-checks revoked/paused)."""
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return (
            db.query(AgentApiKey.id)
            .filter(*active_runtime_key_filters(key_prefix))
            .first()
            is not None
        )


async def watchdog_check_once(
    sink: V1EventStreamSink,
    task_id: int,
    principal: ApiKeyPrincipal,
    *,
    read_task_snapshot: TaskSnapshotReader,
) -> bool:
    """One watchdog check pass. Returns True iff it closed the stream.

    Order: key validity first (the auth axis, fail closed), then the
    task row's own state. All reads go through
    ``run_db_io_cancellation_safe``; none of this runs on the sink's
    ``send_text`` hot path, so the sink itself still never touches the
    database.
    """
    key_active = await run_db_io_cancellation_safe(
        lambda: _is_runtime_key_active(sink.principal_key_prefix)
    )
    if not key_active:
        return sink.enqueue_close(error_frame("unauthorized"))

    try:
        snapshot = await run_db_io_cancellation_safe(
            lambda: read_task_snapshot(task_id, principal)
        )
    except V1ApiError as exc:
        if exc.code is V1ErrorCode.TASK_NOT_FOUND:
            return sink.enqueue_close(error_frame("task_deleted"))
        raise

    status = snapshot.status
    close_reason = _stream_close_reason(status, snapshot.control_state)
    if close_reason == "terminal":
        return sink.enqueue_close(
            completed_frame(
                status=status.value,
                output=snapshot.output,
                error=snapshot.error,
            )
        )
    if close_reason == "input_required":
        return sink.enqueue_close(
            input_required_frame(task_id, snapshot.pending_question)
        )
    if status is TaskStatus.PAUSED:
        # SDK wait() semantics: PAUSED keeps waiting, doesn't close.
        # A "was this paused by an orphaned process?" check was considered
        # and rejected: a normal pause and an orphaned one leave identical
        # values in every lease-tracking column (runner id, lease
        # expiry, last heartbeat, control state) -- both the normal-pause
        # and lease-recovery code paths stamp those columns with "now",
        # so they only measure how long ago the pause happened, not
        # whether anyone is still managing it. Closing on that signal
        # would kill legitimately paused streams too. The 1-hour absolute
        # cap is what actually bounds an orphaned paused stream's lifetime.
        sink.enqueue_status(status.value)
        return False
    return False  # pending/running: keep streaming


async def _watchdog_loop(
    sink: V1EventStreamSink,
    task_id: int,
    principal: ApiKeyPrincipal,
    *,
    read_task_snapshot: TaskSnapshotReader,
    interval_seconds: float,
) -> None:
    """Runs every ``interval_seconds`` and also wakes early on a
    completion hint from a terminal (completed or failed) or
    waiting-for-user broadcast: same check, just run sooner.

    A single failed check (e.g. a transient DB error) must not end
    watchdog coverage for the rest of the stream's lifetime -- that
    would leave an orphaned stream open until the 1-hour absolute cap
    with nobody watching it. So every per-cycle check is wrapped and
    logged; only cancellation (this loop's own teardown, not a check
    failure) ends the loop early.
    """
    while not sink.closing:
        try:
            await asyncio.wait_for(
                sink.completion_hint.wait(), timeout=interval_seconds
            )
        except asyncio.TimeoutError:
            pass
        else:
            sink.completion_hint.clear()
        if sink.closing:
            return
        try:
            if await watchdog_check_once(
                sink, task_id, principal, read_task_snapshot=read_task_snapshot
            ):
                return
        except Exception:
            logger.exception(
                "v1 SSE watchdog check failed for task %s; retrying next cycle",
                task_id,
            )


# -- Response assembly ----------------------------------------------------

# nginx buffers proxied responses by default; X-Accel-Buffering is the
# per-response override that keeps SSE frames flowing immediately instead
# of waiting for the proxy buffer to fill.
_SSE_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


def _sse_response(body: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(body, media_type="text/event-stream", headers=_SSE_HEADERS)


async def _terminal_snapshot_stream(
    snapshot: "_TaskInfoSnapshot",
) -> AsyncIterator[str]:
    """Attach-time fast path for an already-terminal task: emit
    ``task.status`` + ``task.completed`` and end. No sink, no
    registration, no watchdog -- there's nothing left to watch."""
    yield status_frame(snapshot.status.value)
    yield completed_frame(
        status=snapshot.status.value, output=snapshot.output, error=snapshot.error
    )


async def _input_required_snapshot_stream(
    snapshot: "_TaskInfoSnapshot",
) -> AsyncIterator[str]:
    """Attach-time fast path for a task already waiting on user input (and
    not mid-resume): emit ``task.status`` + ``task.input_required`` and
    end. Same rationale as ``_terminal_snapshot_stream`` -- without this,
    the same conclusion is only reached via the watchdog's first cycle,
    up to ``watchdog_interval_seconds`` (30s in production) after attach."""
    yield status_frame(snapshot.status.value)
    yield input_required_frame(snapshot.task_id, snapshot.pending_question)


async def _generate(
    task_id: int,
    principal: ApiKeyPrincipal,
    *,
    key_prefix: str,
    initial_status: str,
    read_task_snapshot: TaskSnapshotReader,
    watchdog_interval_seconds: float,
    stream_max_duration_seconds: float,
    heartbeat_interval_seconds: float,
) -> AsyncIterator[str]:
    """Build the sink, register it, run the stream, tear it all down.

    Sink construction, ``manager`` registration, and the per-principal
    slot reservation all happen *inside* this ``try`` -- deliberately
    not in ``build_event_stream_response`` -- because an async
    generator's body doesn't run at all until it's first iterated. If
    registration/reservation happened at response-construction time
    instead, a ``StreamingResponse`` that gets built but never iterated
    (e.g. the caller closes it before Starlette ever pulls a chunk)
    would leak both: this generator's ``finally`` -- the only code that
    unregisters and releases -- would simply never execute. Deferring
    both into here means whatever starts this generator (even just one
    ``aclose()`` with no frames read) is guaranteed to reach the
    ``finally`` and clean up exactly what it reserved.

    The 429 capacity *checks* still happen earlier, in
    ``build_event_stream_response`` (read-only, no mutation) -- so a
    rejected attach still fails before any stream bytes are sent; only
    the actual counter mutation and registration move here.
    """
    deadline = monotonic() + stream_max_duration_seconds
    sink = V1EventStreamSink(
        task_id=task_id, principal_key_prefix=key_prefix, initial_status=initial_status
    )
    principal_slot_reserved = False
    watchdog_task: "asyncio.Task[None] | None" = None
    try:
        # Soft cap: `principal_slot_available` in `build_event_stream_response`
        # already did the *check* before this generator started, and that
        # check is what carries the normal 429 -- an attach that loses the
        # race here has already been told "yes, come in", so this
        # reservation only accounts for capacity, it never rejects. If a
        # concurrent burst of attaches for the same principal all pass the
        # earlier read-only check and then land here before any of them
        # releases a slot, `try_reserve_principal_slot` returns False for
        # the late arrivals: the number of open streams for this principal
        # can briefly exceed `PER_PRINCIPAL_STREAM_CAP` -- the counter
        # itself never does, since `try_reserve_principal_slot` checks
        # before it increments -- it's logged, and the stream is served
        # anyway. The sentinel (`principal_slot_reserved`) stays False for
        # these streams, so the `finally` below correctly skips
        # `release_principal_slot` for them -- nothing was reserved, so
        # nothing is released. The open-stream count self-heals as soon as
        # any concurrently-open stream for this principal finishes and
        # releases its own slot.
        principal_slot_reserved = try_reserve_principal_slot(key_prefix)
        if not principal_slot_reserved:
            logger.warning(
                "v1 SSE per-principal cap best-effort exceeded for "
                "key_prefix=%s under concurrent attach burst; serving the "
                "stream anyway (soft cap, no reservation held)",
                key_prefix,
            )
        manager.register_connection(cast(WebSocket, sink), task_id)
        watchdog_task = asyncio.create_task(
            _watchdog_loop(
                sink,
                task_id,
                principal,
                read_task_snapshot=read_task_snapshot,
                interval_seconds=watchdog_interval_seconds,
            )
        )
        # The initial yield must be inside this ``try`` too: a generator
        # closed (``aclose()``, e.g. on client disconnect) while suspended
        # here still has to unregister the sink and cancel the watchdog.
        yield status_frame(initial_status)
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                # First-to-close-wins: a no-op if the watchdog already
                # closed the stream first in the same tick.
                sink.enqueue_close(error_frame("stream_expired"))
            # Capped at the heartbeat interval so the deadline check above
            # re-runs often enough, but never let it run past the deadline
            # itself -- otherwise an idle stream whose last ping landed just
            # before the deadline could sleep a full heartbeat interval past
            # ``stream_max_duration_seconds`` before ``stream_expired`` is
            # even enqueued.
            wait_budget = (
                min(
                    heartbeat_interval_seconds, max(remaining, _MIN_WAIT_BUDGET_SECONDS)
                )
                if remaining > 0
                else 1.0
            )
            try:
                frame_text, is_close = await asyncio.wait_for(
                    sink.next_frame(), timeout=wait_budget
                )
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            yield frame_text
            # ``is_close`` is the flag recorded on *this* frame at enqueue
            # time, not ``sink.closing`` re-read now -- see the queue
            # element's docstring in ``V1EventStreamSink.__init__`` for why
            # that distinction is load-bearing under concurrent
            # ``enqueue_close`` calls.
            if is_close:
                return
    finally:
        # The watchdog wait-and-cancel is wrapped in its own try/finally so
        # that `manager.disconnect` and `release_principal_slot` below --
        # the two calls that actually undo what this generator reserved --
        # still run even if awaiting the cancelled watchdog task raises
        # something unexpected. Without this inner `finally`, a raise here
        # would skip straight past both cleanup calls, leaking the sink
        # registration and (if held) the per-principal slot.
        try:
            if watchdog_task is not None:
                watchdog_task.cancel()
                # Narrowed to ``CancelledError`` only (not a blanket
                # ``BaseException``): the watchdog loop itself now catches
                # and logs every per-cycle ``Exception`` internally (see
                # ``_watchdog_loop``) and retries rather than dying, so the
                # only exception this teardown should ever observe here is
                # the cancellation just requested above. If the watchdog
                # task raises anything else, that's a real bug and must
                # propagate instead of being silently swallowed.
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog_task
        finally:
            # ``manager`` is typed against a real ``WebSocket``; this sink only
            # duck-types its ``send_text`` contract (module docstring) -- the
            # cast documents that intentional narrowing instead of suppressing
            # the type error blanket-wide. A safe no-op if registration above
            # never ran or never completed.
            manager.disconnect(cast(WebSocket, sink))
            if principal_slot_reserved:
                release_principal_slot(key_prefix)


async def build_event_stream_response(
    *,
    task_id: int,
    principal: ApiKeyPrincipal,
    initial_snapshot: "_TaskInfoSnapshot",
    read_task_snapshot: TaskSnapshotReader,
    watchdog_interval_seconds: float = WATCHDOG_INTERVAL_SECONDS,
    stream_max_duration_seconds: float = STREAM_MAX_DURATION_SECONDS,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> StreamingResponse:
    """Assemble the SSE ``StreamingResponse`` for one attach.

    ``initial_snapshot`` must already be authorized (i.e. come from
    ``_resolve_task_or_404`` via ``read_task_snapshot``) -- this
    function does no auth of its own. Concurrency caps (429) are
    checked here, before the generator (and therefore the sink and the
    per-principal reservation) ever exists, so a rejected attach never
    touches ``manager`` or the per-principal counter.
    """
    close_reason = _stream_close_reason(
        initial_snapshot.status, initial_snapshot.control_state
    )
    if close_reason == "terminal":
        return _sse_response(_terminal_snapshot_stream(initial_snapshot))

    if close_reason == "input_required":
        return _sse_response(_input_required_snapshot_stream(initial_snapshot))

    if count_task_sinks(task_id) >= PER_TASK_STREAM_CAP:
        raise V1ApiError(V1ErrorCode.RATE_LIMITED, 429)

    key_prefix = principal.key.key_prefix
    if not principal_slot_available(key_prefix):
        raise V1ApiError(V1ErrorCode.RATE_LIMITED, 429)

    return _sse_response(
        _generate(
            task_id,
            principal,
            key_prefix=key_prefix,
            initial_status=initial_snapshot.status.value,
            read_task_snapshot=read_task_snapshot,
            watchdog_interval_seconds=watchdog_interval_seconds,
            stream_max_duration_seconds=stream_max_duration_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
    )
