"""Faithfully reconstruct a task's prior conversation from persisted trace events.

Historically, ``task_execution_context_service.load_task_execution_context_messages``
collapsed all prior turns into a single synthetic ``system`` message: it kept only
the last few ``tool_execution_end`` events and clipped each result to a fixed
character budget with a blind string slice. That slicing could land mid-JSON and
destroy structured data (for example, cutting a continuation handle apart as
``"branch": "review-pr-1392", ...`` -> ``"bran``), which then fed a syntactically
broken fragment back to the model as "context" and caused it to hallucinate.

This module instead reconstructs the true ``assistant``/``tool`` message pairs a
live run would have produced, interleaved chronologically with the persisted
``user``/``assistant``/``system`` transcript rows.

This version reconstructs the full history with no size limiting: it is
deliberately not wired into any caller yet. Bounds (a per-result size cap, a
total exchange-count cap, and a total character budget) and the observability
around which of them fired ship in a follow-up change, alongside the wiring
that puts this service on the hot path.

One known deviation from the live shape, deliberately left for that same
follow-up: a single LLM response that returns several ``tool_calls`` becomes
one assistant message declaring all of them followed by a contiguous run of
tool results in a live run, but is reconstructed here as one
assistant/``tool`` pair per call. B's assistant message then sits after A's
result, which reads as a decision taken with A's result in hand when in fact
both calls were chosen with no results available. This is not gated by
``tool_parallel_enabled`` -- serial execution of a multi-call response has
the same shape. The persisted rows cannot currently distinguish the two
cases: tool trace events carry ``tool_call_id`` and ``turn_id`` but nothing
identifying which assistant response a call belongs to (``step_id`` is per
pattern run, ``turn_id`` per user message, ``parent_event_id`` is always
NULL for tool events), so fixing it means plumbing a per-response
discriminator through the runtime hooks first.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, cast

from sqlalchemy import asc
from sqlalchemy.orm import Session

from ...core.agent.attachments import build_image_context_references
from ...core.agent.result import CONTROL_TOOL_NAMES
from ...core.context_ref import CONTEXT_REFS_KEY, ContextReference
from ..models.chat_message import TaskChatMessage
from ..models.task import TraceEvent
from .chat_history_service import _MAX_HISTORICAL_IMAGE_CONTEXT_REFS

logger = logging.getLogger(__name__)

# Tool-side trace event types persisted by PatternRuntime (runtime.py).
_TOOL_EVENT_TYPES = (
    "tool_execution_start",
    "tool_execution_end",
    "tool_execution_failed",
)

# Transcript roles kept from ``task_chat_messages``, mirroring
# ``normalize_transcript_messages`` (core/agent/transcript.py).
_TRANSCRIPT_ROLES = {"user", "assistant", "system"}


@dataclass
class _TranscriptRow:
    """A single normalized ``task_chat_messages`` row."""

    row_id: int
    role: str
    content: str
    # ``TaskChatMessage.turn_id``, empty string when unset (rows written
    # before this column existed, or a role that never gets one). Used to
    # place this turn's reconstructed tool exchanges immediately after this
    # row -- see ``_merge_chronologically``.
    turn_id: str = ""
    # Raw ``TaskChatMessage.attachments`` payload, kept only long enough for
    # ``_attach_historical_image_context_refs`` to project it into
    # ``context_refs`` below; never read after ``_load_transcript_rows``
    # returns.
    attachments: Any = field(default=None, compare=False)
    # Uploaded-image context refs surviving the historical-image budget (see
    # ``_attach_historical_image_context_refs``). Populated after filtering,
    # so this is empty for any row that never reaches ``_render_messages``.
    context_refs: tuple[ContextReference, ...] = field(default=())


@dataclass
class _PendingToolStart:
    """A ``tool_execution_start`` awaiting its matching end/failure event."""

    assistant_content: str
    tool_name: str
    tool_params: Any
    sort_key: tuple[datetime, int]
    # The turn_id stamped on the "tool_execution_start" event's data (see
    # PatternRuntime.on_tool_start / _with_runtime_turn_id in react.py),
    # empty string when absent (legacy rows, or a call whose start event
    # fell outside this query's window).
    turn_id: str = ""


@dataclass
class _ToolExchange:
    """A reconstructed assistant/tool message pair."""

    call_id: str
    tool_name: str
    tool_params: Any
    result: Any
    assistant_content: str
    sort_key: tuple[datetime, int]
    # The durable turn_id this exchange belongs to (see ``_PendingToolStart
    # .turn_id`` and ``_load_tool_exchanges``), empty string when the
    # producing trace events predate turn_id support. Used by
    # ``_merge_chronologically`` to place this exchange immediately after
    # its turn's user transcript row -- a plain join, since
    # ``TaskChatMessage.turn_id`` and this value come from the same source
    # (see AgentRunner._ensure_user_message_turn_id). An empty turn_id means
    # the exchange is omitted rather than placed by guesswork.
    turn_id: str = ""


@dataclass
class _MergeEntry:
    """One item queued for the transcript/tool-exchange merge (R4)."""

    kind: str  # "group" | "transcript"
    group: Optional[list[_ToolExchange]] = None
    transcript: Optional[_TranscriptRow] = None


@dataclass
class _ReconstructionStats:
    """Counts-only bookkeeping for the single summary log line.

    No message content, tool results, or prose is ever stored here -- every
    field is a count or, where noted, a size. Populated as a side effect
    while the read pipeline runs, then emitted once by
    ``load_task_conversation_context_sync``.
    """

    # Transcript rows retained after role/content filtering (R1) -- the rows
    # that actually feed rendering, not the raw row count from the query.
    transcript_rows: int = 0
    # Tool exchanges that made it into the rendered output.
    exchanges_placed: int = 0
    # Exchanges with no usable turn_id (legacy rows, dropped in
    # ``_group_exchanges_by_turn``).
    exchanges_without_turn_id: int = 0
    # Exchanges whose turn_id never matched a surviving user transcript row
    # (``_merge_chronologically``): either that row's id fell outside
    # ``before_message_id``, or it was an image-only row evicted by
    # ``_attach_historical_image_context_refs``'s ref budget (see the
    # module's F1 fix note). Both causes look identical from here -- a
    # turn_id with no row to anchor on -- so they share one bucket.
    exchanges_with_unmatched_turn: int = 0
    # "tool_execution_start" events that never found a matching end/failure
    # event and so never became a rendered exchange at all (see
    # ``_load_tool_exchanges``). Behavior unchanged: still silently
    # disappears, just now counted.
    dangling_tool_starts: int = 0
    # "tool_execution_end"/"tool_execution_failed" events with no matching
    # start (renders with empty prose and empty tool_params). Behavior
    # unchanged: still renders degraded, just now counted.
    tool_ends_without_start: int = 0


def load_task_conversation_context_sync(
    db: Session,
    task_id: int,
    *,
    before_message_id: int | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct a task's prior conversation as planner-visible messages.

    Returns a chronological list of ``{"role": ..., ...}`` dicts covering the
    full persisted history: transcript rows plus reconstructed tool exchanges.
    This is a pure read: it never mutates the database and performs no async
    I/O, so it can be called from a worker thread that owns a live
    ``Session``.

    Args:
        db: An open, caller-owned database session.
        task_id: The task whose history to reconstruct.
        before_message_id: If given, only ``task_chat_messages`` rows with
            ``id < before_message_id`` are included (used when reconstructing
            context as of a specific point in the conversation).
    """
    stats = _ReconstructionStats()

    transcript_rows = _load_transcript_rows(
        db, task_id, before_message_id=before_message_id
    )
    exchanges = _load_tool_exchanges(db, task_id, stats=stats)

    merged = _merge_chronologically(transcript_rows, exchanges, stats=stats)

    messages = _render_messages(merged)
    messages = _final_pairing_sweep(messages)

    stats.transcript_rows = len(transcript_rows)
    logger.info(
        "task_conversation_context_reconstructed task_id=%s transcript_rows=%d "
        "exchanges_placed=%d exchanges_without_turn_id=%d "
        "exchanges_with_unmatched_turn=%d dangling_tool_starts=%d "
        "tool_ends_without_start=%d",
        task_id,
        stats.transcript_rows,
        stats.exchanges_placed,
        stats.exchanges_without_turn_id,
        stats.exchanges_with_unmatched_turn,
        stats.dangling_tool_starts,
        stats.tool_ends_without_start,
    )

    return messages


# ---------------------------------------------------------------------------
# R1 -- transcript rows
# ---------------------------------------------------------------------------


def _load_transcript_rows(
    db: Session,
    task_id: int,
    *,
    before_message_id: int | None,
) -> list[_TranscriptRow]:
    """Load ``task_chat_messages`` rows for the turn_id-joined reconstruction.

    ``chat_history_service``'s row-loading helpers normalize the row shape
    for a different caller, so the rows are queried directly here instead.
    ``attachments`` is selected alongside the existing narrow column set
    (rather than loading the whole ORM row) so the historical-image budget
    below can reach uploaded-image metadata without widening this query's
    result shape beyond what image support actually needs. ``created_at`` is
    deliberately NOT selected: placement of a turn's tool exchanges is a
    ``turn_id`` join (see ``_merge_chronologically``), not a timestamp
    comparison, and transcript ordering itself comes from ``TaskChatMessage
    .id`` (see ``order_by`` below), so nothing in this module ever needs a
    transcript row's clock time.
    """
    query = db.query(
        TaskChatMessage.id,
        TaskChatMessage.role,
        TaskChatMessage.content,
        TaskChatMessage.attachments,
        TaskChatMessage.turn_id,
    ).filter(TaskChatMessage.task_id == task_id)
    if before_message_id is not None:
        query = query.filter(TaskChatMessage.id < before_message_id)
    query = query.order_by(asc(TaskChatMessage.id))

    rows: list[_TranscriptRow] = []
    for row_id, role, content, attachments, turn_id in query.all():
        normalized_role = str(role or "").strip().lower()
        if normalized_role not in _TRANSCRIPT_ROLES:
            continue
        normalized_content = str(content or "").strip()
        rows.append(
            _TranscriptRow(
                row_id=int(row_id),
                role=normalized_role,
                content=normalized_content,
                turn_id=str(turn_id or ""),
                attachments=attachments,
            )
        )
    # Attach image context refs *before* dropping blank-content rows:
    # ``persist_user_message_no_commit`` (chat_history_service.py) documents
    # that "a row with empty content but non-empty attachments is still
    # persisted" -- an image-only user turn. Filtering on content alone
    # before this call would discard that row, and its attachment, together
    # -- the ref would never get the chance to be built. The retention test
    # below (content OR context refs) mirrors
    # ``normalize_transcript_messages`` (core/agent/transcript.py).
    _attach_historical_image_context_refs(rows)
    return [row for row in rows if row.content or row.context_refs]


def _attach_historical_image_context_refs(rows: list[_TranscriptRow]) -> None:
    """Mirror ``chat_history_service.load_task_transcript``'s image budget.

    Same reverse scan, same global cap: walk the transcript newest-first
    and keep attaching image context refs to each row until
    ``_MAX_HISTORICAL_IMAGE_CONTEXT_REFS`` (shared with
    ``load_task_transcript`` so the two paths can never silently diverge)
    is exhausted, so only the most recent uploaded images survive. This is
    the only bound this module currently applies; the tool-exchange caps
    live in a follow-up change (see the module docstring).
    """
    remaining = _MAX_HISTORICAL_IMAGE_CONTEXT_REFS
    for index in range(len(rows) - 1, -1, -1):
        if remaining <= 0:
            break
        references = build_image_context_references(rows[index].attachments)
        kept = references[:remaining]
        rows[index].context_refs = kept
        remaining -= len(kept)
        rows[index].attachments = None


# ---------------------------------------------------------------------------
# R2/R3 -- tool events and exchange pairing
# ---------------------------------------------------------------------------


def _load_tool_exchanges(
    db: Session, task_id: int, *, stats: _ReconstructionStats | None = None
) -> list[_ToolExchange]:
    stats = stats if stats is not None else _ReconstructionStats()
    trace_rows = (
        db.query(TraceEvent)
        .filter(
            TraceEvent.task_id == task_id,
            TraceEvent.build_id.is_(None),
            TraceEvent.event_type.in_(_TOOL_EVENT_TYPES),
        )
        .order_by(asc(TraceEvent.timestamp), asc(TraceEvent.id))
        .all()
    )

    # ``tool_call_id`` alone is not a safe pending-map key: the DAG pattern
    # runs steps CONCURRENTLY (``dag.py``'s ``asyncio.create_task`` batch),
    # and when a provider omits real tool-call ids, ``_normalize_tool_calls``
    # (react.py) falls back to ``f"tool_call_{index}"`` where ``index`` is
    # only unique WITHIN one LLM response -- two concurrent DAG steps can
    # each emit "tool_call_0". Without a per-step discriminator, step A's
    # pending start can be overwritten by step B's, and A's end then pops
    # B's start: the WRONG tool_name/assistant_content gets attached to a
    # result, corrupting history rather than merely losing it.
    #
    # ``TraceEvent.step_id`` is that discriminator: ``PatternRuntime`` always
    # threads a step id through ``_step_id_from_payload`` (runtime.py) into
    # the dedicated ``step_id`` COLUMN (not ``data``) via
    # ``stage_trace_event_row`` (trace_event_staging.py). The DAG's
    # ``_with_step`` (dag.py) stamps each concurrent step's tool calls with
    # its own unique ``step_id``/``dag_step_id``, so concurrent branches get
    # different keys here. ReAct sets the same step id on both the start and
    # end of a given call (one LLM turn = one step), so this is a no-op for
    # ReAct's already-unique-per-turn ids -- and rows written before this
    # column was populated simply carry ``step_id=None``, which normalizes
    # to the same empty-string discriminator for both events, reproducing
    # the old bare-``tool_call_id`` keying exactly.
    #
    # ``turn_id`` is also folded into this key (when the event carries one;
    # legacy rows without it fall back to the plain
    # ``(step_discriminator, call_id)`` pair, unchanged from before). This
    # is defensive hardening, not a fix for a reachable bug: exploiting a
    # collision on the *current* two-part key would need two DIFFERENT
    # turns' tool calls to interleave in the trace stream with the same
    # ``step_id``/``tool_call_id`` pair, and today that can't happen --
    # a task's turns are serialized by a per-task command gate, a running
    # task refuses a new in-flight command before any DB write, the DB
    # claim is atomic (``RUNNING`` is excluded from ``_APPENDABLE_STATUSES``,
    # so a second writer can't append mid-turn), the pause -> new-message
    # path explicitly drains the in-flight turn before starting the next,
    # a hard cancel emits no end event at all (nothing to interleave), and
    # trace writes are per-event synchronous commits with no batching that
    # could reorder them across turns. ReAct's own ``step_id`` values
    # (``react_{uuid8}``, minted once per pattern start) never repeat
    # either, so only a DAG replan -- which restarts step numbering at
    # ``step_1`` -- can even collide on the current key, and that
    # collision is exactly what ``turn_id`` (unique per user message,
    # shared by every step within one turn) discriminates away. Adding it
    # here is insurance against a future change to any of those
    # invariants, not evidence one is currently broken.
    pending: dict[tuple[str, ...], _PendingToolStart] = {}
    exchanges: list[_ToolExchange] = []

    for trace_row in trace_rows:
        data: dict[str, Any] = (
            trace_row.data if isinstance(trace_row.data, dict) else {}
        )
        row_timestamp = _as_aware_utc(cast(datetime, trace_row.timestamp))
        row_id = int(trace_row.id)
        step_discriminator = str(trace_row.step_id or "")

        if trace_row.event_type == "tool_execution_start":
            call_id = str(data.get("tool_call_id") or "") or f"recon-{row_id}"
            assistant_content = str(data.get("assistant_content") or "").strip()
            start_turn_id = str(data.get("turn_id") or "")
            start_key: tuple[str, ...] = (
                (step_discriminator, call_id, start_turn_id)
                if start_turn_id
                else (step_discriminator, call_id)
            )
            pending[start_key] = _PendingToolStart(
                assistant_content=assistant_content,
                tool_name=str(data.get("tool_name") or ""),
                tool_params=data.get("tool_params"),
                sort_key=(row_timestamp, row_id),
                turn_id=start_turn_id,
            )
            continue

        # tool_execution_end / tool_execution_failed: pop the matching start so
        # each start is consumed by exactly one end.
        raw_call_id = str(data.get("tool_call_id") or "")
        end_turn_id = str(data.get("turn_id") or "")
        start = None
        call_id = raw_call_id
        if raw_call_id:
            end_key: tuple[str, ...] = (
                (step_discriminator, raw_call_id, end_turn_id)
                if end_turn_id
                else (step_discriminator, raw_call_id)
            )
            start = pending.pop(end_key, None)
        if start is None:
            # No id, or id present but no matching start recorded (e.g. the
            # matching start fell outside this query, or lost its id, or its
            # step id doesn't match this end's). Fall back to a synthesized
            # key so pairing always has an identity.
            call_id = raw_call_id or f"recon-{row_id}"
            stats.tool_ends_without_start += 1

        tool_name = str(data.get("tool_name") or "").strip()
        if not tool_name and start is not None:
            tool_name = start.tool_name
        if not tool_name:
            # Cannot identify the tool this exchange belongs to; skip rather
            # than emit an anonymous exchange the model can't reason about.
            continue

        if tool_name in CONTROL_TOOL_NAMES:
            # final_answer / send_message / ask_user_question are pseudo-tools
            # whose observable effect already exists as a TaskChatMessage row.
            # Re-injecting them as tool exchanges would duplicate that content,
            # and replaying a completed final_answer risks the model believing
            # the *current* turn is already answered.
            continue

        tool_params = data.get("tool_params")
        if tool_params is None and start is not None:
            tool_params = start.tool_params
        if tool_params is None:
            tool_params = {}

        result = _resolve_tool_result(cast(str, trace_row.event_type), data)

        # Dedup of prose repeated across a parallel tool-call batch happens
        # below, in final sort-key order -- not here. Trace rows are iterated
        # in the order the *end* events arrive, which for a parallel batch
        # can differ from the *start* order (whichever call finishes first
        # gets processed first), so deduping inline here would attach the
        # prose to the wrong exchange and could blank a duplicate that lands
        # far from its match in final order. Keep the raw value for now.
        assistant_content = start.assistant_content if start is not None else ""

        sort_key = start.sort_key if start is not None else (row_timestamp, row_id)

        # The end/failure event's own "turn_id" (present since
        # PatternRuntime.on_tool_end/on_tool_error also stamp it) wins;
        # fall back to the matching start's value for a pair split across
        # two events where only one carries it (e.g. a start recorded before
        # this field shipped, matched to an end recorded after).
        turn_id = str(data.get("turn_id") or "")
        if not turn_id and start is not None:
            turn_id = start.turn_id

        exchanges.append(
            _ToolExchange(
                call_id=call_id,
                tool_name=tool_name,
                tool_params=tool_params,
                result=result,
                assistant_content=assistant_content,
                sort_key=sort_key,
                turn_id=turn_id,
            )
        )

    # Sort by the *start* time (sort_key), matching the order exchanges will
    # ultimately appear in once merged with the transcript (R4).
    exchanges.sort(key=lambda exchange: exchange.sort_key)
    # Any start left in ``pending`` here never found its end/failure event
    # and so never became an exchange at all; it silently disappears along
    # with its prose (behavior unchanged), but is now counted.
    stats.dangling_tool_starts += len(pending)
    return exchanges


def _resolve_tool_result(event_type: str, data: dict[str, Any]) -> Any:
    if event_type == "tool_execution_end":
        if bool(data.get("interrupted")) and "result" not in data:
            return {
                "success": False,
                "interrupted": True,
                "error": data.get("interrupt_reason"),
            }
        result = data.get("result")
        if result is None:
            # A "tool_execution_end" with neither a "result" key nor the
            # interrupted marker above -- degraded data, but still a
            # completed call. Mirror the shape of the other degraded cases
            # here rather than letting a bare ``None`` flow into
            # ``add_tool_result``.
            return {
                "success": False,
                "status": "unknown",
                "error": "tool result missing from persisted trace event",
            }
        return result

    # tool_execution_failed
    if "result" in data:
        return data.get("result")
    return {
        "success": False,
        "status": "error",
        "error": data.get("error") or data.get("error_message"),
    }


# ---------------------------------------------------------------------------
# R4 -- turn_id join
# ---------------------------------------------------------------------------


def _group_exchanges_by_turn(
    exchanges: list[_ToolExchange],
    *,
    stats: _ReconstructionStats | None = None,
) -> dict[str, list[_ToolExchange]]:
    """Group tool exchanges by the durable turn_id that produced them.

    Unlike the old ``TraceEvent.step_id``-keyed grouping this replaces,
    ``turn_id`` is guaranteed fresh per turn: it is minted once per user
    message (``AgentRunner._ensure_user_message_turn_id``) and threaded
    verbatim into both ``TaskChatMessage.turn_id`` and the tool trace
    events' ``data["turn_id"]`` (``PatternRuntime.on_tool_start`` /
    ``on_tool_end`` / ``on_tool_error``, via ``_with_runtime_turn_id`` in
    react.py and ``_DAGStepRuntime.active_turn_id`` in dag.py). A DAG turn
    that fans out over several steps still shares that one turn_id across
    every step, so it collapses back into a single group here -- unlike
    ``step_id``, which is deliberately per-step and would fragment it.

    Exchanges with no turn_id (``""``, from rows written before this
    column/field existed) are dropped entirely rather than grouped under a
    shared key or placed as singletons: see ``_merge_chronologically`` for
    why omission, not inference, is the deliberate choice here.

    Assistant prose is carried through verbatim, one exchange at a time.
    An earlier revision of this module also deduplicated byte-identical
    prose on adjacent exchanges, on the theory that a parallel tool-call
    batch stamps every concurrent call with the same ``assistant_content``.
    That premise is false: the only writer of that field,
    ``ReActPattern._remember_tool_call_content``, returns after the first
    non-control call in the batch, so exactly one call in a batch ever
    carries prose. A parallel batch therefore cannot produce the adjacent
    duplicate that dedup looked for; the only thing that can is two
    *serialized* iterations of one turn whose prose happens to match (the
    ReAct step_id is minted once per pattern run, so serialized iterations
    share it and the turn_id). Blanking the second of those dropped real
    content the live history had, which is why dedup is gone rather than
    made more precise.
    """
    stats = stats if stats is not None else _ReconstructionStats()
    groups: dict[str, list[_ToolExchange]] = {}
    for exchange in exchanges:
        if not exchange.turn_id:
            stats.exchanges_without_turn_id += 1
            continue
        groups.setdefault(exchange.turn_id, []).append(exchange)
    return groups


def _merge_chronologically(
    transcript_rows: list[_TranscriptRow],
    exchanges: list[_ToolExchange],
    *,
    stats: _ReconstructionStats | None = None,
) -> list[_MergeEntry]:
    """Interleave transcript rows with tool exchanges by turn_id, not timestamp.

    Placement is now a join, not an inference: each user transcript row
    carries the same turn_id its turn's tool exchanges were stamped with
    (see ``_group_exchanges_by_turn``), so a turn's exchanges belong
    unambiguously right after that row and before whatever transcript row
    comes next. This is immune to clock skew between ``TaskChatMessage
    .created_at`` (DB clock) and ``TraceEvent.timestamp`` (app clock) --
    the old timestamp-adjacency approach this replaces could misplace a
    turn either by that skew or, worse, structurally: two DAG re-plans can
    both emit ``step_id="step_1"``, which silently merged two distinct
    turns into one group, and a failed/interrupted turn with no assistant
    row got flushed *after* the next user message by the old relocation
    pass regardless of timestamps. Joining on turn_id makes both cases
    correct by construction instead of narrowing the heuristic further.

    Exchanges whose turn_id has no matching row in ``transcript_rows``
    (already-empty turn_ids are dropped earlier, by
    ``_group_exchanges_by_turn``) are silently omitted rather than placed
    by adjacency or clock guesswork. This happens when the turn's user row
    fell outside ``before_message_id``, or -- F1 -- when it was an
    image-only row (``content == ""``) whose context refs all fell outside
    ``_attach_historical_image_context_refs``'s task-wide budget and so
    were dropped by the retention filter in ``_load_transcript_rows``.
    That row drop itself is correct: it deliberately matches upstream
    (``chat_history_service.load_task_transcript`` /
    ``normalize_transcript_messages`` apply the identical ``content or
    context_refs`` predicate). What would NOT be correct is fabricating a
    placement for the orphaned exchanges once their anchor row is gone --
    there is no other row that legitimately "produced" them. So the
    decision here is a deliberate drop, the same as the
    ``before_message_id`` case: losing that detail is acceptable (the
    conversation reconstructs with transcript text only, exactly today's
    non-reconstructed behavior); fabricating a placement is not, because a
    misplaced exchange can make the model believe a later turn's tool
    results informed an earlier answer, or that a failed turn never
    happened. The one change this module makes for that case is
    observability: every exchange dropped this way is counted in
    ``stats.exchanges_with_unmatched_turn`` (see ``load_task_conversation_
    context_sync``'s summary log line) so the drop is no longer silent,
    even though it is still real.

    Within a turn's group, members keep the relative order already
    established in ``_load_tool_exchanges`` (timestamp/id sort) -- safe
    because it is one turn's own clock, not a cross-turn comparison.
    """
    stats = stats if stats is not None else _ReconstructionStats()
    exchanges_by_turn = _group_exchanges_by_turn(exchanges, stats=stats)

    entries: list[_MergeEntry] = []
    for row in transcript_rows:
        entries.append(_MergeEntry(kind="transcript", transcript=row))
        if row.role != "user" or not row.turn_id:
            continue
        # TaskChatMessage's (task_id, role, turn_id) uniqueness constraint
        # guarantees at most one "user" row per turn_id, so this can never
        # place the same group twice.
        group = exchanges_by_turn.pop(row.turn_id, None)
        if group:
            stats.exchanges_placed += len(group)
            entries.append(_MergeEntry(kind="group", group=group))

    # Whatever is left here never found a surviving user row to anchor on
    # -- see the F1 discussion above. Deliberately dropped, now counted.
    stats.exchanges_with_unmatched_turn += sum(
        len(group) for group in exchanges_by_turn.values()
    )

    return entries


# ---------------------------------------------------------------------------
# Rendering + final pairing sweep
# ---------------------------------------------------------------------------


def _render_messages(entries: list[_MergeEntry]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in entries:
        if entry.kind == "transcript":
            assert entry.transcript is not None
            item: dict[str, Any] = {
                "role": entry.transcript.role,
                "content": entry.transcript.content,
            }
            if entry.transcript.context_refs:
                item[CONTEXT_REFS_KEY] = [
                    reference.durable_dict()
                    for reference in entry.transcript.context_refs
                ]
            messages.append(item)
            continue

        assert entry.group is not None
        for exchange in entry.group:
            messages.append(
                {
                    "role": "assistant",
                    "content": exchange.assistant_content or "",
                    "tool_calls": [
                        {
                            "id": exchange.call_id,
                            "type": "function",
                            "function": {
                                "name": exchange.tool_name,
                                "arguments": json.dumps(
                                    exchange.tool_params or {},
                                    ensure_ascii=False,
                                    default=str,
                                ),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": exchange.call_id,
                    "content": "",
                    "tool_name": exchange.tool_name,
                    "raw_result": exchange.result,
                }
            )
    return messages


def _final_pairing_sweep(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop any ``tool`` message not immediately preceded by its declaring assistant.

    Construction should never produce this, but this mirrors the invariant
    ``ExecutionContext._sanitize_tool_message_pairs`` protects at the live
    context layer.
    """
    sanitized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            sanitized.append(message)
            continue
        previous = messages[index - 1] if index > 0 else None
        if (
            previous is not None
            and previous.get("role") == "assistant"
            and _ids_match(message.get("tool_call_id"), previous.get("tool_calls"))
        ):
            sanitized.append(message)
        # else: orphaned tool message, drop it.
    return sanitized


def _ids_match(tool_call_id: Any, tool_calls: Any) -> bool:
    """Whether ``tool_calls`` declares ``tool_call_id``.

    Compares only present ids. ``str()`` on two ``None``s would render
    ``"None" == "None"`` and report a match, which would let this sweep --
    whose whole job is catching pairs construction should never have
    produced -- pass an orphan through in exactly the case it exists for.
    """
    if tool_call_id is None:
        return False
    target = str(tool_call_id)
    for call in tool_calls or []:
        call_id = call.get("id") if isinstance(call, dict) else None
        if call_id is not None and str(call_id) == target:
            return True
    return False


def _as_aware_utc(value: datetime) -> datetime:
    """Normalize a possibly-naive datetime to aware UTC for safe comparison.

    Applied to every ``TraceEvent.timestamp`` value in ``_load_tool_exchanges``
    before it feeds a ``sort_key`` (used both to order an exchange's start/end
    pairing and to order same-turn exchanges relative to each other).
    ``TraceEvent.timestamp`` is declared ``DateTime(timezone=True)``, but
    SQLite (used by tests) returns naive datetimes regardless of what was
    stored, while Postgres returns aware ones; comparing a naive and an
    aware datetime raises ``TypeError``.

    Placement of a turn's exchanges relative to the transcript is a
    ``turn_id`` join now (see ``_merge_chronologically``), not a timestamp
    comparison, so this function is no longer applied to ``TaskChatMessage
    .created_at`` -- that column isn't even loaded by this module any more.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
