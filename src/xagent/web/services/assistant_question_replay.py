"""Pair replayed assistant question rows with the trace events behind them.

One waiting question is persisted twice by a single producer
(``_persist_agent_outbound_event``): as a ``TraceEvent`` holding the raw
prompt, and as a transcript row whose content additionally carries the
rendered interaction list appended by
:func:`build_assistant_transcript_content`. The two therefore never compare
equal, and historical replay used to ship both -- printing the clarification
form twice, then a third time after another reconnect (#2292).

Replay keeps the transcript row and drops its trace twin. The row is the
sanitized producer: its synthesized payload sets ``expect_response: False``
so a historical question cannot flip the client back into
``waiting_for_user``, whereas the trace payload is passed through untouched.
The surviving event inherits the trace event's id, which gives the client a
stable identity to reconcile against across reconnects.

The pairing is deliberately narrow. Only question rows and only
``agent_message`` traces take part; assistant *answers* are still deduped the
other way round (row dropped, trace kept) by the caller's content check,
because their trace events carry streaming identity this must not disturb.

Not every question row can carry ``source_event_id``. The channel finalizers
write theirs through ``persist_assistant_message_no_commit``, which has no
such parameter, reached from ``execution_result_projection.py`` via
``managed_task_lease.py``; only the Slack, Telegram and Feishu bots consume
that projection, so no ordinary web-chat turn takes this path. Those rows
stay on the derivation path permanently, and because the same waiting
question also goes through the shared outbound handler, a channel task ends
up with two question rows for one trace -- the projection row can never pair,
since the trace is already claimed. That is a pre-existing channel-only
limit tracked separately, not something this pairing introduces; the
symmetry gate below refuses an ambiguous pairing rather than guessing one.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

from ...core.agent.transcript import build_assistant_transcript_content
from ..models.chat_message import TaskChatMessage
from ..models.uploaded_file import UploadedFile
from .chat_history_service import QUESTION_MESSAGE_TYPE, SUPERSEDED_MESSAGE_TYPE
from .file_reference_output_service import (
    load_assistant_file_reference_records,
    reconcile_assistant_file_references,
)
from .public_trace_events import is_audit_only_trace_data

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ``SUPERSEDED_MESSAGE_TYPE`` is the same row relabelled once a structured
# publication has taken over its answer slot; it replays identically.
QUESTION_MESSAGE_TYPES = frozenset({QUESTION_MESSAGE_TYPE, SUPERSEDED_MESSAGE_TYPE})

# Questions are emitted as ``agent_message``. ``ai_message`` is the final
# answer channel and is out of scope by construction.
QUESTION_TRACE_EVENT_TYPE = "agent_message"


@dataclass(frozen=True)
class ReplayQuestionRow:
    """One persisted assistant transcript row, as stored."""

    row_id: int
    content: str
    message_type: str
    source_event_id: Optional[str] = None


@dataclass(frozen=True)
class ReplayTraceQuestion:
    """One assistant trace event competing to represent the same question."""

    event_id: str
    event_type: str
    message: str
    interactions: Optional[Sequence[Any]] = None


@dataclass(frozen=True)
class AssistantQuestionReplayPlan:
    """Which trace events a snapshot's question rows have taken over."""

    superseded_trace_event_ids: frozenset[str]
    trace_event_id_by_row_id: Mapping[int, str]


def _stripped_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def plan_assistant_question_replay(
    *,
    rows: Sequence[ReplayQuestionRow],
    traces: Sequence[ReplayTraceQuestion],
    normalize_message: Optional[Callable[[str], str]] = None,
) -> AssistantQuestionReplayPlan:
    """Decide, for one snapshot, which trace event each question row replaces.

    ``normalize_message`` receives a trace event's raw message and must apply
    whatever rewriting persistence applied before the transcript row was
    built -- file-reference reconciliation, today. Without it a question whose
    text contains a repaired ``file:`` link would not pair.
    """

    candidates = [
        trace
        for trace in traces
        if trace.event_type == QUESTION_TRACE_EVENT_TYPE
        and _stripped_text(trace.message)
    ]
    question_rows = [
        row
        for row in rows
        if row.message_type in QUESTION_MESSAGE_TYPES and _stripped_text(row.content)
    ]
    if not candidates or not question_rows:
        return AssistantQuestionReplayPlan(frozenset(), {})

    by_event_id = {trace.event_id: trace for trace in candidates}
    claimed_by_row: dict[int, str] = {}
    claimed_event_ids: set[str] = set()

    # An explicit ``source_event_id`` is the row's own statement of where it
    # came from, so it is honoured before any derivation can claim the same
    # event. A row naming an event this snapshot does not carry claims
    # nothing: falling back to text here would hand it a different round's
    # identity, and consecutive rounds routinely repeat a question verbatim.
    deriving_rows: list[ReplayQuestionRow] = []
    for row in question_rows:
        source_event_id = _stripped_text(row.source_event_id)
        if not source_event_id:
            deriving_rows.append(row)
            continue
        if source_event_id in by_event_id and source_event_id not in claimed_event_ids:
            claimed_by_row[row.row_id] = source_event_id
            claimed_event_ids.add(source_event_id)

    if deriving_rows:
        derived = _derived_transcript_index(
            candidates, claimed_event_ids, normalize_message
        )
        rows_by_content: dict[str, list[ReplayQuestionRow]] = defaultdict(list)
        for row in deriving_rows:
            rows_by_content[_stripped_text(row.content)].append(row)

        for content, rows_for_content in rows_by_content.items():
            queue = derived.get(content)
            # Symmetry gate. Pair only when this text has exactly one
            # unclaimed trace per row asking it, because without an explicit
            # source_event_id nothing else says which trace belongs to which
            # round. A snapshot missing one round's trace -- filtered out of
            # the public scope, audit-only, pruned, or with a derivation that
            # drifted because file-reference reconciliation now resolves
            # differently -- would otherwise let the earlier row claim a later
            # round's identity, and the client would merge two distinct
            # questions into one bubble. That is #2250's failure reintroduced.
            # Refusing costs only the legacy dedupe for this text: both copies
            # replay, exactly as they did before this fix.
            if queue is None or len(queue) != len(rows_for_content):
                continue
            for row, event_id in zip(rows_for_content, queue):
                claimed_by_row[row.row_id] = event_id
                claimed_event_ids.add(event_id)

    return AssistantQuestionReplayPlan(
        superseded_trace_event_ids=frozenset(claimed_event_ids),
        trace_event_id_by_row_id=claimed_by_row,
    )


@dataclass(frozen=True)
class TranscriptReplay:
    """One snapshot's transcript rows plus the pairing they imply."""

    rows: Sequence["TaskChatMessage"]
    file_reference_records: Sequence["UploadedFile"]
    plan: AssistantQuestionReplayPlan

    def superseded(self, trace_event_id: Any) -> bool:
        return str(trace_event_id) in self.plan.superseded_trace_event_ids

    def event_id_for(self, row_id: Any, fallback: str) -> str:
        return self.plan.trace_event_id_by_row_id.get(int(row_id), fallback)

    def paired(self, row_id: Any) -> bool:
        return int(row_id) in self.plan.trace_event_id_by_row_id


def load_transcript_replay(
    db: "Session",
    *,
    task_id: int,
    task_user_id: int,
    trace_events: Sequence[Any],
    trace_data_by_event_id: Mapping[str, Any],
) -> TranscriptReplay:
    """Read a task's transcript rows and pair its questions with their traces.

    Persistence reconciles file references into a question's text *before*
    building the transcript row, so the trace payload is put through the same
    reconciliation here; otherwise a question carrying a repaired ``file:``
    link would not pair on a row predating ``source_event_id``.
    """

    rows = (
        db.query(TaskChatMessage)
        .filter(TaskChatMessage.task_id == task_id)
        .order_by(TaskChatMessage.created_at, TaskChatMessage.id)
        .all()
    )
    file_reference_records = load_assistant_file_reference_records(
        db,
        task_id=task_id,
        user_id=task_user_id,
    )
    plan = plan_snapshot_question_replay(
        chat_messages=rows,
        trace_events=trace_events,
        trace_data_by_event_id=trace_data_by_event_id,
        normalize_message=lambda text: reconcile_assistant_file_references(
            db,
            task_id=task_id,
            user_id=task_user_id,
            content=text,
            records=file_reference_records,
        ),
    )
    return TranscriptReplay(
        rows=rows,
        file_reference_records=file_reference_records,
        plan=plan,
    )


def plan_snapshot_question_replay(
    *,
    chat_messages: Sequence[Any],
    trace_events: Sequence[Any],
    trace_data_by_event_id: Mapping[str, Any],
    normalize_message: Optional[Callable[[str], str]] = None,
) -> AssistantQuestionReplayPlan:
    """Adapt one historical snapshot's ORM rows onto :func:`plan_assistant_question_replay`.

    ``trace_data_by_event_id`` holds the snapshot's already-normalized trace
    payloads, so this never re-reads ``event.data``.
    """

    rows = [
        ReplayQuestionRow(
            row_id=int(row.id),
            content=str(row.content or ""),
            message_type=str(row.message_type or ""),
            source_event_id=getattr(row, "source_event_id", None),
        )
        for row in chat_messages
        if str(row.role) == "assistant"
    ]

    traces: list[ReplayTraceQuestion] = []
    for event in trace_events:
        data = trace_data_by_event_id.get(str(event.event_id))
        # Audit-only payloads never reach the client, so letting one claim a
        # row would strand that row's real twin.
        if not isinstance(data, dict) or is_audit_only_trace_data(data):
            continue
        metadata = data.get("metadata")
        interactions = (
            metadata.get("interactions")
            if isinstance(metadata, dict)
            and isinstance(metadata.get("interactions"), list)
            else None
        )
        traces.append(
            ReplayTraceQuestion(
                event_id=str(event.event_id),
                event_type=str(event.event_type),
                message=str(data.get("message") or data.get("content") or ""),
                interactions=interactions,
            )
        )

    return plan_assistant_question_replay(
        rows=rows,
        traces=traces,
        normalize_message=normalize_message,
    )


def _derived_transcript_index(
    candidates: Sequence[ReplayTraceQuestion],
    claimed_event_ids: set[str],
    normalize_message: Optional[Callable[[str], str]],
) -> Mapping[str, list[str]]:
    """Index unclaimed trace events by the transcript text they would produce.

    This reproduces the exact transformation that wrote the row rather than
    comparing rendered text loosely -- the pairing is only as wide as
    ``build_assistant_transcript_content`` is deterministic.
    """

    index: dict[str, list[str]] = defaultdict(list)
    for trace in candidates:
        if trace.event_id in claimed_event_ids:
            continue
        message = trace.message
        if normalize_message is not None:
            try:
                message = normalize_message(message)
            except Exception:
                # Pairing degrades to "no match", which replays this question
                # the way it did before the fix rather than mispairing it.
                logger.warning(
                    "Could not normalize trace %s for question pairing; "
                    "falling back to its raw text, which pairs only if the "
                    "row was written before any reconciliation applied",
                    trace.event_id,
                    exc_info=True,
                )
                message = trace.message
        derived = build_assistant_transcript_content(
            message,
            list(trace.interactions) if trace.interactions else None,
        )
        index[derived.strip()].append(trace.event_id)
    return index
