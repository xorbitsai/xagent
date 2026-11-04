"""Root V2 model history, read at one committed event horizon.

This is a model projection, not a recovery reducer or a display API. Business
permissions and file materialization remain at their existing boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import ARRAY, Text, case, func, literal, select, type_coerce
from sqlalchemy.orm import Session

from ...core.agent.attachments import build_image_context_references
from ...core.agent.context.execution import (
    COMPACT_CONTEXT_REFS_METADATA_KEY,
    MODEL_CONTEXT_WATERMARK_METADATA_KEY,
)
from ...core.agent.result import CONTROL_TOOL_NAMES
from ...core.agent.transcript import build_assistant_transcript_content
from ...core.context_ref import (
    CONTEXT_REFS_KEY,
    normalize_context_references,
    split_tool_result_context_references,
)
from ..models.chat_message import TaskChatMessage
from ..models.task import Task
from ..models.task_execution_event import TaskExecutionEvent
from .assistant_history_safety import (
    TASK_FAILURE_MESSAGE_TYPE,
    assistant_history_has_safe_ancillary_payload,
    client_safe_assistant_history_content,
)
from .chat_history_service import _MAX_HISTORICAL_IMAGE_CONTEXT_REFS
from .execution_event_signals import (
    ExecutionEventReadMeasurement,
    measure_execution_event_read,
)
from .task_conversation_context_service import _resolve_tool_result
from .task_execution_context_service import summarize_execution_failure_event
from .task_execution_event_store import MAX_EXECUTION_EVENT_PAGE_SIZE


@dataclass(frozen=True)
class TaskEventContext:
    messages: list[dict[str, Any]]
    watermark: dict[str, Any] | None
    selected_skill_name: str | None


def _data(event: TaskExecutionEvent) -> dict[str, Any]:
    if event.payload_version != 1 or not isinstance(event.payload, dict):
        raise ValueError(
            f"Unsupported model context event {event.event_id} (task_id={event.task_id})"
        )
    data = event.payload.get("data", event.payload)
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid model context event {event.event_id} (task_id={event.task_id})"
        )
    return data


def _coordinate(value: Any, event: TaskExecutionEvent) -> tuple[int, str]:
    if not isinstance(value, dict) or value.get("scope_id") != "root":
        raise ValueError(
            f"Invalid model context coverage scope (task_id={event.task_id}, event_id={event.event_id})"
        )
    sequence = value.get("sequence")
    if type(sequence) is not int or sequence <= 0:
        raise ValueError(
            f"Invalid model context coverage sequence (task_id={event.task_id}, event_id={event.event_id})"
        )
    event_id = value.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError(
            f"Invalid model context coverage identity (task_id={event.task_id}, event_id={event.event_id})"
        )
    return sequence, event_id


def _load_context_anchors(
    db: Session, task_id: int, horizon: int, events: list[TaskExecutionEvent]
) -> dict[str, tuple[int, str, int]]:
    """Fetch referenced identities in bounded batches, without snapshot bodies."""
    references: set[str] = set()
    for event in events:
        if event.kind == "input_applied":
            event_id = _data(event).get("recovery_event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(
                    f"Applied input has no prior root recovery state (task_id={event.task_id}, event_id={event.event_id})"
                )
            references.add(event_id)
        elif event.kind == "action_end_compact":
            data = _data(event)
            content = data.get("summary")
            if not isinstance(content, str) or not content.strip():
                continue
            coordinate = data.get(MODEL_CONTEXT_WATERMARK_METADATA_KEY)
            if coordinate is None:
                coordinate = event.payload.get("transcript_watermark")
            if coordinate is not None:
                references.add(_coordinate(coordinate, event)[1])

    identities = sorted(references)
    anchors: dict[str, tuple[int, str, int]] = {}
    for offset in range(0, len(identities), MAX_EXECUTION_EVENT_PAGE_SIZE):
        # event_id is globally unique: look it up on that index alone and
        # apply the task, scope and horizon predicates here, so the planner
        # cannot scan the task's rows through the (task, scope, sequence) index.
        rows = db.execute(
            select(
                TaskExecutionEvent.event_id,
                TaskExecutionEvent.task_id,
                TaskExecutionEvent.scope_id,
                TaskExecutionEvent.sequence,
                TaskExecutionEvent.kind,
                TaskExecutionEvent.payload_version,
            ).where(
                TaskExecutionEvent.event_id.in_(
                    identities[offset : offset + MAX_EXECUTION_EVENT_PAGE_SIZE]
                ),
            )
        )
        for event_id, owner, scope_id, sequence, kind, version in rows:
            if owner == task_id and scope_id == "root" and sequence <= horizon:
                anchors[event_id] = (sequence, kind, version)
    return anchors


def _selected_skill_name(event: TaskExecutionEvent) -> str | None:
    data = _data(event)
    return str(data.get("skill_name") or "").strip() if data.get("selected") else None


def load_task_event_skill_name(db: Session, task_id: int) -> str | None:
    """Read only the latest root skill selection at the committed horizon."""
    horizon = (
        select(Task.conversation_event_sequence)
        .where(Task.id == task_id)
        .scalar_subquery()
    )
    event = db.scalar(
        select(TaskExecutionEvent)
        .where(
            TaskExecutionEvent.task_id == task_id,
            TaskExecutionEvent.scope_id == "root",
            TaskExecutionEvent.kind == "skill_select_end",
            TaskExecutionEvent.sequence <= horizon,
        )
        .order_by(TaskExecutionEvent.sequence.desc())
        .limit(1)
    )
    return _selected_skill_name(event) if event is not None else None


_MODEL_KINDS = frozenset(
    {
        "input_accepted",
        "input_applied",
        "assistant_message",
        "agent_message",
        "tool_execution_start",
        "tool_execution_end",
        "tool_execution_failed",
        "execution_settled",
        "skill_select_end",
        "action_end_compact",
    }
)
_TOOL_KINDS = frozenset(
    {"tool_execution_start", "tool_execution_end", "tool_execution_failed"}
)
_START_KINDS = frozenset({"tool_execution_start"})
_ACCEPTED_KINDS = frozenset({"input_accepted"})
_OUTCOME_KINDS = _TOOL_KINDS - _START_KINDS


def _page_root_events(
    db: Session, query: Any, after: int, *, scalars: bool
) -> list[Any]:
    rows: list[Any] = []
    cursor = after
    while True:
        paged = (
            query.where(TaskExecutionEvent.sequence > cursor)
            .order_by(TaskExecutionEvent.sequence)
            .limit(MAX_EXECUTION_EVENT_PAGE_SIZE)
        )
        page = list(db.scalars(paged) if scalars else db.execute(paged))
        if not page:
            return rows
        rows.extend(page)
        cursor = int(page[-1].sequence)


def _settlement_payload(db: Session) -> Any:
    """The settlement payload without ``result.agent_result``, computed in SQL.

    The projection never reads the embedded agent context, which dominates the
    size of a production settlement. Unknown dialects fetch the full payload.
    """
    column = TaskExecutionEvent.payload
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        # ``#-`` raises when its path crosses an array, so only object-valued
        # results are stripped; any other payload is returned as stored.
        stripped: Any = case(
            (
                func.jsonb_typeof(column.op("#>")(literal(["result"], ARRAY(Text))))
                == "object",
                column.op("#-")(literal(["result", "agent_result"], ARRAY(Text))),
            ),
            else_=column,
        )
    elif dialect == "sqlite":
        stripped = func.json_remove(column, "$.result.agent_result")
    else:
        return column
    return type_coerce(stripped, column.type)


def _root_events(
    db: Session, task_id: int, *, after: int, through: int, kinds: frozenset[str]
) -> list[TaskExecutionEvent]:
    # Do not load checkpoints or raw LLM prompts to build a model transcript.
    def scoped(*entities: Any) -> Any:
        return select(*entities).where(
            TaskExecutionEvent.task_id == task_id,
            TaskExecutionEvent.scope_id == "root",
            TaskExecutionEvent.sequence <= through,
        )

    others = kinds - {"execution_settled"}
    events: list[TaskExecutionEvent] = []
    if others:
        events = _page_root_events(
            db,
            scoped(TaskExecutionEvent).where(TaskExecutionEvent.kind.in_(others)),
            after,
            scalars=True,
        )
    if len(others) == len(kinds):
        return events
    # Settlements are detached copies built from the stripped payload; the
    # session never holds a row whose payload differs from the database.
    columns = [c for c in TaskExecutionEvent.__table__.c if c.name != "payload"]
    rows = _page_root_events(
        db,
        scoped(*columns, _settlement_payload(db).label("payload")).where(
            TaskExecutionEvent.kind == "execution_settled"
        ),
        after,
        scalars=False,
    )
    events.extend(TaskExecutionEvent(**row._mapping) for row in rows)
    events.sort(key=lambda event: int(event.sequence))
    return events


def _summary_coverage(
    event: TaskExecutionEvent, anchors: dict[str, tuple[int, str, int]]
) -> tuple[dict[str, Any], int, int] | None:
    """Return a usable summary with its transcript and model floors."""
    data = _data(event)
    content = data.get("summary")
    if not isinstance(content, str) or not content.strip():
        return None
    native = data.get(MODEL_CONTEXT_WATERMARK_METADATA_KEY)
    legacy = event.payload.get("transcript_watermark")
    if native is None and legacy is None:
        return None  # Older summaries make no usable coverage claim.
    floor, anchor_id = _coordinate(native if native is not None else legacy, event)
    anchor = anchors.get(anchor_id)
    if anchor is None or anchor[0] != floor:
        raise ValueError(
            f"Invalid model context coverage identity (task_id={event.task_id}, event_id={event.event_id})"
        )
    if floor >= event.sequence:
        raise ValueError(
            f"Summary covers a future event (task_id={event.task_id}, event_id={event.event_id})"
        )
    # Native coverage includes model history; A's coordinate only
    # covers transcript facts, so earlier tool facts remain eligible.
    summary = {
        "role": "system",
        "content": content,
        CONTEXT_REFS_KEY: data.get(COMPACT_CONTEXT_REFS_METADATA_KEY, []),
    }
    return summary, floor, floor if native is not None else 0


def _root_events_matching(
    db: Session,
    task_id: int,
    *,
    through: int,
    kinds: frozenset[str],
    column: Any,
    values: set[Any],
) -> list[TaskExecutionEvent]:
    """Fetch covered facts by an identity column, in bounded IN batches."""
    identities = sorted(value for value in values if value)
    events: list[TaskExecutionEvent] = []
    for offset in range(0, len(identities), MAX_EXECUTION_EVENT_PAGE_SIZE):
        events.extend(
            db.scalars(
                select(TaskExecutionEvent)
                .where(
                    TaskExecutionEvent.task_id == task_id,
                    TaskExecutionEvent.scope_id == "root",
                    TaskExecutionEvent.sequence <= through,
                    TaskExecutionEvent.kind.in_(kinds),
                    column.in_(
                        identities[offset : offset + MAX_EXECUTION_EVENT_PAGE_SIZE]
                    ),
                )
                .order_by(TaskExecutionEvent.sequence)
            )
        )
    return events


def _latest_summary_floor(
    db: Session, task_id: int, horizon: int
) -> tuple[int, bool] | None:
    """Find the summary the projection will keep, newest first.

    Returns its floor and whether the coverage is native (model history), or
    None when no summary at the horizon makes a usable coverage claim.
    """
    query = select(TaskExecutionEvent).where(
        TaskExecutionEvent.task_id == task_id,
        TaskExecutionEvent.scope_id == "root",
        TaskExecutionEvent.kind == "action_end_compact",
    )
    # The newest summary is usually usable: do not page in older bodies first.
    cursor, limit = horizon + 1, 1
    while page := list(
        db.scalars(
            query.where(TaskExecutionEvent.sequence < cursor)
            .order_by(TaskExecutionEvent.sequence.desc())
            .limit(limit)
        )
    ):
        for event in page:
            # Unusable summaries reference no anchor and issue no lookup; the
            # first one with a coordinate returns or raises, so this queries at
            # most once. Batching the page would resolve unused older coordinates.
            anchors = _load_context_anchors(db, task_id, horizon, [event])
            coverage = _summary_coverage(event, anchors)
            if coverage is not None:
                return coverage[1], coverage[2] != 0
        cursor, limit = int(page[-1].sequence), MAX_EXECUTION_EVENT_PAGE_SIZE
    return None


def _covered_facts_still_used(
    db: Session, task_id: int, floor: int, suffix: list[TaskExecutionEvent]
) -> list[TaskExecutionEvent]:
    """Covered facts a native summary's suffix still projects.

    Inputs accepted before the floor but applied after it, and every start and
    outcome of a tool batch with any start or outcome after it. Selection reads
    identity columns only; the projection still validates what it uses.

    This mirrors the pre-floor facts ``_project`` consumes for a native floor.
    Changing which of them the projection uses requires updating this function;
    ``test_task_event_context_equivalence.py`` guards that against a live read
    of the full history.
    """
    turns = {e.turn_id for e in suffix if e.kind == "input_applied"}
    prior = _root_events_matching(
        db,
        task_id,
        through=floor,
        kinds=_ACCEPTED_KINDS,
        column=TaskExecutionEvent.turn_id,
        values=turns,
    )
    suffix_starts = [e for e in suffix if e.kind == "tool_execution_start"]
    started = {e.tool_attempt_id for e in suffix_starts}
    unmatched = {
        e.tool_attempt_id
        for e in suffix
        if e.kind in _OUTCOME_KINDS and e.tool_attempt_id not in started
    }
    early = _root_events_matching(
        db,
        task_id,
        through=floor,
        kinds=_START_KINDS,
        column=TaskExecutionEvent.tool_attempt_id,
        values=unmatched,
    )
    # Batch membership comes from starts: an outcome's own batch id may not
    # match its start, which the projection must still reject.
    early += _root_events_matching(
        db,
        task_id,
        through=floor,
        kinds=_START_KINDS,
        column=TaskExecutionEvent.assistant_message_id,
        values={e.assistant_message_id for e in (*suffix_starts, *early)},
    )
    prior += early
    prior += _root_events_matching(
        db,
        task_id,
        through=floor,
        kinds=_OUTCOME_KINDS,
        column=TaskExecutionEvent.tool_attempt_id,
        values={e.tool_attempt_id for e in (*suffix_starts, *early)},
    )
    return prior


def _load_model_events(
    db: Session, task_id: int, horizon: int
) -> list[TaskExecutionEvent]:
    """Root events the projection needs, ascending by sequence.

    Without a usable summary this is the whole history. Otherwise it is the
    suffix after the summary's floor plus the covered facts that suffix still
    projects; covered facts that are not projected are neither read nor
    revalidated.
    """
    found = _latest_summary_floor(db, task_id, horizon)
    if found is None:
        return _root_events(db, task_id, after=0, through=horizon, kinds=_MODEL_KINDS)
    floor, native = found
    suffix = _root_events(db, task_id, after=floor, through=horizon, kinds=_MODEL_KINDS)
    if native:
        prior = _covered_facts_still_used(db, task_id, floor, suffix)
    else:
        # A legacy coordinate covers transcript facts only: every earlier tool
        # and settlement fact remains eligible, so this read is unbounded.
        prior = _root_events(
            db,
            task_id,
            after=0,
            through=floor,
            kinds=_TOOL_KINDS | {"execution_settled"},
        )
    if not any(event.kind == "skill_select_end" for event in suffix):
        skill = db.scalar(
            select(TaskExecutionEvent)
            .where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == "root",
                TaskExecutionEvent.kind == "skill_select_end",
                TaskExecutionEvent.sequence <= floor,
            )
            .order_by(TaskExecutionEvent.sequence.desc())
            .limit(1)
        )
        if skill is not None:
            prior.append(skill)
    # Starts found by attempt and by batch overlap.
    by_sequence = {int(event.sequence): event for event in (*prior, *suffix)}
    return [by_sequence[sequence] for sequence in sorted(by_sequence)]


def load_task_event_context(
    db: Session,
    task_id: int,
    *,
    before_message_id: int | None = None,
    before_turn_id: str | None = None,
) -> TaskEventContext:
    with measure_execution_event_read("model", task_id=task_id) as read:
        return _load_task_event_context(
            db,
            task_id,
            before_message_id=before_message_id,
            before_turn_id=before_turn_id,
            read=read,
        )


def _load_task_event_context(
    db: Session,
    task_id: int,
    *,
    before_message_id: int | None,
    before_turn_id: str | None,
    read: ExecutionEventReadMeasurement,
) -> TaskEventContext:
    horizon = db.scalar(
        select(Task.conversation_event_sequence).where(Task.id == task_id)
    )
    if horizon is None:
        raise ValueError(f"Task {task_id} does not exist")
    if before_turn_id is not None:
        boundary = db.scalar(
            select(TaskExecutionEvent.sequence).where(
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == "root",
                TaskExecutionEvent.kind == "input_accepted",
                TaskExecutionEvent.turn_id == before_turn_id,
            )
        )
        if boundary is None:
            raise ValueError(
                f"Current turn has no accepted event (task_id={task_id}, turn_id={before_turn_id})"
            )
        horizon = min(horizon, boundary - 1)
    elif before_message_id is not None:
        # Queued commands retain the old integer meaning. This is an identity
        # join only; no legacy transcript content enters the model.
        boundary = db.scalar(
            select(TaskExecutionEvent.sequence)
            .join(
                TaskChatMessage,
                TaskChatMessage.execution_event_id == TaskExecutionEvent.event_id,
            )
            .where(
                TaskChatMessage.task_id == task_id,
                TaskChatMessage.id == before_message_id,
                TaskExecutionEvent.task_id == task_id,
                TaskExecutionEvent.scope_id == "root",
            )
        )
        if boundary is None:
            raise ValueError(
                f"Historical cutoff has no root execution event (task_id={task_id}, message_id={before_message_id})"
            )
        horizon = min(horizon, boundary - 1)

    events = _load_model_events(db, task_id, horizon)
    read.rows = len(events)
    return _project(db, task_id, horizon, events)


def _project(
    db: Session, task_id: int, horizon: int, events: list[TaskExecutionEvent]
) -> TaskEventContext:
    """Project root model history from ``events``, ascending by sequence.

    Changing which pre-floor facts this uses requires updating
    ``_covered_facts_still_used``, guarded by the live-oracle equivalence test.
    """
    anchors = _load_context_anchors(db, task_id, horizon, events)
    accepted: dict[str, TaskExecutionEvent] = {}
    applied: dict[str, int] = {}
    skill_name = None
    summary: dict[str, Any] | None = None
    transcript_floor = model_floor = 0
    for event in events:
        if event.kind == "input_accepted":
            data = _data(event)
            if (
                not event.turn_id
                or data.get("role") != "user"
                or not isinstance(data.get("content"), str)
            ):
                raise ValueError(
                    f"Invalid accepted input (task_id={event.task_id}, event_id={event.event_id})"
                )
            accepted[str(event.turn_id)] = event
        elif event.kind == "input_applied":
            data = _data(event)
            state = anchors.get(cast(str, data.get("recovery_event_id")))
            if (
                state is None
                or state[1] != "recovery_state"
                or state[2] != 1
                or state[0] >= event.sequence
            ):
                raise ValueError(
                    f"Applied input has no prior root recovery state (task_id={event.task_id}, event_id={event.event_id})"
                )
            # Existing/non-transcript starts legitimately have no acceptance.
            # Application alone must not invent a transcript message for them.
            if event.turn_id in accepted:
                applied[str(event.turn_id)] = int(event.sequence)
        elif event.kind == "skill_select_end":
            skill_name = _selected_skill_name(event)
        elif event.kind == "action_end_compact":
            coverage = _summary_coverage(event, anchors)
            if coverage is not None:
                summary, transcript_floor, model_floor = coverage

    entries: list[tuple[int, list[dict[str, Any]]]] = []
    if summary is not None:
        entries.append((0, [summary]))
    for turn_id, position in applied.items():
        event = accepted[turn_id]
        covered = (
            position <= model_floor
            if model_floor
            else event.sequence <= transcript_floor
        )
        if covered:
            continue
        data = _data(event)
        entries.append(
            (
                position,
                [
                    {
                        "role": "user",
                        "content": data["content"],
                        CONTEXT_REFS_KEY: [
                            ref.durable_dict()
                            for ref in build_image_context_references(
                                data.get("attachments")
                            )
                        ],
                    }
                ],
            )
        )

    starts: dict[str, TaskExecutionEvent] = {}
    outcomes: dict[str, TaskExecutionEvent] = {}
    batches: dict[str, list[TaskExecutionEvent]] = {}
    for event in events:
        if event.kind == "assistant_message":
            data = _data(event)
            if (
                event.sequence <= transcript_floor
                or data.get("message_type") == TASK_FAILURE_MESSAGE_TYPE
            ):
                continue
            if not isinstance(data.get("content"), str):
                raise ValueError(
                    f"Invalid assistant message (task_id={event.task_id}, event_id={event.event_id})"
                )
            message_type = str(data.get("message_type") or "")
            refs = (
                build_image_context_references(data.get("attachments"))
                if assistant_history_has_safe_ancillary_payload(message_type)
                else ()
            )
            entries.append(
                (
                    int(event.sequence),
                    [
                        {
                            "role": "assistant",
                            "content": client_safe_assistant_history_content(
                                content=data["content"], message_type=message_type
                            ),
                            CONTEXT_REFS_KEY: [ref.durable_dict() for ref in refs],
                        }
                    ],
                )
            )
        elif event.kind == "agent_message" and event.sequence > transcript_floor:
            data = _data(event)
            if data.get("expect_response"):
                metadata = data.get("metadata")
                if metadata is not None and not isinstance(metadata, dict):
                    raise ValueError(
                        f"Invalid question metadata (task_id={event.task_id}, event_id={event.event_id})"
                    )
                content = build_assistant_transcript_content(
                    content=data.get("message"),
                    interactions=(metadata or {}).get("interactions"),
                )
                if content:
                    entries.append(
                        (
                            int(event.sequence),
                            [{"role": "assistant", "content": content}],
                        )
                    )
        elif event.kind in _TOOL_KINDS:
            data = _data(event)
            if data.get("tool_name") in CONTROL_TOOL_NAMES:
                continue
            if not event.tool_attempt_id or not event.assistant_message_id:
                raise ValueError(
                    f"Tool event has no attempt/batch identity (task_id={event.task_id}, event_id={event.event_id})"
                )
            if event.kind == "tool_execution_start":
                if data.get("settlement_delivery") is True:
                    continue
                starts[str(event.tool_attempt_id)] = event
                batches.setdefault(str(event.assistant_message_id), []).append(event)
            else:
                outcomes[str(event.tool_attempt_id)] = event
        elif event.kind == "execution_settled" and event.sequence > model_floor:
            data = _data(event)
            result = data.get("result")
            if not isinstance(result, dict):
                raise ValueError(
                    f"Invalid execution settlement (task_id={event.task_id}, event_id={event.event_id})"
                )
            status = result.get("status", data.get("status"))
            if status in {"waiting_for_user", "interrupted", "paused"}:
                continue
            if status == "cancelled":
                failure = cast(str | None, "- Previous execution was cancelled.")
            else:
                failure = summarize_execution_failure_event(data)
            if failure:
                entries.append(
                    (int(event.sequence), [{"role": "system", "content": failure}])
                )

    # Tool batches are rendered as one declaration followed by contiguous
    # observations. No turn-text equality or provider call-id heuristic.
    retained_batches = []
    retained_calls = 0
    for batch in reversed(list(batches.values())):
        latest = max(
            int(outcomes.get(str(start.tool_attempt_id), start).sequence)
            for start in batch
        )
        if latest > model_floor and retained_calls < 8:
            retained_batches.append(batch)
            retained_calls += len(batch)
    for batch in reversed(retained_batches):
        calls, results, prose = [], [], []
        last_position = max(int(event.sequence) for event in batch)
        for start in batch:
            data = _data(start)
            end = outcomes.get(str(start.tool_attempt_id))
            if end is not None:
                # A resumed interaction's settlement replaces the waiting
                # outcome; the run that resumed the call commits it.
                settles = _data(end).get("settlement_delivery") is True
                if (
                    end.sequence <= start.sequence
                    or end.assistant_message_id != start.assistant_message_id
                    or (end.run_id != start.run_id and not settles)
                ):
                    raise ValueError(
                        f"Tool outcome does not match its start (task_id={task_id}, event_id={end.event_id}, start_event_id={start.event_id})"
                    )
                last_position = max(last_position, int(end.sequence))
                result = _resolve_tool_result(str(end.kind), _data(end))
            else:
                result = {
                    "success": False,
                    "status": "unknown",
                    "error": "Tool execution has no committed result; external effect is unknown.",
                }
            name, call_id, params = (
                data.get("tool_name"),
                data.get("tool_call_id"),
                data.get("tool_params"),
            )
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(call_id, str)
                or not call_id
                or not isinstance(params, dict)
            ):
                raise ValueError(
                    f"Invalid persisted tool call (task_id={task_id}, event_id={start.event_id})"
                )
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(params, ensure_ascii=False),
                    },
                }
            )
            result, result_refs = split_tool_result_context_references(result)
            results.append(
                {
                    CONTEXT_REFS_KEY: [ref.durable_dict() for ref in result_refs],
                    "role": "tool",
                    "tool_call_id": call_id,
                    "tool_name": name,
                    "content": "",
                    "raw_result": result,
                }
            )
            if data.get("assistant_content"):
                prose.append(data["assistant_content"])
        if last_position > model_floor:
            entries.append(
                (
                    int(batch[0].sequence),
                    [
                        {
                            "role": "assistant",
                            "content": "\n".join(prose),
                            "tool_calls": calls,
                        },
                        *results,
                    ],
                )
            )
    for attempt_id, end in outcomes.items():
        if attempt_id not in starts:
            raise ValueError(
                f"Tool result has no matching start (task_id={task_id}, event_id={end.event_id})"
            )

    messages = [
        message
        for _, group in sorted(entries, key=lambda entry: entry[0])
        for message in group
    ]
    remaining = _MAX_HISTORICAL_IMAGE_CONTEXT_REFS
    for message in reversed(messages):
        refs = normalize_context_references(message.pop(CONTEXT_REFS_KEY, ()))
        retained = refs[:remaining]
        remaining -= len(retained)
        if retained:
            message[CONTEXT_REFS_KEY] = [ref.durable_dict() for ref in retained]
    messages = [
        message
        for message in messages
        if message.get("content")
        or message.get(CONTEXT_REFS_KEY)
        or message.get("tool_calls")
        or message["role"] == "tool"
    ]
    watermark = (
        None
        if not events
        else {
            "scope_id": "root",
            "event_id": events[-1].event_id,
            "sequence": int(events[-1].sequence),
        }
    )
    return TaskEventContext(messages, watermark, skill_name)
