"""Template-driven synthetic V2 histories for the read-budget measurements.

Payload shapes come from ``fixtures/execution_event_templates.json``, exported
from real E2E runs by ``tests/e2e/test_execution_event_template_export.py``.
Only text and result fields are scaled; the event order per run mirrors the
committed sequences in that file.

Context-bearing payloads (``recovery_state``, ``llm_call_start``,
``execution_settled``) embed the context the run holds at that point: the
history the model reader would install at setup plus the run's own messages,
so their size grows with the context (plan F5).

Compaction summary coordinates follow production (plan F8): a compaction
records the event watermark captured at its run's setup, i.e. the last root
model-kind event the setup read covered, and every compaction in the run
reuses it. A setup that read no event writes summaries without a coordinate.
SDK/A2A starts read below the new input (``before_turn_id``); a WebSocket first
run reads the full horizon, so its watermark is its own ``input_accepted``.
Both rules are visible in the exported sequences' ``setup_watermark_sequence``.

``BulkSink`` inserts rows in batches; ``WriterSink`` writes the same plan
through the production writer helpers and the production reader, for the
small-scale cross-check.
"""

from __future__ import annotations

import base64
import copy
import json
import random
import uuid
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from typing import Any, Callable

import sqlalchemy as sa

from xagent.config import get_compact_threshold_default
from xagent.core.agent.context.execution import (
    COMPACT_CONTEXT_REFS_METADATA_KEY,
    COMPACT_SUMMARY_METADATA_KEY,
    MODEL_CONTEXT_WATERMARK_METADATA_KEY,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_execution_event import TaskExecutionEvent
from xagent.web.models.uploaded_file import UploadedFile

TEMPLATES_PATH = Path(__file__).parent / "fixtures" / "execution_event_templates.json"

# Reader-visible kinds, as in task_event_context_service._MODEL_KINDS.
MODEL_KINDS = frozenset(
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
# ReAct keeps the newest eight tool calls after the floor in model history.
HISTORY_TOOL_CALLS = 8
USER_CHARS = 400
ANSWER_CHARS = 1500
SUMMARY_CHARS = 5000
# Drop-fallback compaction keeps CompactConfig.max_messages (template: 20).
DROP_KEEP_MESSAGES = 20


@cache
def templates() -> dict[str, dict[str, Any]]:
    exported = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    picked: dict[str, dict[str, Any]] = {}
    for workflow in exported["template_workflows"]:
        for key, payload in exported["templates"][workflow].items():
            picked.setdefault(key, payload)
    return picked


@cache
def exported_sequences() -> dict[str, list[dict[str, Any]]]:
    return json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))["sequences"]


def template(key: str) -> dict[str, Any]:
    return copy.deepcopy(templates()[key])


@cache
def _corpus() -> str:
    """Deterministic word salad: compresses like prose, unlike repeated bytes."""
    rng = random.Random(2089)
    words = [
        "".join(
            rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 9))
        )
        for _ in range(4000)
    ]
    return " ".join(rng.choice(words) for _ in range(600_000))


def text(size: int, salt: int) -> str:
    corpus = _corpus()
    start = (salt * 7919) % (len(corpus) - size - 1)
    return corpus[start : start + size]


def incompressible(size: int, salt: int) -> str:
    rng = random.Random(salt)
    return base64.b64encode(rng.randbytes(size * 3 // 4)).decode()[:size]


@dataclass(frozen=True)
class Workload:
    """One synthetic task history. ``rounds`` normal runs of one round each,
    then an optional long run; ``long_run_first`` puts the long run first."""

    rounds: int = 10
    tools: int = 3
    result_bytes: int = 16 * 1024
    # "none": no compaction; "summary": threshold-driven LLM summary;
    # "drop": threshold-driven drop fallback, which records no summary.
    compaction: str = "none"
    long_run_tools: int = 0
    long_run_first: bool = False
    first_entry: str = "sdk"
    child_scope: bool = False
    attachments: bool = False
    # With attachments, every ``file_every``-th run's last tool writes a file.
    file_every: int = 10
    threshold_tokens: int = field(default_factory=get_compact_threshold_default)
    # Older runs' recovery states become 4 KiB incompressible stand-ins
    # (still out-of-line on PG, overflow pages on SQLite). None keeps all.
    faithful_recovery_runs: int | None = None
    # The first ``lean_runs`` runs replace every context-bearing payload by a
    # 2.5 KiB incompressible stand-in (out-of-line on PG). Only for covered
    # prefixes in DB-side scan measurements, where payloads are not fetched.
    lean_runs: int = 0

    def label(self) -> str:
        parts = [f"N{self.rounds}", f"k{self.tools}", f"r{self.result_bytes // 1024}k"]
        parts.append(self.compaction)
        if self.long_run_tools:
            where = "first" if self.long_run_first else "last"
            parts.append(f"long{self.long_run_tools}-{where}-{self.first_entry}")
        if self.lean_runs:
            parts.append(f"lean{self.lean_runs}")
        if self.child_scope:
            parts.append("child")
        if self.attachments:
            parts.append("files")
        return "-".join(parts)


@dataclass
class Stored:
    sequence: int
    event_id: str


class BulkSink:
    """Batch INSERT; identities are generated here, sequences are task-global."""

    def __init__(self, factory, task_id: int, batch_bytes: int = 32 << 20):
        self.factory, self.task_id = factory, task_id
        self.sequence = 0
        self.rows: list[dict[str, Any]] = []
        self.pending_bytes = 0
        self.batch_bytes = batch_bytes
        self.kinds: list[str] = []
        self.payload_bytes: list[int] = []
        self.now = datetime(2026, 10, 1, tzinfo=timezone.utc)
        self.files: list[dict[str, Any]] = []

    def append(self, spec: dict[str, Any]) -> Stored:
        self.sequence += 1
        event_id = str(uuid.uuid4())
        size = len(json.dumps(spec["payload"], ensure_ascii=False).encode())
        self.kinds.append(spec["kind"])
        self.payload_bytes.append(size)
        self.rows.append(
            {
                "task_id": self.task_id,
                "sequence": self.sequence,
                "event_id": event_id,
                "occurred_at": self.now + timedelta(milliseconds=self.sequence),
                "payload_version": 1,
                **spec,
            }
        )
        self.pending_bytes += size
        if self.pending_bytes >= self.batch_bytes or len(self.rows) >= 2000:
            self.flush()
        return Stored(self.sequence, event_id)

    def register_file(self, file_id: str, user_id: int) -> None:
        self.files.append({"file_id": file_id, "user_id": user_id})

    def flush(self) -> None:
        if not self.rows:
            return
        with self.factory() as db:
            db.execute(sa.insert(TaskExecutionEvent), self.rows)
            db.execute(
                sa.update(Task)
                .where(Task.id == self.task_id)
                .values(conversation_event_sequence=self.sequence)
            )
            db.commit()
        self.rows, self.pending_bytes = [], 0

    def setup_watermark(self, analytic, turn_id, include_input):
        return analytic

    def finish(self, task_fields: dict[str, Any]) -> None:
        self.flush()
        with self.factory() as db:
            for file in self.files:
                db.add(_uploaded_file(self.task_id, **file))
            # finish() also closes an interrupted run: register files once.
            self.files = []
            db.execute(
                sa.update(Task).where(Task.id == self.task_id).values(**task_fields)
            )
            db.commit()


class WriterSink:
    """The same plan through the production writer helpers, one commit each.

    Inputs and answers go through ``stage_chat_message_no_commit``; applied
    inputs are derived by ``stage_applied_inputs_no_commit`` from the recovery
    state; native summary coordinates are validated by
    ``compact_transcript_event_watermark``; setup watermarks are read with
    ``load_task_event_context`` exactly as task setup does.
    """

    def __init__(self, factory, task_id: int):
        self.factory, self.task_id = factory, task_id
        self.kinds: list[str] = []
        self.payload_bytes: list[int] = []
        self.watermark_mismatches: list[tuple[Any, Any]] = []
        self.files: list[dict[str, Any]] = []

    def _record(self, db, event) -> Stored:
        self.kinds.append(str(event.kind))
        self.payload_bytes.append(
            len(json.dumps(event.payload, ensure_ascii=False).encode())
        )
        return Stored(int(event.sequence), str(event.event_id))

    def append(self, spec: dict[str, Any]) -> Stored:
        from xagent.web.services.task_execution_event_writer import (
            append_fact_no_commit,
            compact_transcript_event_watermark,
            stage_applied_inputs_no_commit,
            stage_chat_message_no_commit,
        )

        kind, payload = spec["kind"], spec["payload"]
        with self.factory() as db:
            if kind == "input_applied":
                # Already derived from the preceding recovery state.
                event = db.scalar(
                    sa.select(TaskExecutionEvent).where(
                        TaskExecutionEvent.task_id == self.task_id,
                        TaskExecutionEvent.idempotency_key == spec["idempotency_key"],
                    )
                )
                assert event is not None, spec["idempotency_key"]
                return self._record(db, event)
            if kind in {"input_accepted", "assistant_message"}:
                task = db.get(Task, self.task_id)
                message = TaskChatMessage(
                    task_id=self.task_id,
                    user_id=task.user_id,
                    role=payload["role"],
                    content=payload["content"],
                    message_type=payload["message_type"],
                    turn_id=spec["turn_id"],
                    attachments=payload.get("attachments"),
                    source_event_id=spec["idempotency_key"],
                )
                if kind == "input_accepted":
                    task.run_id = spec["run_id"]
                staged = stage_chat_message_no_commit(db, message)
                db.flush()
                event = db.scalar(
                    sa.select(TaskExecutionEvent).where(
                        TaskExecutionEvent.event_id == staged.execution_event_id
                    )
                )
                db.commit()
                return self._record(db, event)
            if kind == "action_end_compact" and spec["scope_id"] == "root":
                compact_transcript_event_watermark(
                    db, self.task_id, spec["idempotency_key"], payload["data"]
                )
            event = append_fact_no_commit(
                db,
                task_id=self.task_id,
                kind=kind,
                key=spec["idempotency_key"],
                payload=payload,
                scope_id=spec["scope_id"],
                run_id=spec["run_id"],
                turn_id=spec["turn_id"],
                assistant_message_id=spec["assistant_message_id"],
                tool_attempt_id=spec["tool_attempt_id"],
            )
            stored = self._record(db, event)
            if kind == "recovery_state":
                stage_applied_inputs_no_commit(db, event)
            db.commit()
            return stored

    def register_file(self, file_id: str, user_id: int) -> None:
        self.files.append({"file_id": file_id, "user_id": user_id})

    def setup_watermark(self, analytic, turn_id, include_input):
        from xagent.web.services.task_event_context_service import (
            load_task_event_context,
        )

        with self.factory() as db:
            context = load_task_event_context(
                db, self.task_id, before_turn_id=None if include_input else turn_id
            )
        if context.watermark != analytic:
            self.watermark_mismatches.append((analytic, context.watermark))
        return context.watermark

    def finish(self, task_fields: dict[str, Any]) -> None:
        with self.factory() as db:
            for file in self.files:
                db.add(_uploaded_file(self.task_id, **file))
            # finish() also closes an interrupted run: register files once.
            self.files = []
            db.execute(
                sa.update(Task).where(Task.id == self.task_id).values(**task_fields)
            )
            db.commit()


def _uploaded_file(task_id: int, file_id: str, user_id: int) -> UploadedFile:
    return UploadedFile(
        file_id=file_id,
        user_id=user_id,
        task_id=task_id,
        filename="derived.txt",
        storage_path=f"synthetic/{file_id}/derived.txt",
        workspace_relative_path="output/derived.txt",
        workspace_category="output",
        mime_type="text/plain",
        file_size=20,
    )


@dataclass
class _Message:
    """One context message: the dict persisted in snapshots, plus its tokens."""

    body: dict[str, Any]

    @property
    def tokens(self) -> int:
        content = self.body.get("content") or ""
        return max(1, (len(content) + 48) // 4)


class Generator:
    """Emit one workload's events into a sink, tracking what readers see."""

    def __init__(self, workload: Workload, sink, task_id: int, user_id: int):
        self.w, self.sink, self.task_id, self.user_id = workload, sink, task_id, user_id
        final = template("recovery_state:final")["data"]["snapshot"]["context"]
        self.message_templates = {
            "user": final["messages"][0],
            "tool_call": final["messages"][1],
            "tool": final["messages"][2],
            "answer": final["messages"][-1],
        }
        preview = template("llm_call_start")["data"]["context_preview"]
        self.system_message = preview[0]
        self.system_tokens = max(1, len(preview[0]["content"]) // 4)
        self.salt = 0
        # Reader mirror: root model-kind events and summaries, by sequence.
        self.model_events: list[tuple[int, str, str, dict[str, Any]]] = []
        self.summaries: list[tuple[int, int | None]] = []
        self.applied: dict[str, int] = {}
        # Sequence-ordered mirrors, searched by floor and horizon.
        self.applied_positions: list[int] = []
        self.applied_turns: list[str] = []
        self.answer_positions: list[int] = []
        self.tool_ends: list[int] = []
        self.accepted: dict[str, int] = {}
        self.user_text: dict[str, str] = {}
        self.answers: list[tuple[int, str]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.runs: list[str] = []
        self.run_count = 0
        self.compactions = 0
        self.unusable_summaries = 0
        self.last_run: dict[str, Any] = {}
        self.pending: dict[str, Any] | None = None
        self.last_context: list[_Message] = []

    # ------------------------------------------------------------------ mirror
    def _next_salt(self) -> int:
        self.salt += 1
        return self.salt

    def _emit(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        key: str | None = None,
        scope: str = "root",
        run_id: str | None = None,
        turn_id: str | None = None,
        batch: str | None = None,
        attempt: str | None = None,
    ) -> Stored:
        if "data" in payload and kind not in {"execution_settled"}:
            # Trace-shaped facts: the envelope trace_handlers writes, each
            # with its own protocol event id (templates share a placeholder).
            payload["protocol_event_id"] = str(uuid.uuid4())
            payload.setdefault("event_type", kind)
            payload.setdefault("step_id", None)
            payload.setdefault("parent_event_id", None)
        stored = self.sink.append(
            {
                "kind": kind,
                "scope_id": scope,
                "idempotency_key": key or f"runtime:{uuid.uuid4()}",
                "payload": payload,
                "run_id": run_id,
                "turn_id": turn_id,
                "assistant_message_id": batch,
                "tool_attempt_id": attempt,
            }
        )
        if scope == "root" and kind in MODEL_KINDS:
            self.model_events.append((stored.sequence, stored.event_id, kind, {}))
        return stored

    def _watermark(self, horizon: int) -> dict[str, Any] | None:
        for sequence, event_id, _kind, _ in reversed(self.model_events):
            if sequence <= horizon:
                return {"scope_id": "root", "event_id": event_id, "sequence": sequence}
        return None

    def _floor(self, horizon: int) -> int:
        for sequence, floor in reversed(self.summaries):
            if sequence <= horizon and floor is not None:
                return floor
        return 0

    def _history(self, horizon: int) -> list[_Message]:
        """What load_task_event_context projects at ``horizon``, as context."""
        floor = self._floor(horizon)
        entries: list[tuple[int, list[_Message]]] = []
        if floor:
            entries.append((0, [self._summary_message()]))

        def window(positions: list[int]) -> range:
            return range(
                bisect_right(positions, floor), bisect_right(positions, horizon)
            )

        for i in window(self.applied_positions):
            turn = self.applied_turns[i]
            entries.append(
                (self.applied_positions[i], [self._user(self.user_text[turn], turn)])
            )
        for i in window(self.answer_positions):
            sequence, answer = self.answers[i]
            entries.append((sequence, [self._answer(answer)]))
        calls = window(self.tool_ends)
        for i in range(max(calls.start, calls.stop - HISTORY_TOOL_CALLS), calls.stop):
            call = self.tool_calls[i]
            entries.append((call["start"], self._tool_pair(call)))
        return [m for _, group in sorted(entries, key=lambda e: e[0]) for m in group]

    # -------------------------------------------------------------- messages
    def _user(self, content: str, turn: str | None) -> _Message:
        body = copy.deepcopy(self.message_templates["user"])
        body["content"] = content
        metadata = body.setdefault("metadata", {})
        if turn is None:
            metadata.pop("turn_id", None)
        else:
            metadata["turn_id"] = turn
        return _Message(body)

    def _answer(self, content: str) -> _Message:
        body = copy.deepcopy(self.message_templates["answer"])
        body["content"] = content
        return _Message(body)

    def _summary_message(self) -> _Message:
        body = copy.deepcopy(self.message_templates["answer"])
        body.update(role="system", content=self._summary_text(), tool_calls=None)
        body["metadata"] = {"compacted_context": True}
        return _Message(body)

    def _summary_text(self) -> str:
        return text(SUMMARY_CHARS, 1)

    def _tool_pair(self, call: dict[str, Any]) -> list[_Message]:
        declaration = copy.deepcopy(self.message_templates["tool_call"])
        declaration["tool_calls"] = [
            {
                "id": call["call_id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["params"]),
                },
            }
        ]
        observation = copy.deepcopy(self.message_templates["tool"])
        result = call["result"]
        rendered = result if isinstance(result, str) else json.dumps(result)
        observation["content"] = f"Tool {call['name']} returned: {rendered}"
        observation["metadata"] = {
            **(observation.get("metadata") or {}),
            "raw_result": result,
            "tool_name": call["name"],
        }
        observation["tool_call_id"] = call["call_id"]
        return [_Message(declaration), _Message(observation)]

    def _tokens(self, context: list[_Message]) -> int:
        return self.system_tokens + sum(m.tokens for m in context)

    # ---------------------------------------------------------------- payloads
    def _recovery(
        self,
        label: str,
        run: dict[str, Any],
        context: list[_Message],
        pending: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        key = f"recovery_state:{label}"
        payload = template(key if key in templates() else "recovery_state:before_llm")
        data = payload["data"]
        execution_id = str(self.task_id)
        data.update(
            label=label, execution_id=execution_id, root_execution_id=execution_id
        )
        snapshot = data["snapshot"]
        snapshot.update(label=label, execution_id=execution_id)
        ctx = snapshot["context"]
        ctx["execution_id"] = execution_id
        ctx["messages"] = [m.body for m in context]
        metadata = ctx.setdefault("metadata", {})
        metadata.update(run_id=run["run_id"], turn_id=run["turn_id"])
        if run["watermark"] is None:
            metadata.pop(MODEL_CONTEXT_WATERMARK_METADATA_KEY, None)
        else:
            metadata[MODEL_CONTEXT_WATERMARK_METADATA_KEY] = dict(run["watermark"])
        state = snapshot["pattern_state"]
        # Copy: a buffered bulk row must not see later ledger entries.
        state["tool_ledger"] = dict(run["ledger"])
        state["pending_tool_calls"] = pending or []
        return payload

    def _stub_recovery(self, run: dict[str, Any], label: str) -> dict[str, Any]:
        payload = template("recovery_state:before_llm")
        data = payload["data"]
        data.update(
            label=label,
            execution_id=str(self.task_id),
            root_execution_id=str(self.task_id),
        )
        data["snapshot"]["context"]["messages"] = []
        data["snapshot"]["context"]["metadata"] = {"run_id": run["run_id"]}
        data["snapshot"]["stand_in"] = incompressible(4096, self._next_salt())
        return payload

    def _state(
        self,
        label: str,
        run: dict[str, Any],
        context: list[_Message],
        pending: list[dict[str, Any]] | None = None,
        scope: str = "root",
    ) -> Stored:
        self.last_context = context
        stub = scope == "root" and (
            run["index"] < self.w.lean_runs
            or (
                self.w.faithful_recovery_runs is not None
                and run["index"] < self._total_runs() - self.w.faithful_recovery_runs
            )
        )
        payload = (
            self._stub_recovery(run, label)
            if stub
            else self._recovery(label, run, context, pending)
        )
        return self._emit("recovery_state", payload, scope=scope, run_id=run["run_id"])

    def _total_runs(self) -> int:
        return self.w.rounds + (1 if self.w.long_run_tools else 0)

    def _llm_start(
        self,
        run,
        context: list[_Message],
        *,
        scope: str = "root",
        purpose: str | None = None,
    ) -> None:
        payload = template("llm_call_start")
        data = payload["data"]
        preview = [self.system_message, *[m.body for m in context]]
        data["context_preview"] = preview[-5:]
        data["context_messages_count"] = len(context) + 1
        data["context_tokens"] = self._tokens(context)
        data["context_threshold"] = self.w.threshold_tokens
        if purpose:
            data["purpose"] = purpose
        if run["index"] < self.w.lean_runs:
            data["context_preview"] = []
            data["stand_in"] = incompressible(2560, self._next_salt())
        self._emit("llm_call_start", payload, scope=scope, run_id=run["run_id"])

    def _llm_end(
        self,
        run,
        *,
        call: dict[str, Any] | None = None,
        answer: str = "",
        scope: str = "root",
    ) -> None:
        if call is None:
            payload = template("llm_call_end:answer")
            payload["data"]["response"] = {"content": answer}
        else:
            payload = template("llm_call_end:tool_call")
            payload["data"]["response"] = {
                "content": "",
                "tool_calls": [
                    {
                        "id": call["call_id"],
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["params"]),
                        },
                    }
                ],
            }
        self._emit("llm_call_end", payload, scope=scope, run_id=run["run_id"])

    def _maybe_compact(self, run, context: list[_Message]) -> list[_Message]:
        if (
            self.w.compaction == "none"
            or self._tokens(context) <= self.w.threshold_tokens
        ):
            return context
        latest_user = next(m for m in reversed(context) if m.body.get("role") == "user")
        tokens = self._tokens(context)
        data: dict[str, Any] = {
            "compact_type": "execution_context",
            "original_count": len(context),
            "original_tokens": tokens,
            "threshold": self.w.threshold_tokens,
        }
        if self.w.compaction == "summary":
            self._llm_start(run, context, purpose="context_compaction")
            summary = self._summary_text()
            self._llm_end(run, answer=summary)
            kept = [self._summary_message(), latest_user]
            data.update(
                strategy="llm_summary",
                final_count=len(kept),
                removed_count=len(context) + 1 - len(kept),
                summary_chars=len(summary),
                compact_model="synthetic-compact-model",
                **{
                    COMPACT_SUMMARY_METADATA_KEY: summary,
                    COMPACT_CONTEXT_REFS_METADATA_KEY: [],
                },
            )
            if run["watermark"] is not None:
                data[MODEL_CONTEXT_WATERMARK_METADATA_KEY] = dict(run["watermark"])
        else:
            kept = context[-DROP_KEEP_MESSAGES:]
            if latest_user not in kept:
                kept = [latest_user, *kept[1:]]
            data.update(
                strategy="drop_oldest",
                final_count=len(kept),
                removed_count=len(context) - len(kept),
                fallback_strategy="drop_oldest",
            )
        self._emit(
            "action_start_compact",
            {"data": data, "event_type": "action_start_compact"},
            run_id=run["run_id"],
        )
        end = self._emit(
            "action_end_compact",
            {"data": {**data, "success": True}, "event_type": "action_end_compact"},
            run_id=run["run_id"],
        )
        floor = (
            run["watermark"]["sequence"]
            if (self.w.compaction == "summary" and run["watermark"] is not None)
            else None
        )
        self.summaries.append((end.sequence, floor))
        self.compactions += 1
        if floor is None:
            self.unusable_summaries += 1
        return kept

    # -------------------------------------------------------------------- runs
    def _child(self, run: dict[str, Any]) -> None:
        """A delegated child execution in its own scope (build id)."""
        scope = f"agent:{uuid.uuid4().hex[:12]}"
        child = {**run, "ledger": {}, "watermark": None}
        self._emit(
            "react_task_start",
            template("react_task_start"),
            scope=scope,
            run_id=run["run_id"],
        )
        # The child's task message is not a root transcript input.
        context = [self._user(text(USER_CHARS, self._next_salt()), None)]
        self._state("before_llm", child, context, scope=scope)
        call = self._call()
        self._llm_start(child, context, scope=scope)
        self._llm_end(child, call=call, scope=scope)
        self._state("after_llm", child, context, scope=scope)
        self._tool(child, call, scope=scope)
        context += self._tool_pair(call)
        self._state("after_tool", child, context, scope=scope)
        answer = text(ANSWER_CHARS, self._next_salt())
        self._llm_start(child, context, scope=scope)
        self._llm_end(child, answer=answer, scope=scope)
        self._state("final", child, context + [self._answer(answer)], scope=scope)
        end = template("react_task_end")
        end["data"]["result"].update(output=answer, response=answer)
        self._emit("react_task_end", end, scope=scope, run_id=run["run_id"])

    def _call(self, file_output: bool = False) -> dict[str, Any]:
        name = "write_file" if file_output else "read_file"
        params = (
            {"file_path": "derived.txt", "content": text(200, self._next_salt())}
            if file_output
            else {"file_path": f"source-{self._next_salt()}.txt"}
        )
        if file_output:
            result: Any = template("tool_execution_end:file")["data"]["result"]
            file_id = str(uuid.uuid4())
            link = f"[derived.txt](file:{file_id})"
            for holder in (result, result["file_ref"]):
                holder.update(
                    file_id=file_id,
                    markdown_link=link,
                    download_url=f"/api/files/download/{file_id}",
                    preview_url=f"/api/files/preview/{file_id}",
                )
            self.sink.register_file(file_id, self.user_id)
        elif self.w.result_bytes > 48 * 1024:
            # Stay under the 50k per-string inline limit, as a large dict
            # result (e.g. a parsed document) would.
            parts = -(-self.w.result_bytes // (32 * 1024))
            result = {
                f"part_{i}": text(self.w.result_bytes // parts, self._next_salt())
                for i in range(parts)
            }
        else:
            result = text(self.w.result_bytes, self._next_salt())
        return {
            "call_id": f"call_{uuid.uuid4().hex[:24]}",
            "attempt": str(uuid.uuid4()),
            "batch": str(uuid.uuid4()),
            "name": name,
            "params": params,
            "result": result,
            "file_output": file_output,
        }

    def _tool(
        self, run, call: dict[str, Any], *, scope: str = "root", end: bool = True
    ) -> None:
        start = template("tool_execution_start")
        start["data"].update(
            assistant_message_id=call["batch"],
            tool_attempt_id=call["attempt"],
            tool_call_id=call["call_id"],
            tool_name=call["name"],
            tool_params=call["params"],
            turn_id=run["turn_id"],
        )
        ids = dict(
            scope=scope,
            run_id=run["run_id"],
            turn_id=run["turn_id"],
            batch=call["batch"],
            attempt=call["attempt"],
        )
        started = self._emit(
            "tool_execution_start",
            start,
            key=f"tool:{call['attempt']}:tool_execution_start",
            **ids,
        )
        if scope == "root" and self.w.child_scope and not run.get("child_done"):
            run["child_done"] = True
            self._child(run)
        if not end:
            return
        key = (
            "tool_execution_end:file"
            if call["file_output"]
            else "tool_execution_end:text"
        )
        finish = template(key)
        finish["data"].update(
            assistant_message_id=call["batch"],
            tool_attempt_id=call["attempt"],
            tool_call_id=call["call_id"],
            tool_name=call["name"],
            tool_params=call["params"],
            turn_id=run["turn_id"],
            result=call["result"],
        )
        ended = self._emit(
            "tool_execution_end",
            finish,
            key=f"tool:{call['attempt']}:tool_execution_end",
            **ids,
        )
        run["ledger"][call["call_id"]] = {
            "args": call["params"],
            "result": call["result"],
            "status": "completed",
            "tool_call_id": call["call_id"],
            "tool_name": call["name"],
            "turn_id": run["turn_id"],
        }
        if scope == "root":
            self.tool_calls.append(
                {**call, "start": started.sequence, "end": ended.sequence}
            )
            self.tool_ends.append(ended.sequence)

    def _pending_call(self, call: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": call["call_id"],
            "name": call["name"],
            "args": call["params"],
            "tool_attempt_id": call["attempt"],
            "assistant_message_id": call["batch"],
        }

    def run(self, tools: int, *, stop_after_tool_end: bool = False) -> None:
        index = self.run_count
        self.run_count += 1
        run_id, turn = str(uuid.uuid4()), str(uuid.uuid4())
        websocket_first = index == 0 and self.w.first_entry == "websocket"
        user_text = text(USER_CHARS, self._next_salt())
        accepted = template("input_accepted")
        accepted.update(
            content=user_text, turn_id=turn, user_id=self.user_id, attachments=[]
        )
        accepted_row = self._emit(
            "input_accepted",
            accepted,
            key=f"message:{turn}",
            run_id=run_id if websocket_first else None,
            turn_id=turn,
        )
        self.accepted[turn] = accepted_row.sequence
        self.user_text[turn] = user_text
        command = template("command_accepted")
        command["command_id"] = str(uuid.uuid4())
        command["payload"].update(message=user_text, turn_id=turn, run_id=run_id)
        self._emit("command_accepted", command, key=f"command:{uuid.uuid4()}")
        delivery = template("input_delivery_changed")
        self._emit(
            "input_delivery_changed",
            delivery,
            key=f"delivery:{turn}:{delivery.get('status')}",
            turn_id=turn,
        )
        # Task setup: SDK/A2A read below the new input; a WebSocket first run
        # reads the whole committed horizon, including its own input.
        horizon = self.sink_sequence() if websocket_first else accepted_row.sequence - 1
        analytic = self._watermark(horizon)
        watermark = self.sink.setup_watermark(analytic, turn, websocket_first)
        context = self._history(horizon) + [self._user(user_text, turn)]
        run = {
            "index": index,
            "run_id": run_id,
            "turn_id": turn,
            "ledger": {},
            "watermark": watermark,
        }
        self.runs.append(run_id)
        self.last_run = run
        message = template("user_message")
        message["data"].update(message=user_text, attachments=[], files=[])
        self._emit("user_message", message, run_id=run_id, turn_id=turn)
        self._emit("react_task_start", template("react_task_start"), run_id=run_id)
        state = self._state("before_llm", run, context)
        applied = template("input_applied")
        applied["recovery_event_id"] = state.event_id
        applied_row = self._emit(
            "input_applied",
            applied,
            key=f"input-applied:{turn}",
            run_id=run_id,
            turn_id=turn,
        )
        self.applied[turn] = applied_row.sequence
        self.applied_positions.append(applied_row.sequence)
        self.applied_turns.append(turn)
        file_round = (
            self.w.attachments and tools and (index + 1) % self.w.file_every == 0
        )
        for i in range(tools):
            context = self._maybe_compact(run, context)
            call = self._call(file_output=bool(file_round and i == tools - 1))
            self._llm_start(run, context)
            self._llm_end(run, call=call)
            pending = [self._pending_call(call)]
            self._state("after_llm", run, context, pending)
            self._state("before_tool", run, context, pending)
            self._tool(run, call)
            if stop_after_tool_end and i == tools - 1:
                # Crash after the outcome committed, before the next state:
                # recovery resumes from before_tool and reads the outcome.
                self.pending = self._pending_call(call)
                return
            context = context + self._tool_pair(call)
            self._state("after_tool", run, context)
            self._state("before_llm", run, context)
        context = self._maybe_compact(run, context)
        answer = text(ANSWER_CHARS, self._next_salt())
        if file_round:
            link = self.tool_calls[-1]["result"]["markdown_link"]
            answer = f"{answer} {link}"
        self._llm_start(run, context)
        if not tools:
            self._emit(
                "final_answer_start", template("final_answer_start"), run_id=run_id
            )
        self._llm_end(run, answer=answer)
        stream_id = None
        if not tools:
            # Streamed answers: the completion links the stream it settles.
            stream_id = f"final_answer_{uuid.uuid4().hex}"
            final = template("final_answer_end")
            final["data"].update(
                content=answer, task_id=self.task_id, message_id=stream_id
            )
            self._emit("final_answer_end", final, run_id=run_id)
        context = context + [self._answer(answer)]
        self._state("after_llm", run, context)
        self._state("final", run, context)
        if not tools:
            self._state("completed", run, context)
        end = template("react_task_end")
        end["data"]["result"].update(output=answer, response=answer)
        self._emit("react_task_end", end, run_id=run_id)
        ai = template("ai_message")
        self._emit("ai_message", ai, run_id=run_id)
        completion = template("task_completion")
        completion["data"]["result"] = {"content": answer}
        if stream_id:
            completion["data"]["result"]["stream_message_id"] = stream_id
        self._emit("task_completion", completion, run_id=run_id)
        self._emit(
            "control_state_changed", template("control_state_changed"), run_id=run_id
        )
        assistant = template("assistant_message")
        assistant.update(content=answer, user_id=self.user_id, turn_id=None)
        answer_row = self._emit(
            "assistant_message", assistant, key=f"assistant:{run_id}", run_id=run_id
        )
        self.answers.append((answer_row.sequence, answer))
        self.answer_positions.append(answer_row.sequence)
        settled = template("execution_settled")
        result = settled["result"]
        result.update(output=answer)
        agent_result = result["agent_result"]
        agent_result.update(output=answer, response=answer)
        agent_result["context"]["messages"] = [m.body for m in context]
        if index < self.w.lean_runs:
            agent_result["context"]["messages"] = []
            agent_result["stand_in"] = incompressible(2560, self._next_salt())
        agent_result["context"]["execution_id"] = str(self.task_id)
        if file_round:
            output = copy.deepcopy(result["file_outputs"][0])
            output.update(file_id=self.tool_calls[-1]["result"]["file_id"])
            result["file_outputs"] = [output]
        else:
            result["file_outputs"] = []
        self._emit(
            "execution_settled",
            settled,
            key=f"result:{run_id}:{index}:completed",
            run_id=run_id,
        )
        delivered = template("input_delivery_changed")
        delivered["status"] = "delivered"
        self._emit(
            "input_delivery_changed",
            delivered,
            key=f"delivery:{turn}:delivered",
            turn_id=turn,
        )

    def sink_sequence(self) -> int:
        if isinstance(self.sink, BulkSink):
            return self.sink.sequence
        with self.sink.factory() as db:
            return int(db.get(Task, self.task_id).conversation_event_sequence)

    def generate(self) -> None:
        if self.w.long_run_tools and self.w.long_run_first:
            self.run(self.w.long_run_tools)
        for _ in range(self.w.rounds):
            self.run(self.w.tools)
        if self.w.long_run_tools and not self.w.long_run_first:
            self.run(self.w.long_run_tools)
        self.sink.finish(
            {
                "status": TaskStatus.COMPLETED,
                "run_id": self.runs[-1],
            }
        )

    def interrupted_run(self, tools: int = 1) -> dict[str, Any]:
        """Append a run that stops after its last tool outcome committed."""
        self.run(max(tools, 1), stop_after_tool_end=True)
        self.sink.finish({"status": TaskStatus.RUNNING, "run_id": self.runs[-1]})
        assert self.pending is not None
        return self.pending


def new_task(
    factory, owner_task_id: int, title: str = "read budget"
) -> tuple[int, int]:
    with factory() as db:
        owner = db.get(Task, owner_task_id)
        task = Task(
            user_id=owner.user_id,
            title=title,
            description=title,
            conversation_storage_version=2,
        )
        db.add(task)
        db.commit()
        return int(task.id), int(task.user_id)


def build(
    factory,
    owner_task_id: int,
    workload: Workload,
    *,
    sink_type: Callable[..., Any] = BulkSink,
) -> tuple[int, Generator]:
    task_id, user_id = new_task(factory, owner_task_id, workload.label())
    sink = sink_type(factory, task_id)
    generator = Generator(workload, sink, task_id, user_id)
    generator.generate()
    return task_id, generator


def _rss_main(url: str, task_id: int, measure: str, run_id: str) -> None:
    """Run one read in this fresh process; print peak RSS growth as JSON.

    ``ru_maxrss`` is a high-water mark, so the delta is the read's growth over
    the process peak after imports and one warm-up query. Unlike tracemalloc
    it includes driver buffers and C allocations.
    """
    import resource
    import sys
    import time

    from sqlalchemy.orm import sessionmaker

    from xagent.web.services.hot_path_cache import (
        NoOpCache,
        set_cache_backend_for_testing,
    )

    engine = sa.create_engine(url)
    factory = sessionmaker(engine)
    with factory() as db:
        db.execute(sa.select(Task.id).where(Task.id == task_id)).all()
    set_cache_backend_for_testing(NoOpCache())

    def read() -> Any:
        with factory() as db:
            if measure == "model":
                from xagent.web.services.task_event_context_service import (
                    load_task_event_context,
                )

                return len(load_task_event_context(db, task_id).messages)
            if measure == "display":
                from xagent.web.services.task_event_display import (
                    load_event_display_snapshot,
                )

                return len(load_event_display_snapshot(db, task_id).events)
            if measure == "websocket":
                from xagent.web.api.websocket import (
                    _load_event_historical_stream_snapshot,
                )

                task = db.get(Task, task_id)
                return len(_load_event_historical_stream_snapshot(db, task).events)
            if measure == "checkpoint":
                from xagent.web.services.task_execution_event_recovery import (
                    read_event_checkpoint,
                )

                try:
                    return bool(
                        read_event_checkpoint(
                            db,
                            task_id=task_id,
                            scope_id="root",
                            execution_id=str(task_id),
                            run_id=run_id,
                        )
                    )
                except Exception as error:  # noqa: BLE001 - a refusal is a result
                    return type(error).__name__
        raise ValueError(measure)

    # Import the reader modules before the baseline so imports are excluded.
    import xagent.web.api.websocket  # noqa: F401
    import xagent.web.services.task_event_context_service  # noqa: F401
    import xagent.web.services.task_event_display  # noqa: F401
    import xagent.web.services.task_execution_event_recovery  # noqa: F401

    scale = 1 if sys.platform == "darwin" else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    started = time.perf_counter()
    result = read()
    elapsed = (time.perf_counter() - started) * 1000
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    print(
        json.dumps(
            {
                "measure": measure,
                "result": result,
                "baseline_peak_rss_bytes": before,
                "peak_rss_bytes": after,
                "rss_delta_bytes": after - before,
                "cold_process_elapsed_ms": round(elapsed, 2),
            }
        )
    )


if __name__ == "__main__":
    import sys

    _rss_main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
