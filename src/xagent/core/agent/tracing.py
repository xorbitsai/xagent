from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .attachments import project_file_info_to_chip
from .result import extract_assistant_message
from .trace import (
    get_display_user_message,
    trace_ai_message,
    trace_error,
    trace_task_completion,
    trace_user_message,
)

# Stored on ``context.metadata`` to remember which user messages have already
# been emitted as ``trace_user_message`` events. Uses the latest traced
# message's ISO-8601 UTC timestamp as a high-water mark — ISO strings compare
# lexicographically when timezone-normalized, and the mark survives
# checkpoint round-trips because ``context.metadata`` is persisted in
# ``ExecutionContext.to_dict``.
TRACE_WATERMARK_KEY = "_user_message_trace_watermark"

# Stamped on ``context.metadata`` by ``runner.inject_user_message`` just
# before persisting a freshly-injected user message, and cleared when the
# follow-up persist records the advanced watermark. Carries the injected
# message's ISO-UTC timestamp.
#
# The catch-up loop on resume uses this to disambiguate three cases:
# - both absent  -> old/pre-PR checkpoint, do nothing (don't replay history)
# - pending only -> crashed between persist and trace emit, replay that turn
# - watermark    -> normal "trace already emitted" path, fast-skip
PENDING_MARKER_KEY = "_pending_user_message_trace_timestamp"


@dataclass
class TraceEventCallback:
    """Bridge agent runner callbacks into the existing web trace stream."""

    async def on_run_start(
        self,
        *,
        runner: Any,
        context: Any,
        resume: bool = False,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        tracer = getattr(runner, "tracer", None)
        if tracer is None or not callable(getattr(tracer, "trace_event", None)):
            return

        if not (resume or checkpoint):
            task = self._task_from_context(context)
            files = self._files_from_context(context)
            # File-only turn: the user uploaded files without typing. The
            # transcript row is still persisted (see
            # ``persist_user_message_no_commit``), so the live trace bubble
            # should also fire — otherwise the chip would only show up on
            # next reload via historical replay, mismatching the persist
            # behavior. ``get_display_user_message`` is what handles the
            # "" → "Uploaded file(s)" frontend fallback in the bubble.
            if not task and not files:
                return
            await self._emit_user_message_trace(
                tracer=tracer,
                context=context,
                message=get_display_user_message(context, task or ""),
                files=files,
            )
            # Mark the latest user message (if the runner added one) as
            # traced so a subsequent resume does not re-emit it.
            latest = self._latest_user_message(context)
            if latest is not None:
                self._mark_traced(context, latest)
            return

        # Resume / checkpoint replay: emit any user messages the prior turn
        # did not get to trace. This handles two real scenarios:
        #   1. ``inject_user_message`` was called, the checkpoint was
        #      persisted, but the in-process trace emission was lost
        #      (worker crash between persist and emit).
        #   2. Defensive double-coverage for the continuation flow even
        #      though ``on_user_message_posted`` already covers the happy
        #      path — keeps the chip from disappearing if the callback
        #      somehow didn't fire on the prior worker.
        await self._emit_untraced_user_messages(tracer=tracer, context=context)

    async def on_user_message_posted(
        self,
        *,
        runner: Any,
        context: Any,
        message: Any,
        files: list[dict[str, Any]] | None = None,
    ) -> None:
        """Fire when ``runner.inject_user_message`` lands a fresh user turn.

        ``message`` is the freshly added ``Message`` instance. ``files`` is
        the normalized attachment list the websocket layer received; when
        absent we fall back to ``message.metadata['files']`` (which
        ``inject_user_message`` populates when the caller passes ``files``).
        """
        tracer = getattr(runner, "tracer", None)
        if tracer is None or not callable(getattr(tracer, "trace_event", None)):
            return

        # ``Message.content`` holds the LLM-facing execution text (with
        # any uploaded-files context appended). The chat bubble must
        # render the user-typed string instead, so prefer
        # ``metadata['display_message']`` when the runner stashed one.
        content = getattr(message, "content", None) or ""
        bubble_text = self._display_message_from(message) or content
        resolved_files = files or self._files_from_message(message)
        # File-only continuation: user uploaded files without typing.
        # ``inject_user_message`` still added the (empty-content) Message
        # so the chip survives checkpoints — we mirror that here and let
        # the frontend's ``has_files`` fallback render the bubble.
        if not bubble_text and not resolved_files:
            return
        await self._emit_user_message_trace(
            tracer=tracer,
            context=context,
            message=bubble_text,
            files=resolved_files,
        )
        self._mark_traced(context, message)

    async def on_run_end(
        self, *, runner: Any, context: Any, result: dict[str, Any]
    ) -> None:
        tracer = getattr(runner, "tracer", None)
        if tracer is None or not callable(getattr(tracer, "trace_event", None)):
            return

        execution_id = str(
            result.get("execution_id") or getattr(context, "execution_id", "") or ""
        )
        status = str(result.get("status") or "")
        output = extract_assistant_message(result)
        data: dict[str, Any] = {
            "execution_id": execution_id,
            "status": status or ("completed" if result.get("success") else "failed"),
            "pattern": result.get("pattern"),
        }

        if result.get("success"):
            if output:
                completion_result: dict[str, Any] = {"content": output}
                file_outputs = result.get("file_outputs")
                if file_outputs:
                    completion_result["file_outputs"] = file_outputs
                    completion_result["output"] = output
                await trace_ai_message(tracer, execution_id, output, data)
                await trace_task_completion(
                    tracer,
                    execution_id,
                    result=completion_result,
                    success=True,
                )
            return

        if status in {"interrupted", "waiting_for_user"}:
            # Paused/interrupted executions are resumable control states, not
            # completions. The web trace compatibility layer maps
            # TASK_END_GENERAL to task_completion, so do not emit it here.
            return

        await trace_error(
            tracer,
            execution_id,
            error_type="agent_error",
            error_message=str(result.get("error") or "agent execution failed"),
            data={**data, "context": self._context_payload(context)},
        )

    async def _emit_user_message_trace(
        self,
        *,
        tracer: Any,
        context: Any,
        message: str,
        files: list[dict[str, Any]],
    ) -> None:
        """Emit a user-message trace event with a browser-safe payload.

        Browser-safety contract (rogercloud review): a user-message trace
        event reaches the chat UI / historical-replay client and must not
        carry raw filesystem paths.

        1. We deliberately do NOT attach ``ExecutionContext.to_dict()`` to
           the trace payload. The context dict can contain raw paths under
           ``metadata.request_context.file_info`` and
           ``messages[].metadata.files`` that were not stripped at ingest
           (e.g. when a caller passes raw ``file_info`` straight into
           ``Message.metadata['files']``). Anything internal that needs the
           full context lives in the checkpoint, not the user-facing trace.

        2. ``files`` is funnelled through ``project_file_info_to_chip``
           regardless of how it arrived. This is defence in depth — even
           if a future caller passes raw ``file_info`` with absolute paths
           to ``on_user_message_posted(files=...)``, the projector strips
           anything outside the chip schema (``file_id`` / ``name`` /
           ``size`` / ``type``).
        """
        safe_files = project_file_info_to_chip(files)
        trace_data: dict[str, Any] = {}
        if safe_files:
            # Surface uploaded files at the top level of trace_data so the
            # frontend user-message renderer (which reads ``eventData.files``)
            # can show clickable file chips alongside the user's message.
            trace_data["files"] = safe_files
            trace_data["attachments"] = safe_files
        execution_id = str(getattr(context, "execution_id", "") or "")
        await trace_user_message(tracer, execution_id, message, trace_data)

    async def _emit_untraced_user_messages(self, *, tracer: Any, context: Any) -> None:
        watermark = self._watermark(context)
        pending = self._pending_marker(context)
        # Disambiguate old/pre-PR checkpoints from genuinely-pending turns:
        # a checkpoint that has neither marker is from before this PR (or
        # from a code path that doesn't go through the runner's
        # inject_user_message). Treating it as "everything is untraced"
        # would re-emit every historical user message on resume. The
        # crash-window for newly-injected messages is covered by the
        # pending marker below; long-term per-turn idempotency is tracked
        # in #454.
        if watermark is None and pending is None:
            return

        messages = list(getattr(context, "messages", []) or [])
        # Resolve once outside the loop — every miss inside would otherwise
        # rescan the full message list (O(N^2) in the worst case where many
        # turns lack ``Message.metadata['files']``).
        first_user_idx = self._first_user_message_index(messages)
        for index, message in enumerate(messages):
            if getattr(message, "role", None) != "user":
                continue
            content = getattr(message, "content", None) or ""
            # Same display vs execution split as on_user_message_posted —
            # see the comment there.
            bubble_text = self._display_message_from(message) or content
            ts = self._message_timestamp_iso(message)
            if watermark and ts and ts <= watermark:
                continue
            # Pending marker present without watermark advance: only replay
            # the matching turn. Any other historical turn at this point
            # was already visible to the client on the prior run; skip it
            # so we don't spam re-emissions for everything older than the
            # crash point.
            if pending is not None and ts != pending:
                continue
            files = self._files_from_message(message)
            # For the chronologically first user message we additionally fall
            # back to request_context.file_info — the runner's fresh-start
            # path attaches files to the request_context dict but not to the
            # ``Message`` itself, so a crash-recovery resume needs this
            # fallback to surface chips for the *original* turn.
            if not files and index == first_user_idx:
                files = self._files_from_context(context)
            # File-only message (empty content + non-empty files) is a real
            # turn — the persist layer keeps the row when attachments are
            # present, and the live bubble should match.
            if not bubble_text and not files:
                continue
            await self._emit_user_message_trace(
                tracer=tracer,
                context=context,
                message=bubble_text,
                files=files,
            )
            self._mark_traced(context, message)

    def _watermark(self, context: Any) -> str | None:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get(TRACE_WATERMARK_KEY)
        return value if isinstance(value, str) and value else None

    def _pending_marker(self, context: Any) -> str | None:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get(PENDING_MARKER_KEY)
        return value if isinstance(value, str) and value else None

    def _mark_traced(self, context: Any, message: Any) -> None:
        ts = self._message_timestamp_iso(message)
        if ts is None:
            return
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return
        existing = metadata.get(TRACE_WATERMARK_KEY)
        if not isinstance(existing, str) or ts > existing:
            metadata[TRACE_WATERMARK_KEY] = ts

    def _message_timestamp_iso(self, message: Any) -> str | None:
        ts = getattr(message, "timestamp", None)
        if isinstance(ts, datetime):
            # Normalize to UTC so the watermark's lexicographical comparison
            # is stable even when callers stamp messages with naive datetimes
            # or non-UTC offsets. Without this, an aware ``T12:00:00+00:00``
            # would sort *after* an equivalent naive ``T12:00:00`` and the
            # watermark could let already-traced messages re-emit.
            aware = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
            return aware.astimezone(timezone.utc).isoformat()
        if isinstance(ts, str) and ts:
            return ts
        return None

    def _latest_user_message(self, context: Any) -> Any | None:
        messages = list(getattr(context, "messages", []) or [])
        for message in reversed(messages):
            if getattr(message, "role", None) == "user":
                return message
        return None

    def _first_user_message_index(self, messages: list[Any]) -> int:
        for index, message in enumerate(messages):
            if getattr(message, "role", None) == "user":
                return index
        return -1

    def _files_from_message(self, message: Any) -> list[dict[str, Any]]:
        # Run through the shared projector even though ``inject_user_message``
        # is supposed to hand us already-chip-shaped files. Defense in depth
        # against a caller that drops raw ``file_info`` (with absolute
        # paths) into ``Message.metadata['files']`` directly — the trace
        # event payload reaches the browser, so paths must not leak.
        metadata = getattr(message, "metadata", None)
        if not isinstance(metadata, dict):
            return []
        return project_file_info_to_chip(metadata.get("files"))

    @staticmethod
    def _display_message_from(message: Any) -> str | None:
        """Return ``Message.metadata['display_message']`` if it's a non-empty
        string, else None. The runner sets it when the LLM-facing execution
        text differs from the user-typed bubble text (see
        ``runner.inject_user_message``).
        """
        metadata = getattr(message, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("display_message")
        return value if isinstance(value, str) and value else None

    def _context_payload(self, context: Any) -> dict[str, Any] | None:
        to_dict = getattr(context, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            return dict(payload) if isinstance(payload, dict) else None
        return None

    def _task_from_context(self, context: Any) -> str | None:
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            task = metadata.get("task")
            if isinstance(task, str) and task:
                return task
        messages = getattr(context, "messages", [])
        for message in messages:
            if getattr(message, "role", None) == "user":
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    return content
        return None

    def _files_from_context(self, context: Any) -> list[dict[str, Any]]:
        """Extract uploaded-file chips from the execution context.

        The websocket adapter wraps the raw context dict — including
        ``file_info`` — inside ``metadata["request_context"]`` when starting
        a run. Delegates the projection (and path stripping) to the shared
        ``project_file_info_to_chip`` helper so the chip shape stays
        consistent with the persistence-layer normalization.
        """
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return []
        request_context = metadata.get("request_context")
        if not isinstance(request_context, dict):
            return []
        return project_file_info_to_chip(request_context.get("file_info"))
