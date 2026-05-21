from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, TraceEventCallback
from xagent.core.agent.tracing import PENDING_MARKER_KEY


def _stamp_pending(context: ExecutionContext) -> None:
    """Mark the most-recently added user message as 'pending trace emit'.

    Mirrors what ``runner.inject_user_message`` does just before it
    persists the injected-message checkpoint. The catch-up loop on
    resume requires either a watermark or this marker before it will
    replay anything — otherwise it would re-emit history on pre-PR
    checkpoints. See ``tracing.PENDING_MARKER_KEY`` for the contract.
    """
    callback = TraceEventCallback()
    latest = callback._latest_user_message(context)
    if latest is None:
        return
    ts = callback._message_timestamp_iso(latest)
    if ts:
        context.metadata[PENDING_MARKER_KEY] = ts


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(
        self,
        event_type: Any,
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        self.events.append(
            {
                "event_type": getattr(event_type, "value", str(event_type)),
                "task_id": task_id,
                "data": data or {},
            }
        )
        return str(len(self.events))


@pytest.mark.asyncio
async def test_trace_callback_success_emits_user_assistant_and_completion() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "Write summary"

    await callback.on_run_start(runner=runner, context=context)
    await callback.on_run_end(
        runner=runner,
        context=context,
        result={"success": True, "execution_id": "exec-trace", "answer": "Done"},
    )

    assert [event["event_type"] for event in tracer.events] == [
        "task_start_message",
        "task_end_message",
        "task_end_general",
    ]
    assert tracer.events[1]["data"]["content"] == "Done"


@pytest.mark.asyncio
async def test_trace_callback_uses_display_user_message_when_present() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "Read file\n\n## UPLOADED FILES\nfile_id=file-123"
    context.metadata["request_context"] = {
        "display_user_message": "Read file",
    }

    await callback.on_run_start(runner=runner, context=context)

    assert tracer.events[0]["data"]["message"] == "Read file"


@pytest.mark.asyncio
async def test_trace_callback_prefers_latest_message_display_metadata() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "Original task"
    context.metadata["display_user_message"] = "Original task"
    context.add_user_message(
        "Read file\n\n## UPLOADED FILES\nfile_id=file-123",
        metadata={"display_message": "Read file"},
    )

    await callback.on_run_start(runner=runner, context=context)

    assert tracer.events[0]["data"]["message"] == "Read file"


@pytest.mark.asyncio
async def test_trace_callback_failed_run_emits_error() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")

    await callback.on_run_end(
        runner=runner,
        context=context,
        result={
            "success": False,
            "execution_id": "exec-trace",
            "error": "failed",
        },
    )

    assert tracer.events[0]["event_type"] == "task_error_general"
    assert tracer.events[0]["data"]["error_message"] == "failed"


@pytest.mark.asyncio
async def test_trace_callback_resume_does_not_duplicate_task_start() -> None:
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "Resume task"

    await callback.on_run_start(runner=runner, context=context, resume=True)
    await callback.on_run_start(
        runner=runner, context=context, checkpoint={"context": {}}
    )

    assert tracer.events == []


@pytest.mark.asyncio
async def test_trace_callback_no_tracer_is_noop() -> None:
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=None)
    context = ExecutionContext(execution_id="exec-trace")

    await callback.on_run_start(runner=runner, context=context)
    await callback.on_run_end(
        runner=runner,
        context=context,
        result={"success": True, "output": "Done"},
    )


@pytest.mark.asyncio
async def test_trace_callback_surfaces_uploaded_files_for_chip_rendering() -> None:
    """On run start, file_info from request_context must flow to trace_data.files
    so the frontend can render attachment chips alongside the user bubble."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-trace")
    context.metadata["task"] = "can u generate this?"
    context.metadata["request_context"] = {
        "uploaded_files": ["/abs/path/Q1.xlsx"],
        "file_info": [
            {
                "file_id": "6cdc124b-d758-47e3-9871-284e1c90a98a",
                "name": "normalized.xlsx",
                "original_name": "Q1 Report.xlsx",
                "size": 291953,
                "type": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
                "path": "/abs/leak/should/be/stripped.xlsx",
            }
        ],
    }

    await callback.on_run_start(runner=runner, context=context)

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    assert data["files"] == [
        {
            "file_id": "6cdc124b-d758-47e3-9871-284e1c90a98a",
            "name": "Q1 Report.xlsx",
            "size": 291953,
            "type": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        }
    ]
    assert data["attachments"] == data["files"]
    for f in data["files"]:
        assert "path" not in f


@pytest.mark.asyncio
async def test_on_user_message_posted_emits_trace_event_with_files() -> None:
    """When the websocket calls ``post_user_message`` with attachments, the
    runner's ``on_user_message_posted`` callback must emit a user_message
    trace event with the files surfaced at the top level."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-cont")
    context.metadata["task"] = "Original task"
    files = [
        {
            "file_id": "fid-cont-1",
            "name": "follow-up.pdf",
            "size": 4096,
            "type": "application/pdf",
        }
    ]
    new_message = context.add_user_message(
        "Follow-up question with a file.",
        metadata={"files": files},
    )

    await callback.on_user_message_posted(
        runner=runner,
        context=context,
        message=new_message,
        files=files,
    )

    assert len(tracer.events) == 1
    event = tracer.events[0]
    assert event["event_type"] == "task_start_message"
    assert event["data"]["message"] == "Follow-up question with a file."
    assert event["data"]["files"] == files
    assert event["data"]["attachments"] == files


@pytest.mark.asyncio
async def test_on_user_message_posted_prevents_resume_from_duplicating() -> None:
    """After ``on_user_message_posted`` fires, a subsequent resume must NOT
    re-emit the same user_message trace event — the watermark stored on
    ``context.metadata`` is the claim ticket that prevents duplication."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-cont")
    context.metadata["task"] = "Original task"
    msg = context.add_user_message("Follow-up.", metadata={"files": []})

    await callback.on_user_message_posted(runner=runner, context=context, message=msg)
    await callback.on_run_start(runner=runner, context=context, resume=True)

    assert len(tracer.events) == 1  # not two


@pytest.mark.asyncio
async def test_on_run_start_resume_emits_untraced_user_message() -> None:
    """If a checkpoint contains a user message whose trace event was never
    emitted (e.g., worker crashed between persist and emit), the resume's
    ``on_run_start`` must replay it so the chip still shows up."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-recovery")
    context.metadata["task"] = "Original task"
    files = [
        {
            "file_id": "fid-recover",
            "name": "recover.csv",
            "size": 256,
            "type": "text/csv",
        }
    ]
    context.add_user_message("Continue from here.", metadata={"files": files})
    # Simulate the runner: pending marker stamped just before the
    # crashed checkpoint. Catch-up will only replay turns whose ts
    # matches this marker, not all history.
    _stamp_pending(context)

    await callback.on_run_start(
        runner=runner,
        context=context,
        resume=True,
        checkpoint={"context": context.to_dict()},
    )

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    assert data["message"] == "Continue from here."
    assert data["files"] == files


@pytest.mark.asyncio
async def test_on_run_start_resume_skips_pre_pr_checkpoint_history() -> None:
    """Pre-PR checkpoints (created before the chip-attachments PR landed)
    have neither the trace watermark nor the pending marker on
    ``context.metadata``. Resume catch-up must NOT replay every historical
    user message in that case — otherwise it would re-render bubbles the
    client already saw on the prior worker.
    """
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-pre-pr")
    context.metadata["task"] = "Old task"
    context.add_user_message("First historical turn.")
    context.add_user_message("Second historical turn.")
    # Crucially: no _user_message_trace_watermark, no
    # _pending_user_message_trace_timestamp — this is what an
    # already-running task looks like on first resume after upgrade.

    await callback.on_run_start(runner=runner, context=context, resume=True)
    await callback.on_run_start(
        runner=runner, context=context, checkpoint={"context": context.to_dict()}
    )

    assert tracer.events == []


@pytest.mark.asyncio
async def test_on_run_start_resume_with_pending_marker_replays_only_pending_turn() -> (
    None
):
    """When a checkpoint carries a pending marker but no watermark advance
    (the runner persisted the injected user message and then crashed
    before the trace was emitted), catch-up should fire exactly once for
    the matching turn — not for any older history that lacks the marker.
    """
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-pending-replay")
    context.metadata["task"] = "Original task"
    context.add_user_message("Earlier historical turn.")
    pending_msg = context.add_user_message(
        "The turn that crashed mid-emit.",
        metadata={"files": [{"file_id": "fid-crash", "name": "c.txt"}]},
    )
    # Pending marker points at the second message — only it should replay.
    pending_ts = callback._message_timestamp_iso(pending_msg)
    assert pending_ts is not None
    context.metadata[PENDING_MARKER_KEY] = pending_ts

    await callback.on_run_start(runner=runner, context=context, resume=True)

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    assert data["message"] == "The turn that crashed mid-emit."
    assert data["files"][0]["file_id"] == "fid-crash"


@pytest.mark.asyncio
async def test_on_run_start_resume_skips_already_traced_user_messages() -> None:
    """A pure resume (no new user message since the watermark) must NOT
    emit anything — protects scenario where user clicks "Resume" on a
    paused task without sending a new message."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-pure-resume")
    context.metadata["task"] = "Original task"
    msg = context.add_user_message("Original turn.")
    callback._mark_traced(context, msg)

    await callback.on_run_start(runner=runner, context=context, resume=True)
    await callback.on_run_start(
        runner=runner, context=context, checkpoint={"context": context.to_dict()}
    )

    assert tracer.events == []


@pytest.mark.asyncio
async def test_on_run_start_fresh_emits_for_file_only_initial_turn() -> None:
    """User uploaded files without typing on the first turn — the live
    bubble must still fire (the transcript row is persisted by
    ``persist_user_message_no_commit`` when attachments are present, so the
    trace event should match)."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-file-only-fresh")
    # No ``metadata["task"]`` and no user message in context — only files.
    context.metadata["request_context"] = {
        "file_info": [
            {
                "file_id": "fid-only-upload",
                "name": "report.pdf",
                "size": 1024,
                "type": "application/pdf",
            }
        ]
    }

    await callback.on_run_start(runner=runner, context=context)

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    assert data["files"][0]["file_id"] == "fid-only-upload"


@pytest.mark.asyncio
async def test_on_user_message_posted_emits_for_file_only_continuation() -> None:
    """Continuation where the user only attaches files (no new text) must
    still emit a trace event so the live chip lands — mirrors the
    persistence layer, which keeps file-only rows."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-file-only-cont")
    context.metadata["task"] = "Original task"
    files = [{"file_id": "fid-only", "name": "x.pdf"}]
    new_msg = context.add_user_message("", metadata={"files": files})

    await callback.on_user_message_posted(
        runner=runner, context=context, message=new_msg, files=files
    )

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    assert data["message"] == ""
    # The trace funnels files through ``project_file_info_to_chip`` for
    # browser-safety (see _emit_user_message_trace); ``size``/``type`` are
    # filled with ``None`` when missing so the chip schema stays uniform.
    assert data["files"] == [
        {"file_id": "fid-only", "name": "x.pdf", "size": None, "type": None}
    ]


@pytest.mark.asyncio
async def test_emit_user_message_trace_strips_absolute_paths_from_files() -> None:
    """Browser-safety contract: even if a caller hands the callback raw
    ``file_info`` (with absolute ``path`` keys), the emitted trace payload
    must not contain that path. The trace event reaches the chat UI /
    historical-replay client, which is not allowed to see runtime paths.
    """
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-path-leak")
    raw_files = [
        {
            "file_id": "fid-1",
            "name": "doc.pdf",
            "size": 2048,
            "type": "application/pdf",
            "path": "/abs/secret/doc.pdf",
            "workspace_path": "/workspace/input/doc.pdf",
        }
    ]
    new_msg = context.add_user_message("hello", metadata={"files": raw_files})

    await callback.on_user_message_posted(
        runner=runner, context=context, message=new_msg, files=raw_files
    )

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    # No raw context payload — the full ExecutionContext.to_dict() can
    # also carry path-bearing metadata, so we don't include it.
    assert "context" not in data
    # Files at the top level are projected to the chip schema (no
    # ``path``/``workspace_path``).
    for chip in data["files"]:
        assert "path" not in chip
        assert "workspace_path" not in chip
    assert data["files"] == [
        {
            "file_id": "fid-1",
            "name": "doc.pdf",
            "size": 2048,
            "type": "application/pdf",
        }
    ]
    assert data["attachments"] == data["files"]


@pytest.mark.asyncio
async def test_on_user_message_posted_still_skips_truly_empty_turn() -> None:
    """Regression guard: when there's neither text nor files, the callback
    must stay silent — otherwise an accidental empty inject would emit a
    blank bubble."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-empty")
    empty_msg = context.add_user_message("")

    await callback.on_user_message_posted(
        runner=runner, context=context, message=empty_msg, files=None
    )

    assert tracer.events == []


@pytest.mark.asyncio
async def test_emit_untraced_picks_up_file_only_message_on_resume() -> None:
    """Crash-recovery: a checkpoint with a file-only user message (empty
    content + files in Message.metadata) must surface its chip on resume."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-file-only-recover")
    context.metadata["task"] = "Original task"
    context.add_user_message("Original turn.")
    callback._mark_traced(context, context.messages[-1])
    # Second turn: file-only, never traced before the crash.
    files = [{"file_id": "fid-rec", "name": "rec.csv"}]
    context.add_user_message("", metadata={"files": files})

    await callback.on_run_start(runner=runner, context=context, resume=True)

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    # ``_files_from_message`` runs through ``project_file_info_to_chip``
    # for defense-in-depth, which canonicalizes the chip shape (size/type
    # default to ``None`` when absent in the source).
    assert data["files"] == [
        {"file_id": "fid-rec", "name": "rec.csv", "size": None, "type": None}
    ]


@pytest.mark.asyncio
async def test_files_from_message_strips_paths_from_unprojected_metadata() -> None:
    """Defense in depth: if a caller drops raw ``file_info`` (with absolute
    paths) into ``Message.metadata['files']``, the trace callback must still
    project it through ``project_file_info_to_chip`` so paths don't reach
    the browser. ``inject_user_message`` is *supposed* to hand us
    chip-shaped files, but we don't want to rely on every caller doing
    the right thing for a security-relevant field."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-path-leak")
    context.metadata["task"] = "Original task"
    # Simulate a caller bypassing the websocket-side normalization.
    raw_file_info = [
        {
            "file_id": "fid",
            "name": "x.txt",
            "path": "/abs/secret/should/not/leak.txt",
            "extra_internal_field": "should-not-leak",
        }
    ]
    msg = context.add_user_message("hi", metadata={"files": raw_file_info})

    await callback.on_user_message_posted(runner=runner, context=context, message=msg)

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    assert data["files"] == [
        {"file_id": "fid", "name": "x.txt", "size": None, "type": None}
    ]
    for entry in data["files"]:
        assert "path" not in entry
        assert "extra_internal_field" not in entry


def test_message_timestamp_iso_normalizes_naive_to_utc() -> None:
    """Watermark uses ISO-string lexicographical comparison — naive and
    aware datetimes for the same wall-clock instant must produce the same
    sort key, otherwise a checkpoint with naive timestamps could let an
    already-traced message re-emit on resume."""
    from datetime import datetime, timezone

    callback = TraceEventCallback()
    naive = SimpleNamespace(timestamp=datetime(2026, 5, 18, 12, 0, 0))
    aware_utc = SimpleNamespace(
        timestamp=datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    )
    assert callback._message_timestamp_iso(naive) == callback._message_timestamp_iso(
        aware_utc
    )


@pytest.mark.asyncio
async def test_on_run_start_resume_falls_back_to_request_context_for_initial_files() -> (
    None
):
    """Crash-recovery for the very first turn: runner attaches the initial
    user message but didn't propagate request_context.file_info into
    Message.metadata. The resume catch-up surfaces chips by falling back
    to context.metadata.request_context.file_info for that turn."""
    tracer = TraceRecorder()
    callback = TraceEventCallback()
    runner = SimpleNamespace(tracer=tracer)
    context = ExecutionContext(execution_id="exec-first-recovery")
    context.metadata["task"] = "Initial task"
    context.metadata["request_context"] = {
        "file_info": [
            {
                "file_id": "fid-initial",
                "name": "initial.xlsx",
                "original_name": "initial.xlsx",
                "size": 1024,
                "type": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
            }
        ]
    }
    context.add_user_message("Initial task")
    # Same crash-recovery framing: stamp the pending marker so catch-up
    # knows this is the turn to replay.
    _stamp_pending(context)

    await callback.on_run_start(runner=runner, context=context, resume=True)

    assert len(tracer.events) == 1
    data = tracer.events[0]["data"]
    assert data["files"][0]["file_id"] == "fid-initial"
