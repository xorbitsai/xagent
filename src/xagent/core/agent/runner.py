from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from ...config import get_compact_threshold_default, get_compact_threshold_ratio
from ..model.intent import enter_goal, exit_goal
from ..workspace import WorkspaceManager
from .context import ContextManager, ExecutionContext
from .result import extract_assistant_message
from .runtime import LLMCallInterrupted, PatternRuntime, load_pattern_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ExecutionControl:
    """In-memory control state for an active execution."""

    runtime: PatternRuntime
    task: str | None


class AgentRunner:
    """Execute an agent by materializing an execution context and invoking patterns."""

    def __init__(
        self,
        agent: Any,
        *,
        workspace_manager: WorkspaceManager | None = None,
        memory_manager: Any | None = None,
        tracer: Any | None = None,
        callbacks: list[Any] | None = None,
        context_manager: ContextManager | None = None,
        workspace_base_dir: str = "workspace",
        scope_segments: tuple[str, ...] = (),
        outbound_message_handler: Any | None = None,
    ) -> None:
        self.agent = agent
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.memory_manager = memory_manager
        self.tracer = tracer
        self.callbacks = callbacks or []
        self.context_manager = context_manager or ContextManager()
        self.workspace_base_dir = workspace_base_dir
        self.scope_segments = scope_segments
        self.outbound_message_handler = outbound_message_handler
        self._active_controls: dict[str, ExecutionControl] = {}

    async def run(
        self,
        task: str | None,
        user_id: str | None = None,
        execution_id: str | None = None,
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        allowed_external_dirs: list[str] | None = None,
        base_dir: str | None = None,
        resume: bool = False,
        checkpoint: dict[str, Any] | None = None,
        runtime: PatternRuntime | None = None,
        interrupt_checker: Any | None = None,
        streaming_handler: Any | None = None,
        extra_tools: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        execution_id = execution_id or str(uuid4())
        checkpoint = checkpoint or (
            await self._load_latest_checkpoint(execution_id) if resume else None
        )
        if task is None:
            task = self._resolve_task(
                task=task,
                checkpoint=checkpoint,
                execution_id=execution_id,
            )
        if checkpoint and isinstance(checkpoint.get("context"), dict):
            context = ExecutionContext.from_dict(checkpoint["context"])
            self.context_manager.set_context(context)
            execution_id = context.execution_id
        else:
            context = await self._build_context(
                task=task,
                execution_id=execution_id,
                user_id=user_id,
                session_id=session_id,
                workspace_id=workspace_id,
                allowed_external_dirs=allowed_external_dirs,
                base_dir=base_dir,
                metadata=metadata,
            )
            for message in initial_messages or []:
                role = str(message.get("role") or "").strip()
                content = str(message.get("content") or "").strip()
                if role and content:
                    context.add_message(role, content)
            if task:
                context.add_user_message(
                    task,
                    metadata=self._initial_user_message_metadata(context),
                )

        runtime = runtime or PatternRuntime(
            tracer=self.tracer,
            execution_id=execution_id,
            interrupt_checker=interrupt_checker,
            outbound_message_handler=self.outbound_message_handler,
        )
        self._active_controls[execution_id] = ExecutionControl(
            runtime=runtime,
            task=task,
        )

        # Establish the user's request as the turn's goal. The "auto" model
        # routes on this rather than on the scaffolded sub-prompt a given LLM
        # call carries; finer units (DAG steps) override it with their own goal.
        goal_token = enter_goal(task)

        await self._dispatch_callback(
            "on_run_start",
            runner=self,
            context=context,
            resume=resume,
            checkpoint=checkpoint,
        )

        try:
            patterns = list(getattr(self.agent, "patterns", []))
            if not patterns:
                result = {
                    "success": False,
                    "error": "Agent has no execution patterns configured.",
                    "execution_id": execution_id,
                    "context": context,
                }
                await self._dispatch_callback(
                    "on_run_end", runner=self, context=context, result=result
                )
                return result

            tools = [*getattr(self.agent, "tools", []), *(extra_tools or [])]
            pattern_errors: list[dict[str, Any]] = []

            try:
                await self._setup_tools(tools, task_id=execution_id)
                for pattern in patterns:
                    load_pattern_checkpoint(pattern, checkpoint)
                    try:
                        result = await pattern.run(
                            **self._build_pattern_kwargs(
                                pattern=pattern,
                                task=task or "",
                                context=context,
                                tools=tools,
                                runtime=runtime,
                                streaming_handler=streaming_handler,
                            )
                        )
                    except LLMCallInterrupted as exc:
                        normalized = {
                            "success": False,
                            "status": "interrupted",
                            "error": str(exc),
                            "execution_id": execution_id,
                            "context": context,
                            "pattern": pattern.__class__.__name__,
                        }
                        await self._dispatch_callback(
                            "on_run_end",
                            runner=self,
                            context=context,
                            result=normalized,
                        )
                        return normalized
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "Pattern %s failed", pattern.__class__.__name__
                        )
                        pattern_errors.append(
                            {
                                "pattern": pattern.__class__.__name__,
                                "error": str(exc),
                                "exception_type": exc.__class__.__name__,
                            }
                        )
                        continue

                    normalized = self._normalize_result(
                        result=result,
                        pattern=pattern,
                        context=context,
                        execution_id=execution_id,
                    )
                    if normalized.get("success"):
                        await self._dispatch_callback(
                            "on_run_end",
                            runner=self,
                            context=context,
                            result=normalized,
                        )
                        return normalized
                    if normalized.get("status") in {"interrupted", "waiting_for_user"}:
                        await self._dispatch_callback(
                            "on_run_end",
                            runner=self,
                            context=context,
                            result=normalized,
                        )
                        return normalized

                    pattern_errors.append(
                        {
                            "pattern": pattern.__class__.__name__,
                            "error": normalized.get(
                                "error", "Pattern failed without a detailed error."
                            ),
                            "result": normalized,
                        }
                    )
            finally:
                await self._teardown_tools(tools, task_id=execution_id)

            if len(pattern_errors) == 1:
                single_result = pattern_errors[0].get("result")
                if isinstance(single_result, dict):
                    await self._dispatch_callback(
                        "on_run_end", runner=self, context=context, result=single_result
                    )
                    return single_result

            result = {
                "success": False,
                "error": f"All {len(patterns)} patterns failed or returned unsuccessful results.",
                "pattern_errors": pattern_errors,
                "patterns_attempted": len(patterns),
                "execution_id": execution_id,
                "context": context,
            }
            await self._dispatch_callback(
                "on_run_end", runner=self, context=context, result=result
            )
            return result
        finally:
            self._active_controls.pop(execution_id, None)
            exit_goal(goal_token)

    def pause(self, execution_id: str, reason: str | None = None) -> bool:
        control = self._active_controls.get(execution_id)
        if control is None:
            return False
        control.runtime.request_interrupt(reason or "paused by runner")
        return True

    def cancel(self, execution_id: str, reason: str | None = None) -> bool:
        control = self._active_controls.get(execution_id)
        if control is None:
            return False
        control.runtime.request_interrupt(reason or "cancelled by runner")
        return True

    async def resume(
        self,
        execution_id: str,
        *,
        task: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        workspace_id: str | None = None,
        allowed_external_dirs: list[str] | None = None,
        base_dir: str | None = None,
        streaming_handler: Any | None = None,
        extra_tools: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        interrupt_checker: Any | None = None,
    ) -> dict[str, Any]:
        checkpoint = await self._load_latest_checkpoint(execution_id)
        resolved_task = self._resolve_task(
            task=task,
            checkpoint=checkpoint,
            execution_id=execution_id,
        )
        return await self.run(
            task=resolved_task,
            user_id=user_id,
            execution_id=execution_id,
            session_id=session_id,
            workspace_id=workspace_id,
            allowed_external_dirs=allowed_external_dirs,
            base_dir=base_dir,
            resume=True,
            checkpoint=checkpoint,
            streaming_handler=streaming_handler,
            extra_tools=extra_tools,
            metadata=metadata,
            interrupt_checker=interrupt_checker,
        )

    async def inject_user_message(
        self,
        execution_id: str,
        message: str | None = None,
        *,
        execution_message: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
        request_interrupt: bool = True,
        reason: str | None = None,
    ) -> ExecutionContext | None:
        context = self.context_manager.get_context(execution_id)
        if context is None:
            checkpoint = await self._load_latest_checkpoint(execution_id)
            if not (
                isinstance(checkpoint, dict)
                and isinstance(checkpoint.get("context"), dict)
            ):
                return None
            context = ExecutionContext.from_dict(checkpoint["context"])
            self.context_manager.set_context(context)

        # Display-vs-execution split: ``execution_message`` is the prompt
        # the agent runtime sees (may be enriched with file refs / system
        # context); ``display_message`` is what the chat bubble should
        # show. Both fall back to ``message`` for legacy callers.
        resolved_execution_message = (
            execution_message if execution_message is not None else message
        )
        if resolved_execution_message is None:
            raise ValueError(
                "inject_user_message requires message or execution_message"
            )
        if display_message is None and message is None:
            raise ValueError(
                "inject_user_message requires display_message when "
                "execution_message is provided without legacy message"
            )
        resolved_display_message = (
            display_message if display_message is not None else message
        )
        requested_turn_id = turn_id.strip() if turn_id and turn_id.strip() else None
        if requested_turn_id is not None:
            for existing in context.messages:
                existing_metadata = getattr(existing, "metadata", None)
                if (
                    getattr(existing, "role", None) != "user"
                    or not isinstance(existing_metadata, dict)
                    or existing_metadata.get("turn_id") != requested_turn_id
                ):
                    continue
                if existing.content != resolved_execution_message:
                    raise ValueError(
                        "turn_id is already associated with a different user message"
                    )
                if request_interrupt:
                    self.pause(execution_id, reason=reason or "new user message")
                return context

        # Attach files + display text to the new Message so they survive
        # checkpoint round-trips: Message.metadata is serialized by
        # ExecutionContext. The on_user_message_posted callback reads
        # display_message back from metadata so the chat bubble shows the
        # user-typed text rather than the LLM-augmented prompt.
        metadata: dict[str, Any] = {"display_message": resolved_display_message}
        if files is not None:
            metadata["files"] = files
        if requested_turn_id is not None:
            metadata["turn_id"] = requested_turn_id
        self._ensure_user_message_turn_id(metadata)

        added = context.add_user_message(resolved_execution_message, metadata=metadata)
        # Set a "this turn is waiting to be traced" pending marker before
        # we persist. The resume catch-up logic uses this to disambiguate
        # an old checkpoint that pre-dates this PR (no watermark, no
        # pending marker — should NOT replay history) from a checkpoint
        # that crashed mid-emit (pending marker present — replay this
        # specific turn). Without this, ``_emit_untraced_user_messages``
        # would treat any missing-watermark checkpoint as "everything
        # untraced" and re-render historical user messages on resume.
        self._set_pending_user_message_marker(context, added)
        # Persist BEFORE emitting the trace so the message is durable even
        # if the trace dispatch fails — the resume path's catch-up logic
        # in TraceEventCallback.on_run_start will replay the marked turn.
        await self._persist_injected_context(
            execution_id=execution_id,
            context=context,
            label="user_message_injected",
        )
        # Snapshot the watermark BEFORE the callback so we can detect a
        # change (see comment below) and persist it.
        watermark_before = self._read_trace_watermark(context)
        traced_turn_ids_before = self._read_traced_turn_ids(context)
        await self._dispatch_callback(
            "on_user_message_posted",
            runner=self,
            context=context,
            message=added,
            files=files,
        )
        # Re-persist when the trace callback advanced the watermark —
        # without this a worker crash between trace emission and the next
        # checkpoint would let the resume path replay the same user_message
        # event because the watermark was still living only in memory.
        # The same persist also clears the pending marker since the trace
        # has now been emitted; doing both in one persist keeps the
        # invariant {pending => never traced yet} on every durable state.
        watermark_after = self._read_trace_watermark(context)
        traced_turn_ids_after = self._read_traced_turn_ids(context)
        if (
            watermark_after and watermark_after != watermark_before
        ) or traced_turn_ids_after != traced_turn_ids_before:
            self._clear_pending_user_message_marker(context)
            await self._persist_injected_context(
                execution_id=execution_id,
                context=context,
                label="user_message_trace_watermark",
            )
        if request_interrupt:
            self.pause(execution_id, reason=reason or "new user message")
        return context

    @staticmethod
    def _read_trace_watermark(context: ExecutionContext) -> str | None:
        """Read the user-message trace watermark off context metadata, if any.

        Kept in-runner so we don't import the tracing module (which would
        create a cycle) — the key is a stable contract spelled out in
        ``core.agent.tracing.TRACE_WATERMARK_KEY``.
        """
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("_user_message_trace_watermark")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _read_traced_turn_ids(context: ExecutionContext) -> tuple[str, ...]:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return ()
        value = metadata.get("_user_message_trace_turn_ids")
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str) and item)

    @staticmethod
    def _set_pending_user_message_marker(
        context: ExecutionContext, message: Any
    ) -> None:
        """Stamp ``_pending_user_message_trace_timestamp`` on context.metadata.

        Mirrored in ``core.agent.tracing.PENDING_MARKER_KEY``; kept in-runner
        to avoid an import cycle. The timestamp is the just-added message's
        normalized ISO-UTC timestamp so the catch-up loop can replay this
        specific turn rather than scanning history.
        """
        ts = AgentRunner._message_iso_timestamp(message)
        if ts is None:
            return
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            metadata["_pending_user_message_trace_timestamp"] = ts
            turn_id = AgentRunner._message_turn_id(message)
            if turn_id:
                metadata["_pending_user_message_trace_turn_id"] = turn_id

    @staticmethod
    def _clear_pending_user_message_marker(context: ExecutionContext) -> None:
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            metadata.pop("_pending_user_message_trace_timestamp", None)
            metadata.pop("_pending_user_message_trace_turn_id", None)

    @staticmethod
    def _message_turn_id(message: Any) -> str | None:
        metadata = getattr(message, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("turn_id")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _message_iso_timestamp(message: Any) -> str | None:
        """ISO-UTC string of ``message.timestamp`` — same normalization the
        tracing module uses for its watermark, kept here to avoid a cycle.
        """
        from datetime import datetime, timezone

        ts = getattr(message, "timestamp", None)
        if isinstance(ts, datetime):
            aware = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
            return aware.astimezone(timezone.utc).isoformat()
        if isinstance(ts, str) and ts:
            return ts
        return None

    async def post_user_message(
        self,
        execution_id: str,
        message: str | None = None,
        *,
        execution_message: str | None = None,
        display_message: str | None = None,
        files: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
        request_interrupt: bool = True,
        reason: str | None = None,
    ) -> ExecutionContext | None:
        """Alias for external callers to inject a user message into an execution.

        `send_message` is an agent-side tool (`agent -> user`).
        `post_user_message` is a runner-side control API (`user/system -> execution`).
        """
        return await self.inject_user_message(
            execution_id,
            message,
            execution_message=execution_message,
            display_message=display_message,
            files=files,
            turn_id=turn_id,
            request_interrupt=request_interrupt,
            reason=reason,
        )

    async def _build_context(
        self,
        *,
        task: str | None,
        execution_id: str,
        user_id: str | None,
        session_id: str | None,
        workspace_id: str | None,
        allowed_external_dirs: list[str] | None,
        base_dir: str | None,
        metadata: dict[str, Any] | None,
    ) -> ExecutionContext:
        workspace = self.workspace_manager.get_or_create_workspace(
            base_dir=base_dir or self.workspace_base_dir,
            task_id=workspace_id or execution_id,
            allowed_external_dirs=allowed_external_dirs,
            scope_segments=self.scope_segments,
        )
        if inspect.isawaitable(workspace):
            workspace = await workspace
        context = self.context_manager.create_context(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
            system_prompt=getattr(self.agent, "system_prompt", None),
            workspace_id=workspace.id,
            workspace_path=str(workspace.workspace_dir),
            cwd=str(workspace.workspace_dir),
            workspace_state=self._workspace_state(workspace),
        )
        # Snapshotted at task start. On resume the context (and this threshold)
        # is restored verbatim from the checkpoint, so a context-window or ratio
        # change made after checkpointing only affects newly started tasks.
        context.compact_config.threshold = self._resolve_compact_threshold()
        if metadata:
            context.metadata.update(metadata)
        if task:
            context.metadata.setdefault("task", task)
        request_context = (
            metadata.get("request_context") if isinstance(metadata, dict) else None
        )
        if isinstance(request_context, dict):
            self._apply_request_context(context, request_context)

        memory_session = await self._resolve_memory_session(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )
        if memory_session is not None:
            memory_id, snapshot = memory_session
            context.attach_memory_session(memory_id, snapshot)

        return context

    def _resolve_compact_threshold(self) -> int:
        """Derive the context-compaction threshold from the model's context window.

        When the model declares a context window, compact at
        ``context_window * ratio`` tokens; otherwise fall back to the configured
        default (preserving the historical 32000 behaviour).
        """
        llm = getattr(self.agent, "llm", None)
        context_window = getattr(llm, "context_window", None)
        # context_window is typed int | None end to end (DB Integer -> Pydantic
        # Optional[int]); bool is not a valid value, so a plain int check suffices.
        if isinstance(context_window, int) and context_window > 0:
            return max(1, int(context_window * get_compact_threshold_ratio()))
        return get_compact_threshold_default()

    def _initial_user_message_metadata(
        self, context: ExecutionContext
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        context_metadata = (
            context.metadata if isinstance(context.metadata, dict) else {}
        )
        request_context = context_metadata.get("request_context")

        candidates = []
        if isinstance(request_context, dict):
            candidates.append(request_context)
        candidates.append(context_metadata)

        for candidate in candidates:
            turn_id = candidate.get("turn_id")
            if isinstance(turn_id, str) and turn_id:
                metadata["turn_id"] = turn_id
                break
        self._ensure_user_message_turn_id(metadata)

        for candidate in candidates:
            if "display_message" in candidate:
                display_message = candidate.get("display_message")
                metadata["display_message"] = (
                    display_message if isinstance(display_message, str) else ""
                )
                break
            if "display_user_message" in candidate:
                display_message = candidate.get("display_user_message")
                metadata["display_message"] = (
                    display_message if isinstance(display_message, str) else ""
                )
                break

        for candidate in candidates:
            files = candidate.get("files")
            if isinstance(files, list):
                metadata["files"] = files
                break
            attachments = candidate.get("attachments")
            if isinstance(attachments, list):
                metadata["files"] = attachments
                break

        return metadata

    @staticmethod
    def _ensure_user_message_turn_id(metadata: dict[str, Any]) -> str:
        value = metadata.get("turn_id")
        if isinstance(value, str) and value:
            return value
        turn_id = str(uuid4())
        metadata["turn_id"] = turn_id
        return turn_id

    def _apply_request_context(
        self,
        context: ExecutionContext,
        request_context: dict[str, Any],
    ) -> None:
        system_prompt = request_context.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            prompt = system_prompt.strip()
            if context.system_prompt and context.system_prompt.strip():
                existing = context.system_prompt.strip()
                if prompt not in existing:
                    context.system_prompt = f"{existing}\n\n{prompt}"
            else:
                context.system_prompt = prompt

        for key, value in request_context.items():
            if key == "system_prompt":
                continue
            context.metadata[key] = value

    def _resolve_task(
        self,
        *,
        task: str | None,
        checkpoint: dict[str, Any] | None,
        execution_id: str,
    ) -> str | None:
        if task:
            return task

        if isinstance(checkpoint, dict):
            context_payload = checkpoint.get("context")
            if isinstance(context_payload, dict):
                metadata = context_payload.get("metadata")
                if isinstance(metadata, dict):
                    saved_task = metadata.get("task")
                    if isinstance(saved_task, str) and saved_task:
                        return saved_task
                messages = context_payload.get("messages")
                if isinstance(messages, list):
                    for message in reversed(messages):
                        if (
                            isinstance(message, dict)
                            and message.get("role") == "user"
                            and isinstance(message.get("content"), str)
                            and message["content"]
                        ):
                            content = cast(str, message["content"])
                            return content

        control = self._active_controls.get(execution_id)
        if control is not None:
            return control.task

        return None

    def _workspace_state(self, workspace: Any) -> dict[str, Any]:
        state: dict[str, Any] = {
            "input_dir": str(getattr(workspace, "input_dir", "")),
            "output_dir": str(getattr(workspace, "output_dir", "")),
            "temp_dir": str(getattr(workspace, "temp_dir", "")),
        }
        allowed_dirs = getattr(workspace, "allowed_external_dirs", None)
        if allowed_dirs is not None:
            state["allowed_external_dirs"] = [str(path) for path in allowed_dirs]
        return state

    async def _resolve_memory_session(
        self,
        *,
        execution_id: str,
        user_id: str | None,
        session_id: str | None,
    ) -> tuple[str | None, dict[str, Any] | None] | None:
        if self.memory_manager is None:
            return None

        for method_name in (
            "get_or_create_session",
            "create_session",
            "get_session",
            "load_session",
        ):
            method = getattr(self.memory_manager, method_name, None)
            if method is None:
                continue
            payload = self._call_with_supported_kwargs(
                method,
                execution_id=execution_id,
                user_id=user_id,
                session_id=session_id,
            )
            if inspect.isawaitable(payload):
                payload = await payload
            return self._normalize_memory_session(payload, session_id=session_id)

        return None

    def _normalize_memory_session(
        self,
        payload: Any,
        *,
        session_id: str | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if payload is None:
            return session_id, None
        if isinstance(payload, tuple) and len(payload) == 2:
            return payload[0], payload[1]
        if isinstance(payload, str):
            return payload, None
        if isinstance(payload, dict):
            resolved_id = payload.get("session_id") or payload.get("id") or session_id
            snapshot = payload.get("snapshot")
            if snapshot is None:
                snapshot = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"session_id", "id"}
                }
            return resolved_id, snapshot

        resolved_id = (
            getattr(payload, "session_id", None)
            or getattr(payload, "id", None)
            or session_id
        )
        snapshot = getattr(payload, "snapshot", None)
        if snapshot is None and hasattr(payload, "to_dict"):
            snapshot = payload.to_dict()
        return resolved_id, snapshot

    def _build_pattern_kwargs(
        self,
        *,
        pattern: Any,
        task: str,
        context: ExecutionContext,
        tools: list[Any],
        runtime: PatternRuntime,
        streaming_handler: Any | None,
    ) -> dict[str, Any]:
        return self._call_signature_kwargs(
            pattern.run,
            agent=self.agent,
            task=task,
            context=context,
            llm=getattr(self.agent, "llm", None),
            compact_llm=getattr(self.agent, "compact_llm", None),
            tools=tools,
            tracer=self.tracer,
            runtime=runtime,
            callbacks=self.callbacks,
            streaming_handler=streaming_handler,
            memory_store=getattr(self.agent, "memory_store", None),
            memory_similarity_threshold=getattr(
                self.agent, "memory_similarity_threshold", None
            ),
            skill_manager=getattr(self.agent, "skill_manager", None),
            allowed_skills=getattr(self.agent, "allowed_skills", None),
        )

    async def _setup_tools(self, tools: list[Any], *, task_id: str) -> None:
        for tool in tools:
            setup = getattr(tool, "setup", None)
            if not callable(setup):
                continue
            result = setup(task_id=task_id)
            if inspect.isawaitable(result):
                await result

    async def _teardown_tools(self, tools: list[Any], *, task_id: str) -> None:
        for tool in reversed(tools):
            teardown = getattr(tool, "teardown", None)
            if not callable(teardown):
                continue
            try:
                result = teardown(task_id=task_id)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Tool teardown failed for %s", getattr(tool, "name", tool)
                )

    async def _load_latest_checkpoint(
        self,
        execution_id: str,
    ) -> dict[str, Any] | None:
        if self.tracer is None:
            return None

        for method_name in (
            "load_latest_checkpoint",
            "get_latest_checkpoint",
            "latest_checkpoint",
        ):
            method = getattr(self.tracer, method_name, None)
            if not callable(method):
                continue
            payload = method(execution_id)
            if inspect.isawaitable(payload):
                payload = await payload
            if isinstance(payload, dict):
                return payload
        return None

    async def _persist_injected_context(
        self,
        *,
        execution_id: str,
        context: ExecutionContext,
        label: str,
    ) -> None:
        if self.tracer is None:
            return

        control = self._active_controls.get(execution_id)
        baseline = (
            control.runtime.last_checkpoint
            if control is not None and control.runtime.last_checkpoint is not None
            else await self._load_latest_checkpoint(execution_id)
        )
        payload = dict(baseline or {})
        payload.update(
            {
                "type": "checkpoint",
                "label": label,
                "execution_id": execution_id,
                "context": context.to_dict(),
            }
        )

        checkpoint = getattr(self.tracer, "checkpoint", None)
        if callable(checkpoint):
            result = checkpoint(**payload)
            if inspect.isawaitable(result):
                await result
            return

        write_checkpoint = getattr(self.tracer, "write_checkpoint", None)
        if callable(write_checkpoint):
            result = write_checkpoint(payload)
            if inspect.isawaitable(result):
                await result
            return

    def _normalize_result(
        self,
        *,
        result: Any,
        pattern: Any,
        context: ExecutionContext,
        execution_id: str,
    ) -> dict[str, Any]:
        if isinstance(result, dict):
            normalized = dict(result)
        else:
            normalized = {"success": True, "output": result}

        normalized.setdefault("success", True)
        normalized.setdefault("execution_id", execution_id)
        normalized.setdefault("context", context)
        normalized.setdefault("pattern", pattern.__class__.__name__)

        assistant_message = extract_assistant_message(normalized)
        if assistant_message:
            for key in ("response", "answer", "output", "content", "message"):
                if isinstance(normalized.get(key), str):
                    normalized[key] = assistant_message
        if assistant_message and not self._has_assistant_message(
            context, assistant_message
        ):
            context.add_assistant_message(assistant_message)

        return normalized

    def _has_assistant_message(self, context: ExecutionContext, content: str) -> bool:
        return any(
            message.role == "assistant" and message.content == content
            for message in context.messages
        )

    async def _dispatch_callback(self, event: str, **payload: Any) -> None:
        for callback in self.callbacks:
            handler = getattr(callback, event, None)
            if handler is None:
                continue
            maybe_coroutine = handler(**payload)
            if inspect.isawaitable(maybe_coroutine):
                await maybe_coroutine

    def _call_with_supported_kwargs(self, fn: Any, **kwargs: Any) -> Any:
        return fn(**self._call_signature_kwargs(fn, **kwargs))

    def _call_signature_kwargs(self, fn: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return kwargs
        parameters = signature.parameters.values()
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
            return kwargs
        return {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
