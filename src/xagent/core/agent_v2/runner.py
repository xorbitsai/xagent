from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

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
        outbound_message_handler: Any | None = None,
    ) -> None:
        self.agent = agent
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.memory_manager = memory_manager
        self.tracer = tracer
        self.callbacks = callbacks or []
        self.context_manager = context_manager or ContextManager()
        self.workspace_base_dir = workspace_base_dir
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
                context.add_user_message(task)

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
                        "on_run_end", runner=self, context=context, result=normalized
                    )
                    return normalized
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Pattern %s failed", pattern.__class__.__name__)
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
                        "on_run_end", runner=self, context=context, result=normalized
                    )
                    return normalized
                if normalized.get("status") in {"interrupted", "waiting_for_user"}:
                    await self._dispatch_callback(
                        "on_run_end", runner=self, context=context, result=normalized
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
        )

    async def inject_user_message(
        self,
        execution_id: str,
        message: str,
        *,
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

        context.add_user_message(message)
        await self._persist_injected_context(
            execution_id=execution_id,
            context=context,
            label="user_message_injected",
        )
        if request_interrupt:
            self.pause(execution_id, reason=reason or "new user message")
        return context

    async def post_user_message(
        self,
        execution_id: str,
        message: str,
        *,
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
        if metadata:
            context.metadata.update(metadata)
        if task:
            context.metadata.setdefault("task", task)

        memory_session = await self._resolve_memory_session(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )
        if memory_session is not None:
            memory_id, snapshot = memory_session
            context.attach_memory_session(memory_id, snapshot)

        return context

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
