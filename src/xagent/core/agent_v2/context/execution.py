from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from .components import (
    COMPONENT_LOADERS,
    ExecutionComponent,
    GenericComponent,
    MemoryComponent,
    WorkspaceComponent,
    clone_component,
)
from .enrichment import MEMORY_CONTEXT_METADATA_KEY, SKILL_CONTEXT_METADATA_KEY
from .message import LLMCallRecord, Message

READ_FILE_CONTEXT_LIMIT = 12_000
WRITE_FILE_CONTENT_PREVIEW_LIMIT = 400


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MergeStrategy(str, Enum):
    """Strategies for merging multiple execution contexts."""

    CHRONOLOGICAL = "chronological"
    TOPOLOGICAL = "topological"
    PREFER_FIRST = "prefer_first"


@dataclass
class CompactConfig:
    """Compaction policy for message history."""

    enabled: bool = True
    threshold: int = 8000
    strategy: str = "truncate"
    max_messages: int = 20


@dataclass
class CompactResult:
    """Result returned by context compaction."""

    compacted: bool
    original_count: int
    final_count: int
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Execution state plus pluggable runtime components."""

    execution_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    session_id: str | None = None
    components: dict[str, ExecutionComponent] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    compact_config: CompactConfig = field(default_factory=CompactConfig)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.components.setdefault("workspace", WorkspaceComponent())
        self.components.setdefault("memory", MemoryComponent())

    def get_component(self, name: str) -> ExecutionComponent | None:
        return self.components.get(name)

    def set_component(self, name: str, component: ExecutionComponent) -> None:
        self.components[name] = component

    def _workspace_component(self) -> WorkspaceComponent:
        component = self.components.get("workspace")
        if not isinstance(component, WorkspaceComponent):
            component = WorkspaceComponent()
            self.components["workspace"] = component
        return component

    def _memory_component(self) -> MemoryComponent:
        component = self.components.get("memory")
        if not isinstance(component, MemoryComponent):
            component = MemoryComponent()
            self.components["memory"] = component
        return component

    @property
    def workspace_id(self) -> str | None:
        return self._workspace_component().workspace_id

    @property
    def workspace_path(self) -> str | None:
        return self._workspace_component().workspace_path

    @property
    def cwd(self) -> str | None:
        return self._workspace_component().cwd

    @property
    def workspace_state(self) -> dict[str, Any]:
        return self._workspace_component().state

    @property
    def memory_session_id(self) -> str | None:
        return self._memory_component().session_id

    @property
    def memory_snapshot(self) -> dict[str, Any] | None:
        return self._memory_component().snapshot

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        return message

    def add_user_message(self, content: str, **kwargs: Any) -> Message:
        return self.add_message("user", content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs: Any) -> Message:
        if kwargs.get("tool_calls"):
            kwargs["tool_calls"] = self._sanitize_tool_calls_for_context(
                kwargs["tool_calls"]
            )
        return self.add_message("assistant", content, **kwargs)

    def add_system_message(self, content: str, **kwargs: Any) -> Message:
        return self.add_message("system", content, **kwargs)

    def add_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
    ) -> Message:
        context_result = self._sanitize_tool_result_for_context(tool_name, result)
        content = self._format_tool_result(tool_name, context_result)
        metadata = {
            "tool_name": tool_name,
            "raw_result": context_result,
            "workspace_id": self.workspace_id,
            "workspace_path": self.workspace_path,
            "cwd": self.cwd,
            "memory_session_id": self.memory_session_id,
        }
        return self.add_message(
            "tool",
            content,
            tool_call_id=tool_call_id,
            metadata=metadata,
        )

    def attach_workspace(
        self,
        workspace_id: str | None,
        workspace_path: str | None,
        cwd: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        workspace = self._workspace_component()
        workspace.workspace_id = workspace_id
        workspace.workspace_path = workspace_path
        workspace.cwd = cwd
        if state:
            workspace.state.update(state)

    def attach_memory_session(
        self,
        session_id: str | None,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        memory = self._memory_component()
        memory.session_id = session_id
        memory.snapshot = snapshot

    def _format_tool_result(self, tool_name: str, result: Any) -> str:
        if isinstance(result, dict):
            formatted = result.get("output", result)
        else:
            formatted = result
        return f"Tool {tool_name} returned: {formatted}"

    def _sanitize_tool_result_for_context(self, tool_name: str, result: Any) -> Any:
        if tool_name != "read_file" or not isinstance(result, str):
            return result
        if self._looks_like_binary_text(result):
            return {
                "content_omitted": True,
                "reason": "read_file returned binary-like content",
                "original_chars": len(result),
            }
        if len(result) <= READ_FILE_CONTEXT_LIMIT:
            return result
        return {
            "content_preview": result[:READ_FILE_CONTEXT_LIMIT],
            "content_truncated": True,
            "original_chars": len(result),
        }

    def _sanitize_tool_calls_for_context(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            copied = dict(tool_call)
            function = copied.get("function")
            if isinstance(function, dict):
                function_copy = dict(function)
                if function_copy.get("name") == "write_file":
                    function_copy["arguments"] = self._sanitize_write_file_arguments(
                        function_copy.get("arguments")
                    )
                copied["function"] = function_copy
            elif copied.get("name") == "write_file" and isinstance(
                copied.get("args"), dict
            ):
                copied["args"] = self._sanitize_write_file_args_dict(copied["args"])
            sanitized.append(copied)
        return sanitized

    def _sanitize_write_file_arguments(self, arguments: Any) -> Any:
        if not isinstance(arguments, str):
            return arguments
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return arguments
        if not isinstance(parsed, dict):
            return arguments
        return json.dumps(
            self._sanitize_write_file_args_dict(parsed),
            ensure_ascii=False,
        )

    def _sanitize_write_file_args_dict(self, args: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(args)
        content = sanitized.get("content")
        if not isinstance(content, str):
            return sanitized
        sanitized["content"] = (
            f"[omitted from LLM context: write_file content, {len(content)} chars]"
        )
        if content:
            sanitized["content_preview"] = content[:WRITE_FILE_CONTENT_PREVIEW_LIMIT]
            sanitized["content_truncated"] = (
                len(content) > WRITE_FILE_CONTENT_PREVIEW_LIMIT
            )
        return sanitized

    def _looks_like_binary_text(self, value: str) -> bool:
        if "\x00" in value:
            return True
        sample = value[:4096]
        if not sample:
            return False
        allowed_controls = {"\n", "\r", "\t"}
        control_count = sum(
            1 for char in sample if ord(char) < 32 and char not in allowed_controls
        )
        return control_count / len(sample) > 0.05

    def get_messages_for_llm(
        self,
        include_system: bool = True,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system_parts: list[str] = []
        if include_system and self.system_prompt:
            system_parts.append(self.system_prompt)
        if include_system:
            system_parts.append(self._system_context())

        visible_messages = [message for message in self.messages if not message.hidden]
        if max_tokens:
            visible_messages = self._truncate_by_tokens(visible_messages, max_tokens)
        visible_messages = self._sanitize_tool_message_pairs(visible_messages)

        for message in visible_messages:
            message_dict = message.to_dict()
            waiting_response = message.metadata.get("response_to_waiting_for_user")
            if message_dict.get("role") == "user" and isinstance(
                waiting_response, dict
            ):
                question = str(waiting_response.get("question") or "").strip()
                answer = str(message_dict.get("content") or "").strip()
                message_dict["content"] = (
                    "This user message is the answer to a pending agent question. "
                    "Treat it as the response to that pending question, not as a new "
                    "independent task.\n"
                    f"Pending question: {question}\n"
                    f"User answer: {answer}"
                )
            if include_system and message_dict.get("role") == "system":
                content = str(message_dict.get("content") or "").strip()
                if content:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Previous system-context message retained for "
                                "continuity. Current system instructions above "
                                "take precedence:\n"
                                f"{content}"
                            ),
                        }
                    )
                continue
            messages.append(message_dict)

        if include_system:
            system_content = "\n\n".join(
                part.strip() for part in system_parts if part.strip()
            )
            if system_content:
                messages.insert(0, {"role": "system", "content": system_content})
        return messages

    def _current_time_context(self) -> str:
        current_time = _utcnow()
        return (
            "Current date and time: "
            f"{current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}. "
            "Use this as the reference for relative dates such as today, recent, "
            "latest, yesterday, and tomorrow."
        )

    def _system_context(self) -> str:
        parts = [self._current_time_context()]
        dag_step_id = self.metadata.get("dag_step_id")
        if dag_step_id:
            dag_step_name = str(self.metadata.get("dag_step_name") or "").strip()
            dag_step_description = str(
                self.metadata.get("dag_step_description") or dag_step_name
            ).strip()
            dag_dependencies = self.metadata.get("dag_dependencies")
            if not isinstance(dag_dependencies, list):
                dag_dependencies = []
            dag_tool_names = self.metadata.get("dag_tool_names")
            if not isinstance(dag_tool_names, list):
                dag_tool_names = []
            suggested_tools = (
                ", ".join(str(name) for name in dag_tool_names if str(name).strip())
                or "(none)"
            )
            original_goal = str(self.metadata.get("dag_original_goal") or "").strip()
            parts.append(
                "DAG step execution scope:\n"
                f"- Overall user goal is background context only: "
                f"{original_goal or '(not provided)'}\n"
                f"- Current step id: {dag_step_id}\n"
                f"- Current step title: {dag_step_name or dag_step_id}\n"
                f"- Current step description: "
                f"{dag_step_description or dag_step_name or dag_step_id}\n"
                f"- Current step dependencies: {dag_dependencies}\n"
                f"- Suggested tools for this step: {suggested_tools}\n\n"
                "Only execute the current DAG step. Do not perform sibling, "
                "downstream, final synthesis, rendering, export, or delivery work "
                "unless that work is explicitly part of the current step description. "
                "Use dependency results only as inputs for this step. Treat suggested "
                "tools as the primary tool scope: prefer them and avoid other tools "
                "unless this current step cannot be completed or recovered without "
                "them. If no suggested tools are listed, avoid tool calls unless the "
                "step clearly cannot be completed from provided context and dependency "
                "results."
            )
        memory_context = self.metadata.get(MEMORY_CONTEXT_METADATA_KEY)
        if memory_context:
            parts.append(
                "Relevant memories from previous tasks:\n"
                f"{str(memory_context).strip()}\n\n"
                "Memory usage rules: treat memory as auxiliary context, not as "
                "the current user instruction and not as sufficient evidence for "
                "new factual claims. Memory may inform preferences, prior attempts, "
                "known leads, and failure patterns. For requests that depend on "
                "recent, latest, current, public, source-backed, or otherwise "
                "verifiable facts, use memory only as search or reasoning leads; "
                "verify with available current context, files, knowledge-base "
                "results, or tools before answering. Do not ask the user whether "
                "to use memory or whether to search; decide the appropriate "
                "execution strategy yourself."
            )
        skill_context = self.metadata.get(SKILL_CONTEXT_METADATA_KEY)
        if skill_context:
            parts.append(
                "Selected skill guidance. Use it when relevant to the current task:\n"
                f"{str(skill_context).strip()}"
            )
        return "\n\n".join(part for part in parts if part.strip())

    def get_recent_messages(self, n: int = 10) -> list[Message]:
        if n <= 0:
            return self.messages[:]
        return self.messages[-n:]

    def get_messages_by_role(self, role: str) -> list[Message]:
        return [message for message in self.messages if message.role == role]

    def record_llm_call(
        self,
        response_message: Message,
        input_tokens: int,
        output_tokens: int,
    ) -> Message:
        updated_response = Message(
            role=response_message.role,
            content=response_message.content,
            timestamp=response_message.timestamp,
            metadata=response_message.metadata,
            tool_calls=response_message.tool_calls,
            tool_call_id=response_message.tool_call_id,
            hidden=response_message.hidden,
            output_tokens=output_tokens,
        )
        self.messages.append(updated_response)
        self.llm_calls.append(
            LLMCallRecord(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                message_index=len(self.messages) - 1,
                prompt_message_count=max(0, len(self.messages) - 1),
                prompt_content_chars=self._message_content_chars(
                    self.messages[: max(0, len(self.messages) - 1)]
                ),
            )
        )
        return updated_response

    def record_llm_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        prompt_message_count: int | None = None,
    ) -> None:
        """Record provider usage for an LLM call without appending a message."""

        if input_tokens <= 0 and output_tokens <= 0:
            return

        if prompt_message_count is None:
            prompt_message_count = len(self.messages)
        prompt_message_count = max(0, min(prompt_message_count, len(self.messages)))
        self.llm_calls.append(
            LLMCallRecord(
                input_tokens=max(0, input_tokens),
                output_tokens=max(0, output_tokens),
                total_tokens=max(0, input_tokens) + max(0, output_tokens),
                message_index=max(0, prompt_message_count - 1),
                prompt_message_count=prompt_message_count,
                prompt_content_chars=self._message_content_chars(
                    self.messages[:prompt_message_count]
                ),
            )
        )

    def get_total_token_usage(self) -> dict[str, int]:
        total_input = sum(call.input_tokens for call in self.llm_calls)
        total_output = sum(call.output_tokens for call in self.llm_calls)
        return {
            "total": total_input + total_output,
            "input": total_input,
            "output": total_output,
            "call_count": len(self.llm_calls),
        }

    def extend_with_messages(self, messages: list[Message]) -> None:
        existing: dict[Message, Message] = {
            message: message for message in self.messages
        }
        for message in messages:
            existing[message] = message
        self.messages = list(existing.values())

    @classmethod
    def merge_contexts(
        cls,
        contexts: list["ExecutionContext"],
        strategy: MergeStrategy = MergeStrategy.CHRONOLOGICAL,
    ) -> "ExecutionContext":
        if not contexts:
            return cls()

        base = contexts[0]
        merged = cls(
            execution_id=f"{base.execution_id}_merged_{uuid4().hex[:8]}",
            user_id=base.user_id,
            session_id=base.session_id,
            components={
                name: clone_component(component)
                for name, component in base.components.items()
            },
            system_prompt=base.system_prompt,
            metadata=dict(base.metadata),
            created_at=base.created_at,
            compact_config=replace(base.compact_config),
        )
        merged.llm_calls = [replace(call) for call in base.llm_calls]
        merged._merge_contexts_from_list(contexts, strategy)
        return merged

    def _merge_contexts_from_list(
        self,
        contexts: list["ExecutionContext"],
        strategy: MergeStrategy,
    ) -> None:
        if not contexts:
            self.messages = []
            self.llm_calls = []
            return

        if strategy == MergeStrategy.CHRONOLOGICAL:
            messages_with_source: list[tuple[Message, datetime]] = []
            for context in contexts:
                for message in context.messages:
                    messages_with_source.append((message, message.timestamp))
            messages_with_source.sort(key=lambda item: item[1])
            self._merge_messages_dedup(messages_with_source)
        elif strategy == MergeStrategy.TOPOLOGICAL:
            self.messages = []
            for context in contexts:
                self.extend_with_messages(context.messages)
        elif strategy == MergeStrategy.PREFER_FIRST:
            seen: set[Message] = set()
            ordered: list[Message] = []
            for context in contexts:
                for message in context.messages:
                    if message not in seen:
                        seen.add(message)
                        ordered.append(message)
            self.messages = ordered
        else:
            self.messages = []

        self._merge_llm_calls(contexts)

    def _merge_messages_dedup(
        self, messages_with_source: list[tuple[Message, datetime]]
    ) -> None:
        seen: dict[Message, Message] = {}
        ordered: list[Message] = []
        for message, _ in messages_with_source:
            if message in seen:
                continue
            seen[message] = message
            ordered.append(message)
        self.messages = ordered

    def _merge_llm_calls(self, contexts: list["ExecutionContext"]) -> None:
        if not contexts:
            return

        message_index_map = {message: idx for idx, message in enumerate(self.messages)}
        merged_calls: list[LLMCallRecord] = []
        for context in contexts:
            for call in context.llm_calls:
                new_index = call.message_index
                if 0 <= call.message_index < len(context.messages):
                    original_message = context.messages[call.message_index]
                    new_index = message_index_map.get(original_message, new_index)

                merged_calls.append(
                    LLMCallRecord(
                        input_tokens=call.input_tokens,
                        output_tokens=call.output_tokens,
                        total_tokens=call.total_tokens,
                        message_index=new_index,
                        prompt_message_count=call.prompt_message_count,
                        prompt_content_chars=call.prompt_content_chars,
                        timestamp=call.timestamp,
                    )
                )
        self.llm_calls = merged_calls

    def create_child_context(
        self,
        execution_id: str | None = None,
        task: str | None = None,
        include_system_prompt: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "ExecutionContext":
        child_metadata = dict(self.metadata)
        if metadata:
            child_metadata.update(metadata)
        if task:
            child_metadata["task"] = task

        child = ExecutionContext(
            execution_id=execution_id or f"{self.execution_id}_child_{uuid4().hex[:8]}",
            user_id=self.user_id,
            session_id=self.session_id,
            components={
                name: clone_component(component)
                for name, component in self.components.items()
            },
            messages=self.messages.copy(),
            system_prompt=self.system_prompt if include_system_prompt else None,
            metadata=child_metadata,
            compact_config=replace(self.compact_config),
            llm_calls=[replace(call) for call in self.llm_calls],
        )
        if task:
            child.add_user_message(task)
        return child

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "components": {
                name: component.to_dict() for name, component in self.components.items()
            },
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "metadata": message.metadata,
                    "tool_calls": message.tool_calls,
                    "tool_call_id": message.tool_call_id,
                    "hidden": message.hidden,
                    "output_tokens": message.output_tokens,
                }
                for message in self.messages
            ],
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "llm_calls": [
                {
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "total_tokens": call.total_tokens,
                    "message_index": call.message_index,
                    "prompt_message_count": call.prompt_message_count,
                    "prompt_content_chars": call.prompt_content_chars,
                    "timestamp": call.timestamp.isoformat(),
                }
                for call in self.llm_calls
            ],
            "compact_config": {
                "enabled": self.compact_config.enabled,
                "threshold": self.compact_config.threshold,
                "strategy": self.compact_config.strategy,
                "max_messages": self.compact_config.max_messages,
            },
            # Backward compatibility for older serialized payloads.
            "workspace_id": self.workspace_id,
            "workspace_path": self.workspace_path,
            "cwd": self.cwd,
            "workspace_state": self.workspace_state,
            "memory_session_id": self.memory_session_id,
            "memory_snapshot": self.memory_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionContext":
        messages = [
            Message(
                role=item["role"],
                content=item["content"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
                metadata=item.get("metadata", {}),
                tool_calls=item.get("tool_calls"),
                tool_call_id=item.get("tool_call_id"),
                hidden=item.get("hidden", False),
                output_tokens=item.get("output_tokens"),
            )
            for item in data.get("messages", [])
        ]
        llm_calls = [
            LLMCallRecord(
                input_tokens=call["input_tokens"],
                output_tokens=call["output_tokens"],
                total_tokens=call["total_tokens"],
                message_index=call["message_index"],
                prompt_message_count=call.get("prompt_message_count"),
                prompt_content_chars=call.get("prompt_content_chars"),
                timestamp=datetime.fromisoformat(call["timestamp"]),
            )
            for call in data.get("llm_calls", [])
        ]
        compact = data.get("compact_config", {})
        compact_config = CompactConfig(
            enabled=compact.get("enabled", True),
            threshold=compact.get("threshold", 8000),
            strategy=compact.get("strategy", "truncate"),
            max_messages=compact.get("max_messages", 20),
        )
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else _utcnow()
        )

        components_payload = data.get("components", {})
        components: dict[str, ExecutionComponent] = {}
        for name, payload in components_payload.items():
            loader = COMPONENT_LOADERS.get(name)
            if loader:
                components[name] = loader(payload)
            else:
                components[name] = GenericComponent(data=payload)

        context = cls(
            execution_id=data.get("execution_id", str(uuid4())),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            components=components,
            messages=messages,
            system_prompt=data.get("system_prompt"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            llm_calls=llm_calls,
            compact_config=compact_config,
        )

        if not components_payload:
            if any(
                data.get(key) is not None
                for key in ("workspace_id", "workspace_path", "cwd", "workspace_state")
            ):
                context.attach_workspace(
                    workspace_id=data.get("workspace_id"),
                    workspace_path=data.get("workspace_path"),
                    cwd=data.get("cwd"),
                    state=data.get("workspace_state"),
                )
            if data.get("memory_session_id") or data.get("memory_snapshot") is not None:
                context.attach_memory_session(
                    session_id=data.get("memory_session_id"),
                    snapshot=data.get("memory_snapshot"),
                )

        return context

    def compact_if_needed(self, llm: Any = None) -> CompactResult:
        if not self.compact_config.enabled:
            return CompactResult(
                compacted=False,
                original_count=len(self.messages),
                final_count=len(self.messages),
                strategy="none",
            )

        total_tokens = self._get_total_tokens()
        if total_tokens > self.compact_config.threshold:
            result = self._compact(llm)
            result.metadata.setdefault("original_tokens", total_tokens)
            result.metadata.setdefault("threshold", self.compact_config.threshold)
            return result

        return CompactResult(
            compacted=False,
            original_count=len(self.messages),
            final_count=len(self.messages),
            strategy="none",
        )

    def _compact(self, llm: Any = None) -> CompactResult:
        original_count = len(self.messages)
        if self.compact_config.strategy == "truncate":
            keep_count = min(max(0, self.compact_config.max_messages), original_count)
            self.messages = self._tail_window_preserving_tool_pairs(keep_count)
            removed = max(0, original_count - len(self.messages))
            return CompactResult(
                compacted=True,
                original_count=original_count,
                final_count=len(self.messages),
                strategy="truncate",
                metadata={"removed_count": removed},
            )

        return CompactResult(
            compacted=False,
            original_count=original_count,
            final_count=len(self.messages),
            strategy="none",
        )

    def _get_total_tokens(self) -> int:
        if self.llm_calls:
            latest_call = self.llm_calls[-1]
            if latest_call.input_tokens > 0:
                prompt_message_count = latest_call.prompt_message_count
                prompt_content_chars = latest_call.prompt_content_chars
                if (
                    prompt_message_count is not None
                    and prompt_content_chars is not None
                    and 0 <= prompt_message_count <= len(self.messages)
                    and self._message_content_chars(
                        self.messages[:prompt_message_count]
                    )
                    == prompt_content_chars
                ):
                    delta_chars = self._message_content_chars(
                        self.messages[prompt_message_count:]
                    )
                    return latest_call.input_tokens + max(0, delta_chars // 4)
        return self._estimate_message_tokens(self.messages)

    def _estimate_message_tokens(self, messages: list[Message]) -> int:
        return sum(max(1, len(message.content) // 4) for message in messages)

    def _message_content_chars(self, messages: list[Message]) -> int:
        return sum(len(message.content) for message in messages)

    def _tail_window_preserving_tool_pairs(self, keep_count: int) -> list[Message]:
        """Keep recent messages without cutting a native tool-call exchange."""

        if keep_count <= 0:
            return []

        start = max(0, len(self.messages) - keep_count)
        while start > 0 and self.messages[start].role == "tool":
            start -= 1

        return self._sanitize_tool_message_pairs(self.messages[start:])

    def _sanitize_tool_message_pairs(self, messages: list[Message]) -> list[Message]:
        """Drop native tool protocol fragments that providers reject.

        OpenAI-style chat requires every ``tool`` message to immediately follow an
        assistant message that declared the corresponding ``tool_calls``. Context
        compaction and token truncation must therefore treat an assistant tool-call
        message and its tool results as an atomic block.
        """

        sanitized: list[Message] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role != "assistant" or not message.tool_calls:
                if message.role != "tool":
                    sanitized.append(message)
                index += 1
                continue

            tool_messages: list[Message] = []
            next_index = index + 1
            while next_index < len(messages) and messages[next_index].role == "tool":
                tool_messages.append(messages[next_index])
                next_index += 1

            expected_ids = {
                str(tool_call.get("id"))
                for tool_call in message.tool_calls
                if isinstance(tool_call, dict) and tool_call.get("id")
            }
            received_ids = {
                str(tool_message.tool_call_id)
                for tool_message in tool_messages
                if tool_message.tool_call_id
            }
            if expected_ids and expected_ids.issubset(received_ids):
                sanitized.append(message)
                sanitized.extend(tool_messages)
            elif not expected_ids and len(tool_messages) >= len(message.tool_calls):
                sanitized.append(message)
                sanitized.extend(tool_messages)

            index = next_index

        return sanitized

    def _truncate_by_tokens(
        self,
        messages: list[Message],
        max_tokens: int,
    ) -> list[Message]:
        current_tokens = 0
        start = len(messages)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role == "assistant" and message.output_tokens is not None:
                message_tokens = message.output_tokens
            else:
                message_tokens = max(1, len(message.content) // 4)

            if current_tokens + message_tokens > max_tokens:
                break
            start = index
            current_tokens += message_tokens

        while start > 0 and messages[start].role == "tool":
            start -= 1

        return self._sanitize_tool_message_pairs(messages[start:])
