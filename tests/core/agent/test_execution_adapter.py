from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from xagent.core.agent import AgentExecutionAdapter, AgentExecutionConfig
from xagent.core.agent.execution_adapter import INTERRUPTED_USER_MESSAGE
from xagent.core.agent.service import AgentService


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.model_name = "fake-llm"

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class BlockingLLM:
    def __init__(self, response: Any = "released") -> None:
        self.response = response
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        return self.response


class TracerCheckpointStore:
    def __init__(self) -> None:
        self.by_execution_id: dict[str, dict[str, Any]] = {}

    async def checkpoint(self, **payload: Any) -> None:
        self.by_execution_id[str(payload["execution_id"])] = dict(payload)

    async def load_latest_checkpoint(self, execution_id: str) -> dict[str, Any] | None:
        payload = self.by_execution_id.get(execution_id)
        return dict(payload) if payload is not None else None


class RecordingTracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def trace_event(self, event_type: Any, **kwargs: Any) -> str:
        self.events.append({"event_type": event_type, **kwargs})
        return f"event-{len(self.events)}"


class FakeTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.setup_calls: list[str | None] = []
        self.teardown_calls: list[str | None] = []

        class Metadata:
            name = "noop"
            description = "No-op test tool."

        self.metadata = Metadata()
        self.name = "noop"

    def args_type(self) -> type:
        class Args:
            @staticmethod
            def model_json_schema() -> dict[str, Any]:
                return {"type": "object", "properties": {}}

        return Args

    async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(args)
        return {"args": args}

    async def setup(self, task_id: str | None = None) -> None:
        self.setup_calls.append(task_id)

    async def teardown(self, task_id: str | None = None) -> None:
        self.teardown_calls.append(task_id)


class StartHandle:
    task = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": "running"}


class RecordingRegistry:
    def __init__(self) -> None:
        self.start_kwargs: dict[str, Any] | None = None

    def start(self, *args: Any, **kwargs: Any) -> StartHandle:
        self.start_kwargs = dict(kwargs)
        return StartHandle()


class NoSkillManager:
    async def select_skill(self, **_: Any) -> None:
        return None


class ReusedExecutionAdapter:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.execute_kwargs: dict[str, Any] | None = None

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.execute_kwargs = dict(kwargs)
        return {"success": True}


def auto_decision(
    action: str, *, answer: str = "", reason: str = "test"
) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "call-select",
                "function": {
                    "name": "select_execution_pattern",
                    "arguments": {
                        "action": action,
                        "reason": reason,
                        "response_language": "English",
                        "answer": answer,
                        "requires_current_or_external_facts": False,
                        "existing_context_sufficient": True,
                        "evidence_basis": "test context",
                        "missing_verification": "",
                    },
                },
            }
        ]
    }


def dag_plan(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "call-plan",
                "function": {
                    "name": "generate_execution_plan",
                    "arguments": {"steps": steps, "response_language": "English"},
                },
            }
        ]
    }


def dag_completion(answer: str = "dag done") -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": "call-assess",
                "function": {
                    "name": "assess_dag_completion",
                    "arguments": json.dumps(
                        {
                            "status": "completed",
                            "reason": "Done.",
                            "answer": answer,
                            "missing_work": "",
                            "replan_instruction": "",
                        }
                    ),
                },
            }
        ]
    }


def test_execution_adapter_uses_service_id_as_runner_workspace_id() -> None:
    registry = RecordingRegistry()
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="web-task",
            pattern="react",
            llm=FakeLLM(["unused"]),
            service_id="web_task_458",
            registry=cast(Any, registry),
            skill_manager=NoSkillManager(),
        )
    )

    adapter.start(task="Generate a file", task_id="458")

    assert registry.start_kwargs is not None
    assert registry.start_kwargs["execution_id"] == "458"
    assert registry.start_kwargs["workspace_id"] == "web_task_458"


@pytest.mark.asyncio
async def test_execution_adapter_routes_single_call_to_one_tool_then_final_answer() -> (
    None
):
    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "call-noop",
                        "function": {
                            "name": "noop",
                            "arguments": '{"value":"from tool"}',
                        },
                    }
                ]
            },
            {"content": "done", "done": True},
        ]
    )
    tool = FakeTool()
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="single",
            pattern="single_call",
            llm=llm,
            tools=[tool],
            service_id="single-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Say done", task_id="single-exec")

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["output"] == "done"
    assert result["metadata"]["execution_type"] == "agent_single_call"
    assert tool.calls == [{"value": "from tool"}]
    assert llm.calls[0]["tools"][0]["function"]["name"] == "noop"
    assert llm.calls[0]["tool_choice"] == "required"
    assert [schema["function"]["name"] for schema in llm.calls[1]["tools"]] == [
        "final_answer"
    ]
    assert llm.calls[1]["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_execution_adapter_routes_react_to_react() -> None:
    llm = FakeLLM(["react done"])
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="react",
            pattern="react",
            llm=llm,
            tools=[FakeTool()],
            service_id="react-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Say done", task_id="react-exec")

    assert result["success"] is True
    assert result["output"] == "react done"
    assert result["metadata"]["execution_type"] == "agent_react"
    assert result["agent_result"]["pattern"] == "ReActPattern"
    assert llm.calls[0]["tools"] is not None


@pytest.mark.asyncio
async def test_execution_adapter_propagates_request_context_to_llm() -> None:
    llm = FakeLLM(["context done"])
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="context",
            pattern="react",
            llm=llm,
            tools=[FakeTool()],
            system_prompt="Base system.",
            service_id="context-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(
        task="Say done",
        task_id="context-exec",
        context={
            "system_prompt": "Follow request-specific rules.",
            "process_description": "Use the provided process.",
            "examples": [{"input": "hello", "output": "world"}],
        },
    )

    assert result["success"] is True
    system_messages = [
        message["content"]
        for message in llm.calls[0]["messages"]
        if message["role"] == "system"
    ]
    assert len(system_messages) == 1
    system_prompt = system_messages[0]
    assert "Base system." in system_prompt
    assert "Follow request-specific rules." in system_prompt
    assert "Use the provided process." in system_prompt
    assert "Input: hello" in system_prompt
    assert "Output: world" in system_prompt


@pytest.mark.asyncio
async def test_execution_adapter_runs_tool_lifecycle() -> None:
    llm = FakeLLM(["done"])
    tool = FakeTool()
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="lifecycle",
            pattern="react",
            llm=llm,
            tools=[tool],
            service_id="lifecycle-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Say done", task_id="lifecycle-exec")

    assert result["success"] is True
    assert tool.setup_calls == ["lifecycle-exec"]
    assert tool.teardown_calls == ["lifecycle-exec"]


@pytest.mark.asyncio
async def test_execution_adapter_preserves_waiting_for_user_payload() -> None:
    interactions = [
        {
            "type": "action_cards",
            "field": "kb_source",
            "label": "Choose source",
            "options": [
                {
                    "label": "Upload",
                    "value": "upload",
                    "action_type": "upload",
                }
            ],
        }
    ]
    llm = FakeLLM(
        [
            {
                "tool_calls": [
                    {
                        "id": "call-ask",
                        "function": {
                            "name": "ask_user_question",
                            "arguments": json.dumps(
                                {
                                    "message": "Need FAQ content",
                                    "interactions": interactions,
                                }
                            ),
                        },
                    }
                ]
            }
        ]
    )
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="react",
            pattern="react",
            llm=llm,
            tools=[],
            service_id="react-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Create an FAQ agent", task_id="react-exec")

    assert result["success"] is False
    assert result["status"] == "waiting_for_user"
    assert result["message"] == "Need FAQ content"
    assert result["interactions"] == interactions
    assert result["chat_response"] == {
        "message": "Need FAQ content",
        "interactions": interactions,
    }


@pytest.mark.asyncio
async def test_execution_adapter_includes_persisted_conversation_history() -> None:
    llm = FakeLLM(["generated"])
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="history",
            pattern="react",
            llm=llm,
            tools=[FakeTool()],
            service_id="history-service",
            skill_manager=NoSkillManager(),
            conversation_history=[
                {"role": "user", "content": "用 Python 生成随机整数"},
                {"role": "assistant", "content": "可以用 random.randint。"},
            ],
        )
    )

    result = await adapter.execute(task="生成一个返回给我", task_id="history-exec")

    assert result["success"] is True
    messages = llm.calls[0]["messages"]
    assert [
        message["content"] for message in messages if message["role"] == "user"
    ] == [
        "用 Python 生成随机整数",
        "生成一个返回给我",
    ]
    assert any(
        message["role"] == "assistant"
        and message["content"] == "可以用 random.randint。"
        for message in messages
    )


@pytest.mark.asyncio
async def test_execution_adapter_includes_persisted_execution_context_before_history() -> (
    None
):
    llm = FakeLLM(["updated"])
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="execution-context",
            pattern="react",
            llm=llm,
            tools=[FakeTool()],
            service_id="execution-context-service",
            skill_manager=NoSkillManager(),
            execution_context_messages=[
                {
                    "role": "system",
                    "content": "Previous tool result: output/index_en.html exists.",
                }
            ],
            conversation_history=[
                {"role": "user", "content": "写到文件"},
                {"role": "assistant", "content": "已写入文件。"},
            ],
        )
    )

    result = await adapter.execute(task="继续修改文件", task_id="execution-context")

    assert result["success"] is True
    messages = llm.calls[0]["messages"]
    system_messages = [
        message["content"] for message in messages if message["role"] == "system"
    ]
    assert len(system_messages) == 1
    assert any(
        message["role"] == "user"
        and "Previous tool result: output/index_en.html exists." in message["content"]
        for message in messages
    )
    assert [message["content"] for message in messages if message["role"] == "user"][
        -2:
    ] == ["写到文件", "继续修改文件"]


@pytest.mark.asyncio
async def test_agent_service_passes_conversation_history_to_execution_adapter() -> None:
    llm = FakeLLM(["generated", '{"should_store": false, "reason": "test"}'])
    service = AgentService(
        name="history-service",
        id="history-service",
        pattern="react",
        llm=cast(Any, llm),
        tools=cast(Any, [FakeTool()]),
        tool_config=None,
    )
    service.allowed_skills = []
    service.set_conversation_history(
        [
            {"role": "user", "content": "用 Python 生成随机整数"},
            {"role": "assistant", "content": "可以用 random.randint。"},
        ]
    )

    result = await service.execute_task(
        "生成一个返回给我", task_id="history-service-task"
    )

    assert result["success"] is True
    messages = llm.calls[0]["messages"]
    assert [
        message["content"] for message in messages if message["role"] == "user"
    ] == [
        "用 Python 生成随机整数",
        "生成一个返回给我",
    ]


@pytest.mark.asyncio
async def test_agent_service_passes_execution_context_to_execution_adapter() -> None:
    llm = FakeLLM(["updated", '{"should_store": false, "reason": "test"}'])
    service = AgentService(
        name="execution-context-service",
        id="execution-context-service",
        pattern="react",
        llm=cast(Any, llm),
        tools=cast(Any, [FakeTool()]),
        tool_config=None,
    )
    service.allowed_skills = []
    service.set_execution_context_messages(
        [
            {
                "role": "system",
                "content": "Previous tool result: output/index_zh.html exists.",
            }
        ]
    )

    result = await service.execute_task("继续修改文件", task_id="service-context-task")

    assert result["success"] is True
    system_messages = [
        message["content"]
        for message in llm.calls[0]["messages"]
        if message["role"] == "system"
    ]
    assert len(system_messages) == 1
    messages = llm.calls[0]["messages"]
    assert any(
        message["role"] == "user"
        and "Previous tool result: output/index_zh.html exists." in message["content"]
        for message in messages
    )


@pytest.mark.asyncio
async def test_agent_service_refreshes_compact_llm_on_reused_execution_adapter() -> (
    None
):
    initial_llm = FakeLLM(["initial"])
    updated_llm = FakeLLM(["updated"])
    initial_compact_llm = FakeLLM(["initial compact"])
    updated_compact_llm = FakeLLM(["updated compact"])
    adapter = ReusedExecutionAdapter(
        SimpleNamespace(
            current_task_id=None,
            tools=[],
            llm=initial_llm,
            compact_llm=initial_compact_llm,
            pattern="react",
            outbound_message_handler=None,
            conversation_history=[],
            execution_context_messages=[],
            recovered_skill_context=None,
            memory_store=None,
            allowed_skills=None,
        )
    )
    service = AgentService(
        name="compact-refresh-service",
        id="compact-refresh-service",
        pattern="react",
        llm=cast(Any, initial_llm),
        compact_llm=cast(Any, initial_compact_llm),
        tools=[],
        tool_config=None,
    )
    service._execution_adapter = cast(Any, adapter)

    service.llm = cast(Any, updated_llm)
    service.compact_llm = cast(Any, updated_compact_llm)

    result = await service._execute_agent_task(
        "continue", task_id="task-compact-refresh"
    )

    assert result["success"] is True
    assert adapter.config.current_task_id == "task-compact-refresh"
    assert adapter.config.llm is updated_llm
    assert adapter.config.compact_llm is updated_compact_llm


@pytest.mark.asyncio
async def test_execution_adapter_emits_visible_trace_events() -> None:
    tracer = RecordingTracer()
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="trace",
            pattern="react",
            llm=FakeLLM(["hello"]),
            tracer=tracer,
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="hi", task_id="trace-exec")

    assert result["success"] is True
    event_values = [event["event_type"].value for event in tracer.events]
    assert "task_start_message" in event_values
    assert "task_end_message" in event_values
    user_event = next(
        event
        for event in tracer.events
        if event["event_type"].value == "task_start_message"
    )
    ai_event = next(
        event
        for event in tracer.events
        if event["event_type"].value == "task_end_message"
    )
    assert user_event["data"]["message"] == "hi"
    assert ai_event["data"]["content"] == "hello"


@pytest.mark.asyncio
async def test_execution_adapter_routes_dag_to_dag() -> None:
    llm = FakeLLM(
        [
            dag_plan([{"id": "answer", "task": "Answer directly"}]),
            "dag done",
            dag_completion("dag done"),
        ]
    )
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="dag",
            pattern="dag_plan_execute",
            llm=llm,
            tools=[],
            service_id="dag-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Plan then answer", task_id="dag-exec")

    assert result["success"] is True
    assert result["output"] == "dag done"
    assert result["metadata"]["execution_type"] == "agent_dag"
    assert result["agent_result"]["pattern"] == "DAGPattern"


def test_execution_adapter_passes_dag_max_concurrency_to_pattern() -> None:
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="dag",
            pattern="dag_plan_execute",
            llm=FakeLLM([]),
            dag_max_concurrency=2,
            skill_manager=NoSkillManager(),
        )
    )

    pattern, execution_type = adapter._build_pattern()

    assert execution_type == "agent_dag"
    assert pattern.max_concurrency == 2


def test_execution_adapter_routes_auto_to_auto() -> None:
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="auto",
            pattern="auto",
            llm=FakeLLM([]),
            skill_manager=NoSkillManager(),
        )
    )

    pattern, execution_type = adapter._build_pattern()

    assert execution_type == "agent_auto"
    assert pattern.__class__.__name__ == "AutoPattern"
    assert pattern.dag_pattern.max_concurrency == 4


def test_execution_adapter_passes_tool_concurrency_to_react_children() -> None:
    # The tool-concurrency config must reach the ReAct child on every route that
    # can run ReAct: single_call, react, and auto. auto is the default for
    # standalone non-v1 tasks, so it is the common path for enabling the feature;
    # it previously built a default ReActPattern() and silently dropped the
    # config (regression guard).
    def build_react_child(pattern: str) -> Any:
        adapter = AgentExecutionAdapter(
            AgentExecutionConfig(
                name=pattern,
                pattern=pattern,
                llm=FakeLLM([]),
                tool_parallel_enabled=True,
                tool_max_concurrency=7,
                skill_manager=NoSkillManager(),
            )
        )
        built, _ = adapter._build_pattern()
        return built.react_pattern if pattern == "auto" else built

    for pattern in ("single_call", "react", "auto"):
        react = build_react_child(pattern)
        assert react.tool_parallel_enabled is True, pattern
        assert react.tool_max_concurrency == 7, pattern


@pytest.mark.asyncio
async def test_execution_adapter_executes_auto_final_answer() -> None:
    llm = FakeLLM(
        [auto_decision("final_answer", answer="hello", reason="Greeting only.")]
    )
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="auto",
            pattern="auto",
            llm=llm,
            service_id="auto-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="hi", task_id="auto-final-exec")

    assert result["success"] is True
    assert result["output"] == "hello"
    assert result["metadata"]["execution_type"] == "agent_auto"
    assert result["agent_result"]["pattern"] == "AutoPattern"


@pytest.mark.asyncio
async def test_execution_adapter_executes_auto_react() -> None:
    llm = FakeLLM(
        [
            auto_decision("react", reason="Needs ReAct."),
            "react done",
        ]
    )
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="auto",
            pattern="auto",
            llm=llm,
            service_id="auto-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Use react", task_id="auto-react-exec")

    assert result["success"] is True
    assert result["output"] == "react done"
    assert result["metadata"]["execution_type"] == "agent_auto"
    assert result["agent_result"]["auto_decision"]["action"] == "react"


@pytest.mark.asyncio
async def test_execution_adapter_executes_auto_plan_execute() -> None:
    llm = FakeLLM(
        [
            auto_decision("plan_execute", reason="Needs DAG."),
            dag_plan([{"id": "answer", "task": "Answer directly"}]),
            "dag done",
            dag_completion("dag done"),
        ]
    )
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="auto",
            pattern="auto",
            llm=llm,
            service_id="auto-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Use DAG", task_id="auto-dag-exec")

    assert result["success"] is True
    assert result["output"] == "dag done"
    assert result["metadata"]["execution_type"] == "agent_auto"
    assert result["agent_result"]["auto_decision"]["action"] == "plan_execute"


@pytest.mark.asyncio
async def test_execution_adapter_exposes_pause_and_message_controls() -> None:
    llm = BlockingLLM(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-noop",
                    "name": "noop",
                    "args": {},
                }
            ],
        }
    )
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="control",
            pattern="react",
            llm=llm,
            tools=[FakeTool()],
            service_id="control-service",
            skill_manager=NoSkillManager(),
        )
    )

    status = adapter.start(task="Wait", task_id="control-exec")
    assert status["status"] == "running"

    handle = adapter.registry.get("control-exec")
    assert handle is not None
    assert handle.task is not None
    await llm.started.wait()
    assert adapter.pause("control-exec", reason="pause from test") is True
    assert await adapter.post_user_message(
        "control-exec",
        "Follow-up",
        request_interrupt=False,
    )
    llm.release.set()
    result = await handle.task

    final_status = adapter.get_status("control-exec")
    assert result["status"] == "interrupted"
    assert final_status is not None
    assert final_status["status"] == "interrupted"
    assert final_status["is_resumable"] is True


@pytest.mark.asyncio
async def test_execution_adapter_exposes_cancel_control() -> None:
    llm = BlockingLLM()
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="cancel",
            pattern="react",
            llm=llm,
            service_id="cancel-service",
            skill_manager=NoSkillManager(),
        )
    )

    adapter.start(task="Wait", task_id="cancel-exec")

    handle = adapter.registry.get("cancel-exec")
    assert handle is not None
    assert handle.task is not None
    assert adapter.cancel("cancel-exec", reason="cancel from test") is True
    with pytest.raises(asyncio.CancelledError):
        await handle.task
    for _ in range(20):
        if adapter.get_status("cancel-exec") is None:
            break
        await asyncio.sleep(0)
    assert adapter.get_status("cancel-exec") is None


@pytest.mark.asyncio
async def test_execution_adapter_forwards_outbound_messages() -> None:
    sent_messages: list[dict[str, Any]] = []
    llm = FakeLLM(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-message",
                        "name": "send_message",
                        "args": {
                            "message": "Still working",
                            "message_type": "progress",
                            "expect_response": False,
                        },
                    }
                ],
            },
            "done",
        ]
    )
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="outbound",
            pattern="react",
            llm=llm,
            outbound_message_handler=sent_messages.append,
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Send progress", task_id="outbound-exec")

    assert result["success"] is True
    assert result["output"] == "done"
    assert len(sent_messages) == 1
    outbound_message = sent_messages[0]
    assert outbound_message["type"] == "agent_message"
    assert outbound_message["execution_id"] == "outbound-exec"
    assert outbound_message["message"] == "Still working"
    assert outbound_message["message_type"] == "progress"
    assert outbound_message["expect_response"] is False
    assert outbound_message["visible"] is True
    assert outbound_message["step_id"] == outbound_message["metadata"]["step_id"]


def test_execution_adapter_uses_last_assistant_message_when_output_missing() -> None:
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="fallback",
            pattern="react",
            llm=FakeLLM([]),
            skill_manager=NoSkillManager(),
        )
    )
    context = SimpleNamespace(
        messages=[
            SimpleNamespace(role="user", content="question"),
            SimpleNamespace(role="assistant", content="answer from context"),
        ]
    )

    result = adapter._normalize_result(
        result={"success": True, "context": context},
        execution_type="agent_react",
        execution_id="fallback-exec",
    )

    assert result["output"] == "answer from context"


def test_execution_adapter_hides_internal_error_for_interrupted_result() -> None:
    adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="interrupted",
            pattern="react",
            llm=FakeLLM([]),
            skill_manager=NoSkillManager(),
        )
    )

    result = adapter._normalize_result(
        result={
            "status": "interrupted",
            "success": False,
            "error": "ReActPattern interrupted.",
        },
        execution_type="agent_react",
        execution_id="interrupted-exec",
    )

    assert result["output"] == INTERRUPTED_USER_MESSAGE
    assert result["error"] == "ReActPattern interrupted."


@pytest.mark.asyncio
async def test_execution_adapter_resume_restores_from_tracer_after_restart() -> None:
    tracer = TracerCheckpointStore()
    first_llm = BlockingLLM(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-noop",
                    "name": "noop",
                    "args": {},
                }
            ],
        }
    )
    first_adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="restart",
            pattern="react",
            llm=first_llm,
            tools=[FakeTool()],
            tracer=tracer,
            service_id="restart-service",
            skill_manager=NoSkillManager(),
        )
    )

    first_adapter.start(task="Wait for resume", task_id="restart-exec")
    first_handle = first_adapter.registry.get("restart-exec")
    assert first_handle is not None
    assert first_handle.task is not None
    await first_llm.started.wait()
    assert first_adapter.pause("restart-exec", reason="pause before restart") is True
    first_llm.release.set()

    interrupted = await first_handle.task

    assert interrupted["status"] == "interrupted"
    await first_adapter.post_user_message(
        "restart-exec",
        "Resume with concise answer.",
        request_interrupt=False,
    )

    resumed_llm = FakeLLM(["resumed done"])
    resumed_adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="restart",
            pattern="react",
            llm=resumed_llm,
            tools=[FakeTool()],
            tracer=tracer,
            service_id="restart-service",
            skill_manager=NoSkillManager(),
        )
    )

    resumed = await resumed_adapter.resume("restart-exec")

    assert resumed is not None
    assert resumed["success"] is True
    assert resumed["status"] == "completed"
    assert resumed["output"] == "resumed done"
    assert resumed["metadata"]["execution_type"] == "agent_react"
    assert resumed_adapter.get_status("restart-exec") is None
    context_messages = resumed["agent_result"]["context"].messages
    assert any(
        message.role == "user" and message.content == "Resume with concise answer."
        for message in context_messages
    )


@pytest.mark.asyncio
async def test_execution_adapter_posts_user_message_after_restart() -> None:
    tracer = TracerCheckpointStore()
    first_llm = BlockingLLM(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-noop",
                    "name": "noop",
                    "args": {},
                }
            ],
        }
    )
    first_adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="restart-message",
            pattern="react",
            llm=first_llm,
            tools=[FakeTool()],
            tracer=tracer,
            skill_manager=NoSkillManager(),
        )
    )

    first_adapter.start(task="Wait for message", task_id="restart-message-exec")
    first_handle = first_adapter.registry.get("restart-message-exec")
    assert first_handle is not None
    assert first_handle.task is not None
    await first_llm.started.wait()
    assert first_adapter.pause("restart-message-exec", reason="pause before restart")
    first_llm.release.set()
    interrupted = await first_handle.task
    assert interrupted["status"] == "interrupted"

    restarted_adapter = AgentExecutionAdapter(
        AgentExecutionConfig(
            name="restart-message",
            pattern="react",
            llm=FakeLLM(["resumed after message"]),
            tools=[FakeTool()],
            tracer=tracer,
            skill_manager=NoSkillManager(),
        )
    )

    assert await restarted_adapter.post_user_message(
        "restart-message-exec",
        "New message after process restart.",
        request_interrupt=False,
    )
    resumed = await restarted_adapter.resume("restart-message-exec")

    assert resumed is not None
    assert resumed["success"] is True
    context_messages = resumed["agent_result"]["context"].messages
    assert any(
        message.role == "user"
        and message.content == "New message after process restart."
        for message in context_messages
    )


@pytest.mark.asyncio
async def test_set_interrupt_checker_propagates_to_prebuilt_adapter() -> None:
    # The mid-run quota checker is normally installed before a run, but
    # set_interrupt_checker must also push onto an already-built adapter — the
    # only path that updates enforcement on an already-executing task.
    service = AgentService(
        name="interrupt-service",
        id="interrupt-service",
        pattern="react",
        llm=cast(Any, FakeLLM([])),
        tools=cast(Any, []),
        tool_config=None,
    )
    service.allowed_skills = []
    service._execution_adapter = service._build_execution_adapter()

    def checker() -> None:
        return None

    service.set_interrupt_checker(checker)
    assert service._execution_adapter.config.interrupt_checker is checker
    service.set_interrupt_checker(None)
    assert service._execution_adapter.config.interrupt_checker is None
