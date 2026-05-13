from __future__ import annotations

import asyncio
from typing import Any

import pytest

from xagent.core.agent.service import AgentService
from xagent.core.agent_runtime import AgentV2ExecutionAdapter, AgentV2ExecutionConfig


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
        return {"args": args}


class NoSkillManager:
    async def select_skill(self, **_: Any) -> None:
        return None


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
                    "arguments": {"steps": steps},
                },
            }
        ]
    }


@pytest.mark.asyncio
async def test_v2_adapter_routes_single_call_to_strict_react() -> None:
    llm = FakeLLM(["done"])
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
            name="single",
            pattern="single_call",
            llm=llm,
            tools=[FakeTool()],
            service_id="single-service",
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Say done", task_id="single-exec")

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["output"] == "done"
    assert result["metadata"]["execution_type"] == "agent_v2_single_call"
    assert llm.calls[0]["tools"] is None
    assert llm.calls[0]["tool_choice"] is None


@pytest.mark.asyncio
async def test_v2_adapter_routes_react_to_v2_react() -> None:
    llm = FakeLLM(["react done"])
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
    assert result["metadata"]["execution_type"] == "agent_v2_react"
    assert result["agent_v2_result"]["pattern"] == "ReActPattern"
    assert llm.calls[0]["tools"] is not None


@pytest.mark.asyncio
async def test_v2_adapter_includes_persisted_conversation_history() -> None:
    llm = FakeLLM(["generated"])
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
async def test_v2_adapter_includes_persisted_execution_context_before_history() -> None:
    llm = FakeLLM(["updated"])
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
async def test_agent_service_passes_conversation_history_to_v2_adapter() -> None:
    llm = FakeLLM(["generated", '{"should_store": false, "reason": "test"}'])
    service = AgentService(
        name="history-service",
        id="history-service",
        pattern="react",
        llm=llm,
        tools=[FakeTool()],
        agent_runtime="v2",
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
async def test_agent_service_passes_execution_context_to_v2_adapter() -> None:
    llm = FakeLLM(["updated", '{"should_store": false, "reason": "test"}'])
    service = AgentService(
        name="execution-context-service",
        id="execution-context-service",
        pattern="react",
        llm=llm,
        tools=[FakeTool()],
        agent_runtime="v2",
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
async def test_v2_adapter_emits_visible_trace_events() -> None:
    tracer = RecordingTracer()
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
async def test_v2_adapter_routes_dag_to_v2_dag() -> None:
    llm = FakeLLM(
        [
            dag_plan([{"id": "answer", "task": "Answer directly"}]),
            "dag done",
        ]
    )
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
    assert result["metadata"]["execution_type"] == "agent_v2_dag"
    assert result["agent_v2_result"]["pattern"] == "DAGPattern"


def test_v2_adapter_passes_dag_max_concurrency_to_pattern() -> None:
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
            name="dag",
            pattern="dag_plan_execute",
            llm=FakeLLM([]),
            dag_max_concurrency=2,
            skill_manager=NoSkillManager(),
        )
    )

    pattern, execution_type = adapter._build_pattern()

    assert execution_type == "agent_v2_dag"
    assert pattern.max_concurrency == 2


def test_v2_adapter_routes_auto_to_v2_auto() -> None:
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
            name="auto",
            pattern="auto",
            llm=FakeLLM([]),
            skill_manager=NoSkillManager(),
        )
    )

    pattern, execution_type = adapter._build_pattern()

    assert execution_type == "agent_v2_auto"
    assert pattern.__class__.__name__ == "AutoPattern"
    assert pattern.dag_pattern.max_concurrency == 4


@pytest.mark.asyncio
async def test_v2_adapter_executes_auto_final_answer() -> None:
    llm = FakeLLM(
        [auto_decision("final_answer", answer="hello", reason="Greeting only.")]
    )
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
    assert result["metadata"]["execution_type"] == "agent_v2_auto"
    assert result["agent_v2_result"]["pattern"] == "AutoPattern"


@pytest.mark.asyncio
async def test_v2_adapter_executes_auto_react() -> None:
    llm = FakeLLM(
        [
            auto_decision("react", reason="Needs ReAct."),
            "react done",
        ]
    )
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
    assert result["metadata"]["execution_type"] == "agent_v2_auto"
    assert result["agent_v2_result"]["auto_decision"]["action"] == "react"


@pytest.mark.asyncio
async def test_v2_adapter_executes_auto_plan_execute() -> None:
    llm = FakeLLM(
        [
            auto_decision("plan_execute", reason="Needs DAG."),
            dag_plan([{"id": "answer", "task": "Answer directly"}]),
            "dag done",
        ]
    )
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
    assert result["metadata"]["execution_type"] == "agent_v2_auto"
    assert result["agent_v2_result"]["auto_decision"]["action"] == "plan_execute"


@pytest.mark.asyncio
async def test_v2_adapter_exposes_pause_and_message_controls() -> None:
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
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
async def test_v2_adapter_exposes_cancel_control() -> None:
    llm = BlockingLLM()
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
async def test_v2_adapter_forwards_outbound_messages() -> None:
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
    adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
            name="outbound",
            pattern="react",
            llm=llm,
            outbound_message_handler=sent_messages.append,
            skill_manager=NoSkillManager(),
        )
    )

    result = await adapter.execute(task="Send progress", task_id="outbound-exec")

    assert result["success"] is True
    assert sent_messages == [
        {
            "type": "agent_message",
            "execution_id": "outbound-exec",
            "message": "Still working",
            "message_type": "progress",
            "expect_response": False,
            "metadata": {},
        }
    ]


@pytest.mark.asyncio
async def test_v2_adapter_resume_restores_from_tracer_after_restart() -> None:
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
    first_adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
    resumed_adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
    assert resumed["metadata"]["execution_type"] == "agent_v2_react"
    assert resumed_adapter.get_status("restart-exec") is None
    context_messages = resumed["agent_v2_result"]["context"].messages
    assert any(
        message.role == "user" and message.content == "Resume with concise answer."
        for message in context_messages
    )


@pytest.mark.asyncio
async def test_v2_adapter_posts_user_message_after_restart() -> None:
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
    first_adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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

    restarted_adapter = AgentV2ExecutionAdapter(
        AgentV2ExecutionConfig(
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
    context_messages = resumed["agent_v2_result"]["context"].messages
    assert any(
        message.role == "user"
        and message.content == "New message after process restart."
        for message in context_messages
    )
