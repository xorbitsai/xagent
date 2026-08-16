"""Tests for ReAct-based creation of a Workforce and its agents."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import HTTPException

from xagent.web.services import workforce_creator
from xagent.web.services.workforce_creator import (
    generate_workforce_creation_plan,
    get_localized_description,
)
from xagent.web.services.workforce_prompt_runtime import (
    WorkforcePromptBuilderError,
    WorkforcePromptBuilderState,
    build_workforce_prompt_plan,
    workforce_prompt_builder_system_prompt,
)


class FakeLLM:
    model_name = "fake-workforce-builder"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "id": call_id,
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ]
    }


def _stage_agent(
    state: WorkforcePromptBuilderState,
    *,
    name: str,
    instructions: str,
    mode: str = "balanced",
) -> str:
    result = state.stage_agent(
        {
            "name": name,
            "description": f"何时使用{name}",
            "instructions": instructions,
            "execution_mode": mode,
        }
    )
    assert result["status"] == "success"
    return str(result["agent_ref"])


def test_builder_prompt_requires_multi_agent_react_and_language_harness() -> None:
    prompt = workforce_prompt_builder_system_prompt()
    normalized_prompt = " ".join(prompt.split())

    assert "not a one-shot JSON planning task" in normalized_prompt
    assert "call create_agent separately" in normalized_prompt
    assert "every create_agent call succeeded" in normalized_prompt
    assert "one transaction" in normalized_prompt
    assert "Simplified Chinese" in prompt
    assert "Traditional Chinese" in prompt


def test_builder_state_requires_all_agents_before_finalization() -> None:
    state = WorkforcePromptBuilderState.from_agents([])
    manager_ref = _stage_agent(
        state,
        name="研究经理",
        instructions="拆解任务、委派并综合结果。",
        mode="think",
    )
    worker_ref = _stage_agent(
        state,
        name="研究员",
        instructions="检索并核验资料。",
    )

    before_listing = state.finalize(
        {
            "name": "研究工作组",
            "description": "完成研究任务。",
            "manager_agent_ref": manager_ref,
            "workers": [
                {
                    "agent_ref": worker_ref,
                    "assignment_instructions": "搜集并核验资料。",
                }
            ],
        }
    )
    assert before_listing["status"] == "error"

    state.listed_existing_agents = True
    finalized = state.finalize(
        {
            "name": "研究工作组",
            "description": "完成研究任务。",
            "manager_agent_ref": manager_ref,
            "workers": [
                {
                    "agent_ref": worker_ref,
                    "assignment_instructions": "搜集并核验资料。",
                }
            ],
        }
    )

    assert finalized["status"] == "success"
    plan = state.to_plan()
    assert len(plan["created_agents"]) == 2
    assert plan["manager"]["agent_ref"] == manager_ref
    assert plan["workers"][0]["agent_ref"] == worker_ref


def test_builder_state_rejects_unused_or_failed_staged_agents() -> None:
    state = WorkforcePromptBuilderState.from_agents([])
    failed = state.stage_agent(
        {"name": "无效代理", "description": "说明", "instructions": ""}
    )
    assert failed["status"] == "error"
    assert state.created_agents == {}

    manager_ref = _stage_agent(
        state,
        name="经理",
        instructions="协调工作。",
        mode="think",
    )
    used_worker_ref = _stage_agent(
        state,
        name="研究员",
        instructions="执行研究。",
    )
    _stage_agent(state, name="未使用代理", instructions="执行其他工作。")
    state.listed_existing_agents = True

    result = state.finalize(
        {
            "name": "工作组",
            "description": "执行任务。",
            "manager_agent_ref": manager_ref,
            "workers": [
                {
                    "agent_ref": used_worker_ref,
                    "assignment_instructions": "执行研究。",
                }
            ],
        }
    )

    assert result["status"] == "error"
    assert "Unused refs" in result["message"]
    with pytest.raises(WorkforcePromptBuilderError):
        state.to_plan()


@pytest.mark.asyncio
async def test_react_builder_creates_multiple_agents_before_workforce() -> None:
    llm = FakeLLM(
        [
            _tool_call("list", "list_available_agents", {}),
            _tool_call(
                "manager",
                "create_agent",
                {
                    "name": "产品研究经理",
                    "description": "协调产品研究工作。",
                    "instructions": "拆解目标、委派任务、比较证据并综合结论。",
                    "execution_mode": "think",
                },
            ),
            _tool_call(
                "researcher",
                "create_agent",
                {
                    "name": "市场研究员",
                    "description": "用于收集并核验市场资料。",
                    "instructions": "检索可靠资料，交叉核验并记录来源。",
                    "tool_categories": ["web_search"],
                },
            ),
            _tool_call(
                "analyst",
                "create_agent",
                {
                    "name": "产品分析师",
                    "description": "用于比较产品能力和差异。",
                    "instructions": "建立比较维度，分析差异并指出风险。",
                },
            ),
            _tool_call(
                "workforce",
                "create_workforce",
                {
                    "name": "产品研究工作组",
                    "description": "研究市场并形成产品分析结论。",
                    "manager_agent_ref": "new:1",
                    "workers": [
                        {
                            "agent_ref": "new:2",
                            "alias": "市场研究",
                            "assignment_instructions": "收集并核验市场证据。",
                        },
                        {
                            "agent_ref": "new:3",
                            "alias": "产品分析",
                            "assignment_instructions": "比较产品并总结差异。",
                        },
                    ],
                },
            ),
            _tool_call(
                "final",
                "final_answer",
                {
                    "response_language": "Simplified Chinese",
                    "answer": "工作组已完成配置。",
                    "outcome": "completed",
                },
            ),
        ]
    )

    plan = await build_workforce_prompt_plan(
        prompt="创建一个产品研究工作组，需要市场研究和产品分析。",
        llm=llm,
        available_agents=[],
    )

    assert [agent["agent_ref"] for agent in plan["created_agents"]] == [
        "new:1",
        "new:2",
        "new:3",
    ]
    assert [worker["agent_ref"] for worker in plan["workers"]] == [
        "new:2",
        "new:3",
    ]
    assert plan["name"] == "产品研究工作组"
    assert len(llm.calls) == 6
    first_tool_names = {schema["function"]["name"] for schema in llm.calls[0]["tools"]}
    assert {"create_agent", "create_workforce", "list_available_agents"} <= (
        first_tool_names
    )


@pytest.mark.asyncio
async def test_react_builder_does_not_report_success_without_finalization() -> None:
    llm = FakeLLM(
        [
            _tool_call(
                "final",
                "final_answer",
                {
                    "response_language": "English",
                    "answer": "Done.",
                    "outcome": "completed",
                },
            )
        ]
    )

    with pytest.raises(
        WorkforcePromptBuilderError,
        match="did not call create_workforce successfully",
    ):
        await build_workforce_prompt_plan(
            prompt="Create a research Workforce.",
            llm=llm,
            available_agents=[],
        )


@pytest.mark.asyncio
async def test_generation_fails_clearly_when_no_model_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoModelStorage:
        def __init__(self, _db: object) -> None:
            pass

        def get_configured_defaults(
            self, _user_id: int | None
        ) -> tuple[None, None, None, None]:
            return None, None, None, None

    monkeypatch.setattr(workforce_creator, "UserAwareModelStorage", NoModelStorage)
    monkeypatch.setattr(
        workforce_creator,
        "list_accessible_published_agents",
        lambda _db, _user: [],
    )

    with pytest.raises(HTTPException) as exc_info:
        await generate_workforce_creation_plan(
            object(),
            type("User", (), {"id": 7})(),
            "创建一个研究工作组",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "workforce_prompt_builder_unavailable"


def test_get_localized_description_prefers_english() -> None:
    template = {"descriptions": {"en": "English description", "zh": "中文描述"}}

    assert get_localized_description(template) == "English description"


def test_get_localized_description_falls_back_to_another_locale() -> None:
    """`_parse_yaml_file` only requires an 'en' *key* to be present, not a
    non-empty value, so a template can carry an empty English description
    alongside a populated one in another locale - the Workforce's
    description shouldn't end up empty just because English is blank."""
    template = {"descriptions": {"en": "", "zh": "中文描述"}}

    assert get_localized_description(template) == "中文描述"


def test_get_localized_description_returns_none_when_nothing_is_populated() -> None:
    assert get_localized_description({"descriptions": {"en": "", "zh": ""}}) is None
    assert get_localized_description({"descriptions": "not a dict"}) is None
    assert get_localized_description({}) is None
