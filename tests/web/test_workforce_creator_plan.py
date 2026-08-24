"""Tests for ReAct-based creation of a Workforce and its agents."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from xagent.web.services import workforce_creator, workforce_prompt_runtime
from xagent.web.services.workforce_creator import (
    generate_workforce_creation_plan,
    get_localized_description,
)
from xagent.web.services.workforce_prompt_runtime import (
    MAX_WORKFORCE_BUILDER_AGENTS,
    MAX_WORKFORCE_BUILDER_EXISTING_AGENTS,
    ListAvailableAgentsTool,
    WorkforcePromptBuilderError,
    WorkforcePromptBuilderState,
    WorkforcePromptBuilderUnavailableError,
    _validate_builder_plan_language,
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
    assert all(agent["tool_categories"] == [] for agent in plan["created_agents"])
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


def test_builder_state_enforces_staged_agent_limit() -> None:
    state = WorkforcePromptBuilderState.from_agents([])
    for index in range(MAX_WORKFORCE_BUILDER_AGENTS):
        result = state.stage_agent(
            {
                "name": f"Agent {index}",
                "description": "Performs one bounded role.",
                "instructions": "Complete the assigned bounded role.",
            }
        )
        assert result["status"] == "success"

    rejected = state.stage_agent(
        {
            "name": "One agent too many",
            "description": "Must be rejected.",
            "instructions": "Do not stage this agent.",
        }
    )

    assert rejected["status"] == "error"
    assert "staged-agent limit" in rejected["message"]


def test_builder_language_validation_rejects_wrong_script_before_persistence() -> None:
    with pytest.raises(WorkforcePromptBuilderError, match="workforce.name"):
        _validate_builder_plan_language(
            prompt="创建一个用于分析市场数据的工作组。",
            plan={
                "name": "Market Research Workforce",
                "description": (
                    "Research the market, compare all evidence, and produce a "
                    "complete English report for the user."
                ),
                "created_agents": [],
                "workers": [],
                "builder_response": "The requested Workforce is ready.",
            },
        )


def test_builder_language_validation_allows_explicit_target_language() -> None:
    _validate_builder_plan_language(
        prompt=(
            "Create a research Workforce, but write every persisted field and "
            "the final response in Chinese."
        ),
        plan={
            "name": "市场研究工作组",
            "description": "研究市场数据并总结关键结论。",
            "created_agents": [
                {
                    "name": "市场研究员",
                    "description": "收集并核验市场证据。",
                    "instructions": "检索可靠资料并用中文总结。",
                }
            ],
            "workers": [
                {
                    "alias": "市场研究",
                    "assignment_instructions": "收集并核验市场证据。",
                }
            ],
            "builder_response": "工作组已完成配置。",
        },
    )


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
    assert plan["builder_response"] == "工作组已完成配置。"
    assert len(llm.calls) == 6
    first_tool_names = {schema["function"]["name"] for schema in llm.calls[0]["tools"]}
    assert first_tool_names == {
        "ask_user_question",
        "create_agent",
        "create_workforce",
        "final_answer",
        "list_available_agents",
        "list_available_skills",
        "list_tool_categories",
        "send_message",
    }


def _recording_agent_service_class(captured_kwargs: dict[str, Any]) -> type:
    """A fake AgentService that records its constructor kwargs and reports
    a bare completion with no finalized Workforce - used by the two voice
    tests below, which only need the constructor call to have happened."""

    class RecordingAgentService:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def set_allowed_skills(self, _allowed_skills: list[str]) -> None:
            pass

        async def execute_task(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"success": True, "completion_outcome": "completed"}

    return RecordingAgentService


@pytest.mark.asyncio
async def test_build_workforce_prompt_plan_applies_voice_to_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """builder_response is a free-text assistant reply persisted into the
    Workforce's conversation - the same "every agent this user talks to"
    voice policy that Builder chat and task chat already apply must reach
    this runtime's own AgentService too."""
    captured_kwargs: dict[str, Any] = {}
    monkeypatch.setattr(
        workforce_prompt_runtime,
        "AgentService",
        _recording_agent_service_class(captured_kwargs),
    )
    monkeypatch.setattr(
        workforce_prompt_runtime,
        "extract_assistant_message",
        lambda _result: "工作组已完成配置。",
    )

    with pytest.raises(WorkforcePromptBuilderError):
        # The mock never calls create_workforce, so to_plan() raises - the
        # assertion below only needs the AgentService construction to have
        # already happened, which it has by this point.
        await build_workforce_prompt_plan(
            prompt="创建一个产品研究工作组。",
            llm=FakeLLM([]),
            available_agents=[],
            voice="warm",
        )

    system_prompt = captured_kwargs["system_prompt"]
    assert system_prompt.startswith(
        workforce_prompt_builder_system_prompt() + "\n\n## OUTPUT VOICE\n"
    )
    assert "Empathetic and reassuring" in system_prompt


@pytest.mark.asyncio
async def test_build_workforce_prompt_plan_without_voice_leaves_prompt_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    monkeypatch.setattr(
        workforce_prompt_runtime,
        "AgentService",
        _recording_agent_service_class(captured_kwargs),
    )

    with pytest.raises(WorkforcePromptBuilderError):
        await build_workforce_prompt_plan(
            prompt="创建一个产品研究工作组。",
            llm=FakeLLM([]),
            available_agents=[],
        )

    assert captured_kwargs["system_prompt"] == workforce_prompt_builder_system_prompt()


@pytest.mark.asyncio
async def test_available_agent_tool_bounds_each_model_result() -> None:
    state = WorkforcePromptBuilderState.from_agents(
        [
            {
                "agent_id": index,
                "name": f"Research Agent {index}",
                "description": "Collects and verifies evidence.",
                "status": "published",
            }
            for index in range(1, 31)
        ]
    )

    result = await ListAvailableAgentsTool(state).run_json_async(
        {"query": "research", "limit": 7}
    )

    assert len(result["agents"]) == 7
    assert result["total_matches"] == 30
    assert result["has_more"] is True


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
async def test_react_builder_wraps_execution_boundary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableAgentService:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def set_allowed_skills(self, _allowed_skills: list[str]) -> None:
            pass

        async def execute_task(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("registry initialization failed")

    monkeypatch.setattr(
        workforce_prompt_runtime,
        "AgentService",
        UnavailableAgentService,
    )

    with pytest.raises(
        WorkforcePromptBuilderUnavailableError,
        match="runtime is unavailable",
    ):
        await build_workforce_prompt_plan(
            prompt="Create a research Workforce.",
            llm=FakeLLM([]),
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
        lambda _db, _user, **_kwargs: [],
    )

    with pytest.raises(HTTPException) as exc_info:
        await generate_workforce_creation_plan(
            object(),
            type("User", (), {"id": 7})(),
            "创建一个研究工作组",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "workforce_prompt_builder_unavailable"


@pytest.mark.asyncio
async def test_generation_releases_database_before_react_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM([])
    events: list[str] = []
    catalog_limits: list[int | None] = []

    class ModelStorage:
        def __init__(self, _db: object) -> None:
            pass

        def get_configured_defaults(self, _user_id: int | None) -> tuple[Any, ...]:
            return (llm, None, None, None)

    async def fake_build_workforce_prompt_plan(**kwargs: Any) -> dict[str, Any]:
        events.append("runtime")
        assert kwargs["available_agents"] == [
            {
                "agent_id": 12,
                "name": "Researcher",
                "description": "Collects evidence.",
                "status": "published",
            }
        ]
        return {"name": "Research Workforce"}

    monkeypatch.setattr(workforce_creator, "UserAwareModelStorage", ModelStorage)

    def list_agents(
        _db: object,
        _user: object,
        *,
        limit: int | None = None,
    ) -> list[SimpleNamespace]:
        catalog_limits.append(limit)
        return [
            SimpleNamespace(
                id=12,
                name="Researcher",
                description="Collects evidence.",
                status=SimpleNamespace(value="published"),
            )
        ]

    monkeypatch.setattr(
        workforce_creator,
        "list_accessible_published_agents",
        list_agents,
    )
    monkeypatch.setattr(
        workforce_creator,
        "release_db_connection_if_clean",
        lambda _db: events.append("release") or True,
    )
    monkeypatch.setattr(
        workforce_creator,
        "build_workforce_prompt_plan",
        fake_build_workforce_prompt_plan,
    )

    result = await generate_workforce_creation_plan(
        object(),
        SimpleNamespace(id=7),
        "Create a research Workforce.",
    )

    assert result == {"name": "Research Workforce"}
    assert events == ["release", "runtime"]
    assert catalog_limits == [MAX_WORKFORCE_BUILDER_EXISTING_AGENTS]


@pytest.mark.asyncio
async def test_generation_resolves_voice_from_the_user_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_workforce_creation_plan must resolve voice from the live
    `user` row and thread it into build_workforce_prompt_plan - this is
    the actual wiring under test in G11 (the two build_workforce_prompt_plan
    tests above only exercise that function directly with an explicit
    voice= kwarg, not this caller's own resolution step)."""
    llm = FakeLLM([])
    captured_voice: list[str | None] = []

    class ModelStorage:
        def __init__(self, _db: object) -> None:
            pass

        def get_configured_defaults(self, _user_id: int | None) -> tuple[Any, ...]:
            return (llm, None, None, None)

    async def fake_build_workforce_prompt_plan(**kwargs: Any) -> dict[str, Any]:
        captured_voice.append(kwargs["voice"])
        return {"name": "Research Workforce"}

    monkeypatch.setattr(workforce_creator, "UserAwareModelStorage", ModelStorage)
    monkeypatch.setattr(
        workforce_creator,
        "list_accessible_published_agents",
        lambda _db, _user, **_kwargs: [],
    )
    monkeypatch.setattr(
        workforce_creator, "release_db_connection_if_clean", lambda _db: True
    )
    monkeypatch.setattr(
        workforce_creator,
        "build_workforce_prompt_plan",
        fake_build_workforce_prompt_plan,
    )

    await generate_workforce_creation_plan(
        object(),
        SimpleNamespace(id=7, preferences={"voice": "warm"}),
        "Create a research Workforce.",
    )

    assert captured_voice == ["warm"]


@pytest.mark.asyncio
async def test_generation_maps_runtime_boundary_failure_to_stable_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM([])

    class ModelStorage:
        def __init__(self, _db: object) -> None:
            pass

        def get_configured_defaults(self, _user_id: int | None) -> tuple[Any, ...]:
            return (llm, None, None, None)

    async def unavailable_builder(**_kwargs: Any) -> dict[str, Any]:
        raise WorkforcePromptBuilderUnavailableError("adapter setup failed")

    monkeypatch.setattr(workforce_creator, "UserAwareModelStorage", ModelStorage)
    monkeypatch.setattr(
        workforce_creator,
        "list_accessible_published_agents",
        lambda _db, _user, **_kwargs: [],
    )
    monkeypatch.setattr(
        workforce_creator,
        "release_db_connection_if_clean",
        lambda _db: True,
    )
    monkeypatch.setattr(
        workforce_creator,
        "build_workforce_prompt_plan",
        unavailable_builder,
    )

    with pytest.raises(HTTPException) as exc_info:
        await generate_workforce_creation_plan(
            object(),
            SimpleNamespace(id=7),
            "Create a research Workforce.",
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
