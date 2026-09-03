from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.api import agents as agents_api


class _FakeLLM:
    def __init__(
        self,
        content: str = "请准确理解用户问题，核验关键事实，并用清晰的中文作答。",
    ) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"content": self.content}


class _FakeModelStorage:
    def __init__(self, llm: _FakeLLM) -> None:
        self.llm = llm

    def get_llm_by_id(self, model_id: str, user_id: int) -> _FakeLLM:
        return self.llm

    def get_configured_defaults(self, user_id: int | None = None) -> tuple[Any, ...]:
        return (self.llm, None, None, None)


@pytest.mark.asyncio
async def test_optimize_instructions_preserves_draft_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _FakeLLM()
    monkeypatch.setattr(
        agents_api,
        "UserAwareModelStorage",
        lambda db: _FakeModelStorage(llm),
    )

    result = await agents_api.optimize_instructions(
        agents_api.OptimizeInstructionsRequest(instructions="请用中文回答用户问题。"),
        SimpleNamespace(id=7),
        object(),
    )

    assert result == {
        "optimized_instructions": (
            "请准确理解用户问题，核验关键事实，并用清晰的中文作答。"
        )
    }
    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "Preserve the draft's natural language" in system_prompt
    assert (
        "Use the draft instructions as the baseline language authority" in system_prompt
    )
    assert "explicit or implicit target-language intent" in system_prompt
    assert "unless the draft instructions explicitly asks" not in system_prompt
    assert "Simplified Chinese versus Traditional Chinese" in system_prompt


@pytest.mark.asyncio
async def test_optimize_instructions_falls_back_on_wrong_language_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = "请分析用户问题，核验事实，并用中文给出清晰完整的回答。"
    llm = _FakeLLM(
        "Analyze the user's request, verify every fact, and return a complete "
        "answer in clear English."
    )
    monkeypatch.setattr(
        agents_api,
        "UserAwareModelStorage",
        lambda db: _FakeModelStorage(llm),
    )

    result = await agents_api.optimize_instructions(
        agents_api.OptimizeInstructionsRequest(instructions=draft),
        SimpleNamespace(id=7),
        object(),
    )

    assert result == {"optimized_instructions": draft}
