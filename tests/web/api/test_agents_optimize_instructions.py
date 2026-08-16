from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.api import agents as agents_api


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"content": "请用中文回答用户问题。"}


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

    assert result == {"optimized_instructions": "请用中文回答用户问题。"}
    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "Preserve the draft's natural language" in system_prompt
    assert "same natural language as the draft instructions" in system_prompt
    assert "Simplified Chinese versus Traditional Chinese" in system_prompt
