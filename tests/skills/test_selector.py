from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.skills.selector import SkillSelector


class FakeLLM:
    def __init__(self, response: Any) -> None:
        self.response = response

    async def chat(self, **kwargs: Any) -> dict[str, str]:
        return {"content": json.dumps(self.response)}


SOURCE_BOUND_SKILL = {
    "name": "document-evidence-skill",
    "description": "Answer with source attribution from a knowledge base.",
    "when_to_use": "Use for uploaded files or provided documents.",
    "tags": ["rag"],
}


@pytest.mark.asyncio
async def test_selector_rejects_source_bound_skill_without_explicit_scope() -> None:
    selector = SkillSelector(
        FakeLLM(
            {
                "selected": True,
                "skill_name": "document-evidence-skill",
                "reasoning": "Looks like evidence retrieval.",
                "source_scope_required": True,
                "source_scope_satisfied": False,
                "source_scope_reasoning": "No knowledge base or document scope provided.",
            }
        )
    )

    selected = await selector.select(
        "Summarize recent AI supply chain attacks and list affected vendors.",
        [SOURCE_BOUND_SKILL],
    )

    assert selected is None


@pytest.mark.asyncio
async def test_selector_allows_source_bound_skill_with_explicit_scope() -> None:
    selector = SkillSelector(
        FakeLLM(
            {
                "selected": True,
                "skill_name": "document-evidence-skill",
                "reasoning": "The task names a knowledge base.",
                "source_scope_required": True,
                "source_scope_satisfied": True,
                "source_scope_reasoning": "The user explicitly named uploaded documents.",
            }
        )
    )

    selected = await selector.select(
        "Use the uploaded documents in the knowledge base to summarize risks.",
        [SOURCE_BOUND_SKILL],
    )

    assert selected == SOURCE_BOUND_SKILL


@pytest.mark.asyncio
async def test_selector_ignores_non_object_json_response() -> None:
    selector = SkillSelector(FakeLLM(["not", "an", "object"]))

    selected = await selector.select(
        "Use the uploaded documents in the knowledge base to summarize risks.",
        [SOURCE_BOUND_SKILL],
    )

    assert selected is None


@pytest.mark.asyncio
async def test_selector_normalizes_string_boolean_source_scope() -> None:
    selector = SkillSelector(
        FakeLLM(
            {
                "selected": "true",
                "skill_name": "document-evidence-skill",
                "reasoning": "The task needs evidence retrieval.",
                "source_scope_required": "true",
                "source_scope_satisfied": "false",
                "source_scope_reasoning": "No scoped source was provided.",
            }
        )
    )

    selected = await selector.select(
        "Summarize recent AI supply chain attacks and list affected vendors.",
        [SOURCE_BOUND_SKILL],
    )

    assert selected is None


@pytest.mark.asyncio
async def test_selector_treats_selected_string_false_as_not_selected() -> None:
    selector = SkillSelector(
        FakeLLM(
            {
                "selected": "false",
                "skill_name": "document-evidence-skill",
                "reasoning": "No direct match.",
            }
        )
    )

    selected = await selector.select(
        "Summarize recent AI supply chain attacks and list affected vendors.",
        [SOURCE_BOUND_SKILL],
    )

    assert selected is None
