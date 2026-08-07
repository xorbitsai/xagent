"""Unit tests for the Workforce creation-plan shape (#800).

Workforce-level manager instructions were removed: the cleaned plan must not
carry a top-level ``manager_instructions`` key, while the manager spec itself
keeps its instructions.
"""

from xagent.web.services.workforce_creator import (
    _clean_creation_plan,
    _fallback_creation_plan,
    get_localized_description,
)


def test_clean_creation_plan_has_no_top_level_manager_instructions() -> None:
    plan = _clean_creation_plan(
        {
            "name": "Research Workforce",
            "description": "Coordinates research",
            "manager": {
                "name": "Research Manager",
                "description": "Coordinates workers",
                "instructions": "Delegate and synthesize.",
            },
            "manager_instructions": "Legacy top-level value",
            "workers": [],
        },
        available_agent_ids=set(),
        prompt="research",
    )

    assert "manager_instructions" not in plan
    assert plan["manager"]["instructions"] == "Delegate and synthesize."


def test_fallback_creation_plan_has_no_top_level_manager_instructions() -> None:
    plan = _fallback_creation_plan("research assistant", agents=[])

    assert "manager_instructions" not in plan
    assert plan["manager"]["instructions"]


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
