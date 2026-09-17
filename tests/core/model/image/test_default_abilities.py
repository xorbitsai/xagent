"""Tests for default_image_abilities."""

import pytest

from xagent.core.model.image.base import default_image_abilities

GENERATE = ["generate"]
BOTH = ["generate", "edit"]


@pytest.mark.parametrize(
    ("provider", "model_name", "expected"),
    [
        ("dashscope", "qwen-image-edit", BOTH),
        ("dashscope", "wanx-v1", GENERATE),
        # "3-pro" is Gemini vocabulary; it must not answer for dashscope.
        ("dashscope", "wanx-3-pro", GENERATE),
        ("gemini", "gemini-3-pro-image-preview", BOTH),
        ("gemini", "gemini-2.5-flash-image", GENERATE),
        ("  DashScope  ", "qwen-image-edit", BOTH),
        ("dashscope", "QWEN-IMAGE-EDIT", BOTH),
        # Whole-lineup providers answer the same for any name, so one path cannot
        # hand back a generate-only instance for a row the other calls editable.
        ("openai", "gpt-image-1", BOTH),
        ("openai", "some-unknown-name", BOTH),
        ("xinference", "my-image-edit", BOTH),
        ("xinference", "sd-3.5", BOTH),
        ("unknown-provider", "something-edit", GENERATE),
    ],
)
def test_default_image_abilities(
    provider: str, model_name: str, expected: list[str]
) -> None:
    assert default_image_abilities(provider, model_name) == expected
