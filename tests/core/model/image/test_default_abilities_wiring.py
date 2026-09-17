"""The fallback sites actually reach default_image_abilities.

test_default_abilities.py covers the pure function; without these, deleting any
`or default_image_abilities(...)` wiring leaves the suite green.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.core.model.image.adapter import get_image_model_instance


def _db_model(**overrides: Any) -> SimpleNamespace:
    row = {
        "model_provider": "dashscope",
        "model_name": "qwen-image-edit",
        "api_key": "test-key",
        "base_url": None,
        "abilities": None,
        "timeout": 300.0,
        "max_retries": 3,
    }
    row.update(overrides)
    return SimpleNamespace(**row)


def test_adapter_infers_abilities_for_a_null_row() -> None:
    model = get_image_model_instance(_db_model())
    assert model.abilities == ["generate", "edit"]


def test_adapter_leaves_a_generate_only_name_alone() -> None:
    model = get_image_model_instance(_db_model(model_name="wanx-v1"))
    assert model.abilities == ["generate"]


def test_adapter_never_overrides_a_declared_list() -> None:
    model = get_image_model_instance(
        _db_model(model_name="qwen-image-edit", abilities=["generate"])
    )
    assert model.abilities == ["generate"]


def _service_abilities(
    mocker: Any, provider: str, model_name: str, abilities: Any = None
) -> list[str]:
    from xagent.web.services import model_service

    db_model = SimpleNamespace(
        id=1,
        model_id="img-1",
        category="image",
        is_active=True,
        model_provider=provider,
        model_name=model_name,
        api_key="test-key",
        base_url="https://example.invalid",
        abilities=abilities,
        description=None,
    )
    session = mocker.Mock()
    session.query.return_value.filter.return_value.all.return_value = [db_model]

    models = model_service.get_image_models(session)
    return [m.abilities for m in models.values()][0]


@pytest.mark.parametrize(
    ("provider", "model_name", "expected"),
    [
        ("dashscope", "qwen-image-edit", ["generate", "edit"]),
        ("dashscope", "wanx-v1", ["generate"]),
        # The gemini branch is a separate call site; without a case here reverting
        # it to a literal leaves this suite green.
        ("gemini", "gemini-3-pro-image-preview", ["generate", "edit"]),
        ("gemini", "gemini-2.5-flash-image", ["generate"]),
    ],
)
def test_model_service_infers_abilities_for_a_null_row(
    mocker: Any, provider: str, model_name: str, expected: list[str]
) -> None:
    assert _service_abilities(mocker, provider, model_name) == expected


def test_model_service_never_overrides_a_declared_list(mocker: Any) -> None:
    assert _service_abilities(
        mocker, "dashscope", "qwen-image-edit", abilities=["generate"]
    ) == ["generate"]


@pytest.mark.parametrize(
    ("provider", "model_name", "expected"),
    [
        ("openai", "gpt-image-1", ["generate", "edit"]),
        ("xinference", "sd-3.5", ["generate", "edit"]),
        ("gemini", "gemini-3-pro-image-preview", ["generate", "edit"]),
        ("dashscope", "qwen-image-edit", ["generate", "edit"]),
        ("dashscope", "wanx-v1", ["generate"]),
    ],
)
def test_both_paths_agree_on_a_null_row(
    mocker: Any, provider: str, model_name: str, expected: list[str]
) -> None:
    """A row the two paths read differently is the bug this helper exists to stop.

    Such a row, picked as the default edit model, is built generate-only by one
    path and editable by the other, so _get_edit_model rejects the operator's
    choice and silently serves a different model.

    Both sides are pinned to `expected` rather than only compared to each other,
    so a symmetric regression cannot pass.
    """
    adapter_model = get_image_model_instance(
        _db_model(model_provider=provider, model_name=model_name)
    )
    service_abilities = _service_abilities(mocker, provider, model_name)
    assert adapter_model.abilities == expected
    assert service_abilities == expected
