"""Tests for the typed model-provider exceptions."""

import pytest

from xagent.core.model.chat.exceptions import (
    LLMRetryableError,
    ModelProviderError,
    ModelProviderRetryableError,
)


def _error(**overrides):
    fields = {
        "prefix": "OpenAI API error",
        "kind": "access_denied",
        "status_code": 403,
        "provider_code": "provider_code_4204",
        "provider_message": "Model is decommissioned",
        "sdk_message": "Error code: 403 - Model is decommissioned",
        "details": [],
    }
    fields.update(overrides)
    return ModelProviderError(**fields)


@pytest.mark.parametrize(
    ("status_code", "details", "expected"),
    [
        (
            403,
            ["provider_name=upstream-a", "provider_raw=raw body"],
            "OpenAI API error (403): Error code: 403 - Model is decommissioned"
            " | provider_name=upstream-a | provider_raw=raw body",
        ),
        (
            403,
            [],
            "OpenAI API error (403): Error code: 403 - Model is decommissioned",
        ),
        (
            None,
            [],
            "OpenAI API error: Error code: 403 - Model is decommissioned",
        ),
        (
            None,
            ["provider_name=upstream-a"],
            "OpenAI API error: Error code: 403 - Model is decommissioned"
            " | provider_name=upstream-a",
        ),
    ],
    ids=[
        "status-and-details",
        "status-only",
        "neither",
        "details-only",
    ],
)
def test_text_covers_every_status_and_details_combination(
    status_code, details, expected
):
    exc = ModelProviderError(
        prefix="OpenAI API error",
        kind="access_denied",
        status_code=status_code,
        provider_code="provider_code_4204",
        provider_message="Model is decommissioned",
        sdk_message="Error code: 403 - Model is decommissioned",
        details=details,
    )

    assert str(exc) == expected
    assert exc.args == (expected,)


def test_structured_fields_is_a_literal_dict_and_a_fresh_object_each_call():
    exc = _error()

    first = exc.structured_fields()
    assert first == {
        "kind": "access_denied",
        "status_code": 403,
        "provider_code": "provider_code_4204",
        "message": "Model is decommissioned",
    }

    first["kind"] = "tampered"
    second = exc.structured_fields()
    assert second is not first
    assert second["kind"] == "access_denied"


def test_provider_error_is_not_retryable_by_class():
    assert ModelProviderRetryableError.__mro__[:4] == (
        ModelProviderRetryableError,
        ModelProviderError,
        LLMRetryableError,
        RuntimeError,
    )
    assert not issubclass(ModelProviderError, LLMRetryableError)
    assert not isinstance(_error(), LLMRetryableError)
    assert isinstance(_error(), RuntimeError)
