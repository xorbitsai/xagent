"""Contracts for the wire-safe projection of connector-runtime failures.

The projection has two halves and both are pinned here: the message adapter
(fail-closed on anything that is not a ``ConnectorRuntimeError``) and the
code projector, which is fail-closed the same way.
"""

from __future__ import annotations

import pytest

from xagent.core.model.chat.exceptions import MODEL_PROVIDER_FAILURE_KINDS
from xagent.core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.services import client_error_messages
from xagent.web.services.client_error_messages import (
    CLIENT_SAFE_MODEL_ERROR,
    CLIENT_SAFE_TASK_FAILURE,
    MODEL_ERROR_CLIENT_MESSAGE_MAX_CHARS,
    MODEL_ERROR_KIND_PHRASES,
    MODEL_ERROR_PROVIDER_CODE_MAX_CHARS,
    ClientErrorCode,
    client_error_message,
    connector_runtime_client_code,
    connector_runtime_client_message,
    model_error_client_projection,
    with_task_reference,
)

# --------------------------------------------------------------------------
# connector_runtime_client_message
# --------------------------------------------------------------------------


def test_client_message_returns_the_curated_safe_message() -> None:
    error = ConnectorRuntimeError(
        "missing_runtime_context",
        "Required connector runtime context is missing.",
    )

    assert (
        connector_runtime_client_message(error)
        == "Required connector runtime context is missing."
    )


@pytest.mark.parametrize("safe_message", ["", "   ", "\n\t"])
def test_client_message_falls_back_on_a_blank_safe_message(safe_message: str) -> None:
    error = ConnectorRuntimeError("missing_runtime_context", safe_message)

    assert connector_runtime_client_message(error) == CLIENT_SAFE_TASK_FAILURE


@pytest.mark.parametrize(
    "error",
    [
        ValueError("secret-token-xyz"),
        KeyError("secret-token-xyz"),
        RuntimeError("secret-token-xyz"),
        RequiredMCPUnavailableError("secret-token-xyz"),
    ],
)
def test_client_message_is_fail_closed_for_an_incidental_exception(
    error: BaseException,
) -> None:
    """The specific name is not the gate; the isinstance check is."""

    assert connector_runtime_client_message(error) == CLIENT_SAFE_TASK_FAILURE


# --------------------------------------------------------------------------
# connector_runtime_client_code
# --------------------------------------------------------------------------


def test_client_code_projects_a_connector_runtime_error() -> None:
    error = ConnectorRuntimeError(
        "missing_runtime_context",
        "Required connector runtime context is missing.",
    )

    assert connector_runtime_client_code(error) == "missing_runtime_context"


@pytest.mark.parametrize(
    "error",
    [
        ValueError("secret-token-xyz"),
        KeyError("secret-token-xyz"),
        RuntimeError("secret-token-xyz"),
        RequiredMCPUnavailableError("secret-token-xyz"),
    ],
)
def test_client_code_is_fail_closed_for_an_incidental_exception(
    error: BaseException,
) -> None:
    """The specific name is not the gate; the isinstance check is."""

    assert connector_runtime_client_code(error) is None


# --------------------------------------------------------------------------
# with_task_reference
# --------------------------------------------------------------------------


def test_with_task_reference_appends_positive_integer_id() -> None:
    assert with_task_reference("x", 7) == "x (Task ID: 7)"
    assert (
        with_task_reference(CLIENT_SAFE_TASK_FAILURE, 123)
        == "Task execution failed. (Task ID: 123)"
    )


@pytest.mark.parametrize("task_id", [None, 0, -1, True, "7", 7.0])
def test_with_task_reference_ignores_missing_or_invalid_id(task_id: object) -> None:
    assert with_task_reference("x", task_id) == "x"  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# model_error_client_projection
# --------------------------------------------------------------------------

HEAD = "Model provider call failed"


def _model_error(
    kind: object = "unknown",
    status_code: object = None,
    provider_code: object = None,
    message: object = None,
) -> dict[str, object]:
    return {
        "kind": kind,
        "status_code": status_code,
        "provider_code": provider_code,
        "message": message,
    }


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(
            _model_error(
                "access_denied", 403, "provider_code_4204", "Model is decommissioned"
            ),
            f"{HEAD} (403 provider_code_4204): Model is decommissioned",
            id="403-code-and-body",
        ),
        pytest.param(
            _model_error(
                "authentication_failed",
                401,
                "invalid_api_key",
                "Incorrect API key provided: sk-test-key-echo",
            ),
            f"{HEAD} (401 invalid_api_key): the credential was rejected.",
            id="401-body-withheld",
        ),
        pytest.param(
            _model_error("timeout"),
            f"{HEAD}: the request timed out.",
            id="timeout-without-status",
        ),
        pytest.param(
            _model_error("unknown", None, "invalid_prompt", "echoed prompt text"),
            f"{HEAD} (invalid_prompt).",
            id="status-none-code-only-no-body",
        ),
        pytest.param(
            _model_error("server_error", 503, None, "upstream unavailable"),
            f"{HEAD} (503): upstream unavailable",
            id="503-body",
        ),
        pytest.param(
            _model_error("server_error", 503, None, None),
            f"{HEAD} (503): the provider returned a server error.",
            id="503-without-body-uses-phrase",
        ),
        pytest.param(
            _model_error("rejected", 402, "insufficient_quota", "Quota exhausted"),
            f"{HEAD} (402 insufficient_quota): the provider rejected the request.",
            id="402-body-withheld",
        ),
        pytest.param(
            _model_error("unknown", 404, None, None),
            f"{HEAD} (404).",
            id="phrase-less-kind-without-body",
        ),
        pytest.param(
            _model_error("connection_failed"),
            f"{HEAD}: the provider could not be reached.",
            id="connection-failed",
        ),
    ],
)
def test_projection_text_matrix(value: dict[str, object], expected: str) -> None:
    projection = model_error_client_projection(value)

    assert projection == {
        "error_code": ClientErrorCode.MODEL_ERROR.value,
        "error_message": expected,
        "kind": value["kind"],
        "status_code": value["status_code"],
        "provider_code": value["provider_code"],
    }
    assert "None" not in projection["error_message"]


def test_projection_of_empty_dict_equals_the_fixed_fallback() -> None:
    projection = model_error_client_projection({})

    assert projection is not None
    assert projection["error_message"] == "Model provider call failed."
    assert projection["error_message"] == CLIENT_SAFE_MODEL_ERROR
    assert projection["kind"] == "unknown"
    assert projection["status_code"] is None
    assert projection["provider_code"] is None
    assert client_error_message(ClientErrorCode.MODEL_ERROR) == CLIENT_SAFE_MODEL_ERROR


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param("not a dict", id="str"),
        pytest.param(["kind"], id="list"),
        pytest.param(
            _model_error("bad_request", 400, "invalid_request", "Bad"), id="400"
        ),
        pytest.param(_model_error("bad_request", 400), id="400-bare"),
    ],
)
def test_projection_is_withheld_for_non_dicts_and_http_400(value: object) -> None:
    assert model_error_client_projection(value) is None


class _DictSubclass(dict):  # type: ignore[type-arg]
    pass


def test_projection_requires_an_exact_dict() -> None:
    assert model_error_client_projection(_DictSubclass(kind="timeout")) is None


@pytest.mark.parametrize(
    "status_code",
    [True, False, "403", 403.0, 99, 600, -1, None],
)
def test_projection_treats_a_non_int_or_out_of_range_status_as_missing(
    status_code: object,
) -> None:
    projection = model_error_client_projection(
        _model_error("access_denied", status_code, "provider_code_4204", "Body text")
    )

    assert projection is not None
    assert projection["status_code"] is None
    assert projection["error_message"] == (
        f"{HEAD} (provider_code_4204): access to the model was denied."
    )


@pytest.mark.parametrize("kind", ["not-a-kind", None, 7, ["timeout"], ""])
def test_projection_maps_an_unrecognised_kind_to_unknown(kind: object) -> None:
    projection = model_error_client_projection(_model_error(kind, 503, None, None))

    assert projection is not None
    assert projection["kind"] == "unknown"
    assert projection["error_message"] == f"{HEAD} (503)."


@pytest.mark.parametrize(
    "provider_code",
    [
        "x" * 65,
        "has space",
        "semi;colon",
        "",
        4204,
        None,
        ["a"],
        "line\nbreak",
        "code\n",
    ],
)
def test_projection_drops_a_provider_code_that_fails_the_shape_check(
    provider_code: object,
) -> None:
    projection = model_error_client_projection(
        _model_error("access_denied", 403, provider_code, "Body")
    )

    assert projection is not None
    assert projection["provider_code"] is None
    assert projection["error_message"] == f"{HEAD} (403): Body"


def test_projection_keeps_a_provider_code_of_exactly_the_maximum_length() -> None:
    code = "c" * MODEL_ERROR_PROVIDER_CODE_MAX_CHARS
    projection = model_error_client_projection(
        _model_error("access_denied", 403, code, None)
    )

    assert projection is not None
    assert projection["provider_code"] == code


def test_projection_caps_the_provider_message_at_exactly_200_characters() -> None:
    projection = model_error_client_projection(
        _model_error("server_error", 500, None, "m" * 300)
    )

    assert projection is not None
    prefix = f"{HEAD} (500): "
    assert projection["error_message"].startswith(prefix)
    assert len(projection["error_message"]) - len(prefix) == 200
    assert MODEL_ERROR_CLIENT_MESSAGE_MAX_CHARS == 200


@pytest.mark.parametrize("status_code", [403, 404, 408, 409, 429, 500, 502, 599])
def test_projection_carries_the_body_for_allowlisted_statuses(
    status_code: int,
) -> None:
    projection = model_error_client_projection(
        _model_error("unknown", status_code, None, "visible body")
    )

    assert projection is not None
    assert projection["error_message"] == f"{HEAD} ({status_code}): visible body"


@pytest.mark.parametrize("status_code", [None, 100, 200, 302, 401, 402, 410, 422, 499])
def test_projection_withholds_the_body_for_every_other_status(
    status_code: int | None,
) -> None:
    projection = model_error_client_projection(
        _model_error("unknown", status_code, "provider_code_4204", "RAW_MARKER")
    )

    assert projection is not None
    assert "RAW_MARKER" not in repr(projection)


@pytest.mark.parametrize("message", ["", "   ", "\n\t", None, 12, ["a"]])
def test_projection_ignores_an_empty_or_non_text_message(message: object) -> None:
    projection = model_error_client_projection(
        _model_error("server_error", 503, None, message)
    )

    assert projection is not None
    assert projection["error_message"] == (
        f"{HEAD} (503): the provider returned a server error."
    )


@pytest.mark.parametrize(
    "message, expected",
    [
        pytest.param("line1\n\nline2", "line1 line2", id="newlines-become-one-space"),
        pytest.param("a\tb\r\nc", "a b c", id="tabs-and-crlf"),
        pytest.param("x‮y", "xy", id="bidi-override-removed"),
        pytest.param("x​y", "xy", id="zero-width-space-removed"),
        pytest.param("x\x00\x07y", "xy", id="control-characters-removed"),
        pytest.param(
            "Limit for org-testorgid12345 reached",
            "Limit for org-*** reached",
            id="organization-id-masked",
        ),
        pytest.param(
            "key sk-test-key-echo-123456 bad",
            "key sk-*** bad",
            id="secret-key-masked",
        ),
        pytest.param(
            "key pk_test_key_echo_123456 bad",
            "key pk-*** bad",
            id="underscore-separator-masked-with-dash",
        ),
        pytest.param(
            "token eyJhbGciOiJIUzI1NiJ9.abcdef.ghijkl rejected",
            "token *** rejected",
            id="jwt-masked",
        ),
        pytest.param(
            "key " + "AIza" + "a" * 30 + " rejected",
            "key *** rejected",
            id="google-style-key-masked",
        ),
        pytest.param(
            "token_limit_exceeded_for_model",
            "token_limit_exceeded_for_model",
            id="token-word-is-not-a-prefix",
        ),
        pytest.param(
            "key_management_service_unavailable",
            "key_management_service_unavailable",
            id="key-word-is-not-a-prefix",
        ),
        pytest.param(
            "model proj-assistant-v2-large missing",
            "model proj-*** missing",
            id="proj-prefixed-model-name-masked-as-accepted-trade-off",
        ),
        pytest.param(
            "key gsk_abcdefghijklmnopqrstuvwxyz0123", "key gsk-***", id="gsk-prefix"
        ),
        pytest.param(
            "token hf_abcdefghijklmnopqrstuvwxyzABCDEF rejected",
            "token hf-*** rejected",
            id="hf-prefix",
        ),
        pytest.param(
            "key xai-abcdefghijklmnopqrstuvwxyz0123456789",
            "key xai-***",
            id="xai-prefix",
        ),
        pytest.param(
            "key nvapi-abcdefghijklmnopqrstuvwxyz0123456789",
            "key nvapi-***",
            id="nvapi-prefix",
        ),
        pytest.param(
            "AWS key AKIAIOSFODNN7EXAMPLE denied",
            "AWS key *** denied",
            id="aws-access-key",
        ),
        pytest.param(
            "Bearer abcdefghijklmnopqrstuvwxyz0123456789",
            "Bearer ***",
            id="bare-bearer",
        ),
        pytest.param(
            "Bearer authentication required",
            "Bearer authentication required",
            id="bearer-word-kept",
        ),
        pytest.param(
            "Bearer authentication_required_here",
            "Bearer authentication_required_here",
            id="bearer-long-word-without-digit-kept",
        ),
        pytest.param(
            "Bearer abc123",
            "Bearer abc123",
            id="bearer-short-token-kept",
        ),
        pytest.param(
            "AWS key ASIAIOSFODNN7EXAMPLE denied",
            "AWS key *** denied",
            id="aws-temporary-access-key",
        ),
        pytest.param(
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789",
            "Authorization: Bearer ***6789",
            id="authorization-header-masked-once",
        ),
        pytest.param(
            '{"api_key": "abcdefghij0123456789"}',
            '{"api_key": "***"}',
            id="json-double-quoted",
        ),
        pytest.param(
            "{'api_key': 'abcdefghij0123456789'}",
            "{'api_key': '***'}",
            id="json-single-quoted",
        ),
        pytest.param(
            "my_sk-abcdefghijklmnop1234", "my_sk-***", id="underscore-joined-prefix"
        ),
        pytest.param(
            "task-1234567890 failed",
            "task-1234567890 failed",
            id="ordinary-dash-word-kept",
        ),
    ],
)
def test_projection_cleans_the_provider_message(message: str, expected: str) -> None:
    projection = model_error_client_projection(
        _model_error("server_error", 500, None, message)
    )

    assert projection is not None
    text = projection["error_message"]
    prefix = f"{HEAD} (500): "
    assert text.startswith(prefix)
    assert text[len(prefix) :] == expected


def test_projection_applies_the_shared_credential_assignment_redactor() -> None:
    projection = model_error_client_projection(
        _model_error("server_error", 500, None, "request api_key=sk-live-1234567890")
    )

    assert projection is not None
    assert "sk-live-1234567890" not in projection["error_message"]


def test_projection_survives_a_cleaning_failure_by_dropping_the_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(_text: str) -> str:
        raise RuntimeError("redactor failure")

    monkeypatch.setattr(client_error_messages, "redact_sensitive_text", explode)

    projection = model_error_client_projection(
        _model_error("server_error", 503, None, "RAW_MARKER")
    )

    assert projection is not None
    assert "RAW_MARKER" not in repr(projection)
    assert projection["error_message"] == (
        f"{HEAD} (503): the provider returned a server error."
    )


def test_every_failure_kind_has_exactly_one_client_phrase_entry() -> None:
    assert set(MODEL_ERROR_KIND_PHRASES) == MODEL_PROVIDER_FAILURE_KINDS
    assert MODEL_ERROR_KIND_PHRASES["unknown"] is None
