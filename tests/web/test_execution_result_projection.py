import pytest

from xagent.core.agent.execution_adapter import INTERRUPTED_USER_MESSAGE
from xagent.web.models.task import TaskStatus
from xagent.web.services.client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    ClientErrorCode,
    client_error_message,
)
from xagent.web.services.execution_result_projection import (
    EMPTY_CHANNEL_OUTPUT_FALLBACK,
    execution_result_diagnostic_error,
    present_failed_result,
    project_execution_result_for_channel,
)


def test_project_execution_result_waiting_for_user_uses_chat_message_as_question():
    projection = project_execution_result_for_channel(
        {
            "status": "waiting_for_user",
            "success": False,
            "output": "Need input.",
            "chat_response": {"message": "Choose A or B", "interactions": []},
        }
    )

    assert projection.task_status == TaskStatus.WAITING_FOR_USER
    assert projection.visible_text == "Choose A or B"
    assert projection.transcript_content == "Choose A or B"
    assert projection.message_type == "question"
    assert projection.interactions == []


@pytest.mark.parametrize(
    "outcome", ["completed", "partial", "blocked", None, "invalid", ["partial"]]
)
@pytest.mark.parametrize(
    "status,success",
    [
        ("completed", True),
        ("failed", False),
        ("waiting_for_user", True),
        ("interrupted", True),
    ],
)
def test_projection_keeps_valid_outcome_only_for_completed_execution(
    status, success, outcome
):
    projection = project_execution_result_for_channel(
        {"status": status, "success": success, "completion_outcome": outcome}
    )
    expected = (
        outcome
        if status == "completed" and outcome in ("completed", "partial", "blocked")
        else None
    )
    assert projection.completion_outcome == expected


def test_project_execution_result_appends_interactions_to_visible_text():
    projection = project_execution_result_for_channel(
        {
            "success": True,
            "output": "Need details.",
            "chat_response": {
                "message": "Choose a destination",
                "interactions": [
                    {
                        "label": "Destination",
                        "options": [{"label": "Tokyo"}, {"value": "Osaka"}],
                    }
                ],
            },
        }
    )

    assert projection.task_status == TaskStatus.COMPLETED
    assert projection.transcript_content == "Choose a destination"
    assert projection.message_type == "question"
    assert projection.interactions == [
        {
            "label": "Destination",
            "options": [{"label": "Tokyo"}, {"value": "Osaka"}],
        }
    ]
    assert projection.visible_text == (
        "Choose a destination\n\n• Destination\n  Options: Tokyo, Osaka"
    )


def test_project_execution_result_falls_back_for_empty_output():
    projection = project_execution_result_for_channel({"success": True, "output": None})

    assert projection.task_status == TaskStatus.COMPLETED
    assert projection.visible_text == EMPTY_CHANNEL_OUTPUT_FALLBACK
    assert projection.transcript_content == EMPTY_CHANNEL_OUTPUT_FALLBACK
    assert projection.message_type == "assistant_response"


def test_project_execution_result_maps_interrupted_to_paused():
    projection = project_execution_result_for_channel(
        {
            "status": "interrupted",
            "success": False,
            "output": "ReActPattern interrupted.",
        }
    )

    assert projection.task_status == TaskStatus.PAUSED
    assert projection.visible_text == INTERRUPTED_USER_MESSAGE
    assert projection.transcript_content == ""
    assert projection.interactions == []


def test_project_failed_result_separates_diagnostic_error_from_safe_display() -> None:
    raw_error = "provider token=secret"
    interaction_secret = "interaction token=secret"

    projection = project_execution_result_for_channel(
        {
            "success": False,
            "status": "error",
            "output": raw_error,
            "error": raw_error,
            "chat_response": {
                "message": raw_error,
                "interactions": [{"label": interaction_secret}],
            },
        }
    )

    assert projection.task_status == TaskStatus.FAILED
    assert projection.visible_text == "Task execution failed."
    assert projection.transcript_content == "Task execution failed."
    assert projection.diagnostic_error == raw_error
    assert raw_error not in projection.visible_text
    assert interaction_secret not in projection.visible_text
    assert projection.interactions == []


def test_project_output_only_failure_keeps_original_text_as_diagnostic() -> None:
    projection = project_execution_result_for_channel(
        {
            "success": False,
            "status": "error",
            "output": "provider token=secret",
            "chat_response": {
                "interactions": [{"label": "interaction token=secret"}],
            },
        }
    )

    assert projection.visible_text == "Task execution failed."
    assert projection.transcript_content == "Task execution failed."
    assert projection.diagnostic_error == "provider token=secret"
    assert projection.interactions == []


def test_project_empty_failure_has_no_diagnostic_and_uses_safe_display() -> None:
    projection = project_execution_result_for_channel(
        {"success": False, "status": "error"}
    )

    assert projection.visible_text == "Task execution failed."
    assert projection.transcript_content == "Task execution failed."
    assert projection.diagnostic_error is None
    assert projection.interactions == []


def test_project_failed_result_maps_classified_tool_failure() -> None:
    safe_error = client_error_message(ClientErrorCode.EXCEL_WORKBOOK_ACCESS_DENIED)
    projection = project_execution_result_for_channel(
        {
            "success": False,
            "status": "invalid_tool_protocol",
            "error": "truncated_tool_arguments",
            "failure_code": "excel_workbook_access_denied",
        }
    )

    assert projection.visible_text == safe_error
    assert projection.transcript_content == CLIENT_SAFE_TASK_FAILURE


def test_project_failed_result_rejects_unallowlisted_tool_failure() -> None:
    projection = project_execution_result_for_channel(
        {
            "success": False,
            "status": "invalid_tool_protocol",
            "failure_code": "arbitrary",
        }
    )

    assert projection.visible_text == CLIENT_SAFE_TASK_FAILURE


def test_project_failed_result_keeps_an_existing_error_code_over_tool_failure() -> None:
    projection = project_execution_result_for_channel(
        {
            "success": False,
            "status": "quota_exceeded",
            "error_code": "quota_exceeded",
            "failure_code": "excel_workbook_access_denied",
        }
    )

    assert projection.visible_text == CLIENT_SAFE_TASK_FAILURE


def test_project_failure_appends_task_id_to_visible_text_only() -> None:
    projection = project_execution_result_for_channel(
        {"success": False, "status": "error", "error": "boom"}, task_id=45
    )

    assert projection.task_status == TaskStatus.FAILED
    assert projection.visible_text == "Task execution failed. (Task ID: 45)"
    assert projection.transcript_content == "Task execution failed."
    assert projection.diagnostic_error == "boom"


@pytest.mark.parametrize(
    "result",
    [
        {"success": True, "status": "completed", "output": "done"},
        {"success": True, "status": "interrupted", "output": "partial"},
        {"success": False, "status": "waiting_for_user", "output": "Need input."},
    ],
)
def test_project_non_failure_ignores_task_id(result: dict) -> None:
    assert project_execution_result_for_channel(
        result, task_id=45
    ) == project_execution_result_for_channel(result)
    assert (
        "Task ID"
        not in project_execution_result_for_channel(result, task_id=45).visible_text
    )


@pytest.mark.parametrize(
    "result, expected",
    [
        pytest.param(
            {"diagnostic_error": "owner text", "error": "aggregate"},
            "owner text",
            id="dedicated-key-wins",
        ),
        pytest.param(
            {"diagnostic_error": "  owner text \n", "error": "aggregate"},
            "owner text",
            id="dedicated-key-is-stripped",
        ),
        pytest.param(
            {"diagnostic_error": "", "error": " aggregate "},
            "aggregate",
            id="empty-dedicated-key-falls-back",
        ),
        pytest.param(
            {"diagnostic_error": "   ", "error": "aggregate"},
            "aggregate",
            id="blank-dedicated-key-falls-back",
        ),
        pytest.param(
            {"diagnostic_error": 7, "error": "aggregate"},
            "aggregate",
            id="non-str-dedicated-key-falls-back",
        ),
        pytest.param({"error": "aggregate"}, "aggregate", id="error-only"),
        pytest.param({"diagnostic_error": None, "error": None}, "", id="both-empty"),
        pytest.param({}, "", id="neither-key"),
        pytest.param(
            {
                "error_code": "quota_exceeded",
                "diagnostic_error": "owner text",
                "error": "quota reason",
            },
            "quota reason",
            id="coded-failure-reports-its-own-error-text",
        ),
        pytest.param(
            {"error_code": None, "diagnostic_error": "owner text", "error": "x"},
            "owner text",
            id="none-code-is-not-coded",
        ),
    ],
)
def test_execution_result_diagnostic_error(
    result: dict[str, object], expected: str
) -> None:
    assert execution_result_diagnostic_error(result) == expected


_ECHOED_KEY_DIAGNOSTIC = (
    "OpenAI authentication failed (401): Error code: 401 - "
    "{'error': {'message': 'Incorrect API key provided: "
    "sk-live-ECHOEDKEY1234567890'}} | "
    'provider_raw={"api_key": "rawsecretvalue123"} | '
    "org-testorgid12345 proj-assistant-v2-large"
)


def test_diagnostic_error_masks_credentials_but_keeps_the_diagnosis() -> None:
    masked = execution_result_diagnostic_error(
        {"diagnostic_error": _ECHOED_KEY_DIAGNOSTIC, "error": "aggregate"}
    )

    assert "ECHOEDKEY1234567890" not in masked
    assert "rawsecretvalue123" not in masked
    assert "sk-***" in masked
    assert "org-testorgid12345" in masked
    assert "proj-assistant-v2-large" in masked
    assert "provider_raw=" in masked
    assert masked.startswith("OpenAI authentication failed (401)")


def test_diagnostic_error_returns_the_error_fallback_unmasked() -> None:
    result = {"error": "plain sk-live-ECHOEDKEY1234567890"}

    assert execution_result_diagnostic_error(result) == (
        "plain sk-live-ECHOEDKEY1234567890"
    )


def test_channel_projection_masks_credentials_in_the_dedicated_diagnostic() -> None:
    projection = project_execution_result_for_channel(
        {
            "success": False,
            "status": "error",
            "output": "All 1 patterns failed",
            "error": "All 1 patterns failed",
            "diagnostic_error": _ECHOED_KEY_DIAGNOSTIC,
        }
    )

    assert projection.diagnostic_error is not None
    assert "ECHOEDKEY1234567890" not in projection.diagnostic_error


_FORBIDDEN_MODEL_ERROR = {
    "kind": "access_denied",
    "status_code": 403,
    "provider_code": "provider_code_4204",
    "message": "Model is decommissioned",
}
_PROJECTED_TEXT = (
    "Model provider call failed (403 provider_code_4204): Model is decommissioned"
)


def test_present_failed_result_projects_a_model_error() -> None:
    presentation = present_failed_result(
        {
            "success": False,
            "error": "All 1 patterns failed",
            "diagnostic_error": "OpenAI API error (403): provider_raw=RAW_MARKER",
            "model_error": _FORBIDDEN_MODEL_ERROR,
        }
    )

    assert presentation.diagnostic_error == (
        "OpenAI API error (403): provider_raw=RAW_MARKER"
    )
    assert presentation.visible_text == _PROJECTED_TEXT
    assert presentation.error_code == "model_error"
    assert presentation.error_details == {
        "kind": "access_denied",
        "status_code": 403,
        "provider_code": "provider_code_4204",
        "message": _PROJECTED_TEXT,
    }


@pytest.mark.parametrize(
    "model_error",
    [
        pytest.param(
            {
                "kind": "bad_request",
                "status_code": 400,
                "provider_code": "invalid_request",
                "message": "RAW_MARKER",
            },
            id="complete-400",
        ),
        pytest.param({"status_code": 400}, id="empty-dict-with-400"),
    ],
)
def test_present_failed_result_forces_the_generic_text_when_the_projection_is_withheld(
    model_error: dict[str, object],
) -> None:
    raw = "OpenAI bad request (400): RAW_MARKER"
    presentation = present_failed_result(
        {
            "success": False,
            "output": raw,
            "error": raw,
            "diagnostic_error": raw,
            "model_error": model_error,
        }
    )

    assert presentation.diagnostic_error == raw
    assert presentation.visible_text == CLIENT_SAFE_TASK_FAILURE
    assert presentation.error_code is None
    assert presentation.error_details is None


def test_present_failed_result_projects_a_malformed_model_error_to_the_fallback() -> (
    None
):
    presentation = present_failed_result(
        {"success": False, "error": "x", "model_error": {"status_code": "403"}}
    )

    assert presentation.visible_text == "Model provider call failed."
    assert presentation.error_code == "model_error"


@pytest.mark.parametrize("model_error", ["text", ["a"], 7, None])
def test_present_failed_result_keeps_the_callers_text_when_model_error_is_not_a_dict(
    model_error: object,
) -> None:
    presentation = present_failed_result(
        {"success": False, "error": "failure text", "model_error": model_error}
    )

    assert presentation.diagnostic_error == "failure text"
    assert presentation.visible_text is None
    assert presentation.error_code is None
    assert presentation.error_details is None


def test_present_failed_result_lets_an_existing_error_code_take_precedence() -> None:
    details = {"limit": 5, "used": 5}
    presentation = present_failed_result(
        {
            "success": False,
            "status": "quota_exceeded",
            "output": "Quota reached",
            "error": "Quota reached",
            "error_code": "quota_exceeded",
            "error_details": details,
            "diagnostic_error": "OpenAI API error (403): provider_raw=RAW_MARKER",
            "model_error": _FORBIDDEN_MODEL_ERROR,
        }
    )

    assert presentation.diagnostic_error == "Quota reached"
    assert presentation.visible_text is None
    assert presentation.error_code == "quota_exceeded"
    assert presentation.error_details == details


@pytest.mark.parametrize("error_code, error_details", [(7, {"a": 1}), ("c", "text")])
def test_present_failed_result_passes_only_well_typed_gate_fields(
    error_code: object, error_details: object
) -> None:
    presentation = present_failed_result(
        {
            "success": False,
            "error": "x",
            "error_code": error_code,
            "error_details": error_details,
        }
    )

    assert presentation.error_code == (
        error_code if isinstance(error_code, str) else None
    )
    assert presentation.error_details == (
        error_details if isinstance(error_details, dict) else None
    )


def test_channel_projection_reports_the_dedicated_diagnostic_and_keeps_the_safe_text() -> (
    None
):
    projection = project_execution_result_for_channel(
        {
            "success": False,
            "status": "error",
            "output": "All 1 patterns failed",
            "error": "All 1 patterns failed",
            "diagnostic_error": "OpenAI API error (403): provider_raw=RAW_MARKER",
            "model_error": _FORBIDDEN_MODEL_ERROR,
        }
    )

    assert projection.diagnostic_error == (
        "OpenAI API error (403): provider_raw=RAW_MARKER"
    )
    assert projection.visible_text == CLIENT_SAFE_TASK_FAILURE
    assert projection.transcript_content == CLIENT_SAFE_TASK_FAILURE
