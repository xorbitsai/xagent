from __future__ import annotations

import pytest

from xagent.core.agent.result import (
    ClassifiedToolFailure,
    normalize_tool_failure_code,
    tool_result_requires_authentication,
    tool_result_succeeded,
    unwrap_final_answer_content,
)


class _FailureCodeStringSubclass(str):
    pass


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("authentication_required", "authentication_required"),
        ("oauth_token_required", "oauth_token_required"),
        ("unsupported_nested_interaction", "unsupported_nested_interaction"),
        ("missing_delegated_output", "missing_delegated_output"),
        ("other_valid_code", None),
        (" oauth_token_required", None),
        ("OAUTH_TOKEN_REQUIRED", None),
        (None, None),
        (123, None),
        (_FailureCodeStringSubclass("oauth_token_required"), None),
    ],
)
def test_normalize_tool_failure_code_uses_exact_allowlist(value, expected):
    assert normalize_tool_failure_code(value) == expected


def test_classified_tool_failure_accepts_only_allowlisted_plain_string():
    outcome = ClassifiedToolFailure(failure_code="oauth_token_required")

    assert outcome.failure_code == "oauth_token_required"

    with pytest.raises(ValueError, match="invalid tool failure code"):
        ClassifiedToolFailure(failure_code="other_valid_code")

    with pytest.raises(ValueError, match="invalid tool failure code"):
        ClassifiedToolFailure(
            failure_code=_FailureCodeStringSubclass("oauth_token_required")
        )


@pytest.mark.parametrize(
    "code",
    [
        "authentication_required",
        "unsupported_nested_interaction",
        "missing_delegated_output",
    ],
)
def test_classified_tool_failure_rejects_non_oauth_runtime_codes(code):
    """The runtime allowlist must not widen the OAuth sentinel's validator."""

    assert normalize_tool_failure_code(code) == code
    with pytest.raises(ValueError, match="invalid tool failure code"):
        ClassifiedToolFailure(failure_code=code)


@pytest.mark.parametrize(
    "result",
    [
        {"success": False},
        {"status": "error"},
        {"status": "ERROR"},
        {"is_error": True},
    ],
)
def test_tool_result_succeeded_recognizes_supported_failure_shapes(result):
    assert tool_result_succeeded(result) is False


@pytest.mark.parametrize(
    "result",
    [
        None,
        "ok",
        {},
        {"success": True},
        {"status": "success"},
        {"is_error": False},
        {"is_error": 1},
    ],
)
def test_tool_result_succeeded_preserves_non_failure_results(result):
    assert tool_result_succeeded(result) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"success": True},
        {"is_error": False},
        {"failure_code": "oauth_token_required"},
        {"failure_code": "AUTHENTICATION_REQUIRED"},
        {"failure_code": _FailureCodeStringSubclass("authentication_required")},
    ],
)
def test_authentication_failure_requires_exact_classified_envelope(overrides):
    result = {
        "success": False,
        "is_error": True,
        "failure_code": "authentication_required",
    }
    result.update(overrides)
    assert tool_result_requires_authentication(result) is (not overrides)


@pytest.mark.parametrize(
    "result",
    [
        None,
        "HTTP 401",
        {"status_code": 401},
        {"failure_code": "authentication_required"},
    ],
)
def test_unclassified_results_do_not_request_authentication(result):
    assert not tool_result_requires_authentication(result)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "<final_answer>\nNo, images can't be sent here.\n</final_answer>",
            "No, images can't be sent here.",
        ),
        ("  <FINAL_ANSWER>Done.</Final_Answer>\n", "Done."),
        ("Resolved the blockers.</final_answer>", "Resolved the blockers."),
        (
            "<final_answer>\nStarted without a closing tag.",
            "Started without a closing tag.",
        ),
        (
            '<final_answer>{"action": "final_answer", "action_input": "Hi"}</final_answer>',
            "Hi",
        ),
    ],
)
def test_unwrap_final_answer_content_removes_tags_at_either_end(content, expected):
    assert unwrap_final_answer_content(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        "Wrap the reply in a <final_answer> tag when the tool is unavailable.",
        "Close it with </final_answer> and nothing after.\nThen stop.",
        "  A plain answer with surrounding whitespace.\n",
    ],
)
def test_unwrap_final_answer_content_leaves_text_without_edge_tags_unchanged(content):
    assert unwrap_final_answer_content(content) == content


def test_unwrap_final_answer_content_keeps_legacy_json_unwrapping():
    assert (
        unwrap_final_answer_content('{"action": "final_answer", "action_input": "Hi"}')
        == "Hi"
    )
    assert unwrap_final_answer_content('{"final_answer": {"rows": 3}}') == (
        '{\n  "rows": 3\n}'
    )
