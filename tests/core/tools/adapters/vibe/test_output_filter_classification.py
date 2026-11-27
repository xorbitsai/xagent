"""
Tests that output filtering never changes ``tool_result_succeeded``'s verdict.

Covers the acceptance criteria from issue #3808: a small tool output field cap
can drop the keys that classify a failed call.
"""

from __future__ import annotations

import pytest

from xagent.core.agent.result import tool_result_succeeded
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)


def _make_wrapper(max_fields: int) -> OutputFilteredToolWrapper:
    """Build a wrapper with a tiny field cap and generous string/depth limits."""
    return OutputFilteredToolWrapper(
        target_tool=None,  # not used by _filter_result / _restore_failure_classification
        max_chars=10_000,
        max_fields=max_fields,
        max_recursion=50,
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: #3803's envelope at cap 1 and 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [1, 3])
def test_envelope_with_success_false_still_failure(cap: int):
    """At a cap of 1 and of 3, the envelope result is still a failure."""
    result = {
        "content": "some content",
        "structured_content": {"key": "value"},
        "is_error": False,
        "success": False,
    }
    assert tool_result_succeeded(result) is False

    wrapper = _make_wrapper(cap)
    filtered = wrapper._filter_result(result)

    assert tool_result_succeeded(filtered) is False
    # is_error must not be turned to True by the restoration logic
    assert filtered.get("is_error") is not True


# ---------------------------------------------------------------------------
# Acceptance criterion 2: plain is_error at cap 2
# ---------------------------------------------------------------------------


def test_plain_is_error_true_still_failure():
    """At a cap of 2, a plain is_error=True result is still a failure."""
    result = {
        "content": "some content",
        "structured_content": {"key": "value"},
        "is_error": True,
    }
    assert tool_result_succeeded(result) is False

    wrapper = _make_wrapper(2)
    filtered = wrapper._filter_result(result)

    assert tool_result_succeeded(filtered) is False
    assert filtered.get("is_error") is True


# ---------------------------------------------------------------------------
# Acceptance criterion 3: success=True at a small cap still reads as success
# ---------------------------------------------------------------------------


def test_success_true_still_success():
    """A success=True result at a small cap still reads as a success."""
    result = {
        "content": "some content",
        "structured_content": {"key": "value"},
        "success": True,
        "extra1": "a",
        "extra2": "b",
    }
    assert tool_result_succeeded(result) is True

    wrapper = _make_wrapper(2)
    filtered = wrapper._filter_result(result)

    assert tool_result_succeeded(filtered) is True


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_status_error_restored():
    """A status='error' result that loses the key is restored."""
    result = {
        "content": "content",
        "structured_content": "sc",
        "status": "error",
    }
    assert tool_result_succeeded(result) is False

    wrapper = _make_wrapper(2)
    filtered = wrapper._filter_result(result)

    assert tool_result_succeeded(filtered) is False
    assert filtered.get("status") == "error"


def test_success_false_preferred_over_is_error():
    """When both success=False and is_error=True exist, success=False is restored first."""
    result = {
        "content": "content",
        "structured_content": "sc",
        "is_error": True,
        "success": False,
    }
    wrapper = _make_wrapper(2)  # drops both is_error and success
    filtered = wrapper._filter_result(result)

    assert tool_result_succeeded(filtered) is False
    assert filtered.get("success") is False


def test_non_dict_result_unchanged():
    """Non-dict results pass through without restoration logic."""
    wrapper = _make_wrapper(1)
    assert wrapper._filter_result("ok") == "ok"
    assert wrapper._filter_result(None) is None
    assert wrapper._filter_result(42) == 42


def test_success_result_not_modified():
    """A success result dict is not modified by the restoration logic."""
    result = {"content": "content", "success": True}
    wrapper = _make_wrapper(1)
    filtered = wrapper._filter_result(result)

    # success=True is within the cap, so it should be preserved
    assert filtered.get("success") is True
    assert tool_result_succeeded(filtered) is True


def test_cap_large_enough_no_restoration_needed():
    """With a large cap, no restoration is needed because no keys are dropped."""
    result = {
        "content": "content",
        "structured_content": "sc",
        "is_error": False,
        "success": False,
    }
    wrapper = _make_wrapper(1000)
    filtered = wrapper._filter_result(result)

    assert tool_result_succeeded(filtered) is False
    assert filtered == result
