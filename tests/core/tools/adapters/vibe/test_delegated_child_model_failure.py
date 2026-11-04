"""A delegated child's model-provider failure details stay on the child result.

The parent only ever sees the child's aggregate ``error`` sentence; the
``diagnostic_error`` / ``model_error`` keys never reach the delegation error.
"""

from __future__ import annotations

from xagent.core.tools.adapters.vibe.agent_tool import (
    _classify_delegated_child_failure,
)

AGGREGATE_ERROR = "All 1 patterns failed or returned unsuccessful results."
DIAGNOSTIC_ERROR = (
    "Provider API error (403): Error code: 403 - Model is decommissioned | "
    "request_id=req-1"
)


def test_delegation_error_is_the_aggregate_sentence_without_provider_details() -> None:
    child_result = {
        "status": "failed",
        "success": False,
        "error": AGGREGATE_ERROR,
        "output": AGGREGATE_ERROR,
        "diagnostic_error": DIAGNOSTIC_ERROR,
        "model_error": {
            "kind": "access_denied",
            "status_code": 403,
            "provider_code": "provider_code_4204",
            "message": "Model is decommissioned",
        },
    }

    failure = _classify_delegated_child_failure(child_result)

    assert failure is not None
    assert failure["error"] == AGGREGATE_ERROR
    rendered = repr(failure)
    assert "Model is decommissioned" not in rendered
    assert "provider_code_4204" not in rendered
    assert "request_id=req-1" not in rendered
    assert "access_denied" not in rendered
