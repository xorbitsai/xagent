from __future__ import annotations

import pytest

from xagent.core.tools.user_interaction import (
    ToolInteractionSettlement,
    tool_result_waits_for_user,
    user_interaction_resume_callable,
)


def test_waiting_status_detection_is_normalized() -> None:
    assert tool_result_waits_for_user({"status": "waiting_for_user"}) is True
    assert tool_result_waits_for_user({"status": " WAITING_FOR_USER "}) is True
    assert tool_result_waits_for_user({"status": "completed"}) is False
    assert tool_result_waits_for_user("waiting_for_user") is False
    assert tool_result_waits_for_user(None) is False


def test_resume_capability_detection() -> None:
    class Resumable:
        def resume_user_interaction(
            self,
            *,
            interaction_id: str,
            response: str,
        ) -> None:
            return None

    assert callable(user_interaction_resume_callable(Resumable()))
    assert user_interaction_resume_callable(object()) is None


def test_successful_settlement_preserves_the_tool_result() -> None:
    result = {"success": True, "post_urn": "urn:li:share:123"}

    settlement = ToolInteractionSettlement.succeeded(result)

    assert settlement.projected_result() is result


def test_successful_settlement_rejects_a_failed_tool_result() -> None:
    with pytest.raises(ValueError, match="cannot carry a failed tool result"):
        ToolInteractionSettlement.succeeded({"success": False, "error": "failed"})


@pytest.mark.parametrize(
    "result, message",
    [
        (None, "needs a result"),
        ({"status": "waiting_for_user"}, "cannot wait for user input"),
    ],
)
def test_successful_settlement_rejects_non_terminal_results(
    result, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ToolInteractionSettlement.succeeded(result)


def test_successful_settlement_uses_the_canonical_failure_classifier() -> None:
    result = {"status": " error "}

    assert ToolInteractionSettlement.succeeded(result).result is result


@pytest.mark.parametrize("status", ["rejected", "failed", "dispatch_unknown"])
def test_non_successful_settlement_has_an_unambiguous_failure_shape(
    status: str,
) -> None:
    settlement = ToolInteractionSettlement(status=status, result={"detail": "safe"})  # type: ignore[arg-type]

    assert settlement.projected_result() == {
        "success": False,
        "settlement_status": status,
        "error": {
            "rejected": "The user rejected the tool call.",
            "failed": "The resumed tool call failed.",
            "dispatch_unknown": (
                "The tool call may have reached the external system. Automatic retry "
                "is disabled; verify the external system before trying again."
            ),
        }[status],
        "detail": "safe",
    }


def test_failed_settlement_preserves_a_tool_specific_status() -> None:
    settlement = ToolInteractionSettlement.failed(
        result={"status": "cancelled_by_policy"}
    )

    assert settlement.projected_result()["status"] == "cancelled_by_policy"
    assert settlement.projected_result()["settlement_status"] == "failed"


def test_settlement_rejects_an_unknown_status() -> None:
    with pytest.raises(ValueError, match="Invalid tool interaction settlement"):
        ToolInteractionSettlement(status="unknown")  # type: ignore[arg-type]
