from __future__ import annotations

from xagent.core.tools.user_interaction import (
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
