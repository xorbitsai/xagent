"""The external command error frame admits only its own listed codes.

The admission rule itself is shared with ``create_terminal_task_error_event``
and pinned there; this pins the set this builder passes to it.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from xagent.web.services.client_error_messages import ClientErrorCode
from xagent.web.services.task_execution import create_external_command_error_event


def test_the_command_error_frame_without_a_code_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR):
        event = create_external_command_error_event(7, "x")

    assert set(event) == {"type", "message", "task_id", "timestamp"}
    assert "dropped=asserted_code" not in caplog.text


def test_a_terminal_frame_code_is_dropped_from_the_command_error_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kwargs: dict[str, Any] = {
        "asserted_code": ClientErrorCode.EXTERNAL_TURN_INTERRUPTED
    }
    with caplog.at_level(logging.ERROR):
        event = create_external_command_error_event(7, "x", **kwargs)

    assert set(event) == {"type", "message", "task_id", "timestamp"}
    dropped = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
        and "dropped=asserted_code" in record.getMessage()
    ]
    assert len(dropped) == 1
