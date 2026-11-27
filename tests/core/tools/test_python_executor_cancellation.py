"""Tests for execute_python_code cancellation support (issue #3249).

Cancelling a turn should stop an in-flight ``execute_python_code`` call
within a few seconds, rather than waiting for the full wall-clock limit.
"""

import threading
import time

from xagent.core.tools.core.python_subprocess import run_python_code_in_subprocess


class TestPythonSubprocessCancellation:
    """Test that cancel_event kills the subprocess promptly."""

    def test_normal_execution_still_works(self):
        result = run_python_code_in_subprocess(
            "print('hello')",
            capture_output=True,
            working_directory=None,
            timeout_seconds=10,
        )
        assert result["success"] is True
        assert "hello" in result["output"]

    def test_timeout_still_works(self):
        result = run_python_code_in_subprocess(
            "while True: pass",
            capture_output=True,
            working_directory=None,
            timeout_seconds=1,
        )
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_cancel_event_kills_subprocess(self):
        """Setting cancel_event kills the process within seconds."""
        cancel_event = threading.Event()
        result_holder: dict = {}

        def thread_fn():
            result_holder["result"] = run_python_code_in_subprocess(
                "while True: pass",
                capture_output=True,
                working_directory=None,
                timeout_seconds=30,
                cancel_event=cancel_event,
            )

        thread = threading.Thread(target=thread_fn, daemon=True)
        thread.start()
        time.sleep(1.0)

        cancel_event.set()
        thread.join(timeout=10)

        assert not thread.is_alive(), "Thread should finish after cancellation"
        result = result_holder["result"]
        assert result["success"] is False
        assert "cancelled" in result["error"].lower()

    def test_none_cancel_event_works(self):
        """Passing cancel_event=None should behave like the old code."""
        result = run_python_code_in_subprocess(
            "x = 1 + 1",
            capture_output=True,
            working_directory=None,
            timeout_seconds=5,
            cancel_event=None,
        )
        assert result["success"] is True
