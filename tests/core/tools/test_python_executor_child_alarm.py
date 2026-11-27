"""Tests for the child process fallback deadline (issue #3256).

If the backend crashes or is restarted mid-call, the detached child
process started with ``start_new_session=True`` has no limit left.
The bootstrap now installs ``signal.alarm(timeout + grace)`` so the
child terminates on its own even without the parent.
"""

from xagent.core.tools.core.python_subprocess import run_python_code_in_subprocess


class TestChildAlarmFallback:
    """Test the child-side signal.alarm fallback deadline."""

    def test_normal_execution_still_works(self):
        result = run_python_code_in_subprocess(
            "print('hello')",
            capture_output=True,
            working_directory=None,
            timeout_seconds=10,
        )
        assert result["success"] is True
        assert "hello" in result["output"]

    def test_parent_timeout_fires_before_child_alarm(self):
        """The parent timeout (2s) should fire before the child alarm (7s)."""
        result = run_python_code_in_subprocess(
            "while True: pass",
            capture_output=True,
            working_directory=None,
            timeout_seconds=2,
        )
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_child_has_alarm_set(self):
        """The child process should have signal.alarm configured."""
        result = run_python_code_in_subprocess(
            "import signal\n"
            "alarm_remaining = signal.getitimer(signal.ITIMER_REAL)[0]\n"
            "assert alarm_remaining > 0, 'no alarm set'",
            capture_output=True,
            working_directory=None,
            timeout_seconds=10,
        )
        assert result["success"] is True, (
            f"Child did not have alarm set: {result.get('error', '')}"
        )
