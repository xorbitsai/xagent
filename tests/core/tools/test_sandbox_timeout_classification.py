"""Tests for sandbox SIGKILL classification (issue #3258).

A SIGKILL from the OOM killer after queueing on exec_lock was misreported
as a timeout.  Now we use ``timeout --verbose`` and check stderr for the
"sending signal" message to distinguish a real timeout from other kills.
"""

from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    _SIGKILL_EXIT_CODES,
    _TIMEOUT_VERBOSE_SIGNAL,
)


class TestTimeoutClassification:
    """Test the constants used to classify SIGKILL exits."""

    def test_sigkill_exit_codes_include_docker_and_subprocess(self):
        assert 137 in _SIGKILL_EXIT_CODES
        assert -9 in _SIGKILL_EXIT_CODES

    def test_timeout_verbose_signal_string(self):
        """The signal string matches what GNU timeout --verbose prints."""
        assert "sending signal KILL" in _TIMEOUT_VERBOSE_SIGNAL

    def test_oom_kill_stderr_does_not_contain_signal_message(self):
        """An OOM kill does not produce the timeout verbose message."""
        oom_stderr = "Killed\n"
        assert _TIMEOUT_VERBOSE_SIGNAL not in oom_stderr

    def test_real_timeout_stderr_contains_signal_message(self):
        """A real timeout produces the verbose signal message."""
        timeout_stderr = "timeout: sending signal KILL to command 'python'\n"
        assert _TIMEOUT_VERBOSE_SIGNAL in timeout_stderr
